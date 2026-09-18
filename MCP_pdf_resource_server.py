"""
Author: Stefan Stiller
Affiliation: Leibniz-Centre for Agricultural Landscape Research (ZALF) e.V.
ORCID: https://orcid.org/0009-0004-7468-1678
License: GPL-3.0

MCP PDF Resource Server - Topic-Based Access

This server exposes PDF documents as topic-based resources that can be accessed by MCP clients.
Each PDF is divided into topic blocks defined by page and line ranges, allowing selective
access to specific content sections rather than entire documents.

The server provides access to PDF documents stored in the Inputs directory,
allowing LLMs to read and analyze specific topics from agricultural research documents.
"""

import os
import sys
import logging
import asyncio
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

# PDF processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

# MCP imports
try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import Resource, TextResourceContents
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Create dummy classes for testing without MCP
    class FastMCP:
        def __init__(self, *args, **kwargs):
            pass
        def resource(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def tool(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def run(self):
            pass
    class Context:
        pass


@dataclass
class TopicBlock:
    """Represents a topic block within a PDF document."""
    descriptor: str  # Unique identifier for the topic
    pdf_id: str  # Identifier for the PDF (e.g., "lucas2015", "watson2017")
    title: str  # Human-readable title
    start_page: int  # Starting page (1-indexed)
    start_line: int  # Starting line within the page
    end_page: int  # Ending page (1-indexed)
    end_line: Optional[int]  # Ending line within the page (None means end of page)
    scope: str  # Description of the topic scope
    pdf_filename: str  # Actual PDF filename


@dataclass
class AppContext:
    """Application context for the PDF resource server."""
    pdf_directory: str
    available_pdfs: List[str]
    topic_blocks: Dict[str, List[TopicBlock]]  # pdf_id -> list of TopicBlocks


class TopicParser:
    """Parses topic definition files to extract TopicBlock information."""
    
    @staticmethod
    def parse_lucas_file(file_path: str, pdf_filename: str) -> List[TopicBlock]:
        """
        Parse Lucas 2015 topic definition file.
        
        Format:
        Number) Title
        Pages: X or X–Y
        Lines: Y–Z or "Page X from line Y onward, and most of pages X–Y"
        Descriptor: ...
        Scope: ...
        """
        blocks = []
        pdf_id = "lucas2015"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by numbered entries
        entries = re.split(r'^(\d+)\)', content, flags=re.MULTILINE)
        
        for i in range(1, len(entries), 2):
            if i + 1 >= len(entries):
                break
                
            entry_num = entries[i]
            entry_text = entries[i + 1]
            
            # Extract title (first line)
            lines = entry_text.strip().split('\n')
            if not lines:
                continue
                
            title = lines[0].strip()
            
            # Extract pages
            pages_str = None
            lines_str = None
            descriptor = None
            scope = None
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('Pages:'):
                    pages_str = line.replace('Pages:', '').strip()
                elif line.startswith('Lines:'):
                    lines_str = line.replace('Lines:', '').strip()
                elif line.startswith('Descriptor:'):
                    descriptor = line.replace('Descriptor:', '').strip()
                elif line.startswith('Scope:'):
                    scope = line.replace('Scope:', '').strip()
            
            if not all([pages_str, lines_str, descriptor, scope]):
                continue
            
            # Parse pages
            start_page, end_page = TopicParser._parse_pages(pages_str)
            
            # Parse lines
            start_line, end_line = TopicParser._parse_lines(lines_str, start_page, end_page)
            
            if start_page and start_line:
                block = TopicBlock(
                    descriptor=descriptor,
                    pdf_id=pdf_id,
                    title=title,
                    start_page=start_page,
                    start_line=start_line,
                    end_page=end_page or start_page,
                    end_line=end_line,
                    scope=scope,
                    pdf_filename=pdf_filename
                )
                blocks.append(block)
        
        return blocks
    
    @staticmethod
    def parse_watson_file(file_path: str, pdf_filename: str) -> List[TopicBlock]:
        """
        Parse Watson 2017 topic definition file.
        
        Format:
        Number) Title
        Pages and lines: pX lY → pZ lW or pX lY → pZ l_end
        Descriptor: ...
        Scope: ...
        """
        blocks = []
        pdf_id = "watson2017"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by numbered entries
        entries = re.split(r'^(\d+)\)', content, flags=re.MULTILINE)
        
        for i in range(1, len(entries), 2):
            if i + 1 >= len(entries):
                break
                
            entry_num = entries[i]
            entry_text = entries[i + 1]
            
            # Extract title (first line)
            lines = entry_text.strip().split('\n')
            if not lines:
                continue
                
            title = lines[0].strip()
            
            # Extract pages and lines
            pages_lines_str = None
            descriptor = None
            scope = None
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('Pages and lines:'):
                    pages_lines_str = line.replace('Pages and lines:', '').strip()
                elif line.startswith('Descriptor:'):
                    descriptor = line.replace('Descriptor:', '').strip()
                elif line.startswith('Scope:'):
                    scope = line.replace('Scope:', '').strip()
            
            if not all([pages_lines_str, descriptor, scope]):
                continue
            
            # Parse pages and lines (format: pX lY → pZ lW or pX lY → pZ l_end)
            match = re.match(r'p(\d+)\s+l(\d+)\s*→\s*p(\d+)\s+l(?:(\d+)|_end)', pages_lines_str)
            if match:
                start_page = int(match.group(1))
                start_line = int(match.group(2))
                end_page = int(match.group(3))
                end_line = int(match.group(4)) if match.group(4) else None
                
                block = TopicBlock(
                    descriptor=descriptor,
                    pdf_id=pdf_id,
                    title=title,
                    start_page=start_page,
                    start_line=start_line,
                    end_page=end_page,
                    end_line=end_line,
                    scope=scope,
                    pdf_filename=pdf_filename
                )
                blocks.append(block)
        
        return blocks
    
    @staticmethod
    def _parse_pages(pages_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse page range string. Returns (start_page, end_page)."""
        pages_str = pages_str.strip()
        
        # Handle range like "3–5"
        if '–' in pages_str or '-' in pages_str:
            parts = re.split(r'[–-]', pages_str)
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    return start, end
                except ValueError:
                    pass
        
        # Handle single page
        try:
            page = int(pages_str.strip())
            return page, page
        except ValueError:
            return None, None
    
    @staticmethod
    def _parse_lines(lines_str: str, start_page: int, end_page: int) -> Tuple[Optional[int], Optional[int]]:
        """Parse line range string. Returns (start_line, end_line)."""
        lines_str = lines_str.strip()
        
        # Handle complex format like "Page 3 from line 30 onward, and most of pages 4–5"
        if 'from line' in lines_str.lower() and 'onward' in lines_str.lower():
            match = re.search(r'from line (\d+)', lines_str, re.IGNORECASE)
            if match:
                start_line = int(match.group(1))
                # For "onward" cases, we'll use end_page and None for end_line
                # This means the block continues to the end of the last page
                return start_line, None
        
        # Handle simple range like "46–76"
        if '–' in lines_str or '-' in lines_str:
            parts = re.split(r'[–-]', lines_str)
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    return start, end
                except ValueError:
                    pass
        
        # Handle single line number
        try:
            line = int(lines_str.strip())
            return line, line
        except ValueError:
            return None, None


class PDFReader:
    """Handles PDF reading operations with multiple backend support."""
    
    def __init__(self):
        """Initialize PDF reader with available backends."""
        self.backend = self._select_backend()
        logging.info(f"PDF Reader initialized with backend: {self.backend}")
    
    def _select_backend(self) -> str:
        """Select the best available PDF reading backend."""
        if PYMUPDF_AVAILABLE:
            return "pymupdf"
        elif PDF_AVAILABLE:
            return "pypdf2"
        else:
            raise RuntimeError(
                "No PDF reading library available. Please install PyMuPDF or PyPDF2:\n"
                "pip install PyMuPDF\n"
                "or\n"
                "pip install PyPDF2"
            )
    
    def read_pdf(self, file_path: str) -> str:
        """
        Read PDF content using the selected backend.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content from the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            if self.backend == "pymupdf":
                return self._read_with_pymupdf(file_path)
            elif self.backend == "pypdf2":
                return self._read_with_pypdf2(file_path)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except Exception as e:
            logging.error(f"Error reading PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to read PDF: {str(e)}")
    
    def read_pdf_pages(self, file_path: str) -> Dict[int, List[str]]:
        """
        Read PDF and return pages as lists of lines.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page number (1-indexed) to list of lines
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            if self.backend == "pymupdf":
                return self._read_pages_with_pymupdf(file_path)
            elif self.backend == "pypdf2":
                return self._read_pages_with_pypdf2(file_path)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except Exception as e:
            logging.error(f"Error reading PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to read PDF: {str(e)}")
    
    def extract_topic_block(self, file_path: str, block: TopicBlock) -> str:
        """
        Extract text for a specific topic block from a PDF.
        
        Args:
            file_path: Path to the PDF file
            block: TopicBlock defining the range to extract
            
        Returns:
            Extracted text for the topic block
        """
        pages = self.read_pdf_pages(file_path)
        extracted_lines = []
        
        for page_num in range(block.start_page, block.end_page + 1):
            if page_num not in pages:
                continue
            
            page_lines = pages[page_num]
            
            # Determine start and end line indices for this page
            if page_num == block.start_page:
                start_idx = max(0, block.start_line - 1)  # Convert to 0-indexed
            else:
                start_idx = 0
            
            if page_num == block.end_page:
                if block.end_line is not None:
                    end_idx = min(len(page_lines), block.end_line)
                else:
                    end_idx = len(page_lines)
            else:
                end_idx = len(page_lines)
            
            # Extract lines for this page
            page_extracted = page_lines[start_idx:end_idx]
            if page_extracted:
                extracted_lines.append(f"--- Page {page_num} ---")
                extracted_lines.extend(page_extracted)
        
        return "\n".join(extracted_lines)
    
    def _read_with_pymupdf(self, file_path: str) -> str:
        """Read PDF using PyMuPDF (fitz)."""
        doc = fitz.open(file_path)
        text_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        return "\n\n".join(text_content)
    
    def _read_with_pypdf2(self, file_path: str) -> str:
        """Read PDF using PyPDF2."""
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_content)
    
    def _read_pages_with_pymupdf(self, file_path: str) -> Dict[int, List[str]]:
        """Read PDF pages as line lists using PyMuPDF."""
        doc = fitz.open(file_path)
        pages = {}
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                # Split into lines and filter empty lines
                lines = [line for line in text.split('\n') if line.strip()]
                pages[page_num + 1] = lines  # 1-indexed
        
        doc.close()
        return pages
    
    def _read_pages_with_pypdf2(self, file_path: str) -> Dict[int, List[str]]:
        """Read PDF pages as line lists using PyPDF2."""
        pages = {}
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Split into lines and filter empty lines
                    lines = [line for line in text.split('\n') if line.strip()]
                    pages[page_num + 1] = lines  # 1-indexed
        
        return pages


class TopicManager:
    """Manages topic blocks for all PDFs."""
    
    def __init__(self, inputs_dir: str):
        """Initialize topic manager and load topic definitions."""
        self.inputs_dir = inputs_dir
        self.topic_blocks: Dict[str, List[TopicBlock]] = {}
        self.pdf_id_to_filename: Dict[str, str] = {}
        self.load_topics()
    
    def load_topics(self):
        """Load topic definitions from text files."""
        # Lucas 2015
        lucas_txt = os.path.join(self.inputs_dir, "..", "Lucas et al. - 2015 - The future of lupin as a protein crop in Europe.txt")
        lucas_pdf = "Lucas et al. - 2015 - The future of lupin as a protein crop in Europe.pdf"
        
        if os.path.exists(lucas_txt):
            try:
                blocks = TopicParser.parse_lucas_file(lucas_txt, lucas_pdf)
                self.topic_blocks["lucas2015"] = blocks
                self.pdf_id_to_filename["lucas2015"] = lucas_pdf
                logger.info(f"Loaded {len(blocks)} topic blocks for lucas2015")
            except Exception as e:
                logger.error(f"Error loading Lucas topics: {e}")
        
        # Watson 2017
        watson_txt = os.path.join(self.inputs_dir, "..", "watson2017.txt")
        watson_pdf = "watson2017.pdf"
        
        if os.path.exists(watson_txt):
            try:
                blocks = TopicParser.parse_watson_file(watson_txt, watson_pdf)
                self.topic_blocks["watson2017"] = blocks
                self.pdf_id_to_filename["watson2017"] = watson_pdf
                logger.info(f"Loaded {len(blocks)} topic blocks for watson2017")
            except Exception as e:
                logger.error(f"Error loading Watson topics: {e}")
    
    def get_topic_by_descriptor(self, descriptor: str) -> Optional[TopicBlock]:
        """Get a topic block by its descriptor."""
        for pdf_id, blocks in self.topic_blocks.items():
            for block in blocks:
                if block.descriptor == descriptor:
                    return block
        return None
    
    def get_topics_for_pdf(self, pdf_id: str) -> List[TopicBlock]:
        """Get all topic blocks for a given PDF ID."""
        return self.topic_blocks.get(pdf_id, [])
    
    def list_all_descriptors(self) -> List[str]:
        """List all available topic descriptors."""
        descriptors = []
        for blocks in self.topic_blocks.values():
            descriptors.extend([block.descriptor for block in blocks])
        return descriptors


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('PDF Resource Server')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
inputs_dir = os.path.join(script_dir, "Inputs")

logger.info(f"Project root: {project_root}")
logger.info(f"Inputs directory: {inputs_dir}")

# Initialize topic manager (doesn't require PDF libraries)
topic_manager = TopicManager(inputs_dir)

# Initialize PDF reader (only if PDF libraries are available)
pdf_reader = None
if PYMUPDF_AVAILABLE or PDF_AVAILABLE:
    try:
        pdf_reader = PDFReader()
    except RuntimeError as e:
        logger.error(f"Failed to initialize PDF reader: {e}")
        pdf_reader = None


def get_available_pdfs() -> List[str]:
    """Get list of available PDF files in the inputs directory."""
    if not os.path.exists(inputs_dir):
        logger.warning(f"Inputs directory not found: {inputs_dir}")
        return []
    
    pdf_files = []
    for file in os.listdir(inputs_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(file)
    
    logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    return pdf_files


# Application lifespan management
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle and context."""
    logger.info("Initializing PDF Resource Server...")
    
    # Get available PDFs
    available_pdfs = get_available_pdfs()
    
    # Build topic blocks dictionary
    topic_blocks_dict = {}
    for pdf_id in topic_manager.topic_blocks.keys():
        topic_blocks_dict[pdf_id] = topic_manager.get_topics_for_pdf(pdf_id)
    
    context = AppContext(
        pdf_directory=inputs_dir,
        available_pdfs=available_pdfs,
        topic_blocks=topic_blocks_dict
    )
    
    try:
        logger.info(f"PDF Resource Server initialized with {len(available_pdfs)} PDFs")
        logger.info(f"Total topic blocks: {sum(len(blocks) for blocks in topic_blocks_dict.values())}")
        yield context
    finally:
        logger.info("PDF Resource Server shutting down...")


# Initialize FastMCP server (only if MCP is available)
if MCP_AVAILABLE:
    mcp = FastMCP(
        "PDF Resource Server",
        transport="streamable-http",
        lifespan=app_lifespan,
        logging_level=logging.INFO,
        metadata={
            "serverInfo": {
                "name": "PDF Resource Server",
                "version": "2.0.0",
                "description": "MCP server for exposing PDF documents as topic-based resources"
            },
            "capabilities": {
                "resources": {
                    "subscribe": False,
                    "listChanged": False
                }
            }
        }
    )
else:
    mcp = None


if MCP_AVAILABLE:
    @mcp.resource("pdf://topics/{descriptor}")
    async def get_topic_by_descriptor(descriptor: str) -> str:
        """
        Get a specific topic block by its descriptor.
        
        Args:
            descriptor: Unique topic descriptor (e.g., "lucas2015_europe_context")
            
        Returns:
            Extracted text content for the topic block
        """
        logger.info(f"Reading topic block: {descriptor}")
        
        if pdf_reader is None:
            raise RuntimeError("PDF reader not available. Please install PyMuPDF or PyPDF2.")
        
        block = topic_manager.get_topic_by_descriptor(descriptor)
        if not block:
            raise ValueError(f"Topic descriptor not found: {descriptor}")
        
        pdf_path = os.path.join(inputs_dir, block.pdf_filename)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            content = pdf_reader.extract_topic_block(pdf_path, block)
            
            # Add metadata header
            metadata_header = f"""Topic Block: {block.descriptor}
PDF: {block.pdf_id} ({block.pdf_filename})
Title: {block.title}
Pages: {block.start_page} (line {block.start_line}) to {block.end_page} (line {block.end_line or 'end'})
Scope: {block.scope}
Extracted on: {datetime.now().isoformat()}
Content Length: {len(content)} characters

--- Topic Content ---
"""
            
            full_content = metadata_header + content
            logger.info(f"Successfully extracted topic {descriptor}, content length: {len(content)} characters")
            
            return full_content
            
        except Exception as e:
            logger.error(f"Error extracting topic {descriptor}: {str(e)}")
            raise


    @mcp.resource("pdf://topics/list/{pdf_id}")
    async def list_topics_for_pdf(pdf_id: str) -> str:
        """
        List all available topic blocks for a given PDF.
        
        Args:
            pdf_id: PDF identifier (e.g., "lucas2015", "watson2017")
            
        Returns:
            JSON-formatted list of topic blocks with metadata
        """
        logger.info(f"Listing topics for PDF: {pdf_id}")
        
        topics = topic_manager.get_topics_for_pdf(pdf_id)
        if not topics:
            raise ValueError(f"No topics found for PDF: {pdf_id}")
        
        pdf_filename = topic_manager.pdf_id_to_filename.get(pdf_id, "unknown")
        
        topic_info = []
        for block in topics:
            topic_info.append({
                "descriptor": block.descriptor,
                "title": block.title,
                "start_page": block.start_page,
                "start_line": block.start_line,
                "end_page": block.end_page,
                "end_line": block.end_line,
                "scope": block.scope,
                "resource_uri": f"pdf://topics/{block.descriptor}"
            })
        
        result = {
            "pdf_id": pdf_id,
            "pdf_filename": pdf_filename,
            "topics": topic_info,
            "total_count": len(topics),
            "generated_at": datetime.now().isoformat()
        }
        
        return json.dumps(result, indent=2)


    @mcp.resource("pdf://topics/list")
    async def list_all_topics() -> str:
        """
        List all available topic blocks across all PDFs.
        
        Returns:
            JSON-formatted list of all topic blocks organized by PDF
        """
        logger.info("Listing all topics")
        
        result = {
            "pdfs": {},
            "total_topics": 0,
            "generated_at": datetime.now().isoformat()
        }
        
        for pdf_id in topic_manager.topic_blocks.keys():
            topics = topic_manager.get_topics_for_pdf(pdf_id)
            pdf_filename = topic_manager.pdf_id_to_filename.get(pdf_id, "unknown")
            
            topic_list = []
            for block in topics:
                topic_list.append({
                    "descriptor": block.descriptor,
                    "title": block.title,
                    "start_page": block.start_page,
                    "start_line": block.start_line,
                    "end_page": block.end_page,
                    "end_line": block.end_line,
                    "scope": block.scope,
                    "resource_uri": f"pdf://topics/{block.descriptor}"
                })
            
            result["pdfs"][pdf_id] = {
                "pdf_filename": pdf_filename,
                "topics": topic_list,
                "topic_count": len(topics)
            }
            result["total_topics"] += len(topics)
        
        return json.dumps(result, indent=2)


    @mcp.tool()
    async def read_pdf_topic(descriptor: str) -> str:
        """
        Read a PDF topic block by its descriptor.
        
        Available topic descriptors can be listed using list_pdf_topics().
        
        Args:
            descriptor: Unique topic descriptor (e.g., "lucas2015_europe_context")
            
        Returns:
            The content of the topic block
        """
        logger.info(f"Tool called to read PDF topic: {descriptor}")
        return await get_topic_by_descriptor(descriptor)


    @mcp.tool()
    async def read_multiple_pdf_topics(descriptors: List[str]) -> str:
        """
        Read multiple PDF topic blocks in a single call.
        
        This tool efficiently retrieves multiple topics from PDFs in one operation.
        Use this when you need to access several related topics or compare information
        across multiple sections of a PDF or multiple PDFs.
        
        Available topic descriptors can be listed using list_pdf_topics().
        
        Args:
            descriptors: List of unique topic descriptors to read
                        (e.g., ["lucas2015_europe_context", "lucas2015_yield_factors", "watson2017_nutrition"])
            
        Returns:
            JSON-formatted dictionary containing all requested topics with their content and metadata.
            Each topic includes: descriptor, title, pdf_id, content, pages, scope, and status.
            Topics that fail to load will have a status of "error" with an error message.
        """
        logger.info(f"Tool called to read multiple PDF topics: {len(descriptors)} topics")
        
        if pdf_reader is None:
            raise RuntimeError("PDF reader not available. Please install PyMuPDF or PyPDF2.")
        
        results = {
            "requested_count": len(descriptors),
            "successful_count": 0,
            "failed_count": 0,
            "topics": [],
            "generated_at": datetime.now().isoformat()
        }
        
        for descriptor in descriptors:
            try:
                block = topic_manager.get_topic_by_descriptor(descriptor)
                if not block:
                    results["topics"].append({
                        "descriptor": descriptor,
                        "status": "error",
                        "error": f"Topic descriptor not found: {descriptor}"
                    })
                    results["failed_count"] += 1
                    continue
                
                pdf_path = os.path.join(inputs_dir, block.pdf_filename)
                if not os.path.exists(pdf_path):
                    results["topics"].append({
                        "descriptor": descriptor,
                        "status": "error",
                        "error": f"PDF file not found: {pdf_path}"
                    })
                    results["failed_count"] += 1
                    continue
                
                # Extract topic content
                content = pdf_reader.extract_topic_block(pdf_path, block)
                
                # Build topic result
                topic_result = {
                    "descriptor": block.descriptor,
                    "title": block.title,
                    "pdf_id": block.pdf_id,
                    "pdf_filename": block.pdf_filename,
                    "pages": {
                        "start_page": block.start_page,
                        "start_line": block.start_line,
                        "end_page": block.end_page,
                        "end_line": block.end_line
                    },
                    "scope": block.scope,
                    "content": content,
                    "content_length": len(content),
                    "status": "success"
                }
                
                results["topics"].append(topic_result)
                results["successful_count"] += 1
                
            except Exception as e:
                logger.error(f"Error reading topic {descriptor}: {str(e)}")
                results["topics"].append({
                    "descriptor": descriptor,
                    "status": "error",
                    "error": str(e)
                })
                results["failed_count"] += 1
        
        return json.dumps(results, indent=2, ensure_ascii=False)


    @mcp.tool()
    async def list_pdf_topics(pdf_id: Optional[str] = None) -> str:
        """
        List available topic blocks for PDFs.
        
        Args:
            pdf_id: Optional PDF identifier (e.g., "lucas2015", "watson2017").
                    If not provided, lists all topics across all PDFs.
            
        Returns:
            JSON-formatted list of topic blocks
        """
        logger.info(f"Tool called to list PDF topics for: {pdf_id or 'all'}")
        
        if pdf_id:
            return await list_topics_for_pdf(pdf_id)
        else:
            return await list_all_topics()


if __name__ == "__main__":
    logger.info("Starting PDF Resource Server...")
    
    # Test topic loading
    logger.info(f"Loaded topics for {len(topic_manager.topic_blocks)} PDFs")
    for pdf_id, blocks in topic_manager.topic_blocks.items():
        logger.info(f"  {pdf_id}: {len(blocks)} topic blocks")
        for block in blocks[:3]:  # Show first 3
            logger.info(f"    - {block.descriptor}: {block.title}")
    
    # Test PDF reading functionality
    available_pdfs = get_available_pdfs()
    if available_pdfs:
        logger.info(f"Available PDFs for testing: {available_pdfs}")
        
        # Test reading a topic block if available
        if topic_manager.topic_blocks:
            first_pdf_id = list(topic_manager.topic_blocks.keys())[0]
            first_block = topic_manager.topic_blocks[first_pdf_id][0]
            pdf_path = os.path.join(inputs_dir, first_block.pdf_filename)
            
            try:
                test_content = pdf_reader.extract_topic_block(pdf_path, first_block)
                logger.info(f"Test extraction successful for {first_block.descriptor}: {len(test_content)} characters")
            except Exception as e:
                logger.error(f"Test extraction failed for {first_block.descriptor}: {e}")
    else:
        logger.warning("No PDF files found for testing")
    
    # Run the MCP server
    if mcp is not None:
        logger.info("Starting MCP server with streamable HTTP transport...")
        mcp.run()
    else:
        logger.error("MCP not available. Cannot start server.")
        sys.exit(1)
