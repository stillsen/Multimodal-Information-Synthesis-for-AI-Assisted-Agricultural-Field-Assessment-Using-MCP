"""
Author: Stefan Stiller
Affiliation: Leibniz-Centre for Agricultural Landscape Research (ZALF) e.V.
ORCID: https://orcid.org/0009-0004-7468-1678
License: GPL-3.0

MCP CSV Resource Server

This server exposes CSV data as resources that can be accessed by MCP clients.
It provides access to soil lab results and soil moisture data, with the ability
to filter by Patch_ID (default: 59).

The server handles German CSV files with proper encoding and provides
structured access to agricultural data.
"""

import os
import sys
import logging
import asyncio
import csv
import json
import math
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

# MCP imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import Resource, TextResourceContents


@dataclass
class AppContext:
    """Application context for the CSV resource server."""
    csv_directory: str
    available_csvs: List[str]
    default_patch_id: int = 59


class CSVReader:
    """Handles CSV reading operations with proper encoding and German locale support."""
    
    def __init__(self):
        """Initialize CSV reader."""
        self.encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        logging.info("CSV Reader initialized")
    
    def _german_to_float(self, value: str) -> Optional[float]:
        """
        Convert numeric string to float, handling both international and German formats.
        
        Files are expected to be in international format (dot as decimal separator).
        This function first tries international format, then falls back to German format
        if conversion fails (for backward compatibility).
        
        Args:
            value: String value that may be in international format (6.0) or German format (6,0)
            
        Returns:
            Float value or None if conversion fails
        """
        if not value or not value.strip():
            return None
        
        cleaned = value.strip()
        
        # First, try international format (dot as decimal separator)
        # This is the expected format for the converted CSV files
        try:
            # If value contains a dot, it's likely international format
            if '.' in cleaned:
                # Remove thousand separator commas (they appear before the dot)
                # Example: "1,164,657.50" -> "1164657.50"
                if ',' in cleaned:
                    dot_pos = cleaned.find('.')
                    # Remove commas only before the dot (thousand separators)
                    before_dot = cleaned[:dot_pos].replace(',', '').replace(' ', '')
                    after_dot = cleaned[dot_pos:]
                    cleaned = before_dot + after_dot
                return float(cleaned)
            else:
                # No dot - could be integer in international format
                # Check if commas are thousand separators (multiple commas = thousand separators)
                comma_count = cleaned.count(',')
                if comma_count > 1:
                    # Multiple commas = thousand separators, remove them all
                    cleaned_int = cleaned.replace(',', '').replace(' ', '')
                    return float(cleaned_int)
                elif comma_count == 0:
                    # No commas, direct conversion
                    return float(cleaned.replace(' ', ''))
                # Single comma with no dot - might be German format, fall through
        except ValueError:
            pass
        
        # If international format failed, try German format (comma as decimal separator)
        # This is for backward compatibility with old files
        try:
            comma_count = cleaned.count(',')
            
            if comma_count == 0:
                # No comma, try direct conversion
                return float(cleaned.replace(' ', ''))
            elif comma_count == 1:
                # Single comma - treat as decimal separator (German format)
                return float(cleaned.replace(',', '.'))
            else:
                # Multiple commas - treat all but last as thousand separators
                # Last comma is decimal separator (German format)
                parts = cleaned.rsplit(',', 1)
                integer_part = parts[0].replace(',', '').replace(' ', '')
                decimal_part = parts[1] if len(parts) > 1 else '0'
                return float(integer_part + '.' + decimal_part)
        except (ValueError, AttributeError):
            return None
    
    def _calculate_mean_se(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate mean and standard error for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with 'mean' and 'standard_error' keys
        """
        if not values:
            return {"mean": None, "standard_error": None}
        
        n = len(values)
        mean = sum(values) / n
        
        if n <= 1:
            return {"mean": mean, "standard_error": 0.0}
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_dev = math.sqrt(variance)
        
        # Calculate standard error: SE = SD / sqrt(n)
        standard_error = std_dev / math.sqrt(n)
        
        return {"mean": mean, "standard_error": standard_error}
    
    def read_csv(self, file_path: str) -> List[Dict[str, str]]:
        """
        Read CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries representing CSV rows
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the CSV cannot be read or is corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        for encoding in self.encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
                    # Try to detect delimiter
                    sample = csvfile.read(1024)
                    csvfile.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    
                    reader = csv.DictReader(csvfile, delimiter=delimiter)
                    data = list(reader)
                    
                    logging.info(f"Successfully read CSV {file_path} with encoding {encoding}")
                    return data
                    
            except (UnicodeDecodeError, csv.Error) as e:
                logging.debug(f"Failed to read {file_path} with encoding {encoding}: {e}")
                continue
        
        raise ValueError(f"Could not read CSV file {file_path} with any supported encoding")
    
    def filter_by_patch_id(self, data: List[Dict[str, str]], patch_id: int) -> List[Dict[str, str]]:
        """
        Filter CSV data by Patch_ID.
        
        Args:
            data: List of CSV row dictionaries
            patch_id: Patch ID to filter by
            
        Returns:
            Filtered list of rows matching the Patch_ID
        """
        filtered_data = []
        for row in data:
            try:
                # Handle different possible column names for Patch_ID
                patch_id_cols = ['Patch_ID', 'patch_id', 'PatchID', 'patchid']
                row_patch_id = None
                
                for col in patch_id_cols:
                    if col in row and row[col].strip():
                        row_patch_id = int(row[col].strip())
                        break
                
                if row_patch_id == patch_id:
                    filtered_data.append(row)
                    
            except (ValueError, KeyError) as e:
                logging.debug(f"Error processing row for Patch_ID {patch_id}: {e}")
                continue
        
        return filtered_data
    
    def extract_soil_lab_data(self, data: List[Dict[str, str]], patch_id: int) -> Dict[str, Any]:
        """
        Extract relevant soil lab data for a specific Patch_ID, filtered by depth 0-30.
        Calculates mean and standard error for key parameters.
        
        Args:
            data: List of CSV row dictionaries
            patch_id: Patch ID to extract data for
            
        Returns:
            Dictionary with extracted soil lab data including mean and SE
        """
        # Filter by Patch_ID
        filtered_by_patch = self.filter_by_patch_id(data, patch_id)
        
        if not filtered_by_patch:
            return {
                "patch_id": patch_id,
                "status": "not_found",
                "message": f"No soil lab data found for Patch_ID {patch_id}",
                "available_patch_ids": self._get_available_patch_ids(data)
            }
        
        # Filter by depth = '0-30'
        filtered_data = []
        for row in filtered_by_patch:
            depth = row.get('Depth (cm)', '').strip()
            if depth == '0-30':
                filtered_data.append(row)
        
        if not filtered_data:
            return {
                "patch_id": patch_id,
                "status": "not_found",
                "message": f"No soil lab data found for Patch_ID {patch_id} with depth 0-30",
                "available_patch_ids": self._get_available_patch_ids(data)
            }
        
        # Collect values for each parameter
        parameter_values = {
            'ph_value': [],
            'phosphorus': [],
            'potassium': [],
            'magnesium': [],
            'ammonium_nitrogen': [],
            'nitrate_nitrogen': []
        }
        
        # Helper function to find column name (handles newlines in column names)
        def get_column_value(row: Dict[str, str], possible_names: List[str]) -> Optional[str]:
            """Try multiple possible column name variations."""
            for name in possible_names:
                if name in row:
                    return row[name]
            return None
        
        for row in filtered_data:
            # pH- Wert
            ph_val = self._german_to_float(row.get('pH- Wert', ''))
            if ph_val is not None:
                parameter_values['ph_value'].append(ph_val)
            
            # P (Phosphorus)
            p_val = self._german_to_float(row.get('P', ''))
            if p_val is not None:
                parameter_values['phosphorus'].append(p_val)
            
            # K (Potassium)
            k_val = self._german_to_float(row.get('K', ''))
            if k_val is not None:
                parameter_values['potassium'].append(k_val)
            
            # Mg (Magnesium)
            mg_val = self._german_to_float(row.get('Mg', ''))
            if mg_val is not None:
                parameter_values['magnesium'].append(mg_val)
            
            # Ammonium nitrogen - try different column name variations
            nh4_possible = [
                'mg NH4-NCaCl2/ \n 100g Soil (moist)',
                'mg NH4-NCaCl2/\n 100g Soil (moist)',
                'mg NH4-NCaCl2/ 100g Soil (moist)'
            ]
            nh4_str = get_column_value(row, nh4_possible) or ''
            nh4_val = self._german_to_float(nh4_str)
            if nh4_val is not None:
                parameter_values['ammonium_nitrogen'].append(nh4_val)
            
            # Nitrate nitrogen - try different column name variations
            no3_possible = [
                'mg NO3-NCaCl2/ \n 100g Soil (moist)',
                'mg NO3-NCaCl2/\n 100g Soil (moist)',
                'mg NO3-NCaCl2/ 100g Soil (moist)'
            ]
            no3_str = get_column_value(row, no3_possible) or ''
            no3_val = self._german_to_float(no3_str)
            if no3_val is not None:
                parameter_values['nitrate_nitrogen'].append(no3_val)
        
        # Calculate mean and SE for each parameter
        result = {
            "patch_id": patch_id,
            "depth_cm": "0-30",
            "status": "found",
            "sample_count": len(filtered_data),
            "parameters": {}
        }
        
        for param_name, values in parameter_values.items():
            stats = self._calculate_mean_se(values)
            result["parameters"][param_name] = {
                "mean": round(stats["mean"], 3) if stats["mean"] is not None else None,
                "standard_error": round(stats["standard_error"], 3) if stats["standard_error"] is not None else None,
                "n": len(values)
            }
        
        return result
    
    def extract_soil_moisture_data(self, data: List[Dict[str, str]], patch_id: int) -> Dict[str, Any]:
        """
        Extract soil moisture data for a specific Patch_ID.
        Returns paired entries with date and average moisture value.
        
        Args:
            data: List of CSV row dictionaries
            patch_id: Patch ID to extract data for
            
        Returns:
            Dictionary with extracted soil moisture data as paired entries
        """
        filtered_data = self.filter_by_patch_id(data, patch_id)
        
        if not filtered_data:
            return {
                "patch_id": patch_id,
                "status": "not_found",
                "message": f"No soil moisture data found for Patch_ID {patch_id}",
                "available_patch_ids": self._get_available_patch_ids(data)
            }
        
        # Collect paired entries (date, avg)
        moisture_entries = []
        
        for row in filtered_data:
            date = row.get('Date', '').strip()
            avg_moisture_str = row.get('AVG', '').strip()
            
            # Skip if date or average is missing
            if not date or not avg_moisture_str:
                continue
            
            # Convert average to numeric
            avg_moisture = self._german_to_float(avg_moisture_str)
            if avg_moisture is None:
                continue
            
            moisture_entries.append({
                "date": date,
                "avg": avg_moisture
            })
        
        if not moisture_entries:
            return {
                "patch_id": patch_id,
                "status": "not_found",
                "message": f"No valid soil moisture data found for Patch_ID {patch_id}",
                "available_patch_ids": self._get_available_patch_ids(data)
            }
        
        # Sort by date (assuming DD.MM.YYYY format)
        try:
            moisture_entries.sort(key=lambda x: datetime.strptime(x["date"], "%d.%m.%Y"))
        except (ValueError, KeyError):
            # If date parsing fails, keep original order
            logging.debug("Could not sort moisture entries by date")
        
        result = {
            "patch_id": patch_id,
            "status": "found",
            "data_count": len(moisture_entries),
            "moisture_data": moisture_entries
        }
        
        return result
    
    def _get_available_patch_ids(self, data: List[Dict[str, str]]) -> List[int]:
        """Get list of available Patch_IDs in the data."""
        patch_ids = set()
        patch_id_cols = ['Patch_ID', 'patch_id', 'PatchID', 'patchid']
        
        for row in data:
            for col in patch_id_cols:
                if col in row and row[col].strip():
                    try:
                        patch_ids.add(int(row[col].strip()))
                    except ValueError:
                        continue
                    break
        
        return sorted(list(patch_ids))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('CSV Resource Server')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
inputs_dir = os.path.join(script_dir, "Inputs")

logger.info(f"Project root: {project_root}")
logger.info(f"Inputs directory: {inputs_dir}")

# Initialize CSV reader
csv_reader = CSVReader()


def get_available_csvs() -> List[str]:
    """Get list of available CSV files in the inputs directory."""
    if not os.path.exists(inputs_dir):
        logger.warning(f"Inputs directory not found: {inputs_dir}")
        return []
    
    csv_files = []
    for file in os.listdir(inputs_dir):
        if file.lower().endswith('.csv'):
            csv_files.append(file)
    
    logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")
    return csv_files


# Application lifespan management
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle and context."""
    logger.info("Initializing CSV Resource Server...")
    
    # Get available CSV files
    available_csvs = get_available_csvs()
    
    context = AppContext(
        csv_directory=inputs_dir,
        available_csvs=available_csvs,
        default_patch_id=59
    )
    
    try:
        logger.info(f"CSV Resource Server initialized with {len(available_csvs)} CSV files")
        yield context
    finally:
        logger.info("CSV Resource Server shutting down...")


# Initialize FastMCP server
mcp = FastMCP(
    "CSV Resource Server",
    transport="streamable-http",
    lifespan=app_lifespan,
    logging_level=logging.INFO,
    metadata={
        "serverInfo": {
            "name": "CSV Resource Server",
            "version": "1.0.0",
            "description": "MCP server for exposing CSV agricultural data as resources"
        },
        "capabilities": {
            "resources": {
                "subscribe": False,
                "listChanged": False
            }
        }
    }
)


@mcp.resource("csv://soil-lab-results/{patch_id}")
async def get_soil_lab_results(patch_id: str) -> str:
    """
    Get soil lab results for a specific Patch_ID.
    
    Args:
        patch_id: Patch ID to get data for (default: 59)
        
    Returns:
        JSON-formatted soil lab results data
    """
    logger.info(f"Getting soil lab results for Patch_ID: {patch_id}")
    
    try:
        patch_id_int = int(patch_id)
    except ValueError:
        return json.dumps({
            "error": "Invalid Patch_ID",
            "message": f"Patch_ID must be an integer, got: {patch_id}"
        }, indent=2)
    
    try:
        # Read soil lab results CSV
        csv_path = os.path.join(inputs_dir, "Soil_Lab_Results.csv")
        data = csv_reader.read_csv(csv_path)
        
        # Extract relevant data
        result = csv_reader.extract_soil_lab_data(data, patch_id_int)
        
        # Add metadata
        result["metadata"] = {
            "source_file": "Soil_Lab_Results.csv",
            "extracted_at": datetime.now().isoformat(),
            "total_records_in_file": len(data)
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error getting soil lab results: {str(e)}")
        return json.dumps({
            "error": "Failed to get soil lab results",
            "message": str(e),
            "patch_id": patch_id_int
        }, indent=2)


@mcp.resource("csv://soil-moisture/{patch_id}")
async def get_soil_moisture(patch_id: str) -> str:
    """
    Get soil moisture data for a specific Patch_ID.
    
    Args:
        patch_id: Patch ID to get data for (default: 59)
        
    Returns:
        JSON-formatted soil moisture data
    """
    logger.info(f"Getting soil moisture data for Patch_ID: {patch_id}")
    
    try:
        patch_id_int = int(patch_id)
    except ValueError:
        return json.dumps({
            "error": "Invalid Patch_ID",
            "message": f"Patch_ID must be an integer, got: {patch_id}"
        }, indent=2)
    
    try:
        # Read soil moisture CSV
        csv_path = os.path.join(inputs_dir, "Soil_Moisture.csv")
        data = csv_reader.read_csv(csv_path)
        
        # Extract relevant data
        result = csv_reader.extract_soil_moisture_data(data, patch_id_int)
        
        # Add metadata
        result["metadata"] = {
            "source_file": "Soil_Moisture.csv",
            "extracted_at": datetime.now().isoformat(),
            "total_records_in_file": len(data)
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error getting soil moisture data: {str(e)}")
        return json.dumps({
            "error": "Failed to get soil moisture data",
            "message": str(e),
            "patch_id": patch_id_int
        }, indent=2)


@mcp.resource("csv://soil-lab-results/patch-59")
async def get_soil_lab_results_patch_59() -> str:
    """
    Get soil lab results for Patch_ID 59 (default field).
    
    Returns:
        JSON-formatted soil lab results data for Patch_ID 59
    """
    return await get_soil_lab_results("59")


@mcp.resource("csv://soil-moisture/patch-59")
async def get_soil_moisture_patch_59() -> str:
    """
    Get soil moisture data for Patch_ID 59 (default field).
    
    Returns:
        JSON-formatted soil moisture data for Patch_ID 59
    """
    return await get_soil_moisture("59")


@mcp.resource("csv://available-patch-ids")
async def get_available_patch_ids() -> str:
    """
    Get list of available Patch_IDs in both CSV files.
    
    Returns:
        JSON-formatted list of available Patch_IDs
    """
    logger.info("Getting available Patch_IDs from CSV files")
    
    try:
        result = {
            "available_patch_ids": {},
            "metadata": {
                "generated_at": datetime.now().isoformat()
            }
        }
        
        # Check soil lab results
        try:
            lab_csv_path = os.path.join(inputs_dir, "Soil_Lab_Results.csv")
            lab_data = csv_reader.read_csv(lab_csv_path)
            result["available_patch_ids"]["soil_lab_results"] = csv_reader._get_available_patch_ids(lab_data)
        except Exception as e:
            logger.warning(f"Could not read soil lab results: {e}")
            result["available_patch_ids"]["soil_lab_results"] = []
        
        # Check soil moisture
        try:
            moisture_csv_path = os.path.join(inputs_dir, "Soil_Moisture.csv")
            moisture_data = csv_reader.read_csv(moisture_csv_path)
            result["available_patch_ids"]["soil_moisture"] = csv_reader._get_available_patch_ids(moisture_data)
        except Exception as e:
            logger.warning(f"Could not read soil moisture: {e}")
            result["available_patch_ids"]["soil_moisture"] = []
        
        # Get intersection (Patch_IDs available in both files)
        lab_ids = set(result["available_patch_ids"]["soil_lab_results"])
        moisture_ids = set(result["available_patch_ids"]["soil_moisture"])
        result["available_patch_ids"]["both_files"] = sorted(list(lab_ids.intersection(moisture_ids)))
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting available Patch_IDs: {str(e)}")
        return json.dumps({
            "error": "Failed to get available Patch_IDs",
            "message": str(e)
        }, indent=2)


@mcp.resource("csv://summary/patch-59")
async def get_patch_59_summary() -> str:
    """
    Get a comprehensive summary of all data for Patch_ID 59.
    
    Returns:
        JSON-formatted summary of soil lab results and moisture data for Patch_ID 59
    """
    logger.info("Getting comprehensive summary for Patch_ID 59")
    
    try:
        # Get soil lab results
        lab_results = await get_soil_lab_results("59")
        lab_data = json.loads(lab_results)
        
        # Get soil moisture data
        moisture_data = await get_soil_moisture("59")
        moisture_data = json.loads(moisture_data)
        
        # Create summary
        summary = {
            "patch_id": 59,
            "summary_generated_at": datetime.now().isoformat(),
            "soil_lab_results": lab_data,
            "soil_moisture": moisture_data,
            "key_metrics": {
                "has_lab_data": lab_data.get("status") == "found",
                "has_moisture_data": moisture_data.get("status") == "found",
                "moisture_measurement_count": moisture_data.get("data_count", 0) if moisture_data.get("status") == "found" else 0
            }
        }
        
        # Extract key soil parameters if available
        if lab_data.get("status") == "found" and lab_data.get("parameters"):
            summary["key_metrics"]["soil_parameters"] = lab_data["parameters"]
        
        return json.dumps(summary, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error getting Patch_ID 59 summary: {str(e)}")
        return json.dumps({
            "error": "Failed to get Patch_ID 59 summary",
            "message": str(e)
        }, indent=2)


@mcp.tool()
async def get_soil_summary_patch_59() -> str:
    """
    Retrieve a comprehensive summary of soil lab and moisture data for Patch 59.
    
    Patch 59 is the primary/default experimental field located at approximately 
    52°27′07.5″N, 14°09′42.7″E. Use this tool when field data is requested for 
    this location or when no specific patch ID is mentioned.
    
    Returns:
        JSON-formatted summary including:
        - Soil lab results (pH, P, K, Mg, NH4, NO3) with mean and standard error for depth 0-30 cm
        - Soil moisture time series data (23 weekly measurements)
        - Key metrics and metadata
    """
    logger.info("Tool called to get soil summary for Patch 59")
    return await get_patch_59_summary()


@mcp.tool()
async def read_csv_resource(uri: str) -> str:
    """
    Read CSV agricultural data resources by URI. Access soil lab results, soil moisture data, 
    summaries, and available patch IDs.
    
    Use this tool when you need to access field data for agricultural analysis. 
    Patch 59 (located at ~52°27′07.5″N, 14°09′42.7″E) is the default/primary field.
    
    Available resources:
    - csv://soil-lab-results/{patch_id} - Get soil lab results (pH, P, K, Mg, NH4, NO3) 
      with mean and standard error for depth 0-30 cm for any patch ID
    - csv://soil-moisture/{patch_id} - Get soil moisture time series data (date and average) 
      for any patch ID
    - csv://soil-lab-results/patch-59 - Get soil lab results for Patch 59 (default field)
    - csv://soil-moisture/patch-59 - Get soil moisture data for Patch 59 (default field)
    - csv://summary/patch-59 - Get comprehensive summary for Patch 59 (recommended for 
      field assessments at coordinates 52°27′07.5″N, 14°09′42.7″E)
    - csv://available-patch-ids - List all available Patch_IDs in the dataset
    
    When coordinates are mentioned without a specific patch ID, use Patch 59 resources 
    (the primary experimental field).
    
    Args:
        uri: The resource URI (e.g., "csv://summary/patch-59", "csv://soil-lab-results/89")
        
    Returns:
        The content of the CSV resource as JSON
    """
    logger.info(f"Tool called to read CSV resource: {uri}")
    
    # Parse the URI to determine which resource handler to use
    if uri == "csv://summary/patch-59":
        return await get_patch_59_summary()
    elif uri == "csv://soil-lab-results/patch-59":
        return await get_soil_lab_results_patch_59()
    elif uri == "csv://soil-moisture/patch-59":
        return await get_soil_moisture_patch_59()
    elif uri == "csv://available-patch-ids":
        return await get_available_patch_ids()
    elif uri.startswith("csv://soil-lab-results/"):
        patch_id = uri.replace("csv://soil-lab-results/", "")
        return await get_soil_lab_results(patch_id)
    elif uri.startswith("csv://soil-moisture/"):
        patch_id = uri.replace("csv://soil-moisture/", "")
        return await get_soil_moisture(patch_id)
    else:
        raise ValueError(f"Unknown CSV resource URI: {uri}")


if __name__ == "__main__":
    logger.info("Starting CSV Resource Server...")
    
    # Test CSV reading functionality
    available_csvs = get_available_csvs()
    if available_csvs:
        logger.info(f"Available CSV files for testing: {available_csvs}")
        
        # Test reading the first available CSV
        test_csv = available_csvs[0]
        test_path = os.path.join(inputs_dir, test_csv)
        
        try:
            test_data = csv_reader.read_csv(test_path)
            logger.info(f"Test read successful for {test_csv}: {len(test_data)} records")
        except Exception as e:
            logger.error(f"Test read failed for {test_csv}: {e}")
    else:
        logger.warning("No CSV files found for testing")
    
    # Run the MCP server
    logger.info("Starting MCP server with streamable HTTP transport...")
    mcp.run()
