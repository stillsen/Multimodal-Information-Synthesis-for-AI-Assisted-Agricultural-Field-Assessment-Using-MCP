"""
Author: Stefan Stiller
Affiliation: Leibniz-Centre for Agricultural Landscape Research (ZALF) e.V.
ORCID: https://orcid.org/0009-0004-7468-1678
License: GPL-3.0

MCP Yield Prediction Server

Exposes an image-based crop yield prediction tool to MCP clients, using a
fine-tuned self-supervised ConvNeXt model for lupine yield estimation.
"""

from typing import Any
import torch
import torch.nn as nn
from mcp.server.fastmcp import FastMCP
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import logging
import base64
import io
import binascii
import asyncio
from concurrent.futures import ThreadPoolExecutor
import signal
from contextlib import contextmanager
import uvicorn
import random

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load your CNN model at startup
class YieldModel:
    @classmethod
    def setup_device_and_workers(cls, use_gpu: bool = False, num_workers: int = 1):
        """Setup device and workers based on configuration."""
        # Ensure deterministic behavior
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            workers = num_workers if num_workers > 1 else 1
        else:
            device = torch.device("cpu")
            workers = 1
        
        logger.info(f'Using device: {device}')
        logger.info(f'Number of workers: {workers}')
        return device, workers

    @classmethod
    async def load(cls, model_path: str):
        try:
            # Ensure deterministic behavior
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Setup device and workers
            device, workers = cls.setup_device_and_workers(use_gpu=False, num_workers=1)
            
            # Initialize model wrapper
            model_wrapper = RGBYieldRegressor_Trainer(
                device=device,
                workers=workers,
                architecture='VICRegConvNext',
                pretrained=False,
                tune_fc_only=False
            )
            model_wrapper.model.SSL_training = False
            model_wrapper.reinitialize_fc_layers()

            # Load the model
            model_loaded = model_wrapper.load_model_if_exists(
                model_dir=os.path.dirname(model_path),
                filename=os.path.basename(model_path)
            )
            
            if not model_loaded:
                logger.error(f"Failed to load model from {model_path}")
                raise RuntimeError("Failed to load model")
                
            return cls(model_wrapper.model, device, workers)
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def __init__(self, model, device=None, workers=None):
        # Ensure deterministic behavior
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.model = model
        if device is None or workers is None:
            self.device, self.workers = self.setup_device_and_workers(use_gpu=False, num_workers=1)
        else:
            self.device = device
            self.workers = workers

    def predict(self, image):
        # Ensure deterministic behavior during inference
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        return self.model(image)  # or however inference is done


@dataclass
class AppContext:
    model: YieldModel

# Configure logging to write to stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger('Yield Server')

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Models and Templates are in the same directory as the script
sys.path.append(project_root)

# Add Templates directory to Python path for relative imports within Templates
templates_dir = os.path.join(project_root, "Templates")
sys.path.append(templates_dir)

# Import the trainer class after adding project root to path
from Templates.RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer

model_path = os.path.join(project_root, "Models", "SSL_VICRegConvNeXt-tiny_Lupine_f1.ckpt")

# image_path = os.path.join(project_root, "Output", "Figures", "Patch_59_Lupine", "extracted_image_1_0.png")

logger.info(f"Model path: {model_path}")

# Global variables
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Initialize model wrapper at startup
logger.info("Loading and initializing model...")
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Ensure deterministic behavior during model loading
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    model = await YieldModel.load(model_path)
    try:
        yield AppContext(model=model)
    finally:
        pass  # Cleanup if needed
logger.info("Model initialized successfully")

mcp = FastMCP(
    "Yield Server",
    transport="streamable-http",
    lifespan=app_lifespan,
    logging_level=logging.DEBUG,
    # metadata={
    #     "serverInfo": {
    #         "name": "Yield Server",
    #         "version": "1.0.0"
    #     },
        # "capabilities": {
        #     "experimental": {},
        #     "prompts": {"listChanged": False},
        #     "resources": {"subscribe": False, "listChanged": False},
        #     "tools": {"listChanged": False}
        # }
    # }
)

@mcp.tool("predict_yield")
async def predict_yield(image_path: str, ctx: Context) -> float:
    """Predict lupine yield from a UAV/drone image using deep learning model.
    
    This tool analyzes UAV (unmanned aerial vehicle) or drone images of lupine fields
    to predict crop yield. Use this tool when you need to:
    - Assess expected yield from UAV/drone imagery
    - Predict lupine field yield based on aerial images
    - Analyze field condition and expected productivity from drone photos
    - Get yield predictions for agricultural field assessment
    - Integrate UAV-based yield prediction with other field data
    
    The model uses a trained deep learning CNN (VICRegConvNext) to analyze RGB images
    and predict yield in tons per hectare.

    Args:
        image_path: Full local file path to the UAV/drone image file (PNG, JPG, etc.)
                   Example: "/path/to/debug_image_patch_59_lupine_fold1_test_index_0004.png"

    Returns:
        float: Predicted yield in tons per hectare (t/ha)
    """
    # Ensure deterministic behavior
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logger.info("=== Starting predict_yield function ===")
    logger.info(f"Received image path: {image_path}")
    
    try:
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image file directly
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                logger.info(f"Successfully read image file, size: {len(image_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to read image file: {str(e)}")
            raise ValueError(f"Could not read image file: {image_path}")
        
        # Process image and predict
        logger.info("Starting image processing and prediction")
        model = ctx.request_context.lifespan_context.model
        
        # Load image from bytes
        try:
            logger.info(f"Attempting to open image from {len(image_data)} bytes")
            image_pil = Image.open(io.BytesIO(image_data))
            logger.info(f"Successfully opened image: {image_pil.format}, size: {image_pil.size}")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # Convert to RGB if needed
        try:
            image_pil = image_pil.convert('RGB')
            logger.info("Successfully converted image to RGB")
        except Exception as e:
            logger.error(f"Failed to convert image to RGB: {str(e)}")
            raise ValueError(f"Failed to convert image to RGB: {str(e)}")
        
        # Preprocess image
        try:
            image_tensor = transform(image_pil)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(model.device)
            logger.info(f"Successfully preprocessed image, tensor shape: {image_tensor.shape}")
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
        
        # Make prediction
        try:
            with torch.no_grad():
                prediction = model.predict(image_tensor)
            yield_value = prediction.item()
            logger.info(f"Raw prediction value from model: {yield_value}")
            return float(yield_value)
        except Exception as e:
            logger.error(f"Failed to make prediction: {str(e)}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in predict_yield: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")
    finally:
        logger.info("=== Completed predict_yield function ===")


if __name__ == "__main__":
    # Ensure deterministic behavior at startup
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Test the model directly
    # test_image_path = r"D:\Projects\PatchCROP\Output\Figures\Patch_59_Lupine\extracted_image_1_0.png"
    # test_image_path = r"D:\Projects\DivAg_AIM_WP2\patchCROP\first_image.png"
    # test_prediction(test_image_path)
    
    # Run the MCP server
    logger.info("Starting MCP server with streamable HTTP transport...")
    mcp.run()
