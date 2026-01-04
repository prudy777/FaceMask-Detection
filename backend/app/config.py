"""
Configuration settings for the Face Mask Detection API.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Model configuration
MODEL_PATH = MODEL_DIR / "best_mask_model.pth"
MODEL_INPUT_SIZE = (224, 224)

# Class labels (user's training order)
# Class labels (user's training order)
CLASS_LABELS = ["Face Mask Worn Correctly", "FaceMask Worn Incorrectly", "No FaceMask"]

# Binary Mode Mapping
# Maps original labels to binary categories
BINARY_MAPPING = {
    "Face Mask Worn Correctly": "With Mask",
    "FaceMask Worn Incorrectly": "Without Mask",
    "No FaceMask": "Without Mask"
}

# Image normalization (ImageNet standard)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# API settings
API_TITLE = "Face Mask Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-powered face mask detection using PyTorch"

# CORS settings (includes Cloud Run domains)
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Cloud Run domains (added at deploy time)
    "https://*.run.app",
]

# Allow all origins in production (set via env var)
import os
if os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true":
    CORS_ORIGINS = ["*"]
