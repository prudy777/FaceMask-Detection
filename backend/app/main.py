"""
Face Mask Detection API - Main Application Entry Point.

This module initializes the FastAPI application and loads the model at startup.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS
from .model import get_model
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Loads the model once at startup for efficient inference.
    """
    # Startup: Load the model
    print("🚀 Starting Face Mask Detection API...")
    model = get_model()
    success = model.load_model()
    if success:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model loading had issues, using placeholder model.")
    
    yield
    
    # Shutdown: Cleanup if needed
    print("👋 Shutting down Face Mask Detection API...")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api", tags=["Detection"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "predict_image": "/api/predict-image",
            "predict_frame": "/api/predict-frame",
            "predict_frame_base64": "/api/predict-frame-base64",
        },
    }
