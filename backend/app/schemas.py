"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel
from typing import List, Optional


class ProbabilityScore(BaseModel):
    """Probability score for a single class."""
    label: str
    probability: float


class DetectionResult(BaseModel):
    """Result for a single face detection."""
    label: str
    confidence: float
    bbox: Optional[List[int]] = None  # [x, y, width, height]
    probabilities: List[ProbabilityScore]


class ImagePredictionResponse(BaseModel):
    """Response for image prediction endpoint."""
    success: bool
    message: str
    detections: List[DetectionResult]
    image_width: int
    image_height: int


class FramePredictionResponse(BaseModel):
    """Response for real-time frame prediction."""
    success: bool
    detections: List[DetectionResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
