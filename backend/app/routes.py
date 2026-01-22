"""
FastAPI routes for Face Mask Detection API.
"""
import time
import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List

from .schemas import (
    DetectionResult,
    ProbabilityScore,
    ImagePredictionResponse,
    FramePredictionResponse,
    HealthResponse,
)
from .model import get_model
from .face_detector import get_face_detector

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    model = get_model()
    
    # Check if at least one model is loaded
    is_loaded = (model.multi_model is not None) or (model.binary_model is not None)
    
    # Get device info
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    device = "cuda" if gpus else "cpu"
    
    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded,
        device=device,
    )


@router.post("/predict-image", response_model=ImagePredictionResponse)
async def predict_image(mode: str = "multi", file: UploadFile = File(...)):
    """
    Analyze an uploaded image for face mask detection.
    
    Detects all faces in the image and classifies each for mask wearing.
    Returns bounding boxes and classification for each detected face.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file.",
        )
    
    try:
        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array for face detection
        image_np = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        h, w = image_bgr.shape[:2]
        
        # Detect faces
        face_detector = get_face_detector()
        faces = face_detector.detect_faces(image_bgr)
        
        model = get_model()
        detections: List[DetectionResult] = []
        is_binary = (mode == "binary")
        
        if len(faces) == 0:
            # No faces detected - classify entire image
            label, confidence, all_probs = model.predict(image, binary_mode=is_binary)
            detections.append(DetectionResult(
                label=label,
                confidence=confidence,
                bbox=None,
                probabilities=[
                    ProbabilityScore(label=p["label"], probability=p["probability"])
                    for p in all_probs
                ],
            ))
            message = "No faces detected. Classified entire image."
        else:
            # Classify each detected face
            for face in faces:
                face_region = face_detector.extract_face_region(image_bgr, face)
                face_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                
                label, confidence, all_probs = model.predict(face_pil, binary_mode=is_binary)
                detections.append(DetectionResult(
                    label=label,
                    confidence=confidence,
                    bbox=face.bbox,
                    probabilities=[
                        ProbabilityScore(label=p["label"], probability=p["probability"])
                        for p in all_probs
                    ],
                ))
            message = f"Detected and classified {len(faces)} face(s)."
        
        return ImagePredictionResponse(
            success=True,
            message=message,
            detections=detections,
            image_width=w,
            image_height=h,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}",
        )


@router.post("/predict-frame", response_model=FramePredictionResponse)
async def predict_frame(mode: str = "multi", file: UploadFile = File(...)):
    """
    Process a single video frame for real-time detection.
    
    Optimized for speed - processes frame and returns results quickly.
    """
    start_time = time.time()
    
    try:
        # Read frame
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode frame data.",
            )
        
        # Detect faces
        face_detector = get_face_detector()
        faces = face_detector.detect_faces(
            frame,
            scale_factor=1.2,  # Faster but less accurate
            min_neighbors=4,
        )
        
        model = get_model()
        detections: List[DetectionResult] = []
        is_binary = (mode == "binary")
        
        for face in faces:
            face_region = face_detector.extract_face_region(frame, face)
            label, confidence, all_probs = model.predict_frame(face_region, binary_mode=is_binary)
            
            detections.append(DetectionResult(
                label=label,
                confidence=confidence,
                bbox=face.bbox,
                probabilities=[
                    ProbabilityScore(label=p["label"], probability=p["probability"])
                    for p in all_probs
                ],
            ))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return FramePredictionResponse(
            success=True,
            detections=detections,
            processing_time_ms=round(processing_time, 2),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing frame: {str(e)}",
        )


@router.post("/predict-frame-base64", response_model=FramePredictionResponse)
async def predict_frame_base64(data: dict, mode: str = "multi"):
    """
    Process a base64-encoded video frame for real-time detection.
    
    Alternative to file upload - accepts JSON with base64 image data.
    Expected format: {"image": "base64_encoded_image_data"}
    """
    start_time = time.time()
    
    try:
        if "image" not in data:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field in request body.",
            )
        
        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            # Handle data URL format (e.g., "data:image/jpeg;base64,...")
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode base64 image data.",
            )
        
        # Detect faces
        face_detector = get_face_detector()
        faces = face_detector.detect_faces(
            frame,
            scale_factor=1.2,
            min_neighbors=4,
        )
        
        model = get_model()
        detections: List[DetectionResult] = []
        is_binary = (mode == "binary")
        
        for face in faces:
            face_region = face_detector.extract_face_region(frame, face)
            label, confidence, all_probs = model.predict_frame(face_region, binary_mode=is_binary)
            
            detections.append(DetectionResult(
                label=label,
                confidence=confidence,
                bbox=face.bbox,
                probabilities=[
                    ProbabilityScore(label=p["label"], probability=p["probability"])
                    for p in all_probs
                ],
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return FramePredictionResponse(
            success=True,
            detections=detections,
            processing_time_ms=round(processing_time, 2),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing frame: {str(e)}",
        )
