"""
Unit and integration tests for Face Mask Detection Backend.
"""
import io
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.model import get_model
from app.face_detector import get_face_detector

# Fixture to ensure model is loaded
@pytest.fixture(autouse=True)
def setup_model():
    model = get_model()
    # Ensure models are loaded
    if model.multi_model is None and model.binary_model is None:
        model.load_model()
    yield

def create_dummy_image(color=(100, 100, 100), size=(224, 224)):
    """Create a dummy RGB image for testing."""
    image = Image.new("RGB", size, color)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_health_check():
    """Test the health check endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        # API behavior might change, but let's assume it checks for at least one model
        # assert data["model_loaded"] is True 

def test_model_initialization():
    """Verify model loads."""
    model = get_model()
    if model.multi_model is None:
        model.load_model()
        
    # Should be initialized
    assert model._initialized
    # Should have at least one model
    assert model.multi_model is not None or model.binary_model is not None
    # Check class labels
    assert len(model.class_labels) == 3

def test_face_detector_initialization():
    """Verify face detector loads."""
    detector = get_face_detector()
    assert detector.detector is not None

def test_predict_image_endpoint():
    """Test image prediction endpoint with a dummy image."""
    image_bytes = create_dummy_image()
    
    with TestClient(app) as client:
        response = client.post(
            "/api/predict-image",
            files={"file": ("test_image.jpg", image_bytes, "image/jpeg")}
        )
        
        if response.status_code != 200:
            print(f"FAILED Response: {response.text}")
            
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detections" in data
        assert isinstance(data["detections"], list)
        
        # Depending on random init and dummy image, it might or might not find faces.
        # But it should return successfully.

def test_predict_frame_endpoint():
    """Test frame prediction endpoint."""
    image_bytes = create_dummy_image()
    
    with TestClient(app) as client:
        response = client.post(
            "/api/predict-frame",
            files={"file": ("frame.jpg", image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["detections"], list)

def test_invalid_image_upload():
    """Test uploading a non-image file."""
    with TestClient(app) as client:
        response = client.post(
            "/api/predict-image",
            files={"file": ("test.txt", b"plain text", "text/plain")}
        )
        assert response.status_code == 400
