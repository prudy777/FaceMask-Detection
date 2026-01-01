"""
Face Detection Service using OpenCV.

Detects faces in images so we can run mask classification on each face.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Represents a detected face region."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def bbox(self) -> List[int]:
        """Return bounding box as [x, y, width, height]."""
        return [self.x, self.y, self.width, self.height]
    
    @property
    def bbox_xyxy(self) -> List[int]:
        """Return bounding box as [x1, y1, x2, y2]."""
        return [self.x, self.y, self.x + self.width, self.y + self.height]


class FaceDetector:
    """
    Face detection using OpenCV's Haar Cascade classifier.
    
    Can be extended to use DNN-based detection for better accuracy.
    """
    
    def __init__(self):
        # Load the pre-trained Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            raise RuntimeError("Failed to load face cascade classifier")
    
    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR numpy array.
            scale_factor: How much the image size is reduced at each image scale.
            min_neighbors: Minimum number of neighbors for detection.
            min_size: Minimum possible face size.
        
        Returns:
            List of FaceDetection objects.
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        
        return [
            FaceDetection(x=int(x), y=int(y), width=int(w), height=int(h))
            for (x, y, w, h) in faces
        ]
    
    def extract_face_region(
        self,
        image: np.ndarray,
        face: FaceDetection,
        padding: float = 0.1,
    ) -> np.ndarray:
        """
        Extract a face region from the image with optional padding.
        
        Args:
            image: BGR numpy array.
            face: FaceDetection object.
            padding: Fraction of face size to add as padding.
        
        Returns:
            Cropped face region.
        """
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_w = int(face.width * padding)
        pad_h = int(face.height * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, face.x - pad_w)
        y1 = max(0, face.y - pad_h)
        x2 = min(w, face.x + face.width + pad_w)
        y2 = min(h, face.y + face.height + pad_h)
        
        return image[y1:y2, x1:x2]


# Global detector instance
face_detector = FaceDetector()


def get_face_detector() -> FaceDetector:
    """Get the face detector instance."""
    return face_detector
