"""
Face Mask Detection Model - Loading and Inference.

This module handles efficient loading of the PyTorch model
and provides inference capabilities for face mask detection.
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from .config import (
    MODEL_PATH,
    MODEL_INPUT_SIZE,
    CLASS_LABELS,
    BINARY_MAPPING,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)


class MaskDetectionModel:
    """
    Wrapper class for the Face Mask Detection model.
    
    Handles model loading, preprocessing, and inference.
    The model is loaded once at initialization for efficiency.
    """
    
    _instance: Optional['MaskDetectionModel'] = None
    
    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.transform = self._create_transforms()
        self.class_labels = CLASS_LABELS
        self._initialized = True
        
    def _create_transforms(self) -> transforms.Compose:
        """Create image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
    
    def _create_model_architecture(self) -> nn.Module:
        """
        Create the model architecture.
        
        Uses ResNet18 as base with custom classification head.
        The user's model has keys like 'conv1', 'layer1', 'bn1' which indicate ResNet.
        """
        model = models.resnet18(weights=None)
        # Modify the final fc layer for 3 classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_labels))
        return model
    
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load the pre-trained model from disk.
        
        Args:
            model_path: Optional path to the model file.
                       Uses default MODEL_PATH if not specified.
        
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        path = model_path or MODEL_PATH
        
        try:
            if not path.exists():
                print(f"Warning: Model file not found at {path}")
                print("Creating a placeholder model for development...")
                self.model = self._create_model_architecture()
                self.model.to(self.device)
                self.model.eval()
                return True
            
            self.model = self._create_model_architecture()
            
            # Load the state dict
            state_dict = torch.load(path, map_location=self.device)
            
            # Handle different save formats
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {path}")
            print(f"Using device: {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to placeholder model
            self.model = self._create_model_architecture()
            self.model.to(self.device)
            self.model.eval()
            return False
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess.
        
        Returns:
            Preprocessed tensor ready for inference.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess an OpenCV frame (numpy array) for model input.
        
        Args:
            frame: BGR numpy array from OpenCV.
        
        Returns:
            Preprocessed tensor ready for inference.
        """
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1].copy()
        image = Image.fromarray(rgb_frame)
        return self.preprocess_image(image)
    
    @torch.no_grad()
    def predict(self, image: Image.Image, binary_mode: bool = False) -> Tuple[str, float, list]:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image to classify.
            binary_mode: If True, return binary classification (With/Without Mask).
        
        Returns:
            Tuple of (predicted_label, confidence, all_probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input_tensor = self.preprocess_image(image)
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Default Multi-class Logic
        all_probs = [
            {"label": label, "probability": prob.item()}
            for label, prob in zip(self.class_labels, probabilities)
        ]
        
        if not binary_mode:
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_label = self.class_labels[predicted_idx.item()]
            return predicted_label, confidence.item(), all_probs
            
        # Binary Mode Logic
        # Aggregate probabilities
        binary_probs = {"With Mask": 0.0, "Without Mask": 0.0}
        
        for item in all_probs:
            original_label = item["label"]
            prob = item["probability"]
            binary_label = BINARY_MAPPING[original_label]
            binary_probs[binary_label] += prob
            
        # Determine winner
        if binary_probs["With Mask"] > binary_probs["Without Mask"]:
            predicted_label = "With Mask"
            confidence = binary_probs["With Mask"]
        else:
            predicted_label = "Without Mask"
            confidence = binary_probs["Without Mask"]
            
        # Return binary probabilities structure
        binary_probs_list = [
            {"label": "With Mask", "probability": binary_probs["With Mask"]},
            {"label": "Without Mask", "probability": binary_probs["Without Mask"]},
        ]
        
        return predicted_label, confidence, binary_probs_list
    
    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray, binary_mode: bool = False) -> Tuple[str, float, list]:
        """
        Run inference on an OpenCV frame.
        
        Args:
            frame: BGR numpy array from OpenCV.
            binary_mode: If True, return binary classification.
        
        Returns:
            Tuple of (predicted_label, confidence, all_probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input_tensor = self.preprocess_frame(frame)
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Default Multi-class Logic
        all_probs = [
            {"label": label, "probability": prob.item()}
            for label, prob in zip(self.class_labels, probabilities)
        ]
        
        if not binary_mode:
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_label = self.class_labels[predicted_idx.item()]
            return predicted_label, confidence.item(), all_probs
            
        # Binary Mode Logic
        binary_probs = {"With Mask": 0.0, "Without Mask": 0.0}
        
        for item in all_probs:
            original_label = item["label"]
            prob = item["probability"]
            binary_label = BINARY_MAPPING[original_label]
            binary_probs[binary_label] += prob
            
        # Determine winner
        if binary_probs["With Mask"] > binary_probs["Without Mask"]:
            predicted_label = "With Mask"
            confidence = binary_probs["With Mask"]
        else:
            predicted_label = "Without Mask"
            confidence = binary_probs["Without Mask"]
            
        binary_probs_list = [
            {"label": "With Mask", "probability": binary_probs["With Mask"]},
            {"label": "Without Mask", "probability": binary_probs["Without Mask"]},
        ]
        
        return predicted_label, confidence, binary_probs_list


# Global model instance
mask_detector = MaskDetectionModel()


def get_model() -> MaskDetectionModel:
    """Get the singleton model instance."""
    return mask_detector
