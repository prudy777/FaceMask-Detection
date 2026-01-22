"""
Face Mask Detection Model - Loading and Inference (TensorFlow/Keras).

This module handles efficient loading of the Keras models
and provides inference capabilities for face mask detection.
"""
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from .config import (
    MULTI_MODEL_PATH,
    BINARY_MODEL_PATH,
    MODEL_INPUT_SIZE,
    CLASS_LABELS,
    BINARY_CLASS_LABELS,
)


class MaskDetectionModel:
    """
    Wrapper class for the Face Mask Detection Keras models.
    
    Handles model loading, preprocessing, and inference.
    Loads both binary and multi-class models.
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
            
        self.multi_model = None
        self.binary_model = None
        self.class_labels = CLASS_LABELS
        self.binary_labels = BINARY_CLASS_LABELS
        self._initialized = True
        
    def load_model(self) -> bool:
        """
        Load the pre-trained models from disk.
        
        Returns:
            bool: True if at least one model loaded successfully.
        """
        success = False
        
        # Load Multi-Class Model
        try:
            if MULTI_MODEL_PATH.exists():
                print(f"Loading Multi-class model from {MULTI_MODEL_PATH}...")
                self.multi_model = tf.keras.models.load_model(str(MULTI_MODEL_PATH))
                print("Multi-class model loaded successfully.")
                success = True
            else:
                print(f"Warning: Multi-class model file not found at {MULTI_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading multi-class model: {e}")

        # Load Binary Model
        try:
            if BINARY_MODEL_PATH.exists():
                print(f"Loading Binary model from {BINARY_MODEL_PATH}...")
                self.binary_model = tf.keras.models.load_model(str(BINARY_MODEL_PATH))
                print("Binary model loaded successfully.")
                success = True
            else:
                print(f"Warning: Binary model file not found at {BINARY_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading binary model: {e}")
            
        return success
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess.
        
        Returns:
            Preprocessed numpy array ready for inference.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(MODEL_INPUT_SIZE)
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Assuming standard 0-1 normalization
        
        return img_array
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess an OpenCV frame for model input.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self.preprocess_image(image)
    
    def predict(self, image: Image.Image, binary_mode: bool = False) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image to classify.
            binary_mode: If True, uses the binary model.
        
        Returns:
            Tuple of (predicted_label, confidence, all_probabilities)
        """
        # Select Model
        if binary_mode:
            model = self.binary_model
            labels = self.binary_labels
            model_name = "Binary"
        else:
            model = self.multi_model
            labels = self.class_labels
            model_name = "Multi-class"
            
        if model is None:
            # Fallback or Error? 
            # If the specific model is missing, maybe try the other one or error out.
            # For now, let's error to be explicit.
            raise RuntimeError(f"{model_name} model is not loaded. Cannot perform prediction.")
        
        # Preprocess and Predict
        input_data = self.preprocess_image(image)
        predictions = model.predict(input_data, verbose=0)
        probs = predictions[0]
        
        # Parse Results
        all_probs = []
        for i, prob in enumerate(probs):
            # Handle case where model output size mismatches labels
            if i < len(labels):
                label_name = labels[i]
            else:
                label_name = f"Class {i}"
            
            all_probs.append({"label": label_name, "probability": float(prob)})
            
        # Get winner
        max_idx = np.argmax(probs)
        confidence = float(probs[max_idx])
        predicted_label = labels[max_idx] if max_idx < len(labels) else f"Class {max_idx}"
        
        return predicted_label, confidence, all_probs
    
    def predict_frame(self, frame: np.ndarray, binary_mode: bool = False) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Run inference on an OpenCV frame.
        """
        # Convert BGR to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return self.predict(image, binary_mode=binary_mode)


# Global model instance
mask_detector = MaskDetectionModel()


def get_model() -> MaskDetectionModel:
    """Get the singleton model instance."""
    return mask_detector
