# Face Mask Detection Models

This directory contains the trained Keras/TensorFlow models used by the backend.

## Model Files

1.  **Multi-Class Model** (`model.h5`)
    - Used for standard multi-class detection.
    - Expected classes: "With Mask", "Without Mask", "Mask Worn Incorrectly".
    - Format: Keras `.h5` model.

2.  **Binary Model** (`mask_detector_cnn.h5`)
    - Used for binary detection (With/Without Mask).
    - Format: Keras `.h5` model.

## Input Specifications

- **Input Size**: 224x224 RGB images
- **Preprocessing**: Resizing + Normalization (1/255.0)

## Implementation Details

The backend uses `tensorflow.keras` to load these models. Logic is handled in `app/model.py`.
