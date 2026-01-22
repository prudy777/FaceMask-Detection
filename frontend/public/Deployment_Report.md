# Face Mask Detection System - Deployment & Architecture Report

## 1. Executive Summary
This report documents the successful deployment and major architectural upgrade of the **AI-Powered Face Mask Detection System**. The application has been migrated to a **TensorFlow/Keras** backend to support specialized dual-model inference. It features a responsive React frontend and a robust FastAPI backend containerized for cloud deployment.

## 2. System Architecture

### 2.1 Backend Migration (PyTorch → TensorFlow)
The inference engine has been migrated from PyTorch to **TensorFlow/Keras** to utilize optimized `.h5` model formats. This change aligns with the new requirements for supporting distinct models for different detection modes.

### 2.2 Dual-Model Inference Strategy
To maximize accuracy for specific use cases, the system now simultaneously loads and manages two distinct neural networks:

| Mode | Model File | Purpose | Classes |
| :--- | :--- | :--- | :--- |
| **Multi-Class** | `model.h5` | Detailed Classification | 1. With Mask<br>2. Without Mask<br>3. Worn Incorrectly |
| **Binary** | `mask_detector_cnn.h5` | Fast Pass/Fail Check | 1. With Mask<br>2. Without Mask |

*The backend `MaskDetectionModel` class acts as a singleton manager, routing inference requests to the appropriate model based on the client's `mode` parameter.*

### 2.3 Component Breakdown
*   **Frontend (User Interface)**:
    *   **Tech Stack**: React 19, TypeScript, Vite.
    *   **Features**: Real-time Webcam inference loop, File Upload, Mode Switching (Binary/Multi).
    *   **Service Layer**: `api.ts` configured for dynamic environment switching (Local/Prod).
*   **Backend (Inference Engine)**:
    *   **Tech Stack**: Python 3.11, FastAPI, TensorFlow 2.15, OpenCV.
    *   **Capabilities**: 
        *   Face Detection (Haar Cascades).
        *   Dynamic Model Switching.
        *   Base64 & Multipart-form image processing.

## 3. Deployment Pipeline

### 3.1 Containerization
The application is Dockerized for consistency across environments:
*   **Backend**: `python:3.11-slim` base. Optimized with layered copies to cache `pip install tensorflow` (large dependency) separately from application code.
*   **Frontend**: Multi-stage build (Node.js Build → Nginx Serve). Uses custom Nginx template for dynamic `$PORT` binding on Cloud Run/Railway.

### 3.2 Environment Configuration
Key environment variables for production:
*   `VITE_API_URL`: Points frontend to the backend service.
*   `ALLOW_ALL_ORIGINS`: Enables CORS for the frontend domain.
*   `PORT`: Dynamic port assignment by the hosting provider.

## 4. Testing & Validation
The system has passed the following validation steps:
*   ✅ **Dependencies**: Successful migration to `tensorflow>=2.15.0`.
*   ✅ **Unit Tests**: `pytest` passed for model loading, inference endpoints, and health checks.
*   ✅ **Local Integration**: Validated Frontend <-> Backend communication on `localhost`.
*   ✅ **Model Loading**: Confirmed successful loading of both `model.h5` and `mask_detector_cnn.h5`.

## 5. Future Roadmap
*   **Model Quantization**: Convert `.h5` models to TFLite for potential edge/browser-side inference to reduce latency.
*   **Face Detection Upgrade**: Replace Haar Cascades with MTCNN or a DNN-based face detector for improved robustness in low light.

---
**Current Status**: ✅ Deployed & Verified (Localhost & Ready for Production Push)
