# Face Mask Detection System - Model Deployment Report

## 1. Executive Summary
This report documents the successful deployment of the **AI-Powered Face Mask Detection System**. The project is a full-stack web application capable of real-time face mask classification (Correct/Incorrect/None). It has been containerized using Docker and deployed to a cloud environment (Railway) to ensure accessibility and scalability.

## 2. System Architecture
The application follows a decoupled microservices architecture, ensuring separation of concerns between the user interface and the inference engine.

### 2.1 Component Breakdown
*   **Frontend (User Interface)**:
    *   **Framework**: React (Vite) + TypeScript.
    *   **Functionality**: Handles image uploads, webcam streaming, and visualizing detection results.
    *   **Deployment**: Nginx serving static assets.
*   **Backend (Inference Engine)**:
    *   **Framework**: FastAPI (Python).
    *   **Functionality**: Preprocesses images, runs the PyTorch inference model, and returns JSON predictions.
    *   **Deployment**: Python 3.11 container with Uvicorn server.
*   **Model**:
    *   **Architecture**: ResNet18 (Transfer Learning).
    *   **Framework**: PyTorch.

### 2.2 Data Flow
1.  **Input**: User interacts with the web interface (uploads image or activates webcam).
2.  **Request**: Frontend sends image data (Multipart form or Base64) to the Backend API.
3.  **Preprocessing**: Backend detects faces using OpenCV (Haar Cascades) and crops them.
4.  **Inference**:
    *   Cropped faces are resized to 224x224 and normalized.
    *   ResNet18 model calculates probabilities for 3 classes.
5.  **Response**: API returns bounding boxes, labels, and confidence scores.
6.  **Visualization**: Frontend draws boxes and labels on the UI.

## 3. Model Specification
*   **Base Architecture**: ResNet18 (Residual Neural Network, 18 layers).
*   **Input Resolution**: 224 x 224 pixels (RGB).
*   **Classes**:
    1.  `Face Mask Worn Correctly`
    2.  `FaceMask Worn Incorrectly`
    3.  `No FaceMask`
*   **Optimization**: The model leverages transfer learning, pre-trained on ImageNet, and fine-tuned for face mask dataset features.

## 4. Deployment Strategy
The system is deployed using a **Container-Based Strategy** on **Railway**.

### 4.1 Containerization (Docker)
We utilized Docker to ensure consistency across development and production environments.
*   **Backend Dockerfile**: 
    *   Base Image: `python:3.11-slim`.
    *   Installs system dependencies for OpenCV (`libgl1`, etc.).
    *   Exposes a dynamic port using environment variable expansion to support cloud platforms.
*   **Frontend Dockerfile**:
    *   Multi-stage build:
        1.  **Build Stage**: Node.js compiles the React code to static HTML/JS.
        2.  **Serve Stage**: Nginx serves the static files.
    *   Custom Nginx configuration (`nginx.conf.template`) handles dynamic port binding (`$PORT`) and Single Page Application (SPA) routing.

### 4.2 Cloud Infrastructure (Railway)
*   **Services**: Two distinct services (Frontend & Backend) linked via internal/public networking.
*   **CI/CD Pipeline**: 
    *   Integrated with GitHub.
    *   Automatic deployments triggered on push to `main` branch.
*   **Environment Configuration**:
    *   `ALLOW_ALL_ORIGINS`: Configured on Backend to allow Cross-Origin Resource Sharing (CORS) from the frontend.
    *   `VITE_API_URL`: Configured on Frontend to dynamically point to the production Backend URL.

## 5. Challenges & Solutions
| Challenge | Solution |
| :--- | :--- |
| **Cloud Port Assignment** | Cloud platforms assign random ports. We modified Dockerfiles to listen on `$PORT` dynamically instead of a hardcoded `8000` or `8080`. |
| **Model Size (Git LFS)** | The `.pth` model file is large. We configured Git LFS (Large File Storage) and ensured the Docker build context includes the model directory. |
| **CORS Errors** | Browser security blocked the frontend from calling the backend. We implemented `CORSMiddleware` in FastAPI and configured the `ALLOW_ALL_ORIGINS` environment variable. |

## 6. Conclusion
The Face Mask Detection system is now fully operational in a production environment. The deployment pipeline is robust, automated, and scalable, successfully meeting the project requirements for a modern AI web application.

---
**Status**: ✅ Deployed & Verified
**URL**: [Access Application](https://facemask-detection-production.up.railway.app/)
