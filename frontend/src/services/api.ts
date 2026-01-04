/**
 * API Service for Face Mask Detection Backend
 */

// Use environment variable or default to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export interface ProbabilityScore {
  label: string;
  probability: number;
}

export interface Detection {
  label: string;
  confidence: number;
  bbox: number[] | null; // [x, y, width, height]
  probabilities: ProbabilityScore[];
}

export interface ImagePredictionResponse {
  success: boolean;
  message: string;
  detections: Detection[];
  image_width: number;
  image_height: number;
}

export interface FramePredictionResponse {
  success: boolean;
  detections: Detection[];
  processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
}

/**
 * Check API health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('API health check failed');
  }
  return response.json();
}

/**
 * Upload an image for mask detection
 */
/**
 * Upload an image for mask detection
 */
export async function predictImage(file: File, mode: string = 'multi'): Promise<ImagePredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/predict-image?mode=${mode}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Image prediction failed');
  }

  return response.json();
}

/**
 * Send a video frame for real-time detection
 */
export async function predictFrame(imageBlob: Blob, mode: string = 'multi'): Promise<FramePredictionResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'frame.jpg');

  const response = await fetch(`${API_BASE_URL}/predict-frame?mode=${mode}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Frame prediction failed');
  }

  return response.json();
}

/**
 * Send a base64-encoded frame for real-time detection
 */
export async function predictFrameBase64(base64Image: string, mode: string = 'multi'): Promise<FramePredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/predict-frame-base64?mode=${mode}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: base64Image }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Frame prediction failed');
  }

  return response.json();
}
