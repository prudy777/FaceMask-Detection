import React, { useRef, useState, useEffect, useCallback } from 'react';
import { predictFrame } from '../services/api';
import type { Detection } from '../services/api';

interface WebcamDetectionProps {
    isActive: boolean;
}

const WebcamDetection: React.FC<WebcamDetectionProps> = ({ isActive }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const animationRef = useRef<number | null>(null);

    const [isStreaming, setIsStreaming] = useState(false);
    const [isDetecting, setIsDetecting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [detections, setDetections] = useState<Detection[]>([]);
    const [fps, setFps] = useState(0);
    const [processingTime, setProcessingTime] = useState(0);

    const lastFrameTime = useRef<number>(0);
    const frameCount = useRef<number>(0);

    const startCamera = useCallback(async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;
                setIsStreaming(true);
            }
        } catch (err) {
            setError('Camera access denied. Please allow camera permissions.');
            console.error('Camera error:', err);
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }
        setIsStreaming(false);
        setIsDetecting(false);
        setDetections([]);
    }, []);

    const captureFrame = useCallback(async (): Promise<Blob | null> => {
        const video = videoRef.current;
        const canvas = document.createElement('canvas');

        if (!video || video.readyState !== 4) return null;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        ctx.drawImage(video, 0, 0);

        return new Promise((resolve) => {
            canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.8);
        });
    }, []);

    const drawDetections = useCallback((detections: Detection[]) => {
        const canvas = canvasRef.current;
        const video = videoRef.current;

        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Match canvas size to video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw each detection
        detections.forEach((detection) => {
            if (!detection.bbox) return;

            const [x, y, w, h] = detection.bbox;

            // Color based on detection type
            let color = '#10b981'; // Green for with mask
            if (detection.label.includes('Without')) {
                color = '#ef4444'; // Red
            } else if (detection.label.includes('Incorrect')) {
                color = '#f59e0b'; // Orange
            }

            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Draw label background
            const label = `${detection.label} (${Math.round(detection.confidence * 100)}%)`;
            ctx.font = 'bold 14px Inter, sans-serif';
            const textMetrics = ctx.measureText(label);
            const textHeight = 20;
            const padding = 6;

            ctx.fillStyle = color;
            ctx.fillRect(
                x - 1,
                y - textHeight - padding * 2,
                textMetrics.width + padding * 2,
                textHeight + padding
            );

            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + padding - 1, y - padding - 4);
        });
    }, []);

    const runDetection = useCallback(async () => {
        if (!isDetecting || !isStreaming) return;

        const now = performance.now();

        try {
            const frameBlob = await captureFrame();
            if (frameBlob) {
                const result = await predictFrame(frameBlob);
                setDetections(result.detections);
                setProcessingTime(result.processing_time_ms);
                drawDetections(result.detections);
            }
        } catch (err) {
            console.error('Detection error:', err);
        }

        // Calculate FPS
        frameCount.current++;
        if (now - lastFrameTime.current >= 1000) {
            setFps(frameCount.current);
            frameCount.current = 0;
            lastFrameTime.current = now;
        }

        // Continue detection loop
        if (isDetecting) {
            animationRef.current = requestAnimationFrame(runDetection);
        }
    }, [isDetecting, isStreaming, captureFrame, drawDetections]);

    const toggleDetection = useCallback(() => {
        if (isDetecting) {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }
            setIsDetecting(false);
            setDetections([]);
            // Clear canvas
            const canvas = canvasRef.current;
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx?.clearRect(0, 0, canvas.width, canvas.height);
            }
        } else {
            setIsDetecting(true);
            lastFrameTime.current = performance.now();
            frameCount.current = 0;
        }
    }, [isDetecting]);

    // Start detection loop when isDetecting changes
    useEffect(() => {
        if (isDetecting && isStreaming) {
            runDetection();
        }
        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isDetecting, isStreaming, runDetection]);

    // Cleanup on unmount or when tab switches
    useEffect(() => {
        if (!isActive && isStreaming) {
            stopCamera();
        }
    }, [isActive, isStreaming, stopCamera]);

    const getLabelClass = (label: string): string => {
        if (label.includes('No FaceMask') || label.includes('Without')) return 'without-mask';
        if (label.includes('Incorrectly')) return 'incorrect-mask';
        return 'with-mask';
    };

    const getLabelIcon = (label: string): string => {
        if (label.includes('No FaceMask') || label.includes('Without')) return '❌';
        if (label.includes('Incorrectly')) return '⚠️';
        return '✅';
    };

    return (
        <div className="card">
            <div className="card-header">
                <h2 className="card-title">📹 Real-Time Detection</h2>
                {isStreaming && (
                    <div className="stats-row" style={{ margin: 0 }}>
                        <div className="stat-item">
                            <span className="stat-label">FPS:</span>
                            <span className="stat-value">{fps}</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Latency:</span>
                            <span className="stat-value">{processingTime.toFixed(0)}ms</span>
                        </div>
                    </div>
                )}
            </div>
            <div className="card-body">
                {error && (
                    <div className="detection-item" style={{ marginBottom: '1rem', background: 'var(--color-danger-bg)' }}>
                        <div className="detection-icon without-mask">❌</div>
                        <div className="detection-info">
                            <div className="detection-label" style={{ color: 'var(--color-danger)' }}>{error}</div>
                        </div>
                    </div>
                )}

                <div className="results-container">
                    {/* Webcam Feed */}
                    <div className="webcam-container">
                        <video
                            ref={videoRef}
                            className="webcam-video"
                            autoPlay
                            playsInline
                            muted
                            style={{ display: isStreaming ? 'block' : 'none' }}
                        />
                        <canvas
                            ref={canvasRef}
                            className="webcam-canvas"
                        />
                        {!isStreaming && (
                            <div className="empty-state" style={{ padding: '4rem' }}>
                                <div className="empty-state-icon">📹</div>
                                <h3>Camera not started</h3>
                                <p>Click "Start Camera" to begin</p>
                            </div>
                        )}
                    </div>

                    {/* Live Detection Results */}
                    <div>
                        <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>
                            Live Detections
                        </h3>
                        <div className="detection-list">
                            {detections.length === 0 ? (
                                <div className="empty-state">
                                    <div className="empty-state-icon">👤</div>
                                    <h3>{isDetecting ? 'Scanning...' : 'No detections yet'}</h3>
                                    <p>{isDetecting ? 'Position your face in the camera' : 'Start detection to see results'}</p>
                                </div>
                            ) : (
                                detections.map((detection, idx) => (
                                    <div key={idx} className="detection-item">
                                        <div className={`detection-icon ${getLabelClass(detection.label)}`}>
                                            {getLabelIcon(detection.label)}
                                        </div>
                                        <div className="detection-info">
                                            <div className="detection-label">{detection.label}</div>
                                            <div className="detection-confidence">
                                                <div className="confidence-bar">
                                                    <div
                                                        className={`confidence-fill ${getLabelClass(detection.label)}`}
                                                        style={{ width: `${detection.confidence * 100}%` }}
                                                    />
                                                </div>
                                                <span className="confidence-value">
                                                    {Math.round(detection.confidence * 100)}%
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* Controls */}
                <div className="webcam-controls">
                    {!isStreaming ? (
                        <button className="btn btn-primary" onClick={startCamera}>
                            📷 Start Camera
                        </button>
                    ) : (
                        <>
                            <button
                                className={`btn ${isDetecting ? 'btn-danger' : 'btn-primary'}`}
                                onClick={toggleDetection}
                            >
                                {isDetecting ? '⏹️ Stop Detection' : '▶️ Start Detection'}
                            </button>
                            <button className="btn btn-secondary" onClick={stopCamera}>
                                📴 Stop Camera
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default WebcamDetection;
