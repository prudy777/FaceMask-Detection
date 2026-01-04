import React, { useRef, useState, useCallback } from 'react';
import { predictImage } from '../services/api';
import type { ImagePredictionResponse } from '../services/api';

interface ImageUploadProps {
    detectionMode: 'multi' | 'binary';
    onResult?: (result: ImagePredictionResponse) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ detectionMode, onResult }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isDragOver, setIsDragOver] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [result, setResult] = useState<ImagePredictionResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFile = useCallback(async (file: File) => {
        if (!file.type.startsWith('image/')) {
            setError('Please upload an image file');
            return;
        }

        setError(null);
        setIsLoading(true);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);

        try {
            const response = await predictImage(file, detectionMode);
            setResult(response);
            onResult?.(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to analyze image');
        } finally {
            setIsLoading(false);
        }
    }, [detectionMode, onResult]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    }, [handleFile]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setIsDragOver(false);
    }, []);

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleFile(file);
    };

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
                <h2 className="card-title">📤 Upload Image</h2>
            </div>
            <div className="card-body">
                <div
                    className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
                    onClick={handleClick}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                >
                    <div className="upload-zone-icon">📁</div>
                    <h3>Drop an image here or click to browse</h3>
                    <p>Supports JPG, PNG, WebP</p>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleInputChange}
                    />
                </div>

                {error && (
                    <div className="detection-item" style={{ marginTop: '1rem', background: 'var(--color-danger-bg)' }}>
                        <div className="detection-icon without-mask">❌</div>
                        <div className="detection-info">
                            <div className="detection-label" style={{ color: 'var(--color-danger)' }}>{error}</div>
                        </div>
                    </div>
                )}

                {(previewUrl || result) && (
                    <div className="results-container">
                        {/* Image Preview with Bounding Boxes */}
                        <div className="image-preview">
                            {previewUrl && (
                                <img src={previewUrl} alt="Uploaded preview" />
                            )}
                            {isLoading && (
                                <div className="loading-overlay">
                                    <div className="loading-spinner"></div>
                                    <div className="loading-text">Analyzing image...</div>
                                </div>
                            )}
                            {result && (
                                <div className="detection-overlay">
                                    {result.detections.map((detection, idx) => (
                                        detection.bbox && (
                                            <div
                                                key={idx}
                                                className={`bbox ${getLabelClass(detection.label)}`}
                                                style={{
                                                    left: `${(detection.bbox[0] / result.image_width) * 100}%`,
                                                    top: `${(detection.bbox[1] / result.image_height) * 100}%`,
                                                    width: `${(detection.bbox[2] / result.image_width) * 100}%`,
                                                    height: `${(detection.bbox[3] / result.image_height) * 100}%`,
                                                }}
                                            >
                                                <span className="bbox-label">
                                                    {detection.label} ({Math.round(detection.confidence * 100)}%)
                                                </span>
                                            </div>
                                        )
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Detection Results List */}
                        <div>
                            <h3 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>
                                Detection Results
                            </h3>
                            {result ? (
                                <div className="detection-list">
                                    {result.detections.length === 0 ? (
                                        <div className="empty-state">
                                            <div className="empty-state-icon">🔍</div>
                                            <h3>No faces detected</h3>
                                            <p>Try uploading a clearer image with visible faces</p>
                                        </div>
                                    ) : (
                                        result.detections.map((detection, idx) => (
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
                            ) : (
                                <div className="empty-state">
                                    <div className="loading-spinner" style={{ margin: '0 auto' }}></div>
                                </div>
                            )}

                            {result && (
                                <div className="stats-row">
                                    <div className="stat-item">
                                        <span className="stat-label">Faces Found:</span>
                                        <span className="stat-value">{result.detections.filter(d => d.bbox).length}</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-label">Image Size:</span>
                                        <span className="stat-value">{result.image_width}×{result.image_height}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageUpload;
