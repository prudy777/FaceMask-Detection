import { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import WebcamDetection from './components/WebcamDetection';
import { checkHealth } from './services/api';
import type { HealthResponse } from './services/api';
import './App.css';

type TabType = 'upload' | 'webcam';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await checkHealth();
        setHealth(response);
        setHealthError(false);
      } catch {
        setHealthError(true);
      }
    };

    fetchHealth();
    // Poll health every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          <span>🛡️</span>
          Face Mask Detection
        </h1>
        <div className={`status-badge ${health && !healthError ? 'online' : 'offline'}`}>
          <span className={`status-dot ${health && !healthError ? 'online' : 'offline'}`}></span>
          {health && !healthError ? (
            <>API Online ({health.device})</>
          ) : (
            <>API Offline</>
          )}
        </div>
      </header>

      <main className="app-main">
        {/* Tab Navigation */}
        <div className="tabs">
          <button
            className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📤 Image Upload
          </button>
          <button
            className={`tab-button ${activeTab === 'webcam' ? 'active' : ''}`}
            onClick={() => setActiveTab('webcam')}
          >
            📹 Real-Time
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'upload' && <ImageUpload />}
        {activeTab === 'webcam' && <WebcamDetection isActive={activeTab === 'webcam'} />}
      </main>
    </div>
  );
}

export default App;
