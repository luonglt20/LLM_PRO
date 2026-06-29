import React, { useState, useEffect } from 'react';
import { Mail, Compass, Folder, Calendar, Sparkles, AlertTriangle, Key } from 'lucide-react';
import DailyDigest from './components/DailyDigest';
import ScholarMap from './components/ScholarMap';
import Collections from './components/Collections';
import ConferencePlanner from './components/ConferencePlanner';

export default function App() {
  const [activeTab, setActiveTab] = useState('digest'); // digest, map, collections, planner
  const [papers, setPapers] = useState([]);
  const [digestPapers, setDigestPapers] = useState([]);
  const [ratings, setRatings] = useState({});
  const [collections, setCollections] = useState({ "My Library": [] });
  const [activeLearningPapers, setActiveLearningPapers] = useState([]);
  
  const [loading, setLoading] = useState(true);
  const [backendError, setBackendError] = useState(false);

  // Gemini API Key State
  const [geminiApiKey, setGeminiApiKey] = useState(() => localStorage.getItem('gemini_api_key') || '');
  const [showKeyInput, setShowKeyInput] = useState(false);

  const handleKeyChange = (val) => {
    setGeminiApiKey(val);
    localStorage.setItem('gemini_api_key', val);
  };

  // 1. Fetch main papers and ratings
  const fetchPapersAndRatings = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5001/api/papers');
      if (!res.ok) throw new Error();
      const data = await res.json();
      setPapers(data.papers || []);
      setRatings(data.ratings || {});
      setBackendError(false);
      return data;
    } catch (err) {
      console.warn("Failed to connect to backend server. Re-attempting...");
      setBackendError(true);
      return null;
    }
  };

  // 2. Fetch daily/weekly digest recommendations
  const fetchDigest = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5001/api/digest?range=day');
      if (!res.ok) throw new Error();
      const data = await res.json();
      setDigestPapers(data.papers || []);
    } catch (err) {
      console.warn("Could not fetch digest from backend.");
    }
  };

  // 3. Fetch active learning query (papers near decision boundary)
  const fetchActiveLearning = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5001/api/active-learning');
      if (!res.ok) throw new Error();
      const data = await res.json();
      setActiveLearningPapers(data.papers || []);
    } catch (err) {
      console.warn("Could not fetch active learning papers.");
    }
  };

  // 4. Fetch user paper collections
  const fetchCollections = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5001/api/collections');
      if (!res.ok) throw new Error();
      const data = await res.json();
      setCollections(data.collections || { "My Library": [] });
    } catch (err) {
      console.warn("Could not fetch collections.");
    }
  };

  // Coordinated Initial Load
  const loadAllData = async () => {
    setLoading(true);
    const mainData = await fetchPapersAndRatings();
    if (mainData) {
      await Promise.all([
        fetchDigest(),
        fetchActiveLearning(),
        fetchCollections()
      ]);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadAllData();
  }, []);

  // Submit Paper Rating (Thumbs Up/Down)
  const handleRatePaper = async (paperId, rating) => {
    try {
      const res = await fetch('http://127.0.0.1:5001/api/rate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_id: paperId, rating })
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        // Update states
        setPapers(data.papers);
        setRatings(data.ratings);
        
        // Refresh digest and active learning dynamically
        await Promise.all([
          fetchDigest(),
          fetchActiveLearning()
        ]);
      }
    } catch (err) {
      console.error("Failed to submit rating:", err);
    }
  };

  // Collection Folder additions / removals
  const handleManageCollection = async (action, name, paperId = null) => {
    try {
      const res = await fetch('http://127.0.0.1:5001/api/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, name, paper_id: paperId })
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        setCollections(data.collections);
      }
    } catch (err) {
      console.error("Failed to manage collection:", err);
    }
  };

  return (
    <div className="app-container">
      {/* Top Header & Navigation tab-bar */}
      <header className="header-bar glass-panel">
        <div className="logo-section">
          <div className="logo-icon">
            <Sparkles size={22} fill="#fff" />
          </div>
          <div className="logo-text">Scholar Inbox</div>
        </div>
        
        <nav className="nav-tabs">
          <button 
            className={`nav-tab ${activeTab === 'digest' ? 'active' : ''}`}
            onClick={() => setActiveTab('digest')}
          >
            <Mail size={16} /> Daily Digest
          </button>
          <button 
            className={`nav-tab ${activeTab === 'map' ? 'active' : ''}`}
            onClick={() => setActiveTab('map')}
          >
            <Compass size={16} /> Scholar Maps
          </button>
          <button 
            className={`nav-tab ${activeTab === 'collections' ? 'active' : ''}`}
            onClick={() => setActiveTab('collections')}
          >
            <Folder size={16} /> Collections & Search
          </button>
          <button 
            className={`nav-tab ${activeTab === 'planner' ? 'active' : ''}`}
            onClick={() => setActiveTab('planner')}
          >
            <Calendar size={16} /> Conference Planner
          </button>
        </nav>

        {/* Dynamic API Key Input field */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
          <button 
            className={`action-btn ${geminiApiKey ? 'bookmarked' : ''}`}
            onClick={() => setShowKeyInput(!showKeyInput)}
            title="Configure Gemini API Key for Custom AI Insights"
            style={{ width: '36px', height: '36px' }}
          >
            <Key size={16} />
          </button>
          {showKeyInput && (
            <input 
              type="password" 
              className="glass-input" 
              placeholder="Enter Gemini API Key..." 
              value={geminiApiKey}
              onChange={(e) => handleKeyChange(e.target.value)}
              style={{ padding: '6px 12px', fontSize: '12px', width: '180px' }}
            />
          )}
        </div>
      </header>

      {/* Main Panel Viewport */}
      <main className="main-viewport">
        {/* Backend offline warning banner */}
        {backendError && (
          <div 
            className="glass-panel" 
            style={{
              padding: '16px',
              marginBottom: '24px',
              borderColor: 'rgba(245, 158, 11, 0.4)',
              background: 'rgba(245, 158, 11, 0.1)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              textAlign: 'left'
            }}
          >
            <AlertTriangle size={24} style={{ color: '#f59e0b', flexShrink: 0 }} />
            <div>
              <h4 style={{ margin: '0 0 4px 0', color: '#fff', fontSize: '15px' }}>Backend Offline / Connecting...</h4>
              <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)' }}>
                Please run the backend server in your terminal: 
                <code style={{ marginLeft: '8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', padding: '2px 6px', color: '#c084fc' }}>
                  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 backend/app.py
                </code>
              </p>
            </div>
            <button 
              className="glass-button" 
              style={{ marginLeft: 'auto', padding: '6px 12px', fontSize: '12px' }}
              onClick={loadAllData}
            >
              Retry Connection
            </button>
          </div>
        )}

        {loading ? (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Loading papers and clusters database...</p>
          </div>
        ) : (
          /* Render Active View Component */
          <>
            {activeTab === 'digest' && (
              <DailyDigest 
                papers={digestPapers}
                ratings={ratings}
                onRate={handleRatePaper}
                onBookmark={(pid) => handleManageCollection(collections['My Library']?.includes(pid) ? 'remove' : 'add', 'My Library', pid)}
                bookmarkedIds={collections['My Library'] || []}
                allPapers={papers}
                geminiApiKey={geminiApiKey}
              />
            )}
            
            {activeTab === 'map' && (
              <ScholarMap 
                papers={papers}
                ratings={ratings}
                onRate={handleRatePaper}
                activeLearningPapers={activeLearningPapers}
              />
            )}
            
            {activeTab === 'collections' && (
              <Collections 
                ratings={ratings}
                onRate={handleRatePaper}
                allPapers={papers}
                collections={collections}
                onManageCollection={handleManageCollection}
              />
            )}
            
            {activeTab === 'planner' && (
              <ConferencePlanner 
                ratings={ratings}
                allPapers={papers}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
