import React, { useState, useEffect } from 'react';
import { Mail, Compass, Folder, Calendar, Sparkles, AlertTriangle, Key, Brain, X, ChevronRight, BookOpen } from 'lucide-react';
import DailyDigest from './components/DailyDigest';
import ScholarMap from './components/ScholarMap';
import Collections from './components/Collections';
import ConferencePlanner from './components/ConferencePlanner';
import ResearchWorkspace from './components/ResearchWorkspace';

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

  // Global PDF Viewer & Chat States
  const [activePdfPaper, setActivePdfPaper] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    if (activePdfPaper) {
      setChatHistory([
        {
          sender: 'ai',
          text: `Hi! I am your AI assistant for this paper. Ask me anything about the methodology, benchmarks, or results in "${activePdfPaper.title}".`
        }
      ]);
    } else {
      setChatHistory([]);
    }
    setChatInput('');
    setChatLoading(false);
  }, [activePdfPaper]);

  const handleSendChatMessage = async (e) => {
    if (e) e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMsg = chatInput.trim();
    setChatHistory(prev => [...prev, { sender: 'user', text: userMsg }]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5001/api/chat-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_id: activePdfPaper.id,
          question: userMsg,
          api_key: geminiApiKey
        })
      });
      const data = await response.json();
      if (data.answer) {
        setChatHistory(prev => [...prev, { sender: 'ai', text: data.answer }]);
      } else {
        setChatHistory(prev => [...prev, { sender: 'ai', text: `Error: ${data.error || 'Failed to process question'}` }]);
      }
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, { sender: 'ai', text: 'Error: Failed to connect to server.' }]);
    } finally {
      setChatLoading(false);
    }
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
          <button 
            className={`nav-tab ${activeTab === 'workspace' ? 'active' : ''}`}
            onClick={() => setActiveTab('workspace')}
          >
            <Brain size={16} /> AI Workspace
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
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'map' && (
              <ScholarMap 
                papers={papers}
                ratings={ratings}
                onRate={handleRatePaper}
                activeLearningPapers={activeLearningPapers}
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'collections' && (
              <Collections 
                ratings={ratings}
                onRate={handleRatePaper}
                allPapers={papers}
                collections={collections}
                onManageCollection={handleManageCollection}
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'planner' && (
              <ConferencePlanner 
                ratings={ratings}
                allPapers={papers}
              />
            )}
            
            {activeTab === 'workspace' && (
              <ResearchWorkspace 
                apiKey={geminiApiKey}
                onViewPdf={setActivePdfPaper}
                ratings={ratings}
                onRate={handleRatePaper}
                onBookmark={(pid) => handleManageCollection(collections['My Library']?.includes(pid) ? 'remove' : 'add', 'My Library', pid)}
                bookmarkedIds={collections['My Library'] || []}
              />
            )}
          </>
        )}
      </main>

      {/* Global PDF Document & AI Chat Assistant Modal */}
      {activePdfPaper && (
        <div className="modal-overlay" onClick={() => setActivePdfPaper(null)}>
          <div className="modal-content glass-panel" style={{ maxWidth: '1200px', width: '95%', height: '90vh', display: 'flex', flexDirection: 'column' }} onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setActivePdfPaper(null)}><X size={20} /></button>
            <h3 style={{ margin: '0 0 4px 0', fontFamily: 'var(--font-heading)' }}>Official PDF Document & AI Chat Assistant</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '4px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              Document: <strong>"{activePdfPaper.title}"</strong>
            </p>
            
            <div style={{ display: 'flex', flex: 1, gap: '16px', minHeight: 0, marginTop: '12px' }}>
              {/* Left pane: PDF Iframe */}
              <div style={{ flex: 1, background: '#0f111a', borderRadius: '8px', overflow: 'hidden', height: '100%' }}>
                <iframe 
                  src={`https://arxiv.org/pdf/${activePdfPaper.id}.pdf`}
                  style={{ width: '100%', height: '100%', border: 'none' }}
                  title={activePdfPaper.title}
                />
              </div>
              
              {/* Right pane: AI RAG Chat Sidebar */}
              <div style={{ 
                width: '340px', 
                display: 'flex', 
                flexDirection: 'column', 
                background: 'rgba(0, 0, 0, 0.2)', 
                border: '1px solid rgba(255,255,255,0.05)', 
                borderRadius: '8px', 
                padding: '16px', 
                minHeight: 0,
                height: '100%'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '8px' }}>
                  <Sparkles size={16} style={{ color: 'var(--accent-color)' }} />
                  <span style={{ fontWeight: '700', fontSize: '14px', color: '#fff' }}>Paper Assistant (RAG)</span>
                </div>
                
                {/* Chat Messages Log */}
                <div style={{ 
                  flex: 1, 
                  overflowY: 'auto', 
                  marginBottom: '12px', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '10px',
                  paddingRight: '4px'
                }}>
                  {chatHistory.map((msg, idx) => (
                    <div 
                      key={idx} 
                      style={{ 
                        alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                        maxWidth: '90%',
                        padding: '10px 12px',
                        borderRadius: '8px',
                        fontSize: '13px',
                        lineHeight: '1.4',
                        textAlign: 'left',
                        whiteSpace: msg.sender === 'user' ? 'pre-wrap' : 'normal',
                        background: msg.sender === 'user' ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
                        color: msg.sender === 'user' ? '#fff' : 'var(--text-secondary)',
                        border: msg.sender === 'user' ? 'none' : '1px solid rgba(255,255,255,0.02)'
                      }}
                    >
                      {msg.sender === 'user' ? msg.text : renderChatText(msg.text)}
                    </div>
                  ))}
                  {chatLoading && (
                    <div 
                      style={{ 
                        alignSelf: 'flex-start',
                        background: 'rgba(255,255,255,0.05)',
                        padding: '10px 12px',
                        borderRadius: '8px',
                        fontSize: '13px',
                        color: 'var(--text-muted)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                      }}
                    >
                      <div style={{ width: '12px', height: '12px', border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}></div>
                      Reading PDF & thinking...
                    </div>
                  )}
                </div>
                
                {/* Chat Input Field */}
                <form 
                  onSubmit={handleSendChatMessage}
                  style={{ display: 'flex', gap: '8px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px' }}
                >
                  <input 
                    type="text" 
                    className="glass-input"
                    placeholder="Ask about methodology, results..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    disabled={chatLoading}
                    style={{ flex: 1, fontSize: '12px', padding: '8px 12px' }}
                  />
                  <button 
                    type="submit" 
                    className="glass-button" 
                    disabled={chatLoading || !chatInput.trim()}
                    style={{ padding: '8px' }}
                  >
                    <ChevronRight size={18} />
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Simple bold parser helper
function parseBoldText(text) {
  if (!text) return "";
  const parts = text.split(/\*\*([^*]+)\*\*/g);
  return parts.map((part, index) => {
    return index % 2 === 1 ? <strong style={{ color: '#fff', fontWeight: '600' }} key={index}>{part}</strong> : part;
  });
}

// Simple chat message line splitter helper
function renderChatText(text) {
  if (!text) return null;
  const lines = text.split('\n');
  return lines.map((line, lIdx) => {
    let content = line.trim();
    if (!content) return <div key={lIdx} style={{ height: '4px' }} />;
    
    if (content.startsWith('-') || content.startsWith('*')) {
      const bulletContent = content.substring(1).trim();
      return (
        <div key={lIdx} style={{ marginLeft: '8px', display: 'flex', gap: '6px', marginBottom: '4px', fontSize: '13px', lineHeight: '1.4' }}>
          <span style={{ color: 'var(--accent-color)' }}>•</span>
          <span>{parseBoldText(bulletContent)}</span>
        </div>
      );
    }
    return (
      <p key={lIdx} style={{ margin: '0 0 6px 0', fontSize: '13px', lineHeight: '1.4' }}>
        {parseBoldText(content)}
      </p>
    );
  });
}
