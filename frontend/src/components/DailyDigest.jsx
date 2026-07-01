import React, { useState, useEffect } from 'react';
import { 
  ThumbsUp, ThumbsDown, Bookmark, FileText, Compass, 
  X, Calendar, ChevronRight, Award, Clock, Sparkles, BookOpen
} from 'lucide-react';

export default function DailyDigest({ 
  papers, 
  ratings, 
  onRate, 
  onBookmark, 
  bookmarkedIds, 
  allPapers,
  geminiApiKey,
  onViewPdf
}) {
  const [selectedRange, setSelectedRange] = useState('day');
  const [startDate, setStartDate] = useState('2026-06-01');
  const [endDate, setEndDate] = useState('2026-06-30');
  
  // Modals / Drawers State
  const [activeBibtex, setActiveBibtex] = useState(null);
  const [activeSimilar, setActiveSimilar] = useState(null);

  const [similarRecommendations, setSimilarRecommendations] = useState([]);

  // AI Explanations State
  const [aiExplanations, setAiExplanations] = useState({});
  const [loadingAi, setLoadingAi] = useState({});

  const handleAskAi = async (paperId) => {
    setLoadingAi(prev => ({ ...prev, [paperId]: true }));
    try {
      const response = await fetch('http://127.0.0.1:5001/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_id: paperId, api_key: geminiApiKey })
      });
      const data = await response.json();
      setAiExplanations(prev => ({ ...prev, [paperId]: data }));
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingAi(prev => ({ ...prev, [paperId]: false }));
    }
  };

  // Highlight helper: puts a highlighted CSS span around the key sentence in the abstract
  const renderAbstract = (abstract, highlight) => {
    if (!highlight || !abstract) return abstract;
    const index = abstract.indexOf(highlight);
    if (index === -1) return abstract;
    
    return (
      <>
        {abstract.slice(0, index)}
        <span className="highlighted-sentence">{highlight}</span>
        {abstract.slice(index + highlight.length)}
      </>
    );
  };

  // Cosine similarity in JS for local "Similar Papers" check
  const fetchSimilarLocal = (targetPaper) => {
    if (!targetPaper.embedding) return [];
    
    const candidates = allPapers.filter(p => p.id !== targetPaper.id);
    const sims = candidates.map(p => {
      // Dot product (since embeddings are normalized TF-IDF)
      let dot = 0;
      for (let i = 0; i < targetPaper.embedding.length; i++) {
        dot += targetPaper.embedding[i] * p.embedding[i];
      }
      return { paper: p, similarity: dot };
    });
    
    // Sort and return top 5
    const sortedSims = sims.sort((a, b) => b.similarity - a.similarity);
    setSimilarRecommendations(sortedSims.slice(0, 5));
    setActiveSimilar(targetPaper);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      {/* Upper Control Bar */}
      <div className="digest-header glass-panel" style={{ padding: '16px 24px' }}>
        <div className="digest-title-group" style={{ textAlign: 'left' }}>
          <h1>Daily Digest</h1>
          <p>Ranked updates from open-access preprint servers based on your preferences</p>
        </div>
        
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div className="nav-tabs" style={{ background: 'rgba(0,0,0,0.2)', padding: '4px', borderRadius: '10px' }}>
            <button 
              className={`nav-tab ${selectedRange === 'day' ? 'active' : ''}`}
              onClick={() => setSelectedRange('day')}
              style={{ fontSize: '13px', padding: '8px 14px' }}
            >
              <Clock size={14} /> Daily
            </button>
            <button 
              className={`nav-tab ${selectedRange === 'week' ? 'active' : ''}`}
              onClick={() => setSelectedRange('week')}
              style={{ fontSize: '13px', padding: '8px 14px' }}
            >
              <Calendar size={14} /> Weekly
            </button>
          </div>
          
          <div className="date-picker-bar glass-panel" style={{ border: '1px solid rgba(255,255,255,0.05)' }}>
            <Calendar size={16} style={{ color: 'var(--accent-color)' }} />
            <input 
              type="date" 
              value={startDate} 
              onChange={(e) => setStartDate(e.target.value)} 
            />
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>to</span>
            <input 
              type="date" 
              value={endDate} 
              onChange={(e) => setEndDate(e.target.value)} 
            />
          </div>
        </div>
      </div>

      {/* Recommended Papers List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {papers.length === 0 ? (
          <div className="glass-panel" style={{ padding: '60px', textAlign: 'center' }}>
            <Award size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
            <h3 style={{ margin: '0 0 8px 0', fontFamily: 'var(--font-heading)' }}>No Recommendations Yet</h3>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto' }}>
              Try rating some papers in the **Scholar Map** or adding your research interests to train the recommender!
            </p>
          </div>
        ) : (
          papers.map((paper) => {
            const userRating = ratings[paper.id] || 0;
            const isBookmarked = bookmarkedIds.includes(paper.id);
            const score = paper.relevance_score;
            
            let scoreClass = 'neutral';
            if (score > 15.0) scoreClass = 'positive';
            else if (score < -15.0) scoreClass = 'negative';
            
            return (
              <div key={paper.id} className="paper-card glass-panel">
                <div className="paper-card-header">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
                    <h2 className="paper-title">{paper.title}</h2>
                    <div className="paper-authors">
                      {paper.authors.slice(0, 5).join(', ')}{paper.authors.length > 5 ? ' et al.' : ''}
                    </div>
                    <div className="paper-meta">
                      <span className={`score-badge ${scoreClass}`}>
                        {score > 0 ? `+${score}` : score} Relevance
                      </span>
                      <span className="tag category">{paper.category_name}</span>
                      <span className="tag" style={{ color: 'var(--text-secondary)' }}>{paper.published}</span>
                    </div>
                  </div>
                  
                  {/* Paper Card Actions */}
                  <div className="paper-actions">
                    <button 
                      className={`action-btn upvote ${userRating === 1 ? 'upvoted' : ''}`}
                      onClick={() => onRate(paper.id, userRating === 1 ? 0 : 1)}
                      title="Upvote (This is relevant)"
                    >
                      <ThumbsUp size={16} />
                    </button>
                    <button 
                      className={`action-btn downvote ${userRating === -1 ? 'downvoted' : ''}`}
                      onClick={() => onRate(paper.id, userRating === -1 ? 0 : -1)}
                      title="Downvote (This is not relevant)"
                    >
                      <ThumbsDown size={16} />
                    </button>
                    <button 
                      className={`action-btn ${isBookmarked ? 'bookmarked' : ''}`}
                      onClick={() => onBookmark(paper.id)}
                      title="Bookmark to Collection"
                    >
                      <Bookmark size={16} />
                    </button>
                  </div>
                </div>
                
                {/* Highlighted Abstract */}
                <div className="paper-abstract">
                  {renderAbstract(paper.abstract, paper.highlight)}
                </div>
                
                {/* Paper Footer with Drawers Links */}
                <div className="paper-card-footer">
                  <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                    Source: arXiv preprint
                  </span>
                  
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => setActiveBibtex(paper)}
                    >
                      <FileText size={14} /> BibTeX
                    </button>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => onViewPdf(paper)}
                    >
                      <BookOpen size={14} /> View PDF
                    </button>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => fetchSimilarLocal(paper)}
                    >
                      <Compass size={14} /> Similar Papers
                    </button>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px', borderColor: 'rgba(139,92,246,0.3)' }}
                      onClick={() => handleAskAi(paper.id)}
                      disabled={loadingAi[paper.id]}
                    >
                      <Sparkles size={14} style={{ color: 'var(--accent-color)' }} /> 
                      {loadingAi[paper.id] ? 'Analyzing...' : 'Ask AI'}
                    </button>
                  </div>
                </div>

                {/* AI Explanation Drawer/Box */}
                {aiExplanations[paper.id] && (
                  <div 
                    className="glass-panel" 
                    style={{
                      marginTop: '16px',
                      padding: '16px',
                      background: 'rgba(139, 92, 246, 0.04)',
                      borderLeft: '4px solid var(--accent-color)',
                      fontSize: '13px',
                      textAlign: 'left'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                      <span style={{ fontWeight: '700', color: 'var(--accent-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Sparkles size={14} /> AI Recommendation Insight
                      </span>
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                        powered by {aiExplanations[paper.id].source}
                      </span>
                    </div>
                    <p style={{ color: '#fff', marginBottom: '8px', lineHeight: '1.4' }}>
                      <strong>Relevance:</strong> {aiExplanations[paper.id].explanation}
                    </p>
                    <p style={{ color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
                      <strong>Tailored Summary:</strong> {aiExplanations[paper.id].tailored_summary}
                    </p>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* BibTeX Modal */}
      {activeBibtex && (
        <div className="modal-overlay" onClick={() => setActiveBibtex(null)}>
          <div className="modal-content glass-panel" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setActiveBibtex(null)}><X size={20} /></button>
            <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)' }}>Export BibTeX Reference</h3>
            <pre style={{
              background: '#0a0c14',
              padding: '16px',
              borderRadius: '8px',
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              color: 'var(--accent-secondary)',
              textAlign: 'left',
              overflowX: 'auto',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              {activeBibtex.bibtex}
            </pre>
            <button 
              className="glass-button" 
              style={{ marginTop: '16px', width: '100%', justifyContent: 'center' }}
              onClick={() => {
                navigator.clipboard.writeText(activeBibtex.bibtex);
                alert("BibTeX copied to clipboard!");
              }}
            >
              Copy Reference
            </button>
          </div>
        </div>
      )}



      {/* Similar Papers Drawer */}
      <div className={`similar-drawer glass-panel ${activeSimilar ? 'open' : ''}`}>
        <div className="similar-drawer-header">
          <h3 style={{ margin: 0, fontFamily: 'var(--font-heading)' }}>Similar Papers</h3>
          <button className="action-btn" onClick={() => setActiveSimilar(null)}>
            <X size={16} />
          </button>
        </div>
        
        {activeSimilar && (
          <div style={{ textAlign: 'left', fontSize: '13px' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Recommendations similar to <strong>"{activeSimilar.title}"</strong>
            </p>
          </div>
        )}
        
        <div className="similar-drawer-list">
          {similarRecommendations.map(({ paper, similarity }) => (
            <div 
              key={paper.id} 
              className="glass-panel" 
              style={{ padding: '14px', background: 'rgba(255,255,255,0.01)' }}
            >
              <h4 style={{ margin: '0 0 6px 0', fontSize: '14px', color: '#fff', textAlign: 'left', lineHeight: '1.4' }}>
                {paper.title}
              </h4>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>
                <span>Cosine Sim: {(similarity * 100).toFixed(1)}%</span>
                <button 
                  className="glass-button-secondary"
                  style={{ fontSize: '10px', padding: '2px 8px' }}
                  onClick={() => onViewPdf(paper)}
                >
                  <BookOpen size={10} /> View PDF
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
