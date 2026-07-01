import React, { useState } from 'react';
import { BookOpen, Bookmark, ThumbsUp, ThumbsDown, Sparkles, BrainCircuit } from 'lucide-react';

const workspaceLayoutStyles = {
  container: {
    display: 'flex',
    flexDirection: 'row',
    gap: '24px',
    height: 'calc(100vh - 120px)',
    width: '100%',
    boxSizing: 'border-box',
    overflow: 'hidden',
    textAlign: 'left'
  },
  leftPanel: {
    flex: '1',
    minWidth: '320px',
    maxWidth: '420px',
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    boxSizing: 'border-box',
    padding: '24px'
  },
  rightColumn: {
    flex: '2',
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
    height: '100%',
    boxSizing: 'border-box',
    overflowY: 'auto',
    paddingRight: '8px'
  },
  inputCard: {
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
    padding: '24px'
  },
  form: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
    width: '100%',
    marginTop: '8px'
  },
  input: {
    flex: '1',
    boxSizing: 'border-box',
    height: '40px'
  },
  submitBtn: {
    height: '40px',
    padding: '0 24px',
    borderRadius: '8px',
    background: 'var(--accent-color)',
    color: '#fff',
    border: 'none',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background 0.2s ease'
  },
  reportCard: {
    flex: '1',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
    overflow: 'hidden',
    padding: '24px',
    minHeight: '450px'
  },
  logsContainer: {
    flex: '1',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    paddingRight: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px'
  },
  reportContainer: {
    flex: '1',
    overflowY: 'auto',
    paddingRight: '8px'
  }
};

// Helper function to parse and render styled markdown elements
function renderMarkdown(text) {
  if (!text) return null;
  
  const lines = text.split('\n');
  let inTable = false;
  let tableHeader = null;
  let tableRows = [];
  
  const renderedElements = [];
  
  const parseBoldText = (txt) => {
    if (!txt) return "";
    const parts = txt.split(/\*\*([^*]+)\*\*/g);
    return parts.map((part, index) => {
      return index % 2 === 1 ? (
        <strong style={{ color: '#fff', fontWeight: '600' }} key={index}>{part}</strong>
      ) : part;
    });
  };
  
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    
    // Check for table rows
    if (line.startsWith('|')) {
      inTable = true;
      const cells = line.split('|').map(c => c.trim()).filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
      
      if (line.includes('---')) {
        // Separator row, skip
        continue;
      }
      
      if (!tableHeader) {
        tableHeader = cells;
      } else {
        tableRows.push(cells);
      }
      continue;
    } else {
      // If table ended, render it first
      if (inTable && tableHeader) {
        const headerCells = [...tableHeader];
        const rowsCells = [...tableRows];
        
        renderedElements.push(
          <div style={{ overflowX: 'auto', margin: '16px 0' }} key={`table-${i}`}>
            <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid rgba(255,255,255,0.08)', fontSize: '12px' }}>
              <thead>
                <tr style={{ background: 'rgba(255,255,255,0.04)' }}>
                  {headerCells.map((h, idx) => (
                    <th key={idx} style={{ padding: '8px 12px', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'left', fontWeight: '700', color: '#fff' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rowsCells.map((row, rIdx) => (
                  <tr key={rIdx} style={{ background: rIdx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent' }}>
                    {row.map((c, cIdx) => (
                      <td key={cIdx} style={{ padding: '8px 12px', border: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>{parseBoldText(c)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        inTable = false;
        tableHeader = null;
        tableRows = [];
      }
    }
    
    if (!line) {
      renderedElements.push(<div style={{ height: '8px' }} key={`space-${i}`} />);
      continue;
    }
    
    // Check for headings
    if (line.startsWith('###')) {
      renderedElements.push(
        <h4 style={{ margin: '18px 0 8px 0', color: 'var(--accent-secondary)', fontWeight: '700', fontSize: '13px' }} key={i}>
          {line.replace('###', '').trim()}
        </h4>
      );
    } else if (line.startsWith('##')) {
      renderedElements.push(
        <h3 style={{ margin: '22px 0 12px 0', color: '#fff', fontWeight: '700', fontSize: '15px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }} key={i}>
          {line.replace('##', '').trim()}
        </h3>
      );
    } else if (line.startsWith('#')) {
      renderedElements.push(
        <h2 style={{ margin: '24px 0 16px 0', color: '#fff', fontWeight: '800', fontSize: '18px' }} key={i}>
          {line.replace('#', '').trim()}
        </h2>
      );
    } else if (line.startsWith('-') || line.startsWith('*')) {
      const content = line.substring(1).trim();
      renderedElements.push(
        <li style={{ marginLeft: '16px', marginBottom: '6px', listStyleType: 'disc', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }} key={i}>
          {parseBoldText(content)}
        </li>
      );
    } else if (/^\d+\./.test(line)) {
      const content = line.replace(/^\d+\./, '').trim();
      renderedElements.push(
        <li style={{ marginLeft: '16px', marginBottom: '6px', listStyleType: 'decimal', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }} key={i}>
          {parseBoldText(content)}
        </li>
      );
    } else {
      renderedElements.push(
        <p style={{ margin: '0 0 10px 0', fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }} key={i}>
          {parseBoldText(line)}
        </p>
      );
    }
  }
  
  return renderedElements;
}

export default function ResearchWorkspace({ 
  apiKey,
  onViewPdf,
  ratings,
  onRate,
  onBookmark,
  bookmarkedIds
}) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  const [report, setReport] = useState('');
  const [matchedPapers, setMatchedPapers] = useState([]);
  const [error, setError] = useState('');

  const handleRunAgents = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setLogs([]);
    setReport('');
    setMatchedPapers([]);

    // Add initial log
    setLogs([
      { agent: 'Orchestrator', message: 'Initializing multi-agent workflow state graph...' }
    ]);

    try {
      const res = await fetch('http://localhost:5001/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, api_key: apiKey }),
      });

      if (!res.ok) {
        throw new Error('Failed to execute Multi-Agent research workflow');
      }

      const data = await res.json();
      setLogs(data.logs || []);
      setReport(data.report || '');
      setMatchedPapers(data.papers || []);
    } catch (err) {
      setError(err.message || 'An error occurred during agent execution.');
      setLogs(prev => [
        ...prev,
        { agent: 'Orchestrator', message: 'Error: Agent workflow execution failed.' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyReport = () => {
    if (!report) return;
    navigator.clipboard.writeText(report);
    alert('Report copied to clipboard!');
  };

  return (
    <div style={workspaceLayoutStyles.container}>
      {/* Left Panel: Agent Log Console */}
      <div className="glass-panel" style={workspaceLayoutStyles.leftPanel}>
        <h3 style={{ fontSize: '16px', fontWeight: 'bold', color: '#fff', margin: '0 0 16px 0', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BrainCircuit size={18} style={{ color: 'var(--accent-color)' }} />
          Agent Execution Monitor
        </h3>
        <div style={workspaceLayoutStyles.logsContainer}>
          {logs.length === 0 ? (
            <div style={{ color: 'rgba(255,255,255,0.3)', textAlign: 'center', padding: '48px 0' }}>
              Waiting for agent activation...
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                style={{
                  background: 'rgba(0,0,0,0.2)',
                  border: '1px solid rgba(255,255,255,0.05)',
                  borderRadius: '6px',
                  padding: '12px'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'between', marginBottom: '6px' }}>
                  <span style={{
                    fontSize: '10px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontWeight: 'bold',
                    background: log.agent === 'ArxivSearchAgent' ? 'rgba(59, 130, 246, 0.2)' :
                                log.agent === 'PaperCriticAgent' ? 'rgba(139, 92, 246, 0.2)' :
                                log.agent === 'LiteratureReviewAgent' ? 'rgba(16, 185, 129, 0.2)' :
                                'rgba(255,255,255,0.1)',
                    color: log.agent === 'ArxivSearchAgent' ? '#93c5fd' :
                           log.agent === 'PaperCriticAgent' ? '#c084fc' :
                           log.agent === 'LiteratureReviewAgent' ? '#34d399' :
                           'rgba(255,255,255,0.7)'
                  }}>
                    {log.agent}
                  </span>
                </div>
                <p style={{ color: 'rgba(255,255,255,0.85)', fontSize: '11px', lineHeight: '1.5', margin: 0 }}>
                  {log.message}
                </p>
              </div>
            ))
          )}
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'rgba(255,255,255,0.4)', fontSize: '11px', padding: '8px 0' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-color)' }} className="spinner"></div>
              <span>Agent is working...</span>
            </div>
          )}
        </div>
      </div>

      {/* Right Panels: Query Input, Source Papers, and Report Draft */}
      <div style={workspaceLayoutStyles.rightColumn}>
        {/* Query Input Card */}
        <div className="glass-panel" style={workspaceLayoutStyles.inputCard}>
          <h3 style={{ fontSize: '16px', fontWeight: 'bold', color: '#fff', margin: '0 0 8px 0' }}>
            AI Research Workspace
          </h3>
          <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px', margin: '0 0 16px 0', lineHeight: '1.4' }}>
            Enter a research topic or draft query. The specialized agents will search the database, compile individual paper critiques, and synthesize a complete Phased Research Roadmap.
          </p>
          <form onSubmit={handleRunAgents} style={workspaceLayoutStyles.form}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Robot arm path planning or reinforcement learning policy"
              className="glass-input"
              style={workspaceLayoutStyles.input}
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              style={workspaceLayoutStyles.submitBtn}
            >
              {loading ? 'Executing...' : 'Execute Agents'}
            </button>
          </form>
          {error && (
            <p style={{ color: '#f87171', fontSize: '12px', marginTop: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '8px', borderRadius: '6px', margin: '12px 0 0 0' }}>
              {error}
            </p>
          )}
        </div>

        {/* Matched Source Papers Card */}
        {matchedPapers.length > 0 && (
          <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <h4 style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: 'bold', color: 'var(--accent-color)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BookOpen size={16} />
              Reviewed Source Papers ({matchedPapers.length})
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
              {matchedPapers.map(paper => {
                const isBookmarked = bookmarkedIds.includes(paper.id);
                const currentRating = ratings[paper.id] || 0;
                
                return (
                  <div 
                    key={paper.id} 
                    className="glass-panel" 
                    style={{ 
                      padding: '16px', 
                      background: 'rgba(255,255,255,0.01)', 
                      display: 'flex', 
                      flexDirection: 'column', 
                      justifyContent: 'space-between',
                      gap: '12px'
                    }}
                  >
                    <div>
                      <h5 style={{ margin: '0 0 6px 0', fontSize: '14px', color: '#fff', fontWeight: '600', lineHeight: '1.4', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }} title={paper.title}>
                        {paper.title}
                      </h5>
                      <span className="tag category" style={{ fontSize: '10px' }}>{paper.category_name}</span>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: 'auto' }}>
                      <button 
                        className="glass-button"
                        style={{ fontSize: '11px', padding: '6px 12px', flex: 1, justifyContent: 'center' }}
                        onClick={() => onViewPdf(paper)}
                      >
                        <BookOpen size={13} style={{ marginRight: '4px' }} /> View PDF & AI Chat
                      </button>
                      
                      <button 
                        className={`action-btn ${isBookmarked ? 'bookmarked' : ''}`}
                        style={{ width: '32px', height: '32px', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '6px', background: isBookmarked ? 'rgba(139,92,246,0.2)' : 'transparent', color: isBookmarked ? '#a78bfa' : 'var(--text-secondary)' }}
                        onClick={() => onBookmark(paper.id)}
                        title="Add to Library"
                      >
                        <Bookmark size={13} />
                      </button>
                      
                      <button 
                        className={`action-btn upvote ${currentRating === 1 ? 'upvoted' : ''}`}
                        onClick={() => onRate(paper.id, currentRating === 1 ? 0 : 1)}
                        style={{ width: '32px', height: '32px', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '6px', background: currentRating === 1 ? 'rgba(52,211,153,0.2)' : 'transparent', color: currentRating === 1 ? '#34d399' : 'var(--text-secondary)' }}
                        title="Upvote paper category"
                      >
                        <ThumbsUp size={13} />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Generated Report Card */}
        <div className="glass-panel" style={workspaceLayoutStyles.reportCard}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>
            <h4 style={{ fontSize: '14px', fontWeight: 'bold', color: '#fff', margin: 0, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sparkles size={16} style={{ color: 'var(--accent-color)' }} />
              Synthesized Research Roadmap
            </h4>
            {report && (
              <button
                onClick={handleCopyReport}
                className="glass-button"
                style={{ padding: '4px 12px', fontSize: '11px' }}
              >
                Copy Roadmap
              </button>
            )}
          </div>
          <div style={workspaceLayoutStyles.reportContainer}>
            {report ? (
              <div style={{ color: 'rgba(255,255,255,0.9)', fontSize: '13px', fontFamily: 'var(--font-sans)', lineHeight: '1.6', margin: 0, userSelect: 'text' }}>
                {renderMarkdown(report)}
              </div>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.3)', textAlign: 'center', padding: '80px 0', fontSize: '13px' }}>
                {loading ? 'Synthesizing roadmap, please wait...' : 'The Research Roadmap & Timeline will appear here once the agents complete execution.'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
