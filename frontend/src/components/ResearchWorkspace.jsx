import React, { useState } from 'react';

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
    overflow: 'hidden'
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
    padding: '24px'
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

export default function ResearchWorkspace({ apiKey }) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  const [report, setReport] = useState('');
  const [error, setError] = useState('');

  const handleRunAgents = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setLogs([]);
    setReport('');

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
        <h3 style={{ fontSize: '16px', fontWeight: 'bold', color: '#fff', margin: '0 0 16px 0', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>
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

      {/* Right Panels: Query Input and Report Draft */}
      <div style={workspaceLayoutStyles.rightColumn}>
        {/* Query Input Card */}
        <div className="glass-panel" style={workspaceLayoutStyles.inputCard}>
          <h3 style={{ fontSize: '16px', fontWeight: 'bold', color: '#fff', margin: '0 0 8px 0' }}>
            AI Research Workspace
          </h3>
          <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px', margin: '0 0 16px 0', lineHeight: '1.4' }}>
            Enter a research topic or draft query. The specialized agents will search the database, compile individual paper critiques, and synthesize a complete Literature Review report.
          </p>
          <form onSubmit={handleRunAgents} style={workspaceLayoutStyles.form}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Dexterous robot hand manipulation or continual fine-tuning"
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

        {/* Generated Report Card */}
        <div className="glass-panel" style={workspaceLayoutStyles.reportCard}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>
            <h4 style={{ fontSize: '14px', fontWeight: 'bold', color: '#fff', margin: 0 }}>
              Synthesized Report (Literature Review)
            </h4>
            {report && (
              <button
                onClick={handleCopyReport}
                className="glass-button"
                style={{ padding: '4px 12px', fontSize: '11px' }}
              >
                Copy Review
              </button>
            )}
          </div>
          <div style={workspaceLayoutStyles.reportContainer}>
            {report ? (
              <pre style={{ color: 'rgba(255,255,255,0.9)', fontSize: '13px', fontFamily: 'var(--font-sans)', whiteSpace: 'pre-wrap', lineHeight: '1.6', margin: 0, userSelect: 'text' }}>
                {report}
              </pre>
            ) : (
              <div style={{ color: 'rgba(255,255,255,0.3)', textAlign: 'center', padding: '80px 0', fontSize: '13px' }}>
                {loading ? 'Synthesizing report, please wait...' : 'The Literature Review will appear here once the agents complete execution.'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
