import React, { useState } from 'react';

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
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-140px)]">
      {/* Left Panel: Agent Log Console */}
      <div className="glass-panel p-6 flex flex-col h-full overflow-hidden">
        <h3 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">
          Agent Execution Monitor
        </h3>
        <div className="flex-1 overflow-y-auto space-y-4 pr-2 font-mono text-sm">
          {logs.length === 0 ? (
            <div className="text-white/40 text-center py-12">
              Waiting for agent activation...
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className="bg-black/20 border border-white/5 rounded p-3"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-xs px-2 py-0.5 rounded font-bold ${
                    log.agent === 'ArxivSearchAgent' ? 'bg-blue-500/20 text-blue-300' :
                    log.agent === 'PaperCriticAgent' ? 'bg-purple-500/20 text-purple-300' :
                    log.agent === 'LiteratureReviewAgent' ? 'bg-emerald-500/20 text-emerald-300' :
                    'bg-white/10 text-white/70'
                  }`}>
                    {log.agent}
                  </span>
                </div>
                <p className="text-white/80 text-xs leading-relaxed">{log.message}</p>
              </div>
            ))
          )}
          {loading && (
            <div className="flex items-center space-x-2 text-white/50 text-xs py-2 animate-pulse">
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-ping"></div>
              <span>Agent is working...</span>
            </div>
          )}
        </div>
      </div>

      {/* Right Panels: Query Input and Report Draft */}
      <div className="lg:col-span-2 flex flex-col gap-6 h-full overflow-hidden">
        {/* Query Input Card */}
        <div className="glass-panel p-6">
          <h3 className="text-lg font-bold text-white mb-3">
            AI Research Workspace
          </h3>
          <p className="text-white/60 text-xs mb-4">
            Enter a research topic or draft query. The specialized agents will search the database, compile individual paper critiques, and synthesize a complete Literature Review report.
          </p>
          <form onSubmit={handleRunAgents} className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Dexterous robot hand manipulation or continual fine-tuning"
              className="flex-1 bg-black/40 border border-white/10 rounded px-4 py-2 text-white placeholder-white/30 text-sm focus:outline-none focus:border-purple-500/50"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-purple-600 hover:bg-purple-500 disabled:bg-white/10 disabled:text-white/40 text-white font-bold text-sm px-6 py-2 rounded transition-colors"
            >
              {loading ? 'Executing...' : 'Execute Agents'}
            </button>
          </form>
          {error && (
            <p className="text-red-400 text-xs mt-3 bg-red-500/10 border border-red-500/20 rounded p-2">
              {error}
            </p>
          )}
        </div>

        {/* Generated Report Card */}
        <div className="glass-panel p-6 flex-1 flex flex-col overflow-hidden">
          <div className="flex items-center justify-between mb-4 border-b border-white/10 pb-2">
            <h4 className="text-sm font-bold text-white">
              Synthesized Report (Literature Review)
            </h4>
            {report && (
              <button
                onClick={handleCopyReport}
                className="bg-white/10 hover:bg-white/20 text-white/90 text-xs font-bold px-3 py-1 rounded transition-colors"
              >
                Copy Review
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto pr-2">
            {report ? (
              <pre className="text-white/90 text-sm font-sans whitespace-pre-wrap leading-relaxed select-text">
                {report}
              </pre>
            ) : (
              <div className="text-white/30 text-center py-20 text-sm font-sans">
                {loading ? 'Synthesizing report, please wait...' : 'The Literature Review will appear here once the agents complete execution.'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
