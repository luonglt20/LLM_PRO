import React, { useState, useEffect } from 'react';
import { Calendar, Bookmark, Clock, MapPin, Award, Trash2 } from 'lucide-react';

export default function ConferencePlanner({ ratings, allPapers }) {
  const [sessions, setSessions] = useState([]);
  const [expandedSession, setExpandedSession] = useState(null);
  const [bookmarkedPosters, setBookmarkedPosters] = useState([]);

  // Fetch poster sessions from backend
  const fetchSessions = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/api/conference');
      const data = await response.json();
      setSessions(data.sessions || []);
      if (data.sessions?.length > 0 && !expandedSession) {
        setExpandedSession(data.sessions[0].id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, [ratings]);

  // Handle bookmarking poster to planner schedule
  const toggleBookmarkPoster = (paper) => {
    setBookmarkedPosters(prev => {
      const exists = prev.find(p => p.id === paper.id);
      if (exists) {
        return prev.filter(p => p.id !== paper.id);
      } else {
        // Find matching session for time details
        const session = sessions.find(s => s.papers.some(p => p.id === paper.id));
        return [...prev, { ...paper, time: session?.time || "TBD", sessionName: session?.name || "Poster Session" }];
      }
    });
  };

  const isPosterBookmarked = (paperId) => bookmarkedPosters.some(p => p.id === paperId);

  // Group bookmarked posters by time slots
  const groupedBookmarks = bookmarkedPosters.reduce((groups, poster) => {
    const time = poster.time;
    if (!groups[time]) groups[time] = [];
    groups[time].push(poster);
    return groups;
  }, {});

  return (
    <div className="conf-layout">
      {/* Sessions Left Pane */}
      <div className="conf-sessions-list">
        <div className="glass-panel" style={{ padding: '16px 24px', textAlign: 'left', marginBottom: '8px' }}>
          <h2 style={{ margin: '0 0 6px 0', fontFamily: 'var(--font-heading)' }}>Conference Poster Planner</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '13px', margin: 0 }}>
            Sessions at <strong>ScholarConf 2026</strong> are ranked by your interests. Browse papers and build your itinerary.
          </p>
        </div>

        {sessions.map(session => {
          const isExpanded = expandedSession === session.id;
          
          let scoreColorClass = 'low';
          if (session.avg_relevance > 10) scoreColorClass = 'high';
          else if (session.avg_relevance > -10) scoreColorClass = 'medium';
          
          return (
            <div key={session.id} className="conf-session-card glass-panel" style={{ textAlign: 'left' }}>
              <div 
                className="conf-session-header" 
                style={{ cursor: 'pointer', borderBottom: isExpanded ? '1px solid rgba(255,255,255,0.06)' : 'none' }}
                onClick={() => setExpandedSession(isExpanded ? null : session.id)}
              >
                <div style={{ flex: 1, paddingRight: '16px' }}>
                  <h3 className="conf-session-title">{session.name}</h3>
                  <div className="conf-session-time">
                    <Clock size={13} style={{ color: 'var(--accent-secondary)' }} /> {session.time}
                    <span style={{ color: 'var(--text-muted)' }}>|</span>
                    <MapPin size={13} style={{ color: 'var(--text-muted)' }} /> Exhibition Hall B
                  </div>
                </div>
                
                <div className="conf-session-relevance">
                  <span className={`relevance-score-text ${scoreColorClass}`}>
                    {session.avg_relevance > 0 ? `+${session.avg_relevance}` : session.avg_relevance}
                  </span>
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: '600' }}>AVG RELEVANCE</span>
                </div>
              </div>
              
              {/* Session posters list */}
              {isExpanded && (
                <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {session.papers.slice(0, 10).map((paper, idx) => {
                    const isBookmarked = isPosterBookmarked(paper.id);
                    const score = paper.relevance_score;
                    
                    return (
                      <div 
                        key={paper.id}
                        className="glass-panel" 
                        style={{
                          padding: '12px 16px',
                          background: 'rgba(255,255,255,0.01)',
                          border: '1px solid rgba(255,255,255,0.03)',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}
                      >
                        <div style={{ flex: 1, paddingRight: '16px' }}>
                          <h4 style={{ margin: '0 0 4px 0', fontSize: '14px', color: '#fff', fontWeight: '600' }}>
                            <span style={{ color: 'var(--text-muted)', marginRight: '6px' }}>#{idx+1}</span>
                            {paper.title}
                          </h4>
                          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                            {paper.authors[0]} | <span style={{ color: score > 0 ? '#34d399' : '#9ca3af', fontWeight: '500' }}>Match: {score > 0 ? `+${score}` : score}</span>
                          </div>
                        </div>
                        
                        <button 
                          className={`action-btn ${isBookmarked ? 'bookmarked' : ''}`}
                          onClick={() => toggleBookmarkPoster(paper)}
                          title={isBookmarked ? "Remove from Schedule" : "Add to Schedule"}
                        >
                          <Bookmark size={14} />
                        </button>
                      </div>
                    );
                  })}
                  {session.papers.length > 10 && (
                    <p style={{ color: 'var(--text-muted)', fontSize: '11px', textAlign: 'center', marginTop: '6px' }}>
                      Showing top 10 of {session.papers.length} posters in this session.
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Schedule Right Pane */}
      <div className="conf-schedule-pane glass-panel">
        <h3 style={{ margin: '0 0 8px 0', fontFamily: 'var(--font-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Calendar size={18} style={{ color: 'var(--accent-color)' }} /> My Itinerary
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '12px', margin: '0 0 20px 0', textAlign: 'left', lineHeight: '1.4' }}>
          Select posters to visit. Your itinerary will be sorted chronologically by time slots.
        </p>

        {bookmarkedPosters.length === 0 ? (
          <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Award size={36} style={{ marginBottom: '12px', opacity: 0.5 }} />
            <p style={{ fontSize: '13px' }}>Your poster schedule is empty.</p>
          </div>
        ) : (
          <div className="schedule-timeline">
            {Object.keys(groupedBookmarks).map(timeSlot => (
              <div key={timeSlot} className="timeline-item">
                <div className="timeline-details">
                  <div className="timeline-time">{timeSlot}</div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px', width: '100%' }}>
                    {groupedBookmarks[timeSlot].map(poster => (
                      <div 
                        key={poster.id} 
                        style={{
                          background: 'rgba(255,255,255,0.02)',
                          padding: '10px 12px',
                          borderRadius: '8px',
                          border: '1px solid rgba(255,255,255,0.04)',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          gap: '12px'
                        }}
                      >
                        <div style={{ flex: 1 }}>
                          <div className="timeline-title" style={{ fontSize: '13px', lineHeight: '1.3' }}>
                            {poster.title}
                          </div>
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                            Exhibition Booth {poster.id.substring(0, 4).toUpperCase()}
                          </span>
                        </div>
                        
                        <button 
                          className="active-learning-btn"
                          onClick={() => toggleBookmarkPoster(poster)}
                          style={{ padding: '4px' }}
                        >
                          <Trash2 size={12} style={{ color: 'var(--score-negative-text)' }} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
