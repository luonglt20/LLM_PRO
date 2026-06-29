import React, { useState, useEffect } from 'react';
import { Search, Folder, Plus, Trash2, FolderPlus, ThumbsUp, ThumbsDown, Bookmark } from 'lucide-react';

export default function Collections({ 
  ratings, 
  onRate, 
  allPapers, 
  collections, 
  onManageCollection 
}) {
  const [activeCollection, setActiveCollection] = useState('My Library');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [collectionRecs, setCollectionRecs] = useState([]);
  
  // Perform Search
  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    
    try {
      const response = await fetch('http://127.0.0.1:5001/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery })
      });
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (err) {
      // Local fallback search (word-boundary prefix matching)
      const queryWords = searchQuery.toLowerCase().split(' ').filter(Boolean);
      const localResults = allPapers.filter(p => {
        const text = (p.title + " " + p.abstract).toLowerCase();
        return queryWords.every(word => {
          const escapedWord = word.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
          const regex = new RegExp('\\b' + escapedWord);
          return regex.test(text);
        });
      });
      setSearchResults(localResults);
    }
  };

  // Create Collection
  const handleCreateCollection = () => {
    if (!newCollectionName.trim()) return;
    onManageCollection('create', newCollectionName);
    setActiveCollection(newCollectionName);
    setNewCollectionName('');
  };

  // Fetch proposals to expand collection
  const fetchCollectionRecs = async (colName) => {
    try {
      const response = await fetch(`http://127.0.0.1:5001/api/collections/${encodeURIComponent(colName)}/recommendations`);
      const data = await response.json();
      setCollectionRecs(data.papers || []);
    } catch (err) {
      setCollectionRecs([]);
    }
  };

  useEffect(() => {
    if (activeCollection) {
      fetchCollectionRecs(activeCollection);
    }
  }, [activeCollection, collections]);

  // Find papers inside active collection
  const getCollectionPapers = () => {
    const paperIds = collections[activeCollection] || [];
    return allPapers.filter(p => paperIds.includes(p.id));
  };

  const activePapers = getCollectionPapers();

  return (
    <div className="search-collections-layout">
      {/* Sidebar folders */}
      <div className="collections-sidebar glass-panel">
        <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Folder size={18} style={{ color: 'var(--accent-color)' }} /> Libraries
        </h3>
        
        {/* Create Folder Input */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <input 
            type="text" 
            className="glass-input" 
            placeholder="New folder..." 
            value={newCollectionName}
            onChange={(e) => setNewCollectionName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateCollection()}
            style={{ padding: '6px 12px', fontSize: '13px', width: '100%' }}
          />
          <button 
            className="action-btn" 
            onClick={handleCreateCollection}
            style={{ width: '32px', height: '32px' }}
          >
            <Plus size={14} />
          </button>
        </div>

        {/* Folder List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {Object.keys(collections).map(name => {
            const count = collections[name]?.length || 0;
            return (
              <div 
                key={name}
                className={`collection-item ${activeCollection === name ? 'active' : ''}`}
                onClick={() => setActiveCollection(name)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <FolderPlus size={14} style={{ color: activeCollection === name ? '#fff' : 'var(--text-secondary)' }} />
                  <span style={{ fontSize: '14px', fontWeight: '500' }}>{name}</span>
                </div>
                <span className="tag" style={{ fontSize: '11px', padding: '2px 8px' }}>{count}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Panel */}
      <div className="search-results-pane">
        {/* Search Input Bar */}
        <form onSubmit={handleSearch} className="search-bar-container glass-panel" style={{ padding: '12px' }}>
          <input 
            type="text" 
            className="glass-input" 
            placeholder="Search papers semantically or lexically (e.g. 'Language Agents for coding')..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ background: 'transparent', border: 'none', padding: '6px 12px' }}
          />
          <button className="glass-button" type="submit">
            <Search size={16} /> Search
          </button>
        </form>

        {/* Show Search Results if query exists, else show active collection contents */}
        {searchQuery ? (
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
            <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)' }}>
              Search Results ({searchResults.length})
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {searchResults.length === 0 ? (
                <p style={{ color: 'var(--text-secondary)' }}>No matching papers found.</p>
              ) : (
                searchResults.map(paper => {
                  const rating = ratings[paper.id] || 0;
                  const isBookmarked = collections[activeCollection]?.includes(paper.id);
                  
                  return (
                    <div 
                      key={paper.id} 
                      className="glass-panel" 
                      style={{ padding: '16px', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px' }}
                    >
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: '0 0 6px 0', color: '#fff', fontSize: '16px' }}>{paper.title}</h4>
                        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                          {paper.authors.slice(0, 3).join(', ')}
                        </p>
                        <p style={{ fontSize: '12px', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                          {paper.abstract}
                        </p>
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-end' }}>
                        <span className="tag category" style={{ fontSize: '11px' }}>{paper.category_name}</span>
                        <button 
                          className={`glass-button-secondary ${isBookmarked ? 'active' : ''}`}
                          style={{ fontSize: '11px', padding: '4px 10px' }}
                          onClick={() => onManageCollection(isBookmarked ? 'remove' : 'add', activeCollection, paper.id)}
                        >
                          <Bookmark size={12} /> {isBookmarked ? 'Bookmarked' : 'Add to Collection'}
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        ) : (
          /* Show Collection Contents */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
              <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)' }}>
                {activeCollection} ({activePapers.length} papers)
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {activePapers.length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)' }}>This library is empty. Use Search above or add papers from the Map!</p>
                ) : (
                  activePapers.map(paper => (
                    <div 
                      key={paper.id} 
                      className="glass-panel" 
                      style={{ padding: '16px', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                    >
                      <div>
                        <h4 style={{ margin: '0 0 4px 0', color: '#fff', fontSize: '15px' }}>{paper.title}</h4>
                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                          {paper.authors.slice(0, 2).join(', ')} | {paper.category_name}
                        </span>
                      </div>
                      <button 
                        className="action-btn"
                        onClick={() => onManageCollection('remove', activeCollection, paper.id)}
                        title="Remove from library"
                      >
                        <Trash2 size={14} style={{ color: 'var(--score-negative-text)' }} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Recommendations to Expand Collection */}
            {activePapers.length > 0 && (
              <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
                <h3 style={{ margin: '0 0 6px 0', fontFamily: 'var(--font-heading)', color: 'var(--accent-secondary)' }}>
                  Proposed Additions
                </h3>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  Recommendations based on the papers inside <strong>"{activeCollection}"</strong>:
                </p>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {collectionRecs.map(paper => (
                    <div 
                      key={paper.id} 
                      className="glass-panel" 
                      style={{ padding: '14px', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                    >
                      <div style={{ flex: 1, paddingRight: '16px' }}>
                        <h4 style={{ margin: '0 0 4px 0', color: '#fff', fontSize: '14px' }}>{paper.title}</h4>
                        <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{paper.category_name}</span>
                      </div>
                      <button 
                        className="glass-button" 
                        style={{ padding: '4px 10px', fontSize: '11px' }}
                        onClick={() => onManageCollection('add', activeCollection, paper.id)}
                      >
                        <Plus size={12} /> Add
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
