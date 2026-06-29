import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, BrainCircuit, ThumbsUp, ThumbsDown } from 'lucide-react';

const CATEGORY_COLORS = {
  "Computer Vision": "#06b6d4",
  "Natural Language Processing": "#a78bfa",
  "Machine Learning": "#34d399",
  "Artificial Intelligence": "#ec4899",
  "Other": "#9ca3af"
};

export default function ScholarMap({ papers, ratings, onRate, activeLearningPapers }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  
  // Transform State
  const [scale, setScale] = useState(1.0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  // Interaction State
  const [hoveredPaper, setHoveredPaper] = useState(null);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Calculate centroids of categories dynamically for the outermost labels
  const getCentroids = () => {
    const centroids = {};
    papers.forEach(p => {
      const cat = p.category_name || "Other";
      if (!centroids[cat]) {
        centroids[cat] = { x: 0, y: 0, count: 0 };
      }
      centroids[cat].x += p.x;
      centroids[cat].y += p.y;
      centroids[cat].count += 1;
    });
    
    return Object.keys(centroids).map(name => ({
      name,
      x: centroids[name].x / centroids[name].count,
      y: centroids[name].y / centroids[name].count
    }));
  };

  // Reset View
  const handleReset = () => {
    setScale(1.0);
    setOffset({ x: 0, y: 0 });
    setHoveredPaper(null);
    setSelectedPaper(null);
  };

  // Zoom Helpers
  const handleZoom = (factor) => {
    setScale(prev => Math.min(Math.max(prev * factor, 0.4), 10.0));
  };

  // Convert map coordinates [-100, 100] to canvas screen space coordinates
  const mapToScreen = (mx, my, width, height) => {
    // Center of canvas
    const cx = width / 2;
    const cy = height / 2;
    
    // Scale mapping (at scale=1, [-100, 100] spans 80% of min dimension)
    const viewSize = Math.min(width, height) * 0.4;
    
    const sx = cx + offset.x + (mx / 100) * viewSize * scale;
    const sy = cy + offset.y + (my / 100) * viewSize * scale;
    
    return { x: sx, y: sy };
  };

  // Convert screen coordinates back to map space
  const screenToMap = (sx, sy, width, height) => {
    const cx = width / 2;
    const cy = height / 2;
    const viewSize = Math.min(width, height) * 0.4;
    
    const mx = ((sx - cx - offset.x) / (viewSize * scale)) * 100;
    const my = ((sy - cy - offset.y) / (viewSize * scale)) * 100;
    
    return { x: mx, y: my };
  };

  // Canvas Drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Set resolution
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const w = rect.width;
    const h = rect.height;
    
    ctx.clearRect(0, 0, w, h);
    
    // 1. Draw grid backdrop lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    const gridStep = 50 * scale;
    const startX = (offset.x % gridStep);
    const startY = (offset.y % gridStep);
    
    for (let x = startX; x < w; x += gridStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = startY; y < h; y += gridStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // 2. Draw decision boundary if trained
    const ratedCount = Object.keys(ratings).length;
    if (ratedCount > 0) {
      // Draw simple glowing concentric orbits representing probability rings
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.06)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const origin = mapToScreen(0, 0, w, h);
      ctx.arc(origin.x, origin.y, 120 * scale, 0, 2 * Math.PI);
      ctx.stroke();
      
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.03)';
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, 220 * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }
    
    // 3. Draw paper dots
    papers.forEach(paper => {
      const { x: sx, y: sy } = mapToScreen(paper.x, paper.y, w, h);
      
      // Filter out of bounds dots to optimize drawing
      if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) return;
      
      const isRated = ratings[paper.id];
      const category = paper.category_name || "Other";
      const color = CATEGORY_COLORS[category];
      
      // Determine radius based on relevance score
      let r = 5;
      if (scale > 2) r = 6;
      if (scale > 5) r = 7;
      
      // Make rated papers stand out visually
      if (isRated) {
        ctx.shadowBlur = 10;
        ctx.shadowColor = isRated === 1 ? '#34d399' : '#f87171';
        ctx.fillStyle = isRated === 1 ? '#34d399' : '#f87171';
        r += 2;
      } else {
        ctx.shadowBlur = 0;
        ctx.fillStyle = color;
      }
      
      // Highlight hovered
      const isHovered = hoveredPaper && hoveredPaper.id === paper.id;
      const isSelected = selectedPaper && selectedPaper.id === paper.id;
      
      if (isHovered || isSelected) {
        ctx.shadowBlur = 15;
        ctx.shadowColor = color;
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(sx, sy, r + 3, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, 2 * Math.PI);
      ctx.fill();
      
      // Reset shadows
      ctx.shadowBlur = 0;
      
      // If zoomed in deep, draw paper titles directly next to the dots
      if (scale > 3.5) {
        ctx.fillStyle = isHovered ? '#ffffff' : 'rgba(255,255,255,0.7)';
        ctx.font = `600 ${Math.max(10, Math.min(13, 8 * scale))}px var(--font-sans)`;
        ctx.textAlign = 'left';
        ctx.fillText(paper.title.substring(0, 30) + '...', sx + r + 4, sy + 4);
      }
    });

    // 4. Draw zoom-dependent overlays/labels
    const centroids = getCentroids();
    
    if (scale < 1.8) {
      // Outer zoom: Draw major fields centroids
      centroids.forEach(c => {
        const { x: sx, y: sy } = mapToScreen(c.x, c.y, w, h);
        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = '700 13px var(--font-heading)';
        ctx.textAlign = 'center';
        
        // Draw backing capsule
        const labelWidth = ctx.measureText(c.name).width;
        ctx.fillStyle = 'rgba(10, 12, 20, 0.8)';
        ctx.fillRect(sx - (labelWidth / 2) - 10, sy - 15, labelWidth + 20, 26);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.strokeRect(sx - (labelWidth / 2) - 10, sy - 15, labelWidth + 20, 26);
        
        ctx.fillStyle = CATEGORY_COLORS[c.name] || '#fff';
        ctx.fillText(c.name, sx, sy + 3);
      });
    } else if (scale >= 1.8 && scale <= 3.5) {
      // Mid-zoom: Draw subfields / method labels of some key papers
      // We pick 4 representative paper clusters
      const landmarks = [
        { title: "Generative Models (Diffusion/GAN)", x: -40, y: -40, color: '#06b6d4' },
        { title: "Large Language Models (LLMs)", x: 45, y: -30, color: '#a78bfa' },
        { title: "Reinforcement Learning", x: 30, y: 55, color: '#ec4899' },
        { title: "Deep Optimization & Neural Layers", x: -35, y: 40, color: '#34d399' }
      ];
      
      landmarks.forEach(l => {
        const { x: sx, y: sy } = mapToScreen(l.x, l.y, w, h);
        
        ctx.font = 'bold 11px var(--font-mono)';
        const textWidth = ctx.measureText(l.title).width;
        
        ctx.fillStyle = 'rgba(9, 10, 16, 0.75)';
        ctx.fillRect(sx - (textWidth/2) - 6, sy - 11, textWidth + 12, 20);
        ctx.strokeStyle = l.color;
        ctx.strokeRect(sx - (textWidth/2) - 6, sy - 11, textWidth + 12, 20);
        
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.fillText(l.title, sx, sy + 3);
      });
    }

  }, [papers, scale, offset, hoveredPaper, selectedPaper, ratings]);

  // Mouse Handlers for Dragging & Hover
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
      return;
    }
    
    // Check hit test for hover
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    let found = null;
    papers.forEach(p => {
      const { x: sx, y: sy } = mapToScreen(p.x, p.y, w, h);
      const dist = Math.hypot(sx - mouseX, sy - mouseY);
      if (dist < 10) {
        found = p;
      }
    });
    
    setHoveredPaper(found);
    if (found) {
      setTooltipPos({ x: mouseX + 15, y: mouseY + 15 });
    }
  };

  const handleMouseUp = (e) => {
    setIsDragging(false);
    
    // Hit test for click selection
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    let clicked = null;
    papers.forEach(p => {
      const { x: sx, y: sy } = mapToScreen(p.x, p.y, rect.width, rect.height);
      const dist = Math.hypot(sx - clickX, sy - clickY);
      if (dist < 10) {
        clicked = p;
      }
    });
    
    if (clicked) {
      setSelectedPaper(clicked);
    } else {
      setSelectedPaper(null);
    }
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    handleZoom(zoomFactor);
  };

  return (
    <div className="map-layout">
      {/* 2D Canvas Viewport */}
      <div className="map-canvas-container glass-panel" ref={containerRef}>
        <canvas 
          ref={canvasRef}
          className="map-canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
          style={{ width: '100%', height: '100%', borderRadius: '16px' }}
        />
        
        {/* Floating Controls */}
        <div className="map-zoom-controls">
          <button className="action-btn zoom-btn" onClick={() => handleZoom(1.25)}><ZoomIn size={18} /></button>
          <button className="action-btn zoom-btn" onClick={() => handleZoom(0.8)}><ZoomOut size={18} /></button>
          <button className="action-btn zoom-btn" onClick={handleReset}><RotateCcw size={18} /></button>
        </div>

        {/* Hover Tooltip */}
        {hoveredPaper && !selectedPaper && (
          <div 
            className="glass-panel"
            style={{
              position: 'absolute',
              left: tooltipPos.x,
              top: tooltipPos.y,
              padding: '12px',
              maxWidth: '280px',
              zIndex: 10,
              pointerEvents: 'none',
              textAlign: 'left',
              fontSize: '12px',
              background: 'rgba(9,10,16,0.9)',
              borderColor: CATEGORY_COLORS[hoveredPaper.category_name]
            }}
          >
            <h4 style={{ margin: '0 0 6px 0', fontWeight: '700', color: '#fff' }}>{hoveredPaper.title}</h4>
            <div style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>
              {hoveredPaper.authors[0]} {hoveredPaper.authors.length > 1 && 'et al.'}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px' }}>
              <span style={{ color: CATEGORY_COLORS[hoveredPaper.category_name], fontWeight: '500' }}>
                {hoveredPaper.category_name}
              </span>
              <span style={{ fontWeight: '700', color: hoveredPaper.relevance_score > 0 ? '#34d399' : '#9ca3af' }}>
                Relevance: {hoveredPaper.relevance_score > 0 ? `+${hoveredPaper.relevance_score}` : hoveredPaper.relevance_score}
              </span>
            </div>
          </div>
        )}

        {/* Selected Paper Details Modal Overlay (drawn locally inside container) */}
        {selectedPaper && (
          <div 
            className="glass-panel"
            style={{
              position: 'absolute',
              left: '24px',
              bottom: '24px',
              maxWidth: '420px',
              padding: '20px',
              zIndex: 20,
              textAlign: 'left',
              background: 'rgba(13,16,27,0.92)'
            }}
          >
            <button 
              className="action-btn" 
              style={{ position: 'absolute', top: '12px', right: '12px', width: '24px', height: '24px' }}
              onClick={() => setSelectedPaper(null)}
            >
              ×
            </button>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '15px', color: '#fff', fontWeight: '700' }}>
              {selectedPaper.title}
            </h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
              {selectedPaper.authors.slice(0, 3).join(', ')}{selectedPaper.authors.length > 3 ? ' et al.' : ''}
            </p>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.4', marginBottom: '16px', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' }}>
              {selectedPaper.abstract}
            </p>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="tag category" style={{ fontSize: '11px' }}>
                {selectedPaper.category_name}
              </span>
              
              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  className={`action-btn upvote ${ratings[selectedPaper.id] === 1 ? 'upvoted' : ''}`}
                  onClick={() => onRate(selectedPaper.id, ratings[selectedPaper.id] === 1 ? 0 : 1)}
                  style={{ width: '32px', height: '32px' }}
                >
                  <ThumbsUp size={14} />
                </button>
                <button 
                  className={`action-btn downvote ${ratings[selectedPaper.id] === -1 ? 'downvoted' : ''}`}
                  onClick={() => onRate(selectedPaper.id, ratings[selectedPaper.id] === -1 ? 0 : -1)}
                  style={{ width: '32px', height: '32px' }}
                >
                  <ThumbsDown size={14} />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Active Learning Sidebar Panel */}
      <div className="map-sidebar glass-panel">
        <h3 className="map-sidebar-title">
          <BrainCircuit size={18} style={{ color: 'var(--accent-color)' }} />
          Active Learning
        </h3>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: '0 0 12px 0', textAlign: 'left', lineHeight: '1.4' }}>
          Rate these border papers to help the system refine its decision boundary quickly:
        </p>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {activeLearningPapers.map(paper => {
            const currentRating = ratings[paper.id] || 0;
            return (
              <div key={paper.id} className="active-learning-card">
                <h4 className="active-learning-title" title={paper.title}>{paper.title}</h4>
                <div className="active-learning-meta">
                  <span style={{ color: CATEGORY_COLORS[paper.category_name], fontWeight: '500' }}>
                    {paper.category_name.substring(0, 15)}
                  </span>
                  
                  <div className="active-learning-actions">
                    <button 
                      className={`active-learning-btn upvote ${currentRating === 1 ? 'upvoted' : ''}`}
                      onClick={() => onRate(paper.id, currentRating === 1 ? 0 : 1)}
                    >
                      <ThumbsUp size={12} />
                    </button>
                    <button 
                      className={`active-learning-btn downvote ${currentRating === -1 ? 'downvoted' : ''}`}
                      onClick={() => onRate(paper.id, currentRating === -1 ? 0 : -1)}
                    >
                      <ThumbsDown size={12} />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
