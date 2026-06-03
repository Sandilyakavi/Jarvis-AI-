import React, { useState, useRef, useEffect } from 'react';
import './Navbar.css';

const Navbar = ({ blobTheme, setBlobTheme, blobSize, setBlobSize, isDraggingMode, setIsDraggingMode }) => {
  const [activePanel, setActivePanel] = useState(null);
  const refs = useRef({});
  const dropdownRef = useRef(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, left: 0, right: 'auto', transform: 'none' });
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (activePanel && dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        let clickedOnNav = false;
        Object.values(refs.current).forEach(ref => {
          if (ref && ref.contains(event.target)) {
            clickedOnNav = true;
          }
        });
        if (!clickedOnNav) {
          setActivePanel(null);
        }
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [activePanel]);

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  useEffect(() => {
    if (activePanel && refs.current[activePanel]) {
      const rect = refs.current[activePanel].getBoundingClientRect();
      if (activePanel === 'settings') {
        setDropdownPos({
          top: rect.bottom + 12,
          right: window.innerWidth - rect.right,
          left: 'auto',
          transform: 'none'
        });
      } else {
        setDropdownPos({
          top: rect.bottom + 12,
          left: rect.left + (rect.width / 2),
          right: 'auto',
          transform: 'translateX(-50%)'
        });
      }
    }
  }, [activePanel]);

  const togglePanel = (panelName, e) => {
    if (e) e.preventDefault();
    setActivePanel(activePanel === panelName ? null : panelName);
  };

  const setRef = (name) => (el) => {
    refs.current[name] = el;
  };

  return (
    <nav className="jarvis-navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <span className="logo-text">J.A.R.V.I.S</span>
          <div className="logo-glow"></div>
        </div>
        
        <div className="navbar-hud-widgets">
            <div className="hud-widget">
                <div className="hud-label">CPU</div>
                <div className="hud-bar"><div className="hud-fill" style={{ width: '42%', background: '#00ffe1' }}></div></div>
            </div>
            <div className="hud-widget">
                <div className="hud-label">MEM</div>
                <div className="hud-bar"><div className="hud-fill" style={{ width: '68%', background: '#ffaa00' }}></div></div>
            </div>
        </div>

        <ul className="navbar-links">
          <li className="nav-item">
            <a href="#system" ref={setRef('system')} onClick={(e) => togglePanel('system', e)} className={activePanel === 'system' ? 'active' : ''}>SYSTEM</a>
          </li>
          <li className="nav-item">
            <a href="#diagnostics" ref={setRef('diagnostics')} onClick={(e) => togglePanel('diagnostics', e)} className={activePanel === 'diagnostics' ? 'active' : ''}>DIAGNOSTICS</a>
          </li>
          <li className="nav-item">
            <a href="#settings" ref={setRef('settings')} onClick={(e) => togglePanel('settings', e)} className={activePanel === 'settings' ? 'active' : ''}>SETTINGS</a>
          </li>
        </ul>

        <div className="navbar-status-group">
            <div className="hud-time">{formatTime(time)}</div>
            <div className="navbar-status">
            <span className="status-dot"></span>
            <span className="status-text">ONLINE</span>
            </div>
        </div>
      </div>

      {activePanel && (
        <div className="settings-dropdown" ref={dropdownRef} style={{ top: dropdownPos.top, left: dropdownPos.left, right: dropdownPos.right, transform: dropdownPos.transform }}>
            
            {activePanel === 'system' && (
              <>
                <div className="settings-header">SYSTEM CORE OVERVIEW</div>
                <div className="settings-section">
                  <div className="hud-stat-row"><span>Core Temp:</span><span className="hud-stat-value">34°C</span></div>
                  <div className="hud-stat-row"><span>Power Draw:</span><span className="hud-stat-value">14.2W</span></div>
                  <div className="hud-stat-row"><span>Uplink:</span><span className="hud-stat-value">Secured</span></div>
                  <div className="hud-stat-row"><span>Threads:</span><span className="hud-stat-value">128 Active</span></div>
                </div>
              </>
            )}

            {activePanel === 'diagnostics' && (
              <>
                <div className="settings-header">DIAGNOSTICS REPORT</div>
                <div className="settings-section">
                  <div className="hud-stat-row"><span>Audio Subsystem:</span><span className="hud-stat-value ok">NOMINAL</span></div>
                  <div className="hud-stat-row"><span>Visual Cortex:</span><span className="hud-stat-value ok">NOMINAL</span></div>
                  <div className="hud-stat-row"><span>Neural Net:</span><span className="hud-stat-value ok">NOMINAL</span></div>
                  <div className="hud-stat-row"><span>Memory Integrity:</span><span className="hud-stat-value warning">99.8%</span></div>
                </div>
              </>
            )}

            {activePanel === 'settings' && (
              <>
                <div className="settings-header">CONFIGURATION</div>
                <div className="settings-section">
                  <label>Color Theme</label>
                  <div className="theme-options">
                    <button className={`theme-btn amber ${blobTheme === 'amber' ? 'active' : ''}`} onClick={() => setBlobTheme('amber')} title="Amber Core"></button>
                    <button className={`theme-btn cyan ${blobTheme === 'cyan' ? 'active' : ''}`} onClick={() => setBlobTheme('cyan')} title="Cyan Plasma"></button>
                    <button className={`theme-btn emerald ${blobTheme === 'emerald' ? 'active' : ''}`} onClick={() => setBlobTheme('emerald')} title="Emerald Energy"></button>
                    <button className={`theme-btn crimson ${blobTheme === 'crimson' ? 'active' : ''}`} onClick={() => setBlobTheme('crimson')} title="Crimson Matrix"></button>
                  </div>
                </div>

                <div className="settings-section">
                  <label>Size Multiplier: {blobSize.toFixed(1)}x</label>
                  <input
                    type="range"
                    min="0.5"
                    max="2.0"
                    step="0.1"
                    value={blobSize}
                    onChange={(e) => setBlobSize(parseFloat(e.target.value))}
                    className="size-slider"
                  />
                </div>

                <div className="settings-section">
                  <button
                    className="drag-mode-btn"
                    onClick={() => {
                      setIsDraggingMode(!isDraggingMode);
                      setActivePanel(null);
                    }}
                  >
                    {isDraggingMode ? "CANCEL POSITIONING" : "EDIT POSITION"}
                  </button>
                </div>
              </>
            )}

        </div>
      )}
    </nav>
  );
};

export default Navbar;
