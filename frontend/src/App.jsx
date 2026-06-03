import { useState } from 'react';
import './App.css'
import Blob from "./component/blob.jsx";
import Navbar from "./component/Navbar.jsx";

function App() {
  const [blobTheme, setBlobTheme] = useState('amber'); // amber, cyan, emerald, crimson
  const [blobSize, setBlobSize] = useState(1.0);
  const [blobPosition, setBlobPosition] = useState({ x: 0, y: -0.75 });
  const [isDraggingMode, setIsDraggingMode] = useState(false);

  return (
    <>
      <Navbar 
        blobTheme={blobTheme}
        setBlobTheme={setBlobTheme}
        blobSize={blobSize}
        setBlobSize={setBlobSize}
        isDraggingMode={isDraggingMode}
        setIsDraggingMode={setIsDraggingMode}
      />
      <Blob 
        blobTheme={blobTheme}
        blobSize={blobSize}
        blobPosition={blobPosition}
        setBlobPosition={setBlobPosition}
        isDraggingMode={isDraggingMode}
      />
      {isDraggingMode && (
        <div style={{ position: 'fixed', bottom: '100px', left: '50%', transform: 'translateX(-50%)', zIndex: 9999 }}>
          <button 
            onClick={() => setIsDraggingMode(false)}
            style={{
              padding: '12px 30px', background: 'rgba(0, 255, 225, 0.1)', border: '1px solid #00ffe1',
              color: '#00ffe1', borderRadius: '50px', cursor: 'pointer', backdropFilter: 'blur(10px)',
              fontSize: '16px', letterSpacing: '2px', fontWeight: 'bold', transition: 'all 0.3s'
            }}
          >
            SAVE POSITION
          </button>
        </div>
      )}
    </>
  );
}

export default App
