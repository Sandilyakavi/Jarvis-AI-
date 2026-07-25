import React, { useState, useEffect, useRef } from 'react';
import './ChatInterface.css';

/**
 * ChatInterface Component
 * Renders the sci-fi themed HUD containing chat log and input.
 */
export default function ChatInterface({
  messages = [],
  isThinking = false,
  error = null,
  onSendMessage,
  onClearHistory,
  theme = 'cyan',
  transcript = ''
}) {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking, error]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = inputText.trim();
    if (!text || isThinking) return;

    onSendMessage(text);
    setInputText('');
  };

  return (
    <div className={`jarvis-chat-interface theme-${theme}`}>
      {/* Header */}
      <div className="chat-header">
        <span className="chat-header-title">COMMUNICATION LINK</span>
        <div className="chat-header-status">
          <span className="status-indicator" />
          <span>SECURED_UPLINK</span>
        </div>
      </div>

      {/* Message List */}
      <div className="messages-container">
        {messages.length === 0 && !isThinking && (
          <div style={{ opacity: 0.4, textAlign: 'center', marginTop: '40px', fontSize: '13px' }}>
            System ready. Initiate communication protocols, Sir.
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role}`}>
            <span className={`message-meta ${msg.role}`}>
              {msg.role === 'user' ? '▸ USER' : '▸ J.A.R.V.I.S'}
            </span>
            <div className="message-text">
              {msg.content}
            </div>
          </div>
        ))}

        {/* Real-time speaking transcript preview if not finalized */}
        {transcript && (
          <div className="chat-message user">
            <span className="message-meta user">▸ SPEECH_DECODING...</span>
            <div className="message-text" style={{ fontStyle: 'italic', opacity: 0.6 }}>
              {transcript}
            </div>
          </div>
        )}

        {/* Thinking Indicator */}
        {isThinking && (
          <div className="thinking-container">
            <div className="loading-spinner" />
            <span>ANALYZING PROTOCOLS...</span>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="error-banner">
            <span className="error-banner-title">▸ SYS_ERROR: {error.code || 'CONNECT_FAIL'}</span>
            <span>{error.message || 'An error occurred during transaction.'}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={isThinking}
          placeholder={isThinking ? 'Processing...' : 'Enter transmission input, Sir...'}
          className="message-input"
          autoFocus
        />
        <button
          type="submit"
          disabled={isThinking || !inputText.trim()}
          className="send-button"
        >
          SEND
        </button>
      </form>
    </div>
  );
}
