import React, { useRef, useState } from 'react';
import { Send, RefreshCw, Trash2, ThumbsUp } from 'lucide-react';

export default function LityFooter({
  input,
  setInput,
  handleSend,
  loading,
  backendHealthy,
  theme,
  handleClear,
  handleFeedback, // not used anymore
  messages,
  inputRef
}) {
  // fallback for inputRef if not provided
  const localInputRef = useRef();
  const ref = inputRef || localInputRef;

  // Snackbar state
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const showSnackbar = () => {
    setSnackbarOpen(true);
    setTimeout(() => setSnackbarOpen(false), 2000);
  };

  return (
    <div
      style={{
        padding: '12px 8px',
        background: theme.card,
        borderTop: `1px solid ${theme.border}`,
        display: 'flex',
        gap: '8px',
        alignItems: 'flex-end',
        width: '100%',
        boxSizing: 'border-box',
        maxWidth: '100vw',
      }}
    >
      <div
        style={{
          display: 'flex',
          gap: '6px',
          minWidth: 0,
        }}
      >
        <button
          onClick={handleClear}
          disabled={messages && messages.length <= 1}
          style={{
            background: 'none',
            border: `1px solid ${theme.border}`,
            borderRadius: '8px',
            padding: '8px',
            color: theme.heading,
            cursor: messages && messages.length > 1 ? 'pointer' : 'not-allowed',
            opacity: messages && messages.length > 1 ? 1 : 0.5,
            transition: 'all 0.2s ease'
          }}
        >
          <Trash2 size={18} />
        </button>
        <button
          onClick={showSnackbar}
          style={{
            background: 'none',
            border: `1px solid ${theme.border}`,
            borderRadius: '8px',
            padding: '8px',
            color: theme.heading,
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          <ThumbsUp size={18} />
        </button>
      </div>
      
      <div
        style={{
          flex: 1,
          display: 'flex',
          gap: '6px',
          alignItems: 'flex-end',
          minWidth: 0,
        }}
      >
        <textarea
          ref={ref}
          value={input}
          onChange={e => {
            setInput(e.target.value);
            // Auto-resize
            const el = e.target;
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 120) + 'px';
          }}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="Ask anything finance..."
          aria-label="Chat input"
          disabled={loading || !backendHealthy}
          style={{
            flex: 1,
            background: theme.inputBg,
            border: `1.5px solid ${theme.inputBorder}`,
            borderRadius: '16px',
            padding: '12px 12px',
            color: theme.inputText,
            fontSize: '1rem',
            resize: 'none',
            minHeight: '22px',
            maxHeight: '90px',
            outline: 'none',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
            transition: 'border 0.2s, box-shadow 0.2s',
            fontFamily: 'inherit',
            lineHeight: 1.5,
            fontWeight: 500,
            letterSpacing: 0.05,
            margin: 0,
            width: '100%',
            boxSizing: 'border-box',
            overflowY: 'auto',
            WebkitOverflowScrolling: 'touch',
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
          }}
          rows={1}
          spellCheck={true}
          autoComplete="on"
          autoCorrect="on"
          autoFocus={false}
          onFocus={e => {
            e.target.style.borderColor = theme.accent;
            e.target.style.boxShadow = `0 0 0 3px ${theme.accent}22`;
          }}
          onBlur={e => {
            e.target.style.borderColor = theme.inputBorder;
            e.target.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)';
          }}
        />
        {/* Hide textarea scrollbar for a cleaner look on mobile */}
        <style>{`
          textarea::-webkit-input-placeholder { color: #b0b8c1 !important; opacity: 1; }
          textarea::placeholder { color: #b0b8c1 !important; opacity: 1; }
          textarea::-webkit-scrollbar { display: none; }
          textarea { scrollbar-width: none; -ms-overflow-style: none; }
        `}</style>
        <button
          onClick={handleSend}
          disabled={!input.trim() || loading || !backendHealthy}
          style={{
            background: input.trim() && !loading && backendHealthy ? theme.sendBg : theme.border,
            border: 'none',
            borderRadius: '8px',
            padding: '10px',
            color: 'white',
            cursor: input.trim() && !loading && backendHealthy ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s ease',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minWidth: 36,
            minHeight: 36,
          }}
        >
          {loading ? <RefreshCw size={18} className="animate-spin" /> : <Send size={18} />}
        </button>
      </div>
    {/* Snackbar */}
    {snackbarOpen && (
      <div style={{
        position: 'fixed',
        left: '50%',
        bottom: 32,
        transform: 'translateX(-50%)',
        background: theme.card,
        color: theme.heading,
        border: `1px solid ${theme.border}`,
        borderRadius: 12,
        boxShadow: '0 4px 24px rgba(0,0,0,0.10)',
        padding: '14px 32px',
        fontSize: 16,
        fontWeight: 500,
        zIndex: 99999,
        transition: 'opacity 0.3s',
        opacity: snackbarOpen ? 1 : 0
      }}>
        Nice! Better, coming soon.
      </div>
    )}
    </div>
  );
}
