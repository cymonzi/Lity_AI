import React, { useRef } from 'react';
import { Send, Square } from 'lucide-react';

export default function ChatInput({ input, setInput, handleSend, loading, streaming, handleStop, theme, inputRef, placeholder = 'Message Lity AI...' }) {
  const localRef = useRef();
  const ref = inputRef || localRef;

  const handleChange = e => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  };

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const busy = loading || streaming;
  const canSend = input.trim() && !busy;

  return (
    <div style={{
      width: '100%',
      maxWidth: '760px',
      margin: '0 auto',
      padding: '0 16px',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        gap: '10px',
        background: theme.inputBg,
        border: `1.5px solid ${theme.border}`,
        borderRadius: '16px',
        padding: '10px 12px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
        transition: 'border-color 0.2s',
      }}
        onFocusCapture={e => e.currentTarget.style.borderColor = theme.accent}
        onBlurCapture={e => e.currentTarget.style.borderColor = theme.border}
      >
        <textarea
          ref={ref}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            resize: 'none',
            fontSize: '15px',
            color: theme.text,
            fontFamily: 'inherit',
            lineHeight: '1.6',
            maxHeight: '160px',
            overflowY: 'auto',
            scrollbarWidth: 'none',
          }}
        />
        <button
          onClick={busy ? handleStop : handleSend}
          title={busy ? 'Stop' : 'Send'}
          style={{
            width: '36px', height: '36px', borderRadius: '10px',
            background: busy ? '#ef4444' : canSend ? theme.accent : theme.border,
            border: 'none', cursor: busy ? 'pointer' : canSend ? 'pointer' : 'default',
            color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0, transition: 'background 0.2s',
          }}
        >
          {busy ? <Square size={16} fill="white" /> : <Send size={16} />}
        </button>
      </div>
      <style>{`textarea::-webkit-scrollbar { display: none; }`}</style>
    </div>
  );
}

