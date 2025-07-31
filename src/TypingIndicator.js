import React from 'react';

export default function TypingIndicator({ theme }) {
  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: '8px', 
      padding: '16px 24px', 
      color: theme.inputBorder 
    }}>
      <div className="typing-dots">
        <div></div><div></div><div></div>
      </div>
      <span>Lity is typing...</span>
      <style>{`
        .typing-dots {
          display: flex;
          gap: 4px;
        }
        .typing-dots div {
          width: 8px;
          height: 8px;
          background: ${theme.accent};
          border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out;
        }
        .typing-dots div:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots div:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing {
          0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
