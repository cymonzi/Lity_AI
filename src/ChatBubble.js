import React from 'react';
import ReactMarkdown from 'react-markdown';

export default function ChatBubble({ sender, text, theme, style }) {
  const isUser = sender === 'user';
  return (
    <div
      style={{
        maxWidth: '70%',
        padding: '12px 16px',
        borderRadius: isUser ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
        background: isUser ? theme.userBubble : theme.botBubble,
        color: isUser ? 'white' : theme.heading,
        boxShadow: theme.shadow,
        border: `1px solid ${theme.border}`,
        fontSize: '15px',
        lineHeight: '1.5',
        wordBreak: 'break-word',
        animation: 'fadeInUp 0.3s ease-out',
        textAlign: 'left',
        marginBottom: '16px',
        alignSelf: isUser ? 'flex-end' : 'flex-start',
        ...style,
      }}
    >
      {isUser ? (
        text
      ) : (
        <ReactMarkdown>{text}</ReactMarkdown>
      )}
    </div>
  );
}
