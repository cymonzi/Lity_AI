import React from 'react';
import { MessageSquare, Sun, Moon } from 'lucide-react';

export default function LityHeader({ darkMode, backendHealthy, setDarkMode, showActions, setShowActions }) {
  const theme = darkMode
    ? {
        accent: '#14b8a6', // teal accent
        heading: '#f1f5f9',
        border: '#334155',
        bg: 'rgba(30, 41, 59, 0.8)',
        status: '#14b8a6', // teal status
        offline: '#ef4444',
      }
    : {
        accent: '#14b8a6', // teal accent
        heading: '#0f172a',
        border: '#e2e8f0',
        bg: 'rgba(255, 255, 255, 0.8)',
        status: '#14b8a6', // teal status
        offline: '#ef4444',
      };

  return (
    <div style={{
      padding: '16px 24px',
      background: theme.bg,
      backdropFilter: 'blur(20px)',
      borderBottom: `1px solid ${theme.border}`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          background: theme.accent,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontWeight: '600'
        }}>
          L
        </div>
        <div>
          <h1 style={{ 
            color: theme.heading, 
            fontSize: '20px', 
            fontWeight: '600', 
            margin: 0 
          }}>
            Lity AI
          </h1>
          <p style={{ 
            color: theme.inputBorder || theme.border, 
            fontSize: '14px', 
            margin: 0 
          }}>
            {backendHealthy ? 'Online' : 'Offline'} • Financial Assistant
          </p>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button
          onClick={() => setShowActions && setShowActions((v) => !v)}
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
          <MessageSquare size={18} />
        </button>
        <button
          onClick={() => setDarkMode && setDarkMode((v) => !v)}
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
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </div>
  );
}
