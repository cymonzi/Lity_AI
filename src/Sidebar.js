import React, { useState } from 'react';
import { Plus, MessageSquare, Trash2, Sun, Moon } from 'lucide-react';

export default function Sidebar({
  open,
  chats,
  currentChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  theme,
  darkMode,
  setDarkMode,
  isMobile,
}) {
  const [hoveredId, setHoveredId] = useState(null);
  const panelWidth = isMobile ? '84vw' : '280px';

  const containerStyle = isMobile
    ? {
        width: panelWidth,
        maxWidth: '320px',
        minWidth: '260px',
        background: theme.sidebar,
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        borderRight: `1px solid ${theme.border}`,
        position: 'fixed',
        top: 0,
        left: 0,
        zIndex: 110,
        transform: open ? 'translateX(0)' : 'translateX(-100%)',
        transition: 'transform 0.25s ease',
        boxShadow: open ? '0 10px 35px rgba(0,0,0,0.35)' : 'none',
      }
    : {
        width: open ? panelWidth : '0',
        minWidth: open ? panelWidth : '0',
        background: theme.sidebar,
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        transition: 'width 0.25s ease, min-width 0.25s ease',
        borderRight: `1px solid ${theme.border}`,
        flexShrink: 0,
      };

  return (
    <div style={containerStyle}>
      <div style={{
        opacity: open ? 1 : 0,
        transition: 'opacity 0.2s',
        pointerEvents: open ? 'auto' : 'none',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
        minWidth: '260px',
      }}>
        <div style={{
          padding: '12px',
          borderBottom: `1px solid ${theme.border}`,
          position: 'sticky',
          top: 0,
          background: theme.sidebar,
          zIndex: 2,
        }}>
          <p style={{ margin: '0 0 10px', fontSize: '12px', color: theme.subtext, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            Lity Workspace
          </p>
          <button
            onClick={onNewChat}
            style={{
              width: '100%', padding: '10px 14px',
              background: theme.accent,
              border: 'none', borderRadius: '10px', cursor: 'pointer',
              color: 'white', fontSize: '14px', fontWeight: '600',
              display: 'flex', alignItems: 'center', gap: '8px',
              transition: 'opacity 0.15s', whiteSpace: 'nowrap',
            }}
            aria-label="Start a new chat"
            onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
            onMouseLeave={e => e.currentTarget.style.opacity = '1'}
          >
            <Plus size={16} />
            New Chat
          </button>
        </div>

        {/* Chat history */}
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '8px' }}>
          {chats.length > 0 && (
            <p style={{
              fontSize: '11px', color: theme.subtext, padding: '4px 8px 8px',
              margin: 0, textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap',
            }}>
              Recent Chats
            </p>
          )}
          {chats.map(chat => (
            <div
              key={chat.id}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '10px 10px', borderRadius: '10px', cursor: 'pointer',
                background: currentChatId === chat.id ? theme.activeChat : 'transparent',
                transition: 'background 0.15s', marginBottom: '2px',
              }}
              onMouseEnter={e => {
                setHoveredId(chat.id);
                if (currentChatId !== chat.id) e.currentTarget.style.background = theme.hover;
              }}
              onMouseLeave={e => {
                setHoveredId(null);
                if (currentChatId !== chat.id) e.currentTarget.style.background = 'transparent';
              }}
              onClick={() => onSelectChat(chat.id)}
            >
              <MessageSquare size={14} style={{ color: theme.subtext, flexShrink: 0 }} />
              <span style={{
                fontSize: '13px', color: theme.text, flex: 1, fontWeight: currentChatId === chat.id ? '600' : '500',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {chat.title}
              </span>
              {hoveredId === chat.id && (
                <button
                  onClick={e => { e.stopPropagation(); onDeleteChat(chat.id); }}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: theme.subtext, padding: '2px 4px', borderRadius: '4px',
                    display: 'flex', alignItems: 'center',
                  }}
                  title="Delete chat"
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          ))}

          {chats.length === 0 && (
            <div style={{ padding: '18px 10px', textAlign: 'center', color: theme.subtext, fontSize: '13px' }}>
              No chats yet. Tap New Chat to begin.
            </div>
          )}
        </div>

        {/* Bottom section */}
        <div style={{ padding: '12px', borderTop: `1px solid ${theme.border}`, background: theme.sidebar }}>
          <button
            onClick={() => setDarkMode(v => !v)}
            style={{
              width: '100%', padding: '10px 12px',
              background: 'none', border: 'none', borderRadius: '8px',
              cursor: 'pointer', color: theme.text, fontSize: '14px',
              display: 'flex', alignItems: 'center', gap: '10px',
              transition: 'background 0.15s', whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => e.currentTarget.style.background = theme.hover}
            onMouseLeave={e => e.currentTarget.style.background = 'none'}
          >
            {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            {darkMode ? 'Light mode' : 'Dark mode'}
          </button>
          <div style={{
            marginTop: '8px', padding: '8px 12px',
            display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <div style={{
              width: '30px', height: '30px', borderRadius: '50%',
              background: theme.accent, display: 'flex', alignItems: 'center',
              justifyContent: 'center', color: 'white', fontSize: '13px', fontWeight: '700', flexShrink: 0,
            }}>U</div>
            <div>
              <p style={{ margin: 0, fontSize: '13px', fontWeight: '600', color: theme.text, whiteSpace: 'nowrap' }}>User</p>
              <p style={{ margin: 0, fontSize: '11px', color: theme.subtext, whiteSpace: 'nowrap' }}>SMK Moneykind</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
