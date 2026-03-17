import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, Check, Pencil, Trash2 } from 'lucide-react';

export default function ChatBubble({
  id,
  sender,
  text,
  theme,
  edited,
  isEditNote,
  onSaveEdit,
  onDelete,
}) {
  const isUser = sender === 'user';
  const [hovered, setHovered] = useState(false);
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(text);

  useEffect(() => {
    setDraft(text);
  }, [text]);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleSave = () => {
    const next = draft.trim();
    if (!next || !onSaveEdit) return;
    onSaveEdit(id, next);
    setEditing(false);
  };

  const handleCancel = () => {
    setDraft(text);
    setEditing(false);
  };

  return (
    <div style={
      isUser
        ? { display: 'flex', justifyContent: 'flex-end', width: '100%', marginBottom: '12px' }
        : { display: 'flex', flexDirection: 'column', alignItems: 'flex-start', width: '100%', marginBottom: '12px' }
    }>
      {/* Bot message - plain text, no bubble */}
      {!isUser && (
        <div style={{ width: '100%', maxWidth: '100%' }}>
          <div style={{
            color: isEditNote ? theme.subtext : theme.text,
            fontSize: isEditNote ? '12px' : '15px',
            lineHeight: isEditNote ? '1.45' : '1.65',
            wordBreak: 'break-word',
            fontStyle: isEditNote ? 'italic' : 'normal',
          }}>
            <ReactMarkdown>{text}</ReactMarkdown>
          </div>
          {/* Action buttons under bot messages */}
          {!isEditNote && <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
            <button
              onClick={handleCopy}
              title="Copy"
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: copied ? theme.accent : theme.subtext,
                padding: '4px 8px', borderRadius: '6px',
                fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px',
                transition: 'color 0.15s',
              }}
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>}
        </div>
      )}

      {/* User bubble */}
      {isUser && (
        <div
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', maxWidth: '72%', position: 'relative' }}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          {!editing ? (
            <>
              <div style={{
                background: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
                color: 'white',
                padding: '12px 16px',
                borderRadius: '14px 14px 0 14px',
                fontSize: '15px',
                lineHeight: '1.65',
                wordBreak: 'break-word',
                boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
                transition: 'all 0.2s ease',
              }}>
                {text}
              </div>
              {edited && (
                <span style={{ marginTop: '5px', fontSize: '11px', color: theme.subtext }}>
                  Edited
                </span>
              )}
              <div style={{
                position: 'absolute',
                top: '-10px',
                right: '-10px',
                opacity: hovered ? 1 : 0,
                pointerEvents: hovered ? 'auto' : 'none',
                transition: 'opacity 0.16s ease',
                display: 'flex',
                gap: '4px',
                background: theme.headerBg,
                border: `1px solid ${theme.border}`,
                borderRadius: '999px',
                padding: '3px',
              }}>
                <button
                  onClick={() => setEditing(true)}
                  title="Edit"
                  style={{
                    background: 'transparent', border: 'none', cursor: 'pointer',
                    color: theme.subtext, padding: '5px', borderRadius: '999px',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}
                >
                  <Pencil size={13} />
                </button>
                <button
                  onClick={handleCopy}
                  title="Copy"
                  style={{
                    background: 'transparent', border: 'none', cursor: 'pointer',
                    color: copied ? theme.accent : theme.subtext, padding: '5px', borderRadius: '999px',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}
                >
                  {copied ? <Check size={13} /> : <Copy size={13} />}
                </button>
                {onDelete && (
                  <button
                    onClick={() => onDelete(id)}
                    title="Delete"
                    style={{
                      background: 'transparent', border: 'none', cursor: 'pointer',
                      color: theme.subtext, padding: '5px', borderRadius: '999px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            </>
          ) : (
            <div style={{
              width: '100%',
              background: theme.inputBg,
              border: `1px solid ${theme.border}`,
              borderRadius: '12px',
              padding: '10px',
              transition: 'all 0.2s ease',
            }}>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                rows={3}
                style={{
                  width: '100%',
                  resize: 'vertical',
                  minHeight: '78px',
                  maxHeight: '220px',
                  border: `1px solid ${theme.border}`,
                  borderRadius: '10px',
                  background: theme.bg,
                  color: theme.text,
                  padding: '10px 12px',
                  fontFamily: 'inherit',
                  fontSize: '14px',
                  lineHeight: '1.5',
                  outline: 'none',
                }}
              />
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '8px' }}>
                <button
                  onClick={handleCancel}
                  style={{
                    border: `1px solid ${theme.border}`,
                    background: 'transparent',
                    color: theme.text,
                    borderRadius: '8px',
                    padding: '6px 10px',
                    cursor: 'pointer',
                    fontSize: '12px',
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  style={{
                    border: 'none',
                    background: theme.accent,
                    color: 'white',
                    borderRadius: '8px',
                    padding: '6px 10px',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: '600',
                  }}
                >
                  Save & Resend
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
