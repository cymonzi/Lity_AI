import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Menu, X, Zap } from 'lucide-react';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import ChatInput from './Litybottom';
import Sidebar from './Sidebar';
import { chatWithBot, checkBackendHealth } from './chatLogic';
import { findBestFAQMatch } from './enhancedFAQ';

const QUICK_PROMPTS = [
  { label: 'Build my first budget', query: 'How do I create a simple monthly budget?' },
  { label: 'Explain compound interest', query: 'What is compound interest in simple terms?' },
  { label: 'Start saving consistently', query: 'How can I save money consistently every month?' },
  { label: 'Investment basics', query: 'What are the basics of investing for beginners?' },
  { label: 'Cut unnecessary expenses', query: 'How do I reduce unnecessary spending each month?' },
  { label: 'Build an emergency fund', query: 'How can I build a 3-month emergency fund?' },
];

function getTimeBasedSuggestions(focus) {
  const hour = new Date().getHours();
  const morning = hour >= 5 && hour < 12;
  const evening = hour >= 18;

  if (morning) {
    return ['Plan today\'s spending', 'Quick saving tip', 'Ask a money question'];
  }

  if (evening) {
    return ['Log today\'s expenses', 'Review today\'s spending', 'Plan tomorrow\'s budget'];
  }

  const byFocus = {
    Budgeting: ['Create a monthly budget for me', 'How should I budget my salary?', 'Fix my overspending'],
    'Saving money': ['Help me save 500k', 'Build a 30-day savings plan', 'Where should I keep savings?'],
    Investing: ['Explain compound interest', 'How do I start investing?', 'Beginner investment mistakes to avoid'],
    'Tracking expenses': ['Track my expenses', 'Set expense categories for me', 'How do I cut unnecessary costs?'],
  };

  return byFocus[focus] || [
    'Create a monthly budget for me',
    'Help me save 500k',
    'Explain investing simply',
    'Track my expenses',
  ];
}

function getTheme(dark) {
  if (dark) {
    return {
      bg: '#0b1220',
      headerBg: '#111827',
      sidebarBg: '#0f172a',
      sidebar: '#0f172a',
      inputBg: '#111827',
      botBg: '#111827',
      botBubble: '#111827',
      userBg: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
      text: '#e5e7eb',
      subtext: '#94a3b8',
      border: '#1f2937',
      accent: '#14b8a6',
      activeChat: '#1e293b',
      hover: '#1f2937',
    };
  }

  return {
    bg: '#f8fafc',
    headerBg: '#ffffff',
    sidebarBg: '#f1f5f9',
    sidebar: '#f1f5f9',
    inputBg: '#ffffff',
    botBg: '#ffffff',
    botBubble: '#ffffff',
    userBg: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
    text: '#0f172a',
    subtext: '#475569',
    border: '#e2e8f0',
    accent: '#14b8a6',
    activeChat: '#e2e8f0',
    hover: '#e2e8f0',
  };
}

// â”€â”€â”€ Main Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function LityAI() {
  const [darkMode, setDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [showLandingPage, setShowLandingPage] = useState(true);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const lockAutoScrollRef = useRef(false);
  const theme = getTheme(darkMode);

  useEffect(() => {
    const onResize = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      if (!mobile) setSidebarOpen(true);
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => { checkBackendHealth().then(setBackendHealthy); }, []);
  useEffect(() => {
    if (lockAutoScrollRef.current) return;
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamedText]);

  const findFAQMatch = useCallback((input) => {
    return findBestFAQMatch(input);
  }, []);

  const fallbackResponse = useCallback((input) => {
    const lower = input.toLowerCase();
    const financeWords = ['money','budget','save','invest','loan','bank','finance','profit','income','expense','debt','credit','tax','insurance','stock'];
    if (financeWords.some(w => lower.includes(w))) {
      return `Good question! Here's what I can help with on that topic:\n\n- **Budgeting** â€” simple plans like 50/30/20\n- **Saving** â€” practical methods to save consistently\n- **Investing** â€” beginner-friendly investing basics\n- **Debt & Expenses** â€” cut costs and repay debt faster\n\nTry: "How do I start budgeting?" or "What is compound interest?"`;
    }
    return `I'm Lity AI, focused on **financial literacy**.\n\nAsk me about budgeting, saving, investing, debt, taxes, expenses, and financial habits.`;
  }, []);

  const resolveReply = useCallback(async (text) => {
    let reply = findFAQMatch(text) || '';
    if (!reply && backendHealthy) {
      try {
        const ai = await chatWithBot(text);
        if (ai && ai.length > 15 && !ai.toLowerCase().includes("i don't know")) reply = ai;
      } catch {
        // fall through
      }
    }
    return reply || fallbackResponse(text);
  }, [backendHealthy, findFAQMatch, fallbackResponse]);

  const updateCurrentChatMessages = useCallback((nextMessages, titleHint) => {
    if (!currentChatId) {
      const chatId = Date.now().toString();
      setCurrentChatId(chatId);
      setChats(prev => [{ id: chatId, title: (titleHint || 'New chat').slice(0, 40), messages: nextMessages }, ...prev]);
      return chatId;
    }

    setChats(prev => prev.map(c => (c.id === currentChatId ? { ...c, messages: nextMessages } : c)));
    return currentChatId;
  }, [currentChatId]);

  const streamReplyIntoMessages = useCallback((baseMessages, reply, preserveScrollTop) => {
    setLoading(false);
    setStreaming(true);
    let i = 0;
    streamRef.current = setInterval(() => {
      i += 2;
      setStreamedText(reply.slice(0, i));
      if (i >= reply.length) {
        clearInterval(streamRef.current);
        setStreaming(false);
        setStreamedText('');
        const finalMessages = [...baseMessages, { id: Date.now() + 1, sender: 'bot', text: reply, timestamp: new Date() }];
        setMessages(finalMessages);
        updateCurrentChatMessages(finalMessages, baseMessages.find(m => m.sender === 'user')?.text || 'New chat');

        if (typeof preserveScrollTop === 'number') {
          requestAnimationFrame(() => {
            if (messagesContainerRef.current) {
              messagesContainerRef.current.scrollTop = preserveScrollTop;
            }
            lockAutoScrollRef.current = false;
          });
        }
      }
    }, 12);
  }, [updateCurrentChatMessages]);

  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText || input).trim();
    if (!text || loading || streaming) return;
    lockAutoScrollRef.current = false;

    const userMsg = { id: Date.now(), sender: 'user', text, timestamp: new Date() };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setShowLandingPage(false);
    setLoading(true);
    if (!overrideText) setInput('');
    updateCurrentChatMessages(updatedMessages, text);

    try {
      const reply = await resolveReply(text);
      streamReplyIntoMessages(updatedMessages, reply);
    } catch {
      setLoading(false);
      setStreaming(false);
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'bot', text: fallbackResponse(text), timestamp: new Date() }]);
    }
  }, [input, loading, streaming, messages, resolveReply, streamReplyIntoMessages, fallbackResponse, updateCurrentChatMessages]);

  const handleStop = () => {
    clearInterval(streamRef.current);
    setStreaming(false);
    if (streamedText) {
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'bot', text: streamedText, timestamp: new Date() }]);
      setStreamedText('');
    }
  };

  const handleSaveEditedMessage = useCallback(async (messageId, nextText) => {
    if (loading || streaming) return;

    const idx = messages.findIndex(m => m.id === messageId && m.sender === 'user');
    if (idx < 0) return;

    const preserveScrollTop = messagesContainerRef.current?.scrollTop ?? 0;
    lockAutoScrollRef.current = true;

    const rewritten = {
      ...messages[idx],
      text: nextText,
      edited: true,
      timestamp: new Date(),
    };

    const baseMessages = [
      ...messages.slice(0, idx),
      rewritten,
      {
        id: Date.now() + 1,
        sender: 'bot',
        text: 'Response regenerated based on your edited message.',
        timestamp: new Date(),
        meta: 'edit-note',
      },
    ];

    setMessages(baseMessages);
    setLoading(true);
    setStreaming(false);
    setStreamedText('');

    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = preserveScrollTop;
      }
    });

    try {
      const reply = await resolveReply(nextText);
      streamReplyIntoMessages(baseMessages, reply, preserveScrollTop);
    } catch {
      const fallback = [
        ...baseMessages,
        { id: Date.now() + 2, sender: 'bot', text: fallbackResponse(nextText), timestamp: new Date() },
      ];
      setLoading(false);
      setStreaming(false);
      setStreamedText('');
      setMessages(fallback);
      updateCurrentChatMessages(fallback, nextText);
      requestAnimationFrame(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = preserveScrollTop;
        }
        lockAutoScrollRef.current = false;
      });
    }
  }, [loading, streaming, messages, resolveReply, streamReplyIntoMessages, fallbackResponse, updateCurrentChatMessages]);

  const handleDeleteMessage = useCallback((messageId) => {
    if (loading || streaming) return;
    const idx = messages.findIndex(m => m.id === messageId && m.sender === 'user');
    if (idx < 0) return;

    const preserveScrollTop = messagesContainerRef.current?.scrollTop ?? 0;
    lockAutoScrollRef.current = true;

    const nextMessages = messages.slice(0, idx);
    setMessages(nextMessages);
    updateCurrentChatMessages(nextMessages, nextMessages.find(m => m.sender === 'user')?.text || 'New chat');

    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = preserveScrollTop;
      }
      lockAutoScrollRef.current = false;
    });
  }, [loading, streaming, messages, updateCurrentChatMessages]);

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setInput('');
    if (isMobile) setSidebarOpen(false);
  };

  const handleSelectChat = (id) => {
    const chat = chats.find(c => c.id === id);
    if (chat) { setCurrentChatId(id); setMessages(chat.messages); }
    if (isMobile) setSidebarOpen(false);
  };

  const handleDeleteChat = (id) => {
    setChats(prev => prev.filter(c => c.id !== id));
    if (currentChatId === id) handleNewChat();
  };

  const dynamicSuggestions = getTimeBasedSuggestions();

  if (showLandingPage) {
    return (
      <div style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        width: '100vw',
        background: theme.bg,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        color: theme.text,
        padding: '0',
        overflow: 'hidden',
        boxSizing: 'border-box',
      }}>
        <div style={{
          width: '100%',
          maxWidth: '720px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 16px 72px',
          transform: 'translateY(-4vh)',
        }}>
          {/* Logo + Text */}
          <div style={{ textAlign: 'center', width: '100%', paddingBottom: '28px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '10px',
              background: theme.accent,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '24px',
              fontWeight: '800',
              margin: '0 auto 12px',
            }}>L</div>
            <h1 style={{ margin: '0', fontSize: '24px', fontWeight: '700' }}>Lity AI</h1>
            <p style={{ margin: '4px 0 0', color: theme.subtext, fontSize: '13px', padding: '0 16px' }}>
              Your AI guide for smarter money decisions.
            </p>
          </div>

          {/* Input + Suggestions */}
          <div style={{
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}>
            {/* Input */}
            <div style={{ width: '100%', maxWidth: '600px', flexShrink: 0 }}>
              <ChatInput
                input={input}
                setInput={setInput}
                handleSend={() => handleSend()}
                loading={loading}
                streaming={streaming}
                handleStop={handleStop}
                theme={theme}
                inputRef={inputRef}
                placeholder="Ask anything about money..."
              />
            </div>

            {/* Suggestions dropdown - show only when typing */}
            {input.trim() !== '' && (
              <div style={{
                width: '100%',
                maxWidth: '600px',
                marginTop: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                maxHeight: '200px',
                overflowY: 'auto',
              }}>
                {QUICK_PROMPTS.slice(0, 4).map(p => (
                  <button
                    key={p.query}
                    onClick={() => {
                      setInput(p.query);
                      setTimeout(() => {
                        if (inputRef.current) inputRef.current.focus();
                      }, 0);
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      background: theme.inputBg,
                      border: `1px solid ${theme.border}`,
                      borderRadius: '10px',
                      padding: '12px 14px',
                      cursor: 'pointer',
                      color: theme.text,
                      fontSize: '13px',
                      fontWeight: '400',
                      fontFamily: 'inherit',
                      transition: 'border-color 0.15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = theme.accent}
                    onMouseLeave={e => e.currentTarget.style.borderColor = theme.border}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Disclaimer at bottom */}
        <div style={{
          position: 'absolute',
          left: '50%',
          bottom: '24px',
          transform: 'translateX(-50%)',
          width: '100%',
          maxWidth: '760px',
          padding: '0 16px',
          textAlign: 'center',
        }}>
          <p style={{
            margin: '0',
            fontSize: '11px',
            color: theme.subtext,
            lineHeight: '1.4',
          }}>
            Lity AI can make mistakes. For important financial decisions, consult a professional.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden',
      background: theme.bg, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      color: theme.text, fontSize: '15px',
    }}>
      <Sidebar
        open={sidebarOpen} chats={chats} currentChatId={currentChatId}
        onNewChat={handleNewChat} onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat} theme={theme}
        darkMode={darkMode} setDarkMode={setDarkMode}
      />

      {isMobile && sidebarOpen && (
        <div onClick={() => setSidebarOpen(false)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', zIndex: 99,
        }} />
      )}

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', minWidth: 0 }}>
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '12px',
          padding: '12px 16px', borderBottom: `1px solid ${theme.border}`,
          background: theme.headerBg, flexShrink: 0,
        }}>
          <button
            onClick={() => setSidebarOpen(v => !v)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: theme.subtext, padding: '6px', borderRadius: '8px' }}
          >
            {sidebarOpen && !isMobile ? <X size={20} /> : <Menu size={20} />}
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '8px', background: theme.accent,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'white', fontWeight: '800', fontSize: '15px',
            }}>L</div>
            <span style={{ fontWeight: '600', fontSize: '16px' }}>Lity AI</span>
          </div>
          <div style={{ flex: 1 }} />
          
        </div>

        {/* Messages area */}
        <div ref={messagesContainerRef} style={{ flex: 1, overflowY: 'auto', padding: '24px 0 8px', scrollBehavior: 'smooth' }}>
          <div style={{ maxWidth: '760px', margin: '0 auto', padding: '0 16px' }}>
            {/* All messages */}

            {messages.map((msg) => (
              <div key={msg.id} style={{ marginBottom: '8px' }}>
                <ChatBubble
                  id={msg.id}
                  sender={msg.sender}
                  text={msg.text}
                  theme={theme}
                  edited={Boolean(msg.edited)}
                  isEditNote={msg.meta === 'edit-note'}
                  onSaveEdit={msg.sender === 'user' ? handleSaveEditedMessage : undefined}
                  onDelete={msg.sender === 'user' ? handleDeleteMessage : undefined}
                />
              </div>
            ))}

            {/* Streaming */}
            {streaming && streamedText && (
              <div style={{ marginBottom: '8px' }}>
                <ChatBubble sender="bot" text={streamedText} theme={theme} />
              </div>
            )}

            {/* Loading dots */}
            {loading && <TypingIndicator theme={theme} />}
            <div ref={messagesEndRef} style={{ height: '16px' }} />
          </div>
        </div>

        {/* Input area */}
        <div style={{ padding: '12px 0 16px', background: theme.bg, borderTop: `1px solid ${theme.border}`, flexShrink: 0 }}>
          <div style={{ maxWidth: '760px', margin: '0 auto', padding: '0 16px 10px' }}>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {dynamicSuggestions.map(prompt => (
                <button
                  key={prompt}
                  onClick={() => handleSend(prompt)}
                  style={{
                    border: `1px solid ${theme.border}`,
                    background: theme.inputBg,
                    color: theme.subtext,
                    borderRadius: '999px',
                    padding: '6px 10px',
                    cursor: 'pointer',
                    fontSize: '12px',
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
          <ChatInput
            input={input} setInput={setInput}
            handleSend={() => handleSend()} loading={loading}
            streaming={streaming} handleStop={handleStop}
            theme={theme} inputRef={inputRef}
            placeholder="Ask anything about money..."
          />
        </div>
      </div>

      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 768px) { input, textarea { font-size: 16px !important; } }
      `}</style>
    </div>
  );
}

export default LityAI;
