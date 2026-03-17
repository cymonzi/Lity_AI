import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Menu, X, Zap } from 'lucide-react';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import ChatInput from './Litybottom';
import Sidebar from './Sidebar';
import { chatWithBot, checkBackendHealth } from './chatLogic';
import { findBestFAQMatch } from './enhancedFAQ';

const WELCOME_ID = 'welcome-message';

const QUICK_PROMPTS = [
  { label: 'Build my first budget', query: 'How do I create a simple monthly budget?' },
  { label: 'Explain compound interest', query: 'What is compound interest in simple terms?' },
  { label: 'Start saving consistently', query: 'How can I save money consistently every month?' },
  { label: 'Investment basics', query: 'What are the basics of investing for beginners?' },
  { label: 'Cut unnecessary expenses', query: 'How do I reduce unnecessary spending each month?' },
  { label: 'Build an emergency fund', query: 'How can I build a 3-month emergency fund?' },
];

const ROLE_OPTIONS = ['Student', 'Employed', 'Self-employed', 'Just exploring'];
const FOCUS_OPTIONS = ['Budgeting', 'Saving money', 'Investing', 'Tracking expenses'];
const INCOME_OPTIONS = ['Under 500k UGX', '500k-1M UGX', '1M-3M UGX', '3M+ UGX', 'Skip for now'];

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
  const [messages, setMessages] = useState([{
    id: WELCOME_ID, sender: 'bot',
    text: "Hi! I'm **Lity AI** - your financial literacy assistant.\n\nI can help you with:\n- Budgeting, saving, and investing basics\n- Debt management and expense tracking\n- Financial habits and planning\n- Taxes, insurance, and scam awareness\n\nWhat money topic should we start with?",
    timestamp: new Date()
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [setupStage, setSetupStage] = useState('pending-first');
  const [profile, setProfile] = useState({ role: '', focus: '', income: '' });
  const [showWelcomeBanner, setShowWelcomeBanner] = useState(() => {
    try {
      const done = window.localStorage.getItem('lity_onboarding_done') === 'true';
      return !done;
    } catch {
      return true;
    }
  });

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamRef = useRef(null);
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
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streamedText]);
  useEffect(() => () => clearInterval(streamRef.current), []);
  useEffect(() => {
    if (!currentChatId) return;
    setChats(prev => prev.map(c => (c.id === currentChatId ? { ...c, messages } : c)));
  }, [messages, currentChatId]);

  const findFAQMatch = useCallback((input) => {
    return findBestFAQMatch(input);
  }, []);

  const fallbackResponse = useCallback((input) => {
    const lower = input.toLowerCase();
    const financeWords = ['money','budget','save','invest','loan','bank','finance','profit','income','expense','debt','credit','tax','insurance','stock'];
    if (financeWords.some(w => lower.includes(w))) {
      return `Good question! Here's what I can help with on that topic:\n\n- **Budgeting** - simple plans like 50/30/20\n- **Saving** - practical methods to save consistently\n- **Investing** - beginner-friendly investing basics\n- **Debt & Expenses** - cut costs and repay debt faster\n\nTry: "How do I start budgeting?" or "What is compound interest?"`;
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
  }, [findFAQMatch, backendHealthy, fallbackResponse]);

  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText || input).trim();
    if (!text || loading || streaming) return;

    const userMsg = { id: Date.now(), sender: 'user', text, timestamp: new Date() };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setLoading(true);
    if (!overrideText) setInput('');

    if (!currentChatId) {
      const chatId = Date.now().toString();
      setCurrentChatId(chatId);
      setChats(prev => [{ id: chatId, title: text.slice(0, 40), messages: updatedMessages }, ...prev]);
    } else {
      setChats(prev => prev.map(c => c.id === currentChatId ? { ...c, messages: updatedMessages } : c));
    }

    try {
      const reply = await resolveReply(text);

      const shouldAskRole = setupStage === 'pending-first';

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
          setMessages(prev => {
            const next = [...prev, { id: Date.now() + 1, sender: 'bot', text: reply, timestamp: new Date() }];
            if (shouldAskRole) {
              next.push({
                id: Date.now() + 2,
                sender: 'bot',
                text: "Quick question first: Are you currently a Student, Employed, Self-employed, or Just exploring?",
                timestamp: new Date(),
              });
            }
            return next;
          });
          if (shouldAskRole) {
            setSetupStage('role');
          }
        }
      }, 12);
    } catch {
      const shouldAskRole = setupStage === 'pending-first';
      setLoading(false);
      setStreaming(false);
      setMessages(prev => {
        const next = [...prev, { id: Date.now() + 1, sender: 'bot', text: fallbackResponse(text), timestamp: new Date() }];
        if (shouldAskRole) {
          next.push({
            id: Date.now() + 2,
            sender: 'bot',
            text: "Quick question first: Are you currently a Student, Employed, Self-employed, or Just exploring?",
            timestamp: new Date(),
          });
        }
        return next;
      });
      if (shouldAskRole) {
        setSetupStage('role');
      }
    }
  }, [input, loading, streaming, messages, currentChatId, resolveReply, fallbackResponse, setupStage]);

  const handleStop = () => {
    clearInterval(streamRef.current);
    setStreaming(false);
    if (streamedText) {
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'bot', text: streamedText, timestamp: new Date() }]);
      setStreamedText('');
    }
  };

  const handleRegenerate = useCallback(() => {
    const lastUserIdx = messages.findLastIndex(m => m.sender === 'user');
    if (lastUserIdx < 0 || loading || streaming) return;

    const baseMessages = messages.slice(0, lastUserIdx + 1);
    const lastUser = baseMessages[lastUserIdx];

    setMessages(baseMessages);
    setLoading(true);
    setStreaming(true);
    setStreamedText('');

    resolveReply(lastUser.text)
      .then((reply) => {
        let i = 0;
        clearInterval(streamRef.current);
        streamRef.current = setInterval(() => {
          i += 2;
          setStreamedText(reply.slice(0, i));
          if (i >= reply.length) {
            clearInterval(streamRef.current);
            setStreaming(false);
            setStreamedText('');
            setMessages([...baseMessages, { id: Date.now() + 1, sender: 'bot', text: reply, timestamp: new Date() }]);
          }
        }, 12);
      })
      .catch(() => {
        setLoading(false);
        setStreaming(false);
        setMessages([...baseMessages, { id: Date.now() + 1, sender: 'bot', text: fallbackResponse(lastUser.text), timestamp: new Date() }]);
      })
      .finally(() => setLoading(false));
  }, [messages, loading, streaming, resolveReply, fallbackResponse]);

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([{ id: WELCOME_ID, sender: 'bot', text: "Hi! I'm **Lity AI** - your financial literacy coach. What money topic would you like to explore today?", timestamp: new Date() }]);
    setInput('');
    setSetupStage('pending-first');
    setProfile({ role: '', focus: '', income: '' });
    setShowWelcomeBanner(true);
    if (isMobile) setSidebarOpen(false);
  };

  const startSetup = () => {
    setShowWelcomeBanner(false);
    setSetupStage('role');
    setMessages([
      {
        id: Date.now(),
        sender: 'bot',
        text: `Hi, I'm **Lity AI**. I help you understand and manage your money better.\n\nQuick question: **What best describes you?**`,
        timestamp: new Date(),
      },
    ]);
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

  const handleRoleSelection = (role) => {
    setProfile(prev => ({ ...prev, role }));
    setSetupStage('focus');
    setMessages(prev => [
      ...prev,
      { id: Date.now(), sender: 'user', text: role, timestamp: new Date() },
      {
        id: Date.now() + 1,
        sender: 'bot',
        text: "Nice. What would you like the most help with?",
        timestamp: new Date(),
      },
    ]);
  };

  const handleFocusSelection = (focus) => {
    setProfile(prev => ({ ...prev, focus }));
    setSetupStage('income');
    setMessages(prev => [
      ...prev,
      { id: Date.now(), sender: 'user', text: focus, timestamp: new Date() },
      {
        id: Date.now() + 1,
        sender: 'bot',
        text: `Great. I will focus on helping you with **${focus.toLowerCase()}**. Optional: share your approximate monthly income so I can personalize budget examples.`,
        timestamp: new Date(),
      },
    ]);
  };

  const handleIncomeSelection = (income) => {
    if (income !== 'Skip for now') {
      setProfile(prev => ({ ...prev, income }));
    }
    setSetupStage('done');
    setMessages(prev => [
      ...prev,
      { id: Date.now(), sender: 'user', text: income, timestamp: new Date() },
      {
        id: Date.now() + 1,
        sender: 'bot',
        text: income === 'Skip for now'
          ? 'No problem. We can add your income later. Ask me anything about your money goals.'
          : `Perfect. I will use **${income}** as your working monthly income for personalized guidance.`,
        timestamp: new Date(),
      },
    ]);
    try {
      window.localStorage.setItem('lity_onboarding_done', 'true');
      if (profile.role) window.localStorage.setItem('lity_user_type', profile.role);
      if (profile.focus) window.localStorage.setItem('lity_user_goal', profile.focus);
    } catch {
      // Ignore storage failures
    }
  };

  const lastBotIdx = messages.findLastIndex(m => m.sender === 'bot');
  const isWelcomeOnly = messages.length === 1 && messages[0].id === WELCOME_ID;
  const dynamicSuggestions = getTimeBasedSuggestions(profile.focus);
  const showDynamicSuggestions = !['role', 'focus', 'income'].includes(setupStage);

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
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: backendHealthy ? theme.accent : '#ef4444' }}>
            <Zap size={13} fill={backendHealthy ? theme.accent : '#ef4444'} />
            {backendHealthy ? 'AI Online' : 'FAQ Mode'}
          </div>
        </div>

        {/* Messages area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 0 8px', scrollBehavior: 'smooth' }}>
          <div style={{ maxWidth: '760px', margin: '0 auto', padding: '0 16px' }}>

            {/* Welcome banner + first prompts */}
            {showWelcomeBanner && isWelcomeOnly && (
              <div style={{ textAlign: 'center', paddingBottom: '24px' }}>
                <div style={{
                  width: '56px', height: '56px', borderRadius: '16px', background: theme.accent,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'white', fontSize: '28px', fontWeight: '800', margin: '0 auto 16px',
                }}>L</div>
                <h2 style={{ margin: '0 0 6px', fontSize: '20px', fontWeight: '700', color: theme.text }}>
                  Lity AI
                </h2>
                <p style={{ margin: '0 0 20px', color: theme.subtext, fontSize: '14px' }}>
                  Your AI guide for smarter money decisions.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px', textAlign: 'left', marginBottom: '16px' }}>
                  {QUICK_PROMPTS.slice(0, 4).map(p => (
                    <button key={p.query} onClick={() => handleSend(p.query)} style={{
                      background: theme.inputBg, border: `1px solid ${theme.border}`,
                      borderRadius: '12px', padding: '14px 16px', cursor: 'pointer',
                      color: theme.text, fontSize: '13px', fontWeight: '500',
                      textAlign: 'left', fontFamily: 'inherit', transition: 'border-color 0.15s',
                    }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = theme.accent}
                      onMouseLeave={e => e.currentTarget.style.borderColor = theme.border}
                    >{p.label}</button>
                  ))}
                </div>
                <p style={{ margin: '0 0 12px', color: theme.subtext, fontSize: '12px' }}>Get started in 10 seconds</p>
                <button
                  onClick={startSetup}
                  style={{
                    background: theme.accent,
                    color: 'white',
                    border: 'none',
                    borderRadius: '10px',
                    padding: '10px 16px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                  }}
                >
                  Start Setup
                </button>
              </div>
            )}

            {/* All messages */}
            {messages.map((msg, idx) => (
              <div key={msg.id} style={{ marginBottom: '8px' }}>
                <ChatBubble
                  sender={msg.sender} text={msg.text} theme={theme}
                  isLast={idx === lastBotIdx}
                  onRegenerate={idx === lastBotIdx ? handleRegenerate : undefined}
                />
              </div>
            ))}

            {/* Streaming */}
            {streaming && streamedText && (
              <div style={{ marginBottom: '8px' }}>
                <ChatBubble sender="bot" text={streamedText} theme={theme} isLast={false} />
              </div>
            )}

            {/* Loading dots */}
            {loading && <TypingIndicator theme={theme} />}

            {/* Progressive setup choices */}
            {!loading && !streaming && setupStage === 'role' && (
              <div style={{ marginTop: '10px', marginBottom: '8px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {ROLE_OPTIONS.map(option => (
                  <button
                    key={option}
                    onClick={() => handleRoleSelection(option)}
                    style={{
                      border: `1px solid ${theme.border}`,
                      background: theme.inputBg,
                      color: theme.text,
                      borderRadius: '999px',
                      padding: '8px 12px',
                      cursor: 'pointer',
                      fontSize: '13px',
                    }}
                  >
                    {option}
                  </button>
                ))}
              </div>
            )}

            {!loading && !streaming && setupStage === 'focus' && (
              <div style={{ marginTop: '10px', marginBottom: '8px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {FOCUS_OPTIONS.map(option => (
                  <button
                    key={option}
                    onClick={() => handleFocusSelection(option)}
                    style={{
                      border: `1px solid ${theme.border}`,
                      background: theme.inputBg,
                      color: theme.text,
                      borderRadius: '999px',
                      padding: '8px 12px',
                      cursor: 'pointer',
                      fontSize: '13px',
                    }}
                  >
                    {option}
                  </button>
                ))}
              </div>
            )}

            {!loading && !streaming && setupStage === 'income' && (
              <div style={{ marginTop: '10px', marginBottom: '8px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {INCOME_OPTIONS.map(option => (
                  <button
                    key={option}
                    onClick={() => handleIncomeSelection(option)}
                    style={{
                      border: `1px solid ${theme.border}`,
                      background: theme.inputBg,
                      color: theme.text,
                      borderRadius: '999px',
                      padding: '8px 12px',
                      cursor: 'pointer',
                      fontSize: '13px',
                    }}
                  >
                    {option}
                  </button>
                ))}
              </div>
            )}

            <div ref={messagesEndRef} style={{ height: '16px' }} />
          </div>
        </div>

        {/* Input area */}
        <div style={{ padding: '12px 0 16px', background: theme.bg, borderTop: `1px solid ${theme.border}`, flexShrink: 0 }}>
          <div style={{ maxWidth: '760px', margin: '0 auto', padding: '0 16px 10px' }}>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {showDynamicSuggestions && dynamicSuggestions.map(prompt => (
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
