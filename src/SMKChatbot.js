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
  { label: 'Plan my UGX budget', query: 'I earn UGX 800,000. Build me a weekly budget I can follow.' },
  { label: 'Should I take this loan?', query: 'Should I take a UGX 600,000 loan if repayment is UGX 780,000 in 4 months?' },
  { label: 'Save with irregular income', query: 'My income is irregular. How do I save consistently in Uganda?' },
  { label: 'Check if this is legit', query: 'This investment promises 15% monthly returns. Is it legit?' },
  { label: 'Cut MoMo spending leaks', query: 'Help me cut unnecessary MoMo spending this week.' },
  { label: 'Save or invest first', query: 'Should I save first or start investing right now?' },
];

const ROLE_OPTIONS = ['Student', 'Employed', 'Self-employed', 'Just exploring'];
const FOCUS_OPTIONS = ['Budgeting', 'Saving money', 'Investing', 'Tracking expenses'];
const INCOME_OPTIONS = ['Under 500k UGX', '500k-1M UGX', '1M-3M UGX', '3M+ UGX', 'Skip for now'];

function inferDecisionTopic(text) {
  const t = text.toLowerCase();
  if (/(rent|afford)/.test(t)) return 'rent and buffer';
  if (/(salary|income|paid|paycheck)/.test(t)) return 'salary plan';
  if (/(save|saving|emergency)/.test(t)) return 'savings plan';
  if (/(invest|investment)/.test(t)) return 'investing';
  if (/(loan|debt|borrow)/.test(t)) return 'debt decision';
  return text.trim().slice(0, 60);
}

function getTimeBasedSuggestions(focus) {
  const hour = new Date().getHours();
  const morning = hour >= 5 && hour < 12;
  const evening = hour >= 18;

  if (morning) {
    return ['Set today\'s UGX spending cap', 'Move UGX 20K to savings now', 'What is my top money decision today?'];
  }

  if (evening) {
    return ['Review last 10 MoMo transactions', 'Find one spending leak', 'Set tomorrow\'s budget cap'];
  }

  const byFocus = {
    Budgeting: ['Build my weekly UGX budget', 'Cap my wants spending', 'Fix my cash-flow this month'],
    'Saving money': ['Set a realistic UGX savings plan', 'How do I save with irregular income?', 'Where should I keep emergency money?'],
    Investing: ['Should I save or invest first?', 'How do I start investing safely in Uganda?', 'What is my biggest investing risk?'],
    'Tracking expenses': ['Review my last 10 MoMo transactions', 'Create 3 expense buckets', 'Find my spending leaks'],
  };

  return byFocus[focus] || [
    'Build my weekly UGX budget',
    'Should I take this loan?',
    'Save or invest first?',
    'Check if this deal is legit',
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
    text: "I am **Lity AI**. I help you make better money decisions fast.\n\nShare your decision and amount in UGX, and I will give:\n- What is really happening\n- The best next action\n- The biggest risk to avoid\n\nWhat decision are you making today?",
    timestamp: new Date()
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [setupStage, setSetupStage] = useState('pending-first');
  const [profile, setProfile] = useState({ role: '', focus: '', income: '' });
  const [decisionHistory, setDecisionHistory] = useState(() => {
    try {
      const raw = window.localStorage.getItem('lity_decision_history');
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.slice(-10) : [];
    } catch {
      return [];
    }
  });
  const userStage = decisionHistory.length > 0 ? 'returning' : 'first-time';
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
  useEffect(() => {
    try {
      window.localStorage.setItem('lity_decision_history', JSON.stringify(decisionHistory.slice(-10)));
    } catch {
      // Ignore storage failures
    }
  }, [decisionHistory]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streamedText]);
  useEffect(() => () => clearInterval(streamRef.current), []);
  useEffect(() => {
    if (!currentChatId) return;
    setChats(prev => prev.map(c => (c.id === currentChatId ? { ...c, messages } : c)));
  }, [messages, currentChatId]);

  const findFAQMatch = useCallback((input) => {
    return findBestFAQMatch(input);
  }, []);

  const shouldUseFAQ = useCallback((text) => {
    const lower = text.toLowerCase().trim();
    const wordCount = lower.split(/\s+/).filter(Boolean).length;
    const hasNumber = /\d/.test(lower);
    const directChipMatch = QUICK_PROMPTS.some(
      (p) => p.label.toLowerCase() === lower || p.query.toLowerCase() === lower
    );

    return directChipMatch || (wordCount <= 3 && !hasNumber);
  }, []);

  const fallbackResponse = useCallback((input) => {
    const lower = input.toLowerCase();
    const financeWords = ['money','budget','save','invest','loan','bank','finance','income','expense','debt','tax','insurance','stock','uganda','ugx','momo','airtel','sacco'];

    if (/(can i afford|afford|rent)/.test(lower) && /\d/.test(lower) && /(next month|this month|week|today|tomorrow)/.test(lower)) {
      return 'Good question. You can estimate this now: if that rent leaves enough for essentials plus a buffer, it is manageable; if it eats most of your monthly cash, it is risky. Share your expected income next month in UGX and I will give you a clear yes/no.';
    }

    if (financeWords.some(w => lower.includes(w))) {
      return 'You already gave a useful money question. I will work with what you shared and give a direct recommendation. If one critical detail is missing, I will ask one short follow-up.';
    }
    return 'I am Lity - your financial decision support system. Tell me what you are deciding and I will help you move it forward immediately.';
  }, []);

  const resolveReply = useCallback(async (text) => {
    if (backendHealthy) {
      try {
        const ai = await chatWithBot(text, {
          stage: userStage,
          recentDecisions: decisionHistory.slice(-5),
        });
        if (ai && ai.length > 15 && !ai.toLowerCase().includes("i don't know")) {
          return ai;
        }
      } catch {
        // fall through
      }
    }

    if (shouldUseFAQ(text)) {
      const faqReply = findFAQMatch(text);
      if (faqReply) return faqReply;
    }

    return fallbackResponse(text);
  }, [findFAQMatch, backendHealthy, decisionHistory, fallbackResponse, shouldUseFAQ, userStage]);

  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText || input).trim();
    if (!text || loading || streaming) return;

    const userMsg = { id: Date.now(), sender: 'user', text, timestamp: new Date() };
    const updatedMessages = [...messages, userMsg];
    setDecisionHistory((prev) => [...prev, inferDecisionTopic(text)].slice(-10));
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
        i += 8;
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
      }, 6);
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
          i += 8;
          setStreamedText(reply.slice(0, i));
          if (i >= reply.length) {
            clearInterval(streamRef.current);
            setStreaming(false);
            setStreamedText('');
            setMessages([...baseMessages, { id: Date.now() + 1, sender: 'bot', text: reply, timestamp: new Date() }]);
          }
        }, 6);
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
    setMessages([{ id: WELCOME_ID, sender: 'bot', text: 'I am **Lity AI**. Share your money decision and amount in UGX. I will give your next best action.', timestamp: new Date() }]);
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
        text: `I am **Lity AI**. I coach real money decisions.\n\nFirst, what best describes you?`,
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
      display: 'flex', height: '100dvh', width: '100vw', overflow: 'hidden',
      background: theme.bg, fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      color: theme.text, fontSize: '15px',
    }}>
      <Sidebar
        open={sidebarOpen} chats={chats} currentChatId={currentChatId}
        onNewChat={handleNewChat} onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat} theme={theme}
        darkMode={darkMode} setDarkMode={setDarkMode}
        isMobile={isMobile}
      />

      {isMobile && sidebarOpen && (
        <div onClick={() => setSidebarOpen(false)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', zIndex: 99,
        }} />
      )}

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100dvh', overflow: 'hidden', minWidth: 0, minHeight: 0 }}>
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '12px',
          padding: '12px 16px', borderBottom: `1px solid ${theme.border}`,
          background: theme.headerBg, flexShrink: 0,
          position: 'sticky', top: 0, zIndex: 20,
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
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '24px 0 8px', scrollBehavior: 'smooth' }}>
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
                  Direct support for real money decisions.
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
        <div style={{ padding: '12px 0 16px', background: theme.bg, borderTop: `1px solid ${theme.border}`, flexShrink: 0, position: 'sticky', bottom: 0, zIndex: 20 }}>
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
            placeholder="Ask your money decision in UGX..."
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
