import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Menu, X, Zap } from 'lucide-react';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import ChatInput from './Litybottom';
import Sidebar from './Sidebar';
import { chatWithBot, checkBackendHealth } from './chatLogic';
import { findBestFAQMatch } from './enhancedFAQ';

const QUICK_PROMPTS = [
  { label: 'Spend 200k this week?', query: 'Should I spend UGX 200,000 this week or hold it?' },
  { label: 'Can I afford rent next month?', query: 'Can I afford 500,000 UGX rent next month with my current income?' },
  { label: 'How should I divide my salary?', query: 'I just got paid. How should I divide my salary this month?' },
  { label: 'What to do with 300k', query: 'I have UGX 300,000 now. What is my best next money move?' },
  { label: 'Take this loan or not?', query: 'Should I take this loan now or wait two weeks?' },
  { label: 'Start saving this month', query: 'Help me start saving this month with a simple plan.' },
];

function pickUnique(items, count = 4) {
  return [...new Set(items)].slice(0, count);
}

function isGreeting(text) {
  return /^(hi|hello|hey|hi lity|hello lity)\b/.test(text.trim().toLowerCase());
}

function isRegularityAnswer(text) {
  const t = text.trim().toLowerCase();
  return /^(regular|my income is regular|irregular|my income is irregular)$/.test(t);
}

function inferDecisionTopic(text) {
  const t = text.toLowerCase();
  if (/(rent|afford)/.test(t)) return 'rent and buffer';
  if (/(salary|income|paid|paycheck)/.test(t)) return 'salary plan';
  if (/(save|saving|emergency)/.test(t)) return 'savings plan';
  if (/(invest|investment)/.test(t)) return 'investing';
  if (/(loan|debt|borrow)/.test(t)) return 'debt decision';
  return '';
}

function getContextualSuggestions(messages, decisionContext, hasHistory, decisionHistory) {
  const userMessages = messages.filter((m) => m.sender === 'user').map((m) => m.text.toLowerCase());

  if (decisionContext.awaiting === 'income-pattern') {
    return ['My income is regular', 'My income is irregular'];
  }

  if (decisionContext.awaiting === 'buffer-choice') {
    return ['Yes, calculate buffer', 'No, I\'ll manage'];
  }

  // First-time users: high-probability decision scenarios, not generic prompts.
  if (userMessages.length === 0) {
    if (hasHistory) {
      const historyText = (decisionHistory || []).join(' ').toLowerCase();
      if (/(invest|leftover|surplus)/.test(historyText)) {
        return [
          'Adjust my budget',
          'Next best money move',
          'Invest this leftover UGX?',
        ];
      }

      return [
        'Next best move for my salary',
        'Can I spend 200k this week?',
        'How much should I save this month?',
      ];
    }

    return [
      'Spend 200k this week?',
      'Can I afford rent next month?',
      'Divide my salary',
      'What to do with 300k',
    ];
  }

  const allText = userMessages.join(' ');
  const lastUser = userMessages[userMessages.length - 1] || '';
  const dayOfMonth = new Date().getDate();
  const suggestions = [];

  if (/(phone|buy|purchase|shopping|spend)/.test(lastUser)) {
    suggestions.push('What if I wait 2 weeks?');
    suggestions.push('Cheaper alternative options?');
    suggestions.push('Save before buying?');
  }

  if (/(salary|income|paid|paycheck)/.test(allText)) {
    suggestions.push('Divide my next salary');
    suggestions.push('How much should I save?');
  }

  if (/(budget|expense|overspend|rent|bills)/.test(allText)) {
    suggestions.push('Adjust my budget');
    suggestions.push('Am I overspending?');
    suggestions.push('Can I afford this?');
  }

  if (/(loan|debt|borrow)/.test(allText)) {
    suggestions.push('Can I handle repayments?');
    suggestions.push('Should I delay this loan?');
  }

  if (/(save|saving|invest|investment)/.test(allText)) {
    suggestions.push('Next best money move');
    suggestions.push('Save or invest first?');
  }

  if (dayOfMonth >= 25) {
    suggestions.push('Can I close month on budget?');
  }

  suggestions.push('Next best money move');
  suggestions.push('Can I afford this?');
  suggestions.push('Adjust my budget');
  suggestions.push('Am I overspending?');

  return pickUnique(suggestions, 4);
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
  const [chats, setChats] = useState(() => {
    try {
      const raw = window.localStorage.getItem('lity_chats');
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.slice(0, 50) : [];
    } catch {
      return [];
    }
  });
  const [currentChatId, setCurrentChatId] = useState(() => {
    try {
      return window.localStorage.getItem('lity_current_chat_id');
    } catch {
      return null;
    }
  });
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [showLandingPage, setShowLandingPage] = useState(true);
  const [decisionContext, setDecisionContext] = useState({ awaiting: '', topic: '' });
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
  const [lastTopic, setLastTopic] = useState(() => {
    try {
      return window.localStorage.getItem('lity_last_topic') || '';
    } catch {
      return '';
    }
  });

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
    try {
      window.localStorage.setItem('lity_chats', JSON.stringify(chats.slice(0, 50)));
    } catch {
      // Ignore storage failures
    }
  }, [chats]);
  useEffect(() => {
    try {
      if (currentChatId) {
        window.localStorage.setItem('lity_current_chat_id', currentChatId);
      } else {
        window.localStorage.removeItem('lity_current_chat_id');
      }
    } catch {
      // Ignore storage failures
    }
  }, [currentChatId]);
  useEffect(() => {
    try {
      window.localStorage.setItem('lity_decision_history', JSON.stringify(decisionHistory.slice(-10)));
    } catch {
      // Ignore storage failures
    }
  }, [decisionHistory]);
  useEffect(() => {
    if (lockAutoScrollRef.current) return;
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamedText]);

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

    // FAQ should only handle short, keyword-like inputs.
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
    const lower = text.toLowerCase().trim();

    if (isGreeting(lower)) {
      if (lastTopic) {
        return `Welcome back! Last time we checked your ${lastTopic}.\n\nWhat money decision are you working on today?`;
      }
      return 'Hello! I\'m Lity - your financial decision support AI. I help you make smart money moves fast.\n\nWhat financial decision are you working on right now?';
    }

    if (decisionContext.awaiting === 'income-pattern' && isRegularityAnswer(lower)) {
      const regular = lower.includes('regular') && !lower.includes('irregular');
      setDecisionContext({ awaiting: 'buffer-choice', topic: 'rent and buffer' });
      try {
        window.localStorage.setItem('lity_last_topic', 'rent and buffer');
      } catch {
        // Ignore storage failures
      }
      setLastTopic('rent and buffer');

      if (regular) {
        return 'Great! Here is the assessment:\n\nIf your income comfortably covers rent and still leaves room for food, transport, and a small buffer, you are in a safe zone. A good rule is to keep rent around 30-40% of monthly income.\n\nIf it feels tight, reduce discretionary spending this month and build a small buffer before paying rent.\n\nDo you want me to calculate exactly how much buffer you should have?';
      }

      return 'Thanks - that helps. With irregular income, this rent decision is higher risk unless you keep a stronger buffer. Try to keep at least one month of essentials before committing and avoid locking into costs that depend on your best month.\n\nDo you want me to calculate a safer buffer target for you?';
    }

    if (decisionContext.awaiting === 'buffer-choice' && /(yes|calculate)/.test(lower)) {
      return 'Perfect. Share your expected monthly take-home income in UGX and your target rent amount, and I will calculate the exact buffer you should keep before paying rent.';
    }

    if (decisionContext.awaiting === 'buffer-choice' && /(no|manage|i\'ll manage|ill manage)/.test(lower)) {
      setDecisionContext({ awaiting: '', topic: decisionContext.topic || 'rent and buffer' });
      return 'Good plan. Keep rent disciplined, protect a small buffer, and review your cash flow weekly so pressure does not build quietly.';
    }

    if (/(can i afford|afford|rent)/.test(lower) && /\d/.test(lower) && /(next month|this month|week|today|tomorrow)/.test(lower)) {
      setDecisionContext({ awaiting: 'income-pattern', topic: 'rent and buffer' });
      try {
        window.localStorage.setItem('lity_last_topic', 'rent and buffer');
      } catch {
        // Ignore storage failures
      }
      setLastTopic('rent and buffer');

      return '500,000 UGX for rent next month is a big commitment.\n\nTo give you the clearest advice, I just need a quick check: is your income regular (comes in every month) or irregular?';
    }

    if (backendHealthy) {
      try {
        const ai = await chatWithBot(text, {
          stage: userStage,
          recentDecisions: decisionHistory.slice(-5),
          topic: decisionContext.topic,
        });
        if (ai && ai.length > 15 && !ai.toLowerCase().includes("i don't know")) {
          const topic = inferDecisionTopic(text);
          if (topic) {
            setDecisionContext((prev) => ({ ...prev, awaiting: '', topic }));
            try {
              window.localStorage.setItem('lity_last_topic', topic);
            } catch {
              // Ignore storage failures
            }
            setLastTopic(topic);
          }
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
  }, [backendHealthy, decisionContext.awaiting, decisionContext.topic, decisionHistory, fallbackResponse, findFAQMatch, lastTopic, shouldUseFAQ, userStage]);

  const recordDecision = useCallback((text) => {
    const topic = inferDecisionTopic(text) || text.trim().slice(0, 60);
    if (!topic) return;
    setDecisionHistory((prev) => {
      const next = [...prev, topic];
      return next.slice(-10);
    });
  }, []);

  const updateCurrentChatMessages = useCallback((nextMessages, titleHint) => {
    if (!currentChatId) {
      const chatId = Date.now().toString();
      setCurrentChatId(chatId);
      setChats(prev => [{
        id: chatId,
        title: (titleHint || 'New chat').slice(0, 40),
        messages: nextMessages,
        updatedAt: Date.now(),
      }, ...prev]);
      return chatId;
    }

    setChats(prev => {
      const updated = prev.map(c => (c.id === currentChatId ? { ...c, messages: nextMessages, updatedAt: Date.now() } : c));
      return [...updated].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    });
    return currentChatId;
  }, [currentChatId]);

  const streamReplyIntoMessages = useCallback((baseMessages, reply, preserveScrollTop) => {
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
    }, 6);
  }, [updateCurrentChatMessages]);

  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText || input).trim();
    if (!text || loading || streaming) return;
    lockAutoScrollRef.current = false;

    const userMsg = { id: Date.now(), sender: 'user', text, timestamp: new Date() };
    const updatedMessages = [...messages, userMsg];
    recordDecision(text);
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
  }, [input, loading, streaming, messages, recordDecision, resolveReply, streamReplyIntoMessages, fallbackResponse, updateCurrentChatMessages]);

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
    setDecisionContext({ awaiting: '', topic: '' });
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

  const dynamicSuggestions = getContextualSuggestions(messages, decisionContext, Boolean(lastTopic), decisionHistory);

  if (showLandingPage) {
    return (
      <div style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100dvh',
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
              Financial decision support for real money choices.
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
                placeholder="Ask your money decision..."
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
            Direct. Practical. Action-first.
          </p>
        </div>
      </div>
    );
  }

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

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', minWidth: 0, minHeight: 0 }}>
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
          
        </div>

        {/* Messages area */}
        <div ref={messagesContainerRef} style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '24px 0 8px', scrollBehavior: 'smooth' }}>
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
        <div style={{ padding: '12px 0 16px', background: theme.bg, borderTop: `1px solid ${theme.border}`, flexShrink: 0, position: 'sticky', bottom: 0, zIndex: 20 }}>
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
