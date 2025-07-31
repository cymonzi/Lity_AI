import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, RefreshCw, MessageSquare, Sun, Moon, Trash2, ThumbsUp } from 'lucide-react';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import LityFooter from './Litybottom';
import LityHeader from './LityHeader';
import { chatWithBot, checkBackendHealth } from './chatLogic';

const overview = `🤖 **What Is Lity AI?**\n\nLity AI is a smart, friendly financial assistant designed to help you understand and manage money better — one conversation at a time.\n\n**What can Lity AI do?**\n- Answer financial literacy questions 💡\n- Explain money concepts in simple terms 📚\n- Guide you through tools like Litywise and Nfunayo 🧩\n- Support learners in SMK Moneykind trainings 🎓\n\n**How does it work?**\nLity AI is powered by a fine-tuned language model trained on finance-focused Q&A, using SMK Moneykind's knowledge base. It responds in a warm, engaging tone — like a financial coach or friend 🐾.\n\n**Who is it for?**\n- Students learning about saving, budgeting, or investing\n- Parents teaching kids about money\n- Anyone confused about personal finance terms or tools\n- SMK users seeking help with the app or trainings`;

const faq = {
  litywise: "Litywise is our gamified financial literacy app with a guide named Lity. It helps kids, teens, and adults learn money skills through fun, interactive lessons.",
  nfunayo: "Nfunayo is our personal finance tracker for students to manage income and expenses with ease.",
  trainings: "Yes! We offer financial literacy workshops in schools and universities through our SMK Trainings program.",
  contact: "You can reach SMK Moneykind via our website or socials. Type 'overview' to learn more.",
};



function SMKChatbot() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: '👋 Hi, I\'m Lity AI — your friendly financial assistant!\n\nAsk me anything about money, saving, budgeting, or SMK Moneykind. Type "overview" to learn what I can do.',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState(true);
  const [darkMode, setDarkMode] = useState(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  const [showActions, setShowActions] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Enhanced theme with better colors and transitions
  const theme = darkMode
    ? {
        bg: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        card: '#1e293b',
        border: '#334155',
        shadow: '0 8px 32px rgba(0,0,0,0.4)',
        inputBg: '#334155',
        inputBorder: '#475569',
        inputText: '#f1f5f9',
        sendBg: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
        sendText: '#ffffff',
        heading: '#f1f5f9',
        error: '#ef4444',
        botBubble: '#334155',
        userBubble: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
        accent: '#0ea5e9'
      }
    : {
        bg: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
        card: '#ffffff',
        border: '#e2e8f0',
        shadow: '0 8px 32px rgba(0,0,0,0.1)',
        inputBg: '#ffffff',
        inputBorder: '#cbd5e1',
        inputText: '#334155',
        sendBg: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
        sendText: '#ffffff',
        heading: '#0f172a',
        error: '#ef4444',
        botBubble: '#ffffff',
        userBubble: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
        accent: '#0ea5e9'
      };

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Check backend health
  useEffect(() => {
    checkBackendHealth().then(setBackendHealthy);
  }, []);

  // Handle send message
  const handleSend = async () => {
    if (!input.trim() || loading || !backendHealthy) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    const currentInput = input;
    setInput('');

    try {
      let reply = '';
      const lowerInput = currentInput.toLowerCase();
      if (lowerInput.includes('overview')) {
        reply = overview;
      } else if (faq[lowerInput]) {
        reply = faq[lowerInput];
      } else {
        reply = await chatWithBot(currentInput);
      }
      const botMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        text: reply,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        text: '❌ Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Action handlers
  const handleClear = () => {
    setMessages([messages[0]]);
    setShowActions(false);
  };

  const handleFeedback = () => {
    alert('Thank you for your interest in providing feedback! This feature is coming soon.');
  };

  return (
    <div style={{
      minHeight: '100vh',
      height: '100vh',
      width: '100vw',
      background: theme.bg,
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      transition: 'all 0.3s ease',
      overflow: 'hidden',
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      margin: 0,
      padding: 0,
      zIndex: 9999
    }}>
      {/* Header */}
      <LityHeader
        darkMode={darkMode}
        backendHealthy={backendHealthy}
        setDarkMode={setDarkMode}
        showActions={showActions}
        setShowActions={setShowActions}
      />

      {/* Backend Status Warning */}
      {!backendHealthy && (
        <div style={{
          background: theme.error,
          color: 'white',
          padding: '12px 24px',
          textAlign: 'center',
          fontSize: '14px',
          fontWeight: '500'
        }}>
          ⚠️ Unable to connect to backend. Please check your server connection.
        </div>
      )}

      {/* Messages Container */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        margin: 0,
        padding: 0,
        minHeight: 0,
        height: '100%'
      }}>
        {/* Today Divider */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          margin: 0,
          gap: '16px',
          padding: '0 16px'
        }}>
          <div style={{ flex: 1, height: '1px', background: theme.border }} />
          <span style={{ 
            color: theme.inputBorder, 
            fontSize: '14px', 
            fontWeight: '500',
            padding: '0 8px'
          }}>
            Today
          </span>
          <div style={{ flex: 1, height: '1px', background: theme.border }} />
        </div>

        {/* Messages */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              display: 'flex',
              justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              margin: 0,
              padding: '0 16px'
            }}
          >
            <ChatBubble
              sender={msg.sender}
              text={msg.text}
              theme={theme}
              style={{
                maxWidth: '70%',
                padding: '12px 16px',
                borderRadius: msg.sender === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                background: msg.sender === 'user' ? theme.userBubble : theme.botBubble,
                color: msg.sender === 'user' ? 'white' : theme.heading,
                boxShadow: theme.shadow,
                border: `1px solid ${theme.border}`,
                fontSize: '15px',
                lineHeight: '1.5',
                wordBreak: 'break-word',
                animation: 'fadeInUp 0.3s ease-out'
              }}
            />
          </div>
        ))}

        {loading && <TypingIndicator theme={theme} />}
        <div ref={messagesEndRef} />
      </div>


      {/* Quick Actions */}
      {showActions && (
        <div style={{
          padding: '16px 24px',
          background: theme.card,
          borderTop: `1px solid ${theme.border}`,
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap'
        }}>
          {['overview', 'litywise', 'nfunayo', 'trainings'].map((action) => (
            <button
              key={action}
              onClick={() => {
                setInput(action);
                setShowActions(false);
                inputRef.current?.focus();
              }}
              style={{
                background: 'none',
                border: `1px solid ${theme.border}`,
                borderRadius: '20px',
                padding: '6px 12px',
                color: theme.heading,
                cursor: 'pointer',
                fontSize: '14px',
                transition: 'all 0.2s ease',
                textTransform: 'capitalize'
              }}
            >
              {action}
            </button>
          ))}
        </div>
      )}

      {/* Footer Input Area */}
      <LityFooter
        input={input}
        setInput={setInput}
        handleSend={handleSend}
        loading={loading}
        backendHealthy={backendHealthy}
        theme={theme}
        handleClear={handleClear}
        handleFeedback={handleFeedback}
        messages={messages}
        inputRef={inputRef}
      />



      {/* CSS Animations */}
      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
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
        
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        button:hover {
          transform: translateY(-1px);
        }
        
        textarea:focus {
          border-color: ${theme.accent} !important;
          box-shadow: 0 0 0 3px ${theme.accent}20 !important;
        }
      `}</style>
    </div>
  );
}

export default SMKChatbot;
