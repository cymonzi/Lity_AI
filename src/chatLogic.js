// chatLogic.js - FastAPI backend integration with fallback responses
// API URL configuration - uses environment variable or defaults to FastAPI localhost
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

// Fallback responses for SMK Moneykind financial education
const fallbackResponses = {
  "budget": "🎯 **Budget Like a Pro!** Create a simple budget: list your income, then expenses (needs first, then wants). Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Track everything with Nfunayo and adjust as needed!",
  
  "save": "💰 **Smart Saving Tips!** Start small - even 500 shillings weekly adds up! Set specific goals, automate if possible, and use Nfunayo to track progress. Remember: pay yourself first before any expenses!",
  
  "invest": "📈 **Investment Basics!** Investing means putting money to work for growth. Understand risk vs reward first. Start with savings accounts, then learn about bonds and stocks. Only invest money you won't need soon!",
  
  "litywise": "🎮 **Litywise - Gamified Learning!** Our fun financial education app! Choose your path: 🌱 Saver (beginners), 📊 Investor (intermediate), or 👑 Boss (advanced). Earn XP, collect badges, and master money skills through interactive lessons!",
  
  "nfunayo": "📱 **Nfunayo - Expense Tracker!** Track income, spending, and savings goals in real-time. Perfect for students and young adults to build financial awareness. Categories, budgets, and insights all in one place!",
  
  "smk": "🏆 **SMK Moneykind!** We're empowering African youth with financial literacy through Litywise (gamified learning) and Nfunayo (expense tracking). Building confident, financially literate young adults across Africa!",
  
  "greeting": "👋 Welcome to Lity AI! I'm here to help you master financial literacy with SMK Moneykind. Ask me about budgeting, saving, investing, or our awesome apps Litywise and Nfunayo!",
  
  "default": "💡 Great question about financial literacy! I specialize in helping with budgeting, saving, investing, and our SMK Moneykind educational tools. What specific financial topic would you like to explore? Ask about Litywise, Nfunayo, or any money management topic!"
};

// Finance-related keywords for better response matching
const financeKeywords = {
  budget: ['budget', 'budgeting', 'planning', 'expense', 'income'],
  saving: ['save', 'saving', 'savings', 'emergency fund', 'goal'],
  investing: ['invest', 'investment', 'stock', 'bond', 'portfolio', 'return'],
  apps: ['litywise', 'nfunayo', 'smk', 'moneykind', 'app'],
  general: ['money', 'finance', 'financial', 'cash', 'debt', 'loan', 'credit']
};

function getSmartFallbackResponse(userMessage) {
  const lowerMessage = userMessage.toLowerCase();
  
  // Check for greetings
  if (lowerMessage.includes('hi') || lowerMessage.includes('hello') || lowerMessage.includes('hey')) {
    return fallbackResponses.greeting;
  }
  
  // Check for specific topics
  for (const [topic, keywords] of Object.entries(financeKeywords)) {
    if (keywords.some(keyword => lowerMessage.includes(keyword))) {
      if (topic === 'budget') return fallbackResponses.budget;
      if (topic === 'saving') return fallbackResponses.save;
      if (topic === 'investing') return fallbackResponses.invest;
      if (topic === 'apps') {
        if (lowerMessage.includes('litywise')) return fallbackResponses.litywise;
        if (lowerMessage.includes('nfunayo')) return fallbackResponses.nfunayo;
        return fallbackResponses.smk;
      }
    }
  }
  
  return fallbackResponses.default;
}

// Main chat function: tries Azure backend, falls back to smart responses
export async function chatWithBot(userMessage) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: userMessage }),
    });
    if (!response.ok) {
      // Backend error, use fallback
      return getSmartFallbackResponse(userMessage);
    }
    const data = await response.json();
    return data.reply || getSmartFallbackResponse(userMessage);
  } catch (error) {
    // Network or other error, use fallback
    console.error("Chat error:", error);
    return getSmartFallbackResponse(userMessage);
  }
}

// Health check: returns true if backend is reachable
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: "GET",
      headers: { "Content-Type": "application/json" }
    });
    return response.ok;
  } catch (error) {
    console.warn("Backend health check failed:", error);
    return false;
  }
}

// Additional utility for deployment info
export function getDeploymentInfo() {
  return {
    mode: "fastapi-backend",
    backend: API_BASE_URL,
    features: ["fastapi_backend", "smart_fallback", "financial_education", "smk_apps", "model_fallback"],
    lastUpdated: new Date().toISOString()
  };
}
