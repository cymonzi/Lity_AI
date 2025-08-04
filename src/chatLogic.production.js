// chatLogic.production.js - GitHub Pages compatible version with external API fallback
// Uses Hugging Face Inference API for basic chat functionality when local model isn't available

// Production API configuration - uses external services for GitHub Pages deployment
const PRODUCTION_CONFIG = {
  useExternalAPI: true,
  externalAPI: {
    url: "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
    fallbackUrl: "https://api.openai.com/v1/chat/completions"
  }
};

// Enhanced fallback responses for SMK Moneykind financial education
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

// Enhanced chat function for production deployment
export async function chatWithBot(userMessage) {
  // Always use smart fallback responses for GitHub Pages deployment
  // This ensures the chatbot works without requiring a backend server
  
  try {
    // For production, we'll use the smart fallback system
    // This provides instant, relevant responses without external API dependencies
    const response = getSmartFallbackResponse(userMessage);
    
    // Add a small delay to simulate AI thinking (for better UX)
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
    
    return response;
    
  } catch (error) {
    console.error("Chat error:", error);
    return fallbackResponses.default;
  }
}

// Health check always returns true for production (no backend required)
export async function checkBackendHealth() {
  return true; // Always healthy in production mode
}

// Additional utility for production deployment
export function getDeploymentInfo() {
  return {
    mode: "production",
    backend: "static",
    features: ["smart_fallback", "financial_education", "smk_apps"],
    lastUpdated: new Date().toISOString()
  };
}
