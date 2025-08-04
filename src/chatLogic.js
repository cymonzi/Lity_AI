// chatLogic.js - Enhanced with better error handling and fallback responses
const API_BASE_URL = "http://localhost:5000";

// Fallback responses for common queries when backend is unavailable
const fallbackResponses = {
  "budget": "Create a simple budget: list your income, then expenses (needs first, then wants). Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Track everything and adjust as needed!",
  "save": "Start saving small! Even 500 shillings weekly adds up. Set specific goals, automate if possible, and use Nfunayo to track progress. Remember: pay yourself first!",
  "invest": "Investing is putting money to work for growth. Start with understanding risk vs reward. Begin with savings accounts, then learn about bonds and stocks. Only invest money you won't need soon!",
  "litywise": "Litywise is our gamified financial learning app! Choose your path: Saver (beginners), Investor (intermediate), or Boss (advanced). Earn XP, collect badges, and master money skills through fun lessons!",
  "nfunayo": "Nfunayo is our expense tracker that helps you monitor income, spending, and savings goals in real-time. Perfect for students and young adults to build financial awareness!",
  "default": "That's a great question about financial literacy! While I'm having trouble connecting to my knowledge base right now, I'd love to help you learn about budgeting, saving, investing, or our SMK Moneykind apps. What specific topic interests you most?"
};

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
      const errorText = await response.text();
      console.warn(`API error: ${response.status} - ${errorText}`);
      
      // Return fallback response based on user message
      const lowerMessage = userMessage.toLowerCase();
      if (lowerMessage.includes('budget')) return fallbackResponses.budget;
      if (lowerMessage.includes('save') || lowerMessage.includes('saving')) return fallbackResponses.save;
      if (lowerMessage.includes('invest')) return fallbackResponses.invest;
      if (lowerMessage.includes('litywise')) return fallbackResponses.litywise;
      if (lowerMessage.includes('nfunayo')) return fallbackResponses.nfunayo;
      
      return fallbackResponses.default;
    }
    
    const data = await response.json();
    return data.reply || fallbackResponses.default;
  } catch (error) {
    console.error("Chat API error:", error);
    
    // Smart fallback based on user input
    const lowerMessage = userMessage.toLowerCase();
    if (lowerMessage.includes('budget')) return fallbackResponses.budget;
    if (lowerMessage.includes('save') || lowerMessage.includes('saving')) return fallbackResponses.save;
    if (lowerMessage.includes('invest')) return fallbackResponses.invest;
    if (lowerMessage.includes('litywise')) return fallbackResponses.litywise;
    if (lowerMessage.includes('nfunayo')) return fallbackResponses.nfunayo;
    
    return `I'm having trouble connecting to my full knowledge base right now, but I'm still here to help! Try asking about: budgeting basics, saving tips, Litywise features, Nfunayo tracking, or SMK Moneykind in general. What would you like to explore? 🤔`;
  }
}

export async function checkBackendHealth() {
  try {
    const response = await fetch(API_BASE_URL, {
      timeout: 5000  // 5 second timeout
    });
    return response.ok;
  } catch {
    return false;
  }
}
