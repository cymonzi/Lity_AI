// chatLogic.js - FastAPI backend integration with decision-oriented fallback responses
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

function formatLityResponse({ ack, diagnose, explain, steps, warning }) {
  const lines = [
    `${ack} ${diagnose}`,
    '',
    explain,
    '',
    'Do this now:',
    ...steps.map((step, idx) => `${idx + 1}. ${step}`),
  ];

  if (warning) {
    const cleanedWarning = warning.replace('Biggest risk:', '').trim();
    lines.push('', `Big risk: ${cleanedWarning}`);
  }

  return lines.join('\n');
}

function getSmartFallbackResponse(userMessage) {
  const lower = userMessage.toLowerCase();

  if (!lower.trim()) {
    return 'I need your exact money decision. Tell me your income source, amount in UGX, and deadline, then I will give your next move.';
  }

  if (/(hi|hello|hey)\b/.test(lower)) {
    return 'Hello. I am Lity - your financial decision support system. Tell me your money decision in one line with amount in UGX and your deadline, and I will give your best next action and biggest risk.';
  }

  if (/(should|loan|borrow|debt|credit)/.test(lower)) {
    return formatLityResponse({
      ack: 'You are deciding whether to take debt.',
      diagnose: 'Most bad loans come from urgency, not math.',
      explain: 'The real decision is whether this loan creates value or just delays a cash-flow problem.',
      steps: [
        'Open MoMo or Airtel Money and total your last 30 days of income and expenses.',
        'If repayment is above 20% of monthly income, do not take the loan.',
        'Only borrow for income-generating use and compare total repayment across at least 2 lenders or SACCO options.',
      ],
      warning: 'Biggest risk: taking a loan without fixing the spending pattern that caused the shortfall.',
    });
  }

  if (/(save|saving|emergency)/.test(lower)) {
    return formatLityResponse({
      ack: 'You want to save consistently.',
      diagnose: 'Savings fail when money sits in the same wallet as daily spending.',
      explain: 'You need separation and automation, not motivation.',
      steps: [
        'Move UGX 20,000 today to a separate wallet or SACCO account.',
        'Set a weekly transfer day right after income hits.',
        'Track progress every Sunday and increase by UGX 5,000 every 2 weeks if stable.',
      ],
      warning: 'Keep emergency savings liquid. Do not lock it into high-risk investments.',
    });
  }

  if (/(invest|investment|compound|interest|sacco|treasury|bond)/.test(lower)) {
    return formatLityResponse({
      ack: 'You want to grow money through investing.',
      diagnose: 'Most beginners invest before they build cash stability.',
      explain: 'First create a base, then start small and consistent.',
      steps: [
        'Build at least 1 month of emergency expenses first.',
        'Start with a low-risk option you understand (for example SACCO shares or treasury products via your bank).',
        'Invest a fixed amount monthly in UGX, then review every 90 days.',
      ],
      warning: 'If someone promises guaranteed high returns, treat it as a scam until proven otherwise.',
    });
  }

  if (/(scam|legit|real|fake|ponzi)/.test(lower)) {
    return formatLityResponse({
      ack: 'You are trying to avoid getting scammed.',
      diagnose: 'Scams usually push urgency and guaranteed returns.',
      explain: 'The decision is not return size. The decision is verification before payment.',
      steps: [
        'Do not send money now.',
        'Verify company registration and physical presence in Uganda.',
        'Ask how returns are generated; if unclear, walk away.',
      ],
      warning: 'Biggest risk: paying first because the offer expires today.',
    });
  }

  if (/(budget|expense|spending|income|salary)/.test(lower)) {
    return formatLityResponse({
      ack: 'You want better control of your budget.',
      diagnose: 'You cannot optimize what you do not track.',
      explain: 'A working budget starts from real cash-flow, not guesses.',
      steps: [
        'Open MoMo or Airtel Money and review the last 10 outgoing transactions.',
        'Label each as Need, Want, or Waste.',
        'Set a 7-day spending cap in UGX and move savings first before spending.',
      ],
      warning: 'Lifestyle inflation will erase progress unless you cap wants before payday.',
    });
  }

  return formatLityResponse({
    ack: 'You need a clear money decision now.',
    diagnose: 'Generic advice will waste your time.',
    explain: 'Give your exact context and I will give a direct recommendation.',
    steps: [
      'Tell me your decision in one sentence.',
      'Include amount in UGX and your timeline.',
      'I will give you the best next move and the main risk.',
    ],
  });
}

// Main chat function: sends free-text user input plus optional context for backend reasoning.
export async function chatWithBot(userMessage, context = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: userMessage, context }),
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
    features: ["fastapi_backend", "decision_engine_fallback", "uganda_first", "model_fallback"],
    lastUpdated: new Date().toISOString()
  };
}
