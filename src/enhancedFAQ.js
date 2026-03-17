// Finance-only FAQ responses for Lity AI.
// This file intentionally excludes company/careers/partnership metadata.

export const enhancedFAQ = {
  'budget': 'A simple budget starts with three buckets: needs, wants, and savings. Track your income, set spending limits per bucket, and review weekly to stay on target.',
  'how do i budget': 'List your monthly income first, then fixed costs, then variable costs. Set a realistic savings target and keep at least a small emergency amount every month.',
  'set budget nfunayo': 'In Nfunayo, add your income, create expense categories, and set limits per category. Review your category spend weekly and adjust when needed.',
  'save money': 'Start small and stay consistent. Automate a fixed amount after each income event, then increase your savings rate as your income grows.',
  'how do i save': 'Use a clear savings goal, a deadline, and weekly contribution targets. Keep savings separate from daily spending so you are less tempted to use it.',
  'set savings goals': 'Set one short-term goal and one long-term goal. Give each a target amount and date, then track progress weekly to stay motivated.',
  'emergency fund': 'Aim for at least 3 months of essential expenses over time. Start with a mini goal first, then build steadily.',
  'expenses': 'Track every expense for 30 days to see spending patterns. Cut low-value spending first and redirect that money to savings or debt repayment.',
  'track expenses': 'Use categories like food, transport, bills, data, and savings. Category-level tracking helps you make faster better decisions.',
  'debt': 'Pay minimums on all debts, then attack the highest-interest debt first. Avoid taking new debt until your repayment plan is stable.',
  'loan': 'Before taking a loan, compare total repayment cost, interest rate, fees, and repayment flexibility. Borrow only for a clear useful purpose.',
  'credit score': 'A stronger credit profile comes from on-time payments, low missed installments, and borrowing only what you can comfortably repay.',
  'interest': 'Interest is the cost of borrowing or the reward for saving. Higher rates grow debt faster, so compare rates before borrowing.',
  'compound interest': 'Compound interest means earnings generate additional earnings. Starting early and staying consistent creates stronger long-term growth.',
  'investing': 'Investing is long-term money growth with risk. Begin with goals, time horizon, and risk level, then diversify instead of betting on one option.',
  'how do i invest': 'Start by building an emergency fund first. Then invest regularly in diversified options that match your risk tolerance and timeline.',
  'mobile money': 'Treat mobile money like a bank wallet: track transfers, cash-outs, and fees. Frequent fee checks help reduce hidden costs.',
  'nfunayo': 'Nfunayo helps you track income, expenses, and savings goals so you can apply financial literacy in daily life.',
  'litywise': 'Litywise teaches practical money skills through structured learning paths and short quizzes you can apply to real decisions.',
  'saver role': 'Saver path focuses on basics: needs vs wants, saving habits, and smart spending decisions.',
  'investor role': 'Investor path focuses on budgeting, risk awareness, and long-term wealth building fundamentals.',
  'boss role': 'Boss path focuses on advanced skills: credit, debt management, taxes, business finance, and long-term planning.',
  'xp': 'XP rewards consistent learning behavior. Use it as motivation to build steady financial habits over time.',
  'quiz': 'Quizzes help reinforce practical money decisions. Retake missed questions and focus on why the right answer works in real life.',
  'financial literacy': 'Financial literacy is the ability to earn, manage, save, invest, and protect money effectively for long-term stability.',
  'insurance': 'Insurance protects you from large unexpected costs. Prioritize coverage for risks that could seriously damage your finances.',
  'tax': 'Taxes are mandatory contributions on income or transactions. Keep records and plan ahead so tax periods do not disrupt your cash flow.',
  'wealth': 'Wealth grows through consistent saving, smart investing, controlled debt, and disciplined long-term decisions.',
  'income': 'Grow income by improving skills, increasing value, and adding reliable side income streams where possible.',
  'spending': 'Spend intentionally by asking whether each purchase supports your goals. Delay non-urgent buys to reduce impulse spending.',
};

export const financialLiteracyFAQ = enhancedFAQ;

const FINANCE_TERMS = [
  'money', 'finance', 'financial', 'budget', 'budgeting', 'save', 'saving', 'savings',
  'invest', 'investing', 'investment', 'expense', 'expenses', 'income', 'debt', 'credit',
  'loan', 'loans', 'tax', 'taxes', 'insurance', 'interest', 'compound', 'wealth', 'cash',
  'spend', 'spending', 'mobile money', 'nfunayo', 'litywise', 'saver', 'investor', 'boss',
  'quiz', 'xp', 'financial literacy',
];

const tokenize = (value) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 2);

const countTermHits = (text, terms) =>
  terms.reduce((count, term) => (text.includes(term) ? count + 1 : count), 0);

const isFinancialIntent = (input) => {
  const text = input.toLowerCase();
  return countTermHits(text, FINANCE_TERMS) > 0;
};

const faqEntries = Object.entries(enhancedFAQ);

export const findBestFAQMatch = (userInput) => {
  const input = userInput.toLowerCase().trim();
  if (!input || !isFinancialIntent(input)) return null;

  if (enhancedFAQ[input]) return enhancedFAQ[input];

  const inputWords = tokenize(input);
  let bestMatch = null;
  let bestScore = 0;

  for (const [key, response] of faqEntries) {
    const keyWords = tokenize(key);
    const overlap = inputWords.filter((word) => keyWords.includes(word)).length;
    const phraseBoost = input.includes(key) ? 2 : 0;
    const financeBoost = countTermHits(`${key} ${response}`.toLowerCase(), FINANCE_TERMS);
    const practicalBoost = /(budget|save|invest|expense|debt|loan|interest|tax|insurance)/.test(`${key} ${response}`.toLowerCase()) ? 2 : 0;
    const score = overlap * 3 + phraseBoost + Math.min(financeBoost, 3) + practicalBoost;

    if (score > bestScore) {
      bestScore = score;
      bestMatch = response;
    }
  }

  return bestScore >= 4 ? bestMatch : null;
};
