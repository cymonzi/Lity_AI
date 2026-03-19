// Decision-oriented, Uganda-first FAQ responses for Lity AI.

export const enhancedFAQ = {
  'budget': 'Set a weekly spending cap in UGX today. Open MoMo or Airtel Money, review your last 10 transfers, and label each as Need, Want, or Waste. Cut one Waste item immediately.',
  'how do i budget': 'Use this order: income first, essentials second, savings third, wants last. Move savings first, not what is left over.',
  'save money': 'Move UGX 20,000 now into a separate wallet or SACCO account. Repeat on the same day every week.',
  'how do i save': 'Do not keep savings in your daily spending wallet. Separate account first, then automate a fixed transfer amount.',
  'set savings goals': 'Pick one goal, one number, one deadline. Example: UGX 600,000 in 3 months means UGX 50,000 weekly.',
  'emergency fund': 'Start with UGX 200,000 as a mini emergency fund. Build to 1 month of essential expenses, then 3 months.',
  'expenses': 'Track every expense for 14 days before making a new budget. If you do not track, your plan is a guess.',
  'track expenses': 'Check your MoMo or Airtel Money history nightly and update three buckets: essentials, wants, debt.',
  'debt': 'Pay all minimums, then push extra cash to the highest-interest debt first. Freeze new borrowing for 30 days.',
  'loan': 'Only take a loan if repayment stays under 20% of monthly income and the loan increases income. If not, do not take it.',
  'interest': 'Always check total repayment, not just monthly installment. Small installments can hide expensive loans.',
  'compound interest': 'Compound interest rewards time and consistency. Start with small monthly investing in UGX and avoid skipping months.',
  'investing': 'Invest only after building emergency savings. Start with options you understand through a SACCO or regulated bank product.',
  'how do i invest': 'Define amount, timeline, and risk first. Then set an automatic monthly amount you can sustain.',
  'mobile money': 'Mobile money leaks cash through many small fees. Check weekly charges and cut unnecessary withdrawals.',
  'momo': 'Treat MoMo like a bank account. Every transfer must have a reason or it is likely impulse spending.',
  'airtel money': 'Use Airtel Money transaction history to spot spending leaks. Cancel one low-value habit this week.',
  'sacco': 'A good SACCO can support disciplined saving and lower-cost credit. Confirm governance and withdrawal rules before joining.',
  'stanbic': 'If using Stanbic, compare account fees and transfer costs before choosing where salary lands.',
  'dfcu': 'If using DFCU, ask for total loan cost breakdown before signing any facility.',
  'centenary': 'Centenary can be useful for savings discipline. Set a standing order so saving happens before spending.',
  'insurance': 'Buy insurance for risks that can wipe out your cash flow, not for everything. Start with the highest impact risk first.',
  'tax': 'Keep income and expense records weekly so tax time does not become an emergency.',
  'income': 'Increase income with one high-value skill and one side channel. Track extra income separately so it is not absorbed by lifestyle inflation.',
  'spending': 'Impulse spending is usually a trigger problem, not a money problem. Add a 24-hour rule before non-essential purchases.',
  'financial literacy': 'Financial literacy means making better money decisions fast: earn, protect, grow, and avoid avoidable losses.',
  'is this legit': 'If returns are guaranteed and unclear, treat it as a scam. Verify registration, business model, and payout source first.',
  'save or invest': 'If you lack emergency cash, save first. If emergency fund exists, split new money between saving and investing.',
  'should i': 'Decide using cash-flow impact first. If the decision hurts basic expenses next month, do not do it.',
};

export const financialLiteracyFAQ = enhancedFAQ;

const FINANCE_TERMS = [
  'money', 'finance', 'financial', 'budget', 'budgeting', 'save', 'saving', 'savings',
  'invest', 'investing', 'investment', 'expense', 'expenses', 'income', 'debt', 'credit',
  'loan', 'loans', 'tax', 'taxes', 'insurance', 'interest', 'compound', 'wealth', 'cash',
  'spend', 'spending', 'mobile money', 'momo', 'airtel', 'sacco', 'ugx', 'uganda',
  'stanbic', 'dfcu', 'centenary', 'financial literacy', 'scam',
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
    const practicalBoost = /(budget|save|invest|expense|debt|loan|interest|tax|insurance|ugx|momo|airtel|sacco|scam)/.test(`${key} ${response}`.toLowerCase()) ? 2 : 0;
    const score = overlap * 3 + phraseBoost + Math.min(financeBoost, 3) + practicalBoost;

    if (score > bestScore) {
      bestScore = score;
      bestMatch = response;
    }
  }

  return bestScore >= 4 ? bestMatch : null;
};
