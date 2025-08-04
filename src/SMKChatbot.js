import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, RefreshCw, MessageSquare, Sun, Moon, Trash2, ThumbsUp } from 'lucide-react';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import LityFooter from './Litybottom';
import LityHeader from './LityHeader';
import { chatWithBot, checkBackendHealth } from './chatLogic';
import { enhancedFAQ, findBestFAQMatch } from './enhancedFAQ';

const overview = `What Is Lity AI?

Lity AI is a smart, friendly financial assistant designed to help you understand and manage money better, one conversation at a time.

What can Lity AI do?
- Answer financial literacy questions
- Explain money concepts in simple terms
- Guide you through tools like Litywise and Nfunayo
- Support learners in SMK Moneykind trainings
- Help with budgeting, saving, and investment basics
- Provide personalized financial advice

How does it work?
Lity AI is powered by a fine-tuned language model trained on finance-focused Q&A, using SMK Moneykind's comprehensive knowledge base. It responds in a warm, engaging tone like a financial coach or friend.

Who is it for?
- Students learning about saving, budgeting, or investing
- Parents teaching kids about money
- Anyone confused about personal finance terms or tools
- SMK users seeking help with the app or trainings
- Young entrepreneurs and professionals

Try asking me about: "What is SMK Moneykind?", "How do I budget?", "Tell me about Litywise", or "Investment basics"`;

// Enhanced FAQ system with comprehensive responses
const faq = {
  // About SMK Moneykind
  "smk moneykind": "SMK Moneykind is a youth-focused financial literacy initiative built to empower a financially resilient generation. Through a unique ecosystem of gamified digital tools, engaging trainings, and practical resources, SMK Moneykind helps young people learn how to save, manage, and grow their money. Our platform includes the Litywise learning app, the Nfunayo expense tracker, and physical trainings offered in schools and communities.",
  
  "what is smk moneykind": "SMK Moneykind is a youth-focused financial literacy initiative built to empower a financially resilient generation. Through a unique ecosystem of gamified digital tools, engaging trainings, and practical resources, SMK Moneykind helps young people learn how to save, manage, and grow their money.",
  
  // About Lity
  "who is lity": "Lity is the cheerful face of the Litywise app, a lovable animated guide who makes learning about money fun and relatable. Lity is designed to engage users emotionally, using expressions and voice to deliver financial wisdom in bite-sized, interactive ways. Whether you're a kid just starting your money journey or a teen stepping into investing, Lity is right there with you, encouraging progress every step of the way.",
  
  "lity": "Lity is the cheerful face of the Litywise app, a lovable animated guide who makes learning about money fun and relatable. Lity is designed to engage users emotionally, using expressions and voice to deliver financial wisdom in bite-sized, interactive ways.",
  
  // About Litywise
  "litywise": "The Litywise app offers a fun, gamified learning journey where users can choose a financial path: Saver, Investor, or Boss, each tailored to a different age or skill level. As you progress, you unlock quizzes, stories, and mini-challenges that teach core money principles. You earn XP, collect badges, complete levels, and build your personal financial knowledge through consistent practice. It's like Duolingo meets financial literacy, with Lity as your money buddy.",
  
  "how does litywise work": "The Litywise app offers a fun, gamified learning journey where users can choose a financial path: Saver, Investor, or Boss, each tailored to a different age or skill level. As you progress, you unlock quizzes, stories, and mini-challenges that teach core money principles. You earn XP, collect badges, complete levels, and build your personal financial knowledge through consistent practice.",
  
  // Learning Paths
  "saver role": "The Saver role in Litywise is perfect for children or beginners. It focuses on teaching the basics of money: understanding needs vs wants, saving small amounts regularly, and spending wisely. The Saver path includes colorful visuals, short stories, and beginner-friendly quizzes that help lay a strong foundation for financial habits in a playful and stress-free environment.",
  
  "investor role": "The Investor role is designed for teens and intermediate learners who want to level up their financial skills. This path covers budgeting, understanding different types of investments, compound interest, and building wealth over time. You'll learn about stocks, bonds, savings accounts, and how to make your money work for you through smart financial decisions.",
  
  "boss role": "The Boss role is for young adults and advanced learners ready to tackle complex financial topics. This path covers entrepreneurship, business finance, credit management, loans, taxes, retirement planning, and long-term wealth building strategies. It's perfect for those stepping into adult financial responsibilities or starting their own ventures.",
  
  // About Nfunayo
  "nfunayo": "Nfunayo is now live and available through the SMK Moneykind platform. It's a lightweight, easy-to-use expense tracking tool that helps you monitor your income, spending, and saving goals in real time. Whether you're a student budgeting lunch money or a young adult managing your salary, Nfunayo helps you build clarity and confidence in your financial life.",
  
  "is nfunayo available": "Nfunayo is now live and available through the SMK Moneykind platform. It's a lightweight, easy-to-use expense tracking tool that helps you monitor your income, spending, and saving goals in real time.",
  
  "budget setup": "Setting up your budget in Nfunayo is simple. Start by adding your monthly income, then categorize your expenses like food, transport, entertainment, and savings. The app will help you track your spending against these categories and send gentle reminders when you're approaching your limits. You can adjust your budget anytime as your financial situation changes.",
  
  "savings goals": "Nfunayo lets you create multiple savings goals, whether it's for a new phone, school fees, or an emergency fund. You can set target amounts, deadlines, and track your progress with visual indicators. The app will even suggest how much to save weekly or monthly to reach your goals on time.",
  
  // Trainings
  "trainings": "Our physical trainings are a vital part of the SMK Moneykind experience. We work with schools, universities, and youth organizations to deliver interactive workshops that make finance engaging and accessible. These sessions are led by trained facilitators and sometimes our interns, bringing financial concepts to life with activities, role-plays, and discussions.",
  
  "physical trainings": "Our physical trainings are a vital part of the SMK Moneykind experience. We work with schools, universities, and youth organizations to deliver interactive workshops that make finance engaging and accessible.",
  
  "training topics": "Our physical trainings cover a wide range of topics including basic money management, budgeting, saving strategies, understanding bank accounts, digital payments, avoiding financial scams, entrepreneurship basics, and investment fundamentals. We tailor the content to match the audience's age and experience level.",
  
  // Financial Basics
  "budgeting": "Budgeting is about planning how to spend your money before you spend it. Start by listing your income, then your essential expenses (needs), followed by your wants. A simple rule: try the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. Track your spending and adjust as needed. Nfunayo can help you with this.",
  
  "saving": "Saving is putting money aside for future use. Start small, even 500 shillings a week adds up. Set specific goals (like 'save for a phone'), automate your savings if possible, and celebrate milestones. Remember: pay yourself first by saving before spending on wants.",
  
  "investment basics": "Investing means putting your money to work to grow over time. Start with understanding the difference between saving (keeping money safe) and investing (growing money with some risk). Popular beginner options include savings accounts, government bonds, and later stocks. Always invest money you won't need immediately.",
  
  "compound interest": "Compound interest is like magic for your money. It's when you earn interest not just on your original money, but also on the interest you've already earned. For example, if you save 100,000 shillings at 10% interest yearly, after year 1 you have 110,000, but year 2 you earn interest on the full 110,000. Time is your friend with compound interest.",
  
  "emergency fund": "An emergency fund is money saved for unexpected situations like medical bills, job loss, or urgent repairs. Aim to save 3-6 months of your basic expenses. Start small, even 10,000 shillings is better than nothing. Keep it in a separate, easily accessible account.",
  
  // SMK Ecosystem
  "ecosystem": "The SMK Moneykind ecosystem is made up of three core components:\n1. Litywise (Learn): a gamified learning app with quizzes and animations.\n2. Nfunayo (Track): an expense tracker that builds financial awareness.\n3. SMK Trainings (Practice): hands-on workshops and sessions in schools.\nTogether, these tools offer a full-circle approach to learning, applying, and mastering money skills.",
  
  // Age Groups
  "age group": "SMK Moneykind is designed to grow with you. It serves:\nKids (around 6-12) using the Saver path in Litywise.\nTeens (13-19) through the Investor path and workshops.\nYoung Adults (20+) with the Boss path and tools like Nfunayo.\nEach tool is built with age-appropriate content to ensure a smooth and relevant learning journey.",
  
  // Support and Contact
  "contact": "If you have any questions or need help, you can reach out through our website's contact form, email us at support@smkmoneykind.com, or connect with us on social media.",
  
  "support": "Our support team is available Monday through Friday, 8 AM to 6 PM East Africa Time. We respond to most inquiries within 24 hours. For urgent issues, you can also reach out through our social media channels for faster response.",
  
  // Technical
  "download apps": "You can download Litywise and Nfunayo from the Google Play Store for Android devices or the App Store for iOS devices. Simply search for 'SMK Moneykind', 'Litywise', or 'Nfunayo' and look for our official apps. Both apps are free to download and use.",
  
  "system requirements": "Litywise works on Android devices running Android 5.0 (API level 21) or higher, and iOS devices running iOS 10.0 or later. It requires about 100MB of storage space and works best with at least 2GB of RAM for smooth performance.",
  
  // Fun responses
  "joke": "Why did the dollar bill break up with the penny? Because it found someone who made more 'cents'!",
  
  "money joke": "Why did the dollar bill break up with the penny? Because it found someone who made more 'cents'!",
  
  "founder": "Our founder is Simon, also known as Cymon Zi. He's passionate about making money fun and easy to learn. He started SMK Moneykind after seeing too many young people struggle with financial decisions due to lack of proper money education.",
  
  // Greetings and common phrases
  "hi": "Hello! I'm Lity, your friendly money buddy from SMK Moneykind. I'm here to help you learn, grow, and win with money. What would you like to know about today?",
  
  "hello": "Hello! I'm Lity, your friendly money buddy from SMK Moneykind. I'm here to help you learn, grow, and win with money. What would you like to know about today?",
  
  "help": "I'm here to help you with all things money. I can explain financial concepts, guide you through our apps (Litywise and Nfunayo), tell you about our trainings, or answer specific questions about budgeting, saving, and investing. What would you like to learn about?",
  
  // Mobile Money specific (African context)
  "mobile money": "Nfunayo is designed with African mobile money systems in mind. You can easily track transactions from popular services like MTN Mobile Money, Airtel Money, and others. The app helps you categorize mobile money expenses and see where your money goes.",

  // Foundations of Money
  "what is money": "Money is a tool we use to trade for things we need and want. It can be coins, bills, or even digital numbers in your phone (like mobile money). Money helps us buy food, clothes, and fun things, but it's important to use it wisely.",
  
  "needs vs wants": "Needs are things you MUST have to live safely and healthily: food, water, shelter, clothes, and education. Wants are nice to have but not essential: toys, candy, or the latest phone. Smart money management means taking care of needs first, then using leftover money for wants.",
  
  "counting money": "Start by learning your local currency. Know the different coins and bills, practice making change, and always count twice. Use games to make it fun, like 'store' where you practice buying and selling with real or play money.",
  
  "smart spending": "Before buying anything, ask yourself: Do I really need this? Is this the best price? Can I wait and save up for something better? Compare prices, read reviews, and never rush into big purchases.",

  // Earning & Income  
  "ways to earn money": "For young people: help with chores at home, sell crafts or snacks to friends, offer services like car washing or tutoring younger kids. For adults: get a job, start a business, freelance your skills, or invest your money to make it grow.",
  
  "allowance system": "An allowance can teach money skills. Parents might give weekly money for completing chores or just for learning to manage money. Use it to practice budgeting: save some, spend some wisely, and maybe even give some to help others.",
  
  "entrepreneurship basics": "Starting a business means solving a problem for people and getting paid for it. Start small, maybe selling snacks at school, tutoring friends, or making things to sell. Learn about customers, costs, and profits. Every big business started with one small idea.",

  // Saving & Budgeting
  "why save money": "Saving gives you POWER. Power to buy bigger things later, power to handle emergencies, and power to reach your dreams. It's like planting seeds that grow into trees: a little saving today becomes a lot of money tomorrow.",
  
  "saving goals": "Make your goals SMART: Specific (I want a bike), Measurable (costs 200,000 shillings), Achievable (I can save 10,000 monthly), Relevant (I need transport), and Time-bound (in 20 months). Write it down and track your progress.",
  
  "emergency fund": "An emergency fund is money saved for unexpected problems like medical bills, job loss, or urgent repairs. Start with saving 10,000 shillings, then grow to 3-6 months of expenses. Keep it separate from spending money so you're not tempted to use it.",
  
  "budgeting basics": "A budget is a plan for your money. List your income (money coming in), then your expenses (money going out). Make sure income is higher than expenses. Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",

  // Spending Wisely
  "impulse buying": "Impulse buying is when you buy something suddenly without thinking. Fight it with the 24-hour rule: wait a day before buying anything over 20,000 shillings. Ask yourself: Will I still want this tomorrow? Do I have something similar already?",
  
  "delayed gratification": "This means waiting for something better instead of getting something smaller now. Like saving for a laptop instead of buying a cheap tablet. It's hard but SO worth it. Practice with small things first.",
  
  "comparing prices": "Never buy the first thing you see. Check at least 3 different places (online and offline). Look for sales, discounts, and reviews. Sometimes spending a bit more gets you much better quality that lasts longer.",

  // Banking & Financial Tools
  "what is a bank": "A bank is like a safe place for your money that also helps it grow. Banks keep your money secure, pay you interest for saving, and offer services like loans and money transfers. They're regulated by government to protect your money.",
  
  "bank vs mobile money": "Banks offer full services: savings, loans, investments, and physical locations. Mobile money (like MTN, Airtel) is great for quick payments and transfers but has fewer services. SACCOs are community-owned and often offer better rates for members.",
  
  "how interest works": "Interest is money the bank pays YOU for keeping your money with them, or money YOU pay for borrowing. If you save 100,000 at 5% interest yearly, you'll have 105,000 after one year. The longer you save, the more it grows.",

  // Money Ethics & Habits
  "lending to friends": "Only lend money you can afford to lose. Be clear about when and how it will be repaid. Write it down if it's a large amount. Remember: money can sometimes hurt friendships, so think carefully before lending or borrowing.",
  
  "money mistakes": "Everyone makes money mistakes, it's how we learn. Common ones: impulse buying, not saving for emergencies, or borrowing too much. The key is learning from mistakes and making better choices next time.",

  // Safety & Security
  "keeping money safe": "For physical money: use banks or secure hiding places, never carry large amounts. For digital money: use strong passwords, enable two-factor authentication, never share PINs, and only use trusted apps and websites.",
  
  "avoiding scams": "Remember: if it sounds too good to be true, it probably is. Never give personal info to strangers, don't click suspicious links, and verify everything independently. Common scams: 'get rich quick' schemes, fake investment opportunities, and phishing messages.",

  // Credit & Debt
  "what is credit": "Credit is borrowed money that you promise to pay back later, usually with interest. Good credit history helps you get loans for big purchases like cars or homes. Bad credit makes borrowing expensive or impossible.",
  
  "loans and borrowing": "Only borrow for things that improve your life long-term (education, business, home) or true emergencies. Understand the total cost including interest. Have a clear payback plan before borrowing anything.",

  // Investing Basics
  "what is investing": "Investing means using your money to buy things that might become worth more over time, like company shares, property, or bonds. It's riskier than saving but can help your money grow faster. Never invest money you need soon.",
  
  "stocks": "Stocks are tiny pieces of ownership in companies. When companies do well, stock prices often go up. When they struggle, prices can fall. It's like betting on which companies will succeed: exciting but risky.",
  
  "diversification": "Don't put all your eggs in one basket. Spread your investments across different companies, industries, and types of investments. If one fails, others might succeed. This reduces your overall risk.",

  // Taxes & Insurance
  "what are taxes": "Taxes are money we pay to the government for public services like roads, schools, hospitals, and security. Everyone who earns money pays taxes. Understanding taxes helps you plan better and avoid problems with authorities.",
  
  "insurance": "Insurance protects you from big financial losses. You pay small amounts regularly (premiums) so the insurance company pays big amounts if something bad happens (like accidents, illness, or theft). It's like a safety net.",

  // Retirement Planning
  "what is retirement": "Retirement is when you stop working but still need money to live. Smart people save throughout their working years so they have enough money for retirement. The earlier you start saving, the easier it becomes.",
  
  "compound interest": "This is the magic of money growing on money. If you save 100,000 at 10% interest, year 1 you have 110,000. Year 2, you earn interest on 110,000, getting 121,000. Your money grows faster and faster over time.",

  // Business & Entrepreneurship
  "how businesses make money": "Businesses make money by selling products or services for more than they cost to make. The difference is profit. They need to cover costs (materials, salaries, rent) and still have money left over to grow and reward owners.",
  
  "revenue vs profit": "Revenue is all the money coming in from sales. Profit is what's left after paying all expenses. A business might have high revenue but low profit if costs are too high. Profit is what really matters.",

  // Financial Psychology
  "financial habits": "Good money habits include: saving regularly, tracking expenses, avoiding debt, comparing prices, and learning about money. Start small, even saving 1,000 shillings weekly builds the habit. Habits become automatic with practice.",
  
  "money emotions": "Money can make us feel excited, scared, proud, or ashamed. These emotions affect our decisions. When feeling emotional about money, take a break, think logically, and maybe talk to someone you trust before making big financial choices."
};



function SMKChatbot() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: 'Hi there! I\'m Lity AI, your friendly financial assistant from SMK Moneykind.\n\nI\'m here to help you master money skills through fun, interactive conversations. Whether you\'re just starting your financial journey or looking to level up your money game, I\'ve got you covered.\n\nQuick Start Options:\n• Type "overview" to learn what I can do\n• Ask "What is SMK Moneykind?" to learn about our platform\n• Try "budgeting tips" or "how to save money"\n• Ask about "Litywise" or "Nfunayo" apps\n\nReady to become financially awesome? What would you like to explore first?',
      timestamp: new Date()
    }
  ]);
  const [streaming, setStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const streamRef = useRef(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState(true);
  const [darkMode, setDarkMode] = useState(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  const [showActions, setShowActions] = useState(false);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Handle window resize for responsive design
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

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
        sendBg: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)', // teal gradient for send button
        sendText: '#ffffff',
        heading: '#f1f5f9',
        error: '#ef4444',
        botBubble: '#334155',
        userBubble: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)', // teal gradient for user only
        accent: '#14b8a6' // teal accent for glowing border
      }
    : {
        bg: 'linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%)', // light teal gradient
        card: '#ffffff',
        border: '#5eead4', // light teal border
        shadow: '0 8px 32px rgba(20,184,166,0.08)',
        inputBg: '#ffffff',
        inputBorder: '#5eead4',
        inputText: '#134e4a',
        sendBg: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)', // teal gradient for send button
        sendText: '#134e4a',
        heading: '#134e4a',
        error: '#ef4444',
        botBubble: '#ffffff',
        userBubble: 'linear-gradient(135deg, #5eead4 0%, #14b8a6 100%)', // teal gradient
        accent: '#14b8a6' // teal accent for glowing border
      };

  // Auto-scroll to bottom with mobile optimization
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      // Immediate scroll without delay for real-time typing visibility
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
  }, []);

  // Separate effect for messages (immediate scroll)
  useEffect(() => {
    if (messagesEndRef.current) {
      // Immediate scroll for new messages
      setTimeout(() => {
        messagesEndRef.current.scrollIntoView({ 
          behavior: 'smooth',
          block: 'end',
          inline: 'nearest'
        });
      }, 50);
    }
  }, [messages]);

  // Separate effect for streaming text (real-time scroll during typing)
  useEffect(() => {
    if (streaming && streamedText && messagesEndRef.current) {
      // Real-time scroll during typing - no delay
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
  }, [streamedText, streaming]);

  // Check backend health
  useEffect(() => {
    checkBackendHealth().then(setBackendHealthy);
  }, []);

  // Enhanced FAQ matching function with dataset integration
  const findFAQMatch = (input) => {
    const lowerInput = input.toLowerCase().trim();
    
    // First try the enhanced FAQ from our dataset
    const datasetMatch = findBestFAQMatch(input);
    if (datasetMatch) {
      return datasetMatch;
    }
    
    // Then try our curated FAQ
    if (faq[lowerInput]) {
      return faq[lowerInput];
    }
    
    // Check for partial matches and keywords
    const keywords = {
      'overview': ['overview', 'what can you do', 'what do you do', 'capabilities', 'features'],
      'smk moneykind': ['smk', 'moneykind', 'smk moneykind', 'what is smk', 'about smk'],
      'litywise': ['litywise', 'lity wise', 'learning app', 'gamified app'],
      'nfunayo': ['nfunayo', 'expense tracker', 'track expenses', 'budget app'],
      'trainings': ['training', 'trainings', 'workshop', 'workshops', 'physical training'],
      'budgeting': ['budget', 'budgeting', 'how to budget', 'budget tips', 'budgeting tips'],
      'saving': ['save', 'saving', 'how to save', 'saving tips', 'save money'],
      'investment basics': ['invest', 'investment', 'investing', 'investment basics', 'how to invest'],
      'saver role': ['saver', 'saver role', 'saver path', 'beginner path'],
      'investor role': ['investor', 'investor role', 'investor path', 'intermediate path'],
      'boss role': ['boss', 'boss role', 'boss path', 'advanced path'],
      'compound interest': ['compound', 'compound interest', 'interest compound'],
      'emergency fund': ['emergency', 'emergency fund', 'emergency savings'],
      'contact': ['contact', 'support', 'help', 'get help'],
      'download apps': ['download', 'get app', 'install', 'app store', 'play store'],
      'founder': ['founder', 'who founded', 'simon', 'cymon', 'cymon zi'],
      'joke': ['joke', 'funny', 'humor', 'make me laugh'],
      'hi': ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon'],
      'mobile money': ['mobile money', 'mtn', 'airtel money', 'mobile payments']
    };
    
    // Find matching keywords
    for (const [key, variations] of Object.entries(keywords)) {
      for (const variation of variations) {
        if (lowerInput.includes(variation)) {
          return faq[key] || null;
        }
      }
    }
    
    // Special cases for complex queries
    if (lowerInput.includes('what is') && lowerInput.includes('moneykind')) {
      return faq['smk moneykind'];
    }
    
    if (lowerInput.includes('how') && (lowerInput.includes('work') || lowerInput.includes('use')) && lowerInput.includes('litywise')) {
      return faq['litywise'];
    }
    
    if ((lowerInput.includes('age') || lowerInput.includes('old')) && (lowerInput.includes('group') || lowerInput.includes('for'))) {
      return faq['age group'];
    }
    
    if (lowerInput.includes('ecosystem') || (lowerInput.includes('all') && lowerInput.includes('tools'))) {
      return faq['ecosystem'];
    }
    
    return null;
  };

  // Enhanced fallback response generator
  const generateFallbackResponse = (input) => {
    const lowerInput = input.toLowerCase();
    
    // Check if question is finance-related but not in our data
    const financeKeywords = ['money', 'budget', 'save', 'invest', 'loan', 'bank', 'finance', 'financial', 'currency', 'profit', 'income', 'expense', 'debt', 'credit', 'tax', 'insurance'];
    const isFinanceRelated = financeKeywords.some(keyword => lowerInput.includes(keyword));
    
    if (isFinanceRelated) {
      return `💡 That's a great financial question! While I don't have specific information about "${input}" in my current knowledge base, I can help you with:

📚 **Financial Basics**: budgeting, saving, investing, emergency funds
🎮 **Our Apps**: Litywise and Nfunayo features 
🏫 **SMK Programs**: trainings, workshops, learning paths
💰 **Money Management**: spending wisely, earning tips, mobile money

🔍 Try asking: 
• "How do I start budgeting?"
• "What is compound interest?"
• "Tell me about Litywise"
• "How can I save money?"

Or type "overview" to see everything I can help with!`;
    }
    
    // For non-finance questions
    return `🤔 I'm Lity AI, your financial assistant! I'm specifically designed to help with money and financial topics.

I'd love to help you with:
💰 **Personal Finance**: budgeting, saving, investing
🎓 **Financial Education**: learning about money management
🏢 **SMK Moneykind**: our apps, trainings, and programs
📱 **Practical Tips**: mobile money, entrepreneurship, financial safety

💡 Try asking about financial topics like:
• "How should I budget my money?"
• "What are good saving strategies?"
• "Tell me about investment basics"
• "What is SMK Moneykind?"

What financial topic would you like to explore?`;
  };

  // Validate bot response quality
  const isValidResponse = (response, input) => {
    if (!response || response.length < 10) return false;
    
    // Check if response is too generic or seems like hallucination
    const genericPhrases = [
      'i don\'t know',
      'i\'m not sure',
      'i cannot',
      'i can\'t help',
      'as an ai',
      'i don\'t have access',
      'i apologize'
    ];
    
    const lowerResponse = response.toLowerCase();
    const seemsGeneric = genericPhrases.some(phrase => lowerResponse.includes(phrase));
    
    // Check if response is relevant to input (basic check)
    const inputWords = input.toLowerCase().split(' ').filter(word => word.length > 3);
    const responseWords = response.toLowerCase().split(' ');
    const relevanceScore = inputWords.filter(word => 
      responseWords.some(respWord => respWord.includes(word) || word.includes(respWord))
    ).length / Math.max(inputWords.length, 1);
    
    return !seemsGeneric && relevanceScore > 0.1; // At least 10% relevance
  };

  // Handle send message with enhanced logic
  const handleSend = async () => {
    if (!input.trim() || loading || streaming) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setStreaming(false);
    setStreamedText('');
    const currentInput = input;
    setInput('');

    try {
      let reply = '';
      
      // First check for overview
      if (currentInput.toLowerCase().includes('overview')) {
        reply = overview;
      } else {
        // Try FAQ matching first
        const faqResponse = findFAQMatch(currentInput);
        if (faqResponse) {
          reply = faqResponse;
        } else if (backendHealthy) {
          // Try AI model
          try {
            const aiResponse = await chatWithBot(currentInput);
            
            // Validate the AI response quality
            if (isValidResponse(aiResponse, currentInput)) {
              reply = aiResponse;
            } else {
              // AI response seems invalid, use fallback
              reply = generateFallbackResponse(currentInput);
            }
          } catch (error) {
            console.log('AI model error, using fallback:', error);
            reply = generateFallbackResponse(currentInput);
          }
        } else {
          // Backend not healthy, use fallback
          reply = generateFallbackResponse(currentInput);
        }
      }

      // Start typewriter effect for bot response
      setStreaming(true);
      setLoading(false);
      let i = 0;
      streamRef.current = setInterval(() => {
        i++;
        setStreamedText(reply.slice(0, i));
        
        if (i >= reply.length) {
          clearInterval(streamRef.current);
          setStreaming(false);
          setMessages(prev => [...prev, {
            id: Date.now() + 1,
            sender: 'bot',
            text: reply,
            timestamp: new Date()
          }]);
          setStreamedText('');
        }
      }, 18); // speed of typewriter
    } catch (error) {
      setStreaming(false);
      setLoading(false);
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        text: generateFallbackResponse(currentInput),
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  // Stop streaming response
  const handleStop = () => {
    if (streaming && streamRef.current) {
      clearInterval(streamRef.current);
      setStreaming(false);
      setLoading(false);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'bot',
        text: streamedText,
        timestamp: new Date()
      }]);
      setStreamedText('');
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
      maxWidth: '100vw',
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
      zIndex: 9999,
      WebkitOverflowScrolling: 'touch', // iOS smooth scrolling
      touchAction: 'manipulation' // Better touch handling
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
        padding: windowWidth <= 480 ? '0 0 160px 0' : '0 0 150px 0', // Much more space for full visibility
        minHeight: 0,
        height: '100%',
        WebkitOverflowScrolling: 'touch', // iOS smooth scrolling
        scrollBehavior: 'smooth', // Smooth scrolling for all browsers
        position: 'relative'
      }}>
        {/* Today Divider */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          margin: '16px 0 12px 0',
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
        {/* Render messages as usual */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              display: 'flex',
              justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              margin: '8px 0',
              padding: '0 16px',
              width: '100%',
              boxSizing: 'border-box'
            }}
          >
            <ChatBubble
              sender={msg.sender}
              text={msg.text}
              theme={theme}
              style={{
                maxWidth: windowWidth <= 768 ? '85%' : '70%', // More space on mobile
                minWidth: windowWidth <= 480 ? '200px' : '250px',
                padding: windowWidth <= 480 ? '10px 14px' : '12px 16px',
                borderRadius: msg.sender === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                background: msg.sender === 'user' ? theme.userBubble : theme.botBubble,
                color: msg.sender === 'user' ? 'white' : theme.heading,
                boxShadow: theme.shadow,
                border: `1px solid ${theme.border}`,
                fontSize: windowWidth <= 480 ? '14px' : '15px',
                lineHeight: '1.5',
                wordBreak: 'break-word',
                overflowWrap: 'break-word',
                hyphens: 'auto',
                animation: 'fadeInUp 0.3s ease-out',
                position: 'relative'
              }}
            />
          </div>
        ))}

        {/* Streaming bot response (typewriter effect) */}
        {streaming && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start',
            margin: '8px 0',
            padding: '0 16px',
            width: '100%',
            boxSizing: 'border-box'
          }}>
            <ChatBubble
              sender="bot"
              text={streamedText}
              theme={theme}
              style={{
                maxWidth: windowWidth <= 768 ? '85%' : '70%',
                minWidth: windowWidth <= 480 ? '200px' : '250px',
                padding: windowWidth <= 480 ? '10px 14px' : '12px 16px',
                borderRadius: '20px 20px 20px 4px',
                background: theme.botBubble,
                color: theme.heading,
                boxShadow: theme.shadow,
                border: `1px solid ${theme.border}`,
                fontSize: windowWidth <= 480 ? '14px' : '15px',
                lineHeight: '1.5',
                wordBreak: 'break-word',
                overflowWrap: 'break-word',
                hyphens: 'auto',
                animation: 'fadeInUp 0.3s ease-out',
                position: 'relative'
              }}
            />
          </div>
        )}
        <div ref={messagesEndRef} style={{ height: '40px', minHeight: '40px' }} />
      </div>


      {/* Quick Actions */}
      {showActions && (
        <div style={{
          padding: windowWidth <= 480 ? '12px 16px' : '16px 24px',
          background: theme.card,
          borderTop: `1px solid ${theme.border}`,
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          justifyContent: 'center',
          maxHeight: '120px',
          overflowY: 'auto',
          WebkitOverflowScrolling: 'touch'
        }}>
          {[
            { key: 'overview', label: '📋 Overview', query: 'overview' },
            { key: 'budgeting', label: '💰 Budgeting', query: 'budgeting tips' },
            { key: 'saving', label: '🏦 Saving', query: 'how to save money' },
            { key: 'litywise', label: '🎮 Litywise', query: 'Tell me about Litywise' },
            { key: 'nfunayo', label: '📊 Nfunayo', query: 'What is Nfunayo?' },
            { key: 'investing', label: '📈 Investing', query: 'investment basics' },
            { key: 'trainings', label: '🎓 Trainings', query: 'physical trainings' },
            { key: 'smk', label: '🏢 About SMK', query: 'What is SMK Moneykind?' }
          ].map((action) => (
            <button
              key={action.key}
              onClick={() => {
                setInput(action.query);
                setShowActions(false);
                inputRef.current?.focus();
              }}
              style={{
                background: 'none',
                border: `1px solid ${theme.border}`,
                borderRadius: '20px',
                padding: windowWidth <= 480 ? '8px 12px' : '6px 12px',
                color: theme.heading,
                cursor: 'pointer',
                fontSize: windowWidth <= 480 ? '11px' : '12px',
                transition: 'all 0.2s ease',
                fontWeight: '500',
                minWidth: 'fit-content',
                whiteSpace: 'nowrap',
                touchAction: 'manipulation'
              }}
            >
              {action.label}
            </button>
          ))}
        </div>
      )}


      {/* Footer Input Area - Integrated with disclaimer */}
      <div style={{
        position: 'fixed',
        left: 0,
        right: 0,
        bottom: 0,
        background: theme.bg,
        borderTop: `1px solid ${theme.border}`,
        zIndex: 10001,
        paddingTop: '12px'
      }}>
        {/* LityFooter */}
        <LityFooter
          input={input}
          setInput={setInput}
          handleSend={handleSend}
          loading={loading || streaming}
          backendHealthy={backendHealthy}
          theme={theme}
          handleClear={handleClear}
          handleFeedback={handleFeedback}
          messages={messages}
          inputRef={inputRef}
          streaming={streaming}
          handleStop={handleStop}
        />
        
        {/* Integrated Disclaimer */}
        <div style={{
          textAlign: 'center',
          fontSize: windowWidth <= 480 ? '10px' : '11px',
          color: darkMode ? '#94a3b8' : '#64748b', // More subtle color
          padding: windowWidth <= 480 ? '8px 12px 12px 12px' : '10px 16px 16px 16px',
          fontWeight: 400,
          lineHeight: 1.3,
          opacity: 0.8
        }}>
          Lity AI can make mistakes. Take Care.
        </div>
      </div>



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
        
        /* Mobile-specific improvements */
        @media (max-width: 768px) {
          body {
            -webkit-overflow-scrolling: touch;
            overflow-scrolling: touch;
          }
          
          /* Prevent zoom on input focus on iOS */
          input, textarea, select {
            font-size: 16px !important;
          }
          
          /* Better touch targets */
          button {
            min-height: 44px;
            min-width: 44px;
          }
        }
        
        @media (max-width: 480px) {
          /* Smaller screens adjustments */
          .chat-bubble {
            font-size: 14px;
            padding: 10px 14px;
          }
          
          .quick-action-button {
            font-size: 11px;
            padding: 8px 12px;
          }
        }
        
        /* Smooth scrolling for all elements */
        * {
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
        }
        
        /* Improve text rendering on mobile */
        @media (max-width: 768px) {
          * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
          }
        }
      `}</style>
    </div>
  );
}

export default SMKChatbot;
