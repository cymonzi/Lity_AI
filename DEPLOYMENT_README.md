# 🚀 Lity AI - Deployment Guide

## 📖 Overview
Lity AI is SMK Moneykind's intelligent financial literacy chatbot, powered by a custom-trained DialoGPT model. This guide covers both local development and GitHub Pages deployment.

## 🏠 Local Development (Full AI Model)

### Prerequisites
- Node.js (v16 or higher)
- Python 3.8+ 
- Git

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/cymonzi/Lity_AI.git
   cd Lity_AI
   ```

2. **Install dependencies:**
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. **Start the full application:**
   ```bash
   # Start the AI model server (Terminal 1)
   start_local_model.bat
   
   # Start the React frontend (Terminal 2)
   start_lity_ai.bat
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## 🌐 GitHub Pages Deployment (Static Version)

### Features Available on GitHub Pages
- ✅ Full UI/UX experience
- ✅ Smart fallback responses for common financial topics
- ✅ Mobile-friendly design
- ✅ SMK Moneykind branding and content
- ❌ Custom DialoGPT model (replaced with intelligent fallbacks)

### Deployment Steps

1. **Automatic Deployment (Recommended):**
   ```bash
   # Run the deployment script
   deploy_to_github.bat
   ```

2. **Manual Deployment:**
   ```bash
   # Switch to production mode
   copy src\chatLogic.production.js src\chatLogic.js
   
   # Build and deploy
   npm run build
   npm run deploy
   ```

3. **Access your deployed site:**
   - 🔗 **Live Site:** https://cymonzi.github.io/Lity_AI/

## 📁 Project Structure

```
Lity_AI/
├── src/
│   ├── SMKChatbot.js          # Main chatbot component
│   ├── chatLogic.js           # Chat API logic (local)
│   ├── chatLogic.production.js # Production fallback logic
│   ├── LityHeader.js          # Header component
│   └── Litybottom.js          # Footer/input component
├── public/
│   └── index.html             # HTML template
├── lity-ai-final-model/       # Custom DialoGPT model (excluded from git)
├── local_model_server.py      # Python Flask server
├── requirements.txt           # Python dependencies
├── deploy_to_github.bat       # Deployment script
└── .gitignore                # Git exclusions
```

## 🔧 Configuration

### Environment Modes
- **Development:** Uses local DialoGPT model for authentic AI responses
- **Production:** Uses smart fallback system for instant, relevant responses

### Key Features
- 🎯 Financial literacy focus (budgeting, saving, investing)
- 📱 Mobile-responsive design
- 🎮 SMK Moneykind app integration (Litywise, Nfunayo)
- 💬 Typewriter effect for engaging conversations
- 📝 Smart disclaimer and user guidance

## 🛠️ Troubleshooting

### Local Development Issues
1. **Python model server fails:**
   - Check Python version (3.8+ required)
   - Install missing dependencies: `pip install -r requirements.txt`
   - Verify model files exist in `lity-ai-final-model/`

2. **React app won't start:**
   - Delete `node_modules` and run `npm install`
   - Check Node.js version (v16+ required)

3. **CORS errors:**
   - Ensure backend server is running on port 5000
   - Check firewall settings

### Deployment Issues
1. **GitHub Pages build fails:**
   - Check that `gh-pages` is installed: `npm install --save-dev gh-pages`
   - Verify repository settings allow GitHub Pages

2. **Site not updating:**
   - GitHub Pages can take 5-10 minutes to update
   - Check GitHub Actions tab for build status

## 📊 Performance

### Local Development
- ✅ Full AI model responses
- ✅ Contextual conversations
- ✅ Custom financial training data
- ⚠️ Requires local server setup

### GitHub Pages
- ✅ Instant loading
- ✅ No server dependencies
- ✅ Smart topical responses
- ✅ Always available
- ⚠️ Fallback responses only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test locally: `npm start`
5. Deploy and test: `npm run deploy`
6. Submit a pull request

## 📞 Support

For questions about SMK Moneykind or Lity AI:
- 📧 Email: support@smkmoneykind.com
- 🌐 Website: https://smkmoneykind.com
- 📱 Apps: Litywise & Nfunayo

---

**Made with ❤️ by SMK Moneykind Team**
*Empowering African youth through financial literacy*
