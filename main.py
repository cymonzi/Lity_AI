from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allows all origins
    allow_methods=["*"],      # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],      # Allows all headers
)

@app.get("/")
async def health():
    return {"status": "ok"}

# --- Conversational AI setup ---
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Enhanced model loading with fallback
model = None
tokenizer = None
model_loaded = False

try:
    # Check if model directory exists
    model_path = "smk_moneykind_dialoGPT"
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Check if model files exist
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']
        
        config_exists = any(os.path.exists(os.path.join(model_path, f)) for f in required_files)
        model_file_exists = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
        
        if config_exists and model_file_exists:
            print("Loading custom SMK Moneykind model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model_loaded = True
            print("✅ Custom model loaded successfully!")
        else:
            print("⚠️ Model directory exists but missing required files")
    else:
        print("⚠️ Custom model directory not found")
        
except Exception as e:
    print(f"❌ Error loading custom model: {e}")

# Fallback to a public model if custom model fails
if not model_loaded:
    try:
        print("🔄 Falling back to DialoGPT-medium...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        model_loaded = True
        print("✅ Fallback model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading fallback model: {e}")
        print("⚠️ Running without AI model - will use predefined responses only")

# Enhanced FAQ responses for better offline experience
predefined_responses = {
    "what is smk moneykind": "SMK Moneykind is a youth-focused financial literacy initiative built to empower a financially resilient generation through gamified digital tools, engaging trainings, and practical resources.",
    "tell me about litywise": "Litywise is our gamified learning app where you can choose from three paths - Saver (beginners), Investor (intermediate), or Boss (advanced). You earn XP, collect badges, and learn money skills through fun quizzes and stories!",
    "what is nfunayo": "Nfunayo is our lightweight expense tracking tool that helps you monitor income, spending, and saving goals in real time. Perfect for students and young adults managing their money!",
    "how to save money": "Start with the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. Set specific goals, start small (even 500 shillings weekly helps!), and use Nfunayo to track your progress.",
    "budgeting tips": "Create a simple budget by listing income first, then essential expenses (needs), then wants. Track everything for a month to see patterns. Use the 50/30/20 rule as a starting point and adjust based on your situation.",
    "investment basics": "Investing means putting money to work to grow over time. Start by understanding the difference between saving (keeping money safe) and investing (growing money with some risk). Begin with savings accounts, then explore bonds and stocks as you learn more.",
    "hello": "Hello! 👋 I'm Lity AI from SMK Moneykind. I'm here to help you master money skills! Ask me about budgeting, saving, our apps, or type 'what can you do' to see all my capabilities.",
    "hi": "Hi there! 👋 I'm your friendly financial assistant Lity. Ready to learn about money management, budgeting, saving, or our awesome apps? What would you like to explore first?",
    "what can you do": "I can help you with: 💰 Budgeting & saving tips, 🎮 Learn about Litywise app, 📊 Nfunayo expense tracker, 🎓 SMK Moneykind trainings, 📈 Investment basics, 🏦 Financial literacy concepts. What interests you most?"
}

@app.post("/chat/")
async def chat(request: Request):
    global chat_history_ids
    data = await request.json()
    user_text = data.get("text", "").lower().strip()

    # Check for predefined responses first
    for key, response in predefined_responses.items():
        if key in user_text or any(word in user_text for word in key.split()):
            return {"reply": response}
    
    # Enhanced keyword matching
    if any(word in user_text for word in ["budget", "budgeting"]):
        return {"reply": predefined_responses["budgeting tips"]}
    elif any(word in user_text for word in ["save", "saving"]):
        return {"reply": predefined_responses["how to save money"]}
    elif any(word in user_text for word in ["invest", "investment"]):
        return {"reply": predefined_responses["investment basics"]}
    elif any(word in user_text for word in ["hello", "hi", "hey"]):
        return {"reply": predefined_responses["hello"]}
    
    # If model is loaded, try to generate response
    if model_loaded and model is not None and tokenizer is not None:
        try:
            # Format input for fine-tuned model (match training format)
            formatted_input = f"User: {data.get('text', '')}\nBot:"
            new_input_ids = tokenizer.encode(formatted_input, return_tensors="pt")
            
            # Generate response
            chat_history_ids = model.generate(
                new_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            reply = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            # Remove 'Bot: ' prefix if present
            if reply.startswith("Bot: "):
                reply = reply[len("Bot: "):]
                
            if reply.strip():
                return {"reply": reply}
        except Exception as e:
            print(f"Error generating AI response: {e}")
    
    # Fallback response
    return {"reply": "I'd love to help you with that! I'm great at explaining budgeting, saving, investing, and our SMK Moneykind tools. Try asking about 'budgeting tips', 'how to save money', 'what is Litywise', or 'investment basics'. What specific topic interests you most?"}
