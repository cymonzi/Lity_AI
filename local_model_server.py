#!/usr/bin/env python3
"""
Local Model Server for Lity AI Chatbot
Uses the trained DialoGPT model for financial conversations
"""

import os
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

class LityAIModel:
    def __init__(self, model_path="./lity-ai-final-model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def generate_response(self, user_input, max_length=150, temperature=0.7, do_sample=True):
        """Generate a response using the trained model"""
        try:
            # Prepare input text with conversation format
            input_text = f"User: {user_input} Bot:"
            
            # Encode input
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the bot's response
            if "Bot:" in full_response:
                bot_response = full_response.split("Bot:")[-1].strip()
                
                # Clean up the response
                bot_response = bot_response.replace("User:", "").strip()
                
                # Limit response length
                if len(bot_response) > 500:
                    bot_response = bot_response[:500] + "..."
                
                return bot_response if bot_response else self.get_fallback_response(user_input)
            else:
                return self.get_fallback_response(user_input)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self.get_fallback_response(user_input)
    
    def get_fallback_response(self, user_input):
        """Provide fallback responses based on keywords"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['budget', 'budgeting']):
            return "Budgeting is about planning your money before you spend it! Start by listing your income, then your essential expenses (needs), followed by wants. Try the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings."
        
        elif any(word in user_lower for word in ['save', 'saving', 'savings']):
            return "Saving is putting money aside for future use. Start small - even 500 shillings a week adds up! Set specific goals, automate your savings if possible, and celebrate milestones. Remember: pay yourself first!"
        
        elif any(word in user_lower for word in ['invest', 'investment', 'investing']):
            return "Investing means putting your money to work to grow over time. Start with understanding the difference between saving (keeping money safe) and investing (growing money with some risk). Popular beginner options include savings accounts, government bonds, and later stocks."
        
        elif 'litywise' in user_lower:
            return "The Litywise app offers a fun, gamified learning journey where you can choose a financial path: Saver, Investor, or Boss, each tailored to different skill levels. You earn XP, collect badges, and build your financial knowledge through consistent practice!"
        
        elif 'nfunayo' in user_lower:
            return "Nfunayo is our expense tracking tool that helps you monitor your income, spending, and saving goals in real time. Whether you're budgeting lunch money or managing your salary, Nfunayo helps build financial awareness!"
        
        else:
            return "I'm here to help you learn about money management! I can explain budgeting, saving, investing, and guide you through our SMK Moneykind tools like Litywise and Nfunayo. What would you like to explore?"

# Initialize the model
try:
    lity_model = LityAIModel()
    logger.info("Lity AI model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    lity_model = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Lity AI Local Model Server is running",
        "model_loaded": lity_model is not None
    })

@app.route('/chat/', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('text', '').strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if lity_model is None:
            return jsonify({
                "reply": "I'm having trouble loading my AI model right now. But I can still help with basic financial questions! Try asking about budgeting, saving, or investing."
            })
        
        # Generate response using the model
        response = lity_model.generate_response(user_message)
        
        logger.info(f"User: {user_message}")
        logger.info(f"Bot: {response}")
        
        return jsonify({"reply": response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "reply": "I encountered an error processing your message. Let me help you with financial basics instead! What would you like to learn about?"
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint with model info"""
    return jsonify({
        "status": "running",
        "model_loaded": lity_model is not None,
        "device": str(lity_model.device) if lity_model else "none",
        "model_path": lity_model.model_path if lity_model else "none"
    })

if __name__ == '__main__':
    print("🚀 Starting Lity AI Local Model Server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🤖 Model ready for financial conversations!")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
