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

tokenizer = AutoTokenizer.from_pretrained("./smk_moneykind_dialoGPT")
model = AutoModelForCausalLM.from_pretrained("./smk_moneykind_dialoGPT")

# Simple global history (for demo, not per user)
chat_history_ids = None

@app.post("/chat/")
async def chat(request: Request):
    global chat_history_ids
    data = await request.json()
    user_text = data.get("text", "")

    # Format input for fine-tuned model (match training format)
    formatted_input = f"User: {user_text}\nBot:"
    new_input_ids = tokenizer.encode(formatted_input, return_tensors="pt")
    # For single-turn, reset chat_history_ids each time
    bot_input_ids = new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"User: {user_text} | Bot: {reply}")  # Debug log
    # Remove 'Bot: ' prefix if present
    if reply.startswith("Bot: "):
        reply = reply[len("Bot: "):]

    return {"reply": reply}
