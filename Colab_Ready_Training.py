# Enhanced Lity AI Training - Google Colab Version
# Copy this entire code into a Google Colab notebook cell and run

# 1. Install required libraries
!pip install transformers datasets accelerate peft bitsandbytes
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings('ignore')

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 2. Upload your enhanced dataset files using Colab file browser:
# - lity_ai_training_data.json (the new enhanced dataset)
# - evaluation_prompts.json (for testing)

# 3. Load and preprocess the enhanced dataset
print("Loading enhanced training dataset...")

# Load the new structured training data
with open('lity_ai_training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data)} enhanced training examples")

# New preprocessing function for instruction-tuned format
def format_instruction(example):
    """Format data for instruction tuning (much better than simple User/Bot format)"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

# Format all examples
formatted_data = [{"text": format_instruction(ex)} for ex in training_data]

# Create dataset and split for validation
dataset = Dataset.from_list(formatted_data)
train_dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Training examples: {len(train_dataset['train'])}")
print(f"Validation examples: {len(train_dataset['test'])}")

# Preview formatted example
print("\n📝 Sample formatted training example:")
print(formatted_data[0]["text"][:300] + "...")

# 4. Load model and tokenizer with enhanced configuration
model_name = "microsoft/DialoGPT-medium"  # Upgraded from small to medium
print(f"Loading model: {model_name}")

# Enhanced quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("✅ Model and tokenizer loaded successfully")

# 5. Setup LoRA for efficient fine-tuning (NEW!)
print("Setting up LoRA configuration...")

lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # DialoGPT specific
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# 6. Enhanced tokenization function
def tokenize_function(examples):
    """Enhanced tokenization with proper handling"""
    # Increased max_length for longer, more detailed responses
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # Increased from 128 to handle detailed responses
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset['train'].map(tokenize_function, batched=True)
tokenized_val = train_dataset['test'].map(tokenize_function, batched=True)
print("✅ Tokenization complete")

# 7. Enhanced training arguments
training_args = TrainingArguments(
    output_dir="./lity-ai-results",
    num_train_epochs=3,  # Reduced from 5 to prevent overfitting
    per_device_train_batch_size=1,  # Reduced due to longer sequences
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # NEW: Effective batch size = 4
    warmup_steps=50,  # NEW: Warmup for stable training
    learning_rate=2e-4,  # NEW: Optimized learning rate
    weight_decay=0.01,  # NEW: Regularization
    logging_steps=10,  # More frequent logging
    evaluation_strategy="steps",  # NEW: Evaluate during training
    eval_steps=25,  # NEW: Evaluate every 25 steps
    save_steps=50,  # More frequent saves
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    fp16=True,  # NEW: Mixed precision for efficiency
    gradient_checkpointing=True,  # NEW: Memory optimization
    dataloader_pin_memory=False,
)

print("✅ Training configuration set")

# 8. Enhanced trainer with validation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,  # NEW: Validation dataset
    tokenizer=tokenizer,
)

print("🚀 Starting enhanced training...")
print("This will take longer but produce much better results!")

# Train the model
trainer.train()

print("✅ Training completed!")

# 9. Enhanced model saving and testing
print("Saving enhanced model...")

# Save the LoRA adapter
trainer.save_model("./lity-ai-lora-adapter")

# Merge LoRA weights for deployment
print("Merging LoRA weights for deployment...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./lity-ai-final-model")
tokenizer.save_pretrained("./lity-ai-final-model")

# 10. Test the enhanced model
print("\n🧪 Testing the enhanced model...")

def generate_response(prompt, max_length=300):
    """Generate response using the trained model"""
    instruction = "You are Lity AI, a friendly financial assistant from SMK Moneykind. Provide helpful, accurate, and engaging financial guidance suitable for young people in Africa."
    
    input_text = f"""### Instruction:
{instruction}

### Input:
{prompt}

### Response:
"""
    
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = merged_model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Test with sample questions
test_questions = [
    "What is SMK Moneykind?",
    "How should I start saving money as a student?",
    "What's the difference between saving and investing?",
    "How can I use mobile money safely?"
]

print("\n📋 Testing enhanced model responses:")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n🤔 Question {i}: {question}")
    response = generate_response(question)
    print(f"🤖 Lity AI: {response}")
    print("-" * 40)

# 11. Package and download the enhanced model
print("\n📦 Packaging enhanced model for download...")

!zip -r lity_ai_enhanced_model.zip lity-ai-final-model

# Create enhanced inference script
inference_script = '''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the enhanced trained model
model_path = "./lity-ai-final-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def ask_lity(question, max_length=300):
    """Ask Lity AI a financial question"""
    instruction = "You are Lity AI, a friendly financial assistant from SMK Moneykind. Provide helpful, accurate, and engaging financial guidance suitable for young people in Africa."
    
    input_text = f"""### Instruction:
{instruction}

### Input:
{question}

### Response:
"""
    
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Example usage
if __name__ == "__main__":
    while True:
        question = input("Ask Lity AI: ")
        if question.lower() in ['quit', 'exit', 'bye']:
            break
        answer = ask_lity(question)
        print(f"Lity AI: {answer}\\n")
'''

with open("enhanced_lity_inference.py", "w") as f:
    f.write(inference_script)

print("✅ Enhanced inference script created: enhanced_lity_inference.py")

# Download everything
from google.colab import files
files.download("lity_ai_enhanced_model.zip")
files.download("enhanced_lity_inference.py")

print("\n🎉 TRAINING COMPLETE!")
print("=" * 50)
print("✅ Enhanced model trained successfully")
print("✅ Model saved and packaged")
print("✅ Inference script created")
print("✅ Files ready for download")
print("\n📈 Improvements in this version:")
print("• Used enhanced dataset with detailed responses")
print("• Implemented LoRA for efficient fine-tuning")
print("• Added validation dataset and monitoring")
print("• Increased sequence length for detailed responses")
print("• Used instruction-tuned format for better performance")
print("• Added memory optimization techniques")
print("• Created comprehensive testing framework")
print("\n🚀 Your model is now much smarter and ready for deployment!")
