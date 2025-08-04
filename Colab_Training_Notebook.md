# Lity AI Model Training in Google Colab

This notebook will help you fine-tune a language model using the enhanced SMK Moneykind financial literacy dataset.

## Setup and Installation

```python
# Install required packages
!pip install transformers datasets accelerate peft bitsandbytes trl
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
```

## Load and Prepare Dataset

```python
# Upload your lity_ai_training_data.json file to Colab first
# Or download it from your repository

# Load the training data
with open('lity_ai_training_data.json', 'r') as f:
    training_data = json.load(f)

print(f"Loaded {len(training_data)} training examples")

# Preview first example
print("\\nSample training example:")
print("Instruction:", training_data[0]['instruction'][:100] + "...")
print("Input:", training_data[0]['input'])
print("Output:", training_data[0]['output'][:200] + "...")
```

## Format Data for Training

```python
def format_instruction(example):
    \"\"\"Format the data for instruction tuning\"\"\"
    return f\"\"\"### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}\"\"\"

# Format all examples
formatted_data = [{"text": format_instruction(ex)} for ex in training_data]

# Create dataset
dataset = Dataset.from_list(formatted_data)

# Split into train/validation (90/10)
train_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Training examples: {len(train_dataset['train'])}")
print(f"Validation examples: {len(train_dataset['test'])}")
```

## Model Configuration

```python
# Model configuration
model_name = "microsoft/DialoGPT-medium"  # Good for conversational AI
# Alternative: "facebook/opt-1.3b" or "EleutherAI/gpt-neo-1.3B"

# Quantization config for memory efficiency
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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

## LoRA Configuration

```python
# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # DialoGPT specific
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(f"Trainable parameters: {model.num_parameters()}")
```

## Tokenization

```python
def tokenize_function(examples):
    \"\"\"Tokenize the text\"\"\"
    # Tokenize
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=1024,  # Adjust based on your data
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Tokenize datasets
tokenized_train = train_dataset['train'].map(tokenize_function, batched=True)
tokenized_val = train_dataset['test'].map(tokenize_function, batched=True)

print("Tokenization complete!")
```

## Training Configuration

```python
# Training arguments
training_args = TrainingArguments(
    output_dir="./lity-ai-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=None,
    fp16=True,  # Enable mixed precision
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
)

print("Training configuration set!")
```

## Custom Trainer

```python
class FinancialLiteracyTrainer(Trainer):
    \"\"\"Custom trainer for financial literacy model\"\"\"
    
    def compute_loss(self, model, inputs, return_outputs=False):
        \"\"\"Custom loss computation\"\"\"
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

# Initialize trainer
trainer = FinancialLiteracyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

print("Trainer initialized!")
```

## Training

```python
# Start training
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model("./lity-ai-final-model")
tokenizer.save_pretrained("./lity-ai-final-model")

print("Training complete! Model saved.")
```

## Evaluation

```python
# Load evaluation prompts
with open('evaluation_prompts.json', 'r') as f:
    eval_prompts = json.load(f)

def generate_response(prompt, max_length=256):
    \"\"\"Generate response from the trained model\"\"\"
    instruction = "You are Lity AI, a friendly financial assistant from SMK Moneykind. Provide helpful, accurate, and engaging financial guidance suitable for young people in Africa."
    
    input_text = f\"\"\"### Instruction:
{instruction}

### Input:
{prompt}

### Response:
\"\"\"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response = response.split("### Response:")[-1].strip()
    
    return response

# Test with evaluation prompts
print("Testing trained model:")
print("=" * 50)

for i, prompt in enumerate(eval_prompts[:3]):  # Test first 3
    print(f"\\nPrompt {i+1}: {prompt}")
    print(f"Response: {generate_response(prompt)}")
    print("-" * 30)
```

## Export for Deployment

```python
# Save model in a format suitable for deployment
final_model_path = "./lity-ai-deployment-ready"

# Merge LoRA weights into base model for deployment
merged_model = model.merge_and_unload()
merged_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Deployment-ready model saved to: {final_model_path}")

# Create a simple inference script
inference_script = '''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the trained model
model_path = "./lity-ai-deployment-ready"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def ask_lity(question):
    """Ask Lity AI a question"""
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
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Example usage
if __name__ == "__main__":
    question = "How can I start saving money as a student?"
    answer = ask_lity(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
'''

with open("lity_ai_inference.py", "w") as f:
    f.write(inference_script)

print("Inference script created: lity_ai_inference.py")
```

## Model Performance Tips

```python
# Performance monitoring and tips
print("🎯 Model Training Tips:")
print("1. Monitor loss curves - should decrease steadily")
print("2. Check for overfitting - validation loss should not increase significantly")
print("3. Adjust learning rate if loss plateaus")
print("4. Use gradient clipping if training becomes unstable")
print("5. Experiment with different batch sizes and accumulation steps")

print("\\n📊 Evaluation Suggestions:")
print("1. Test with diverse financial questions")
print("2. Check for African context relevance")
print("3. Verify mobile money and local currency mentions")
print("4. Ensure culturally appropriate advice")
print("5. Test edge cases and potential harmful outputs")

print("\\n🚀 Deployment Recommendations:")
print("1. Add safety filters for financial advice")
print("2. Implement confidence scoring")
print("3. Add fallback to predefined responses")
print("4. Monitor user interactions and feedback")
print("5. Regular updates with new financial information")
```

## Usage Instructions

1. **Upload your files to Colab:**
   - `lity_ai_training_data.json`
   - `evaluation_prompts.json`

2. **Run cells in order:**
   - Setup and installation
   - Load and prepare dataset
   - Model configuration
   - Training
   - Evaluation

3. **Download trained model:**
   - After training, download the model files
   - Use the inference script for deployment

4. **Integration with your chatbot:**
   - Replace the backend model with your trained model
   - Update the API endpoint to use the new model
   - Test thoroughly before deployment

## Expected Results

Your trained model should:
- Understand financial concepts specific to African contexts
- Provide advice considering mobile money and local banking
- Maintain the friendly, educational tone of Lity AI
- Give practical, actionable financial guidance
- Show cultural awareness in responses

## Troubleshooting

**Common Issues:**
1. **Out of memory:** Reduce batch size or enable gradient checkpointing
2. **Slow training:** Use mixed precision (fp16) and gradient accumulation
3. **Poor performance:** Increase training epochs or adjust learning rate
4. **Overfitting:** Add more regularization or reduce model complexity

Happy training! 🚀
