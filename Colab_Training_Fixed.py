# ✅ Enhanced Lity AI Training Script (Colab Optimized - Error Resistant)

# 1. Memory Management and Setup
import gc
import psutil
import torch
torch.cuda.empty_cache()
gc.collect()

# Check available resources
def check_resources():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)} | Memory: {gpu_memory:.1f}GB")
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f"💾 RAM: {ram_gb:.1f}GB available")
    return gpu_memory if torch.cuda.is_available() else 0, ram_gb

gpu_memory, ram_memory = check_resources()

# 2. Install Dependencies with Better Error Handling
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        print(f"✅ {package} installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False
    return True

# Install packages one by one for better error tracking
packages = [
    "transformers==4.40.0",
    "datasets==2.19.0", 
    "accelerate==0.30.0",
    "peft==0.8.2",
    "bitsandbytes==0.43.0"
]

for package in packages:
    if not install_package(package):
        print(f"⚠️ Continuing without {package}...")

# Install PyTorch separately
try:
    !pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print("✅ PyTorch installed successfully")
except Exception as e:
    print(f"❌ PyTorch installation failed: {e}")

# 3. Import Libraries with Error Handling
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import json
    import os
    import random
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
        BitsAndBytesConfig, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    print("✅ All libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Trying alternative imports...")

# 4. Enhanced Seed Setting
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# 5. Enhanced Environment Check
def detailed_environment_check():
    print("🔍 Environment Details:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Colab: {'google.colab' in str(get_ipython())}")

detailed_environment_check()

# 6. Safe File Upload with Validation
def safe_file_upload():
    try:
        from google.colab import files
        print("📁 Please upload your training data file...")
        uploaded = files.upload()
        
        if not uploaded:
            raise ValueError("No file uploaded")
            
        filename = list(uploaded.keys())[0]
        print(f"✅ File uploaded: {filename}")
        
        # Validate file
        if not filename.endswith('.json'):
            print("⚠️ Warning: File doesn't end with .json")
        
        file_size = len(uploaded[filename]) / 1024 / 1024  # MB
        print(f"📊 File size: {file_size:.2f} MB")
        
        return filename
        
    except Exception as e:
        print(f"❌ File upload failed: {e}")
        print("📝 Creating sample data for testing...")
        
        # Create sample data if upload fails
        sample_data = [
            {
                "instruction": "Respond as Lity AI assistant",
                "input": "Hello, how are you?",
                "output": "Hello! I'm Lity AI, and I'm doing well. How can I assist you today?"
            },
            {
                "instruction": "Provide helpful information",
                "input": "What can you help me with?",
                "output": "I can help you with various tasks including answering questions, providing information, and assisting with problem-solving."
            }
        ] * 10  # Duplicate for training
        
        with open("lity_ai_training_data.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        return "lity_ai_training_data.json"

filename = safe_file_upload()

# 7. Enhanced Data Loading with Validation
def load_and_validate_data(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"📊 Loaded {len(data)} examples")
        
        # Validate data structure
        required_keys = ["instruction", "input", "output"]
        valid_examples = []
        
        for i, example in enumerate(data):
            if all(key in example for key in required_keys):
                # Clean and validate text
                for key in required_keys:
                    if not isinstance(example[key], str):
                        example[key] = str(example[key])
                    example[key] = example[key].strip()
                
                valid_examples.append(example)
            else:
                print(f"⚠️ Skipping invalid example {i}: missing required keys")
        
        print(f"✅ {len(valid_examples)} valid examples after validation")
        
        if len(valid_examples) < 10:
            raise ValueError("Too few valid examples for training")
        
        return valid_examples
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise

training_data = load_and_validate_data(filename)

# 8. Smart Data Formatting
def format_instruction(ex):
    template = f"""### Instruction:
{ex['instruction']}

### Input:
{ex['input']}

### Response:
{ex['output']}"""
    return template

def prepare_dataset(data):
    try:
        formatted_data = []
        max_length = 0
        
        for ex in data:
            formatted_text = format_instruction(ex)
            formatted_data.append({"text": formatted_text})
            max_length = max(max_length, len(formatted_text))
        
        print(f"📏 Max text length: {max_length} characters")
        
        dataset = Dataset.from_list(formatted_data)
        
        # Smart split based on dataset size
        test_size = min(0.2, max(0.05, 50 / len(formatted_data)))
        train_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        
        print(f"📚 Train: {len(train_dataset['train'])}, Test: {len(train_dataset['test'])}")
        
        return train_dataset
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        raise

train_dataset = prepare_dataset(training_data)

# 9. Smart Model Selection and Loading
def select_model_and_config():
    # Choose model based on available resources
    if gpu_memory >= 15:  # High-end GPU
        model_name = "microsoft/DialoGPT-large"
        use_quantization = False
        batch_size = 2
    elif gpu_memory >= 10:  # Mid-range GPU
        model_name = "microsoft/DialoGPT-medium"
        use_quantization = True
        batch_size = 1
    elif gpu_memory >= 6:  # Lower-end GPU
        model_name = "microsoft/DialoGPT-small"
        use_quantization = True
        batch_size = 1
    else:  # CPU or very limited GPU
        model_name = "microsoft/DialoGPT-small"
        use_quantization = False
        batch_size = 1
    
    print(f"🤖 Selected model: {model_name}")
    print(f"⚡ Quantization: {use_quantization}")
    print(f"📦 Batch size: {batch_size}")
    
    return model_name, use_quantization, batch_size

model_name, use_quantization, batch_size = select_model_and_config()

# 10. Enhanced Model Loading with Fallbacks
def load_model_and_tokenizer(model_name, use_quantization):
    try:
        # Configure quantization
        bnb_config = None
        if use_quantization and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with fallbacks
        print("🤖 Loading model...")
        
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
        }
        
        if use_quantization and bnb_config:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32
        
        try:
            model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        except Exception as e:
            print(f"⚠️ Failed with device_map, trying without: {e}")
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        print("✅ Model loaded successfully")
        return model, tokenizer, bnb_config is not None
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("🔄 Trying fallback model...")
        
        # Fallback to smaller model
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                torch_dtype=torch.float32
            )
            
            print("✅ Fallback model loaded")
            return model, tokenizer, False
            
        except Exception as e2:
            print(f"❌ Fallback model also failed: {e2}")
            raise

model, tokenizer, is_quantized = load_model_and_tokenizer(model_name, use_quantization)

# 11. Smart LoRA Configuration
def setup_lora(model, is_quantized):
    try:
        if is_quantized:
            model = prepare_model_for_kbit_training(model)
        
        # Adaptive LoRA config based on model size
        model_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {model_params:,}")
        
        if model_params > 500_000_000:  # Large model
            lora_r = 32
            lora_alpha = 64
        elif model_params > 100_000_000:  # Medium model
            lora_r = 16
            lora_alpha = 32
        else:  # Small model
            lora_r = 8
            lora_alpha = 16
        
        # Find target modules dynamically
        target_modules = []
        for name, module in model.named_modules():
            if any(target in name for target in ["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "out_proj"]):
                module_type = name.split('.')[-1]
                if module_type not in target_modules:
                    target_modules.append(module_type)
        
        if not target_modules:
            target_modules = ["c_attn", "c_proj"]  # Default fallback
        
        print(f"🎯 LoRA targets: {target_modules}")
        print(f"📐 LoRA r={lora_r}, alpha={lora_alpha}")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model.train()
        model = get_peft_model(model, lora_config)
        
        print("✅ LoRA setup complete")
        return model
        
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        print("🔄 Continuing without LoRA...")
        return model

model = setup_lora(model, is_quantized)

# 12. Smart Tokenization with Length Analysis
def smart_tokenize(dataset, tokenizer):
    try:
        # Analyze text lengths first
        sample_texts = [dataset['train'][i]['text'] for i in range(min(100, len(dataset['train'])))]
        lengths = [len(tokenizer.encode(text)) for text in sample_texts]
        
        avg_length = np.mean(lengths)
        max_length = min(max(lengths), 1024)  # Cap at 1024
        
        print(f"📏 Token lengths - Avg: {avg_length:.0f}, Max: {max_length}")
        
        # Choose appropriate max_length
        if avg_length < 128:
            max_length = 256
        elif avg_length < 256:
            max_length = 512
        else:
            max_length = min(max_length, 1024)
        
        print(f"🎯 Using max_length: {max_length}")
        
        def tokenize_function(examples):
            texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
            tokenized = tokenizer(
                texts, 
                truncation=True, 
                padding="max_length", 
                max_length=max_length,
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        print("🔄 Tokenizing train dataset...")
        tokenized_train = dataset['train'].map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"],
            desc="Tokenizing train"
        )
        
        print("🔄 Tokenizing validation dataset...")
        tokenized_val = dataset['test'].map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"],
            desc="Tokenizing validation"
        )
        
        print("✅ Tokenization complete")
        return tokenized_train, tokenized_val
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        raise

tokenized_train, tokenized_val = smart_tokenize(train_dataset, tokenizer)

# 13. Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 14. Adaptive Training Configuration
def create_training_args(batch_size, is_quantized, dataset_size):
    # Adaptive learning rate based on dataset size
    if dataset_size < 100:
        learning_rate = 5e-4
        epochs = 5
    elif dataset_size < 1000:
        learning_rate = 2e-4
        epochs = 3
    else:
        learning_rate = 1e-4
        epochs = 2
    
    # Adaptive gradient accumulation
    gradient_acc_steps = max(1, 8 // batch_size)
    
    print(f"📚 Dataset size: {dataset_size}")
    print(f"📊 Learning rate: {learning_rate}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📈 Gradient accumulation: {gradient_acc_steps}")
    
    training_args = TrainingArguments(
        output_dir="./lity-ai-results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_acc_steps,
        warmup_steps=min(50, dataset_size // 10),
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=max(1, dataset_size // 20),
        eval_strategy="steps",
        eval_steps=max(10, dataset_size // 10),
        save_steps=max(25, dataset_size // 5),
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        fp16=is_quantized and torch.cuda.is_available(),
        gradient_checkpointing=gpu_memory < 12,  # Use for memory efficiency
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        optim="adamw_torch",
    )
    
    return training_args

training_args = create_training_args(batch_size, is_quantized, len(tokenized_train))

# 15. Enhanced Training with Error Recovery
def train_with_fallbacks(model, training_args, tokenized_train, tokenized_val, tokenizer, data_collator):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("🚀 Starting training...")
    
    # Training with progressive fallbacks
    attempts = [
        ("original", {}),
        ("reduce_batch", {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2}),
        ("no_fp16", {"fp16": False}),
        ("minimal", {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 1, "fp16": False})
    ]
    
    for attempt_name, modifications in attempts:
        try:
            print(f"🎯 Attempt: {attempt_name}")
            
            # Apply modifications
            for key, value in modifications.items():
                setattr(training_args, key, value)
            
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            trainer.train()
            print("✅ Training completed successfully!")
            return trainer
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ {attempt_name} failed - Out of memory")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"⚠️ {attempt_name} failed - {e}")
                continue
        except Exception as e:
            print(f"⚠️ {attempt_name} failed - {e}")
            continue
    
    raise RuntimeError("All training attempts failed")

trainer = train_with_fallbacks(model, training_args, tokenized_train, tokenized_val, tokenizer, data_collator)

# 16. Enhanced Model Saving
def save_model_safely(trainer, model, tokenizer):
    try:
        print("💾 Saving model...")
        
        # Save the trained adapter
        trainer.save_model("./lity-ai-lora-adapter")
        print("✅ LoRA adapter saved")
        
        # Try to merge and save full model
        try:
            if hasattr(model, 'merge_and_unload'):
                print("🔄 Merging LoRA weights...")
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained("./lity-ai-final-model")
                tokenizer.save_pretrained("./lity-ai-final-model")
                print("✅ Merged model saved")
                return True
            else:
                raise AttributeError("No merge_and_unload method")
                
        except Exception as e:
            print(f"⚠️ Merge failed ({e}), saving adapter only")
            model.save_pretrained("./lity-ai-final-model")
            tokenizer.save_pretrained("./lity-ai-final-model")
            print("✅ Adapter model saved")
            return False
            
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False

merge_success = save_model_safely(trainer, model, tokenizer)

# 17. Smart Model Packaging and Download
def package_and_download():
    try:
        print("📦 Packaging model...")
        
        # Create info file
        info = {
            "model_name": model_name,
            "is_merged": merge_success,
            "quantized": is_quantized,
            "training_samples": len(tokenized_train),
            "lora_config": {
                "r": getattr(model.peft_config.get('default', None), 'r', 'N/A'),
                "alpha": getattr(model.peft_config.get('default', None), 'lora_alpha', 'N/A')
            } if hasattr(model, 'peft_config') else "No LoRA"
        }
        
        with open("./lity-ai-final-model/model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        # Create zip
        !zip -r lity_ai_enhanced_model.zip lity-ai-final-model
        
        # Download
        from google.colab import files
        files.download("lity_ai_enhanced_model.zip")
        
        print("✅ Model packaged and downloaded successfully!")
        print("📋 Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Packaging failed: {e}")
        print("💡 You can manually download the 'lity-ai-final-model' folder")

package_and_download()

# 18. Final Summary and Usage Instructions
print("\n🎉 Training Complete!")
print("\n📖 Usage Instructions:")
print("1. Extract the downloaded zip file")
if merge_success:
    print("2. Load the model using: AutoModelForCausalLM.from_pretrained('./lity-ai-final-model')")
else:
    print("2. Load the model with LoRA adapter using PEFT library")
print("3. Use the tokenizer from the same folder")
print("\n💡 Tips:")
print("- Test the model with small inputs first")
print("- Adjust generation parameters for better outputs")
print("- Consider fine-tuning further with more data")

# Memory cleanup
torch.cuda.empty_cache()
gc.collect()
print("\n🧹 Memory cleaned up")