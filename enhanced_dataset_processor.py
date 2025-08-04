"""
Enhanced Dataset Processor for Lity AI Model Training
This script processes the enhanced training dataset for use in Google Colab model training.
"""

import json
from collections import Counter
import re

def load_and_analyze_dataset(file_path):
    """Load and analyze the enhanced training dataset"""
    
    print("Loading Enhanced Training Dataset...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"✅ Loaded {len(data)} training examples")
    
    # Analyze the data
    input_lengths = [len(item['input'].split()) for item in data]
    output_lengths = [len(item['output'].split()) for item in data]
    
    print(f"\n📊 Dataset Analysis:")
    print(f"Average input length: {sum(input_lengths) / len(input_lengths):.1f} words")
    print(f"Average output length: {sum(output_lengths) / len(output_lengths):.1f} words")
    print(f"Max input length: {max(input_lengths)} words")
    print(f"Max output length: {max(output_lengths)} words")
    
    # Extract topics and themes
    topics = []
    for item in data:
        # Extract key topics from input questions
        input_text = item['input'].lower()
        if 'smk' in input_text or 'moneykind' in input_text:
            topics.append('SMK Moneykind Platform')
        elif 'budget' in input_text:
            topics.append('Budgeting')
        elif 'save' in input_text or 'saving' in input_text:
            topics.append('Saving Strategies')
        elif 'invest' in input_text:
            topics.append('Investment')
        elif 'mobile money' in input_text or 'digital' in input_text:
            topics.append('Digital Finance')
        elif 'entrepreneur' in input_text or 'business' in input_text:
            topics.append('Entrepreneurship')
        elif 'credit' in input_text or 'debt' in input_text:
            topics.append('Credit & Debt')
        elif 'scam' in input_text or 'fraud' in input_text:
            topics.append('Financial Security')
        elif 'insurance' in input_text:
            topics.append('Insurance')
        elif 'economic' in input_text or 'africa' in input_text:
            topics.append('Economic Development')
        elif 'government' in input_text or 'policy' in input_text:
            topics.append('Policy & Governance')
        elif 'technology' in input_text or 'fintech' in input_text:
            topics.append('Financial Technology')
        else:
            topics.append('General Financial Literacy')
    
    topic_counts = Counter(topics)
    print(f"\n🎯 Topic Distribution:")
    for topic, count in topic_counts.most_common():
        print(f"  {topic}: {count} examples")
    
    return data, topic_counts

def create_colab_training_format(data, output_file):
    """Convert data to format suitable for Colab training"""
    
    print(f"\n🔄 Converting to Colab training format...")
    
    # Create training format with instruction, input, and output
    training_data = []
    
    for item in data:
        training_example = {
            "instruction": "You are Lity AI, a friendly financial assistant from SMK Moneykind. Provide helpful, accurate, and engaging financial guidance suitable for young people in Africa. Focus on practical advice that considers local financial realities like mobile money, family obligations, and entrepreneurship opportunities.",
            "input": item["input"],
            "output": item["output"]
        }
        training_data.append(training_example)
    
    # Save as JSON for easy loading in Colab
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(training_data)} training examples to {output_file}")
    
    # Also create a simplified JSONL format
    jsonl_file = output_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Also saved as JSONL format: {jsonl_file}")

def create_evaluation_prompts(output_file):
    """Create evaluation prompts to test the trained model"""
    
    evaluation_prompts = [
        "What is SMK Moneykind and how can it help young people?",
        "How should a university student in Uganda create a monthly budget?",
        "What are the best ways for teenagers to start saving money?",
        "Explain compound interest in simple terms for a young person",
        "How can young people use mobile money safely and effectively?",
        "What should I know before starting my first small business?",
        "How do I protect myself from financial scams online?",
        "What's the difference between saving and investing?",
        "How can families teach children about money management?",
        "What role does financial literacy play in Africa's development?",
        "How can technology help young people access financial services?",
        "What are the most important financial skills for young people?",
        "How do cultural factors affect financial decisions in Africa?",
        "What investment options work best for young people with little money?",
        "How can digital financial services help rural communities?"
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_prompts, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(evaluation_prompts)} evaluation prompts: {output_file}")

def generate_training_insights():
    """Generate insights for model training optimization"""
    
    insights = {
        "training_recommendations": [
            "Use a learning rate between 1e-4 and 3e-4 for optimal convergence",
            "Train for 3-5 epochs to avoid overfitting with this dataset size",
            "Use gradient accumulation if batch size is limited by GPU memory",
            "Implement early stopping based on validation loss",
            "Use warmup steps (10% of total training steps) for stable training"
        ],
        "model_architecture_tips": [
            "Consider using models like Llama2-7B or Mistral-7B as base models",
            "Implement LoRA (Low-Rank Adaptation) for efficient fine-tuning",
            "Use 4-bit quantization to reduce memory requirements",
            "Set max sequence length to 2048 tokens to handle longer responses",
            "Enable gradient checkpointing to reduce memory usage"
        ],
        "evaluation_metrics": [
            "BLEU score for response quality",
            "ROUGE scores for content relevance",
            "Human evaluation for financial accuracy",
            "Cultural appropriateness assessment",
            "Response coherence and engagement metrics"
        ],
        "deployment_considerations": [
            "Test responses for financial accuracy before deployment",
            "Ensure responses maintain cultural sensitivity",
            "Implement safety filters for inappropriate content",
            "Monitor for hallucinations in financial advice",
            "Regular updates with new financial information"
        ]
    }
    
    print("\n🎯 Training Insights and Recommendations:")
    for category, items in insights.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  • {item}")
    
    return insights

def main():
    """Main function to process the enhanced dataset"""
    
    print("🚀 Enhanced Dataset Processor for Lity AI")
    print("=" * 50)
    
    # Load and analyze the enhanced dataset
    input_file = "Enhanced_Training_Dataset.jsonl"
    data, topic_counts = load_and_analyze_dataset(input_file)
    
    # Create Colab training format
    training_output = "lity_ai_training_data.json"
    create_colab_training_format(data, training_output)
    
    # Create evaluation prompts
    eval_output = "evaluation_prompts.json"
    create_evaluation_prompts(eval_output)
    
    # Generate training insights
    insights = generate_training_insights()
    
    print(f"\n✅ Dataset processing complete!")
    print(f"\nFiles created:")
    print(f"  • {training_output} - Training data for Colab")
    print(f"  • lity_ai_training_data.jsonl - JSONL format")
    print(f"  • {eval_output} - Evaluation prompts")
    
    print(f"\n🎓 Ready for Colab Training!")
    print(f"Your enhanced dataset contains {len(data)} high-quality examples")
    print(f"covering {len(topic_counts)} different financial literacy topics.")

if __name__ == "__main__":
    main()
