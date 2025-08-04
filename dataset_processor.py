#!/usr/bin/env python3
"""
SMK Moneykind Dataset Processor
Processes the JSONL dataset for enhanced FAQ responses and training data preparation
"""

import json
import re
from typing import Dict, List, Tuple

def load_dataset(file_path: str) -> List[Dict]:
    """Load the JSONL dataset"""
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        print(f"✅ Loaded {len(dataset)} entries from dataset")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from input text for better matching"""
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Common stop words to exclude
    stop_words = {'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'can', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'my', 'your', 'i', 'you', 'we', 'they', 'he', 'she', 'it'}
    
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words

def create_enhanced_faq(dataset: List[Dict]) -> Dict[str, str]:
    """Create an enhanced FAQ dictionary with multiple key variations"""
    faq = {}
    
    for entry in dataset:
        input_text = entry['input'].lower().strip()
        output_text = entry['output']
        
        # Add the exact input as a key
        faq[input_text] = output_text
        
        # Extract keywords for alternative matching
        keywords = extract_keywords(input_text)
        
        # Create shorter key variations
        if len(keywords) > 0:
            # Use the first few important keywords
            short_key = ' '.join(keywords[:3])
            if short_key not in faq:
                faq[short_key] = output_text
        
        # Special handling for common question patterns
        if input_text.startswith('what is'):
            topic = input_text.replace('what is', '').strip().replace('?', '')
            if topic:
                faq[topic] = output_text
        
        if input_text.startswith('how do i') or input_text.startswith('how can i'):
            action = input_text.replace('how do i', '').replace('how can i', '').strip().replace('?', '')
            if action:
                faq[action] = output_text
    
    return faq

def categorize_responses(dataset: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize responses by topic for better organization"""
    categories = {
        'about_smk': [],
        'apps_litywise': [],
        'apps_nfunayo': [],
        'financial_basics': [],
        'trainings': [],
        'technical_support': [],
        'business_info': [],
        'greetings': [],
        'other': []
    }
    
    for entry in dataset:
        input_lower = entry['input'].lower()
        
        if any(word in input_lower for word in ['smk', 'moneykind', 'founder', 'company', 'started']):
            categories['about_smk'].append(entry)
        elif any(word in input_lower for word in ['litywise', 'lity', 'gamified', 'saver', 'investor', 'boss']):
            categories['apps_litywise'].append(entry)
        elif any(word in input_lower for word in ['nfunayo', 'expense', 'tracker', 'budget']):
            categories['apps_nfunayo'].append(entry)
        elif any(word in input_lower for word in ['save', 'invest', 'money', 'budget', 'financial', 'compound', 'emergency']):
            categories['financial_basics'].append(entry)
        elif any(word in input_lower for word in ['training', 'workshop', 'school', 'physical']):
            categories['trainings'].append(entry)
        elif any(word in input_lower for word in ['support', 'help', 'crash', 'password', 'data', 'download']):
            categories['technical_support'].append(entry)
        elif any(word in input_lower for word in ['partnership', 'hire', 'invest', 'career', 'office']):
            categories['business_info'].append(entry)
        elif any(word in input_lower for word in ['hi', 'hello', 'joke', 'tell me']):
            categories['greetings'].append(entry)
        else:
            categories['other'].append(entry)
    
    return categories

def generate_javascript_faq(dataset: List[Dict], output_file: str):
    """Generate a JavaScript file with enhanced FAQ responses"""
    faq = create_enhanced_faq(dataset)
    
    js_content = """// Enhanced FAQ responses generated from SMK Moneykind dataset
// This file contains comprehensive responses for Lity AI

export const enhancedFAQ = {\n"""
    
    for key, value in sorted(faq.items()):
        # Escape quotes and newlines for JavaScript
        escaped_key = key.replace("'", "\\'").replace('"', '\\"')
        escaped_value = value.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        
        js_content += f"  '{escaped_key}': '{escaped_value}',\n"
    
    js_content += "};\n\n"
    
    # Add keyword matching function
    js_content += """
// Enhanced keyword matching for better FAQ responses
export const findBestFAQMatch = (userInput) => {
  const input = userInput.toLowerCase().trim();
  
  // Direct match
  if (enhancedFAQ[input]) {
    return enhancedFAQ[input];
  }
  
  // Keyword matching
  const inputWords = input.split(' ').filter(word => word.length > 2);
  let bestMatch = null;
  let maxMatches = 0;
  
  for (const [key, response] of Object.entries(enhancedFAQ)) {
    const keyWords = key.split(' ');
    const matches = inputWords.filter(word => keyWords.includes(word)).length;
    
    if (matches > maxMatches && matches > 0) {
      maxMatches = matches;
      bestMatch = response;
    }
  }
  
  return bestMatch;
};
"""
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(js_content)
        print(f"✅ Generated JavaScript FAQ file: {output_file}")
    except Exception as e:
        print(f"❌ Error generating JavaScript file: {e}")

def analyze_dataset(dataset: List[Dict]):
    """Analyze the dataset and provide insights"""
    print("\n📊 Dataset Analysis")
    print("=" * 50)
    
    categories = categorize_responses(dataset)
    
    print(f"Total entries: {len(dataset)}")
    print("\nCategory breakdown:")
    for category, entries in categories.items():
        print(f"  {category.replace('_', ' ').title()}: {len(entries)} entries")
    
    # Find most common keywords
    all_keywords = []
    for entry in dataset:
        all_keywords.extend(extract_keywords(entry['input']))
    
    keyword_counts = {}
    for keyword in all_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    print(f"\nTop 10 keywords:")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {keyword}: {count}")

def main():
    """Main function to process the dataset"""
    dataset_file = "SMK_Moneykind_Custom_Dataset.jsonl"
    js_output_file = "src/enhancedFAQ.js"
    
    print("🚀 SMK Moneykind Dataset Processor")
    print("=" * 50)
    
    # Load dataset
    dataset = load_dataset(dataset_file)
    
    if not dataset:
        print("❌ No dataset loaded. Exiting.")
        return
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    # Generate JavaScript FAQ
    generate_javascript_faq(dataset, js_output_file)
    
    print("\n✅ Dataset processing complete!")
    print(f"📁 Generated files:")
    print(f"  - {js_output_file}")

if __name__ == "__main__":
    main()
