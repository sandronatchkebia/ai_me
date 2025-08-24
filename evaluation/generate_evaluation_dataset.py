#!/usr/bin/env python3
"""
AI Me - Evaluation Dataset Generator

This script generates the evaluation dataset by:
1. Loading selected emails
2. Generating GPT-4 responses
3. Generating LoRA responses
4. Saving all three versions for analysis

Run this in Colab where you have GPU access for the LoRA model.
"""

import os
import json
import openai
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import tqdm
import time

# Configuration
OPENAI_API_KEY = "your_openai_api_key_here"  # Set your OpenAI API key
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_MODEL_PATH = "./fine_tuning/out/ai_me_lora_llama3p1_8b"  # Adjust path as needed

# OpenAI client setup
openai.api_key = OPENAI_API_KEY

# Generation parameters
MAX_TOKENS = 500
TEMPERATURE = 0.7
NUM_SAMPLES = 50  # Number of email samples to evaluate

# Rate limiting
OPENAI_DELAY = 1.0  # Seconds between OpenAI API calls

print("🚀 Starting Evaluation Dataset Generation...")
print("="*60)

def load_evaluation_emails(file_path: str = "evaluation_data/selected_evaluation_emails.json") -> List[Dict]:
    """Load the selected evaluation emails from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            emails = json.load(f)
        print(f"✅ Loaded {len(emails)} evaluation emails from {file_path}")
        return emails
    except FileNotFoundError:
        print(f"❌ Evaluation emails file not found: {file_path}")
        print("Please run select_evaluation_emails.py first to generate the email selection.")
        return []
    except Exception as e:
        print(f"❌ Error loading evaluation emails: {e}")
        return []

def load_models():
    """Load the base model and fine-tuned LoRA model."""
    print("🔧 Loading models...")
    
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("  Loading fine-tuned LoRA model...")
    lora_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    lora_model = PeftModel.from_pretrained(lora_model, LORA_MODEL_PATH)
    
    print("✅ Models loaded successfully!")
    return tokenizer, base_model, lora_model

def generate_gpt4_response(partner_email: str, attempt: int = 1) -> str:
    """Generate a response using GPT-4 API with retry logic."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are writing an email response. Write a natural, conversational response that would be appropriate for a colleague or friend. Keep it casual and friendly. Respond as if you were the person receiving this email."},
                {"role": "user", "content": f"Write an email response to this message: {partner_email}"}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if attempt <= 3:
            print(f"  ⚠️  GPT-4 attempt {attempt} failed: {e}")
            time.sleep(OPENAI_DELAY * 2)  # Longer delay on retry
            return generate_gpt4_response(partner_email, attempt + 1)
        else:
            print(f"  ❌ GPT-4 failed after 3 attempts: {e}")
            return f"ERROR: {str(e)}"

def generate_lora_response(partner_email: str, lora_model, tokenizer, attempt: int = 1) -> str:
    """Generate a response using your fine-tuned LoRA model with retry logic."""
    try:
        prompt = f"<|user|>\nWrite an email response to this message: {partner_email}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
        
        with torch.no_grad():
            outputs = lora_model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Clean up response
        if response and len(response) > 10:
            return response
        else:
            raise ValueError("Generated response too short or empty")
            
    except Exception as e:
        if attempt <= 3:
            print(f"  ⚠️  LoRA attempt {attempt} failed: {e}")
            time.sleep(1)  # Short delay on retry
            return generate_lora_response(partner_email, lora_model, tokenizer, attempt + 1)
        else:
            print(f"  ❌ LoRA failed after 3 attempts: {e}")
            return f"ERROR: {str(e)}"

def rewrite_gpt4_with_lora(gpt4_response: str, lora_model, tokenizer, attempt: int = 1) -> str:
    """Rewrite a GPT-4 response using the LoRA model to apply your personal style."""
    try:
        # Create a more explicit prompt that clearly specifies what to rewrite
        prompt = f"<|user|>\nRewrite ONLY this email response in my personal writing style. Do not add any new content, do not include the original message, do not change the meaning. Just rewrite this text to sound like I wrote it:\n\n{gpt4_response}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
        with torch.no_grad():
            outputs = lora_model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        if response and len(response) > 10:
            return response
        else:
            raise ValueError("Generated rewrite too short or empty")
            
    except Exception as e:
        if attempt <= 3:
            print(f"  ⚠️  LoRA rewrite attempt {attempt} failed: {e}")
            time.sleep(1)  # Short delay on retry
            return rewrite_gpt4_with_lora(gpt4_response, lora_model, tokenizer, attempt + 1)
        else:
            print(f"  ❌ LoRA rewrite failed after 3 attempts: {e}")
            return f"ERROR: {str(e)}"

def generate_dataset(evaluation_emails: List[Dict], tokenizer, lora_model) -> List[Dict]:
    """Generate the complete evaluation dataset."""
    dataset = []
    
    print(f"\n🔄 Generating responses for {len(evaluation_emails)} emails...")
    print("="*60)
    
    for i, email_sample in enumerate(tqdm.tqdm(evaluation_emails, desc="Generating responses")):
        print(f"\n📧 Processing email {i+1}/{len(evaluation_emails)}")
        print(f"ID: {email_sample.get('original_id', f'sample_{i+1}')}")
        
        partner_email = email_sample["partner_email"]
        original_response = email_sample["your_response"]
        
        # Generate GPT-4 response
        print("  🤖 Generating GPT-4 response...")
        gpt4_response = generate_gpt4_response(partner_email)
        print(f"    GPT-4: {gpt4_response[:100]}...")
        
        # Rate limiting for OpenAI
        time.sleep(OPENAI_DELAY)
        
        # Generate direct LoRA response
        print("  🎯 Generating direct LoRA response...")
        lora_response = generate_lora_response(partner_email, lora_model, tokenizer)
        print(f"    LoRA Direct: {lora_response[:100]}...")
        
        # Generate LoRA rewrite of GPT-4 response (hybrid approach)
        print("  🎯 Rewriting GPT-4 response with LoRA style...")
        lora_rewrite = rewrite_gpt4_with_lora(gpt4_response, lora_model, tokenizer)
        print(f"    LoRA Rewrite: {lora_rewrite[:100]}...")
        
        # Create dataset entry
        dataset_entry = {
            "sample_id": i + 1,
            "original_id": email_sample.get("original_id", f"sample_{i+1}"),
            "partner_email": partner_email,
            "original_response": original_response,
            "gpt4_response": gpt4_response,
            "lora_response": lora_response,  # Direct LoRA response
            "lora_rewrite": lora_rewrite,   # Hybrid: GPT-4 content + LoRA style
            "meta": email_sample.get("meta", {}),
            "generation_timestamp": time.time()
        }
        
        dataset.append(dataset_entry)
        print(f"  ✅ Completed email {i+1}")
        
        # Progress update every 10 emails
        if (i + 1) % 10 == 0:
            print(f"\n📊 Progress: {i+1}/{len(evaluation_emails)} emails completed")
    
    return dataset

def save_dataset(dataset: List[Dict], output_file: str = "evaluation_data/evaluation_dataset.json"):
    """Save the generated dataset to JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Dataset saved to {output_file}")
    print(f"📊 Total samples: {len(dataset)}")

def create_summary_report(dataset: List[Dict]):
    """Create a summary report of the generated dataset."""
    print(f"\n📋 DATASET SUMMARY REPORT")
    print("="*60)
    
    # Count successful generations
    successful_gpt4 = sum(1 for item in dataset if not item['gpt4_response'].startswith('ERROR'))
    successful_lora_direct = sum(1 for item in dataset if not item['lora_response'].startswith('ERROR'))
    successful_lora_rewrites = sum(1 for item in dataset if not item['lora_rewrite'].startswith('ERROR'))
    
    print(f"Total samples: {len(dataset)}")
    print(f"Successful GPT-4 generations: {successful_gpt4}/{len(dataset)} ({successful_gpt4/len(dataset)*100:.1f}%)")
    print(f"Successful LoRA direct responses: {successful_lora_direct}/{len(dataset)} ({successful_lora_direct/len(dataset)*100:.1f}%)")
    print(f"Successful LoRA rewrites: {successful_lora_rewrites}/{len(dataset)} ({successful_lora_rewrites/len(dataset)*100:.1f}%)")
    
    # Response length statistics
    original_lengths = [len(item['original_response'].split()) for item in dataset]
    gpt4_lengths = [len(item['gpt4_response'].split()) for item in dataset if not item['gpt4_response'].startswith('ERROR')]
    lora_direct_lengths = [len(item['lora_response'].split()) for item in dataset if not item['lora_response'].startswith('ERROR')]
    lora_rewrite_lengths = [len(item['lora_rewrite'].split()) for item in dataset if not item['lora_rewrite'].startswith('ERROR')]
    
    print(f"\n📏 Response Length Statistics:")
    print(f"  Original responses: {np.mean(original_lengths):.1f} ± {np.std(original_lengths):.1f} words")
    if gpt4_lengths:
        print(f"  GPT-4 responses: {np.mean(gpt4_lengths):.1f} ± {np.std(gpt4_lengths):.1f} words")
    if lora_direct_lengths:
        print(f"  LoRA direct responses: {np.mean(lora_direct_lengths):.1f} ± {np.std(lora_direct_lengths):.1f} words")
    if lora_rewrite_lengths:
        print(f"  LoRA rewrites: {np.mean(lora_rewrite_lengths):.1f} ± {np.std(lora_rewrite_lengths):.1f} words")
    
    # Sample preview
    print(f"\n📧 SAMPLE GENERATIONS:")
    print("-"*60)
    
    for i, item in enumerate(dataset[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Partner: {item['partner_email'][:80]}...")
        print(f"  Original: {item['original_response']}")
        print(f"  GPT-4: {item['gpt4_response'][:80]}...")
        print(f"  LoRA Direct: {item['lora_response'][:80]}...")
        print(f"  LoRA Rewrite: {item['lora_rewrite'][:80]}...")

def main():
    """Main function to generate the evaluation dataset."""
    print("🎯 AI Me - Evaluation Dataset Generator")
    print("="*60)
    
    # Load evaluation emails
    evaluation_emails = load_evaluation_emails()
    if not evaluation_emails:
        print("❌ No evaluation emails loaded. Exiting.")
        return
    
    # Limit to specified number of samples
    evaluation_emails = evaluation_emails[:NUM_SAMPLES]
    print(f"📧 Using {len(evaluation_emails)} emails for evaluation")
    
    # Load models
    tokenizer, base_model, lora_model = load_models()
    
    # Generate dataset
    dataset = generate_dataset(evaluation_emails, tokenizer, lora_model)
    
    # Save dataset
    save_dataset(dataset)
    
    # Create summary report
    create_summary_report(dataset)
    
    print(f"\n🎉 DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"📁 Next steps:")
    print(f"1. Download evaluation_dataset.json from Colab")
    print(f"2. Run the analysis script: python3 analyze_evaluation_dataset.py")
    print(f"3. Review the comprehensive style analysis results")
    
    # Clean up memory
    del base_model, lora_model
    torch.cuda.empty_cache()
    print("\n🧹 Memory cleaned up!")

if __name__ == "__main__":
    main()
