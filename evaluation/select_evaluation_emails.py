#!/usr/bin/env python3
"""
Select appropriate emails from preprocessed data for style evaluation.

This script analyzes the pairs_train.jsonl file to find emails that are good
for style evaluation - ones that focus on style rather than specialized knowledge.
Only includes emails from 2020 onwards.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

def analyze_email_content(email_text: str) -> Dict:
    """Analyze email content to determine if it's good for style evaluation."""
    
    # Convert to lowercase for analysis
    text_lower = email_text.lower()
    
    # Check for specialized knowledge indicators
    technical_terms = [
        'algorithm', 'api', 'database', 'framework', 'protocol', 'architecture',
        'deployment', 'infrastructure', 'microservices', 'kubernetes', 'docker',
        'machine learning', 'ai', 'neural network', 'tensorflow', 'pytorch',
        'javascript', 'python', 'react', 'node.js', 'sql', 'nosql', 'regression',
        'excel', 'model', 'case', 'assignment', 'project', 'client', 'consulting'
    ]
    
    business_terms = [
        'quarterly', 'revenue', 'kpi', 'roi', 'stakeholder', 'deliverable',
        'milestone', 'deadline', 'budget', 'forecast', 'strategy', 'tactical',
        'membership fee', 'paypal', 'zelle', 'venmo', 'collaborative plan'
    ]
    
    # Count specialized terms
    technical_count = sum(1 for term in technical_terms if term in text_lower)
    business_count = sum(1 for term in business_terms if term in text_lower)
    
    # Check for style-focused indicators
    style_indicators = [
        'how are you', 'how\'s it going', 'catch up', 'lunch', 'coffee',
        'weekend', 'family', 'travel', 'weather', 'thanks', 'appreciate',
        'sorry', 'congratulations', 'good luck', 'take care', 'hope you are doing great',
        'thrilled to have you', 'excited about', 'warm regards', 'best regards'
    ]
    
    style_count = sum(1 for indicator in style_indicators if indicator in text_lower)
    
    # Check for personal pronouns and casual language
    personal_pronouns = ['i', 'you', 'we', 'us', 'our', 'my', 'your']
    personal_count = sum(1 for pronoun in personal_pronouns if pronoun in text_lower)
    
    # Check for contractions and informal language
    contractions = ['don\'t', 'can\'t', 'won\'t', 'i\'m', 'you\'re', 'we\'re', 'it\'s', 'that\'s']
    contraction_count = sum(1 for contraction in contractions if contraction in text_lower)
    
    # Check for casual expressions
    casual_expressions = ['hey', 'hi', 'hello', 'thanks', 'cool', 'awesome', 'great', 'nice', 'good', 'bad', 'stuff', 'thing', 'haha', 'lol']
    casual_count = sum(1 for expr in casual_expressions if expr in text_lower)
    
    # Calculate scores
    specialized_score = technical_count + business_count
    style_score = style_count + (personal_count / 10) + (contraction_count * 2) + (casual_count * 1.5)
    
    # Determine if email is good for style evaluation
    is_good_for_style = (
        specialized_score < 4 and      # Not too technical/business-heavy
        style_score > 1.5 and         # Has style elements
        len(email_text.split()) > 5 and   # Not too short
        len(email_text.split()) < 150     # Not too long
    )
    
    return {
        'specialized_score': specialized_score,
        'style_score': style_score,
        'is_good_for_style': is_good_for_style,
        'technical_terms': technical_count,
        'business_terms': business_count,
        'style_indicators': style_count,
        'personal_pronouns': personal_count,
        'contractions': contraction_count,
        'casual_expressions': casual_count,
        'word_count': len(email_text.split())
    }

def is_email_from_2020_onwards(meta: Dict) -> bool:
    """Check if the email is from 2020 or later."""
    try:
        date_str = meta.get('date', '')
        if not date_str:
            return False
        
        # Parse the date string
        if 'T' in date_str:
            # ISO format like "2023-09-08T00:21:51+00:00"
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            # Try other common formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return False
        
        # Check if date is 2020 or later
        return date_obj.year >= 2020
        
    except Exception as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return False

def load_pairs_data(data_file: str) -> List[Dict]:
    """Load email pairs from the preprocessed data, filtering for 2020+ emails."""
    emails = []
    total_emails = 0
    filtered_emails = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_emails += 1
                
                # Extract the target_text (your response) and context
                if 'target_text' in data and 'context' in data:
                    target_text = data['target_text']
                    context = data['context']
                    meta = data.get('meta', {})
                    
                    # Filter for emails from 2020 onwards
                    if not is_email_from_2020_onwards(meta):
                        continue
                    
                    filtered_emails += 1
                    
                    # Find the partner's message (the one you're responding to)
                    partner_message = None
                    for ctx in context:
                        if ctx.get('role') == 'partner':
                            partner_message = ctx.get('text', '')
                            break
                    
                    if partner_message and target_text:
                        emails.append({
                            'id': data.get('id', f'line_{line_num}'),
                            'partner_email': partner_message,
                            'your_response': target_text,
                            'meta': meta,
                            'line_number': line_num
                        })
                        
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"📊 Date filtering results:")
    print(f"  Total emails processed: {total_emails}")
    print(f"  Emails from 2020+: {filtered_emails}")
    print(f"  Filtered out: {total_emails - filtered_emails}")
    
    return emails

def select_evaluation_emails(emails: List[Dict], num_samples: int = 50) -> List[Dict]:
    """Select the best emails for style evaluation."""
    
    # Analyze all emails
    analyzed_emails = []
    for email in emails:
        analysis = analyze_email_content(email['your_response'])
        analyzed_emails.append({
            **email,
            'analysis': analysis
        })
    
    # Sort by style score (descending) and then by specialized score (ascending)
    analyzed_emails.sort(
        key=lambda x: (x['analysis']['style_score'], -x['analysis']['specialized_score']),
        reverse=True
    )
    
    # Filter for good style evaluation candidates
    good_candidates = [
        email for email in analyzed_emails 
        if email['analysis']['is_good_for_style']
    ]
    
    # Return top candidates
    return good_candidates[:num_samples]

def save_evaluation_emails(selected_emails: List[Dict], output_file: str):
    """Save selected emails in a format suitable for the evaluation script."""
    
    # Create the format expected by the evaluation script
    evaluation_format = []
    
    for email in selected_emails:
        evaluation_format.append({
            "partner_email": email['partner_email'],
            "your_response": email['your_response'],
            "original_id": email['id'],
            "meta": email['meta']
        })
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_format, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(evaluation_format)} emails to {output_file}")

def main():
    """Main function to select evaluation emails."""
    print("🔍 Loading email pairs from preprocessed data (2020 onwards only)...")
    
    # Load email data
    data_file = "../data/preprocessed/english/pairs_train.jsonl"
    emails = load_pairs_data(data_file)
    
    if not emails:
        print("❌ No email data found. Please check the data file path.")
        return
    
    print(f"✅ Loaded {len(emails)} email pairs from 2020 onwards")
    
    # Select evaluation emails
    print(f"\n🎯 Selecting best emails for style evaluation...")
    selected_emails = select_evaluation_emails(emails, num_samples=50)
    
    if not selected_emails:
        print("❌ No suitable emails found for style evaluation.")
        print("Consider adjusting the selection criteria.")
        return
    
    print(f"✅ Found {len(selected_emails)} good candidates for evaluation")
    
    # Display sample of selected emails
    print(f"\n📧 SAMPLE OF SELECTED EMAILS:")
    print("="*80)
    
    for i, email in enumerate(selected_emails[:5], 1):
        print(f"\nEmail {i}:")
        print(f"ID: {email['id']}")
        print(f"Date: {email['meta'].get('date', 'Unknown')}")
        print(f"Style Score: {email['analysis']['style_score']:.2f}")
        print(f"Specialized Score: {email['analysis']['specialized_score']:.2f}")
        print(f"Word Count: {email['analysis']['word_count']}")
        print(f"Partner: {email['partner_email'][:100]}...")
        print(f"Your Response: {email['your_response']}")
        print("-"*80)
    
    # Save selected emails
    output_file = "evaluation_data/selected_evaluation_emails.json"
    save_evaluation_emails(selected_emails, output_file)
    
    # Show statistics
    print(f"\n📊 SELECTION STATISTICS:")
    print(f"Total emails from 2020+: {len(emails)}")
    print(f"Good candidates found: {len(selected_emails)}")
    print(f"Selection rate: {len(selected_emails)/len(emails)*100:.1f}%")
    
    if selected_emails:
        avg_style_score = sum(e['analysis']['style_score'] for e in selected_emails) / len(selected_emails)
        avg_specialized_score = sum(e['analysis']['specialized_score'] for e in selected_emails) / len(selected_emails)
        print(f"Average style score: {avg_style_score:.2f}")
        print(f"Average specialized score: {avg_specialized_score:.2f}")
    
    print(f"\n📝 NEXT STEPS:")
    print(f"1. Review the sample emails above to ensure they meet your criteria")
    print(f"2. Check the full selection in {output_file}")
    print(f"3. Use these emails in your style evaluation script")

if __name__ == "__main__":
    main()
