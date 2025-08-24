#!/usr/bin/env python3
"""
Script to clean existing data files by removing "<This message was edited>" strings
and similar patterns that are causing issues with model generation.
"""

import json
import re
import os
from pathlib import Path
import argparse

def clean_edited_messages(text):
    """Remove all variations of 'message was edited' strings."""
    if not text:
        return text
    
    # Remove various forms of "message was edited" strings
    patterns = [
        r'\s*<This message was edited>\s*',
        r'\s*<This message was edited\.>\s*',
        r'\s*<Message was edited>\s*',
        r'\s*<Message was edited\.>\s*',
        r'\s*<Edited>\s*',
        r'\s*<Edited\.>\s*',
        r'\s*This message was edited\s*',
        r'\s*This message was edited\.\s*',
        r'\s*Message was edited\s*',
        r'\s*Message was edited\.\s*',
        r'\s*Edited\s*',
        r'\s*Edited\.\s*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_jsonl_file(file_path, output_path=None):
    """Clean a single JSONL file."""
    if output_path is None:
        output_path = file_path
    
    cleaned_count = 0
    total_count = 0
    
    print(f"Cleaning {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                total_count += 1
                
                # Clean different possible text fields
                if isinstance(data, dict):
                    # Clean body_text field
                    if 'body_text' in data and data['body_text']:
                        original = data['body_text']
                        data['body_text'] = clean_edited_messages(data['body_text'])
                        if data['body_text'] != original:
                            cleaned_count += 1
                    
                    # Clean target_text field (for pairs)
                    if 'target_text' in data and data['target_text']:
                        original = data['target_text']
                        data['target_text'] = clean_edited_messages(data['target_text'])
                        if data['target_text'] != original:
                            cleaned_count += 1
                    
                    # Clean text field (for mono)
                    if 'text' in data and data['text']:
                        original = data['text']
                        data['text'] = clean_edited_messages(data['text'])
                        if data['text'] != original:
                            cleaned_count += 1
                    
                    # Clean context text (for pairs)
                    if 'context' in data and isinstance(data['context'], list):
                        for ctx_item in data['context']:
                            if isinstance(ctx_item, dict) and 'text' in ctx_item:
                                original = ctx_item['text']
                                ctx_item['text'] = clean_edited_messages(ctx_item['text'])
                                if ctx_item['text'] != original:
                                    cleaned_count += 1
                
                # Write cleaned data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at line {line_num}: {e}")
                # Write the original line if we can't parse it
                outfile.write(line + '\n')
    
    print(f"  Processed {total_count} records, cleaned {cleaned_count} text fields")
    return cleaned_count, total_count

def clean_directory(directory_path, recursive=True):
    """Clean all JSONL files in a directory."""
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory {directory_path} does not exist")
        return
    
    total_cleaned = 0
    total_files = 0
    
    # Find all JSONL files
    if recursive:
        jsonl_files = list(directory.rglob("*.jsonl"))
    else:
        jsonl_files = list(directory.glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files in {directory_path}")
    
    for file_path in jsonl_files:
        try:
            cleaned, total = clean_jsonl_file(file_path)
            total_cleaned += cleaned
            total_files += total
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nTotal: Cleaned {total_cleaned} text fields across {len(jsonl_files)} files")

def main():
    parser = argparse.ArgumentParser(description="Clean existing data files by removing 'message was edited' strings")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory to clean")
    parser.add_argument("--output", "-o", help="Output file (only for single file input)")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, help="Recursively process directories (default: True)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        if args.output:
            clean_jsonl_file(input_path, args.output)
        else:
            clean_jsonl_file(input_path)
    elif input_path.is_dir():
        # Directory
        clean_directory(input_path, args.recursive)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
