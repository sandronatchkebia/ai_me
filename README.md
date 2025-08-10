# AI_ME – Personal Conversational Fine-Tuning Pipeline

AI_ME is an end-to-end pipeline for fine-tuning a Large Language Model (LLM) to emulate a user’s communication style.  
It parses personal conversation data from multiple platforms, preprocesses it, and prepares it for model fine-tuning.

## 🚀 Pipeline Overview

1. **Data Export**  
   Export your data from:
   - Gmail – Google Takeout (.mbox format)
   - Instagram – Meta data export (JSON)
   - WhatsApp – Chat export (JSON/ZIP)
   - Facebook Messenger – Meta data export (JSON)

2. **Parsing (Implemented)**  
   Each platform parser:
   - Reads the raw export format
   - Extracts key metadata
   - Cleans and normalizes message text
   - Outputs a `.jsonl` file (one JSON object per message) in `data/processed`

3. **Pre-processing (Planned)**  
   - Deduplicate messages  
   - Filter by language  
   - Remove low-value/system messages  
   - Split into train/validation/test sets  

4. **Fine-tuning (Planned)**  
   - Format dataset into instruction–response pairs  
   - Train using LoRA or full fine-tuning on an open-source LLM  
   - Evaluate output quality  

---

## 📜 Running the Parsers

**Example – Parse Gmail**
```bash
python parsers/gmail_parser.py \
  --in data/raw/.gmail.mbox.icloud \
  --out data/processed/gmail_parsed.jsonl \
  --me "myemail@gmail.com"
Example – Parse Instagram

bash
Copy
Edit
python parsers/instagram_parser.py \
  --in data/raw/instagram \
  --out data/processed/instagram_parsed.jsonl \
  --me "myusername"
📄 Output JSONL Schema
Each message record follows:

json
Copy
Edit
{
  "platform": "gmail",
  "source_file": "path/to/file",
  "conversation_id": "unique-id",
  "participants": ["me", "other_person"],
  "account": "me",
  "message_id": "uuid",
  "reply_to_message_id": null,
  "date": "2025-07-09T18:34:00Z",
  "from": "me",
  "to": ["other_person"],
  "direction": "outbound",
  "turn_role": "you",
  "content_type": "text",
  "attachments": {"count": 0, "types": []},
  "body_raw": "Original raw text",
  "body_text": "Cleaned text",
  "lang": "en",
  "thread_index": 0,
  "thread_len": 42
}
🧹 Pre-processing (Planned)
Example workflow (to be implemented):

bash
Copy
Edit
python pre_processing/clean_data.py \
  --in data/processed/gmail_parsed.jsonl \
  --out data/processed/gmail_clean.jsonl \
  --languages en ka \
  --min-length 5
🤖 Fine-tuning (Planned)
Example LoRA fine-tuning (to be implemented):

bash
Copy
Edit
python fine_tuning/train_lora.py \
  --dataset data/processed/all_clean.jsonl \
  --model llama3-8b \
  --output-dir ./lora_model \
  --epochs 3