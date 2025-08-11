# AI_ME – Personal Conversational Fine-Tuning Pipeline

AI_ME is an end-to-end pipeline for fine-tuning a Large Language Model (LLM) to emulate a user's communication style. It parses personal conversation data from multiple platforms, preprocesses it, and prepares it for model fine-tuning using LoRA (Low-Rank Adaptation).

## 🎯 Project Vision

The goal is to create an AI assistant that sounds and writes like you by learning from your actual communication patterns across email, social media, and messaging platforms. This involves:

- **Data Collection**: Parsing your personal communication data
- **Style Analysis**: Understanding your writing patterns, vocabulary, and communication style
- **Model Training**: Fine-tuning an open-source LLM using LoRA to preserve your unique voice
- **Evaluation**: Testing how well the model captures your communication style

## 🚀 Pipeline Overview

## 📈 Current Status

**✅ COMPLETED:**
- Data parsing from Gmail, Instagram, and WhatsApp
- Advanced data preprocessing with email cleaning
- Dataset preparation in HuggingFace chat format
- 33,024 training examples and 6,405 validation examples ready

**🔄 IN PROGRESS:**
- LoRA fine-tuning setup (requires cloud VM due to PyTorch compatibility)

**🔮 PLANNED:**
- Model deployment and evaluation
- Style consistency monitoring

### Phase 1: Data Export & Parsing ✅ **IMPLEMENTED**
1. **Data Export**  
   Export your data from:
   - Gmail – Google Takeout (.mbox format)
   - Instagram – Meta data export (JSON)
   - WhatsApp – Chat export (TXT files)
   - Facebook Messenger – Meta data export (JSON)

2. **Parsing & Standardization**  
   Each platform parser:
   - Reads the raw export format
   - Extracts key metadata (timestamps, participants, content)
   - Cleans and normalizes message text
   - Handles encoding issues and emoji processing
   - Outputs standardized `.jsonl` files in `data/processed`

### Phase 2: Data Preprocessing ✅ **IMPLEMENTED**
- **Deduplication**: Remove duplicate messages and conversations
- **Language Filtering**: Focus on primary languages (e.g., English, Georgian)
- **Quality Filtering**: Remove low-value/system messages, short responses
- **Conversation Structuring**: Organize messages into proper conversation flows
- **Dataset Splitting**: Create train/validation/test sets for fine-tuning
- **Instruction Formatting**: Convert conversations to instruction-response pairs
- **Email Cleaning**: Advanced signature/quote removal, HTML entity cleaning
- **Chat Formatting**: Convert to HuggingFace chat format with system/user/assistant messages

### Phase 3: Model Fine-tuning 🔮 **PLANNED**
- **Model Selection**: Choose appropriate open-source LLM (e.g., Llama 3, Mistral)
- **LoRA Configuration**: Set up low-rank adaptation for efficient fine-tuning
- **Training Pipeline**: Implement training with conversation data
- **Style Preservation**: Ensure the model maintains your unique communication patterns
- **Evaluation**: Test output quality and style consistency

### Phase 4: Deployment & Integration 🔮 **PLANNED**
- **Model Serving**: Deploy fine-tuned model for inference
- **API Integration**: Create endpoints for real-time communication
- **Style Monitoring**: Track how well the model maintains your voice over time

---

## 📊 Data Schema

All parsers output records with consistent fields for unified processing:

```json
{
  "conversation_id": "stable_hash_of_participants",
  "message_id": "unique_message_identifier", 
  "timestamp_ms": 1234567890000,
  "date_iso": "2023-01-01T12:00:00Z",
  "sender": "normalized_sender_name",
  "direction": "inbound|outbound",
  "participants": ["user1", "user2"],
  "body_raw": "original_message_content",
  "body_text": "cleaned_and_redacted_content",
  "subject": "email_subject_or_null",
  "subject_norm": "normalized_subject_stripping_re_fwd",
  "reply_to_message_id": "in_reply_to_header_or_null",
  "account": "your_email_address_for_outbound",
  "sender_domain": "domain.com",
  "partner_id": "hashed_other_participants",
  "is_reply": true|false,
  "turn_role": "you|partner",
  "content_type": "text|photo|video|link|etc",
  "lang": "detected_language",
  "char_len": 150,
  "token_est": 38,
  "has_emoji": true|false,
  "thread_index": 1,
  "thread_len": 5,
  "platform": "gmail|instagram|whatsapp|messenger"
}
```

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/sandronatchkebia/ai_me.git
cd ai_me

# Install dependencies using uv (recommended) or pip
uv add transformers datasets tokenizers huggingface-hub
# OR
pip install -r requirements.txt

# Note: PEFT and TRL require PyTorch which may not be available on all platforms
# These will be installed on the training VM/cloud instance

# Set up data directories
mkdir -p data/raw data/processed
```

## 📜 Running the Parsers

### Gmail Parser
```bash
python3 parsers/gmail_parser.py \
  --in data/raw/gmail.mbox \
  --out data/processed/gmail_parsed.jsonl \
  --me "your.email@gmail.com" \
  --progress
```

### Instagram Parser
```bash
python3 parsers/instagram_parser.py \
  --in data/raw/instagram \
  --out data/processed/instagram_parsed.jsonl \
  --me "Your Name" \
  --progress
```

### WhatsApp Parser
```bash
python3 parsers/whatsapp_parser.py \
  --in data/raw/whatsapp/messages \
  --out data/processed/whatsapp_parsed.jsonl \
  --me "Your Name" \
  --tz "+04:00" \
  --progress
```

---

## 🔄 Data Preprocessing (Planned)

The preprocessing pipeline will handle data quality and prepare conversations for fine-tuning:

```bash
# Example workflow (to be implemented)
python preprocessing/clean_data.py \
  --in data/processed/*_parsed.jsonl \
  --out data/processed/all_clean.jsonl \
  --languages en ka \
  --min-length 10 \
  --remove-duplicates

python preprocessing/conversation_builder.py \
  --in data/processed/all_clean.jsonl \
  --out data/processed/conversations.jsonl \
  --min-conversation-length 3

python preprocessing/dataset_splitter.py \
  --in data/processed/conversations.jsonl \
  --train data/fine_tuning/train.jsonl \
  --val data/fine_tuning/val.jsonl \
  --test data/fine_tuning/test.jsonl \
  --split-ratio 0.8 0.1 0.1
```

---

## 🤖 Fine-tuning Pipeline (Planned)

### LoRA Configuration
```bash
# Example LoRA fine-tuning (to be implemented)
python fine_tuning/train_lora.py \
  --dataset data/fine_tuning/train.jsonl \
  --val_dataset data/fine_tuning/val.jsonl \
  --model "meta-llama/Llama-3-8B-Instruct" \
  --output_dir ./models/ai_me_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --max_seq_length 2048
```

### Training Features
- **LoRA Adaptation**: Efficient fine-tuning with minimal parameter updates
- **Style Preservation**: Focus on maintaining your unique communication patterns
- **Conversation Context**: Train on full conversation flows, not just individual messages
- **Multi-platform Learning**: Combine data from all sources for comprehensive style modeling

---

## 📁 Project Structure

```
ai_me/
├── parsers/                 # Data parsing modules ✅
│   ├── gmail_parser.py
│   ├── instagram_parser.py
│   ├── whatsapp_parser.py
│   └── messenger_parser.py
├── preprocessing/           # Data cleaning & preparation 🔄
│   ├── clean_data.py       # Deduplication, filtering
│   ├── conversation_builder.py  # Structure conversations
│   └── dataset_splitter.py # Train/val/test splits
├── fine_tuning/            # Model training pipeline 🔮
│   ├── train_lora.py       # LoRA fine-tuning
│   ├── evaluate.py         # Model evaluation
│   └── configs/            # Training configurations
├── models/                 # Trained models & checkpoints 🔮
├── data/                   # Data storage
│   ├── raw/               # Original exports
│   └── processed/         # Parsed & cleaned data
├── notebooks/              # Analysis & experimentation 🔮
├── tests/                  # Unit tests 🔮
└── docs/                   # Documentation 🔮
```

---

## 🔒 Privacy & Security

- **Local Processing**: All data processing happens locally, no external API calls
- **PII Redaction**: Automatic removal of sensitive information (URLs, emails, phone numbers)
- **Participant Hashing**: Other users' identities converted to privacy-preserving hashes
- **Content Sanitization**: Filtering of potentially malicious content patterns
- **Data Retention**: Processed data can be deleted after fine-tuning

---

## 📈 Performance & Scalability

### Parsing Performance
- **Gmail**: ~1000 messages/second on typical hardware
- **Instagram**: ~800 messages/second with mojibake correction
- **WhatsApp**: ~1000 messages/second for TXT parsing
- **Memory Efficient**: Streams data without loading entire datasets

### Training Considerations
- **LoRA Efficiency**: Reduces memory requirements by ~80% compared to full fine-tuning
- **Dataset Size**: Typically 10K-100K messages for effective style learning
- **Hardware Requirements**: Can run on consumer GPUs with 8GB+ VRAM

---

## 🚧 Current Status

- ✅ **Data Parsing**: Gmail, Instagram, WhatsApp parsers fully implemented
- ✅ **Data Standardization**: Consistent JSONL schema across all platforms
- ✅ **Encoding Handling**: Robust UTF-8 and mojibake correction
- 🔄 **Preprocessing**: Data cleaning and conversation structuring (in development)
- 🔮 **Fine-tuning**: LoRA training pipeline (planned)
- 🔮 **Model Deployment**: Inference serving and API (planned)

---

## 🛠️ Development & Contributing

### Prerequisites
- Python 3.8+
- Git
- Basic understanding of LLM fine-tuning concepts

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black parsers/
flake8 parsers/
```

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

---

## 📚 Resources & References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Hugging Face**: [LoRA Fine-tuning Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- **Data Privacy**: [GDPR Data Export Guidelines](https://gdpr.eu/data-export/)
- **Platform APIs**: [Gmail](https://developers.google.com/gmail/api), [Instagram](https://developers.facebook.com/docs/instagram-basic-display-api)

---