# AI Personal Data Parser

A comprehensive toolkit for parsing and standardizing personal communication data from various platforms into a unified JSONL format for AI/ML analysis and fine-tuning.

## Overview

This project converts raw data exports from Gmail, Instagram, and WhatsApp into a standardized, privacy-preserving format that maintains conversation context while protecting personal information.

## Features

- **Multi-Platform Support**: Gmail (MBOX), Instagram Takeout, WhatsApp TXT exports
- **Privacy-First**: Automatic PII redaction, hashed participant IDs, content sanitization
- **Standardized Output**: Consistent JSONL schema across all platforms
- **Conversation Context**: Thread identification, reply tracking, turn-based analysis
- **Rich Metadata**: Language detection, emoji analysis, content classification, timing data
- **Performance Optimized**: Efficient parsing for large datasets with progress tracking

## Data Schema

All parsers output records with the following core fields:

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
  "thread_len": 5
}
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai_me

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- `tqdm` - Progress bars
- `langdetect` - Language detection (optional)
- Standard library modules: `json`, `re`, `datetime`, `hashlib`, `unicodedata`, `mailbox`

## Usage

### Gmail Parser

Parse MBOX files from Gmail Takeout:

```bash
python3 parsers/gmail_parser.py \
  --in data/raw/gmail.mbox \
  --out data/processed/gmail_parsed.jsonl \
  --me "your.email@gmail.com" \
  --progress
```

### Instagram Parser

Parse Instagram Takeout data:

```bash
python3 parsers/instagram_parser.py \
  --in data/raw/instagram \
  --out data/processed/instagram_parsed.jsonl \
  --me "Your Name" \
  --progress
```

### WhatsApp Parser

Parse extracted TXT chat files:

```bash
python3 parsers/whatsapp_parser.py \
  --in data/raw/whatsapp/messages \
  --out data/processed/whatsapp_parsed.jsonl \
  --me "Your Name" \
  --tz "+04:00" \
  --progress
```

## Data Preparation

### Gmail
- Export data from Gmail Takeout as MBOX format
- Place `.mbox` files in `data/raw/`

### Instagram  
- Download Instagram Takeout data
- Extract to `data/raw/instagram/` folder
- Ensure JSON files are accessible

### WhatsApp
- Export chats as TXT files from WhatsApp
- Extract ZIP files to `data/raw/whatsapp/messages/`
- Parser expects individual `.txt` files

## Output Structure

```
data/
├── raw/                    # Original data exports
│   ├── gmail.mbox
│   ├── instagram/
│   └── whatsapp/
│       ├── messages/       # Extracted TXT files
│       └── zipped/         # Original ZIP archives
└── processed/              # Parsed JSONL outputs
    ├── gmail_parsed.jsonl
    ├── instagram_parsed.jsonl
    └── whatsapp_parsed.jsonl
```

## Privacy & Security

- **PII Redaction**: URLs, emails, phone numbers automatically replaced with placeholders
- **Participant Hashing**: Other users' identities converted to privacy-preserving hashes
- **Content Sanitization**: Malicious content patterns filtered out
- **Local Processing**: All data processing happens locally, no external API calls

## Performance

- **Gmail**: ~1000 messages/second on typical hardware
- **Instagram**: ~800 messages/second with mojibake correction
- **WhatsApp**: ~1000 messages/second for TXT parsing
- **Memory Efficient**: Streams data without loading entire datasets into memory

## Troubleshooting

### Common Issues

1. **Encoding Problems**: Instagram data may have UTF-8 mojibake - the parser includes automatic correction
2. **Large Files**: Use `--limit` parameter for testing on subsets
3. **Timezone Issues**: Specify `--tz` for WhatsApp if timestamps seem incorrect

### Debug Mode

Add `--debug` flag to see sample data processing:

```bash
python3 parsers/instagram_parser.py --in data/raw/instagram --out test.jsonl --me "Test" --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your chosen license]

## Acknowledgments

- Built for personal data analysis and AI model fine-tuning
- Inspired by the need for standardized communication data formats
- Uses established libraries for robust text processing and privacy protection
