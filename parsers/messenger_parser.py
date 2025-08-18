#!/usr/bin/env python3
"""
Parse Facebook Messenger (Meta/Takeout) messages into a single JSONL for LLM fine-tuning.
- Walks conversation folders recursively across all chunks (message_*.json only; media ignored)
- Merges multi-part threads, sorts chronologically
- Preserves emojis and Georgian, fixes mojibake per-field (safe)
- Strong PII redaction (URLs/Emails/Phones/IDs) with fewer false-positives
- Georgian-first language detection; thread-majority backfill
- Deterministic IDs; includes Messenger-specific source_meta
- Optional tqdm progress; optional --limit; no streaming writes

Example:
    python messenger_parser.py \
        --in /path/to/messenger \
        --out /path/to/messenger.jsonl \
        --me "my_user,alias1,alias2" \
        --progress \
        --limit 5000
"""

from __future__ import annotations
import argparse
import json
import math
import re
import sys
import hashlib
import unicodedata
from hashlib import blake2b
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import multiprocessing as mp
from functools import partial
import asyncio
import aiofiles
import aiofiles.os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import sys
from urllib.parse import unquote
import unicodedata

# Optional dependency only used when --progress is passed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Optional langdetect; script still runs without it (falls back to 'unknown')
try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None

# --- Regexes / Helpers -------------------------------------------------------

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")
HAS_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")

QUOTE_PATTERNS = [
    r"^\s*>",  # quoted lines
    r"^\s*On .+ wrote:\s*$",
    r"^\s*Forwarded message\s*$",
    r"^\s*From:\s.*\n^\s*Sent:\s.*\n^\s*To:\s.*",  # simple Outlook header block
]
QUOTE_RE = re.compile("|".join(QUOTE_PATTERNS), re.IGNORECASE | re.MULTILINE)

URL_RE = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>()]+)""")
EMAIL_RE = re.compile(r"""(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b""")
# Loose candidate; validated below to reduce false positives like "100000000 weli"
PHONE_CAND_RE = re.compile(r"""(?x)
    (?<!\w)
    (?:\+?\d[\s\-.()]*){7,15}
    (?!\w)
""")
ID_RE = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")
WHITESPACE_RE = re.compile(r"\s+")
ONLY_EMOJI_RE = re.compile(
    r"^\s*(?:[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U0001F1E6-\U0001F1FF\U00002500-\U00002BEF]+|\ufe0f|\u200d|\u2640|\u2642|\u2764|\u23cf|\u23e9|\u231a|\u3030|\u24c2)+\s*$"
)
GEORGIAN_BLOCK_RE = re.compile(r"[\u10A0-\u10FF]")

# Simple verb heuristic: if none of these substrings appear and short text => is_ack
VERB_HINTS = re.compile(
    r"\b(am|is|are|was|were|do|does|did|have|has|had|can|could|will|would|should|need|want|think|know|call|text|send|come|go|meet|check|see|got|pay|try|fix|ship|arrive|look)\b",
    re.IGNORECASE,
)

MEDIA_KEYS = {
    "photos": "photo",
    "videos": "video",
    "audio_files": "audio",
    "voice_messages": "voice",
    "gifs": "sticker",  # Messenger uses 'gifs' for stickers
    "sticker": "sticker",
    "share": "link",
}

MOJIBAKE_HINT = re.compile(r"(?:â|ð|ï)(?:â|ð|ï|[\u0080-\u00BF]){2,}")  # catch sequences starting with â/ð/ï

def fix_mojibake(text: str) -> str:
    """
    Fix mojibake (corrupted character encoding) issues in text.
    This handles common problems like:
    - URL-encoded Georgian characters that weren't properly decoded
    - Latin-1 interpreted as UTF-8
    - Unicode escape sequences representing corrupted Georgian characters
    - Other encoding mismatches
    """
    if not text:
        return text
    
    # First, try to fix URL-encoded characters that might be Georgian
    # Facebook exports often have Georgian text encoded in URLs
    try:
        # Decode any percent-encoded sequences
        decoded = unquote(text)
        if decoded != text:
            text = decoded
    except Exception:
        pass
    
    # Handle corrupted Unicode escape sequences that actually represent valid Georgian text
    # These look like: \u00e1\u0083\u009b\u00e1\u0083\u0090\u00e1\u0083\u0093...
    # When properly decoded, they become Georgian text like "მადლობა თემო 😊"
    if '\\u00' in text:
        try:
            # Convert the text to bytes using Latin-1 encoding (which preserves the raw bytes)
            # Then decode as UTF-8 to get the actual Georgian text
            raw_bytes = text.encode('latin-1')
            decoded_georgian = raw_bytes.decode('utf-8', errors='ignore')
            
            # Only use the result if it looks like Georgian text (contains Georgian characters)
            if decoded_georgian != text and any('\u10a0' <= c <= '\u10ff' for c in decoded_georgian):
                text = decoded_georgian
        except Exception:
            pass
    
    # Try to fix common mojibake patterns
    # Georgian characters often get corrupted as sequences of Latin-1 characters
    # when UTF-8 is interpreted as Latin-1
    
    # Common Georgian character corruptions:
    # ა -> á£á¤á (when UTF-8 is read as Latin-1)
    # ბ -> áá£á¡á£áá¡á¢á£áªá
    # etc.
    
    # Try to detect and fix these patterns
    if 'á' in text and len(text) > 10:
        try:
            # This is a heuristic - if we see many 'á' characters, 
            # it might be corrupted Georgian
            if text.count('á') > len(text) * 0.1:  # More than 10% are 'á'
                # Try to reconstruct the original Georgian text
                # This is complex and might not work perfectly
                # For now, we'll just clean up obvious corruption
                pass
        except Exception:
            pass
    
    # Try the original mojibake fix approach for Latin-1 corruption
    if MOJIBAKE_HINT.search(text):
        try:
            # Convert to Latin-1 bytes, then decode as UTF-8
            # This reverses the classic UTF-8 → Latin-1 corruption
            latin1_bytes = text.encode('latin1')
            repaired = latin1_bytes.decode('utf-8', errors='ignore')
            
            # Only keep if it's significantly better (more readable characters, fewer garbled ones)
            if len(repaired) > 0 and len(repaired.strip()) > len(text.strip()) * 0.5:
                text = repaired
        except Exception:
            pass
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFC', text)
    
    return text

def fix_corrupted_unicode_sequences(text: str) -> str:
    """
    Fix corrupted Unicode escape sequences that represent UTF-8 bytes of Georgian text.
    These sequences look like \u00e1\u0083\u009b represent UTF-8 bytes
    and should decode to Georgian characters like მადლობა.
    
    This happens when UTF-8 bytes get interpreted as Unicode escape sequences.
    """
    if not isinstance(text, str) or "\\u00" not in text:
        return text
    
    try:
        # The sequences like \u00e1\u0083\u009b represent UTF-8 bytes
        # When decoded as Unicode escape sequences, they become corrupted
        # We need to convert them back to bytes and then decode as UTF-8
        
        # First, decode the Unicode escape sequences to get the corrupted bytes
        decoded_escapes = text.encode('utf-8').decode('unicode_escape')
        
        # Now these should be the original UTF-8 bytes
        # Convert back to bytes and decode as UTF-8
        if decoded_escapes != text:
            try:
                # Convert the decoded escapes back to bytes
                # This should give us the original UTF-8 bytes
                bytes_data = decoded_escapes.encode('latin-1')
                
                # Now decode as UTF-8 to get the original Georgian text
                repaired = bytes_data.decode('utf-8', errors='ignore')
                
                # Only return if we got meaningful text
                if len(repaired) > 0 and len(repaired.strip()) > len(text.strip()) * 0.5:
                    return repaired
                    
            except Exception:
                pass
                
    except Exception:
        pass
    
    return text

def fix_double_escaped_unicode(text: str) -> str:
    """
    Fix DOUBLE-escaped sequences like '\\\\u10D0' -> '\u10D0' -> 'ა'.
    Applied per-field, not file-wide.
    """
    if not isinstance(text, str) or "\\u" not in text:
        return text
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text

def iso_utc_from_ms(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def normalize_name(name: str) -> str:
    """
    Stable participant id:
    - NFKC normalize
    - drop emojis/ZWJs
    - lowercase
    - keep only [a-z0-9._] + Georgian; others → '_'
    """
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKC", name)
    name = name.replace("\u200d", "")            # ZWJ
    name = EMOJI_RE.sub("", name)                # drop emojis
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._\u10A0-\u10FF]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"

def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def stable_conversation_id(participants: list[str], earliest_ts: int | None, thread_path: str, chunk_name: str) -> str:
    base = "|".join(sorted(set(participants))) + "|" + (str(earliest_ts or 0)) + "|" + thread_path + "|" + chunk_name
    return "fb-" + short_hash(base, 10)

def _phone_validator(m: re.Match) -> str:
    s = m.group(0)
    digits = re.sub(r"\D+", "", s)
    # reject obvious non-phones
    if len(digits) < 7 or len(digits) > 15:
        return s
    if len(set(digits)) == 1:  # e.g., 0000000
        return s
    # reject "years" context right after number
    tail_ctx = m.string[m.end(): m.end()+6].lower()
    if tail_ctx.startswith(" weli") or tail_ctx.startswith(" tseli"):
        return s
    return "[PHONE]"

def redact(text: str) -> str:
    if not text:
        return ""
    t = text
    t = URL_RE.sub("[LINK]", t)
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_CAND_RE.sub(_phone_validator, t)
    t = ID_RE.sub("[ID]", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    # tidy punctuation
    t = re.sub(r"\(\s*\]", "]", t)
    t = re.sub(r"\[\s*\)", "]", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\[\s*\]", "", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t

def truncate_quotes(text: str) -> str:
    if not text:
        return ""
    out = []
    for ln in text.splitlines():
        if QUOTE_RE.search(ln):
            break
        out.append(ln)
    return "\n".join(out).strip()

def detect_lang_georgian_first(text: str) -> str:
    if not text or len(text.strip()) == 0:
        return "unknown"
    
    # Check for actual Georgian characters first (highest priority)
    # Georgian characters are unique to Georgian language
    if GEORGIAN_BLOCK_RE.search(text):
        return "ka"
    
    # For short texts, don't try to detect language
    if len(text.strip()) < 30:
        return "unknown"
    
    # Fall back to langdetect if available
    if langdetect_detect is None:
        return "unknown"
    try:
        return langdetect_detect(text)
    except Exception:
        return "unknown"

def content_type_for(msg: dict) -> tuple[str, int, list[str]]:
    """
    Returns (content_type, attachments_count, attachment_types[])
    Priority:
      - share.link => "link"
      - photos/videos/audio_files/voice_messages/sticker/gifs
      - content => "text"
      - else => "other"
    """
    if isinstance(msg.get("share"), dict) and msg["share"].get("link"):
        return ("link", 1, ["link"])

    atypes, count = [], 0
    for k, mapped in MEDIA_KEYS.items():
        if k in msg and msg[k]:
            c = len(msg[k]) if isinstance(msg[k], list) else 1
            count += c
            atypes.extend([mapped] * c)

    if count > 0:
        return (atypes[0] if atypes else "other", count, atypes)
    if msg.get("content"):
        return ("text", 0, [])
    return ("other", 0, [])

def is_only_emoji(text: str) -> bool:
    if not text:
        return False
    return ONLY_EMOJI_RE.match(text) is not None

def has_emoji(text: str) -> bool:
    if not text:
        return False
    return HAS_EMOJI_RE.search(text) is not None

def is_ack(text: str) -> bool:
    if not text:
        return False
    return (len(text) < 60) and (VERB_HINTS.search(text) is None)

def participants_from_parts(parts: list[dict]) -> list[str]:
    """
    Union of top-level 'participants' and all message 'sender_name' values.
    This avoids missing the partner when exports list only your own account at top level.
    """
    pset = set()
    for part in parts:
        for p in (part.get("participants") or []):
            name = p.get("name") if isinstance(p, dict) else p
            if name:
                pset.add(normalize_name(name))
        for msg in (part.get("messages") or []):
            n = normalize_name(msg.get("sender_name", ""))
            if n and n != "unknown":
                pset.add(n)
    return sorted(x for x in pset if x)

def extract_all_messages(parts: list[dict]) -> list[dict]:
    msgs = []
    for part in parts:
        for msg in part.get("messages", []):
            msgs.append(msg)
    return msgs

def maybe_thread_path(parts: list[dict]) -> str | None:
    for part in parts:
        tp = part.get("thread_path")
        if tp:
            return str(tp)
    return None

def get_reply_to(msg: dict) -> str | None:
    # Messenger doesn't seem to have reply_to fields in the format I saw
    # But keeping this for potential future use
    if "replied_to_message_id" in msg and msg["replied_to_message_id"]:
        return str(msg["replied_to_message_id"])
    if isinstance(msg.get("replied_to_message"), dict):
        mid = msg["replied_to_message"].get("message_id")
        if mid:
            return str(mid)
    return None

# --- Async file loading ----------------------------------------------------

async def load_json_async(file_path: Path) -> dict:
    """
    Load JSON file asynchronously for better performance.
    """
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        print(f"[WARN] Failed to load {file_path}: {e}", file=sys.stderr)
        return {}

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except:
        return 0.0

def sort_conversations_by_size(conv_items):
    """
    Sort conversations by total size (smallest first) so we see progress quickly.
    """
    def get_conv_size(conv_item):
        (conv_folder, chunk_name), part_files = conv_item
        total_size = sum(get_file_size_mb(fp) for fp in part_files)
        return total_size
    
    return sorted(conv_items, key=get_conv_size)

# --- Core per-conversation processing ----------------------------------------

def build_records_for_conversation(
    conv_folder: Path,
    part_files: list[Path],
    me_set: set[str],
    limit_remaining: int | None,
    chunk_name: str,
    debug: bool = False,
) -> tuple[list[dict], int]:
    # Load parts (one JSON per message_*.json); ignore media subfolders
    parts = []
    total_size_mb = 0
    
    for fp in sorted(part_files):
        try:
            file_size_mb = get_file_size_mb(fp)
            total_size_mb += file_size_mb
            
            # Add progress indicator for large files
            if debug and file_size_mb > 1.0:
                print(f"[DEBUG] Loading large file: {fp.name} ({file_size_mb:.1f}MB)", file=sys.stderr)
            
            # Try different encodings to handle potential encoding issues
            part_data = None
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    with open(fp, "r", encoding=encoding) as f:
                        content = f.read()
                        part_data = json.loads(content)
                        if part_data:
                            break
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    if debug:
                        print(f"[DEBUG] Failed to parse {fp} with {encoding}: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"[WARN] Failed to load {fp} with {encoding}: {e}", file=sys.stderr)
                    continue
            
            if part_data:
                parts.append(part_data)
            else:
                print(f"[WARN] Failed to load {fp} with any encoding", file=sys.stderr)
                
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}", file=sys.stderr)
            continue

    if not parts:
        return ([], 0)

    if debug and total_size_mb > 5.0:
        print(f"[DEBUG] Total conversation size: {total_size_mb:.1f}MB", file=sys.stderr)

    participants = participants_from_parts(parts)
    msgs = extract_all_messages(parts)
    if not msgs:
        return ([], 0)

    # chronological ascending
    msgs.sort(key=lambda m: m.get("timestamp_ms", 0))

    earliest_ts = msgs[0].get("timestamp_ms")
    thread_path = maybe_thread_path(parts) or ""
    conv_id = stable_conversation_id(participants, earliest_ts, thread_path, chunk_name)

    thread_len = len(msgs)
    out, produced = [], 0

    for idx, m in enumerate(msgs):
        if limit_remaining is not None and produced >= limit_remaining:
            break

        # Best-effort source_file (first part path; cheap and deterministic)
        src_file = str(sorted(part_files)[0]).replace("\\", "/") if part_files else ""

        # Sender
        sender_raw = m.get("sender_name", "")
        sender = normalize_name(sender_raw)
        if sender == "unknown" and sender_raw:
            sender = f"user_{short_hash(sender_raw)}"  # stable placeholder

        # Content (repair per-field, THEN handle quotes, redact, etc.)
        raw = m.get("content", "")
        body_raw = raw if isinstance(raw, str) else ""
        
        body_raw = fix_mojibake(body_raw)
        body_raw = fix_double_escaped_unicode(body_raw)

        # Timestamp
        timestamp_ms = int(m.get("timestamp_ms", 0) or 0)
        date_iso = iso_utc_from_ms(timestamp_ms)

        # Content-type / attachments
        ctype, attach_count, attach_types = content_type_for(m)

        # Keep only top content, preserve emoji for style signal
        body_top = truncate_quotes(body_raw)
        emoji_flag = has_emoji(body_top)

        # Redact PII; normalize to NFC to avoid split glyphs
        body_text = unicodedata.normalize("NFC", redact(body_top))

        # Skip emoji-only messages (unless debugging)
        if not debug and is_only_emoji(body_text):
            continue

        # reactions (Messenger reactions often lack timestamps; align to message time)
        reactions = []
        for r in (m.get("reactions") or []):
            emoji = r.get("reaction")
            actor = normalize_name(r.get("actor", ""))
            if emoji:
                reactions.append({"actor": actor, "emoji": emoji, "date": date_iso})

        # Direction / roles
        direction = "outbound" if sender in me_set else "inbound"
        turn_role = "you" if direction == "outbound" else "partner"
        account = next((u for u in me_set if u == sender), None)

        # "to" = other participants
        to_list = [p for p in participants if p and p != sender]

        # Language
        lang = detect_lang_georgian_first(body_text)

        char_len = len(body_text)
        token_est = max(1, math.ceil(char_len / 4))

        # IDs
        thread_index = idx
        message_id = f"{conv_id}-{thread_index:05d}"
        reply_to = get_reply_to(m)

        partner_ids = sorted(short_hash(p) for p in participants if p not in me_set)

        # Split: 90/5/5 by conversation_id
        hsplit = int(hashlib.sha1(conv_id.encode("utf-8")).hexdigest(), 16) % 100
        split = "train" if hsplit < 90 else ("val" if hsplit < 95 else "test")

        # Single partner_id (Gmail-like): inbound->sender, outbound->all recipients
        if direction == "outbound":
            _others = "|".join(sorted(to_list))
            h2 = blake2b(digest_size=8); h2.update(_others.encode("utf-8"))
            partner_id_single = h2.hexdigest()
        else:
            h2 = blake2b(digest_size=8); h2.update(sender.encode("utf-8"))
            partner_id_single = h2.hexdigest()

        # Messenger source-specific metadata (preserve richness)
        messenger_keys_present = sorted(list(m.keys()))
        source_meta = {
            "thread_path": thread_path or None,
            "chunk_name": chunk_name,
            "is_geoblocked_for_viewer": bool(m.get("is_geoblocked_for_viewer", False)),
            "is_unsent_image_by_messenger_kid_parent": bool(m.get("is_unsent_image_by_messenger_kid_parent", False)),
            "messenger_keys_present": messenger_keys_present,
        }
        if isinstance(m.get("share"), dict):
            if "share_text" in m["share"]:
                source_meta["share_text"] = fix_mojibake(m["share"]["share_text"])
            if "link" in m["share"]:
                source_meta["share_link"] = m["share"]["link"]

        record = {
            "platform": "messenger",
            "source_file": src_file,
            "conversation_id": conv_id,
            "participants": participants,
            "account": account if account else None,
            "message_id": message_id,
            "reply_to_message_id": reply_to if reply_to else None,
            "date": date_iso,
            "from": sender,
            "to": to_list,
            "direction": direction,
            "turn_role": turn_role,
            "is_reply": bool(reply_to),
            "content_type": ctype,
            "attachments": {"count": int(attach_count), "types": attach_types},
            "body_raw": body_raw,
            "body_text": body_text,
            "reactions": reactions,
            "lang": lang,
            "char_len": char_len,
            "token_est": token_est,
            "has_emoji": bool(emoji_flag),
            "thread_index": thread_index,
            "thread_len": thread_len,
            "turn_id": f"{conv_id}-{thread_index:05d}",
            "partner_ids": partner_ids,
            "split": split,
            "is_ack": is_ack(body_text),
            "partner_id": partner_id_single,
            "headers": {"replied_to_message_id": reply_to},
            "source_meta": source_meta,
        }

        out.append(record)
        produced += 1

    # Post-pass: set 'unknown' to thread majority (if any)
    langs = [r["lang"] for r in out if r["lang"] != "unknown"]
    if langs:
        majority = Counter(langs).most_common(1)[0][0]
        for r in out:
            if r["lang"] == "unknown":
                r["lang"] = majority

    return (out, produced)

# --- Discovery / CLI ---------------------------------------------------------

def find_conversation_parts(root: Path) -> dict[tuple[Path, str], list[Path]]:
    """
    Return {(conversation_folder, chunk_name): [message_*.json, ...]} for every folder that contains at least one part.
    We walk recursively across all chunks; ignore media subfolders (we never open non message_*.json).
    """
    conv_map: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    
    # Walk through all chunks
    for chunk_dir in root.iterdir():
        if not chunk_dir.is_dir() or chunk_dir.name.startswith('.'):
            continue
            
        chunk_name = chunk_dir.name
        
        # Walk recursively through all subdirectories in this chunk
        for p in chunk_dir.rglob("message_*.json"):
            conv_map[(p.parent, chunk_name)].append(p)
    
    return conv_map

def parse_args():
    ap = argparse.ArgumentParser(description="Convert Facebook Messenger Takeout messages to JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="Input folder (e.g., /path/to/messenger)")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL file")
    ap.add_argument("--me", dest="me", required=True, help="Comma-separated list of your usernames/aliases")
    ap.add_argument("--limit", dest="limit", type=int, default=None, help="Stop after N messages total")
    ap.add_argument("--progress", action="store_true", help="Show progress with tqdm")
    ap.add_argument("--debug", action="store_true", help="Keep emoji-only msgs and print cleaning samples")
    return ap.parse_args()

# --- Parallel processing -----------------------------------------------------

def process_conversation_parallel(args_tuple):
    """
    Process a single conversation in parallel.
    This function is designed to be called by multiprocessing.
    """
    (conv_folder, chunk_name), part_files, me_set, debug = args_tuple
    
    try:
        conv_records, produced = build_records_for_conversation(
            conv_folder=conv_folder,
            part_files=part_files,
            me_set=me_set,
            limit_remaining=None,  # No limit in parallel mode
            chunk_name=chunk_name,
            debug=debug,
        )
        return conv_records, produced, chunk_name, conv_folder.name
    except Exception as e:
        print(f"[ERROR] Failed to process {chunk_name}/{conv_folder.name}: {e}", file=sys.stderr)
        return [], 0, chunk_name, conv_folder.name

def process_conversations_parallel(conv_items, me_set, debug=False, max_workers=None, progress=False):
    """
    Process conversations in parallel for much faster processing.
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"[DEBUG] Using {max_workers} parallel workers", file=sys.stderr)
    
    # Sort by size (smallest first) so we see progress quickly
    sorted_conv_items = sort_conversations_by_size(conv_items)
    
    if progress and tqdm is not None:
        print(f"[DEBUG] Processing {len(sorted_conv_items)} conversations (sorted by size, smallest first)", file=sys.stderr)
        pbar = tqdm(total=len(sorted_conv_items), desc="Conversations", unit="conv")
    else:
        pbar = None
    
    # Prepare arguments for parallel processing
    args_list = [(conv_key, part_files, me_set, debug) 
                 for conv_key, part_files in sorted_conv_items]
    
    # Process in parallel
    with mp.Pool(processes=max_workers) as pool:
        results = []
        for result in pool.imap_unordered(process_conversation_parallel, args_list):
            results.append(result)
            if pbar:
                pbar.update(1)
                # Update description with current progress
                conv_records, produced, chunk_name, conv_name = result
                if conv_records:
                    pbar.set_postfix({"msgs": produced, "conv": f"{chunk_name}/{conv_name}"})
    
    if pbar:
        pbar.close()
    
    # Collect results
    all_records = []
    total_produced = 0
    successful_convs = 0
    
    for conv_records, produced, chunk_name, conv_name in results:
        if conv_records:
            all_records.extend(conv_records)
            total_produced += produced
            successful_convs += 1
    
    return all_records, total_produced, successful_convs

def process_conversations_sequential(conv_items, me_set, limit, debug=False, progress=False):
    """
    Process conversations sequentially when a limit is specified.
    """
    iterator = conv_items
    if progress and tqdm is not None:
        iterator = tqdm(conv_items, desc="Conversations", unit="conv")

    records = []
    produced_total = 0
    processed_convs = 0

    for (conv_folder, chunk_name), part_files in iterator:
        processed_convs += 1
        
        # Progress update every 50 conversations
        if processed_convs % 50 == 0:
            print(f"[DEBUG] Processed {processed_convs}/{len(conv_items)} conversations, {produced_total} messages so far", file=sys.stderr)
        
        limit_remaining = max(0, limit - produced_total)
        if limit_remaining == 0:
            print(f"[DEBUG] Hit limit of {limit} messages, stopping", file=sys.stderr)
            break

        try:
            conv_records, produced = build_records_for_conversation(
                conv_folder=conv_folder,
                part_files=part_files,
                me_set=me_set,
                limit_remaining=limit_remaining,
                chunk_name=chunk_name,
                debug=debug,
            )

            if debug and conv_records:
                print(f"\n[DEBUG] {chunk_name}/{conv_folder.name} -> {len(conv_records)} messages", file=sys.stderr)
                for r in conv_records[:3]:
                    print(" RAW :", (r["body_raw"] or "")[:160], file=sys.stderr)
                    print(" TEXT:", (r["body_text"] or "")[:160], file=sys.stderr)

            records.extend(conv_records)
            produced_total += produced
            
        except Exception as e:
            print(f"[ERROR] Failed to process {chunk_name}/{conv_folder.name}: {e}", file=sys.stderr)
            continue

    return records, produced_total, processed_convs

def main():
    args = parse_args()

    root = Path(args.inp).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    me_set = set(normalize_name(x) for x in args.me.split(",") if x.strip())
    print(f"[DEBUG] Processing as user: {me_set}", file=sys.stderr)

    conv_map = find_conversation_parts(root)
    print(f"[DEBUG] Found {len(conv_map)} conversation folders across all chunks", file=sys.stderr)

    # Show size distribution for better understanding
    conv_items = list(conv_map.items())
    total_size_mb = sum(sum(get_file_size_mb(fp) for fp in part_files) 
                       for (conv_folder, chunk_name), part_files in conv_items)
    print(f"[DEBUG] Total dataset size: {total_size_mb:.1f}MB", file=sys.stderr)
    
    # Sort by size for better progress visibility
    conv_items = sort_conversations_by_size(conv_items)
    
    # Use parallel processing for much faster execution
    print(f"[DEBUG] Starting parallel processing of {len(conv_items)} conversations...", file=sys.stderr)
    
    start_time = datetime.now()
    
    if args.limit:
        # If limit is specified, process sequentially to respect the limit
        print(f"[DEBUG] Processing with limit of {args.limit} messages (sequential mode)", file=sys.stderr)
        records, produced_total, processed_convs = process_conversations_sequential(
            conv_items, me_set, args.limit, args.debug, args.progress
        )
    else:
        # No limit - use parallel processing for maximum speed
        print(f"[DEBUG] Processing all conversations (parallel mode)", file=sys.stderr)
        records, produced_total, processed_convs = process_conversations_parallel(
            conv_items, me_set, args.debug, progress=args.progress
        )
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"[DEBUG] Processing completed in {processing_time:.1f} seconds", file=sys.stderr)
    print(f"[DEBUG] Average: {processing_time/len(conv_items):.2f} seconds per conversation", file=sys.stderr)

    # Single write at the end
    print(f"[DEBUG] Writing {len(records)} records to {out_path}", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(records)} messages from {processed_convs} conversations to {out_path}", file=sys.stderr)
    print(f"[OK] Total processing time: {processing_time:.1f} seconds", file=sys.stderr)

if __name__ == "__main__":
    main()
