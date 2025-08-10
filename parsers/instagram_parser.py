#!/usr/bin/env python3
"""
Parse Instagram (Meta/Takeout) messages into a single JSONL for LLM fine-tuning.
- Walks conversation folders recursively (message_*.json only; media ignored)
- Merges multi-part threads, sorts chronologically
- Preserves emojis and Georgian, fixes mojibake per-field (safe)
- Strong PII redaction (URLs/Emails/Phones/IDs) with fewer false-positives
- Georgian-first language detection; thread-majority backfill
- Deterministic IDs; includes IG-specific source_meta
- Optional tqdm progress; optional --limit; no streaming writes

Example:
    python instagram_takeout_to_jsonl.py \
        --in /path/to/messages/inbox \
        --out /path/to/instagram.jsonl \
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
    "gifs": "sticker",  # IG sometimes uses 'gifs' for stickers
    "sticker": "sticker",
    "share": "link",
}

MOJIBAKE_HINT = re.compile(r"(?:â|ð|ï)(?:â|ð|ï|[\u0080-\u00BF]){2,}")  # catch sequences starting with â/ð/ï

def fix_mojibake(s: str) -> str:
    """Repair aggressive mojibake by attempting Latin-1 → UTF-8 conversion."""
    if not isinstance(s, str) or not s:
        return s
    
    # Look for sequences of corrupted UTF-8 bytes that got interpreted as Latin-1
    if not MOJIBAKE_HINT.search(s):
        return s
    
    try:
        # Convert to Latin-1 bytes, then decode as UTF-8
        # This reverses the classic UTF-8 → Latin-1 corruption
        latin1_bytes = s.encode('latin1')
        repaired = latin1_bytes.decode('utf-8', errors='ignore')
        
        # Only keep if it's significantly better (more readable characters, fewer garbled ones)
        if len(repaired) > 0 and len(repaired.strip()) > len(s.strip()) * 0.5:
            return repaired
            
    except Exception:
        pass
    
    return s

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

def stable_conversation_id(participants: list[str], earliest_ts: int | None, folder: str) -> str:
    base = "|".join(sorted(set(participants))) + "|" + (str(earliest_ts or 0)) + "|" + folder
    return "ig-" + short_hash(base, 10)

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
    if GEORGIAN_BLOCK_RE.search(text):
        return "ka"
    if len(text.strip()) < 30:
        return "unknown"
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
    if "replied_to_message_id" in msg and msg["replied_to_message_id"]:
        return str(msg["replied_to_message_id"])
    if isinstance(msg.get("replied_to_message"), dict):
        mid = msg["replied_to_message"].get("message_id")
        if mid:
            return str(mid)
    return None

# --- Core per-conversation processing ----------------------------------------

def build_records_for_conversation(
    conv_folder: Path,
    part_files: list[Path],
    me_set: set[str],
    limit_remaining: int | None,
    debug: bool = False,
) -> tuple[list[dict], int]:
    # Load parts (one JSON per message_*.json); ignore media subfolders
    parts = []
    for fp in sorted(part_files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                parts.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}", file=sys.stderr)

    participants = participants_from_parts(parts)
    msgs = extract_all_messages(parts)
    if not msgs:
        return ([], 0)

    # chronological ascending
    msgs.sort(key=lambda m: m.get("timestamp_ms", 0))

    earliest_ts = msgs[0].get("timestamp_ms")
    thread_path = maybe_thread_path(parts) or ""
    conv_id = stable_conversation_id(participants, earliest_ts, thread_path or conv_folder.name)

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

        # reactions (IG reactions often lack timestamps; align to message time)
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
        ig_mid = str(m.get("message_id")) if m.get("message_id") else None
        thread_index = idx
        message_id = ig_mid or f"{conv_id}-{thread_index:05d}"
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

        # IG source-specific metadata (preserve richness)
        ig_keys_present = sorted(list(m.keys()))
        source_meta = {
            "thread_path": thread_path or None,
            "is_geoblocked_for_viewer": bool(m.get("is_geoblocked_for_viewer", False)),
            "ig_keys_present": ig_keys_present,
        }
        if isinstance(m.get("share"), dict):
            if "share_text" in m["share"]:
                source_meta["share_text"] = fix_mojibake(m["share"]["share_text"])
            if "link" in m["share"]:
                source_meta["share_link"] = m["share"]["link"]

        record = {
            "platform": "instagram",
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

def find_conversation_parts(root: Path) -> dict[Path, list[Path]]:
    """
    Return {conversation_folder: [message_*.json, ...]} for every folder that contains at least one part.
    We walk recursively; ignore media subfolders (we never open non message_*.json).
    """
    conv_map: dict[Path, list[Path]] = defaultdict(list)
    for p in root.rglob("message_*.json"):
        conv_map[p.parent].append(p)
    return conv_map

def parse_args():
    ap = argparse.ArgumentParser(description="Convert Instagram Takeout messages to JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="Input folder (e.g., messages/inbox)")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL file")
    ap.add_argument("--me", dest="me", required=True, help="Comma-separated list of your usernames/aliases")
    ap.add_argument("--limit", dest="limit", type=int, default=None, help="Stop after N messages total")
    ap.add_argument("--progress", action="store_true", help="Show progress with tqdm")
    ap.add_argument("--debug", action="store_true", help="Keep emoji-only msgs and print cleaning samples")
    return ap.parse_args()

def main():
    args = parse_args()

    root = Path(args.inp).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    me_set = set(normalize_name(x) for x in args.me.split(",") if x.strip())
    print(f"[DEBUG] Processing as user: {me_set}", file=sys.stderr)

    conv_map = find_conversation_parts(root)
    print(f"[DEBUG] Found {len(conv_map)} conversation folders", file=sys.stderr)

    conv_items = sorted(conv_map.items(), key=lambda kv: str(kv[0]).lower())
    iterator = conv_items
    if args.progress and tqdm is not None:
        iterator = tqdm(conv_items, desc="Conversations", unit="conv")

    records = []
    produced_total = 0

    for conv_folder, part_files in iterator:
        limit_remaining = None if args.limit is None else max(0, args.limit - produced_total)
        if limit_remaining == 0:
            print(f"[DEBUG] Hit limit of {args.limit} messages, stopping", file=sys.stderr)
            break

        conv_records, produced = build_records_for_conversation(
            conv_folder=conv_folder,
            part_files=part_files,
            me_set=me_set,
            limit_remaining=limit_remaining,
            debug=args.debug,
        )

        if args.debug and conv_records:
            print(f"\n[DEBUG] {conv_folder} -> {len(conv_records)} messages", file=sys.stderr)
            for r in conv_records[:3]:
                print(" RAW :", (r["body_raw"] or "")[:160], file=sys.stderr)
                print(" TEXT:", (r["body_text"] or "")[:160], file=sys.stderr)

        records.extend(conv_records)
        produced_total += produced

    # Single write at the end
    print(f"[DEBUG] Writing {len(records)} records to {out_path}", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(records)} messages to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()