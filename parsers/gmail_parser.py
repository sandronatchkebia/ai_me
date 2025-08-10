#!/usr/bin/env python3
"""
Parse Gmail Takeout MBOX files (personal + school) into a clean JSONL corpus
suited for style/adaptation fine-tuning.

Features
- Extracts text from multipart emails (prefers text/plain; falls back to HTML -> text)
- Strips quoted reply blocks and common signatures
- Preserves a raw body copy for debugging
- Normalizes participants (from/to/cc/bcc) and timestamps (UTC ISO8601)
- Marks direction (outbound/inbound) based on provided --my addresses
- Captures Gmail-specific headers like X-GM-THRID and X-Gmail-Labels when present
- Threads messages via X-GM-THRID; falls back to References/In-Reply-To chaining
- Emits per-message JSONL with conversation_id and thread_index for ordering
- Optional language detection (langdetect). If unavailable, sets "unknown".

Usage
    python parse_gmail_mbox.py \
        --mbox ~/Takeout/Mail/personal.mbox ~/Takeout/Mail/berkeley.mbox \
        --my "me@example.com, first.last@berkeley.edu" \
        --out ./gmail_parsed.jsonl

Tip
- Provide ALL of your addresses/aliases in --my (comma-separated). Case-insensitive.
- You can re-run safely; output is deterministic given the same inputs.
"""

from __future__ import annotations
import argparse
import email
import email.policy
import html
import json
import mailbox
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, getaddresses
from hashlib import blake2b
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    from langdetect import detect as lang_detect  # pip install langdetect
except Exception:
    lang_detect = None

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_WS_RE = re.compile(r"\s+")

# Rough heuristics for quoted replies and signatures
RE_QUOTED_INTRO = re.compile(
    r"(^|\n)On\s.+?wrote:\n.*$", re.IGNORECASE | re.DOTALL
)
RE_SIG_SPLIT = re.compile(r"\n--\s?$")  # RFC 3676 signature delimiter
RE_LINE_QUOTE = re.compile(r"^>(.*)$", re.MULTILINE)

# Gmail headers often present in Takeout MBOX
GMAIL_THRID = "X-GM-THRID"
GMAIL_MSGID = "X-GM-MSGID"
GMAIL_LABELS = "X-Gmail-Labels"


def decode_str(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        return str(make_header(decode_header(s)))
    except Exception:
        return s


def to_iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def extract_addresses(hval: str) -> List[str]:
    addrs = [addr.lower() for _, addr in getaddresses([hval or ""]) if addr]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for a in addrs:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def html_to_text(s: str) -> str:
    # minimalist HTML -> text conversion without external deps
    s = re.sub(r"<\s*(br|/p)\s*>", "\n", s, flags=re.IGNORECASE)
    s = HTML_TAG_RE.sub("", s)
    s = html.unescape(s)
    return s


def choose_part(msg: email.message.Message) -> Tuple[str, str]:
    """Return (mime_type, text) choosing text/plain, else HTML->text, else empty."""
    if msg.is_multipart():
        # prefer the first text/plain part that isn't an attachment
        for part in msg.walk():
            cdisp = (part.get("Content-Disposition") or "").lower()
            if part.get_content_maintype() == "text" and "attachment" not in cdisp:
                subtype = part.get_content_subtype()
                payload = part.get_payload(decode=True) or b""
                try:
                    text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    text = payload.decode("utf-8", errors="replace")
                if subtype == "plain":
                    return ("text/plain", text)
        # fall back to first text/html
        for part in msg.walk():
            cdisp = (part.get("Content-Disposition") or "").lower()
            if part.get_content_type() == "text/html" and "attachment" not in cdisp:
                payload = part.get_payload(decode=True) or b""
                try:
                    text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    text = payload.decode("utf-8", errors="replace")
                return ("text/html", html_to_text(text))
        return ("text/plain", "")
    else:
        ctype = (msg.get_content_type() or "").lower()
        payload = msg.get_payload(decode=True) or b""
        try:
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            text = payload.decode("utf-8", errors="replace")
        if ctype == "text/html":
            return ("text/html", html_to_text(text))
        return ("text/plain", text)


def strip_signatures_and_quotes(text: str) -> str:
    if not text:
        return text
    # Drop everything after a standard signature delimiter if it's near the end
    parts = RE_SIG_SPLIT.split(text)
    if len(parts) > 1:
        text = parts[0]
    # Remove quoted-intro blocks ("On … wrote:")
    text = RE_QUOTED_INTRO.sub("\n", text)
    # Remove purely quoted lines (starting with '>') when a majority of block lines are quoted
    lines = text.splitlines()
    if lines:
        # If more than 60% of non-empty lines are quoted, keep only non-quoted
        non_empty = [ln for ln in lines if ln.strip()]
        if non_empty:
            quoted = [ln for ln in non_empty if ln.lstrip().startswith(">")]
            if len(quoted) / max(1, len(non_empty)) > 0.6:
                lines = [ln for ln in lines if not ln.lstrip().startswith(">")]
    text = "\n".join(lines)
    # Normalize whitespace
    text = MULTI_WS_RE.sub(" ", text)
    text = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", text)  # collapse >2 blank lines
    return text.strip()


def pii_redact(s: str) -> str:
    if not s:
        return s
    # Pass 1: emails
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", s)
    # Pass 2: phone numbers (very rough, international)
    s = re.sub(r"(?:(?<!\w)\+?\d[\d\s().-]{8,}\d)(?!\w)", "[PHONE]", s)
    # Pass 3: URLs
    s = re.sub(r"https?://\S+", "[LINK]", s)
    # Pass 4: simple IDs and ticket refs like ABC-1234
    s = re.sub(r"\b[A-Z]{2,5}-\d{2,6}\b", "[ID]", s)
    return s


def safe_lang(s: str) -> str:
    if not s or len(s) < 20:
        return "unknown"
    if lang_detect is None:
        return "unknown"
    try:
        return lang_detect(s)
    except Exception:
        return "unknown"


def hash_fallback_thread(subject: str, refs: List[str]) -> str:
    key = (subject or "").strip().lower()
    root = refs[0] if refs else ""
    h = blake2b(digest_size=10)
    h.update((key + "\u241f" + root).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def make_conversation_id(msg: email.message.Message) -> str:
    th = msg.get(GMAIL_THRID)
    if th:
        return th.strip()
    refs = []
    for h in (msg.get_all("References") or []):
        refs.extend(re.findall(r"<[^>]+>", h))
    in_reply = msg.get("In-Reply-To") or ""
    if in_reply:
        refs.append(in_reply)
    subject = decode_str(msg.get("Subject"))
    return hash_fallback_thread(subject, refs)


def message_id(msg: email.message.Message) -> str:
    mid = (msg.get("Message-ID") or msg.get("Message-Id") or "").strip()
    if mid:
        return mid
    # fabricate a stable ID
    h = blake2b(digest_size=10)
    h.update((decode_str(msg.get("Subject")) + (msg.get("Date") or "") + (msg.get("From") or "")).encode("utf-8", errors="ignore"))
    return f"<gen-{h.hexdigest()}>"


def parse_one(msg: email.message.Message, my_set: set, src_path: str) -> Dict:
    subj = decode_str(msg.get("Subject"))
    # Normalize subject for deduplication
    subject_norm = subj
    if subject_norm:
        # Strip common prefixes (case-insensitive)
        subject_norm = re.sub(r'^(Re|Fwd|Fw|Forward|Reply):\s*', '', subject_norm, flags=re.IGNORECASE)
        # Strip extra whitespace
        subject_norm = re.sub(r'\s+', ' ', subject_norm).strip()
    
    date_hdr = msg.get("Date")
    try:
        dt = parsedate_to_datetime(date_hdr) if date_hdr else None
    except Exception:
        dt = None
    dt_iso = to_iso_utc(dt) if dt else None

    from_addr = extract_addresses(msg.get("From") or "")
    to_addrs = extract_addresses(
        ", ".join((msg.get("To") or "", msg.get("Delivered-To") or ""))
    )
    cc_addrs = extract_addresses(msg.get("Cc") or "")
    bcc_addrs = extract_addresses(msg.get("Bcc") or "")

    ctype, body = choose_part(msg)
    body_raw = body
    body = strip_signatures_and_quotes(body)
    body = pii_redact(body)

    labels = [l.strip() for l in (msg.get(GMAIL_LABELS) or "").split(",") if l.strip()]

    direction = "outbound" if any(a in my_set for a in from_addr) else "inbound"
    
    # Identify which account sent the message (for outbound messages)
    account = None
    sender_domain = None
    if direction == "outbound" and from_addr:
        # Find which of my addresses matches
        for addr in from_addr:
            if addr in my_set:
                account = addr
                # Extract domain from email address
                if '@' in addr:
                    sender_domain = addr.split('@')[1].lower()
                break
    
    # Create privacy-preserving partner ID hash
    partner_id = None
    if direction == "outbound":
        # For outbound messages, hash the recipients
        other_participants = []
        other_participants.extend(to_addrs)
        other_participants.extend(cc_addrs)
        other_participants.extend(bcc_addrs)
        # Remove duplicates and sort for consistent hashing
        other_participants = sorted(list(set(other_participants)))
        if other_participants:
            h = blake2b(digest_size=8)
            h.update(('\u241f'.join(other_participants)).encode('utf-8'))
            partner_id = h.hexdigest()
    elif direction == "inbound":
        # For inbound messages, hash the sender
        if from_addr:
            h = blake2b(digest_size=8)
            h.update(('\u241f'.join(sorted(from_addr))).encode('utf-8'))
            partner_id = h.hexdigest()

    # Flags for instruction building
    is_reply = msg.get("In-Reply-To") is not None
    turn_role = "you" if direction == "outbound" else "partner"

    # Length features for sampling/weighting
    char_len = len(body) if body else 0
    token_est = max(1, char_len // 4)  # Rough estimate: ~4 chars per token
    has_emoji = bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF]', body or ''))

    lang = safe_lang(body)

    doc = {
        "platform": "gmail",
        "source_file": src_path,
        "conversation_id": make_conversation_id(msg),
        "message_id": message_id(msg),
        "date": dt_iso,
        "subject": subj,
        "subject_norm": subject_norm,
        "from": from_addr[0] if from_addr else None,
        "to": to_addrs,
        "cc": cc_addrs,
        "bcc": bcc_addrs,
        "direction": direction,
        "account": account,
        "sender_domain": sender_domain,
        "partner_id": partner_id,
        "is_reply": is_reply,
        "turn_role": turn_role,
        "char_len": char_len,
        "token_est": token_est,
        "has_emoji": has_emoji,
        "labels": labels or None,
        "content_type": ctype,
        "body_raw": body_raw.strip() if body_raw else "",
        "body_text": body.strip() if body else "",
        "lang": lang,
        "reply_to_message_id": msg.get("In-Reply-To"),
        "headers": {
            "In-Reply-To": msg.get("In-Reply-To"),
            "References": msg.get("References"),
        },
    }
    return doc


def thread_and_index(records: List[Dict]) -> List[Dict]:
    # group by conversation_id and sort by date, then add thread_index
    print(f"Grouping {len(records)} messages into threads...")
    threads: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        threads[r["conversation_id"]].append(r)
    
    print(f"Found {len(threads)} unique conversations")
    out = []
    for cid, msgs in threads.items():
        msgs.sort(key=lambda r: (r.get("date") or "", r.get("message_id")))
        for i, r in enumerate(msgs):
            r["thread_index"] = i
            out.append(r)
    # stable order by (date, conversation_id, thread_index)
    out.sort(key=lambda r: (r.get("date") or "", r["conversation_id"], r["thread_index"]))
    return out


def parse_mbox(paths: List[Path], my_addresses: List[str]) -> List[Dict]:
    policy = email.policy.default
    my_set = {a.strip().lower() for a in my_addresses if a.strip()}
    all_records: List[Dict] = []

    print(f"Processing {len(paths)} mbox files...")
    for p in paths:
        print(f"Opening mbox: {p}")
        print(f"Processing all messages from {p.name}...")
        
        # Use mailbox library for efficient MBOX parsing
        mbox = mailbox.mbox(p)
        temp_records = []
        
        print(f"Found {len(mbox)} messages in {p.name}")
        for i, msg in tqdm(enumerate(mbox), total=len(mbox), desc=f"Processing {p.name}"):
            try:
                rec = parse_one(msg, my_set, str(p))
                if rec["body_text"] or rec["body_raw"]:
                    temp_records.append(rec)
            except Exception as e:
                print(f"[warn] failed to parse message {i} in {p.name}: {e}", file=sys.stderr)
        
        print(f"Collected {len(temp_records)} messages from {p.name}")
        
        # Filter: only keep conversations where you participated
        print(f"Filtering conversations where you participated...")
        conversations_with_me = set()
        for rec in temp_records:
            if rec["direction"] == "outbound":
                conversations_with_me.add(rec["conversation_id"])
        
        # Keep only messages from conversations where you sent at least one message
        filtered_records = [rec for rec in temp_records if rec["conversation_id"] in conversations_with_me]
        print(f"After filtering: {len(filtered_records)} messages from {len(conversations_with_me)} conversations")
        
        all_records.extend(filtered_records)
    
    print("Threading and indexing messages...")
    all_records = thread_and_index(all_records)
    print(f"Total messages after threading: {len(all_records)}")
    return all_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbox", nargs="+", required=True, help="Paths to one or more .mbox files")
    ap.add_argument("--my", type=str, required=True, help="Comma-separated list of your email addresses/aliases")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    args = ap.parse_args()

    mbox_paths = [Path(p).expanduser() for p in args.mbox]
    my_addrs = [a.strip() for a in args.my.split(",") if a.strip()]
    out_path = Path(args.out).expanduser()

    print(f"Starting Gmail parser...")
    print(f"Input files: {[str(p) for p in mbox_paths]}")
    print(f"My addresses: {my_addrs}")
    print(f"Output file: {out_path}")
    print("-" * 50)

    records = parse_mbox(mbox_paths, my_addrs)
    
    print(f"Writing {len(records)} records to {out_path}...")
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Done! Summary:")
    # Quick summary
    total = len(records)
    inbound = sum(1 for r in records if r["direction"] == "inbound")
    outbound = total - inbound
    convs = len({r["conversation_id"] for r in records})
    print(json.dumps({
        "messages": total,
        "conversations": convs,
        "inbound": inbound,
        "outbound": outbound
    }, indent=2))

if __name__ == "__main__":
    main()
