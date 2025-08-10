#!/usr/bin/env python3
# whatsapp_txt_to_jsonl.py
"""
Convert WhatsApp TXT exports (one chat per .txt) into a single JSONL for LLM fine-tuning.

Example:
    python whatsapp_txt_to_jsonl.py \
        --in /path/to/whatsapp/folder \
        --out /path/to/whatsapp.jsonl \
        --me "my_user,alias1,alias2" \
        --tz "+04:00" \
        --progress \
        --limit 10000
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import hashlib
import unicodedata
from hashlib import blake2b
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

# Optional progress
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Optional langdetect
try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None

# ============== Regex / helpers ==============

# Line formats (iOS & Android)
RE_BRACKET = re.compile(r'^\[(?P<dt>[^\]]+)\]\s(?:(?P<sender>[^:]+):\s)?(?P<text>.*)$')
RE_DASH    = re.compile(r'^(?P<dt>[^-]+?)\s-\s(?:(?P<sender>[^:]+):\s)?(?P<text>.*)$')

# Normalize weird spaces/marks WhatsApp uses
WS_WEIRD = dict.fromkeys(map(ord, "\u202F\u00A0\u200E\u200F\u2066\u2067\u2068\u2069"), " ")

EMOJI_RE      = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")
HAS_EMOJI_RE  = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
ONLY_EMOJI_RE = re.compile(r"^\s*(?:[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U0001F1E6-\U0001F1FF\U00002500-\U00002BEF]+|\ufe0f|\u200d|\u2640|\u2642|\u2764|\u23cf|\u23e9|\u231a|\u3030|\u24c2)+\s*$")

GEORGIAN_BLOCK_RE = re.compile(r"[\u10A0-\u10FF]")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE   = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>()]+)""")
EMAIL_RE = re.compile(r"""(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b""")

PHONE_CAND_RE = re.compile(r"""(?x)
    (?<!\w)
    (?:\+?\d[\s\-.()]*){7,15}
    (?!\w)
""")
ID_RE = re.compile(r"\b[A-Z]{2,}-\d{2,}\b")

# Media markers seen in TXT exports
MEDIA_HINTS = [
    ("voice",   re.compile(r"(?i)\b(voice note|voice message)\b")),
    ("photo",   re.compile(r"(?i)\b(image|photo)\b.*\bomitted\b|<\s*media\s*omitted\s*>")),
    ("video",   re.compile(r"(?i)\bvideo\b.*\bomitted\b")),
    ("audio",   re.compile(r"(?i)\baudio\b.*\bomitted\b")),
    ("sticker", re.compile(r"(?i)\bsticker\b.*\bomitted\b|\bgif\b.*\bomitted\b")),
    ("other",   re.compile(r"(?i)\b(contact card|document|location|file)\b.*\bomitted\b")),
]

SYSTEM_HINTS = (
    "Messages and calls are end-to-end encrypted",
    "You created this group",
    "changed this group's icon",
    "changed the subject",
    "added",
    "removed",
    "joined using this group's invite link",
    "left",
)

VERB_HINTS = re.compile(
    r"\b(am|is|are|was|were|do|does|did|have|has|had|can|could|will|would|should|need|want|think|know|call|text|send|come|go|meet|check|see|got|pay|try|fix|ship|arrive|look)\b",
    re.IGNORECASE,
)

def normalize_name(name: str) -> str:
    if not isinstance(name, str): return ""
    s = unicodedata.normalize("NFKC", name).translate(WS_WEIRD).replace("\u200d","")
    s = EMOJI_RE.sub("", s).strip().lower()
    s = re.sub(r"[^a-z0-9._\u10A0-\u10FF]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def has_emoji(text: str) -> bool:
    return bool(HAS_EMOJI_RE.search(text or ""))

def is_only_emoji(text: str) -> bool:
    return bool(ONLY_EMOJI_RE.match(text or ""))

def is_ack(text: str) -> bool:
    if not text: return False
    return (len(text) < 60) and (VERB_HINTS.search(text) is None)

def redact(text: str) -> str:
    if not text: return ""
    t = URL_RE.sub("[LINK]", text)
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_CAND_RE.sub(_phone_validator, t)
    t = ID_RE.sub("[ID]", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    t = re.sub(r"\(\s*\]","]", t); t = re.sub(r"\[\s*\)","]", t)
    t = re.sub(r"\(\s*\)","", t);  t = re.sub(r"\[\s*\]","", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t

def _phone_validator(m: re.Match) -> str:
    s = m.group(0)
    digits = re.sub(r"\D+", "", s)
    if len(digits) < 7 or len(digits) > 15: return s
    if len(set(digits)) == 1:               return s
    tail = m.string[m.end(): m.end()+6].lower()
    if tail.startswith(" weli") or tail.startswith(" year"): return s
    return "[PHONE]"

def truncate_quotes(text: str) -> str:
    if not text: return ""
    # WhatsApp sometimes has quoted prefixes like "⤷" or ">"; keep top lines until typical quote markers
    lines = (text or "").splitlines()
    out = []
    for ln in lines:
        if ln.startswith("> ") or ln.startswith("⤷"):
            break
        out.append(ln)
    return "\n".join(out).strip()

def detect_lang_georgian_first(text: str) -> str:
    if not text or len(text.strip()) == 0: return "unknown"
    if GEORGIAN_BLOCK_RE.search(text):     return "ka"
    if len(text.strip()) < 30:             return "unknown"
    if langdetect_detect is None:          return "unknown"
    try:
        return langdetect_detect(text)
    except Exception:
        return "unknown"

def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:n]

def stable_conversation_id(participants: list[str], earliest_ts: int | None, fname: str) -> str:
    base = "|".join(sorted(set(p for p in participants if p))) + "|" + str(earliest_ts or 0) + "|" + fname
    return "wa-" + short_hash(base, 10)

# ----- time parsing -----

DT_FORMATS = [
    # month/day first (US style), 12h
    "%m/%d/%y, %I:%M:%S %p",
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M:%S %p",
    "%m/%d/%Y, %I:%M %p",
    # day/month first (many locales), 24h
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%y, %H:%M",
    "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%Y, %H:%M",
    # Loose fallback (some exports omit seconds/comma variance)
    "%m/%d/%y %I:%M %p",
    "%d/%m/%y %H:%M",
]

def parse_dt_local(dt_str: str, tz_local: timezone) -> int:
    s = unicodedata.normalize("NFKC", dt_str or "").translate(WS_WEIRD)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove trailing commas if present like "4/18/25, 1:48 AM" -> keep as-is for formats above
    for fmt in DT_FORMATS:
        try:
            dt_naive = datetime.strptime(s, fmt)
            dt_local = dt_naive.replace(tzinfo=tz_local)
            return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)
        except Exception:
            continue
    # last resort: try to split date + time heuristically
    raise ValueError(f"Unrecognized date format: {dt_str}")

def parse_tz(arg: str | None) -> timezone:
    if not arg:
        # use local offset
        return datetime.now().astimezone().tzinfo or timezone.utc
    arg = arg.strip()
    m = re.match(r"^([+-])(\d{2}):(\d{2})$", arg)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hh = int(m.group(2)); mm = int(m.group(3))
        return timezone(sign * timedelta(hours=hh, minutes=mm))
    # fallback: treat unknown as UTC
    return timezone.utc

# ----- content-type heuristics -----

def classify_content(text: str) -> tuple[str, int, list[str], bool]:
    """
    Returns (ctype, count, types, reply_hint)
    """
    t = (text or "").strip()
    # reply hint (heuristic)
    reply_hint = t.startswith("> ") or t.startswith("⤷")
    # media omitted markers
    for mapped, rx in MEDIA_HINTS:
        if rx.search(t):
            return (mapped, 1, [mapped], reply_hint)
    # pure/mostly link?
    if URL_RE.search(t):
        stripped = URL_RE.sub("", t).strip()
        if len(stripped) <= 4:  # almost only link(s)
            return ("link", 1, ["link"], reply_hint)
    # default text / other
    if t:
        return ("text", 0, [], reply_hint)
    return ("other", 0, [], reply_hint)

# ============== Core parsing per TXT ==============

def parse_whatsapp_txt(fp: Path, tz_local: timezone) -> tuple[list[dict], list[str]]:
    """
    Returns (msgs, participants)
    msgs: [{ms:int, sender_raw:str|None, text:str, system:bool, source_file:str}]
    participants: normalized names seen as human senders
    """
    msgs: list[dict] = []
    participants_set = set()
    with open(fp, "r", encoding="utf-8", errors="strict") as f:
        cur = None  # current message accumulator for multiline
        for raw_line in f:
            line = raw_line.rstrip("\n").translate(WS_WEIRD)
            if not line.strip():
                # blank lines extend current message with newline for fidelity
                if cur is not None:
                    cur["text"] += "\n"
                continue

            m = RE_BRACKET.match(line) or RE_DASH.match(line)
            if m:
                # commit previous message
                if cur is not None:
                    msgs.append(cur); cur = None

                dt_str  = m.group("dt")
                sender  = m.group("sender")
                text    = m.group("text") or ""

                try:
                    ms = parse_dt_local(dt_str, tz_local)
                except Exception:
                    # if date is unparsable, treat line as continuation
                    if msgs:
                        msgs[-1]["text"] += "\n" + line
                        continue
                    else:
                        # drop it
                        continue

                system = False
                sender_norm = None
                if sender:
                    # Some system notices prefix with group name as "sender"; detect by text
                    if any(text.startswith(h) for h in SYSTEM_HINTS):
                        system = True
                        sender_norm = "system"
                    else:
                        sender_norm = normalize_name(sender)
                        if sender_norm and sender_norm != "unknown":
                            participants_set.add(sender_norm)
                else:
                    system = True
                    sender_norm = "system"

                cur = {
                    "ms": ms,
                    "sender_raw": sender_norm,
                    "text": text,
                    "system": system,
                    "source_file": str(fp).replace("\\", "/"),
                }
            else:
                # continuation of previous message
                if cur is None:
                    # rare header noise -> skip
                    continue
                cur["text"] += "\n" + line

        if cur is not None:
            msgs.append(cur)

    msgs.sort(key=lambda x: x["ms"])
    participants = sorted(participants_set)
    return msgs, participants

# ============== Build records ==============

def build_records_for_file(
    fp: Path,
    tz_local: timezone,
    me_set: set[str],
    limit_remaining: int | None,
    debug: bool = False,
) -> tuple[list[dict], int]:
    msgs, participants = parse_whatsapp_txt(fp, tz_local)
    if not msgs:
        return ([], 0)

    earliest_ts = msgs[0]["ms"]
    conv_id = stable_conversation_id(participants, earliest_ts, fp.name)
    thread_len = len(msgs)

    out = []
    produced = 0

    for idx, m in enumerate(msgs):
        if limit_remaining is not None and produced >= limit_remaining:
            break

        sender = m["sender_raw"] or "system"
        date_iso = datetime.fromtimestamp(m["ms"]/1000.0, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

        body_raw = m["text"]
        # keep only top content
        body_top = truncate_quotes(body_raw)
        emoji_flag = has_emoji(body_top)
        # redact + normalize
        body_text = unicodedata.normalize("NFC", redact(body_top))

        # content-type & attachments
        ctype, attach_count, attach_types, reply_hint = classify_content(body_top)

        # skip emoji-only unless debug
        if not debug and is_only_emoji(body_text):
            continue

        # direction/roles
        direction = "outbound" if (sender in me_set) else "inbound"
        if sender == "system":
            direction = "inbound"
        turn_role = "you" if direction == "outbound" else "partner"
        account = sender if sender in me_set else None

        # to-list (others)
        to_list = [p for p in participants if p and p != sender]
        if sender == "system":
            to_list = [p for p in participants]  # system addressed to all

        lang = detect_lang_georgian_first(body_text)
        char_len = len(body_text)
        token_est = max(1, math.ceil(char_len / 4))

        thread_index = idx
        message_id = f"{conv_id}-{thread_index:05d}"
        reply_to = None  # TXT exports don't carry stable IDs

        partner_ids = sorted(short_hash(p) for p in participants if p not in me_set)

        # split 90/5/5
        hsplit = int(hashlib.sha1(conv_id.encode("utf-8")).hexdigest(), 16) % 100
        split = "train" if hsplit < 90 else ("val" if hsplit < 95 else "test")

        # partner_id single (gmail-like)
        if direction == "outbound":
            others = "|".join(sorted(to_list))
            h2 = blake2b(digest_size=8); h2.update(others.encode("utf-8"))
            partner_id_single = h2.hexdigest()
        else:
            h2 = blake2b(digest_size=8); h2.update(sender.encode("utf-8"))
            partner_id_single = h2.hexdigest()

        source_meta = {
            "system": bool(m["system"]),
            "wa_format": "bracket" if RE_BRACKET else "dash",  # informational only
        }

        record = {
            "platform": "whatsapp",
            "source_file": m["source_file"],
            "conversation_id": conv_id,
            "participants": participants,
            "account": account if account else None,
            "message_id": message_id,
            "reply_to_message_id": reply_to,
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
            "reactions": [],  # not present in TXT exports
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
            "headers": {"reply_hint": bool(reply_hint)},
            "source_meta": source_meta,
        }

        if debug and idx < 3:
            print(" RAW :", (body_raw or "")[:160], file=sys.stderr)
            print(" TEXT:", (body_text or "")[:160], file=sys.stderr)

        out.append(record)
        produced += 1

    # majority-lang backfill
    langs = [r["lang"] for r in out if r["lang"] != "unknown"]
    if langs:
        majority = Counter(langs).most_common(1)[0][0]
        for r in out:
            if r["lang"] == "unknown":
                r["lang"] = majority

    return (out, produced)

# ============== Walk + CLI ==============

def find_txt_chats(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])

def parse_args():
    ap = argparse.ArgumentParser(description="Convert WhatsApp TXT exports to JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="Input folder containing .txt chats")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL file")
    ap.add_argument("--me", dest="me", required=True, help="Comma-separated list of your usernames/aliases")
    ap.add_argument("--tz", dest="tz", default=None, help='Timezone of timestamps in TXT, e.g. "+04:00". Default: local.')
    ap.add_argument("--limit", dest="limit", type=int, default=None, help="Stop after N messages total")
    ap.add_argument("--progress", action="store_true", help="Show progress with tqdm")
    ap.add_argument("--debug", action="store_true", help="Print a few cleaning samples")
    return ap.parse_args()

def main():
    args = parse_args()

    root = Path(args.inp).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tz_local = parse_tz(args.tz)
    me_set = set(normalize_name(x) for x in args.me.split(",") if x.strip())
    print(f"[DEBUG] TZ local={tz_local}, me={me_set}", file=sys.stderr)

    files = find_txt_chats(root)
    print(f"[DEBUG] Found {len(files)} .txt chats", file=sys.stderr)

    iterator = tqdm(files, desc="Chats", unit="chat") if (args.progress and tqdm is not None) else files

    records = []
    produced_total = 0

    for fp in iterator:
        limit_remaining = None if args.limit is None else max(0, args.limit - produced_total)
        if limit_remaining == 0:
            print(f"[DEBUG] Hit limit of {args.limit} messages, stopping", file=sys.stderr)
            break

        conv_records, produced = build_records_for_file(
            fp=fp,
            tz_local=tz_local,
            me_set=me_set,
            limit_remaining=limit_remaining,
            debug=args.debug,
        )
        records.extend(conv_records)
        produced_total += produced

    print(f"[DEBUG] Writing {len(records)} records to {out_path}", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(records)} messages to {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
