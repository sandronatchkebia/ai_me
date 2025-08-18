#!/usr/bin/env python3
# prepare_dataset.py
# Build chat-ready HF datasets from ai_me preprocessed JSONLs.

import json, random, argparse
from pathlib import Path
from typing import Union, List, Dict, Any
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# -----------------------
# I/O
# -----------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file with error handling and validation."""
    items = []
    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return items
    
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        # Basic validation
                        if isinstance(item, dict):
                            items.append(item)
                        else:
                            print(f"[WARNING] Skipping non-dict item at line {line_num}")
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] JSON decode error at line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return []
    
    print(f"[INFO] Loaded {len(items)} items from {path.name}")
    return items

def detect_files(folder: Path) -> Dict[str, List[Path]]:
    """Detect available files with language-based directory structure."""
    expected_files = {
        "pairs_train": [],
        "pairs_val":   [],
        "pairs_test":  [],
        "mono_train":  [],
        "mono_val":    [],
        "mono_test":   [],
    }
    
    # Look for language subdirectories
    language_dirs = [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not language_dirs:
        print(f"[WARNING] No language subdirectories found in {folder}")
        return {}
    
    print(f"[INFO] Found language directories: {[d.name for d in language_dirs]}")
    
    for lang_dir in language_dirs:
        for split_name in expected_files.keys():
            file_path = lang_dir / f"{split_name}.jsonl"
            if file_path.exists():
                expected_files[split_name].append(file_path)
                print(f"[INFO] Found {split_name} in {lang_dir.name}")
            else:
                print(f"[WARNING] Missing {split_name} in {lang_dir.name}")
    
    return expected_files

def load_jsonl_multiple(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load multiple JSONL files and combine them."""
    all_items = []
    for path in paths:
        items = load_jsonl(path)
        all_items.extend(items)
    return all_items

# -----------------------
# Data validation
# -----------------------
def validate_pair_example(ex: Dict[str, Any]) -> bool:
    """Validate that a pair example has the expected structure."""
    required_fields = ["context", "target_text", "meta"]
    if not all(field in ex for field in required_fields):
        return False
    
    if not isinstance(ex["context"], list) or len(ex["context"]) == 0:
        return False
    
    if not ex["target_text"] or not isinstance(ex["target_text"], str):
        return False
    
    # Check for clean content (no obvious artifacts)
    target_text = ex["target_text"].strip()
    if len(target_text) < 3:  # Too short
        return False
    
    # Check for common artifacts that might have slipped through
    artifacts = ["virus-free", "www.avast.com", "sent from my", "get outlook for"]
    if any(artifact.lower() in target_text.lower() for artifact in artifacts):
        return False
    
    return True

def validate_mono_example(ex: Dict[str, Any]) -> bool:
    """Validate that a mono example has the expected structure."""
    if "text" not in ex:
        return False
    
    text = ex["text"].strip()
    if len(text) < 3:  # Too short
        return False
    
    # Check for artifacts
    artifacts = ["virus-free", "www.avast.com", "sent from my", "get outlook for"]
    if any(artifact.lower() in text.lower() for artifact in artifacts):
        return False
    
    return True

# -----------------------
# Mapping to chat format
# -----------------------
def to_chat_from_pair(ex: Dict[str, Any], drop_short_targets: int = 0, short_min: int = 1e9) -> Union[Dict[str, Any], None]:
    """
    Convert a 'pair' example into chat messages.
    We keep full context and set your line as the assistant label.
    """
    # Validate example first
    if not validate_pair_example(ex):
        return None
    
    # Optional guard: drop ultra-short targets if requested
    if drop_short_targets and len(ex["meta"].get("len_chars", 0)) < short_min:
        return None

    msgs = []
    for turn in ex["context"]:
        role = "assistant" if turn["role"] == "you" else "user"
        msgs.append({"role": role, "content": turn["text"]})
    # Target (your reply)
    msgs.append({"role": "assistant", "content": ex["target_text"]})

    return {"messages": msgs, "labels": ex["target_text"]}

def to_chat_from_mono(ex: Dict[str, Any], style_prompt: str = None) -> Union[Dict[str, Any], None]:
    """
    Convert a 'mono' sample into a single-turn assistant answer with a light cue.
    """
    # Validate example first
    if not validate_mono_example(ex):
        return None
    
    system = style_prompt or "You respond concisely in Aleks' natural tone."
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "(Write a short message continuing the thread.)"},
        {"role": "assistant", "content": ex["text"]},
    ]
    return {"messages": msgs, "labels": ex["text"]}

# -----------------------
# Mixing
# -----------------------
def weighted_mix(pairs: List[Dict[str, Any]], mono: List[Dict[str, Any]], 
                pair_weight: int = 3, mono_weight: int = 1, 
                max_train: int = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Make a train set with the requested ratio (pairs:mono).
    We sample without replacement up to availability.
    """
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rng.shuffle(mono)

    if not pairs and not mono:
        return []

    # Decide target sizes
    if max_train:
        total = max_train
    else:
        # default: use as many as available respecting ratio
        total = len(pairs) + len(mono)

    if pair_weight + mono_weight == 0:
        pair_target = min(len(pairs), total)
        mono_target = max(0, total - pair_target)
    else:
        pair_target = int(total * (pair_weight / (pair_weight + mono_weight)))
        mono_target = total - pair_target

    # Cap by availability
    pair_target = min(pair_target, len(pairs))
    mono_target = min(mono_target, len(mono))

    mixed = pairs[:pair_target] + mono[:mono_target]
    rng.shuffle(mixed)
    return mixed

# -----------------------
# Build DatasetDict
# -----------------------
def build_dataset_dict(
    pre_dir: Path,
    pair_weight: int,
    mono_weight: int,
    max_train: Union[int, None],
    style_prompt: Union[str, None],
    seed: int
) -> DatasetDict:
    files = detect_files(pre_dir)
    
    if not files:
        print("[ERROR] No valid files found!")
        return DatasetDict({"train": Dataset.from_list([]), "validation": Dataset.from_list([])})

    # Load raw with progress bars
    print("[INFO] Loading data files...")
    pairs_train = load_jsonl_multiple(files.get("pairs_train", []))
    pairs_val   = load_jsonl_multiple(files.get("pairs_val", []))
    mono_train  = load_jsonl_multiple(files.get("mono_train", []))
    mono_val    = load_jsonl_multiple(files.get("mono_val", []))

    # Map to chat with progress bars and validation
    print("[INFO] Converting pairs to chat format...")
    pairs_train_m = [to_chat_from_pair(x) for x in tqdm(pairs_train, desc="Pairs train")]
    pairs_val_m   = [to_chat_from_pair(x) for x in tqdm(pairs_val, desc="Pairs val")]
    
    print("[INFO] Converting mono to chat format...")
    mono_train_m  = [to_chat_from_mono(x, style_prompt) for x in tqdm(mono_train, desc="Mono train")]
    mono_val_m    = [to_chat_from_mono(x, style_prompt) for x in tqdm(mono_val, desc="Mono val")]

    # Drop Nones (safety)
    pairs_train_m = [x for x in pairs_train_m if x]
    pairs_val_m   = [x for x in pairs_val_m if x]
    mono_train_m  = [x for x in mono_train_m if x]
    mono_val_m    = [x for x in mono_val_m if x]

    print(f"[INFO] Valid examples: {len(pairs_train_m)} pairs train, {len(mono_train_m)} mono train")
    print(f"[INFO] Valid examples: {len(pairs_val_m)} pairs val, {len(mono_val_m)} mono val")

    # Mix splits
    train_mixed = weighted_mix(
        pairs_train_m, mono_train_m,
        pair_weight=pair_weight,
        mono_weight=mono_weight,
        max_train=max_train,
        seed=seed
    )
    # For val, keep the natural distribution (don't overfit to ratio)
    val_mixed = pairs_val_m + mono_val_m
    random.Random(seed).shuffle(val_mixed)

    ds_train = Dataset.from_list(train_mixed) if train_mixed else Dataset.from_list([])
    ds_val   = Dataset.from_list(val_mixed)   if val_mixed   else Dataset.from_list([])

    return DatasetDict({"train": ds_train, "validation": ds_val})

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare HF datasets from ai_me preprocessed JSONLs.")
    ap.add_argument("--preprocessed_dir", required=True, help="Folder with pairs_*.jsonl and mono_*.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output folder for DatasetDict (save_to_disk)")
    ap.add_argument("--pair_weight", type=int, default=3, help="Relative weight for pairs in train mix")
    ap.add_argument("--mono_weight", type=int, default=1, help="Relative weight for mono in train mix")
    ap.add_argument("--max_train", type=int, default=None, help="Optional cap on total train samples")
    ap.add_argument("--style_prompt", type=str, default="You respond concisely in Aleks' natural tone.",
                    help="System prompt used for mono samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pre_dir = Path(args.preprocessed_dir)
    out_dir = Path(args.out_dir)
    
    if not pre_dir.exists():
        print(f"[ERROR] Preprocessed directory not found: {pre_dir}")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing data from: {pre_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Weights: pairs={args.pair_weight}, mono={args.mono_weight}")
    if args.max_train:
        print(f"[INFO] Max train samples: {args.max_train}")

    dsd = build_dataset_dict(
        pre_dir=pre_dir,
        pair_weight=args.pair_weight,
        mono_weight=args.mono_weight,
        max_train=args.max_train,
        style_prompt=args.style_prompt,
        seed=args.seed
    )

    # Quick stats
    n_train = len(dsd["train"])
    n_val = len(dsd["validation"])
    print(f"\n[OK] Final dataset sizes:")
    print(f"  Train: {n_train:,}")
    print(f"  Validation: {n_val:,}")
    print(f"  Total: {n_train + n_val:,}")
    
    if n_train:
        ex = dsd["train"][0]
        print("\n[Sample: train]")
        print(json.dumps(ex, indent=2, ensure_ascii=False))
    if n_val:
        exv = dsd["validation"][0]
        print("\n[Sample: val]")
        print(json.dumps(exv, indent=2, ensure_ascii=False))

    dsd.save_to_disk(str(out_dir))
    print(f"\n[SAVED] DatasetDict -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()