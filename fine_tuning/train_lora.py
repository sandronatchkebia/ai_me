#!/usr/bin/env python3
# train_lora.py
# LoRA/QLoRA fine-tuning for ai_me on chat-formatted DatasetDict saved by prepare_dataset.py

import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def build_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_model(model_id: str, load_4bit: bool = True):
    quant = None
    if load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
    )
    return model


def formatting_func_builder(tokenizer):
    """
    Returns a function that converts one dataset row -> rendered chat string
    using the model's chat template. Each row has:
      - messages: [{role, content}, ...]  (last is assistant target)
      - labels:   assistant text (not used here directly)
    """
    def fmt(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,  # assistant target is already included
        )
    return fmt


def build_response_template_ids(tokenizer):
    """
    Create token IDs for the assistant header so the collator can mask loss
    to ONLY the final assistant completion.
    Trick: render a single empty assistant message and use its prefix.
    """
    tmpl = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Tokenize without adding BOS/EOS; we want raw prefix ids
    ids = tokenizer(tmpl, add_special_tokens=False).input_ids
    return ids


def main():
    ap = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for ai_me.")
    ap.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset_dir", required=True, help="Path to DatasetDict saved by prepare_dataset.py")
    ap.add_argument("--output_dir", default="out/ai_me_lora")
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="*", default=[
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ])
    # Training
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=1.5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--load_4bit", action="store_true", default=True)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset
    dsd = load_from_disk(args.dataset_dir)
    print(f"Loaded dataset: train={len(dsd['train'])}, val={len(dsd['validation'])}")

    # 2) Tokenizer & response template (for loss masking)
    tokenizer = build_tokenizer(args.model_id)
    response_tmpl_ids = build_response_template_ids(tokenizer)

    # 3) Model + LoRA
    model = build_model(args.model_id, load_4bit=args.load_4bit)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )

    # 4) Collator to apply loss only on assistant completion
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_tmpl_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) Trainer
    sft_cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_len,
        packing=True,  # efficient packing of examples
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        report_to=[],  # set to ["wandb"] if you use it
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        formatting_func=lambda ex: [formatting_func_builder(tokenizer)(e) for e in ex],
        data_collator=collator,
        args=sft_cfg,
    )

    # 6) Train
    trainer.train()
    trainer.save_model()         # saves adapter weights
    tokenizer.save_pretrained(out)

    print(f"Done. LoRA adapter saved to: {out.resolve()}")


if __name__ == "__main__":
    main()