#!/usr/bin/env python3
# train_lora.py
# LoRA/QLoRA fine-tuning for ai_me on chat-formatted DatasetDict saved by prepare_dataset.py
# Optimized for GPU cluster training with distributed training support

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from accelerate import Accelerator
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def setup_accelerator():
    """Setup accelerator for distributed training."""
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,  # Will be set in training args
    )
    return accelerator


def build_tokenizer(model_id: str, trust_remote_code: bool = True):
    """Build and configure tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            use_fast=True,
            trust_remote_code=trust_remote_code
        )
        
        # Handle padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "</s>"
        
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        
        logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, pad_token={tokenizer.pad_token}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def build_model(
    model_id: str, 
    load_4bit: bool = True,
    load_8bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: str = "auto"
):
    """Build model with quantization if requested."""
    try:
        quantization_config = None
        
        if load_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Using 4-bit quantization")
        elif load_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Using 8-bit quantization")
        
        # Set default dtype if not specified
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {model.config.model_type}, {param_count:,} parameters")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
):
    """Create LoRA configuration."""
    if target_modules is None:
        # Default target modules for common architectures
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3",  # For some models
        ]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=getattr(TaskType, task_type),
        inference_mode=False,
    )
    
    logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    return config


def prepare_model_for_training(model, lora_config, load_4bit: bool = False):
    """Prepare model for training with LoRA."""
    try:
        if load_4bit:
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for 4-bit training")
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to prepare model for training: {e}")
        raise


def create_data_collator(tokenizer, max_length: int = 2048):
    """Create data collator for training."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )


def format_chat_example(example, tokenizer):
    """Format chat example for training."""
    try:
        messages = example["messages"]
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": formatted}
    except Exception as e:
        logger.warning(f"Failed to format example: {e}")
        return None


def create_training_arguments(
    output_dir: str,
    num_train_epochs: float = 2.0,
    learning_rate: float = 1.5e-4,
    warmup_ratio: float = 0.02,
    weight_decay: float = 0.05,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    eval_steps: int = 200,
    save_steps: int = 200,
    logging_steps: int = 20,
    save_total_limit: int = 3,
    bf16: bool = True,
    dataloader_pin_memory: bool = True,
    remove_unused_columns: bool = False,
    report_to: Optional[List[str]] = None,
    ddp_find_unused_parameters: bool = False,
):
    """Create training arguments with backward-compatible kwargs filtering."""
    if report_to is None:
        report_to = []

    # Construct kwargs and filter by TrainingArguments signature for compatibility
    import inspect
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        bf16=bf16,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=remove_unused_columns,
        report_to=report_to,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{output_dir}/logs",
        run_name="ai_me_lora_training",
        dataloader_num_workers=4,
        group_by_length=True,
        save_safetensors=True,
    )
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
    return TrainingArguments(**filtered_kwargs)


def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for ai_me on GPU clusters.")
    
    # Model and data
    parser.add_argument("--model_id", default="meta-llama/Llama-2-7b-chat-hf", 
                       help="HuggingFace model ID")
    parser.add_argument("--dataset_dir", required=True, 
                       help="Path to DatasetDict saved by prepare_dataset.py")
    parser.add_argument("--output_dir", default="out/ai_me_lora", 
                       help="Output directory for checkpoints and final model")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                       help="LoRA dropout")
    parser.add_argument("--target_modules", nargs="*", 
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help="Target modules for LoRA")
    
    # Training configuration
    parser.add_argument("--epochs", type=float, default=2.0, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, 
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, 
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.05, 
                       help="Weight decay")
    parser.add_argument("--max_seq_len", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                       help="Per device training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, 
                       help="Per device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                       help="Gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=200, 
                       help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=200, 
                       help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=20, 
                       help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=3, 
                       help="Maximum number of checkpoints to keep")
    
    # Quantization and precision
    parser.add_argument("--load_4bit", action="store_true", 
                       help="Load model in 4-bit precision")
    parser.add_argument("--load_8bit", action="store_true", 
                       help="Load model in 8-bit precision")
    parser.add_argument("--bf16", action="store_true", default=True, 
                       help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", 
                       help="Use float16 precision")
    
    # Distributed training
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", 
                       help="Find unused parameters in DDP")
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="Local rank for distributed training")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, 
                       help="Trust remote code from model")
    parser.add_argument("--report_to", nargs="*", default=[], 
                       help="Report to (e.g., wandb, tensorboard)")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup accelerator
    accelerator = setup_accelerator()
    
    logger.info(f"Starting training with model: {args.model_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {accelerator.device}")
    
    try:
        # 1) Load dataset
        logger.info("Loading dataset...")
        dsd = load_from_disk(args.dataset_dir)
        logger.info(f"Dataset loaded: train={len(dsd['train']):,}, val={len(dsd['validation']):,}")
        
        # 2) Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = build_tokenizer(args.model_id, args.trust_remote_code)
        # Respect CLI sequence length even if TrainingArguments lacks max_seq_length
        try:
            tokenizer.model_max_length = args.max_seq_len
        except Exception:
            pass
        
        # 3) Load model
        logger.info("Loading model...")
        torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        model = build_model(
            args.model_id, 
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
            torch_dtype=torch_dtype
        )
        
        # 4) Create LoRA config and prepare model
        logger.info("Setting up LoRA...")
        lora_config = create_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        model = prepare_model_for_training(model, lora_config, args.load_4bit)
        
        # 5) Format dataset
        logger.info("Formatting dataset...")
        def format_dataset(examples):
            messages_batch = examples["messages"]
            texts = []
            for messages in messages_batch:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(formatted)
            return {"text": texts}
        
        train_dataset = dsd["train"].map(
            format_dataset,
            batched=True,
            batch_size=1000,
            remove_columns=dsd["train"].column_names,
            desc="Formatting training data"
        )
        
        eval_dataset = dsd["validation"].map(
            format_dataset,
            batched=True,
            batch_size=1000,
            remove_columns=dsd["validation"].column_names,
            desc="Formatting validation data"
        )
        
        # 6) Create data collator
        data_collator = create_data_collator(tokenizer, args.max_seq_len)
        
        # 7) Create training arguments
        training_args = create_training_arguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_seq_length=args.max_seq_len,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            bf16=args.bf16,
            report_to=args.report_to,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        )
        
        # 8) Create trainer
        logger.info("Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # 9) Train
        logger.info("Starting training...")
        trainer.train()
        
        # 10) Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training config
        training_args.save_to_json(output_dir / "training_args.json")
        
        logger.info(f"Training completed! Model saved to: {output_dir.resolve()}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()