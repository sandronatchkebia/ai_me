# AI Me Fine-tuning Guide

This directory contains scripts for fine-tuning language models on the ai_me dataset using LoRA (Low-Rank Adaptation) on GPU clusters.

## 🚀 Quick Start

### 1. Prepare Your Dataset
First, ensure you have a prepared dataset from `prepare_dataset.py`:
```bash
python prepare_dataset.py \
    --preprocessed_dir ../data/preprocessed \
    --out_dir dataset/ai_me_chat \
    --pair_weight 3 \
    --mono_weight 1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training
```bash
python train_lora.py \
    --model_id "meta-llama/Llama-2-7b-chat-hf" \
    --dataset_dir dataset/ai_me_chat \
    --output_dir out/ai_me_lora \
    --epochs 2.0 \
    --learning_rate 1.5e-4 \
    --load_4bit \
    --bf16
```

## 🏗️ Architecture Overview

### LoRA Configuration
- **Rank (r)**: 16 (default) - Controls the complexity of adaptation
- **Alpha**: 32 (default) - Scaling factor for LoRA weights
- **Dropout**: 0.05 (default) - Regularization during training
- **Target Modules**: Attention and MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

### Training Configuration
- **Batch Size**: 2 per GPU (adjustable)
- **Gradient Accumulation**: 8 steps (effective batch size = 2 × 8 × num_gpus)
- **Sequence Length**: 2048 tokens
- **Precision**: bfloat16 (recommended for modern GPUs)
- **Quantization**: 4-bit (QLoRA) for memory efficiency

## 🖥️ GPU Cluster Setup

### Supported Cluster Systems
- **SLURM**: Use `slurm_job.sh` template
- **PBS**: Use `pbs_job.sh` template
- **Direct**: Use `run_training.sh` for single-node execution

### Resource Requirements
- **GPU Memory**: 24GB+ per GPU (for 7B models with 4-bit quantization)
- **System Memory**: 32GB+ per node
- **Storage**: 100GB+ for dataset and checkpoints
- **Network**: High-speed interconnect for multi-node training

### Environment Setup
```bash
# Create conda environment
conda create -n ai_me_env python=3.10
conda activate ai_me_env

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## 📊 Training Configuration

### Model Selection
| Model | Parameters | GPU Memory (4-bit) | Recommended GPUs |
|-------|------------|-------------------|------------------|
| Llama-2-7b | 7B | 16GB | 1-2x A100 |
| Llama-2-13b | 13B | 24GB | 2-4x A100 |
| Llama-2-70b | 70B | 80GB | 8x A100 |

### Hyperparameter Tuning
```bash
# Conservative training (stable)
python train_lora.py \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01

# Aggressive training (faster convergence)
python train_lora.py \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.1
```

### Multi-GPU Training
```bash
# Enable distributed training
torchrun --nproc_per_node=4 train_lora.py \
    --model_id "meta-llama/Llama-2-7b-chat-hf" \
    --dataset_dir dataset/ai_me_chat \
    --output_dir out/ai_me_lora \
    --ddp_find_unused_parameters
```

## 📈 Monitoring and Logging

### Weights & Biases Integration
```bash
# Enable W&B logging
python train_lora.py \
    --report_to wandb \
    --run_name "ai_me_lora_experiment_001"
```

### TensorBoard Integration
```bash
# Enable TensorBoard logging
python train_lora.py \
    --report_to tensorboard \
    --logging_dir logs/tensorboard
```

### Checkpoint Management
- **Save Strategy**: Every 200 steps
- **Save Total Limit**: 3 checkpoints (prevents disk overflow)
- **Best Model**: Automatically loads best model at end of training

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 16

# Use 4-bit quantization
--load_4bit
```

#### Slow Training
```bash
# Enable Flash Attention
--attn_implementation flash_attention_2

# Increase batch size if memory allows
--per_device_train_batch_size 4

# Reduce sequence length
--max_seq_len 1024
```

#### Convergence Issues
```bash
# Lower learning rate
--learning_rate 5e-5

# Increase warmup
--warmup_ratio 0.1

# Adjust LoRA rank
--lora_r 32
```

### Performance Optimization
- **Flash Attention**: Automatically enabled for compatible models
- **Gradient Checkpointing**: Built into the training loop
- **Mixed Precision**: bfloat16 for optimal performance
- **Data Loading**: Multi-worker dataloaders with pin memory

## 📁 Output Structure

After training, your output directory will contain:
```
out/ai_me_lora/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── training_args.json           # Training arguments
├── tokenizer/                   # Tokenizer files
├── logs/                        # Training logs
└── checkpoints/                 # Training checkpoints
```

## 🚀 Deployment

### Loading Fine-tuned Model
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "out/ai_me_lora")
```

### Inference
```python
# Format input
messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

# Generate response
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📚 Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Training Guide](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review training logs in the output directory
3. Verify your cluster configuration
4. Ensure all dependencies are properly installed

Happy training! 🎉
