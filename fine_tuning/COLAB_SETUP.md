# Google Colab Setup for LoRA Training

This guide explains how to set up and run LoRA fine-tuning on Google Colab.

## Prerequisites

1. **Google Colab Account** with GPU runtime access
2. **Hugging Face Account** with access to Llama models
3. **Prepared Dataset** from `prepare_dataset.py`

## Quick Start

1. **Open the Colab notebook**: `run_lora_colab.ipynb`
2. **Set your HF API key** in Colab secrets
3. **Upload your dataset** when prompted
4. **Run all cells** to start training

## Detailed Setup

### 1. Set Hugging Face API Key

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. In Colab: Runtime → Manage secrets
4. Add secret with key: `hf_key`, value: your token

### 2. Prepare Dataset

Your dataset should be prepared using `prepare_dataset.py` and contain:
- `train/` split with chat-formatted conversations
- `validation/` split for evaluation
- Proper tokenization and formatting

### 3. GPU Requirements

- **T4 (16GB)**: Basic training, may need smaller batch sizes
- **V100 (16GB)**: Good performance, stable training
- **A100 (40GB)**: Best performance, larger batch sizes possible

### 4. Training Parameters

The notebook uses these optimized settings:
- **Model**: Meta-Llama-3.1-8B-Instruct
- **Method**: LoRA with 4-bit quantization
- **Epochs**: 2.0
- **Learning Rate**: 1.5e-4
- **Sequence Length**: 2048 tokens
- **Batch Size**: 2 per device, 8 gradient accumulation steps

## Expected Training Time

- **T4**: ~8-12 hours
- **V100**: ~6-8 hours  
- **A100**: ~4-6 hours

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_seq_len` or batch size
2. **HF Login Failed**: Check your API key in Colab secrets
3. **Dataset Loading Error**: Ensure dataset format matches expected structure

### Performance Tips

1. **Use A100 if available** for fastest training
2. **Monitor GPU memory** with `!nvidia-smi`
3. **Adjust batch sizes** based on your GPU memory
4. **Use gradient checkpointing** for memory efficiency

## Output

After training, you'll get:
- Fine-tuned LoRA weights
- Training logs and metrics
- Model configuration files
- Training arguments summary

## Next Steps

1. **Test the model** with inference
2. **Merge LoRA weights** if needed
3. **Deploy** to your preferred platform
4. **Share** your fine-tuned model on Hugging Face Hub

## Support

For issues or questions:
1. Check the training logs
2. Verify your setup matches requirements
3. Review the main README.md
4. Open an issue on GitHub
