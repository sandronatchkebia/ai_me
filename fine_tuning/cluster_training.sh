#!/bin/bash
# Cluster Training Script for ai_me LoRA Fine-tuning
# This script can be adapted for different cluster systems (SLURM, PBS, etc.)

# =============================================================================
# CONFIGURATION - Modify these variables for your cluster
# =============================================================================

# Job configuration
JOB_NAME="ai_me_lora_training"
NODES=1
GPUS_PER_NODE=4
CPUS_PER_GPU=8
MEM_PER_GPU="32G"
TIME_LIMIT="24:00:00"

# Model and training configuration
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"  # Change to your preferred model
DATASET_DIR="fine_tuning/dataset/ai_me_chat"
OUTPUT_DIR="fine_tuning/out/ai_me_lora_$(date +%Y%m%d_%H%M%S)"

# LoRA configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Training configuration
EPOCHS=2.0
LEARNING_RATE=1.5e-4
BATCH_SIZE=2  # Per GPU batch size
GRADIENT_ACCUMULATION=8
MAX_SEQ_LEN=2048

# Quantization
LOAD_4BIT=true
BF16=true

# =============================================================================
# SLURM JOB SCRIPT (uncomment and modify for SLURM clusters)
# =============================================================================

cat << 'EOF' > slurm_job.sh
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_GPU}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --mem=${MEM_PER_GPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --partition=gpu  # Change to your partition name

# Load modules (modify for your cluster)
module load cuda/11.8
module load python/3.10
module load anaconda3

# Activate conda environment
source activate ai_me_env  # Change to your environment name

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export HF_HOME="/tmp/hf_home"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"

# Run training
python fine_tuning/train_lora.py \
    --model_id ${MODEL_ID} \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 20 \
    --save_total_limit 3 \
    --load_4bit \
    --bf16 \
    --ddp_find_unused_parameters \
    --report_to wandb

echo "Training completed successfully!"
EOF

# =============================================================================
# PBS JOB SCRIPT (uncomment and modify for PBS clusters)
# =============================================================================

cat << 'EOF' > pbs_job.sh
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -l nodes=${NODES}:ppn=${GPUS_PER_NODE}:gpus=${GPUS_PER_NODE}
#PBS -l mem=${MEM_PER_GPU}
#PBS -l walltime=${TIME_LIMIT}
#PBS -o logs/${JOB_NAME}.out
#PBS -e logs/${JOB_NAME}.err
#PBS -q gpu  # Change to your queue name

# Load modules (modify for your cluster)
module load cuda/11.8
module load python/3.10
module load anaconda3

# Activate conda environment
source activate ai_me_env  # Change to your environment name

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export HF_HOME="/tmp/hf_home"

# Print job info
echo "Job ID: $PBS_JOBID"
echo "Node: $PBS_NODEFILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"

# Run training
python fine_tuning/train_lora.py \
    --model_id ${MODEL_ID} \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 20 \
    --save_total_limit 3 \
    --load_4bit \
    --bf16 \
    --ddp_find_unused_parameters \
    --report_to wandb

echo "Training completed successfully!"
EOF

# =============================================================================
# DIRECT EXECUTION (for testing or single-node clusters)
# =============================================================================

cat << 'EOF' > run_training.sh
#!/bin/bash
# Direct execution script for single-node training

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="/tmp/transformers_cache"
export HF_HOME="/tmp/hf_home"

# Print configuration
echo "Model: ${MODEL_ID}"
echo "Dataset: ${DATASET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS_PER_NODE}"

# Run training
python fine_tuning/train_lora.py \
    --model_id ${MODEL_ID} \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 20 \
    --save_total_limit 3 \
    --load_4bit \
    --bf16 \
    --report_to wandb

echo "Training completed successfully!"
EOF

# Make scripts executable
chmod +x slurm_job.sh pbs_job.sh run_training.sh

echo "Cluster training scripts created:"
echo "  - slurm_job.sh (for SLURM clusters)"
echo "  - pbs_job.sh (for PBS clusters)"
echo "  - run_training.sh (for direct execution)"
echo ""
echo "To submit jobs:"
echo "  SLURM: sbatch slurm_job.sh"
echo "  PBS: qsub pbs_job.sh"
echo "  Direct: ./run_training.sh"
echo ""
echo "Remember to:"
echo "  1. Modify configuration variables at the top"
echo "  2. Adjust module loads for your cluster"
echo "  3. Set correct partition/queue names"
echo "  4. Install dependencies: pip install -r requirements.txt"
