#!/bin/bash
#SBATCH --job-name=Pre-Train-LLM
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --time=01-00:00:00
#SBATCH --mem=64gb
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:4
#SBATCH --export=ALL

# Activate conda environment
source /etc/profile
eval "$(conda shell.bash hook)"
conda activate LLM

# Go to project
cd /mathbiospace/data01/a/a.kikaganeshwala001/LLM/GPT_torch
mkdir -p checkpoints

# Run with DDP across 3 GPUs
torchrun --nproc_per_node=4 Pre-TrainingLLM.py