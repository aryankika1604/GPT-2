# GPT Model Implementation in PyTorch

A PyTorch implementation of the GPT (Generative Pre-trained Transformer) architecture from scratch, including distributed training support.

## Project Structure

```
GPT_torch/
├── GPTmodel_torch.py              # Main GPT model architecture
├── TransformerBlock.py            # Transformer block implementation
├── MultiHeadAttention.py          # Multi-head attention mechanism
├── FeedForward.py                 # Feed-forward network
├── LayerNorm.py                   # Layer normalization
├── Pre-TrainingLLM.py             # Main training script with DDP support
├── gpt_architecture_Data_Preparation.py  # Data preparation utilities
├── gpt_architecture_Embedding_Tokens.py  # Token embedding utilities
├── Scripts/
│   └── Pre-Train.sh               # SLURM training script
└── checkpoints/                   # Model checkpoints (not tracked)

## Features

- **GPT Architecture**: Complete implementation of GPT model with configurable parameters
- **Distributed Training**: Support for Distributed Data Parallel (DDP) training
- **Modular Design**: Separate modules for attention, feed-forward, and transformer blocks
- **Training Scripts**: Ready-to-use SLURM scripts for cluster training

## Model Configuration

The default configuration (GPT_CONFIG_124M) includes:
- Vocabulary size: 50,257
- Context length: 1,024
- Embedding dimension: 768
- Number of layers: 12
- Number of attention heads: 12
- Dropout: 0.1

## Requirements

- PyTorch
- tiktoken (for tokenization)
- plotly (optional, for visualization)

## Usage

### Training

```bash
# Using SLURM
sbatch Scripts/Pre-Train.sh

# Or run directly
python Pre-TrainingLLM.py
```

## Components

- **GPTModel**: Main model class implementing the GPT architecture
- **TransformerBlock**: Single transformer block with attention and feed-forward layers
- **MultiHeadAttention**: Multi-head self-attention mechanism
- **FeedForward**: Position-wise feed-forward network
- **LayerNorm**: Layer normalization implementation




