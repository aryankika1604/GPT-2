import torch
from gpt_architecture_Data_Preparation import *

def embedding_tokens(token_ids, vocab_size=50257, embedding_size = 256):
    """
    Create token embeddings with positional encoding for GPT architecture.
    
    Args:
        token_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_length]
        vocab_size (int): Size of vocabulary (default: 50257 for GPT-2)
        embedding_size (int): Dimension of embedding vectors (default: 256)
        max_length (int): Maximum sequence length for positional embeddings
    """

    batch_size = token_ids.shape[0] # No of input sequences given at a time
    seq_length = token_ids.shape[1] # No of tokens in each input sequence

    embedding_layer = torch.nn.Embedding(vocab_size,embedding_size)

    token_embeddings = embedding_layer(token_ids)

    positional_embedding_layer = torch.nn.Embedding(seq_length, embedding_size)
    positional_embeddings = positional_embedding_layer(torch.arange(seq_length, device=token_ids.device))

    input_embeddings = token_embeddings + positional_embeddings

    return input_embeddings




