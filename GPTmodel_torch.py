import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from LayerNorm import LayerNorm

class GPTModel(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['dropout'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range (cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, sequence_length = in_idx.shape
        token_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(sequence_length, device=in_idx.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits




 

