import torch
import torch.nn as nn 
from MultiHeadAttention import MultiHeadAttention, FlashAttention
from LayerNorm import LayerNorm
from FeedForward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg, Flash_Attn = True):
        super().__init__()
        if Flash_Attn:
            self.attn = FlashAttention(
                d_in = cfg['emb_dim'],
                d_out = cfg["emb_dim"],
                context_len = cfg['context_length'],
                num_heads = cfg['n_heads'],
                dropout = cfg['dropout'],
                qkv_bias=cfg['qkv_bias']
            )
        else:
            self.attn = MultiHeadAttention(
                d_in = cfg['emb_dim'],
                d_out = cfg["emb_dim"],
                context_len = cfg['context_length'],
                num_heads = cfg['n_heads'],
                dropout = cfg['dropout'],
                qkv_bias=cfg['qkv_bias']
            )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['dropout'])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
