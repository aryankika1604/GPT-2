import torch
import torch.nn as nn
import torch.nn.functional as F
# --- try Dao's flash-attn; fallback to PyTorch Flash SDP ---
try:
    # flash-attn v2 API
    from flash_attn.flash_attn_interface import flash_attn_func  # ✅ DAO path
    HAVE_DAO_FLASH = True
except Exception:
    HAVE_DAO_FLASH = False

# ✅ enable fast kernels when using PyTorch Flash SDP
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    # PyTorch 2.x Flash SDP toggle
    from torch.backends.cuda import sdp_kernel
    HAVE_SDP = True
except Exception:
    HAVE_SDP = False
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout,num_heads, qkv_bias = False):
        super().__init__()

        #Sanity Check: Asserting that the output dimension is divisible by the number of heads
        assert (d_out % num_heads == 0),\
            "d_out must be divisible by num_heads"

        self.num_heads = num_heads

        # Calculating the head dimension d_out//num_heads
        self.head_dim = d_out // num_heads

        # Setting the output dimension d_out
        self.d_out = d_out
        self.dropout = torch.nn.Dropout(dropout)
        self.W_q = torch.nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_k =  torch.nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_v = torch.nn.Linear(d_in,d_out,bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal = 1))

    def forward(self,x):
        b, num_token, d_in = x.shape # d_in = 768

        # Calculating the queries, keys and values matrices => shape: [b, num_token, d_out]
        queries = self.W_q(x) 
        keys=  self.W_k(x)
        values = self.W_v(x)

        # Unrolling the last dim d_out into num_heads and head_dim => d_out = num_heads * head_dim => shape: [b, num_tokens, num_heads, head_dim]
        queries = queries.view(b,num_token,self.num_heads, self.head_dim) # Grouped by num_tokens
        keys = keys.view(b,num_token,self.num_heads, self.head_dim)
        values = values.view(b, num_token, self.num_heads, self.head_dim)

        # Tranpsoing the num_heads and num_token to group by num_heads
        queries =queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # Calculating the attention scores => shape: [b, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2,3)
        d_k = keys.shape[-1]

        # Applying the mask to the attention scores
        mask_bool = self.mask.bool()[:num_token, :num_token]

        attn_scores.masked_fill_(mask_bool,float('-inf'))

        # Calculating the attention weights => shape: [b, num_heads, num_tokens, num_tokens]
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim= -1)

        # Calculating the context vectors => shape: [b, num_heads, num_tokens, head_dim]
        context_vec = (attn_weights @ values).transpose(1,2) 

        # Combining the num_heads and head_dim back into d_out => shape: [b, num_token, d_out]
        context_vec = context_vec.contiguous().view(b, num_token,self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec

class FlashAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.d_out     = d_out
        self.dropout   = nn.Dropout(dropout)  # holds .p

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        # TF32 toggles (help other matmuls)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def forward(self, x):
        B, L, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # [B,L,H,Hd] -> [B,H,L,Hd]
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()

        p = self.dropout.p if self.training else 0.0  # ✅ float, not module

        # safer while iterating (allows fallback if dtype isn't bf16/fp16)
        with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=p,
                is_causal=True,
            )  # [B,H,L,Hd]

        out = attn.transpose(1, 2).contiguous().view(B, L, self.d_out)
        out = self.out_proj(out)
        out = self.dropout(out)  # optional residual dropout
        return out
