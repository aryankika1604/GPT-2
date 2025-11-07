import sys
sys.path.append('/mathbiospace/data01/a/a.kikaganeshwala/LLM/GPT_torch')
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from GPTmodel_torch import GPTModel
import tiktoken
from gpt_architecture_Data_Preparation import create_dataloader_v1
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False

# ----------------- Config -----------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "dropout": 0.1,
    "qkv_bias": False,
}

# ----------------- Setup DDP -----------------
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ----------------- Forward + Loss -----------------
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        probabs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probabs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
        

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ----------------- Training Loop -----------------
from torch.cuda.amp import autocast, GradScaler   # ✅ FIXED: correct AMP import

def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer,
                       train_sampler=None, val_sampler=None, rank: int = 0,
                       accumulation_steps: int = 4):   # ✅ NEW: gradient accumulation
    train_losses, val_losses, tracked_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    scaler = GradScaler()   # ✅ FIXED: AMP scaler

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        model.train()

        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            # ✅ AMP autocast
            with autocast(dtype=torch.bfloat16):  
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / accumulation_steps   # ✅ Normalize for gradient accumulation

            scaler.scale(loss).backward()

            # ✅ Update only every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tracked_tokens_seen.append(tokens_seen)
                if rank == 0:
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train Loss: {train_loss:.3f}, "
                          f"Val Loss: {val_loss:.3f}")

        if rank == 0:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, tracked_tokens_seen


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    context_size = base_model.pos_emb.weight.shape[0]
    encoded_ids = tokenizer.encode(start_context)
    idx = torch.tensor(encoded_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=idx, max_new_tokens=50, context_size=context_size)
    decoded_text = tokenizer.decode(token_ids[0].tolist())
    print(decoded_text.replace('\n', ' '))
    model.train()


# ----------------- Plot -----------------
def plot_train_val_losses(tokens_seen, train_losses, val_losses, out_html_path=None):
    if not _PLOTLY_AVAILABLE:
        print("Plotly not available; skipping interactive loss plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tokens_seen, y=train_losses, mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=tokens_seen, y=val_losses, mode='lines+markers', name='Val Loss'))
    fig.update_layout(title='Training vs Validation Loss', xaxis_title='Tokens Seen', yaxis_title='Loss', template='plotly_white')
    if out_html_path:
        fig.write_html(out_html_path, include_plotlyjs='cdn')
        print(f"Saved Plotly loss curve to: {out_html_path}")
    else:
        fig.show()


# ----------------- Main -----------------
file_path = '/mathbiospace/data01/a/a.kikaganeshwala001/LLM/Harry-Potter.txt'
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

print('Total Characters: ', len(text_data))
tokenizer = tiktoken.get_encoding("gpt2")
total_tokens = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
print('Total Tokens: ', len(total_tokens))

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data, batch_size=8,   # ✅ REDUCED batch size
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    shuffle=True, drop_last=True, num_workers=0)

val_loader = create_dataloader_v1(
    val_data, batch_size=8,   # ✅ REDUCED batch size
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    shuffle=False, drop_last=False, num_workers=0)

# ----------------- DDP init -----------------
is_ddp = "LOCAL_RANK" in os.environ
if is_ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M).to(device)
if is_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 100
rank = dist.get_rank() if is_ddp else 0
if rank == 0:
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Model device: {next(model.parameters()).device}")

train_sampler = DistributedSampler(train_loader.dataset) if is_ddp else None
val_sampler = DistributedSampler(val_loader.dataset, shuffle=False) if is_ddp else None
train_loader = DataLoader(train_loader.dataset, batch_size=8, sampler=train_sampler if is_ddp else None,
                          shuffle=(not is_ddp), drop_last=True, num_workers=4)
val_loader = DataLoader(val_loader.dataset, batch_size=8, sampler=val_sampler if is_ddp else None,
                        shuffle=False, drop_last=False, num_workers=4)

train_losses, val_losses, tracked_tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=len(train_loader), eval_iter=50,
    start_context='The boy who lived', tokenizer=tokenizer,
    train_sampler=train_sampler, val_sampler=val_sampler, rank=rank,
    accumulation_steps=4   # ✅ GRAD ACCUM (8*4 = effective 32 batch)
)

if rank == 0:
    default_html = '/mathbiospace/data01/a/a.kikaganeshwala001/LLM/GPT_torch/train_val_loss.html' if 'SLURM_JOB_ID' in os.environ else None
    plot_train_val_losses(tracked_tokens_seen, train_losses, val_losses, out_html_path=default_html)

if is_ddp:
    dist.destroy_process_group()

