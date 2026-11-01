"""
GPT-2 to gpt-oss, One Change at a Time

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fetch_stories
import os
import tempfile
import urllib.request

STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"


def fetch_stories(max_chars):
    # TODO: download to the temp cache once, read, return the first max_chars characters
    path = os.path.join(tempfile.gettempdir(), "gptoss_tinystories_valid.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(STORIES_URL, path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars)

# Step 2 - CharTokenizer
import torch


class CharTokenizer:
    def __init__(self, text):
        self.itos = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
       
    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return torch.tensor([self.stoi[ch] for ch in s if ch in self.stoi], dtype=torch.long)

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join([self.itos[i] for i in ids])

# Step 3 - get_batch
import torch


def get_batch(data, block_size, batch_size, generator=None):
    # TODO: random starts, input windows x and next-character targets y
    ids = torch.randint(0, len(data) - block_size - 1, (batch_size, ), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ids])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ids])
    return x, y

# Step 4 - CausalSelfAttention
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        # TODO: qkv, proj, attn_drop, resid_drop, n_heads, head_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model) # project the vector into Q, K, V - dxd matrices
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _split_heads(self, t):
        """(B, T, C) -> (B, H, T, D): each head becomes an independent batch entry."""
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t):
        """(B, H, T, D) -> (B, T, C): concatenate the heads back. Inverse of _split_heads."""
        B, H, T, D = t.shape
        return t.transpose(1, 2).reshape(B, T, H * D)

    def forward(self, x):
        B, T, C = x.shape

        # Project x into Q, K, V and split on heads
        q, k, v = self.qkv(x).split(C, dim=-1)            # each (B, T, C)
        q = self._split_heads(q)                          # (B, H, T, D)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Compute the scores
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # (B, H, T, T)

        # Causal mask: position i may not look at positions j > i
        future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))

        # Compute softmax and apply dropout
        weights = F.softmax(scores, dim=-1)               # (B, H, T, T)
        weights = self.attn_drop(weights)

        # Weighted average of the values
        out = weights @ v                                 # (B, H, T, D)

        # Concatenate heads, mix them, dropout
        out = self._merge_heads(out)                      # (B, T, C)
        return self.resid_drop(self.proj(out))

# Step 5 - GPT2MLP
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        # TODO: fc (d -> 4d), proj (4d -> d), drop
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x), approximate = 'tanh')))

# Step 6 - GPT2
import torch
import torch.nn as nn


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        # TODO: ln_1, attn, ln_2, mlp
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, block_size, dropout=0.1):
        super().__init__()
        # TODO: wte, wpe, drop, blocks, ln_f, head (tied), GPT-2 init
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model) # Word Token Embedding
        self.wpe = nn.Embedding(block_size, d_model) # Word Position Embedding
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPT2Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Wight tying
        self.head.weight = self.wte.weight

        # Initialize weights
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, idx): # (B, T), B - batch_size, T - block_size
        B, T = idx.size()
        assert T <= self.block_size, f"Can't resolve the sequence greater than block_size"

        # Positional indices
        pos = torch.arange(0, T, dtype=torch.long)

        # Embedding
        token_embedding = self.wte(idx) # token embeddings of shape (B, T, d_model)
        positional_embedding = self.wpe(pos) # position embeddings of shape (T, d_model)

        x = self.drop(token_embedding + positional_embedding)

        # Pass through transformer
        for block in self.blocks:
            x = block(x)
        
        # Layer norm + head projection to vocabulary logits
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)

        return logits

# Step 7 - evaluate_bpc
import math
import torch
import torch.nn.functional as F


def evaluate_bpc(model, data, block_size, batch_size=16, n_batches=8, seed=0):
    # TODO: eval mode, fixed-seed batches, mean cross-entropy / ln 2, restore mode

    # Switch to eval mode
    was_training = model.training
    model.eval()

    # Create seed
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, block_size, batch_size, generator=generator)
            # Move the tensors on the same device as model's
            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device)
            # Get logits
            logits = model(x)

            # Compute cross-entropy (B * T, vocab_size) & (B * T, ) 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    # Average the loss along the batches
    average_loss = total_loss / n_batches
    # Convert to bitwise loss
    bpc = average_loss / math.log(2)

    # Set the model back to training mode
    model.train(was_training)

    return float(bpc)

# Step 8 - train_lm
import torch
import torch.nn.functional as F


def train_lm(model, data, steps, block_size, batch_size=32, lr=3e-3, seed=0):
    # TODO: seed, AdamW(0.9, 0.95, wd 0.1), loop with clipping, return losses
    # Initialize generator, optimizer
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, betas = (0.9, 0.95), weight_decay = 0.1)

    losses = []
    for _ in range(steps): 
        # Get the batch
        x, y = get_batch(data, block_size, batch_size, generator = generator)

        # Move data to model's device
        device = next(model.parameters()).device
        x, y = x.to(device), y.to(device)

        # Compute logits, loss
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backprop
        optimizer.zero_grad() 
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()

        # Record the losses
        losses.append(float(loss.item()))
    
    return losses

# Step 9 - set_dropout
import torch.nn as nn


def set_dropout(model, p):
    # TODO: set p on every nn.Dropout and return how many there are
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p
            count += 1
    return count

# Step 17 - RMSNorm
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        # TODO: scale parameter (ones) and eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        t = x.to(torch.float32)

        # Compute variance of squares along the hidden dimension, keeping dimensions
        variance = t.pow(2).mean(dim=-1, keepdim=True)
        # Multiply by rsqrt (reciprocal square root) and scale
        output = t * torch.rsqrt(variance + self.eps) * self.scale

        return output.to(input_dtype)

