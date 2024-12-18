import torch
import torch.nn as nn
from llama2 import MultiHeadAttention, RMSNorm, SiLU, FeedForward

import torch


def precompute_rope_params_2d(head_dim, theta_base=10_000, max_height=64, max_width=64):
    assert head_dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D RoPE"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))

    # Generate position indices for height and width
    h_positions = torch.arange(max_height)
    w_positions = torch.arange(max_width)

    # Compute the angles for height and width
    h_angles = h_positions[:, None] * inv_freq[None, :]  # Shape: (max_height, head_dim // 4)
    w_angles = w_positions[:, None] * inv_freq[None, :]  # Shape: (max_width, head_dim // 4)

    # Expand angles to match the head_dim
    h_angles = torch.cat([h_angles, h_angles], dim=1)  # Shape: (max_height, head_dim // 2)
    w_angles = torch.cat([w_angles, w_angles], dim=1)  # Shape: (max_width, head_dim // 2)

    # Precompute sine and cosine
    h_cos, h_sin = torch.cos(h_angles), torch.sin(h_angles)
    w_cos, w_sin = torch.cos(w_angles), torch.sin(w_angles)

    return h_cos, h_sin, w_cos, w_sin


def compute_rope_2d(x, h_cos, h_sin, w_cos, w_sin, height, width):
    # x: (batch_size, num_heads, height * width, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE"
    assert seq_len == height * width, "Sequence length must equal height * width"

    # Reshape x to separate height and width dimensions
    x = x.view(batch_size, num_heads, height, width, head_dim)

    # Split x into quarters
    x1, x2, x3, x4 = torch.split(x, head_dim // 4, dim=-1)

    # Apply rotary embedding for height
    h_cos = h_cos[:height, :].unsqueeze(1).unsqueeze(1).unsqueeze(0)  # Shape: (1, 1, height, 1, head_dim // 2)
    h_sin = h_sin[:height, :].unsqueeze(1).unsqueeze(1).unsqueeze(0)
    x1, x2 = (x1 * h_cos) - (x2 * h_sin), (x2 * h_cos) + (x1 * h_sin)

    # Apply rotary embedding for width
    w_cos = w_cos[:width, :].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, width, head_dim // 2)
    w_sin = w_sin[:width, :].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x3, x4 = (x3 * w_cos) - (x4 * w_sin), (x4 * w_cos) + (x3 * w_sin)

    # Concatenate the quarters back together
    x_rotated = torch.cat([x1, x2, x3, x4], dim=-1)

    # Reshape back to original dimensions
    x_rotated = x_rotated.view(batch_size, num_heads, height * width, head_dim)

    return x_rotated.to(dtype=x.dtype)


def create_neighbor_mask(height, width, include_diagonal=True):
    mask = torch.zeros(height * width, height * width)
    for i in range(height * width):
        row, col = i // width, i % width
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if include_diagonal or dr == 0 or dc == 0:  # Check if we should include diagonal
                    if 0 <= row + dr < height and 0 <= col + dc < width:
                        neighbor_idx = (row + dr) * width + (col + dc)
                        mask[i, neighbor_idx] = 1
    return mask


class ContextLengthReductionBlock(nn.Module):
    # Apply a linear tranformation to the embedding vectors in the context dimension
    # so that the output context length is the one we expect
    def __init__(self, cfg):
        super().__init__()
        self.in_contex_length = cfg["max_in_width"] * cfg["max_in_height"]
        self.out_contex_length = cfg["max_out_width"] * cfg["max_out_height"]
        self.emb_dim = cfg["emb_dim"]
        self.linear = nn.Linear(self.in_contex_length, self.out_contex_length, bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        # x is [b x self.in_contex_length x self.emd_dim]
        batch_size = x.size(0)
        # make sure linear is applied only to sequence direction
        x_reshaped = x.permute(0, 2, 1).reshape(-1, self.in_contex_length)
        y = self.linear(x_reshaped)
        y = y.reshape(batch_size, self.emb_dim, self.out_context_length).permute(0, 2, 1)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            width=cfg["max_width"],
            height=cfg["max_height"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)

        # RMS Norm in Llama
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class LlamaLikeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # no embedding layer: it is assumed inputs are embeddings from another model (H-optimus-0)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # RMS Norm
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.reduction = ContextLengthReductionBlock(cfg)
        self.hazard_head = nn.Linear(cfg["emb_dim"], 1, bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        # x is now expected to be the pre-computed embeddings
        # Shape of x: [batch_size, seq_len, emb_dim]

        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.reduction(x)
        log_hazards = self.hazard_head(logits)
        return log_hazards, logits
