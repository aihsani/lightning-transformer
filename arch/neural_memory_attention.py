import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NeuralMemoryAttention(nn.Module):
    """
    TITANS-style Neural Memory attention.

    - Replaces softmax attention with an online Delta Rule update:
        e_t = S_{t-1} k_t - v_t
        S_t = (1 - σ(α)) * S_{t-1} - η * G_t ⊙ momentum(e_t ⊗ k_t)
        z_t = S_t q_t

      where G_t is a surprise-dependent gain and momentum is an exponential
      moving average over outer(e_t, k_t).

    - memory_state S has shape (batch, num_heads, head_dim, head_dim)
      and does NOT grow with sequence length.
    - Forward returns (output, memory_state).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        dropout: float,
        qkv_bias: bool = False,
        inner_momentum_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Learned inner-loop hyperparameters
        # η > 0 (effective learning rate for the Delta Rule)
        self.log_eta = nn.Parameter(torch.zeros(1))  # softplus -> (0, ∞)

        # α (gating how much memory is retained vs updated), passed through sigmoid
        self.alpha = nn.Parameter(torch.zeros(1))

        # β (momentum coefficient) in (0, 1) via sigmoid
        self.logit_beta = nn.Parameter(
            torch.logit(torch.tensor(inner_momentum_init).clamp(1e-4, 1 - 1e-4))
        )

        # Surprise scaling (how strongly surprise amplifies the update)
        self.surprise_scale = nn.Parameter(torch.ones(1))

        # Persistent memory buffer; dynamically shaped per batch at runtime
        # Shape when allocated: (batch, num_heads, head_dim, head_dim)
        self.register_buffer("memory_state", None, persistent=False)

    def _init_memory(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def delta_rule_update(
        self,
        S_prev: torch.Tensor,
        V_prev: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One-step Delta Rule update with momentum and surprise modulation.

        Args:
            S_prev: (B, H, D, D) previous memory matrix
            V_prev: (B, H, D, D) previous momentum state
            k_t:    (B, H, D) key at time t
            v_t:    (B, H, D) value at time t

        Returns:
            S_t:        (B, H, D, D) updated memory
            V_t:        (B, H, D, D) updated momentum
            surprise_t: (B, H, 1)    surprise signal (for optional monitoring)
        """
        # Hyperparameters
        eta = F.softplus(self.log_eta)  # > 0
        forget = torch.sigmoid(self.alpha)  # in (0, 1)
        beta = torch.sigmoid(self.logit_beta)  # in (0, 1)

        # Predict v_t from current memory: v_hat_t = S_{t-1} k_t
        # S_prev: (B, H, D, D), k_t: (B, H, D) -> v_hat: (B, H, D)
        v_hat = torch.einsum("bhde,bhe->bhd", S_prev, k_t)

        # Error and surprise
        e_t = v_hat - v_t  # (B, H, D)
        surprise_t = (e_t ** 2).mean(dim=-1, keepdim=True)  # (B, H, 1)

        # Surprise-dependent gain: higher surprise => larger update
        gain_t = 1.0 + self.surprise_scale * surprise_t  # (B, H, 1)

        # Raw outer product: e_t ⊗ k_t -> (B, H, D, D)
        delta_raw = torch.einsum("bhd,bhe->bhde", e_t, k_t)

        # Momentum over updates
        V_t = beta * V_prev + (1.0 - beta) * delta_raw  # (B, H, D, D)

        # Scale by surprise gain (broadcast over last two dims)
        gain_t_expanded = gain_t.unsqueeze(-1)  # (B, H, 1, 1)
        scaled_update = gain_t_expanded * V_t  # (B, H, D, D)

        # Final Delta Rule update with forget gate
        S_t = (1.0 - forget) * S_prev - eta * scaled_update

        return S_t, V_t, surprise_t

    def forward(
        self,
        x: torch.Tensor,
        past_memory_state: Optional[torch.Tensor] = None,
        reset_memory: bool = False,
        return_memory_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: input tensor of shape (B, L, d_in)
            past_memory_state: optional previous S of shape (B, H, D, D).
                               If provided, used as S_0; otherwise we either
                               reset or reuse internal memory_state.
            reset_memory: if True, ignore stored memory and start from zeros.
            return_memory_state: if True, return (output, S_T); else just output.

        Returns:
            output: (B, L, d_out)
            memory_state: (B, H, D, D) final S_T (if return_memory_state is True)
        """
        B, L, _ = x.shape
        device, dtype = x.device, x.dtype

        # Projections
        keys = self.W_key(x)  # (B, L, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to (B, H, L, D)
        keys = keys.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Initialize memory S_0
        if past_memory_state is not None:
            S = past_memory_state
        elif (self.memory_state is None) or reset_memory:
            S = self._init_memory(B, device, dtype)
        else:
            # Reuse stored memory_state, expanding or slicing batch if needed
            if self.memory_state.shape[0] == B:
                S = self.memory_state.to(device=device, dtype=dtype)
            else:
                S = self._init_memory(B, device, dtype)

        # Initialize momentum state V_0 (local to this sequence)
        V = torch.zeros_like(S)

        # Recurrent inner loop over sequence length (causal by construction)
        outputs = []
        for t in range(L):
            q_t = queries[:, :, t, :]  # (B, H, D)
            k_t = keys[:, :, t, :]  # (B, H, D)
            v_t = values[:, :, t, :]  # (B, H, D)

            # Update memory with current (k_t, v_t)
            S, V, _ = self.delta_rule_update(S, V, k_t, v_t)

            # Retrieve with updated memory: z_t = S_t q_t
            z_t = torch.einsum("bhde,bhe->bhd", S, q_t)  # (B, H, D)
            outputs.append(z_t)

        # Stack over time: (B, H, L, D) -> (B, L, H, D) -> (B, L, d_out)
        z = torch.stack(outputs, dim=2)  # (B, H, L, D)
        z = z.transpose(1, 2).contiguous()  # (B, L, H, D)
        context_vec = z.view(B, L, self.d_out)  # (B, L, d_out)

        context_vec = self.out_proj(context_vec)
        context_vec = self.dropout(context_vec)

        # Persist memory across calls in inference mode
        # (optional: you can comment this out if you prefer stateless behavior)
        if not self.training:
            self.memory_state = S.detach()

        if return_memory_state:
            return context_vec, S
        else:
            return context_vec, None

