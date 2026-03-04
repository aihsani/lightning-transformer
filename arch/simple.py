import torch
import torch.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dtype=torch.float16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim, dtype=dtype),
        )

    def forward(self, x):
        return self.layers(x)


class StatefulConvolutionLayer(nn.Module):

    def __init__(self, in_dim, out_dim, context_length, window_length, window_stride=None, device=torch.device('cpu'), xs0=None, dtype=torch.float16):
        super().__init__()
        self.device = device

        # if xs0 is None:
        #     self.xs = torch.zeros((1, 1, in_dim), device=device)
        # else:
        #     self.xs = xs0.to(device)

        # self.xs = nn.Parameter(self.xs)

        self.window_length = window_length
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_length = context_length
        self.window_stride = window_stride or window_length

        self.T_proj = nn.Linear(in_dim * window_length, out_dim * out_dim, bias=False, device=device, dtype=dtype)
        self.W_proj = nn.Linear(in_dim * window_length, out_dim * out_dim * window_length, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(in_dim * window_length, out_dim * window_length, bias=False, device=device, dtype=dtype)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.gelu = nn.GELU()
        self.ffn = FeedForwardLayer(out_dim, 4 * out_dim, dtype=dtype)
        self.fbn = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            nn.GELU(),
        )

    def forward(self, x):

        b, context_length, in_dim = x.shape

        # xs = self.xs
        y = [None] * context_length
        norm = [0.0] * context_length

        for loc in range(0, context_length - self.window_length + 1, self.window_stride):

            T = self.T_proj(x[:, loc:loc+self.window_length, ...].view(b, self.window_length * in_dim))
            W = self.W_proj(x[:, loc:loc+self.window_length, ...].view(b, self.window_length * in_dim))
            v = self.v_proj(x[:, loc:loc+self.window_length, ...].view(b, self.window_length * in_dim))

            T = T.view(b, self.out_dim, self.out_dim)
            W = W.view(b, self.window_length, self.out_dim, self.out_dim)
            v = v.view(b, self.window_length, self.out_dim)

            # vs = self.v_proj(xs)
            # vs = vs.view(self.in_dim, self.window_length, self.out_dim)

            # everything below happens in the context window
            for i in range(0, self.window_length):

                norm[i + loc] += 1.0

                # y[i] = nn.functional.softmax(T, dim=2) @ vs
                if y[i + loc] is not None:
                    y[i + loc] = y[i + loc] + self.gelu(nn.functional.softmax(T @ W[:, i, ...], dim=1) @ v[:, i, ...].unsqueeze(-1)).squeeze(-1)
                else:
                    y[i + loc] = self.gelu(nn.functional.softmax(T @ W[:, i, ...], dim=1) @ v[:, i, ...].unsqueeze(-1)).squeeze(-1)

                for j in range(1, i):

                    y[i + loc] = y[i + loc] \
                        + self.gelu(nn.functional.softmax(T @ W[:, j, ...], dim=1) @ v[:, j, ...].unsqueeze(-1)).squeeze(-1)

        y = [x / n for (x, n) in zip(y, norm)]
        y = torch.stack(y, dim=1)
        y = self.layer_norm(y)

        y = self.ffn(y)

        # xs = self.fbn(y)

        return y


if __name__ == "__main__":
    # Initialize model
    model = StatefulConvolutionLayer(
        in_dim=3,           # Input embedding dimension
        out_dim=64,         # Hidden state dimension
        context_length=10,  # Sequence length
        window_length=5,    # Convolution window size
        window_stride=1,    # Stride for window
    )

    # Generate dummy input
    batch_size = 2
    input_data = torch.randn(batch_size, 10, 3, dtype=torch.float16)

    # Forward pass
    output = model(input_data)

    # Validation checks
    print(f"\nInput shape: {input_data.shape}")  # (2,10,3)
    print(f"Output shape: {output.shape}")  # (2,10,3) [Same as input due to design]

    # Show first 3 output features
    print("\nFirst time step features:")
    print(input_data[0, 0, :3], "→", output[0, 0, :3])

    # Final time step features
    print("\nLast time step features:")
    print(input_data[0, -1, :3], "→", output[0, -1, :3])
