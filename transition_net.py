import torch
import torch.nn as nn


class TransitionNet(nn.Module):
    def __init__(self, hidden_size: int, rnn_hidden: int = None):
        super().__init__()
        rnn_hidden = rnn_hidden or hidden_size
        self.rnn = nn.LSTM(
            input_size=2 * hidden_size + 1,
            hidden_size=rnn_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(2 * rnn_hidden, hidden_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_a, z_b, w.unsqueeze(-1)], dim=-1)
        h, _ = self.rnn(x)
        return self.head(h)


def hold_window(w: torch.Tensor) -> torch.Tensor:
    return 4.0 * w * (1.0 - w)


def blended_latent(z_a: torch.Tensor, z_b: torch.Tensor, w: torch.Tensor,
                    transition_net: TransitionNet) -> torch.Tensor:
    naive = (1 - w).unsqueeze(-1) * z_a + w.unsqueeze(-1) * z_b
    delta_z = transition_net(z_a, z_b, w)
    window = hold_window(w).unsqueeze(-1)
    return naive + window * delta_z
