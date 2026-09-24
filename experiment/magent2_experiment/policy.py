from __future__ import annotations

import torch
from torch import nn


ATTACK_OFFSETS = {
    13: (-1, -1), 14: (0, -1), 15: (1, -1), 16: (-1, 0),
    17: (1, 0), 18: (-1, 1), 19: (0, 1), 20: (1, 1),
}


def legal_action_mask(observations: torch.Tensor) -> torch.Tensor:
    """Mask attack directions that do not contain an adjacent opponent."""
    batch = observations.shape[0]
    mask = torch.ones((batch, 21), dtype=torch.bool, device=observations.device)
    for action, (dx, dy) in ATTACK_OFFSETS.items():
        mask[:, action] = observations[:, 6 + dy, 6 + dx, 3] > 0
    return mask


def apply_action_mask(logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal_action_mask(observations), -1e9)


class SharedActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int):
        super().__init__()
        height, width, channels = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            encoded = self.encoder(torch.zeros(1, channels, height, width)).shape[-1]
        self.body = nn.Sequential(nn.Linear(encoded, 256), nn.ReLU())
        self.actor = nn.Linear(256, action_count)
        self.critic = nn.Linear(256, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 4:
            raise ValueError("observations must have shape (batch, height, width, channels)")
        x = observations.float().permute(0, 3, 1, 2)
        hidden = self.body(self.encoder(x))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)
