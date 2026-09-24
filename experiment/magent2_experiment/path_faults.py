from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FaultSpec:
    operator: str
    strength: float
    level: int = 0

    @property
    def fault_id(self) -> str:
        suffix = f"L{self.level}" if self.level else str(self.strength).replace(".", "p")
        return f"{self.operator}_{suffix}"


MUTANTS: tuple[FaultSpec, ...] = tuple(
    FaultSpec(operator, strength, level)
    for operator, strengths in {
        "attack_dropout": (0.25, 0.50, 0.75),
        "observation_mask": (0.25, 0.50, 1.00),
        "action_shift": (0.10, 0.25, 0.50),
        "reward_offset": (0.05, 0.10, 0.20),
    }.items()
    for level, strength in enumerate(strengths, start=1)
)


class FaultRuntime:
    def __init__(self, spec: FaultSpec, seed: int):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.activation_count = 0

    def mutate_policy_observations(
        self,
        observations: Mapping[str, np.ndarray],
        red_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        if self.spec.operator != "observation_mask":
            return dict(observations)
        mutated = dict(observations)
        for name in red_names:
            if self.rng.random() >= self.spec.strength:
                continue
            observation = np.asarray(observations[name]).copy()
            observation[..., 3] = 0
            mutated[name] = observation
            self.activation_count += 1
        return mutated

    def mutate_actions(self, actions: Mapping[str, int]) -> dict[str, int]:
        mutated = dict(actions)
        for name, action in actions.items():
            if not name.startswith("red_"):
                continue
            if self.spec.operator == "attack_dropout":
                if action >= 13 and self.rng.random() < self.spec.strength:
                    mutated[name] = 6
                    self.activation_count += 1
            elif self.spec.operator == "action_shift":
                if self.rng.random() < self.spec.strength:
                    mutated[name] = (int(action) + 1) % 21
                    self.activation_count += 1
        return mutated

    def mutate_rewards(self, rewards: Mapping[str, float]) -> dict[str, float]:
        if self.spec.operator != "reward_offset":
            return dict(rewards)
        mutated = dict(rewards)
        for name, reward in rewards.items():
            if name.startswith("red_"):
                mutated[name] = float(reward) - self.spec.strength
                self.activation_count += 1
        return mutated
