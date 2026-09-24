from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


ATTACK_ACTION_START = 13
NO_OP_ACTION = 6


def apply_reward_offset(
    rewards: Mapping[str, float], selected_red: Sequence[str], offset: int
) -> tuple[dict[str, float], bool]:
    changed = dict(rewards)
    selected = [name for name in selected_red if name in rewards]
    if len(selected) < 2:
        return changed, False
    positive = [(name, rewards[name]) for name in selected if rewards[name] > 0.05]
    if not positive:
        return changed, False
    for source, value in positive:
        target = selected[(selected.index(source) + offset) % len(selected)]
        if target == source:
            continue
        changed[source], changed[target] = changed[target], value
    return changed, changed != dict(rewards)


def apply_attack_dropout(
    actions: Mapping[str, int],
    selected_red: Sequence[str],
    probability: float,
    rng: np.random.Generator,
) -> tuple[dict[str, int], list[str]]:
    changed = dict(actions)
    dropped: list[str] = []
    selected = set(selected_red)
    for name, action in actions.items():
        if name in selected and int(action) >= ATTACK_ACTION_START and rng.random() < probability:
            changed[name] = NO_OP_ACTION
            dropped.append(name)
    return changed, dropped


def apply_observation_freeze(
    current: Mapping[str, np.ndarray],
    snapshot: Mapping[str, np.ndarray],
    selected_red: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    changed = {name: np.array(value, copy=True) for name, value in current.items()}
    stale: list[str] = []
    for name in selected_red:
        if name in current and name in snapshot and not np.array_equal(current[name], snapshot[name]):
            changed[name] = np.array(snapshot[name], copy=True)
            stale.append(name)
    return changed, stale


def update_ghost_agents(
    ghosts: Mapping[str, int], newly_dead: Sequence[str], delay: int
) -> dict[str, int]:
    updated = {name: remaining - 1 for name, remaining in ghosts.items() if remaining - 1 > 0}
    for name in newly_dead:
        updated[name] = max(updated.get(name, 0), delay)
    return updated


MUTANTS = [
    *(dict(name=f"reward_offset_{value}", operator="reward_offset", strength=value) for value in (1, 2, 3)),
    *(dict(name=f"death_delay_{value}", operator="death_delay", strength=value) for value in (1, 3, 5)),
    *(dict(name=f"attack_dropout_{value}", operator="attack_dropout", strength=value) for value in (0.2, 0.4, 0.6)),
    *(dict(name=f"observation_freeze_{value}", operator="observation_freeze", strength=value) for value in (5, 15, 30)),
]
