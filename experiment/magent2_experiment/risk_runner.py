from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from magent2.environments import battlefield_v5
from torch.distributions import Categorical

from .policy import ATTACK_OFFSETS, SharedActorCritic, apply_action_mask
from .risk_catalog import RiskTestCase
from .runner import MOVE_OFFSETS, _alive_counts, _team_positions


def determine_terminal_scenarios(red_alive: int, blue_alive: int, reached_limit: bool) -> set[str]:
    if red_alive > 0 and blue_alive == 0:
        return {"S10"}
    if blue_alive > 0 and red_alive == 0:
        return {"S11"}
    if reached_limit:
        return {"S12"}
    return set()


def evaluate_oracle(
    actions: list[int], rewards: list[float], observation_valid: bool, terminal_sync: bool
) -> list[str]:
    failures: list[str] = []
    if any(action < 0 or action > 20 for action in actions):
        failures.append("动作越界")
    if any(not math.isfinite(float(value)) for value in rewards):
        failures.append("数值输出异常")
    if not observation_valid:
        failures.append("观测结构异常")
    if not terminal_sync:
        failures.append("终止不同步")
    return failures


@dataclass
class RiskEpisodeResult:
    test_case: dict[str, object]
    covered_scenarios: list[str]
    nonconforming_scenarios: list[str]
    oracle_failures: list[str]
    winner: str
    red_alive: int
    blue_alive: int
    team_return: float
    steps: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _attack_opportunities(
    red_positions: dict[str, tuple[float, float]], blue_positions: dict[str, tuple[float, float]]
) -> tuple[bool, bool]:
    offsets = set(ATTACK_OFFSETS.values())
    attackers_by_target: dict[str, set[str]] = {}
    for red_name, (rx, ry) in red_positions.items():
        for blue_name, (bx, by) in blue_positions.items():
            delta = (int(round(bx - rx)), int(round(by - ry)))
            if delta in offsets:
                attackers_by_target.setdefault(blue_name, set()).add(red_name)
    any_attack = any(attackers_by_target.values())
    focus = any(len(attackers) >= 2 for attackers in attackers_by_target.values())
    return any_attack, focus


def _red_hp(env) -> np.ndarray:
    state = np.asarray(env.state())
    present = state[..., 1] > 0
    return np.asarray(state[..., 2][present], dtype=float)


def red_hp_or_empty(env, red_alive: int, blue_alive: int) -> np.ndarray:
    """MAgent2 0.3.4 cannot build a global state after either team is empty."""
    if red_alive == 0 or blue_alive == 0:
        return np.asarray([], dtype=float)
    return _red_hp(env)


@torch.no_grad()
def run_risk_episode(model: SharedActorCritic, case: RiskTestCase, device: str = "cpu") -> RiskEpisodeResult:
    started = time.perf_counter()
    env = battlefield_v5.parallel_env(map_size=case.map_size, max_cycles=case.max_cycles, render_mode=None)
    observations, _ = env.reset(seed=case.environment_seed)
    rng = np.random.default_rng(case.environment_seed + 400_009)
    covered: set[str] = set()
    failures: set[str] = set()
    team_return = 0.0
    previous_red_alive, previous_blue_alive = _alive_counts(env)
    previous_hp = _red_hp(env)
    steps = 0
    while env.agents:
        steps += 1
        red_names = [name for name in env.agents if name.startswith("red_")]
        blue_names = [name for name in env.agents if name.startswith("blue_")]
        red_positions = _team_positions(env, 0, red_names)
        blue_positions = _team_positions(env, 1, blue_names)
        enemy_visible = any(float(np.asarray(observations[name])[..., 3].sum()) > 0 for name in red_names)
        if steps == 3 and not enemy_visible:
            covered.add("S01")
        if enemy_visible:
            covered.add("S02")
        attack_opportunity, focus_opportunity = _attack_opportunities(red_positions, blue_positions)
        if attack_opportunity:
            covered.add("S03")
        if focus_opportunity:
            covered.add("S04")
        if enemy_visible and any(float(np.asarray(observations[name])[4:9, 4:9, 0].sum()) > 0 for name in red_names):
            covered.add("S05")

        actions: dict[str, int] = {}
        if red_names:
            batch = torch.as_tensor(np.stack([observations[name] for name in red_names]), device=device)
            logits, _ = model(batch)
            logits = apply_action_mask(logits, batch)
            selected = logits.argmax(dim=-1) if case.deterministic else Categorical(logits=logits).sample()
            actions.update({name: int(action) for name, action in zip(red_names, selected.cpu().numpy())})
        if case.blue_policy == "random":
            actions.update({name: int(rng.integers(env.action_space(name).n)) for name in blue_names})
        elif case.blue_policy == "shared_model" and blue_names:
            batch = torch.as_tensor(np.stack([observations[name] for name in blue_names]), device=device)
            logits, _ = model(batch)
            logits = apply_action_mask(logits, batch)
            selected = logits.argmax(dim=-1) if case.deterministic else Categorical(logits=logits).sample()
            actions.update({name: int(action) for name, action in zip(blue_names, selected.cpu().numpy())})
        elif case.blue_policy == "aggressive":
            for name in blue_names:
                attack_choices = [
                    action for action, (dx, dy) in ATTACK_OFFSETS.items()
                    if observations[name][6 + dy, 6 + dx, 3] > 0
                ]
                if attack_choices:
                    actions[name] = attack_choices[0]
                elif red_positions:
                    x, y = blue_positions[name]
                    tx, ty = min(red_positions.values(), key=lambda pos: (pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                    actions[name] = min(
                        MOVE_OFFSETS,
                        key=lambda action: (x + MOVE_OFFSETS[action][0] - tx) ** 2 + (y + MOVE_OFFSETS[action][1] - ty) ** 2,
                    )
                else:
                    actions[name] = 6
        else:
            if case.blue_policy not in {"random", "shared_model", "aggressive"}:
                env.close()
                raise ValueError(f"unknown blue policy: {case.blue_policy}")

        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        team_return += sum(float(value) for name, value in rewards.items() if name.startswith("red_"))
        red_alive, blue_alive = _alive_counts(env)
        if blue_alive - red_alive >= 2:
            covered.add("S06")
        hp = red_hp_or_empty(env, red_alive, blue_alive)
        if len(hp) and float(np.mean(hp < 0.4)) >= 0.25:
            covered.add("S07")
        if red_alive == previous_red_alive and len(hp) and len(previous_hp) and hp.sum() > previous_hp.sum() + 1e-7:
            covered.add("S08")
        if red_alive < previous_red_alive and red_alive > 0 and blue_alive > 0:
            covered.add("S09")

        observation_valid = all(np.asarray(value).shape == (13, 13, 5) and np.isfinite(value).all() for value in next_observations.values())
        failures.update(evaluate_oracle(list(actions.values()), list(rewards.values()), observation_valid, True))
        observations = next_observations
        previous_red_alive, previous_blue_alive = red_alive, blue_alive
        previous_hp = hp

    red_alive, blue_alive = _alive_counts(env)
    reached_limit = steps >= case.max_cycles and red_alive > 0 and blue_alive > 0
    covered.update(determine_terminal_scenarios(red_alive, blue_alive, reached_limit))
    terminal_sync = reached_limit or red_alive == 0 or blue_alive == 0
    failures.update(evaluate_oracle([], [], True, terminal_sync))
    env.close()
    if red_alive > 0 and blue_alive == 0:
        winner = "red"
    elif blue_alive > 0 and red_alive == 0:
        winner = "blue"
    else:
        winner = "draw"
    return RiskEpisodeResult(
        test_case=case.to_dict(),
        covered_scenarios=sorted(covered),
        nonconforming_scenarios=sorted(covered) if failures else [],
        oracle_failures=sorted(failures),
        winner=winner,
        red_alive=red_alive,
        blue_alive=blue_alive,
        team_return=float(team_return),
        steps=steps,
        elapsed_seconds=time.perf_counter() - started,
    )
