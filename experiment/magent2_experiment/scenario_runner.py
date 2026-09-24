from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .policy import ATTACK_OFFSETS, SharedActorCritic, apply_action_mask
from .risk_runner import (
    RiskEpisodeResult,
    _attack_opportunities,
    determine_terminal_scenarios,
    evaluate_oracle,
    red_hp_or_empty,
)
from .runner import MOVE_OFFSETS, _alive_counts, _team_positions
from .scenario_config import ScenarioConfig
from .scenario_environment import make_scenario_env
from .scenario_geometry import minimum_team_distance
from .scenario_monitor import has_local_numerical_disadvantage, updates_no_contact_coverage


def inspect_scenario(config: ScenarioConfig) -> dict[str, object]:
    env = make_scenario_env(config)
    try:
        observations, _ = env.reset(seed=config.environment_seed)
        names = list(observations)
        first = names[0]
        return {
            "case_id": config.case_id,
            "red_agents": sum(name.startswith("red_") for name in names),
            "blue_agents": sum(name.startswith("blue_") for name in names),
            "minimum_distance": minimum_team_distance(
                env.scenario_red_positions, env.scenario_blue_positions
            ),
            "obstacle_count": len(env.scenario_walls),
            "observation_shape": tuple(observations[first].shape),
            "action_count": int(env.action_space(first).n),
            "red_positions": list(env.scenario_red_positions),
            "blue_positions": list(env.scenario_blue_positions),
            "walls": list(env.scenario_walls),
        }
    finally:
        env.close()


def _health_by_name(env, positions: dict[str, tuple[float, float]], channel: int) -> dict[str, float]:
    state = np.asarray(env.state())
    return {name: float(state[int(round(x)), int(round(y)), channel]) for name, (x, y) in positions.items()}


def _random_legal_action(observation: np.ndarray, rng: np.random.Generator) -> int:
    valid = list(range(13))
    valid.extend(
        action
        for action, (dx, dy) in ATTACK_OFFSETS.items()
        if observation[6 + dy, 6 + dx, 3] > 0
    )
    return int(rng.choice(valid))


def _move_toward(position: tuple[float, float], target: tuple[float, float]) -> int:
    x, y = position
    tx, ty = target
    return min(
        MOVE_OFFSETS,
        key=lambda action: (x + MOVE_OFFSETS[action][0] - tx) ** 2
        + (y + MOVE_OFFSETS[action][1] - ty) ** 2,
    )


def _move_away(position: tuple[float, float], red_positions: dict[str, tuple[float, float]]) -> int:
    if not red_positions:
        return 6
    x, y = position
    return max(
        MOVE_OFFSETS,
        key=lambda action: min(
            (x + MOVE_OFFSETS[action][0] - rx) ** 2 + (y + MOVE_OFFSETS[action][1] - ry) ** 2
            for rx, ry in red_positions.values()
        ),
    )


def _rule_based_blue_actions(
    env,
    config: ScenarioConfig,
    observations: dict[str, np.ndarray],
    blue_names: list[str],
    red_names: list[str],
) -> dict[str, int]:
    blue_positions = _team_positions(env, 1, blue_names)
    red_positions = _team_positions(env, 0, red_names)
    red_health = _health_by_name(env, red_positions, 2) if red_positions else {}
    retreat = config.phase_rule == "retreat_after_low_hp" and red_health and min(red_health.values()) < 0.4
    actions: dict[str, int] = {}
    for name in blue_names:
        if retreat:
            actions[name] = _move_away(blue_positions[name], red_positions)
            continue
        if not red_positions:
            actions[name] = 6
            continue
        if config.blue_target_rule == "lowest_health":
            target_name = min(red_health, key=lambda item: (red_health[item], item))
        else:
            x, y = blue_positions[name]
            target_name = min(
                red_positions,
                key=lambda item: (red_positions[item][0] - x) ** 2 + (red_positions[item][1] - y) ** 2,
            )
        x, y = blue_positions[name]
        tx, ty = red_positions[target_name]
        delta = (int(round(tx - x)), int(round(ty - y)))
        attack = next((action for action, offset in ATTACK_OFFSETS.items() if offset == delta), None)
        actions[name] = attack if attack is not None else _move_toward((x, y), (tx, ty))
    return actions


@torch.no_grad()
def run_scenario_episode(
    model: SharedActorCritic,
    config: ScenarioConfig,
    device: str = "cpu",
) -> RiskEpisodeResult:
    started = time.perf_counter()
    torch.manual_seed(config.environment_seed)
    rng = np.random.default_rng(config.environment_seed + 400_009)
    env = make_scenario_env(config)
    observations, _ = env.reset(seed=config.environment_seed)
    covered: set[str] = set()
    failures: set[str] = set()
    team_return = 0.0
    previous_red_alive, previous_blue_alive = _alive_counts(env)
    previous_hp = red_hp_or_empty(env, previous_red_alive, previous_blue_alive)
    steps = 0
    no_contact_streak = 0
    try:
        while env.agents:
            steps += 1
            red_names = [name for name in env.agents if name.startswith("red_")]
            blue_names = [name for name in env.agents if name.startswith("blue_")]
            red_positions = _team_positions(env, 0, red_names)
            blue_positions = _team_positions(env, 1, blue_names)
            enemy_visible = any(float(np.asarray(observations[name])[..., 3].sum()) > 0 for name in red_names)
            no_contact_streak, no_contact_covered = updates_no_contact_coverage(
                no_contact_streak, enemy_visible
            )
            if no_contact_covered:
                covered.add("S01")
            if enemy_visible:
                covered.add("S02")
            attack_opportunity, focus_opportunity = _attack_opportunities(red_positions, blue_positions)
            if attack_opportunity:
                covered.add("S03")
            if focus_opportunity:
                covered.add("S04")
            if enemy_visible and any(
                float(np.asarray(observations[name])[4:9, 4:9, 0].sum()) > 0 for name in red_names
            ):
                covered.add("S05")

            actions: dict[str, int] = {}
            if red_names:
                batch = torch.as_tensor(np.stack([observations[name] for name in red_names]), device=device)
                logits, _ = model(batch)
                logits = apply_action_mask(logits, batch)
                selected = logits.argmax(dim=-1) if config.deterministic else Categorical(logits=logits).sample()
                actions.update({name: int(action) for name, action in zip(red_names, selected.cpu().numpy())})

            if config.blue_controller == "random":
                actions.update({name: _random_legal_action(np.asarray(observations[name]), rng) for name in blue_names})
            elif config.blue_controller == "shared_model" and blue_names:
                batch = torch.as_tensor(np.stack([observations[name] for name in blue_names]), device=device)
                logits, _ = model(batch)
                logits = apply_action_mask(logits, batch)
                selected = logits.argmax(dim=-1) if config.deterministic else Categorical(logits=logits).sample()
                actions.update({name: int(action) for name, action in zip(blue_names, selected.cpu().numpy())})
            elif config.blue_controller == "rule_based":
                actions.update(_rule_based_blue_actions(env, config, observations, blue_names, red_names))

            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            team_return += sum(float(value) for name, value in rewards.items() if name.startswith("red_"))
            red_alive, blue_alive = _alive_counts(env)
            if has_local_numerical_disadvantage(red_positions, blue_positions, radius=6, margin=2):
                covered.add("S06")
            hp = red_hp_or_empty(env, red_alive, blue_alive)
            if len(hp) and float(np.mean(hp < 0.4)) >= 0.25:
                covered.add("S07")
            if red_alive == previous_red_alive and len(hp) and len(previous_hp) and hp.sum() > previous_hp.sum() + 1e-7:
                covered.add("S08")
            if red_alive < previous_red_alive and red_alive > 0 and blue_alive > 0:
                covered.add("S09")
            observation_valid = all(
                np.asarray(value).shape == (13, 13, 5) and np.isfinite(value).all()
                for value in next_observations.values()
            )
            failures.update(evaluate_oracle(list(actions.values()), list(rewards.values()), observation_valid, True))
            observations = next_observations
            previous_red_alive, previous_blue_alive = red_alive, blue_alive
            previous_hp = hp

        red_alive, blue_alive = _alive_counts(env)
        reached_limit = steps >= config.max_cycles and red_alive > 0 and blue_alive > 0
        covered.update(determine_terminal_scenarios(red_alive, blue_alive, reached_limit))
        terminal_sync = reached_limit or red_alive == 0 or blue_alive == 0
        failures.update(evaluate_oracle([], [], True, terminal_sync))
    finally:
        env.close()

    if red_alive > 0 and blue_alive == 0:
        winner = "red"
    elif blue_alive > 0 and red_alive == 0:
        winner = "blue"
    else:
        winner = "draw"
    if not math.isfinite(team_return):
        failures.add("数值输出异常")
    return RiskEpisodeResult(
        test_case=config.to_dict(),
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
