from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import magent2
import numpy as np
import torch
from magent2.environments import battlefield_v5
from torch.distributions import Categorical

from .faults import (
    apply_attack_dropout,
    apply_observation_freeze,
    apply_reward_offset,
    update_ghost_agents,
)
from .generators import TestCase
from .path_model import KEY_PATHS, StepFacts, classify_state, coordinated_focus_fire, path_is_covered, progress_by_path, terminal_state
from .policy import ATTACK_OFFSETS, SharedActorCritic, apply_action_mask


MOVE_OFFSETS = {
    0: (0, -2), 1: (-1, -1), 2: (0, -1), 3: (1, -1),
    4: (-2, 0), 5: (-1, 0), 6: (0, 0), 7: (1, 0),
    8: (2, 0), 9: (-1, 1), 10: (0, 1), 11: (1, 1), 12: (0, 2),
}


def inspect_environment(seed: int = 123) -> dict[str, object]:
    env = battlefield_v5.parallel_env(map_size=80, max_cycles=10, render_mode=None)
    observations, _ = env.reset(seed=seed)
    names = list(observations)
    first = names[0]
    facts = {
        "magent2_version": magent2.__version__,
        "red_agents": sum(name.startswith("red_") for name in names),
        "blue_agents": sum(name.startswith("blue_") for name in names),
        "observation_shape": [int(value) for value in observations[first].shape],
        "action_count": int(env.action_space(first).n),
    }
    env.close()
    return facts


def _agent_number(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def select_agents(
    observations: dict[str, np.ndarray],
    count: int,
    selection: str,
    rng: np.random.Generator,
    positions: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    red_names = sorted((name for name in observations if name.startswith("red_")), key=_agent_number)
    count = min(count, len(red_names))
    if selection == "random":
        return sorted(rng.choice(red_names, size=count, replace=False).tolist(), key=_agent_number) if count else []
    scores = {name: float(np.asarray(observations[name])[..., 3].sum()) for name in red_names}
    frontline = sorted(red_names, key=lambda name: (-scores[name], _agent_number(name)))
    if selection == "frontline" or not positions:
        return frontline[:count]
    anchor = frontline[0]
    ax, ay = positions[anchor]
    return sorted(red_names, key=lambda name: ((positions[name][0] - ax) ** 2 + (positions[name][1] - ay) ** 2, _agent_number(name)))[:count]


def _team_positions(env, handle_index: int, names: list[str]) -> dict[str, tuple[float, float]]:
    try:
        backend = env.unwrapped.env
        handle = env.unwrapped.handles[handle_index]
        ids = backend.get_agent_id(handle)
        raw = backend.get_pos(handle)
        possible_agents = env.unwrapped.possible_agents
        mapped = {
            possible_agents[int(agent_id)]: (float(pos[0]), float(pos[1]))
            for agent_id, pos in zip(ids, raw)
            if possible_agents[int(agent_id)] in names
        }
        if len(mapped) == len(names):
            return mapped
    except Exception:
        pass
    return {name: (float(index), 0.0) for index, name in enumerate(sorted(names, key=_agent_number))}


def _positions(env, red_names: list[str]) -> dict[str, tuple[float, float]]:
    return _team_positions(env, 0, red_names)


def _alive_counts(env) -> tuple[int, int]:
    return tuple(int(env.unwrapped.env.get_num(handle)) for handle in env.unwrapped.handles)  # type: ignore[return-value]


def _disadvantaged(env, red_alive: int, blue_alive: int) -> bool:
    if blue_alive - red_alive >= 2:
        return True
    try:
        state = env.state()
        present = state[..., 1] > 0
        hp = state[..., 2][present]
        return bool(len(hp) and np.mean(hp < 0.4) >= 0.25)
    except Exception:
        return False


@dataclass
class EpisodeResult:
    environment_seed: int
    team_return: float
    red_alive: int
    blue_alive: int
    winner: str
    engaged: bool
    oracle_failure: bool
    fault_activated: bool
    shadow_eligible: bool
    shadow_conditions: dict[str, bool]
    first_enemy_step: int | None
    first_attack_step: int | None
    first_lifecycle_step: int | None
    states: list[str]
    covered_paths: dict[str, bool]
    path_progress: dict[str, int]
    elapsed_seconds: float
    steps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_policy(checkpoint: str | Path, device: str = "cpu") -> SharedActorCritic:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SharedActorCritic((13, 13, 5), 21).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def run_episode(
    model: SharedActorCritic,
    test_case: TestCase,
    mutant: dict[str, object] | None = None,
    device: str = "cpu",
    deterministic: bool = True,
    red_policy: str = "shared_model",
    blue_policy: str = "random",
    map_size: int = 80,
    max_cycles: int = 1_000,
) -> EpisodeResult:
    start = time.perf_counter()
    env = battlefield_v5.parallel_env(map_size=map_size, max_cycles=max_cycles, render_mode=None)
    observations, _ = env.reset(seed=test_case.environment_seed)
    rng = np.random.default_rng(test_case.environment_seed + 100_003)
    states = ["M0"]
    team_return = 0.0
    engaged = False
    fault_activated = False
    oracle_failure = False
    oracle_next_step = False
    shadow_eligible = False
    shadow_conditions = {
        "reward_offset": False,
        "death_delay": False,
        "attack_dropout": False,
        "observation_freeze": False,
    }
    first_enemy_step = None
    first_attack_step = None
    first_lifecycle_step = None
    selected: list[str] = []
    snapshot: dict[str, np.ndarray] = {}
    ghosts: dict[str, int] = {}
    ghost_observations: dict[str, np.ndarray] = {}
    previous_agents = set(env.agents)
    steps = 0
    attack_history: deque[dict[str, tuple[int, int]]] = deque(maxlen=3)
    while env.agents:
        steps += 1
        red_names = [name for name in env.agents if name.startswith("red_")]
        blue_names = [name for name in env.agents if name.startswith("blue_")]
        if steps == test_case.trigger_step:
            selected = select_agents(
                {name: observations[name] for name in red_names},
                test_case.affected_agents,
                test_case.selection,
                rng,
                _positions(env, red_names),
            )
            snapshot = {name: np.array(observations[name], copy=True) for name in selected if name in observations}
        active = test_case.trigger_step <= steps < test_case.trigger_step + test_case.duration
        actions: dict[str, int] = {}
        policy_names = red_names + [name for name in ghosts if name in ghost_observations]
        if policy_names and red_policy == "shared_model":
            policy_observations = [observations[name] if name in observations else ghost_observations[name] for name in policy_names]
            policy_batch = torch.as_tensor(np.stack(policy_observations), device=device)
            logits, _ = model(policy_batch)
            logits = apply_action_mask(logits, policy_batch)
            chosen = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
            actions.update({name: int(action) for name, action in zip(policy_names, chosen.cpu().numpy()) if name in env.agents})
        elif policy_names and red_policy == "random":
            actions.update({name: int(rng.integers(env.action_space(name).n)) for name in policy_names if name in env.agents})
        elif red_policy not in {"shared_model", "random"}:
            raise ValueError(f"unknown red policy: {red_policy}")
        if blue_policy == "random":
            for name in blue_names:
                actions[name] = int(rng.integers(env.action_space(name).n))
        elif blue_policy == "shared_model" and blue_names:
            blue_batch = torch.as_tensor(np.stack([observations[name] for name in blue_names]), device=device)
            blue_logits, _ = model(blue_batch)
            blue_logits = apply_action_mask(blue_logits, blue_batch)
            blue_chosen = blue_logits.argmax(dim=-1) if deterministic else Categorical(logits=blue_logits).sample()
            actions.update({name: int(action) for name, action in zip(blue_names, blue_chosen.cpu().numpy())})
        elif blue_policy == "aggressive" and blue_names:
            red_positions_now = _team_positions(env, 0, red_names)
            blue_positions_now = _team_positions(env, 1, blue_names)
            for name in blue_names:
                observation = observations[name]
                attack_choices = [
                    action
                    for action, (dx, dy) in ATTACK_OFFSETS.items()
                    if observation[6 + dy, 6 + dx, 3] > 0
                ]
                if attack_choices:
                    actions[name] = attack_choices[0]
                elif red_positions_now:
                    x, y = blue_positions_now[name]
                    tx, ty = min(red_positions_now.values(), key=lambda pos: (pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                    actions[name] = min(
                        MOVE_OFFSETS,
                        key=lambda action: (x + MOVE_OFFSETS[action][0] - tx) ** 2 + (y + MOVE_OFFSETS[action][1] - ty) ** 2,
                    )
                else:
                    actions[name] = 6
        elif blue_policy not in {"shared_model", "aggressive"}:
            raise ValueError(f"unknown blue policy: {blue_policy}")
        operator = str(mutant["operator"]) if mutant else ""
        strength = mutant["strength"] if mutant else 0
        activated_this_step = False
        if active and operator == "attack_dropout":
            actions, dropped = apply_attack_dropout(actions, selected, float(strength), rng)
            activated_this_step = bool(dropped)
        red_positions = _positions(env, red_names)
        attack_targets = {
            name: (int(round(red_positions[name][0] + ATTACK_OFFSETS[action][0])), int(round(red_positions[name][1] + ATTACK_OFFSETS[action][1])))
            for name, action in actions.items()
            if name in red_positions and action in ATTACK_OFFSETS
        }
        if active and any(name in selected and action in ATTACK_OFFSETS for name, action in actions.items()):
            shadow_conditions["attack_dropout"] = True
        attack_history.append(attack_targets)
        focus_fire = coordinated_focus_fire(list(attack_history))
        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        current_agents = set(next_observations)
        newly_dead = sorted(name for name in previous_agents - current_agents if name.startswith("red_"))
        lifecycle_event = current_agents != previous_agents
        if active and any(name in selected for name in newly_dead):
            shadow_conditions["death_delay"] = True
        if active and any(float(rewards.get(name, 0.0)) > 1.0 for name in selected):
            shadow_conditions["reward_offset"] = True
        if active and any(
            name in next_observations
            and name in snapshot
            and not np.array_equal(next_observations[name], snapshot[name])
            and (
                float(observations.get(name, next_observations[name])[..., 3].sum()) > 0
                or float(next_observations[name][..., 3].sum()) > 0
            )
            for name in selected
        ):
            shadow_conditions["observation_freeze"] = True
        if active and operator == "reward_offset" and any(value > 1.0 for name, value in rewards.items() if name.startswith("red_")):
            rewards, shifted = apply_reward_offset(rewards, selected, int(strength))
            activated_this_step |= shifted
            oracle_next_step |= shifted
        if active and operator == "death_delay":
            affected_deaths = [name for name in newly_dead if name in selected]
            if affected_deaths:
                for name in affected_deaths:
                    if name in observations:
                        ghost_observations[name] = np.array(observations[name], copy=True)
                ghosts = update_ghost_agents(ghosts, affected_deaths, int(strength))
                activated_this_step = True
                oracle_next_step = True
        elif ghosts:
            ghosts = update_ghost_agents(ghosts, [], int(strength) if strength else 1)
            ghost_observations = {name: value for name, value in ghost_observations.items() if name in ghosts}
        if active and operator == "observation_freeze":
            freeze_end = test_case.trigger_step + min(test_case.duration, int(strength))
            if steps < freeze_end:
                next_observations, stale = apply_observation_freeze(next_observations, snapshot, selected)
                activated_this_step |= bool(stale)
        fault_activated |= activated_this_step
        shadow_eligible = any(shadow_conditions.values())
        team_return += sum(value for name, value in rewards.items() if name.startswith("red_"))
        effective_attack = any(value > 0.05 for name, value in rewards.items() if name.startswith("red_"))
        engaged |= effective_attack
        red_alive, blue_alive = _alive_counts(env)
        enemy_visible = any(float(obs[..., 3].sum()) > 0 for name, obs in next_observations.items() if name.startswith("red_"))
        if enemy_visible and first_enemy_step is None:
            first_enemy_step = steps
        if (effective_attack or focus_fire) and first_attack_step is None:
            first_attack_step = steps
        if lifecycle_event and first_lifecycle_step is None:
            first_lifecycle_step = steps
        explicit_anomaly = oracle_next_step and not lifecycle_event
        facts = StepFacts(
            initialized=False,
            enemy_visible=enemy_visible,
            effective_attack=effective_attack or focus_fire,
            disadvantaged=_disadvantaged(env, red_alive, blue_alive),
            fault_active=active and activated_this_step and operator in {"attack_dropout", "observation_freeze"},
            lifecycle_event=lifecycle_event,
            red_alive=red_alive,
            blue_alive=blue_alive,
            oracle_failure=explicit_anomaly,
        )
        states.append(classify_state(facts))
        if explicit_anomaly:
            oracle_failure = True
            oracle_next_step = False
        observations = next_observations
        previous_agents = current_agents
    red_alive, blue_alive = _alive_counts(env)
    env.close()
    terminal = terminal_state(red_alive, blue_alive)
    if terminal is not None and states[-1] != terminal:
        states.append(terminal)
    if red_alive > 0 and blue_alive == 0:
        winner = "red"
    elif blue_alive > 0 and red_alive == 0:
        winner = "blue"
    else:
        winner = "draw"
    if oracle_next_step:
        oracle_failure = True
        states.append("M8")
    return EpisodeResult(
        environment_seed=test_case.environment_seed,
        team_return=float(team_return),
        red_alive=red_alive,
        blue_alive=blue_alive,
        winner=winner,
        engaged=engaged,
        oracle_failure=oracle_failure,
        fault_activated=fault_activated,
        shadow_eligible=shadow_eligible,
        shadow_conditions=shadow_conditions,
        first_enemy_step=first_enemy_step,
        first_attack_step=first_attack_step,
        first_lifecycle_step=first_lifecycle_step,
        states=states,
        covered_paths={name: path_is_covered(states, path) for name, path in KEY_PATHS.items()},
        path_progress=progress_by_path(states),
        elapsed_seconds=time.perf_counter() - start,
        steps=steps,
    )
