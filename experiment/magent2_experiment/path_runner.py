from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .coverage_model import CoverageMonitor, StateEvidence, classify_state_flags
from .path_faults import FaultRuntime
from .policy import SharedActorCritic, apply_action_mask
from .risk_runner import _attack_opportunities, determine_terminal_scenarios, evaluate_oracle, red_hp_or_empty
from .runner import _alive_counts, _team_positions
from .scenario_config import ScenarioConfig
from .scenario_environment import make_scenario_env
from .scenario_monitor import has_local_numerical_disadvantage, updates_no_contact_coverage
from .scenario_runner import _random_legal_action, _rule_based_blue_actions


@dataclass
class PathEpisodeResult:
    test_case: dict[str, object]
    state_trace: list[str]
    state_flags: list[list[str]]
    covered_states: list[str]
    transitions: list[dict[str, object]]
    covered_paths: list[str]
    path_progress: dict[str, float]
    covered_scenarios: list[str]
    oracle_failures: list[str]
    transition_conformance: float
    winner: str
    red_alive: int
    blue_alive: int
    team_return: float
    steps: int
    mutant_id: str | None
    fault_activations: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _result_from_monitor(
    monitor: CoverageMonitor,
    *,
    test_case: dict[str, object],
    covered_scenarios: set[str] | None = None,
    team_return: float = 0.0,
    steps: int | None = None,
    elapsed_seconds: float = 0.0,
    fault: FaultRuntime | None = None,
) -> PathEpisodeResult:
    transitions = [asdict(item) for item in monitor.transitions]
    legal_count = sum(bool(item["legal"]) for item in transitions)
    conformance = legal_count / len(transitions) if transitions else 1.0
    oracle_failures = sorted(
        {failure for record in monitor.records for failure in record.oracle_failures}
    )
    final_evidence_state = monitor.records[-1].primary_state if monitor.records else "M0"
    if final_evidence_state == "M7":
        winner = "red"
    elif final_evidence_state == "M8":
        winner = "blue" if test_case.get("terminal_tag") == "red_failure" else "draw"
    else:
        winner = "unfinished"

    red_alive = int(test_case.get("red_alive", 0))
    blue_alive = int(test_case.get("blue_alive", 0))
    if monitor.transitions:
        final_guard = monitor.transitions[-1].guard_evidence
        red_alive = int(final_guard.get("red_alive", red_alive))
        blue_alive = int(final_guard.get("blue_alive", blue_alive))
    if winner == "unfinished" and red_alive > 0 and blue_alive == 0:
        winner = "red"
    elif winner == "unfinished" and blue_alive > 0 and red_alive == 0:
        winner = "blue"

    return PathEpisodeResult(
        test_case=test_case,
        state_trace=[record.primary_state for record in monitor.records],
        state_flags=[list(record.flags) for record in monitor.records],
        covered_states=sorted(
            {flag for record in monitor.records for flag in record.flags},
            key=lambda item: int(item[1:]),
        ),
        transitions=transitions,
        covered_paths=sorted(monitor.covered_paths),
        path_progress=monitor.path_progress,
        covered_scenarios=sorted(covered_scenarios or set()),
        oracle_failures=oracle_failures,
        transition_conformance=conformance,
        winner=winner,
        red_alive=red_alive,
        blue_alive=blue_alive,
        team_return=float(team_return),
        steps=len(monitor.records) - 1 if steps is None else steps,
        mutant_id=fault.spec.fault_id if fault is not None else None,
        fault_activations=fault.activation_count if fault is not None else 0,
        elapsed_seconds=float(elapsed_seconds),
    )


def trace_evidence_sequence(
    evidence_sequence: list[StateEvidence], *, test_case: dict[str, object]
) -> PathEpisodeResult:
    monitor = CoverageMonitor()
    for step, evidence in enumerate(evidence_sequence):
        monitor.observe(step, evidence)
    result = _result_from_monitor(monitor, test_case=dict(test_case))
    if evidence_sequence:
        final = evidence_sequence[-1]
        result.red_alive = final.red_alive
        result.blue_alive = final.blue_alive
        if final.terminal_tag == "red_victory":
            result.winner = "red"
        elif final.terminal_tag == "red_failure":
            result.winner = "blue"
        elif final.terminal_tag == "timeout":
            result.winner = "draw"
    return result


def _near_obstacle_or_boundary(
    positions: dict[str, tuple[float, float]], walls: list[tuple[int, int]], map_size: int
) -> bool:
    for x, y in positions.values():
        if x <= 2 or y <= 2 or x >= map_size - 3 or y >= map_size - 3:
            return True
        if any(max(abs(x - wall_x), abs(y - wall_y)) <= 2 for wall_x, wall_y in walls):
            return True
    return False


@torch.no_grad()
def run_path_episode(
    model: SharedActorCritic,
    config: ScenarioConfig,
    device: str = "cpu",
    fault: FaultRuntime | None = None,
) -> PathEpisodeResult:
    started = time.perf_counter()
    torch.manual_seed(config.environment_seed)
    rng = np.random.default_rng(config.environment_seed + 400_009)
    env = make_scenario_env(config)
    observations, _ = env.reset(seed=config.environment_seed)
    monitor = CoverageMonitor()
    covered_scenarios: set[str] = set()
    failures: set[str] = set()
    team_return = 0.0
    red_alive, blue_alive = _alive_counts(env)
    previous_red_alive = red_alive
    previous_hp = red_hp_or_empty(env, red_alive, blue_alive)
    no_contact_streak = 0
    recovery_streak = 0
    has_seen_m5 = False
    steps = 0
    monitor.observe(0, StateEvidence(initialized=True, red_alive=red_alive, blue_alive=blue_alive))

    try:
        while env.agents:
            steps += 1
            red_names = [name for name in env.agents if name.startswith("red_")]
            blue_names = [name for name in env.agents if name.startswith("blue_")]
            red_positions = _team_positions(env, 0, red_names)
            blue_positions = _team_positions(env, 1, blue_names)
            enemy_visible = any(
                float(np.asarray(observations[name])[..., 3].sum()) > 0 for name in red_names
            )
            no_contact_streak, no_contact_covered = updates_no_contact_coverage(
                no_contact_streak, enemy_visible
            )
            if no_contact_covered:
                covered_scenarios.add("S01")
            if enemy_visible:
                covered_scenarios.add("S02")

            attack_opportunity, focus_opportunity = _attack_opportunities(
                red_positions, blue_positions
            )
            if attack_opportunity:
                covered_scenarios.add("S03")
            if focus_opportunity:
                covered_scenarios.add("S04")
            obstacle_engagement = enemy_visible and _near_obstacle_or_boundary(
                red_positions, env.scenario_walls, config.map_size
            )
            if obstacle_engagement:
                covered_scenarios.add("S05")
            local_disadvantage = has_local_numerical_disadvantage(
                red_positions, blue_positions, radius=6, margin=2
            )
            if local_disadvantage:
                covered_scenarios.add("S06")

            actions: dict[str, int] = {}
            if red_names:
                policy_observations = (
                    fault.mutate_policy_observations(observations, red_names)
                    if fault is not None
                    else observations
                )
                batch = torch.as_tensor(
                    np.stack([policy_observations[name] for name in red_names]), device=device
                )
                logits, _ = model(batch)
                logits = apply_action_mask(logits, batch)
                selected = (
                    logits.argmax(dim=-1)
                    if config.deterministic
                    else Categorical(logits=logits).sample()
                )
                actions.update(
                    {name: int(action) for name, action in zip(red_names, selected.cpu().numpy())}
                )

            if config.blue_controller == "random":
                actions.update(
                    {
                        name: _random_legal_action(np.asarray(observations[name]), rng)
                        for name in blue_names
                    }
                )
            elif config.blue_controller == "shared_model" and blue_names:
                batch = torch.as_tensor(
                    np.stack([observations[name] for name in blue_names]), device=device
                )
                logits, _ = model(batch)
                logits = apply_action_mask(logits, batch)
                selected = (
                    logits.argmax(dim=-1)
                    if config.deterministic
                    else Categorical(logits=logits).sample()
                )
                actions.update(
                    {name: int(action) for name, action in zip(blue_names, selected.cpu().numpy())}
                )
            elif config.blue_controller == "rule_based":
                actions.update(
                    _rule_based_blue_actions(env, config, observations, blue_names, red_names)
                )

            if fault is not None:
                actions = fault.mutate_actions(actions)
            next_observations, rewards, _, _, _ = env.step(actions)
            if fault is not None:
                rewards = fault.mutate_rewards(rewards)
            team_return += sum(
                float(value) for name, value in rewards.items() if name.startswith("red_")
            )
            red_alive, blue_alive = _alive_counts(env)
            hp = red_hp_or_empty(env, red_alive, blue_alive)
            low_health_fraction = float(np.mean(hp < 0.4)) if len(hp) else 0.0
            if low_health_fraction >= 0.25:
                covered_scenarios.add("S07")
            recovered_health = (
                red_alive == previous_red_alive
                and len(hp) > 0
                and len(previous_hp) > 0
                and hp.sum() > previous_hp.sum() + 1e-7
            )
            if recovered_health:
                covered_scenarios.add("S08")
            continued_after_death = (
                red_alive < previous_red_alive and red_alive > 0 and blue_alive > 0
            )
            if continued_after_death:
                covered_scenarios.add("S09")

            is_m5 = local_disadvantage or low_health_fraction >= 0.25
            has_seen_m5 = has_seen_m5 or is_m5
            recovered_risk = (
                has_seen_m5
                and not is_m5
                and red_alive > 0
                and blue_alive > 0
            )
            recovery_streak = recovery_streak + 1 if recovered_risk else 0

            observation_valid = all(
                np.asarray(value).shape == (13, 13, 5) and np.isfinite(value).all()
                for value in next_observations.values()
            )
            failures.update(
                evaluate_oracle(
                    list(actions.values()), list(rewards.values()), observation_valid, True
                )
            )
            reached_limit = steps >= config.max_cycles and red_alive > 0 and blue_alive > 0
            terminated = red_alive == 0 or blue_alive == 0
            terminal_tag = None
            if red_alive > 0 and blue_alive == 0:
                terminal_tag = "red_victory"
            elif red_alive == 0:
                terminal_tag = "red_failure"
            elif reached_limit:
                terminal_tag = "timeout"

            evidence = StateEvidence(
                no_contact_steps=no_contact_streak,
                enemy_visible=enemy_visible,
                legal_attack=attack_opportunity,
                coordinated_attack=focus_opportunity,
                obstacle_engagement=obstacle_engagement,
                local_disadvantage=local_disadvantage,
                low_health_fraction=low_health_fraction,
                recovery_steps=recovery_streak,
                continued_after_red_death=continued_after_death,
                red_alive=red_alive,
                blue_alive=blue_alive,
                terminated=terminated,
                truncated=reached_limit,
                terminal_tag=terminal_tag,
                oracle_failures=tuple(sorted(failures)),
            )
            if classify_state_flags(evidence):
                monitor.observe(steps, evidence)

            observations = next_observations
            previous_red_alive = red_alive
            previous_hp = hp

        reached_limit = steps >= config.max_cycles and red_alive > 0 and blue_alive > 0
        covered_scenarios.update(determine_terminal_scenarios(red_alive, blue_alive, reached_limit))
        terminal_sync = reached_limit or red_alive == 0 or blue_alive == 0
        failures.update(evaluate_oracle([], [], True, terminal_sync))
    finally:
        env.close()

    if not math.isfinite(team_return):
        failures.add("数值输出异常")
    result = _result_from_monitor(
        monitor,
        test_case={
            **config.to_dict(),
            "terminal_tag": (
                "red_victory"
                if red_alive > 0 and blue_alive == 0
                else "red_failure"
                if red_alive == 0
                else "timeout"
            ),
            "red_alive": red_alive,
            "blue_alive": blue_alive,
        },
        covered_scenarios=covered_scenarios,
        team_return=team_return,
        steps=steps,
        elapsed_seconds=time.perf_counter() - started,
        fault=fault,
    )
    result.oracle_failures = sorted(failures)
    result.winner = (
        "red"
        if red_alive > 0 and blue_alive == 0
        else "blue"
        if blue_alive > 0 and red_alive == 0
        else "draw"
    )
    result.red_alive = red_alive
    result.blue_alive = blue_alive
    return result
