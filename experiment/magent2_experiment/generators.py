from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .path_model import KEY_PATHS, PATH_RISK


@dataclass(frozen=True)
class TestCase:
    __test__ = False
    environment_seed: int
    trigger_step: int
    duration: int
    affected_agents: int
    selection: str

    def as_tuple(self) -> tuple[Any, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class ParameterDomain:
    environment_seeds: tuple[int, ...] = tuple(range(1000))
    trigger_steps: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160)
    durations: tuple[int, ...] = (5, 10, 15, 20, 30)
    affected_counts: tuple[int, ...] = (1, 3, 6)
    selections: tuple[str, ...] = ("random", "frontline", "cluster")

    def contains(self, case: TestCase) -> bool:
        return (
            case.environment_seed in self.environment_seeds
            and case.trigger_step in self.trigger_steps
            and case.duration in self.durations
            and case.affected_agents in self.affected_counts
            and case.selection in self.selections
        )


COMMON_INITIAL_CASES = (
    TestCase(17, 20, 5, 1, "random"),
    TestCase(193, 60, 10, 3, "frontline"),
    TestCase(389, 100, 15, 6, "cluster"),
    TestCase(617, 140, 20, 3, "random"),
    TestCase(881, 160, 30, 6, "frontline"),
)


def aligned_trigger(event_step: int | None, domain: ParameterDomain | None = None) -> int:
    """Choose a declared trigger whose 30-step window contains or precedes an event."""
    triggers = (domain or ParameterDomain()).trigger_steps
    if event_step is None:
        return 80
    containing = [value for value in triggers if value <= event_step < value + 30]
    if containing:
        return max(containing)
    return min(triggers, key=lambda value: abs(value - event_step))


class BaseGenerator:
    def __init__(self, seed: int, domain: ParameterDomain | None = None):
        self.rng = np.random.default_rng(seed)
        self.domain = domain or ParameterDomain()

    def sample(self) -> TestCase:
        return TestCase(
            int(self.rng.choice(self.domain.environment_seeds)),
            int(self.rng.choice(self.domain.trigger_steps)),
            int(self.rng.choice(self.domain.durations)),
            int(self.rng.choice(self.domain.affected_counts)),
            str(self.rng.choice(self.domain.selections)),
        )


class RandomGenerator(BaseGenerator):
    def next_case(self, history: list[Any]) -> TestCase:
        del history
        return self.sample()


class RewardGenerator(BaseGenerator):
    def next_case(self, history: list[tuple[TestCase, float]]) -> TestCase:
        if not history:
            return self.sample()
        incumbent = min(history, key=lambda item: item[1])[0]
        values = list(incumbent.as_tuple())
        index = int(self.rng.integers(0, len(values)))
        domains = (
            self.domain.environment_seeds,
            self.domain.trigger_steps,
            self.domain.durations,
            self.domain.affected_counts,
            self.domain.selections,
        )
        alternatives = [value for value in domains[index] if value != values[index]]
        choice = self.rng.choice(alternatives)
        values[index] = choice.item() if hasattr(choice, "item") else choice
        return TestCase(*values)


class ProposedGenerator(BaseGenerator):
    last_target: str | None = None

    def _unique_related(self, case: TestCase, used: set[TestCase], target: str) -> TestCase:
        if case not in used:
            return case
        if target in {"P2", "P4", "P5"}:
            trigger_order = (case.trigger_step,) + tuple(value for value in self.domain.trigger_steps if value != case.trigger_step)
            duration_order = (30, 20, 15, 10, 5)
            count_order = (6, 3, 1)
            selection_order = (case.selection,) + tuple(value for value in self.domain.selections if value != case.selection)
            for trigger in trigger_order:
                for duration in duration_order:
                    for count in count_order:
                        for selection in selection_order:
                            candidate = TestCase(case.environment_seed, trigger, duration, count, selection)
                            if candidate not in used:
                                return candidate
        for environment_seed in self.domain.environment_seeds:
            candidate = TestCase(environment_seed, case.trigger_step, case.duration, case.affected_agents, case.selection)
            if candidate not in used:
                return candidate
        raise RuntimeError("declared parameter domain exhausted")

    def next_case(
        self,
        history: list[tuple[TestCase, dict[str, Any]]],
        path_progress: dict[str, int],
        attempts: dict[str, int],
    ) -> TestCase:
        eligible = [
            name
            for name, path in KEY_PATHS.items()
            if path_progress.get(name, 0) < len(path) and attempts.get(name, 0) < 3
        ]
        if not eligible:
            eligible = [name for name, path in KEY_PATHS.items() if path_progress.get(name, 0) < len(path)] or list(KEY_PATHS)
        target = max(
            eligible,
            key=lambda name: (
                PATH_RISK[name],
                path_progress.get(name, 0) / len(KEY_PATHS[name]),
                -attempts.get(name, 0),
            ),
        )
        self.last_target = target
        if history:
            relevant_shadow = {
                "P2": ("reward_offset", "death_delay"),
                "P4": ("attack_dropout", "observation_freeze"),
                "P5": ("attack_dropout", "observation_freeze"),
            }.get(target, ())
            event_field = {
                "P2": "first_lifecycle_step",
                "P4": "first_attack_step",
                "P5": "first_attack_step",
            }.get(target, "first_enemy_step")
            case, incumbent_result = max(
                history,
                key=lambda item: (
                    item[1].get(event_field) is not None,
                    any(bool(item[1].get("shadow_conditions", {}).get(name)) for name in relevant_shadow),
                    int(item[1].get("path_progress", {}).get(target, 0)),
                    -int(item[1].get(event_field) or 10_000),
                ),
            )
        else:
            case = self.sample()
            incumbent_result = {}
        if target in {"P4", "P5"}:
            case = TestCase(
                case.environment_seed,
                aligned_trigger(incumbent_result.get("first_attack_step")),
                30,
                6,
                str(self.rng.choice(("frontline", "cluster"))),
            )
        elif target == "P2":
            case = TestCase(
                case.environment_seed,
                aligned_trigger(incumbent_result.get("first_lifecycle_step")),
                30,
                6,
                "frontline",
            )
        elif target == "P3":
            case = TestCase(case.environment_seed, 40, 20, 6, "frontline")
        elif target == "P1":
            case = TestCase(
                int(self.rng.choice(self.domain.environment_seeds)),
                case.trigger_step,
                case.duration,
                case.affected_agents,
                case.selection,
            )
        return self._unique_related(case, {item[0] for item in history}, target)
