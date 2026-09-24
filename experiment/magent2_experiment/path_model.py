from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


KEY_PATHS: dict[str, list[str]] = {
    "P1": ["M0", "M1", "M2", "M5", "M6"],
    "P2": ["M0", "M1", "M2", "M5", "M8"],
    "P3": ["M0", "M1", "M3", "M1", "M2", "M5", "M6"],
    "P4": ["M0", "M1", "M4", "M3", "M1"],
    "P5": ["M0", "M1", "M4", "M7"],
}

PATH_RISK = {"P2": 5, "P5": 5, "P4": 4, "P3": 3, "P1": 2}


@dataclass(frozen=True)
class StepFacts:
    initialized: bool = False
    enemy_visible: bool = False
    effective_attack: bool = False
    disadvantaged: bool = False
    fault_active: bool = False
    lifecycle_event: bool = False
    red_alive: int = 1
    blue_alive: int = 1
    oracle_failure: bool = False


def classify_state(facts: StepFacts) -> str:
    """Map observable step facts to exactly one abstract state."""
    if facts.oracle_failure:
        return "M8"
    if facts.blue_alive == 0 and facts.red_alive > 0:
        return "M6"
    if facts.red_alive == 0 and facts.blue_alive > 0:
        return "M7"
    if facts.lifecycle_event:
        return "M5"
    if facts.fault_active:
        return "M4"
    if facts.disadvantaged:
        return "M3"
    if facts.effective_attack:
        return "M2"
    if facts.enemy_visible:
        return "M1"
    return "M0"


def terminal_state(red_alive: int, blue_alive: int) -> str | None:
    """Return the explicit terminal state that follows the final M5 event."""
    if red_alive > 0 and blue_alive == 0:
        return "M6"
    if red_alive == 0 and blue_alive > 0:
        return "M7"
    return None


def coordinated_focus_fire(history: Sequence[Mapping[str, Hashable]]) -> bool:
    """Whether two distinct red agents targeted one cell within the recent window."""
    by_target: dict[Hashable, set[str]] = {}
    for step in history[-3:]:
        for agent, target in step.items():
            by_target.setdefault(target, set()).add(agent)
    return any(len(agents) >= 2 for agents in by_target.values())


def compress_states(states: Iterable[str]) -> list[str]:
    result: list[str] = []
    for state in states:
        if not result or result[-1] != state:
            result.append(state)
    return result


def path_is_covered(trace: Sequence[str], path: Sequence[str]) -> bool:
    compact = compress_states(trace)
    width = len(path)
    return any(compact[index : index + width] == list(path) for index in range(len(compact) - width + 1))


def longest_prefix(trace: Sequence[str], path: Sequence[str]) -> int:
    """Longest path prefix occurring contiguously anywhere in a compact trace."""
    compact = compress_states(trace)
    best = 0
    for start in range(len(compact)):
        matched = 0
        for actual, expected in zip(compact[start:], path):
            if actual != expected:
                break
            matched += 1
        best = max(best, matched)
    return best


def progress_by_path(trace: Sequence[str]) -> dict[str, int]:
    return {name: longest_prefix(trace, path) for name, path in KEY_PATHS.items()}
