from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .coverage_model import ALLOWED_TRANSITIONS, KEY_PATHS, STATE_ORDER


def _field(result: object, name: str, default: Any) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def cumulative_coverage(
    results: Sequence[object],
    *,
    state_universe: Iterable[str] = STATE_ORDER,
    transition_universe: Iterable[tuple[str, str]] = ALLOWED_TRANSITIONS,
    path_universe: Iterable[str] = KEY_PATHS,
) -> list[dict[str, int | float]]:
    states_all = set(state_universe)
    transitions_all = set(transition_universe)
    paths_all = set(path_universe)
    covered_states: set[str] = set()
    covered_transitions: set[tuple[str, str]] = set()
    covered_paths: set[str] = set()
    covered_scenarios: set[str] = set()
    curve: list[dict[str, int | float]] = []

    for budget, result in enumerate(results, start=1):
        covered_states.update(set(_field(result, "covered_states", [])) & states_all)
        for transition in _field(result, "transitions", []):
            edge = (str(transition["source"]), str(transition["target"]))
            if bool(transition.get("legal", False)) and edge in transitions_all:
                covered_transitions.add(edge)
        covered_paths.update(set(_field(result, "covered_paths", [])) & paths_all)
        covered_scenarios.update(_field(result, "covered_scenarios", []))
        curve.append(
            {
                "budget": budget,
                "state_count": len(covered_states),
                "state_coverage": len(covered_states) / len(states_all) if states_all else 1.0,
                "transition_count": len(covered_transitions),
                "transition_coverage": (
                    len(covered_transitions) / len(transitions_all) if transitions_all else 1.0
                ),
                "path_count": len(covered_paths),
                "key_path_coverage": len(covered_paths) / len(paths_all) if paths_all else 1.0,
                "scenario_count": len(covered_scenarios),
            }
        )
    return curve


def normalized_auc(curve: Sequence[Mapping[str, int | float]], key: str) -> float:
    if not curve:
        return 0.0
    return sum(float(point[key]) for point in curve) / len(curve)


def transition_conformance(results: Sequence[object]) -> float:
    transitions = [
        transition
        for result in results
        for transition in _field(result, "transitions", [])
    ]
    if not transitions:
        return 1.0
    return sum(bool(item.get("legal", False)) for item in transitions) / len(transitions)


def summarize_suite(
    results: Sequence[object],
    *,
    budgets: Sequence[int] = (10, 20, 30),
    state_universe: Iterable[str] = STATE_ORDER,
    transition_universe: Iterable[tuple[str, str]] = ALLOWED_TRANSITIONS,
    path_universe: Iterable[str] = KEY_PATHS,
) -> dict[str, object]:
    curve = cumulative_coverage(
        results,
        state_universe=state_universe,
        transition_universe=transition_universe,
        path_universe=path_universe,
    )
    final = curve[-1] if curve else {
        "state_coverage": 0.0,
        "transition_coverage": 0.0,
        "key_path_coverage": 0.0,
    }
    snapshots: dict[str, Mapping[str, int | float]] = {}
    for budget in budgets:
        if curve:
            snapshots[str(budget)] = curve[min(budget, len(curve)) - 1]
    first_path_budget = next(
        (int(point["budget"]) for point in curve if float(point["key_path_coverage"]) > 0),
        None,
    )
    return {
        "episode_count": len(results),
        "curve": curve,
        "state_auc": normalized_auc(curve, "state_coverage"),
        "transition_auc": normalized_auc(curve, "transition_coverage"),
        "key_path_auc": normalized_auc(curve, "key_path_coverage"),
        "final_state_coverage": float(final["state_coverage"]),
        "final_transition_coverage": float(final["transition_coverage"]),
        "final_key_path_coverage": float(final["key_path_coverage"]),
        "transition_conformance": transition_conformance(results),
        "first_path_budget": first_path_budget,
        "at_budget": snapshots,
    }


def evaluate_clean_gate(summary: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "all_states_reachable": float(summary["final_state_coverage"]) == 1.0,
        "all_key_paths_reachable": float(summary["final_key_path_coverage"]) == 1.0,
        "transition_conformance": float(summary["transition_conformance"]) >= 0.99,
        "nonzero_key_path_coverage": float(summary["final_key_path_coverage"]) > 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}
