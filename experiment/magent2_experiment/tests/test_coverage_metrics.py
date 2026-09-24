from __future__ import annotations

import pytest

from magent2_experiment.coverage_metrics import (
    cumulative_coverage,
    evaluate_clean_gate,
    normalized_auc,
    summarize_suite,
)


def episode(
    states: list[str], transitions: list[tuple[str, str]], paths: list[str]
) -> dict[str, object]:
    return {
        "covered_states": states,
        "transitions": [
            {"source": source, "target": target, "legal": True, "guard_evidence": {}}
            for source, target in transitions
        ],
        "covered_paths": paths,
        "covered_scenarios": [],
    }


def test_cumulative_coverage_uses_suite_union_not_episode_mean() -> None:
    results = [
        episode(["M0", "M1"], [("M0", "M1")], []),
        episode(["M1", "M2"], [("M1", "M2")], ["P1"]),
    ]

    curve = cumulative_coverage(
        results,
        state_universe={"M0", "M1", "M2"},
        transition_universe={("M0", "M1"), ("M1", "M2")},
        path_universe={"P1", "P2"},
    )

    assert curve == [
        {
            "budget": 1,
            "state_count": 2,
            "state_coverage": pytest.approx(2 / 3),
            "transition_count": 1,
            "transition_coverage": 0.5,
            "path_count": 0,
            "key_path_coverage": 0.0,
            "scenario_count": 0,
        },
        {
            "budget": 2,
            "state_count": 3,
            "state_coverage": 1.0,
            "transition_count": 2,
            "transition_coverage": 1.0,
            "path_count": 1,
            "key_path_coverage": 0.5,
            "scenario_count": 0,
        },
    ]


def test_normalized_auc_is_mean_of_cumulative_rates() -> None:
    curve = [{"transition_coverage": 0.25}, {"transition_coverage": 0.75}]

    assert normalized_auc(curve, "transition_coverage") == 0.5


def test_suite_summary_reports_budget_snapshots() -> None:
    results = [
        episode(["M0"], [], []),
        episode(["M1"], [("M0", "M1")], ["P1"]),
        episode(["M2"], [("M1", "M2")], ["P2"]),
    ]

    summary = summarize_suite(
        results,
        budgets=(1, 3),
        state_universe={"M0", "M1", "M2"},
        transition_universe={("M0", "M1"), ("M1", "M2")},
        path_universe={"P1", "P2"},
    )

    assert summary["at_budget"]["1"]["key_path_coverage"] == 0.0
    assert summary["at_budget"]["3"]["key_path_coverage"] == 1.0
    assert summary["first_path_budget"] == 2


def test_clean_gate_requires_reachability_paths_and_conformance() -> None:
    complete = {
        "final_state_coverage": 1.0,
        "final_key_path_coverage": 1.0,
        "transition_conformance": 0.995,
    }
    failed = {**complete, "transition_conformance": 0.98}

    assert evaluate_clean_gate(complete)["passed"] is True
    verdict = evaluate_clean_gate(failed)
    assert verdict["passed"] is False
    assert verdict["checks"]["transition_conformance"] is False
