from __future__ import annotations

import pytest

from magent2_experiment.coverage_model import (
    CoverageMonitor,
    StateEvidence,
    classify_state_flags,
    select_primary_state,
)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (StateEvidence(initialized=True), {"M0"}),
        (StateEvidence(no_contact_steps=3), {"M1"}),
        (StateEvidence(enemy_visible=True), {"M2"}),
        (StateEvidence(enemy_visible=True, legal_attack=True), {"M2", "M3"}),
        (StateEvidence(coordinated_attack=True), {"M4"}),
        (StateEvidence(obstacle_engagement=True), {"M4"}),
        (StateEvidence(local_disadvantage=True), {"M5"}),
        (StateEvidence(low_health_fraction=0.25), {"M5"}),
        (StateEvidence(recovery_steps=3), {"M6"}),
        (StateEvidence(continued_after_red_death=True), {"M6"}),
        (StateEvidence(terminated=True, red_alive=2, blue_alive=0), {"M7"}),
        (StateEvidence(truncated=True, red_alive=2, blue_alive=2), {"M8"}),
        (StateEvidence(terminated=True, red_alive=0, blue_alive=2), {"M8"}),
    ],
)
def test_each_state_flag_has_independent_guard(evidence: StateEvidence, expected: set[str]) -> None:
    assert classify_state_flags(evidence) == expected


def test_primary_state_uses_frozen_risk_precedence() -> None:
    evidence = StateEvidence(
        enemy_visible=True,
        legal_attack=True,
        obstacle_engagement=True,
        local_disadvantage=True,
        recovery_steps=3,
    )

    assert classify_state_flags(evidence) == {"M2", "M3", "M4", "M5", "M6"}
    assert select_primary_state(classify_state_flags(evidence)) == "M6"


def test_oracle_failure_does_not_create_task_failure_state() -> None:
    evidence = StateEvidence(enemy_visible=True, oracle_failures=("reward_nan",))

    assert classify_state_flags(evidence) == {"M2"}


def test_transition_records_guard_evidence_and_simultaneous_flags() -> None:
    monitor = CoverageMonitor()
    monitor.observe(0, StateEvidence(initialized=True))
    monitor.observe(1, StateEvidence(no_contact_steps=3))
    record = monitor.observe(2, StateEvidence(enemy_visible=True, legal_attack=True))

    assert record.flags == ("M2", "M3")
    assert [(item.source, item.target, item.legal) for item in monitor.transitions] == [
        ("M0", "M1", True),
        ("M1", "M2", True),
    ]
    assert monitor.transitions[-1].guard_evidence["legal_attack"] is True


def test_primary_progression_does_not_skip_overlapping_milestones() -> None:
    monitor = CoverageMonitor()
    monitor.observe(0, StateEvidence(initialized=True))
    first = monitor.observe(
        1,
        StateEvidence(
            enemy_visible=True,
            legal_attack=True,
            coordinated_attack=True,
            local_disadvantage=True,
        ),
    )
    second = monitor.observe(
        2,
        StateEvidence(
            enemy_visible=True,
            legal_attack=True,
            coordinated_attack=True,
            local_disadvantage=True,
        ),
    )
    third = monitor.observe(
        3,
        StateEvidence(
            enemy_visible=True,
            legal_attack=True,
            coordinated_attack=True,
            local_disadvantage=True,
        ),
    )

    assert [first.primary_state, second.primary_state, third.primary_state] == ["M2", "M3", "M4"]
    assert all(item.legal for item in monitor.transitions)


def test_m5_holds_until_recovery_guard_reaches_m6() -> None:
    monitor = CoverageMonitor()
    monitor.observe(0, StateEvidence(initialized=True))
    monitor.observe(1, StateEvidence(enemy_visible=True))
    monitor.observe(2, StateEvidence(local_disadvantage=True))
    warmup = monitor.observe(3, StateEvidence(enemy_visible=True, recovery_steps=2))
    recovered = monitor.observe(4, StateEvidence(enemy_visible=True, recovery_steps=3))

    assert warmup.primary_state == "M5"
    assert recovered.primary_state == "M6"


def test_self_loops_do_not_count_as_transitions() -> None:
    monitor = CoverageMonitor()
    monitor.observe(0, StateEvidence(initialized=True))
    monitor.observe(1, StateEvidence(initialized=True))

    assert monitor.transitions == []


def evidence_for(state: str, *, terminal_tag: str | None = None) -> StateEvidence:
    cases = {
        "M0": StateEvidence(initialized=True),
        "M1": StateEvidence(no_contact_steps=3),
        "M2": StateEvidence(enemy_visible=True),
        "M3": StateEvidence(legal_attack=True),
        "M4": StateEvidence(obstacle_engagement=True),
        "M5": StateEvidence(local_disadvantage=True),
        "M6": StateEvidence(recovery_steps=3),
        "M7": StateEvidence(terminated=True, red_alive=1, blue_alive=0, terminal_tag="red_victory"),
        "M8": StateEvidence(truncated=terminal_tag == "timeout", terminated=terminal_tag != "timeout", red_alive=0, blue_alive=1, terminal_tag=terminal_tag),
    }
    return cases[state]


@pytest.mark.parametrize(
    ("path_id", "trace", "terminal_tag"),
    [
        ("P1", ["M0", "M1", "M2", "M3", "M4", "M7"], None),
        ("P2", ["M0", "M1", "M2", "M3", "M5", "M6"], None),
        ("P3", ["M0", "M2", "M3", "M4", "M5", "M6"], None),
        ("P4", ["M0", "M1", "M8"], "timeout"),
        ("P5", ["M0", "M2", "M3", "M4", "M8"], "red_failure"),
        ("P5", ["M0", "M2", "M3", "M5", "M8"], "red_failure"),
    ],
)
def test_key_path_finite_state_monitors_accept_declared_paths(
    path_id: str, trace: list[str], terminal_tag: str | None
) -> None:
    monitor = CoverageMonitor()
    for step, state in enumerate(trace):
        tag = terminal_tag if step == len(trace) - 1 else None
        monitor.observe(step, evidence_for(state, terminal_tag=tag))

    assert path_id in monitor.covered_paths


def test_key_path_monitor_allows_self_loops_and_declared_backtrack() -> None:
    monitor = CoverageMonitor()
    trace = ["M0", "M1", "M1", "M2", "M3", "M2", "M3", "M4", "M7"]
    for step, state in enumerate(trace):
        monitor.observe(step, evidence_for(state))

    assert "P1" in monitor.covered_paths


@pytest.mark.parametrize(
    ("path_id", "trace", "terminal_tag"),
    [
        ("P2", ["M0", "M1", "M2", "M3", "M4", "M5", "M6"], None),
        ("P5", ["M0", "M2", "M3", "M4", "M5", "M6", "M8"], "red_failure"),
    ],
)
def test_key_path_monitor_accepts_declared_risk_detours(
    path_id: str, trace: list[str], terminal_tag: str | None
) -> None:
    monitor = CoverageMonitor()
    for step, state in enumerate(trace):
        tag = terminal_tag if step == len(trace) - 1 else None
        monitor.observe(step, evidence_for(state, terminal_tag=tag))

    assert path_id in monitor.covered_paths


def test_key_path_monitor_rejects_arbitrary_subsequence() -> None:
    monitor = CoverageMonitor()
    trace = ["M0", "M1", "M6", "M2", "M3", "M4", "M7"]
    for step, state in enumerate(trace):
        monitor.observe(step, evidence_for(state))

    assert "P1" not in monitor.covered_paths


def test_terminal_tags_are_part_of_path_guards() -> None:
    timeout_monitor = CoverageMonitor()
    failure_monitor = CoverageMonitor()
    for step, state in enumerate(["M0", "M1", "M8"]):
        timeout_monitor.observe(step, evidence_for(state, terminal_tag="red_failure" if state == "M8" else None))
    for step, state in enumerate(["M0", "M2", "M3", "M5", "M8"]):
        failure_monitor.observe(step, evidence_for(state, terminal_tag="timeout" if state == "M8" else None))

    assert "P4" not in timeout_monitor.covered_paths
    assert "P5" not in failure_monitor.covered_paths


@pytest.mark.parametrize(
    "trace",
    [
        ["M0", "M2", "M3", "M8"],
        ["M0", "M2", "M3", "M4", "M6"],
        ["M0", "M2", "M3", "M4", "M2"],
        ["M0", "M2", "M3", "M1"],
        ["M0", "M2", "M5", "M7"],
        ["M0", "M2", "M8"],
        ["M0", "M2", "M7"],
        ["M0", "M5"],
        ["M0", "M2", "M6"],
        ["M0", "M2", "M3", "M6"],
        ["M0", "M2", "M3", "M4", "M1"],
    ],
)
def test_semantically_adjacent_recovery_disengagement_and_terminal_edges_are_legal(
    trace: list[str],
) -> None:
    monitor = CoverageMonitor()
    for step, state in enumerate(trace):
        terminal_tag = "timeout" if state == "M8" else None
        monitor.observe(step, evidence_for(state, terminal_tag=terminal_tag))

    assert all(item.legal for item in monitor.transitions)
