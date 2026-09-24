from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


STATE_ORDER = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
PRIMARY_PRECEDENCE = ("M7", "M8", "M6", "M5", "M4", "M3", "M2", "M1", "M0")

ALLOWED_TRANSITIONS = frozenset(
    {
        ("M0", "M1"),
        ("M0", "M2"),
        ("M0", "M5"),
        ("M1", "M2"),
        ("M1", "M8"),
        ("M2", "M1"),
        ("M2", "M3"),
        ("M2", "M5"),
        ("M2", "M6"),
        ("M2", "M7"),
        ("M2", "M8"),
        ("M3", "M1"),
        ("M3", "M2"),
        ("M3", "M4"),
        ("M3", "M5"),
        ("M3", "M6"),
        ("M3", "M7"),
        ("M3", "M8"),
        ("M4", "M1"),
        ("M4", "M2"),
        ("M4", "M3"),
        ("M4", "M5"),
        ("M4", "M6"),
        ("M4", "M7"),
        ("M4", "M8"),
        ("M5", "M6"),
        ("M5", "M7"),
        ("M5", "M8"),
        ("M6", "M2"),
        ("M6", "M3"),
        ("M6", "M7"),
        ("M6", "M8"),
    }
)

KEY_PATHS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "P1": (("M0", "M1", "M2", "M3", "M4", "M7"),),
    "P2": (("M0", "M1", "M2", "M3", "M5", "M6"),),
    "P3": (("M0", "M2", "M3", "M4", "M5", "M6"),),
    "P4": (("M0", "M1", "M8"),),
    "P5": (
        ("M0", "M2", "M3", "M4", "M8"),
        ("M0", "M2", "M3", "M5", "M8"),
    ),
}

TERMINAL_PATH_TAGS = {"P4": "timeout", "P5": "red_failure"}

# Keyed by (path id, variant index, last matched index).
PATH_DETOURS: Mapping[tuple[str, int, int], frozenset[str]] = {
    ("P1", 0, 4): frozenset({"M5", "M6"}),
    ("P2", 0, 3): frozenset({"M4"}),
    ("P5", 0, 3): frozenset({"M5", "M6"}),
    ("P5", 1, 2): frozenset({"M4"}),
    ("P5", 1, 3): frozenset({"M6"}),
}


@dataclass(frozen=True)
class StateEvidence:
    initialized: bool = False
    no_contact_steps: int = 0
    enemy_visible: bool = False
    legal_attack: bool = False
    coordinated_attack: bool = False
    obstacle_engagement: bool = False
    local_disadvantage: bool = False
    low_health_fraction: float = 0.0
    recovery_steps: int = 0
    continued_after_red_death: bool = False
    red_alive: int = 0
    blue_alive: int = 0
    terminated: bool = False
    truncated: bool = False
    terminal_tag: str | None = None
    oracle_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionRecord:
    step: int
    source: str
    target: str
    legal: bool
    guard_evidence: Mapping[str, object]


@dataclass(frozen=True)
class StepRecord:
    step: int
    primary_state: str
    flags: tuple[str, ...]
    oracle_failures: tuple[str, ...]


def classify_state_flags(evidence: StateEvidence) -> set[str]:
    flags: set[str] = set()
    if evidence.initialized:
        flags.add("M0")
    if evidence.no_contact_steps >= 3 and not evidence.enemy_visible:
        flags.add("M1")
    if evidence.enemy_visible:
        flags.add("M2")
    if evidence.legal_attack:
        flags.add("M3")
    if evidence.coordinated_attack or evidence.obstacle_engagement:
        flags.add("M4")
    if evidence.local_disadvantage or evidence.low_health_fraction >= 0.25:
        flags.add("M5")
    if evidence.recovery_steps >= 3 or evidence.continued_after_red_death:
        flags.add("M6")
    if evidence.terminated and evidence.red_alive > 0 and evidence.blue_alive == 0:
        flags.add("M7")
    if evidence.truncated or (evidence.terminated and evidence.red_alive == 0 and evidence.blue_alive > 0):
        flags.add("M8")
    return flags


def select_primary_state(flags: set[str]) -> str:
    for state in PRIMARY_PRECEDENCE:
        if state in flags:
            return state
    raise ValueError("state evidence did not activate any M-state")


def select_progressive_state(previous: str | None, flags: set[str]) -> str:
    if previous is None:
        return select_primary_state(flags)
    if "M7" in flags:
        return "M7"
    if "M8" in flags:
        return "M8"

    if previous == "M0":
        if "M2" in flags:
            return "M2"
        if "M1" in flags:
            return "M1"
        if "M0" in flags:
            return "M0"
        return select_primary_state(flags)
    if previous == "M1":
        if "M2" in flags:
            return "M2"
        if "M1" in flags:
            return "M1"
        return select_primary_state(flags)
    if previous == "M2":
        if "M3" in flags:
            return "M3"
        if "M5" in flags:
            return "M5"
        if "M1" in flags:
            return "M1"
        if "M2" in flags:
            return "M2"
        return select_primary_state(flags)
    if previous == "M3":
        if "M4" in flags:
            return "M4"
        if "M5" in flags:
            return "M5"
        if "M3" in flags:
            return "M3"
        if "M2" in flags:
            return "M2"
        return select_primary_state(flags)
    if previous == "M4":
        if "M5" in flags:
            return "M5"
        if "M4" in flags:
            return "M4"
        if "M3" in flags:
            return "M3"
        return select_primary_state(flags)
    if previous == "M5":
        return "M6" if "M6" in flags else "M5"
    if previous == "M6":
        if "M6" in flags:
            return "M6"
        if "M3" in flags:
            return "M3"
        if "M2" in flags:
            return "M2"
        return "M6"
    return previous


@dataclass
class _VariantTracker:
    path_id: str
    variant_index: int
    states: tuple[str, ...]
    matched_index: int = -1
    valid: bool = True
    covered: bool = False

    def observe(self, state: str, transition_legal: bool, terminal_tag: str | None) -> None:
        if not self.valid or self.covered:
            return

        if self.matched_index == -1:
            if state != self.states[0]:
                self.valid = False
                return
            self.matched_index = 0
            return

        current = self.states[self.matched_index]
        if state == current:
            return

        next_index = self.matched_index + 1
        if next_index < len(self.states) and state == self.states[next_index]:
            if not transition_legal:
                self.valid = False
                return
            self.matched_index = next_index
            if self.matched_index == len(self.states) - 1:
                required_tag = TERMINAL_PATH_TAGS.get(self.path_id)
                self.covered = required_tag is None or terminal_tag == required_tag
                if not self.covered:
                    self.valid = False
            return

        prefix_states = set(self.states[: self.matched_index + 1])
        detours = PATH_DETOURS.get((self.path_id, self.variant_index, self.matched_index), frozenset())
        if transition_legal and (state in prefix_states or state in detours):
            return
        self.valid = False


@dataclass
class CoverageMonitor:
    records: list[StepRecord] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    _trackers: list[_VariantTracker] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._trackers = [
            _VariantTracker(path_id, variant_index, states)
            for path_id, variants in KEY_PATHS.items()
            for variant_index, states in enumerate(variants)
        ]

    @property
    def covered_paths(self) -> set[str]:
        return {tracker.path_id for tracker in self._trackers if tracker.covered}

    @property
    def path_progress(self) -> dict[str, float]:
        progress: dict[str, float] = {}
        for path_id in KEY_PATHS:
            candidates = [tracker for tracker in self._trackers if tracker.path_id == path_id]
            progress[path_id] = max(
                (tracker.matched_index + 1) / len(tracker.states) if tracker.valid else 0.0
                for tracker in candidates
            )
        return progress

    def observe(self, step: int, evidence: StateEvidence) -> StepRecord:
        flags = classify_state_flags(evidence)
        previous = self.records[-1].primary_state if self.records else None
        primary = select_progressive_state(previous, flags)
        ordered_flags = tuple(state for state in STATE_ORDER if state in flags)
        record = StepRecord(step, primary, ordered_flags, evidence.oracle_failures)

        transition_legal = True
        if self.records and self.records[-1].primary_state != primary:
            source = self.records[-1].primary_state
            transition_legal = (source, primary) in ALLOWED_TRANSITIONS
            self.transitions.append(
                TransitionRecord(
                    step=step,
                    source=source,
                    target=primary,
                    legal=transition_legal,
                    guard_evidence=asdict(evidence),
                )
            )

        for tracker in self._trackers:
            tracker.observe(primary, transition_legal, evidence.terminal_tag)
        self.records.append(record)
        return record
