from magent2_experiment.scenario_monitor import (
    has_local_numerical_disadvantage,
    updates_no_contact_coverage,
)


def test_no_contact_requires_three_consecutive_steps():
    streak = 0
    covered = False
    for enemy_visible in (False, True, False):
        streak, newly_covered = updates_no_contact_coverage(streak, enemy_visible)
        covered = covered or newly_covered
    assert streak == 1
    assert not covered


def test_three_consecutive_steps_without_contact_cover_s01():
    streak = 0
    covered = False
    for enemy_visible in (False, False, False):
        streak, newly_covered = updates_no_contact_coverage(streak, enemy_visible)
        covered = covered or newly_covered
    assert streak == 3
    assert covered


def test_local_disadvantage_uses_agents_within_six_cells_not_global_counts():
    red = {"red_0": (10, 10), "red_1": (40, 40)}
    blue = {
        "blue_0": (11, 10),
        "blue_1": (12, 10),
        "blue_2": (13, 10),
        "blue_3": (60, 60),
    }
    assert has_local_numerical_disadvantage(red, blue, radius=6, margin=2)


def test_local_disadvantage_requires_blue_minus_red_at_least_two():
    red = {"red_0": (10, 10), "red_1": (11, 10)}
    blue = {"blue_0": (12, 10), "blue_1": (13, 10), "blue_2": (14, 10)}
    assert not has_local_numerical_disadvantage(red, blue, radius=6, margin=2)
