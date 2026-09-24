import pytest

from magent2_experiment.scenario_config import ScenarioConfig
from magent2_experiment.scenario_geometry import (
    build_obstacles,
    build_team_positions,
    formation_offsets,
    minimum_team_distance,
    validate_geometry,
)


def test_compact_formation_is_a_three_by_four_block():
    assert formation_offsets("compact") == [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
    ]


def test_line_and_split_formations_have_twelve_unique_positions():
    line = formation_offsets("line")
    split = formation_offsets("split")
    assert len(line) == len(set(line)) == 12
    assert len(split) == len(set(split)) == 12
    assert max(y for _, y in split) - min(y for _, y in split) == 11


@pytest.mark.parametrize("distance", [2, 7, 12, 16])
def test_front_placement_has_requested_minimum_distance(distance):
    config = ScenarioConfig(minimum_distance=distance, obstacle_layout="open")
    red, blue = build_team_positions(config)
    assert minimum_team_distance(red, blue) == distance


def test_single_gap_wall_has_exact_three_cell_opening():
    walls = set(build_obstacles("single_gap", map_size=80))
    assert (40, 38) in walls
    assert (40, 39) not in walls
    assert (40, 40) not in walls
    assert (40, 41) not in walls
    assert (40, 42) in walls


def test_single_gap_front_placement_puts_teams_on_opposite_sides_of_wall():
    config = ScenarioConfig(
        minimum_distance=7,
        red_formation="compact",
        blue_formation="line",
        relative_position="front",
        obstacle_layout="single_gap",
    )
    red, blue = build_team_positions(config)
    walls = build_obstacles("single_gap")
    validate_geometry(config, walls, red, blue)
    assert max(x for x, _ in red) < 40 < min(x for x, _ in blue)
    assert set(y for _, y in red).intersection({39, 40, 41})
    assert set(y for _, y in blue).intersection({39, 40, 41})


def test_corridor_front_placement_is_inside_corridor_not_on_walls():
    config = ScenarioConfig(
        minimum_distance=7,
        red_formation="compact",
        blue_formation="compact",
        relative_position="front",
        obstacle_layout="corridor",
    )
    red, blue = build_team_positions(config)
    walls = build_obstacles("corridor")
    validate_geometry(config, walls, red, blue)
    assert all(37 < y < 42 for _, y in red + blue)


def test_validation_rejects_agent_wall_overlap():
    config = ScenarioConfig()
    red, blue = build_team_positions(config)
    with pytest.raises(ValueError, match="障碍"):
        validate_geometry(config, [red[0]], red, blue)


def test_validation_rejects_unreachable_teams():
    config = ScenarioConfig(minimum_distance=7)
    red, blue = build_team_positions(config)
    full_wall = [(40, y) for y in range(1, 79)]
    # Translate teams onto opposite sides of the complete wall while preserving separation.
    red = [(35 + x - min(px for px, _ in red), y) for x, y in red]
    blue = [(42 + x - min(px for px, _ in blue), y) for x, y in blue]
    config = ScenarioConfig(minimum_distance=4)
    with pytest.raises(ValueError, match="不可达"):
        validate_geometry(config, full_wall, red, blue)


def test_config_rejects_variable_map_or_team_size():
    with pytest.raises(ValueError, match="80"):
        ScenarioConfig(map_size=100)
    with pytest.raises(ValueError, match="12"):
        ScenarioConfig(red_count=10)
