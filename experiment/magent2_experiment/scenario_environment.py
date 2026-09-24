from __future__ import annotations

from magent2.environments.battlefield.battlefield import _parallel_env, default_reward_args

from .scenario_config import ScenarioConfig
from .scenario_geometry import build_obstacles, build_team_positions, validate_geometry


class ScenarioBattlefieldEnv(_parallel_env):
    """Battlefield with externally supplied map geometry and unchanged transition rules."""

    def __init__(self, config: ScenarioConfig, render_mode: str | None = None):
        self.scenario_config = config
        self.scenario_walls: list[tuple[int, int]] = []
        self.scenario_red_positions: list[tuple[int, int]] = []
        self.scenario_blue_positions: list[tuple[int, int]] = []
        super().__init__(
            map_size=config.map_size,
            minimap_mode=False,
            reward_args=dict(default_reward_args),
            max_cycles=config.max_cycles,
            extra_features=False,
            render_mode=render_mode,
            seed=config.environment_seed,
        )

    def generate_map(self) -> None:
        walls = build_obstacles(self.scenario_config.obstacle_layout, self.scenario_config.map_size)
        red, blue = build_team_positions(self.scenario_config)
        validate_geometry(self.scenario_config, walls, red, blue)

        self.scenario_walls = list(walls)
        self.scenario_red_positions = list(red)
        self.scenario_blue_positions = list(blue)
        if walls:
            self.env.add_walls(method="custom", pos=walls)
        self.env.add_agents(self.handles[0], method="custom", pos=[[x, y, 0] for x, y in red])
        self.env.add_agents(self.handles[1], method="custom", pos=[[x, y, 0] for x, y in blue])


def make_scenario_env(config: ScenarioConfig, render_mode: str | None = None) -> ScenarioBattlefieldEnv:
    return ScenarioBattlefieldEnv(config, render_mode=render_mode)
