from __future__ import annotations

from dataclasses import asdict, dataclass


FORMATIONS = ("compact", "line", "split")
RELATIVE_POSITIONS = ("front", "flank", "near_red_group_1", "encircle")
OBSTACLE_LAYOUTS = ("open", "single_gap", "double_gap", "corridor")
ENGAGEMENT_REGIONS = ("center", "top", "bottom", "left", "right")
BLUE_CONTROLLERS = ("random", "rule_based", "shared_model")
BLUE_TARGET_RULES = ("nearest", "lowest_health")
PHASE_RULES = ("constant", "retreat_after_low_hp")
RED_ACTION_RULES = ("argmax", "sample")


@dataclass(frozen=True)
class ScenarioConfig:
    case_id: str = "SC-DEFAULT"
    target_scenario: str = "S01"
    map_size: int = 80
    red_count: int = 12
    blue_count: int = 12
    minimum_distance: int = 7
    red_formation: str = "compact"
    blue_formation: str = "compact"
    relative_position: str = "front"
    obstacle_layout: str = "open"
    engagement_region: str = "center"
    blue_controller: str = "random"
    blue_target_rule: str = "nearest"
    phase_rule: str = "constant"
    red_action_rule: str = "argmax"
    max_cycles: int = 200
    environment_seed: int = 101

    def __post_init__(self) -> None:
        if self.map_size != 80:
            raise ValueError("场景实验固定使用80×80地图")
        if self.red_count != 12 or self.blue_count != 12:
            raise ValueError("场景实验固定使用红蓝双方各12个代理")
        if self.minimum_distance < 1:
            raise ValueError("双方最小距离必须大于0")
        if self.max_cycles < 1:
            raise ValueError("最大周期必须大于0")
        choices = {
            "red_formation": (self.red_formation, FORMATIONS),
            "blue_formation": (self.blue_formation, FORMATIONS),
            "relative_position": (self.relative_position, RELATIVE_POSITIONS),
            "obstacle_layout": (self.obstacle_layout, OBSTACLE_LAYOUTS),
            "engagement_region": (self.engagement_region, ENGAGEMENT_REGIONS),
            "blue_controller": (self.blue_controller, BLUE_CONTROLLERS),
            "blue_target_rule": (self.blue_target_rule, BLUE_TARGET_RULES),
            "phase_rule": (self.phase_rule, PHASE_RULES),
            "red_action_rule": (self.red_action_rule, RED_ACTION_RULES),
        }
        for field, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{field}不支持取值{value!r}，允许值为{allowed}")

    @property
    def deterministic(self) -> bool:
        return self.red_action_rule == "argmax"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
