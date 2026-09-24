from __future__ import annotations

from collections import deque
from typing import Iterable

from .scenario_config import ScenarioConfig

Position = tuple[int, int]


def formation_offsets(name: str) -> list[Position]:
    if name == "compact":
        return [(x, y) for y in range(3) for x in range(4)]
    if name == "line":
        return [(x, 0) for x in range(12)]
    if name == "split":
        first = [(x, y) for y in range(2) for x in range(3)]
        second = [(x, y + 10) for y in range(2) for x in range(3)]
        return first + second
    raise ValueError(f"未知队形: {name}")


def _translate(points: Iterable[Position], dx: int, dy: int) -> list[Position]:
    return [(x + dx, y + dy) for x, y in points]


def _bounds(points: Iterable[Position]) -> tuple[int, int, int, int]:
    values = list(points)
    xs = [x for x, _ in values]
    ys = [y for _, y in values]
    return min(xs), max(xs), min(ys), max(ys)


def _align_y(reference: list[Position], moving: list[Position]) -> int:
    _, _, rmin, rmax = _bounds(reference)
    _, _, mmin, mmax = _bounds(moving)
    return round((rmin + rmax - mmin - mmax) / 2)


def _align_x(reference: list[Position], moving: list[Position]) -> int:
    rmin, rmax, _, _ = _bounds(reference)
    mmin, mmax, _, _ = _bounds(moving)
    return round((rmin + rmax - mmin - mmax) / 2)


def _red_anchor(region: str) -> Position:
    return {
        "center": (20, 34),
        "top": (20, 2),
        "bottom": (20, 65),
        "left": (2, 34),
        "right": (52, 34),
    }[region]


def build_team_positions(config: ScenarioConfig) -> tuple[list[Position], list[Position]]:
    red_offsets = formation_offsets(config.red_formation)
    blue_offsets = formation_offsets(config.blue_formation)
    d = config.minimum_distance

    anchor_x, anchor_y = _red_anchor(config.engagement_region)
    if config.relative_position in {"front", "near_red_group_1"} and config.obstacle_layout in {
        "single_gap", "double_gap"
    }:
        # Put the teams on opposite sides of the x=40 wall and align them with a real opening.
        _, red_width_max, red_height_min, red_height_max = _bounds(red_offsets)
        anchor_x = 39 - red_width_max
        opening_center = 40 if config.obstacle_layout == "single_gap" else 25
        anchor_y = round(opening_center - (red_height_min + red_height_max) / 2)
    elif config.relative_position == "front" and config.obstacle_layout == "corridor":
        if config.red_formation == "split" or config.blue_formation == "split":
            raise ValueError("split队形无法完整放入4格宽走廊")
        _, _, red_height_min, red_height_max = _bounds(red_offsets)
        anchor_y = round(40 - (red_height_min + red_height_max) / 2)
    red = _translate(red_offsets, anchor_x, anchor_y)

    if config.relative_position == "front":
        _, red_max_x, _, _ = _bounds(red)
        blue_min_x, _, _, _ = _bounds(blue_offsets)
        blue = _translate(blue_offsets, red_max_x + d - blue_min_x, _align_y(red, blue_offsets))
    elif config.relative_position == "flank":
        _, _, _, red_max_y = _bounds(red)
        _, _, blue_min_y, _ = _bounds(blue_offsets)
        blue = _translate(blue_offsets, _align_x(red, blue_offsets), red_max_y + d - blue_min_y)
    elif config.relative_position == "near_red_group_1":
        first_group = red[:6] if config.red_formation == "split" else red
        _, group_max_x, _, _ = _bounds(first_group)
        blue_min_x, _, _, _ = _bounds(blue_offsets)
        blue = _translate(blue_offsets, group_max_x + d - blue_min_x, _align_y(first_group, blue_offsets))
    else:  # encircle
        red_min_x, red_max_x, _, _ = _bounds(red)
        group_offsets = [(x, y) for y in range(2) for x in range(3)]
        _, group_max_x, _, _ = _bounds(group_offsets)
        left_reference = red[:6] if config.red_formation == "split" else red
        right_reference = red[6:] if config.red_formation == "split" else red
        left = _translate(
            group_offsets,
            red_min_x - d - group_max_x,
            _align_y(left_reference, group_offsets),
        )
        right = _translate(
            group_offsets,
            red_max_x + d,
            _align_y(right_reference, group_offsets),
        )
        blue = left + right

    return red, blue


def build_obstacles(layout: str, map_size: int = 80) -> list[Position]:
    if map_size != 80:
        raise ValueError("障碍模板固定按80×80地图定义")
    if layout == "open":
        return []
    if layout == "single_gap":
        return [(40, y) for y in range(10, 70) if y not in {39, 40, 41}]
    if layout == "double_gap":
        gaps = {24, 25, 26, 53, 54, 55}
        return [(40, y) for y in range(10, 70) if y not in gaps]
    if layout == "corridor":
        return [(x, 37) for x in range(20, 60)] + [(x, 42) for x in range(20, 60)]
    raise ValueError(f"未知障碍布局: {layout}")


def minimum_team_distance(red: Iterable[Position], blue: Iterable[Position]) -> int:
    red_values = list(red)
    blue_values = list(blue)
    if not red_values or not blue_values:
        raise ValueError("两队坐标不能为空")
    return min(max(abs(rx - bx), abs(ry - by)) for rx, ry in red_values for bx, by in blue_values)


def _teams_reachable(map_size: int, walls: set[Position], red: list[Position], blue: list[Position]) -> bool:
    targets = set(blue)
    queue = deque(red)
    visited = set(red)
    while queue:
        x, y = queue.popleft()
        if (x, y) in targets:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (0 < nxt[0] < map_size - 1 and 0 < nxt[1] < map_size - 1):
                continue
            if nxt in walls or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False


def validate_geometry(
    config: ScenarioConfig,
    walls: Iterable[Position],
    red: Iterable[Position],
    blue: Iterable[Position],
) -> None:
    wall_values = list(walls)
    red_values = list(red)
    blue_values = list(blue)
    if len(red_values) != config.red_count or len(blue_values) != config.blue_count:
        raise ValueError("代理数量与场景配置不一致")
    agents = red_values + blue_values
    if len(set(agents)) != len(agents):
        raise ValueError("红蓝代理位置发生重叠")
    if set(wall_values).intersection(agents):
        raise ValueError("代理位置与障碍发生重叠")
    for x, y in agents + wall_values:
        if not (0 < x < config.map_size - 1 and 0 < y < config.map_size - 1):
            raise ValueError(f"坐标越界: {(x, y)}")
    actual = minimum_team_distance(red_values, blue_values)
    if actual != config.minimum_distance:
        raise ValueError(f"双方最小距离应为{config.minimum_distance}，实际为{actual}")
    if not _teams_reachable(config.map_size, set(wall_values), red_values, blue_values):
        raise ValueError("红蓝双方在当前障碍布局下不可达")
