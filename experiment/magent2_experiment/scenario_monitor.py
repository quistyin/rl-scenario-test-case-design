from __future__ import annotations

from collections.abc import Mapping


Position = tuple[float, float]


def updates_no_contact_coverage(streak: int, enemy_visible: bool) -> tuple[int, bool]:
    """Update the consecutive no-contact count and report whether S01 is covered."""
    next_streak = 0 if enemy_visible else streak + 1
    return next_streak, next_streak >= 3


def has_local_numerical_disadvantage(
    red_positions: Mapping[str, Position],
    blue_positions: Mapping[str, Position],
    *,
    radius: int = 6,
    margin: int = 2,
) -> bool:
    """Return true when any red agent has at least ``margin`` more nearby blue agents."""
    for center_x, center_y in red_positions.values():
        local_red = sum(
            max(abs(red_x - center_x), abs(red_y - center_y)) <= radius
            for red_x, red_y in red_positions.values()
        )
        local_blue = sum(
            max(abs(blue_x - center_x), abs(blue_y - center_y)) <= radius
            for blue_x, blue_y in blue_positions.values()
        )
        if local_blue - local_red >= margin:
            return True
    return False
