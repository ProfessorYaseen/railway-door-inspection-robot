"""UR5e arm helpers: inspection poses and joint-delta-based move timing."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

JOINT_NAMES: List[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _rad(degrees: Sequence[float]) -> List[float]:
    return [math.radians(x) for x in degrees]


# Joint-space poses taught on the pendant (degrees -> radians).
# The inspection sequence traces the rectangle of the door with the
# wrist-mounted camera at a 1.0 m standoff.
POSES: Dict[str, List[float]] = {
    "initial_pose":         _rad([210.40, -91.71, -148.71, -8.097, 89.11, 89.81]),
    "stow_forward":         _rad([208.08, -87.67, -92.83, -3.88, 89.19, 92.72]),
    "armed_pose":           _rad([210.40, -49.17, -124.53, -9.06, 88.88, 89.81]),
    "top_looking":          _rad([210.31, -67.93, -79.32, -4.46, 89.00, 89.81]),
    "top_corner_right":     _rad([193.71, -93.14, -56.06, -13.08, 88.99, 89.81]),
    "top_corner_middle":    _rad([193.68, -98.54, -67.12, -17.23, 88.85, 89.82]),
    "bottom_corner_right":  _rad([197.20, -111.39, -82.16, -17.21, 88.81, 89.82]),
    "bottom_bottom":        _rad([208.78, -110.69, -86.16, -16.81, 88.81, 89.82]),
    "bottom_corner_left":   _rad([222.19, -108.90, -85.95, -16.82, 88.81, 89.81]),
    "bottom_corner_middle": _rad([222.08, -95.91, -80.94, -16.81, 88.80, 89.81]),
    "bottom_top":           _rad([222.09, -89.80, -55.33, -16.78, 88.81, 89.81]),
}

INSPECT_SEQUENCE: List[str] = [
    "armed_pose", "top_looking",
    "top_corner_right", "top_corner_middle",
    "bottom_corner_right", "bottom_bottom",
    "bottom_corner_left", "bottom_corner_middle", "bottom_top",
    "armed_pose", "initial_pose",
]


def arm_move_time(
    current: Optional[Sequence[float]],
    target: Sequence[float],
    joint_speed: float = 0.90,
    time_pad: float = 0.50,
    min_time: float = 1.6,
    max_time: float = 6.0,
) -> float:
    """Trajectory duration proportional to the largest joint motion.

    Small hops between neighbouring inspection poses take ~2 s instead of a
    flat 6 s, while large reconfigurations stay smooth.

    Args:
        current: current joint angles (rad) or ``None`` if unknown.
        target: target joint angles (rad).
        joint_speed: rad/s budget for the fastest-moving joint.
        time_pad: seconds added for acceleration / deceleration.
        min_time, max_time: clamp range in seconds.
    """
    if current is None:
        return max_time  # unknown start -> be conservative
    dmax = max(abs(t - c) for t, c in zip(target, current))
    return max(min_time, min(max_time, dmax / joint_speed + time_pad))
