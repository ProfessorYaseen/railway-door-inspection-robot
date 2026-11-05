"""Perception helpers: decide whether a detection is a real door, and how far away it is.

Both functions were extracted unchanged (behaviour-wise) from the field-tested
mission node so that they can be unit-tested without a camera.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def door_candidate_ok(
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    conf_thresh: float = 0.45,
    min_h_frac: float = 0.15,
    min_area_frac: float = 0.030,
    max_aspect: float = 1.50,
) -> bool:
    """Clutter gate used when *acquiring* a new door.

    Lab scenes contain many door-like rectangles (cabinets, fences, posters).
    A detection is only accepted as a door if it is confident enough, tall
    enough, large enough and taller-than-wide.

    Args:
        conf: detector confidence in [0, 1].
        x1, y1, x2, y2: bounding box in pixels.
        img_w, img_h: image size in pixels.
        conf_thresh: minimum confidence.
        min_h_frac: minimum box height as a fraction of image height.
        min_area_frac: minimum box area as a fraction of image area.
        max_aspect: maximum width / height ratio.
    """
    if conf < conf_thresh:
        return False
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    if h < min_h_frac * img_h:  # too short -> a speck, not a door
        return False
    if (w * h) < min_area_frac * (img_w * img_h):
        return False
    if (w / h) > max_aspect:  # wider than tall -> not a door
        return False
    return True


def robust_door_depth(
    get_distance: Callable[[int, int], float],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    depth_min: float = 0.30,
    depth_max: float = 9.0,
    grid_x: int = 9,
    grid_y: int = 7,
    near_fraction: float = 0.40,
    box_top_frac: float = 0.08,
    box_bot_frac: float = 0.30,
    max_v_frac: float = 0.88,
) -> Optional[float]:
    """Robust distance (metres) to the near surface inside a door bounding box.

    Why not just read the depth at the box centre?
      * For an OPEN door the centre pixel looks *through* the doorway, so we
        keep only the nearest ``near_fraction`` of samples (the door panel/frame).
      * Up close, the bottom of the box contains the FLOOR in front of the
        robot, which made an earlier version stop at 1.5 m instead of 1.0 m.
        The bottom ``box_bot_frac`` of the box and every image row below
        ``max_v_frac`` are therefore skipped.

    Args:
        get_distance: ``f(u, v) -> metres`` (e.g. ``rs.depth_frame.get_distance``).
        x1, y1, x2, y2: bounding box in pixels (colour-aligned depth).
        img_w, img_h: image size in pixels.

    Returns:
        Median of the nearest samples in metres, or ``None`` if too few valid
        depth returns were found.
    """
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w - 1, int(x2))
    y2 = min(img_h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    h = y2 - y1
    ys = y1 + int(box_top_frac * h)
    ye = y2 - int(box_bot_frac * h)
    ye = min(ye, int(max_v_frac * img_h))  # hard floor cut-off in image rows
    if ye <= ys:  # degenerate (very short box) -> fall back to the full box
        ys, ye = y1, y2

    def sample(sy: int, ey: int) -> list[float]:
        vals = []
        for vv in np.linspace(sy, ey, grid_y):
            for uu in np.linspace(x1, x2, grid_x):
                dd = get_distance(int(uu), int(vv))
                if depth_min < dd < depth_max:
                    vals.append(dd)
        return vals

    vals = sample(ys, ye)
    if len(vals) < 6:
        # Open door up close: the panel is only at the extremes -> retry full box.
        vals = sample(y1, y2)
    if len(vals) < 6:
        return None

    vals.sort()
    k = max(3, int(len(vals) * near_fraction))
    return float(np.median(vals[:k]))
