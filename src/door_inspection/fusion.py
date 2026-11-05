"""Camera + LiDAR distance fusion with intruder rejection.

The robot must stop exactly ``SAFE_STOP_DISTANCE`` (1.0 m) in front of the
door. Two sensors measure that distance and each fails in a different way:

* **RealSense depth (camera)** knows *which* object is the door, but up close
  the box can include the floor, and the box disappears when the door fills
  the frame.
* **2D LiDAR (straight-ahead sector)** is metrically accurate, but it does not
  know what it is looking at: a person walking in front of the robot, or the
  empty space *through* an open doorway, both look like "the door".

``fuse_distance`` arbitrates between them. It is a pure function so every
rule can be unit-tested (see ``tests/test_fusion.py``).
"""

from __future__ import annotations

from typing import Optional, Tuple


def fuse_distance(
    vision_d: Optional[float],
    lidar_d: Optional[float],
    pred: Optional[float],
    door_is_open: bool,
    vision_stale: bool,
    intruder_jump: float = 0.30,
    vision_short_guard: float = 0.35,
    coast_farther_tol: float = 0.50,
) -> Tuple[Optional[float], bool]:
    """Fuse camera and LiDAR range to the door.

    Args:
        vision_d: camera depth to the tracked door box (m) or ``None``.
        lidar_d: straight-ahead LiDAR range (m) or ``None`` (only passed when
            the robot is centred on the door, otherwise the beam hits the wall).
        pred: dead-reckoned prediction ``last_distance - v * dt`` or ``None``.
        door_is_open: last detected door class was ``Door_Open``.
        vision_stale: no fresh camera distance for ``OPEN_VISION_STALE`` seconds.
        intruder_jump: LiDAR this much closer than expected => something stepped in.
        vision_short_guard: camera this much closer than LiDAR => floor in the box.
        coast_farther_tol: LiDAR this much farther than expected => looking through
            a doorway / noise.

    Returns:
        ``(measurement, intruder)``. ``measurement`` is the distance to use this
        cycle (``None`` if nothing trustworthy). ``intruder`` is ``True`` when
        something is closer than the door and the base should HOLD.

    Decision table (both sensors valid):

    ============================================  =================  ==========
    condition                                      meaning            trust
    ============================================  =================  ==========
    lidar < camera - intruder_jump                 person cut in      HOLD
    open door and lidar > camera + coast_tol       beam thru doorway  camera
    camera < lidar - short_guard                   floor in the box   lidar
    otherwise (they agree)                         --                 lidar
    ============================================  =================  ==========
    """
    if vision_d is not None and lidar_d is not None:
        if lidar_d < vision_d - intruder_jump:
            return None, True  # something closer than the door we see
        if door_is_open and lidar_d > vision_d + coast_farther_tol:
            return vision_d, False  # lidar sees through the open doorway
        if vision_d < lidar_d - vision_short_guard:
            return lidar_d, False  # camera is reading floor/step -> lidar is the door
        return lidar_d, False  # they agree -> lidar is metric truth

    if vision_d is not None:
        return vision_d, False

    if lidar_d is not None:
        # Box dropped (usually up close) -> terminal drive-in on LiDAR.
        if pred is not None and lidar_d < pred - intruder_jump:
            return None, True  # someone cut in closer than expected
        if door_is_open and vision_stale:
            return pred, False  # OPEN door: never drive blind through the doorway
        if pred is not None and lidar_d > pred + coast_farther_tol:
            return pred, False  # reads far (doorway/noise) -> coast on prediction
        return lidar_d, False

    return pred, False  # not centred / no lidar -> coast briefly on prediction
