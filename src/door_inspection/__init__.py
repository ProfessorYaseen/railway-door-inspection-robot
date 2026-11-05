"""Hardware-independent core of the autonomous railway door inspection robot.

Everything in this package is pure Python (NumPy only) so it can be unit-tested
on any machine without ROS 2, a RealSense camera or the UR5e arm attached.
The ROS 2 mission node in ``robot/autonomous_mission.py`` imports from here.
"""

from .arm import arm_move_time
from .fusion import fuse_distance
from .perception import door_candidate_ok, robust_door_depth

__all__ = [
    "door_candidate_ok",
    "robust_door_depth",
    "fuse_distance",
    "arm_move_time",
]

__version__ = "1.0.0"
