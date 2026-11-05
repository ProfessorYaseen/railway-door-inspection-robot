import importlib.util
import math
from pathlib import Path

import pytest

from door_inspection.arm import INSPECT_SEQUENCE, JOINT_NAMES, POSES, arm_move_time

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- arm
def test_every_sequence_pose_is_defined():
    assert all(name in POSES for name in INSPECT_SEQUENCE)


def test_poses_have_six_joints_in_radians():
    for name, q in POSES.items():
        assert len(q) == len(JOINT_NAMES), name
        assert all(-2 * math.pi <= j <= 2 * math.pi for j in q[1:]), name


def test_unknown_start_is_conservative():
    assert arm_move_time(None, POSES["armed_pose"]) == 6.0


def test_small_hop_is_fast():
    t = arm_move_time(POSES["top_corner_right"], POSES["top_corner_middle"])
    assert t == pytest.approx(1.6)  # clamped to the floor


def test_move_time_scales_with_largest_joint_delta():
    cur = [0.0] * 6
    tgt = [0.0, 0.0, 2.7, 0.0, 0.0, 0.0]  # 2.7 rad / 0.9 rad/s + 0.5 s pad
    assert arm_move_time(cur, tgt) == pytest.approx(3.5)


def test_move_time_is_capped():
    assert arm_move_time([0.0] * 6, [10.0] + [0.0] * 5) == 6.0


# ---------------------------------------------------------------- training utils
def _load_train_module():
    spec = importlib.util.spec_from_file_location("train", ROOT / "training" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_to_polygon_corners():
    train = _load_train_module()
    out = train.box_to_polygon("3 0.5 0.5 0.2 0.4").split()
    assert out[0] == "3"
    xs = list(map(float, out[1::2]))
    ys = list(map(float, out[2::2]))
    assert min(xs) == pytest.approx(0.4) and max(xs) == pytest.approx(0.6)
    assert min(ys) == pytest.approx(0.3) and max(ys) == pytest.approx(0.7)


def test_polygon_lines_are_untouched():
    train = _load_train_module()
    line = "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"
    assert train.box_to_polygon(line) == line
