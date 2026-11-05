"""Every rule of the camera/LiDAR fusion table, as executable specification."""

import pytest

from door_inspection.fusion import fuse_distance


def fuse(vision, lidar, pred=None, open_=False, stale=False):
    return fuse_distance(vision, lidar, pred, door_is_open=open_, vision_stale=stale)


class TestBothSensors:
    def test_agree_trusts_lidar(self):
        assert fuse(2.05, 2.00) == (2.00, False)

    def test_person_steps_in_is_intruder(self):
        # LiDAR suddenly much closer than the door the camera is tracking.
        meas, intruder = fuse(2.0, 1.2)
        assert intruder and meas is None

    def test_floor_in_box_trusts_lidar(self):
        # Up close the camera box includes the floor -> camera reads short.
        assert fuse(1.5, 2.0) == (2.0, False)

    def test_open_doorway_trusts_camera(self):
        # Open door: the LiDAR beam passes through the doorway and reads far.
        assert fuse(1.6, 3.5, open_=True) == (1.6, False)

    def test_closed_door_far_lidar_still_trusts_lidar(self):
        # Same geometry but the door is NOT open -> floor contamination is the
        # likelier explanation, so keep trusting the metric sensor.
        assert fuse(1.6, 3.5, open_=False) == (3.5, False)

    def test_open_door_small_gap_trusts_lidar(self):
        assert fuse(1.6, 1.9, open_=True) == (1.9, False)


class TestCameraOnly:
    def test_uses_camera(self):
        assert fuse(2.4, None) == (2.4, False)


class TestLidarOnly:
    def test_terminal_drive_in_on_lidar(self):
        # Box lost up close; lidar consistent with prediction -> use lidar.
        assert fuse(None, 1.10, pred=1.15) == (1.10, False)

    def test_intruder_against_prediction(self):
        meas, intruder = fuse(None, 0.6, pred=1.2)
        assert intruder and meas is None

    def test_open_door_never_drives_blind(self):
        assert fuse(None, 3.0, pred=1.4, open_=True, stale=True) == (1.4, False)

    def test_reads_far_coasts_on_prediction(self):
        assert fuse(None, 2.5, pred=1.5) == (1.5, False)

    def test_no_prediction_uses_lidar(self):
        assert fuse(None, 1.8, pred=None) == (1.8, False)


class TestNoSensors:
    def test_coasts_on_prediction(self):
        assert fuse(None, None, pred=1.3) == (1.3, False)

    def test_nothing_at_all(self):
        assert fuse(None, None, pred=None) == (None, False)


@pytest.mark.parametrize("vision,lidar", [(1.0, 1.0), (3.0, 3.1), (5.0, 4.8)])
def test_never_flags_intruder_when_sensors_agree(vision, lidar):
    _, intruder = fuse(vision, lidar)
    assert not intruder
