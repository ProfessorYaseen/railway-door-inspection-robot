import numpy as np
import pytest

from door_inspection.perception import door_candidate_ok, robust_door_depth

W, H = 640, 480


# ---------------------------------------------------------------- door gate
class TestDoorGate:
    def test_accepts_typical_door(self):
        assert door_candidate_ok(0.8, 200, 60, 380, 420, W, H)

    def test_rejects_low_confidence(self):
        assert not door_candidate_ok(0.30, 200, 60, 380, 420, W, H)

    def test_rejects_tiny_speck(self):
        assert not door_candidate_ok(0.9, 300, 200, 320, 240, W, H)

    def test_rejects_wide_object(self):
        # A cabinet / fence: wider than tall.
        assert not door_candidate_ok(0.9, 20, 200, 620, 400, W, H)

    def test_accepts_full_frame_door(self):
        # Up close the door fills the frame (aspect 640/480 = 1.33 <= 1.5).
        assert door_candidate_ok(0.9, 0, 0, 639, 479, W, H)


# ---------------------------------------------------------------- depth
def depth_fn(img):
    return lambda u, v: float(img[v, u])


class TestRobustDepth:
    def test_flat_door(self):
        img = np.full((H, W), 2.0)
        assert robust_door_depth(depth_fn(img), 200, 60, 400, 420, W, H) == pytest.approx(2.0)

    def test_ignores_floor_at_bottom_of_box(self):
        # Door at 1.5 m; bottom 25% of the image is floor reading 1.0 m.
        # A naive "nearest pixels" estimate would say 1.0 m and stop too early.
        img = np.full((H, W), 1.5)
        img[int(0.75 * H):, :] = 1.0
        d = robust_door_depth(depth_fn(img), 150, 40, 490, 479, W, H)
        assert d == pytest.approx(1.5)

    def test_open_door_uses_near_panel_not_doorway(self):
        # Open doorway: centre of the box sees far (6 m), frame/panel at the
        # left and right edges is at 1.8 m.
        img = np.full((H, W), 6.0)
        img[:, 200:240] = 1.8
        img[:, 360:400] = 1.8
        d = robust_door_depth(depth_fn(img), 200, 60, 400, 420, W, H)
        assert d == pytest.approx(1.8)

    def test_invalid_depth_returns_none(self):
        img = np.zeros((H, W))  # RealSense returns 0 for "no data"
        assert robust_door_depth(depth_fn(img), 200, 60, 400, 420, W, H) is None

    def test_degenerate_box_returns_none(self):
        img = np.full((H, W), 2.0)
        assert robust_door_depth(depth_fn(img), 300, 300, 300, 300, W, H) is None

    def test_box_is_clipped_to_image(self):
        img = np.full((H, W), 2.0)
        assert robust_door_depth(depth_fn(img), -50, -50, 900, 900, W, H) == pytest.approx(2.0)
