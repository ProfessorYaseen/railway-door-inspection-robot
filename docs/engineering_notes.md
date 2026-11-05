# Engineering notes: from v1 to v3

These notes record what broke in field tests, why it broke, and how each problem was fixed.
The fixes live in `src/door_inspection/` and have tests in `tests/`.

---

## 1. Robot stopped at about 1.5 m instead of the 1.0 m standoff

**Symptom.** At the end of the approach the robot reported "arrived at 1.00 m", but a tape measure showed about 1.5 m.

**Root cause.** Distance comes from the RealSense depth *inside the door's bounding box*. To handle open doors (see §3), the sampler keeps only the **nearest 40%** of depth samples. Up close, the box grows until its bottom edge includes the **floor** in front of the robot. The floor is nearer than the door, so the "nearest 40%" became floor pixels, and the reading said 1.0 m at a real 1.5 m.

**Fix** (`perception.robust_door_depth`):

* skip the bottom 30% of the box and the top 8% (ceiling and header bleed);
* never sample below 88% of the image height;
* if that leaves fewer than 6 valid samples (a very short or open box), fall back to the full box;
* cross-check with LiDAR: if the camera reads much **closer** than LiDAR (more than 0.35 m), assume floor contamination and trust LiDAR.

Tested in `tests/test_perception.py::test_ignores_floor_at_bottom_of_box`.

## 2. Door "lost" at about 2 m

**Symptom.** As the robot closed in, detection confidence dropped and the box disappeared, so the robot stopped or turned back to searching.

**Root cause.** When the door fills the frame there is no context left around it. The detector was trained mostly on doors with surroundings visible.

**Fix.**

* **Two-stage tracking.** *Acquiring* a door is strict (confidence ≥ 0.45, height ≥ 15% of frame, area ≥ 3%, aspect ≤ 1.5, seen for 3 frames in a row). Once a door is *locked*, tracking is loose: the nearest box within 220 px of the last position is accepted, whatever its confidence, for up to 8 s.
* **Terminal drive-in on LiDAR.** When the box drops while the robot is centred, the last stretch is driven on the straight-ahead LiDAR range, checked against a dead-reckoned prediction `d - v·dt`.
* Heading error decays (`err *= 0.85`) rather than snapping to zero, so the robot straightens out smoothly.

## 3. Open door: LiDAR sees through the doorway

**Symptom (found by simulation during the code clean-up).** With an open door, the LiDAR beam passes through the doorway and reads several metres farther than the door frame.

**Root cause.** v3 had two rules for "camera closer than LiDAR":

```
camera << lidar  -> floor in the box     -> trust LiDAR    (threshold 0.35 m)
lidar  >> camera -> beam through doorway -> trust camera   (threshold 0.50 m)
```

These describe the **same geometric condition**. Because the first threshold (0.35) is smaller than the second (0.50), the "trust camera" rule could never run. In a simulated open-door approach the old logic kept driving past the 1 m standoff, and only the separate 0.55 m obstacle stop would have caught it.

**Fix** (`fusion.fuse_distance`): use the *door state* to tell the two cases apart. If the door is `Door_Open` and LiDAR reads more than 0.5 m beyond the camera, trust the camera. Otherwise keep the floor-contamination rule. Closed-door behaviour is unchanged. Open doors also never drive in on LiDAR alone once the camera has been blind for more than 1 s.

Tested in `tests/test_fusion.py::test_open_doorway_trusts_camera` and `test_closed_door_far_lidar_still_trusts_lidar`.

## 4. A person stepping in front of the robot

If both sensors are valid and LiDAR reads **more than 0.30 m closer** than the camera's door distance, something is between the robot and the door. With only LiDAR available, the same check runs against the motion prediction. The robot HOLDs after 2 consecutive frames (debounced, so one noisy LiDAR return does not cause a stop), keeps the door lock alive, and continues once the path is clear.

Independently, anything within **0.55 m** in a ±22° front fan always causes a HOLD, and the arm will not start while anything is closer than 0.8 m straight ahead.

## 5. Speed

| | v2 | v3 |
|---|---|---|
| Max forward speed | 0.45 m/s | 0.60 m/s |
| Distance gain | 0.55 | 0.90 |
| Off-centre behaviour | hard drop to 0.12 m/s creep | speed scales with heading error |
| Settle before arm | 3.0 s | 1.5 s |
| Arm move time | fixed 6 s per pose | `max joint change / 0.9 rad/s + 0.5 s`, clamped to [1.6, 6] s |

The UR *scaled* trajectory controller also multiplies by the pendant speed slider, which must be at 100%.

## 6. Robustness

* **Vision thread** runs inside a restart loop. A RealSense USB hiccup no longer kills detection silently (which used to leave START doing nothing), and a single Telegram warning is sent.
* **Camera-ready handshake** uses a `threading.Event`. v2 checked whether the principal point had changed from 320.0, which deadlocked on a camera whose principal point really was 320.0.
* **START is idempotent.** Pressing it again during a mission is ignored.
* **Short dwell** (0.5 s) at each inspection pose, so recorded frames are sharp rather than motion-blurred.

## 7. Networking

The web UI was first opened on the NUC's wired IP, so unplugging the tether killed the video (Telegram kept working because it connects *out* to the internet). `start_robot.sh` now detects and prints the Wi-Fi IP. If a phone still cannot reach it, the Wi-Fi network is probably isolating clients from each other; put both devices on the Panther's own access point.
