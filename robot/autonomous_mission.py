#!/usr/bin/env python3
"""
================================================================================
 autonomous_mission.py  --  ROS 2 node for autonomous railway door inspection
================================================================================
Runs on the robot's Intel NUC (ROS 2 Humble). One process, four threads:

  * vision   : RealSense D435i colour + aligned depth -> YOLOE-26s detector ->
               door tracking (strict acquisition gate, loose lock-on tracking),
               defect alerts to Telegram, annotated MP4 recording.
  * control  : mission state machine
               IDLE -> STOW -> SEARCH (rotate / patrol) -> APPROACH (camera +
               LiDAR fused servo to a 1.0 m standoff) -> ARRIVED -> INSPECT
               (UR5e traces the door rectangle) -> DONE / FAILED
  * ROS spin : /scan (LiDAR), /joint_states, arm FollowJointTrajectory action
  * web UI   : Flask on :5000 -- live MJPEG video + START / ABORT / Step / NEXT
               from any phone or laptop on the same network.

Safety layers (in priority order):
  1. ABORT button stops the base and the arm sequence at any time.
  2. Wide LiDAR fan: anything within OBSTACLE_STOP -> HOLD.
  3. Camera/LiDAR disagreement -> intruder -> HOLD (debounced).
  4. Too close to the door -> HOLD.  5. Arm never starts until the path is clear.
  6. UR5e pendant must be in Remote mode with external_control running (manual).

Configuration: the tunables below. Paths and secrets come from environment
variables (see .env.example); nothing secret is stored in this file.
Engineering history (v1 -> v3 field fixes): docs/engineering_notes.md
================================================================================
"""

import datetime
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, redirect
from ultralytics import YOLO

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Make the hardware-independent core importable without `pip install`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from door_inspection.arm import INSPECT_SEQUENCE, JOINT_NAMES, POSES, arm_move_time  # noqa: E402
from door_inspection.fusion import fuse_distance  # noqa: E402
from door_inspection.notify import TelegramNotifier  # noqa: E402
from door_inspection.perception import door_candidate_ok, robust_door_depth  # noqa: E402

# =============================================================================
#  CONFIG
# =============================================================================

# ---- vision ----
MODEL_PATH    = os.environ.get("MODEL_PATH", str(REPO_ROOT / "weights" / "best.pt"))
CLASS_NAMES   = ["Door_Open", "Door_closed", "Door_not_closed", "Rubber_missing"]
DOOR_CLASSES  = {"Door_Open", "Door_closed", "Door_not_closed"}
CONF_THRESH   = 0.45        # model inference floor (also governs Rubber_missing alerts)

# ---- door-detection gate (stops lab clutter being seen as a "door") ----
# Applied ONLY when ACQUIRING a new door (not yet locked). Once locked, tracking
# is loose so the door is not dropped as its confidence dips / it fills the frame.
DOOR_CONF_THRESH     = 0.45   # acquire a door above this
DOOR_MIN_H_FRAC      = 0.15   # box height >= 15% of the frame (rejects tiny specks)
DOOR_MIN_AREA_FRAC   = 0.030  # box area   >= 3% of the frame
DOOR_MAX_ASPECT      = 1.50   # width/height <= this (full-frame door = 1.33, so this passes)
DOOR_DEBOUNCE_FRAMES = 3      # must persist this many frames before we ACQUIRE it

ALERT_CLASSES  = ["Door_Open", "Rubber_missing"]
ALERT_COOLDOWN = 15
SAVE_DIR       = os.environ.get("SAVE_DIR", str(Path.home() / "inspection_runs"))
IMG_W, IMG_H   = 640, 480

# ---- standoff ----
SAFE_STOP_DISTANCE = 1.0     # m: stop the base this far from the door

# ---- depth sampling (robust: open doors OK, floor in the box rejected) ----
DEPTH_MIN, DEPTH_MAX = 0.30, 9.0
DEPTH_GRID_X, DEPTH_GRID_Y = 9, 7   # samples across the door box
NEAR_FRACTION = 0.40                # use nearest 40% of returns = door panel
DEPTH_BOX_TOP_FRAC = 0.08           # skip top 8% of the box (header/ceiling bleed)
DEPTH_BOX_BOT_FRAC = 0.30           # skip bottom 30% of the box (floor creeps in up close)
DEPTH_MAX_V_FRAC   = 0.88           # never sample below this image row at all

# ---- approach servo ----
KP_LIN   = 0.90            # distance error -> forward speed
KP_ANG   = 1.0
MAX_VX   = 0.60            # top forward speed toward the door (m/s)
MIN_VX   = 0.06
CREEP_VX = 0.12            # forward floor while badly off-centre (keeps door in view)
MAX_WZ   = 0.50
MIN_WZ   = 0.12
CENTRE_TOL = 0.10          # |err| under this while moving = drive straight at the door
TURN_SIGN  = -1.0          # if it turns the WRONG way (door slides off), use +1.0
ARRIVE_CONFIRM = 4

# ---- final centring at the door (arrive squarely, no left-right wag) ----
ARRIVE_ALIGN_TOL = 0.08    # must be squarer than this to call it "arrived & centred"
ARRIVE_DIST_TOL  = 0.12    # m of slack on the standoff while doing the final centre
KP_ANG_FINE = 0.55         # gentle gain for centring in place (no MIN_WZ floor -> no wag)
MAX_WZ_FINE = 0.25
ANG_DEADBAND = 0.025       # inside this heading error, command ZERO turn (kills hunting)
APPROACH_TIMEOUT = 120.0
LOST_TIMEOUT = 8.0         # s with NO door detected AND no usable range before giving up
TARGET_LOCK_PIX = 220      # commit to the door nearest the last one, within this many px
LOCK_TTL        = 8.0      # s: keep the lock alive (don't hop to a door on the other side)

# ---- search ----
SCAN_OMEGA = 0.40
SWEEP_TIME = (2 * math.pi / SCAN_OMEGA) + 3.0
PATROL_VX  = 0.20
PATROL_TIME = 4.0
MAX_PATROL = 3

# ---- lidar obstacle gate ----
CMD_VEL_TOPIC  = "/panther/cmd_vel"
SCAN_TOPIC     = "/scan"
FRONT_OFFSET_DEG = 0.0
FRONT_HALF_DEG   = 22.0
OBSTACLE_STOP    = 0.55
OBSTACLE_MIN_VALID = 0.20   # ignore returns closer than this (robot's own body / noise)

# ---- approach: lidar/vision fusion + intruder rejection ----
FRONT_HALF_DEG_NARROW = 10.0   # tight straight-ahead sector = distance to the door / intruder
INTRUDER_JUMP   = 0.30         # m closer-than-expected (debounced) = a person stepped in -> HOLD
INTRUDER_DEBOUNCE = 2          # frames the intruder must persist before we hold (kills lidar noise)
VISION_SHORT_GUARD = 0.35      # camera reads this much CLOSER than lidar -> floor/step in the box
COAST_FARTHER_TOL = 0.50       # lidar reads FARTHER than camera/prediction (open doorway / noise)
PRE_INSPECT_CLEAR = 0.20       # m below standoff that must be clear before the arm may start
OPEN_VISION_STALE = 1.0        # s: OPEN door + no fresh camera distance -> never drive blind

# ---- OPTIONAL best-effort sidestep (leave OFF for demos) ----
ENABLE_SIDESTEP = False        # True = try to arc around a long-standing blocker
SIDESTEP_AFTER  = 6.0          # s blocked before an arc attempt
SIDESTEP_TIME   = 2.5          # s of arcing per attempt

# ---- arm ----
ARM_ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
ARM_JOINT_SPEED    = 0.90      # rad/s budget for the fastest joint
ARM_TIME_PAD       = 0.50      # s added to every move for accel/decel
ARM_MIN_MOVE_TIME  = 1.6       # s floor  (never snap)
ARM_MAX_MOVE_TIME  = 6.0       # s ceiling (also used when the start pose is unknown)
ARM_TIMEOUT   = 30.0
INSPECT_DWELL = 0.5            # s hold at each inspection pose -> sharp recorded frames
STEP_MODE_ARM = False          # True = pause per pose; False = smooth run (toggle from web UI)
SETTLE_SECONDS = 1.5           # base settle before the arm moves
STOW_POSE_NAME = "stow_forward"

# ---- web UI ----
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "5000"))

# =============================================================================
#  SHARED STATE
# =============================================================================
os.makedirs(SAVE_DIR, exist_ok=True)
app = Flask(__name__)
model = YOLO(MODEL_PATH)
tg = TelegramNotifier(alert_classes=ALERT_CLASSES, cooldown_s=ALERT_COOLDOWN)

latest_frame = None
frame_lock = threading.Lock()

det_lock = threading.Lock()
det = {"seen": False, "u": 0, "v": 0, "depth": None, "name": "", "t": 0.0}
last_target = {"u": None, "v": None, "t": 0.0}   # for door locking
door_streak = 0                                   # consecutive trusted-door frames (debounce)

state_lock = threading.Lock()
sm_state  = "IDLE"
sm_status = "idle -- press START"
run_requested   = False
mission_active  = False           # True while a mission is running -> extra START presses ignored
abort_requested = False
step_mode       = STEP_MODE_ARM   # runtime: pause after each inspection pose (toggle from web UI)
step_advance    = False           # web "NEXT" pulse: advance one pose while paused

CAM_CX = IMG_W / 2.0
cam_ready = threading.Event()     # set once real intrinsics are read


def send_telegram(text, img=None):
    tg.send(text, img)


def snapshot():
    with frame_lock:
        return None if latest_frame is None else latest_frame.copy()


# =============================================================================
#  ROS NODE
# =============================================================================
class MissionNode(Node):
    def __init__(self):
        super().__init__("autonomous_mission")
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._scan_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.scan = None
        self.joint_pos = None            # dict name -> position (from /joint_states)
        self.arm = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self.arm_done = True
        self.arm_success = False

    def _scan_cb(self, msg):
        self.scan = msg

    def _js_cb(self, msg):
        self.joint_pos = dict(zip(msg.name, msg.position))

    def current_joints(self):
        """Actual arm joints in JOINT_NAMES order, or None if not yet received."""
        jp = self.joint_pos
        if jp is None:
            return None
        try:
            return [jp[n] for n in JOINT_NAMES]
        except KeyError:
            return None

    def set_vel(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.cmd_pub.publish(t)

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def hard_stop(self, n=20):
        z = Twist()
        for _ in range(n):
            self.cmd_pub.publish(z)
            time.sleep(0.05)

    def sector_min_range(self, centre_deg, half_deg):
        """Min valid lidar range in a fan of +/- half_deg around centre_deg (deg)."""
        s = self.scan
        if s is None:
            return None
        half = math.radians(half_deg)
        off = math.radians(FRONT_OFFSET_DEG + centre_deg)
        rmin = math.inf
        for i, r in enumerate(s.ranges):
            if not math.isfinite(r) or r <= OBSTACLE_MIN_VALID:
                continue
            ang = s.angle_min + i * s.angle_increment
            ang = math.atan2(math.sin(ang - off), math.cos(ang - off))
            if abs(ang) <= half and r < rmin:
                rmin = r
        return rmin if math.isfinite(rmin) else None

    def front_min_range(self, half_deg=FRONT_HALF_DEG):
        """Straight-ahead min range over +/- half_deg."""
        return self.sector_min_range(0.0, half_deg)

    def send_arm(self, positions, move_time):
        if not self.arm.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server not available "
                                    "(UR driver up? pendant Remote + program playing?)")
            self.arm_done = True
            self.arm_success = False
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start = Duration(sec=int(move_time),
                                      nanosec=int((move_time % 1) * 1e9))
        goal.trajectory.points = [pt]
        self.arm_done = False
        self.arm_success = False
        fut = self.arm.send_goal_async(goal)
        fut.add_done_callback(self._arm_resp)

    def _arm_resp(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error("Arm goal REJECTED.")
            self.arm_done = True
            self.arm_success = False
            return
        gh.get_result_async().add_done_callback(self._arm_result)

    def _arm_result(self, fut):
        try:
            self.arm_success = (fut.result().result.error_code == 0)
        except Exception:
            self.arm_success = True
        self.arm_done = True


# =============================================================================
#  HELPERS
# =============================================================================
def set_state(s, msg):
    global sm_state, sm_status
    with state_lock:
        sm_state = s
        sm_status = msg
    print(f"[{s}] {msg}")


def fresh_det():
    with det_lock:
        return dict(det)


def aborted():
    with state_lock:
        return abort_requested


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _door_ok(conf, x1, y1, x2, y2):
    return door_candidate_ok(conf, x1, y1, x2, y2, IMG_W, IMG_H,
                             conf_thresh=DOOR_CONF_THRESH,
                             min_h_frac=DOOR_MIN_H_FRAC,
                             min_area_frac=DOOR_MIN_AREA_FRAC,
                             max_aspect=DOOR_MAX_ASPECT)


def _door_depth(depth_frame, x1, y1, x2, y2):
    return robust_door_depth(depth_frame.get_distance, x1, y1, x2, y2, IMG_W, IMG_H,
                             depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
                             grid_x=DEPTH_GRID_X, grid_y=DEPTH_GRID_Y,
                             near_fraction=NEAR_FRACTION,
                             box_top_frac=DEPTH_BOX_TOP_FRAC,
                             box_bot_frac=DEPTH_BOX_BOT_FRAC,
                             max_v_frac=DEPTH_MAX_V_FRAC)


def move_arm_blocking(node, name):
    tgt = POSES[name]
    mt = arm_move_time(node.current_joints(), tgt,
                       joint_speed=ARM_JOINT_SPEED, time_pad=ARM_TIME_PAD,
                       min_time=ARM_MIN_MOVE_TIME, max_time=ARM_MAX_MOVE_TIME)
    with state_lock:
        cur = sm_state
    set_state(cur if cur == "INSPECT" else "STOW", f"arm -> '{name}' ({mt:.1f}s)")
    node.send_arm(tgt, mt)
    t0 = time.time()
    while not node.arm_done and not aborted():
        if time.time() - t0 > ARM_TIMEOUT:
            print(f"    arm timeout reaching '{name}'")
            return False
        time.sleep(0.05)
    if node.arm_success:
        print(f"    reached '{name}'")
    return node.arm_success


# =============================================================================
#  VISION THREAD
# =============================================================================
def vision_loop():
    """Restart loop: a camera hiccup never silently kills detection."""
    warned = False
    while True:
        try:
            _vision_once()
        except Exception as e:
            print(f"[VISION] crashed: {e!r} -- restarting in 3 s")
            if not warned:
                warned = True
                send_telegram(f"Camera/vision error ({e}); auto-restarting. "
                              "If video stays frozen, replug the RealSense.")
            time.sleep(3.0)


def _vision_once():
    global latest_frame, CAM_CX, door_streak
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    CAM_CX = float(intr.ppx)
    cam_ready.set()
    print(f"Intrinsics fx={intr.fx:.1f} ppx={intr.ppx:.1f}  (CAM_CX={CAM_CX:.1f})")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{SAVE_DIR}/run_{ts}.mp4"
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (IMG_W, IMG_H))
    print(f"Recording to {out}")
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            img = np.asanyarray(color.get_data())
            res = model.predict(img, conf=CONF_THRESH, verbose=False)[0]
            annotated = res.plot()

            # Collect ALL door boxes the model emits (already >= CONF_THRESH).
            # The strict clutter gate is applied later, ONLY when acquiring.
            raw = []
            for b in res.boxes:
                cid = int(b.cls[0])
                name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                tg.maybe_alert(name, annotated)       # alerts on ALL boxes (incl. Rubber_missing)
                if name in DOOR_CLASSES:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0])
                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)
                    raw.append((conf, u, v, name, x1, y1, x2, y2))

            now = time.time()
            chosen = None
            locked = (last_target["u"] is not None and now - last_target["t"] < LOCK_TTL)

            if locked:
                # LOOSE TRACKING: already committed to a door -> nearest box to it.
                # Up close confidence dips and the box goes wide; staying near the
                # last position is evidence it is the same door.
                if raw:
                    near = min(raw, key=lambda c: abs(c[1] - last_target["u"]))
                    if abs(near[1] - last_target["u"]) <= TARGET_LOCK_PIX:
                        chosen = near
            else:
                # STRICT ACQUISITION: only lock onto something that really is a door.
                good = [c for c in raw if _door_ok(c[0], c[4], c[5], c[6], c[7])]
                if good:
                    chosen = max(good, key=lambda c: c[0])   # most confident real door

            if chosen is not None:
                last_target.update(u=chosen[1], v=chosen[2], t=now)

            # Debounce gates ACQUISITION only; once locked a 1-frame blink never
            # throws the door away mid-approach.
            door_streak = door_streak + 1 if chosen is not None else 0
            trusted = locked or (door_streak >= DOOR_DEBOUNCE_FRAMES)

            if chosen is not None and trusted:
                _, u, v, name, x1, y1, x2, y2 = chosen
                d = _door_depth(depth, x1, y1, x2, y2)
                cv2.circle(annotated, (u, v), 6, (0, 255, 255), -1)
                with det_lock:
                    det.update(seen=True, u=u, v=v, depth=d, name=name, t=now)
            else:
                with det_lock:
                    det.update(seen=False, depth=None, t=now)

            with state_lock:
                txt = f"[{sm_state}] {sm_status}"
            cv2.putText(annotated, txt, (8, IMG_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            writer.write(annotated)
            with frame_lock:
                latest_frame = annotated.copy()
    finally:
        writer.release()
        pipeline.stop()


# =============================================================================
#  MISSION PHASES
# =============================================================================
def phase_search(node):
    for hop in range(MAX_PATROL + 1):
        set_state("SEARCH_SCAN", f"sweeping (hop {hop}/{MAX_PATROL})")
        t0 = time.time()
        while time.time() - t0 < SWEEP_TIME and not aborted():
            node.set_vel(0.0, SCAN_OMEGA)
            d = fresh_det()
            if d["seen"] and d["depth"] is not None:
                node.stop_base()
                set_state("APPROACH", f"door {d['name']} at {d['depth']:.2f} m")
                return True
            time.sleep(0.05)
        node.stop_base()
        if aborted() or hop == MAX_PATROL:
            break
        set_state("SEARCH_PATROL", f"hop {hop+1}: creep forward")
        t0 = time.time()
        while time.time() - t0 < PATROL_TIME and not aborted():
            fr = node.front_min_range()
            if fr is not None and fr < OBSTACLE_STOP:
                node.stop_base()
                set_state("SEARCH_PATROL", f"obstacle {fr:.2f} m -- holding")
                time.sleep(0.3)
                continue
            node.set_vel(PATROL_VX, 0.0)
            time.sleep(0.05)
        node.stop_base()
    return False


def _sidestep(node):
    """OPTIONAL best-effort: blocked too long -> arc toward the more open side.
    Skid-steer cannot strafe, so this creeps + arcs; it can fail in tight spaces.
    Only runs if ENABLE_SIDESTEP=True."""
    left = node.sector_min_range(+40.0, 25.0)
    right = node.sector_min_range(-40.0, 25.0)
    lo = left if left is not None else 99.0
    ro = right if right is not None else 99.0
    sign = +1.0 if lo >= ro else -1.0          # arc toward the more open side
    set_state("APPROACH", "blocked too long -- attempting sidestep around obstacle")
    t0 = time.time()
    while time.time() - t0 < SIDESTEP_TIME and not aborted():
        f = node.front_min_range(FRONT_HALF_DEG_NARROW)
        if f is not None and f < OBSTACLE_STOP:
            node.set_vel(-0.08, sign * 0.4)    # too close: ease back while turning out
        else:
            node.set_vel(CREEP_VX, sign * 0.5)  # creep + arc around the obstacle
        time.sleep(0.05)
    node.stop_base()


def phase_approach(node):
    """Drive to the door and stop SQUARELY at SAFE_STOP_DISTANCE, then hand to the arm.

    Distance comes from door_inspection.fusion.fuse_distance (camera + LiDAR
    arbitration with intruder rejection -- see that module for the rule table).
    """
    t_start    = time.time()
    last_t     = time.time()
    door_dist  = None          # best estimate of metres to the door panel dead ahead
    last_err   = 0.0
    last_vx    = 0.0
    arrive_count    = 0
    unseen_since    = None
    intruder_streak = 0
    blocked_since   = None
    door_is_open    = False        # OPEN doors: lidar sees THROUGH the doorway
    last_vision_t   = time.time()  # last time the camera gave a real distance to the door

    while not aborted():
        now = time.time()
        dt = max(1e-3, now - last_t)
        last_t = now
        if now - t_start > APPROACH_TIMEOUT:
            node.stop_base()
            set_state("APPROACH", "timed out before standoff")
            return False

        wide   = node.front_min_range(FRONT_HALF_DEG)         # general safety fan
        narrow = node.front_min_range(FRONT_HALF_DEG_NARROW)  # straight ahead = door / intruder

        # (1) HARD SAFETY: anything very close anywhere in the front fan -> HOLD
        if wide is not None and wide < OBSTACLE_STOP:
            node.stop_base()
            last_vx = 0.0
            if last_target["u"] is not None:
                last_target["t"] = now   # keep the door lock warm
            set_state("APPROACH", f"obstacle {wide:.2f} m -- holding for clearance")
            if ENABLE_SIDESTEP:
                if blocked_since is None:
                    blocked_since = now
                if now - blocked_since > SIDESTEP_AFTER:
                    _sidestep(node)
                    blocked_since = None
            time.sleep(0.12)
            continue
        blocked_since = None

        d = fresh_det()
        seen = d["seen"]
        if seen:
            last_err = (d["u"] - CAM_CX) / (IMG_W / 2.0)
            err_x = last_err
            door_is_open = (d["name"] == "Door_Open")
            if d["depth"] is not None:
                last_vision_t = now
        else:
            last_err *= 0.85        # straighten out gradually when the box drops up close
            err_x = last_err

        pred = (door_dist - last_vx * dt) if door_dist is not None else None

        # (2) DISTANCE FUSION + INTRUDER REJECTION
        centred  = (abs(err_x) <= CENTRE_TOL)
        vision_d = d["depth"] if (seen and d["depth"] is not None) else None
        lidar_d  = narrow if (narrow is not None and centred) else None
        meas, intruder = fuse_distance(
            vision_d, lidar_d, pred,
            door_is_open=door_is_open,
            vision_stale=(now - last_vision_t > OPEN_VISION_STALE),
            intruder_jump=INTRUDER_JUMP,
            vision_short_guard=VISION_SHORT_GUARD,
            coast_farther_tol=COAST_FARTHER_TOL,
        )

        intruder_streak = intruder_streak + 1 if intruder else 0
        if intruder_streak >= INTRUDER_DEBOUNCE:
            node.stop_base()
            last_vx = 0.0
            if last_target["u"] is not None:
                last_target["t"] = now
            set_state("APPROACH", f"path blocked ({narrow:.2f} m) -- waiting to clear")
            time.sleep(0.12)
            continue

        # (3) GENUINELY LOST (no box and no usable range)
        if meas is None and not seen:
            if unseen_since is None:
                unseen_since = now
            node.stop_base()
            last_vx = 0.0
            if now - unseen_since > LOST_TIMEOUT:
                set_state("APPROACH", "lost the door -- returning to search")
                return False
            time.sleep(0.05)
            continue
        unseen_since = None

        if meas is not None:
            door_dist = meas if door_dist is None else (0.6 * meas + 0.4 * door_dist)  # light smoothing

        if door_dist is None:
            if seen:                                            # turn toward an off-centre door while edging in
                wz = clamp(TURN_SIGN * KP_ANG * err_x, -MAX_WZ, MAX_WZ)
                if abs(err_x) > CENTRE_TOL and abs(wz) < MIN_WZ:
                    wz = math.copysign(MIN_WZ, wz)
                node.set_vel(CREEP_VX, wz)
            else:
                node.set_vel(CREEP_VX, 0.0)
            last_vx = CREEP_VX
            time.sleep(0.05)
            continue

        # (4) CLOSER THAN THE STANDOFF but not flagged (belt & braces) -> HOLD
        if door_dist < SAFE_STOP_DISTANCE - 0.30:
            node.stop_base()
            last_vx = 0.0
            arrive_count = 0
            if last_target["u"] is not None:
                last_target["t"] = now
            set_state("APPROACH", f"too close ({door_dist:.2f} m) -- holding (not the standoff)")
            time.sleep(0.12)
            continue

        # (5) IN THE STANDOFF BAND -> centre squarely (if visible) then ARRIVE
        if door_dist <= SAFE_STOP_DISTANCE + ARRIVE_DIST_TOL:
            if (seen and abs(err_x) <= ARRIVE_ALIGN_TOL) or (not seen):
                node.stop_base()
                last_vx = 0.0
                arrive_count += 1
                if arrive_count >= ARRIVE_CONFIRM:
                    tag = d["name"] if seen else "door (lidar)"
                    set_state("ARRIVED", f"at {door_dist:.2f} m, centred on {tag}")
                    return True
            else:
                arrive_count = 0
                wz = clamp(TURN_SIGN * KP_ANG_FINE * err_x, -MAX_WZ_FINE, MAX_WZ_FINE)
                if abs(err_x) < ANG_DEADBAND:
                    wz = 0.0
                node.set_vel(0.0, wz)
                last_vx = 0.0
                set_state("APPROACH", f"centring  err {err_x:+.2f}  wz {wz:+.2f}")
            time.sleep(0.05)
            continue
        arrive_count = 0

        # (6) STILL APPROACHING: steer onto the door centre while driving in
        wz = clamp(TURN_SIGN * KP_ANG * err_x, -MAX_WZ, MAX_WZ)
        if abs(err_x) > CENTRE_TOL and abs(wz) < MIN_WZ:
            wz = math.copysign(MIN_WZ, wz)
        elif abs(err_x) <= ANG_DEADBAND:
            wz = 0.0
        # Forward speed scales smoothly with heading error.
        vx_full = clamp(KP_LIN * (door_dist - SAFE_STOP_DISTANCE), MIN_VX, MAX_VX)
        if not seen or abs(err_x) <= CENTRE_TOL:
            vx = vx_full
        else:
            vx = max(CREEP_VX, vx_full * (1.0 - min(1.0, abs(err_x) / 0.6)))
        node.set_vel(vx, wz)
        last_vx = vx
        set_state("APPROACH",
                  f"{(d['name'] if seen else 'door')}  {door_dist:.2f} m  "
                  f"err {err_x:+.2f}  vx {vx:.2f} wz {wz:+.2f}")
        time.sleep(0.05)
    return False


def phase_inspect(node):
    """Trace the door rectangle. If step-mode is ON (toggle from the phone), pause
    after each pose until NEXT is tapped."""
    global step_advance
    set_state("INSPECT", "tracing door rectangle")
    for name in INSPECT_SEQUENCE:
        if aborted():
            return False
        if not move_arm_blocking(node, name):
            set_state("INSPECT", f"arm failed at '{name}'")
            return False
        time.sleep(INSPECT_DWELL)          # brief hold -> sharp recorded frames
        with state_lock:
            paused = step_mode
        if paused and not aborted():
            with state_lock:
                step_advance = False
            set_state("INSPECT", f"paused at '{name}' -- tap NEXT on the phone (or turn Step off)")
            while not aborted():
                with state_lock:
                    go_next = step_advance or (not step_mode)
                if go_next:
                    break
                time.sleep(0.1)
            with state_lock:
                step_advance = False
    return True


# =============================================================================
#  CONTROL THREAD
# =============================================================================
def control_loop(node):
    global run_requested, abort_requested, mission_active
    cam_ready.wait()

    while True:
        with state_lock:
            mission_active = False                 # between missions: a new START is allowed
            go = run_requested
            run_requested = False
            abort_requested = False
        if not go:
            time.sleep(0.2)
            continue
        with state_lock:
            mission_active = True                  # running: ignore any further START presses

        set_state("STOW", f"moving arm to stow '{STOW_POSE_NAME}'")
        if not move_arm_blocking(node, STOW_POSE_NAME):
            set_state("FAILED", "arm/stow not ready -- check UR driver + pendant")
            send_telegram("Mission aborted: arm not ready (UR driver / Remote / program).")
            continue
        time.sleep(1.0)
        send_telegram("Autonomous mission started: searching for the door.")

        if not phase_search(node):
            node.stop_base()
            if aborted():
                set_state("IDLE", "aborted")
                continue
            set_state("FAILED", "door not found in hall")
            send_telegram("Door not found after full search.")
            continue
        if aborted():
            node.stop_base()
            set_state("IDLE", "aborted")
            continue

        if not phase_approach(node):
            node.stop_base()
            if aborted():
                set_state("IDLE", "aborted")
                continue
            set_state("FAILED", "could not reach standoff")
            send_telegram("Could not converge to 1 m standoff. Stopped safely.")
            continue
        if aborted():
            node.stop_base()
            set_state("IDLE", "aborted")
            continue

        set_state("ARRIVED", "base hard-stop + settle before arm moves")
        node.hard_stop(20)
        time.sleep(SETTLE_SECONDS)

        # FINAL SAFETY: never start the arm while something is closer than the door.
        while not aborted():
            nf = node.front_min_range(FRONT_HALF_DEG_NARROW)
            if nf is None or nf >= SAFE_STOP_DISTANCE - PRE_INSPECT_CLEAR:
                break
            node.stop_base()
            set_state("ARRIVED", f"clear the path ({nf:.2f} m) before inspection")
            time.sleep(0.3)
        if aborted():
            node.stop_base()
            set_state("IDLE", "aborted")
            continue

        send_telegram(f"Arrived ~{SAFE_STOP_DISTANCE:.1f} m from the door. Inspecting.",
                      snapshot())

        ok = phase_inspect(node)

        node.stop_base()
        if ok and not aborted():
            set_state("DONE", "inspection complete")
            send_telegram("Inspection complete. Annotated video saved on the NUC.", snapshot())
        elif aborted():
            set_state("IDLE", "aborted during inspection")
        else:
            set_state("FAILED", "inspection did not complete")
            send_telegram("Inspection ended early -- check UR driver / pendant Remote.")


# =============================================================================
#  WEB UI
# =============================================================================
def mjpeg():
    while True:
        f = snapshot()
        if f is not None:
            ok, buf = cv2.imencode(".jpg", f)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


BTN = "font-size:20px;padding:12px 24px;margin:4px"


@app.route("/")
def index():
    with state_lock:
        s = f"[{sm_state}] {sm_status}"
        sm_on = step_mode
    step_btn = (f'<a href="/step_off"><button style="{BTN}">Step: ON &#8594; turn OFF</button></a>'
                if sm_on else
                f'<a href="/step_on"><button style="{BTN}">Step: OFF &#8594; turn ON</button></a>')
    next_btn = (f'<a href="/next"><button style="{BTN}">NEXT pose</button></a>' if sm_on else "")
    return ('<html><head><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Panther Door Inspection</title></head>'
            '<body style="font-family:sans-serif;text-align:center;margin:12px">'
            '<h2>Panther &#8212; Autonomous Door Inspection</h2>'
            f'<p style="font-size:18px">Status: <b>{s}</b></p>'
            f'<p><a href="/start"><button style="{BTN}">START</button></a>'
            f'<a href="/abort"><button style="{BTN}">ABORT</button></a></p>'
            f'<p>{step_btn} {next_btn}</p>'
            '<p style="color:#666;font-size:14px">Step-mode pauses the arm after each pose '
            'until you tap NEXT. Leave OFF for a smooth hands-off run.</p>'
            '<img src="/video" style="width:100%;max-width:820px;border:1px solid #ccc">'
            '</body></html>')


@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start")
def start():
    global run_requested, abort_requested
    with state_lock:
        if not mission_active:                     # double-click / re-press during a run -> ignored
            run_requested = True
            abort_requested = False
    return redirect("/")


@app.route("/abort")
def abort():
    global abort_requested
    with state_lock:
        abort_requested = True
    return redirect("/")


@app.route("/step_on")
def step_on():
    global step_mode
    with state_lock:
        step_mode = True
    return redirect("/")


@app.route("/step_off")
def step_off():
    global step_mode
    with state_lock:
        step_mode = False
    return redirect("/")


@app.route("/next")
def next_pose():
    global step_advance
    with state_lock:
        step_advance = True
    return redirect("/")


# =============================================================================
#  MAIN
# =============================================================================
def _spin(node):
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def main():
    rclpy.init()
    node = MissionNode()
    threading.Thread(target=_spin, args=(node,), daemon=True).start()
    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=control_loop, args=(node,), daemon=True).start()
    send_telegram("Autonomous mission node online. Open the web UI and press START.")
    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_base()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
