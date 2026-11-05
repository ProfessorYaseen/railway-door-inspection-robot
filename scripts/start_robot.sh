#!/usr/bin/env bash
# =============================================================================
#  start_robot.sh  --  launch the full autonomous robot stack on the Intel NUC.
# =============================================================================
#  Starts LiDAR + UR5e driver + the mission node in a background tmux session,
#  then returns your prompt. The robot keeps running after you close SSH.
#  Control the mission from any browser on the same network:
#      http://<NUC-WIFI-IP>:5000   ->  START / ABORT  (and Step / NEXT)
#
#  USAGE (on the NUC, from the repository root):
#      cp .env.example .env && nano .env      # first time: add Telegram token etc.
#      ./scripts/start_robot.sh
#
#  NETWORKING: connect over the NUC's Wi-Fi IP, not the wired 10.15.20.x
#  address -- otherwise the live video dies when the cable is unplugged.
#  If a phone cannot load the page, the network probably uses client
#  isolation: join both devices to the Panther's own access point.
#
#  PENDANT (manual by design -- a physical safety step):
#      1. Load external_control.urp   2. Press play   3. Set Remote
#
#  Logs:  tmux attach -t demo   (Ctrl+b then d to detach)
#  Stop:  tmux kill-session -t demo
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-demo}"

# Load secrets / settings from .env (never committed to git).
if [ -f "$REPO_DIR/.env" ]; then
  set -a; source "$REPO_DIR/.env"; set +a
else
  echo "  [!] No .env found -- Telegram alerts disabled. See .env.example."
fi
ROBOT_IP="${ROBOT_IP:-10.15.20.50}"      # UR5e arm IP -- override in .env
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

# Auto-detect the Wi-Fi IP (anything that is NOT the wired/loopback/docker range).
WIFI_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' \
  | grep -vE '^(10\.15\.20\.|127\.|169\.254\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.)' \
  | head -n1 || true)

tmux kill-session -t "$SESSION" 2>/dev/null || true

# Window 0: LiDAR
tmux new-session -d -s "$SESSION" -n lidar
tmux send-keys -t "$SESSION:lidar" "source $ROS_SETUP && ros2 run rplidar_ros rplidar_node" C-m

# Window 1: UR5e driver  (then PENDANT: load program -> play -> Remote)
tmux new-window -t "$SESSION" -n ur
tmux send-keys -t "$SESSION:ur" \
  "source $ROS_SETUP && ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=$ROBOT_IP launch_rviz:=false" C-m

# Window 2: mission node (waits 8 s so LiDAR/driver come up first).
# Environment from .env is inherited by tmux windows created after it was sourced.
tmux new-window -t "$SESSION" -n mission
tmux send-keys -t "$SESSION:mission" \
  "cd $REPO_DIR && source $ROS_SETUP && set -a && [ -f .env ] && source .env; set +a; sleep 8; python3 robot/autonomous_mission.py" C-m

echo ""
echo "  ============================================================"
echo "  Robot stack launching in tmux '$SESSION'"
echo "    win0 = lidar | win1 = UR driver | win2 = mission"
echo "  ============================================================"
echo ""
echo "  ALL NUC IPs:  $(hostname -I 2>/dev/null)"
if [ -n "$WIFI_IP" ]; then
  echo "  >> OPEN THIS (laptop + phone): http://$WIFI_IP:5000"
  echo "  >> SSH over Wi-Fi:              ssh <user>@$WIFI_IP"
else
  echo "  >> Run 'hostname -I' and use the IP that is NOT 10.15.20.x -> http://<IP>:5000"
fi
cat <<'EOF'

  PRE-FLIGHT CHECKLIST
   [ ] Mission window printed 'Intrinsics fx=... ppx=...' (RealSense up).
       If not, START does nothing -- replug the camera.
   [ ] Pendant: external_control.urp loaded -> play -> Remote; speed slider 100%.
   [ ] Web UI loads on the PHONE (if not: client isolation on the Wi-Fi).
   [ ] Robot 3-5 m from the train, square-on, clear line of sight.
   [ ] Step-mode OFF for a smooth run; ON to step the arm pose-by-pose with NEXT.
   [ ] Watch the APPROACH line: the metres must FALL to ~1.00.
   [ ] Off-centre door slides off screen? flip TURN_SIGN (-1.0 <-> +1.0).

  You can now close SSH / unplug the cable. The robot keeps running.
EOF
