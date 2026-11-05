# Robot setup

## Software on the Intel NUC

Ubuntu 22.04 with **ROS 2 Humble**.

```bash
sudo apt install ros-humble-ur-robot-driver ros-humble-rplidar-ros tmux
pip install ultralytics opencv-python pyrealsense2 flask requests numpy
git clone https://github.com/ProfessorYaseen/railway-door-inspection-robot.git
cd railway-door-inspection-robot
mkdir -p weights && cp /path/to/best.pt weights/      # or best_openvino_model/
cp .env.example .env && nano .env                     # Telegram token, ROBOT_IP, MODEL_PATH
python3 scripts/check_camera.py --width 640 --height 480
```

> Faster on the NUC CPU: set `MODEL_PATH=weights/best_openvino_model` to use the FP16 OpenVINO export.

## Telegram alerts (optional)

1. In Telegram, talk to **@BotFather**, send `/newbot` and copy the token into `.env` as `TELEGRAM_TOKEN`.
2. Send your new bot any message. The chat ID is discovered automatically, or you can set `TELEGRAM_CHAT_ID`.
3. Never commit `.env`. It is already listed in `.gitignore`.

## Launch

```bash
ssh <user>@<NUC-WIFI-IP>        # use Wi-Fi, not the wired 10.15.20.x address
cd railway-door-inspection-robot
./scripts/start_robot.sh
```

This opens a `tmux` session with three windows: LiDAR, UR driver, mission.

**On the UR pendant** (a manual safety step by design): load `external_control.urp`, press ▶, then switch to **Remote**. Set the speed slider to 100%.

Open `http://<NUC-WIFI-IP>:5000` on a laptop or phone and press **START**.

| Button | Effect |
|---|---|
| START | stow arm → search → approach → inspect |
| ABORT | stop base and arm sequence immediately |
| Step ON/OFF | pause after each arm pose |
| NEXT | advance one pose while paused |

```bash
tmux attach -t demo         # watch logs (Ctrl+b then d to detach)
tmux kill-session -t demo   # stop everything
```

Annotated videos of every run are saved to `SAVE_DIR` (default `~/inspection_runs`).

## Tuning cheatsheet

All tunables are at the top of `robot/autonomous_mission.py`.

| Symptom | Change |
|---|---|
| Door slides off-screen while turning | flip `TURN_SIGN` (-1.0 ↔ +1.0) |
| Stops short or long | check the APPROACH log; `SAFE_STOP_DISTANCE`, `DEPTH_BOX_BOT_FRAC` |
| Lab clutter detected as a door | raise `DOOR_CONF_THRESH` / `DOOR_MIN_H_FRAC` |
| Too many Telegram alerts | raise `ALERT_COOLDOWN` |
| Arm too slow | pendant slider at 100%, then `ARM_JOINT_SPEED` |
| Stops for people too eagerly | `INTRUDER_JUMP`, `INTRUDER_DEBOUNCE` |
