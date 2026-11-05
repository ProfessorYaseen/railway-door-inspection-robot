#!/usr/bin/env python3
"""Pre-flight check for the Intel RealSense D435i.

Verifies that the camera is detected, is on a USB 3.x port, and can stream
colour at the requested resolution.

    python3 scripts/check_camera.py                 # 1920x1080 @ 15 fps
    python3 scripts/check_camera.py --width 1280 --height 720
"""

import argparse
import sys

import numpy as np
import pyrealsense2 as rs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    devices = rs.context().query_devices()
    print("RealSense devices found:", len(devices))
    if len(devices) == 0:
        print("  -> no camera. Check the cable and try another USB port.")
        return 1

    dev = devices[0]
    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    print("  model :", dev.get_info(rs.camera_info.name))
    print("  USB   :", usb)
    if not usb.startswith("3"):
        print("  [!] USB 2 link -- high resolutions will fail. Use a USB 3 port/cable.")

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe = rs.pipeline()
    try:
        pipe.start(cfg)
        frame = pipe.wait_for_frames().get_color_frame()
        img = np.asanyarray(frame.get_data())
        print(f"  stream: GOT {img.shape[1]} x {img.shape[0]} @ {args.fps} fps  -> OK")
    except RuntimeError as e:
        print(f"  stream: FAILED at {args.width}x{args.height} ({e})")
        print("  -> try --width 1280 --height 720")
        return 1
    finally:
        try:
            pipe.stop()
        except RuntimeError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
