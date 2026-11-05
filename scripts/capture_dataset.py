#!/usr/bin/env python3
"""Capture a training dataset from the robot's RealSense D435i.

Saves one JPEG every ``--interval`` seconds while you move the camera around
the door. Run it on the NUC (USB 3.2) -- a laptop may not sustain 1080p.

    python3 scripts/capture_dataset.py --out ~/railway_capture --num 200

Capture protocol (what made the dataset generalise):
  * keep moving: vary distance, angle, height and lighting the whole time;
  * stand back so the door fills ~1/4 of the frame, not all of it;
  * include deliberately hard shots: dim, oblique, partly occluded;
  * include frames with NO train door (lab doors, cabinets, fences) --
    these become hard negatives that stop the model hallucinating doors.

Then copy to your laptop and annotate (we used Roboflow):
    scp -r <user>@<nuc-ip>:~/railway_capture ~/railway_capture
"""

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/railway_capture"), help="output folder")
    ap.add_argument("--num", type=int, default=200, help="number of images to save")
    ap.add_argument("--interval", type=float, default=0.6, help="seconds between saves (1.5 = slower walk)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--prefix", default="img", help="filename prefix")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe = rs.pipeline()
    pipe.start(cfg)

    n = 0
    try:
        for _ in range(30):  # let auto-exposure settle
            pipe.wait_for_frames()
        print(f"Capturing {args.num} images to {args.out} -- keep moving the camera!")
        while n < args.num:
            f = pipe.wait_for_frames().get_color_frame()
            if not f:
                continue
            img = np.asanyarray(f.get_data())
            path = os.path.join(args.out, f"{args.prefix}_{n:04d}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            n += 1
            print(f"{n:4d}  {img.shape[1]} x {img.shape[0]}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipe.stop()
        print(f"\nSaved {n} images to: {args.out}")


if __name__ == "__main__":
    main()
