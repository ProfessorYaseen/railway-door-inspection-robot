#!/usr/bin/env python3
"""Run the door-inspection detector on images, a video, a webcam or a RealSense.

No robot needed -- this is the quickest way to try the model.

Examples:
    # a folder of images (annotated copies are written to outputs/)
    python3 scripts/run_inference.py --weights weights/best.pt --source assets/samples

    # a video file
    python3 scripts/run_inference.py --weights weights/best.pt --source door.mp4

    # laptop webcam / RealSense colour stream, live window
    python3 scripts/run_inference.py --weights weights/best.pt --source 0 --show
    python3 scripts/run_inference.py --weights weights/best.pt --source realsense --show

    # faster on Intel CPUs: the OpenVINO export
    python3 scripts/run_inference.py --weights weights/best_openvino_model --source 0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}
DEFECT_CLASSES = {"Door_Open", "Door_not_closed", "Rubber_missing"}


def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def realsense_frames(width=640, height=480, fps=30):
    import numpy as np
    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    try:
        while True:
            f = pipe.wait_for_frames().get_color_frame()
            if f:
                yield np.asanyarray(f.get_data())
    finally:
        pipe.stop()


def capture_frames(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def summarise(result) -> list[tuple[str, float]]:
    names = result.names
    return [(names[int(b.cls)], float(b.conf)) for b in result.boxes]


def run_images(model, paths, out_dir: Path, conf: float, imgsz: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    for p in paths:
        t0 = time.perf_counter()
        r = model.predict(str(p), conf=conf, imgsz=imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = summarise(r)
        totals.update(n for n, _ in dets)
        flag = "  <-- DEFECT" if any(n in DEFECT_CLASSES for n, _ in dets) else ""
        pretty = ", ".join(f"{n} {c:.2f}" for n, c in dets) or "no detections"
        print(f"{p.name:40s} {ms:7.1f} ms  {pretty}{flag}")
        cv2.imwrite(str(out_dir / p.name), r.plot())
    print(f"\nAnnotated images -> {out_dir}/")
    if totals:
        print("Totals:", dict(totals))


def run_stream(model, frames, out_path: Path | None, show: bool, conf: float, imgsz: int) -> None:
    writer = None
    n, t_start = 0, time.perf_counter()
    try:
        for frame in frames:
            r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            vis = r.plot()
            n += 1
            fps = n / (time.perf_counter() - t_start)
            cv2.putText(vis, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if out_path is not None:
                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                writer.write(vis)
            if show:
                cv2.imshow("door inspection", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"Annotated video -> {out_path}")
        if show:
            cv2.destroyAllWindows()
        if n:
            print(f"{n} frames, mean {n / (time.perf_counter() - t_start):.1f} FPS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/best.pt", help=".pt, .onnx or *_openvino_model dir")
    ap.add_argument("--source", required=True,
                    help="image, folder, video file, webcam index (0) or 'realsense'")
    ap.add_argument("--conf", type=float, default=0.45, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--show", action="store_true", help="display a live window")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"Weights not found: {args.weights}\n"
              "Put best.pt in weights/ (see README > 'Model weights').")
        return 1
    model = load_model(args.weights)
    out = Path(args.out)
    src = args.source

    if src == "realsense":
        run_stream(model, realsense_frames(), out / "realsense.mp4", args.show, args.conf, args.imgsz)
    elif src.isdigit():
        run_stream(model, capture_frames(int(src)), out / "webcam.mp4", args.show, args.conf, args.imgsz)
    else:
        p = Path(src)
        if p.is_dir():
            imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
            if not imgs:
                print(f"No images in {p}")
                return 1
            run_images(model, imgs, out, args.conf, args.imgsz)
        elif p.suffix.lower() in IMG_EXT:
            run_images(model, [p], out, args.conf, args.imgsz)
        elif p.suffix.lower() in VID_EXT:
            run_stream(model, capture_frames(str(p)), out / f"{p.stem}_pred.mp4", args.show, args.conf, args.imgsz)
        else:
            print(f"Unsupported source: {src}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
