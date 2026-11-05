#!/usr/bin/env python3
"""Train, evaluate and export the YOLOE-26s door-inspection model.

Reproduces the Colab notebook (training/train_door_detector.ipynb) as a CLI.

    # 0) dataset exported from Roboflow in "YOLOv8/YOLO11" format, unzipped to data/
    # 1) fix data.yaml paths + convert box labels to polygons (needed by the seg trainer)
    python3 training/train.py prepare --data data/data.yaml
    # 2) train (Tesla T4: ~36 min for 100 epochs)
    python3 training/train.py train   --data data/data.yaml
    # 3) held-out TEST split metrics
    W=runs/segment/train_door_detector/yoloe26s/weights/best.pt
    python3 training/train.py eval    --data data/data.yaml --weights $W
    # 4) export for deployment (OpenVINO for the Intel NUC, ONNX for anything else)
    python3 training/train.py export  --weights $W
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CLASS_NAMES = ["Door_Open", "Door_closed", "Door_not_closed", "Rubber_missing"]


# --------------------------------------------------------------------------- #
#  prepare
# --------------------------------------------------------------------------- #
def box_to_polygon(line: str) -> str:
    """'cls cx cy w h' (YOLO box) -> 'cls x1 y1 x2 y2 x3 y3 x4 y4' (rectangle polygon).

    The YOLOE *seg* trainer refuses mixed box/polygon datasets, and most of our
    labels are boxes, so boxes are converted to 4-corner polygons. Lines that
    are already polygons are returned unchanged.
    """
    parts = line.split()
    if len(parts) != 5:
        return line
    cls = parts[0]
    cx, cy, w, h = map(float, parts[1:])
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    pts = [x1, y1, x2, y1, x2, y2, x1, y2]
    return cls + " " + " ".join(f"{p:.6f}" for p in pts)


def prepare(args) -> None:
    data_yaml = Path(args.data).resolve()
    root = data_yaml.parent
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    # Roboflow writes paths relative to its own folder layout -> make them absolute.
    data["path"] = str(root)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    data.pop("roboflow", None)
    if data.get("names") != CLASS_NAMES:
        print(f"[!] class order in data.yaml is {data.get('names')} (expected {CLASS_NAMES})")
    with open(data_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"data.yaml fixed -> {data_yaml}")

    converted = kept = 0
    counts = {n: 0 for n in data["names"]}
    for split in ("train", "valid", "test"):
        n_img = len(list((root / split / "images").glob("*")))
        n_bg = 0
        for lf in (root / split / "labels").glob("*.txt"):
            lines = [ln.strip() for ln in lf.read_text().splitlines() if ln.strip()]
            if not lines:
                n_bg += 1
                continue
            new = []
            for ln in lines:
                counts[data["names"][int(ln.split()[0])]] += 1
                out = box_to_polygon(ln)
                converted += out != ln
                kept += out == ln
                new.append(out)
            lf.write_text("\n".join(new) + "\n")
        print(f"  {split:5s}: {n_img} images ({n_bg} empty-label backgrounds)")
    print(f"labels: {converted} boxes -> polygons, {kept} polygons kept")
    print("instances per class:", counts)


# --------------------------------------------------------------------------- #
#  train / eval / export
# --------------------------------------------------------------------------- #
def train(args) -> None:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

    model = YOLOE(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=20,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=0.025,
        momentum=0.9,
        warmup_epochs=5,
        warmup_bias_lr=0.0,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        trainer=YOLOEPESegTrainer,
        device=args.device,
        project="train_door_detector",
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        plots=True,
    )


def evaluate(args) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    m = model.val(data=str(Path(args.data).resolve()), split=args.split, imgsz=args.imgsz, plots=True)
    print(f"\n{args.split.upper()} split:")
    print(f"  box  mAP50 {m.box.map50:.3f}   mAP50-95 {m.box.map:.3f}")
    print(f"  mask mAP50 {m.seg.map50:.3f}   mAP50-95 {m.seg.map:.3f}")


def export(args) -> None:
    from ultralytics import YOLOE

    model = YOLOE(args.weights)
    model.set_classes(CLASS_NAMES)
    model.export(format="openvino", imgsz=args.imgsz, half=True, simplify=True)  # Intel NUC
    model.export(format="onnx", imgsz=args.imgsz, half=False, simplify=True, opset=17)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="fix data.yaml + convert box labels to polygons")
    p.add_argument("--data", required=True)
    p.set_defaults(func=prepare)

    p = sub.add_parser("train", help="fine-tune YOLOE-26s-seg")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yoloe-26s-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="yoloe26s")
    p.set_defaults(func=train)

    p = sub.add_parser("eval", help="metrics on a split (default: test)")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.set_defaults(func=evaluate)

    p = sub.add_parser("export", help="export to OpenVINO (FP16) + ONNX")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.set_defaults(func=export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
