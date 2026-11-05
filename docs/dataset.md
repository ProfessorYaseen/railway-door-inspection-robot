# Dataset

> The images are institutional data and are **not published** in this repository.
> Contact the author for research access.

## Capture

* **Sensor:** the robot's own Intel RealSense D435i, so training images match deployment images.
* **Tool:** `scripts/capture_dataset.py` (one frame every 0.6 s while the operator moves the camera).
* **Protocol:**
  * vary distance, angle, height and lighting continuously;
  * keep the door at about ¼ of the frame (not filling it) for most shots;
  * deliberately include dim, oblique and partly occluded views;
  * include **hard negatives**: frames with no train door (lab doors, cabinets, fences, pillars).

The deployed model was trained on 640×480 captures. A later 1920×1080 recapture (running from the NUC's USB 3.2 port) is supported by the same script (`--width 1920 --height 1080`).

## Annotation

Annotated in Roboflow with 4 classes:

| id | class | instances |
|---|---|---:|
| 0 | `Door_Open` | 120 |
| 1 | `Door_closed` | 213 |
| 2 | `Door_not_closed` | 151 |
| 3 | `Rubber_missing` | 125 |

Most labels are boxes and a few are polygons. The YOLOE segmentation trainer rejects a dataset that mixes the two, so `training/train.py prepare` converts every box into a 4-corner rectangle polygon (`box_to_polygon`, unit-tested).

## Split

| split | images | labelled | hard negatives |
|---|---:|---:|---:|
| train | 489 | 448 | 41 |
| valid | 61 | 56 | 5 |
| test | 63 | 57 | 6 |
| **total** | **613** | **561** | **52** |

Labelled and background images were shuffled separately (seed 42) and split 80/10/10, so every split contains hard negatives.

## Expected layout

```
data/
├── data.yaml          # names: [Door_Open, Door_closed, Door_not_closed, Rubber_missing]
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```
