# Model weights

Weights are not stored in git. Download them from the
[Releases page](https://github.com/ProfessorYaseen/railway-door-inspection-robot/releases)
and place them here:

```
weights/
├── best.pt                    # PyTorch checkpoint (YOLOE-26s-seg, 10.4 M params, 22 MB)
└── best_openvino_model/       # optional: FP16 OpenVINO export, fastest on Intel CPUs
```

You can also train your own with `training/train.py` (see the main README).
