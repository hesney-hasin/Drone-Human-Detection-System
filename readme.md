#  Drone Human Detection & Counting System

Fine-tuned YOLOv8s on the VisDrone2019-DET dataset for aerial human and vehicle detection, with multi-tracker comparison using ByteTrack and BoT-SORT.

---

##  Overview

This project builds an end-to-end drone vision pipeline:

- **Dataset**: VisDrone2019-DET (10 classes, aerial imagery)
- **Model**: YOLOv8s fine-tuned for 50 epochs on dual T4 GPUs
- **Detection**: Frame-by-frame human and vehicle detection with count overlay
- **Tracking**: Multi-tracker comparison — ByteTrack vs BoT-SORT
- **Counting**: Class-aware human counting (pedestrian, people, bicycle, tricycle, motor)

## Repository Structure

```text
drone-human-detection-counting-system/
├── notebook.ipynb                  ← Training & detection notebook
├── notebook_extended.ipynb         ← Tracking & evaluation notebook
├── visdrone_yolo.yaml              ← Dataset config (auto-generated)
├── outputs/
│   ├── eda_class_distribution.png
│   ├── sample_gt_annotations.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxPR_curve.png
│   ├── BoxF1_curve.png
│   ├── tracker_comparison.png
│   ├── tracker_comparison_table.png
│   └── fps_benchmark.png
├── runs/detect/visdrone_yolov8s/
│   ├── weights/best.pt             ← Fine-tuned YOLOv8s weights
│   ├── results.csv
│   └── *.png
└── README.md
```

## Dataset

**VisDrone2019-DET** — drone-captured imagery with 10 object classes:

| Class ID | Name | Class ID | Name |
|---|---|---|---|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

- **Train**: 6,471 images
- **Val**: 548 images
- **Test**: 1,610 images (held out)
- Ignored regions (`score=0`) excluded from training labels
- Data split verified mutually exclusive by path separation

---

## Model — YOLOv8s

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | YOLOv8s |
| Epochs | 50 |
| Image size | 960px |
| Batch size | 16 (8 per GPU) |
| Device | DDP — dual T4 GPU |
| Optimizer | AdamW |
| Learning rate | 0.01 |
| Augmentation | Mosaic, copy-paste (0.3), HSV, mixup (0.1) |
| AMP | ✓ float16 |

### Results

| Metric | Value |
|---|---|
| mAP50 | **0.48** |
| mAP50-95 | **0.284** |

### Per-Class mAP50

| Class | mAP50 | Class | mAP50 |
|---|---|---|---|
| car | 0.848 | truck | 0.449 |
| bus | 0.629 | tricycle | 0.354 |
| pedestrian | 0.552 | awning-tricycle | 0.216 |
| motor | 0.557 | bicycle | 0.238 |
| van | 0.529 | people | 0.428 |

---

## Detection & Counting

Frame-by-frame detection on aerial traffic video with:
- Custom color-coded bounding boxes per class
- Human count overlay (pedestrian + people + bicycle + tricycle + motor)
- Vehicle count overlay (car + van + truck + bus)
- Smoothed count using 10-frame rolling average
- Inference at `conf=0.25`, `iou=0.4`

---

## Multi-Tracker Comparison

| Tracker | Type | Speed | Occlusion Handling | Re-ID |
|---|---|---|---|---|
| **ByteTrack** | IoU + Kalman + low-conf detections | ★★★★★ Fast | ★★★★ Good | ✗ No |
| **BoT-SORT** | ByteTrack + Camera Motion Compensation + ReID | ★★★★ Good | ★★★★★ Best | ✓ Partial |

**Recommendation**: BoT-SORT performs best on drone footage due to global motion compensation (GMC) that handles aerial camera pan and tilt — a unique challenge not addressed by ByteTrack.

---

##  Quick Start

```bash
pip install ultralytics supervision albumentations
```

```python
from ultralytics import YOLO

model = YOLO('runs/detect/visdrone_yolov8s/weights/best.pt')

# Inference on image
model.predict('your_drone_image.jpg', conf=0.25, save=True)

# ByteTrack on video
model.track('your_drone_video.mp4', tracker='bytetrack.yaml', conf=0.25, save=True)

# BoT-SORT on video
model.track('your_drone_video.mp4', tracker='botsort.yaml', conf=0.25, save=True)
```

---

## Limitations

- **Small object recall**: YOLOv8s misses objects smaller than 8×8 px at aerial altitude (bicycle mAP50 = 0.238)
- **Class imbalance**: Rare classes (bus, awning-tricycle) have very few training samples → low AP
- **Identity switches**: All trackers lose IDs under heavy occlusion in dense crowds
- **No MOT metrics**: MOTA/MOTP/IDF1 require MOT-format ground truth not available in VisDrone-DET

---

## Challenges Faced

- **VisDrone annotation format**: 1-indexed categories and `score=0` ignored regions required careful parser logic to avoid label leakage
- **DDP on Kaggle**: `device=[0,1]` returns `None` for the results object — fixed with `exist_ok=True` and absolute output paths
- **Session crashes**: RAM-cached dataset lost on OOM restart — solved by saving checkpoints to `/kaggle/working` with `save_period=5`
- **Cross-session transfer**: Used Kaggle notebook output mounting to carry `best.pt` into a new session without retraining

---

## Sample Outputs

### Training Curves
![Training Curves](Output/training_curves.png)

### Per-Class AP
![Per Class AP](Output/per_class_ap.png)

### Results (from Ultralytics)
![Results](Output/results.png)

### YOLOv8s Metrics + Tracker Comparison
![Results Summary](Output/results_summary.png)

### Tracker Quantitative Metrics
![Tracker Quantitative](Output/tracker_quantitative.png)

### Multi-Tracker Visual Comparison (ByteTrack | BoT-SORT)
![Tracker Comparison](Outputs/tracker_comparison.png)

### Inference Speed Benchmark
![FPS Benchmark](Outputs/fps_benchmark.png)

---

## Environment

- Platform: Kaggle (dual T4 GPU)
- Python: 3.12
- PyTorch: 2.10.0+cu128
- Ultralytics: latest
- CUDA: 12.8

---

## Author

**Maliha Hasin** — BSc CSE (Data Science), East West University
GitHub: [@hesney-hasin](https://github.com/hesney-hasin)
