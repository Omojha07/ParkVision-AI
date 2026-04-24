# 🅿️ Smart Parking Detection System
### YOLOv8 · PKLot Dataset · Real-time Space Classification

---

## Overview

Detects and classifies parking spaces as **empty** or **occupied** using
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) fine-tuned on
the [PKLot dataset](https://universe.roboflow.com/roboflow-100/pklot-640).

| Class | Color | Description |
|---|---|---|
| `space-empty` | 🟩 Neon green | Free parking slot |
| `space-occupied` | 🟥 Red | Occupied slot |

---

## Project Structure

```
smart_parking/
├── smart_parking_system.py   # Main script (all modules)
├── parking.yaml              # Dataset configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── dataset/                  # PKLot dataset (download separately)
│   ├── train/images/
│   ├── valid/images/
│   └── test/images/
├── runs/                     # Training output (auto-created)
│   └── detect/parking/
│       └── weights/
│           ├── best.pt       ← best model weights
│           └── last.pt
├── output_folder/            # Folder prediction results
├── parking_log.csv           # Auto-generated activity log
└── folder_results.csv        # Batch prediction summary
```

---

## Installation

### 1 · Clone / download this project

```bash
git clone https://github.com/yourname/smart-parking-detection
cd smart_parking
```

### 2 · Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

**GPU support (optional but recommended):**

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4 · Download the PKLot dataset

```bash
# Option A — Roboflow CLI
pip install roboflow
python - <<'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
proj = rf.workspace("roboflow-100").project("pklot-640")
proj.version(1).download("yolov8", location="./dataset")
EOF

# Option B — Manual
# Download from https://universe.roboflow.com/roboflow-100/pklot-640
# Extract to ./dataset/ preserving train/valid/test structure
```

Update `parking.yaml` → `path:` to match your local path.

---

## Usage

### Train

```bash
# Quick start (YOLOv8 Nano, 50 epochs, GPU auto-detect)
python smart_parking_system.py --mode train --data parking.yaml

# Full options
python smart_parking_system.py \
  --mode train \
  --data parking.yaml \
  --model yolov8s.pt \    # yolov8n.pt (fastest) or yolov8s.pt (better accuracy)
  --epochs 100 \
  --batch 32 \
  --imgsz 640 \
  --device 0              # 0 = first GPU, cpu = CPU only
```

Best weights are saved automatically to `runs/detect/parking/weights/best.pt`.

---

### Evaluate

```bash
python smart_parking_system.py \
  --mode eval \
  --weights runs/detect/parking/weights/best.pt \
  --data parking.yaml \
  --conf 0.25
```

Prints: **mAP@0.5**, **mAP@0.5:0.95**, **Precision**, **Recall**.

---

### Predict — Single Image

```bash
python smart_parking_system.py \
  --mode predict \
  --weights best.pt \
  --source parking_lot.jpg \
  --conf 0.3
```

Annotated image saved as `output_parking_lot.jpg`.

---

### Predict — Folder of Images

```bash
python smart_parking_system.py \
  --mode predict \
  --weights best.pt \
  --source ./test_images/
```

Results written to `output_folder/` + `folder_results.csv`.

---

### Predict — Video File

```bash
python smart_parking_system.py \
  --mode predict \
  --weights best.pt \
  --source parking_video.mp4
```

Output video: `output_parking_video.mp4`. Press **Q** to stop early.

---

### Live Webcam

```bash
python smart_parking_system.py \
  --mode webcam \
  --weights best.pt \
  --cam 0                 # camera index
  --conf 0.25
```

Press **Q** to quit.

---

### Streamlit GUI

```bash
pip install streamlit
streamlit run smart_parking_system.py -- --mode gui
```

Opens a browser UI for uploading images and viewing annotated results.

---

## Expected Performance (YOLOv8n, PKLot)

| Metric | Approx. Value |
|---|---|
| mAP@0.5 | ~0.94 |
| mAP@0.5:0.95 | ~0.72 |
| Precision | ~0.93 |
| Recall | ~0.91 |

*Values vary with training duration, hardware, and dataset split.*

---

## HUD Overlay

Every output frame includes:

```
┌──────────────────────────────────────────────────┐
│  Available: 12 / 50          Occupied: 38        │
│  76.0% Free                  Empty: 12           │
└──────────────────────────────────────────────────┘
⚠  ALL PARKING SPACES ARE OCCUPIED  ⚠  (alert bar, if full)
```

---

## Logging

Every prediction appends a row to `parking_log.csv`:

```
timestamp,source,total,occupied,empty,availability_pct,is_full
2024-01-15T14:32:01,lot_a.jpg,50,38,12,24.00,False
```

---

## CLI Reference

```
usage: smart_parking_system.py [-h] --mode {train,eval,predict,webcam,gui}
                                [--weights WEIGHTS] [--conf CONF]
                                [--imgsz IMGSZ] [--device DEVICE]
                                [--data DATA] [--epochs EPOCHS]
                                [--batch BATCH] [--model MODEL]
                                [--source SOURCE] [--cam CAM] [--no-save]

Options:
  --mode       train | eval | predict | webcam | gui
  --weights    Path to .pt weights (default: yolov8n.pt)
  --conf       Confidence threshold 0–1 (default: 0.25)
  --imgsz      Inference size (default: 640)
  --device     cpu | 0 | auto (default: auto)
  --data       Dataset YAML for train/eval (default: parking.yaml)
  --epochs     Training epochs (default: 50)
  --batch      Batch size (default: 16)
  --model      Base model for training (default: yolov8n.pt)
  --source     Image / folder / video path for predict mode
  --cam        Webcam index (default: 0)
  --no-save    Skip saving annotated output files
```

---

## License

MIT — free to use, modify, and distribute.
