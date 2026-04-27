
  # 🅿️ ParkVision AI
  **A real-time, ultra-precise Smart Parking Detection System powered by YOLOv8 and Streamlit.**
  
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/ultralytics)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
</div>

---

## 🌟 Overview

**ParkVision AI** is a production-ready computer vision application designed to monitor CCTV and surveillance feeds to automatically classify parking spaces in real-time. By leveraging a fine-tuned **YOLOv8** model trained on the PKLot dataset, the system accurately distinguishes between empty and occupied spaces, even in highly dense parking lots.

### ✨ Key Features
- **Real-Time Video Analytics:** Stream live RTSP/HTTP camera feeds or upload `.mp4` files for instantaneous analysis.
- **Dynamic Auto-Calibration:** Built-in tensor calibration normalizes raw YOLOv8 probabilities to provide a highly accurate, user-friendly 0-100% confidence slider.
- **Agnostic Non-Maximum Suppression (NMS):** Smart overlapping box removal ensures that tightly parked cars aren't double-counted with conflicting red/green boxes.
- **High-Density Capable:** Process massive 4K drone feeds with the ability to detect and track over 1,000 independent parking spaces simultaneously (`max_det=1000`).
- **Interactive Streamlit GUI:** A sleek, dark-themed dashboard providing live metrics (Total, Available, Occupied).

---

## 🛠️ Project Architecture

```text
ParkVision/
├── app.py                      # Streamlit interactive dashboard & auto-calibration engine
├── smart_parking_system.py     # Core YOLOv8 OOP Engine (Train/Predict modules)
├── parking.yaml                # Dataset configuration matrix
├── best_model.pt               # Compiled production weights (add this after training)
├── requirements.txt            # Environment dependencies
└── .gitignore                  # Keeps your repo clean from massive datasets
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/ParkVision-AI.git
cd ParkVision-AI
```

### 2. Install Dependencies
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Launch the Dashboard
Ensure you have your weights (`best_model.pt`) in the root folder, then boot the system:
```bash
streamlit run app.py
```

---

## 🧠 Model Training

If you wish to fine-tune the model yourself or adjust for a new camera angle (Domain Shift adaptation), you can utilize the core engine:

```bash
# Train the model (Optimized for RTX 3000/4000 series via FP32 math)
python smart_parking_system.py --mode train --epochs 100 --batch 4
```
*Note: Mixed Precision (AMP) is disabled by default to prevent FP16 underflow bugs on dense datasets.*

---

## 📸 Expected Output

The UI provides a live heads-up display overlay on all processed frames:
- 🟩 **Green Boxes:** `space-empty`
- 🟥 **Red Boxes:** `space-occupied`

```text
┌──────────────────────────────────────────────────┐
│  Available: 67 / 199         Occupied: 132       │
│  Free: 33.7%                 Empty: 67           │
└──────────────────────────────────────────────────┘
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

