# 🎯 Ultimate AI Motion Detection System

A real-time motion and AI-powered person detection application built with Python, OpenCV, and YOLOv8. Designed to run entirely on CPU — no GPU required.

---

## 📸 Features

| Feature | Description |
|---|---|
| 🎥 Live Preview | Real-time webcam feed displayed in the app window |
| 🟢 Motion Detection | Background subtraction via OpenCV MOG2 |
| 🤖 AI Person Detection | YOLOv8n model detects people in each frame |
| ⏺ Auto Recording | Automatically records video when motion/person is detected |
| 🖐 Manual Recording | Start/stop recording manually with a button |
| 📷 Snapshot | Save a still image from the live feed at any time |
| 🎚 Adjustable Sensitivity | Tune motion area, threshold, and recording duration |
| 📂 Recording Browser | Browse and open saved recordings from inside the app |
| 🔴 REC Indicator | On-screen red dot + "REC" label while recording |
| 📊 Status Bar | Live display of current detection state |

---

## 🖥 Requirements

- Python **3.9 or newer**
- A working **webcam**
- Windows, macOS, or Linux

### Python Libraries

Install all dependencies with a single command:

```bash
pip install opencv-python pillow numpy ultralytics
```

---

## 🚀 Quick Start

### 1. Clone or download this project

Place all files in a folder, e.g.:

```
Project_1/
└── ultimate_motion_detector_v2.py
```

### 2. Install dependencies

```bash
pip install opencv-python pillow numpy ultralytics
```

### 3. Run the application

```bash
python ultimate_motion_detector_v2.py
```

> **Note:** On the very first run, YOLOv8 will automatically download the model weights file `yolov8n.pt` (~6 MB). Internet access is required for this one-time download.

---

## 🗂 Folder Structure

```
Project_1/
│
├── ultimate_motion_detector_v2.py   ← Main application
├── README.md                        ← This file
│
├── recordings/                      ← Auto-created; .avi video files saved here
│   └── 20240101_120000.avi
│
└── snapshots/                       ← Auto-created; .jpg snapshot images saved here
    └── 20240101_120015.jpg
```

---

## 🎛 UI Controls

| Control | Description |
|---|---|
| **Min Motion Area** | Minimum contour size (pixels²) to count as motion. Increase to ignore small movements. |
| **Threshold** | Sensitivity of the background subtractor. Lower = more sensitive. |
| **Record Seconds** | How many extra seconds to keep recording after motion stops. |
| **Enable AI Person Detection** | Toggle YOLOv8 person detection on/off. |
| **📷 Snapshot** | Saves a JPEG image to the `snapshots/` folder. |
| **⏺ Manual Record** | Starts/stops recording regardless of motion. Button label updates to show state. |
| **📂 Browse Recordings** | Opens a list of saved recordings with an "Open Folder" shortcut. |

---

## 🔍 How It Works

```
Webcam Frame
    │
    ├─► Background Subtraction (MOG2)
    │       └─ Contours too small?  →  Ignore
    │       └─ Large contour found? →  Motion Detected ✅
    │
    ├─► YOLOv8 (background thread, if enabled)
    │       └─ Class 0 (person) found? → Person Detected ✅
    │
    └─► Trigger = Motion OR Person OR Manual Record
            └─ Start/continue recording to recordings/
            └─ Stop recording after N seconds of silence
```

The AI detection runs in a **separate background thread** so it never blocks the live preview, keeping the UI smooth even on CPU-only machines.

---

## 🎨 Color Legend (On-screen overlays)

| Color | Meaning |
|---|---|
| 🟢 Green rectangle | Motion contour detected |
| 🔵 Blue/Red rectangle | AI-detected person |
| 🔴 Red dot + "REC" | Currently recording |

---

## ⚙️ Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot open camera` error | Check webcam is connected and not used by another app |
| AI detection checkbox greyed out | Run `pip install ultralytics` and restart |
| YOLOv8 download fails | Check your internet connection on first launch |
| Video is choppy | Disable AI detection to reduce CPU load |
| Recordings folder is empty | Make sure motion/person was detected, or use Manual Record |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture, background subtraction, drawing |
| `pillow` | Converting OpenCV frames for Tkinter display |
| `numpy` | Array operations |
| `ultralytics` | YOLOv8 AI person detection |

---

## 📄 License

This project is provided as-is for personal and educational use.
