# 🎯 Ultimate AI Motion Detection System V2

A high-performance, real-time motion and AI-powered behavior analysis system built with Python, OpenCV, and YOLO. Modernized with a premium side-by-side UI and optimized for smooth execution on CPU-only hardware.

---

## 📸 Core Features

| Category | Features |
|---|---|
| 🖥 **Modern UI** | Sleek Dark Mode interface with a **75% / 25% side-by-side layout** maximizing camera real-estate. |
| 🤖 **AI Person & Object Detection** | Identifies people and 80+ object categories (Laptops, Backpacks, Bottles, etc.) in real-time. |
| 🎭 **Action Recognition** | Advanced human behavior analysis: detects **Standing, Walking, Bending, Waving, Sitting, and Falling**. |
| 📱 **Smartphone Optimization** | Specifically calibrated for **Smartphone** detection with specialized "Holding" interaction logic. |
| 🛡 **Safety Alerts** | Automatic red-border visual alarm and log entry for critical actions like **FALL DETECTED**. |
| 🎚 **AI Accuracy Slider** | Real-time confidence control (10–99%) with **dynamic color-coded logic hints** for optimal tuning. |
| ⚡ **CPU Optimization** | Native **720p 16:9** streaming and **Proxy Frame Downscaling** for lag-free motion analysis. |
| ⏺ **Flexible Recording** | Dedicated **Auto-Record** toggle + **Manual Record** button with high-fidelity 720p output. |

---

## 🤖 Detailed AI Capabilities

### 🔍 Supported Actions
The system uses skeletal keypoint analysis to categorize exactly what a person is doing:
- **Falling**: Detects sudden horizontal orientation changes (Triggers "FALL DETECTED" alarm).
- **Sitting**: Analyzes hip-to-knee-to-ankle ratios.
- **Waving**: Detects hands raised above the shoulder line.
- **Bending**: Recognizes upper-body angle shifts.
- **Holding**: Detects when a specific object (like a **Smartphone**) is within a person's hand/wrist radius.
- **Picking Up**: Combines bending logic with hand-object proximity.

### 📦 Object Recognition
Leverages the COCO dataset to identify a wide range of objects:
- **Electronic Devices**: Smartphones, Laptops, TVs, Mice, Keyboards.
- **Furniture**: Chairs, Tables, Couches, Beds.
- **Personal Items**: Backpacks, Umbrellas, Handbags, Ties.
- **Kitchenware**: Cups, Forks, Knives, Spoons, Bottles.

---

## 🖥 Requirements

- Python **3.12 or newer**
- A working **webcam**
- Windows 10/11 (Preferred)

### Python Libraries

```bash
pip install opencv-python pillow numpy ultralytics customtkinter darkdetect
```

---

## 🚀 Quick Start

1. **Clone the Project** and navigate to the directory.
2. **Install Dependencies**:
   ```bash
   pip install opencv-python pillow numpy ultralytics customtkinter darkdetect
   ```
3. **Run the App**:
   ```bash
   python ultimate_motion_detector_v2.py
   ```

> **Note:** On the very first run, the system will automatically download the optimized `yolov9t.pt` and `yolov8n-pose.pt` models (approx 12MB total).

---

## 🎛 Pro Control Tuning

- **AI Accuracy (%)**: 
  - 🟢 **25–40%**: The "Sweet Spot". Detects both people and complex objects reliably.
  - 🟡 **41–65%**: Medium Zone. Filters out background noise; great for busy environments.
  - 🔴 **66%+**: Strict Mode. Displays only extremely confident human detections.
- **Auto-Record**: When enabled, the system automatically saves video files to the `/recordings` folder whenever motion or human activity is confirmed.

---

## 🔍 How It Works

1. **Native Stream**: Captures a wide 1280x720 feed for high visual fidelity.
2. **Proxy Analysis**: Automatically creates a low-res 640x360 "proxy" frame for background subtraction. This removes CPU lag regardless of how large the actual viewing window is.
3. **Hybrid AI Engine**:
   - **YOLOv9**: Handles high-speed object and person detection.
   - **YOLOv8-Pose**: Maps 17 skeletal keypoints for behavior analysis.
   - **Action Engine**: Computes geometric relationships between bones and objects.
4. **Majority-Vote Smoothing**: Applies a 10-frame rolling buffer to action labels to prevent flickering.

---

## 📄 License

Modernized version released 2026. Provided for personal, educational, and security research use.
