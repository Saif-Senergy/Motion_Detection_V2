"""
Ultimate AI Motion Detection System (CPU Version)
==================================================
Features:
  - Live camera preview with detection overlays
  - Motion detection via OpenCV background subtraction
  - AI person detection via YOLOv8 (runs in background thread)
  - Auto-recording when motion/person detected
  - Manual recording toggle
  - Snapshot capture
  - Adjustable sensitivity and recording duration
  - Recording browser
  - Graceful shutdown / resource cleanup
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import datetime
import os
import time
import threading
import queue

from action_engine import ActionEngine

# ── Optional YOLO import (graceful fallback if ultralytics not installed) ──────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
RECORD_DIR = "recordings"
SNAP_DIR   = "snapshots"
WIDTH      = 640
HEIGHT     = 480
FPS_TARGET = 20          # target display / recording FPS

os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(SNAP_DIR,   exist_ok=True)

# ── Application class ─────────────────────────────────────────────────────────
class MotionDetectorApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Ultimate AI Motion Detection System")
        self.root.geometry("1100x800")
        self.root.minsize(800, 600)
        
        # Maximize the window on launch
        self.root.after(0, lambda: self.root.state('zoomed'))
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── State variables ───────────────────────────────────────────────────
        self.recording        = False
        self.manual_recording = False
        self.out: cv2.VideoWriter | None = None
        self.last_motion_time = 0.0
        self.running          = True
        self.motion_counter   = 0  # for temporal filtering

        # AI detection state (updated from bg thread)
        self._ai_result_lock    = threading.Lock()
        self._ai_boxes: list    = []          # list of (x1,y1,x2,y2) ints
        self._ai_poses: list    = []          # list of keypoint arrays
        self._ai_objects: list  = []          # list of (x1,y1,x2,y2, cls_id)
        self._ai_actions: list  = []          # list of current action strings
        
        self.action_engine      = ActionEngine()
        self.action_history: list = []        # list of (timestamp, text)

        self._ai_pending        = False       # True while a frame is being processed
        self._ai_queue: queue.Queue = queue.Queue(maxsize=1)
        self._ai_result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._ai_stale_frames    = 0
        self._ai_max_stale      = 5

        # ── Camera ────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Camera Error",
                                         "Cannot open camera. Please check your webcam connection.")
                    self.root.destroy()
                    return

        # Try to set camera properties for widescreen 720p (perfect 16:9 aspect fit)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Test camera by grabbing a frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            messagebox.showerror("Camera Error",
                                 "Cannot read from camera. Please check camera permissions and ensure no other app is using it.")
            self.cap.release()
            self.root.destroy()
            return

        # ── Background subtractor ─────────────────────────────────────────────
        self.fgbg = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=400.0)

        # ── YOLO models ───────────────────────────────────────────────────────
        self.model = None
        self.pose_model = None
        self.depth_model = None
        self.class_names = {}
        if YOLO_AVAILABLE:
            # Use hybrid approach: YOLOv9 for detection (faster) + YOLOv8 for pose (reliable)
            model_variants = [
                ("yolov9t.pt", "yolov8n-pose.pt"),  # YOLOv9 detection + YOLOv8 pose (hybrid - best)
                ("yolov8n.pt", "yolov8n-pose.pt"),  # Fallback: YOLOv8 detection + YOLOv8 pose
            ]
            load_error = None
            for det_path, pose_path in model_variants:
                try:
                    if os.path.exists(det_path) and os.path.exists(pose_path):
                        self.model = YOLO(det_path)
                        self.pose_model = YOLO(pose_path)
                        if self.model:
                            self.class_names = self.model.names
                        load_error = None
                        break
                    else:
                        missing = []
                        if not os.path.exists(det_path):
                            missing.append(det_path)
                        if not os.path.exists(pose_path):
                            missing.append(pose_path)
                        raise FileNotFoundError(f"Missing model file(s): {', '.join(missing)}")
                except Exception as e:
                    load_error = e
                    self.model = None
                    self.pose_model = None
            if load_error:
                messagebox.showwarning(
                    "YOLO Warning",
                    f"Could not load YOLO models:\n{load_error}\n\n"
                    "AI detection will be disabled.\n"
                    "Place yolov9t.pt/yolov9t-pose.pt or yolov8n.pt/yolov8n-pose.pt in the app folder."
                )

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()

        # ── Start AI worker thread ────────────────────────────────────────────
        self._ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self._ai_thread.start()

        # ── Start frame loop ──────────────────────────────────────────────────
        self._frame_interval_ms = max(1, int(1000 / FPS_TARGET))
        self._update_frame()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Apply CustomTkinter styling constraints natively
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root.configure(fg_color="#181825") # Deep dark background

        # ── Grid Layout ───────────────────────────────────────────────────────
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=1)  # Allow log frame to expand if needed
        
        self.root.grid_columnconfigure(0, weight=3) # Video takes roughly 75% width
        self.root.grid_columnconfigure(1, weight=1) # UI controls take 25% width

        # ── Video area ────────────────────────────────────────────────────────
        video_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        video_frame.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=0, pady=0)

        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self.root, corner_radius=10, fg_color="#1e1e2e")
        ctrl.grid(row=0, column=1, padx=16, pady=(16, 8), sticky="new")
        ctrl.columnconfigure(1, weight=1)

        # Min Motion Area
        self.min_area = tk.IntVar(value=1200)
        ctk.CTkLabel(ctrl, text="Min Motion Area", font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w", padx=12, pady=4)
        ctk.CTkSlider(ctrl, from_=100, to=5000, variable=self.min_area).grid(row=0, column=1, sticky="ew", padx=12, pady=4)
        self.min_area_val_lbl = ctk.CTkLabel(ctrl, text="1200", width=40, font=("Segoe UI", 12))
        self.min_area_val_lbl.grid(row=0, column=2, sticky="w", padx=12, pady=4)
        self.min_area.trace_add("write", lambda *_: self.min_area_val_lbl.configure(text=str(self.min_area.get())))

        # Threshold
        self.threshold_val = tk.IntVar(value=25)
        ctk.CTkLabel(ctrl, text="Threshold", font=("Segoe UI", 12)).grid(row=1, column=0, sticky="w", padx=12, pady=4)
        ctk.CTkSlider(ctrl, from_=1, to=100, variable=self.threshold_val).grid(row=1, column=1, sticky="ew", padx=12, pady=4)
        self.thresh_val_lbl = ctk.CTkLabel(ctrl, text="25", width=40, font=("Segoe UI", 12))
        self.thresh_val_lbl.grid(row=1, column=2, sticky="w", padx=12, pady=4)
        self.threshold_val.trace_add("write", lambda *_: self.thresh_val_lbl.configure(text=str(self.threshold_val.get())))

        # Record Seconds
        self.record_duration = tk.IntVar(value=10)
        ctk.CTkLabel(ctrl, text="Record Seconds", font=("Segoe UI", 12)).grid(row=2, column=0, sticky="w", padx=12, pady=4)
        ctk.CTkSlider(ctrl, from_=5, to=60, variable=self.record_duration, number_of_steps=55).grid(row=2, column=1, sticky="ew", padx=12, pady=4)
        self.rec_dur_lbl = ctk.CTkLabel(ctrl, text="10", width=40, font=("Segoe UI", 12))
        self.rec_dur_lbl.grid(row=2, column=2, sticky="w", padx=12, pady=4)
        self.record_duration.trace_add("write", lambda *_: self.rec_dur_lbl.configure(text=str(self.record_duration.get())))

        # AI Accuracy Level Configurable Slider (default 30%)
        self.ai_accuracy = tk.DoubleVar(value=30.0)
        ctk.CTkLabel(ctrl, text="AI Accuracy (%)", font=("Segoe UI", 12, "bold"), text_color="#a6e3a1").grid(row=3, column=0, sticky="w", padx=12, pady=4)
        ctk.CTkSlider(ctrl, from_=10.0, to=99.0, variable=self.ai_accuracy, number_of_steps=89, command=lambda v: self._on_accuracy_change(v)).grid(row=3, column=1, sticky="ew", padx=12, pady=4)
        self.ai_acc_lbl = ctk.CTkLabel(ctrl, text="30%", width=40, font=("Segoe UI", 12, "bold"), text_color="#a6e3a1")
        self.ai_acc_lbl.grid(row=3, column=2, sticky="w", padx=12, pady=4)

        # Accuracy hint note
        self.acc_hint_lbl = ctk.CTkLabel(
            ctrl,
            text="✦ Sweet spot: 25–40% detects people & objects  |  60%+ for strict person-only mode  |  90%+ rarely detects anything",
            font=("Segoe UI", 10, "italic"),
            text_color="#89b4fa",
            anchor="w",
            wraplength=520
        )
        self.acc_hint_lbl.grid(row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 6))

        # Toggles
        self.ai_enabled = tk.BooleanVar(value=True)
        self.alarm_enabled = tk.BooleanVar(value=False)
        self.auto_record_enabled = tk.BooleanVar(value=False) # Off by default
        
        ai_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        ai_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=4)
        
        ai_chk = ctk.CTkCheckBox(ai_frame, text="AI Action", variable=self.ai_enabled, font=("Segoe UI", 12))
        ai_chk.pack(side="left", padx=8, pady=4)
        
        alarm_chk = ctk.CTkCheckBox(ai_frame, text="Alarm", variable=self.alarm_enabled, font=("Segoe UI", 12))
        alarm_chk.pack(side="left", padx=8, pady=4)

        record_chk = ctk.CTkCheckBox(ai_frame, text="Auto-Record", variable=self.auto_record_enabled, font=("Segoe UI", 12))
        record_chk.pack(side="left", padx=8, pady=4)

        if not (YOLO_AVAILABLE and self.model):
            self.ai_enabled.set(False)
            ai_chk.configure(state="disabled")
            alarm_chk.configure(state="disabled")

        # ── Button row ────────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self.root, fg_color="transparent")
        btn_row.grid(row=1, column=1, padx=16, pady=4, sticky="new")
        
        ctk.CTkButton(btn_row, text="📷 Snapshot", command=self._snapshot, height=36, font=("Segoe UI", 13, "bold")).pack(side="top", pady=4, expand=True, fill="x")
        self.rec_btn = ctk.CTkButton(btn_row, text="⏺ Manual Record", command=self._toggle_record, height=36, font=("Segoe UI", 13, "bold"), fg_color="#313244", hover_color="#45475a")
        self.rec_btn.pack(side="top", pady=4, expand=True, fill="x")
        ctk.CTkButton(btn_row, text="📂 Browse Recordings", command=self._open_recordings, height=36, font=("Segoe UI", 13, "bold"), fg_color="#313244", hover_color="#45475a").pack(side="top", pady=4, expand=True, fill="x")

        # ── Action Log & Status Bar ───────────────────────────────────────────
        bot_frame = ctk.CTkFrame(self.root, fg_color="#11111b", corner_radius=10)
        bot_frame.grid(row=2, column=1, sticky="nsew", padx=16, pady=(4, 16))
        
        status_row = ctk.CTkFrame(bot_frame, fg_color="transparent")
        status_row.pack(fill="x", padx=12, pady=(12, 4))

        self.status_lbl = ctk.CTkLabel(status_row, text="● Ready", text_color="#a6e3a1", font=("Segoe UI", 12, "bold"))
        self.status_lbl.pack(side="left")

        self.rec_status_lbl = ctk.CTkLabel(status_row, text="", text_color="#f38ba8", font=("Segoe UI", 12, "bold"))
        self.rec_status_lbl.pack(side="right")
        
        log_frame = ctk.CTkFrame(bot_frame, fg_color="transparent")
        log_frame.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        
        self.log_list = tk.Listbox(log_frame, bg="#11111b", fg="#cdd6f4", relief="flat", bd=0, font=("Consolas", 10), highlightthickness=0)
        self.log_list.pack(fill="both", side="left", expand=True)
        
        scrollbar = ctk.CTkScrollbar(log_frame, orientation="vertical", command=self.log_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_list.config(yscrollcommand=scrollbar.set)

    # ─────────────────────────────────────────────────────────────────────────
    # Accuracy slider callback
    # ─────────────────────────────────────────────────────────────────────────
    def _on_accuracy_change(self, v):
        val = int(v)
        self.ai_acc_lbl.configure(text=f"{val}%")
        # Update hint color dynamically based on zone
        if val <= 40:
            color, tip = "#a6e3a1", "✦ Sweet spot: 25–40% detects people & objects  |  60%+ for strict person-only mode  |  90%+ rarely detects anything"
        elif val <= 65:
            color, tip = "#f9e2af", "⚠ Medium zone: 40–65% — mostly detects people, fewer objects. Good for busy scenes."
        else:
            color, tip = "#f38ba8", "✕ Strict mode: 65%+ — only very confident detections shown. Objects rarely detected above 75%."
        self.ai_acc_lbl.configure(text_color=color)
        self.acc_hint_lbl.configure(text=tip, text_color=color)

    # ─────────────────────────────────────────────────────────────────────────
    # Frame update loop
    # ─────────────────────────────────────────────────────────────────────────
    def _update_frame(self):
        if not self.running:
            return

        ret, raw = self.cap.read()
        if not ret or raw is None:
            # Camera error - try to reopen
            print("Camera read failed, attempting to reopen...")
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                ret, raw = self.cap.read()
            if not ret or raw is None:
                # Still failed - show error and stop
                self.status_lbl.configure(text="● Camera Error")
                self.root.after(self._frame_interval_ms, self._update_frame)
                return

        # Get current video label size
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        if label_width <= 1 or label_height <= 1:
            label_width, label_height = WIDTH, HEIGHT  # fallback

        raw_height, raw_width = raw.shape[:2]
        
        # Maintain native aspect ratio without stretching
        aspect = raw_width / raw_height
        if label_width / label_height > aspect:
            display_width = int(label_height * aspect)
            display_height = label_height
        else:
            display_width = label_width
            display_height = int(label_width / aspect)

        frame = cv2.resize(raw, (display_width, display_height))
        original = raw.copy()   # clean copy → written to video

        # Scale factors for AI coordinates
        self.scale_x = display_width / raw_width
        self.scale_y = display_height / raw_height

        # Motion detection
        frame, motion_detected = self._process_frame(frame)

        # Pull latest AI result (non-blocking)
        try:
            results = self._ai_result_queue.get_nowait()
            if isinstance(results, dict):
                if not results:  # AI was turned off, clear memory
                    with self._ai_result_lock:
                        self._ai_boxes.clear()
                        self._ai_poses.clear()
                        self._ai_objects.clear()
                        self._ai_actions.clear()
                        self._ai_stale_frames = 0
                else:
                    with self._ai_result_lock:
                        new_boxes   = results.get("boxes", [])
                        new_poses   = results.get("poses", [])
                        new_objects = results.get("objects", [])
                        new_actions = results.get("actions", [])

                        # keep detections for a few frames if the model misses intermittently
                        # (Now correctly tracks generic objects alongside humans)
                        if new_boxes or new_objects or self._ai_stale_frames >= self._ai_max_stale:
                            self._ai_boxes   = new_boxes
                            self._ai_poses   = new_poses
                            self._ai_objects = new_objects
                            self._ai_actions = new_actions
                            self._ai_stale_frames = 0
                        else:
                            self._ai_stale_frames += 1

                        # Update Action Log
                        for action in self._ai_actions:
                            if action not in ["Standing", "Detecting...", "Unknown"]:
                                ts = datetime.datetime.now().strftime("%H:%M:%S")
                                log_entry = f"[{ts}] {action}"
                                if not self.action_history or self.action_history[-1] != action:
                                    self.log_list.insert(0, log_entry)
                                    self.action_history.append(action)
                                    if len(self.action_history) > 50: self.action_history.pop(0)
            self._ai_pending = False
        except queue.Empty:
            pass

        # Visual Alarm
        alarm_trigger = False

        # Draw AI info
        person_detected = False
        if self.ai_enabled.get():
            with self._ai_result_lock:
                boxes   = list(self._ai_boxes)
                poses   = list(self._ai_poses)
                actions = list(self._ai_actions)
                objects = list(self._ai_objects)

            # Draw generic objects (boxes, laptops, etc.)
            for (ox1, oy1, ox2, oy2, cls, name, _, conf) in objects:
                ox1 = int(ox1 * self.scale_x)
                oy1 = int(oy1 * self.scale_y)
                ox2 = int(ox2 * self.scale_x)
                oy2 = int(oy2 * self.scale_y)
                box_color = (180, 180, 180)
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), box_color, 2)
                
                label = name.replace('_', ' ').title()
                if label == "Cell Phone": label = "Smartphone"
                label = f"{label} {int(conf*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(oy1 - th - 5, 15)
                # Background rect for text
                cv2.rectangle(frame, (ox1, ty - 5), (ox1 + tw + 6, ty + th + 5), box_color, -1)
                cv2.putText(frame, label, (ox1 + 3, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

            # Draw persons and actions
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                x1 = int(x1 * self.scale_x)
                y1 = int(y1 * self.scale_y)
                x2 = int(x2 * self.scale_x)
                y2 = int(y2 * self.scale_y)
                person_detected = True
                action = actions[i] if i < len(actions) else "..."

                # Highlight alarm actions
                color = (255, 140, 0) # Normal person color (Orange/Blueish)
                text_color = (255, 255, 255)
                if "FALL DETECTED" in action or "Picking Up" in action:
                    color = (0, 0, 255) # Red for critical
                    alarm_trigger = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                label_p = f"Person: {action.replace('cell phone', 'Smartphone')}"
                (pw, ph), _ = cv2.getTextSize(label_p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                py = max(y1 - ph - 6, 20)
                # Background rect for person text
                cv2.rectangle(frame, (x1, py - 6), (x1 + pw + 8, py + ph + 6), color, -1)
                cv2.putText(frame, label_p, (x1 + 4, py + ph), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

                # Draw skeleton if available
                if i < len(poses):
                    for kp in poses[i]:
                        x, y, conf = kp
                        if conf > 0.5:
                            x = int(x * self.scale_x)
                            y = int(y * self.scale_y)
                            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Submit next frame into the AI queue if there is room.
            try:
                self._ai_queue.put_nowait({
                    "frame": original.copy(),
                    "motion": motion_detected,
                    "accuracy": self.ai_accuracy.get(),  # Send the slider's value to the worker
                    "enabled": self.ai_enabled.get()
                })
            except queue.Full:
                pass

        # Visual Alarm border
        if alarm_trigger and self.alarm_enabled.get():
            cv2.rectangle(frame, (0, 0), (WIDTH, HEIGHT), (0, 0, 255), 10)

        # Recording logic
        auto_trigger_condition = (motion_detected or person_detected) and self.auto_record_enabled.get()
        trigger = auto_trigger_condition or self.manual_recording
        if trigger:
            self.last_motion_time = time.time()
            if not self.recording:
                self._start_recording()

        if self.recording:
            if self.out:
                self.out.write(original)
            elapsed_since_motion = time.time() - self.last_motion_time
            if elapsed_since_motion > self.record_duration.get() and not self.manual_recording:
                self._stop_recording()

        # Recording indicator on frame
        if self.recording:
            cv2.circle(frame, (WIDTH - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (WIDTH - 55, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Convert and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb)
        imgtk     = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Update status bar
        parts = []
        if motion_detected:  parts.append("Motion")
        if person_detected:  parts.append("Person")
        if self.manual_recording: parts.append("Manual REC")
        status_text = "● " + " | ".join(parts) if parts else "● Monitoring…"
        self.status_lbl.configure(text=status_text)
        self.rec_status_lbl.configure(
            text="⏺ RECORDING" if self.recording else "")

        self.root.after(self._frame_interval_ms, self._update_frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Motion detection
    # ─────────────────────────────────────────────────────────────────────────
    def _process_frame(self, frame):
        motion_detected = False
        
        # PERFORMANCE OPTIMIZATION: Calculate motion against a fixed 640x360 proxy structure.
        # This prevents exponential CPU lag when maximizing the display window while keeping 
        # your sensitivity sliders permanently calibrated regardless of window scale.
        proxy = cv2.resize(frame, (640, 360))
        
        mask   = self.fgbg.apply(proxy)
        mask   = cv2.medianBlur(mask, 5)
        _, thresh = cv2.threshold(mask, self.threshold_val.get(), 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_motion = False
        for c in contours:
            if cv2.contourArea(c) > self.min_area.get():
                raw_motion = True
                break

        # Temporal filtering to reduce flickering
        if raw_motion:
            self.motion_counter = min(self.motion_counter + 1, 7)
        else:
            self.motion_counter = max(self.motion_counter - 1, 0)

        if self.motion_counter >= 5:
            motion_detected = True

        return frame, motion_detected

    # ─────────────────────────────────────────────────────────────────────────
    # AI worker thread
    # ─────────────────────────────────────────────────────────────────────────
    def _ai_worker(self):
        """Processes detection and pose in background."""
        frame_counter = 0
        # Per-person action vote history for smoothing (keyed by person index slot)
        from collections import deque, Counter as _Counter
        action_votes = {}  # slot_idx -> deque of recent action strings
        VOTE_WINDOW = 10   # smooth over last 10 inference results
        while True:
            try:
                data = self._ai_queue.get(timeout=1)
                frame = data["frame"]
                motion = data["motion"]
                accuracy_req = data.get("accuracy", 30.0) / 100.0  # Convert 30.0 to 0.30
            except queue.Empty:
                if not self.running: break
                continue

            if not (self.model and data.get("enabled", True)):
                try:
                    self._ai_result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._ai_result_queue.put_nowait({})
                except queue.Full:
                    pass
                frame_counter += 1
                continue

            result_payload = {}
            try:
                # ── Step 1: Detect Persons & Objects ──────────────────────────
                # Optimization: Run detection every 2 frames or if motion detected
                boxes = []
                objects = []
                if frame_counter % 2 == 0 or motion:
                    # Dynamically adjust threshold based on user slider!
                    results_det = self.model(frame, verbose=False, iou=0.45, conf=accuracy_req)
                    
                    for r in results_det:
                        for b in r.boxes:
                            cls = int(b.cls)
                            coords = map(int, b.xyxy[0])
                            x1, y1, x2, y2 = coords
                            name = self.class_names.get(cls, f"class_{cls}")
                            confidence = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                            if confidence < accuracy_req:
                                continue
                            if cls == 0:  # person
                                boxes.append((x1, y1, x2, y2))
                            else:
                                objects.append((x1, y1, x2, y2, cls, name, 0.0, confidence))


                # ── Step 1.6: Depth Estimation for Objects ─────────────────────
                if self.depth_model and objects and frame_counter % 10 == 0:  # every 10 frames for performance
                    try:
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        depth_result = self.depth_model(pil_image)
                        depth_map = depth_result['depth']
                        for i, obj in enumerate(objects):
                            x1, y1, x2, y2, cls, name, _, confidence = obj
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
                                depth_val = float(depth_map[center_y, center_x])
                            else:
                                depth_val = 0.0
                            objects[i] = (x1, y1, x2, y2, cls, name, depth_val, confidence)
                    except Exception as e:
                        print(f"Depth estimation error: {e}")

                # ── Step 2: Pose Estimation for detected persons ──────────────
                poses = []
                actions = []
                if boxes and self.pose_model and frame_counter % 3 == 0:
                    # Run pose only when people are detected, to reduce CPU use.
                    results_pose = self.pose_model(frame, verbose=False, conf=accuracy_req)
                    for r in results_pose:
                         if r.keypoints is not None:
                            all_kpts = r.keypoints.data.cpu().numpy()
                            for i in range(len(all_kpts)):
                                kpts = all_kpts[i]
                                raw_action = self.action_engine.get_action(kpts, objects, None)

                                # Per-person vote smoothing
                                if i not in action_votes:
                                    action_votes[i] = deque(maxlen=VOTE_WINDOW)
                                action_votes[i].append(raw_action)
                                # Pick majority vote; fall back to raw if tie
                                most_common = _Counter(action_votes[i]).most_common(1)[0][0]
                                poses.append(kpts)
                                actions.append(most_common)

                # Prune vote history to current person count
                for stale in [k for k in action_votes if k >= len(boxes)]:
                    del action_votes[stale]

                result_payload = {
                    "boxes": boxes,
                    "poses": poses,
                    "objects": objects,
                    "actions": actions
                }
            except Exception as e:
                print(f"AI Error: {e}")
                result_payload = {}
            finally:
                try:
                    self._ai_result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._ai_result_queue.put_nowait(result_payload)
                except queue.Full:
                    pass
                frame_counter += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Recording helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _start_recording(self):
        name   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
        path   = os.path.join(RECORD_DIR, name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # Use camera resolution for recording
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(path, fourcc, FPS_TARGET, (width, height))
        self.recording = True

    def _stop_recording(self):
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None

    # ─────────────────────────────────────────────────────────────────────────
    # Button handlers
    # ─────────────────────────────────────────────────────────────────────────
    def _snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            path = os.path.join(SNAP_DIR, name)
            cv2.imwrite(path, frame)
            messagebox.showinfo("Snapshot Saved", f"Saved to:\n{os.path.abspath(path)}")

    def _toggle_record(self):
        self.manual_recording = not self.manual_recording
        if self.manual_recording:
            self.rec_btn.configure(text="⏹  Stop Recording")
            if not self.recording:
                self._start_recording()
        else:
            self.rec_btn.configure(text="⏺  Manual Record")
            # Let the auto-stop timer handle release naturally
            self.last_motion_time = time.time()

    def _open_recordings(self):
        files = sorted(os.listdir(RECORD_DIR), reverse=True)
        win   = ctk.CTkToplevel(self.root)
        win.title("Saved Recordings")
        win.configure(fg_color="#1e1e2e")
        win.geometry("380x320")

        ctk.CTkLabel(win, text="Saved Recordings", font=("Segoe UI", 14, "bold")).pack(pady=8)

        listbox = tk.Listbox(win, bg="#313244", fg="#cdd6f4",
                             selectbackground="#585b70",
                             relief="flat", bd=0, font=("Segoe UI", 9))
        listbox.pack(fill="both", expand=True, padx=12, pady=4)

        if files:
            for f in files:
                listbox.insert("end", f)
        else:
            listbox.insert("end", "(no recordings yet)")

        def _open_folder():
            os.startfile(os.path.abspath(RECORD_DIR))

        ctk.CTkButton(win, text="Open Folder", command=_open_folder).pack(pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        self._stop_recording()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = ctk.CTk()
    app = MotionDetectorApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("App interrupted by user")
        root.destroy()
