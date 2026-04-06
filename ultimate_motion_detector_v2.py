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
from PIL import Image, ImageTk
import datetime
import os
import time
import threading
import queue

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
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ultimate AI Motion Detection System")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── State variables ───────────────────────────────────────────────────
        self.recording        = False
        self.manual_recording = False
        self.out: cv2.VideoWriter | None = None
        self.last_motion_time = 0.0
        self.running          = True

        # AI detection state (updated from bg thread)
        self._ai_result_lock    = threading.Lock()
        self._ai_boxes: list    = []          # list of (x1,y1,x2,y2) ints
        self._ai_pending        = False       # True while a frame is being processed
        self._ai_queue: queue.Queue = queue.Queue(maxsize=1)
        self._ai_result_queue: queue.Queue = queue.Queue(maxsize=1)

        # ── Camera ────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Cannot open camera. Please check your webcam connection.")
            self.root.destroy()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        # ── Background subtractor ─────────────────────────────────────────────
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)

        # ── YOLO model ────────────────────────────────────────────────────────
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")   # downloads weights automatically
            except Exception as e:
                messagebox.showwarning("YOLO Warning",
                                       f"Could not load YOLO model:\n{e}\n\n"
                                       "AI detection will be disabled.")

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
        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".",        background="#1e1e2e", foreground="#cdd6f4",
                        font=("Segoe UI", 9))
        style.configure("TLabel",   background="#1e1e2e", foreground="#cdd6f4")
        style.configure("TFrame",   background="#1e1e2e")
        style.configure("TButton",  background="#313244", foreground="#cdd6f4",
                        padding=4, relief="flat")
        style.map("TButton",
                  background=[("active", "#45475a"), ("pressed", "#585b70")])
        style.configure("TCheckbutton", background="#1e1e2e", foreground="#cdd6f4")
        style.configure("TScale",       background="#1e1e2e")
        style.configure("Status.TLabel", background="#181825", foreground="#a6e3a1",
                        font=("Segoe UI", 9, "bold"), padding=(6, 3))
        style.configure("Rec.TLabel",   background="#181825", foreground="#f38ba8",
                        font=("Segoe UI", 9, "bold"), padding=(6, 3))

        self.root.configure(bg="#1e1e2e")

        # ── Video area ────────────────────────────────────────────────────────
        video_frame = ttk.Frame(self.root)
        video_frame.grid(row=0, column=0, columnspan=3, padx=8, pady=(8, 4))

        self.video_label = tk.Label(video_frame, bg="black",
                                    width=WIDTH, height=HEIGHT)
        self.video_label.pack()

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ttk.Frame(self.root)
        ctrl.grid(row=1, column=0, columnspan=3, padx=8, pady=4, sticky="ew")
        ctrl.columnconfigure(1, weight=1)

        # Min Motion Area
        self.min_area = tk.IntVar(value=800)
        ttk.Label(ctrl, text="Min Motion Area").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Scale(ctrl, from_=100, to=5000, variable=self.min_area,
                  orient="horizontal").grid(row=0, column=1, sticky="ew", padx=4)
        self.min_area_val_lbl = ttk.Label(ctrl, text="800", width=6)
        self.min_area_val_lbl.grid(row=0, column=2, sticky="w")
        self.min_area.trace_add("write",
            lambda *_: self.min_area_val_lbl.config(text=str(self.min_area.get())))

        # Threshold
        self.threshold_val = tk.IntVar(value=25)
        ttk.Label(ctrl, text="Threshold").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Scale(ctrl, from_=1, to=100, variable=self.threshold_val,
                  orient="horizontal").grid(row=1, column=1, sticky="ew", padx=4)
        self.thresh_val_lbl = ttk.Label(ctrl, text="25", width=6)
        self.thresh_val_lbl.grid(row=1, column=2, sticky="w")
        self.threshold_val.trace_add("write",
            lambda *_: self.thresh_val_lbl.config(text=str(self.threshold_val.get())))

        # Record Seconds
        self.record_duration = tk.IntVar(value=10)
        ttk.Label(ctrl, text="Record Seconds").grid(row=2, column=0, sticky="w", padx=4)
        ttk.Scale(ctrl, from_=5, to=60, variable=self.record_duration,
                  orient="horizontal").grid(row=2, column=1, sticky="ew", padx=4)
        self.rec_dur_lbl = ttk.Label(ctrl, text="10", width=6)
        self.rec_dur_lbl.grid(row=2, column=2, sticky="w")
        self.record_duration.trace_add("write",
            lambda *_: self.rec_dur_lbl.config(text=str(self.record_duration.get())))

        # AI enable checkbox
        self.ai_enabled = tk.BooleanVar(value=True)
        ai_chk = ttk.Checkbutton(ctrl, text="Enable AI Person Detection",
                                  variable=self.ai_enabled)
        ai_chk.grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=4)
        if not (YOLO_AVAILABLE and self.model):
            self.ai_enabled.set(False)
            ai_chk.config(state="disabled")

        # ── Button row ────────────────────────────────────────────────────────
        btn_row = ttk.Frame(self.root)
        btn_row.grid(row=2, column=0, columnspan=3, padx=8, pady=4, sticky="ew")

        ttk.Button(btn_row, text="📷  Snapshot",
                   command=self._snapshot).pack(side="left", padx=4, expand=True, fill="x")

        self.rec_btn = ttk.Button(btn_row, text="⏺  Manual Record",
                                  command=self._toggle_record)
        self.rec_btn.pack(side="left", padx=4, expand=True, fill="x")

        ttk.Button(btn_row, text="📂  Browse Recordings",
                   command=self._open_recordings).pack(side="left", padx=4, expand=True, fill="x")

        # ── Status bar ────────────────────────────────────────────────────────
        status_row = ttk.Frame(self.root, style="TFrame")
        status_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=0, pady=(4, 0))

        self.status_lbl = ttk.Label(status_row, text="● Ready", style="Status.TLabel")
        self.status_lbl.pack(side="left")

        self.rec_status_lbl = ttk.Label(status_row, text="", style="Rec.TLabel")
        self.rec_status_lbl.pack(side="right")

    # ─────────────────────────────────────────────────────────────────────────
    # Frame update loop
    # ─────────────────────────────────────────────────────────────────────────
    def _update_frame(self):
        if not self.running:
            return

        ret, raw = self.cap.read()
        if not ret:
            self.root.after(self._frame_interval_ms, self._update_frame)
            return

        frame = cv2.resize(raw, (WIDTH, HEIGHT))
        original = frame.copy()   # clean copy → written to video

        # Motion detection
        frame, motion_detected = self._process_frame(frame)

        # Pull latest AI result (non-blocking)
        try:
            ai_boxes = self._ai_result_queue.get_nowait()
            with self._ai_result_lock:
                self._ai_boxes = ai_boxes
            self._ai_pending = False
        except queue.Empty:
            pass

        # Draw AI boxes from last result
        person_detected = False
        if self.ai_enabled.get():
            with self._ai_result_lock:
                boxes = list(self._ai_boxes)
            for (x1, y1, x2, y2) in boxes:
                person_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
                cv2.putText(frame, "Person", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 2)

            # Submit new frame to AI thread if it's idle
            if not self._ai_pending:
                try:
                    self._ai_queue.put_nowait(frame.copy())
                    self._ai_pending = True
                except queue.Full:
                    pass

        # Recording logic
        trigger = motion_detected or person_detected or self.manual_recording
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
        self.status_lbl.config(text=status_text)
        self.rec_status_lbl.config(
            text="⏺ RECORDING" if self.recording else "")

        self.root.after(self._frame_interval_ms, self._update_frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Motion detection
    # ─────────────────────────────────────────────────────────────────────────
    def _process_frame(self, frame):
        motion_detected = False
        mask   = self.fgbg.apply(frame)
        _, thresh = cv2.threshold(mask, self.threshold_val.get(), 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) > self.min_area.get():
                motion_detected = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 100), 2)

        return frame, motion_detected

    # ─────────────────────────────────────────────────────────────────────────
    # AI worker thread
    # ─────────────────────────────────────────────────────────────────────────
    def _ai_worker(self):
        """Runs in a background thread; processes frames from _ai_queue."""
        while True:
            try:
                frame = self._ai_queue.get(timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                continue

            if not (self.model and self.ai_enabled.get()):
                self._ai_result_queue.put([])
                continue

            try:
                results = self.model(frame, verbose=False)
                boxes = []
                for r in results:
                    for b in r.boxes:
                        if int(b.cls) == 0:   # class 0 = person
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            boxes.append((x1, y1, x2, y2))
                # Drain old result and push new one
                try:
                    self._ai_result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._ai_result_queue.put(boxes)
            except Exception:
                pass   # silently ignore inference errors

    # ─────────────────────────────────────────────────────────────────────────
    # Recording helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _start_recording(self):
        name   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
        path   = os.path.join(RECORD_DIR, name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(path, fourcc, FPS_TARGET, (WIDTH, HEIGHT))
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
            self.rec_btn.config(text="⏹  Stop Recording")
            if not self.recording:
                self._start_recording()
        else:
            self.rec_btn.config(text="⏺  Manual Record")
            # Let the auto-stop timer handle release naturally
            self.last_motion_time = time.time()

    def _open_recordings(self):
        files = sorted(os.listdir(RECORD_DIR), reverse=True)
        win   = tk.Toplevel(self.root)
        win.title("Saved Recordings")
        win.configure(bg="#1e1e2e")
        win.geometry("380x320")

        ttk.Label(win, text="Saved Recordings", font=("Segoe UI", 11, "bold")).pack(pady=8)

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

        ttk.Button(win, text="Open Folder", command=_open_folder).pack(pady=8)

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
    root = tk.Tk()
    app  = MotionDetectorApp(root)
    root.mainloop()
