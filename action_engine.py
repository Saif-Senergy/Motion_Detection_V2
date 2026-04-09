import numpy as np
import time
from collections import Counter

class ActionEngine:
    def __init__(self):
        # Rolling history of keypoints for each tracked person {track_id: list of points}
        self.history = {}
        self.max_history = 30  # ~1.5 seconds at 20fps
        
        # Action history for stability
        self.action_history = []
        self.action_history_max = 30  # keep last 30 actions
        self.stable_action = "Unknown"
        self.change_threshold = 0.8  # require 80% consistency to change
        
        # Action labels
        self.LABEL_STABLE = "Standing"
        self.LABEL_WALK   = "Walking"
        self.LABEL_RUN    = "Running"
        self.LABEL_SIT    = "Sitting"
        self.LABEL_BEND   = "Bending"
        self.LABEL_PICK   = "Picking Up"
        self.LABEL_FALL   = "FALL DETECTED"
        self.LABEL_WAVE   = "Waving"

    def get_action(self, keypoints, objects, person_box):
        """
        keypoints: list of (x, y, conf) or numpy array (17, 3)
        objects: list of (x1, y1, x2, y2, cls_id, name, depth, confidence)
        person_box: (x1, y1, x2, y2)
        """
        if keypoints is None or len(keypoints) < 17:
            return "Unknown"
        
        return self._classify_action(keypoints, objects, person_box)

    def _classify_action(self, keypoints, objects, person_box):
        # Extract useful points (YOLO Pose indices: 
        # 0:nose, 5:l_shoulder, 6:r_shoulder, 11:l_hip, 12:r_hip, 13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle, 9:l_wrist, 10:r_wrist)
        def get_pt(idx): return keypoints[idx][:2]
        def get_conf(idx): return keypoints[idx][2]
        
        try:
            l_shoulder, r_shoulder = get_pt(5), get_pt(6)
            l_hip, r_hip           = get_pt(11), get_pt(12)
            l_knee, r_knee         = get_pt(13), get_pt(14)
            l_ankle, r_ankle       = get_pt(15), get_pt(16)
            
            # Confidence check for critical structural points
            has_knees = get_conf(13) > 0.4 and get_conf(14) > 0.4
            has_ankles = get_conf(15) > 0.4 and get_conf(16) > 0.4
            
            # 1. Height analysis
            hip_height   = (l_hip[1] + r_hip[1]) / 2
            knee_height  = (l_knee[1] + r_knee[1]) / 2
            ankle_height = (l_ankle[1] + r_ankle[1]) / 2
            shoulder_height = (l_shoulder[1] + r_shoulder[1]) / 2
            
            # 2. Falling (Sudden horizontal orientation or vertical drop)
            h_dist = abs(l_shoulder[0] - r_shoulder[0])
            v_dist = abs(shoulder_height - hip_height)
            if h_dist > v_dist * 2.0:  # more horizontal than vertical
                return self.LABEL_FALL

            # 3. Sitting vs Standing
            if has_knees and has_ankles:
                upper_leg = abs(hip_height - knee_height)
                lower_leg = abs(knee_height - ankle_height)
                if upper_leg < lower_leg * 0.6:
                    return self.LABEL_SIT

            # 4. Bending / Picking Up
            is_bending = False
            if shoulder_height > hip_height - 20: # Shoulders dropped near hips
                is_bending = True

            # Check interaction with any object
            sorted_objects = sorted(objects, key=lambda o: len(o) > 6 and o[6] or 0)
            for obj in sorted_objects:
                ox1, oy1, ox2, oy2 = obj[:4]
                name = obj[5] if len(obj)>5 else "Object"
                for wrist_idx in [9, 10]:
                    if get_conf(wrist_idx) > 0.4:
                        wx, wy = get_pt(wrist_idx)
                        if (ox1-20 <= wx <= ox2+20) and (oy1-20 <= wy <= oy2+20):
                            if is_bending:
                                return f"Picking Up {name}"
                            elif (ox1-20 <= wx <= ox2+20) and (oy1-20 <= wy <= oy2+20):
                                return f"Holding {name}"
            
            if is_bending:
                return self.LABEL_BEND

            # 5. Waving
            for wrist_idx in [9, 10]:
                if get_conf(wrist_idx) > 0.4:
                    wx, wy = get_pt(wrist_idx)
                    if wy < shoulder_height - 30:
                        return self.LABEL_WAVE

            return self.LABEL_STABLE

        except Exception:
            return "Detecting..."

    def calculate_angle(self, a, b, c):
        """Calculates angle abc."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return angle
