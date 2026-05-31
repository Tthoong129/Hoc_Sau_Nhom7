import time
import math
from collections import deque

# Dai dien cho mot doi tuong khuon mat dang duoc theo doi
class TrackedFace:
    def __init__(self, tid, centroid, width, yaw_ratio=1.0):
        self.tid = tid
        self.anchor_x = centroid[0]
        self.anchor_y = centroid[1]
        self.anchor_yaw = yaw_ratio
        self.last_seen = time.time()
        self.width = width
        self.anchor_width = width
    
    def update(self, centroid, width, yaw_ratio, is_violating):
        if not is_violating:
            self.anchor_x = self.anchor_x * 0.9 + centroid[0] * 0.1
            self.anchor_y = self.anchor_y * 0.9 + centroid[1] * 0.1
            self.anchor_yaw = self.anchor_yaw * 0.9 + yaw_ratio * 0.1
            self.anchor_width = getattr(self, 'anchor_width', width) * 0.9 + width * 0.1
        self.last_seen = time.time()
        self.width = width

# Quan ly va dinh danh (ID) cho cac vi tri hoc sinh ngoi trong lop
class SeatTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.timeout = 2.0
        
    # Ghep cap khuon mat moi voi ID track cu dua tren khoang cach
    def match_faces(self, faces):
        matched = []
        now = time.time()
        
        # Xoa tracks cu da roi khoi khung hinh qua thoi gian timeout
        to_remove = [tid for tid, tr in self.tracks.items() if now - tr.last_seen > self.timeout]
        for tid in to_remove:
            del self.tracks[tid]
            
        assigned_ids = set()
        for face in faces:
            cx, cy = face["centroid"]
            best_id = None
            best_dist = float('inf')
            for tid, tr in self.tracks.items():
                if tid in assigned_ids:
                    continue
                # Kiem tra ty le do rong mat de tranh nhay ID sang hang ban khac
                width_ratio = face["face_width"] / max(tr.width, 1e-6)
                if not (0.7 <= width_ratio <= 1.45):
                    continue
                    
                dist = math.hypot(cx - tr.anchor_x, cy - tr.anchor_y)
                if dist < max(tr.width, face["face_width"]) * 0.9 and dist < best_dist:
                    best_dist = dist
                    best_id = tid
        
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = TrackedFace(best_id, (cx, cy), face["face_width"], face.get("yaw_ratio", 1.0))
            else:
                assigned_ids.add(best_id)
            
            face["track_id"] = best_id
            matched.append((best_id, face))
        return matched
        
    def get_track(self, tid):
        return self.tracks.get(tid)
        
    def reset(self):
        self.tracks.clear()
        self.next_id = 1

# Bo dem thoi gian duy tri trang thai vi pham lien tuc
class SustainedTimer:
    def __init__(self, threshold):
        self.threshold = threshold
        self.timers = {}
        
    # Kiem tra dieu kien da duy tri du lau chua
    def check(self, key, condition):
        if condition:
            if key not in self.timers:
                self.timers[key] = time.time()
            return (time.time() - self.timers[key]) >= self.threshold
        else:
            self.timers.pop(key, None)
            return False
            
    def elapsed(self, key):
        if key in self.timers:
            return time.time() - self.timers[key]
        return 0.0
        
    def reset(self, key):
        self.timers.pop(key, None)
        
    def reset_all(self):
        self.timers.clear()

# Bo loc nhieu bang cua so truot (Sliding Window) cho hanh vi pose
class ViolationVoter:
    def __init__(self, window=10, vote_thresh=0.45, min_frames=3):
        self.window = window
        self.vote_thresh = vote_thresh
        self.min_frames = min_frames
        self.history = {}
        
    # Nhap ket qua tung frame va kiem tra ty le vi pham trong cua so truot
    def vote(self, key, is_violating):
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        self.history[key].append(1 if is_violating else 0)
        
        if len(self.history[key]) < self.min_frames:
            return False
        ratio = sum(self.history[key]) / len(self.history[key])
        return ratio >= self.vote_thresh
        
    def reset(self, key):
        self.history.pop(key, None)
        
    def reset_all(self):
        self.history.clear()

# Theo doi van toc thay doi goc quay de phat hien liec bai (liên tục quay đầu)
class SignalVelocityTracker:
    def __init__(self):
        self._prev_signals = {}
        self._yaw_dir_history = {}
        self.OSC_DIR_CHANGES = 6
        
    # Tinh so lan doi huong quay cua dau hoc sinh
    def update(self, tid, face_data):
        cur_yaw = face_data.get("yaw_ratio", 1.0)
        cur_roll = face_data.get("roll_angle", 0.0)
        cur_twist = abs(face_data.get("body_twist", 0.0))
        cur_arm = face_data.get("arm_reach_dist", 0.0)
        
        result = {"oscillating": False, "velocity_bad": False}
        if tid not in self._prev_signals:
            self._prev_signals[tid] = {"yaw": cur_yaw, "roll": cur_roll, "twist": cur_twist, "arm": cur_arm}
            self._yaw_dir_history[tid] = deque(maxlen=10)
            return result
            
        prev = self._prev_signals[tid]
        yaw_diff = cur_yaw - prev["yaw"]
        
        if abs(yaw_diff) > 0.05:
            d = 1 if yaw_diff > 0 else -1
            self._yaw_dir_history[tid].append(d)
            
        dir_changes = 0
        if len(self._yaw_dir_history[tid]) >= 3:
            hist = list(self._yaw_dir_history[tid])
            prev_dir = hist[0]
            for d in hist[1:]:
                if d != prev_dir:
                    dir_changes += 1
                prev_dir = d
                
        result["oscillating"] = dir_changes >= self.OSC_DIR_CHANGES
        result["velocity_bad"] = result["oscillating"]
        
        self._prev_signals[tid] = {
            "yaw": cur_yaw, "roll": cur_roll,
            "twist": cur_twist, "arm": cur_arm,
        }
        return result
        
    def remove(self, tid: int):
        self._prev_signals.pop(tid, None)
        self._yaw_dir_history.pop(tid, None)

    def reset(self):
        self._prev_signals.clear()
        self._yaw_dir_history.clear()

# Bo loc nhieu bang cua so truot rieng cho phat hien doi tuong YOLO
class YOLOClassVoter:
    def __init__(self, window=8, presence_thresh=0.55, conf_thresh=0.0):
        self.window = window
        self.presence_thresh = presence_thresh
        self.conf_thresh = conf_thresh
        self._history = {}

    # Bieu quyet do tin cay cua vat the qua nhieu frame truoc khi dua ra canh bao
    def vote(self, class_name: str, detected: bool, confidence: float = 0.0) -> bool:
        if class_name not in self._history:
            self._history[class_name] = deque(maxlen=self.window)
        self._history[class_name].append(confidence if detected else 0.0)
        buf = self._history[class_name]
        if len(buf) < 3:
            return False
        present_frames = sum(1 for c in buf if c > 0)
        presence_ratio = present_frames / len(buf)
        avg_conf = sum(c for c in buf if c > 0) / max(present_frames, 1)
        return presence_ratio >= self.presence_thresh and avg_conf >= self.conf_thresh

    def reset(self, class_name: str = None):
        if class_name:
            self._history.pop(class_name, None)
        else:
            self._history.clear()
