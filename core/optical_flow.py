import cv2
import numpy as np
import math

# Theo doi chuyen dong dot ngot (Optical Flow dua tren Centroid)
class MotionTracker:
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.prev_centroids = {}
        self.motion_data = {}
        
    def _default_motion(self):
        return {"magnitude": 0.0, "sudden": False, "moving": False, "still": True}
        
    # Cap nhat toa do centroid va tinh do dich chuyen
    def update(self, frame, faces):
        info = {}
        for f in faces:
            tid = f.get("track_id", -1)
            if tid == -1:
                continue
            cx, cy = f["centroid"]
            fw = f["face_width"]
            
            if tid not in self.prev_centroids:
                self.prev_centroids[tid] = (cx, cy)
                info[tid] = self._default_motion()
                self.motion_data[tid] = info[tid]
                continue
                
            pcx, pcy = self.prev_centroids[tid]
            dx = cx - pcx
            dy = cy - pcy
            dist = math.hypot(dx, dy)
            
            # Chuan hoa khoang cach dich chuyen theo chieu rong mat
            norm_dist = dist / max(fw, 1.0)
            
            # Kiem tra chuyen dong dot ngot (vuot 4% do rong mat trong 1 frame)
            is_sudden = norm_dist > 0.04
            is_moving = norm_dist > 0.01
            
            info[tid] = {
                "magnitude": dist,
                "sudden": is_sudden,
                "moving": is_moving,
                "still": not is_moving
            }
            self.motion_data[tid] = info[tid]
            self.prev_centroids[tid] = (cx, cy)
            
        return info
        
    def get(self, tid):
        return self.motion_data.get(tid, self._default_motion())
        
    def reset(self):
        self.prev_centroids.clear()
        self.motion_data.clear()
