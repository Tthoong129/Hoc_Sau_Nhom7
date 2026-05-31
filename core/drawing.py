import cv2
import numpy as np
from .config import (
    CLASS_COLORS, SEV_COLOR,
    YAW_THRESHOLD, YAW_THRESHOLD_LOW, LEAN_SHIFT_RATIO,
)

_CLASS_LABEL_ASCII = {
    "CheatingPaper": "GIAY PHAO",
    "earphone":      "TAI NGHE",
    "HandGestures":  "RA DAU TAY",
    "Phone":         "DIEN THOAI",
    "Rotation":      "QUAY DAU",
    "Sleeping":      "NGU GAT",
    "SmartWatch":    "DONG HO",
    "normal":        "BINH THUONG",
}

_ALERT_LABEL_ASCII = {
    "HEAD_TURN":     "QUAY DAU",
    "BODY_LEAN":     "NHOAI NGUOI",
    "MULTI_FACE":    "NHIEU NGUOI",
    "CheatingPaper": "GIAY PHAO",
    "earphone":      "TAI NGHE",
    "HandGestures":  "RA DAU TAY",
    "Phone":         "DIEN THOAI",
    "Rotation":      "QUAY DAU (YOLO)",
    "Sleeping":      "NGU GAT",
    "SmartWatch":    "DONG HO",
}

# Ve hop nhan dien cho cac doi tuong YOLO (Dien thoai, Phao thi,...)
def draw_yolo_box(frame: np.ndarray, det: dict):
    x1, y1, x2, y2 = det["bbox"]
    cls = det["class_name"]
    conf = det["confidence"]
    sev = det.get("severity", "LOW")
    
    # Mau sac theo do nghiem trong (High = Do, Medium = Cam, Low = Xanh)
    if sev == "HIGH":
        color = (0, 0, 255)       
    elif sev == "MEDIUM":
        color = (0, 200, 255)    
    else:
        color = (0, 255, 0)     
        
    label = _CLASS_LABEL_ASCII.get(cls, cls)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf:.2f}"
    _draw_label(frame, txt, x1, y1, color)

# Ve hop nhan dien khuon mat hoc sinh kem trang thai vi pham
def draw_face_box(frame: np.ndarray, bbox: tuple, tid: int,
                  violation_level: str, label_parts: list[str]):
    x1, y1, x2, y2 = bbox
    if violation_level == "HIGH":
        color = (0, 0, 255)     
        thick = 2
    elif violation_level == "MEDIUM":
        color = (0, 200, 255)  
        thick = 2
    elif violation_level == "WARNING":
        color = (0, 255, 255)    
        thick = 1
    else:
        color = (0, 255, 0)     
        thick = 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    label = f"ID:{tid} " + " | ".join(label_parts) if label_parts else f"ID:{tid} OK"
    _draw_label(frame, label, x1, y1, color)

# Ve thanh trang thai phia tren dau
def draw_head_bars(frame: np.ndarray, idx: int, yaw_r: float, x_diff_ratio: float):
    pass

# Hien thi so luong vi pham o goc tren ben trai man hinh
def draw_status_overlay(frame: np.ndarray, alerts: list[dict]):
    if not alerts:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    priority = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    top = min(alerts, key=lambda a: priority.get(a.get("severity", "LOW"), 2))
    short = _ALERT_LABEL_ASCII.get(top["type"], top["type"])
    n = len(alerts)
    label = f"! {short}" if n == 1 else f"! {n} VI PHAM ({short}...)"
    bg = SEV_COLOR.get(top.get("severity", "LOW"), (80, 80, 80))
    scale = 0.42
    thick = 1
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
    pad_x, pad_y = 8, 5
    bx1 = 6
    by1 = 6
    bx2 = bx1 + tw + pad_x * 2
    by2 = by1 + th + pad_y * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    cv2.putText(frame, label, (bx1 + pad_x, by2 - pad_y - 1),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# Hien thi thong tin tong hop YOLO
def draw_yolo_summary(frame: np.ndarray, detections: list[dict]):
    pass 

# Ve nhan chu thich (Label) di kem cac hop rectangle
def _draw_label(frame: np.ndarray, label: str, x: int, y: int, bg_color: tuple):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
    ly = max(y, th + 4)
    cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 4, ly + 2), bg_color, -1)
    cv2.putText(frame, label, (x + 2, ly - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# Ve panel thong tin phan tich chuyen dong pose de debug (khi bat che do xem chi tiet)
def draw_pose_debug(frame: np.ndarray, face_data: dict,
                    right_factors: int = 0, left_factors: int = 0,
                    yaw_bad: bool = False, lean_bad: bool = False,
                    of_mag: float = 0.0, of_sudden: bool = False):
    x1, y1, x2, y2 = face_data["bbox"]
    roll    = face_data.get("roll_angle", 0.0)
    yaw_r   = face_data.get("yaw_ratio", 1.0)
    reach   = face_data.get("arm_reach_dist", 0.0)
    twist   = abs(face_data.get("body_twist", 0.0))
    fwd     = face_data.get("lean_forward", False)
    pitch   = face_data.get("pitch_down", False)
    arm_s   = face_data.get("arm_reach_side", 0)
    tid     = face_data.get("track_id", -1)
    if yaw_bad or lean_bad:
        bg = (0, 0, 160)
    elif of_sudden:
        bg = (0, 80, 180)   
    elif right_factors >= 1 or left_factors >= 1:
        bg = (0, 100, 160)
    else:
        bg = (30, 30, 30)
    of_label = f"OF={of_mag:.1f}px" + (" [SUDDEN!]" if of_sudden else "")
    lines = [
        f"ID:{tid}  R={right_factors} L={left_factors}",
        f"roll={roll:+.1f}  yaw={yaw_r:.2f}",
        f"reach={reach:.2f}({'+'if arm_s>0 else'-'if arm_s<0 else'0'})"
        f"  twist={twist:.1f}",
        f"fwd={'Y' if fwd else 'N'}  pitch={'Y' if pitch else 'N'}"
        f"  {'!VIOLATION!' if (yaw_bad or lean_bad) else ''}",
        of_label,
    ]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thick = 1
    line_h = 14
    pad = 4
    max_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + pad * 2
    panel_h = line_h * len(lines) + pad * 2
    max_w = max(10, max_w)
    px1 = x1
    py1 = y2 + 2
    px2 = px1 + max_w
    py2 = py1 + panel_h
    fh, fw = frame.shape[:2]
    if py2 > fh:
        py1 = y1 - panel_h - 2
        py2 = y1 - 2
    cv2.rectangle(frame, (px1, py1), (px2, py2), bg, -1)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (80, 80, 80), 1)
    for i, line in enumerate(lines):
        ty = py1 + pad + (i + 1) * line_h - 2
        if i == len(lines) - 1 and of_sudden:
            color = (0, 220, 255)
        elif yaw_bad or lean_bad:
            color = (0, 255, 100)
        else:
            color = (200, 200, 200)
        cv2.putText(frame, line, (px1 + pad, ty), font, scale, color, thick, cv2.LINE_AA)
