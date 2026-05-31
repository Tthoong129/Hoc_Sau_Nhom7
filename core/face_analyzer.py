import cv2
import math
import numpy as np
from ultralytics import YOLO

class PoseAnalyzer:
    def __init__(self):
        print("Khoi tao YOLO Pose...")
        try:
            self.model = YOLO('yolov8s-pose.pt')
            self.available = True
        except Exception as e:
            print(f"Loi load YOLO Pose: {e}")
            self.model = None
            self.available = False

    def analyze(self, rgb_frame, w, h):
        if not self.available:
            return []

        inference_sz = 1280 if max(w, h) >= 1200 else (1024 if max(w, h) >= 1000 else 640)
        results = self.model(rgb_frame, imgsz=inference_sz, verbose=False)
        faces = []

        for r in results:
            if r.keypoints is None or len(r.keypoints) == 0:
                continue

            for i, kpts in enumerate(r.keypoints.xy):
                if len(kpts) < 17:
                    continue

                points = kpts.cpu().numpy()
                confs = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)

                nose          = points[0]
                left_eye      = points[1]
                right_eye     = points[2]
                left_ear      = points[3]
                right_ear     = points[4]
                left_shoulder = points[5]
                right_shoulder= points[6]

                box = r.boxes[i]
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                body_h = by2 - by1

                head_pts = [points[j] for j in range(5) if confs[j] > 0.35]
                if len(head_pts) >= 2:
                    xs = [pt[0] for pt in head_pts]
                    ys = [pt[1] for pt in head_pts]
                    h_xmin, h_xmax = min(xs), max(xs)
                    h_ymin, h_ymax = min(ys), max(ys)
                    head_w = h_xmax - h_xmin
                    head_h = h_ymax - h_ymin
                    pad_w = max(12.0, head_w * 0.3)
                    pad_h = max(15.0, head_h * 0.4)

                    x1 = int(max(0, h_xmin - pad_w))
                    y1 = int(max(0, h_ymin - pad_h * 1.5))
                    x2 = int(min(w, h_xmax + pad_w))
                    y2 = int(min(h, h_ymax + pad_h))
                else:
                    x1 = int(bx1)
                    y1 = int(by1)
                    x2 = int(bx2)
                    y2 = int(by1 + body_h * 0.22)

                l_sh_ok = confs[5] > 0.3
                r_sh_ok = confs[6] > 0.3

                if l_sh_ok and r_sh_ok:
                    cx = (left_shoulder[0] + right_shoulder[0]) / 2.0
                    cy = (left_shoulder[1] + right_shoulder[1]) / 2.0
                    fw = math.hypot(
                        right_shoulder[0] - left_shoulder[0],
                        right_shoulder[1] - left_shoulder[1]
                    )
                else:
                    nose_ok = confs[0] > 0.3
                    if nose_ok:
                        cx = nose[0]
                        cy = nose[1]
                    else:
                        cx = (bx1 + bx2) / 2.0
                        cy = (by1 + by2) / 2.0
                    fw = (bx2 - bx1) * 0.4

                yaw_ratio      = 1.0
                roll_angle     = 0.0
                yaw_dir        = "Thang"
                yaw_str        = "Thang"
                pitch_down     = False
                body_twist_dir = 0
                body_twist     = 0.0
                arm_reach_side = 0
                arm_reach_dist = 0.0
                lean_forward   = False

                # 1. Yaw (quay dau)
                d_l = max(abs(nose[0] - left_eye[0]),  1.0)
                d_r = max(abs(nose[0] - right_eye[0]), 1.0)
                yaw_ratio = max(d_l, d_r) / min(d_l, d_r)
                if d_l > d_r * 1.5:
                    yaw_dir = "Quay PHAI"
                    yaw_str = "Ngo PHAI"
                elif d_r > d_l * 1.5:
                    yaw_dir = "Quay TRAI"
                    yaw_str = "Ngo TRAI"

                # 2. Roll (nghieng dau)
                dx_eye = right_eye[0] - left_eye[0]
                dy_eye = right_eye[1] - left_eye[1]
                if dx_eye != 0:
                    roll_angle = math.degrees(math.atan2(dy_eye, dx_eye))

                shoulder_dist = 0.0

                if l_sh_ok and r_sh_ok:
                    # 3. Pitch Down & Lean Forward (cui dau & chom nguoi)
                    avg_sh_y   = (left_shoulder[1] + right_shoulder[1]) / 2.0
                    sh_mid_x   = (left_shoulder[0] + right_shoulder[0]) / 2.0
                    head_pitch = avg_sh_y - nose[1]
                    shoulder_dist = math.hypot(
                        right_shoulder[0] - left_shoulder[0],
                        right_shoulder[1] - left_shoulder[1]
                    )

                    if shoulder_dist > 0:
                        nose_lateral_offset = abs(nose[0] - sh_mid_x) / max(shoulder_dist, 1.0)
                        head_pitch_ratio    = head_pitch / max(shoulder_dist, 1.0)
                        if (head_pitch > 0 and head_pitch_ratio < 0.55
                                and nose_lateral_offset < 0.70) or head_pitch <= 0:
                            pitch_down = True

                        # Lean Forward: mũi rất gần mức vai -> cúi sâu thật sự
                        if pitch_down and head_pitch_ratio < 0.28:
                            lean_forward = True

                    # 4. Body Twist (xoay nguoi)
                    hip_l   = points[11] if len(points) > 11 else [0, 0]
                    hip_r   = points[12] if len(points) > 12 else [0, 0]
                    hip_dx  = hip_r[0] - hip_l[0]
                    hip_dy  = hip_r[1] - hip_l[1]
                    hip_angle = 0.0
                    if abs(hip_dx) > 5:
                        hip_angle = math.degrees(math.atan2(hip_dy, hip_dx))

                    body_twist = roll_angle - hip_angle
                    if body_twist > 12.0:
                        body_twist_dir = 1
                    elif body_twist < -12.0:
                        body_twist_dir = -1

                # Stable scale: dùng khoảng cách mũi-tai thay vì vai (không bị sai khi cúi)
                stable_scale = fw
                head_size_l = (
                    math.hypot(nose[0] - left_ear[0],  nose[1] - left_ear[1])
                    if (confs[0] > 0.3 and confs[3] > 0.3) else 0
                )
                head_size_r = (
                    math.hypot(nose[0] - right_ear[0], nose[1] - right_ear[1])
                    if (confs[0] > 0.3 and confs[4] > 0.3) else 0
                )
                head_size = max(head_size_l, head_size_r)
                if head_size > 5:
                    stable_scale = max(head_size * 2.8, fw)
                else:
                    stable_scale = max((bx2 - bx1) * 0.35, fw)

                # 5. Arm Reach (voi tay)
                if len(points) > 10 and l_sh_ok and r_sh_ok:
                    img_left_x  = min(left_shoulder[0], right_shoulder[0])
                    img_right_x = max(left_shoulder[0], right_shoulder[0])
                    left_wrist, right_wrist = points[9], points[10]
                    best_right, best_left = 0.0, 0.0

                    for wrist in [left_wrist, right_wrist]:
                        if wrist[0] > 0 and wrist[1] > 0:
                            overshoot_r = (wrist[0] - img_right_x) / max(stable_scale, 1.0)
                            overshoot_l = (img_left_x - wrist[0])  / max(stable_scale, 1.0)
                            best_right  = max(best_right, overshoot_r)
                            best_left   = max(best_left,  overshoot_l)

                    # Khi pitch_down (cúi viết), ngưỡng phải rất lớn (0.70)
                    reach_thresh = 0.70 if pitch_down else 0.10
                    if best_right > reach_thresh and best_right > best_left:
                        arm_reach_side = 1
                        arm_reach_dist = best_right
                    elif best_left > reach_thresh and best_left > best_right:
                        arm_reach_side = -1
                        arm_reach_dist = best_left

                # Wrist near face (dien thoai sat mat)
                wrist_near_face = False
                if len(points) > 10 and l_sh_ok and r_sh_ok:
                    avg_sh_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
                    for wrist in [points[9], points[10]]:
                        if wrist[0] > 0 and wrist[1] > 0:
                            wrist_above_sh = wrist[1] < avg_sh_y
                            wrist_x_near   = abs(wrist[0] - nose[0]) < shoulder_dist * 1.0
                            wrist_y_near   = abs(wrist[1] - nose[1]) < shoulder_dist * 1.2
                            if wrist_above_sh and wrist_x_near and wrist_y_near:
                                wrist_near_face = True
                                break

                face_data = {
                    "centroid":       (cx, cy),
                    "face_width":     fw,
                    "yaw_str":        yaw_str,
                    "yaw_ratio":      yaw_ratio,
                    "roll_angle":     roll_angle,
                    "yaw_dir":        yaw_dir,
                    "pitch_down":     pitch_down,
                    "body_twist_dir": body_twist_dir,
                    "body_twist":     body_twist,
                    "arm_reach_side": arm_reach_side,
                    "arm_reach_dist": arm_reach_dist,
                    "lean_forward":   lean_forward,
                    "stable_scale":   stable_scale,
                    "wrist_near_face":wrist_near_face,
                    "bbox":           (max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                    "landmarks":      points,
                }
                faces.append(face_data)

        # NMS: loc trung lap
        unique_faces = []
        for f in faces:
            fcx, fcy = f["centroid"]
            ffw = f["face_width"]
            is_dup = False
            for uf in unique_faces:
                ufcx, ufcy = uf["centroid"]
                uffw = uf["face_width"]
                dist = math.hypot(fcx - ufcx, fcy - ufcy)
                if dist < 0.4 * max(ffw, uffw):
                    is_dup = True
                    break
            if not is_dup:
                unique_faces.append(f)
        return unique_faces

    def draw_face_contours(self, frame, landmarks):
        pass

    def release(self):
        self.model = None
        self.available = False
