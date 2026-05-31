import cv2
import numpy as np
import time
import math
from .config import (
    MODEL_PATH, LEAN_SHIFT_RATIO, HEAD_TILT_THRESH,
    SUSTAINED_SECS, FUSION_BOOST, CLASS_SEVERITY,
    VIDEO_FRAME_SKIP,
)
from .yolo_detector import YOLODetector, PhoneAuxDetector
from .face_analyzer import PoseAnalyzer
from .tracker import SeatTracker, SustainedTimer, ViolationVoter, SignalVelocityTracker, YOLOClassVoter
from .optical_flow import MotionTracker
from .drawing import (
    draw_yolo_box, draw_face_box, draw_head_bars,
    draw_status_overlay, draw_yolo_summary, draw_pose_debug,
)

# Engine chinh phat hien hanh vi gian lan cua hoc sinh
class CheatingDetectorEngine:
    def __init__(self, yolo_model_path: str = None):
        print("Khoi dong AI Engine v7.0")
        model_path = yolo_model_path or MODEL_PATH
        self.yolo = YOLODetector(model_path)
        self.face = PoseAnalyzer()
        self.seat_tracker = SeatTracker()
        self.timer = SustainedTimer(SUSTAINED_SECS)
        self.voter = ViolationVoter(window=12, vote_thresh=0.60, min_frames=4)
        self.motion = MotionTracker(history_len=5)
        self.velocity = SignalVelocityTracker()
        self.yolo_voter = YOLOClassVoter(window=8, presence_thresh=0.45)
        self.phone_aux = PhoneAuxDetector()
        self.debug_mode = False
        self._frame_count = 0
        self._last_yolo_result = []
        self._last_alerts = []
        self._last_annotated = None
        print("AI Engine v7.0 san sang!")
        yolo_status = "OK" if self.yolo.available else "FAIL"
        print(f"YOLO: {yolo_status} | Pose: OK | OpticalFlow: OK | Velocity: OK\n")

    # Xu ly tung khung hinh tu camera hoac video de trich xuat vi pham
    def process_frame(self, frame: np.ndarray, conf_override: float = None,
                      skip_mode: bool = False) -> tuple[np.ndarray, list, list]:
        self._frame_count += 1
        h, w = frame.shape[:2]
        annotated = frame
        alerts = []
        raw_detections = []
        
        # Doc keypoints tu model Pose theo chu ky skip frame
        run_face = True
        if skip_mode and VIDEO_FRAME_SKIP > 1:
            run_face = (self._frame_count % VIDEO_FRAME_SKIP != 0)
        if run_face:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face.analyze(rgb, w, h)
            self._last_faces = faces
        else:
            faces = getattr(self, '_last_faces', [])

        # Chay model YOLO phat hien do vat (Dien thoai, Tai lieu, Nguoi quay bai)
        run_yolo = True
        if skip_mode and VIDEO_FRAME_SKIP > 1:
            run_yolo = (self._frame_count % VIDEO_FRAME_SKIP == 0)

        if run_yolo and self.yolo.available:
            yolo_dets = self.yolo.detect(frame, conf=conf_override)
            if self.phone_aux.available:
                aux_phones = self.phone_aux.detect(frame, conf=0.40)
                filtered_yolo_dets = []
                for d in yolo_dets:
                    if d["class_name"] == "Phone":
                        ex1, ey1, ex2, ey2 = d["bbox"]
                        ecx, ecy = (ex1 + ex2) / 2, (ey1 + ey2) / 2

                        # Tinh khoang cach tu dien thoai den co tay hoc sinh
                        near_hand = False
                        for face in faces:
                            points = face["landmarks"]
                            for w_idx in [9, 10]:  # left_wrist, right_wrist
                                if w_idx < len(points):
                                    wx, wy = points[w_idx]
                                    if wx > 0 and wy > 0:
                                        dist = math.hypot(ecx - wx, ecy - wy)
                                        scale = face.get("stable_scale", face.get("face_width", 100))
                                        if dist < scale * 0.9:
                                            near_hand = True
                                            break
                            if near_hand:
                                break

                        if near_hand:
                            # Neu dien thoai gan tay: doi chieu model COCO phu de chong nham Casio
                            if d["confidence"] < 0.96:
                                is_confirmed = False
                                for ap in aux_phones:
                                    ax1, ay1, ax2, ay2 = ap["bbox"]
                                    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                                    if (abs(acx - ecx) < (ax2 - ax1) * 0.8
                                            and abs(acy - ecy) < (ay2 - ay1) * 0.8):
                                        is_confirmed = True
                                        break
                                if not is_confirmed:
                                    continue
                        else:
                            # Neu dien thoai de ban: phai dat nguong confidence tu thanh cuon slider
                            if d["confidence"] < conf_override:
                                continue
                    filtered_yolo_dets.append(d)
                yolo_dets = filtered_yolo_dets

                # Gop cac dien thoai de ban duoc bat boi COCO detector
                for ap in aux_phones:
                    ax1, ay1, ax2, ay2 = ap["bbox"]
                    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                    is_dup = False
                    for d in yolo_dets:
                        if d["class_name"] == "Phone":
                            ex1, ey1, ex2, ey2 = d["bbox"]
                            ecx, ecy = (ex1 + ex2) / 2, (ey1 + ey2) / 2
                            if (abs(acx - ecx) < (ax2 - ax1) * 0.8
                                    and abs(acy - ecy) < (ay2 - ay1) * 0.8):
                                is_dup = True
                                break
                    if not is_dup:
                        user_conf = conf_override if conf_override is not None else 0.55
                        standalone_thresh = min(user_conf, 0.60)
                        if ap["confidence"] >= standalone_thresh:
                            yolo_dets.append(ap)
            self._last_yolo_result = yolo_dets
        else:
            yolo_dets = self._last_yolo_result

        # Phantich ty le hop bbox va bo qua cac vat qua to hoac sai mau sac (Casio)
        detected_classes = {det["class_name"] for det in yolo_dets if det["is_cheating"]}
        for det in yolo_dets:
            x1, y1, x2, y2 = det["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            cls = det["class_name"]
            conf = det["confidence"]
            if cls == "Phone" and bh > 0:
                ar = bw / bh
                if not (0.20 <= ar <= 3.5):
                    self.yolo_voter.vote(cls, False)
                    continue
                if bw * bh > h * w * 0.12:
                    self.yolo_voter.vote(cls, False)
                    continue
                # Loc Casio: kiem tra mau sac/sat man hinh cua bbox
                if not self._is_phone_screen(frame, (x1, y1, x2, y2)):
                    self.yolo_voter.vote(cls, False)
                    continue

            is_temporally_valid = self.yolo_voter.vote(cls, det["is_cheating"], conf)

            draw_yolo_box(annotated, det)
            if det["is_cheating"] and is_temporally_valid:
                raw_detections.append(f"YOLO: {det['label_vi']} ({det['confidence']:.0%})")
                yolo_key = f"yolo_{cls}"
                if self.timer.check(yolo_key, True):
                    elapsed = self.timer.elapsed(yolo_key)
                    alerts.append({
                        "type":     cls,
                        "msg":      f"{det['label_vi']} ({det['confidence']:.0%}) [{elapsed:.1f}s]",
                        "severity": det["severity"],
                        "source":   "YOLO",
                    })
            elif not det["is_cheating"] or not is_temporally_valid:
                self.timer.reset(f"yolo_{cls}")

        # Tinh toan chuyen dong va phan tich cac chi so hinh hoc tu pose hoc sinh
        motion_info = self.motion.update(frame, faces)
        if faces:
            n_faces = len(faces)
            raw_detections.append(f"Nguoi: {n_faces} phat hien")
            matched = self.seat_tracker.match_faces(faces)
            for tid, face_data in matched:
                track = self.seat_tracker.get_track(tid)
                if track is None:
                    continue
                self.face.draw_face_contours(annotated, face_data["landmarks"])
                cx, cy   = face_data["centroid"]
                fw       = face_data["face_width"]
                yaw_str  = face_data["yaw_str"]
                yaw_ratio= face_data["yaw_ratio"]
                roll_angle = face_data["roll_angle"]
                x_diff   = cx - track.anchor_x
                stable_scale = face_data.get("stable_scale", fw)
                x_diff_ratio = abs(x_diff) / max(stable_scale, 1e-6)
                yaw_dir  = face_data.get("yaw_dir", "Thang")
                pitch_down = face_data.get("pitch_down", False)
                dir_yaw  = 1 if yaw_dir == "Quay PHAI" else (-1 if yaw_dir == "Quay TRAI" else 0)
                if pitch_down:
                    dir_yaw = 0
                anchor_yaw = max(track.anchor_yaw, 1.0)
                relative_yaw_ratio = (
                    yaw_ratio / anchor_yaw if yaw_ratio > anchor_yaw
                    else anchor_yaw / max(yaw_ratio, 1.0)
                )
                if pitch_down:
                    relative_yaw_ratio = 1.0
                dir_roll  = 0 if pitch_down else (1 if roll_angle > 10.0 else (-1 if roll_angle < -10.0 else 0))
                dir_shift = 1 if x_diff < -stable_scale*0.1 else (-1 if x_diff > stable_scale*0.1 else 0)
                body_twist_dir = face_data.get("body_twist_dir", 0)
                body_twist     = abs(face_data.get("body_twist", 0.0))
                arm_reach_side = face_data.get("arm_reach_side", 0)
                arm_reach_dist = face_data.get("arm_reach_dist", 0.0)
                lean_forward   = face_data.get("lean_forward", False)

                minfo      = motion_info.get(tid, self.motion.get(tid))
                of_sudden  = minfo["sudden"]
                of_moving  = minfo["moving"]
                of_mag     = minfo["magnitude"]
                vinfo = self.velocity.update(tid, face_data)
                if not of_moving:
                    vinfo["velocity_bad"] = False
                has_yolo_rotation = any(
                    d["class_name"] == "Rotation"
                    and abs(((d["bbox"][0] + d["bbox"][2]) / 2) - cx) < fw * 1.5
                    for d in yolo_dets
                )
                
                # Setup cac nguong vi pham (lon/nho tuy theo cu ly gan hay xa camera)
                if fw >= 80:
                    yaw_solo_thresh   = 2.0
                    yaw_combo_thresh  = 1.4
                    min_combo_factors = 2
                elif fw >= 40:
                    yaw_solo_thresh   = 2.4
                    yaw_combo_thresh  = 1.6
                    min_combo_factors = 2
                else:
                    yaw_solo_thresh   = 2.8
                    yaw_combo_thresh  = 1.8
                    min_combo_factors = 3
                if anchor_yaw > 2.0:
                    side_cam_factor  = min(anchor_yaw / 2.0, 2.0)
                    yaw_solo_thresh  = yaw_solo_thresh  * side_cam_factor
                    yaw_combo_thresh = yaw_combo_thresh * side_cam_factor
                if has_yolo_rotation:
                    yaw_solo_thresh  = yaw_solo_thresh  * 0.75
                    yaw_combo_thresh = yaw_combo_thresh * 0.75
                roll_thresh_solo  = 45.0 if pitch_down else 30.0
                shift_thresh_solo = 0.80 if pitch_down else 0.70
                severe_yaw   = relative_yaw_ratio > yaw_solo_thresh
                severe_roll  = abs(roll_angle)  > roll_thresh_solo
                severe_shift = x_diff_ratio     > shift_thresh_solo
                lean_bonus        = 1 if lean_forward else 0
                roll_combo_thresh = 22.0
                
                # Gom cac hanh vi vi pham theo huong trai/phai
                arm_reach_combo = arm_reach_side if not pitch_down else 0
                right_factors = (
                    (1 if dir_yaw        ==  1 and relative_yaw_ratio > yaw_combo_thresh else 0) +
                    (1 if dir_roll       ==  1 and abs(roll_angle)     > roll_combo_thresh else 0) +
                    (1 if dir_shift      ==  1 and x_diff_ratio        > 0.35 else 0) +
                    (1 if body_twist_dir ==  1 and body_twist          > 15.0 else 0) +
                    (1 if arm_reach_combo ==  1 and arm_reach_dist     > 0.25 else 0) +
                    lean_bonus
                )
                left_factors = (
                    (1 if dir_yaw        == -1 and relative_yaw_ratio > yaw_combo_thresh else 0) +
                    (1 if dir_roll       == -1 and abs(roll_angle)     > roll_combo_thresh else 0) +
                    (1 if dir_shift      == -1 and x_diff_ratio        > 0.35 else 0) +
                    (1 if body_twist_dir == -1 and body_twist          > 15.0 else 0) +
                    (1 if arm_reach_combo == -1 and arm_reach_dist     > 0.25 else 0) +
                    lean_bonus
                )
                yaw_bad  = False
                lean_bad = False
                if severe_yaw:
                    yaw_bad = True
                elif severe_shift:
                    lean_bad = True
                elif right_factors >= min_combo_factors:
                    yaw_bad  = True
                    lean_bad = True
                    yaw_str  = "Ngo Nghieng PHAI"
                elif left_factors >= min_combo_factors:
                    yaw_bad  = True
                    lean_bad = True
                    yaw_str  = "Ngo Nghieng TRAI"
                is_currently_timing_tilt = f"tilt_{tid}" in self.timer.timers
                significant_arm_reach = arm_reach_side != 0 and arm_reach_dist > 0.40 and not pitch_down
                if significant_arm_reach:
                    lean_bad = True
                is_currently_timing_tilt = f"tilt_{tid}" in self.timer.timers
                if not of_sudden and not is_currently_timing_tilt:
                    if not severe_shift and not lean_forward and not significant_arm_reach:
                        lean_bad = False
                
                # Reset nghi ngo nhao nguoi khi dang cui viet bai
                if pitch_down and not severe_shift and not significant_arm_reach:
                    lean_bad = False
                    self.timer.reset(f"tilt_{tid}")
                if fw < 15:
                    yaw_bad = False
                    if not significant_arm_reach:
                        lean_bad = False
                
                # Bo qua rung dong nho neu camera bi nghieng tu truoc
                if anchor_yaw > 1.8:
                    if relative_yaw_ratio < 2.5:
                        yaw_bad = False
                    if fw < 50:
                        significant_arm_reach = False
                        if not severe_shift and not lean_forward:
                            lean_bad = False
                yaw_bad  = self.voter.vote(f"yaw_{tid}",  yaw_bad)
                lean_bad = self.voter.vote(f"lean_{tid}", lean_bad)
                if self.debug_mode:
                    draw_pose_debug(
                        annotated, face_data,
                        right_factors=right_factors,
                        left_factors=left_factors,
                        yaw_bad=yaw_bad,
                        lean_bad=lean_bad,
                        of_mag=of_mag,
                        of_sudden=of_sudden,
                    )
                track.update((cx, cy), fw, yaw_ratio=yaw_ratio, is_violating=lean_bad)
                
                # Check thoi gian duy tri vi pham Quay dau
                if self.timer.check(f"yaw_{tid}", yaw_bad):
                    elapsed = self.timer.elapsed(f"yaw_{tid}")
                    alerts.append({
                        "type":     "HEAD_TURN",
                        "msg":      f"QUAY DAU: {yaw_str} ({elapsed:.1f}s)",
                        "severity": "MEDIUM",
                        "source":   "Pose" + (" + YOLO" if has_yolo_rotation else ""),
                    })
                elif not yaw_bad:
                    self.timer.reset(f"yaw_{tid}")
                
                # Check thoi gian duy tri vi pham Chom nguoi / Voi tay
                if self.timer.check(f"tilt_{tid}", lean_bad):
                    elapsed = self.timer.elapsed(f"tilt_{tid}")
                    is_severe_lean = lean_forward or (x_diff_ratio > 0.70) or arm_reach_side != 0
                    sev = "HIGH" if is_severe_lean else "MEDIUM"
                    if is_severe_lean:
                        msg = "CHOM NGUOI/VOI TAY"
                    else:
                        msg = "NHOAI NGUOI sang TRAI" if x_diff < 0 else "NHOAI NGUOI sang PHAI"
                    alerts.append({
                        "type":     "BODY_LEAN",
                        "msg":      f"{msg} ({elapsed:.1f}s)",
                        "severity": sev,
                        "source":   "Pose",
                    })
                elif not lean_bad:
                    self.timer.reset(f"tilt_{tid}")
                
                # Phat hien dao mat lien tuc (liec bai)
                if fw >= 60 and vinfo["oscillating"]:
                    if not any(a["type"] == "OSCILLATION" for a in alerts):
                        alerts.append({
                            "type":     "OSCILLATION",
                            "msg":      f"LIEC DI LIEC LAI (ID:{tid})",
                            "severity": "MEDIUM",
                            "source":   "Velocity",
                        })
                
                # Phat hien dien thoai sat mat bang tay
                wrist_near_face  = face_data.get("wrist_near_face", False)
                wrist_phone_valid = self.voter.vote(f"wrist_{tid}", wrist_near_face)
                has_nearby_phone = any(
                    d["class_name"] == "Phone"
                    and abs(((d["bbox"][0] + d["bbox"][2]) / 2) - cx) < fw * 2.0
                    for d in yolo_dets
                )
                if self.timer.check(f"wrist_{tid}", wrist_phone_valid and has_nearby_phone):
                    elapsed = self.timer.elapsed(f"wrist_{tid}")
                    if not any(a["type"] == "Phone" for a in alerts):
                        alerts.append({
                            "type":     "Phone",
                            "msg":      f"DUNG DIEN THOAI (Pose+YOLO) ({elapsed:.1f}s)",
                            "severity": "HIGH",
                            "source":   "Pose+YOLO",
                        })
                elif not (wrist_phone_valid and has_nearby_phone):
                    self.timer.reset(f"wrist_{tid}")
                
                # Ve hop bao quanh mat hoc sinh kem canh bao phu hop
                label_parts = []
                if yaw_bad: label_parts.append(yaw_str)
                if lean_bad:
                    if lean_forward:         label_parts.append("Chom nguoi")
                    elif arm_reach_side != 0: label_parts.append("Voi tay")
                    elif x_diff_ratio > LEAN_SHIFT_RATIO: label_parts.append("Nhoai nguoi")
                    else:                    label_parts.append("Sai tu the")
                is_currently_bad = yaw_bad or lean_bad
                is_violating_time = False
                if is_currently_bad:
                    if self.timer.elapsed(f"yaw_{tid}")  >= self.timer.threshold and yaw_bad:
                        is_violating_time = True
                    if self.timer.elapsed(f"tilt_{tid}") >= self.timer.threshold and lean_bad:
                        is_violating_time = True
                if is_violating_time:
                    is_severe = lean_forward or (x_diff_ratio > 0.70) or arm_reach_side != 0
                    draw_face_box(
                        annotated, face_data["bbox"], tid,
                        violation_level="HIGH" if is_severe else "MEDIUM",
                        label_parts=label_parts,
                    )
                elif is_currently_bad:
                    draw_face_box(
                        annotated, face_data["bbox"], tid,
                        violation_level="WARNING",
                        label_parts=[f"Nghi ngo: {p}" for p in label_parts],
                    )
                draw_head_bars(annotated, tid, yaw_ratio, x_diff_ratio)
        draw_yolo_summary(annotated, yolo_dets)
        draw_status_overlay(annotated, alerts)
        return annotated, alerts, raw_detections

    # Kiem tra do bao hoa mau (Saturation) cua vung box de phan biet Dien thoai vs Casio
    def _is_phone_screen(self, frame: np.ndarray, bbox: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(fw, int(x2)), min(fh, int(y2))
        if x2 - x1 < 15 or y2 - y1 < 15:
            return True
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        frac_colored = float(np.mean(sat > 50))
        return frac_colored > 0.15

    # Giai phong tai nguyen
    def release(self):
        self.face.release()
        self.yolo.release()
        self.phone_aux.release()
        self.seat_tracker.reset()
        self.timer.reset_all()
        self.voter.reset_all()
        self.yolo_voter.reset()
        self.motion.reset()
        self.velocity.reset()
