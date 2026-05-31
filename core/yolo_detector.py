from ultralytics import YOLO
from .config import YOLO_CONF_DEFAULT, YOLO_IOU_THRESH, YOLO_INFER_SIZE, CHEATING_CLASSES, CLASS_SEVERITY, CLASS_LABELS_VI

# Bo nhan dien YOLO chinh (Custom Model cho Cac lop gian lan)
class YOLODetector:
    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            self.available = True
        except Exception as e:
            print(f"Loi load YOLO: {e}")
            self.model = None
            self.available = False

    # Nguyen tac do tu tin rieng cho tung lop hanh vi
    SENSITIVITY_OVERRIDE = {
        "Rotation":     0.38,
        "HandGestures": 0.40,
        "Sleeping":     0.40,
        "Phone":        0.50,
    }

    # Phat hien vat the tren khung hinh
    def detect(self, frame, conf=None) -> list:
        if not self.available:
            return []
        c = conf if conf is not None else YOLO_CONF_DEFAULT
        base_conf = min(c, 0.35)
        results = self.model.predict(
            source=frame,
            conf=base_conf,
            iou=YOLO_IOU_THRESH,
            imgsz=YOLO_INFER_SIZE,
            verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[cls_id]
                if class_name in self.SENSITIVITY_OVERRIDE:
                    min_conf = self.SENSITIVITY_OVERRIDE[class_name]
                else:
                    min_conf = c
                if confidence < min_conf:
                    continue
                det = {
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": confidence,
                    "severity": CLASS_SEVERITY.get(class_name, "LOW"),
                    "is_cheating": class_name in CHEATING_CLASSES,
                    "label_vi": CLASS_LABELS_VI.get(class_name, class_name)
                }
                detections.append(det)
        return detections

    def release(self):
        self.model = None
        self.available = False


# Bo nhan dien phu de kiem tra cheo dien thoai qua bo du lieu COCO
class PhoneAuxDetector:
    COCO_PHONE_CLASS_ID = 67

    def __init__(self):
        self.model = None
        self.available = False
        try:
            self.model = YOLO('yolov8s.pt')
            self.available = True
            print("PhoneAuxDetector (COCO yolov8s) loaded OK")
        except Exception as e:
            print(f"PhoneAuxDetector load failed (optional): {e}")

    # Chay dự doan rieng cho lop dien thoai cua COCO
    def detect(self, frame, conf=0.35) -> list:
        if not self.available:
            return []
        results = self.model.predict(
            source=frame,
            conf=conf,
            classes=[self.COCO_PHONE_CLASS_ID],
            iou=0.45,
            imgsz=640,
            verbose=False
        )
        phones = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != self.COCO_PHONE_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = x2 - x1, y2 - y1
                if bh <= 0 or bw <= 0:
                    continue
                confidence = float(box.conf[0])
                phones.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": "Phone",
                    "confidence": confidence,
                    "severity": "HIGH",
                    "is_cheating": True,
                    "label_vi": "DIEN THOAI (COCO)",
                    "source": "aux"
                })
        return phones

    def release(self):
        self.model = None
        self.available = False
