MODEL_PATH = 'models/best.pt'
YOLO_CLASSES = ['CheatingPaper', 'earphone', 'HandGestures', 'normal', 'Phone', 'Rotation', 'Sleeping', 'SmartWatch']
CHEATING_CLASSES = frozenset({'Phone', 'CheatingPaper', 'earphone', 'SmartWatch', 'Sleeping', 'HandGestures', 'Rotation'})

CLASS_SEVERITY = {
    'CheatingPaper': 'HIGH',
    'earphone': 'HIGH',
    'HandGestures': 'MEDIUM',
    'Phone': 'HIGH',
    'Rotation': 'MEDIUM',
    'Sleeping': 'LOW',
    'SmartWatch': 'HIGH',
    'normal': 'OK'
}

CLASS_LABELS_VI = {
    'CheatingPaper': 'GIẤY PHAO',
    'earphone': 'TAI NGHE',
    'HandGestures': 'RA DẤU TAY',
    'Phone': 'ĐIỆN THOẠI',
    'Rotation': 'QUAY ĐẦU',
    'Sleeping': 'NGỦ GẬT',
    'SmartWatch': 'ĐỒNG HỒ THÔNG MINH',
    'normal': 'BÌNH THƯỜNG'
}

CLASS_COLORS = {
    'CheatingPaper': (0, 140, 255),
    'earphone': (0, 0, 220),
    'HandGestures': (255, 165, 0),
    'Phone': (0, 0, 255),
    'Rotation': (0, 200, 255),
    'Sleeping': (200, 100, 200),
    'SmartWatch': (0, 50, 200),
    'normal': (40, 200, 40)
}

YAW_THRESHOLD = 1.55
YAW_THRESHOLD_LOW = 0.65
HEAD_TILT_THRESH = 35.0
LEAN_SHIFT_RATIO = 0.55
SUSTAINED_SECS = 2.0
YOLO_INFER_SIZE = 640
YOLO_CONF_DEFAULT = 0.4
YOLO_IOU_THRESH = 0.45
VIDEO_FRAME_SKIP = 3
TARGET_FPS = 30
MAX_FACES = 8
FUSION_BOOST = True
MIN_YOLO_CONF = 0.3
SCREENSHOT_DIR = 'violations'

SEV_COLOR = {
    'HIGH': (0, 0, 255),
    'MEDIUM': (0, 200, 255),
    'LOW': (30, 130, 30),
    'OK': (40, 200, 40)
}
