import cv2
import time
import os
import json
import base64
import shutil
import threading
import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from core.ai_engine import CheatingDetectorEngine

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "proctoring-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

MODEL_PATH = "models/best.pt"
SESSION_DIR = "screenshots/session"
HISTORY_DIR = "screenshots/history"

os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

engine = None
cap = None
running = False
paused = False
seek_target = -1
auto_capture = True
debug_mode = False
conf_threshold = 0.75
playback_speed = 1.0
latest_frame = None
session_log = []
violations_count = 0
last_capture_time = {}
current_run_id = 0

# Khoi tao AI Engine
def init_engine():
    global engine
    print("Loading AI Engine...")
    engine = CheatingDetectorEngine(MODEL_PATH)
    print("AI Engine ready.")

# Don dep thu muc anh chup vi pham trong phien hien tai
def clear_session_folder():
    if os.path.exists(SESSION_DIR):
        shutil.rmtree(SESSION_DIR)
    os.makedirs(SESSION_DIR, exist_ok=True)

# Luu anh chup bang chung vi pham (Session va History)
def save_violation_screenshot(frame, reason, to_history=False):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"{reason}_{ts}.jpg"
    folder = HISTORY_DIR if to_history else SESSION_DIR
    path = os.path.join(folder, fname)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return fname

# Luong doc khung hinh camera/video va chay model AI realtime
def capture_loop(source, video_path=None, run_id=0):
    global cap, running, latest_frame, violations_count, current_run_id, seek_target
    if source == "camera":
        local_cap = cv2.VideoCapture(0)
        local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        is_video = False
        video_fps = 30.0
    else:
        local_cap = cv2.VideoCapture(video_path)
        is_video = True
    if not local_cap.isOpened():
        if run_id == current_run_id:
            socketio.emit("error", {"msg": "Khong mo duoc nguon video!"})
            running = False
            socketio.emit("stopped", {})
        return
    cap = local_cap
    if is_video:
        video_fps = local_cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(local_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        socketio.emit("video_info", {
            "total_frames": total_frames,
            "fps": video_fps,
            "duration": total_frames / video_fps if video_fps > 0 else 0
        })
    target_dt = (1.0 / video_fps) if is_video else 0
    frame_count = 0
    fps_count = 0
    fps_timer = time.time()
    frame_accumulator = 0.0
    last_time = time.time()
    while running and run_id == current_run_id:
        if paused:
            last_time = time.time()
            if seek_target != -1:
                local_cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                seek_target = -1
                ret, frame = local_cap.read()
                if ret:
                    h_orig, w_orig = frame.shape[:2]
                    if max(h_orig, w_orig) > 960:
                        scale_r = 960 / max(h_orig, w_orig)
                        frame = cv2.resize(frame, None, fx=scale_r, fy=scale_r)
                    annotated, alerts, raw = engine.process_frame(
                        frame, conf_override=conf_threshold, skip_mode=is_video
                    )
                    latest_frame = annotated.copy()
                    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    socketio.emit("frame", {"data": b64})
                    pos = int(local_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    socketio.emit("video_pos", {"pos": pos})
            time.sleep(0.05)
            continue
        loop_start = time.time()
        dt = loop_start - last_time
        last_time = loop_start
        if seek_target != -1:
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
            seek_target = -1
            frame_accumulator = 0.0
        if is_video:
            frames_to_advance = dt * video_fps * playback_speed + frame_accumulator
            skip_count = int(frames_to_advance) - 1
            if skip_count < 0:
                skip_count = 0
            frame_accumulator = frames_to_advance - (skip_count + 1)
            if skip_count > 0:
                if skip_count > 60:
                    current_pos = local_cap.get(cv2.CAP_PROP_POS_FRAMES)
                    local_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + skip_count)
                else:
                    for _ in range(skip_count):
                        local_cap.grab()
        ret, frame = local_cap.read()
        if not ret:
            if is_video:
                socketio.emit("video_ended", {})
            break
        frame_count += 1
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = fps_count / (now - fps_timer)
            fps_count = 0
            fps_timer = now
            socketio.emit("fps", {"value": round(fps)})
        h_orig, w_orig = frame.shape[:2]
        if max(h_orig, w_orig) > 960:
            scale_r = 960 / max(h_orig, w_orig)
            frame = cv2.resize(frame, None, fx=scale_r, fy=scale_r)
        annotated, alerts, raw = engine.process_frame(
            frame, conf_override=conf_threshold, skip_mode=is_video
        )
        latest_frame = annotated.copy()
        if alerts:
            for a in alerts:
                sev = a.get("severity", "LOW")
                atype = a["type"]
                cooldown = 3.0
                if time.time() - last_capture_time.get(atype, 0) < cooldown:
                    continue
                last_capture_time[atype] = time.time()
                log_entry = {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": atype,
                    "msg": a["msg"],
                    "severity": sev,
                    "source": a.get("source", ""),
                    "screenshot": None,
                }
                violations_count += 1
                if auto_capture and sev in ("HIGH", "MEDIUM"):
                    fname = save_violation_screenshot(annotated, atype)
                    save_violation_screenshot(annotated, atype, to_history=True)
                    log_entry["screenshot"] = fname
                session_log.append(log_entry)
                socketio.emit("alert", log_entry)
                
                # Luu nhat ky vi pham vao file log
                log_path = os.path.join(HISTORY_DIR, "violation_log.txt")
                try:
                    with open(log_path, "a", encoding="utf-8") as lf:
                        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        lf.write(f"[{time_str}] [{sev}] Type: {atype} | Msg: {a['msg']} | Screen: {log_entry['screenshot'] or 'None'}\n")
                except Exception as e:
                    print(f"Loi ghi violation_log: {e}")
        if is_video and frame_count % 5 == 0:
            pos = int(local_cap.get(cv2.CAP_PROP_POS_FRAMES))
            socketio.emit("video_pos", {"pos": pos})
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buffer).decode("utf-8")
        socketio.emit("frame", {"data": b64})
        if is_video and target_dt > 0:
            if playback_speed < 1.0:
                effective_dt = target_dt / playback_speed
            else:
                effective_dt = target_dt
            elapsed = time.time() - now
            wait = effective_dt - elapsed
            if wait > 0.001:
                time.sleep(wait)
    local_cap.release()
    if run_id == current_run_id:
        cap = None
        running = False
        socketio.emit("stopped", {})

# Router tra ve giao dien chinh (index.html)
@app.route("/")
def index():
    return render_template("index.html")

# API truy cap anh chup vi pham phien hien tai
@app.route("/screenshots/session/<path:filename>")
def serve_session(filename):
    return send_from_directory(SESSION_DIR, filename)

# API truy cap kho anh lich su tat ca cac phien
@app.route("/screenshots/history/<path:filename>")
def serve_history(filename):
    return send_from_directory(HISTORY_DIR, filename)

# API danh sach anh chup vi pham
@app.route("/api/screenshots/<folder>")
def list_screenshots(folder):
    d = SESSION_DIR if folder == "session" else HISTORY_DIR
    if not os.path.exists(d):
        return jsonify([])
    files = sorted(os.listdir(d), reverse=True)
    files = [f for f in files if f.endswith((".jpg", ".png"))]
    return jsonify(files)

# API quet danh sach cac file video co trong thu muc nguon
@app.route("/api/videos")
def list_videos():
    extensions = (".mp4", ".avi", ".mkv", ".mov", ".wmv")
    files = [f for f in os.listdir(".") if f.lower().endswith(extensions) and os.path.isfile(f)]
    return jsonify(files)

# API thong ke so luong vi pham (Tong hop, Cao, Trung binh)
@app.route("/api/stats")
def get_stats():
    high = sum(1 for x in session_log if x["severity"] == "HIGH")
    med = sum(1 for x in session_log if x["severity"] == "MEDIUM")
    return jsonify({
        "total": violations_count,
        "high": high,
        "medium": med,
        "running": running,
        "paused": paused,
        "auto_capture": auto_capture,
    })

# API nhan va luu file video tu may tinh nguoi dung upload len
@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    return jsonify({"path": filepath})

# API xuat file bao cao gian lan (json) ra kho luu tru
@app.route("/api/export")
def export_report():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"total": violations_count, "log": session_log}
    path = os.path.join(HISTORY_DIR, f"report_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return jsonify({"path": path, "ok": True})

# SocketIO: Bat dau giam sat (Camera hoac Video File)
@socketio.on("start")
def handle_start(data):
    global running, paused, violations_count, session_log, current_run_id
    current_run_id += 1
    run_id = current_run_id
    running = True
    paused = False
    violations_count = 0
    session_log = []
    last_capture_time.clear()
    clear_session_folder()
    source = data.get("source", "camera")
    video_path = data.get("video_path", None)
    thread = threading.Thread(target=capture_loop, args=(source, video_path, run_id), daemon=True)
    thread.start()
    socketio.emit("started", {"source": source})

# SocketIO: Dung giam sat
@socketio.on("stop")
def handle_stop():
    global running, current_run_id, cap
    running = False
    current_run_id += 1
    cap = None
    socketio.emit("stopped", {})

# SocketIO: Tam dung hoac tiep tuc chay video
@socketio.on("pause")
def handle_pause():
    global paused
    paused = not paused
    socketio.emit("paused", {"paused": paused})

# SocketIO: Cap nhat do tu tin detection tu thanh keo (slider)
@socketio.on("set_conf")
def handle_conf(data):
    global conf_threshold
    conf_threshold = float(data.get("value", 0.75))

# SocketIO: Cap nhat toc do phat video (1x, 2x, etc.)
@socketio.on("set_speed")
def handle_speed(data):
    global playback_speed
    playback_speed = float(data.get("value", 1.0))

# SocketIO: Bat/tat tu dong chup anh man hinh khi co vi pham
@socketio.on("toggle_capture")
def handle_toggle_capture():
    global auto_capture
    auto_capture = not auto_capture
    socketio.emit("capture_status", {"auto_capture": auto_capture})

# SocketIO: Bat/tat giao dien debug hien thi khung xuong pose hoc sinh
@socketio.on("toggle_debug")
def handle_toggle_debug():
    global debug_mode
    if engine:
        engine.debug_mode = not engine.debug_mode
        debug_mode = engine.debug_mode
    socketio.emit("debug_status", {"debug": debug_mode})

# SocketIO: Chup anh bang chung thu cong (khi thay co click nut chup anh)
@socketio.on("manual_shot")
def handle_manual_shot():
    if latest_frame is not None:
        fname = save_violation_screenshot(latest_frame, "manual")
        save_violation_screenshot(latest_frame, "manual", to_history=True)
        socketio.emit("screenshot_taken", {"filename": fname})

# SocketIO: Tua den khung hinh mong muon tren thanh progress
@socketio.on("seek")
def handle_seek(data):
    global seek_target
    seek_target = int(data.get("pos", 0))

if __name__ == "__main__":
    init_engine()
    print("\n  Web UI: http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
