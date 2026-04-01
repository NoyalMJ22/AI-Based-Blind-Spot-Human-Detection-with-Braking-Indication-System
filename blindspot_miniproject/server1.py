# =============================================================================
# EDGE EYE v3.0 — Flask Backend Server
# Team Tiki-Takas | Camera Stream + SQLite DB + REST API
# =============================================================================
# Run this INSTEAD of blindspot_v2.py:
#   pip install flask ultralytics pygame pyserial
#   python server.py
#
# Then open http://localhost:5000 in your browser.
# The dashboard will auto-open. Camera feed streams via MJPEG.
# All detections are saved to edge_eye.db (SQLite).
# =============================================================================

import cv2
import json
import os
import sqlite3
import tempfile
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, send_from_directory, request

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "edge_eye.db"
STATIC_DIR = BASE_DIR  # serve dashboard.html from same folder

# Detection thresholds (can be updated via API)
CONFIG = {
    "alert_distance_cm":    200,
    "critical_distance_cm": 100,
    "confidence_threshold": 0.55,
    "focal_length":         700.0,
    "camera_height_cm":     300,  # High mount typical for buses
    "adult_height_cm":      170,
    "child_height_cm":      120,
    "mirror_camera":        True,
}

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR))

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                model       TEXT,
                total_detections INTEGER DEFAULT 0,
                total_alerts     INTEGER DEFAULT 0,
                total_critical   INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                type        TEXT NOT NULL,
                distance_cm INTEGER NOT NULL,
                confidence  REAL NOT NULL,
                box_height  INTEGER,
                alert_level TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id);
            CREATE INDEX IF NOT EXISTS idx_det_ts      ON detections(timestamp);
        """)
    print("✅ Database ready:", DB_PATH)

init_db()

# ─────────────────────────────────────────────
# YOLO MODEL
# ─────────────────────────────────────────────
model      = None
model_name = "none"

def load_model():
    global model, model_name
    try:
        from ultralytics import YOLO
        for name in ["yolov8n.pt", "yolov10n.pt"]:
            path = BASE_DIR / name
            if path.exists():
                model = YOLO(str(path))
                model_name = name
                print(f"✅ Loaded model: {name}")
                return
        model = YOLO("yolov8s.pt")
        model_name = "yolov8s.pt"
        print("✅ Downloaded + loaded yolov8s.pt")
    except Exception as e:
        print(f"⚠️  YOLO load failed: {e}  — running in mock mode")
        model = None
        model_name = "mock"

threading.Thread(target=load_model, daemon=True).start()

# ─────────────────────────────────────────────
# ARDUINO (optional)
# ─────────────────────────────────────────────
arduino        = None
ARDUINO_STATUS = "Not Connected"
try:
    import serial
    arduino = serial.Serial('COM5', 9600)
    time.sleep(2)
    ARDUINO_STATUS = "Connected (COM5)"
    print("✅ Arduino Connected")
except Exception:
    print("⚠️  Arduino not connected — software mode")

# ─────────────────────────────────────────────
# AUDIO (optional)
# ─────────────────────────────────────────────
alert_sound  = None
sound_playing = False
try:
    import pygame
    pygame.mixer.init()
    snd = BASE_DIR / "alert_final.wav"
    if snd.exists():
        alert_sound = pygame.mixer.Sound(str(snd))
        print("✅ Alert sound loaded")
except Exception:
    pass

# ─────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────
def classify_person(x1, y1, x2, y2):
    bh = y2 - y1
    bw = max(x2 - x1, 1)
    ratio = bh / bw
    if ratio > 2.2:
        return "ADULT", CONFIG["adult_height_cm"]
    elif ratio > 1.5:
        return "CHILD", CONFIG["child_height_cm"]
    return "HUMAN", CONFIG["adult_height_cm"]

def estimate_distance(box_h_px, real_h_cm, box_w_px=None):
    if box_h_px <= 0:
        return 99999
        
    # Correct for truncated bodies (e.g. only showing upper body when extremely close to the bus)
    # Average human aspect ratio (width:height) is approx 1:2.2
    if box_w_px is not None:
        expected_h = box_w_px * 2.2
        if expected_h > box_h_px:
            box_h_px = expected_h  # Correct for truncation
    # Direct camera-to-person distance
    direct_distance = (real_h_cm * CONFIG["focal_length"]) / box_h_px
    # Ground distance using Pythagorean theorem for bus camera altitude
    cam_h = CONFIG.get("camera_height_cm", 300)
    
    if direct_distance > cam_h:
        ground_dist = (direct_distance**2 - cam_h**2)**0.5
    else:
        # if direct_distance is shorter than the camera mount height,
        # it's physically impossible for the Pythagorean theorem to apply.
        # Fallback to direct_distance directly to prevent artificial shrinking during localized testing.
        ground_dist = direct_distance
    
    return int(max(ground_dist, 1))

adaptive_focal_history = deque(maxlen=30)

# ─────────────────────────────────────────────
# SESSION STATE  (shared across threads)
# ─────────────────────────────────────────────
session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
session_lock = threading.Lock()
session_stats = {
    "total_detections": 0,
    "total_alerts":     0,
    "total_critical":   0,
}

# Insert session row
with get_db() as conn:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at, model) VALUES(?,?,?)",
        (session_id, datetime.now().isoformat(), model_name)
    )

def log_detection(det_type, distance, confidence, box_h):
    alert_level = (
        "CRITICAL" if distance <= CONFIG["critical_distance_cm"] else
        "ALERT"    if distance <= CONFIG["alert_distance_cm"]    else
        "CLEAR"
    )
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')

    with session_lock:
        session_stats["total_detections"] += 1
        if alert_level in ("ALERT", "CRITICAL"):
            session_stats["total_alerts"] += 1
        if alert_level == "CRITICAL":
            session_stats["total_critical"] += 1

    with get_db() as conn:
        conn.execute(
            """INSERT INTO detections
               (session_id, timestamp, type, distance_cm, confidence, box_height, alert_level)
               VALUES(?,?,?,?,?,?,?)""",
            (session_id, ts, det_type, distance, round(confidence, 4), box_h, alert_level)
        )
        conn.execute(
            """UPDATE sessions SET
               total_detections=?, total_alerts=?, total_critical=?, model=?
               WHERE id=?""",
            (session_stats["total_detections"],
             session_stats["total_alerts"],
             session_stats["total_critical"],
             model_name, session_id)
        )

# ─────────────────────────────────────────────
# LIVE STATUS  (updated every frame)
# ─────────────────────────────────────────────
live_status = {
    "ts": datetime.now().isoformat(),
    "status": "OFFLINE",
    "fps": 0.0,
    "closest_cm": None,
    "focal_length": CONFIG["focal_length"],
    "model": model_name,
    "arduino": ARDUINO_STATUS,
    "session_id": session_id,
    "cam_resolution": "---",
    "detections_this_frame": 0,
    **session_stats
}
live_lock = threading.Lock()

def update_live_status(**kwargs):
    with live_lock:
        live_status.update(kwargs)
        live_status["ts"] = datetime.now().isoformat(sep=' ', timespec='milliseconds')
        live_status["total_detections"] = session_stats["total_detections"]
        live_status["total_alerts"]     = session_stats["total_alerts"]
        live_status["total_critical"]   = session_stats["total_critical"]

# ─────────────────────────────────────────────
# CAMERA + DETECTION THREAD
# ─────────────────────────────────────────────
frame_lock      = threading.Lock()
latest_frame    = None   # encoded JPEG bytes
camera_running  = False

def put_text_bg(frame, text, pos, scale=0.60, color=(255,255,255), bg=(0,0,0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = pos
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def camera_loop():
    global latest_frame, camera_running, sound_playing, model_name

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    update_live_status(cam_resolution=f"{cam_w}×{cam_h}")

    fps_buf   = deque(maxlen=30)
    prev_time = time.time()
    camera_running = True

    print(f"📷 Camera started: {cam_w}×{cam_h}")

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if CONFIG["mirror_camera"]:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-5))
        prev_time = now
        fps = sum(fps_buf) / len(fps_buf)

        human_in_range   = False
        human_critical   = False
        closest_dist     = 99999
        dets_this_frame  = 0

        if model is not None:
            results = model(frame, verbose=False,
                            conf=CONFIG["confidence_threshold"], classes=[0])
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIG["confidence_threshold"]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_h = y2 - y1
                    box_w = x2 - x1
                    person_type, real_h = classify_person(x1, y1, x2, y2)
                    distance_cm = estimate_distance(box_h, real_h, box_w)

                    closest_dist = min(closest_dist, distance_cm)
                    dets_this_frame += 1

                    if distance_cm <= CONFIG["critical_distance_cm"]:
                        label = f"⚠ {person_type} {distance_cm}cm CRITICAL"
                        color = (0, 0, 255) # Pure Red (BGR)
                        human_in_range = True
                        human_critical = True
                    elif distance_cm <= CONFIG["alert_distance_cm"]:
                        label = f"⚠ {person_type} {distance_cm}cm WARNING"
                        color = (0, 165, 255) # Standard Orange (BGR)
                        human_in_range = True
                    else:
                        label = f"✓ {person_type} {distance_cm}cm SAFE"
                        color = (0, 255, 0) # Pure Green (BGR)

                    thickness = 3 if distance_cm <= CONFIG["alert_distance_cm"] else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    put_text_bg(frame, label, (x1, max(y1-12, 20)), scale=0.6, color=(255,255,255))
                    put_text_bg(frame, f"{conf*100:.0f}%", (x2-45, max(y1-12, 20)),
                                scale=0.5, color=(200,200,200), bg=(40,40,40))

                    log_detection(person_type, distance_cm, conf, box_h)

        # HUD
        status_color = (0,0,200) if human_critical else (0,80,200) if human_in_range else (20,120,20)
        status_text  = ("⚠ CRITICAL — HUMAN VERY CLOSE" if human_critical else
                        "⚠ ALERT — HUMAN IN BLIND SPOT" if human_in_range else
                        "✓  CLEAR")
        cv2.rectangle(frame, (0, 0), (w, 42), status_color, -1)
        cv2.putText(frame, status_text, (10, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

        info_lines = [
            f"FPS: {fps:.1f}",
            f"Model: {model_name}",
            f"Closest: {closest_dist if closest_dist < 9999 else '---'} cm",
            f"Alert: {CONFIG['alert_distance_cm']} cm",
            f"Detections: {session_stats['total_detections']}",
        ]
        for i, line in enumerate(info_lines):
            put_text_bg(frame, line, (w-270, 65 + i*28),
                        scale=0.52, color=(220,220,220), bg=(10,10,10))

        put_text_bg(frame, "EDGE EYE v3.0  |  edge_eye.db",
                    (10, h-12), scale=0.45, color=(140,140,140), bg=(0,0,0))

        # Update live status
        update_live_status(
            fps=round(fps, 1),
            status="CRITICAL" if human_critical else "ALERT" if human_in_range else "CLEAR",
            closest_cm=closest_dist if closest_dist < 9999 else None,
            focal_length=round(CONFIG["focal_length"], 1),
            camera_height_cm=CONFIG.get("camera_height_cm", 300),
            model=model_name,
            alert_distance_cm=CONFIG["alert_distance_cm"],
            critical_distance_cm=CONFIG["critical_distance_cm"],
            confidence_threshold=CONFIG["confidence_threshold"],
            detections_this_frame=dets_this_frame,
        )

        # Arduino
        if arduino:
            try:
                arduino.write(b'2' if human_critical else b'1' if human_in_range else b'0')
            except Exception:
                pass

        # Audio
        if alert_sound:
            if human_in_range and not sound_playing:
                alert_sound.play(-1)
                sound_playing = True
            elif not human_in_range and sound_playing:
                alert_sound.stop()
                sound_playing = False

        # Encode JPEG for streaming
        ret2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            with frame_lock:
                latest_frame = buf.tobytes()

    cap.release()
    update_live_status(status="OFFLINE", fps=0.0)
    print("📷 Camera stopped")

# Start camera thread on import
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()

# ─────────────────────────────────────────────
# MJPEG GENERATOR
# ─────────────────────────────────────────────
def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 fps cap

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'edge_eye_dashboard_v3.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with live_lock:
        data = dict(live_status)
    return jsonify(data)

@app.route('/api/detections')
def api_detections():
    limit    = int(request.args.get('limit', 200))
    sess     = request.args.get('session', session_id)
    offset   = int(request.args.get('offset', 0))
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM detections WHERE session_id=?
               ORDER BY id DESC LIMIT ? OFFSET ?""",
            (sess, limit, offset)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/sessions')
def api_sessions():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 50"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/sessions/<sid>')
def api_session_detail(sid):
    with get_db() as conn:
        sess = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not sess:
            return jsonify({"error": "not found"}), 404
        dets = conn.execute(
            "SELECT * FROM detections WHERE session_id=? ORDER BY id",
            (sid,)
        ).fetchall()
    return jsonify({"session": dict(sess), "detections": [dict(r) for r in dets]})

@app.route('/api/stats/hourly')
def api_hourly():
    """Detections grouped by hour for the current session"""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT strftime('%H:%M', timestamp) as t,
                      COUNT(*) as total,
                      SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END) as critical,
                      SUM(CASE WHEN alert_level='ALERT' THEN 1 ELSE 0 END) as alert,
                      AVG(distance_cm) as avg_dist,
                      AVG(confidence)  as avg_conf
               FROM detections WHERE session_id=?
               GROUP BY strftime('%H:%M', timestamp)
               ORDER BY t""",
            (session_id,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/stats/summary')
def api_summary():
    sess = request.args.get('session', session_id)
    with get_db() as conn:
        row = conn.execute(
            """SELECT
               COUNT(*)                                    as total,
               SUM(CASE WHEN alert_level='ALERT' OR alert_level='CRITICAL' THEN 1 ELSE 0 END) as alerts,
               SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END) as critical,
               AVG(confidence)*100                         as avg_conf,
               MAX(confidence)*100                         as max_conf,
               MIN(confidence)*100                         as min_conf,
               AVG(distance_cm)                            as avg_dist,
               MIN(distance_cm)                            as min_dist,
               MAX(distance_cm)                            as max_dist
               FROM detections WHERE session_id=?""",
            (sess,)
        ).fetchone()
    return jsonify(dict(row) if row else {})

@app.route('/api/config', methods=['GET'])
def api_config_get():
    return jsonify(CONFIG)

@app.route('/api/config', methods=['POST'])
def api_config_set():
    data = request.get_json(force=True)
    for k, v in data.items():
        if k in CONFIG:
            CONFIG[k] = type(CONFIG[k])(v)
    return jsonify({"ok": True, "config": CONFIG})

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    """Trigger focal-length recalibration on next detection frame"""
    CONFIG["_calibrate"] = True
    return jsonify({"ok": True, "message": "Stand 150 cm from camera"})

@app.route('/api/db/export')
def api_db_export():
    return send_from_directory(str(BASE_DIR), 'edge_eye.db',
                               as_attachment=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚍 EDGE EYE v3.0 — Flask Backend")
    print(f"📂 DB: {DB_PATH}")
    print(f"🌐 Dashboard: http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    # Open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
