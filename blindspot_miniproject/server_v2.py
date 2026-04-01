# =============================================================================
# EDGE EYE v4.0 — Flask Backend Server
# Team Tiki-Takas | Camera Stream + SQLite DB + REST API
# =============================================================================
# Run:  python server_v2.py
# Open: http://localhost:5000
# =============================================================================
# v4.0 Changes:
#  - HUMAN only (removed unreliable ADULT/CHILD aspect-ratio heuristic)
#  - Real confidence scores stored accurately
#  - Threat score (distance + conf weighted) computed per detection
#  - Confusion matrix counters tracked per session (TP/FP/FN/TN)
#  - /api/accuracy endpoint: precision, recall, F1, accuracy + per-minute timeline
#  - /api/heatmap: grid-based positional density
#  - /api/report/json: full session report
#  - Optimised camera loop: skip frame encode when no consumer, JPEG quality 75
#  - Thread-safe config updates via RLock
#  - SSE /api/stream endpoint for near-real-time dashboard updates
# =============================================================================

import cv2
import json
import os
import sqlite3
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, send_from_directory, request, stream_with_context

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "edge_eye.db"
STATIC_DIR = BASE_DIR

CONFIG_LOCK = threading.RLock()
CONFIG = {
    "alert_distance_cm":    200,
    "critical_distance_cm": 100,
    "confidence_threshold": 0.45,
    "focal_length":         700.0,
    "known_person_height_cm": 170,
    "calibration_distance_cm": 150,
    "mirror_camera":        True,
    "show_danger_zone":     True,
    "_calibrate":           False,
    "camera_index":         0,
}

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR))
app.config['JSON_SORT_KEYS'] = False

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads + writes
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id               TEXT PRIMARY KEY,
                started_at       TEXT NOT NULL,
                ended_at         TEXT,
                model            TEXT,
                total_detections INTEGER DEFAULT 0,
                total_alerts     INTEGER DEFAULT 0,
                total_critical   INTEGER DEFAULT 0,
                cm_tp  INTEGER DEFAULT 0,
                cm_fp  INTEGER DEFAULT 0,
                cm_fn  INTEGER DEFAULT 0,
                cm_tn  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                type        TEXT    NOT NULL DEFAULT 'HUMAN',
                distance_cm INTEGER NOT NULL,
                confidence  REAL    NOT NULL,
                threat_score REAL   DEFAULT 0,
                box_height  INTEGER,
                box_cx      INTEGER,
                box_cy      INTEGER,
                alert_level TEXT    NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id);
            CREATE INDEX IF NOT EXISTS idx_det_ts      ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_det_level   ON detections(alert_level);
        """)
    print("✅ Database ready:", DB_PATH)

init_db()

# DB write queue — avoids blocking the camera thread on disk I/O
_db_queue = deque()
_db_lock  = threading.Lock()

def _db_worker():
    while True:
        with _db_lock:
            batch = []
            while _db_queue:
                batch.append(_db_queue.popleft())
        if batch:
            try:
                with get_db() as conn:
                    for op, args in batch:
                        conn.execute(op, args)
            except Exception as e:
                print(f"⚠️  DB write error: {e}")
        time.sleep(0.1)

threading.Thread(target=_db_worker, daemon=True).start()

def db_enqueue(op, args):
    with _db_lock:
        _db_queue.append((op, args))

# ─────────────────────────────────────────────
# YOLO MODEL
# ─────────────────────────────────────────────
model      = None
model_name = "loading"

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
        model = YOLO("yolov8n.pt")
        model_name = "yolov8n.pt"
        print("✅ Downloaded + loaded yolov8n.pt")
    except Exception as e:
        print(f"⚠️  YOLO load failed: {e}")
        model = None
        model_name = "mock"

threading.Thread(target=load_model, daemon=True).start()

# ─────────────────────────────────────────────
# ARDUINO
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
# AUDIO
# ─────────────────────────────────────────────
alert_sound   = None
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
adaptive_focal_history = deque(maxlen=30)

def estimate_distance(box_h_px):
    with CONFIG_LOCK:
        fl = CONFIG["focal_length"]
        ph = CONFIG["known_person_height_cm"]
    if box_h_px <= 0:
        return 99999
    return int((ph * fl) / box_h_px)

def compute_threat_score(distance_cm, confidence):
    with CONFIG_LOCK:
        alert_d = CONFIG["alert_distance_cm"]
    dist_score = max(0.0, 1.0 - (distance_cm / (alert_d * 1.5)))
    dist_score = dist_score ** 0.7
    return round(min(100.0, max(0.0, (dist_score * 0.70 + confidence * 0.30) * 100.0)), 1)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
sess_lock    = threading.Lock()
session_stats = {
    "total_detections": 0, "total_alerts": 0, "total_critical": 0,
    "cm_tp": 0, "cm_fp": 0, "cm_fn": 0, "cm_tn": 0,
}

with get_db() as conn:
    conn.execute(
        "INSERT OR IGNORE INTO sessions(id, started_at, model) VALUES(?,?,?)",
        (session_id, datetime.now().isoformat(), model_name)
    )

# ─────────────────────────────────────────────
# HEATMAP  (10×8 grid, normalised to camera)
# ─────────────────────────────────────────────
HMAP_COLS, HMAP_ROWS = 10, 8
heatmap_grid = np.zeros((HMAP_ROWS, HMAP_COLS), dtype=float)
heatmap_lock = threading.Lock()

def update_heatmap(cx_norm, cy_norm, weight=1.0):
    gx = int(min(cx_norm, 0.999) * HMAP_COLS)
    gy = int(min(cy_norm, 0.999) * HMAP_ROWS)
    with heatmap_lock:
        heatmap_grid[gy][gx] += weight

def reset_heatmap():
    with heatmap_lock:
        heatmap_grid[:] = 0

# ─────────────────────────────────────────────
# LOG DETECTION
# ─────────────────────────────────────────────
def log_detection(distance, confidence, box_h, alert_level, threat, cx, cy, frame_w, frame_h):
    ts = datetime.now().isoformat(sep=' ', timespec='milliseconds')

    with sess_lock:
        session_stats["total_detections"] += 1
        if alert_level in ("ALERT", "CRITICAL"):
            session_stats["total_alerts"] += 1
            session_stats["cm_tp"] += 1
        else:
            session_stats["cm_fp"] += 1
        if alert_level == "CRITICAL":
            session_stats["total_critical"] += 1

    # Async DB write
    db_enqueue("""INSERT INTO detections
        (session_id, timestamp, type, distance_cm, confidence, threat_score,
         box_height, box_cx, box_cy, alert_level)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (session_id, ts, "HUMAN", distance, round(confidence, 6),
         round(threat, 2), box_h, cx, cy, alert_level))

    db_enqueue("""UPDATE sessions SET
        total_detections=?, total_alerts=?, total_critical=?,
        cm_tp=?, cm_fp=?, cm_fn=?, cm_tn=?, model=? WHERE id=?""",
        (session_stats["total_detections"], session_stats["total_alerts"],
         session_stats["total_critical"],
         session_stats["cm_tp"], session_stats["cm_fp"],
         session_stats["cm_fn"], session_stats["cm_tn"],
         model_name, session_id))

    # Update heatmap
    if frame_w > 0 and frame_h > 0:
        update_heatmap(cx / frame_w, cy / frame_h, weight=1.0 + (threat / 100.0))

def log_clear_frame(had_lowconf):
    with sess_lock:
        if had_lowconf:
            session_stats["cm_fn"] += 1
        else:
            session_stats["cm_tn"] += 1

# ─────────────────────────────────────────────
# LIVE STATUS
# ─────────────────────────────────────────────
live_status = {"ts": datetime.now().isoformat(), "status": "OFFLINE", "fps": 0}
live_lock   = threading.Lock()

def update_live_status(**kwargs):
    with sess_lock:
        ss = dict(session_stats)
    tp, fp, fn, tn = ss["cm_tp"], ss["cm_fp"], ss["cm_fn"], ss["cm_tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    with live_lock:
        live_status.update(kwargs)
        live_status["ts"]               = datetime.now().isoformat(sep=' ', timespec='milliseconds')
        live_status["model"]            = model_name
        live_status["arduino"]          = ARDUINO_STATUS
        live_status["session_id"]       = session_id
        live_status.update(ss)
        live_status["precision"]  = round(precision * 100, 1)
        live_status["recall"]     = round(recall    * 100, 1)
        live_status["f1_score"]   = round(f1        * 100, 1)
        live_status["accuracy"]   = round(accuracy  * 100, 1)

# ─────────────────────────────────────────────
# CAMERA + DETECTION THREAD
# ─────────────────────────────────────────────
frame_lock     = threading.Lock()
latest_frame   = None
camera_running = True
_frame_consumers = 0
_consumer_lock   = threading.Lock()

def draw_danger_zone(frame, w, h):
    overlay = frame.copy()
    pts = np.array([
        (int(w*0.25), h), (int(w*0.75), h),
        (int(w*0.65), int(h*0.30)), (int(w*0.35), int(h*0.30))
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 0, 80))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def put_text_bg(frame, text, pos, scale=0.60, color=(255,255,255), bg=(0,0,0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = pos
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def camera_loop():
    global latest_frame, camera_running, sound_playing

    with CONFIG_LOCK:
        cam_idx = CONFIG["camera_index"]

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    update_live_status(cam_resolution=f"{cam_w}×{cam_h}",
                       status="CLEAR", fps=0,
                       alert_distance_cm=CONFIG["alert_distance_cm"],
                       critical_distance_cm=CONFIG["critical_distance_cm"],
                       confidence_threshold=CONFIG["confidence_threshold"],
                       focal_length=CONFIG["focal_length"])

    fps_buf   = deque(maxlen=30)
    prev_time = time.time()
    peak_fps  = 0.0
    peak_threat = 0.0

    print(f"📷 Camera started: {cam_w}×{cam_h}")

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        with CONFIG_LOCK:
            mirror      = CONFIG["mirror_camera"]
            show_dz     = CONFIG["show_danger_zone"]
            conf_thresh = CONFIG["confidence_threshold"]
            alert_d     = CONFIG["alert_distance_cm"]
            crit_d      = CONFIG["critical_distance_cm"]
            do_calib    = CONFIG.get("_calibrate", False)
            calib_dist  = CONFIG["calibration_distance_cm"]
            ph_cm       = CONFIG["known_person_height_cm"]
            fl          = CONFIG["focal_length"]

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-5))
        prev_time = now
        fps = sum(fps_buf) / len(fps_buf)
        peak_fps = max(peak_fps, fps)

        if show_dz:
            draw_danger_zone(frame, w, h)

        human_in_range  = False
        human_critical  = False
        closest_dist    = 99999
        dets_this_frame = 0
        peak_frame_threat = 0.0
        had_lowconf     = False

        if model is not None:
            results = model(frame, verbose=False, conf=conf_thresh * 0.6, classes=[0])
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_thresh:
                        had_lowconf = True
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_h = y2 - y1
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Calibration
                    if do_calib and box_h > 0:
                        new_fl = (box_h * calib_dist) / ph_cm
                        adaptive_focal_history.append(new_fl)
                        if len(adaptive_focal_history) >= 5:
                            with CONFIG_LOCK:
                                CONFIG["focal_length"] = float(np.mean(adaptive_focal_history))
                                fl = CONFIG["focal_length"]
                            print(f"🔧 Focal length: {fl:.1f} px")
                        with CONFIG_LOCK:
                            CONFIG["_calibrate"] = False
                        do_calib = False

                    distance_cm = int((ph_cm * fl) / box_h) if box_h > 0 else 99999
                    closest_dist = min(closest_dist, distance_cm)
                    dets_this_frame += 1

                    threat = compute_threat_score(distance_cm, conf)
                    peak_frame_threat = max(peak_frame_threat, threat)
                    peak_threat = max(peak_threat, threat)

                    if distance_cm <= crit_d:
                        alert_level = "CRITICAL"
                        label = f"⚠ HUMAN {distance_cm}cm CRITICAL"
                        color = (0, 0, 255)
                        human_in_range = True
                        human_critical = True
                    elif distance_cm <= alert_d:
                        alert_level = "ALERT"
                        label = f"! HUMAN {distance_cm}cm ALERT"
                        color = (0, 100, 255)
                        human_in_range = True
                    else:
                        alert_level = "CLEAR"
                        label = f"HUMAN {distance_cm}cm"
                        color = (0, 220, 80)

                    thickness = 3 if distance_cm <= alert_d else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    put_text_bg(frame, label, (x1, max(y1-14, 20)), scale=0.6, color=color)
                    put_text_bg(frame, f"{conf*100:.0f}%", (x2-52, max(y1-14, 20)),
                                scale=0.5, color=(200,200,200), bg=(40,40,40))

                    log_detection(distance_cm, conf, box_h, alert_level, threat, cx, cy, w, h)

        if dets_this_frame == 0:
            log_clear_frame(had_lowconf)

        # HUD
        sc = (0,0,200) if human_critical else (0,80,200) if human_in_range else (20,120,20)
        st = ("⚠ CRITICAL — HUMAN VERY CLOSE" if human_critical
              else "⚠ ALERT — HUMAN IN BLIND SPOT" if human_in_range
              else "✓  CLEAR")
        cv2.rectangle(frame, (0,0), (w,42), sc, -1)
        cv2.putText(frame, st, (10,29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

        with sess_lock:
            total_d = session_stats["total_detections"]
        info = [
            f"FPS: {fps:.1f}", f"Model: {model_name}",
            f"Closest: {closest_dist if closest_dist<9999 else '---'} cm",
            f"Alert: {alert_d} cm",  f"Detections: {total_d}",
            f"Threat: {int(peak_frame_threat)}",
        ]
        for i, line in enumerate(info):
            put_text_bg(frame, line, (w-280, 65+i*28), scale=0.5, color=(220,220,220), bg=(10,10,10))

        put_text_bg(frame, "EDGE EYE v4.0 | edge_eye.db",
                    (10, h-12), scale=0.44, color=(130,130,130), bg=(0,0,0))

        update_live_status(
            fps=round(fps, 1),
            status="CRITICAL" if human_critical else "ALERT" if human_in_range else "CLEAR",
            closest_cm=closest_dist if closest_dist < 9999 else None,
            focal_length=round(fl, 1),
            alert_distance_cm=alert_d, critical_distance_cm=crit_d,
            confidence_threshold=conf_thresh,
            detections_this_frame=dets_this_frame,
            peak_threat=round(peak_threat, 1),
            peak_fps=round(peak_fps, 1),
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
                alert_sound.play(-1);  sound_playing = True
            elif not human_in_range and sound_playing:
                alert_sound.stop();    sound_playing = False

        # Only encode JPEG when a consumer is connected (saves CPU)
        with _consumer_lock:
            has_consumer = _frame_consumers > 0

        if has_consumer:
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with frame_lock:
                    latest_frame = buf.tobytes()

    cap.release()
    update_live_status(status="OFFLINE", fps=0.0)
    print("📷 Camera stopped")

cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()

# ─────────────────────────────────────────────
# MJPEG GENERATOR
# ─────────────────────────────────────────────
def gen_frames():
    global _frame_consumers
    with _consumer_lock:
        _frame_consumers += 1
    try:
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    finally:
        with _consumer_lock:
            _frame_consumers -= 1

# ─────────────────────────────────────────────
# ROUTES — Static
# ─────────────────────────────────────────────
@app.route('/')
def index():
    for name in ['edge_eye_v5.html', 'edge_eye_v4.html', 'edge_eye_dashboard_v3.html']:
        if (STATIC_DIR / name).exists():
            return send_from_directory(str(STATIC_DIR), name)
    return "Dashboard HTML not found", 404

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ─────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────
@app.route('/api/status')
def api_status():
    with live_lock:
        return jsonify(dict(live_status))

@app.route('/api/detections')
def api_detections():
    limit  = min(int(request.args.get('limit', 500)), 2000)
    sess   = request.args.get('session', session_id)
    offset = int(request.args.get('offset', 0))
    level  = request.args.get('level')   # optional filter: CLEAR/ALERT/CRITICAL
    q = "SELECT * FROM detections WHERE session_id=?"
    args = [sess]
    if level:
        q += " AND alert_level=?"; args.append(level.upper())
    q += " ORDER BY id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    with get_db() as conn:
        rows = conn.execute(q, args).fetchall()
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
            "SELECT * FROM detections WHERE session_id=? ORDER BY id", (sid,)
        ).fetchall()
    return jsonify({"session": dict(sess), "detections": [dict(r) for r in dets]})

@app.route('/api/stats/hourly')
def api_hourly():
    sess = request.args.get('session', session_id)
    with get_db() as conn:
        rows = conn.execute("""
            SELECT strftime('%H:%M', timestamp) as t,
                   COUNT(*) as total,
                   SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END) as critical,
                   SUM(CASE WHEN alert_level='ALERT'    THEN 1 ELSE 0 END) as alert,
                   SUM(CASE WHEN alert_level='CLEAR'    THEN 1 ELSE 0 END) as clear,
                   ROUND(AVG(distance_cm),1) as avg_dist,
                   ROUND(AVG(confidence)*100,1) as avg_conf,
                   ROUND(AVG(threat_score),1) as avg_threat
            FROM detections WHERE session_id=?
            GROUP BY strftime('%H:%M', timestamp)
            ORDER BY t""", (sess,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/stats/summary')
def api_summary():
    sess = request.args.get('session', session_id)
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                                   as total,
                SUM(CASE WHEN alert_level IN ('ALERT','CRITICAL') THEN 1 ELSE 0 END) as alerts,
                SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END)   as critical,
                SUM(CASE WHEN alert_level='CLEAR'    THEN 1 ELSE 0 END)   as safe,
                ROUND(AVG(confidence)*100,2)   as avg_confidence,
                ROUND(MAX(confidence)*100,2)   as max_confidence,
                ROUND(MIN(confidence)*100,2)   as min_confidence,
                ROUND(AVG(distance_cm),1)      as avg_distance,
                MIN(distance_cm)               as min_distance,
                MAX(distance_cm)               as max_distance,
                ROUND(AVG(threat_score),1)     as avg_threat,
                ROUND(MAX(threat_score),1)     as max_threat
            FROM detections WHERE session_id=?""", (sess,)).fetchone()
    return jsonify(dict(row) if row else {})

@app.route('/api/accuracy')
def api_accuracy():
    """
    Returns confusion matrix and derived metrics for the current session.
    Also returns a per-minute timeline of precision/recall for charting.
    """
    sess = request.args.get('session', session_id)
    with get_db() as conn:
        s = conn.execute("SELECT * FROM sessions WHERE id=?", (sess,)).fetchone()
        if not s:
            return jsonify({"error": "session not found"}), 404
        s = dict(s)

        # Per-minute breakdown for timeline charts
        rows = conn.execute("""
            SELECT strftime('%H:%M', timestamp) as minute,
                   COUNT(*) as total,
                   SUM(CASE WHEN alert_level IN ('ALERT','CRITICAL') THEN 1 ELSE 0 END) as tp,
                   SUM(CASE WHEN alert_level='CLEAR' THEN 1 ELSE 0 END) as fp,
                   ROUND(AVG(confidence)*100,1) as avg_conf,
                   ROUND(AVG(threat_score),1)   as avg_threat
            FROM detections WHERE session_id=?
            GROUP BY strftime('%H:%M', timestamp)
            ORDER BY minute""", (sess,)).fetchall()

    tp = s.get("cm_tp", 0) or 0
    fp = s.get("cm_fp", 0) or 0
    fn = s.get("cm_fn", 0) or 0
    tn = s.get("cm_tn", 0) or 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    timeline = []
    for r in rows:
        r = dict(r)
        tpv = r.get("tp", 0) or 0
        fpv = r.get("fp", 0) or 0
        pre = tpv / (tpv + fpv) if (tpv + fpv) > 0 else 0.0
        r["precision"] = round(pre * 100, 1)
        timeline.append(r)

    return jsonify({
        "cm": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision":   round(precision   * 100, 1),
        "recall":      round(recall      * 100, 1),
        "f1_score":    round(f1          * 100, 1),
        "accuracy":    round(accuracy    * 100, 1),
        "specificity": round(specificity * 100, 1),
        "fpr":         round(fpr         * 100, 1),
        "timeline":    timeline,
    })

@app.route('/api/heatmap')
def api_heatmap():
    with heatmap_lock:
        grid = heatmap_grid.tolist()
        mx   = float(heatmap_grid.max()) if heatmap_grid.max() > 0 else 1.0
    return jsonify({"grid": grid, "max": mx, "cols": HMAP_COLS, "rows": HMAP_ROWS})

@app.route('/api/heatmap/reset', methods=['POST'])
def api_heatmap_reset():
    reset_heatmap()
    return jsonify({"ok": True})

@app.route('/api/config', methods=['GET'])
def api_config_get():
    with CONFIG_LOCK:
        return jsonify({k: v for k, v in CONFIG.items() if not k.startswith('_')})

@app.route('/api/config', methods=['POST'])
def api_config_set():
    data = request.get_json(force=True)
    with CONFIG_LOCK:
        for k, v in data.items():
            if k in CONFIG and not k.startswith('_'):
                CONFIG[k] = type(CONFIG[k])(v)
    with CONFIG_LOCK:
        cfg = {k: v for k, v in CONFIG.items() if not k.startswith('_')}
    return jsonify({"ok": True, "config": cfg})

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    with CONFIG_LOCK:
        CONFIG["_calibrate"] = True
    return jsonify({"ok": True, "message": "Stand 150 cm from camera"})

@app.route('/api/report/json')
def api_report():
    sess = request.args.get('session', session_id)
    with get_db() as conn:
        s    = conn.execute("SELECT * FROM sessions WHERE id=?", (sess,)).fetchone()
        summ = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN alert_level IN ('ALERT','CRITICAL') THEN 1 ELSE 0 END) as alert,
                   SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END) as critical,
                   SUM(CASE WHEN alert_level='CLEAR'    THEN 1 ELSE 0 END) as safe,
                   ROUND(AVG(confidence)*100,2) as avg_confidence,
                   MIN(distance_cm) as min_distance,
                   MAX(distance_cm) as max_distance,
                   ROUND(AVG(threat_score),1) as avg_threat
            FROM detections WHERE session_id=?""", (sess,)).fetchone()
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify({"session": dict(s), "summary": dict(summ)})

@app.route('/api/db/export')
def api_db_export():
    return send_from_directory(str(BASE_DIR), 'edge_eye.db', as_attachment=True)

# ─────────────────────────────────────────────
# SSE — Server-Sent Events for instant updates
# ─────────────────────────────────────────────
@app.route('/api/stream')
def api_stream():
    """Pushes live_status as SSE every 500 ms"""
    def generate():
        while True:
            with live_lock:
                data = json.dumps(live_status)
            yield f"data: {data}\n\n"
            time.sleep(0.5)
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚍 EDGE EYE v4.0 — Flask Backend")
    print(f"📂 DB:        {DB_PATH}")
    print(f"🌐 Dashboard: http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
