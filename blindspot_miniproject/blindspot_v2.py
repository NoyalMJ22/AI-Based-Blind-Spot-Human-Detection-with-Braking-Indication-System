# =============================================================================
# EDGE EYE v2.1 - Bus Blind Spot Human Detection System
# Team Tiki-Takas | Adaptive YOLOv8 + Arduino + Sound Alert
# =============================================================================
# v2.1 CHANGES:
#  - Live status JSON written every frame  → dashboard reads in real-time
#  - Auto-opens dashboard in browser on startup
#  - Atomic JSON writes (temp file + rename) to prevent partial reads
#  - Session summary JSON updated on every detection (not just at exit)
#  - Logs folder auto-created, csv_latest always flushed
# =============================================================================

import cv2
from ultralytics import YOLO
import pygame
import serial
import time
import os
import csv
import json
import tempfile
import webbrowser
import numpy as np
from datetime import datetime
from collections import deque

# ─────────────────────────────────────────────
# ARDUINO SERIAL SETUP
# ─────────────────────────────────────────────
try:
    arduino = serial.Serial('COM5', 9600)
    time.sleep(2)
    ARDUINO_STATUS = "Connected (COM5)"
    print("✅ Arduino Connected")
except Exception:
    arduino = None
    ARDUINO_STATUS = "Not Connected"
    print("⚠️  Arduino Not Connected — Software Mode Only")

# ─────────────────────────────────────────────
# MODEL LOADING (dual model with fallback)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    for model_name in ["yolov8n.pt", "yolov10n.pt"]:
        path = os.path.join(BASE_DIR, model_name)
        if os.path.exists(path):
            try:
                m = YOLO(path)
                print(f"✅ Loaded model: {model_name}")
                return m, model_name
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    print("⬇️  Downloading yolov8n.pt ...")
    m = YOLO("yolov8n.pt")
    return m, "yolov8n.pt"

model, model_name = load_model()

# ─────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ─────────────────────────────────────────────
# ADAPTIVE DISTANCE CALIBRATION
# ─────────────────────────────────────────────
ADULT_HEIGHT_CM  = 170
CHILD_HEIGHT_CM  = 120
FOCAL_LENGTH     = 700.0
ALERT_DISTANCE_CM    = 200
CRITICAL_DISTANCE_CM = 100
adaptive_focal_history = deque(maxlen=30)
CALIBRATION_DISTANCE_CM = 150

def classify_and_estimate_height(x1, y1, x2, y2):
    box_h = y2 - y1
    box_w = x2 - x1
    ratio = box_h / max(box_w, 1)
    if ratio > 2.2:
        return "ADULT", ADULT_HEIGHT_CM
    elif ratio > 1.5:
        return "CHILD", CHILD_HEIGHT_CM
    else:
        return "HUMAN", ADULT_HEIGHT_CM

def estimate_distance(box_height_px, real_height_cm):
    if box_height_px <= 0:
        return 99999
    return int((real_height_cm * FOCAL_LENGTH) / box_height_px)

# ─────────────────────────────────────────────
# AUDIO SETUP
# ─────────────────────────────────────────────
pygame.mixer.init()
sound_path = os.path.join(BASE_DIR, "alert_final.wav")
if os.path.exists(sound_path):
    alert_sound = pygame.mixer.Sound(sound_path)
    print("✅ Alert sound loaded")
else:
    alert_sound = None
    print("⚠️  alert_final.wav not found — sound disabled")

sound_playing = False

# ─────────────────────────────────────────────
# LOGGING SETUP (CSV + JSON)
# ─────────────────────────────────────────────
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path     = os.path.join(log_dir, f"detections_{session_id}.csv")
csv_latest   = os.path.join(BASE_DIR, "detections_log.csv")
json_path    = os.path.join(log_dir, f"session_{session_id}.json")

# ── live status file: read by dashboard every second ──
live_status_path = os.path.join(BASE_DIR, "live_status.json")

csv_file        = open(csv_path, 'w', newline='')
csv_latest_file = open(csv_latest, 'w', newline='')

def _make_writer(f):
    w = csv.writer(f)
    w.writerow(["timestamp", "type", "distance_cm", "confidence", "box_height_px"])
    return w

csv_writer        = _make_writer(csv_file)
csv_latest_writer = _make_writer(csv_latest_file)

session_data = {
    "session_id": session_id,
    "model": model_name,
    "start_time": datetime.now().isoformat(),
    "alert_distance_cm": ALERT_DISTANCE_CM,
    "critical_distance_cm": CRITICAL_DISTANCE_CM,
    "total_detections": 0,
    "total_alerts": 0,
    "total_critical": 0,
    "events": []
}

def log_detection(det_type, distance, confidence, box_h):
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    row = [ts, det_type, distance, f"{confidence:.3f}", box_h]
    csv_writer.writerow(row)
    csv_file.flush()
    csv_latest_writer.writerow(row)
    csv_latest_file.flush()
    session_data["total_detections"] += 1
    if distance <= CRITICAL_DISTANCE_CM:
        session_data["total_critical"] += 1
    if distance <= ALERT_DISTANCE_CM:
        session_data["total_alerts"] += 1
    session_data["events"].append({
        "t": ts, "type": det_type,
        "dist": distance, "conf": round(confidence, 3)
    })
    # Keep event list to last 500 entries for memory safety
    if len(session_data["events"]) > 500:
        session_data["events"] = session_data["events"][-500:]

# ─────────────────────────────────────────────
# LIVE STATUS JSON — written atomically each frame
# Dashboard polls this file every second
# ─────────────────────────────────────────────
def write_live_status(fps, closest_dist, human_in_range, human_critical,
                      detections_this_frame, focal_length):
    status = "CRITICAL" if human_critical else "ALERT" if human_in_range else "CLEAR"
    payload = {
        "ts": datetime.now().isoformat(sep=' ', timespec='milliseconds'),
        "status": status,
        "fps": round(fps, 1),
        "closest_cm": closest_dist if closest_dist < 9999 else None,
        "focal_length": round(focal_length, 1),
        "alert_distance_cm": ALERT_DISTANCE_CM,
        "critical_distance_cm": CRITICAL_DISTANCE_CM,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model": model_name,
        "arduino": ARDUINO_STATUS,
        "session_id": session_id,
        "session_start": session_data["start_time"],
        "total_detections": session_data["total_detections"],
        "total_alerts": session_data["total_alerts"],
        "total_critical": session_data["total_critical"],
        "detections_this_frame": detections_this_frame,
        "cam_resolution": f"{CAM_W}×{CAM_H}",
    }
    # Atomic write: write to temp then rename so dashboard never reads partial JSON
    tmp_path = live_status_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp_path, live_status_path)

# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45

def draw_danger_zone(frame, w, h):
    overlay = frame.copy()
    pts = np.array([(int(w*0.25), h), (int(w*0.75), h),
                    (int(w*0.65), int(h*0.3)), (int(w*0.35), int(h*0.3))], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 0, 80))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def put_text_bg(frame, text, pos, scale=0.65, color=(255,255,255), bg=(0,0,0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = pos
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# ─────────────────────────────────────────────
# FULLSCREEN LETTERBOX
# ─────────────────────────────────────────────
WINDOW_NAME = "EDGE EYE — Blind Spot Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def letterbox_frame(frame, target_w=1920, target_h=1080):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype='uint8')
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas

# ─────────────────────────────────────────────
# FPS TRACKER
# ─────────────────────────────────────────────
fps_buffer = deque(maxlen=30)
prev_time  = time.time()

# ─────────────────────────────────────────────
# AUTO-LAUNCH DASHBOARD IN BROWSER
# ─────────────────────────────────────────────
dashboard_path = os.path.join(BASE_DIR, "edge_eye_dashboard.html")
if os.path.exists(dashboard_path):
    webbrowser.open(f"file://{dashboard_path}")
    print(f"🌐 Dashboard opened: {dashboard_path}")
else:
    print(f"⚠️  Dashboard not found at: {dashboard_path}")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\n🚍 EDGE EYE v2.1 Started")
print(f"📡 Model: {model_name}")
print(f"🔔 Alert threshold: {ALERT_DISTANCE_CM} cm")
print(f"🚨 Critical threshold: {CRITICAL_DISTANCE_CM} cm")
print("📷 Press 'C' to calibrate distance at 150 cm")
print("Press 'Q' to quit\n")

calibration_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera feed lost")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ── FPS ──
    now = time.time()
    fps_buffer.append(1.0 / max(now - prev_time, 1e-5))
    prev_time = now
    fps = sum(fps_buffer) / len(fps_buffer)

    draw_danger_zone(frame, w, h)

    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=[0])

    human_in_range      = False
    human_critical      = False
    closest_distance    = 99999
    detections_in_frame = 0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_h = y2 - y1

            person_type, real_h = classify_and_estimate_height(x1, y1, x2, y2)
            distance_cm = estimate_distance(box_h, real_h)

            if calibration_mode:
                new_fl = (box_h * CALIBRATION_DISTANCE_CM) / real_h
                adaptive_focal_history.append(new_fl)
                if len(adaptive_focal_history) >= 5:
                    FOCAL_LENGTH = float(np.mean(adaptive_focal_history))
                    print(f"🔧 Focal length updated: {FOCAL_LENGTH:.1f} px")
                calibration_mode = False

            closest_distance = min(closest_distance, distance_cm)
            detections_in_frame += 1

            if distance_cm <= CRITICAL_DISTANCE_CM:
                label = f"⚠ {person_type} {distance_cm}cm CRITICAL"
                color = (0, 0, 255)
                human_in_range = True
                human_critical = True
            elif distance_cm <= ALERT_DISTANCE_CM:
                label = f"! {person_type} {distance_cm}cm ALERT"
                color = (0, 100, 255)
                human_in_range = True
            else:
                label = f"{person_type} {distance_cm}cm"
                color = (0, 220, 80)

            thickness = 3 if distance_cm <= ALERT_DISTANCE_CM else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            put_text_bg(frame, label, (x1, y1 - 12), scale=0.6, color=color)
            conf_text = f"{conf*100:.0f}%"
            put_text_bg(frame, conf_text, (x2 - 45, y1 - 12),
                        scale=0.5, color=(200, 200, 200), bg=(40, 40, 40))

            log_detection(person_type, distance_cm, conf, box_h)

    # ── HUD Overlay ──
    status_color = (0, 0, 200) if human_critical else \
                   (0, 100, 255) if human_in_range else (30, 150, 30)
    status_text  = "⚠ CRITICAL — AUTO BRAKE" if human_critical else \
                   "⚠ ALERT — HUMAN IN BLIND SPOT" if human_in_range else \
                   "✓  CLEAR"
    cv2.rectangle(frame, (0, 0), (w, 40), status_color, -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    info_lines = [
        f"FPS: {fps:.1f}",
        f"Model: {model_name}",
        f"Closest: {closest_distance if closest_distance < 9999 else '---'} cm",
        f"Alert: {ALERT_DISTANCE_CM} cm",
        f"Session alerts: {session_data['total_alerts']}",
        f"FL: {FOCAL_LENGTH:.0f} px",
    ]
    for i, line in enumerate(info_lines):
        put_text_bg(frame, line, (w - 280, 65 + i * 28),
                    scale=0.55, color=(220, 220, 220), bg=(10, 10, 10))

    put_text_bg(frame, "Press C = calibrate | Q = quit",
                (10, h - 12), scale=0.5, color=(180, 180, 180), bg=(0, 0, 0))

    # ── Write live status JSON ──
    write_live_status(fps, closest_distance, human_in_range, human_critical,
                      detections_in_frame, FOCAL_LENGTH)

    # ── Arduino control ──
    if arduino is not None:
        try:
            arduino.write(b'2' if human_critical else b'1' if human_in_range else b'0')
        except Exception:
            pass

    # ── Audio alert ──
    if alert_sound is not None:
        if human_in_range and not sound_playing:
            alert_sound.play(-1)
            sound_playing = True
        elif not human_in_range and sound_playing:
            alert_sound.stop()
            sound_playing = False

    display_frame = letterbox_frame(frame)
    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('c'), ord('C')):
        calibration_mode = True
        print(f"🔧 Calibration mode: stand {CALIBRATION_DISTANCE_CM}cm from camera")

# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────
session_data["end_time"] = datetime.now().isoformat()
with open(json_path, 'w') as jf:
    json.dump(session_data, jf, indent=2)

# Write final OFFLINE status
write_live_status(0, 99999, False, False, 0, FOCAL_LENGTH)

csv_file.close()
csv_latest_file.close()
if alert_sound:
    alert_sound.stop()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print(f"\n✅ Session saved:")
print(f"   CSV  → {csv_path}")
print(f"   JSON → {json_path}")
print(f"   Total detections : {session_data['total_detections']}")
print(f"   Total alerts     : {session_data['total_alerts']}")
print(f"   Total critical   : {session_data['total_critical']}")
print("System closed successfully.")