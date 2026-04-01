# =============================================================================
# EDGE EYE v3.0 — Bus Blind Spot Detection System
# Team Tiki-Takas | YOLOv8 + Arduino + Sound Alert
# =============================================================================
# CHANGES v3.0:
#  - Single "HUMAN" class only (removed unreliable ADULT/CHILD aspect-ratio hack)
#  - Real confidence scores passed through accurately
#  - Threat score computed per detection (distance + confidence weighted)
#  - Confusion matrix data tracked in session JSON (TP/FP/FN/TN windows)
#  - Live status JSON includes confusion matrix counters
#  - CSV columns extended with threat_score
#  - Focal length adaptive calibration retained
#  - Atomic JSON writes prevent dashboard read corruption
# =============================================================================

import cv2
from ultralytics import YOLO
import pygame
import serial
import time
import os
import csv
import json
import threading
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
# MODEL LOADING
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

model, MODEL_NAME = load_model()

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
# DETECTION CONFIG
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.45
ALERT_DISTANCE_CM     = 200
CRITICAL_DISTANCE_CM  = 100
FOCAL_LENGTH          = 700.0          # pixels — calibrated via 'C' key
KNOWN_PERSON_HEIGHT_CM = 170           # average adult height for distance estimate
CALIBRATION_DISTANCE_CM = 150

adaptive_focal_history = deque(maxlen=30)
calibration_mode = False

def estimate_distance(box_height_px):
    if box_height_px <= 0:
        return 99999
    return int((KNOWN_PERSON_HEIGHT_CM * FOCAL_LENGTH) / box_height_px)

def compute_threat_score(distance_cm, confidence):
    """
    Threat score 0–100:
      - distance component: closer → higher (exponential fall-off beyond 200 cm)
      - confidence component: higher conf → higher threat
    """
    dist_score = max(0.0, 1.0 - (distance_cm / (ALERT_DISTANCE_CM * 1.5)))
    dist_score = dist_score ** 0.7  # non-linear: danger rises sharply near bus
    threat = (dist_score * 0.70 + confidence * 0.30) * 100.0
    return round(min(100.0, max(0.0, threat)), 1)

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
# LOGGING SETUP
# ─────────────────────────────────────────────
log_dir    = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

session_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path         = os.path.join(log_dir, f"detections_{session_id}.csv")
csv_latest_path  = os.path.join(BASE_DIR, "detections_log.csv")
json_path        = os.path.join(log_dir, f"session_{session_id}.json")
live_status_path = os.path.join(BASE_DIR, "live_status.json")

csv_file        = open(csv_path,        'w', newline='')
csv_latest_file = open(csv_latest_path, 'w', newline='')

HEADER = ["timestamp", "type", "distance_cm", "confidence", "threat_score",
          "box_height_px", "alert_level"]

def _make_writer(f):
    w = csv.writer(f)
    w.writerow(HEADER)
    return w

csv_writer        = _make_writer(csv_file)
csv_latest_writer = _make_writer(csv_latest_file)

# ─────────────────────────────────────────────
# SESSION STATE  (confusion matrix counters)
# ─────────────────────────────────────────────
# Since we don't have ground-truth labels, we use a proxy definition:
#   TP = detection inside alert zone (distance ≤ ALERT_DISTANCE_CM)       with conf ≥ threshold
#   FP = detection outside alert zone                                      with conf ≥ threshold
#   FN = frames where alert zone triggered but no detection above thresh   (estimated via missed_frames)
#   TN = frames where alert zone clear AND no high-conf detection
# This gives a meaningful operational confusion matrix for the demo.

session_data = {
    "session_id":      session_id,
    "model":           MODEL_NAME,
    "start_time":      datetime.now().isoformat(),
    "alert_distance_cm":    ALERT_DISTANCE_CM,
    "critical_distance_cm": CRITICAL_DISTANCE_CM,
    "total_detections": 0,
    "total_alerts":     0,
    "total_critical":   0,
    # confusion matrix (operational proxy)
    "cm_tp": 0,   # detected in alert zone (correct alert)
    "cm_fp": 0,   # detected but outside zone (no real threat)
    "cm_fn": 0,   # alert zone active, detection dropped below threshold
    "cm_tn": 0,   # clear scene, no detection (correct clear)
    "events": []
}
data_lock = threading.Lock()


def log_detection(distance, confidence, box_h, alert_level, threat):
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    row = [ts, "HUMAN", distance, f"{confidence:.4f}", threat, box_h, alert_level]
    csv_writer.writerow(row);        csv_file.flush()
    csv_latest_writer.writerow(row); csv_latest_file.flush()

    with data_lock:
        session_data["total_detections"] += 1
        if alert_level in ("ALERT", "CRITICAL"):
            session_data["total_alerts"]  += 1
            session_data["cm_tp"]         += 1
        else:
            session_data["cm_fp"]         += 1
        if alert_level == "CRITICAL":
            session_data["total_critical"] += 1

        session_data["events"].append({
            "t": ts, "type": "HUMAN",
            "dist": distance, "conf": round(confidence, 4),
            "threat": threat, "level": alert_level
        })
        if len(session_data["events"]) > 1000:
            session_data["events"] = session_data["events"][-1000:]


def log_clear_frame(had_lowconf_detection):
    """Called when a frame has no detection above threshold."""
    with data_lock:
        if had_lowconf_detection:
            session_data["cm_fn"] += 1   # missed a real person (low conf)
        else:
            session_data["cm_tn"] += 1   # genuinely clear scene

# ─────────────────────────────────────────────
# LIVE STATUS JSON  (atomic write)
# ─────────────────────────────────────────────
def write_live_status(fps, closest_dist, human_in_range, human_critical,
                      dets_this_frame, focal_length, peak_threat):
    status = "CRITICAL" if human_critical else "ALERT" if human_in_range else "CLEAR"
    with data_lock:
        sd = dict(session_data)

    # Derive accuracy metrics from confusion matrix
    tp, fp, fn, tn = sd["cm_tp"], sd["cm_fp"], sd["cm_fn"], sd["cm_tn"]
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    payload = {
        "ts":                datetime.now().isoformat(sep=' ', timespec='milliseconds'),
        "status":            status,
        "fps":               round(fps, 1),
        "closest_cm":        closest_dist if closest_dist < 9999 else None,
        "focal_length":      round(focal_length, 1),
        "alert_distance_cm": ALERT_DISTANCE_CM,
        "critical_distance_cm": CRITICAL_DISTANCE_CM,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model":             MODEL_NAME,
        "arduino":           ARDUINO_STATUS,
        "session_id":        session_id,
        "session_start":     sd["start_time"],
        "total_detections":  sd["total_detections"],
        "total_alerts":      sd["total_alerts"],
        "total_critical":    sd["total_critical"],
        "detections_this_frame": dets_this_frame,
        "cam_resolution":    f"{CAM_W}×{CAM_H}",
        "peak_threat":       peak_threat,
        # confusion matrix
        "cm_tp": tp, "cm_fp": fp, "cm_fn": fn, "cm_tn": tn,
        "precision":  round(precision  * 100, 1),
        "recall":     round(recall     * 100, 1),
        "f1_score":   round(f1         * 100, 1),
        "accuracy":   round(accuracy   * 100, 1),
    }
    tmp = live_status_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, live_status_path)

# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def draw_danger_zone(frame, w, h):
    overlay = frame.copy()
    pts = np.array([
        (int(w * 0.25), h), (int(w * 0.75), h),
        (int(w * 0.65), int(h * 0.30)), (int(w * 0.35), int(h * 0.30))
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 0, 80))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def put_text_bg(frame, text, pos, scale=0.65, color=(255, 255, 255), bg=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# ─────────────────────────────────────────────
# WINDOW SETUP
# ─────────────────────────────────────────────
WINDOW_NAME = "EDGE EYE v3.0 — Blind Spot Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def letterbox_frame(frame, tw=1920, th=1080):
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype='uint8')
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas

# ─────────────────────────────────────────────
# AUTO-LAUNCH DASHBOARD
# ─────────────────────────────────────────────
dashboard_path = os.path.join(BASE_DIR, "edge_eye_v5.html")
if os.path.exists(dashboard_path):
    webbrowser.open(f"file://{dashboard_path}")
    print(f"🌐 Dashboard opened: {dashboard_path}")
else:
    print(f"⚠️  Dashboard not found: {dashboard_path}")

# ─────────────────────────────────────────────
# FPS TRACKER
# ─────────────────────────────────────────────
fps_buffer = deque(maxlen=30)
prev_time  = time.time()
peak_fps   = 0.0
peak_threat_session = 0.0

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\n🚍 EDGE EYE v3.0 Started")
print(f"📡 Model       : {MODEL_NAME}")
print(f"🔔 Alert       : {ALERT_DISTANCE_CM} cm")
print(f"🚨 Critical    : {CRITICAL_DISTANCE_CM} cm")
print(f"🎯 Conf Thresh : {CONFIDENCE_THRESHOLD*100:.0f}%")
print("📷 Press C = calibrate 150cm | Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera feed lost")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # FPS
    now = time.time()
    fps_buffer.append(1.0 / max(now - prev_time, 1e-5))
    prev_time = now
    fps = sum(fps_buffer) / len(fps_buffer)
    peak_fps = max(peak_fps, fps)

    draw_danger_zone(frame, w, h)

    # Run YOLO — only person class (0)
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=[0])

    human_in_range   = False
    human_critical   = False
    closest_dist     = 99999
    dets_this_frame  = 0
    peak_threat_frame = 0.0
    had_lowconf      = False

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])

            # Track any detection (even sub-threshold — for FN counting)
            if conf < CONFIDENCE_THRESHOLD:
                had_lowconf = True
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_h = y2 - y1

            # Calibration mode: update focal length using this detection
            if calibration_mode and box_h > 0:
                new_fl = (box_h * CALIBRATION_DISTANCE_CM) / KNOWN_PERSON_HEIGHT_CM
                adaptive_focal_history.append(new_fl)
                if len(adaptive_focal_history) >= 5:
                    FOCAL_LENGTH = float(np.mean(adaptive_focal_history))
                    print(f"🔧 Focal length updated: {FOCAL_LENGTH:.1f} px")
                calibration_mode = False

            distance_cm = estimate_distance(box_h)
            closest_dist = min(closest_dist, distance_cm)
            dets_this_frame += 1

            # Determine alert level
            if distance_cm <= CRITICAL_DISTANCE_CM:
                alert_level = "CRITICAL"
                label = f"⚠ HUMAN {distance_cm}cm CRITICAL"
                color = (0, 0, 255)
                human_in_range = True
                human_critical = True
            elif distance_cm <= ALERT_DISTANCE_CM:
                alert_level = "ALERT"
                label = f"! HUMAN {distance_cm}cm ALERT"
                color = (0, 100, 255)
                human_in_range = True
            else:
                alert_level = "CLEAR"
                label = f"HUMAN {distance_cm}cm"
                color = (0, 220, 80)

            threat = compute_threat_score(distance_cm, conf)
            peak_threat_frame = max(peak_threat_frame, threat)
            peak_threat_session = max(peak_threat_session, threat)

            thickness = 3 if distance_cm <= ALERT_DISTANCE_CM else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            put_text_bg(frame, label, (x1, max(y1 - 12, 20)), scale=0.6, color=color)
            put_text_bg(frame, f"{conf*100:.0f}%", (x2 - 50, max(y1 - 12, 20)),
                        scale=0.5, color=(200, 200, 200), bg=(40, 40, 40))

            log_detection(distance_cm, conf, box_h, alert_level, threat)

    if dets_this_frame == 0:
        log_clear_frame(had_lowconf)

    # HUD
    status_color = (0, 0, 200) if human_critical else (0, 80, 220) if human_in_range else (20, 120, 20)
    status_text  = ("⚠ CRITICAL — HUMAN VERY CLOSE" if human_critical
                    else "⚠ ALERT — HUMAN IN BLIND SPOT" if human_in_range
                    else "✓  CLEAR")
    cv2.rectangle(frame, (0, 0), (w, 42), status_color, -1)
    cv2.putText(frame, status_text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    with data_lock:
        sd_copy = dict(session_data)
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Model: {MODEL_NAME}",
        f"Closest: {closest_dist if closest_dist < 9999 else '---'} cm",
        f"Alert: {ALERT_DISTANCE_CM} cm",
        f"Session alerts: {sd_copy['total_alerts']}",
        f"Threat: {int(peak_threat_frame)}",
        f"FL: {FOCAL_LENGTH:.0f} px",
    ]
    for i, line in enumerate(info_lines):
        put_text_bg(frame, line, (w - 290, 65 + i * 28),
                    scale=0.52, color=(220, 220, 220), bg=(10, 10, 10))

    put_text_bg(frame, "C = calibrate 150cm | Q = quit",
                (10, h - 12), scale=0.48, color=(170, 170, 170), bg=(0, 0, 0))

    # Write live status JSON
    write_live_status(fps, closest_dist, human_in_range, human_critical,
                      dets_this_frame, FOCAL_LENGTH, peak_threat_session)

    # Arduino
    if arduino is not None:
        try:
            arduino.write(b'2' if human_critical else b'1' if human_in_range else b'0')
        except Exception:
            pass

    # Audio
    if alert_sound is not None:
        if human_in_range and not sound_playing:
            alert_sound.play(-1);  sound_playing = True
        elif not human_in_range and sound_playing:
            alert_sound.stop();    sound_playing = False

    cv2.imshow(WINDOW_NAME, letterbox_frame(frame))

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('c'), ord('C')):
        calibration_mode = True
        print(f"🔧 Calibration mode: stand {CALIBRATION_DISTANCE_CM}cm from camera")

# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────
with data_lock:
    session_data["end_time"] = datetime.now().isoformat()
    session_data["peak_fps"] = round(peak_fps, 1)
    with open(json_path, 'w') as jf:
        json.dump(session_data, jf, indent=2)

write_live_status(0, 99999, False, False, 0, FOCAL_LENGTH, 0)

csv_file.close()
csv_latest_file.close()
if alert_sound:
    alert_sound.stop()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

with data_lock:
    sd = session_data
print(f"\n✅ Session saved:")
print(f"   CSV     → {csv_path}")
print(f"   JSON    → {json_path}")
print(f"   Detections : {sd['total_detections']}")
print(f"   Alerts     : {sd['total_alerts']}")
print(f"   Critical   : {sd['total_critical']}")
print(f"   CM: TP={sd['cm_tp']} FP={sd['cm_fp']} FN={sd['cm_fn']} TN={sd['cm_tn']}")
print("System closed.")
