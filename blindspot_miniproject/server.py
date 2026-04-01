# =============================================================================
# EDGE EYE v4.0 — Enhanced Flask Backend
# Team Tiki-Takas | Person Tracking + Heatmap + Threat Scoring + Full API
# =============================================================================
# Run:  python server.py
# Open: http://localhost:5000
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
from flask import Flask, Response, jsonify, send_from_directory, request, send_file

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "edge_eye.db"
STATIC_DIR = BASE_DIR

CONFIG = {
    "alert_distance_cm":    200,
    "critical_distance_cm": 100,
    "confidence_threshold": 0.45,
    "focal_length":         700.0,
    "adult_height_cm":      170,
    "child_height_cm":      120,
    "mirror_camera":        True,
    "camera_index":         0,
    "show_danger_zone":     True,
    "show_tracking":        True,
}

app = Flask(__name__, static_folder=str(STATIC_DIR))

# ── DATABASE ──────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
                model TEXT, total_detections INTEGER DEFAULT 0,
                total_alerts INTEGER DEFAULT 0, total_critical INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0, peak_fps REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                type TEXT NOT NULL, distance_cm INTEGER NOT NULL,
                confidence REAL NOT NULL, box_height INTEGER,
                box_x INTEGER, box_y INTEGER, box_w INTEGER,
                alert_level TEXT NOT NULL, threat_score REAL DEFAULT 0,
                track_id INTEGER DEFAULT 0,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id);
            CREATE INDEX IF NOT EXISTS idx_det_ts      ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_det_level   ON detections(alert_level);
        """)
    print("✅ Database ready:", DB_PATH)

init_db()

# ── MODEL ──────────────────────────────────────
model = None
model_name = "loading..."

def load_model():
    global model, model_name
    try:
        from ultralytics import YOLO
        for name in ["yolov8n.pt", "yolov10n.pt", "yolov8s.pt"]:
            p = BASE_DIR / name
            if p.exists():
                model = YOLO(str(p)); model_name = name
                print(f"✅ Loaded: {name}"); return
        model = YOLO("yolov8n.pt"); model_name = "yolov8n.pt"
    except Exception as e:
        print(f"⚠️  YOLO failed: {e}"); model_name = "mock"

threading.Thread(target=load_model, daemon=True).start()

# ── ARDUINO ────────────────────────────────────
arduino = None; ARDUINO_STATUS = "Not Connected"
try:
    import serial
    arduino = serial.Serial('COM5', 9600); time.sleep(2)
    ARDUINO_STATUS = "Connected (COM5)"
except: pass

# ── AUDIO ──────────────────────────────────────
alert_sound = None; sound_playing = False
try:
    import pygame; pygame.mixer.init()
    snd = BASE_DIR / "alert_final.wav"
    if snd.exists(): alert_sound = pygame.mixer.Sound(str(snd))
except: pass

# ── SESSION ────────────────────────────────────
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_lock = threading.Lock()
session_stats = {"total_detections":0,"total_alerts":0,"total_critical":0,
                 "peak_fps":0.0,"conf_sum":0.0}

with get_db() as conn:
    conn.execute("INSERT OR IGNORE INTO sessions(id,started_at,model) VALUES(?,?,?)",
                 (session_id, datetime.now().isoformat(), model_name))

# ── HEATMAP (32×18 grid) ───────────────────────
HEAT_W, HEAT_H = 32, 18
heatmap_grid   = np.zeros((HEAT_H, HEAT_W), dtype=np.float32)
heatmap_lock   = threading.Lock()

def update_heatmap(cx, cy, fw, fh, w=1.0):
    gx = max(0, min(HEAT_W-1, int(cx/fw*HEAT_W)))
    gy = max(0, min(HEAT_H-1, int(cy/fh*HEAT_H)))
    with heatmap_lock:
        heatmap_grid[gy,gx] += w
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = gx+dx, gy+dy
            if 0<=nx<HEAT_W and 0<=ny<HEAT_H:
                heatmap_grid[ny,nx] += w*0.3

# ── TRACKER ────────────────────────────────────
class Tracker:
    def __init__(self, max_age=12):
        self.tracks={}; self.nid=1; self.max_age=max_age
    def update(self, dets):
        for tid in list(self.tracks):
            self.tracks[tid]['age']+=1
            if self.tracks[tid]['age']>self.max_age: del self.tracks[tid]
        used=set()
        result={}
        for (cx,cy,dist,dtype) in dets:
            best_id,best_d=None,90
            for tid,tr in self.tracks.items():
                if tid in used: continue
                d=((tr['cx']-cx)**2+(tr['cy']-cy)**2)**0.5
                if d<best_d: best_d,best_id=d,tid
            if best_id:
                used.add(best_id)
                self.tracks[best_id].update({'cx':cx,'cy':cy,'dist':dist,'type':dtype,'age':0})
                result[best_id]=self.tracks[best_id]
            else:
                self.tracks[self.nid]={'cx':cx,'cy':cy,'dist':dist,'type':dtype,'age':0,'id':self.nid}
                result[self.nid]=self.tracks[self.nid]; self.nid+=1
        return result

tracker = Tracker()

def threat_score(dist, conf, ptype):
    s = max(0,(300-dist)/300*60) + conf*25 + (15 if ptype=="CHILD" else 0)
    return min(100, round(s,1))

def classify_person(x1,y1,x2,y2):
    r=(y2-y1)/max(x2-x1,1)
    if r>2.2: return "ADULT",CONFIG["adult_height_cm"]
    if r>1.5: return "CHILD",CONFIG["child_height_cm"]
    return "HUMAN",CONFIG["adult_height_cm"]

def estimate_dist(bh, rh):
    return int((rh*CONFIG["focal_length"])/bh) if bh>0 else 99999

def log_det(dtype,dist,conf,bh,bx,by,bw,threat,tid=0):
    level=("CRITICAL" if dist<=CONFIG["critical_distance_cm"] else
           "ALERT"    if dist<=CONFIG["alert_distance_cm"] else "CLEAR")
    ts=datetime.now().isoformat(sep=' ',timespec='seconds')
    with session_lock:
        session_stats["total_detections"]+=1
        session_stats["conf_sum"]+=conf
        if level in ("ALERT","CRITICAL"): session_stats["total_alerts"]+=1
        if level=="CRITICAL":             session_stats["total_critical"]+=1
    with get_db() as conn:
        conn.execute("""INSERT INTO detections
            (session_id,timestamp,type,distance_cm,confidence,box_height,
             box_x,box_y,box_w,alert_level,threat_score,track_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id,ts,dtype,dist,round(conf,4),bh,bx,by,bw,level,threat,tid))
        n=session_stats["total_detections"]
        avg_c=session_stats["conf_sum"]/n if n else 0
        conn.execute("""UPDATE sessions SET
            total_detections=?,total_alerts=?,total_critical=?,avg_confidence=?,model=?
            WHERE id=?""",
            (n,session_stats["total_alerts"],session_stats["total_critical"],
             round(avg_c,4),model_name,session_id))

# ── LIVE STATUS ────────────────────────────────
live_status={"ts":datetime.now().isoformat(),"status":"OFFLINE","fps":0.0,
             "closest_cm":None,"focal_length":CONFIG["focal_length"],
             "model":model_name,"arduino":ARDUINO_STATUS,
             "session_id":session_id,"cam_resolution":"---",
             "detections_this_frame":0,"peak_threat":0,"tracks":[],
             "total_detections":0,"total_alerts":0,"total_critical":0,"peak_fps":0.0}
live_lock=threading.Lock()

def update_live(**kw):
    with live_lock:
        live_status.update(kw)
        live_status["ts"]=datetime.now().isoformat(sep=' ',timespec='milliseconds')
        live_status["total_detections"]=session_stats["total_detections"]
        live_status["total_alerts"]    =session_stats["total_alerts"]
        live_status["total_critical"]  =session_stats["total_critical"]

# ── CAMERA LOOP ────────────────────────────────
frame_lock=threading.Lock(); latest_frame=None; camera_running=False
COLORS={"CLEAR":(0,210,100),"ALERT":(0,140,255),"CRITICAL":(40,40,255)}

def put_bg(frame,text,pos,scale=0.58,col=(255,255,255),bg=(0,0,0)):
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
    x,y=pos
    cv2.rectangle(frame,(x-3,y-th-4),(x+tw+4,y+3),bg,-1)
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,col,2)

def camera_loop():
    global latest_frame,camera_running,sound_playing,model_name
    idx=CONFIG.get("camera_index",0)
    cap=cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)
    cam_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cam_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    update_live(cam_resolution=f"{cam_w}×{cam_h}",status="CLEAR")
    fps_buf=deque(maxlen=30); prev_time=time.time(); camera_running=True
    print(f"📷 Camera {idx}: {cam_w}×{cam_h}")

    while camera_running:
        ret,frame=cap.read()
        if not ret: time.sleep(0.05); continue
        if CONFIG.get("mirror_camera",True): frame=cv2.flip(frame,1)
        h,w=frame.shape[:2]
        now=time.time(); fps_buf.append(1.0/max(now-prev_time,1e-5)); prev_time=now
        fps=sum(fps_buf)/len(fps_buf)

        # Danger zone
        if CONFIG.get("show_danger_zone",True):
            ov=frame.copy()
            pts=np.array([(int(w*.2),h),(int(w*.8),h),(int(w*.65),int(h*.28)),(int(w*.35),int(h*.28))],np.int32)
            cv2.fillPoly(ov,[pts],(0,0,100)); cv2.addWeighted(ov,0.14,frame,0.86,0,frame)
            cv2.polylines(frame,[pts],True,(0,60,140),1)

        human_in_range=False; human_critical=False
        closest_dist=99999; dets_frame=0; peak_thr=0
        det_list=[]

        if model is not None:
            try:
                results=model(frame,verbose=False,conf=CONFIG["confidence_threshold"],classes=[0])
                for result in results:
                    for box in result.boxes:
                        conf=float(box.conf[0])
                        if conf<CONFIG["confidence_threshold"]: continue
                        x1,y1,x2,y2=map(int,box.xyxy[0])
                        bh=y2-y1; bw2=x2-x1; cx=(x1+x2)//2; cy=(y1+y2)//2
                        ptype,rh=classify_person(x1,y1,x2,y2)
                        dist=estimate_dist(bh,rh)
                        thr=threat_score(dist,conf,ptype)
                        peak_thr=max(peak_thr,thr)
                        closest_dist=min(closest_dist,dist)
                        dets_frame+=1
                        det_list.append((cx,cy,dist,ptype))
                        update_heatmap(cx,cy,w,h,1.0 if dist<=200 else 0.4)
                        level=("CRITICAL" if dist<=CONFIG["critical_distance_cm"] else
                               "ALERT" if dist<=CONFIG["alert_distance_cm"] else "CLEAR")
                        color=COLORS[level]
                        if level=="CRITICAL": human_critical=True; human_in_range=True
                        elif level=="ALERT":  human_in_range=True
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if level!="CLEAR" else 2)
                        lbl=f"{'⚠ ' if level!='CLEAR' else ''}{ptype} {dist}cm"
                        ly=max(y1-8,20)
                        put_bg(frame,lbl,(x1+2,ly),col=color)
                        cv2.putText(frame,f"{conf*100:.0f}%",(x2-38,ly),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.44,(150,150,150),1)
                        # Threat bar under box
                        blen=int(bw2*min(thr/100,1.0))
                        cv2.rectangle(frame,(x1,y2+2),(x1+bw2,y2+6),(25,25,25),-1)
                        bc=(40,40,255) if thr>70 else (0,140,255) if thr>40 else (0,200,80)
                        if blen>0: cv2.rectangle(frame,(x1,y2+2),(x1+blen,y2+6),bc,-1)
                        log_det(ptype,dist,conf,bh,x1,y1,bw2,thr)
            except Exception as e:
                print(f"Det error: {e}")

        tracks=tracker.update(det_list) if CONFIG.get("show_tracking",True) else {}
        tlist=[{"id":tid,"cx":t["cx"],"cy":t["cy"],"dist":t["dist"],"type":t["type"]}
               for tid,t in tracks.items()]

        # HUD
        sc=(30,30,200) if human_critical else (0,70,180) if human_in_range else (10,90,35)
        cv2.rectangle(frame,(0,0),(w,44),sc,-1)
        st=("⚠ CRITICAL — BRAKE NOW" if human_critical else
            "⚠ ALERT — HUMAN IN BLIND SPOT" if human_in_range else "✓  ALL CLEAR")
        cv2.putText(frame,st,(12,30),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
        infos=[f"FPS {fps:.1f}",f"Model:{model_name}",
               f"Dets:{dets_frame}",f"Closest:{'---' if closest_dist>=9999 else str(closest_dist)+' cm'}"]
        for i,ln in enumerate(infos):
            (tw,th),_=cv2.getTextSize(ln,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
            x=w-tw-16
            cv2.rectangle(frame,(x-3,52+i*23-th-2),(x+tw+4,52+i*23+3),(0,0,0),-1)
            cv2.putText(frame,ln,(x,52+i*23),cv2.FONT_HERSHEY_SIMPLEX,0.48,(190,190,190),1)
        cv2.putText(frame,"EDGE EYE v4.0  |  edge_eye.db",
                    (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,(70,70,70),1)

        if fps>session_stats["peak_fps"]: session_stats["peak_fps"]=fps
        update_live(fps=round(fps,1),peak_fps=round(session_stats["peak_fps"],1),
                    status="CRITICAL" if human_critical else "ALERT" if human_in_range else "CLEAR",
                    closest_cm=closest_dist if closest_dist<9999 else None,
                    focal_length=round(CONFIG["focal_length"],1),
                    model=model_name,alert_distance_cm=CONFIG["alert_distance_cm"],
                    critical_distance_cm=CONFIG["critical_distance_cm"],
                    confidence_threshold=CONFIG["confidence_threshold"],
                    detections_this_frame=dets_frame,peak_threat=round(peak_thr,1),
                    tracks=tlist,arduino=ARDUINO_STATUS)

        if arduino:
            try: arduino.write(b'2' if human_critical else b'1' if human_in_range else b'0')
            except: pass
        if alert_sound:
            if human_in_range and not sound_playing: alert_sound.play(-1); sound_playing=True
            elif not human_in_range and sound_playing: alert_sound.stop(); sound_playing=False

        ret2,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,82])
        if ret2:
            with frame_lock: latest_frame=buf.tobytes()

    cap.release(); update_live(status="OFFLINE",fps=0.0)

threading.Thread(target=camera_loop, daemon=True).start()

def gen_frames():
    while True:
        with frame_lock: frame=latest_frame
        if frame: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
        time.sleep(0.033)

# ── ROUTES ─────────────────────────────────────
@app.route('/')
def index(): return send_from_directory(str(STATIC_DIR),'edge_eye_v4.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with live_lock: return jsonify(dict(live_status))

@app.route('/api/detections')
def api_detections():
    limit=int(request.args.get('limit',300)); sess=request.args.get('session',session_id)
    offset=int(request.args.get('offset',0))
    with get_db() as conn:
        rows=conn.execute("SELECT * FROM detections WHERE session_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
                          (sess,limit,offset)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/sessions')
def api_sessions():
    with get_db() as conn:
        rows=conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 50").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/heatmap')
def api_heatmap():
    with heatmap_lock: data=heatmap_grid.tolist()
    mx=float(heatmap_grid.max()) if heatmap_grid.max()>0 else 1.0
    return jsonify({"grid":data,"max":mx,"cols":HEAT_W,"rows":HEAT_H})

@app.route('/api/heatmap/reset', methods=['POST'])
def api_heatmap_reset():
    with heatmap_lock: heatmap_grid[:]=0
    return jsonify({"ok":True})

@app.route('/api/stats/summary')
def api_summary():
    sess=request.args.get('session',session_id)
    with get_db() as conn:
        row=conn.execute("""SELECT COUNT(*) as total,
          SUM(CASE WHEN alert_level IN('ALERT','CRITICAL') THEN 1 ELSE 0 END) as alerts,
          SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END) as critical,
          AVG(confidence)*100 as avg_conf, MAX(confidence)*100 as max_conf,
          MIN(confidence)*100 as min_conf, AVG(distance_cm) as avg_dist,
          MIN(distance_cm) as min_dist, MAX(distance_cm) as max_dist,
          AVG(threat_score) as avg_threat, MAX(threat_score) as peak_threat
          FROM detections WHERE session_id=?""",(sess,)).fetchone()
    return jsonify(dict(row) if row else {})

@app.route('/api/stats/timeline')
def api_timeline():
    sess=request.args.get('session',session_id)
    with get_db() as conn:
        rows=conn.execute("""SELECT strftime('%H:%M',timestamp) as t,
          COUNT(*) as total,
          SUM(CASE WHEN alert_level='CRITICAL' THEN 1 ELSE 0 END) as critical,
          SUM(CASE WHEN alert_level='ALERT' THEN 1 ELSE 0 END) as alert,
          AVG(distance_cm) as avg_dist, AVG(threat_score) as avg_threat
          FROM detections WHERE session_id=?
          GROUP BY strftime('%H:%M',timestamp) ORDER BY t""",(sess,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/stats/threat_timeline')
def api_threat_timeline():
    sess=request.args.get('session',session_id)
    with get_db() as conn:
        rows=conn.execute("""SELECT id,timestamp,threat_score,distance_cm,type,alert_level
          FROM detections WHERE session_id=? ORDER BY id DESC LIMIT 200""",(sess,)).fetchall()
    return jsonify([dict(r) for r in reversed(rows)])

@app.route('/api/config',methods=['GET'])
def api_config_get(): return jsonify(CONFIG)

@app.route('/api/config',methods=['POST'])
def api_config_set():
    data=request.get_json(force=True)
    for k,v in data.items():
        if k in CONFIG: CONFIG[k]=type(CONFIG[k])(v)
    return jsonify({"ok":True,"config":CONFIG})

@app.route('/api/calibrate',methods=['POST'])
def api_calibrate():
    CONFIG["_calibrate"]=True; return jsonify({"ok":True})

@app.route('/api/db/export')
def api_db_export():
    return send_file(str(DB_PATH),as_attachment=True,download_name='edge_eye.db')

@app.route('/api/report/json')
def api_report_json():
    sess=request.args.get('session',session_id)
    with get_db() as conn:
        s=conn.execute("SELECT * FROM sessions WHERE id=?",(sess,)).fetchone()
        dets=conn.execute("SELECT * FROM detections WHERE session_id=? ORDER BY id",(sess,)).fetchall()
    report={"generated_at":datetime.now().isoformat(),"session":dict(s) if s else {},
            "summary":{"total":len(dets),
                        "critical":sum(1 for d in dets if d["alert_level"]=="CRITICAL"),
                        "alert":sum(1 for d in dets if d["alert_level"]=="ALERT"),
                        "safe":sum(1 for d in dets if d["alert_level"]=="CLEAR"),
                        "avg_confidence":round(sum(d["confidence"] for d in dets)/len(dets)*100,1) if dets else 0,
                        "min_distance":min((d["distance_cm"] for d in dets),default=0)},
            "detections":[dict(d) for d in dets]}
    return jsonify(report)

if __name__=='__main__':
    print("\n🚍 EDGE EYE v4.0")
    print(f"📂 DB  : {DB_PATH}")
    print(f"🌐 URL : http://localhost:5000\nCtrl+C to stop\n")
    def _open():
        time.sleep(1.8); webbrowser.open("http://localhost:5000")
    threading.Thread(target=_open,daemon=True).start()
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
