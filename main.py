"""
main.py — VisionThreatRobot Central Execution Module
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

from threat_logic  import calculate_threat, is_threat, threat_level_label, THREAT_THRESHOLD
from anomaly       import get_anomaly_score, get_anomaly_score_from_box
from secure_comm   import build_alert_payload, print_alert
from robot_control import threat_response, cleanup

try:
    import cv2
    from ultralytics import YOLO
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False
    print("[main] OpenCV/YOLO not found — demo mode only.")


# ── CONFIG ─────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"
PERSON_CLASS_ID = 0
ZONE_BOUNDARY   = (100, 100, 540, 380)
ALERT_COOLDOWN  = 3.0
DISPLAY_W, DISPLAY_H = 640, 480


# ── HELPERS ────────────────────────────────────────
def _in_zone(box_xyxy):
    x1,y1,x2,y2 = box_xyxy
    cx,cy = (x1+x2)/2, (y1+y2)/2
    return int(ZONE_BOUNDARY[0] <= cx <= ZONE_BOUNDARY[2] and
               ZONE_BOUNDARY[1] <= cy <= ZONE_BOUNDARY[3])


def _draw_zone(frame):
    x1,y1,x2,y2 = ZONE_BOUNDARY
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (x1+5, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


def _draw_hud(frame, frame_id, fps, conf, anomaly, threat, label):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (320,140), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Conf: {conf:.2f}",
        f"Anomaly: {anomaly:.2f}",
        f"Threat: {threat:.2f} [{label}]",
        f"Thresh: {THREAT_THRESHOLD}"
    ]

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 20 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


def _load_model():
    if not os.path.exists(YOLO_MODEL):
        print("❌ YOLO model not found")
        sys.exit(1)
    return YOLO(YOLO_MODEL)


# ── CSV LOGGER ─────────────────────────────────────
class CSVLogger:
    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frame","conf","anomaly","zone","threat","label","alert"])

    def log(self, f, c, a, z, t, l, al):
        self.writer.writerow([f,c,a,z,t,l,al])

    def close(self):
        self.file.close()


# ── DEMO MODE ──────────────────────────────────────
def run_demo(iterations=10):
    print("\n--- DEMO MODE ---")

    for i in range(iterations):
        conf = np.random.uniform(0.5, 1.0)
        anomaly = get_anomaly_score()
        zone = np.random.randint(0,2)

        threat = calculate_threat(conf, anomaly, zone)
        label = threat_level_label(threat)

        print(f"Frame {i} | Conf={conf:.2f} Anom={anomaly:.2f} Threat={threat:.2f}")

        if is_threat(threat):
            payload = build_alert_payload(threat, conf, anomaly, zone, i)
            print_alert(payload)
            threat_response(threat)

        time.sleep(0.2)


# ── LIVE MODE ──────────────────────────────────────
def run_live(source=0, save=None):

    if not _CV_AVAILABLE:
        print("❌ OpenCV not available")
        return

    model = _load_model()

    # ✅ FIXED CAMERA
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("⚠ Camera failed → using demo video")
        cap = cv2.VideoCapture("demo_video.mp4")

    logger = CSVLogger(save) if save else None

    frame_id = 0
    last_alert = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        results = model(frame)

        conf = 0
        anomaly = 0
        zone = 0

        for box in results[0].boxes:
            if int(box.cls[0]) == PERSON_CLASS_ID:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                zone = _in_zone(xyxy)

                anomaly = get_anomaly_score_from_box(
                    conf, xyxy, DISPLAY_W, DISPLAY_H, frame_id
                )

        threat = calculate_threat(conf, anomaly, zone)
        label = threat_level_label(threat)

        # 🔥 LOG
        print(f"\nFrame {frame_id}")
        print(f"Conf: {conf:.2f}, Anomaly: {anomaly:.2f}, Threat: {threat:.2f}")

        if is_threat(threat) and (now - last_alert > ALERT_COOLDOWN):
            payload = build_alert_payload(threat, conf, anomaly, zone, frame_id)
            print_alert(payload)
            threat_response(threat)
            last_alert = now

        if logger:
            logger.log(frame_id, conf, anomaly, zone, threat, label, int(is_threat(threat)))

        annotated = results[0].plot()
        _draw_zone(annotated)
        _draw_hud(annotated, frame_id, fps, conf, anomaly, threat, label)

        cv2.imshow("VisionThreatRobot", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cleanup()
    cv2.destroyAllWindows()

    if logger:
        logger.close()


# ── MAIN ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--save", default=None)

    args = parser.parse_args()

    if args.demo:
        run_demo(args.iterations)
    else:
        try:
            src = int(args.source)
        except:
            src = args.source

        run_live(src, args.save)


if __name__ == "__main__":
    main()