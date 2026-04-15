"""
main.py
-------
VisionThreatRobot — Central Execution Module

Pipeline:
  Camera → YOLOv8 Detection → Anomaly Score → Threat Scoring
         → AES-Encrypted Alert → Robot Response → Annotated Display

Run:
    python main.py                  # live webcam (default)
    python main.py --source 0       # webcam index 0
    python main.py --source video.mp4
    python main.py --demo           # demo mode (no camera needed)
"""

import argparse
import time
import sys
import numpy as np

# ── Project modules ────────────────────────────────────────────────────────
from threat_logic import calculate_threat, is_threat, threat_level_label
from anomaly      import get_anomaly_score
from secure_comm  import build_alert_payload, print_alert
from robot_control import threat_response, cleanup

# ── Optional: OpenCV + YOLO (may not be installed in demo mode) ────────────
try:
    import cv2
    from ultralytics import YOLO
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False
    print("[main] OpenCV/YOLO not found — demo mode only.")


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"   # nano model — fastest
PERSON_CLASS_ID = 0              # COCO class 0 = person
ZONE_BOUNDARY   = (100, 100,     # (x1, y1, x2, y2) restricted zone rectangle
                   540, 380)
DISPLAY_WIDTH   = 960
DISPLAY_HEIGHT  = 540
ALERT_COOLDOWN  = 3.0            # seconds between repeated alerts


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _in_zone(box_xyxy, zone=ZONE_BOUNDARY) -> int:
    """Return 1 if bounding box centre is inside the restricted zone."""
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1, zx2, zy2 = zone
    return int(zx1 <= cx <= zx2 and zy1 <= cy <= zy2)


def _draw_zone(frame):
    """Draw the restricted zone rectangle on the frame."""
    zx1, zy1, zx2, zy2 = ZONE_BOUNDARY
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE",
                (zx1 + 4, zy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)


def _draw_hud(frame, frame_id: int, fps: float,
              confidence: float, anomaly: float,
              threat: float, label: str):
    """Overlay HUD information on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    color = (0, 255, 0) if label == "LOW" else \
            (0, 165, 255) if label == "MEDIUM" else (0, 0, 255)

    lines = [
        (f"Frame   : {frame_id}",         (10, 22)),
        (f"FPS     : {fps:.1f}",           (10, 44)),
        (f"Conf    : {confidence:.3f}",    (10, 66)),
        (f"Anomaly : {anomaly:.3f}",       (10, 88)),
        (f"Threat  : {threat:.3f}  [{label}]", (10, 110)),
    ]
    for text, pos in lines:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


# ──────────────────────────────────────────────────────────────────────────
# DEMO MODE (no camera / YOLO required)
# ──────────────────────────────────────────────────────────────────────────
def run_demo(iterations: int = 10):
    """Simulate the full pipeline without a real camera."""
    print("\n" + "═" * 60)
    print("  VisionThreatRobot  —  DEMO MODE")
    print("═" * 60)

    last_alert_time = 0.0

    for i in range(1, iterations + 1):
        # Simulate YOLO detection
        confidence   = round(np.random.uniform(0.5, 0.99), 3)
        zone_flag    = int(np.random.random() > 0.5)
        anomaly      = get_anomaly_score()
        threat       = calculate_threat(confidence, anomaly, zone_flag)
        label        = threat_level_label(threat)

        print(f"\n  Frame {i:>3}  │  Conf={confidence:.3f}  "
              f"Anomaly={anomaly:.3f}  Zone={zone_flag}  "
              f"Threat={threat:.3f}  [{label}]")

        if is_threat(threat):
            now = time.time()
            if now - last_alert_time >= ALERT_COOLDOWN:
                payload = build_alert_payload(
                    threat, confidence, anomaly, zone_flag, i)
                print_alert(payload)
                last_alert_time = now

            threat_response(threat)

        time.sleep(0.3)

    print("\n  ✔ Demo complete.\n")


# ──────────────────────────────────────────────────────────────────────────
# LIVE MODE (webcam / video file)
# ──────────────────────────────────────────────────────────────────────────
def _load_yolo_safe(model_name: str):
    """Load YOLO model, auto-deleting corrupted file and re-downloading."""
    import os, urllib.request, zipfile
    model_path = model_name

    # Detect corrupted file (less than 1 MB is definitely broken)
    if os.path.exists(model_path) and os.path.getsize(model_path) < 1_000_000:
        print(f"[main] Corrupted model file detected — deleting and re-downloading …")
        os.remove(model_path)

    if not os.path.exists(model_path):
        print(f"[main] Downloading {model_name} from Ultralytics …")
        url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[main] Download complete: {model_path}")
        except Exception as e:
            print(f"[main] ERROR downloading model: {e}")
            print("[main] TIP: Run --demo mode instead (no model needed).")
            sys.exit(1)

    return YOLO(model_path)


def run_live(source=0):
    """Run the full pipeline on a live video source."""
    if not _CV_AVAILABLE:
        print("[main] ERROR: OpenCV/YOLO not installed. Use --demo flag.")
        sys.exit(1)

    print(f"\n[main] Loading YOLO model: {YOLO_MODEL} …")
    model = _load_yolo_safe(YOLO_MODEL)

    print(f"[main] Opening video source: {source} …")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[main] ERROR: Cannot open source '{source}'.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    frame_id        = 0
    last_alert_time = 0.0
    prev_time       = time.time()

    print("[main] Detection running. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[main] End of stream.")
            break

        frame_id  += 1
        cur_time   = time.time()
        fps        = 1.0 / max(cur_time - prev_time, 1e-6)
        prev_time  = cur_time

        # ── YOLOv8 inference ──────────────────────────────────────────────
        results = model(frame, verbose=False)

        best_conf  = 0.0
        best_zone  = 0
        detected   = False

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id != PERSON_CLASS_ID:
                continue
            detected = True
            conf     = float(box.conf[0])
            xyxy     = box.xyxy[0].tolist()
            zone     = _in_zone(xyxy)

            if conf > best_conf:
                best_conf = conf
                best_zone = zone

        # ── Anomaly + Threat ──────────────────────────────────────────────
        anomaly = get_anomaly_score() if detected else 0.0
        threat  = calculate_threat(best_conf, anomaly, best_zone)
        label   = threat_level_label(threat)

        # ── Console log ───────────────────────────────────────────────────
        if detected:
            print(f"  Frame {frame_id:>4}  │  "
                  f"Conf={best_conf:.3f}  Anomaly={anomaly:.3f}  "
                  f"Zone={best_zone}  Threat={threat:.3f}  [{label}]")

        # ── Alert & Robot Response ─────────────────────────────────────────
        if is_threat(threat):
            now = time.time()
            if now - last_alert_time >= ALERT_COOLDOWN:
                payload = build_alert_payload(
                    threat, best_conf, anomaly, best_zone, frame_id)
                print_alert(payload)
                last_alert_time = now

            threat_response(threat)

        # ── Draw annotations ──────────────────────────────────────────────
        annotated = results[0].plot()
        _draw_zone(annotated)
        _draw_hud(annotated, frame_id, fps,
                  best_conf, anomaly, threat, label)

        cv2.imshow("VisionThreatRobot — Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[main] User exit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    cleanup()


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="VisionThreatRobot — Cyber-Physical Threat Detection")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 (webcam), or path to video file")
    parser.add_argument("--demo",   action="store_true",
                        help="Run in demo mode (no camera/YOLO needed)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of demo iterations (default: 10)")
    args = parser.parse_args()

    if args.demo:
        run_demo(iterations=args.iterations)
    else:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        run_live(source=source)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received — exiting cleanly. Bye!")
