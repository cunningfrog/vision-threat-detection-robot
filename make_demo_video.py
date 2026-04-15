"""
make_demo_video.py  — save this in your vision threat robot files folder
Run: python make_demo_video.py
Then: python main.py --source demo_video.mp4
"""
import cv2
import numpy as np

W, H = 640, 480
FPS = 20
DURATION_SEC = 15
TOTAL_FRAMES = FPS * DURATION_SEC
OUT = "demo_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

ZX1, ZY1, ZX2, ZY2 = 160, 120, 480, 360

for i in range(TOTAL_FRAMES):
    t = i / FPS
    frame = np.full((H, W, 3), (30, 30, 30), dtype=np.uint8)

    for gx in range(0, W, 80):
        cv2.line(frame, (gx, 0), (gx, H), (42, 42, 42), 1)
    for gy in range(0, H, 80):
        cv2.line(frame, (0, gy), (W, gy), (42, 42, 42), 1)

    cv2.rectangle(frame, (ZX1, ZY1), (ZX2, ZY2), (0, 0, 200), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (ZX1 + 4, ZY1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    px = int(50 + (W - 150) * (0.5 + 0.45 * np.sin(t * 0.6)))
    py = int(80 + (H - 200) * (0.5 + 0.35 * np.cos(t * 0.4)))
    pw, ph = 80, 160
    conf = round(0.72 + 0.18 * abs(np.sin(t * 0.9)), 2)
    cx, cy = px + pw // 2, py + ph // 2
    in_zone = ZX1 <= cx <= ZX2 and ZY1 <= cy <= ZY2
    box_color = (0, 0, 255) if in_zone else (0, 60, 220)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), box_color, 2)
    cv2.putText(frame, f"person {conf:.2f}", (px, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

    anomaly = round(0.3 + 0.4 * abs(np.sin(t * 1.1)), 2)
    zone_flag = 1 if in_zone else 0
    threat = round(0.4 * conf + 0.4 * anomaly + 0.2 * zone_flag, 2)
    level = "HIGH" if threat >= 0.6 else ("MEDIUM" if threat >= 0.3 else "LOW")
    level_color = (0, 0, 255) if level == "HIGH" else (
                  (0, 165, 255) if level == "MEDIUM" else (0, 200, 0))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Frame   : {i+1}",          (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
    cv2.putText(frame, f"Conf    : {conf}",          (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
    cv2.putText(frame, f"Anomaly : {anomaly}",       (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
    cv2.putText(frame, f"Zone    : {zone_flag}",     (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
    cv2.putText(frame, f"Threat  : {threat} [{level}]", (10,102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, level_color, 1)

    if level == "HIGH":
        al = frame.copy()
        cv2.rectangle(al, (0,0),(W,H),(0,0,180),6)
        cv2.putText(al, "!! THREAT DETECTED !!", (100, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.addWeighted(al, 0.35, frame, 0.65, 0, frame)

    cv2.putText(frame, f"CAM-01  REC  {int(t//60):02d}:{int(t%60):02d}",
                (W-200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,80), 1)
    writer.write(frame)

writer.release()
print(f"[OK] {OUT} created — {TOTAL_FRAMES} frames, {DURATION_SEC}s")
print("[OK] Now run:  python main.py --source demo_video.mp4")
