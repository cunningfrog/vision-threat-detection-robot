# VisionThreatRobot 🤖🔐
## Cyber-Physical Threat Detection System

A modular AI-powered robot system integrating:
- **YOLOv8** real-time object detection
- **Autoencoder** anomaly detection (unsupervised)
- **Weighted threat scoring** model
- **AES-256** encrypted secure alerts
- **Rule-based** autonomous robot navigation

---

## 📁 Project Structure

```
VisionThreatRobot/
├── main.py                  ← Central execution (run this)
├── threat_logic.py          ← Weighted threat scoring T = w1·O + w2·A + w3·Z
├── anomaly.py               ← NumPy autoencoder anomaly detection
├── secure_comm.py           ← AES-256 EAX encrypted alert system
├── robot_control.py         ← Motor control (simulation + RPi GPIO)
├── performance_metrics.py   ← Precision / Recall / F1 + plots
└── requirements.txt         ← Python dependencies
```

---

## ⚙️ Setup Instructions

### Step 1 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run Demo (No Camera Needed)
```bash
python main.py --demo
```
This simulates the full AI pipeline: detection → anomaly → threat → encrypted alert → robot response.

### Step 4 — Run with Webcam
```bash
python main.py --source 0
```

### Step 5 — Run with Video File
```bash
python main.py --source path/to/video.mp4
```

---

## 🧠 Algorithm Overview

| Component           | Algorithm              | Type                    |
|---------------------|------------------------|-------------------------|
| Object Detection    | YOLOv8n                | CNN (Deep Learning)     |
| Anomaly Detection   | Autoencoder            | Unsupervised NN         |
| Threat Scoring      | Weighted Linear Model  | Mathematical Model      |
| Encryption          | AES-256 EAX            | Symmetric Cryptography  |
| Navigation          | Rule-based FSM         | Control Algorithm       |

---

## 📐 Threat Score Formula

```
T = w1 × O + w2 × A + w3 × Z

Where:
  O  = Object detection confidence (YOLOv8)
  A  = Anomaly reconstruction loss (Autoencoder, normalised)
  Z  = Zone violation flag (0 or 1)
  w1 = 0.4, w2 = 0.4, w3 = 0.2
```

**Threshold:** T ≥ 0.6 → ALERT triggered

---

## 📊 Run Performance Metrics
```bash
python performance_metrics.py
```
Generates confusion matrix + bar chart saved as `results.png`.

---

## 🔒 Test Secure Communication
```bash
python secure_comm.py
```

## 🤖 Test Robot Control
```bash
python robot_control.py
```

## 🔍 Test Anomaly Detection
```bash
python anomaly.py
```

## ⚡ Test Threat Scoring
```bash
python threat_logic.py
```

---

## 📈 Results (Sample)

| Metric    | Value  |
|-----------|--------|
| Precision | 0.904  |
| Recall    | 0.863  |
| F1 Score  | 0.883  |
| Accuracy  | 0.900  |
| Latency   | ~5 ms  |

*(Update with your actual measured values)*

---

## 🔮 Future Scope
- Thermal camera integration (multi-modal sensing)
- Trained autoencoder on UCSD anomaly dataset
- Cloud + edge hybrid communication (TLS/MQTT)
- Real industrial Raspberry Pi 4 deployment
- Multi-robot coordination

---

## ⚠️ Limitations
- Autoencoder trained on synthetic normal data
- Lighting sensitivity in YOLO detection
- Simulated motor control on non-Pi hardware
- Limited real-world threat dataset

---

## 🎯 Research Novelty
> "A scalable cyber-physical security framework integrating AI-based visual threat detection, unsupervised anomaly scoring, and cryptographically secure alert transmission for Industry 4.0 autonomous environments."
