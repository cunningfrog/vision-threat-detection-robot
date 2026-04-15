"""
anomaly.py  —  VisionThreatRobot
Anomaly Detection Module (Phase 1: NumPy autoencoder, no GPU required)
 
Key improvements over previous version:
  - build_feature_vector()  extracts a meaningful 128-dim vector from
    real YOLO box data (position, size, aspect ratio, motion delta)
  - Weights saved to disk → no retraining on every run
  - is_anomalous() uses the adaptive threshold from training
  - get_anomaly_score_from_box() connects directly to main pipeline
"""
 
import os
import numpy as np
 
np.random.seed(42)
 
INPUT_DIM    = 128
HIDDEN_DIM   = 32
LEARNING_RATE = 0.01
EPOCHS       = 200
WEIGHTS_FILE  = "autoencoder_weights.npz"
 
 
class NumpyAutoencoder:
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        s_e = np.sqrt(2.0 / (input_dim + hidden_dim))
        s_d = np.sqrt(2.0 / (hidden_dim + input_dim))
        self.W_enc = np.random.randn(input_dim, hidden_dim) * s_e
        self.b_enc = np.zeros(hidden_dim)
        self.W_dec = np.random.randn(hidden_dim, input_dim) * s_d
        self.b_dec = np.zeros(input_dim)
        self.threshold = 0.05
        self.trained   = False
 
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
 
    @staticmethod
    def _mse(a, b):
        return float(np.mean((a - b) ** 2))
 
    def forward(self, x):
        self.encoded = self._sigmoid(x @ self.W_enc + self.b_enc)
        self.decoded = self._sigmoid(self.encoded @ self.W_dec + self.b_dec)
        return self.decoded
 
    def _backward(self, x, lr):
        err   = self.decoded - x
        d_dec = err * self.decoded * (1 - self.decoded)
        d_enc = (d_dec @ self.W_dec.T) * self.encoded * (1 - self.encoded)
        self.W_dec -= lr * self.encoded.reshape(-1, 1) * d_dec
        self.b_dec -= lr * d_dec
        self.W_enc -= lr * x.reshape(-1, 1) * d_enc
        self.b_enc -= lr * d_enc
 
    def train(self, normal_data, epochs=EPOCHS, lr=LEARNING_RATE, verbose=True):
        for epoch in range(epochs):
            total = 0.0
            for sample in normal_data:
                recon = self.forward(sample)
                self._backward(sample, lr)
                total += self._mse(sample, recon)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:>4}/{epochs}  |  Avg Loss: {total/len(normal_data):.6f}")
        losses = [self._mse(s, self.forward(s)) for s in normal_data]
        self.threshold = float(np.mean(losses) + 2 * np.std(losses))
        self.trained   = True
        print(f"\n  Training complete. Anomaly threshold = {self.threshold:.6f}\n")
 
    def reconstruction_loss(self, sample):
        return self._mse(sample, self.forward(sample))
 
    def save(self, path=WEIGHTS_FILE):
        np.savez(path, W_enc=self.W_enc, b_enc=self.b_enc,
                 W_dec=self.W_dec, b_dec=self.b_dec,
                 threshold=np.array([self.threshold]))
        print(f"  [anomaly] Weights saved -> {path}")
 
    def load(self, path=WEIGHTS_FILE):
        d = np.load(path)
        self.W_enc, self.b_enc = d["W_enc"], d["b_enc"]
        self.W_dec, self.b_dec = d["W_dec"], d["b_dec"]
        self.threshold = float(d["threshold"][0])
        self.trained   = True
        print(f"  [anomaly] Weights loaded <- {path}  (threshold={self.threshold:.6f})")
 
 
_autoencoder = NumpyAutoencoder()
 
def _initialise_model():
    if os.path.exists(WEIGHTS_FILE):
        _autoencoder.load(WEIGHTS_FILE)
        return
    print("[anomaly.py] Training autoencoder on synthetic normal data ...")
    normal_data = np.random.normal(0.5, 0.05, (300, INPUT_DIM)).clip(0, 1)
    _autoencoder.train(normal_data, verbose=True)
    _autoencoder.save(WEIGHTS_FILE)
 
_initialise_model()
 
 
# ── Feature Engineering ────────────────────────────────────────────────────
_prev_base: np.ndarray = None
 
def build_feature_vector(confidence=0.0, box_xyxy=None,
                          frame_w=640, frame_h=480, frame_idx=0):
    """
    Build a 128-dim feature vector from YOLO detection output.
 
    Base 8 features (meaningful):
      cx, cy          – normalised bounding-box centre
      bw, bh          – normalised box dimensions
      confidence      – YOLO detection confidence
      aspect_ratio    – height/width (normalised 0-1)
      area_fraction   – box area as fraction of frame
      frame_phase     – frame_idx mod 100 / 100
    Motion 8 features:
      abs(current_base - previous_base)  -- inter-frame change
    Tiled 8x to fill 128 dims.
    """
    global _prev_base
    if box_xyxy is None:
        box_xyxy = [0, 0, 0, 0]
 
    x1, y1, x2, y2 = box_xyxy
    bw  = (x2 - x1) / max(frame_w, 1)
    bh  = (y2 - y1) / max(frame_h, 1)
    cx  = ((x1 + x2) / 2) / max(frame_w, 1)
    cy  = ((y1 + y2) / 2) / max(frame_h, 1)
    asp = min(bh / max(bw, 1e-6), 5.0) / 5.0
    area = min(bw * bh, 1.0)
    phase = (frame_idx % 100) / 100.0
 
    base = np.array([cx, cy, bw, bh, confidence, asp, area, phase],
                    dtype=np.float32)
    motion = np.abs(base - _prev_base[:8]) if _prev_base is not None else np.zeros(8, np.float32)
    _prev_base = base.copy()
 
    tile = np.concatenate([base, motion])          # 16-dim
    vec  = np.tile(tile, INPUT_DIM // 16)[:INPUT_DIM]
    vec += np.random.normal(0, 0.002, INPUT_DIM)   # tiny jitter
    return np.clip(vec, 0.0, 1.0).astype(np.float32)
 
 
# ── Public API ─────────────────────────────────────────────────────────────
def get_anomaly_score(feature_vector=None):
    """Normalised anomaly score in [0, 1]. Pass None for demo/sim mode."""
    if feature_vector is None:
        feature_vector = np.random.rand(INPUT_DIM).astype(np.float32)
    fv = np.array(feature_vector, np.float32).flatten()
    if len(fv) != INPUT_DIM:
        fv = np.resize(fv, INPUT_DIM)
    loss  = _autoencoder.reconstruction_loss(fv)
    score = min(1.0, loss / max(_autoencoder.threshold * 1.5, 1e-9))
    return round(score, 4)
 
 
def get_anomaly_score_from_box(confidence, box_xyxy, frame_w=640,
                                frame_h=480, frame_idx=0):
    """Direct wrapper: YOLO box -> anomaly score."""
    vec = build_feature_vector(confidence, box_xyxy, frame_w, frame_h, frame_idx)
    return get_anomaly_score(vec)
 
 
def is_anomalous(score):
    return score >= 0.5
 
 
if __name__ == "__main__":
    print("-- Anomaly Module Self-Test --")
    normal_vec = build_feature_vector(0.85, [220,150,340,400], frame_idx=10)
    attack_vec = build_feature_vector(0.20, [580,400,630,470], frame_idx=11)
    s_n = get_anomaly_score(normal_vec)
    s_a = get_anomaly_score(attack_vec)
    print(f"  Normal : {s_n:.4f}  -> {'ANOMALOUS' if is_anomalous(s_n) else 'NORMAL'}")
    print(f"  Attack : {s_a:.4f}  -> {'ANOMALOUS' if is_anomalous(s_a) else 'NORMAL'}")
    print(f"  Threshold: {_autoencoder.threshold:.6f}")
 