"""
threat_logic.py  —  VisionThreatRobot
Risk Fusion Module
 
Formula:  r = alpha*c + beta*a + gamma*z
  c = YOLOv8 detection confidence    (alpha = 0.4)
  a = autoencoder anomaly score      (beta  = 0.4)
  z = zone violation flag 0/1        (gamma = 0.2)
Alert threshold: r >= 0.6
"""
 
W_OBJECT   = 0.35
W_ANOMALY  = 0.45
W_ZONE     = 0.20
THREAT_THRESHOLD = 0.55
 
 
def calculate_threat(confidence, anomaly_score, zone_violation):
    c = max(0.0, min(1.0, float(confidence)))
    a = max(0.0, min(1.0, float(anomaly_score)))
    z = 1 if zone_violation else 0
    return round(W_OBJECT*c + W_ANOMALY*a + W_ZONE*z, 4)
 
 
def is_threat(threat_score):
    return threat_score >= THREAT_THRESHOLD
 
 
def threat_level_label(threat_score):
    if threat_score < 0.3:
        return "LOW"
    elif threat_score < 0.6:
        return "MEDIUM"
    return "HIGH"
 
 
if __name__ == "__main__":
    cases = [(0.85,0.70,1),(0.50,0.30,0),(0.20,0.10,0),(0.60,0.60,0)]
    print(f"{'Conf':>6} {'Anomaly':>8} {'Zone':>6} | {'Score':>7} {'Level':>8} {'Alert':>6}")
    print("-"*52)
    for c,a,z in cases:
        s = calculate_threat(c,a,z)
        print(f"{c:>6.2f} {a:>8.2f} {z:>6}   {s:>7.4f} {threat_level_label(s):>8} {'YES' if is_threat(s) else 'no':>6}")
 