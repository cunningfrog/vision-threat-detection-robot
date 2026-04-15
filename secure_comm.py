"""
secure_comm.py
--------------
Secure Communication Module for VisionThreatRobot.

Uses AES-256 (EAX mode) for authenticated encryption.
EAX mode provides:
  - Confidentiality  (AES block cipher)
  - Integrity        (MAC tag verification)
  - Authenticity     (prevents tampering)

Suitable for resource-constrained embedded / edge devices.
"""

import json
import time
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# ── Key management (in production: load from secure vault / HSM) ───────────
_MASTER_KEY = get_random_bytes(32)   # AES-256 → 32-byte key


# ──────────────────────────────────────────────────────────────────────────
# Core encryption / decryption
# ──────────────────────────────────────────────────────────────────────────
def encrypt_alert(message: str, key: bytes = None) -> dict:
    """
    Encrypt a plaintext alert message using AES-256 EAX mode.

    Parameters
    ----------
    message : str   – plaintext alert string
    key     : bytes – 32-byte key (uses module master key if None)

    Returns
    -------
    dict with keys: ciphertext, nonce, tag  (all base64-encoded strings)
    """
    if key is None:
        key = _MASTER_KEY

    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(message.encode("utf-8"))

    return {
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "nonce":      base64.b64encode(cipher.nonce).decode(),
        "tag":        base64.b64encode(tag).decode(),
    }


def decrypt_alert(payload: dict, key: bytes = None) -> str:
    """
    Decrypt and verify an AES-256 EAX encrypted payload.

    Parameters
    ----------
    payload : dict – output of encrypt_alert()
    key     : bytes – must match the encryption key

    Returns
    -------
    str – decrypted plaintext, or raises ValueError on tamper detection
    """
    if key is None:
        key = _MASTER_KEY

    ciphertext = base64.b64decode(payload["ciphertext"])
    nonce      = base64.b64decode(payload["nonce"])
    tag        = base64.b64decode(payload["tag"])

    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext.decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Alert builder
# ──────────────────────────────────────────────────────────────────────────
def build_alert_payload(threat_score: float,
                        confidence: float,
                        anomaly: float,
                        zone: int,
                        frame_id: int = 0) -> dict:
    """
    Build a structured JSON alert, then encrypt it.

    Returns dict with:
      - encrypted_payload (ciphertext/nonce/tag)
      - timestamp
      - frame_id
    """
    alert_data = {
        "event":        "THREAT_DETECTED",
        "threat_score": round(threat_score, 4),
        "confidence":   round(confidence, 4),
        "anomaly":      round(anomaly, 4),
        "zone":         zone,
        "frame_id":     frame_id,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    message_json = json.dumps(alert_data)
    encrypted    = encrypt_alert(message_json)

    return {
        "encrypted_payload": encrypted,
        "timestamp":         alert_data["timestamp"],
        "frame_id":          frame_id,
    }


def print_alert(payload: dict):
    """Pretty-print an encrypted alert to console (for demo/debug)."""
    print("\n" + "═" * 55)
    print("  🚨  SECURE ALERT GENERATED")
    print("═" * 55)
    print(f"  Timestamp : {payload['timestamp']}")
    print(f"  Frame     : {payload['frame_id']}")
    print(f"  Nonce     : {payload['encrypted_payload']['nonce'][:24]}…")
    print(f"  Ciphertext: {payload['encrypted_payload']['ciphertext'][:32]}…")
    print(f"  Auth Tag  : {payload['encrypted_payload']['tag']}")
    print("═" * 55 + "\n")


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[secure_comm.py] Self-test …")

    payload = build_alert_payload(
        threat_score=0.75, confidence=0.88,
        anomaly=0.62, zone=1, frame_id=42
    )

    print_alert(payload)

    # Verify round-trip
    decrypted = decrypt_alert(payload["encrypted_payload"])
    data = json.loads(decrypted)
    print("  ✔ Decrypted alert:")
    for k, v in data.items():
        print(f"    {k:>15} : {v}")

    # Tamper detection test
    print("\n  Testing tamper detection …")
    tampered = dict(payload["encrypted_payload"])
    raw = base64.b64decode(tampered["ciphertext"])
    raw = bytes([raw[0] ^ 0xFF]) + raw[1:]          # flip first byte
    tampered["ciphertext"] = base64.b64encode(raw).decode()
    try:
        decrypt_alert(tampered)
        print("  ✘ Tamper NOT detected (unexpected).")
    except Exception:
        print("  ✔ Tamper detected and rejected correctly.")
