"""
robot_control.py
----------------
Navigation / Hardware Control Module for VisionThreatRobot.

Provides:
  - Simulation mode   : runs on any PC / laptop (default)
  - GPIO mode         : activates on Raspberry Pi with RPi.GPIO installed

Movement commands:
  move_forward()   – advance toward target
  move_backward()  – retreat
  turn_left()      – rotate left
  turn_right()     – rotate right
  stop()           – halt all motors
  return_to_base() – navigate back to safe zone
"""

import time
import sys

# ── Detect hardware environment ────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    _HARDWARE_MODE = True
    print("[robot_control] RPi.GPIO detected — hardware mode active.")
except ImportError:
    _HARDWARE_MODE = False
    print("[robot_control] No GPIO library — running in SIMULATION mode.")

# ── Demo speed factor ──────────────────────────────────────────────────────
# 0.15 = 15% of real duration → fast demo on PC, no long waits
# Change to 1.0 for actual robot deployment on Raspberry Pi
DEMO_SPEED_FACTOR = 0.15

# ── GPIO pin mapping (BCM numbering) ──────────────────────────────────────
PIN_MOTOR_LEFT_FWD  = 17
PIN_MOTOR_LEFT_BWD  = 18
PIN_MOTOR_RIGHT_FWD = 22
PIN_MOTOR_RIGHT_BWD = 23
PIN_ENABLE_LEFT     = 25
PIN_ENABLE_RIGHT    = 24

DEFAULT_SPEED = 75

# ── Hardware setup ─────────────────────────────────────────────────────────
if _HARDWARE_MODE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    _pins = [PIN_MOTOR_LEFT_FWD, PIN_MOTOR_LEFT_BWD,
             PIN_MOTOR_RIGHT_FWD, PIN_MOTOR_RIGHT_BWD,
             PIN_ENABLE_LEFT, PIN_ENABLE_RIGHT]
    for pin in _pins:
        GPIO.setup(pin, GPIO.OUT)
    _pwm_left  = GPIO.PWM(PIN_ENABLE_LEFT,  100)
    _pwm_right = GPIO.PWM(PIN_ENABLE_RIGHT, 100)
    _pwm_left.start(0)
    _pwm_right.start(0)


# ── Helpers ────────────────────────────────────────────────────────────────
def _set_motors(left_fwd, left_bwd, right_fwd, right_bwd, speed=DEFAULT_SPEED):
    if _HARDWARE_MODE:
        GPIO.output(PIN_MOTOR_LEFT_FWD,  left_fwd)
        GPIO.output(PIN_MOTOR_LEFT_BWD,  left_bwd)
        GPIO.output(PIN_MOTOR_RIGHT_FWD, right_fwd)
        GPIO.output(PIN_MOTOR_RIGHT_BWD, right_bwd)
        _pwm_left.ChangeDutyCycle(speed)
        _pwm_right.ChangeDutyCycle(speed)


def _log(action: str):
    print(f"  [robot] 🤖 {action}", flush=True)


def _safe_sleep(seconds: float):
    """
    Sleep that scales duration in simulation mode and handles
    KeyboardInterrupt cleanly — no ugly tracebacks on Ctrl+C.
    """
    effective = seconds * (DEMO_SPEED_FACTOR if not _HARDWARE_MODE else 1.0)
    try:
        time.sleep(effective)
    except KeyboardInterrupt:
        _log("Interrupted — stopping safely.")
        stop()
        print("\n[robot_control] Clean exit on Ctrl+C.")
        sys.exit(0)


# ── Movement API ───────────────────────────────────────────────────────────
def move_forward(duration: float = 0.0, speed: int = DEFAULT_SPEED):
    """Drive forward. duration=0 → until stop() is called."""
    _log(f"FORWARD  │ speed={speed}%  │ {duration}s")
    _set_motors(True, False, True, False, speed)
    if duration > 0:
        _safe_sleep(duration)
        stop()


def move_backward(duration: float = 0.0, speed: int = DEFAULT_SPEED):
    """Drive backward."""
    _log(f"BACKWARD │ speed={speed}%  │ {duration}s")
    _set_motors(False, True, False, True, speed)
    if duration > 0:
        _safe_sleep(duration)
        stop()


def turn_left(duration: float = 0.5, speed: int = DEFAULT_SPEED):
    """Rotate left (counter-clockwise)."""
    _log(f"TURN LEFT  │ speed={speed}%  │ {duration}s")
    _set_motors(False, True, True, False, speed)
    _safe_sleep(duration)
    stop()


def turn_right(duration: float = 0.5, speed: int = DEFAULT_SPEED):
    """Rotate right (clockwise)."""
    _log(f"TURN RIGHT │ speed={speed}%  │ {duration}s")
    _set_motors(True, False, False, True, speed)
    _safe_sleep(duration)
    stop()


def stop():
    """Halt all motors immediately."""
    _log("STOP")
    _set_motors(False, False, False, False, 0)


# ── High-level behaviours ──────────────────────────────────────────────────
def return_to_base():
    """Reverse → 180° pivot → advance to safe zone."""
    _log("RETURN TO BASE initiated")
    try:
        move_backward(duration=1.0)
        turn_right(duration=1.0)
        turn_right(duration=1.0)
        move_forward(duration=2.0)
        stop()
        _log("RETURN TO BASE complete ✔")
    except KeyboardInterrupt:
        _log("Interrupted during return-to-base — stopping.")
        stop()
        sys.exit(0)


def threat_response(threat_score: float):
    """
    Autonomous response based on threat level:
      LOW    (T < 0.3)  → patrol forward
      MEDIUM (0.3–0.6)  → halt and observe
      HIGH   (T ≥ 0.6)  → return to safe zone
    """
    try:
        if threat_score >= 0.6:
            _log(f"HIGH threat ({threat_score:.2f}) → RETREATING")
            return_to_base()
        elif threat_score >= 0.3:
            _log(f"MEDIUM threat ({threat_score:.2f}) → OBSERVING")
            stop()
            _safe_sleep(2.0)
        else:
            _log(f"LOW threat ({threat_score:.2f}) → PATROLLING")
            move_forward(duration=1.0)
    except KeyboardInterrupt:
        stop()
        sys.exit(0)


def cleanup():
    """Release GPIO resources on program exit."""
    if _HARDWARE_MODE:
        stop()
        _pwm_left.stop()
        _pwm_right.stop()
        GPIO.cleanup()
        print("[robot_control] GPIO cleaned up.")


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Robot Control Self-Test (Simulation) ──────────")
    move_forward(duration=1.0)
    turn_left(duration=0.5)
    turn_right(duration=0.5)
    move_backward(duration=0.5)
    return_to_base()

    print("\n  Threat response tests:")
    for score in [0.15, 0.45, 0.75]:
        print(f"\n  → threat_score = {score}")
        threat_response(score)
