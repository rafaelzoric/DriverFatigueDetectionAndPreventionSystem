"""
SleepAway — Eye State Detection (CNN + EAR fusion)
===================================================
Runs on the Raspberry Pi 5. Replaces pure-EAR detection with a
CNN classifier, using EAR as a cross-check.

Requires:  eye_state.tflite  (from train_eye_classifier.py)

Install on Pi:
    pip install tflite-runtime --break-system-packages
    # if that fails:
    pip install tensorflow --break-system-packages

Works with both the USB webcam and the CSI XC9021 camera.
Set USE_CSI = True to use Picamera2 instead of OpenCV capture.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# TFLite runtime — try the lightweight one first
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "eye_state.tflite"
IMG_SIZE     = 64          # must match training
USE_CSI      = False       # True = XC9021 via Picamera2, False = USB webcam
FRAME_W      = 640
FRAME_H      = 480

# Detection thresholds
CNN_CLOSED_THRESH = 0.45   # below this probability = eye closed
EAR_CLOSED_THRESH = 0.25   # EAR fallback threshold
SMOOTH_FRAMES     = 5      # rolling window to suppress single-frame noise
CLOSED_TIME_WARN  = 1.0    # seconds -> score 50
CLOSED_TIME_ALERT = 2.0    # seconds -> score 100

# ─────────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye landmark indices (MediaPipe 468-point model)
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

# Wider set used for cropping the eye region for the CNN
LEFT_EYE_BOX  = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE_BOX = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                 173, 157, 158, 159, 160, 161, 246]


# ─────────────────────────────────────────────────────────────────
# TFLITE MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────
class EyeStateClassifier:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.in_index  = self.input_details[0]["index"]
        self.out_index = self.output_details[0]["index"]
        self.in_dtype  = self.input_details[0]["dtype"]

        # Quantisation params (present if int8 model)
        self.in_scale, self.in_zero = self.input_details[0].get(
            "quantization", (0.0, 0)
        )
        self.out_scale, self.out_zero = self.output_details[0].get(
            "quantization", (0.0, 0)
        )

        print(f"Loaded {model_path}")
        print(f"  input : {self.input_details[0]['shape']} {self.in_dtype}")
        print(f"  output: {self.output_details[0]['shape']}")

    def predict(self, gray_crop):
        """
        gray_crop: single-channel uint8 image of the eye region.
        Returns probability the eye is OPEN (0.0 - 1.0).
        """
        img = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE))
        x = img.astype(np.float32)
        x = np.expand_dims(x, axis=(0, -1))   # -> (1, H, W, 1)

        # Handle quantised input
        if self.in_dtype == np.int8:
            if self.in_scale:
                x = x / self.in_scale + self.in_zero
            x = np.clip(x, -128, 127).astype(np.int8)
        elif self.in_dtype == np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)

        self.interpreter.set_tensor(self.in_index, x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_index)

        # Dequantise output if needed
        if self.output_details[0]["dtype"] in (np.int8, np.uint8):
            if self.out_scale:
                out = (out.astype(np.float32) - self.out_zero) * self.out_scale

        return float(np.squeeze(out))


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def calculate_ear(landmarks, idx, w, h):
    """Classic Eye Aspect Ratio from 6 landmarks."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idx]
    p1, p2, p3, p4, p5, p6 = pts
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def crop_eye(gray_frame, landmarks, idx, w, h, pad=0.35):
    """
    Crop a square region around the eye, padded so the model sees
    the eyelid and surrounding skin (matches MRL dataset framing).
    """
    xs = [landmarks[i].x * w for i in idx]
    ys = [landmarks[i].y * h for i in idx]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    size   = max(x_max - x_min, y_max - y_min) * (1 + pad * 2)
    half   = size / 2

    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, (x1, y1, x2, y2)

    return gray_frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def score_from_duration(closed_seconds):
    """
    Maps sustained eye closure to a 0-100 fatigue score.
    Matches the camera scoring table in the design report.
    """
    if closed_seconds <= 0:                 return 0
    if closed_seconds < 0.4:                return 25    # long blink
    if closed_seconds < CLOSED_TIME_WARN:   return 50
    if closed_seconds < CLOSED_TIME_ALERT:  return 75
    return 100


# ─────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────
def open_camera():
    if USE_CSI:
        from picamera2 import Picamera2
        picam = Picamera2()
        cfg = picam.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
        )
        picam.configure(cfg)
        picam.start()
        time.sleep(1)
        return picam
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        time.sleep(1)
        return cap


def read_frame(cam):
    if USE_CSI:
        return True, cam.capture_array()
    return cam.read()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    classifier = EyeStateClassifier(MODEL_PATH)
    cam = open_camera()

    # Rolling window of recent open/closed decisions
    history = deque(maxlen=SMOOTH_FRAMES)

    closed_start = None
    fps_times    = deque(maxlen=30)

    print("Running — press Q to quit")

    while True:
        t0 = time.time()
        ret, frame = read_frame(cam)
        if not ret:
            print("Camera read failed")
            break

        h, w = frame.shape[:2]

        # KEY STEP: grayscale unifies daylight and IR night input
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        cnn_prob   = None
        ear_avg    = None
        eye_closed = False
        face_found = False

        if results.multi_face_landmarks:
            face_found = True
            lm = results.multi_face_landmarks[0].landmark

            # ---- CNN branch -------------------------------------
            l_crop, l_box = crop_eye(gray, lm, LEFT_EYE_BOX,  w, h)
            r_crop, r_box = crop_eye(gray, lm, RIGHT_EYE_BOX, w, h)

            probs = []
            if l_crop is not None and l_crop.size:
                probs.append(classifier.predict(l_crop))
            if r_crop is not None and r_crop.size:
                probs.append(classifier.predict(r_crop))

            if probs:
                cnn_prob = float(np.mean(probs))   # avg both eyes

            # ---- EAR branch (cross-check) -----------------------
            l_ear = calculate_ear(lm, LEFT_EYE_EAR,  w, h)
            r_ear = calculate_ear(lm, RIGHT_EYE_EAR, w, h)
            ear_avg = (l_ear + r_ear) / 2.0

            # ---- Fusion -----------------------------------------
            # CNN is primary. EAR only overrides when the CNN is
            # uncertain (0.4-0.6), which happens on partial blinks
            # and heavy head rotation.
            if cnn_prob is None:
                eye_closed = ear_avg < EAR_CLOSED_THRESH
            elif 0.40 <= cnn_prob <= 0.60:
                eye_closed = ear_avg < EAR_CLOSED_THRESH
            else:
                eye_closed = cnn_prob < CNN_CLOSED_THRESH

            # Draw eye boxes
            for (x1, y1, x2, y2) in (l_box, r_box):
                colour = (0, 0, 255) if eye_closed else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # ---- Temporal smoothing ---------------------------------
        history.append(1 if eye_closed else 0)
        smoothed_closed = (sum(history) / len(history)) > 0.6 if history else False

        # ---- Duration tracking ----------------------------------
        if smoothed_closed and face_found:
            if closed_start is None:
                closed_start = time.time()
            closed_duration = time.time() - closed_start
        else:
            closed_start = None
            closed_duration = 0.0

        camera_score = score_from_duration(closed_duration)

        # ---- Overlay --------------------------------------------
        fps_times.append(time.time() - t0)
        fps = 1.0 / np.mean(fps_times) if fps_times else 0

        if not face_found:
            status, colour = "NO FACE", (0, 0, 255)
        elif smoothed_closed:
            status, colour = "CLOSED", (0, 165, 255)
        else:
            status, colour = "OPEN", (0, 255, 0)

        y = 28
        cv2.putText(frame, f"Eyes: {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2); y += 28

        if cnn_prob is not None:
            cv2.putText(frame, f"CNN open prob: {cnn_prob:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 26
        if ear_avg is not None:
            cv2.putText(frame, f"EAR: {ear_avg:.3f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += 26

        cv2.putText(frame, f"Closed: {closed_duration:.1f}s", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 26
        cv2.putText(frame, f"Camera score: {camera_score}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2); y += 26
        cv2.putText(frame, f"{fps:.1f} FPS", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        if camera_score >= 100:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(frame, "FATIGUE ALERT", (w // 2 - 170, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)

        cv2.imshow("SleepAway - Eye State Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    if USE_CSI:
        cam.stop()
    else:
        cam.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Stopped")


if __name__ == "__main__":
    main()
