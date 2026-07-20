import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import psutil, os

# ── Constants ──────────────────────────────────────────────────────────────────
EAR_THRESHOLD  = 0.20
CLOSED_FRAMES  = 20
frame_count    = 0

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ── EAR calculation ────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in eye_indices
    ])
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# ── Model download ─────────────────────────────────────────────────────────────
import urllib.request

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# ── Set up FaceLandmarker ──────────────────────────────────────────────────────
options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

cap          = cv2.VideoCapture(0)
closed_count = 0
alert_active = False
process      = psutil.Process(os.getpid())

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                cx = int(lm[idx].x * w)
                cy = int(lm[idx].y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            left_ear  = eye_aspect_ratio(lm, LEFT_EYE_IDX,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if avg_ear < EAR_THRESHOLD:
                closed_count += 1
            else:
                closed_count = 0
                alert_active = False

            if closed_count >= CLOSED_FRAMES:
                alert_active = True

        else:
            closed_count = 0

        if alert_active:
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(frame, "Fatigue Detected",
                        (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

        # ── RAM monitor (every 100 frames) ─────────────────────────────────────
        if frame_count % 100 == 0:
            ram_mb = process.memory_info().rss / 1024 / 1024
            print(f"Frame {frame_count} | RAM usage: {ram_mb:.1f} MB")

        cv2.imshow("Driver Fatigue Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()