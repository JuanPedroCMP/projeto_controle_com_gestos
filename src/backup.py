import cv2
import mediapipe as mp
import pyautogui as ag
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
)
detector = vision.HandLandmarker.create_from_options(options)

# 21 conexões padrão da mão (pares de índices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),       # indicador
    (0, 9), (9,10), (10,11), (11,12),     # médio
    (0,13), (13,14), (14,15), (15,16),    # anelar
    (0,17), (17,18), (18,19), (19,20)     # mindinho
]

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)

    # Desenhar landmarks se tiver mão
    if result.hand_landmarks:
        for hand in result.hand_landmarks:  # cada mão
            # Converte landmarks normalizados (0–1) para coords de pixel
            pts = []
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append((x, y))
                # ponto
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # linhas (dedos)
            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, pts[i], pts[j], (255, 0, 0), 2)

    cv2.imshow("Hand Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

