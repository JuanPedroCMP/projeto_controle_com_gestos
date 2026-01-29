import cv2
import mediapipe as mp
import pyautogui as ag
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
   
last_move = 0

def main():

    MODEL_PATH = "hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
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

        x0_ant = 0
        y0_ant = 0

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:  # cada mão
                lm0 = hand[0]

                x0 = int(lm0.x * w)
                y0 = int(lm0.y * h)

               ## hand_move_cursor(x0, x0_ant, y0, y0_ant)

                x0_ant = x0
                y0_ant = y0

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

def hand_move_cursor(x, x_ant, y, y_ant): 
    global last_move 

    agora = time.time()

    if agora - last_move < 0.8:
        return
    
    ## TODO Mover com direção do movimento e aceleração

   ## x_move, y_move = 0




    ag.move(x - x_ant, y - y_ant, 0.4)

    last_move = agora

if __name__ == "__main__":
    main()
 
