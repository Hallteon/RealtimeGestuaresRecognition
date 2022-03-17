from keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import mediapipe

gest_map = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    4: "forward",
    5: "back"
}

columns = ['x11', 'x21', 'x12', 'x22', 'x13', 'x23', 'x14', 'x24', 'x15', 'x25',
           'x16', 'x26', 'x17', 'x27', 'x18', 'x28', 'x19', 'x29', 'x110', 'x210', 'x111',
           'x211', 'x112', 'x212', 'x113', 'x213', '114', '214', '115', 'x215', 'x116',
           'x216', 'x117', 'x217', 'x118', 'x218', 'x119', 'x219', 'x120', 'x220', 'x121',
           'x221']


def mapper(val):
    return gest_map[val]


drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = load_model("gestures_model.h5")


with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=1) as hands:
    while (True):
        # frame == 480 640
        # roi == 310 307
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(1)

        cv2.imshow('Hands recognizer', frame)
        hands_roi = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height_frame, width_frame, channels_frame = frame.shape

        if results.multi_hand_landmarks != None:
            new_row = []
            for handLandmarks in results.multi_hand_landmarks:
                try:
                    for point in handsModule.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                              normalizedLandmark.y,
                                                                                              width_frame, height_frame)
                        new_row.extend(list(pixelCoordinatesLandmark))
                except TypeError:
                    break

            if new_row:
                try:
                    data = []
                    data.append(new_row)
                    if len(data[data.index(new_row)]) > 2:
                        df = pd.DataFrame(data, columns=columns)
                        df = df.fillna(0)
                        df = df / 310
                        pred = model.predict(df)
                        move_code = np.argmax(pred[0])
                        user_move_name = mapper(move_code)
                        print(user_move_name)
                except ValueError:
                    continue

        if k == ord('a'):
            start = not start

        if k == ord('q'):
            break

cv2.destroyAllWindows()
capture.release()