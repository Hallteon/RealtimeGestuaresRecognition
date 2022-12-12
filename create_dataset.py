from csv import writer
import cv2
import mediapipe
import sys
import os.path
import pandas as pd

signs = {'up': 0, 'down': 1, 'right': 2, 'left': 3, 'forward': 4, 'back': 5, 'rotate_clockwise': 6,
         'rotate_anti_clockwise': 7}

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

start = False
max_count = int(sys.argv[1])
name = sys.argv[2]
fetch = sys.argv[3]
counter = 0
time_counter = 0

columns = ['x11', 'x21', 'x12', 'x22', 'x13', 'x23', 'x14', 'x24', 'x15', 'x25',
           'x16', 'x26', 'x17', 'x27', 'x18', 'x28', 'x19', 'x29', 'x110', 'x210', 'x111',
           'x211', 'x112', 'x212', 'x113', 'x213', '114', '214', '115', 'x215', 'x116',
           'x216', 'x117', 'x217', 'x118', 'x218', 'x119', 'x219', 'x120', 'x220', 'x121',
           'x221', 'y']

if fetch == 'train':
    name_file = 'train_dataset.csv'
elif fetch == 'test':
    name_file = 'test_dataset.csv'

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=1) as hands:
    while (True):
        # frame == 480 640
        # roi == 310 307
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height_frame, width_frame, channels_frame = frame.shape
        k = cv2.waitKey(1)

        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)

        cv2.imshow('Hands recognizer', frame)

        if counter == max_count:
            break

        if start:
            new_row = []
            hands_frame = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Collecting {}".format(counter),
                        (10, 20), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in handsModule.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                                  normalizedLandmark.y,
                                                                                                  width_frame,
                                                                                                  height_frame)
                        new_row.extend(list(pixelCoordinatesLandmark))

                if os.path.exists(name_file):
                    with open(r'{0}'.format(name_file), 'a', newline='') as file:
                        new_row.append(signs[name])
                        if len(new_row) > 2:
                            writer_file = writer(file)
                            writer_file.writerow(new_row)

                else:
                    data = []
                    data.append(new_row)
                    data[data.index(new_row)].append(signs[name])
                    if len(data[data.index(new_row)]) > 0:
                        df = pd.DataFrame(data, columns=columns)
                        df = df.fillna(0)
                        df.to_csv(r'{0}'.format(name_file), index=False)

                counter += 1

        if k == ord('a'):
            start = not start

        if k == ord('q'):
            break

cv2.destroyAllWindows()
capture.release()
