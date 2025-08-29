import cv2
import numpy as np
import mediapipe as mp
from collections import deque

bpoints, gpoints, rpoints, ypoints = (
    [deque(maxlen=1024)],
    [deque(maxlen=1024)],
    [deque(maxlen=1024)],
    [deque(maxlen=1024)],
)
bindex = gindex = rindex = yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.putText(
    paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
)
cv2.putText(
    paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
)
cv2.putText(
    paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
)
cv2.putText(
    paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2
)

cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(
        frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2
    )

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handLms in result.multi_hand_landmarks:
            for lm in handLms.landmark:
                lmx, lmy = int(lm.x * w), int(lm.y * h)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        thumb = (landmarks[4][0], landmarks[4][1])
        center = fore_finger
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        if abs(thumb[1] - center[1]) < 30:
            bpoints.append(deque(maxlen=1024))
            bindex += 1
            gpoints.append(deque(maxlen=1024))
            gindex += 1
            rpoints.append(deque(maxlen=1024))
            rindex += 1
            ypoints.append(deque(maxlen=1024))
            yindex += 1
        elif center[1] <= 65:
            if 40 <= center[0] <= 140:
                bpoints, gpoints, rpoints, ypoints = (
                    [deque(maxlen=1024)],
                    [deque(maxlen=1024)],
                    [deque(maxlen=1024)],
                    [deque(maxlen=1024)],
                )
                bindex = gindex = rindex = yindex = 0
                paintWindow[:] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0
            elif 275 <= center[0] <= 370:
                colorIndex = 1
            elif 390 <= center[0] <= 485:
                colorIndex = 2
            elif 505 <= center[0] <= 600:
                colorIndex = 3
        else:
            if colorIndex == 0:
                bpoints[bindex].appendleft(center)
            elif colorIndex == 1:
                gpoints[gindex].appendleft(center)
            elif colorIndex == 2:
                rpoints[rindex].appendleft(center)
            elif colorIndex == 3:
                ypoints[yindex].appendleft(center)
    else:
        bpoints.append(deque(maxlen=1024))
        bindex += 1
        gpoints.append(deque(maxlen=1024))
        gindex += 1
        rpoints.append(deque(maxlen=1024))
        rindex += 1
        ypoints.append(deque(maxlen=1024))
        yindex += 1

    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(
                    paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2
                )

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
