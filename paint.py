import cv2
import mediapipe as mp
import numpy as np
import os
import math

# MediaPipe Hands and Drawing utilities initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Capture webcam input and set camera properties
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 5)
width, height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Canvas to draw and pass onto the camera image
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Load header images
folderPath = 'Header'
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in os.listdir(folderPath)]
header = overlayList[0]

# Drawing parameters
drawColor = (0, 0, 255)
thickness = 20  # Drawing thickness
xp, yp = 0, 0  # Coordinates for tracking last finger position

# Indexes for fingertips
tipIds = [4, 8, 12, 16, 20]

# Setup Hand detection
with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Preprocessing image
        image = cv2.flip(image, 1)  # Mirror the image for selfie view
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False  # Improve performance by marking image as not writeable
        results = hands.process(rgb_image)

        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]

                if points:
                    x1, y1 = points[8]  # Index finger tip
                    x2, y2 = points[12]  # Middle finger tip
                    x3, y3 = points[4]  # Thumb tip
                    x4, y4 = points[20]  # Pinky tip

                    # Determine which fingers are up
                    fingers = [
                        int(points[tipIds[0]][0] < points[tipIds[0] - 1][0]),  # Thumb
                        *[int(points[tipIds[i]][1] < points[tipIds[i] - 2][1]) for i in range(1, 5)]  # Fingers
                    ]

                    # Selection Mode (Index and Middle finger up)
                    if fingers[1] and fingers[2] and all(fingers[i] == 0 for i in [0, 3, 4]):
                        xp, yp = x1, y1
                        if y1 < 125:
                            if 170 < x1 < 295:
                                header, drawColor = overlayList[0], (0, 0, 255)
                            elif 436 < x1 < 561:
                                header, drawColor = overlayList[1], (255, 0, 0)
                            elif 700 < x1 < 825:
                                header, drawColor = overlayList[2], (0, 255, 0)
                            elif 980 < x1 < 1105:
                                header, drawColor = overlayList[3], (0, 0, 0)
                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                    # Standby Mode (Index and Pinky finger up)
                    elif fingers[1] and fingers[4] and all(fingers[i] == 0 for i in [0, 2, 3]):
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5)
                        xp, yp = x1, y1

                    # Draw Mode (Only Index finger up)
                    elif fingers[1] and all(fingers[i] == 0 for i in [0, 2, 3, 4]):
                        cv2.circle(image, (x1, y1), thickness // 2, drawColor, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        xp, yp = x1, y1

                    # Clear canvas when fist is closed (no fingers up)
                    elif all(finger == 0 for finger in fingers):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = x1, y1

                    # Adjust thickness using Index and Thumb
                    elif fingers[0] and fingers[1]:
                        dist = int(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) / 3)
                        x0, y0 = (x1 + x3) // 2, (y1 + y3) // 2
                        cv2.circle(image, (x0, y0), dist // 2, drawColor, -1)
                        if fingers[4]:  # Confirm selection with Pinky
                            thickness = dist
                            cv2.putText(image, 'Thickness Set', (x4-25, y4-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        # Update the header and merge the drawing canvas with the camera image
        image[0:125, 0:width] = header
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Display the image
        cv2.imshow('Hand Drawing', img)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
