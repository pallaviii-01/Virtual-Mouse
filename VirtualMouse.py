import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

# Constants
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7

# Previous and current location
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize webcam
cap = cv2.VideoCapture(0)  # Changed to 0 to ensure default camera is used
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(maxHands=1)

# Get screen size
wScr, hScr = pyautogui.size()

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image")
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
    fingers = detector.fingersUp() if len(lmList) != 0 else []

    # Draw rectangle around frame
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # 4. Only Index Finger: Moving Mode
    if len(fingers) > 1 and fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Move Mouse
        pyautogui.moveTo(clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up: Clicking Mode
    if len(fingers) > 1 and fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            pyautogui.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
