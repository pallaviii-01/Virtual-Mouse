import cv2
import time
import os
import HandTrackingModule as htm  # Ensure this module is available

# Camera settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, wCam)
cap.set(4, hCam)

# Folder containing finger images
folderPath = "FingerImages"
if not os.path.exists(folderPath):
    print(f"The directory {folderPath} does not exist.")
    exit()

myList = os.listdir(folderPath)
if not myList:
    print("No images found in the directory.")
    exit()

overlayList = []
for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    if image is None:
        print(f"Error loading image: {imPath}")
        continue
    overlayList.append(image)

if not overlayList:
    print("No valid images to display.")
    exit()

print(f"Loaded {len(overlayList)} images.")
pTime = 0

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        if totalFingers > 0 and totalFingers <= len(overlayList):
            h, w, c = overlayList[totalFingers - 1].shape
            img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
