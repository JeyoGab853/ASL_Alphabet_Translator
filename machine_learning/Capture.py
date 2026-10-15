import os
import sys
import time
import pickle
import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path='../task/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,running_mode=VisionRunningMode.IMAGE,num_hands=1,min_hand_detection_confidence=0.8)

detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = '../ManualAlphabetData'
IMAGE_DIR = '../AlphabetImageData'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 300
j = 'z'

if not os.path.exists(os.path.join(DATA_DIR,str(j))):
    os.makedirs(os.path.join(DATA_DIR,str(j)))
else:
    print(F"ACTION {j} ALREADY EXISTS IN {DATA_DIR}")

if not os.path.exists(os.path.join(IMAGE_DIR,str(j))):
    os.makedirs(os.path.join(IMAGE_DIR, str(j)))
else:
    print(F"ACTION {j} ALREADY EXISTS IN {IMAGE_DIR}")

    while True:
        ynInput = input("continue y/n?")

        if ynInput == 'y':
            break
        else:
            sys.exit()


loop = 0
Timer = 0

Done = False
Starting = False

cap = cv2.VideoCapture(0)

data = []
labels = []
imgName = []

timerLoop = True

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def standardize(landMarkData):
    # --------------------------------------- Resize Values
    targetDim = 300

    xCords = [val * 480 for idx, val in enumerate(landMarkData) if idx % 2 != 0]
    yCords = [val * 480 for idx, val in enumerate(landMarkData) if idx % 2 == 0]

    highestYVal = min(yCords)
    lowestYVal = max(yCords)

    leftMostXVal = min(xCords)
    rightMostXVal = max(xCords)

    Height = lowestYVal - highestYVal
    Width = rightMostXVal - leftMostXVal

    higherDim = max(Height,Width)
    ScaleFactor = targetDim/higherDim

    ResizedX = [(val - Width/2) * ScaleFactor + Width/2  for idx, val in enumerate(xCords)]
    ResizedY = [(val - Height/2) * ScaleFactor + Width/2 for idx, val in enumerate(yCords)]

    # ---------------------------------------

    highestYVal = min(ResizedY)
    leftMostXVal = min(ResizedX)

    lowestYVal = max(ResizedY)
    rightMostXVal = max(ResizedX)

    middleY = targetDim/2 - (lowestYVal - highestYVal)/2
    middleX = targetDim/2 - (rightMostXVal - leftMostXVal)/2

    shiftedX = middleX if rightMostXVal < lowestYVal else 0
    shiftedY = middleY if lowestYVal < rightMostXVal else 0

    StandardX = [clamp(val - leftMostXVal,0,targetDim) + shiftedX  for idx, val in enumerate(ResizedX)]
    StandardY = [clamp(val - highestYVal,0,targetDim) + shiftedY for idx, val in enumerate(ResizedY)]

    StandardX = [val/targetDim for idx, val in enumerate(StandardX)]
    StandardY = [val/targetDim for idx, val in enumerate(StandardY)]

    standardizedLandMarks = []

    for idx in range(len(StandardX)):
        standardizedLandMarks.append(StandardX[idx])
        standardizedLandMarks.append(StandardY[idx])

    image2 = np.zeros((300, 300, 3), dtype=np.uint8)

    for i in range(len(StandardX)):
        point = (int(round(StandardX[i])), int(round(StandardY[i])))
        cv2.circle(image2, point, radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Point2", image2)


while True:
    data_aux = []

    ret, frame = cap.read()
    height, width = frame.shape[:2]
    min_dim = min(height, width)

    startX = int((width - min_dim) / 2)
    startY = int((height - min_dim) / 2)

    croppedFrame = frame[startY: startY + min_dim,startX: startX + min_dim]
    bgrIMG = cv2.cvtColor(croppedFrame, cv2.COLOR_RGB2BGR)

    mpIMG = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgrIMG)
    results = detector.detect(mpIMG)

    if results.hand_landmarks:
        for hand_landmark in results.hand_landmarks:
            draw_landmarks(croppedFrame, hand_landmark)

    cv2.imshow('frame', croppedFrame)

    if loop == dataset_size:
        break

    waitkey = cv2.waitKey(10)

    if waitkey == ord('q') or waitkey == ord('o'):
        if results.hand_landmarks:
            loop += 1
            print(f"Video{j}" + f"Frame{loop}")

            cv2.imwrite(f'{IMAGE_DIR}/{j}/{j.upper()}_{loop}.jpg', croppedFrame)
            #cv2.imwrite(f'{IMAGE_DIR}/{j}/{j.upper()}_{loop}.jpg', croppedFrame)
            for hand_landmark in results.hand_landmarks:
                for i in range(len(hand_landmark)):
                    LandMarkX = hand_landmark[i].x
                    LandMarkY = hand_landmark[i].y

                    data_aux.append(LandMarkX)
                    data_aux.append(LandMarkY)

            data.append(data_aux)
            labels.append(j)
            imgName.append(f'{j.upper()}_{loop}')

    elif waitkey == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()

print(data)
print(labels)

f = open(f"{DATA_DIR}/{j}/landmarkData{j.upper()}.pickle",'wb')
pickle.dump({'data':data,'labels':labels,'imgName':imgName}, f)
f.close()

