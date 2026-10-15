#Code the captures and stores 300 sets of landmark data about a specific ASL alphabet in pickle file

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
j = 'z' #what ASL letter the data is for (e.g: z)

#checks if the directory of that specific letter is already made to avoid overwriting data
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

#Loops until a specific theshold (300) of data is reached
while True:
    data_aux = []

    ret, frame = cap.read()
    height, width = frame.shape[:2]
    min_dim = min(height, width)

    startX = int((width - min_dim) / 2)
    startY = int((height - min_dim) / 2)

    #Cropping image for standardization
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

    #Collects data if q or o is pressed
    if waitkey == ord('q') or waitkey == ord('o'):
        if results.hand_landmarks:
            loop += 1
            print(f"Video{j}" + f"Frame{loop}")

            #Stores image data for further checking after data collection to see if its valid
            cv2.imwrite(f'{IMAGE_DIR}/{j}/{j.upper()}_{loop}.jpg', croppedFrame)
            
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

#Stores data inside pickle file
f = open(f"{DATA_DIR}/{j}/landmarkData{j.upper()}.pickle",'wb')
pickle.dump({'data':data,'labels':labels,'imgName':imgName}, f)
f.close()

