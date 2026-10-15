#Code for the Translator

import os.path

import numpy as np
import sys
from tensorflow.keras.models import load_model
import math

import cv2
from mediapipe import Image as ImageMP
from mediapipe import ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks
from pathlib import Path

#checks if the file is in a EXE or in testing
def get_resource_path(relative_path):
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        base_path = Path(__file__).resolve().parent

    return base_path / relative_path

landMarkerPath = os.path.join(get_resource_path('task'),'hand_landmarker.task')

VisionRunningMode = python.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=landMarkerPath)
options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.IMAGE, num_hands=1,
                                       min_hand_detection_confidence=0.8)

detector = vision.HandLandmarker.create_from_options(options)

actions = np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
ModelPath = os.path.join(get_resource_path('Model'),'ASL_Model.h5')
model = load_model(ModelPath)

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

#detects if the hand is moving
def detectmovement(dataSequence):
    # print(len(currentData),len(prevData))
    if len(dataSequence) < 3:
        return 0

    prev2XCords = [val for idx, val in enumerate(dataSequence[2]) if idx % 2 == 0]
    prev2YCords = [val for idx, val in enumerate(dataSequence[2]) if idx % 2 != 0]

    prevXCords = [val for idx, val in enumerate(dataSequence[1]) if idx % 2 == 0]
    prevYCords = [val for idx, val in enumerate(dataSequence[1]) if idx % 2 != 0]

    xCords = [val for idx, val in enumerate(dataSequence[0]) if idx % 2 == 0]
    yCords = [val for idx, val in enumerate(dataSequence[0]) if idx % 2 != 0]

    cordDelta = []

    for i in range(len(prevXCords)):
        x0 = round(prev2XCords[i], 1)
        y0 = round(prev2YCords[i], 1)

        x1 = round(prevXCords[i], 1)
        y1 = round(prevYCords[i], 1)

        x2 = round(xCords[i], 1)
        y2 = round(yCords[i], 1)

        distance0 = math.dist((x0, y0), (x1, y1))
        distance = math.dist((x1, y1), (x2, y2))
        # print(x1, y1, x2, y2, ' thresh: ', distance)

        cordDelta.append(distance0)
        cordDelta.append(distance)

    return sum(cordDelta)

#standardizes landmarks to be in the dimension of 300x300
def standardize(landMarkData, frame):
    # --------------------------------------- Resize Values
    frameHeight, frameWidth = frame.shape[:2]
    targetDim = 300

    xCords = [val * frameWidth for idx, val in enumerate(landMarkData) if idx % 2 == 0]
    yCords = [val * frameHeight for idx, val in enumerate(landMarkData) if idx % 2 != 0]

    highestYVal = min(yCords)
    lowestYVal = max(yCords)

    leftMostXVal = min(xCords)
    rightMostXVal = max(xCords)

    Height = lowestYVal - highestYVal
    Width = rightMostXVal - leftMostXVal

    higherDim = max(Height, Width)
    ScaleFactor = targetDim / higherDim

    ResizedX = [(val - Width / 2) * ScaleFactor + Width / 2 for idx, val in enumerate(xCords)]
    ResizedY = [(val - Height / 2) * ScaleFactor + Width / 2 for idx, val in enumerate(yCords)]

    # ---------------------------------------

    highestYVal = min(ResizedY)
    leftMostXVal = min(ResizedX)

    lowestYVal = max(ResizedY)
    rightMostXVal = max(ResizedX)

    middleY = clamp(targetDim / 2 - (lowestYVal - highestYVal) / 2, 0, targetDim)
    middleX = clamp(targetDim / 2 - (rightMostXVal - leftMostXVal) / 2, 0, targetDim)

    StandardX = [clamp(val - leftMostXVal, 0, targetDim) + middleX for idx, val in enumerate(ResizedX)]
    StandardY = [clamp(val - highestYVal, 0, targetDim) + middleY for idx, val in enumerate(ResizedY)]

    StandardX = [val / targetDim for idx, val in enumerate(StandardX)]
    StandardY = [val / targetDim for idx, val in enumerate(StandardY)]

    standardizedLandMarks = []

    for idx in range(len(StandardX)):
        standardizedLandMarks.append(StandardX[idx])
        standardizedLandMarks.append(StandardY[idx])

    return standardizedLandMarks

#translator class

class Translator:
    def __init__(self):
        self.frame = None
        self.currentLetter = None
        self.dataSequence = []

    def translate(self):
        movement_list = ['j', 'z']

        if not isinstance(self.frame,np.ndarray):
            return None

        data_aux = []

        bgrIMG = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        toRGB = cv2.cvtColor(bgrIMG,cv2.COLOR_BGR2RGB)

        mpIMG = ImageMP(image_format=ImageFormat.SRGB, data=toRGB)
        results = detector.detect(mpIMG)

        if results.hand_landmarks:
            for hand_landmark in results.hand_landmarks:
                draw_landmarks(self.frame, hand_landmark)

                for i in range(len(hand_landmark)):
                    LandMarkX = hand_landmark[i].x
                    LandMarkY = hand_landmark[i].y

                    data_aux.append(LandMarkX)
                    data_aux.append(LandMarkY)

            self.dataSequence.insert(0, data_aux)
            self.dataSequence = self.dataSequence[:3]
            movementThreshold = detectmovement(self.dataSequence)

            res = model.predict(np.expand_dims(standardize(data_aux, self.frame), axis=0), verbose=0)[0]

            predictedLetter = actions[np.argmax(res)]
            predAccuracy = res[np.argmax(res)]

            if movementThreshold > .1:
                if predictedLetter in movement_list and predictedLetter != self.currentLetter and predAccuracy > .9:
                    self.currentLetter = predictedLetter
                    return predictedLetter
            else:
                if res[np.argmax(res)] > .8 and predictedLetter != self.currentLetter:
                    self.currentLetter = actions[np.argmax(res)]
                    return predictedLetter

    def cleardata(self):
        self.frame = None
        self.currentLetter = None
        self.dataSequence = self.dataSequence.clear()

