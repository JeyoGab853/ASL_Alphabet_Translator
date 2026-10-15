import sys

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import os
import numpy as np
import pickle

Actions = os.listdir('./ManualAlphabetData')
labelMap = {label:num for num,label in enumerate(Actions)}

landmarks, labels = [],[]

fileName = 'ProcessedAlphData/fingerSignStandard2.pickle'

if os.path.exists(fileName):
    print(f"file {fileName} already exists")
    while True:
        agInput = input("exit? y/n:")
        if agInput == 'y':
            sys.exit()
        elif agInput == 'n':
            break

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def standardize(landMarkData):
    # --------------------------------------- Resize Values
    targetDim = 300

    xCords = [val * 480 for idx, val in enumerate(landMarkData) if idx % 2 == 0]
    yCords = [val * 480 for idx, val in enumerate(landMarkData) if idx % 2 != 0]

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

    middleY = clamp(targetDim/2 - (lowestYVal - highestYVal)/2,0,targetDim)
    middleX = clamp(targetDim/2 - (rightMostXVal - leftMostXVal)/2,0,targetDim)

    StandardX = [clamp(val - leftMostXVal,0,targetDim) + middleX for idx, val in enumerate(ResizedX)]
    StandardY = [clamp(val - highestYVal,0,targetDim) + middleY for idx, val in enumerate(ResizedY)]

    StandardX = [val/targetDim for idx, val in enumerate(StandardX)]
    StandardY = [val/targetDim for idx, val in enumerate(StandardY)]

    standardizedLandMarks = []

    for idx in range(len(StandardX)):
        standardizedLandMarks.append(StandardX[idx])
        standardizedLandMarks.append(StandardY[idx])

    return standardizedLandMarks

for action in Actions:
    DataInAction = len(os.listdir(f'./ManualAlphabetData/{action}'))
    print(f'{action}:',DataInAction)

    Data = os.listdir(os.path.join('./ManualAlphabetData',action))[0]
    DataDir = os.path.join('./ManualAlphabetData',f'{action}',f'{Data}')

    with open(DataDir, 'rb') as file:
       array = pickle.load(file)

    for dat in array['data']:
        landmarks.append(standardize(dat))

    for key in array['labels']:
        labels.append(labelMap[key])

x = np.array(landmarks)
y = to_categorical(labels).astype(int)

print(x.shape,y.shape)

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1, shuffle=True, random_state=42, stratify=y)

processedData = {'x_train':x_train,'x_val':x_val,
                 'y_train':y_train,'y_val':y_val}

f = open(fileName, 'wb')
pickle.dump(processedData, f)
f.close()
