import random
import sys

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time

import os
import numpy as np
import pickle

Actions = os.listdir('../ManualAlphabetData')
labelMap = {label:num for num,label in enumerate(Actions)}

landmarks, labels = [],[]

fileName = '../ProcessedAlphData/fingerSign.pickle'

if os.path.exists(fileName):
    print(f"file {fileName} already exists")
    while True:
        agInput = input("exit? y/n:")
        if agInput == 'y':
            sys.exit()
        elif agInput == 'n':
            break

for action in Actions:
    DataInAction = len(os.listdir(f'./ManualAlphabetData/{action}'))
    print(f'{action}:',DataInAction)

    data = os.listdir(os.path.join('../ManualAlphabetData', action))[0]
    dataDir = os.path.join('../ManualAlphabetData', f'{action}', f'{data}')

    print(dataDir)

    with open(dataDir, 'rb') as file:
       array = pickle.load(file)

    for dat in array['data']:
        landmarks.append(dat)

    for key in array['labels']:
        labels.append(labelMap[key])

x = np.array(landmarks)
y = to_categorical(labels).astype(int)

print(x[:2],y[:2])

print('x:',x.shape,'y:',y.shape)

# X_train_val, x_test, y_train_val, y_test = train_test_split(x,y, test_size=0.1, shuffle=True, random_state=42, stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1, shuffle=True, random_state=42, stratify=y)
#
# processedData = {'x_train':x_train,'x_val':x_val,
#                  'y_train':y_train,'y_val':y_val}
#
# f = open(fileName, 'wb')
# pickle.dump(processedData, f)
# f.close()