import random
import sys

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time

import os
import numpy as np
import pickle

Actions = os.listdir('./TestDir')
labelMap = {label:num for num,label in enumerate(Actions)}

sequences, labels = [],[]

fileName = 'ProcessedData/data10_large_2.pickle'

if os.path.exists(fileName):
    print(f"file {fileName} already exists")
    while True:
        agInput = input("exit? y/n:")
        if agInput == 'y':
            sys.exit()
        elif agInput == 'n':
            break

for action in Actions:
    SequenceInAction = len(os.listdir(f'./DataAction/{action}'))
    print(f'{action}:',SequenceInAction)

    for sequence in os.listdir(os.path.join('./DataAction',action)):
        window = []
        for frame_num in range(30):
            res = np.load(os.path.join('./DataAction',action,sequence,"{}.npy".format(frame_num)))
            window.append(res)

        sequences.append(window)
        labels.append(labelMap[action])


x = np.array(sequences)
y = to_categorical(labels).astype(int)

print('x:',x.shape,'y:',y.shape)

# X_train_val, x_test, y_train_val, y_test = train_test_split(x,y, test_size=0.1, shuffle=True, random_state=42, stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(X_train_val,y_train_val, test_size=0.1, shuffle=True, random_state=42, stratify=y_train_val)
#
# processedData = {'x_train':x_train,'x_val':x_val,'x_test':x_test,
#                  'y_train':y_train,'y_val':y_val,'y_test':y_test}
#
#
#
# f = open(fileName, 'wb')
# pickle.dump(processedData, f)
# f.close()

# x,y = shuffle(x,y)

# f = open('ProcessedData/largerData.pickle', 'wb')
# pickle.dump({'X_train':x, 'Y_train':y}, f)
# f.close()

