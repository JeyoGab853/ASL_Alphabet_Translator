import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import keras.callbacks
import keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization,Input, Flatten
from tensorflow.keras.callbacks import TensorBoard

with open('../ProcessedAlphData/fingerSignStandard2.pickle', 'rb') as file:
    organizedData = pickle.load(file)

actions = np.array(os.listdir('../AlphabetImageData'))

X_train,X_val,y_train,y_val = [data for data in organizedData.values()] #gets training and validation data

model = Sequential([ 
    Input(shape=(42,)), #The hand has 21 landmarks in total with coordinates of x and y (therefore if flattened, the shape is 42)
    LayerNormalization(axis=1), #to normalize data
    Dropout(0.01), #Dropout to prevent overfitting
    Dense(48, activation='elu'), #elu for more stable training
    Dropout(0.05),
    Dense(64, activation='elu'),
    Dropout(0.1),
    Dense(56, activation='elu'),

    Dense(actions.shape[0], activation='softmax', name='Softmax')
])

model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['categorical_accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath='../Log/CheckPointFiles2.keras',
    monitor='val_loss',
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True
)

#callback that stops after val loss if below a certain threshold (0.01)
class StopAtLossThreshold(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get('val_loss')

        if val_loss is not None and val_loss < 0.01:
            print(f"\nVal_Loss reached {val_loss:.4f}. Terminating training mid-epoch.")
            self.model.stop_training = True

history = model.fit(X_train, y_train,validation_data=(X_val,y_val), epochs=2000,callbacks=[StopAtLossThreshold(),checkpoint_cb])

model.save('./Model/AlphStandZ3.h5')
