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

X_train,X_val,y_train,y_val = [data for data in organizedData.values()]

model = Sequential([
    Input(shape=(42,)),
    LayerNormalization(axis=1),
    Dropout(0.01),
    Dense(48, activation='elu'),
    Dropout(0.05),
    Dense(64, activation='elu'),
    Dropout(0.1),
    Dense(56, activation='elu'),

    Dense(actions.shape[0], activation='softmax', name='Softmax')
])
costumAdam = optimizers.Adam(learning_rate=0.001)
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

class StopAtLossThreshold(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get('val_loss')

        if val_loss is not None and val_loss < 0.01:
            print(f"\nVal_Loss reached {val_loss:.4f}. Terminating training mid-epoch.")
            self.model.stop_training = True

history = model.fit(X_train, y_train,validation_data=(X_val,y_val), epochs=2000,callbacks=[StopAtLossThreshold(),checkpoint_cb])

plt.figure(figsize=(12, 4))
# First subplot: Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  # Plot training accuracy over epochs
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')  # Plot validation accuracy over epochs
plt.title('Model Accuracy')  # Title for the accuracy plot
plt.xlabel('Epoch')  # Label for the x-axis (epochs)
plt.ylabel('Accuracy')  # Label for the y-axis (accuracy)
plt.legend()  # Show the legend to distinguish between training and validation curves

# Second subplot: Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')  # Plot training loss over epochs
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss over epochs
plt.title('Model Loss')  # Title for the loss plot
plt.xlabel('Epoch')  # Label for the x-axis (epochs)
plt.ylabel('Loss')  # Label for the y-axis (loss)
plt.legend()  # Show the legend to distinguish between training and validation curves

# Adjust the layout for a cleaner view
plt.tight_layout()

# Display the plots showing the model's accuracy and loss over time
# plt.show()

model.save('./Model/AlphStandZ3.h5')
