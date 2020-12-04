import librosa
import tensorflow as tf
#import keras
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.model_selection import train_test_split

path_file_1 = './dataset/ff1010bird/wav/'
path_file_2 = './dataset/warblrb10k_public/wav/'

#Audio data (X)
data1 = []
data2 = []

#Load labels DataFrame
label1 = pd.read_csv('./dataset/ff1010bird_metadata_2018.csv')
label2 = pd.read_csv('./dataset/warblrb10k_public_metadata_2018.csv')

#Initialise label array(y)
y1 = []
y2 = []

for subdir, dirs, files in os.walk(path_file_1):
    for filename in files:
        y, sr = librosa.load(path_file_1 + filename)
        y_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        y_mel_log = np.log(y_spec)
        data1.append(y_mel_log)
        has_bird = label1[label1['itemid'] == int(os.path.splitext(filename)[0])]
        y1.append(has_bird)
        print('Processed ' + str(len(data1)+len(data2)) + ' audio files')

for subdir, dirs, files in os.walk(path_file_2):
    for filename in files:
        y, sr = librosa.load(path_file_2 + filename)
        y_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        y_mel_log = np.log(y_spec)
        data2.append(y_mel_log)
        has_bird = label2[label2['itemid'] == os.path.splitext(filename)[0]]
        y2.append(has_bird)
        print('Processed ' + str(len(data1)+len(data2)) + ' audio files')
        
data1 = np.array(data1)
data2 = np.array(data2)

y1 = np.array([y['hasbird'].values[0] for y in y1])
y2 = np.array([y['hasbird'].values[0] for y in y2])\

X_train, X_test, y_train, y_test = train_test_split(data1, y1, test_size=0.2)
        
from tensorflow.keras.layers import Conv2D, Conv1D, GRU, Dense, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def bird_model1():
   input1 = Input(shape=(data1.shape[1], data1.shape[2],))
#   conv1 = Conv2D()
   gru1 = GRU(64, return_sequences= True)(input1)
   gru2 = GRU(64, return_sequences=False)(gru1)
#   flatten1 = Flatten()(input1)
   dense1 = Dense(10, activation='relu')(gru2)
   sigmoid = Dense(1, activation='sigmoid')(dense1)
   
   model = Model(inputs=input1, outputs=sigmoid)
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

model = bird_model1()
model.summary()

callbacks = [
    EarlyStopping(patience=2),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001),
]

history = model.fit(X_train, y_train, batch_size=128, epochs=150, callbacks=callbacks, validation_data=(X_test, y_test))



                                                                    