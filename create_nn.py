import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Dropout

df_train = pd.read_csv('train_dataset.csv')

x_train = df_train.drop(['y'], axis=1)
y_train = df_train['y']

x_train = x_train / 310

y_train_cat = keras.utils.to_categorical(y_train, 6)

model = keras.models.Sequential([Dense(85, input_shape=(42,), activation='relu'),
                                Dense(170, activation='relu'),
                                Dense(340, activation='relu'),
                                Dense(680, activation='relu'),
                                Dense(1340, activation='relu'),
                                Dense(6, activation='softmax')])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=16, epochs=10, validation_split=0.2)

model.save("gestures_model.h5")
