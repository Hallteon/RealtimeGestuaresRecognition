import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Flatten, Input

df_train = pd.read_csv('train_dataset.csv')
df_train = df_train.drop(columns=['Unnamed: 0'], axis=1)
df_train = df_train.fillna(0)

df_test = pd.read_csv('test_dataset.csv')
df_test = df_test.drop(columns=['Unnamed: 0'], axis=1)
df_test = df_test.fillna(0)

x_train = df_train.drop(['y'], axis=1)
y_train = df_train['y']

x_test = df_test.drop(['y'], axis=1)
y_test = df_test['y']

x_train = x_train / 310
x_test = x_test / 310

y_train_cat = keras.utils.to_categorical(y_train, 6)

y_test_cat = keras.utils.to_categorical(y_test, 6)

model = keras.models.Sequential([Dense(32, input_shape=(42,), activation='relu'),
                                Dense(6, activation='softmax')])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=16, epochs=7, validation_split=0.2)

model.save("gestures_model.h5")
