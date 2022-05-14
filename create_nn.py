import pandas as pd
from tensorflow import keras
from keras.layers import Dense

df_train = pd.read_csv('train_dataset.csv')

x_train = df_train.drop(['y'], axis=1)
y_train = df_train['y']

x_train = x_train / 640

y_train_cat = keras.utils.to_categorical(y_train, 8)

model = keras.models.Sequential([Dense(256, input_shape=(42,), activation='relu'),
                                Dense(524, activation='relu'),
                                Dense(1048, activation='relu'),
                                Dense(8, activation='softmax')])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.3)

model.save("gestures_model.h5")
