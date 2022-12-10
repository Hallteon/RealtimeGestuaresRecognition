import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Dropout

df_train = pd.read_csv('train_dataset.csv')

x_train = df_train.drop(['y'], axis=1)
y_train = df_train['y']

x_train = x_train / 640

y_train_cat = keras.utils.to_categorical(y_train, 8)

model = keras.models.Sequential([Dense(32, input_shape=(42,), activation='relu'),
                                Dense(54, activation='relu'),
                                Dense(86, activation='relu'),
                                Dense(156, activation='relu'),
                                Dropout(rate=0.8),
                                Dense(8, activation='softmax')])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=400, epochs=12, validation_split=0.2)

df_test = pd.read_csv('train_dataset.csv')

x_test = df_test.drop(['y'], axis=1)
y_test = df_test['y']

x_test = x_test / 640

y_test_cat = keras.utils.to_categorical(y_test, 8)

result = model.evaluate(x_test, y_test_cat)

print(result)

model.save("gestures_model.h5")
