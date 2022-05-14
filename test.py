import pandas as pd
import keras
from tensorflow import keras

model = keras.models.load_model("gestures_model.h5")

df_test = pd.read_csv('train_dataset.csv')

x_test = df_test.drop(['y'], axis=1)
y_test = df_test['y']

x_test = x_test / 640

y_test_cat = keras.utils.to_categorical(y_test, 6)

result = model.evaluate(x_test, y_test_cat)

print(result)