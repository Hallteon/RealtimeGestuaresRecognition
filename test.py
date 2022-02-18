import numpy as np
import pandas as pd
from keras.models import load_model

REV_CLASS_MAP = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    4: "forward",
    5: "back"
}

def mapper(val):
    return REV_CLASS_MAP[val]

df_test = pd.read_csv('test_dataset.csv')
df_test = df_test.fillna(0)
df_test = df_test.drop(columns=['Unnamed: 0'], axis=1)

x_test = df_test.drop(['y'], axis=1)
y_test = df_test['y']

model = load_model("gestures_model.h5")

predicted_list = model.predict(x_test)

print(predicted_list)

# preds = model.predict(x_test)
# for pred in preds:
#     move_code = np.argmax(pred[0])
#     user_move_name = mapper(move_code)
#     print(user_move_name)
# print(len(preds))
