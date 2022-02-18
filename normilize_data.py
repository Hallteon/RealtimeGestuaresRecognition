import os

data_content = []

with open("train_dataset.csv", "r") as data:
    lines = data.readlines()
    for line in lines:
        if len(line) > 2:
            data_content.append(line)

with open("train_dataset.csv", "w") as data:
    for line in data_content:
        data.write(line)





