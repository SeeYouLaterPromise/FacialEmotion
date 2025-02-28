import pandas as pd
import cv2
import numpy as np
import os
import shutil

def separate():
    path = "train.csv"

    df = pd.read_csv(path)

    label = df[['label']]
    feature = df[['feature']]

    label.to_csv("label.csv", index=False, header=False)
    feature.to_csv("data.csv", index=False, header=False)

def toImage():
    path = "face"
    data = np.loadtxt("data.csv")
    print(data.shape)
    for i in range(data.shape[0]):
        print(i)
        face_array = data[i, :].reshape((48, 48))
        cv2.imwrite(path + "//" + f"{i}.jpg", face_array)

def transport():
    path = "face"
    for file in os.listdir(path):
        source_file = os.path.join(path, file)
        num = int(file.split('.')[0])
        if num > 23999:
            shutil.move(source_file, "val")
        else:
            shutil.move(source_file, "train")
    print("over!")


transport()
