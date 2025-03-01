import pandas as pd
import cv2
import numpy as np
import os
import shutil


DATA_DIR = "data"

def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

class DataPrepare:
    def __init__(self, dataset_name="fer2013"):
        self.dataset_name = dataset_name
        self.csv_filepath = os.path.join(DATA_DIR, dataset_name + ".csv")
        self.dataset_dir = os.path.join(DATA_DIR, self.dataset_name)
        create_directory(self.dataset_dir)
        self.separate()
        self.csv2image()
        self.split(0.8)
        self.gen_data_label("train")
        self.gen_data_label("val")

    # the dataset csv file should at least contain "feature" and "label" columns.
    def separate(self):
        path = os.path.join(self.csv_filepath)

        # check
        if not os.path.exists(path):
            raise FileNotFoundError(f"Check existence for {path}. ")

        # read from csv
        df = pd.read_csv(path)
        label = df[['label']]
        feature = df[['feature']]

        # split into two csv files
        label.to_csv(os.path.join(self.dataset_dir, "label.csv"), index=False, header=False)
        feature.to_csv(os.path.join(self.dataset_dir, "data.csv"), index=False, header=False)

    def csv2image(self):
        data_path = os.path.join(self.dataset_dir, "data.csv")
        data = np.loadtxt(data_path)
        for i in range(data.shape[0]):
            face_array = data[i, :].reshape((48, 48))
            face_dir = os.path.join(self.dataset_dir, "face")
            # create if not exist
            create_directory(face_dir)
            cv2.imwrite(os.path.join(face_dir, f"{i}.jpg"), face_array)

    # split for train and validate
    def split(self, train_val_ratio=0.8):
        source_dir = os.path.join(self.dataset_dir, "face")
        train_dst = os.path.join(self.dataset_dir, "train")
        val_dst = os.path.join(self.dataset_dir, "val")
        # check if not exist
        create_directory(train_dst)
        create_directory(val_dst)

        file_list = os.listdir(source_dir)
        stop_num = train_val_ratio * len(file_list)
        for file in file_list:
            source_file = os.path.join(source_dir, file)
            num = int(file.split('.')[0])
            if num > stop_num:
                shutil.copy(source_file, val_dst)
            else:
                shutil.copy(source_file, train_dst)
        print("over!")


    def gen_data_label(self, split_type):
        label_path = os.path.join(self.dataset_dir, "label.csv")
        path = os.path.join(self.dataset_dir, split_type)
        # check if not exist
        create_directory(path)
        # 读取label文件
        df_label = pd.read_csv(label_path, header = None)
        # 查看该文件夹下所有文件
        files_dir = os.listdir(path)
        # 用于存放图片名
        path_list = []
        # 用于存放图片对应的label
        label_list = []
        # 遍历该文件夹下的所有文件
        for file_dir in files_dir:
            # 如果某文件是图片，则将其文件名以及对应的label取出，分别放入path_list和label_list这两个列表中
            if os.path.splitext(file_dir)[1] == ".jpg":
                path_list.append(file_dir)
                index = int(os.path.splitext(file_dir)[0])
                label_list.append(df_label.iat[index, 0])

        # 将两个列表写进dataset.csv文件
        path_s = pd.Series(path_list)
        label_s = pd.Series(label_list)
        df = pd.DataFrame()
        df['path'] = path_s
        df['label'] = label_s
        df.to_csv(os.path.join(path, "dataset.csv"), index=False, header=False)


def main():
    DataPrepare()

if __name__ == "__main__":
    main()