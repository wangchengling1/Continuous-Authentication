import glob
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
class MyDataset(Dataset):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __getitem__(self, index):
        accelerations = self.data[index]
        label = self.labels[index]
        return accelerations, label

    def __len__(self):
        return len(self.data)


# 创建一个带通滤波器
def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 应用带通滤波器
def bandpass_filter(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def preprocess_data(data_path, window_size):
    data = []
    labels = []
    file1 = os.path.join(data_path, 'w002day1-raw.csv')
    file2 = os.path.join(data_path, 'w005day1-raw.csv')
    file3 = os.path.join(data_path, 'acc_jumping_chest.csv')
    files = [file2]

    for i, file in enumerate(files):
        # 对于 'w0014day2-raw'文件，标签为1，对于 'chest2.csv'文件，标签为0
        label = 1 if 'w005day1-raw' in file else -1
        print(f"Processing file: {file} with label {label}")  # 打印文件名和对应的标签

        df = pd.read_csv(file, delimiter='\t')
        acc_data = df[['accel_x', 'accel_y', 'accel_z']].values
        # 应用带通滤波器
        acc_data = bandpass_filter(acc_data, 0.5, 12, 50, order=8)

        # 对整个数据集进行归一化
        max_acc_x = np.max(acc_data[:, 0])
        min_acc_x = np.min(acc_data[:, 0])
        norm_acc_x = (acc_data[:, 0] - min_acc_x) / (max_acc_x - min_acc_x)

        max_acc_y = np.max(acc_data[:, 1])
        min_acc_y = np.min(acc_data[:, 1])
        norm_acc_y = (acc_data[:, 1] - min_acc_y) / (max_acc_y - min_acc_y)

        max_acc_z = np.max(acc_data[:, 2])
        min_acc_z = np.min(acc_data[:, 2])
        norm_acc_z = (acc_data[:, 2] - min_acc_z) / (max_acc_z - min_acc_z)

        normalized_data = np.column_stack((norm_acc_x, norm_acc_y, norm_acc_z))
        # 对每一个窗口进行划分，并为每个窗口添加相应的标签
        for j in range(0, len(normalized_data) - window_size + 1, window_size):
            window_data = normalized_data[j:j + window_size]
            data.append(window_data)
            labels.append(label)
    return data, labels

# def create_dataloader(data, labels, batch_size=128, shuffle=True):
#     dataset = MyDataset(data, labels, batch_size)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=None)
#     return dataloader


def create_dataloader(data, labels, batch_size=32, shuffle=True, train_ratio=0.5):
    dataset = MyDataset(data, labels, batch_size)

    # 计算划分训练集和测试集的样本数量
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # 划分训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建训练集和测试集的数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=None)

    return train_dataloader, test_dataloader

# data_path = 'C:\\Users\wcl\\Desktop\MTGAN\\dataset'
# # data, labels = preprocess_data(data_path)
# # test_Label = labels[10,:]
# # dataloader = create_dataloader(data, labels, batch_size=batch_size, shuffle=shuffle)
# # print(dataloader)

