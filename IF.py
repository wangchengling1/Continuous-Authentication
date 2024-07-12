import torch
import torch.nn as nn
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
import Tripletdataset
import Read_Acc
import CNN_model
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import Read_new
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
import Read
import new
import torch.nn.functional as F
# 使用PCA进行降维
pca = PCA(n_components=20)  # 设定降维后的特征数量为20

# 使用孤立森林（Isolation Forest）分类器
iforest = IsolationForest(contamination=0.05)

#混淆电路的定义
def confusion_matrix_new(true_labels, pred_labels):
    # 确保输入是 tensor
    true_labels = torch.tensor(true_labels)
    pred_labels = torch.tensor(pred_labels)

    # 计算 TP, FN, FP, TN
    TP = torch.sum((pred_labels == 1) & (true_labels == 1)).item()
    FN = torch.sum((pred_labels == -1) & (true_labels == 1)).item()
    FP = torch.sum((pred_labels == 1) & (true_labels == -1)).item()
    TN = torch.sum((pred_labels == -1) & (true_labels == -1)).item()

    # 返回一个 numpy 数组
    return np.array([[TN, FP], [FN, TP]])

# GPU运行
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print('device',device)

# 数据处理
window_size = 50
data_path = 'F:\\2\\gait-dataset\\gait-dataset'
data, labels = Read.preprocess_data(data_path, window_size)
train_loader, test_loader = Read.create_dataloader(data, labels, batch_size=32, shuffle=True)

data_path = 'F:\\2\\gait-dataset\\gait-dataset'
data, labels = new.preprocess_data(data_path, window_size)
train_loader1, test_loader1 = new.create_dataloader(data, labels, batch_size=32, shuffle=True)


# CNN模型调用
model = CNN_model.CNN().to(device)
model.load_state_dict(torch.load('model.pth'))
# 将模型设置为评估模式
model.eval()

# 存储测试样本的特征表示和标签
features = []
labels = []

# 对测试集进行推断
with torch.no_grad():
    for i, (data, _) in enumerate(train_loader):

        if data.size(0) < 32:
            continue  # 如果数据批次小于32，跳过当前循环，继续下一次循环
        data = data.to(device)
        data = data.float()
        data = data.unsqueeze(3)
        data = data.reshape(32, 1, 50, 3)
        # 前向传播获取特征表示
        batch_data_output = model(data)
        batch_data_output = batch_data_output.to('cpu')  # 将张量复制到主机内存
        batch_data_output_np = batch_data_output.detach().numpy()  # 将复制后的张量转换为 NumPy 数组
        print('batch_data_output_np', batch_data_output_np.shape)
        anchor_output_np = batch_data_output_np.reshape(32, 256)
        anchor_output_pca = pca.fit_transform(anchor_output_np)
        iforest.fit(anchor_output_pca)




all_test_labels = []
all_y_pred = []
# 对新样本进行预测
for k,(batch_data, batch_label) in enumerate(test_loader1):

    # 前向传播获取特征表示
    if batch_data.size(0) < 32:
        continue  # 如果数据批次小于32，跳过当前循环，继续下一次循环
    batch_data = batch_data.to(device)
    noise = torch.randn_like(batch_data) * 0.05
    noisy_batch_data = batch_data + noise
    noisy_batch_data = torch.clamp(noisy_batch_data, 0, 1)
    noisy_batch_data = noisy_batch_data.float()
    noisy_batch_data = noisy_batch_data.unsqueeze(3)
    noisy_batch_data = noisy_batch_data.reshape(32, 1, 50, 3)
    batch_data_output = model(noisy_batch_data)
    batch_data_output = batch_data_output.to('cpu')  # 将张量复制到主机内存
    batch_data_output_np = batch_data_output.detach().numpy()  # 将复制后的张量转换为 NumPy 数组
    print('batch_data_output_np', batch_data_output_np.shape)
    anchor_output_np = batch_data_output_np.reshape(32, 256)
    anchor_output_pca = pca.transform(anchor_output_np)
    predictions = iforest.predict(anchor_output_pca)
    all_y_pred.extend(predictions)
    all_test_labels.extend(batch_label)

# 计算混淆矩阵
print('all_test_labels', all_test_labels, 'all_y_pred', all_y_pred)
cm = confusion_matrix_new(all_test_labels, all_y_pred)
accuracy = accuracy_score(all_test_labels, all_y_pred)
TN, FP, FN, TP = cm.ravel()

# 计算 FRR 和 FAR
FRR = FN / (FN + TP)
FAR = FP / (FP + TN)
# 计算 F1 分数
F1 = 2 * TP / (2 * TP + FP + FN)

# 打印分类结果，混淆矩阵，FRR 和 FAR
print("分类结果: \n", predictions)
print("混淆矩阵: \n", cm)
print("误报率 (FAR): {:.2f}%".format(FAR * 100))
print("误拒率 (FRR): {:.2f}%".format(FRR * 100))
print("准确率 (Accuracy): {:.2f}%".format(accuracy * 100))
# 打印 F1 分数
print("F1 分数: {:.2f}%".format(F1 * 100))
EER=(FRR+FAR)/2
print("EER : {:.2f}%".format(EER * 100))


# 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# 打印更详细的性能报告
report = classification_report(all_test_labels, all_y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("分类性能报告: \n", report_df)

# 在报告中加入准确率和 F1 分数
report_df.loc['accuracy', :] = [accuracy, accuracy, accuracy, len(all_test_labels)]
report_df.loc['f1 score', :] = [2 * TP / (2 * TP + FP + FN)] * 3 + [len(all_test_labels)]

# 绘制报告的热图
plt.figure(figsize=(10, 7))
sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f')  # 排除最后一行的支持（support）数值
plt.title('Classification Report')
plt.show()