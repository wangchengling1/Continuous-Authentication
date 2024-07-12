from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
import Read_Acc
import CNN_model

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

batch_size=32
shuffle = True
learning_rate = 0.000001
num_epochs = 800


# 准备原始数据集和标签
data_path = 'C:\\Users\\wcl\\Desktop\\CNN\\dataset_1'
data_test, labels_test = Read_Acc.preprocess_data(data_path)
test_dataloader,evaluate_dataloader = Read_Acc.create_dataloader(data_test, labels_test, batch_size=batch_size, shuffle=shuffle)


# # 加载和处理CTGAN产生的加速度数据
# predictions_2 = np.load("predictions_2.npy")
# # 找到最大能被3整除的元素数量
# max_elements_divisible_by_3 = len(predictions_2) - (len(predictions_2) % 3)
# # 只保留能被3整除的部分
# predictions_2_divisible_by_3 = predictions_2[:max_elements_divisible_by_3]
# predictions_2 = predictions_2_divisible_by_3.reshape(-1, 3)
# print(predictions_2.shape)
# #print(predictions_2.shape)
# print('predictions_2', predictions_2)
# label_array_2 = np.ones(15744)

predictions_1= np.load("predictions_1.npy")
predictions_1 = predictions_1.reshape(-1, 3)
print(predictions_1.shape)
#print(predictions_1.shape)
print('predictions_1', predictions_1)
# predictions = np.concatenate((predictions_2, predictions_1), axis=0)
# #print(predictions.shape)
# label_set = np.concatenate((label_array_2, label_array_1))


model=CNN_model.CNN().to(device)


# 创建 SVM 分类器实例，这里使用默认的RBF核
oc_svm = OneClassSVM(nu=0.0011, kernel='rbf', gamma=0.001)


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

# 假设 'train_dataloader' 是一个包含训练数据和标签的 DataLoader
data_list = []
label_list = []
for k,(batch_data, batch_label) in enumerate(test_dataloader):
    # 训练 SVM 分类器
    data_list.append(batch_data)
    label_list.append(batch_label)
    oc_svm.fit(batch_data)


oc_svm.fit(predictions_1)
# y_pred_2 = oc_svm.predict(predictions_2)
# print('y_pred_2', y_pred_2)

# 使用模型进行预测
# 初始化列表
all_test_labels = []
all_y_pred = []
for i, (test_data, test_label) in enumerate(evaluate_dataloader):
    y_pred = oc_svm.predict(test_data)
    print('y_pred',y_pred)
    print('y_pred', len(y_pred))
    all_y_pred.extend(y_pred)
    all_test_labels.extend(test_label)


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
print("分类结果: \n", y_pred)
print("混淆矩阵: \n", cm)
print("误报率 (FAR): {:.2f}%".format(FAR * 100))
print("误拒率 (FRR): {:.2f}%".format(FRR * 100))
print("准确率 (Accuracy): {:.2f}%".format(accuracy * 100))
# 打印 F1 分数
print("F1 分数: {:.2f}%".format(F1 * 100))

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