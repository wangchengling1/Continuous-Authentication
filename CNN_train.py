import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import Tripletdataset
import CNN_model
import Read_Acc
import Read_new
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np


# GPU运行
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print('device:', device)

# 数据处理
window_size = 50
data_path = 'F:\\2\\gait-dataset\\gait-dataset'
data, labels = Read_Acc.preprocess_data(data_path, window_size)
train_dataloader, test_dataloader = Read_Acc.create_dataloader(data, labels, batch_size=32, shuffle=True)

# CNN模型调用
model = CNN_model.CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
'''
# 打印每一层的输入大小
def print_layer_input_size(module, input):
    print(f"Layer: {module.__class__.__name__}, Input Size: {input[0].size()}")

# 注册钩子来打印每一层的输入大小
for module in model.modules():
    module.register_forward_pre_hook(print_layer_input_size)

# 打印每一层的参数数量
total_params = 0
for name, param in model.named_parameters():
    param_params = param.numel()
    print(f"Layer: {name}, Parameter Count: {param_params}")
    total_params += param_params

print(f"Total Parameters: {total_params}")
'''


# 使用数据加载器进行训练
num_epochs = 1000
losses = []  # 用于保存每一轮的损失值
for epoch in range(num_epochs):
    epoch_loss = 0.0  # 用于记录当前轮的总损失

    for i, (data, _) in enumerate(test_dataloader):

        # 清零梯度
        optimizer.zero_grad()
        if data.size(0) < 32:
            continue  # 如果数据批次小于32，跳过当前循环，继续下一次循环
        data = data.to(device)  # 将输入张量移动到设备（device）
        data = data.float()
        data = data.unsqueeze(3)
        data = data.reshape(32, 1, 50, 3)
        # 前向传播
        batch_output = model(data)
        # print('batch_output',batch_output.shape)
        # 计算损失（假设使用均方差损失函数）
        loss = nn.MSELoss()(batch_output, data)  # 将target替换为对应的目标张量
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累加当前批次的损失
        epoch_loss += loss.item()

    # 计算平均损失
    epoch_loss /= len(train_dataloader)

    # 打印当前轮的损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存当前轮的损失值
    losses.append(epoch_loss)

# 可视化损失值
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


# 加载和处理CTGAN产生的加速度数据
generated_data = np.load("predictions_1.npy")
print('generated_data', generated_data)
# 转换数据为张量
generated_data_tensor = torch.from_numpy(generated_data).float()
# 创建一个 TensorDataset 对象，该对象用于将数据和标签封装在一起：
dataset_generated = TensorDataset(generated_data_tensor)
# 创建一个 DataLoader 对象，用于批量加载数据
dataloader_generated = DataLoader(dataset_generated, batch_size=32, shuffle=False)
# 使用 dataloader 对象进行批量预测
num_epochs_new = 1000
losses_new = []  # 用于保存每一轮的损失值
for epoch in range(num_epochs_new):
    new_loss = 0.0  # 用于记录当前轮的总损失
    for i, batch_new in enumerate(dataloader_generated):
        # 将批次中的每个元素逐个转换为张量并放入列表中
        #batch_new_tensor = [torch.tensor(data).to(device).float().unsqueeze(0) for data in batch_new]
        # 清零梯度
        optimizer.zero_grad()
        batch_new_tensor = [torch.as_tensor(data).to(device).float().unsqueeze(0) for data in batch_new]
        batch_new = torch.cat(batch_new_tensor, dim=0)
        batch_new = batch_new.to(device)
        batch_new = batch_new.float()
        batch_new = batch_new.unsqueeze(3)
        # 添加条件语句，如果数据批次小于32，跳过当前循环
        if batch_new.size(0) < 32:
            continue
        batch_new = batch_new.reshape(32, 1, 50, 3)
        # 前向传播获取特征表示
        batch_new_output = model(batch_new)
        loss_new = nn.MSELoss()(batch_new_output, batch_new)  # 将target替换为对应的目标张量
        # 反向传播和优化
        loss_new.backward()
        optimizer.step()
        # 累加当前批次的损失
        new_loss += loss_new.item()
    # 计算平均损失
    new_loss /= len(dataloader_generated)
    # 打印当前轮的损失
    print(f"Epoch [{epoch + 1}/{num_epochs_new}], Loss: {new_loss:.4f}")

    # 保存当前轮的损失值
    losses_new.append(new_loss)

# 可视化第二个训练的损失值
plt.plot(losses_new)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (Second Training)')
plt.show()

#模型的保存
torch.save(model.state_dict(), 'model.pth')
