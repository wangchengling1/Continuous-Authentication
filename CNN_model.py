import torch
import torch.nn as nn
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print('device',device)

# 卷积层conv2d输入参数 (batch size, channels(通道数，使用传感器数量), height(窗口大小), width(特征数量))
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,2), stride=(1,1), padding = 0)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.35)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,2), stride=(1,1), padding = 0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,1), stride=(1,1), padding = 0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(2, 1), stride=(1, 1), padding=0)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))  # 全局最大池化层

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.global_maxpool(x)
        return x