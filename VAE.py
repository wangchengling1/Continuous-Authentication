import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dropout_rate =0.35
# 定义变分自编码器（VAE）模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            #nn.BatchNorm1d(hidden_dim),  # 添加批量归一化层
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout层
            nn.Linear(128, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),  # 添加批量归一化层
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout层
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),  # 添加批量归一化层
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout层
            nn.Linear(hidden_dim,128),
            # nn.BatchNorm1d(hidden_dim),  # 添加批量归一化层
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout层
            nn.Linear(128, input_dim),
            nn.LeakyReLU()
        )
        # 可训练的标准差
        self.std = nn.Parameter(torch.ones(1, latent_dim))

    def reparameterize(self, mu):
        std = self.std.exp()  # 使用可训练的标准差
        eps = torch.randn_like(mu)  # 从标准正态分布中采样噪声
        z = mu + eps * std  # 重参数化技巧
        return z

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)  # 将输出分割为均值和方差
        z = self.reparameterize(mu)  # 重参数化

        # 解码
        decoded = self.decoder(z)
        return decoded, mu, logvar
