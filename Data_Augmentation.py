import torch
from torch import nn
import torch.nn.functional as F
import VAE
import numpy as np
import matplotlib.pyplot as plt
import Read_Acc
import Read_new
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


# GPU运行
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print('device',device)

# 模型调用
epoch = 1000
input_dim = 150
hidden_dim = 64
latent_dim = 32
batch_size=32   
shuffle = True
Model = VAE.VAE(input_dim, hidden_dim, latent_dim).to(device)

beta = 0.40
def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta*KLD

# 准备原始数据集和标签
window_size = 50
data_path = 'F:\\2\\gait-dataset\\gait-dataset'
data, labels = Read_new.preprocess_data(data_path, window_size)
train_dataloader, test_dataloader = Read_new.create_dataloader(data,labels, batch_size=batch_size, shuffle=shuffle)


train_losses_VAE = []
test_losses_VAE = []


def train(num_epochs, model, optimizer, train_loader):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for j, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            if data.size(0) < 32:  # 如果数据长度小于32，跳过这个批次
                continue
            data = data.to(device)
            optimizer.zero_grad()
            data = data.float().to(device)
            data = data.reshape(32, 150)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, avg_train_loss))
        train_losses_VAE.append(avg_train_loss)

generated_data = []


def test(num_epochs, model, test_loader):
    model.eval()
    for epoch in range(num_epochs):
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if data.shape[0] < 32:  # 如果数据长度小于32，跳过这个批次
                    continue
                data = data.to(device)
                data = data.float().to(device)
                data = data.reshape(32, 150)
                recon_batch, mu, logvar = model(data)
                generated_data.append(recon_batch.detach().cpu().numpy())
                test_loss += loss_function(recon_batch, data, mu, logvar).item()


            test_loss /= len(test_loader.dataset)
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, test_loss))
            test_losses_VAE.append(test_loss)


    np.save("predictions_1.npy", generated_data)


optimizer = torch.optim.Adam(Model.parameters(),lr=0.0001)
train(epoch, Model, optimizer, train_dataloader)
test(epoch, Model, test_dataloader)

#print('generated_data',generated_data)
# 可视化损失函数损失值
plt.figure(figsize=(10, 6))
plt.plot(train_losses_VAE)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend()
plt.show()





