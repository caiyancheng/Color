import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import pandas as pd
from ReadData import Read_Data
from MLPs import create_mlp
# 设置随机种子以复现结果
torch.manual_seed(8)

inputs, targets = Read_Data()
dataset = TensorDataset(inputs, targets)

# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset = test_dataset = dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)

model = create_mlp([2, 10, 10, 1], dropout_rate=0).double()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 测试模型性能
model.eval()  # 将模型设置为评估模式
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader)}')

torch.save(model, 'MLP.pth')

