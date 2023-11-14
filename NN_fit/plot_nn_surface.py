import torch
import numpy as np
import matplotlib.pyplot as plt
from ReadData import Read_Data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

inputs, targets = Read_Data()
model = torch.load('MLP.pth')
model.eval()

# 创建一个输入网格
x = torch.linspace(inputs[:,0].min(), inputs[:,0].max(), 100)
y = torch.linspace(inputs[:,1].min(), inputs[:,1].max(), 100)
xx, yy = torch.meshgrid(x, y)
grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)

# 预测输出
with torch.no_grad():
    zz = model(grid.double()).reshape(xx.shape)

# 可视化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制神经网络的预测结果作为表面，设置透明度为半透明
surf = ax.plot_surface(xx.numpy(), yy.numpy(), zz.numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.5)

# 绘制数据点
ax.scatter(inputs[:,0].numpy(), inputs[:,1].numpy(), targets.numpy(), color='r', label='Data points')

# 设置标签和标题
ax.set_xlabel('E_mean')
ax.set_ylabel('E_std')
ax.set_zlabel('Score_std')

# 显示图例
ax.legend()

plt.show()