import torch
from torch import nn

def create_mlp(layer_sizes, dropout_rate=None):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:  # 不在输出层之前添加激活函数和Dropout
            layers.append(nn.ReLU())
            if dropout_rate is not None:  # 如果指定了dropout_rate，则添加Dropout层
                layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)

if __name__ == '__main__':
    # 创建一个带有dropout的MLP，dropout率为0.5
    mlp = create_mlp([3, 5, 4, 2], dropout_rate=0.5)
    print(mlp)
