import matplotlib.pyplot as plt
import torch
import pandas as pd


data = pd.read_csv('data.csv',header=None)
x_train = data[0].values.reshape(-1, 1)
y_train = data[1].values.reshape(-1, 1)

#转换为tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

#定义参数w和b
w =torch.randn(1, requires_grad=True)
b =torch.zeros(1, requires_grad=True)  #使用0初始化

optimizer = torch.optim.SGD([w, b], lr=0.0001)

# 构建线性回归模型
def linear_model(x):
    return x * w + b

# 绘制实际散点图
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')

# 计算损失
def get_loss(y_pred, y):
    return torch.mean((y_pred - y_train) ** 2)

# 训练模型
def run(epochs):
    for e in range(epochs):  # 进行 epochs 次更新
        optimizer.zero_grad()
        y_pred = linear_model(x_train)
        loss = get_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {e}, Loss: {loss.item()}")

if __name__ == '__main__':
    run(10)
    y_pred = linear_model(x_train)
    plt.plot(x_train.data.numpy(), y_pred.data.numpy(), 'go', label='estimated')
    plt.legend()
    plt.show()
