import torch
import torch.nn as nn
import torch.optim as optim

#数据
# torch.tensor是专门用于神经网络的数据结构，
# 自动求导（神经网络训练核心），GPU加速矩阵运算，参与反向传播，如果用list只能拼接，不是数学加法
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0]
])

y = torch.tensor([
    [10.0],
    [20.0],
    [90.0],
    [15.0],
    [95.0],
    [18.0],
    [97.0],
    [20.0]
])
#创建一个由多层线性层和非线性激活函数组成的神经网络模型
model=nn.Sequential(
    nn.Linear(1,64),
    nn.ReLU(),   
    nn.Linear(64,64),#64个特征再组合成64个更复杂的特征，
    #深层学习复杂特征，由64个高级特征输出一个预测值
    nn.ReLU(),
    nn.Linear(64,1)
)

#损失函数 这个是均方误差，而不是绝对误差
criterion=nn.MSELoss()
#训练前提前准备参数更新的规则
optimizer=optim.Adam(
    #优化模型的参数 例如linear的weight和bias，优化器会自动更新
    model.parameters(),
    #learning rate每次参数更新的步子有多大
    lr=0.001
)
#训练循环 

for epoch in range(2000):
    #前向传播 计算预测值
    prediction=model(x)
    #计算损失
    loss=criterion(prediction,y)
    #梯度清零 梯度就是参数应该往大调，还是往小调，
    # 清零：在本轮训练开始前，把上一轮的梯度清空
    #pytorch默认累加梯度，所以需要清零，
    #否则每次更新都不是单下训练的梯度，而上加上了之前的梯度
    optimizer.zero_grad()
    #计算所有参数的梯度
    loss.backward()
    #更新参数
    optimizer.step()
#测试 输入二维[[5.0]]
test=torch.tensor([[5.0]])
result=model(test)
print(result)
#记录我的错误
#我创建损失函数时criterion=nn.MSELoss()忘记加括号，所以没有识别出创建函数，
# 训练用到的函数都有括号，只有预测值和计算损失括号里面有实参
# 一定是先创建损失函数，然后才能计算损失，否则会报错
#大小写敏感 module 'torch.nn' has no attribute 'sequential' 应为Sequential
#没有加nn.Linear(64,64)和nn.ReLU()时,tensor([[107.4971]], grad_fn=<AddmmBackward0>)
#grad_fn代表tensor还可以继续求导

