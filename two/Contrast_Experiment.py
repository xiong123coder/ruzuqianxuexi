import torch
import torch.nn as nn
import torch.optim as optim

#数据
x=torch.tensor([
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
    [19.0], 
    [12.0], 
    [5.0], 
    [4.0], 
    [15.0], 
    [44.0], 
    [97.0], 
    [180.0]
])

#三个对照模型
models=[
    nn.Sequential(
        nn.Linear(1,64),nn.ReLU(),
        nn.Linear(64,64),nn.ReLU(),
        nn.Linear(64,1)
    ),
    nn.Sequential(
        nn.Linear(1,64),
        nn.Linear(64,64),
        nn.Linear(64,1)
    ),
    nn.Sequential(
        nn.Linear(1,64),
        nn.ReLU(),
        nn.Linear(64,1)
    )
]
criterion=nn.MSELoss()
print("开始训练\n")
for i,model in enumerate(models):
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(2000):
        prediction=model(x)
        loss=criterion(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test=torch.tensor([[5.0]])
    result=model(test)
    print(f"模型{i+1}的训练结果\n")
    print(f"最终训练loss:{loss.item():.4f}")
    print(f"对[[5.0]]的预测值：{result.item():.4f}")  
    print("-"*30)      
"""
1. 激活函数（ReLU）是灵魂
模型 2（无激活函数） 虽然有三层，但最终的 Loss 高达 1255.5，预测值（57.0）严重偏离真实值。
原理解析：没有激活函数，多个线性层 Linear 堆叠在数学上等同于单层线性回归。
无论网络多深，它只能画出一条“直线”，根本无法拟合你数据中的曲线弯曲。
2. 网络的“深度”与“表达能力”
模型 1（双层ReLU，2000轮）：Loss 降到了 12.2869，预测值 15.0301 极其接近真实值 15.0。
模型 3（单层ReLU，2000轮）：Loss 却高达 1182.9672。
原理解析：复杂的非线性函数需要足够的网络深度和参数量来逼近。
模型 3 虽然有激活函数，但由于只有一层隐藏层，其拟合如此陡峭的曲线时显得“力不从心”；
而模型 1 通过两层 ReLU 的组合，具备了更强的函数逼近能力。
"""