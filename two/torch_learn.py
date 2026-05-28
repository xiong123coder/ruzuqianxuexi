import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 非交互后端，纯保存图片，不弹窗，兼容 Windows 通义灵码环境
import os
import sys

import matplotlib.pyplot as plt

print("Current Python interpreter:")
print(sys.executable)

# ===============================
# 训练数据
# ===============================
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
    [30.0],
    [55.0],
    [70.0],
    [80.0],
    [88.0],
    [93.0],
    [97.0],
    [99.0]
])

# ===============================
# 神经网络模型
# ===============================
model = nn.Sequential(

    # 输入层 -> 隐藏层1
    nn.Linear(1, 64),

    # 激活函数（非线性）
    nn.ReLU(),

    # 隐藏层1 -> 隐藏层2
    nn.Linear(64, 64),

    nn.ReLU(),

    # 输出层
    nn.Linear(64, 1)
)

# ===============================
# Xavier 权重初始化
# ===============================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# ===============================
# 损失函数
# ===============================
criterion = nn.MSELoss()

# ===============================
# 优化器
# lr 调小，更稳定
# ===============================
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ===============================
# 开始训练
# ===============================
print("\n========== Training Started ==========\n")

for epoch in range(10000):

    # 前向传播
    prediction = model(x)

    # 计算损失
    loss = criterion(prediction, y)

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印训练过程
    if epoch % 1000 == 0:
        print(f"epoch: {epoch}    loss: {loss.item():.4f}")

# ===============================
# 测试预测
# ===============================
print("\n========== Prediction Results ==========\n")

test_hours = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0]
])

# 关闭梯度计算（推理阶段）
with torch.no_grad():

    predictions = model(test_hours)

    # 限制在 0~100
    predictions = torch.clamp(predictions, 0, 100)

# 打印结果
for i in range(len(test_hours)):

    hour = int(test_hours[i].item())

    score = predictions[i].item()

    print(f"Study {hour}h -> Predicted Score: {score:.2f}")
print("Entering plotting phase...")

# ===============================
# 生成拟合曲线
# ===============================
x_line = torch.linspace(0, 9, 200).view(-1, 1)

with torch.no_grad():

    y_line = model(x_line)

    y_line = torch.clamp(y_line, 0, 100)

# ===============================
# Plotting
# ===============================
print("Creating figure...")
plt.figure(figsize=(8, 5))

# Raw data scatter
print("Adding scatter plot...")
plt.scatter(
    x.numpy(),
    y.numpy(),
    s=80,
    label="Raw Data"
)

# NN fitted curve
print("Adding fitting curve...")
plt.plot(
    x_line.numpy(),
    y_line.numpy(),
    linewidth=2,
    label="NN Fitting Curve"
)

# Chart info (English only, avoid Chinese font rendering)
print("Configuring chart labels...")
plt.title("Study Time vs Score (NN Non-linear Fit)")

plt.xlabel("Study Hours")

plt.ylabel("Predicted Score")

plt.ylim(0, 105)

plt.grid(True, alpha=0.3)

print("Adding legend...")
plt.legend()

# ===============================
# 自动创建保存目录
# ===============================
output_dir = r"E:\ml_output"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "fit_curve.png"
)

# ===============================
# 保存图片
# ===============================
print("savefig starting...")
try:

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight'
    )

    print(f"\n[OK] Image saved to: {output_path}")

except Exception as e:

    print("\n[ERROR] Failed to save image!")

    print("Error details:")

    print(e)

# ===============================
# Verify backend
# ===============================
print(f"\nCurrent matplotlib backend: {matplotlib.get_backend()} (non-interactive mode)")
print(f"Output image: {output_path}")

