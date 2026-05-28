import torch
import torch.nn as nn
import torch.optim as optim
import sys

# ===== 修复 1：在 import pyplot 之前强制指定交互式后端 =====
# 这是 Windows 下 matplotlib 窗口不弹出的最常见原因
import matplotlib
matplotlib.use('TkAgg')   # 如果报错缺少 tkinter，可改为 'Qt5Agg'（需 pip install pyqt5）
import matplotlib.pyplot as plt

print("当前正在使用的python路径:")
print(sys.executable)
print(f"matplotlib 后端: {matplotlib.get_backend()}")

x = torch.tensor([
    [1.0], [2.0], [3.0], [4.0],
    [5.0], [6.0], [7.0], [8.0]
])
y = torch.tensor([
    [30.0], [55.0], [70.0], [80.0],
    [88.0], [93.0], [97.0], [99.0]
])

# ⚠️ 注意：ReLU 被注释掉后，三层 Linear 堆叠退化为纯线性 y=Wx+b，只能拟合直线
model = nn.Sequential(
    nn.Linear(1, 64),
#   nn.ReLU(),
    nn.Linear(64, 64),
#   nn.ReLU(),
    nn.Linear(64, 1)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10000):
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"epoch:{epoch}  loss:{loss.item():.4f}")

# 验证多个学习时间
print("\n========== 各学习时间预测分数 ==========")
test_hours = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
predictions = model(test_hours).detach()
predictions_clamped = torch.clamp(predictions, 0, 100)

for i in range(len(test_hours)):
    h = int(test_hours[i].item())
    print(f"  学习 {h} 小时 → {predictions_clamped[i].item():.2f} 分")
print(f"\n(最大学时 8h 预测: {predictions_clamped[-1].item():.2f} 分，未超过 100)")
sys.stdout.flush()

# ===== 可视化 =====
print("\n正在生成可视化图表...")
sys.stdout.flush()

x_line = torch.linspace(0, 9, 100).view(-1, 1)
y_line = model(x_line).detach()

# 修复 2：检查预测值是否含 NaN/Inf（防止静默绘图失败）
if torch.isnan(y_line).any() or torch.isinf(y_line).any():
    print("\n⚠️ 警告：预测值包含 NaN 或 Inf！可能因 ReLU 缺失梯度异常。")
    print("模型权重统计：")
    for name, param in model.named_parameters():
        print(f"  {name}: min={param.data.min():.4f}, max={param.data.max():.4f}")
    y_line = torch.nan_to_num(y_line, nan=0.0, posinf=100.0, neginf=0.0)

y_line_clamped = torch.clamp(y_line, 0, 100)

try:
    # 绘图
    fig, ax = plt.subplots()
    ax.scatter(x.numpy(), y.numpy(), color='blue', label='原始数据', s=60)
    ax.plot(x_line.numpy(), y_line_clamped.numpy(), color='red', linewidth=2, label='拟合曲线')
    ax.set_title('学习时间与预测分数关系（ReLU 已注释，纯线性拟合）')
    ax.set_xlabel('学习时间 (小时)')
    ax.set_ylabel('预测分数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # ===== 修复 3：优先保存图片文件 =====
    # 即使窗口因环境问题没弹出，也能在 PNG 文件中看到图表
    output_path = r"e:\组会记录\第二周\fit_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 图表已保存至: {output_path}")
    sys.stdout.flush()

    # 修复 4：尝试弹出交互式窗口（无 GUI 环境会自动跳过）
    print("正在显示交互窗口（若无 GUI 环境请直接关闭，图表已保存为 PNG）...")
    sys.stdout.flush()
    plt.show(block=True)

except Exception as e:
    print(f"\n❌ 绘图过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
