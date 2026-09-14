import torch
import matplotlib.pyplot as plt
# 生成数据
x = torch.linspace(-5, 5, 100).reshape(-1, 1)

true_w = 3.0
true_b = 2.0

y = true_w * x + true_b
losses = []
# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.001    #对比不同学习率的影响，学习率过大会导致loss不收敛，学习率过小会导致收敛速度慢
optimizer = torch.optim.SGD([w, b], lr=lr)

for epoch in range(100):
    # 前向传播
    y_pred = w * x + b

    # 计算损失
    loss = ((y_pred - y) ** 2).mean()
    losses.append(loss.item())
    optimizer.zero_grad()
    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()
    


    #print(f"Epoch {epoch+1}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")
# 同时设置颜色、虚线样式和图例
plt.plot(losses, color='red', linestyle='-', label='Loss Curve')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()  # 必须加上这行，label='Loss Curve' 才会显示在图中
plt.grid(True) # 加上网格辅助看图
plt.show()