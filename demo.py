import torch
import torch.nn as nn
from torch.nn import functional as F

print("=" * 50)
print("1. 张量 vs C++ 的 vector<vector<...>>")
print("=" * 50)

# C++: vector<vector<vector<double>>> logits(B, vector<vector<double>>(T, vector<double>(C)));
# PyTorch: 一个三维张量，不管多少维，类型都是 torch.Tensor
logits = torch.randn(2,4, 4)  # (B, T, C)
print(f"形状: {logits.shape}")         # torch.Size([4, 8, 65])
print(f"维度数: {logits.ndim}")
print(f"元素总数: {logits.numel()}")
print(f"数据类型: {logits.dtype}")

print(logits)

# 变形 — C++ 需要重新创建数组、拷数据，PyTorch 一个 view 原地改元信息
flat = logits.view(-1, 4)           # (8, 4)
print(f"\nview(-1, 4) → {flat.shape}")  # 8 = 2*4 自动算
print(f"同一块内存: {flat.data_ptr() == logits.data_ptr()}")  # True，没拷贝

print("\n" + "=" * 50)
print("2. nn.Embedding — 查表+自动求导的包裹")
print("=" * 50)

vocab_size = 65
embed = nn.Embedding(vocab_size, vocab_size)  # 等价于 W[65][65]
print(f"权重形状: {embed.weight.shape}")       # torch.Size([65, 65])
print(f"可训练参数数: {sum(p.numel() for p in embed.parameters())}")  # 4225
print(f"requires_grad: {embed.weight.requires_grad}")  # True，自动求导开启

# 查表 — C++: 两层循环取 W[idx[b][t]]，PyTorch: 一行
idx = torch.randint(0, vocab_size, (4, 8))  # 随机输入 [B, T]
logits = embed(idx)                           # 自动把 idx 展开，取出对应行
print(f"\n输入 idx 形状: {idx.shape}")        # [4, 8]
print(f"查表后 logits 形状: {logits.shape}")  # [4, 8, 65]

print("\n" + "=" * 50)
print("3. F.cross_entropy — softmax + log + mean 一步到位")
print("=" * 50)

# C++: 要写 softmax_1d() + softmax_all() + cross_entropy() 三个函数
# PyTorch: 一行，而且内部做了 log_softmax 防止数值溢出
targets = torch.randint(0, vocab_size, (4, 8))  # 正确答案
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # (B*T, C)
    targets.view(-1)              # (B*T,)
)
print(f"Cross entropy loss: {loss.item():.4f}")

# 验证：手算一遍确认理解
probs = F.softmax(logits.view(-1, vocab_size), dim=-1)   # softmax
correct_probs = probs[range(32), targets.view(-1)]         # 取正确类别概率
manual_loss = -torch.log(correct_probs).mean()             # -log 再平均
print(f"手动算的 loss:        {manual_loss.item():.4f}")  # 跟上面一致

print("\n" + "=" * 50)
print("4. 自动求导 — 一行 backward() 替代所有手推梯度")
print("=" * 50)

# 简化：只用 1 个样本，确保没有 token 重复出现
W_demo = torch.randn(65, 65, requires_grad=True)
idx_demo = torch.tensor([[12, 5, 30, 41, 7, 22, 18, 3]])  # 8 个不同 token
target_demo = torch.randint(0, 65, (1, 8))

# 前向
logits_demo = W_demo[idx_demo.view(-1)]    # (8, 65)
loss_demo = F.cross_entropy(logits_demo, target_demo.view(-1))

print(f"前向 loss: {loss_demo.item():.4f}")
print(f"W_demo.grad 存在吗? {W_demo.grad}")  # None

loss_demo.backward()
print(f"反向传播后 W_demo.grad 形状: {W_demo.grad.shape}")  # [65, 65]

# 验证梯度公式: ∂L/∂z_i = p_i - (1 if i==target else 0)
# 每个输入 token 只在序列里出现一次，梯度不会叠加
token = 12  # idx_demo 的第一个值
pos = 0     # 对应 logits_demo[0]
t = target_demo[0, pos]

with torch.no_grad():
    p = F.softmax(logits_demo[pos], dim=-1)       # softmax 概率
    p[t] -= 1                                      # 正确位减 1
    p /= 8                                          # loss 对 8 个位置取平均！
    autograd_grad = W_demo.grad[token]
    print(f"\n∂L/∂z[{token}] 手算:  {p[:5]}")
    print(f"∂L/∂z[{token}] autograd: {autograd_grad[:5]}")
    print(f"匹配吗? {torch.allclose(p, autograd_grad, atol=1e-6)}")  # True!

print("\n" + "=" * 50)
print("5. GPU/MPS 加速 — 一行 .to(device)")
print("=" * 50)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"当前设备: {device}")

# 同样的计算，搬到 GPU/MPS 上
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# 矩阵乘法在 MPS 上并行执行
z = x @ y
print(f"结果形状: {z.shape}, 设备: {z.device}")
print(f"结果样例: {z[0, :3]}")

print("\n" + "=" * 50)
print("总结：PyTorch 做的事")
print("=" * 50)
print("1. Tensor — 替代 vector<vector<vector<...>>>")
print("2. nn.Embedding / F.cross_entropy — 封装常用操作")
print("3. backward() — 自动求导，不用手推梯度")
print("4. .to('mps') — 一行搬上 GPU 并行计算")
