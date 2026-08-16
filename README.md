# LLM Learning

个人 LLM 自学仓库。大二 CS 学生,跟着 Andrej Karpathy 的教程从零手写神经网络与语言模型,这里存放学习笔记与跟写代码,仅作个人学习记录。

## 目录

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [`micrograd.ipynb`](micrograd.ipynb) | 从零实现反向传播引擎(micrograd)与一个小型 MLP | ✅ |
| [`demo.py`](demo.py) | PyTorch 基础笔记:张量、`nn.Embedding`、`F.cross_entropy`、自动求导、MPS 加速(结合 C++ 视角对比理解) | ✅ |
| [`demo.ipynb`](demo.ipynb) | 字符级 GPT(nanoGPT):数据集、字符级 tokenizer、批次采样、Bigram 语言模型 | 🚧 进行中 |
| [`input.txt`](input.txt) | tiny Shakespeare 数据集(约 1.1 MB,来自 Karpathy 的 char-rnn 仓库) | — |

## 学习路线

主要跟学 Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列:

1. **micrograd** — 手写自动求导引擎,理解反向传播的本质([视频](https://www.youtube.com/watch?v=VMj-3S1tku0) / [代码](https://github.com/karpathy/micrograd))
2. **nanoGPT** — 从零搭建字符级 GPT,理解 Transformer 的每一行([视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) / [代码](https://github.com/karpathy/nanoGPT))

## 环境

- Python 3 + PyTorch
- 本机为 Apple Silicon,使用 MPS 加速(`demo.py` 中会自动检测)
- `demo.py` 可直接运行:

```bash
python demo.py
```

- 两个 `.ipynb` 用 Jupyter 打开:

```bash
jupyter notebook
```

## 说明

- 本仓库仅为个人学习记录,代码基于 Karpathy 教程跟写整理,并附有自己的注释与理解。
- 内容会随学习进度持续更新。
