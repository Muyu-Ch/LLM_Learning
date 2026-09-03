# LLM Learning — Andrej Karpathy 教程中文注解版

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 从零手写并理解神经网络与语言模型。全程跟随 [Andrej Karpathy 的 Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列,代码逐行配上中文注释与自己的思考,希望为中文学习者提供一个可以对照教程轻松读懂的伴读仓库。

## ✨ 这个仓库有什么

- **逐行中文注释**:不只是翻译"这行代码做了什么",更记录"为什么这样写"——包括踩过的坑(梯度不清零、tanh 饱和、广播机制等)
- **实验驱动**:每个关键结论都尽量用具体数字和可视化验证,而不是"书上这么说"
- **与教程一一对应**:目录按 Zero to Hero 的课程顺序组织,方便边看视频边对照

## 📁 目录与进度

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [`micrograd/micrograd.ipynb`](micrograd/micrograd.ipynb) | 从零实现反向传播引擎(micrograd)与小型 MLP | ✅ |
| [`makemore/Makemore.ipynb`](makemore/Makemore.ipynb) | 名字生成模型:bigram 计数版(频数矩阵、`multinomial` 采样、并行化生成)已完成;接下来是 loss 与神经网络版 | 🚧 |
| [`demo.py`](demo.py) | PyTorch 基础笔记:张量、`nn.Embedding`、`F.cross_entropy`、自动求导、MPS 加速 | ✅ |
| [`demo.ipynb`](demo.ipynb) | 字符级 GPT(nanoGPT):数据集、字符级 tokenizer、批次采样、Bigram 语言模型 | 🚧 |
| [`input.txt`](input.txt) | tiny Shakespeare 数据集(约 1.1 MB,来自 Karpathy 的 char-rnn 仓库) | — |

## 🧭 学习路线(Zero to Hero)

1. **micrograd** — 手写自动求导引擎,理解反向传播的本质([视频](https://www.youtube.com/watch?v=VMj-3S1tku0) / [代码](https://github.com/karpathy/micrograd))
2. **makemore** — 从 bigram 计数到多层感知机,再到 Transformer 的前身([视频](https://www.youtube.com/watch?v=PaCmpygFfXo) / [代码](https://github.com/karpathy/makemore))
3. **nanoGPT** — 从零搭建字符级 GPT,理解 Transformer 的每一行([视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) / [代码](https://github.com/karpathy/nanoGPT))
4. *(计划中)* **GPT-2 from scratch** — 复刻 GPT-2 的完整训练([视频](https://www.youtube.com/watch?v=l8pRSuU81PU) / [代码](https://github.com/karpathy/build-nanogpt))

## 🚀 快速开始

- 环境:Python 3 + PyTorch(本机 Apple Silicon,自动使用 MPS 加速)
- `demo.py` 可直接运行:

```bash
python demo.py
```

- 两个 `.ipynb` 用 Jupyter 打开:

```bash
jupyter notebook
```

## 📚 致谢与说明

- 代码跟写自 Andrej Karpathy 的公开教程与仓库(MIT License),中文注释与整理为个人学习记录
- 本人是初学者,注释里可能有理解不到位的地方,欢迎提 [issue](https://github.com/Muyu-Ch/LLM_Learning/issues) 或 PR 指正

## 📄 License

[MIT](LICENSE)
