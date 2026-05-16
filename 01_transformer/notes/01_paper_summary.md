# 《Attention Is All You Need》精读笔记

> Vaswani et al., NeurIPS 2017 · [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
>
> 阅读视角：**为什么每个设计选择是这样的**，而不是"它是什么"。

---

## 0. 一句话总结

> 在序列建模任务上，**完全用 attention 替代 recurrence 和 convolution**，可以训练更快、效果更好。

这句话拆开看每个词都有分量：
- "完全替代"：不是用 attention 当 RNN 的补丁，是从根上换掉
- "训练更快"：解放了 RNN 的时序串行性，整个序列可以并行
- "效果更好"：WMT'14 EN-DE 上 +2 BLEU，EN-FR 41.8 BLEU（当年 SOTA）

---

## 1. 动机：RNN/CNN 哪里不够好？

理解 Transformer 必须先理解它要解决什么痛点。

### 1.1 RNN 的两个根本问题

**(a) 串行计算无法并行**
RNN 计算 $h_t$ 必须先有 $h_{t-1}$。一个长度 $n$ 的序列，前向至少要 $O(n)$ 个串行步。GPU 的并行能力被时序依赖锁死。

**(b) 长距离依赖路径太长**
信息从位置 $i$ 传到位置 $j$ 要经过 $|j-i|$ 个隐状态。即便有 LSTM/GRU 的门控机制，梯度仍然会沿路径衰减或膨胀。

### 1.2 CNN（如 ConvS2S）的折中

CNN 解决了并行问题，但**感受野受卷积核限制**。要让两个远端位置交互，需要堆叠 $O(\log_k n)$ 层（kernel size $k$ 的 dilated conv）或 $O(n/k)$ 层（普通 conv）。

### 1.3 Attention 的承诺

| 模型 | 单层串行步 | 任意两点最大路径 | 单层复杂度 |
|------|-----------|----------------|----------|
| RNN  | $O(n)$    | $O(n)$         | $O(n \cdot d^2)$ |
| CNN  | $O(1)$    | $O(\log_k n)$  | $O(k \cdot n \cdot d^2)$ |
| **Self-Attention** | $O(1)$ | $O(1)$ | $O(n^2 \cdot d)$ |

关键洞察：**self-attention 把任意两个位置的交互路径压缩到常数**，代价是 $n^2$ 的内存/计算（后来这是 LLM 长上下文优化的核心战场，FlashAttention/线性 attention 都是攻击这个 $n^2$）。

---

## 2. 架构总览

原论文是 encoder–decoder（为机器翻译设计）。本项目按基线决策走 **decoder-only**（GPT 形态），原因见 [`../../README.md`](../README.md) 和 tech_choices 记忆。

```
Decoder-only Transformer（GPT 形态）

         Token IDs              Output logits (over vocab)
             │                          ▲
        ┌────▼─────┐               ┌────┴────┐
        │ Embedding│               │ Linear  │  ← 通常和 embedding 权重共享
        └────┬─────┘               └────▲────┘
             │                          │
        ┌────▼─────┐              ┌─────┴─────┐
        │ + Pos.   │              │  LayerNorm│
        │  Encoding│              └─────▲─────┘
        └────┬─────┘                    │
             │                          │
        ╔════▼══════════════════════════╧════╗
        ║         × N (e.g. 6 or 12)         ║
        ║                                    ║
        ║   ┌────────────────────────┐       ║
        ║   │  Masked Multi-Head     │       ║
        ║   │  Self-Attention        │       ║
        ║   └──────────┬─────────────┘       ║
        ║              + (residual, LayerNorm)║
        ║   ┌──────────▼─────────────┐       ║
        ║   │  FFN (Linear → GELU →  │       ║
        ║   │       Linear)          │       ║
        ║   └──────────┬─────────────┘       ║
        ║              + (residual, LayerNorm)║
        ╚══════════════╧═════════════════════╝
```

构成元素只有 5 种：
1. Embedding（含 positional encoding）
2. Masked multi-head self-attention
3. Position-wise FFN
4. Residual connection
5. LayerNorm

第 2 项是真正的"新东西"，剩下都是工程脚手架。

---

## 3. Scaled Dot-Product Attention：为什么是 $\sqrt{d_k}$

### 3.1 公式

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### 3.2 直觉一句话

> Attention 是"用 query 去查询一组 key，按相似度加权聚合 value"——一种**可微的键值查询**。

类比哈希表 `dict[key] = value`：传统查询是精确匹配（key 完全一样才返回 value），attention 是软查询（每个 key 都返回一些权重，越像权重越大）。

### 3.3 为什么是点积？为什么除以 $\sqrt{d_k}$？

**点积**：计算便宜（矩阵乘法），且在向量已归一化的情况下等价于余弦相似度。论文里和"additive attention"（Bahdanau attention）做了对比，**当 $d_k$ 较小时两者效果接近，但点积更快**——因为可以走高度优化的 matmul。

**$\sqrt{d_k}$ 缩放的方差论证**：

> 假设 $q, k \in \mathbb{R}^{d_k}$，每个分量是均值 0、方差 1 的独立随机变量。则 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$（独立项方差相加），标准差为 $\sqrt{d_k}$。

后果：当 $d_k$ 大时，点积值的尺度也大。softmax 对大数值非常敏感——它会把最大值的概率推到接近 1，其它推到接近 0。这意味着：
- **梯度消失**：softmax 在饱和区导数极小，反向传播几乎没信号
- **训练不稳定**：early training 时随机权重就可能把 softmax 推到饱和

除以 $\sqrt{d_k}$ 把方差拉回 1，softmax 保持在"中间梯度合理"的区间。

### 3.4 关键代码片段（先脑内实现）

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # [B, h, n, n]
scores = scores.masked_fill(causal_mask, -inf)       # decoder: 未来位置置 -inf
attn   = scores.softmax(dim=-1)
out    = attn @ V                                    # [B, h, n, d_v]
```

---

## 4. Multi-Head Attention：为什么不用一个大头

### 4.1 论文原话

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

### 4.2 翻译成人话

单头 attention 对每个位置只能产出**一个**加权和。如果不同的语义关系（句法主语、共指、词义消歧……）需要不同的加权模式，单头被迫求平均，互相抵消。

多头 = **把 $d_{\text{model}}$ 拆成 $h$ 份独立的子空间，每个子空间独立做 attention**，然后拼回去。

```
单头：    Q,K,V ∈ R^{n × d_model}                   → 一种 attention 模式
多头(h=8): 切成 8 份  Q_i,K_i,V_i ∈ R^{n × (d_model/h)}  → 8 种 attention 模式
                     ↓
                  concat
                     ↓
                output projection
```

### 4.3 注意：计算量近似不变

每个头 $d_k = d_v = d_{\text{model}}/h$，所以总 FLOPs 和单头跑 $d_{\text{model}}$ 维度差不多。多头是 **结构归纳偏置**，不是堆算力。

### 4.4 实证发现（论文 §5 + 后续研究）

- 不同头确实学到了不同的"专长"（句法、共指、邻接等）
- BERT/GPT 时代的可视化工作（Clark et al. 2019）进一步证实了这一点
- 但也有研究表明大部分头可以剪掉（Voita et al. 2019）——说明初始化的多样性比头数本身更关键

→ **博客点子**：可视化训练完的小 Transformer 不同头的 attention 模式，证明它们确实分工

---

## 5. Positional Encoding：为什么是正弦

### 5.1 问题：attention 是 permutation-invariant 的

如果你把序列 `[A, B, C]` 打乱成 `[C, A, B]`，输出（在不考虑位置时）完全一样。RNN 天然有顺序信息（隐状态时序流动），attention 没有。所以必须**显式注入位置**。

### 5.2 论文方案：固定正弦/余弦

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- 每一维是不同频率的正弦/余弦
- 低频维度变化慢（捕捉宏观位置），高频维度变化快（捕捉局部细节）
- 类似"位置的傅里叶基"

### 5.3 为什么不用 learned positional embedding？

论文做了对比："learned 和 sinusoidal 几乎一样好"。选 sinusoidal 的理由：

> "...for any fixed offset k, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$."

也就是说，**相对位置关系是位置编码的线性变换**——理论上模型应该更容易学到"距离"概念。

另一个理由是**外推**：训练时见过最长 $L$，推理时遇到 $> L$ 的序列，learned PE 没法处理（embedding 表里没有），sinusoidal 公式可以无缝外延。**但**实际外推效果一般，后来的 RoPE/ALiBi 才真正解决了这个问题。

### 5.4 后续演进（值得在博客里铺一条时间线）

| 方法 | 出处 | 思想 |
|------|------|------|
| Sinusoidal (绝对) | Vaswani 2017 | 固定正弦 |
| Learned (绝对)    | GPT/BERT | 可学习查表 |
| Relative position bias | Shaw 2018 / T5 | attention 分数加位置偏置 |
| **RoPE**          | Su 2021    | 旋转位置编码，自然内嵌相对距离 |
| **ALiBi**         | Press 2022 | softmax 前减去距离 × 斜率 |

现代大模型（LLaMA、Qwen）几乎都用 RoPE 了。

---

## 6. LayerNorm 位置：Post-LN vs Pre-LN

### 6.1 原论文：Post-LN

```
x → Sublayer(x) → +x → LayerNorm → 下一层
            残差         规范化
```

公式：`out = LayerNorm(x + Sublayer(x))`

### 6.2 问题：训练不稳定

Post-LN 的 Transformer 训练对学习率非常敏感，**必须配 warmup**（论文里 4000 步线性 warmup），否则容易发散。

原因（Xiong et al. 2020 的分析）：早期训练时，残差路径上的梯度尺度随层数线性放大，深层网络梯度爆炸。

### 6.3 Pre-LN 的修正

```
x → LayerNorm → Sublayer → +x → 下一层
    规范化        残差
```

公式：`out = x + Sublayer(LayerNorm(x))`

Pre-LN 把 LayerNorm 移到子层内部，**残差是恒等路径**，梯度可以无衰减传到底层。代价是表达能力略有损失（理论上 Pre-LN 等价于一个更窄的网络），但训练稳定性大幅提高，**可以省掉 warmup**。

### 6.4 现代选择

- GPT-2 起开始用 Pre-LN
- LLaMA 使用 **RMSNorm + Pre-LN**（RMSNorm 是 LayerNorm 的简化版，省掉均值中心化）
- 训练超大模型时，Pre-LN 几乎是默认选择

→ **MVP 决策**：第一版用 Pre-LN（更容易训稳），写完后专门跑一个 Post-LN 对比实验，作为博客的"实验小节"。

---

## 7. FFN：被低估的部分

每个 Transformer block 里除了 attention 还有一个 FFN：

```python
FFN(x) = Linear(GELU(Linear(x)))
# 维度: d_model → d_ff → d_model
# 论文: d_model=512, d_ff=2048  (4× 扩张)
```

虽然结构平平无奇，但 **FFN 占了 Transformer 大部分参数**（attention 部分是 $4 d_{\text{model}}^2$，FFN 是 $8 d_{\text{model}}^2$）。

近年的研究（如 Geva 2021）发现 FFN 起到了 **key-value memory** 的作用，存储着模型的事实知识。这也是 MoE（Mixture of Experts）专门拆 FFN 的原因。

---

## 8. 训练细节

| 项目 | 设置 | 为什么 |
|------|------|--------|
| Optimizer | Adam, $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$ | $\beta_2$ 比常见的 0.999 小，对二阶动量更敏感 |
| LR schedule | $\text{lr} = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$，warmup = 4000 | warmup 防 Post-LN 早期爆炸 |
| Dropout | 0.1（子层输出 + embedding 加和后） | 标准正则化 |
| Label smoothing | $\epsilon_{ls}=0.1$ | 让模型不过度自信，提升 BLEU（但增加 PPL） |
| Batch | ~25K source + 25K target tokens | 注意：是 token 数而不是 sequence 数 |

**关于学习率公式**：这是个工程奇技，本质是"warmup 阶段线性涨，然后按 $1/\sqrt{step}$ 衰减"。decoder-only 训练时，现代做法更常用 **cosine decay + warmup**，更直观。

---

## 9. 关键超参一览（Base 模型）

| 超参 | 值 |
|------|---|
| $d_{\text{model}}$ | 512 |
| $d_{ff}$ | 2048 |
| 头数 $h$ | 8 |
| $d_k = d_v$ | 64（= 512/8） |
| 层数 $N$ | 6（encoder 和 decoder 各 6） |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| Vocab | BPE，~37K |

---

## 10. 学习者视角的关键问题清单

这些问题既是博客的好材料，也是 MVP 实现完后要能回答的：

### 已经在论文/笔记里回答的
- [x] 为什么 attention 比 RNN 强？（并行 + 路径短）
- [x] 为什么除 $\sqrt{d_k}$？（softmax 不饱和）
- [x] 为什么多头？（多种 attention 模式并存）
- [x] 为什么正弦 PE？（外推 + 相对位置容易学）

### 需要在 MVP 阶段亲手验证的
- [ ] 去掉 $\sqrt{d_k}$ 缩放，训练会怎样不稳？（消融实验）
- [ ] 头数从 1 → 8 → 32，loss 曲线如何变化？算力差别？
- [ ] Pre-LN vs Post-LN，同样 LR 下谁更稳？
- [ ] Sinusoidal vs learned PE，同任务上有差别吗？
- [ ] 训练好的 attention 头到底学了什么？（可视化）

### 不在本模块范围内（但要知道是开放问题）
- [ ] $n^2$ 复杂度怎么破？→ 后续 FlashAttention / 线性 attention 模块
- [ ] 怎么扩展上下文长度？→ 后续 RoPE / ALiBi / YaRN 模块
- [ ] FFN 里到底存了什么？→ 后续可解释性模块

---

## 11. 下一步（衔接 Task 2：MVP 实现）

带着这份理解去写代码时的设计基线：

1. **架构**：decoder-only，4–6 层，$d_{\text{model}}$ 128–256（toy 任务够用）
2. **PE**：先实现 sinusoidal（原汁原味），消融时再加 learned 对比
3. **LayerNorm**：用 **Pre-LN**（稳定优先）
4. **Attention**：手写 scaled dot-product + multi-head，**不调用 `nn.MultiheadAttention`**
5. **Toy 任务**：tiny Shakespeare 字符级语言建模
6. **训练**：AdamW + cosine warmup（不抠原论文 LR 公式）
7. **打印/日志**：每个组件 forward 时记录 shape，便于调试和后续可视化

代码骨架预想（具体在 Task 2 落地）：

```
01_transformer/
├── notes/
│   └── 01_paper_summary.md       ← 本文件
├── model.py                       ← Block / MultiHeadAttention / FFN / 完整模型
├── data.py                        ← tiny Shakespeare 加载 + tokenizer
├── train.py                       ← 训练循环 + 日志
├── visualize.py                   ← attention 热力图、PE 模式图
├── ablation.py                    ← 消融实验：±sqrt(dk), Post-LN, learned PE
└── notebooks/
    └── walkthrough.ipynb          ← 逐步演示
```
