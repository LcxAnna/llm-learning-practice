# LLM 学习实践：第一周学习框架

> 主题：从语言模型基础到最小 Transformer 实现  
> 周期：第 1 周  
> 目标：建立 LLM 的最小知识闭环与代码闭环，为后续深入 Attention、Transformer、预训练、微调、RAG、Agent 等内容打基础。

---

## 1. 本周总体目标

第一周不追求大模型训练，也不追求复杂工程系统，而是聚焦一个核心问题：

> 大语言模型是如何从文本数据中学习“预测下一个 token”的？

围绕这个问题，本周需要完成三个层面的目标。

### 1.1 概念目标

理解以下基础概念：

- token、vocabulary、tokenization
- context window / block size
- next-token prediction
- logits、softmax、cross entropy loss
- embedding
- positional encoding / positional embedding
- self-attention
- causal mask
- multi-head attention
- feed-forward network
- residual connection
- layer normalization
- decoder-only Transformer

### 1.2 代码目标

能够用 PyTorch 从零实现一个最小语言模型训练流程：

```text
文本数据
  ↓
Tokenizer
  ↓
训练样本 x, y
  ↓
Batch
  ↓
模型 forward
  ↓
Cross Entropy Loss
  ↓
Backward
  ↓
Optimizer Step
  ↓
文本生成
```

### 1.3 产出目标

本周结束时，产出一个可以运行的 MiniGPT 项目：

```text
llm_week1/
├── day1_tokenizer.py
├── day2_bigram_lm.py
├── day3_embedding_model.py
├── day4_self_attention.py
├── day5_transformer_block.py
├── day6_minigpt.py
└── week1_summary.md
```

最终模型虽然很小，但必须具备完整训练和生成能力。

---

## 2. 为什么第一周这样安排

LLM 的学习很容易陷入两个误区：

1. 一开始就看复杂论文，例如 GPT、BERT、T5、LLaMA、MoE、RLHF。
2. 一开始就调用现成框架，例如 Hugging Face Transformers，但并不理解内部机制。

这两种方式都容易导致“会用 API，但不理解模型本质”。

因此，第一周采用从底层到整体的学习路线：

```text
Tokenizer
  ↓
Bigram Language Model
  ↓
Embedding
  ↓
Self-Attention
  ↓
Multi-Head Attention
  ↓
Transformer Block
  ↓
MiniGPT
```

这样安排的原因是：

### 2.1 先理解数据，再理解模型

LLM 的输入不是原始文本，而是 token id。

如果不先理解 tokenization，就很难理解：

- 为什么模型输入是整数张量；
- 为什么词表大小会影响输出层；
- 为什么模型输出的是 vocabulary 上的概率分布；
- 为什么生成文本时需要反复采样 token。

所以第 1 天先学习 tokenizer 和训练样本构造。

---

### 2.2 先做最简单语言模型，再做 Transformer

Bigram Language Model 只根据当前 token 预测下一个 token。

它非常简单，但可以完整覆盖语言模型训练的核心流程：

```text
input ids -> logits -> loss -> backward -> optimizer -> generate
```

在理解 Bigram 之后，再引入 embedding、attention、Transformer，学习难度会小很多。

---

### 2.3 先理解 Attention，再组合 Transformer

Transformer 看起来复杂，但本质上由几个模块组合而成：

```text
Token Embedding
+ Positional Embedding
+ Causal Self-Attention
+ Feed-Forward Network
+ Residual Connection
+ LayerNorm
```

如果直接看完整 Transformer，很容易被结构细节淹没。

所以本周把 Transformer 拆成几天逐步实现：

- Day 3：Embedding
- Day 4：Self-Attention
- Day 5：Transformer Block
- Day 6：完整 MiniGPT

---

### 2.4 本周只做 Decoder-only Transformer

LLM 中最典型的生成式模型，例如 GPT 系列，采用的是 Decoder-only Transformer。

第一周优先学习 Decoder-only 架构，而不是同时学习 Encoder、Decoder、Seq2Seq，是为了降低认知负担。

本周关注：

```text
给定前文 token，预测下一个 token
```

暂时不展开：

- BERT 的 Masked Language Modeling
- T5 的 Encoder-Decoder 架构
- Instruction Tuning
- RLHF / DPO
- RAG
- Agent
- 多模态模型

这些内容放到后续周学习。

---

## 3. 本周学习路线总览

| 天数 | 主题 | 核心问题 | 主要产出 |
|---|---|---|---|
| Day 1 | Tokenization 与训练样本构造 | 文本如何变成模型可以处理的数据？ | 字符级 tokenizer、x/y 样本、batch |
| Day 2 | Bigram Language Model | 如何训练一个最小语言模型？ | 完整训练闭环与生成函数 |
| Day 3 | Embedding 与位置表示 | token id 如何变成向量？模型如何知道顺序？ | Token embedding + positional embedding |
| Day 4 | Self-Attention | 模型如何从上下文中聚合信息？ | 单头 causal self-attention |
| Day 5 | Multi-Head Attention 与 Transformer Block | 如何组合多个 attention 头和非线性层？ | Transformer block |
| Day 6 | MiniGPT 完整实现 | 如何搭建 decoder-only Transformer？ | 可训练、可生成的 MiniGPT |
| Day 7 | 复盘与实验 | 如何验证自己真的理解了？ | week1_summary.md、调参实验记录 |

---

## 4. 每天的学习重点

## Day 1：Tokenization 与训练样本构造

### 学习重点

- 什么是 token
- 什么是 vocabulary
- 字符级 tokenizer 的 encode / decode
- 什么是 block size
- 什么是 batch size
- 如何构造语言模型训练样本

### 关键理解

语言模型训练时，输入和目标是错位一位的序列：

```text
x: hello wor
y: ello worl
```

模型在每个位置都要预测下一个 token。

### 当日产出

```text
day1_tokenizer.py
```

要求：

- 能将文本编码为 token id；
- 能将 token id 解码回文本；
- 能构造 x, y；
- 能构造 batch；
- 能解释每个张量的 shape。

---

## Day 2：Bigram Language Model

### 学习重点

- logits
- softmax
- cross entropy
- model forward
- loss.backward()
- optimizer.step()
- 文本生成 generate

### 关键理解

Bigram 模型虽然简单，但它已经包含语言模型训练的完整闭环。

它学习的是：

```text
P(next_token | current_token)
```

### 当日产出

```text
day2_bigram_lm.py
```

要求：

- 能训练一个 Bigram 模型；
- 能计算 loss；
- 能执行反向传播；
- 能生成文本；
- 能解释 logits 的 shape。

---

## Day 3：Embedding 与位置表示

### 学习重点

- token embedding
- positional embedding
- embedding table
- context length
- 为什么 Transformer 需要位置信息

### 关键理解

Transformer 本身没有天然的序列顺序感，因此需要显式加入位置信息。

模型输入从 token id 变为向量：

```text
token id -> token embedding vector
```

再加上位置向量：

```text
token embedding + positional embedding
```

### 当日产出

```text
day3_embedding_model.py
```

要求：

- 能使用 `nn.Embedding`；
- 能理解 embedding table 的 shape；
- 能将 token embedding 与 position embedding 相加；
- 能解释 batch、time、channel 三个维度。

---

## Day 4：Self-Attention

### 学习重点

- query
- key
- value
- attention score
- scaled dot-product attention
- causal mask
- softmax attention weights

### 关键理解

Self-Attention 的核心是：

> 每个 token 根据自己和其他 token 的相关性，从上下文中加权聚合信息。

Decoder-only LM 中必须使用 causal mask，防止当前位置看到未来 token。

### 当日产出

```text
day4_self_attention.py
```

要求：

- 能实现单头 self-attention；
- 能实现 causal mask；
- 能解释 Q、K、V 的作用；
- 能解释 attention weights 的 shape。

---

## Day 5：Multi-Head Attention 与 Transformer Block

### 学习重点

- multi-head attention
- feed-forward network
- residual connection
- layer normalization
- dropout
- Transformer block

### 关键理解

Multi-head attention 允许模型从多个子空间理解上下文关系。

Transformer block 的基本结构是：

```text
x = x + self_attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

这是现代 Decoder-only Transformer 的基本构件。

### 当日产出

```text
day5_transformer_block.py
```

要求：

- 能实现多个 attention head；
- 能拼接多个 head 的输出；
- 能实现 FFN；
- 能实现 residual connection；
- 能实现 LayerNorm；
- 能组合出 Transformer block。

---

## Day 6：MiniGPT 完整实现

### 学习重点

- decoder-only Transformer
- 多层 Transformer block
- final layer norm
- language modeling head
- 训练循环
- 文本生成

### 关键理解

MiniGPT 的整体结构是：

```text
token ids
  ↓
token embedding + positional embedding
  ↓
Transformer blocks
  ↓
final layer norm
  ↓
linear head
  ↓
logits over vocabulary
```

### 当日产出

```text
day6_minigpt.py
```

要求：

- 能完整训练 MiniGPT；
- 能根据 prompt 生成文本；
- 能调整模型超参数；
- 能观察 loss 下降；
- 能解释模型每个模块的作用。

---

## Day 7：复盘、调参与总结

### 学习重点

- 回顾本周核心概念
- 梳理代码结构
- 对比 Bigram 与 Transformer
- 尝试调参实验
- 记录学习总结

### 建议实验

至少完成 3 个实验：

1. 修改 `block_size`，观察生成效果变化；
2. 修改 `n_embd`，观察 loss 和训练速度变化；
3. 修改 `n_layer` 或 `n_head`，观察模型能力变化。

### 当日产出

```text
week1_summary.md
```

内容包括：

- 本周学到的核心概念；
- 每个模块的作用；
- 遇到的问题；
- 解决方式；
- 模型训练结果；
- 后续改进方向。

---

## 5. 本周代码实现原则

本周代码遵循以下原则：

### 5.1 少依赖

除 PyTorch 外，尽量不使用复杂框架。

暂时不使用：

- Hugging Face Transformers
- Datasets
- Tokenizers
- Lightning
- Accelerate
- DeepSpeed

原因是第一周的目标是理解底层机制，而不是快速调用高层 API。

---

### 5.2 小数据

本周使用极小文本数据即可。

可以使用：

- 一段英文文本；
- 一段中文文本；
- 自己写的小语料；
- Tiny Shakespeare 一类的小型文本。

不要一开始使用大规模语料。

---

### 5.3 小模型

模型参数要小，优先保证能跑通。

建议初始配置：

```python
batch_size = 16
block_size = 32
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1
```

如果电脑性能有限，可以进一步减小：

```python
batch_size = 8
block_size = 16
n_embd = 32
n_head = 4
n_layer = 1
```

---

## 6. 本周不重点学习的内容

为了避免第一周过载，以下内容暂时不深入：

- BPE tokenizer 原理与实现
- GPT-2 / GPT-3 / LLaMA 论文细节
- Hugging Face Transformers 源码
- 分布式训练
- 混合精度训练
- FlashAttention
- KV Cache
- LoRA / QLoRA
- SFT / RLHF / DPO
- RAG
- Agent
- 多模态 LLM

这些内容将在后续学习阶段逐步展开。

---

## 7. 学习方法建议

每天按照下面顺序学习：

```text
先理解概念
  ↓
再运行代码
  ↓
再修改参数
  ↓
最后写总结
```

不要只复制代码。

每一天都要回答三个问题：

1. 今天这个模块解决了什么问题？
2. 它的输入和输出 shape 是什么？
3. 如果去掉这个模块，模型会发生什么变化？

---

## 8. 本周完成标准

完成第一周后，应该达到以下标准：

### 8.1 概念标准

能够清楚解释：

- LLM 为什么可以用 next-token prediction 训练；
- tokenization 的作用；
- cross entropy 在语言模型中的含义；
- self-attention 的 Q、K、V 分别是什么；
- causal mask 为什么必要；
- Transformer block 的组成；
- Decoder-only Transformer 的基本结构。

### 8.2 代码标准

能够独立写出或解释：

- tokenizer；
- get_batch；
- Bigram 模型；
- embedding 模型；
- self-attention；
- multi-head attention；
- Transformer block；
- MiniGPT；
- generate 函数。

### 8.3 产出标准

GitHub 仓库中至少包含：

```text
README.md
llm_week1/day1_tokenizer.py
llm_week1/day2_bigram_lm.py
llm_week1/day3_embedding_model.py
llm_week1/day4_self_attention.py
llm_week1/day5_transformer_block.py
llm_week1/day6_minigpt.py
llm_week1/week1_summary.md
```

README.md 中应说明：

- 本周学习目标；
- 每天学习内容；
- 如何运行代码；
- 当前完成进度；
- 遇到的问题与解决方案。

---

## 9. 推荐 GitHub README 结构

可以将本文作为第一周 README 的主体内容，也可以拆分为：

```text
README.md
docs/week1_plan.md
llm_week1/
```

推荐结构：

```text
llm-learning-practice/
├── README.md
├── docs/
│   └── week1_plan.md
└── llm_week1/
    ├── day1_tokenizer.py
    ├── day2_bigram_lm.py
    ├── day3_embedding_model.py
    ├── day4_self_attention.py
    ├── day5_transformer_block.py
    ├── day6_minigpt.py
    └── week1_summary.md
```

---

## 10. 后续学习展开方式

接下来可以按照每日节奏推进。

每一天的学习内容建议包含四部分：

1. **核心概念讲解**
2. **代码逐行实现**
3. **关键 shape 分析**
4. **练习题与总结模板**

第一周的第一步是 Day 1：

> 实现字符级 tokenizer，理解文本如何变成语言模型训练数据。

完成本周框架文档后，就可以正式开始 Day 1 的具体学习。
