---
title: "21.6 Reinforcement Attention Learning（论文）"
source_docx: "第3部分 大语言模型/21.强化微调的前沿探索/21.6 Reinforcement Attention Learning（论文）.docx"
status: "image-reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 21.6 Reinforcement Attention Learning（论文）


> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

## 一、提出背景

当前的后训练范式主要使用强化学习（RL）来优化策略，通过奖励来偏好高实用性的词元序列。这种基于下一个词元预测的优化在纯文本推理中效果显著，但在多模态任务（如图像或视频问答）中，仅靠生成冗长的文本思考过程不仅对感知能力的提升有限，甚至可能降低模型的基础感知能力。

多模态任务要求模型精准地识别并关注与任务相关的视觉信息，这一过程由Transformer的注意力机制控制。然而，标准的RL优化的是结果（生成的词元），而非内部信息分配的过程。单纯在词元级别进行优化容易导致“奖励作弊”，即模型过度拟合语言的表面形式，而不是掌握真正的底层跨模态逻辑。

## 二、Reinforced Attention Learning（RAL）的原理与工作流

### （一）策略的定义

- 设总序列为 $S=(x_1,\cdots,x_T)$，其中前 $P$ 个为提示词，后续为生成的回复。
- 从模型最后一层 Transformer 中提取各个注意力头的权重并求平均，得到生成词元 $t$ 对之前位置 $i$ 的注意力权重 $\alpha_{t,i}$。
- 对于每个生成的词元 $t \in [P+1,T]$，将其注意力转化为分布策略：

$$
p_\theta^t(i)=\frac{\alpha_{t,i}}{\sum_{\tau}\alpha_{t,\tau}}
$$

该公式捕捉了模型如何关注初始指令、视觉输入以及自身正在生成的推理过程。

### （二）计算优势加权的注意力散度

- 算法借鉴了 PPO/GRPO 中的重要性采样，通过比较当前策略 $p_\theta^t$ 与旧策略 $p_{\mathrm{old}}^t$ 之间的 Jensen-Shannon 散度（JSD）来推导每词元损失。
- 该散度通过序列级别的优势函数 $A_t$ 进行加权，目标函数定义为：

$$
L_{\mathrm{AttnRL}}=\mathbb{E}_t\left[A_t \cdot D\left(p_\theta^t \middle\| p_{\mathrm{old}}^t\right)\right]
$$

这部分计算的是当前注意力分布 $p_\theta^t$ 与行为策略（旧分布）$p_{\mathrm{old}}^t$ 之间的距离。

- **为什么使用 JSD**：论文特别指出 $D(\cdot)$ 应该是一个对称且有界的散度度量，例如 Jensen-Shannon 散度（JSD）。与通常缺乏边界的 KL 散度相比，使用 JSD 能够极大地保证训练的稳定性。
- **防止策略崩溃**：传统的基于词元的 RL 往往会导致模型过度拟合某些高奖励的文本表面形式（即“奖励作弊”），从而丧失多样性。通过约束内部的注意力散度，模型实际上获得了一种结构化正则（structural regularization），它只规范模型“如何收集和整合信息”，而不死板地限制最终输出哪一个特定的词元。

### （三）利用奖励信号引导注意力更新

- **当 $A_t>0$ 时（获得高奖励）**：这说明当前的注意力分配模式促成了一个好的回答。此时最小化 $L_{\mathrm{AttnRL}}$ 会产生“拉力”，将当前正在更新的注意力策略 $p_\theta$ 拉向那个成功的旧策略 $p_{\mathrm{old}}$。
- **当 $A_t<0$ 时（获得低奖励）**：这说明当前的注意力分配导致了错误或低质量的输出。此时优化该目标会产生“推力”，迫使当前策略 $p_\theta$ 远离这种次优的注意力分配模式，从而鼓励模型去探索上下文中其他的多模态线索。

最终的训练目标是标准词元级别策略梯度损失 $\mathcal{L}_{\mathrm{RL}}$ 与内部注意力正则化项的结合：

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{RL}}+\lambda_{\mathrm{attn}}L_{\mathrm{AttnRL}}
$$

## 三、在模型蒸馏中的应用

- **目标**：让学生模型 $\pi_\theta$ 继承教师模型 $\pi_\phi$ 的注意力分布，以学习深层的结构化推理模式。
- **损失函数**：该目标不涉及优势项 $A_t$，追求纯粹的结构模仿。损失定义为学生与教师注意力策略间的散度总和：

$$
\mathcal{L}_{\mathrm{AttnDistill}}=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=P+1}^{T}
JSD\left(p_\theta^t \middle\| p_\phi^t\right)
\right]
$$

- **整体目标**：最终的同策略蒸馏目标结合了强化学习目标、输出逻辑的广义知识蒸馏（GKD）损失以及注意力蒸馏损失。

## 参考文献

- Li, B., Ni, J., Qu, C., et al. (2026). [Reinforced Attention Learning](https://arxiv.org/abs/2602.04884). arXiv:2602.04884.
