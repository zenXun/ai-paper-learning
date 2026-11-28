# 03_rl_from_scratch — 阅读资料与 NotebookLM 索引

本目录存放强化学习（RL）专题的两批核心文献（PDF 由 `.gitignore` 排除，不入库；Markdown 清洗稿与本索引保留）。

## Part A：RL 原理与工程驯服（`part_A_fundamentals/`）

| 编号 | 本地文件名 | 来源 | 在主线中的作用 |
|---|---|---|---|
| A1 | `A1_REINFORCE_Williams_1992.pdf` | [UMass 镜像 PDF](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | Policy Gradient 奠基，暴露高方差与信用分配问题 |
| A1b | `A1b_Pong_from_Pixels_Karpathy_2016.md` | [Karpathy 博客](https://karpathy.github.io/2016/05/31/rl/) | REINFORCE 的直觉化拆解（加权监督学习） |
| A2 | `A2_GAE_Schulman_2015.pdf` | [arXiv:1506.02438](https://arxiv.org/pdf/1506.02438.pdf) | Baseline 与 Advantage 降方差（精读 §1–3） |
| A3 | `A3_PPO_Schulman_2017.pdf` | [arXiv:1707.06347](https://arxiv.org/pdf/1707.06347.pdf) | Clip 安全绳防止分布漂移与策略跳崖 |
| A4 | `A4_PPO_37_Implementation_Details_2022.md` | [ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | 决定 RL 生死的 37 个工程实现细节 |
| A4b | `A4b_Deep_RL_That_Matters_Henderson_2017.pdf` | [arXiv:1709.06560](https://arxiv.org/pdf/1709.06560.pdf) | 选读对照：随机种子方差与复现陷阱 |

## Part B：现代 LLM 强化学习实战（`part_B_llm_rl/`）

| 编号 | 本地文件名 | 来源 | 在主线中的作用 |
|---|---|---|---|
| B1 | `B1_InstructGPT_RLHF_Ouyang_2022.pdf` | [arXiv:2203.02155](https://arxiv.org/pdf/2203.02155.pdf) | RLHF 三阶段管线、显存开销与 Reward Hacking |
| B2 | `B2_DPO_Rafailov_2023.pdf` | [arXiv:2305.18290](https://arxiv.org/pdf/2305.18290.pdf) | 消掉 Reward Model 的离线对比优化及其边界 |
| B3 | `B3_DeepSeekMath_GRPO_2024.pdf` | [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300.pdf) | GRPO 组内相对优势，省掉 Critic 模型（精读 §4.1） |
| B3b | `B3b_DeepSeek_R1_2025.pdf` | [arXiv:2501.12948](https://arxiv.org/pdf/2501.12948.pdf) | 可验证奖励（RLVR）与纯 RL 涌现推理 |
| B4 | `B4_DAPO_ByteDance_2025.pdf` | [arXiv:2503.14476](https://arxiv.org/pdf/2503.14476.pdf) | 大规模 LLM RL 工程四坑与修法 |
