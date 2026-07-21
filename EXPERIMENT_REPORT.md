# 基于鲁棒行为克隆与门控残差强化学习的灵巧手抓取

## 1. 任务与实验设置

本实验基于 DexGraspMotionChallenge2025，在 Ubuntu 20.04、RTX 4060 Laptop 8 GB 和 Isaac Gym 上训练 Shadow Hand 抓取策略。数据包含 bottle、bowl、camera、mug 四类物体；选取 16 个物体训练，另冻结 4 个未见物体作为最终测试。训练物体划分为 395 条 BC 训练轨迹和 100 条同物体留出轨迹，未见测试集共 95 条。所有正式结果均使用未修改的官方 `successes` 成功标志，自定义 reward 仅用于 PPO 训练。

## 2. 方法

首先从官方权重 warm-start 一个共享多物体 BC。诊断发现，BC 在专家状态上的动作 MAE 仅为 0.0072，但在自身闭环状态上增至 0.2648（36.8 倍），且误差在预抓取、闭合和抬升阶段继续放大。这说明主要问题不是离线拟合不足，而是策略偏离示范后缺少恢复能力。受 DAgger 所讨论的序列决策误差累积启发，我在 100 维本体观测上加入均匀噪声 ±0.02，使 BC 学习示范邻域内的纠正动作。

随后参考 Residual Reinforcement Learning，在冻结 BC 上训练共享 PPO 残差。为避免残差破坏 BC 已有成功动作，本文加入两个状态相关门：腕部门和手指门。最终动作写为

\[
a=\operatorname{clip}\left(a_{BC}+s\odot(g\odot\delta a)\right),
\]

其中残差 \(\delta a\) 为 28 维，腕部与手指最大尺度分别为 0.05 和 0.10，门值 \(g_w,g_f\in[0,1]\)。门作为 PPO 随机动作的一部分参与策略梯度，初始值设为 0.1，并加入权重 0.01 的开启代价。Actor 使用当前 DexRep、BC 动作及 3 帧本体/动作历史；critic 额外使用物体状态、接触力等特权信息。64 个训练环境覆盖 16 个物体，每个物体包含 2 条 BC 成功锚点、1 条高抬升失败和 1 条普通失败。无门和门控方法使用相同 seed、轨迹和 PPO 主参数。

## 3. 结果

同物体留出集结果如下。普通多物体 BC 只有 2/100；观测噪声将其提高到 16/100。门控 PPO 第 50 轮达到 16/100，明显优于相同设置的无门 PPO（11/100），但按预先固定的“宏平均成功率→抬升→失败率”规则，噪声 BC 的抬升更高、失败率更低，因此最终基础模型仍选噪声 BC。

| 方法 | 成功数 | 宏平均成功率 | 平均最大抬升 | 失败率 |
| --- | ---: | ---: | ---: | ---: |
| 普通多物体 BC | 2/100 | 3.65% | 2.81 cm | 10.94% |
| 原共享残差 PPO | 8/100 | 8.82% | 5.27 cm | 11.72% |
| 噪声 BC | **16/100** | **18.71%** | **10.76 cm** | **7.81%** |
| 噪声 BC + 无门 PPO（50轮） | 11/100 | 13.13% | 8.44 cm | 9.49% |
| 噪声 BC + 门控 PPO（50轮） | **16/100** | **18.71%** | 9.41 cm | 8.59% |

门控 PPO 相对噪声 BC 保留 12 个原成功、新增 4 个、丢失 4 个；无门 PPO 仅保留 3 个、新增 8 个、丢失 13 个，说明门控显著缓解了负迁移。腕部和手指门从约 0.1 缓慢增至约 0.19，没有无条件完全打开。训练 reward 后期门控保持稳定，而无门策略继续下降；失败惩罚约占绝对 reward 的 50%，接近与抬升分别约占 34% 和 12%。

<img src="custom_tools/results/noisebc_gated_comparison/training_reward_comparison.png" width="49%" alt="训练 reward 对比"><img src="custom_tools/results/noisebc_gated_comparison/heldout_success_curve.png" width="49%" alt="留出集成功率曲线">

最终未见物体只评测一次，不再用于调参。噪声 BC、无门 PPO、门控 PPO 分别为 28/95、26/95、27/95，门控仍优于无门但没有超过 BC。噪声 BC 的分类别成功为 bottle 4/37、bowl 19/24、camera 3/17、mug 2/17，性能主要由 bowl 拉高，说明跨几何泛化仍不均衡。

| 未见物体方法 | 总成功率 | bottle | bowl | camera | mug |
| --- | ---: | ---: | ---: | ---: | ---: |
| 噪声 BC | **29.47%** | 10.81% | 79.17% | 17.65% | 11.76% |
| 无门 PPO | 27.37% | 13.51% | 75.00% | 5.88% | 11.76% |
| 门控 PPO | 28.42% | 13.51% | **83.33%** | 11.76% | 0% |

成功案例中物体被手指稳定包络并离开桌面；典型失败则表现为接近误差导致未形成有效接触，随后手与物体分离。完整四类成功/失败视频位于 `custom_tools/results/report_renders/`。

<img src="custom_tools/results/report_renders/bottle_success_candidate_ba4_idx6/env000_success.png" width="24%" alt="bottle成功"><img src="custom_tools/results/report_renders/noisebc_final/02_bottle_failure_6/env000_final.png" width="24%" alt="bottle失败"><img src="custom_tools/results/report_renders/noisebc_final/07_mug_success_5/env000_success.png" width="24%" alt="mug成功"><img src="custom_tools/results/report_renders/noisebc_final/08_mug_failure_2/env000_final.png" width="24%" alt="mug失败">

## 4. 理解与思考

实验表明，灵巧手抓取的低成功率不能只用离线 loss 判断：很小的一步动作误差会通过接触动力学迅速累积。简单观测噪声直接提高了闭环鲁棒性，是本实验收益最大的改动。残差 PPO 能探索 BC 之外的动作，但无门残差容易“以新换旧”；状态门使策略只进行有限修正，显著保留已有技能。不过门控仍会在类别间重新分配成功，尤其在未见 mug 上发生遗忘，说明两个全局门还不足以描述不同抓取阶段和物体几何。后续可尝试按接近/闭合/抬升阶段设置门，加入成功锚点上的行为约束，或训练类别专家后蒸馏为统一学生策略。

参考：[挑战 Wiki](https://github.com/DexGraspMotionChallenge/DexGraspMotionChallenge2025/wiki)、[Residual Reinforcement Learning](https://arxiv.org/abs/1812.03201)、[DAgger](https://proceedings.mlr.press/v15/ross11a.html)。
