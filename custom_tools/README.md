# 自定义工具

- 官方仓库代码作为基准保留，不直接加入我们的实验逻辑。
- 自定义诊断、评测和数据工具放在 `custom_tools/`。
- 必须扩展官方类时，优先使用新文件中的包装器、子类或独立配置副本。
- 10 个曾修改的官方源码和配置已恢复为 Git 基准版本。
- 训练和评测入口带显存余量检查；显存不足时会在创建模型/仿真前退出。
- 命令行中的相对路径按启动命令时的目录解析，不受内部工作目录切换影响。

当前工具：

- `preflight_check.py`：只读检查环境、数据、mesh、模型和已有结果。
- `select_object_split.py`：CPU 只读检查轨迹和 mesh，生成几何多样的训练/测试候选清单。
- `stage_object_meshes.py`：用非破坏性符号链接把选定 mesh 接到仿真任务所需路径。
- `preprocess_graspm3.py`：预处理到指定新目录；默认复现官方末帧筛选。
- `summarize_preprocessed_split.py`：汇总候选物体的官方轨迹保留率并标出需替换对象。
- `finalize_object_split.py`：按同几何组备选规则替换低数据量训练物体并冻结最终清单。
- `prepare_bc_dataset.py`：按整条轨迹切分 BC 训练/验证集，并隔离未见测试物体。
- `train_bc.py`：独立 BC 训练，保存配置、元数据和 checkpoint。
- `evaluate_bc.py`：独立评测；默认使用官方峰值成功率。
- `diagnose_graspm3_replay.py`：只读诊断轨迹抬升、越界和成功时刻。
- `export_experiment_curves.py`：从 TensorBoard/YAML 导出 CSV 和 PNG。
- `summarize_bc_evaluations.py`：汇总多组 BC 的总体、分类别和逐物体成功率。
- `train_residual_ppo.py`：冻结 BC，训练共享有界残差 PPO，保存 reward 分项、PPO 指标和资源统计。
- `evaluate_residual_ppo.py`：按未修改的官方成功标志评测零残差对照或残差策略。
- `evaluate_residual_isolated.py`：每个物体使用独立Python/CUDA进程评测并汇总，避免Isaac Gym反复重建仿真导致非法访问。
- `consolidate_residual_runs.py`：合并checkpoint恢复前后的实际训练分段及独立验证结果，用于连续曲线导出。
- `test_residual_ppo.py`：不创建仿真的 PPO 数学与梯度 CPU 测试。
- `calibrate_residual_reward.py`：冻结 BC、令残差为 0，统计 reward 各分项的均值、标准差、绝对贡献和逐物体 episode 累计值。
- `test_category_advantage_normalization.py`：用不同尺度的合成类别信号验证分类别 advantage 平衡和小噪声保护。
- `test_validation_checkpoint_rule.py`：验证最佳模型按“官方成功率、最大抬升、失败率”的顺序选择。
- `diagnose_residual_reset.py`：用零残差检查跨 episode 局部重置后的高度、失败和重复终止。
- `export_residual_ppo_curves.py`：从残差 PPO 的 CSV 导出 reward、验证成功率/抬升曲线和 reward 分项汇总。
- `test_lift_progress_reward.py`：CPU 验证有符号高度差和历史最高高度进步两种抬升奖励。

指标约定：

- `official_peak`：未修改官方任务代码得到的成功标志，作为默认和正式对比指标。
- `ever`：曾经成功的诊断口径，必须显式指定。
- reward 默认保持官方的 0；稠密诊断 reward 必须显式开启并单独标记。
- 预处理默认 `official_final`；其他筛选口径只用于诊断。
- 任务描述以抬升 30 cm 为目标；官方实现同时接受距目标不超过 12 cm。严格相对抬升 30 cm 单独标为诊断指标。

常用入口（先 `conda activate dexgrasp`，再回到仓库根目录）：

```bash
python custom_tools/preflight_check.py

python custom_tools/preprocess_graspm3.py \
  --object-id <物体编号> \
  --output-root dexgrasp/dataset/train_custom

python custom_tools/prepare_bc_dataset.py

python custom_tools/train_bc.py \
  --config custom_tools/configs/smoke_bc.yaml \
  --run-name <新实验名>

python custom_tools/evaluate_bc.py \
  --object-id <物体编号> \
  --bc-checkpoint <权重路径> \
  --result-tag <结果标签>

python custom_tools/train_residual_ppo.py \
  --config custom_tools/configs/residual_ppo_stage1.yaml \
  --trajectory-selection custom_tools/configs/residual_stage1_trajectory_selection.yaml \
  --run-name <新实验名>

python custom_tools/export_residual_ppo_curves.py \
  --run-dir custom_tools/runs/residual_ppo/<实验名>

python custom_tools/evaluate_residual_ppo.py \
  --zero-residual \
  --object-id <物体编号> \
  --trajectory-root dexgrasp/dataset/object_split_candidates_preprocessed
```

`--init-checkpoint` 只加载网络参数并开始新实验；`--resume-checkpoint` 同时恢复 epoch 和优化器。两者不能同时使用。

残差 PPO 的 reward 只用于训练，结果文件明确标为 `custom_residual_ppo_training_reward`；正式成功率仍来自任务原有的 `successes`。零残差严格配对时，必须同时固定 BC checkpoint、seed、轨迹根目录和轨迹索引。

训练 CSV 对每个 reward 分项同时记录 mean、std 和 absolute contribution fraction，并按类别记录 reward 波动、成功和结束次数；不能只用总 reward 判断权重是否合理。

使用 `--trajectory-selection` 时，训练目录会自动保存 `trajectory_selection.yaml` 副本，避免外部清单后续修改导致旧实验无法复现。

残差PPO支持 `--resume-checkpoint` 恢复网络、Adam优化器、迭代数和全局步数；每次进入验证前先保存周期checkpoint，防止验证仿真异常导致已完成训练丢失。

隔离评测支持 `--use-selection-indices` 按冻结清单逐物体取非连续轨迹；未完成的逐物体结果可直接续跑，单个CUDA子进程默认最多尝试2次。

PPO 按类别对 advantage 去均值，并除以 `max(类别标准差, 1.0)` 后裁剪到 `[-5, 5]`。这样可减弱容易类别的尺度优势，同时不会把困难类别中的微小噪声放大。

第一阶段每 50 轮暂停训练，在同 4 个物体的 28 条留出轨迹上确定性评价。优先按类别宏平均官方成功率保存 `best.pt`；成功率相同时依次比较平均最大抬升（至少相差 1 mm）和失败率。连续 3 次没有改善则早停。验证时先释放训练仿真，验证后再重建，避免两套 Isaac Gym 同时占显存；验证 CSV、周期 checkpoint、`best.pt` 和 `last.pt` 均自动保存。

局部 episode 重置后使用 4 个控制步（约 12 个物理步）使物体重新落稳；期间 reward 和终止信号保持中性并更新高度基准。原因是官方首次 reset 会先推进 10 个物理步，而直接局部重置会把自然落稳误判成掉落。该处理不修改官方成功条件，也不影响单次正式评测。

第一阶段正式 300 轮实验保存在 `custom_tools/runs/residual_ppo/residual_ppo_stage1_history3_seed2025_i300_resetfix_v1/`。按留出集选出的 `best.pt` 为第 150 轮（1/28），`last.pt` 为 0/28；报告曲线位于该目录的 `plots/`，正式比较应使用 `best.pt`。

探索噪声 0.15 的单变量配置为 `custom_tools/configs/residual_ppo_stage1_std015.yaml`。150 轮三次验证均为 0/28，低于噪声 0.25 的对照；该配置作为负结果保留，不作为后续默认设置。

接近奖励权重 5 的配置为 `custom_tools/configs/residual_ppo_stage1_approach5.yaml`。150 轮最佳模型位于第 50 轮（2/28，mug 和 bowl 各 1 条），优于原权重的正式最佳 1/28；结果与曲线保存在 `custom_tools/runs/residual_ppo/residual_ppo_stage1_history3_approach5_seed2025_i150_resetfix_v1/`。

抬升权重 80 的配置为 `custom_tools/configs/residual_ppo_stage1_approach5_lift80.yaml`。最佳仅 1/28，低于权重 40 的 2/28；该配置作为负结果保留，后续仍使用抬升权重 40。

历史最高高度奖励配置为 `custom_tools/configs/residual_ppo_stage1_approach5_maxheight.yaml`。CPU 测试和 Isaac Gym 标定通过，150 轮最佳为 1/28（camera），但总体低于默认有符号高度差的 2/28；结果保存在 `custom_tools/runs/residual_ppo/residual_ppo_stage1_history3_approach5_maxheight_seed2025_i150_resetfix_v1/`。

接近权重 5 的最佳模型在 16 个训练物体各 10 条轨迹上的结果为 9/160（5.625%），低于 warm-start BC 的 12/160（7.5%）。课程内 4 个物体从 4/40 提高到 5/40，课程外 12 个物体从 8/120 降到 4/120，说明当前主要问题是小课程过拟合；结果文件为 `custom_tools/results/evaluations/residual_approach5_best_train16_eval10.yaml`。

16物体扩展课程的轨迹清单位于 `custom_tools/configs/residual_full16_trajectory_selection.yaml`：每个物体从 BC 训练划分中选择2条索引不小于10的轨迹，固定评测索引0～9继续保留，且与 `bc_multicategory_valid` 无重叠。32环境、2轮资源测试保存在 `custom_tools/runs/residual_ppo/residual_ppo_full16_2traj_profile_seed2025_i2_v1/`；2048步约25.5秒，PyTorch 峰值预留显存912 MiB，无NaN或动作饱和。该测试只验证资源与数值稳定性，不作为成功率结论。

16物体正式残差PPO的第150轮模型保存在 `custom_tools/runs/residual_ppo/residual_ppo_full16_2traj_approach5_seed2025_resume100_to150_v1/last.pt`，选模记录为 `custom_tools/results/residual_full16_model_selection.yaml`。严格留出集BC为2/100、残差为8/100；固定16×10配对集BC为13/160、残差为8/160，说明留出泛化提高但仍有基础技能遗忘。连续曲线位于 `custom_tools/results/residual_full16_approach5_consolidated/plots/`。

BC闭环误差诊断命令为 `python custom_tools/diagnose_bc_closed_loop.py`。它先在同一仿真回放专家动作，再分别计算专家观测上的一步动作误差和BC自身观测上的闭环误差；正式结果位于 `custom_tools/results/bc_closed_loop_diagnostic_aligned/`。70帧预处理数据的最后动作槽是全零占位，工具只使用0～68帧有效配对。当前教师状态MAE为0.0072，闭环MAE为0.2648（36.8倍），说明主要瓶颈是闭环分布偏移而非离线拟合不足。

观测噪声BC配置为 `custom_tools/configs/multicategory_bc_noise002.yaml`，只相对正式warm-start开启100维本体观测均匀噪声±0.02。`train_bc.py`现仅在配置缺失时使用`add_noise: false`默认值，不再覆盖显式配置。epoch100权重位于 `custom_tools/runs/bc/multicategory_bc_noise002_seed2025_e100/last.ckpt`：固定集25/160、独立验证集16/100，均优于原BC的13/160与2/100；训练曲线位于 `custom_tools/results/bc_noise002_e100_curves/`，闭环诊断位于 `custom_tools/results/bc_noise002_closed_loop_diagnostic_aligned/`。

门控残差使用 `select_gated_training_trajectories.py` 从完整BC审计中固定16×4条课程。配置 `residual_ppo_noisebc_balanced64_control.yaml` 是28维无门对照，`residual_ppo_noisebc_balanced64_gated.yaml` 额外输出腕部和手指两个门。门属于PPO随机动作的一部分，因而能由策略梯度学习；初始值0.1用于保护冻结BC，正式成功口径不变。

64环境两轮smoke中，无门/门控峰值预留显存为1724/1720 MiB；门控腕部和手指门从约0.108开始正常更新，最大KL 0.0133，无NaN或动作饱和。短smoke只有8步rollout，不用于判断抓取成功率。
