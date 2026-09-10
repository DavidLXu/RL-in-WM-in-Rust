# 世界模型实验

## 三种策略必须分开

1. **Real PPO**：在解析物理仿真器训练。
2. **World-model fine-tuned PPO**：加载 Real PPO 权重，在学到的 dynamics 上继续 PPO。
3. **World-model cold-start PPO**：随机初始化权重，在学到的 dynamics 上训练。

另设 **BC-only** 基线：网络随机初始化，只根据记录的 observation/action 做监督学习。它不加载 Real PPO 权重，但动作标签仍来自 Real PPO；`--updates 0` 时没有任何 world-model PPO 更新，不能把它的成功率当成纯模型 RL 的成绩。

当前结果摘要和原始指标在 [2026-09-10 实验目录](../experiments/2026-09-10-cold-start)。1000 回合确认中 BC20 seed 1 为 82.0%、Real PPO 为 81.5%。0.5 个百分点的差距不足以说明方法稳定优于基线。

## BC 基线命令

先按 README 生成 dataset 和 world model。当前 CLI 的 BC 位于 `--imagine` 中，所以仍要求传入 world model 文件，但 `--updates 0` 不做模型 PPO：

```bash
cargo run --release -p push-cli -- \
  --imagine --mode cold-start --imagine-seed 1 \
  --dataset runs/dataset-4000.json --world-model runs/world-model.json \
  --bc-epochs 20 --bc-samples 120000 --updates 0 \
  --imagined-checkpoint runs/bc20-seed1.json \
  --metrics runs/bc20-metrics.jsonl

cargo run --release -p push-cli -- \
  --evaluate --policy-checkpoint runs/bc20-seed1.json \
  --episodes 1000 --eval-seed 80000 --trajectory runs/eval-bc20.json
```

`--bc-samples` 是上限，当前通过 stride 均匀抽样。旧数据集有 146,761 个 transitions，设上限 120,000 时实际取 73,381 个。新的数据集或 checkpoint 不一定重复旧结果。BC-only 没有逐 epoch 曲线，终端记录样本数和聚合 loss；updates 为 0 的 RL metrics 文件为空。

## 短 rollout 与不确定性

```bash
PUSH_NUM_ENVS=13 cargo run --release -p push-cli -- \
  --imagine --mode finetune --init-policy checkpoints/ppo-1000.json \
  --dataset runs/dataset-4000.json --world-model runs/world-model.json \
  --updates 500 --rollout-steps 30 --imagine-horizon 30 \
  --imagine-starts replay --uncertainty-limit 0.75 \
  --uncertainty-penalty 0 \
  --imagined-checkpoint runs/short30-replay.json \
  --metrics runs/short30-replay-metrics.jsonl
```

30 步是约 2 秒，完整回合是 75 步、约 5 秒。`--rollout-steps` 控制 PPO 每轮每个环境的采样长度，`--imagine-horizon` 控制模型环境重置频率。两者是不同参数。达到 horizon 或 uncertainty limit 会截断，PPO 使用 value bootstrap。`replay` 从数据的不同时间段选择 reset 状态；它不代表模型自己学会了后半段，也不自动降低 dynamics MSE。

`--uncertainty-penalty X` 将 ensemble disagreement 乘 X 从 reward 扣除。当前 guard 的 disagreement 没有校准，不能解释为出错概率。RL metrics 中旧的 `uncertainty_guarded` 字段实际统计未成功结束的片段；精确 guard 计数应看独立 `--evaluate-world` 报告。

## 模型预测与物理结果配对

`--evaluate-world` 采用 dataset 前 N 条 episode 的初始状态，不会自动限制到 world-model test split，适合诊断而不自动构成独立泛化测试。最好用新的 `--data-seed` 单独收集一个评估 dataset。

下面对 README 中 `--data-seed 29000` 的 dataset 做配对诊断：

```bash
cargo run --release -p push-cli -- \
  --evaluate-world --dataset runs/dataset-4000.json \
  --world-model runs/world-model.json \
  --policy-checkpoint runs/cold-start.json \
  --episodes 100 --world-horizon 75 --uncertainty-limit 0.75 \
  --world-trajectory runs/eval-world-cold.json

cargo run --release -p push-cli -- \
  --evaluate --policy-checkpoint runs/cold-start.json \
  --episodes 100 --eval-seed 29000 --trajectory runs/eval-real-paired.json
```

此处两边从同一初始 seed 开始，各自的 policy 根据各自状态产生 action。它与 `--compare-world` 的“共享整条记录动作序列”不同。历史上 random cold-start 在模型里达到 82%，回物理仿真器却为 0%；这类偏差需要检查接触 dynamics 和分布外预测，不能仅靠提高模型内 return 解决。

## 后续实验记录约定

每次实验建立独立的 `runs/<run-id>/`，保存完整命令、环境数量、seed、输入 checkpoint/dataset 的哈希、模型配置和输出文件。至少记录：

- 物理仿真成功率、评估 episode 数和 seed；候选和基线采用相同测试集。
- 模型内成功率、模型 horizon、guard 次数；区分预测性能与真实仿真性能。
- 固定 held-out episodes 上的 1/5/10/30/75 步误差，注明归一化和聚合方法。
- 是否使用示范 action、加载策略权重，以及是否增加了真实环境数据。

本仓库暂未实现自动实验调度、全配置 manifest 或自动 best promotion。现有历史记录是整理后的有限实验矩阵，并不覆盖所有参数组合。
