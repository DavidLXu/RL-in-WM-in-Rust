# 配置与文件格式

从仓库根目录执行 `cargo run --release -p push-cli -- --help` 查看命令。

## CPU 和训练量级

| 模式 | 未设置 `PUSH_NUM_ENVS` 时 |
| --- | --- |
| 物理环境 PPO | `max(available_parallelism - 2, 1)` |
| 物理环境 SAC | 8 个环境，当前 CLI 串行推进 |
| 世界模型 PPO | 4 个环境 |

`PUSH_NUM_ENVS` 控制环境数量，`RAYON_NUM_THREADS` 控制 Rayon 工作线程数。两者不等同于物理核心数。应使用大于 0 的环境数量。跨方法比较时显式固定环境数，例如 `PUSH_NUM_ENVS=13`；在自己的 Mac 上用 `push-bench` 检查合适的数量。

物理 PPO 固定每轮每环境采样 256 个决策步，默认 10 次 update；模型 PPO 默认 500 次 update，采样长度可用 `--rollout-steps` 修改。一次决策包含四个 60 Hz 子步，即 15 Hz 决策频率。默认回合 75 次决策、约 5 秒。

release profile 使用优化级别 3、thin LTO 和一个 codegen unit。当前没有专门的 GPU、Metal 或 MPS 路径，也不依赖手写 NEON 内核。

## SAC

```bash
PUSH_NUM_ENVS=13 cargo run --release -p push-cli -- \
  --algorithm sac --updates 1000 \
  --checkpoint runs/sac.json --metrics runs/sac-metrics.jsonl

cargo run --release -p push-cli -- \
  --evaluate --algorithm sac --policy-checkpoint runs/sac.json \
  --episodes 1000 --eval-seed 80000 --trajectory runs/eval-sac.json
```

SAC 可在物理环境训练、采集轨迹和评估；模型内的 `--imagine` / `--evaluate-world` 目前只接入 PPO。

## Observation 和 action

默认 observation 为 16 维：

| Group | 维数 | 默认启用 |
| --- | ---: | --- |
| `jointAngles` | 2 | 是 |
| `jointVelocities` | 2 | 是 |
| `blockPosition` | 2 | 是 |
| `goalPosition` | 2 | 是 |
| `blockRelative` | 2 | 是 |
| `goalRelative` | 2 | 是 |
| `blockVelocity` | 2 | 是 |
| `blockAngularVelocity` | 1 | 是 |
| `episodeTime` | 1 | 是 |
| `blockOrientation` | 2 | 否 |
| `bias` | 1 | 否 |
| `commandVelocity` | 2 | 否 |

通过 `--obs jointAngles,jointVelocities,blockPosition,goalPosition,...` 选择输入。组的顺序也是网络输入顺序；不要重复组名。布局仅在启动时解析，不进入采样循环。

action 始终是两个连续分量，支持 `--action position`（默认）、`joint-delta`、`absolute`、`velocity` 和 `acceleration`。各模式的控制器定义在 `env/src/lib.rs`。reward 使用内置的接近、推向目标、时间、控制量和成功项，当前没有 reward slider。

checkpoint 不完整保存环境配置。训练、数据采集、评估必须使用相同的 action 模式和 observation 顺序；相同维度并不能保证配置相同。随仓库提供的两个 checkpoint 都使用默认 position 和默认 16 维 observation。

## Artifact

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| PPO/SAC checkpoint JSON | 网络权重、优化器、随机数状态、算法配置 | 评估、采集；PPO 可作模型内微调初始化 |
| Dataset JSON | 完整 episodes；state/action/next_state/reward/end flags | 独立训练 dynamics |
| World model JSON | ensemble、归一化参数、配置与训练指标 | 模型 rollout |
| Evaluation JSON | 每回合状态、动作、reward、成功率等 | 回放和策略质量评估 |
| Comparison JSON | 相同初始状态和动作下的真实/预测状态 | 并排回放 |
| Metrics JSON/JSONL | dynamics 指标或每轮 RL 指标 | 训练曲线 |

世界模型用 18 维完整状态，而 policy 使用所选 observation。完整状态包含关节角/速度、方块位置/速度/朝向/角速度、控制器目标/速度、goal、时间和成功保持时间。

大文件写在被忽略的 `runs/`，手工保留重要 checkpoint 后再做下一轮实验，避免默认输出路径覆盖。`--updates` 不提供物理 RL 续训；模型 fine-tune 通过 `--init-policy` 加载 PPO。

## 当前指标口径

- RL metrics 的 `return` 是该轮采样总 reward，受环境数和 rollout 长度影响，不是平均 episode return。
- `--evaluate` 的 `mean_return` 是平均 episode return，比较策略时更合适。
- `--compare-world` 连续自回归，旧字段 `mean_one_step_mse` 实际聚合了 rollout 各步误差。它不是 teacher-forced 单步 MSE；保留该字段是为了旧 artifact 兼容。
- world training 的归一化 MSE 和原始状态 MSE 不同，比较曲线必须固定指标、数据分区和 horizon。
- CLI 的 dataset `config_hash` 当前是固定标识，还不是完整配置的校验哈希。
