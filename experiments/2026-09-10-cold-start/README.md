# Cold-start 实验记录 · 2026-09-10

此目录保留历史实验的轻量记录，包含 19 个配置结果和 2 个扩大评估规模的确认记录；没有上传大数据集或完整 rollout。

| 文件 | 内容 |
| --- | --- |
| `coldstart-experiment-log.csv` / `.json` | 方法、BC 轮数、模型 PPO updates、horizon、penalty、成功率 |
| `confirmation-metrics.json` | 历史 1000 回合评估摘要与原始输出 SHA-256 |
| `standalone-confirmation.json` | 拆分为独立仓库后重新评估两个示例 checkpoint |
| `world-vs-real-success-summary.csv` / `.json` | 相同初始种子的模型预测/物理结果诊断 |
| `provenance.json` | 源码快照来源与示例 checkpoint SHA-256 |

BC-only 是 imitation baseline，使用 demonstrator 的 action 标签，模型 PPO updates 为 0。短 rollout 的三项实验是从 Real PPO 权重 fine-tune，不能混入随机 cold-start 的结果。

主矩阵 `world_success_100` 使用从 29000 起的 dataset starts；`real_success_100/500/1000` 使用从 80000 起的 seed，两列不是逐 episode 配对。独立的 `world-vs-real-success-summary` 才是配对诊断。不要直接把主矩阵的差值当配对误差。

新仓库整理时修正了两条 1000 回合确认行误填的 `real_success_500`，将其置空，并把短 rollout 行明确标为 fine-tune。原始未记录的参数保留为空，不补猜值。部分行的 `mse30` 对应另外采集数据重新训练后的 dynamics，不是该行 policy 训练所用固定模型的误差。

随仓库提供 BC20 seed 1 和 Real PPO 两份 checkpoint，可重新执行评估。其他历史 checkpoint、大型 dataset 和 world model 没有打包。历史 100/500/1000 回合分别采用同一 seed 范围的前缀，因此扩大评估并非全新的独立测试集。
