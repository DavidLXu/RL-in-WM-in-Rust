# 示例策略

| 文件 | 来源 | 确认结果 |
| --- | --- | --- |
| `ppo-1000.json` | 解析物理仿真器中的 PPO | 1000 回合成功 815 次 |
| `bc20-seed1.json` | 随机初始化后用示范动作做 BC20，模型 PPO updates 为 0 | 同一批 1000 回合成功 820 次 |

两者都使用默认 16 维 observation 和 `position` action。评估 seeds 为 80000 到 80999。它们是小型 JSON checkpoint，包含网络、优化器和随机数状态，没有数据集或个人信息。

```bash
cargo run --release -p push-cli -- \
  --evaluate --policy-checkpoint checkpoints/bc20-seed1.json \
  --episodes 1000 --eval-seed 80000 --trajectory runs/eval-bc.json
```

用仓库根目录的 `visualize.html` 加载生成的 trajectory JSON。页面不直接解析网络 checkpoint。

来源和 SHA-256 记录在 [provenance.json](../experiments/2026-09-10-cold-start/provenance.json)。这两份 checkpoint 在独立仓库整理后再次进行 1000 回合确认，结果见该实验目录的 `standalone-confirmation.json`。
