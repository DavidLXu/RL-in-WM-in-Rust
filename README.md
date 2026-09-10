# RL-in-WM-in-Rust

用 Rust 研究 **先在仿真器训练 RL → 采集轨迹 → 训练世界模型 → 对比预测 → 在世界模型里训练 RL**。

任务是固定底座的两关节平面机械臂，把随机位置的方块推到随机目标。训练使用本机 CPU，PPO 通过 Rayon 并行采样；适合 Apple Silicon Mac，也可在 Linux 上构建。无需 Python、CUDA 或外部仿真器。

这里的“真实环境”指解析物理仿真器 `push-env`，不是实体机器人。World model 输入当前完整状态和动作，预测下一状态的增量；动作由 policy 产生。

```text
policy:      observation → action
world model: state + action → next state
```

## 先体验已有策略

安装 stable Rust 工具链后，从仓库根目录执行：

```bash
git clone https://github.com/DavidLXu/RL-in-WM-in-Rust.git
cd RL-in-WM-in-Rust
cargo test --workspace --locked
cargo run --release -p push-cli -- --help

# 随仓库提供的 PPO 策略；生成 100 回合的回放文件
cargo run --release -p push-cli -- \
  --evaluate --policy-checkpoint checkpoints/ppo-1000.json \
  --episodes 100 --eval-seed 80000 --trajectory runs/eval-ppo.json

# macOS：在本地浏览器打开，选择 runs/eval-ppo.json
open visualize.html
```

其他系统直接用浏览器打开 `visualize.html`。页面读取 Rust 导出的文件，可回放策略、并排比较物理仿真和模型预测、查看 JSON/JSONL 训练曲线；页面不执行训练。也可把 checkpoint 换成 `checkpoints/bc20-seed1.json`，体验行为克隆基线。

## 五个独立阶段

每个阶段独立读写文件，产物默认放在被 Git 忽略的 `runs/` 中。下面按标准实验量级执行。`PUSH_NUM_ENVS=13` 是可复现实验设置，并非所有电脑都应使用 13；详见[配置说明](docs/configuration.md)。

### 1. 在物理仿真器训练 PPO

```bash
PUSH_NUM_ENVS=13 cargo run --release -p push-cli -- \
  --updates 1000 \
  --checkpoint runs/ppo-1000.json \
  --metrics runs/ppo-metrics.jsonl
```

每次 update 每个环境采样 256 个决策步。13 个环境 × 1000 次更新约为 333 万 transitions。checkpoint 保存最后一次更新的策略，尚未自动选择验证集 best。真实环境也支持 `--algorithm sac`，见[配置说明](docs/configuration.md)。

### 2. 采集 4000 条完整轨迹

```bash
cargo run --release -p push-cli -- \
  --collect-data --policy-checkpoint runs/ppo-1000.json \
  --episodes 4000 --data-seed 29000 \
  --dataset runs/dataset-4000.json
```

可用 `checkpoints/ppo-1000.json` 跳过阶段 1。数据记录 18 维完整状态、动作、下一状态、reward 和终止标记。每条轨迹最多 75 个决策步；transition 数随提前成功而变化。

### 3. 训练状态预测 world model

```bash
cargo run --release -p push-cli -- \
  --train-world --dataset runs/dataset-4000.json \
  --world-epochs 10 --world-members 3 \
  --world-checkpoint runs/world-model.json \
  --world-metrics runs/world-metrics.json
```

标准配置为 **4000 条轨迹 / 10 epochs / 3 个 ensemble members**。按完整 episode 切分 train/validation/test，避免同一轨迹出现在不同分区。模型学习状态增量，目标、时间和成功保持时间等字段按规则保留或更新；reward 在模型环境中按预测状态解析计算。

### 4. 并排比较 world model 与仿真器

```bash
cargo run --release -p push-cli -- \
  --compare-world --dataset runs/dataset-4000.json \
  --world-model runs/world-model.json \
  --comparison runs/world-compare.json
```

打开 `visualize.html`，在“真实 vs world model”中加载 `runs/world-compare.json`。两边从相同初始状态开始，执行记录中的相同动作；模型端连续使用自己的预测状态，不在每一步接回真实状态。页面显示方块误差、末端误差和 ensemble disagreement。

在“训练 metrics”中可加载 `runs/ppo-metrics.jsonl`、`runs/world-metrics.json` 或阶段 5 的 metrics。大数据集生成的对比文件也可能很大；交互预览时可额外采集一个较小的 dataset。

### 5. 在 world model 中训练 PPO

从阶段 1 策略微调：

```bash
PUSH_NUM_ENVS=13 cargo run --release -p push-cli -- \
  --imagine --mode finetune --init-policy runs/ppo-1000.json \
  --dataset runs/dataset-4000.json --world-model runs/world-model.json \
  --updates 500 \
  --imagined-checkpoint runs/finetuned.json \
  --metrics runs/finetuned-metrics.jsonl
```

随机初始化策略，在模型中从零训练：

```bash
PUSH_NUM_ENVS=13 cargo run --release -p push-cli -- \
  --imagine --mode cold-start --imagine-seed 777 \
  --dataset runs/dataset-4000.json --world-model runs/world-model.json \
  --updates 500 \
  --imagined-checkpoint runs/cold-start.json \
  --metrics runs/cold-start-metrics.jsonl
```

最终必须回到物理仿真器评估，两个 checkpoint 使用相同 episode 数和 seed：

```bash
cargo run --release -p push-cli -- \
  --evaluate --policy-checkpoint runs/finetuned.json \
  --episodes 1000 --eval-seed 80000 --trajectory runs/eval-finetuned.json

cargo run --release -p push-cli -- \
  --evaluate --policy-checkpoint runs/cold-start.json \
  --episodes 1000 --eval-seed 80000 --trajectory runs/eval-cold-start.json
```

短 rollout、replay 起点、uncertainty guard/penalty 和 BC 初始化的命令见[实验说明](docs/experiments.md)。`--world-model` 仍有旧的一键兼容入口；正式实验使用以上明确的阶段参数。

## 已有实验结果

| 策略 | 物理仿真器成功率 | 评估规模 |
| --- | ---: | ---: |
| 随仓库提供的阶段 1 PPO | 81.5% | 1000 回合 |
| 随仓库提供的 BC20 seed 1 | 82.0% | 同一批 1000 回合 |
| 随机初始化、world-model PPO | 0% | 100 回合 |

**82% 是行为克隆基线：使用示范动作标签，模型 PPO updates 为 0。它不是纯 world-model cold-start RL 的成果。** 目前没有证据表明模型内训练超过了物理仿真器训练。随机 cold-start 曾出现模型内预测成功率 82%、回物理仿真器却为 0% 的情况。

完整的精简记录、评估口径、checkpoint 哈希和确认结果见 [experiments/2026-09-10-cold-start](experiments/2026-09-10-cold-start)。原始大数据集和全量回放未随仓库上传，可用独立命令重新生成。

## 代码结构

```text
env/                  确定性 2D 接触后端和 observation 编码
rl/                   MLP、Adam、PPO、SAC
world/                数据集、episode 切分、ensemble dynamics
cli/src/main.rs       独立阶段的命令入口
cli/src/adapters.rs   物理环境和模型环境的 PPO 接口
cli/src/imitation.rs  行为克隆初始化
bench/                物理与 PPO 端到端吞吐测试
checkpoints/          两份小型示例策略
experiments/          已记录的实验摘要和指标
docs/                 配置和实验说明
scripts/smoke.sh       全流程烟雾测试
visualize.html        Rust 产物的本地回放器
```

## 开发与验证

```bash
cargo fmt --all -- --check
cargo test --workspace --locked
bash scripts/smoke.sh
cargo run --release -p push-bench
```

CI 在 macOS 和 Linux 上执行格式检查、单元测试和缩小规模的全流程测试。`push-bench` 测吞吐，`--evaluate` 测策略质量；两者不能混用。PPO 采用 Rayon 并行环境采样，SAC 的当前 CLI 采样循环为串行。网络是手写 MLP/Adam，尚未接入 Metal、MPS 或 GPU 后端。

## 当前边界

- 物理后端用圆盘近似方块接触，不是高精度刚体引擎。
- 模型内训练只支持 PPO，SAC 目前用于物理仿真器训练。
- checkpoint 尚未完整保存 action/observation 配置；训练、采集和评估必须保持同一配置。
- CLI 没有真实环境验证后的自动 best promotion，`--updates` 也不表示从上次训练继续。
- ensemble disagreement 不是经过校准的正确率，模型内高回报不能替代真实仿真评估。
