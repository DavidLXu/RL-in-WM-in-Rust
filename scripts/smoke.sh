#!/usr/bin/env bash
# Exercise every explicit stage in a fresh temporary artifact directory.
# This checks pipeline wiring, not learning performance.
set -euo pipefail
cd "$(dirname "$0")/.."
smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/push-rust-smoke.XXXXXX")"
trap 'rm -rf "$smoke_dir"' EXIT
export PUSH_NUM_ENVS=2
cargo build --release -p push-cli --locked
cli=./target/release/push-cli
"$cli" --help
"$cli" --updates 1 --checkpoint "$smoke_dir/ppo.json" --metrics "$smoke_dir/ppo.jsonl"
"$cli" --evaluate --policy-checkpoint "$smoke_dir/ppo.json" --episodes 3 --trajectory "$smoke_dir/eval.json"
"$cli" --collect-data --policy-checkpoint "$smoke_dir/ppo.json" --episodes 6 --dataset "$smoke_dir/data.json"
"$cli" --train-world --dataset "$smoke_dir/data.json" --world-epochs 1 --world-members 2 --world-checkpoint "$smoke_dir/world.json" --world-metrics "$smoke_dir/world-metrics.json"
"$cli" --compare-world --dataset "$smoke_dir/data.json" --world-model "$smoke_dir/world.json" --comparison "$smoke_dir/compare.json"
"$cli" --evaluate-world --dataset "$smoke_dir/data.json" --world-model "$smoke_dir/world.json" --policy-checkpoint "$smoke_dir/ppo.json" --episodes 3 --world-horizon 5 --world-trajectory "$smoke_dir/eval-world.json"
"$cli" --imagine --mode cold-start --dataset "$smoke_dir/data.json" --world-model "$smoke_dir/world.json" --updates 1 --rollout-steps 8 --imagine-horizon 5 --bc-epochs 1 --bc-samples 64 --imagined-checkpoint "$smoke_dir/cold.json" --metrics "$smoke_dir/cold.jsonl"
"$cli" --imagine --mode finetune --init-policy "$smoke_dir/ppo.json" --dataset "$smoke_dir/data.json" --world-model "$smoke_dir/world.json" --updates 1 --rollout-steps 8 --imagine-horizon 5 --imagined-checkpoint "$smoke_dir/fine.json" --metrics "$smoke_dir/fine.jsonl"
"$cli" --algorithm sac --updates 1 --checkpoint "$smoke_dir/sac.json" --metrics "$smoke_dir/sac.jsonl"
"$cli" --evaluate --algorithm sac --policy-checkpoint "$smoke_dir/sac.json" --episodes 3 --trajectory "$smoke_dir/eval-sac.json"
printf 'Pipeline smoke checks passed.\n'
