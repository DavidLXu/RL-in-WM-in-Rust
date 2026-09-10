//! Independent stages for simulator RL, datasets, learned dynamics and evaluation.
mod adapters;
mod imitation;

use adapters::{RealEnv, WorldEnv};
use imitation::behavior_clone_policy;

use push_env::{decode_state, encode_state, ActionMode, Env, EnvConfig, ObsLayout, STATE_DIM};
use push_rl::{Agent, Environment, Ppo, PpoConfig, Sac, SacConfig};
use push_world::{Dataset, Episode, Transition, WorldConfig, WorldModel};
use serde_json::json;
use std::fs;
use std::io::Write;

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|x| x == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}
fn arg_u64(args: &[String], name: &str, default: u64) -> u64 {
    args.iter()
        .position(|x| x == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}
fn string_arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|x| x == name)
        .and_then(|i| args.get(i + 1))
        .map(String::clone)
        .unwrap_or_else(|| default.to_owned())
}
fn checkpoint(args: &[String]) -> String {
    args.iter()
        .position(|x| x == "--checkpoint")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "runs/ppo.json".into())
}
fn policy_checkpoint(args: &[String]) -> String {
    args.iter()
        .position(|x| x == "--policy-checkpoint")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "runs/ppo.json".into())
}
fn dataset_path(args: &[String]) -> String {
    string_arg(args, "--dataset", "runs/dataset.json")
}
fn collect_dataset(
    episodes: usize,
    seed: u64,
    env_config: EnvConfig,
    layout: &ObsLayout,
    policy: &mut Ppo,
) -> Dataset {
    collect_dataset_with_action(episodes, seed, env_config, layout, |obs| policy.action(obs))
}
fn collect_dataset_with_action<F>(
    episodes: usize,
    seed: u64,
    env_config: EnvConfig,
    layout: &ObsLayout,
    mut choose_action: F,
) -> Dataset
where
    F: FnMut(&[f32]) -> [f32; 2],
{
    let mut all = Vec::with_capacity(episodes);
    for episode_id in 0..episodes {
        let mut env = Env::with_config(seed + episode_id as u64, env_config);
        let episode_seed = seed + episode_id as u64;
        let mut transitions = Vec::new();
        for _ in 0..env_config.max_decisions {
            let before = encode_state(&env.full_state());
            let mut obs = vec![0.0; layout.dim];
            env.observe_layout(layout, &mut obs);
            let action = choose_action(&obs);
            let result = env.step_result(action);
            let after = encode_state(&env.full_state());
            transitions.push(Transition {
                state: before.to_vec(),
                action,
                next_state: after.to_vec(),
                reward: result.reward,
                terminated: result.terminated,
                truncated: result.truncated,
            });
            if result.terminated || result.truncated {
                break;
            }
        }
        all.push(Episode {
            id: episode_id as u64,
            seed: episode_seed,
            transitions,
        });
    }
    Dataset {
        version: "push-state-dataset-v1".into(),
        config_hash: "rust-env-v2-default-obs16-position".into(),
        state_dim: STATE_DIM,
        episodes: all,
    }
}
fn run_imagined_ppo(model: &WorldModel, dataset: &Dataset, layout: &ObsLayout) {
    let starts: Vec<Vec<f32>> = dataset
        .episodes
        .iter()
        .filter_map(|e| e.transitions.first().map(|t| t.state.clone()))
        .collect();
    if starts.is_empty() {
        return;
    }
    let mut envs = (0..4)
        .map(|i| {
            WorldEnv::new(
                model.clone(),
                vec![
                    starts[i % starts.len()].clone(),
                    starts[(i + 1) % starts.len()].clone(),
                ],
                layout.clone(),
                32,
                0.75,
            )
        })
        .collect::<Vec<_>>();
    let mut p = Ppo::new(
        layout.dim,
        PpoConfig {
            epochs: 2,
            minibatch_size: 64,
            ..Default::default()
        },
        777,
    );
    let stats = p.train(&mut envs, 32);
    println!("world_model_ppo imagined_steps={} imagined_return={:.3} uncertainty_guarded_episodes={} real_validation_required=true",stats.steps,stats.reward_sum,stats.episodes);
}

fn run_sac(args: &[String], env_config: EnvConfig, layout: &ObsLayout) {
    let updates = arg(args, "--updates", 10);
    let metrics_path = string_arg(args, "--metrics", "runs/sac-metrics.jsonl");
    reset_metrics(&metrics_path);
    let env_count = std::env::var("PUSH_NUM_ENVS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(8);
    let mut config = SacConfig::default();
    config.obs_dim = layout.dim;
    config.batch_size = 64;
    config.replay_capacity = 100_000;
    config.warmup_steps = (env_count * 64) as u64;
    let mut sac = Sac::new(config, 42);
    let mut envs = (0..env_count)
        .map(|i| Env::with_config(i as u64, env_config))
        .collect::<Vec<_>>();
    let mut obs = vec![vec![0.0; layout.dim]; env_count];
    for (e, o) in envs.iter().zip(&mut obs) {
        e.observe_layout(layout, o);
    }
    for update in 0..updates {
        let mut raw = 0.0;
        let mut completed = 0;
        for i in 0..env_count * 256 {
            let lane = i % env_count;
            let action = sac.action(&obs[lane], false);
            let result = envs[lane].step_result(action);
            let mut next = vec![0.0; layout.dim];
            envs[lane].observe_layout(layout, &mut next);
            raw += result.reward;
            sac.observe(
                &obs[lane],
                action,
                result.reward,
                &next,
                result.terminated,
                result.truncated,
            );
            obs[lane] = next;
            if result.terminated || result.truncated {
                completed += 1;
                envs[lane].reset();
                envs[lane].observe_layout(layout, &mut obs[lane]);
            }
            let _ = sac.update();
        }
        println!(
            "sac_update={} env_steps={} return={:.3} episodes={} replay={} alpha={:.4}",
            update + 1,
            sac.environment_steps,
            raw,
            completed,
            sac.replay_len(),
            sac.alpha()
        );
        append_metric(
            &metrics_path,
            json!({
                "stage": "train-real",
                "algorithm": "sac",
                "update": update + 1,
                "env_steps": sac.environment_steps,
                "return": raw,
                "episodes": completed,
                "replay": sac.replay_len(),
                "alpha": sac.alpha()
            }),
        );
    }
    let path = checkpoint(args).replace("ppo", "sac");
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(
        &path,
        serde_json::to_string(&sac).expect("serialize SAC checkpoint"),
    )
    .expect("write SAC checkpoint");
    println!("checkpoint={path}");
}
fn evaluate_policy<F>(
    episodes: usize,
    seed: u64,
    env_config: EnvConfig,
    layout: &ObsLayout,
    checkpoint_path: &str,
    algorithm: &str,
    mut action: F,
) -> serde_json::Value
where
    F: FnMut(&[f32]) -> [f32; 2],
{
    let mut records = Vec::with_capacity(episodes);
    let mut successes = 0usize;
    let mut returns = 0.0;
    let mut steps_sum = 0usize;
    let mut final_distance = 0.0;
    let mut min_distance = 0.0;
    for episode in 0..episodes {
        let episode_seed = seed + episode as u64;
        let mut env = Env::with_config(episode_seed, env_config);
        let mut obs = vec![0.0; layout.dim];
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut distances = Vec::new();
        let mut score = 0.0;
        let mut success = false;
        for _ in 0..env_config.max_decisions {
            states.push(json!(encode_state(&env.full_state())));
            env.observe_layout(layout, &mut obs);
            let a = action(&obs);
            actions.push(json!([a[0], a[1]]));
            let r = env.step_result(a);
            score += r.reward;
            rewards.push(r.reward);
            distances.push(env.full_state().block.dist(env.full_state().goal));
            if r.success {
                success = true;
            }
            if r.terminated || r.truncated {
                break;
            }
        }
        let steps = actions.len();
        let end_distance = distances.last().copied().unwrap_or(f32::INFINITY);
        let min_d = distances.iter().copied().fold(f32::INFINITY, f32::min);
        if success {
            successes += 1;
        }
        returns += score;
        steps_sum += steps;
        final_distance += end_distance;
        min_distance += min_d;
        records.push(json!({"seed":episode_seed,"success":success,"return":score,"steps":steps,"final_distance":end_distance,"min_distance":min_d,"states":states,"actions":actions,"rewards":rewards}));
    }
    let n = episodes.max(1) as f32;
    json!({"format":"push-rust-eval-v1","checkpoint":checkpoint_path,"algorithm":algorithm,"observation_dim":layout.dim,"episodes":records,"metrics":{"episodes":episodes,"success_rate":successes as f32/n,"mean_return":returns/n,"mean_steps":steps_sum as f32/n,"mean_final_distance":final_distance/n,"mean_min_distance":min_distance/n}})
}
fn run_evaluate(args: &[String], env_config: EnvConfig, layout: &ObsLayout) {
    let path = policy_checkpoint(args);
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read checkpoint {path}: {e}"));
    let episodes = arg(args, "--episodes", 100).max(1);
    let seed = arg_u64(args, "--eval-seed", 80000);
    let output = string_arg(args, "--trajectory", "runs/eval.json");
    let is_sac = args.iter().any(|x| x == "sac");
    let report = if is_sac {
        let mut p: Sac = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("invalid SAC checkpoint {path}: {e}"));
        if p.config.obs_dim != layout.dim {
            panic!(
                "checkpoint observation dimension {} does not match selected layout {}",
                p.config.obs_dim, layout.dim
            );
        }
        evaluate_policy(episodes, seed, env_config, layout, &path, "sac", |o| {
            p.action(o, true)
        })
    } else {
        let mut p =
            Ppo::load_json(&text).unwrap_or_else(|e| panic!("invalid PPO checkpoint {path}: {e}"));
        if p.actor.input_dim() != layout.dim {
            panic!(
                "checkpoint observation dimension {} does not match selected layout {}",
                p.actor.input_dim(),
                layout.dim
            );
        }
        evaluate_policy(episodes, seed, env_config, layout, &path, "ppo", |o| {
            p.deterministic_action(o)
        })
    };
    if let Some(parent) = std::path::Path::new(&output).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&output, serde_json::to_string_pretty(&report).unwrap())
        .expect("write evaluation report");
    let m = &report["metrics"];
    println!("evaluation checkpoint={} algorithm={} episodes={} success_rate={:.3} mean_return={:.3} mean_steps={:.1} mean_final_distance={:.4} mean_min_distance={:.4} trajectory={}",path,report["algorithm"].as_str().unwrap(),m["episodes"],m["success_rate"],m["mean_return"],m["mean_steps"],m["mean_final_distance"],m["mean_min_distance"],output);
}

/// Evaluate a policy entirely inside a learned world model.
///
/// The initial states are taken from the supplied dataset, so this is a fixed
/// start-state comparison rather than a real-environment benchmark.  No calls
/// to `Env::step_result` are made after the initial state is loaded.
fn evaluate_world_policy<F>(
    dataset_path: &str,
    model_path: &str,
    checkpoint_path: &str,
    episodes: usize,
    horizon: usize,
    uncertainty_limit: f32,
    layout: &ObsLayout,
    dataset: &Dataset,
    model: &WorldModel,
    mut action: F,
) -> serde_json::Value
where
    F: FnMut(&[f32]) -> [f32; 2],
{
    let selected = dataset.episodes.iter().take(episodes);
    let mut records = Vec::new();
    let mut successes = 0usize;
    let mut returns = 0.0f32;
    let mut steps_sum = 0usize;
    let mut final_distance = 0.0f32;
    let mut min_distance = 0.0f32;
    let mut uncertainty_guarded_episodes = 0usize;
    let mut uncertainty_guarded_steps = 0usize;
    let mut horizon_truncated_episodes = 0usize;
    let mut terminated_episodes = 0usize;

    for episode in selected {
        let Some(first) = episode.transitions.first() else {
            continue;
        };
        let mut env = WorldEnv::new(
            model.clone(),
            vec![first.state.clone()],
            layout.clone(),
            horizon,
            uncertainty_limit,
        );
        let mut obs = vec![0.0; layout.dim];
        let mut states = vec![first.state.clone()];
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut distances = Vec::new();
        let mut score = 0.0f32;
        let mut success = false;
        let mut uncertainty_guarded = false;
        let mut horizon_truncated = false;
        let mut terminated = false;

        for _ in 0..horizon {
            env.observe(&mut obs);
            let a = action(&obs);
            actions.push(json!([a[0], a[1]]));
            let result = env.step_env(a);
            score += result.reward;
            rewards.push(result.reward);
            states.push(env.state.clone());
            let current = decode_state(env.state.clone().try_into().expect("state dim"));
            distances.push(current.block.dist(current.goal));
            uncertainty_guarded_steps += usize::from(env.last_uncertainty_guarded);
            uncertainty_guarded |= env.last_uncertainty_guarded;
            horizon_truncated |= env.last_horizon_truncated;
            if result.success {
                success = true;
            }
            if result.terminated || result.truncated {
                terminated = result.terminated;
                break;
            }
        }

        let steps = actions.len();
        let end_distance = distances.last().copied().unwrap_or(f32::INFINITY);
        let min_d = distances.iter().copied().fold(f32::INFINITY, f32::min);
        successes += usize::from(success);
        returns += score;
        steps_sum += steps;
        final_distance += end_distance;
        min_distance += min_d;
        uncertainty_guarded_episodes += usize::from(uncertainty_guarded);
        horizon_truncated_episodes += usize::from(horizon_truncated);
        terminated_episodes += usize::from(terminated);
        records.push(json!({
            "id": episode.id,
            "seed": episode.seed,
            "success": success,
            "return": score,
            "steps": steps,
            "final_distance": end_distance,
            "min_distance": min_d,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "model_uncertainty_guarded": uncertainty_guarded,
            "horizon_truncated": horizon_truncated,
        }));
    }

    let n = records.len().max(1) as f32;
    json!({
        "format": "push-world-eval-v1",
        "dataset": dataset_path,
        "world_model": model_path,
        "checkpoint": checkpoint_path,
        "algorithm": "ppo",
        "observation_dim": layout.dim,
        "horizon": horizon,
        "uncertainty_limit": uncertainty_limit,
        "episodes": records,
        "metrics": {
            "episodes": records.len(),
            "success_rate": successes as f32 / n,
            "mean_return": returns / n,
            "mean_steps": steps_sum as f32 / n,
            "mean_final_distance": final_distance / n,
            "mean_min_distance": min_distance / n,
            "terminated_episodes": terminated_episodes,
            "horizon_truncated_episodes": horizon_truncated_episodes,
            "uncertainty_guarded_episodes": uncertainty_guarded_episodes,
            "uncertainty_guarded_steps": uncertainty_guarded_steps,
        }
    })
}

fn run_evaluate_world(args: &[String], layout: &ObsLayout) {
    let input = dataset_path(args);
    let dataset = load_dataset(&input);
    let model_path = string_arg(args, "--world-model", "runs/world-model.json");
    let model_text = fs::read_to_string(&model_path)
        .unwrap_or_else(|e| panic!("cannot read world model {model_path}: {e}"));
    let model: WorldModel = serde_json::from_str(&model_text)
        .unwrap_or_else(|e| panic!("invalid world model {model_path}: {e}"));
    if model.state_dim != dataset.state_dim {
        panic!(
            "world model state dimension {} does not match dataset {}",
            model.state_dim, dataset.state_dim
        );
    }
    let path = policy_checkpoint(args);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read policy checkpoint {path}: {e}"));
    let episodes = arg(args, "--episodes", dataset.episodes.len()).max(1);
    let horizon = arg(args, "--world-horizon", 75).max(1);
    let uncertainty_limit = args
        .iter()
        .position(|x| x == "--uncertainty-limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse::<f32>().ok())
        .unwrap_or(0.75)
        .max(0.0);
    let mut policy = Ppo::load_json(&text)
        .unwrap_or_else(|e| panic!("invalid PPO policy checkpoint {path}: {e}"));
    if policy.actor.input_dim() != layout.dim {
        panic!(
            "checkpoint observation dimension {} does not match selected layout {}",
            policy.actor.input_dim(),
            layout.dim
        );
    }
    let report = evaluate_world_policy(
        &input,
        &model_path,
        &path,
        episodes,
        horizon,
        uncertainty_limit,
        layout,
        &dataset,
        &model,
        |o| policy.deterministic_action(o),
    );
    let output = string_arg(args, "--world-trajectory", "runs/eval-world.json");
    save_json_file(&output, &report);
    let m = &report["metrics"];
    println!("world_evaluation checkpoint={} world_model={} episodes={} success_rate={:.3} mean_return={:.3} mean_steps={:.1} mean_final_distance={:.4} mean_min_distance={:.4} uncertainty_guarded_episodes={} uncertainty_guarded_steps={} horizon_truncated_episodes={} trajectory={}", path, model_path, m["episodes"], m["success_rate"], m["mean_return"], m["mean_steps"], m["mean_final_distance"], m["mean_min_distance"], m["uncertainty_guarded_episodes"], m["uncertainty_guarded_steps"], m["horizon_truncated_episodes"], output);
}
fn save_json_file(path: &str, value: &impl serde::Serialize) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(
        path,
        serde_json::to_string_pretty(value).expect("serialize artifact"),
    )
    .unwrap_or_else(|e| panic!("write artifact {path}: {e}"));
}
fn reset_metrics(path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, "").unwrap_or_else(|e| panic!("cannot initialize metrics {path}: {e}"));
}
fn append_metric(path: &str, value: serde_json::Value) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("cannot open metrics {path}: {e}"));
    writeln!(
        file,
        "{}",
        serde_json::to_string(&value).expect("serialize metric")
    )
    .unwrap_or_else(|e| panic!("cannot write metrics {path}: {e}"));
}
fn load_dataset(path: &str) -> Dataset {
    let text =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read dataset {path}: {e}"));
    let dataset: Dataset =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("invalid dataset {path}: {e}"));
    dataset
        .validate()
        .unwrap_or_else(|e| panic!("invalid dataset {path}: {e}"));
    dataset
}
fn run_collect_data(args: &[String], env_config: EnvConfig, layout: &ObsLayout) {
    let episodes = arg(args, "--episodes", 1000).max(3);
    let seed = arg_u64(args, "--data-seed", 9000);
    let policy_path = policy_checkpoint(args);
    let text = fs::read_to_string(&policy_path)
        .unwrap_or_else(|e| panic!("cannot read policy checkpoint {policy_path}: {e}"));
    let algorithm = if args.iter().any(|x| x == "sac") {
        "sac"
    } else {
        "ppo"
    };
    let dataset = if algorithm == "sac" {
        let mut policy: Sac = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("invalid SAC checkpoint {policy_path}: {e}"));
        if policy.config.obs_dim != layout.dim {
            panic!(
                "checkpoint observation dimension {} does not match selected layout {}",
                policy.config.obs_dim, layout.dim
            );
        }
        collect_dataset_with_action(episodes, seed, env_config, layout, |obs| {
            policy.action(obs, false)
        })
    } else {
        let mut policy = Ppo::load_json(&text)
            .unwrap_or_else(|e| panic!("invalid PPO checkpoint {policy_path}: {e}"));
        if policy.actor.input_dim() != layout.dim {
            panic!(
                "checkpoint observation dimension {} does not match selected layout {}",
                policy.actor.input_dim(),
                layout.dim
            );
        }
        collect_dataset(episodes, seed, env_config, layout, &mut policy)
    };
    let output = dataset_path(args);
    save_json_file(&output, &dataset);
    println!("dataset policy_checkpoint={} algorithm={} episodes={} transitions={} state_dim={} dataset={}", policy_path, algorithm, dataset.episodes.len(), dataset.transitions(), dataset.state_dim, output);
}
fn train_world_model(dataset: &Dataset, args: &[String]) -> WorldModel {
    let mut config = WorldConfig::default();
    config.epochs = arg(args, "--world-epochs", config.epochs);
    config.members = arg(args, "--world-members", config.members).max(2);
    WorldModel::train(dataset, config, arg_u64(args, "--world-seed", 1234))
        .unwrap_or_else(|e| panic!("world model training failed: {e}"))
}
fn write_world_model(model: &WorldModel, path: &str) {
    save_json_file(path, model);
    println!("world_model train_mse={:.6} val_mse={:.6} test_1step_mse={:.6} test_5step_mse={:.6} checkpoint={}", model.metrics.train_normalized_mse, model.metrics.validation_normalized_mse, model.metrics.test_one_step_mse, model.metrics.test_five_step_mse, path);
}
fn run_train_world(args: &[String]) {
    let input = dataset_path(args);
    let dataset = load_dataset(&input);
    let model = train_world_model(&dataset, args);
    let output = string_arg(args, "--world-checkpoint", "runs/world-model.json");
    write_world_model(&model, &output);
    let metrics = string_arg(args, "--world-metrics", "runs/world-metrics.json");
    save_json_file(
        &metrics,
        &json!({
            "stage": "train-world",
            "dataset": input,
            "checkpoint": output,
            "train_mse": model.metrics.train_normalized_mse,
            "validation_mse": model.metrics.validation_normalized_mse,
            "test_one_step_mse": model.metrics.test_one_step_mse,
            "test_five_step_mse": model.metrics.test_five_step_mse,
            "train_episodes": model.metrics.train_episodes,
            "validation_episodes": model.metrics.validation_episodes,
            "test_episodes": model.metrics.test_episodes
        }),
    );
    println!("world_metrics={metrics}");
}
fn run_compare_world(args: &[String]) {
    let dataset = load_dataset(&dataset_path(args));
    let model_path = string_arg(args, "--world-model", "runs/world-model.json");
    let model_text = fs::read_to_string(&model_path)
        .unwrap_or_else(|e| panic!("cannot read world model {model_path}: {e}"));
    let model: WorldModel = serde_json::from_str(&model_text)
        .unwrap_or_else(|e| panic!("invalid world model {model_path}: {e}"));
    if model.state_dim != dataset.state_dim {
        panic!(
            "world model state dimension {} does not match dataset {}",
            model.state_dim, dataset.state_dim
        );
    }
    let mut ws = model.workspace();
    let mut episodes = Vec::with_capacity(dataset.episodes.len());
    let mut one_step_sq = 0.0f32;
    let mut count = 0usize;
    for episode in &dataset.episodes {
        let mut predicted = Vec::with_capacity(episode.transitions.len() + 1);
        let mut uncertainties = Vec::with_capacity(episode.transitions.len());
        if let Some(first) = episode.transitions.first() {
            predicted.push(first.state.clone());
        }
        for t in &episode.transitions {
            let mut next = vec![0.0; model.state_dim];
            // Roll the model forward from its own previous prediction. This
            // makes the comparison expose compounding multi-step error rather
            // than hiding it behind teacher forcing with the real state.
            let model_input = predicted.last().unwrap_or(&t.state);
            let uncertainty = model.predict(model_input, t.action, &mut next, None, &mut ws);
            one_step_sq += next
                .iter()
                .zip(&t.next_state)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / model.state_dim as f32;
            count += 1;
            predicted.push(next);
            uncertainties.push(uncertainty);
        }
        let mut real_states = Vec::with_capacity(episode.transitions.len() + 1);
        if let Some(first) = episode.transitions.first() {
            real_states.push(first.state.clone());
        }
        real_states.extend(episode.transitions.iter().map(|t| t.next_state.clone()));
        episodes.push(json!({"id":episode.id,"seed":episode.seed,"real_states":real_states,"predicted_states":predicted,"actions":episode.transitions.iter().map(|t| t.action).collect::<Vec<_>>(),"rewards":episode.transitions.iter().map(|t| t.reward).collect::<Vec<_>>(),"uncertainties":uncertainties}));
    }
    let output = string_arg(args, "--comparison", "runs/world-compare.json");
    let report = json!({"format":"push-world-compare-v1","dataset":dataset_path(args),"world_model":model_path,"episodes":episodes,"metrics":{"episodes":dataset.episodes.len(),"transitions":count,"mean_one_step_mse":one_step_sq / count.max(1) as f32}});
    save_json_file(&output, &report);
    println!("world_compare dataset={} world_model={} episodes={} transitions={} mean_one_step_mse={:.6} comparison={}", dataset_path(args), model_path, dataset.episodes.len(), count, one_step_sq / count.max(1) as f32, output);
}
fn run_imagine(args: &[String], layout: &ObsLayout) {
    let dataset = load_dataset(&dataset_path(args));
    let model_path = string_arg(args, "--world-model", "runs/world-model.json");
    let model_text = fs::read_to_string(&model_path)
        .unwrap_or_else(|e| panic!("cannot read world model {model_path}: {e}"));
    let model: WorldModel = serde_json::from_str(&model_text)
        .unwrap_or_else(|e| panic!("invalid world model {model_path}: {e}"));
    let start_mode = string_arg(args, "--imagine-starts", "episode-start");
    let starts: Vec<Vec<f32>> = if start_mode == "replay" {
        // Short model rollouts need states from the whole replay buffer. If we
        // only reset to episode starts, a 30-step horizon never trains on the
        // late approach/success part of a 75-step task.
        let all: Vec<Vec<f32>> = dataset
            .episodes
            .iter()
            .flat_map(|e| e.transitions.iter().map(|t| t.state.clone()))
            .collect();
        let stride = (all.len() / 512).max(1);
        all.into_iter().step_by(stride).collect()
    } else {
        dataset
            .episodes
            .iter()
            .filter_map(|e| e.transitions.first().map(|t| t.state.clone()))
            .collect()
    };
    if starts.is_empty() {
        panic!("dataset has no episode starts");
    }
    let mode = string_arg(args, "--mode", "cold-start");
    let policy_hidden = arg(args, "--policy-hidden", 32).max(4);
    let mut policy = if mode == "finetune" {
        let init_path = string_arg(args, "--init-policy", "runs/ppo.json");
        let text = fs::read_to_string(&init_path)
            .unwrap_or_else(|e| panic!("cannot read init policy {init_path}: {e}"));
        Ppo::load_json(&text).unwrap_or_else(|e| panic!("invalid init policy {init_path}: {e}"))
    } else {
        Ppo::new(
            layout.dim,
            PpoConfig {
                hidden: policy_hidden,
                minibatch_size: 256,
                ..Default::default()
            },
            arg_u64(args, "--imagine-seed", 777),
        )
    };
    if policy.actor.input_dim() != layout.dim {
        panic!("policy observation dimension does not match selected layout");
    }
    let bc_epochs = arg(args, "--bc-epochs", 0);
    let bc_samples = arg(args, "--bc-samples", 50_000).max(1);
    let (bc_sample_count, bc_loss) = if mode == "cold-start" && bc_epochs > 0 {
        let result = behavior_clone_policy(&mut policy, &dataset, layout, bc_epochs, bc_samples);
        println!(
            "behavior_clone samples={} epochs={} loss={:.6}",
            result.0, bc_epochs, result.1
        );
        result
    } else {
        (0, 0.0)
    };
    let env_count = std::env::var("PUSH_NUM_ENVS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(4)
        .max(1);
    // Preserve the original 256-step PPO update and full 75-step task horizon
    // unless the short-rollout experiment is requested explicitly.
    let rollout_steps = arg(args, "--rollout-steps", 256).max(1);
    let rollout_horizon = arg(args, "--imagine-horizon", 75).max(1);
    let uncertainty_limit = args
        .iter()
        .position(|x| x == "--uncertainty-limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse::<f32>().ok())
        .unwrap_or(0.75)
        .max(0.0);
    let uncertainty_penalty = args
        .iter()
        .position(|x| x == "--uncertainty-penalty")
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse::<f32>().ok())
        .unwrap_or(0.0)
        .max(0.0);
    let mut envs: Vec<WorldEnv> = (0..env_count)
        .map(|i| {
            let env_starts = if start_mode == "replay" {
                // Rotate each pool so parallel lanes do not begin at exactly
                // the same replay state, while retaining all sampled phases.
                starts
                    .iter()
                    .cycle()
                    .skip(i % starts.len())
                    .take(starts.len())
                    .cloned()
                    .collect()
            } else {
                vec![
                    starts[i % starts.len()].clone(),
                    starts[(i + 1) % starts.len()].clone(),
                ]
            };
            WorldEnv::with_penalty(
                model.clone(),
                env_starts,
                layout.clone(),
                rollout_horizon,
                uncertainty_limit,
                uncertainty_penalty,
            )
        })
        .collect();
    let updates = arg(args, "--updates", 500);
    let metrics_path = string_arg(args, "--metrics", "runs/imagined-metrics.jsonl");
    reset_metrics(&metrics_path);
    for update in 0..updates {
        let stats = policy.train(&mut envs, rollout_steps);
        if update == 0 || (update + 1) % 10 == 0 {
            println!("imagine_update={} steps={} imagined_return={:.3} episodes={} uncertainty_guarded={}", update + 1, stats.steps, stats.reward_sum, stats.episodes, stats.episodes.saturating_sub(stats.successes));
        }
        append_metric(
            &metrics_path,
            json!({
                "stage": "imagine",
                "mode": mode,
                "start_mode": start_mode,
                "bc_epochs": bc_epochs,
                "bc_samples": bc_sample_count,
                "bc_loss": bc_loss,
                "uncertainty_penalty": uncertainty_penalty,
                "algorithm": "ppo",
                "update": update + 1,
                "steps": stats.steps,
                "return": stats.reward_sum,
                "episodes": stats.episodes,
                "successes": stats.successes,
                "success_rate": stats.successes as f32 / stats.episodes.max(1) as f32,
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "approx_kl": stats.approx_kl,
                "uncertainty_guarded": stats.episodes.saturating_sub(stats.successes)
            }),
        );
    }
    let output = string_arg(args, "--imagined-checkpoint", "runs/imagined-ppo.json");
    save_json_file(
        &output,
        &serde_json::from_str::<serde_json::Value>(&policy.save_json().unwrap()).unwrap(),
    );
    println!(
        "imagined_policy mode={} world_model={} updates={} checkpoint={}",
        mode, model_path, updates, output
    );
    println!("imagined_metrics={metrics_path}");
}
fn run_world_model(args: &[String], env_config: EnvConfig, layout: &ObsLayout) {
    let episodes = arg(args, "--episodes", 24).max(3);
    let policy_path = policy_checkpoint(args);
    let policy_text = fs::read_to_string(&policy_path).unwrap_or_else(|error| {
        panic!("cannot read stage-1 PPO checkpoint {policy_path}: {error}; run push-cli first or pass --policy-checkpoint PATH")
    });
    let mut policy = Ppo::load_json(&policy_text)
        .unwrap_or_else(|error| panic!("invalid PPO policy checkpoint {policy_path}: {error}"));
    if policy.actor.input_dim() != layout.dim {
        panic!(
            "checkpoint observation dimension {} does not match selected layout {}",
            policy.actor.input_dim(),
            layout.dim
        );
    }
    let dataset = collect_dataset(episodes, 9000, env_config, layout, &mut policy);
    let path = string_arg(args, "--world-checkpoint", "runs/world-model.json");
    let model = train_world_model(&dataset, args);
    write_world_model(&model, &path);
    // A model-only rollout: this loop calls WorldModel::predict and never Env::step.
    let first = &dataset.episodes[0].transitions[0];
    let mut state = first.state.clone();
    let mut next = vec![0.; STATE_DIM];
    let mut ws = model.workspace();
    let mut uncertainty_sum = 0.;
    for t in 0..5 {
        let u = model.predict(&state, first.action, &mut next, None, &mut ws);
        uncertainty_sum += u;
        state.copy_from_slice(&next);
        println!(
            "imagined_step={} uncertainty={:.5} reward_source=analytic",
            t + 1,
            u
        );
    }
    println!(
        "imagined_rollout_mean_uncertainty={:.5} real_physics_calls=0",
        uncertainty_sum / 5.0
    );
    run_imagined_ppo(&model, &dataset, layout);
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{}", include_str!("help.txt"));
        return;
    }
    let action_mode = args
        .iter()
        .position(|x| x == "--action")
        .and_then(|i| args.get(i + 1))
        .map(|x| match x.as_str() {
            "velocity" => ActionMode::Velocity,
            "acceleration" => ActionMode::Acceleration,
            "absolute" => ActionMode::Absolute,
            "joint-delta" => ActionMode::JointDelta,
            _ => ActionMode::Position,
        })
        .unwrap_or(ActionMode::Position);
    let env_config = EnvConfig {
        action_mode,
        max_decisions: 75,
    };
    let layout = args
        .iter()
        .position(|x| x == "--obs")
        .and_then(|i| args.get(i + 1))
        .map(|x| ObsLayout::from_csv(x).expect("invalid --obs selection"))
        .unwrap_or_default();
    if args.iter().any(|x| x == "--evaluate-world") {
        run_evaluate_world(&args, &layout);
        return;
    }
    if args.iter().any(|x| x == "--evaluate") {
        run_evaluate(&args, env_config, &layout);
        return;
    }
    if args.iter().any(|x| x == "--collect-data") || args.iter().any(|x| x == "--collect") {
        run_collect_data(&args, env_config, &layout);
        return;
    }
    if args.iter().any(|x| x == "--train-world") {
        run_train_world(&args);
        return;
    }
    if args.iter().any(|x| x == "--compare-world") {
        run_compare_world(&args);
        return;
    }
    if args.iter().any(|x| x == "--imagine") {
        run_imagine(&args, &layout);
        return;
    }
    if args.iter().any(|x| x == "--world-model") {
        run_world_model(&args, env_config, &layout);
        return;
    }
    if args.iter().any(|x| x == "--algorithm") && args.iter().any(|x| x == "sac") {
        run_sac(&args, env_config, &layout);
        return;
    }
    let updates = arg(&args, "--updates", 10);
    let metrics_path = string_arg(&args, "--metrics", "runs/ppo-metrics.jsonl");
    reset_metrics(&metrics_path);
    let envs_n = std::env::var("PUSH_NUM_ENVS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|x| x.get().saturating_sub(2).max(1))
                .unwrap_or(1)
        });
    let mut envs = (0..envs_n)
        .map(|i| RealEnv::new(i as u64, env_config, layout.clone()))
        .collect::<Vec<_>>();
    let mut p = Ppo::new(
        layout.dim,
        PpoConfig {
            minibatch_size: 256,
            ..Default::default()
        },
        42,
    );
    for update in 0..updates {
        let s = p.train(&mut envs, 256);
        println!(
            "update={} steps={} return={:.3} episodes={} success={} rate={:.3} rejected={}",
            update + 1,
            s.steps,
            s.reward_sum,
            s.episodes,
            s.successes,
            s.successes as f32 / s.episodes.max(1) as f32,
            s.rejected_updates
        );
        append_metric(
            &metrics_path,
            json!({
                "stage": "train-real",
                "algorithm": "ppo",
                "update": update + 1,
                "steps": s.steps,
                "return": s.reward_sum,
                "episodes": s.episodes,
                "successes": s.successes,
                "success_rate": s.successes as f32 / s.episodes.max(1) as f32,
                "policy_loss": s.policy_loss,
                "value_loss": s.value_loss,
                "entropy": s.entropy,
                "approx_kl": s.approx_kl,
                "rejected_updates": s.rejected_updates
            }),
        );
    }
    let path = checkpoint(&args);
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&path, p.save_json().expect("serialize checkpoint")).expect("write checkpoint");
    println!("checkpoint={path} metrics={metrics_path}");
}
