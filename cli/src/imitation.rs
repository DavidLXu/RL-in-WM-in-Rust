//! Supervised policy initialization using action labels from recorded data.
//! BC-only runs (zero PPO updates) are imitation baselines, not model-based RL.
use push_env::{decode_state, Env, EnvConfig, ObsLayout};
use push_rl::{MlpGrad, Ppo};
use push_world::Dataset;

/// Initialize a policy from the logged action distribution without loading a
/// stage-1 checkpoint. The subsequent PPO updates still happen in the model.
pub(crate) fn behavior_clone_policy(
    policy: &mut Ppo,
    dataset: &Dataset,
    layout: &ObsLayout,
    epochs: usize,
    max_samples: usize,
) -> (usize, f32) {
    let mut env = Env::with_config(1, EnvConfig::default());
    let mut observations = Vec::<Vec<f32>>::new();
    let mut actions = Vec::<[f32; 2]>::new();
    for episode in &dataset.episodes {
        for transition in &episode.transitions {
            env.set_state(decode_state(
                transition
                    .state
                    .clone()
                    .try_into()
                    .expect("dataset state dimension"),
            ));
            let mut obs = vec![0.0; layout.dim];
            env.observe_layout(layout, &mut obs);
            observations.push(obs);
            actions.push(transition.action);
        }
    }
    if observations.is_empty() {
        return (0, 0.0);
    }
    if max_samples > 0 && observations.len() > max_samples {
        let stride = (observations.len() + max_samples - 1) / max_samples;
        let mut sampled_obs = Vec::with_capacity(max_samples);
        let mut sampled_actions = Vec::with_capacity(max_samples);
        for i in (0..observations.len()).step_by(stride) {
            sampled_obs.push(observations[i].clone());
            sampled_actions.push(actions[i]);
            if sampled_obs.len() == max_samples {
                break;
            }
        }
        observations = sampled_obs;
        actions = sampled_actions;
    }
    let mut indices: Vec<usize> = (0..observations.len()).collect();
    let mut ws = policy.actor.workspace();
    let mut grad = MlpGrad::new(&policy.actor);
    let batch_size = policy.config.minibatch_size.max(1);
    let mut loss_sum = 0.0;
    let mut loss_count = 0usize;
    for _ in 0..epochs {
        policy.rng.shuffle(&mut indices);
        for chunk in indices.chunks(batch_size) {
            grad.zero();
            for &i in chunk {
                let output = policy.actor.forward(&observations[i], &mut ws).to_vec();
                let target = [
                    actions[i][0].clamp(-0.999, 0.999).atanh(),
                    actions[i][1].clamp(-0.999, 0.999).atanh(),
                    -1.0,
                    -1.0,
                ];
                let mut output_grad = [0.0; 4];
                for j in 0..4 {
                    let error = output[j] - target[j];
                    loss_sum += 0.5 * error * error;
                    loss_count += 1;
                    output_grad[j] = error / chunk.len() as f32;
                }
                policy.actor.backward(&mut ws, &output_grad, &mut grad);
            }
            let _ = policy.actor_optimizer.step(&mut policy.actor, &grad);
        }
    }
    (observations.len(), loss_sum / loss_count.max(1) as f32)
}
