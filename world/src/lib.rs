//! Episode-separated datasets and learned ensembles. This crate has no physics dependency.
use push_rl::neural::{Adam, Mlp, MlpGrad, Rng, Workspace};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: [f32; 2],
    pub next_state: Vec<f32>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    pub id: u64,
    pub seed: u64,
    pub transitions: Vec<Transition>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub version: String,
    pub config_hash: String,
    pub state_dim: usize,
    pub episodes: Vec<Episode>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeSplit {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub test: Vec<usize>,
}
impl Dataset {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != "push-state-dataset-v1" || self.state_dim == 0 || self.episodes.len() < 3
        {
            return Err(
                "dataset needs version v1, a state dimension and at least 3 episodes".into(),
            );
        }
        let mut ids = std::collections::HashSet::new();
        for e in &self.episodes {
            if !ids.insert(e.id) || e.transitions.is_empty() {
                return Err("episode IDs must be unique and episodes nonempty".into());
            }
            for t in &e.transitions {
                if t.state.len() != self.state_dim
                    || t.next_state.len() != self.state_dim
                    || !t
                        .state
                        .iter()
                        .chain(&t.next_state)
                        .chain(&t.action)
                        .all(|x| x.is_finite())
                    || !t.reward.is_finite()
                {
                    return Err("invalid/nonfinite transition".into());
                }
            }
        }
        Ok(())
    }
    pub fn split(&self, seed: u64) -> EpisodeSplit {
        let n = self.episodes.len();
        assert!(n >= 3);
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = Rng::new(seed);
        for i in (1..n).rev() {
            indices.swap(i, rng.index(i + 1));
        }
        let valid = (n / 10).max(1);
        let test = (n / 10).max(1);
        EpisodeSplit {
            train: indices[..n - valid - test].to_vec(),
            validation: indices[n - valid - test..n - test].to_vec(),
            test: indices[n - test..].to_vec(),
        }
    }
    pub fn transitions(&self) -> usize {
        self.episodes.iter().map(|e| e.transitions.len()).sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    pub members: usize,
    pub hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// Only these state fields are learned. Goal, branch, time and success hold are analytical.
    pub predicted_indices: Vec<usize>,
    pub time_index: Option<usize>,
    pub hold_index: Option<usize>,
    pub decision_dt: f32,
}
impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            members: 3,
            hidden: 32,
            epochs: 30,
            batch_size: 128,
            learning_rate: 1e-3,
            predicted_indices: (0..14).collect(),
            time_index: Some(16),
            hold_index: Some(17),
            decision_dt: 1.0 / 15.0,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorldMetrics {
    pub train_normalized_mse: f32,
    pub validation_normalized_mse: f32,
    pub test_one_step_mse: f32,
    pub test_five_step_mse: f32,
    pub train_episodes: usize,
    pub validation_episodes: usize,
    pub test_episodes: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct WorldModel {
    pub version: String,
    pub config_hash: String,
    pub config: WorldConfig,
    pub state_dim: usize,
    pub members: Vec<Mlp>,
    pub input_mean: Vec<f32>,
    pub input_std: Vec<f32>,
    pub delta_mean: Vec<f32>,
    pub delta_std: Vec<f32>,
    pub split: EpisodeSplit,
    pub metrics: WorldMetrics,
}
pub struct PredictionWorkspace {
    input: Vec<f32>,
    outputs: Vec<f32>,
    member_workspaces: Vec<Workspace>,
}
impl WorldModel {
    pub fn workspace(&self) -> PredictionWorkspace {
        PredictionWorkspace {
            input: vec![0.0; self.state_dim + 2],
            outputs: vec![0.0; self.members.len() * self.config.predicted_indices.len()],
            member_workspaces: self.members.iter().map(Mlp::workspace).collect(),
        }
    }
    /// Returns ensemble standard deviation in normalized delta units. No physics or allocations.
    pub fn predict(
        &self,
        state: &[f32],
        action: [f32; 2],
        next: &mut [f32],
        member: Option<usize>,
        ws: &mut PredictionWorkspace,
    ) -> f32 {
        let n = self.config.predicted_indices.len();
        next.copy_from_slice(state);
        for i in 0..self.state_dim {
            ws.input[i] = (state[i] - self.input_mean[i]) / self.input_std[i];
        }
        for i in 0..2 {
            ws.input[self.state_dim + i] = (action[i] - self.input_mean[self.state_dim + i])
                / self.input_std[self.state_dim + i];
        }
        for (m, net) in self.members.iter().enumerate() {
            let output = net.forward(&ws.input, &mut ws.member_workspaces[m]);
            ws.outputs[m * n..(m + 1) * n].copy_from_slice(output);
        }
        let mut variance = 0.0;
        for (k, &index) in self.config.predicted_indices.iter().enumerate() {
            let mean = (0..self.members.len())
                .map(|m| ws.outputs[m * n + k])
                .sum::<f32>()
                / self.members.len() as f32;
            variance += (0..self.members.len())
                .map(|m| (ws.outputs[m * n + k] - mean).powi(2))
                .sum::<f32>()
                / self.members.len() as f32;
            let prediction = member.map_or(mean, |m| ws.outputs[(m % self.members.len()) * n + k]);
            next[index] = state[index] + prediction * self.delta_std[k] + self.delta_mean[k];
        }
        if let Some(i) = self.config.time_index {
            next[i] = state[i] + self.config.decision_dt;
        }
        // The task adapter recomputes success hold from the predicted geometry.
        if let Some(i) = self.config.hold_index {
            next[i] = state[i];
        }
        (variance / n as f32).sqrt()
    }
    pub fn train(dataset: &Dataset, config: WorldConfig, seed: u64) -> Result<Self, String> {
        dataset.validate()?;
        if config.members < 2
            || config.hidden == 0
            || config.epochs == 0
            || config.batch_size == 0
            || config.predicted_indices.is_empty()
            || config
                .predicted_indices
                .iter()
                .any(|&i| i >= dataset.state_dim)
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
        {
            return Err("invalid world configuration".into());
        }
        let split = dataset.split(seed);
        let training: Vec<&Transition> = split
            .train
            .iter()
            .flat_map(|&i| dataset.episodes[i].transitions.iter())
            .collect();
        let validation: Vec<&Transition> = split
            .validation
            .iter()
            .flat_map(|&i| dataset.episodes[i].transitions.iter())
            .collect();
        let input_dim = dataset.state_dim + 2;
        let output_dim = config.predicted_indices.len();
        let mut input_mean = vec![0.0; input_dim];
        let mut input_sq = vec![0.0; input_dim];
        let mut delta_mean = vec![0.0; output_dim];
        let mut delta_sq = vec![0.0; output_dim];
        for t in &training {
            for i in 0..input_dim {
                let x = if i < dataset.state_dim {
                    t.state[i]
                } else {
                    t.action[i - dataset.state_dim]
                };
                input_mean[i] += x;
                input_sq[i] += x * x;
            }
            for (j, &i) in config.predicted_indices.iter().enumerate() {
                let delta = t.next_state[i] - t.state[i];
                delta_mean[j] += delta;
                delta_sq[j] += delta * delta;
            }
        }
        let count = training.len() as f32;
        for x in &mut input_mean {
            *x /= count;
        }
        for x in &mut delta_mean {
            *x /= count;
        }
        let input_std: Vec<f32> = input_sq
            .iter()
            .zip(&input_mean)
            .map(|(s, m)| (s / count - m * m).max(0.0).sqrt().max(1e-3))
            .collect();
        let delta_std: Vec<f32> = delta_sq
            .iter()
            .zip(&delta_mean)
            .map(|(s, m)| (s / count - m * m).max(0.0).sqrt().max(1e-4))
            .collect();
        let normalize = |transitions: &[&Transition]| -> (Vec<f32>, Vec<f32>) {
            let mut xs = Vec::with_capacity(transitions.len() * input_dim);
            let mut ys = Vec::with_capacity(transitions.len() * output_dim);
            for t in transitions {
                for i in 0..input_dim {
                    let x = if i < dataset.state_dim {
                        t.state[i]
                    } else {
                        t.action[i - dataset.state_dim]
                    };
                    xs.push((x - input_mean[i]) / input_std[i]);
                }
                for (j, &i) in config.predicted_indices.iter().enumerate() {
                    ys.push((t.next_state[i] - t.state[i] - delta_mean[j]) / delta_std[j]);
                }
            }
            (xs, ys)
        };
        let (train_x, train_y) = normalize(&training);
        let (val_x, val_y) = normalize(&validation);
        let mut rng = Rng::new(seed ^ 0xDEADBEEF);
        let mut members = Vec::with_capacity(config.members);
        for _ in 0..config.members {
            let mut net = Mlp::new(input_dim, config.hidden, output_dim, &mut rng);
            let mut optimizer = Adam::new(&net, config.learning_rate);
            let mut gradient = MlpGrad::new(&net);
            let mut ws = net.workspace();
            let mut output_grad = vec![0.0; output_dim];
            // Bootstrap entire episodes; normalization still uses the original training split only.
            let mut bootstrap = Vec::with_capacity(training.len());
            let offsets: Vec<usize> = split
                .train
                .iter()
                .scan(0usize, |sum, &e| {
                    let offset = *sum;
                    *sum += dataset.episodes[e].transitions.len();
                    Some(offset)
                })
                .collect();
            for _ in 0..split.train.len() {
                let e = rng.index(split.train.len());
                bootstrap.extend(
                    offsets[e]..offsets[e] + dataset.episodes[split.train[e]].transitions.len(),
                );
            }
            let mut best = net.clone();
            let mut best_loss = f32::INFINITY;
            for _ in 0..config.epochs {
                for i in (1..bootstrap.len()).rev() {
                    bootstrap.swap(i, rng.index(i + 1));
                }
                for batch in bootstrap.chunks(config.batch_size) {
                    gradient.zero();
                    for &i in batch {
                        let pred =
                            net.forward(&train_x[i * input_dim..(i + 1) * input_dim], &mut ws);
                        for k in 0..output_dim {
                            output_grad[k] =
                                2.0 * (pred[k] - train_y[i * output_dim + k]) / output_dim as f32;
                        }
                        net.backward(&mut ws, &output_grad, &mut gradient);
                    }
                    gradient.scale(1.0 / batch.len() as f32);
                    optimizer.step(&mut net, &gradient);
                }
                let loss = network_mse(&net, &val_x, &val_y, input_dim, output_dim);
                if loss.is_finite() && loss < best_loss {
                    best_loss = loss;
                    best = net.clone();
                }
            }
            if !best_loss.is_finite() {
                return Err("nonfinite world-model loss".into());
            }
            members.push(best);
        }
        let mut model = Self {
            version: "push-delta-ensemble-v1".into(),
            config_hash: dataset.config_hash.clone(),
            config,
            state_dim: dataset.state_dim,
            members,
            input_mean,
            input_std,
            delta_mean,
            delta_std,
            split,
            metrics: WorldMetrics::default(),
        };
        model.metrics.train_normalized_mse = model
            .members
            .iter()
            .map(|m| network_mse(m, &train_x, &train_y, input_dim, output_dim))
            .sum::<f32>()
            / model.members.len() as f32;
        model.metrics.validation_normalized_mse = model
            .members
            .iter()
            .map(|m| network_mse(m, &val_x, &val_y, input_dim, output_dim))
            .sum::<f32>()
            / model.members.len() as f32;
        model.metrics.train_episodes = model.split.train.len();
        model.metrics.validation_episodes = model.split.validation.len();
        model.metrics.test_episodes = model.split.test.len();
        model.metrics.test_one_step_mse = model.rollout_mse(dataset, 1);
        model.metrics.test_five_step_mse = model.rollout_mse(dataset, 5);
        Ok(model)
    }
    pub fn rollout_mse(&self, dataset: &Dataset, horizon: usize) -> f32 {
        let mut ws = self.workspace();
        let mut pred = vec![0.0; self.state_dim];
        let mut next = pred.clone();
        let mut sq = 0.0;
        let mut n = 0;
        for &e in &self.split.test {
            let trajectory = &dataset.episodes[e].transitions;
            for start in (0..trajectory.len()).step_by(horizon) {
                pred.copy_from_slice(&trajectory[start].state);
                for t in trajectory.iter().skip(start).take(horizon) {
                    self.predict(&pred, t.action, &mut next, None, &mut ws);
                    for &k in &self.config.predicted_indices {
                        sq += (next[k] - t.next_state[k]).powi(2);
                        n += 1;
                    }
                    std::mem::swap(&mut pred, &mut next);
                }
            }
        }
        sq / n.max(1) as f32
    }
}
fn network_mse(net: &Mlp, x: &[f32], y: &[f32], input: usize, output: usize) -> f32 {
    let mut ws = net.workspace();
    let mut sq = 0.0;
    for (a, b) in x.chunks_exact(input).zip(y.chunks_exact(output)) {
        sq += net
            .forward(a, &mut ws)
            .iter()
            .zip(b)
            .map(|(u, v)| (u - v).powi(2))
            .sum::<f32>();
    }
    sq / y.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    fn synthetic() -> Dataset {
        let mut rng = Rng::new(7);
        let episodes = (0..20)
            .map(|id| {
                let mut state = vec![rng.uniform() * 2.0 - 1.0, rng.uniform() * 2.0 - 1.0];
                let transitions = (0..20)
                    .map(|_| {
                        let action = [rng.uniform() * 2.0 - 1.0, rng.uniform() * 2.0 - 1.0];
                        let next = vec![state[0] + 0.1 * action[0], state[1] + 0.2 * action[1]];
                        let t = Transition {
                            state: state.clone(),
                            action,
                            next_state: next.clone(),
                            reward: 0.0,
                            terminated: false,
                            truncated: false,
                        };
                        state = next;
                        t
                    })
                    .collect();
                Episode {
                    id,
                    seed: id,
                    transitions,
                }
            })
            .collect();
        Dataset {
            version: "push-state-dataset-v1".into(),
            config_hash: "test".into(),
            state_dim: 2,
            episodes,
        }
    }
    #[test]
    fn split_is_episode_disjoint() {
        let d = synthetic();
        let s = d.split(7);
        let all: std::collections::HashSet<_> =
            s.train.iter().chain(&s.validation).chain(&s.test).collect();
        assert_eq!(all.len(), d.episodes.len());
        assert!(!s.test.is_empty());
    }
    #[test]
    fn learns_action_conditioned_delta_and_roundtrips() {
        let d = synthetic();
        let c = WorldConfig {
            members: 2,
            hidden: 16,
            epochs: 35,
            batch_size: 32,
            predicted_indices: vec![0, 1],
            time_index: None,
            hold_index: None,
            ..Default::default()
        };
        let m = WorldModel::train(&d, c, 4).unwrap();
        assert!(m.metrics.test_one_step_mse < 0.0006, "{:?}", m.metrics);
        let encoded = serde_json::to_string(&m).unwrap();
        let restored: WorldModel = serde_json::from_str(&encoded).unwrap();
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        let mut ws = m.workspace();
        let mut ws2 = restored.workspace();
        m.predict(&[0.1, 0.2], [0.7, -0.4], &mut a, None, &mut ws);
        restored.predict(&[0.1, 0.2], [0.7, -0.4], &mut b, None, &mut ws2);
        assert_eq!(a, b);
        assert!((a[0] - 0.17).abs() < 0.05 && (a[1] - 0.12).abs() < 0.05);
    }
}
