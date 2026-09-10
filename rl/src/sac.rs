//! Soft actor-critic for two bounded continuous actions.
//!
//! `Sac` is a complete, serializable trainer: checkpoints retain the online and
//! target networks, Adam moments, temperature optimizer, random generator, and
//! replay ring. Scratch storage is rebuilt once after deserialization. Call
//! `observe` with the observation *before resetting* a finished environment;
//! time-limit truncations bootstrap, true terminations do not.

use crate::neural::{Adam, Mlp, MlpGrad, Rng, Workspace};
use serde::{Deserialize, Serialize};

/// Number of continuous controls supported by the push environment.
pub const ACTION_DIM: usize = 2;
const LOG_STD_MIN: f32 = -5.0;
const LOG_STD_MAX: f32 = 2.0;
const LOG_TWO_PI: f32 = 1.837_877;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SacConfig {
    pub obs_dim: usize,
    pub hidden: usize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    /// Initial environment interactions use uniform random actions.
    pub warmup_steps: u64,
    pub gamma: f32,
    /// Target interpolation: target <- (1 - tau) target + tau online.
    pub tau: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub temperature_lr: f32,
    pub alpha: f32,
    pub auto_temperature: bool,
    /// Differential entropy target; the usual two-action default is -2.
    pub target_entropy: f32,
    pub max_grad_norm: f32,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            obs_dim: 16,
            hidden: 32,
            replay_capacity: 100_000,
            batch_size: 64,
            warmup_steps: 1_000,
            gamma: 0.99,
            tau: 0.005,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            temperature_lr: 3e-4,
            alpha: 0.2,
            auto_temperature: false,
            target_entropy: -(ACTION_DIM as f32),
            max_grad_norm: 10.0,
        }
    }
}

impl SacConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.obs_dim == 0 || self.hidden == 0 {
            return Err("observation and hidden dimensions must be positive");
        }
        if self.batch_size == 0 || self.replay_capacity < self.batch_size {
            return Err("replay capacity must be at least the positive batch size");
        }
        if !(0.0..=1.0).contains(&self.gamma) || !(0.0..=1.0).contains(&self.tau) {
            return Err("gamma and tau must lie in [0, 1]");
        }
        for value in [
            self.actor_lr,
            self.critic_lr,
            self.temperature_lr,
            self.alpha,
            self.max_grad_norm,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(
                    "learning rates, alpha, and gradient limit must be positive and finite",
                );
            }
        }
        if !self.target_entropy.is_finite() {
            return Err("target entropy must be finite");
        }
        Ok(())
    }
}

/// Preallocated structure-of-arrays replay storage. No allocation occurs on push.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayBuffer {
    capacity: usize,
    obs_dim: usize,
    observations: Vec<f32>,
    next_observations: Vec<f32>,
    actions: Vec<[f32; ACTION_DIM]>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    len: usize,
    cursor: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayTransition<'a> {
    pub observation: &'a [f32],
    pub action: [f32; ACTION_DIM],
    pub reward: f32,
    pub next_observation: &'a [f32],
    pub terminated: bool,
    pub truncated: bool,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, obs_dim: usize) -> Self {
        assert!(capacity > 0 && obs_dim > 0);
        let scalars = capacity.checked_mul(obs_dim).expect("replay size overflow");
        Self {
            capacity,
            obs_dim,
            observations: vec![0.0; scalars],
            next_observations: vec![0.0; scalars],
            actions: vec![[0.0; ACTION_DIM]; capacity],
            rewards: vec![0.0; capacity],
            terminated: vec![false; capacity],
            truncated: vec![false; capacity],
            len: 0,
            cursor: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        observation: &[f32],
        action: [f32; ACTION_DIM],
        reward: f32,
        next_observation: &[f32],
        terminated: bool,
        truncated: bool,
    ) {
        assert_eq!(observation.len(), self.obs_dim);
        assert_eq!(next_observation.len(), self.obs_dim);
        assert!(reward.is_finite());
        assert!(observation
            .iter()
            .chain(next_observation)
            .all(|x| x.is_finite()));
        assert!(action.iter().all(|x| x.is_finite()));
        let offset = self.cursor * self.obs_dim;
        self.observations[offset..offset + self.obs_dim].copy_from_slice(observation);
        self.next_observations[offset..offset + self.obs_dim].copy_from_slice(next_observation);
        self.actions[self.cursor] = action;
        self.rewards[self.cursor] = reward;
        self.terminated[self.cursor] = terminated;
        self.truncated[self.cursor] = truncated;
        self.cursor = (self.cursor + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Read an occupied physical slot; slot order is not chronological after wrap.
    pub fn transition(&self, index: usize) -> ReplayTransition<'_> {
        assert!(index < self.len);
        let offset = index * self.obs_dim;
        ReplayTransition {
            observation: &self.observations[offset..offset + self.obs_dim],
            action: self.actions[index],
            reward: self.rewards[index],
            next_observation: &self.next_observations[offset..offset + self.obs_dim],
            terminated: self.terminated[index],
            truncated: self.truncated[index],
        }
    }

    fn sample_index(&self, rng: &mut Rng) -> usize {
        assert!(!self.is_empty());
        ((rng.uniform() * self.len as f32) as usize).min(self.len - 1)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct SacMetrics {
    pub critic_loss: f32,
    pub q1_loss: f32,
    pub q2_loss: f32,
    pub actor_loss: f32,
    pub temperature_loss: f32,
    pub alpha: f32,
    pub entropy: f32,
    pub mean_q: f32,
    pub mean_target: f32,
    pub actor_grad_norm: f32,
    pub critic_grad_norm: f32,
    pub updates: u64,
    pub samples: usize,
    /// False means a nonfinite update was rejected by the numerical guard.
    pub applied: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TemperatureAdam {
    moment: f32,
    variance: f32,
    step: u64,
}

impl TemperatureAdam {
    fn new() -> Self {
        Self {
            moment: 0.0,
            variance: 0.0,
            step: 0,
        }
    }

    fn update(&mut self, parameter: &mut f32, gradient: f32, learning_rate: f32) -> bool {
        if !gradient.is_finite() {
            return false;
        }
        self.step += 1;
        self.moment = 0.9 * self.moment + 0.1 * gradient;
        self.variance = 0.999 * self.variance + 0.001 * gradient * gradient;
        let m = self.moment / (1.0 - 0.9_f32.powf(self.step as f32));
        let v = self.variance / (1.0 - 0.999_f32.powf(self.step as f32));
        // Keep exp(log_alpha) representable, including in very long runs.
        *parameter = (*parameter - learning_rate * m / (v.sqrt() + 1e-8)).clamp(-20.0, 5.0);
        true
    }
}

/// Squashed Gaussian sample and the values needed for its pathwise derivative.
#[derive(Clone, Copy, Debug)]
struct PolicySample {
    action: [f32; ACTION_DIM],
    std_noise: [f32; ACTION_DIM],
    log_std_derivative: [f32; ACTION_DIM],
    log_prob: f32,
}

fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn sample_policy(output: &[f32], noise: [f32; ACTION_DIM]) -> PolicySample {
    let mut result = PolicySample {
        action: [0.0; ACTION_DIM],
        std_noise: [0.0; ACTION_DIM],
        log_std_derivative: [0.0; ACTION_DIM],
        log_prob: 0.0,
    };
    for i in 0..ACTION_DIM {
        let raw_log_std = output[ACTION_DIM + i];
        let log_std = raw_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX);
        let std_noise = log_std.exp() * noise[i];
        let pre_tanh = output[i] + std_noise;
        result.action[i] = pre_tanh.tanh();
        result.std_noise[i] = std_noise;
        result.log_std_derivative[i] = if (LOG_STD_MIN..=LOG_STD_MAX).contains(&raw_log_std) {
            1.0
        } else {
            0.0
        };
        // This stable log-Jacobian remains finite when tanh rounds to +/-1.
        let log_jacobian = 2.0 * (std::f32::consts::LN_2 - pre_tanh - softplus(-2.0 * pre_tanh));
        result.log_prob += -0.5 * (noise[i] * noise[i] + LOG_TWO_PI) - log_std - log_jacobian;
    }
    result
}

fn policy_output_gradient(
    sample: &PolicySample,
    q_action_gradient: &[f32],
    alpha: f32,
    scale: f32,
) -> [f32; 2 * ACTION_DIM] {
    let mut gradient = [0.0; 2 * ACTION_DIM];
    for i in 0..ACTION_DIM {
        let a = sample.action[i];
        // Reparameterized derivative of alpha log pi(a|s) - Q(s,a).
        let dz = 2.0 * alpha * a - q_action_gradient[i] * (1.0 - a * a);
        gradient[i] = scale * dz;
        gradient[ACTION_DIM + i] =
            scale * (dz * sample.std_noise[i] - alpha) * sample.log_std_derivative[i];
    }
    gradient
}

fn bellman_target(reward: f32, terminated: bool, gamma: f32, next_value: f32) -> f32 {
    if terminated {
        reward
    } else {
        reward + gamma * next_value
    }
}

fn clip_gradient(gradient: &mut MlpGrad, max_norm: f32) -> f32 {
    let norm = gradient.norm();
    if norm > max_norm && norm.is_finite() {
        gradient.scale(max_norm / norm);
    }
    norm
}

struct SacScratch {
    acting: Workspace,
    target_actor: Workspace,
    target_q1: Workspace,
    target_q2: Workspace,
    q1: Workspace,
    q2: Workspace,
    actor: Workspace,
    actor_q1: Workspace,
    actor_q2: Workspace,
    actor_gradient: MlpGrad,
    q1_gradient: MlpGrad,
    q2_gradient: MlpGrad,
    unused_q1_gradient: MlpGrad,
    unused_q2_gradient: MlpGrad,
    q_input: Vec<f32>,
    next_q_input: Vec<f32>,
    indices: Vec<usize>,
}

impl SacScratch {
    fn new(sac: &Sac) -> Self {
        Self {
            acting: sac.actor.workspace(),
            target_actor: sac.actor.workspace(),
            target_q1: sac.target_q1.workspace(),
            target_q2: sac.target_q2.workspace(),
            q1: sac.q1.workspace(),
            q2: sac.q2.workspace(),
            actor: sac.actor.workspace(),
            actor_q1: sac.q1.workspace(),
            actor_q2: sac.q2.workspace(),
            actor_gradient: MlpGrad::new(&sac.actor),
            q1_gradient: MlpGrad::new(&sac.q1),
            q2_gradient: MlpGrad::new(&sac.q2),
            unused_q1_gradient: MlpGrad::new(&sac.q1),
            unused_q2_gradient: MlpGrad::new(&sac.q2),
            q_input: vec![0.0; sac.config.obs_dim + ACTION_DIM],
            next_q_input: vec![0.0; sac.config.obs_dim + ACTION_DIM],
            indices: vec![0; sac.config.batch_size],
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Sac {
    pub config: SacConfig,
    pub actor: Mlp,
    pub q1: Mlp,
    pub q2: Mlp,
    pub target_q1: Mlp,
    pub target_q2: Mlp,
    actor_optimizer: Adam,
    q1_optimizer: Adam,
    q2_optimizer: Adam,
    log_alpha: f32,
    temperature_optimizer: TemperatureAdam,
    rng: Rng,
    pub replay: ReplayBuffer,
    pub environment_steps: u64,
    pub updates: u64,
    #[serde(skip)]
    scratch: Option<SacScratch>,
}

impl Default for Sac {
    fn default() -> Self {
        Self::new(SacConfig::default(), 0)
    }
}

impl Sac {
    pub fn new(config: SacConfig, seed: u64) -> Self {
        config.validate().expect("invalid SAC configuration");
        let mut rng = Rng::new(seed);
        let actor = Mlp::new(config.obs_dim, config.hidden, 2 * ACTION_DIM, &mut rng);
        let q1 = Mlp::new(config.obs_dim + ACTION_DIM, config.hidden, 1, &mut rng);
        let q2 = Mlp::new(config.obs_dim + ACTION_DIM, config.hidden, 1, &mut rng);
        let mut sac = Self {
            actor_optimizer: Adam::new(&actor, config.actor_lr),
            q1_optimizer: Adam::new(&q1, config.critic_lr),
            q2_optimizer: Adam::new(&q2, config.critic_lr),
            target_q1: q1.clone(),
            target_q2: q2.clone(),
            actor,
            q1,
            q2,
            log_alpha: config.alpha.ln(),
            temperature_optimizer: TemperatureAdam::new(),
            rng,
            replay: ReplayBuffer::new(config.replay_capacity, config.obs_dim),
            environment_steps: 0,
            updates: 0,
            config,
            scratch: None,
        };
        sac.scratch = Some(SacScratch::new(&sac));
        sac
    }

    fn ensure_scratch(&mut self) {
        if self.scratch.is_none() {
            self.scratch = Some(SacScratch::new(self));
        }
    }

    pub fn alpha(&self) -> f32 {
        if self.config.auto_temperature {
            self.log_alpha.exp()
        } else {
            self.config.alpha
        }
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Deterministic evaluation returns tanh(mean), without consuming RNG state.
    /// Stochastic training actions are uniform during the configured warmup.
    pub fn action(&mut self, observation: &[f32], deterministic: bool) -> [f32; ACTION_DIM] {
        assert_eq!(observation.len(), self.config.obs_dim);
        if !deterministic && self.environment_steps < self.config.warmup_steps {
            return [
                2.0 * self.rng.uniform() - 1.0,
                2.0 * self.rng.uniform() - 1.0,
            ];
        }
        self.ensure_scratch();
        let scratch = self.scratch.as_mut().unwrap();
        let output = self.actor.forward(observation, &mut scratch.acting);
        if deterministic {
            [output[0].tanh(), output[1].tanh()]
        } else {
            sample_policy(output, [self.rng.normal(), self.rng.normal()]).action
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        observation: &[f32],
        action: [f32; ACTION_DIM],
        reward: f32,
        next_observation: &[f32],
        terminated: bool,
        truncated: bool,
    ) {
        self.replay.push(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        );
        self.environment_steps += 1;
    }

    /// One replay minibatch update. Returns None until warmup and batch size are met.
    pub fn update(&mut self) -> Option<SacMetrics> {
        if self.replay.len() < self.config.batch_size
            || self.environment_steps < self.config.warmup_steps
        {
            return None;
        }
        self.ensure_scratch();
        let alpha = self.alpha();
        let scratch = self.scratch.as_mut().unwrap();
        let obs_dim = self.config.obs_dim;
        let inverse_batch = 1.0 / self.config.batch_size as f32;
        let mut metrics = SacMetrics {
            alpha,
            samples: self.config.batch_size,
            updates: self.updates,
            ..SacMetrics::default()
        };
        scratch.q1_gradient.zero();
        scratch.q2_gradient.zero();
        scratch.actor_gradient.zero();
        scratch.unused_q1_gradient.zero();
        scratch.unused_q2_gradient.zero();

        for index in &mut scratch.indices {
            *index = self.replay.sample_index(&mut self.rng);
        }

        // Critic targets have no gradient into either the actor or target critics.
        for &index in &scratch.indices {
            let transition = self.replay.transition(index);
            let output = self
                .actor
                .forward(transition.next_observation, &mut scratch.target_actor);
            let next_sample = sample_policy(output, [self.rng.normal(), self.rng.normal()]);
            scratch.next_q_input[..obs_dim].copy_from_slice(transition.next_observation);
            scratch.next_q_input[obs_dim..].copy_from_slice(&next_sample.action);
            let next_q1 = self
                .target_q1
                .forward(&scratch.next_q_input, &mut scratch.target_q1)[0];
            let next_q2 = self
                .target_q2
                .forward(&scratch.next_q_input, &mut scratch.target_q2)[0];
            let target = bellman_target(
                transition.reward,
                transition.terminated,
                self.config.gamma,
                next_q1.min(next_q2) - alpha * next_sample.log_prob,
            );
            scratch.q_input[..obs_dim].copy_from_slice(transition.observation);
            scratch.q_input[obs_dim..].copy_from_slice(&transition.action);
            let q1 = self.q1.forward(&scratch.q_input, &mut scratch.q1)[0];
            let q2 = self.q2.forward(&scratch.q_input, &mut scratch.q2)[0];
            let error1 = q1 - target;
            let error2 = q2 - target;
            metrics.q1_loss += inverse_batch * error1 * error1;
            metrics.q2_loss += inverse_batch * error2 * error2;
            metrics.mean_target += inverse_batch * target;
            self.q1.backward(
                &mut scratch.q1,
                &[2.0 * inverse_batch * error1],
                &mut scratch.q1_gradient,
            );
            self.q2.backward(
                &mut scratch.q2,
                &[2.0 * inverse_batch * error2],
                &mut scratch.q2_gradient,
            );
        }
        metrics.critic_loss = metrics.q1_loss + metrics.q2_loss;
        let q1_norm = clip_gradient(&mut scratch.q1_gradient, self.config.max_grad_norm);
        let q2_norm = clip_gradient(&mut scratch.q2_gradient, self.config.max_grad_norm);
        metrics.critic_grad_norm = q1_norm.hypot(q2_norm);
        if !metrics.critic_loss.is_finite() || !metrics.critic_grad_norm.is_finite() {
            return Some(metrics);
        }
        if !self.q1_optimizer.step(&mut self.q1, &scratch.q1_gradient)
            || !self.q2_optimizer.step(&mut self.q2, &scratch.q2_gradient)
        {
            return Some(metrics);
        }

        let mut temperature_gradient = 0.0;
        // Differentiate the selected Q through its action input into the actor.
        // Q parameter gradients below are discarded, so actor updates cannot
        // optimize critic parameters to make their own loss smaller.
        for &index in &scratch.indices {
            let transition = self.replay.transition(index);
            let output = self
                .actor
                .forward(transition.observation, &mut scratch.actor);
            let sample = sample_policy(output, [self.rng.normal(), self.rng.normal()]);
            scratch.q_input[..obs_dim].copy_from_slice(transition.observation);
            scratch.q_input[obs_dim..].copy_from_slice(&sample.action);
            let q1 = self.q1.forward(&scratch.q_input, &mut scratch.actor_q1)[0];
            let q2 = self.q2.forward(&scratch.q_input, &mut scratch.actor_q2)[0];
            let q = q1.min(q2);
            let input_gradient = if q1 <= q2 {
                self.q1.backward(
                    &mut scratch.actor_q1,
                    &[1.0],
                    &mut scratch.unused_q1_gradient,
                )
            } else {
                self.q2.backward(
                    &mut scratch.actor_q2,
                    &[1.0],
                    &mut scratch.unused_q2_gradient,
                )
            };
            let gradient =
                policy_output_gradient(&sample, &input_gradient[obs_dim..], alpha, inverse_batch);
            self.actor
                .backward(&mut scratch.actor, &gradient, &mut scratch.actor_gradient);
            metrics.actor_loss += inverse_batch * (alpha * sample.log_prob - q);
            metrics.entropy -= inverse_batch * sample.log_prob;
            metrics.mean_q += inverse_batch * q;
            let entropy_error = sample.log_prob + self.config.target_entropy;
            temperature_gradient -= inverse_batch * entropy_error;
            metrics.temperature_loss -= inverse_batch * self.log_alpha * entropy_error;
        }
        metrics.actor_grad_norm =
            clip_gradient(&mut scratch.actor_gradient, self.config.max_grad_norm);
        if !metrics.actor_loss.is_finite()
            || !metrics.actor_grad_norm.is_finite()
            || !temperature_gradient.is_finite()
        {
            return Some(metrics);
        }
        if !self
            .actor_optimizer
            .step(&mut self.actor, &scratch.actor_gradient)
        {
            return Some(metrics);
        }
        if self.config.auto_temperature {
            self.temperature_optimizer.update(
                &mut self.log_alpha,
                temperature_gradient,
                self.config.temperature_lr,
            );
            metrics.alpha = self.log_alpha.exp();
        } else {
            metrics.temperature_loss = 0.0;
        }
        self.target_q1.soft_update(&self.q1, self.config.tau);
        self.target_q2.soft_update(&self.q2, self.config.tau);
        self.updates += 1;
        metrics.updates = self.updates;
        metrics.applied = true;
        Some(metrics)
    }
}

impl crate::Agent for Sac {
    fn action(&mut self, observation: &[f32]) -> [f32; ACTION_DIM] {
        Sac::action(self, observation, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SacConfig {
        SacConfig {
            obs_dim: 3,
            hidden: 8,
            replay_capacity: 32,
            batch_size: 8,
            warmup_steps: 0,
            ..SacConfig::default()
        }
    }

    fn fill_replay(sac: &mut Sac) {
        for i in 0..24 {
            let x = i as f32 / 24.0;
            sac.observe(
                &[x, x.sin(), 1.0],
                [0.3 * x, -0.2],
                1.0 - x,
                &[x + 0.01, (x + 0.01).sin(), 1.0],
                i % 7 == 0,
                i % 5 == 0,
            );
        }
    }

    #[test]
    fn replay_overwrites_oldest_and_preserves_end_flags() {
        let mut replay = ReplayBuffer::new(3, 1);
        for i in 0..5 {
            replay.push(
                &[i as f32],
                [0.1, -0.1],
                i as f32,
                &[i as f32 + 1.0],
                i == 4,
                i == 3,
            );
        }
        assert_eq!(replay.len(), 3);
        assert_eq!(replay.cursor, 2);
        assert_eq!(replay.transition(0).observation, &[3.0]);
        assert!(replay.transition(0).truncated);
        assert!(!replay.transition(0).terminated);
        assert_eq!(replay.transition(1).observation, &[4.0]);
        assert!(replay.transition(1).terminated);
        assert_eq!(replay.transition(2).observation, &[2.0]);
        let mut rng = Rng::new(9);
        for _ in 0..100 {
            assert!(replay.sample_index(&mut rng) < 3);
        }
    }

    #[test]
    fn truncation_bootstraps_but_termination_does_not() {
        let mut replay = ReplayBuffer::new(2, 1);
        replay.push(&[0.0], [0.0, 0.0], 2.0, &[1.0], false, true);
        replay.push(&[0.0], [0.0, 0.0], 2.0, &[1.0], true, false);
        let truncated = replay.transition(0);
        let terminated = replay.transition(1);
        assert_eq!(
            bellman_target(truncated.reward, truncated.terminated, 0.9, 10.0),
            11.0
        );
        assert_eq!(
            bellman_target(terminated.reward, terminated.terminated, 0.9, 10.0),
            2.0
        );
    }

    #[test]
    fn pathwise_policy_gradient_matches_finite_difference() {
        let output = [0.3, -0.4, -0.2, 0.1];
        let noise = [0.7, -0.9];
        let q_gradient = [1.7, -0.8];
        let alpha = 0.2;
        let sample = sample_policy(&output, noise);
        let analytic = policy_output_gradient(&sample, &q_gradient, alpha, 1.0);
        let objective = |values: &[f32]| {
            let sample = sample_policy(values, noise);
            alpha * sample.log_prob
                - q_gradient[0] * sample.action[0]
                - q_gradient[1] * sample.action[1]
        };
        for i in 0..4 {
            let mut plus = output;
            let mut minus = output;
            plus[i] += 1e-3;
            minus[i] -= 1e-3;
            let numeric = (objective(&plus) - objective(&minus)) / 2e-3;
            assert!(
                (numeric - analytic[i]).abs() < 3e-4,
                "gradient {i}: {numeric} != {}",
                analytic[i]
            );
        }
        let saturated = sample_policy(&[50.0, -50.0, 3.0, -10.0], noise);
        assert!(saturated.log_prob.is_finite());
        assert_eq!(saturated.log_std_derivative, [0.0, 0.0]);
    }

    #[test]
    fn updates_change_actor_critics_targets_and_keep_losses_finite() {
        let mut sac = Sac::new(test_config(), 19);
        fill_replay(&mut sac);
        let actor_before = sac.actor.parameters().to_vec();
        let q1_before = sac.q1.parameters().to_vec();
        let q2_before = sac.q2.parameters().to_vec();
        let target_before = sac.target_q1.parameters().to_vec();
        let alpha_before = sac.alpha();
        for _ in 0..12 {
            let metrics = sac.update().unwrap();
            assert!(metrics.applied);
            assert!(metrics.actor_loss.is_finite() && metrics.critic_loss.is_finite());
            assert!(metrics.actor_grad_norm > 0.0 && metrics.critic_grad_norm > 0.0);
        }
        assert_ne!(actor_before, sac.actor.parameters());
        assert_ne!(q1_before, sac.q1.parameters());
        assert_ne!(q2_before, sac.q2.parameters());
        assert_ne!(target_before, sac.target_q1.parameters());
        assert_eq!(alpha_before, sac.alpha());
        assert_eq!(sac.updates, 12);
        assert!(sac
            .action(&[0.2, 0.1, 1.0], false)
            .iter()
            .all(|a| (-1.0..=1.0).contains(a)));
    }

    #[test]
    fn target_network_interpolates_after_online_update() {
        let mut config = test_config();
        config.tau = 0.25;
        let mut sac = Sac::new(config, 31);
        fill_replay(&mut sac);
        let old_target = sac.target_q1.parameters().to_vec();
        assert!(sac.update().unwrap().applied);
        for ((old, online), target) in old_target
            .iter()
            .zip(sac.q1.parameters())
            .zip(sac.target_q1.parameters())
        {
            assert!((target - (0.75 * old + 0.25 * online)).abs() < 1e-6);
        }
    }

    #[test]
    fn automatic_temperature_updates_and_checkpoint_resumes_exactly() {
        let mut config = test_config();
        config.auto_temperature = true;
        let mut sac = Sac::new(config, 51);
        fill_replay(&mut sac);
        let original_alpha = sac.alpha();
        assert!(sac.update().unwrap().applied);
        assert_ne!(original_alpha, sac.alpha());
        let bytes = serde_json::to_vec(&sac).unwrap();
        let mut resumed: Sac = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(sac.replay_len(), resumed.replay_len());
        for _ in 0..4 {
            assert_eq!(
                sac.action(&[0.2, 0.1, 1.0], false),
                resumed.action(&[0.2, 0.1, 1.0], false)
            );
            let a = sac.update().unwrap();
            let b = resumed.update().unwrap();
            assert_eq!(a.actor_loss, b.actor_loss);
            assert_eq!(a.critic_loss, b.critic_loss);
            assert_eq!(a.alpha, b.alpha);
            assert_eq!(sac.actor.parameters(), resumed.actor.parameters());
            assert_eq!(sac.q1.parameters(), resumed.q1.parameters());
            assert_eq!(sac.target_q2.parameters(), resumed.target_q2.parameters());
        }
    }

    #[test]
    fn warmup_requires_sufficient_data_and_deterministic_action_does_not_use_rng() {
        let mut config = test_config();
        config.warmup_steps = 25;
        let mut sac = Sac::new(config, 81);
        fill_replay(&mut sac);
        assert!(sac.update().is_none());
        let bytes = serde_json::to_vec(&sac).unwrap();
        let mut other: Sac = serde_json::from_slice(&bytes).unwrap();
        let deterministic = sac.action(&[0.2, 0.1, 1.0], true);
        assert_eq!(deterministic, sac.action(&[0.2, 0.1, 1.0], true));
        assert_eq!(
            sac.action(&[0.2, 0.1, 1.0], false),
            other.action(&[0.2, 0.1, 1.0], false)
        );
        sac.observe(&[0.0; 3], [0.0; 2], 0.0, &[0.0; 3], false, false);
        assert!(sac.update().unwrap().applied);
    }
}
