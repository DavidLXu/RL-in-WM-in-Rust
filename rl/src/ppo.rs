use crate::neural::{Adam, Mlp, MlpGrad, Rng, Workspace};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

pub trait Environment: Send {
    fn obs_dim(&self) -> usize;
    fn observe(&self, out: &mut [f32]);
    fn step_env(&mut self, action: [f32; 2]) -> StepResult;
    fn reset_env(&mut self);
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StepResult {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub success: bool,
}
pub trait Agent {
    fn action(&mut self, obs: &[f32]) -> [f32; 2];
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PpoConfig {
    pub hidden: usize,
    pub learning_rate: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_ratio: f32,
    pub value_clip: f32,
    pub entropy_coef: f32,
    pub value_coef: f32,
    pub max_grad_norm: f32,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub parallel: bool,
}
impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            hidden: 32,
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_ratio: 0.2,
            value_clip: 0.2,
            entropy_coef: 0.0,
            value_coef: 0.5,
            max_grad_norm: 0.5,
            epochs: 4,
            minibatch_size: 256,
            parallel: true,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainStats {
    pub steps: usize,
    pub episodes: usize,
    pub successes: usize,
    pub reward_sum: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub updates: usize,
    pub rejected_updates: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ppo {
    pub actor: Mlp,
    pub critic: Mlp,
    pub actor_optimizer: Adam,
    pub critic_optimizer: Adam,
    pub rng: Rng,
    pub config: PpoConfig,
    #[serde(skip)]
    actor_workspace: Option<Workspace>,
    #[serde(skip)]
    critic_workspace: Option<Workspace>,
}
struct Trajectory {
    obs: Vec<f32>,
    actions: Vec<[f32; 2]>,
    old_logp: Vec<f32>,
    old_values: Vec<f32>,
    rewards: Vec<f32>,
    next_values: Vec<f32>,
    dones: Vec<bool>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    episodes: usize,
    successes: usize,
    reward_sum: f32,
}
impl Ppo {
    pub fn new(obs_dim: usize, config: PpoConfig, seed: u64) -> Self {
        let mut r = Rng::new(seed);
        let actor = Mlp::new(obs_dim, config.hidden, 4, &mut r);
        let critic = Mlp::new(obs_dim, config.hidden, 1, &mut r);
        let ao = Adam::new(&actor, config.learning_rate);
        let co = Adam::new(&critic, config.learning_rate);
        Self {
            actor,
            critic,
            actor_optimizer: ao,
            critic_optimizer: co,
            rng: r,
            config,
            actor_workspace: None,
            critic_workspace: None,
        }
    }
    pub fn deterministic_action(&mut self, obs: &[f32]) -> [f32; 2] {
        let ws = self
            .actor_workspace
            .get_or_insert_with(|| self.actor.workspace());
        let o = self.actor.forward(obs, ws);
        [o[0].tanh(), o[1].tanh()]
    }
    fn ensure_workspaces(&mut self) {
        if self.actor_workspace.is_none() {
            self.actor_workspace = Some(self.actor.workspace());
        }
        if self.critic_workspace.is_none() {
            self.critic_workspace = Some(self.critic.workspace());
        }
    }
    fn logp_and_grads(
        mean: [f32; 2],
        logstd: [f32; 2],
        action: [f32; 2],
    ) -> (f32, [f32; 2], [f32; 2]) {
        let mut lp = 0.;
        let mut gm = [0.; 2];
        let mut gs = [0.; 2];
        for i in 0..2 {
            let a = action[i].clamp(-0.999999, 0.999999);
            let u = 0.5 * ((1. + a) / (1. - a)).ln();
            let s = logstd[i].clamp(-5., 2.).exp();
            let z = (u - mean[i]) / s;
            lp += -0.5 * z * z - logstd[i] - 0.5 * (2. * PI).ln() + (1. - a * a).max(1e-7).ln();
            gm[i] = z / s;
            gs[i] = z * z - 1.;
        }
        (lp, gm, gs)
    }
    fn collect<E: Environment>(
        env: &mut E,
        actor: &Mlp,
        critic: &Mlp,
        steps: usize,
        mut rng: Rng,
        gamma: f32,
        lambda: f32,
    ) -> Trajectory {
        let dim = env.obs_dim();
        let mut obs = vec![0.; dim];
        env.observe(&mut obs);
        let mut t = Trajectory {
            obs: Vec::with_capacity(steps * dim),
            actions: Vec::with_capacity(steps),
            old_logp: Vec::with_capacity(steps),
            old_values: Vec::with_capacity(steps),
            rewards: Vec::with_capacity(steps),
            next_values: Vec::with_capacity(steps),
            dones: Vec::with_capacity(steps),
            advantages: vec![0.; steps],
            returns: vec![0.; steps],
            episodes: 0,
            successes: 0,
            reward_sum: 0.,
        };
        let mut aws = actor.workspace();
        let mut cws = critic.workspace();
        for _ in 0..steps {
            t.obs.extend_from_slice(&obs);
            let ao = actor.forward(&obs, &mut aws);
            let mut action = [0.; 2];
            let mut mean = [0.; 2];
            let mut ls = [0.; 2];
            for j in 0..2 {
                mean[j] = ao[j];
                ls[j] = ao[j + 2].clamp(-5., 2.);
                action[j] = (mean[j] + ls[j].exp() * rng.normal()).tanh();
            }
            let (lp, _, _) = Self::logp_and_grads(mean, ls, action);
            let value = critic.forward(&obs, &mut cws)[0];
            t.actions.push(action);
            t.old_logp.push(lp);
            t.old_values.push(value);
            let sr = env.step_env(action);
            let reward = if sr.reward.is_finite() { sr.reward } else { 0. };
            t.rewards.push(reward);
            t.reward_sum += reward;
            let done = sr.terminated || sr.truncated;
            t.dones.push(done);
            env.observe(&mut obs);
            let next_v = if sr.terminated {
                0.
            } else {
                critic.forward(&obs, &mut cws)[0]
            };
            t.next_values.push(next_v);
            if done {
                t.episodes += 1;
                if sr.success {
                    t.successes += 1;
                }
                env.reset_env();
                env.observe(&mut obs);
            }
        }
        let mut gae = 0.;
        for i in (0..steps).rev() {
            let delta = t.rewards[i]
                + if t.next_values[i].is_finite() {
                    gamma * t.next_values[i]
                } else {
                    0.
                }
                - t.old_values[i];
            gae = delta + if t.dones[i] { 0. } else { gamma * lambda * gae };
            t.advantages[i] = gae;
            t.returns[i] = t.old_values[i] + gae;
        }
        t
    }
    pub fn train<E: Environment>(&mut self, envs: &mut [E], rollout_steps: usize) -> TrainStats {
        self.ensure_workspaces();
        if envs.is_empty() || rollout_steps == 0 {
            return TrainStats::default();
        }
        let mut seeds = Vec::with_capacity(envs.len());
        for _ in 0..envs.len() {
            seeds.push(self.rng.next_u64());
        }
        let actor = &self.actor;
        let critic = &self.critic;
        let c = self.config.clone();
        let mut trajectories: Vec<Trajectory> = if c.parallel {
            envs.par_iter_mut()
                .zip(seeds.par_iter())
                .map(|(e, s)| {
                    Self::collect(
                        e,
                        actor,
                        critic,
                        rollout_steps,
                        Rng::new(*s),
                        c.gamma,
                        c.gae_lambda,
                    )
                })
                .collect()
        } else {
            envs.iter_mut()
                .zip(seeds.iter())
                .map(|(e, s)| {
                    Self::collect(
                        e,
                        actor,
                        critic,
                        rollout_steps,
                        Rng::new(*s),
                        c.gamma,
                        c.gae_lambda,
                    )
                })
                .collect()
        };
        let total = trajectories.iter().map(|t| t.rewards.len()).sum();
        let dim = self.actor.input_dim();
        let mut obs = Vec::with_capacity(total * dim);
        let mut acts = Vec::with_capacity(total);
        let mut oldlp = Vec::with_capacity(total);
        let mut oldv = Vec::with_capacity(total);
        let mut adv = Vec::with_capacity(total);
        let mut ret = Vec::with_capacity(total);
        let mut stats = TrainStats {
            steps: total,
            ..Default::default()
        };
        for t in &mut trajectories {
            stats.episodes += t.episodes;
            stats.successes += t.successes;
            stats.reward_sum += t.reward_sum;
            let mean = t.advantages.iter().sum::<f32>() / (t.advantages.len().max(1) as f32);
            let sd = (t
                .advantages
                .iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f32>()
                / (t.advantages.len().max(1) as f32))
                .sqrt()
                .max(1e-6);
            for i in 0..t.rewards.len() {
                obs.extend_from_slice(&t.obs[i * dim..(i + 1) * dim]);
                acts.push(t.actions[i]);
                oldlp.push(t.old_logp[i]);
                oldv.push(t.old_values[i]);
                adv.push((t.advantages[i] - mean) / sd);
                ret.push(t.old_values[i] + t.advantages[i]);
            }
        }
        let mut indices: Vec<usize> = (0..total).collect();
        let mut pg = MlpGrad::new(&self.actor);
        let mut vg = MlpGrad::new(&self.critic);
        let mut aws = self.actor.workspace();
        let mut cws = self.critic.workspace();
        for _ in 0..c.epochs {
            self.rng.shuffle(&mut indices);
            for chunk in indices.chunks(c.minibatch_size.max(1)) {
                pg.zero();
                vg.zero();
                let scale = 1.0 / chunk.len() as f32;
                let mut pl = 0.;
                let mut vl = 0.;
                for &i in chunk {
                    let ao = self
                        .actor
                        .forward(&obs[i * dim..(i + 1) * dim], &mut aws)
                        .to_vec();
                    let mean = [ao[0], ao[1]];
                    let ls = [ao[2].clamp(-5., 2.), ao[3].clamp(-5., 2.)];
                    let (lp, gm, gs) = Self::logp_and_grads(mean, ls, acts[i]);
                    let ratio = (lp - oldlp[i]).exp();
                    let clipped = ratio.clamp(1. - c.clip_ratio, 1. + c.clip_ratio);
                    let use_ratio = if adv[i] >= 0. {
                        ratio <= clipped
                    } else {
                        ratio >= clipped
                    };
                    let coeff = if use_ratio {
                        -adv[i] * ratio * scale
                    } else {
                        0.
                    };
                    let mut go = [coeff * gm[0], coeff * gm[1], coeff * gs[0], coeff * gs[1]];
                    for j in 0..2 {
                        go[j + 2] += -c.entropy_coef
                            * (1.
                                - 2. * ((acts[i][j].atanh() - mean[j]).tanh()
                                    * (acts[i][j].atanh() - mean[j])));
                    }
                    self.actor.backward(&mut aws, &go, &mut pg);
                    pl += -adv[i]
                        * if use_ratio {
                            ratio.ln()
                        } else {
                            (clipped).ln()
                        };
                    let v = self.critic.forward(&obs[i * dim..(i + 1) * dim], &mut cws)[0];
                    let dv = v - ret[i];
                    let clipped_v = oldv[i] + (v - oldv[i]).clamp(-c.value_clip, c.value_clip);
                    let use_v = (v - ret[i]).abs() <= (clipped_v - ret[i]).abs();
                    let dvc = if use_v { dv } else { 0. };
                    self.critic
                        .backward(&mut cws, &[c.value_coef * dvc * scale], &mut vg);
                    vl += 0.5 * dvc * dvc;
                }
                if c.max_grad_norm > 0. {
                    let n = pg.norm();
                    if n > c.max_grad_norm {
                        pg.scale(c.max_grad_norm / n);
                    }
                    let n = vg.norm();
                    if n > c.max_grad_norm {
                        vg.scale(c.max_grad_norm / n);
                    }
                }
                let ok1 = self.actor_optimizer.step(&mut self.actor, &pg);
                let ok2 = self.critic_optimizer.step(&mut self.critic, &vg);
                if !ok1 {
                    stats.rejected_updates += 1;
                }
                if !ok2 {
                    stats.rejected_updates += 1;
                }
                stats.policy_loss += pl / chunk.len() as f32;
                stats.value_loss += vl / chunk.len() as f32;
                stats.updates += 1;
            }
        }
        let denom = stats.updates.max(1) as f32;
        stats.policy_loss /= denom;
        stats.value_loss /= denom;
        stats
    }
    pub fn save_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    pub fn load_json(s: &str) -> Result<Self, serde_json::Error> {
        let mut p: Self = serde_json::from_str(s)?;
        p.actor_workspace = None;
        p.critic_workspace = None;
        Ok(p)
    }
}
impl Agent for Ppo {
    fn action(&mut self, obs: &[f32]) -> [f32; 2] {
        let ws = self
            .actor_workspace
            .get_or_insert_with(|| self.actor.workspace());
        let o = self.actor.forward(obs, ws);
        let mut a = [0.; 2];
        for i in 0..2 {
            a[i] = (o[i] + o[i + 2].clamp(-5., 2.).exp() * self.rng.normal()).tanh();
        }
        a
    }
}
impl Default for Ppo {
    fn default() -> Self {
        Self::new(16, PpoConfig::default(), 0x5eed)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Bandit {
        x: f32,
        n: usize,
    }
    impl Environment for Bandit {
        fn obs_dim(&self) -> usize {
            1
        }
        fn observe(&self, out: &mut [f32]) {
            out[0] = self.x
        }
        fn step_env(&mut self, a: [f32; 2]) -> StepResult {
            self.n += 1;
            StepResult {
                reward: 1.0 - (a[0] - 0.7).powi(2),
                terminated: true,
                truncated: false,
                success: true,
            }
        }
        fn reset_env(&mut self) {
            self.n = 0;
        }
    }
    #[test]
    fn ppo_updates_weights_and_checkpoint_rehydrates_workspaces() {
        let mut cfg = PpoConfig::default();
        cfg.hidden = 8;
        cfg.epochs = 2;
        cfg.minibatch_size = 8;
        cfg.learning_rate = 1e-3;
        let mut p = Ppo::new(1, cfg, 9);
        let before = p.actor.parameters().to_vec();
        let mut e = (0..4).map(|_| Bandit { x: 1., n: 0 }).collect::<Vec<_>>();
        let st = p.train(&mut e, 8);
        assert!(st.updates > 0);
        assert!(p
            .actor
            .parameters()
            .iter()
            .zip(before)
            .any(|(a, b)| a != &b));
        let s = p.save_json().unwrap();
        let mut q = Ppo::load_json(&s).unwrap();
        assert_eq!(q.actor.parameters(), p.actor.parameters());
        assert_eq!(q.deterministic_action(&[1.]), p.deterministic_action(&[1.]));
    }
}
