//! Adapters exposing the analytic simulator and learned dynamics to PPO.
//!
//! WorldEnv uses Env only for observation encoding and analytic reward logic;
//! its state transitions come exclusively from WorldModel::predict.
use push_env::{
    decode_state, encode_state, evaluate_transition, Env, EnvConfig, ObsLayout, STATE_DIM,
};
use push_rl::{Environment, StepResult};
use push_world::WorldModel;

pub(crate) struct RealEnv {
    env: Env,
    layout: ObsLayout,
}
impl RealEnv {
    pub(crate) fn new(seed: u64, config: EnvConfig, layout: ObsLayout) -> Self {
        Self {
            env: Env::with_config(seed, config),
            layout,
        }
    }
}
impl Environment for RealEnv {
    fn obs_dim(&self) -> usize {
        self.layout.dim
    }
    fn observe(&self, out: &mut [f32]) {
        self.env.observe_layout(&self.layout, out)
    }
    fn step_env(&mut self, a: [f32; 2]) -> StepResult {
        let r = self.env.step_result(a);
        StepResult {
            reward: r.reward,
            terminated: r.terminated,
            truncated: r.truncated,
            success: r.success,
        }
    }
    fn reset_env(&mut self) {
        self.env.reset()
    }
}
pub(crate) struct WorldEnv {
    model: WorldModel,
    workspace: push_world::PredictionWorkspace,
    pub(crate) state: Vec<f32>,
    starts: Vec<Vec<f32>>,
    cursor: usize,
    shadow: Env,
    uncertainty_limit: f32,
    uncertainty_penalty: f32,
    /// Maximum number of imagined steps before a receding-horizon reset.
    /// A truncated model rollout is bootstrapped by PPO's value function.
    rollout_horizon: usize,
    steps_since_reset: usize,
    last_uncertainty: f32,
    pub(crate) last_uncertainty_guarded: bool,
    pub(crate) last_horizon_truncated: bool,
    layout: ObsLayout,
}
impl WorldEnv {
    pub(crate) fn new(
        model: WorldModel,
        starts: Vec<Vec<f32>>,
        layout: ObsLayout,
        rollout_horizon: usize,
        uncertainty_limit: f32,
    ) -> Self {
        Self::with_penalty(
            model,
            starts,
            layout,
            rollout_horizon,
            uncertainty_limit,
            0.0,
        )
    }
    pub(crate) fn with_penalty(
        model: WorldModel,
        starts: Vec<Vec<f32>>,
        layout: ObsLayout,
        rollout_horizon: usize,
        uncertainty_limit: f32,
        uncertainty_penalty: f32,
    ) -> Self {
        let mut shadow = Env::with_config(1, EnvConfig::default());
        shadow.set_state(decode_state(
            starts[0].clone().try_into().expect("state dim"),
        ));
        let workspace = model.workspace();
        Self {
            model,
            workspace,
            state: starts[0].clone(),
            starts,
            cursor: 0,
            shadow,
            uncertainty_limit,
            uncertainty_penalty,
            rollout_horizon,
            steps_since_reset: 0,
            last_uncertainty: 0.0,
            last_uncertainty_guarded: false,
            last_horizon_truncated: false,
            layout,
        }
    }
}
impl Environment for WorldEnv {
    fn obs_dim(&self) -> usize {
        self.layout.dim
    }
    fn observe(&self, out: &mut [f32]) {
        let mut shadow = self.shadow.clone();
        shadow.set_state(decode_state(
            self.state.clone().try_into().expect("state dim"),
        ));
        shadow.observe_layout(&self.layout, out);
    }
    fn step_env(&mut self, action: [f32; 2]) -> StepResult {
        let prev = decode_state(self.state.clone().try_into().expect("state dim"));
        let mut next = vec![0.0; STATE_DIM];
        let uncertainty =
            self.model
                .predict(&self.state, action, &mut next, None, &mut self.workspace);
        let mut ns = decode_state(next.clone().try_into().expect("state dim"));
        let distance = ns.block.dist(ns.goal);
        ns.success_hold_seconds = if distance <= 0.07 {
            prev.success_hold_seconds + 1.0 / 15.0
        } else {
            0.0
        };
        next.copy_from_slice(&encode_state(&ns));
        self.state = next;
        self.steps_since_reset += 1;
        let result = evaluate_transition(&prev, &ns, action, &EnvConfig::default());
        let horizon_truncated =
            self.rollout_horizon > 0 && self.steps_since_reset >= self.rollout_horizon;
        self.last_uncertainty = uncertainty;
        self.last_uncertainty_guarded = uncertainty > self.uncertainty_limit;
        self.last_horizon_truncated = horizon_truncated;
        StepResult {
            reward: result.reward - self.uncertainty_penalty * uncertainty,
            terminated: result.terminated,
            truncated: result.truncated
                || horizon_truncated
                || uncertainty > self.uncertainty_limit,
            success: result.success,
        }
    }
    fn reset_env(&mut self) {
        self.cursor = (self.cursor + 1) % self.starts.len();
        self.state = self.starts[self.cursor].clone();
        self.steps_since_reset = 0;
        self.last_uncertainty = 0.0;
        self.last_uncertainty_guarded = false;
        self.last_horizon_truncated = false;
    }
}
