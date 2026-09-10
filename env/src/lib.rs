//! Deterministic lightweight disk-contact backend. It approximates square-block contact
//! with a disk and does not claim parity with the browser's rigid-body engine.
use std::f32::consts::PI;
pub const DT: f32 = 1.0 / 60.0;
pub const STATE_DIM: usize = 18;
pub const BACKEND: &str = "kinematic-arm-disk-contact-v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObservationKey {
    Bias,
    JointAngles,
    JointVelocities,
    BlockPosition,
    GoalPosition,
    BlockRelative,
    GoalRelative,
    BlockVelocity,
    BlockOrientation,
    BlockAngularVelocity,
    EpisodeTime,
    CommandVelocity,
}
impl ObservationKey {
    pub fn dims(self) -> usize {
        match self {
            Self::Bias | Self::EpisodeTime | Self::BlockAngularVelocity => 1,
            Self::JointAngles
            | Self::JointVelocities
            | Self::BlockPosition
            | Self::GoalPosition
            | Self::BlockRelative
            | Self::GoalRelative
            | Self::BlockVelocity
            | Self::BlockOrientation
            | Self::CommandVelocity => 2,
        }
    }
    pub fn parse(id: &str) -> Option<Self> {
        Some(match id {
            "bias" => Self::Bias,
            "jointAngles" => Self::JointAngles,
            "jointVelocities" => Self::JointVelocities,
            "blockPosition" => Self::BlockPosition,
            "goalPosition" => Self::GoalPosition,
            "blockRelative" => Self::BlockRelative,
            "goalRelative" => Self::GoalRelative,
            "blockVelocity" => Self::BlockVelocity,
            "blockOrientation" => Self::BlockOrientation,
            "blockAngularVelocity" => Self::BlockAngularVelocity,
            "episodeTime" => Self::EpisodeTime,
            "commandVelocity" => Self::CommandVelocity,
            _ => return None,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsLayout {
    pub keys: Vec<ObservationKey>,
    pub dim: usize,
}
impl Default for ObsLayout {
    fn default() -> Self {
        Self::from_keys(&[
            ObservationKey::JointAngles,
            ObservationKey::JointVelocities,
            ObservationKey::BlockPosition,
            ObservationKey::GoalPosition,
            ObservationKey::BlockRelative,
            ObservationKey::GoalRelative,
            ObservationKey::BlockVelocity,
            ObservationKey::BlockAngularVelocity,
            ObservationKey::EpisodeTime,
        ])
    }
}
impl ObsLayout {
    pub fn from_keys(keys: &[ObservationKey]) -> Self {
        assert!(!keys.is_empty());
        let mut seen = std::collections::HashSet::new();
        for k in keys {
            assert!(seen.insert(*k));
        }
        Self {
            keys: keys.to_vec(),
            dim: keys.iter().map(|k| k.dims()).sum(),
        }
    }
    pub fn from_csv(csv: &str) -> Result<Self, String> {
        let mut keys = Vec::new();
        for id in csv.split(',').filter(|x| !x.is_empty()) {
            keys.push(
                ObservationKey::parse(id)
                    .ok_or_else(|| format!("unknown observation key: {id}"))?,
            );
        }
        if keys.is_empty() {
            return Err("observation selection is empty".into());
        }
        Ok(Self::from_keys(&keys))
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}
impl Vec2 {
    pub fn dist(self, o: Self) -> f32 {
        ((self.x - o.x).powi(2) + (self.y - o.y).powi(2)).sqrt()
    }
}
pub fn wrap(a: f32) -> f32 {
    a.sin().atan2(a.cos())
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionMode {
    Position,
    JointDelta,
    Absolute,
    Velocity,
    Acceleration,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvConfig {
    pub action_mode: ActionMode,
    pub max_decisions: u32,
}
impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            action_mode: ActionMode::Position,
            max_decisions: 75,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FullState {
    pub q: [f32; 2],
    pub qv: [f32; 2],
    pub block: Vec2,
    pub block_v: Vec2,
    pub block_angle: f32,
    pub block_angular_velocity: f32,
    pub controller_target: [f32; 2],
    pub controller_velocity: [f32; 2],
    pub goal: Vec2,
    pub elapsed_seconds: f32,
    pub success_hold_seconds: f32,
}
pub fn encode_state(s: &FullState) -> [f32; STATE_DIM] {
    [
        s.q[0],
        s.q[1],
        s.qv[0],
        s.qv[1],
        s.block.x,
        s.block.y,
        s.block_v.x,
        s.block_v.y,
        s.block_angle,
        s.block_angular_velocity,
        s.controller_target[0],
        s.controller_target[1],
        s.controller_velocity[0],
        s.controller_velocity[1],
        s.goal.x,
        s.goal.y,
        s.elapsed_seconds,
        s.success_hold_seconds,
    ]
}
pub fn decode_state(x: [f32; STATE_DIM]) -> FullState {
    FullState {
        q: [x[0], x[1]],
        qv: [x[2], x[3]],
        block: Vec2 { x: x[4], y: x[5] },
        block_v: Vec2 { x: x[6], y: x[7] },
        block_angle: x[8],
        block_angular_velocity: x[9],
        controller_target: [x[10], x[11]],
        controller_velocity: [x[12], x[13]],
        goal: Vec2 { x: x[14], y: x[15] },
        elapsed_seconds: x[16],
        success_hold_seconds: x[17],
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepResult {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub success: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Reward {
    pub total: f32,
    pub goal: f32,
    pub approach: f32,
    pub time: f32,
    pub effort: f32,
    pub terminal: f32,
}

pub fn encode_observation(
    state: &FullState,
    layout: &ObsLayout,
    max_seconds: f32,
    out: &mut [f32],
) {
    assert_eq!(out.len(), layout.dim);
    let base = Vec2 { x: -1.15, y: -0.55 };
    let tip = fk(state.q);
    let mut at = 0;
    for key in &layout.keys {
        let n = key.dims();
        match key {
            ObservationKey::Bias => {
                out[at] = 1.0;
            }
            ObservationKey::JointAngles => {
                out[at] = state.q[0] / PI;
                out[at + 1] = state.q[1] / PI;
            }
            ObservationKey::JointVelocities => {
                out[at] = state.qv[0] / 3.0;
                out[at + 1] = state.qv[1] / 3.0;
            }
            ObservationKey::BlockPosition => {
                out[at] = state.block.x - base.x;
                out[at + 1] = state.block.y - base.y;
            }
            ObservationKey::GoalPosition => {
                out[at] = state.goal.x - base.x;
                out[at + 1] = state.goal.y - base.y;
            }
            ObservationKey::BlockRelative => {
                out[at] = state.block.x - tip.x;
                out[at + 1] = state.block.y - tip.y;
            }
            ObservationKey::GoalRelative => {
                out[at] = state.goal.x - state.block.x;
                out[at + 1] = state.goal.y - state.block.y;
            }
            ObservationKey::BlockVelocity => {
                out[at] = state.block_v.x;
                out[at + 1] = state.block_v.y;
            }
            ObservationKey::BlockOrientation => {
                out[at] = state.block_angle.sin();
                out[at + 1] = state.block_angle.cos();
            }
            ObservationKey::BlockAngularVelocity => {
                out[at] = state.block_angular_velocity / 3.0;
            }
            ObservationKey::EpisodeTime => {
                out[at] = (state.elapsed_seconds / max_seconds.max(1e-6)).clamp(0.0, 1.0);
            }
            ObservationKey::CommandVelocity => {
                out[at] = state.controller_velocity[0] / 1.5;
                out[at + 1] = state.controller_velocity[1] / 1.5;
            }
        };
        at += n;
    }
}
fn fk(q: [f32; 2]) -> Vec2 {
    Vec2 {
        x: -1.15 + 1.05 * q[0].cos() + 0.95 * (q[0] + q[1]).cos(),
        y: -0.55 + 1.05 * q[0].sin() + 0.95 * (q[0] + q[1]).sin(),
    }
}
fn push_point_distance(s: &FullState) -> f32 {
    let dx = s.goal.x - s.block.x;
    let dy = s.goal.y - s.block.y;
    let d = (dx * dx + dy * dy).sqrt().max(1e-6);
    let p = Vec2 {
        x: s.block.x - 0.2 * dx / d,
        y: s.block.y - 0.2 * dy / d,
    };
    fk(s.q).dist(p)
}
pub fn transition_reward(
    prev: &FullState,
    next: &FullState,
    action: [f32; 2],
    config: &EnvConfig,
) -> Reward {
    let success = next.success_hold_seconds + 1e-6 >= 0.25;
    let failure = next.block.dist(Vec2 { x: -1.15, y: -0.55 }) > 1.6;
    let mut r = Reward {
        goal: 100.0 * (prev.block.dist(prev.goal) - next.block.dist(next.goal)),
        approach: 12.0 * (push_point_distance(prev) - push_point_distance(next)),
        time: -0.035,
        effort: -0.008 * (action[0].powi(2) + action[1].powi(2)),
        terminal: if success {
            100.0 + 0.1 * (config.max_decisions as f32 * 4.0 * DT - next.elapsed_seconds).max(0.0)
        } else if failure {
            -30.0
        } else {
            0.0
        },
        total: 0.0,
    };
    r.total = r.goal + r.approach + r.time + r.effort + r.terminal;
    r
}
pub fn evaluate_transition(
    prev: &FullState,
    next: &FullState,
    action: [f32; 2],
    config: &EnvConfig,
) -> StepResult {
    let success = next.success_hold_seconds + 1e-6 >= 0.25;
    let terminated = success || next.block.dist(Vec2 { x: -1.15, y: -0.55 }) > 1.6;
    StepResult {
        reward: transition_reward(prev, next, action, config).total,
        terminated,
        truncated: !terminated
            && next.elapsed_seconds + 1e-5 >= config.max_decisions as f32 * 4.0 * DT,
        success,
    }
}
#[derive(Clone, Debug)]
pub struct Env {
    pub q: [f32; 2],
    pub qv: [f32; 2],
    pub block: Vec2,
    pub goal: Vec2,
    pub block_v: Vec2,
    pub t: u32,
    pub seed: u64,
    pub success_hold: u32,
    pub block_angle: f32,
    pub block_angular_velocity: f32,
    pub controller_target: [f32; 2],
    pub controller_velocity: [f32; 2],
    elapsed_seconds: f32,
    success_hold_seconds: f32,
    config: EnvConfig,
}
impl Env {
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, EnvConfig::default())
    }
    pub fn with_config(seed: u64, config: EnvConfig) -> Self {
        assert!(config.max_decisions > 0);
        let mut e = Self {
            q: [0.0; 2],
            qv: [0.0; 2],
            block: Vec2::default(),
            goal: Vec2::default(),
            block_v: Vec2::default(),
            t: 0,
            seed,
            success_hold: 0,
            block_angle: 0.0,
            block_angular_velocity: 0.0,
            controller_target: [0.0; 2],
            controller_velocity: [0.0; 2],
            elapsed_seconds: 0.0,
            success_hold_seconds: 0.0,
            config,
        };
        e.reset();
        e
    }
    pub fn config(&self) -> EnvConfig {
        self.config
    }
    pub fn full_state(&self) -> FullState {
        FullState {
            q: self.q,
            qv: self.qv,
            block: self.block,
            block_v: self.block_v,
            goal: self.goal,
            block_angle: self.block_angle,
            block_angular_velocity: self.block_angular_velocity,
            controller_target: self.controller_target,
            controller_velocity: self.controller_velocity,
            elapsed_seconds: self.elapsed_seconds,
            success_hold_seconds: self.success_hold_seconds,
        }
    }
    pub fn set_state(&mut self, s: FullState) {
        self.q = s.q;
        self.qv = s.qv;
        self.block = s.block;
        self.block_v = s.block_v;
        self.goal = s.goal;
        self.block_angle = s.block_angle;
        self.block_angular_velocity = s.block_angular_velocity;
        self.controller_target = s.controller_target;
        self.controller_velocity = s.controller_velocity;
        self.elapsed_seconds = s.elapsed_seconds;
        self.success_hold_seconds = s.success_hold_seconds;
        self.t = (s.elapsed_seconds / (4.0 * DT)).round() as u32;
        self.success_hold = (s.success_hold_seconds / (4.0 * DT)).floor() as u32;
    }
    fn rng(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.seed >> 40) as f32) / 16777216.0
    }
    pub fn reset(&mut self) {
        let a = self.rng() * 2.0 * PI;
        let r = 0.95 + self.rng() * 0.25;
        let delta = 0.5 + self.rng() * 0.12;
        self.block = Vec2 {
            x: -1.15 + r * a.cos(),
            y: -0.55 + r * a.sin(),
        };
        self.goal = Vec2 {
            x: -1.15 + r * (a + delta).cos(),
            y: -0.55 + r * (a + delta).sin(),
        };
        let ee = Vec2 {
            x: -1.15 + r * (a - 0.2).cos(),
            y: -0.55 + r * (a - 0.2).sin(),
        };
        self.q = [0.0; 2];
        self.q = self.ik(ee);
        self.qv = [0.0; 2];
        self.block_v = Vec2::default();
        self.t = 0;
        self.success_hold = 0;
        self.elapsed_seconds = 0.0;
        self.success_hold_seconds = 0.0;
        self.block_angle = 0.0;
        self.block_angular_velocity = 0.0;
        self.controller_target = self.q;
        self.controller_velocity = [0.0; 2];
    }
    pub fn ik(&self, p: Vec2) -> [f32; 2] {
        let x = p.x + 1.15;
        let y = p.y + 0.55;
        let c =
            ((x * x + y * y - 1.05 * 1.05 - 0.95 * 0.95) / (2.0 * 1.05 * 0.95)).clamp(-1.0, 1.0);
        let q2 = c.acos().clamp(-2.75, 2.75);
        let q1 = y.atan2(x) - (0.95 * q2.sin()).atan2(1.05 + 0.95 * q2.cos());
        [self.q[0] + wrap(q1 - self.q[0]), q2]
    }
    pub fn end_effector(&self) -> Vec2 {
        fk(self.q)
    }
    pub fn obs_dim(&self) -> usize {
        16
    }
    pub fn observe(&self, out: &mut [f32]) {
        assert_eq!(out.len(), 16);
        self.observe_layout(&ObsLayout::default(), out);
    }
    pub fn observe_layout(&self, layout: &ObsLayout, out: &mut [f32]) {
        encode_observation(
            &self.full_state(),
            layout,
            self.config.max_decisions as f32 * 4.0 * DT,
            out,
        );
    }
    pub fn obs(&self) -> [f32; 16] {
        let mut out = [0.0; 16];
        self.observe(&mut out);
        out
    }
    /// One 60 Hz contact step. Goal is read only to update the success timer.
    pub fn physics_step(&mut self, target: [f32; 2]) {
        let old = self.end_effector();
        for (j, value) in target.iter().enumerate() {
            let error = if j == 0 {
                wrap(*value - self.q[j])
            } else {
                *value - self.q[j]
            };
            self.qv[j] = (error * 20.0).clamp(-2.8, 2.8);
            self.q[j] += self.qv[j] * DT;
        }
        self.q[1] = self.q[1].clamp(-2.75, 2.75);
        let ee = self.end_effector();
        self.block.x += self.block_v.x * DT;
        self.block.y += self.block_v.y * DT;
        self.block_angle += self.block_angular_velocity * DT;
        self.block_v.x *= 0.82;
        self.block_v.y *= 0.82;
        self.block_angular_velocity *= 0.82;
        let dx = self.block.x - ee.x;
        let dy = self.block.y - ee.y;
        let d = (dx * dx + dy * dy).sqrt();
        let radius = 0.11;
        if d < radius {
            let (nx, ny) = if d > 1e-6 {
                (dx / d, dy / d)
            } else {
                (1.0, 0.0)
            };
            self.block.x += nx * (radius - d);
            self.block.y += ny * (radius - d);
            let vx = (ee.x - old.x) / DT;
            let vy = (ee.y - old.y) / DT;
            let normal_speed = (vx - self.block_v.x) * nx + (vy - self.block_v.y) * ny;
            let impulse = normal_speed.max(0.0);
            let tangential_speed = (vx - self.block_v.x) * (-ny) + (vy - self.block_v.y) * nx
                - self.block_angular_velocity * 0.07;
            let friction = tangential_speed.clamp(-0.3 * impulse, 0.3 * impulse);
            self.block_v.x += impulse * nx - friction * ny;
            self.block_v.y += impulse * ny + friction * nx;
            self.block_angular_velocity -= friction * 2.0 / 0.07;
        }
        self.elapsed_seconds += DT;
        if self.block.dist(self.goal) <= 0.07 {
            self.success_hold_seconds += DT;
        } else {
            self.success_hold_seconds = 0.0;
        }
    }
    pub fn step_result(&mut self, action: [f32; 2]) -> StepResult {
        let a = [action[0].clamp(-1.0, 1.0), action[1].clamp(-1.0, 1.0)];
        let prev = self.full_state();
        match self.config.action_mode {
            ActionMode::Position => {
                let ee = self.end_effector();
                self.controller_target = self.ik(Vec2 {
                    x: ee.x + 0.04 * a[0],
                    y: ee.y + 0.04 * a[1],
                });
            }
            ActionMode::JointDelta => {
                self.controller_target = [
                    self.q[0] + 0.15 * a[0],
                    (self.q[1] + 0.15 * a[1]).clamp(-2.75, 2.75),
                ];
            }
            ActionMode::Absolute => {
                self.controller_target = [self.q[0] + wrap(PI * a[0] - self.q[0]), 2.75 * a[1]];
            }
            ActionMode::Velocity => {
                self.controller_velocity = [0.9 * a[0], 0.9 * a[1]];
            }
            ActionMode::Acceleration => {
                for (j, value) in a.iter().enumerate() {
                    self.controller_velocity[j] =
                        (self.controller_velocity[j] + 3.0 * value * DT * 4.0).clamp(-1.5, 1.5);
                }
            }
        }
        for _ in 0..4 {
            if matches!(
                self.config.action_mode,
                ActionMode::Velocity | ActionMode::Acceleration
            ) {
                for j in 0..2 {
                    self.controller_target[j] = self.q[j] + self.controller_velocity[j] / 20.0;
                }
            }
            self.physics_step(self.controller_target);
        }
        self.t += 1;
        self.success_hold = (self.success_hold_seconds / (4.0 * DT)).floor() as u32;
        evaluate_transition(&prev, &self.full_state(), a, &self.config)
    }
    pub fn step(&mut self, a: [f32; 2]) -> (f32, bool) {
        let r = self.step_result(a);
        (r.reward, r.terminated || r.truncated)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic() {
        let mut a = Env::new(99);
        let mut b = Env::new(99);
        for i in 0..150 {
            let act = [(i as f32).sin(), (i as f32).cos()];
            assert_eq!(a.step(act), b.step(act));
            assert_eq!(a.full_state(), b.full_state());
        }
    }
    #[test]
    fn layouts() {
        for seed in 0..1000 {
            let e = Env::new(seed);
            let base = Vec2 { x: -1.15, y: -0.55 };
            assert!((0.65..=1.65).contains(&e.block.dist(base)));
            assert!((0.65..=1.65).contains(&e.goal.dist(base)));
            assert!((0.4..=0.8).contains(&e.block.dist(e.goal)));
            assert!(e.block.dist(e.end_effector()) > 0.11);
        }
    }
    #[test]
    fn observations_include_velocity() {
        let mut e = Env::new(0);
        e.block_v = Vec2 { x: 0.3, y: -0.4 };
        e.block_angular_velocity = 0.7;
        assert_eq!(&e.obs()[12..15], &[0.3, -0.4, 0.7 / 3.0]);
    }
    #[test]
    fn goal_does_not_affect_dynamics() {
        let mut a = Env::new(123);
        let mut b = a.clone();
        b.goal = Vec2 { x: 100., y: 100. };
        for _ in 0..20 {
            a.step([1., 0.2]);
            b.step([1., 0.2]);
            assert_eq!(a.block, b.block);
            assert_eq!(a.block_v, b.block_v);
        }
    }
    #[test]
    fn timeout_75_decisions() {
        let mut e = Env::new(1);
        for _ in 0..74 {
            assert!(!e.step([0., 0.]).1);
        }
        assert!(e.step_result([0., 0.]).truncated);
    }
    #[test]
    fn success_requires_hold() {
        let mut e = Env::new(1);
        e.goal = e.block;
        for _ in 0..3 {
            assert!(!e.step([0., 0.]).1);
        }
        assert!(e.step_result([0., 0.]).success);
    }
    #[test]
    fn shoulder_wrap() {
        assert!((wrap(-PI + 0.01 - (PI - 0.01)) - 0.02).abs() < 1e-5);
    }
    #[test]
    fn state_roundtrip_resumes() {
        let mut a = Env::with_config(
            8,
            EnvConfig {
                action_mode: ActionMode::Acceleration,
                max_decisions: 75,
            },
        );
        a.step([0.4, -0.1]);
        let state = decode_state(encode_state(&a.full_state()));
        let mut b = a.clone();
        b.set_state(state);
        for _ in 0..20 {
            assert_eq!(a.step([0.2, -0.5]), b.step([0.2, -0.5]));
            assert_eq!(a.full_state(), b.full_state());
        }
    }
    #[test]
    fn velocity_units_and_continuous_shoulder() {
        let mut e = Env::with_config(
            4,
            EnvConfig {
                action_mode: ActionMode::Velocity,
                max_decisions: 75,
            },
        );
        e.q[0] = PI - 0.001;
        let before = e.q[0];
        e.step([1.0, 0.0]);
        assert!(e.q[0] > PI);
        assert!((e.q[0] - before - 0.9 * 4.0 * DT).abs() < 1e-5);
    }
    #[test]
    fn every_action_mode_finite() {
        for action_mode in [
            ActionMode::Position,
            ActionMode::JointDelta,
            ActionMode::Absolute,
            ActionMode::Velocity,
            ActionMode::Acceleration,
        ] {
            let mut e = Env::with_config(
                7,
                EnvConfig {
                    action_mode,
                    max_decisions: 75,
                },
            );
            for _ in 0..75 {
                let result = e.step_result([0.5, -0.7]);
                assert!(result.reward.is_finite());
                assert!(encode_state(&e.full_state()).iter().all(|x| x.is_finite()));
            }
        }
    }
    #[test]
    fn free_spin_and_decay() {
        let mut e = Env::new(2);
        e.block_angular_velocity = 1.0;
        e.physics_step(e.q);
        assert!((e.block_angle - DT).abs() < 1e-6);
        assert!((e.block_angular_velocity - 0.82).abs() < 1e-6);
    }
}
