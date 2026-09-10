use push_env::{Env, EnvConfig};
use push_rl::StepResult;
use push_rl::{Environment, Ppo, PpoConfig};
use std::time::Instant;
struct RealEnv(Env);
impl Environment for RealEnv {
    fn obs_dim(&self) -> usize {
        self.0.obs_dim()
    }
    fn observe(&self, out: &mut [f32]) {
        self.0.observe(out)
    }
    fn step_env(&mut self, a: [f32; 2]) -> StepResult {
        {
            let r = self.0.step_result(a);
            StepResult {
                reward: r.reward,
                terminated: r.terminated,
                truncated: r.truncated,
                success: r.success,
            }
        }
    }
    fn reset_env(&mut self) {
        self.0.reset()
    }
}
fn main() {
    let cpus = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let max_envs = cpus.saturating_sub(2).max(1);
    println!(
        "backend={} logical_cpus={} max_parallel_envs={}",
        push_env::BACKEND,
        cpus,
        max_envs
    );
    for &n in [1, 2, 4, 8, 16, 32, 64]
        .iter()
        .filter(|&&n| n <= max_envs.max(1))
    {
        let mut envs = (0..n)
            .map(|i| RealEnv(Env::with_config(i as u64, EnvConfig::default())))
            .collect::<Vec<_>>();
        let physics_start = Instant::now();
        let physics_steps = 20_000usize;
        for _ in 0..physics_steps {
            for e in &mut envs {
                let _ = e.0.step_result([0.11, -0.07]);
            }
        }
        let physics_sps = (physics_steps * n) as f64 / physics_start.elapsed().as_secs_f64();
        for e in &mut envs {
            e.0.reset();
        }
        let mut p = Ppo::new(
            16,
            PpoConfig {
                epochs: 1,
                minibatch_size: 256,
                ..Default::default()
            },
            7,
        );
        let train_start = Instant::now();
        let stats = p.train(&mut envs, 256);
        let policy_sps = stats.steps as f64 / train_start.elapsed().as_secs_f64();
        println!("envs={n:>3} physics_steps_s={physics_sps:>12.0} ppo_end_to_end_steps_s={policy_sps:>12.0} episodes={} success={}",stats.episodes,stats.successes);
    }
}
