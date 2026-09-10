//! Small, allocation-free-at-inference, two-hidden-layer tanh networks.
use serde::{Deserialize, Serialize};

/// Serializable SplitMix64 stream; Box-Muller normal samples use an open unit interval.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rng {
    state: u64,
    spare: Option<f32>,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f32 {
        // 23 random bits: the half-bin offset stays strictly below one in f32.
        (((self.next_u64() >> 41) as f32) + 0.5) * (1.0 / 8388608.0)
    }
    pub fn normal(&mut self) -> f32 {
        if let Some(x) = self.spare.take() {
            return x;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let theta = std::f32::consts::TAU * self.uniform();
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
    pub fn index(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            // Rejection sampling avoids modulo bias.
            let range = (i + 1) as u64;
            let threshold = range.wrapping_neg() % range;
            let r = loop {
                let r = self.next_u64();
                if r >= threshold {
                    break r;
                }
            };
            values.swap(i, (r % range) as usize);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mlp {
    input: usize,
    hidden: usize,
    output: usize,
    parameters: Vec<f32>,
}
#[derive(Clone, Debug)]
pub struct Workspace {
    input: Vec<f32>,
    h1: Vec<f32>,
    h2: Vec<f32>,
    output: Vec<f32>,
    d1: Vec<f32>,
    d2: Vec<f32>,
    input_gradient: Vec<f32>,
}
#[derive(Clone, Debug)]
pub struct MlpGrad {
    pub data: Vec<f32>,
}
impl MlpGrad {
    pub fn new(model: &Mlp) -> Self {
        Self {
            data: vec![0.0; model.parameters.len()],
        }
    }
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }
    pub fn scale(&mut self, factor: f32) {
        for g in &mut self.data {
            *g *= factor;
        }
    }
    pub fn norm(&self) -> f32 {
        self.data
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt() as f32
    }
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }
}
impl Mlp {
    pub fn new(input: usize, hidden: usize, output: usize, rng: &mut Rng) -> Self {
        assert!(input > 0 && hidden > 0 && output > 0);
        let n = hidden * input + hidden + hidden * hidden + hidden + output * hidden + output;
        let mut m = Self {
            input,
            hidden,
            output,
            parameters: vec![0.0; n],
        };
        let mut offset = 0;
        for (ins, outs) in [(input, hidden), (hidden, hidden), (hidden, output)] {
            let scale = (6.0 / (ins + outs) as f32).sqrt();
            for w in &mut m.parameters[offset..offset + ins * outs] {
                *w = (2.0 * rng.uniform() - 1.0) * scale;
            }
            offset += ins * outs + outs;
        }
        m
    }
    pub fn input_dim(&self) -> usize {
        self.input
    }
    pub fn output_dim(&self) -> usize {
        self.output
    }
    pub fn hidden_dim(&self) -> usize {
        self.hidden
    }
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    pub fn parameters_mut(&mut self) -> &mut [f32] {
        &mut self.parameters
    }
    pub fn workspace(&self) -> Workspace {
        Workspace {
            input: vec![0.0; self.input],
            h1: vec![0.0; self.hidden],
            h2: vec![0.0; self.hidden],
            output: vec![0.0; self.output],
            d1: vec![0.0; self.hidden],
            d2: vec![0.0; self.hidden],
            input_gradient: vec![0.0; self.input],
        }
    }
    pub fn forward<'a>(&self, input: &[f32], w: &'a mut Workspace) -> &'a [f32] {
        assert_eq!(input.len(), self.input);
        w.input.copy_from_slice(input);
        let h = self.hidden;
        let p = &self.parameters;
        dense(
            &w.input,
            &p[..h * self.input],
            &p[h * self.input..h * self.input + h],
            &mut w.h1,
            true,
        );
        let b = h * self.input + h;
        dense(
            &w.h1,
            &p[b..b + h * h],
            &p[b + h * h..b + h * h + h],
            &mut w.h2,
            true,
        );
        let b = b + h * h + h;
        dense(
            &w.h2,
            &p[b..b + self.output * h],
            &p[b + self.output * h..],
            &mut w.output,
            false,
        );
        &w.output
    }
    /// Accumulates parameter gradients and returns d(loss)/d(input). Must follow forward.
    pub fn backward<'a>(
        &self,
        w: &'a mut Workspace,
        output_gradient: &[f32],
        grad: &mut MlpGrad,
    ) -> &'a [f32] {
        assert_eq!(output_gradient.len(), self.output);
        assert_eq!(grad.data.len(), self.parameters.len());
        let h = self.hidden;
        let b1 = h * self.input;
        let w2 = b1 + h;
        let b2 = w2 + h * h;
        let w3 = b2 + h;
        let b3 = w3 + self.output * h;
        w.d2.fill(0.0);
        for (j, &d) in output_gradient.iter().enumerate() {
            grad.data[b3 + j] += d;
            for k in 0..h {
                grad.data[w3 + j * h + k] += d * w.h2[k];
                w.d2[k] += self.parameters[w3 + j * h + k] * d;
            }
        }
        for k in 0..h {
            w.d2[k] *= 1.0 - w.h2[k] * w.h2[k];
        }
        w.d1.fill(0.0);
        for j in 0..h {
            let d = w.d2[j];
            grad.data[b2 + j] += d;
            for k in 0..h {
                grad.data[w2 + j * h + k] += d * w.h1[k];
                w.d1[k] += self.parameters[w2 + j * h + k] * d;
            }
        }
        for k in 0..h {
            w.d1[k] *= 1.0 - w.h1[k] * w.h1[k];
        }
        w.input_gradient.fill(0.0);
        for j in 0..h {
            let d = w.d1[j];
            grad.data[b1 + j] += d;
            for k in 0..self.input {
                grad.data[j * self.input + k] += d * w.input[k];
                w.input_gradient[k] += self.parameters[j * self.input + k] * d;
            }
        }
        &w.input_gradient
    }
    pub fn soft_update(&mut self, source: &Mlp, tau: f32) {
        assert_eq!(
            (self.input, self.hidden, self.output),
            (source.input, source.hidden, source.output)
        );
        assert!((0.0..=1.0).contains(&tau));
        for (dst, src) in self.parameters.iter_mut().zip(&source.parameters) {
            *dst = (1.0 - tau) * *dst + tau * src;
        }
    }
    pub fn is_finite(&self) -> bool {
        self.parameters.iter().all(|x| x.is_finite())
    }
    pub fn validate(&self) -> bool {
        let Some(n) = self
            .hidden
            .checked_mul(self.input)
            .and_then(|n| n.checked_add(self.hidden))
            .and_then(|n| {
                self.hidden
                    .checked_mul(self.hidden)
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| n.checked_add(self.hidden))
            .and_then(|n| {
                self.output
                    .checked_mul(self.hidden)
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| n.checked_add(self.output))
        else {
            return false;
        };
        self.input > 0
            && self.hidden > 0
            && self.output > 0
            && n == self.parameters.len()
            && self.is_finite()
    }
}
#[inline]
fn dense(input: &[f32], weights: &[f32], bias: &[f32], out: &mut [f32], tanh: bool) {
    for ((row, &b), y) in weights.chunks_exact(input.len()).zip(bias).zip(out) {
        let mut sum = b;
        for (&x, &v) in input.iter().zip(row) {
            sum += x * v;
        }
        *y = if tanh { sum.tanh() } else { sum };
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adam {
    pub learning_rate: f32,
    step: u64,
    beta1_power: f32,
    beta2_power: f32,
    m: Vec<f32>,
    v: Vec<f32>,
}
impl Adam {
    pub fn new(model: &Mlp, learning_rate: f32) -> Self {
        assert!(learning_rate.is_finite() && learning_rate > 0.0);
        Self {
            learning_rate,
            step: 0,
            beta1_power: 1.0,
            beta2_power: 1.0,
            m: vec![0.0; model.parameters.len()],
            v: vec![0.0; model.parameters.len()],
        }
    }
    /// Applies Adam atomically; a nonfinite gradient/update leaves model and optimizer unchanged.
    pub fn step(&mut self, model: &mut Mlp, grad: &MlpGrad) -> bool {
        if self.m.len() != model.parameters.len()
            || self.v.len() != self.m.len()
            || grad.data.len() != self.m.len()
            || !grad.is_finite()
        {
            return false;
        }
        let p1 = self.beta1_power * 0.9;
        let p2 = self.beta2_power * 0.999;
        let update = |i: usize| {
            let g = grad.data[i];
            let m = 0.9 * self.m[i] + 0.1 * g;
            let v = 0.999 * self.v[i] + 0.001 * g * g;
            let next = model.parameters[i]
                - self.learning_rate * (m / (1.0 - p1)) / ((v / (1.0 - p2)).sqrt() + 1e-8);
            (m, v, next)
        };
        for i in 0..self.m.len() {
            let (m, v, p) = update(i);
            if !m.is_finite() || !v.is_finite() || !p.is_finite() {
                return false;
            }
        }
        for i in 0..self.m.len() {
            let g = grad.data[i];
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            model.parameters[i] -= self.learning_rate * (self.m[i] / (1.0 - p1))
                / ((self.v[i] / (1.0 - p2)).sqrt() + 1e-8);
        }
        self.step += 1;
        self.beta1_power = p1;
        self.beta2_power = p2;
        true
    }
    pub fn steps(&self) -> u64 {
        self.step
    }
    pub fn validate(&self, model: &Mlp) -> bool {
        self.m.len() == model.parameters.len()
            && self.v.len() == self.m.len()
            && self.m.iter().all(|x| x.is_finite())
            && self.v.iter().all(|x| x.is_finite() && *x >= 0.0)
            && self.learning_rate.is_finite()
            && self.learning_rate > 0.0
            && self.beta1_power.is_finite()
            && (0.0..=1.0).contains(&self.beta1_power)
            && self.beta2_power.is_finite()
            && (0.0..=1.0).contains(&self.beta2_power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backward_matches_finite_differences_for_parameters_and_inputs() {
        let mut m = Mlp::new(3, 4, 2, &mut Rng::new(7));
        let mut w = m.workspace();
        let input = [0.2, -0.4, 0.7];
        let dout = [0.7, -0.3];
        let mut grad = MlpGrad::new(&m);
        m.forward(&input, &mut w);
        let dx = m.backward(&mut w, &dout, &mut grad).to_vec();
        let eval = |m: &Mlp, input: &[f32], w: &mut Workspace| -> f32 {
            m.forward(input, w)
                .iter()
                .zip(dout)
                .map(|(a, b)| a * b)
                .sum()
        };
        let eps = 0.001;
        for i in 0..m.parameters.len() {
            let orig = m.parameters[i];
            m.parameters[i] = orig + eps;
            let hi = eval(&m, &input, &mut w);
            m.parameters[i] = orig - eps;
            let lo = eval(&m, &input, &mut w);
            m.parameters[i] = orig;
            assert!(
                ((hi - lo) / (2.0 * eps) - grad.data[i]).abs() < 0.0002,
                "parameter {i}"
            );
        }
        for i in 0..3 {
            let mut x = input;
            x[i] += eps;
            let hi = eval(&m, &x, &mut w);
            x[i] -= 2.0 * eps;
            let lo = eval(&m, &x, &mut w);
            assert!(((hi - lo) / (2.0 * eps) - dx[i]).abs() < 0.0002);
        }
    }
    #[test]
    fn adam_rejects_nonfinite_atomically_and_resumes_exactly() {
        let mut m = Mlp::new(2, 3, 1, &mut Rng::new(8));
        let mut a = Adam::new(&m, 0.001);
        let mut g = MlpGrad::new(&m);
        g.data.fill(0.2);
        assert!(a.step(&mut m, &g));
        let encoded = serde_json::to_string(&(m.clone(), a.clone())).unwrap();
        let (mut m2, mut a2): (Mlp, Adam) = serde_json::from_str(&encoded).unwrap();
        g.data[0] = f32::NAN;
        assert!(!a.step(&mut m, &g));
        assert_eq!(m.parameters, m2.parameters);
        assert_eq!(a.steps(), a2.steps());
        g.data[0] = 0.2;
        a.step(&mut m, &g);
        a2.step(&mut m2, &g);
        assert_eq!(m.parameters, m2.parameters);
    }
}
