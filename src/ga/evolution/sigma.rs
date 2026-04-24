/// Configuration for the built-in [`super::default_evolution::DefaultEvolution`] evolver.
/// Kept separate from [`crate::Config`] so GA config stays evolver-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct SigmaConfig {
    /// Initial standard deviation for Gaussian mutation noise.
    pub max: f32,

    /// Floor for annealed sigma — prevents mutation from freezing completely.
    pub min: f32,
}

impl SigmaConfig {
    /// Linear annealing: sigma decreases from `max` to
    /// `min` over `ctx.config.max_generation` generations.
    pub fn sigma<GaState>(&self, ctx: &super::Context<'_, GaState>) -> f32 {
        let max_gen = ctx.config.max_generation;
        if max_gen <= 1 {
            return self.min;
        }
        let step = (self.max - self.min) / (max_gen - 1) as f32;
        (self.max - step * ctx.generation as f32).max(self.min)
    }
}

impl Default for SigmaConfig {
    fn default() -> Self {
        Self {
            max: 3.0,
            min: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Context};
    use spectral::prelude::*;

    /// Sigma at gen 0 = max; at last gen ≈ min.
    #[test]
    fn sigma_anneals_from_max_to_min() {
        let cfg = SigmaConfig { max: 10.0, min: 1.0 };
        let ga_cfg = Config { max_generation: 10, ..Config::default() };
        let ctx0 = Context { generation: 0, diversity: 0.0, stagnation: 0.0, config: &ga_cfg, state: &None::<()> };
        let ctx9 = Context { generation: 9, diversity: 0.0, stagnation: 0.0, config: &ga_cfg, state: &None::<()> };
        assert_that!(cfg.sigma(&ctx0)).is_close_to(10.0, 1e-5);
        assert_that!(cfg.sigma(&ctx9)).is_close_to(1.0, 1e-5);
    }

    /// Sigma never drops below min even past max_generation.
    #[test]
    fn sigma_floors_at_min() {
        let cfg = SigmaConfig { max: 5.0, min: 2.0 };
        let ga_cfg = Config { max_generation: 5, ..Config::default() };
        let ctx = Context { generation: 9999, diversity: 0.0, stagnation: 0.0, config: &ga_cfg, state: &None::<()> };
        assert_that!(cfg.sigma(&ctx)).is_equal_to(2.0);
    }

    /// Single-generation config always returns min.
    #[test]
    fn sigma_single_generation_returns_min() {
        let cfg = SigmaConfig { max: 8.0, min: 0.5 };
        let ga_cfg = Config { max_generation: 1, ..Config::default() };
        let ctx = Context { generation: 0, diversity: 0.0, stagnation: 0.0, config: &ga_cfg, state: &None::<()> };
        assert_that!(cfg.sigma(&ctx)).is_equal_to(0.5);
    }
}
