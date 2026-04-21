/// Configuration for the built-in [`super::default_evolution::DefaultEvolution`] evolver.
/// Kept separate from [`crate::Config`] so GA config stays evolver-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct DefaultEvolutionConfig {
    /// Initial standard deviation for Gaussian mutation noise.
    pub max_mutation_sigma: f32,

    /// Floor for annealed sigma — prevents mutation from freezing completely.
    pub min_mutation_sigma: f32,

    /// Total number of generations; used to compute the annealing step.
    pub max_generation: usize,
}

impl DefaultEvolutionConfig {
    /// Linear annealing: sigma decreases from `max_mutation_sigma` to
    /// `min_mutation_sigma` over `max_generation` generations.
    pub fn sigma(&self, generation: usize) -> f32 {
        if self.max_generation <= 1 {
            return self.min_mutation_sigma;
        }
        let step = (self.max_mutation_sigma - self.min_mutation_sigma)
            / (self.max_generation - 1) as f32;
        (self.max_mutation_sigma - step * generation as f32).max(self.min_mutation_sigma)
    }
}

impl Default for DefaultEvolutionConfig {
    fn default() -> Self {
        Self {
            max_mutation_sigma: 3.0,
            min_mutation_sigma: 1.0,
            max_generation: 999, // starting with 0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral::prelude::*;

    /// Sigma at gen 0 = max; at last gen ≈ min.
    #[test]
    fn sigma_anneals_from_max_to_min() {
        let cfg = DefaultEvolutionConfig {
            max_mutation_sigma: 10.0,
            min_mutation_sigma: 1.0,
            max_generation: 10,
        };
        assert_that!(cfg.sigma(0)).is_close_to(10.0, 1e-5);
        assert_that!(cfg.sigma(9)).is_close_to(1.0, 1e-5);
    }

    /// Sigma never drops below min even past max_generation.
    #[test]
    fn sigma_floors_at_min() {
        let cfg = DefaultEvolutionConfig {
            max_mutation_sigma: 5.0,
            min_mutation_sigma: 2.0,
            max_generation: 5,
        };
        assert_that!(cfg.sigma(9999)).is_equal_to(2.0);
    }

    /// Single-generation config always returns min.
    #[test]
    fn sigma_single_generation_returns_min() {
        let cfg = DefaultEvolutionConfig {
            max_mutation_sigma: 8.0,
            min_mutation_sigma: 0.5,
            max_generation: 1,
        };
        assert_that!(cfg.sigma(0)).is_equal_to(0.5);
    }
}
