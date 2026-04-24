/// Configuration for the built-in default evolver.
/// Kept separate from [`crate::Config`] so GA config stays evolver-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct Sigma {
    /// Initial standard deviation for Gaussian mutation noise.
    pub max: f32,

    /// Floor for annealed sigma — prevents mutation from freezing completely.
    pub min: f32,
}

impl Sigma {
    /// Linear annealing: sigma decreases from `max` to `min` over `max_generation` generations.
    pub fn get(&self, generation: usize, max_generation: usize) -> f32 {
        if max_generation <= 1 {
            return self.min;
        }
        let step = (self.max - self.min) / (max_generation - 1) as f32;
        (self.max - step * generation as f32).max(self.min)
    }
}

impl Default for Sigma {
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
    use spectral::prelude::*;

    /// Sigma at gen 0 = max; at last gen ≈ min.
    #[test]
    fn sigma_anneals_from_max_to_min() {
        let cfg = Sigma { max: 10.0, min: 1.0 };
        assert_that!(cfg.get(0, 10)).is_close_to(10.0, 1e-5);
        assert_that!(cfg.get(9, 10)).is_close_to(1.0, 1e-5);
    }

    /// Sigma never drops below min even past max_generation.
    #[test]
    fn sigma_floors_at_min() {
        let cfg = Sigma { max: 5.0, min: 2.0 };
        assert_that!(cfg.get(9999, 5)).is_equal_to(2.0);
    }

    /// Single-generation config always returns min.
    #[test]
    fn sigma_single_generation_returns_min() {
        let cfg = Sigma { max: 8.0, min: 0.5 };
        assert_that!(cfg.get(0, 1)).is_equal_to(0.5);
    }
}
