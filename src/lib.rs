//! Domain agnostic genetic algorithm implementation.

mod ga;
mod individual;

pub use ga::*;
pub use individual::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};

// Uses Gaussian distribution to generate deviations from the current values on mutations.
pub struct Random {
    normal: Normal<f32>,
    rng: SmallRng,
}

impl Random {
    pub fn new(mean: f32, std_dev: f32) -> Self {
        Self {
            normal: Normal::new(mean, std_dev).unwrap(),
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }

    /// Set new sigma.
    pub fn std_dev(&mut self, val: f32) {
        self.normal = Normal::new(self.normal.mean(), val).unwrap();
    }

    pub fn random_range(&mut self, range: &GeneRange) -> Gene {
        self.rng.random_range(range.0..=range.1)
    }

    pub fn next(&mut self, noise: f32) -> Gene {
        (self.normal.sample(&mut self.rng) * noise).round() as Gene
    }
}
