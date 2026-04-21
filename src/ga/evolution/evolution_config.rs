/// Configuration for the built-in [`super::default_evolution::DefaultEvolution`] evolver.
/// Kept separate from [`crate::Config`] so GA config stays evolver-agnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionConfig {
    /// Initial standard deviation for Gaussian mutation noise.
    pub max_mutation_sigma: f32,

    /// Floor for annealed sigma — prevents mutation from freezing completely.
    pub min_mutation_sigma: f32,

    /// Relative noise factor applied to offspring after crossover.
    pub cross_noise_factor: f32,

    /// Total number of generations; used to compute the annealing step.
    pub max_generation: usize,
}

impl EvolutionConfig {
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

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            max_mutation_sigma: 3.0,
            min_mutation_sigma: 1.0,
            cross_noise_factor: 1.0,
            max_generation: 999,
        }
    }
}
