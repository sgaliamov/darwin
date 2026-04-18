use crate::{GeneRanges, Genome};
use serde::Deserialize;

// tbd: [future, ga] type of ranges can be generic.
//      it will allow to define them in domain-specific way, like percentages, time intervals, etc.
/// Generic settings for any genetic algorithms.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Config {
    /// Defines amount of genes and their ranges.
    pub ranges: Vec<GeneRanges>,

    /// How many mutants will be generated from top individuals.
    pub mutation_ratio: f32,

    /// Fraction of fresh, completely random individuals injected per generation.
    /// Works like a mutation reset that re-introduces diversity when the search
    /// space is rugged.
    pub random_ratio: f32,

    /// How many of the fittest individuals are finally returned to the caller.
    pub bests: usize,

    /// Hard cap on generations. Bounds runtime even when the fitness plateaus.
    pub max_generation: usize,

    /// Lower bound for `mutation_sigma` reached linearly at `max_generations`. A
    /// small, non-zero sigma keeps the walk from freezing completely.
    pub min_mutation_sigma: f32,

    /// Initial standard deviation for Gaussian mutation noise applied to gene
    /// values.
    pub max_mutation_sigma: f32,

    /// Number of isolated sub-populations evolved in parallel. Migration
    /// is implemented explicitly via `immigration_ratio` and crossover between
    /// pools.
    pub pools: usize,

    /// Members per pool after truncation. Population size *per pool* – not total.
    pub population_size: usize,

    /// Evolution stops if the global best has not improved in this many
    /// generations. Prevents wasting compute after convergence.
    pub stagnation_count: usize,

    /// Number of contenders sampled during tournament selection.
    /// Larger value increases the probability,
    /// that the same mom and dad will be used for crossover.
    pub tournament_size: usize,

    /// Fraction of each pool replaced by round-robin crossover offspring.
    pub crossover_ratio: f32,

    /// Relative amount of mutation noise added *after* crossover.
    pub mutation_noise_factor: f32,

    /// Predefined seed.
    pub seed: Vec<Genome>,

    /// Defines how children are distributed between parent's pools.
    /// 0 - all goes to dad, 1 - all goes to mom.
    pub migration_factor: f64,
}

impl Config {
    /// Mutation sigma decreases linearly from `max_mutation_sigma` to
    /// `min_mutation_sigma` over the run. Simulates a simple annealing schedule.
    pub fn sigma(&self, generation: usize) -> f32 {
        let step =
            (self.max_mutation_sigma - self.min_mutation_sigma) / (self.max_generation - 1) as f32;

        let sigma = self.max_mutation_sigma - step * generation as f32;

        sigma.max(self.min_mutation_sigma)
    }

    pub fn mutant_count(&self) -> usize {
        (self.population_size as f32 * self.mutation_ratio)
            .ceil()
            .max(1.0) as usize
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ranges: Default::default(), // need to be defined explicitly
            mutation_ratio: 0.2,
            crossover_ratio: 0.2,
            random_ratio: 0.25,
            max_generation: 999,
            stagnation_count: 100,
            pools: 16,
            population_size: 128,
            max_mutation_sigma: 3.0,
            min_mutation_sigma: 1.0,
            mutation_noise_factor: 1.0,
            tournament_size: 4,
            bests: 5,
            seed: Default::default(),
            // tbd: [future, ga] find a better default, now it looks like, that 0 is the best.
            //      moving all children to one parent pool allows to use seed safely,
            //      as it won't affect other pools.
            //      need to collect all linage to see how migration happens.
            migration_factor: 0.000_000_1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use spectral::prelude::*;

    #[test]
    fn test_defaults_from_empty_json() {
        let config: Config =
            serde_json::from_value(json!({})).expect("failed to deserialize empty JSON");

        asserting("default values match")
            .that(&config)
            .is_equal_to(Config::default());
    }
}
