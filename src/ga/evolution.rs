use crate::{Gene, GeneRanges, GeneRangesRef, Genome, GenomeRef};
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use std::iter;

/// Context passed to [`Evolver`] methods on each mutation / crossover call.
/// Carries GA-level signals an evolver may use to tune its behaviour.
/// All pressure values are normalised to `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy)]
pub struct Context {
    /// Current generation number.
    pub generation: usize,

    /// Pool gene diversity: `0.0` = fully converged, `1.0` = maximally diverse.
    pub diversity: f32,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,
}

/// Trait for pluggable genome operation strategies.
///
/// Implementations must be `Send + Sync` so a single instance can be shared
/// across Rayon threads without cloning or locking.
pub trait Evolver: Send + Sync {
    /// Generate a fully random genome.
    fn random(&self) -> Genome;

    /// Return a mutated copy of `genome`, or `None` if the mutant falls outside range.
    fn mutant(&self, genome: GenomeRef, ctx: &Context) -> Option<Genome>;

    /// Produce offspring by crossing two parent genomes.
    /// Returns one or more child genomes.
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, ctx: &Context) -> Vec<Genome>;
}

/// Configuration for the built-in [`Evolution`] evolver.
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

/// Built-in evolution engine.
///
/// Stateless — all randomness is drawn from `rand::rng()` (the thread-local RNG),
/// so a single `Evolution` instance is `Sync` and can be shared freely across
/// Rayon threads without cloning or per-thread initialisation.
pub struct Evolution {
    ranges: GeneRanges,
    groups: Vec<usize>,
    config: EvolutionConfig,
}

impl Evolution {
    pub fn new(ranges: GeneRangesRef, groups: &[usize], config: EvolutionConfig) -> Self {
        assert!(!groups.is_empty());
        Self {
            ranges: ranges.to_vec(),
            groups: groups.to_vec(),
            config,
        }
    }

    fn sigma(&self, generation: usize) -> f32 {
        self.config.sigma(generation)
    }
}

impl Evolver for Evolution {
    /// Create a random genome.
    fn random(&self) -> Genome {
        let mut rng = rand::rng();
        self.ranges
            .iter()
            .map(|range| {
                if range.0 == range.1 {
                    range.0
                } else {
                    rng.random_range(range.0..=range.1)
                }
            })
            .collect()
    }

    /// Return a *mutated copy* of the given genome.
    /// Mutants that fall outside the allowed range are discarded (returns `None`).
    fn mutant(&self, genome: GenomeRef, ctx: &Context) -> Option<Genome> {
        let mut rng = rand::rng();
        let sigma = self.sigma(ctx.generation);
        // High diversity -> lower noise (exploit). High stagnation -> higher noise (explore).
        let noise_factor = (1.0 - ctx.diversity) + ctx.stagnation * ctx.diversity;
        // μ (mean) is 0 so shifts can go left or right.
        let normal = Normal::new(0.0_f32, sigma).expect("`sigma` should be valid.");
        genome
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let range = self.ranges.get(i)?;

                if range.0 == range.1 {
                    return Some(*g);
                }

                let sample = normal.sample(&mut rng);
                let shift = (sample * noise_factor).round() as Gene;
                let new = g + shift;

                if new < range.0 || new > range.1 {
                    None
                } else {
                    Some(new)
                }
            })
            .collect()
    }

    /// For each group in `self.groups`, copy that contiguous chunk from one of the
    /// parents (50 / 50), preserving group boundaries.
    /// Returns `[maybe_mutant, pure_child]` — mutant first if produced.
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, ctx: &Context) -> Vec<Genome> {
        debug_assert_eq!(dad.len(), mom.len(), "parents must be same length");
        debug_assert_eq!(
            self.groups.iter().sum::<usize>(),
            dad.len(),
            "group sizes must sum to genome length"
        );

        let mut rng = rand::rng();
        let mut child = Vec::with_capacity(dad.len());

        self.groups
            .iter()
            .scan(0usize, |i, &g| {
                let start = *i;
                *i += g;
                Some((start, *i)) // tbd: [ga] no need to swap groups with genes with static ranges
            })
            .for_each(|(start, end)| {
                let src = if rng.random_bool(0.5) {
                    &dad[start..end]
                } else {
                    &mom[start..end]
                };
                child.extend_from_slice(src);
            });

        // Cross-noise: use a reduced stagnation so crossover children mutate gently.
        let cross_ctx = Context {
            stagnation: self.config.cross_noise_factor,
            ..*ctx
        };
        self.mutant(&child, &cross_ctx)
            .into_iter()
            .chain(iter::once(child))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_keeps_group_chunks_from_either_parent() {
        let groups = vec![2, 1];
        let evolver = Evolution::new(
            &[(0, 9), (10, 19), (20, 29)],
            &groups,
            EvolutionConfig { cross_noise_factor: 0.0, ..Default::default() },
        );
        let mom = evolver.random();
        let dad = evolver.random();

        let children = evolver.cross(
            &dad,
            &mom,
            &Context { generation: 0, diversity: 0.5, stagnation: 0.0 },
        );
        let child = &children[1];

        let bounds: Vec<(usize, usize)> = groups
            .iter()
            .scan(0usize, |i, &g| {
                let s = *i;
                *i += g;
                Some((s, *i))
            })
            .collect();

        for (start, end) in bounds {
            let c = &child[start..end];
            let a = &dad[start..end];
            let m = &mom[start..end];
            assert!(
                c == a || c == m,
                "segment [{start},{end}) didn't match either parent"
            );
        }
    }
}
