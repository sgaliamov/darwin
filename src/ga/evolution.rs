use crate::{Gene, GeneRanges, GeneRangesRef, Genome, GenomeRef};
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use std::iter;

/// Trait for pluggable genome operation strategies.
///
/// Implementations must be `Send + Sync` so a single instance can be shared
/// across Rayon threads without cloning or locking.
pub trait Evolver: Send + Sync {
    /// Generate a fully random genome within the configured gene ranges.
    fn random(&self) -> Genome;

    /// Return a mutated copy of `genome`, or `None` if the mutant falls outside range.
    ///
    /// - `generation`   — current generation number; used to derive mutation magnitude.
    /// - `noise_factor` — per-pool scaling driven by diversity / stagnation; pass `1.0` to ignore.
    fn mutant(&self, genome: GenomeRef, generation: usize, noise_factor: f32) -> Option<Genome>;

    /// Produce offspring by crossing two parent genomes.
    ///
    /// `generation` is the current generation number; implementations may use it
    /// to scale mutation applied to the child. Returns one or more child genomes.
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, generation: usize) -> Vec<Genome>;
}

/// Built-in evolution engine.
///
/// Stateless — all randomness is drawn from `rand::rng()` (the thread-local RNG),
/// so a single `Evolution` instance is `Sync` and can be shared freely across
/// Rayon threads without cloning or per-thread initialisation.
pub struct Evolution {
    ranges: GeneRanges,
    cross_noise_factor: f32,
    groups: Vec<usize>,
    max_mutation_sigma: f32,
    min_mutation_sigma: f32,
    max_generation: usize,
}

impl Evolution {
    pub fn new(
        ranges: GeneRangesRef,
        cross_noise_factor: f32,
        groups: &[usize],
        max_mutation_sigma: f32,
        min_mutation_sigma: f32,
        max_generation: usize,
    ) -> Self {
        assert!(!groups.is_empty());
        Self {
            ranges: ranges.to_vec(),
            cross_noise_factor,
            groups: groups.to_vec(),
            max_mutation_sigma,
            min_mutation_sigma,
            max_generation,
        }
    }

    /// Linear annealing: sigma decreases from `max_mutation_sigma` to
    /// `min_mutation_sigma` over `max_generation` generations.
    fn sigma(&self, generation: usize) -> f32 {
        if self.max_generation <= 1 {
            return self.min_mutation_sigma;
        }
        let step = (self.max_mutation_sigma - self.min_mutation_sigma)
            / (self.max_generation - 1) as f32;
        (self.max_mutation_sigma - step * generation as f32).max(self.min_mutation_sigma)
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
    fn mutant(&self, genome: GenomeRef, generation: usize, noise_factor: f32) -> Option<Genome> {
        let mut rng = rand::rng();
        let sigma = self.sigma(generation);
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
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, generation: usize) -> Vec<Genome> {
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

        self.mutant(&child, generation, self.cross_noise_factor)
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
        let evolver = Evolution::new(&[(0, 9), (10, 19), (20, 29)], 0.0, &groups, 1.0, 1.0, 100);
        let mom = evolver.random();
        let dad = evolver.random();

        let children = evolver.cross(&dad, &mom, 0);
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
