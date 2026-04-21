use super::super::genome::{Gene, GeneRanges, GeneRangesRef, Genome, GenomeRef};
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use std::iter;

use super::context::Context;
use super::evolver::Evolver;
use super::evolution_config::EvolutionConfig;

/// Built-in evolution engine.
///
/// Stateless — all randomness is drawn from `rand::rng()` (the thread-local RNG),
/// so a single `DefaultEvolution` instance is `Sync` and can be shared freely across
/// Rayon threads without cloning or per-thread initialisation.
pub struct DefaultEvolution {
    ranges: GeneRanges,
    groups: Vec<usize>,
    config: EvolutionConfig,
}

impl DefaultEvolution {
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

impl Evolver for DefaultEvolution {
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
        let evolver = DefaultEvolution::new(
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
