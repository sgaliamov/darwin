use super::super::genome::{Gene, GeneRanges, GeneRangesRef, Genome, GenomeRef};
use super::config::DefaultEvolutionConfig;
use super::context::Context;
use super::evolver::Evolver;
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use std::iter;

/// Built-in evolution engine.
///
/// Stateless — all randomness is drawn from `rand::rng()` (the thread-local RNG),
/// so a single `DefaultEvolution` instance is `Sync` and can be shared freely across
/// Rayon threads without cloning or per-thread initialization short flight.  I need a short flight. .
pub struct DefaultEvolution {
    ranges: GeneRanges,
    groups: Vec<usize>,
    config: DefaultEvolutionConfig,
}

impl DefaultEvolution {
    pub fn new(ranges: GeneRangesRef, groups: &[usize], config: DefaultEvolutionConfig) -> Self {
        assert!(!groups.is_empty());
        Self {
            ranges: ranges.to_vec(),
            groups: groups.to_vec(),
            config,
        }
    }

    /// Core mutation logic — applies Gaussian noise derived from `ctx`.
    fn mutant_with_noise<GaState>(
        &self,
        genome: GenomeRef,
        ctx: &Context<'_, GaState>,
        noise_factor: f32,
        rng: &mut impl rand::Rng,
    ) -> Option<Genome> {
        // μ = 0 so shifts are symmetric around the original value.
        let normal = Normal::new(0.0_f32, self.config.sigma(ctx.generation))
            .expect("`sigma` should be valid.");
        genome
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let range = self.ranges.get(i)?;
                if range.0 == range.1 {
                    return Some(*g);
                }
                let shift = (normal.sample(rng) * noise_factor).round() as Gene;
                let new = g + shift;
                if new < range.0 || new > range.1 {
                    None
                } else {
                    Some(new)
                }
            })
            .collect()
    }
}

impl<GaState: Sync> Evolver<GaState> for DefaultEvolution {
    /// Create a random genome.
    fn generate(&self) -> Genome {
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
    fn mutant(&self, genome: GenomeRef, ctx: &Context<'_, GaState>) -> Option<Genome> {
        self.mutant_with_noise(genome, ctx, ctx.noise_factor(), &mut rand::rng())
    }

    /// For each group in `self.groups`, copy that contiguous chunk from one of the
    /// parents (50 / 50), preserving group boundaries.
    /// Returns `[maybe_mutant, pure_child]` — mutant first if produced.
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, ctx: &Context<'_, GaState>) -> Vec<Genome> {
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

        self.mutant_with_noise(&child, ctx, ctx.noise_factor(), &mut rng)
            .into_iter()
            .chain(iter::once(child))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral::prelude::*;

    fn make_evo(ranges: &[(i64, i64)]) -> DefaultEvolution {
        let groups = vec![ranges.len()];
        DefaultEvolution::new(ranges, &groups, DefaultEvolutionConfig::default())
    }

    /// Every gene of `random()` must stay within its declared range.
    #[test]
    fn random_genome_stays_in_range() {
        let ranges: Vec<_> = vec![(0, 9), (10, 19), (100, 200)];
        let evo = make_evo(&ranges);
        for _ in 0..100 {
            let g = <DefaultEvolution as Evolver<()>>::generate(&evo);
            for (gene, &(lo, hi)) in g.iter().zip(ranges.iter()) {
                assert!(*gene >= lo && *gene <= hi, "gene {gene} out of [{lo},{hi}]");
            }
        }
    }

    /// `mutant` with low sigma and high diversity (noise≈0) almost always returns the same genome.
    #[test]
    fn mutant_with_zero_noise_returns_same_genome() {
        let evo = make_evo(&[(0, 1_000_000)]);
        let genome = vec![500_000i64];
        // diversity=1 stagnation=0 → noise_factor=0 → shift is ~0 almost always
        let ctx = Context {
            generation: 0,
            diversity: 1.0,
            stagnation: 0.0,
            state: &None::<()>,
        };
        let mut same = 0usize;
        for _ in 0..100 {
            if let Some(m) = evo.mutant(&genome, &ctx) {
                if m == genome {
                    same += 1;
                }
            }
        }
        assert_that!(same).is_greater_than(80);
    }

    /// `mutant` result always stays within range even with huge sigma.
    #[test]
    fn mutant_never_exceeds_range() {
        let ranges: Vec<_> = vec![(0, 10)];
        let evo = DefaultEvolution::new(
            &ranges,
            &[1],
            DefaultEvolutionConfig {
                max_mutation_sigma: 1000.0,
                min_mutation_sigma: 500.0,
                max_generation: 100,
            },
        );
        let genome = vec![5i64];
        let ctx = Context {
            generation: 0,
            diversity: 0.0,
            stagnation: 0.0,
            state: &None::<()>,
        };
        // mutants that land outside the range return None — verify any Some is in range
        for _ in 0..200 {
            if let Some(m) = evo.mutant(&genome, &ctx) {
                assert!(m[0] >= 0 && m[0] <= 10, "mutant {} out of range", m[0]);
            }
        }
    }

    #[test]
    fn cross_keeps_group_chunks_from_either_parent() {
        let groups = vec![2, 1];
        let evolver = DefaultEvolution::new(
            &[(0, 9), (10, 19), (20, 29)],
            &groups,
            DefaultEvolutionConfig::default(),
        );
        let mom = <DefaultEvolution as Evolver<()>>::generate(&evolver);
        let dad = <DefaultEvolution as Evolver<()>>::generate(&evolver);

        let children = evolver.cross(
            &dad,
            &mom,
            // diversity=1.0, stagnation=0.0 → noise_factor=0.0 → no mutation; pure child is unmodified
            &Context {
                generation: 0,
                diversity: 1.0,
                stagnation: 0.0,
                state: &None::<()>,
            },
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
