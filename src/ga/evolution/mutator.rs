use super::super::genome::{GeneRanges, GeneRangesRef, Genome, GenomeRef};
use super::config::DefaultEvolutionConfig;
use super::context::Context;
use super::Mutator;

/// Produces mutated genome copies via Gaussian noise.
pub struct DefaultMutator {
    ranges: GeneRanges,
    config: DefaultEvolutionConfig,
}

impl DefaultMutator {
    pub fn new(ranges: GeneRangesRef, config: DefaultEvolutionConfig) -> Self {
        Self { ranges: ranges.to_vec(), config }
    }
}

impl<GaState: Sync> Mutator<GaState> for DefaultMutator {
    /// Return a *mutated copy* of the given genome.
    /// Mutants that fall outside the allowed range are discarded (returns `None`).
    fn mutant(&self, genome: GenomeRef, ctx: &Context<'_, GaState>) -> Option<Genome> {
        super::mutant_with_noise(
            &self.ranges,
            &self.config,
            genome,
            ctx,
            ctx.noise_factor(),
            &mut rand::rng(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, DefaultEvolutionConfig};
    use spectral::prelude::*;

    /// `mutant` with low sigma and high diversity (noise≈0) almost always returns the same genome.
    #[test]
    fn mutant_with_zero_noise_returns_same_genome() {
        let mutator = DefaultMutator::new(&[(0, 1_000_000)], DefaultEvolutionConfig::default());
        let genome = vec![500_000i64];
        // diversity=1, stagnation=0 → noise_factor=0 → shift is ~0 almost always
        let ctx = Context { generation: 0, diversity: 1.0, stagnation: 0.0, state: &None::<()> };
        let mut same = 0usize;
        for _ in 0..100 {
            if let Some(m) = mutator.mutant(&genome, &ctx) {
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
        let mutator = DefaultMutator::new(
            &[(0, 10)],
            DefaultEvolutionConfig {
                max_mutation_sigma: 1000.0,
                min_mutation_sigma: 500.0,
                max_generation: 100,
            },
        );
        let genome = vec![5i64];
        let ctx = Context { generation: 0, diversity: 0.0, stagnation: 0.0, state: &None::<()> };
        // mutants that land outside the range return None — verify any Some is in range
        for _ in 0..200 {
            if let Some(m) = mutator.mutant(&genome, &ctx) {
                assert!(m[0] >= 0 && m[0] <= 10, "mutant {} out of range", m[0]);
            }
        }
    }
}
