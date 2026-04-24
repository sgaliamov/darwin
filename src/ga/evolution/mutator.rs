use crate::{Context, Gene, GeneRanges, GeneRangesRef, Genome, GenomeRef, Mutator};
use rand::distr::uniform::SampleUniform;
use std::ops::Add;
use super::{Sigma, noise_factor, mutant_with_noise};

/// Produces mutated genome copies via Gaussian noise.
pub struct DefaultMutator<G> {
    ranges: GeneRanges<G>,
    config: Sigma,
}

impl<G: Gene + SampleUniform + Add<Output = G> + TryFrom<i64>> DefaultMutator<G> {
    pub fn new(ranges: GeneRangesRef<G>, config: Sigma) -> Self {
        Self {
            ranges: ranges.to_vec(),
            config,
        }
    }
}

impl<G, GaState> Mutator<G, GaState> for DefaultMutator<G>
where
    G: Gene + Add<Output = G> + TryFrom<i64>,
    GaState: Sync,
{
    /// Return a *mutated copy* of the given genome.
    /// Mutants that fall outside the allowed range are discarded (returns `None`).
    fn mutant(&self, genome: GenomeRef<G>, ctx: &Context<'_, G, GaState>) -> Option<Genome<G>> {
        let sigma = self.config.get(ctx.generation, ctx.config.max_generation);
        let noise = noise_factor(ctx.diversity, ctx.stagnation);
        mutant_with_noise(&self.ranges, sigma, genome, noise, &mut rand::rng())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Context, Sigma};
    use spectral::prelude::*;

    /// `mutant` with low sigma and high diversity (noise≈0) almost always returns the same genome.
    #[test]
    fn mutant_with_zero_noise_returns_same_genome() {
        let mutator = DefaultMutator::new(&[(0i64, 1_000_000)], Sigma::default());
        let genome = vec![500_000i64];
        // diversity=1, stagnation=0 → noise_factor=0 → shift is ~0 almost always
        let ga_cfg = Config::default();
        let ctx = Context {
            generation: 0,
            diversity: 1.0,
            stagnation: 0.0,
            config: &ga_cfg,
            state: &None::<()>,
        };
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
            &[(0i64, 10)],
            Sigma {
                max: 1000.0,
                min: 500.0,
            },
        );
        let genome = vec![5i64];
        let ga_cfg = Config::<i64> {
            max_generation: 100,
            ..Config::default()
        };
        let ctx = Context {
            generation: 0,
            diversity: 0.0,
            stagnation: 0.0,
            config: &ga_cfg,
            state: &None::<()>,
        };
        // mutants that land outside the range return None — verify any Some is in range
        for _ in 0..200 {
            if let Some(m) = mutator.mutant(&genome, &ctx) {
                assert!(m[0] >= 0 && m[0] <= 10, "mutant {} out of range", m[0]);
            }
        }
    }
}
