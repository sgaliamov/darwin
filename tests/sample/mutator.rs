use super::noisy_mutant;
use darwin::{Context, Gene, GeneRanges, GeneRangesRef, Genome, Individual, Mutator};
use rand::distr::uniform::SampleUniform;
use std::ops::Add;

/// Produces mutated genome copies via Gaussian noise.
pub struct DefaultMutator<G> {
    ranges: GeneRanges<G>,
}

impl<G: Gene + SampleUniform + Add<Output = G> + TryFrom<i64>> DefaultMutator<G> {
    pub fn new(ranges: GeneRangesRef<G>) -> Self {
        Self {
            ranges: ranges.to_vec(),
        }
    }
}

impl<G, GaState, IndState> Mutator<G, GaState, IndState> for DefaultMutator<G>
where
    G: Gene + Add<Output = G> + TryFrom<i64>,
    GaState: Sync,
{
    /// Return a *mutated copy* of the given individual's genome.
    /// Mutants that fall outside the allowed range are discarded (returns `None`).
    fn mutant(
        &self,
        individual: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> Option<Genome<G>> {
        noisy_mutant(&self.ranges, individual, ctx, &mut rand::rng())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use darwin::{Context, Individual};
    use spectral::prelude::*;

    /// `mutant` with tiny sigma and zero stagnation almost always returns the same genome.
    #[test]
    fn mutant_with_zero_noise_returns_same_genome() {
        let mutator = DefaultMutator::new(&[(0i64, 1_000_000)]);
        let genome = vec![500_000i64];
        // sigma=0.01 → shift rounds to 0 for integers almost always
        let pools = darwin::Pools::<i64, ()>::from_vec(vec![]);
        let ctx = Context {
            epoch: darwin::Epoch { generation: 0, stagnation: 0.0, normal: rand_distr::Normal::new(0.0_f32, 0.01).unwrap() },
            state: &None::<()>,
            pools: &pools,
            __: std::marker::PhantomData,
        };
        let ind = Individual::firstborn(0, 0, genome.clone());
        let mut same = 0usize;
        for _ in 0..100 {
            if let Some(m) = mutator.mutant(&ind, &ctx)
                && m == genome
            {
                same += 1;
            }
        }
        assert_that!(same).is_greater_than(80);
    }

    /// `mutant` result always stays within range even with huge sigma.
    #[test]
    fn mutant_never_exceeds_range() {
        let mutator = DefaultMutator::new(
            &[(0i64, 10)],
        );
        let genome = vec![5i64];
        let pools = darwin::Pools::<i64, ()>::from_vec(vec![]);
        let ctx = Context {
            epoch: darwin::Epoch { generation: 0, stagnation: 0.0, normal: rand_distr::Normal::new(0.0_f32, 1000.0_f32).unwrap() },
            state: &None::<()>,
            pools: &pools,
            __: std::marker::PhantomData,
        };
        let ind = Individual::firstborn(0, 0, genome.clone());
        // mutants that land outside the range return None — verify any Some is in range
        for _ in 0..200 {
            if let Some(m) = mutator.mutant(&ind, &ctx) {
                assert!(m[0] >= 0 && m[0] <= 10, "mutant {} out of range", m[0]);
            }
        }
    }
}
