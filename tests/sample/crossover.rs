use super::noisy_mutant;
use darwin::{Context, Crossover, Gene, GeneRanges, Genome, Individual, RangeSet};
use rand::RngExt;
use std::iter;
use std::ops::Add;

/// Produces offspring by group-chunked crossover, with optional post-cross mutation.
pub struct DefaultCrossover<G> {
    flat_ranges: GeneRanges<G>,
    groups: Vec<usize>,
}

impl<G: Gene + Add<Output = G> + TryFrom<i64>> DefaultCrossover<G> {
    /// `range_set` — one `GeneRanges` per group; groups and flat ranges are derived from it.
    pub fn new(range_set: &RangeSet<G>) -> Self {
        assert!(!range_set.is_empty());
        Self {
            groups: range_set.iter().map(|r| r.len()).collect(),
            flat_ranges: range_set.iter().flatten().copied().collect(),
        }
    }
}

impl<G, GaState, IndState> Crossover<G, GaState, IndState> for DefaultCrossover<G>
where
    G: Gene + Add<Output = G> + TryFrom<i64>,
    GaState: Sync,
{
    /// For each group in `self.groups`, copy that contiguous chunk from one of the
    /// parents (50 / 50), preserving group boundaries.
    /// Returns `[maybe_mutant, pure_child]` — mutant first if produced.
    fn cross(
        &self,
        dad: &Individual<G, IndState>,
        mom: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> Vec<Genome<G>> {
        debug_assert_eq!(dad.genome.len(), mom.genome.len(), "parents must be same length");
        debug_assert_eq!(
            self.groups.iter().sum::<usize>(),
            dad.genome.len(),
            "group sizes must sum to genome length"
        );

        let mut rng = rand::rng();
        let mut child = Vec::with_capacity(dad.genome.len());

        self.groups
            .iter()
            .scan(0usize, |i, &g| {
                let start = *i;
                *i += g;
                Some((start, *i))
            })
            .for_each(|(start, end)| {
                let src = if rng.random_bool(0.5) {
                    &dad.genome[start..end]
                } else {
                    &mom.genome[start..end]
                };
                child.extend_from_slice(src);
            });

        noisy_mutant(&self.flat_ranges, &child, ctx, &mut rng)
            .into_iter()
            .chain(iter::once(child))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::DefaultGenerator;
    use darwin::{Config, Context, Generator, Individual, Pools};
    use std::marker::PhantomData;

    #[test]
    fn cross_keeps_group_chunks_from_either_parent() {
        let range_set: RangeSet<i64> = vec![vec![(0, 9), (10, 19)], vec![(20, 29)]];
        let ranges: Vec<(i64, i64)> = range_set.iter().flatten().copied().collect();
        let groups: Vec<usize> = range_set.iter().map(|r| r.len()).collect();
        let generator = DefaultGenerator::new(&ranges);
        let crossover = DefaultCrossover::new(&range_set);

        let ga_cfg = Config::default();
        let pools = Pools::from_vec(vec![]);
        let sigma = ga_cfg.sigma.get(0, ga_cfg.max_generation);
        let ctx = Context::<i64, (), ()> {
            generation: 0,
            diversity: 1.0,
            stagnation: 0.0,
            normal: rand_distr::Normal::new(0.0_f32, sigma).unwrap(),
            config: &ga_cfg,
            state: &None::<()>,
            pools: &pools,
            __: PhantomData,
        };
        let mom = Individual::firstborn(0, 0, generator.generate(&ctx));
        let dad = Individual::firstborn(0, 0, generator.generate(&ctx));

        let children = crossover.cross(
            &dad,
            &mom,
            // diversity=1.0, stagnation=0.0 → noise=0.0 → no mutation; pure child is last
            &ctx,
        );
        let child = &children[children.len() - 1];

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
            let a = &dad.genome[start..end];
            let m = &mom.genome[start..end];
            assert!(
                c == a || c == m,
                "segment [{start},{end}) didn't match either parent"
            );
        }
    }
}
