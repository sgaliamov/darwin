use crate::{Context, Crossover, Gene, GeneRanges, GeneRangesRef, Genome, GenomeRef};
use rand::RngExt;
use std::iter;
use std::ops::Add;
use super::{Sigma, noise_factor, mutant_with_noise};

/// Produces offspring by group-chunked crossover, with optional post-cross mutation.
pub struct DefaultCrossover<G> {
    ranges: GeneRanges<G>,
    groups: Vec<usize>,
    config: Sigma,
}

impl<G: Gene + Add<Output = G> + TryFrom<i64>> DefaultCrossover<G> {
    // todo: use vector of ranges and calculate groups from it
    pub fn new(ranges: GeneRangesRef<G>, groups: &[usize], config: Sigma) -> Self {
        assert!(!groups.is_empty());
        Self {
            ranges: ranges.to_vec(),
            groups: groups.to_vec(),
            config,
        }
    }
}

impl<G, GaState> Crossover<G, GaState> for DefaultCrossover<G>
where
    G: Gene + Add<Output = G> + TryFrom<i64>,
    GaState: Sync,
{
    /// For each group in `self.groups`, copy that contiguous chunk from one of the
    /// parents (50 / 50), preserving group boundaries.
    /// Returns `[maybe_mutant, pure_child]` — mutant first if produced.
    fn cross(
        &self,
        dad: GenomeRef<G>,
        mom: GenomeRef<G>,
        ctx: &Context<'_, G, GaState>,
    ) -> Vec<Genome<G>> {
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
                Some((start, *i))
            })
            .for_each(|(start, end)| {
                let src = if rng.random_bool(0.5) {
                    &dad[start..end]
                } else {
                    &mom[start..end]
                };
                child.extend_from_slice(src);
            });

        let sigma = self.config.get(ctx.generation, ctx.config.max_generation);
        let noise = noise_factor(ctx.diversity, ctx.stagnation);
        mutant_with_noise(&self.ranges, sigma, &child, noise, &mut rng)
            .into_iter()
            .chain(iter::once(child))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultGenerator, Generator};

    #[test]
    fn cross_keeps_group_chunks_from_either_parent() {
        let groups = vec![2, 1];
        let ranges: Vec<(i64, i64)> = vec![(0, 9), (10, 19), (20, 29)];
        let config = Sigma::default();
        let generator = DefaultGenerator::new(&ranges);
        let crossover = DefaultCrossover::new(&ranges, &groups, config);

        let mom = generator.generate();
        let dad = generator.generate();

        let ga_cfg = crate::Config::default();
        let children = crossover.cross(
            &dad,
            &mom,
            // diversity=1.0, stagnation=0.0 → noise_factor=0.0 → no mutation; pure child is last
            &Context {
                generation: 0,
                diversity: 1.0,
                stagnation: 0.0,
                config: &ga_cfg,
                state: &None::<()>,
            },
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
            let a = &dad[start..end];
            let m = &mom[start..end];
            assert!(
                c == a || c == m,
                "segment [{start},{end}) didn't match either parent"
            );
        }
    }
}
