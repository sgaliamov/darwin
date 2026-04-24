use super::super::genome::{GeneRanges, GeneRangesRef, Genome, GenomeRef};
use super::config::DefaultEvolutionConfig;
use super::context::Context;
use super::Crossover;
use rand::RngExt;
use std::iter;

/// Produces offspring by group-chunked crossover, with optional post-cross mutation.
pub struct DefaultCrossover {
    ranges: GeneRanges,
    groups: Vec<usize>,
    config: DefaultEvolutionConfig,
}

impl DefaultCrossover {
    pub fn new(ranges: GeneRangesRef, groups: &[usize], config: DefaultEvolutionConfig) -> Self {
        assert!(!groups.is_empty());
        Self { ranges: ranges.to_vec(), groups: groups.to_vec(), config }
    }
}

impl<GaState: Sync> Crossover<GaState> for DefaultCrossover {
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
                Some((start, *i))
            })
            .for_each(|(start, end)| {
                let src = if rng.random_bool(0.5) { &dad[start..end] } else { &mom[start..end] };
                child.extend_from_slice(src);
            });

        super::mutant_with_noise(&self.ranges, &self.config, &child, ctx, ctx.noise_factor(), &mut rng)
            .into_iter()
            .chain(iter::once(child))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::Generator;
    use super::*;
    use super::super::generator::DefaultGenerator;

    #[test]
    fn cross_keeps_group_chunks_from_either_parent() {
        let groups = vec![2, 1];
        let ranges = vec![(0, 9), (10, 19), (20, 29)];
        let config = DefaultEvolutionConfig::default();
        let generator = DefaultGenerator::new(&ranges);
        let crossover = DefaultCrossover::new(&ranges, &groups, config);

        let mom = generator.generate();
        let dad = generator.generate();

        let children = crossover.cross(
            &dad,
            &mom,
            // diversity=1.0, stagnation=0.0 → noise_factor=0.0 → no mutation; pure child is last
            &Context { generation: 0, diversity: 1.0, stagnation: 0.0, state: &None::<()> },
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
