use crate::{Gene, GeneRangesRef, Genome, GenomeRef};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use std::iter;

/// Evolution engine.
/// Works with pure genomes.
/// Need to be created per thread.
pub struct Evolution<'a> {
    rng: SmallRng,
    ranges: GeneRangesRef<'a>,
    normal: Normal<f32>,
    cross_noise_factor: f32,
    groups: &'a [usize],
}

impl<'a> Evolution<'a> {
    pub fn new(
        ranges: GeneRangesRef<'a>,
        sigma: f32,
        cross_noise_factor: f32,
        groups: &'a [usize],
    ) -> Self {
        assert!(!groups.is_empty());

        Self {
            rng: SmallRng::from_rng(&mut rand::rng()),
            ranges,
            // μ (mean) is 0 to be able to shift left and right.
            normal: Normal::new(0.0, sigma).expect("`sigma` should be valid."),
            cross_noise_factor,
            groups,
        }
    }

    /// Create random genome.
    pub fn random(&mut self) -> Genome {
        self.ranges
            .iter()
            .map(|range| {
                if range.0 == range.1 {
                    range.0
                } else {
                    self.rng.random_range(range.0..=range.1)
                }
            })
            .collect()
    }

    /// Return a *mutated copy* of the given DNA. `noise_factor` scales the sigma
    /// passed in from the GA (allows weaker noise for crossover offspring).
    /// Mutants who goes too far from the range will be discarded.
    /// `noise_factor` is a parameter as it depends on the pool diversity or cross mutation factor.
    pub fn mutant(&mut self, genome: GenomeRef, noise_factor: f32) -> Option<Genome> {
        genome
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let range = self.ranges.get(i)?;

                if range.0 == range.1 {
                    return Some(*g);
                }

                let sample = self.normal.sample(&mut self.rng);
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

    /// For each group size in `self.groups`, copy that contiguous chunk
    /// from one of the parents (50/50), preserving group boundaries.
    /// Returns `[maybe_mutant, pure_child]` — mutant first if produced.
    pub fn cross(&mut self, dad: GenomeRef, mom: GenomeRef) -> Vec<Genome> {
        debug_assert_eq!(dad.len(), mom.len(), "parents must be same length");
        debug_assert_eq!(
            self.groups.iter().sum::<usize>(),
            dad.len(),
            "group sizes must sum to genome length"
        );

        let mut child = Vec::with_capacity(dad.len());

        self.groups
            .iter()
            .scan(0usize, |i, &g| {
                let start = *i;
                *i += g;
                Some((start, *i)) // tbd: [ga] no need to swap groups with genes with static ranges
            })
            .for_each(|(start, end)| {
                let src = if self.rng.random_bool(0.5) {
                    &dad[start..end]
                } else {
                    &mom[start..end]
                };
                child.extend_from_slice(src);
            });

        self.mutant(&child, self.cross_noise_factor)
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
        let mut evolver = Evolution::new(&[(0, 9), (10, 19), (20, 29)], 1.0, 0.0, &groups);
        let mom = evolver.random();
        let dad = evolver.random();

        let children = evolver.cross(&dad, &mom);
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
