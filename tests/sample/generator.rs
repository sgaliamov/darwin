use darwin::{Context, Gene, GeneRanges, GeneRangesRef, Generator, Genome};
use rand::RngExt;
use rand::distr::uniform::SampleUniform;

/// Generates random genomes from declared ranges.
pub struct DefaultGenerator<G> {
    ranges: GeneRanges<G>,
}

impl<G: Gene + SampleUniform> DefaultGenerator<G> {
    pub fn new(ranges: GeneRangesRef<G>) -> Self {
        Self {
            ranges: ranges.to_vec(),
        }
    }
}

impl<G: Gene + SampleUniform, GaState, IndState> Generator<G, GaState, IndState>
    for DefaultGenerator<G>
{
    /// Produce a random genome; each gene is drawn uniformly from its range.
    fn generate(&self, _ctx: &Context<'_, G, GaState, IndState>) -> Genome<G> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every gene of `generate()` must stay within its declared range.
    #[test]
    fn random_genome_stays_in_range() {
        use darwin::{Config, Context};
        let ranges: Vec<(i64, i64)> = vec![(0, 9), (10, 19), (100, 200)];
        let generator = DefaultGenerator::new(&ranges);
        let cfg = Config::<i64>::default();
        let pools = darwin::Pools::<i64, ()>::from_vec(vec![]);
        let epoch = darwin::Epoch {
            generation: 0,
            stagnation: 0.0,
            normal: rand_distr::Normal::new(0.0_f32, cfg.sigma.get(0, cfg.max_generation)).unwrap(),
        };
        let ctx = Context::<i64, (), ()>::new(&epoch, &None::<()>, &pools);
        for _ in 0..100 {
            let g = generator.generate(&ctx);
            for (gene, &(lo, hi)) in g.iter().zip(ranges.iter()) {
                assert!(*gene >= lo && *gene <= hi, "gene {gene} out of [{lo},{hi}]");
            }
        }
    }
}
