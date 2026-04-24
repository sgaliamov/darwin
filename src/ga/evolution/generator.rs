use super::super::genome::{GeneRanges, GeneRangesRef, Genome};
use super::Generator;
use rand::RngExt;

/// Generates random genomes from declared ranges.
pub struct DefaultGenerator {
    ranges: GeneRanges,
}

impl DefaultGenerator {
    pub fn new(ranges: GeneRangesRef) -> Self {
        Self { ranges: ranges.to_vec() }
    }
}

impl Generator for DefaultGenerator {
    /// Produce a random genome; each gene is drawn uniformly from its range.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every gene of `generate()` must stay within its declared range.
    #[test]
    fn random_genome_stays_in_range() {
        let ranges = vec![(0, 9), (10, 19), (100, 200)];
        let generator = DefaultGenerator::new(&ranges);
        for _ in 0..100 {
            let g = generator.generate();
            for (gene, &(lo, hi)) in g.iter().zip(ranges.iter()) {
                assert!(*gene >= lo && *gene <= hi, "gene {gene} out of [{lo},{hi}]");
            }
        }
    }
}
