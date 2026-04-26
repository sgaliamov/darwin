use crate::{Gene, GeneRangesRef, Individual};
use rand::{Rng, seq::IteratorRandom};
use std::cmp::Ordering;

/// Evolves in isolation except for explicit migration/crossover.
#[derive(Debug)]
pub struct Pool<G, State> {
    pub number: usize,
    pub individuals: Vec<Individual<G, State>>,
    diversity: f32,
}

impl<G, State> Pool<G, State> {
    pub fn new(number: usize, individuals: Vec<Individual<G, State>>) -> Self {
        Self {
            number,
            individuals,
            diversity: f32::NAN,
        }
    }

    /// Returns current diversity value.
    pub fn diversity(&self) -> f32 {
        self.diversity
    }
}

impl<G: Gene, State> Pool<G, State> {
    /// It's assumed that new individuals inserted last,
    /// which mean that only new duplicates should be removed.
    /// Changes order.
    pub fn dedup(&mut self) {
        self.individuals
            .sort_unstable_by(|a, b| match a.genome.cmp(&b.genome) {
                Ordering::Equal => match (a.fitness.is_nan(), b.fitness.is_nan()) {
                    (false, true) => Ordering::Less,
                    (true, false) => Ordering::Greater,
                    _ => Ordering::Equal,
                },
                other => other,
            });

        self.individuals.dedup_by(|a, b| a.genome == b.genome);
    }

    /// Helper function for tournament selection.
    /// Skip `mutant_count` to not cross the bests, they mutate only.
    pub fn tournament_selection<R: Rng>(
        &self,
        tournament_size: usize,
        mutant_count: usize,
        rng: &mut R,
    ) -> Option<&Individual<G, State>> {
        let k = tournament_size.min(self.individuals.len());

        self.individuals
            .iter()
            .filter(|ind| ind.fitness.is_finite())
            .skip(mutant_count)
            .sample(rng, k)
            .into_iter()
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
    }

    /// Average variance over loci.
    /// Result varies from 0.0 (no diversity) to 1.0 (all different).
    /// Need to be calculated after all changes with the pool.
    pub fn calc_diversity(&mut self, ranges: GeneRangesRef<G>) -> f32 {
        let n = self.individuals.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let loci = ranges.len();
        if loci == 0 {
            return 0.0;
        }

        let variances: Vec<f64> = (0..loci)
            .filter_map(|gn_idx| {
                let range = ranges[gn_idx];
                let (min, max) = (range.0, range.1);
                let span = max.to_f64() - min.to_f64();

                // Skip constant genes (zero span) - they contribute nothing to diversity
                if span == 0.0 {
                    return None;
                }

                let (sum, sum_sq) = self
                    .individuals
                    .iter()
                    .fold((0_f64, 0_f64), |(s, ss), ind| {
                        debug_assert!((min..=max).contains(&ind.genome[gn_idx]));

                        let x = (ind.genome[gn_idx].to_f64() - min.to_f64()) / span;
                        debug_assert!((0.0..=1.0).contains(&x));

                        (s + x, ss + x * x)
                    });

                let mean = sum / n;
                Some((sum_sq / n) - mean * mean)
            })
            .collect();

        // If all genes are constant, diversity is zero
        if variances.is_empty() {
            return 0.0;
        }

        let diversity = variances.iter().sum::<f64>()
            / variances.len() as f64 // 0.25 is max value here, that why need to normalize to 1
            * 4.0;

        // need to clamp as it accumulates a floating-point error
        debug_assert!((0.0 - 1e-6..=1.0 + 1e-6).contains(&diversity));
        self.diversity = diversity.clamp(0.0, 1.0) as f32;
        self.diversity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pool;
    #[allow(dead_code)]
    type G = i64;
    use itertools::Itertools;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use spectral::prelude::*;

    #[test]
    fn test_diversity() {
        let ranges = &[(0, 1_000); 100];

        let individuals = (0..100)
            .map(|_| Individual::<_, ()>::firstborn(0, 0, random_genome(ranges)))
            .collect_vec();

        let mut pool = Pool::new(0, individuals);
        let actual = pool.calc_diversity(ranges);

        // completely random items gives ~0.33
        assert_that!(actual).is_close_to(0.333, 0.02);
    }

    #[test]
    fn diversity_identical_individuals_is_zero() {
        let mut rng = StdRng::from_rng(&mut rand::rng());
        let ranges = &[(0, 100)];
        let genome = ranges
            .iter()
            .map(|range| rng.random_range(range.0..=range.1))
            .collect_vec();

        let individuals = (0..10)
            .map(|_| Individual::firstborn(0, 0, genome.clone()))
            .collect_vec();
        let mut pool = Pool::<_, ()>::new(0, individuals);

        let actual = pool.calc_diversity(ranges);

        assert!((actual - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diversity_cases() {
        let ranges = &[(0, 100)];
        let test_cases = [
            // (genes, expected_diversity)
            (&[vec![1], vec![1]], 0.0),
            (&[vec![0], vec![10]], 0.01),
            (&[vec![0], vec![25]], 0.0625),
            (&[vec![0], vec![50]], 0.25),
            (&[vec![0], vec![75]], 0.5625),
            (&[vec![0], vec![100]], 1.0),
            (&[vec![0, 0], vec![100, 100]], 1.0),
            (&[vec![0, 100], vec![100, 100]], 0.5),
            (&[vec![0, 100], vec![100, 0]], 1.0),
            (&[vec![0; 10], vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 100]], 0.1),
            (&[vec![0; 10], vec![100; 10]], 1.0),
        ];
        for (genes, expected) in test_cases {
            let individuals = genes
                .iter()
                .cloned()
                .map(|genome| Individual::<_, ()>::firstborn(0, 0, genome))
                .collect();

            let mut pool = Pool::new(0, individuals);
            let ranges = vec![ranges[0]; genes[0].len()];

            asserting(&format!("Diversity for {genes:?}"))
                .that(&pool.calc_diversity(&ranges))
                .is_close_to(expected, 1e-6);
        }
    }

    #[test]
    fn test_diversity_with_constant_genes() {
        // Constant gene (1000, 1000) + variable gene (0, 100)
        let ranges = &[(1000, 1000), (0, 100)];
        let test_cases = [
            // First gene is fixed at 1000, diversity comes only from second gene
            (&[vec![1000, 0], vec![1000, 100]], 1.0),
            (&[vec![1000, 50], vec![1000, 50]], 0.0),
            (&[vec![1000, 0], vec![1000, 50]], 0.25),
        ];

        for (genes, expected) in test_cases {
            let individuals = genes
                .iter()
                .cloned()
                .map(|genome| Individual::<_, ()>::firstborn(0, 0, genome))
                .collect();

            let mut pool = Pool::new(0, individuals);

            asserting(&format!("Diversity for {genes:?} with constant gene"))
                .that(&pool.calc_diversity(ranges))
                .is_close_to(expected, 1e-6);
        }
    }

    #[test]
    fn test_diversity_all_constant_genes() {
        // All genes are constant
        let ranges = &[(1000, 1000), (500, 500), (42, 42)];
        let individuals = vec![
            Individual::<_, ()>::firstborn(0, 0, vec![1000i64, 500, 42]),
            Individual::<_, ()>::firstborn(0, 0, vec![1000i64, 500, 42]),
            Individual::<_, ()>::firstborn(0, 0, vec![1000i64, 500, 42]),
        ];

        let mut pool = Pool::new(0, individuals);
        let actual = pool.calc_diversity(ranges);

        // All constant genes should result in zero diversity
        assert!((actual - 0.0).abs() < f32::EPSILON);
    }

    #[test]

    fn test_remove_duplicates() {
        let ranges = &[(0, 1_000); 10];
        let i1 = Individual::firstborn(0, 0, random_genome(ranges));
        let i2 = Individual::firstborn(0, 0, random_genome(ranges));
        let i3 = Individual::new(i2.genome.clone(), i2.lineage.clone());

        let mut pool = Pool::<_, ()>::new(0, vec![i1, i2, i3]);
        pool.dedup();

        assert_that!(pool.individuals.len()).is_equal_to(2);
    }

    #[test]
    fn test_deduplication_keeps_with_fitness() {
        let ranges = &[(0, 1_000); 10];
        let i1 = Individual::firstborn(0, 0, random_genome(ranges));
        let mut i2 = Individual::<_, ()>::new(i1.genome.clone(), i1.lineage.clone());
        i2.fitness = 1.0;
        let mut pool = Pool::new(0, vec![i1, i2]);

        pool.dedup();

        assert_that!(pool.individuals).has_length(1);
        assert_that!(pool.individuals[0].fitness).is_equal_to(1.0);
    }

    fn random_genome(ranges: &[(i64, i64)]) -> Vec<i64> {
        let mut rng = rand::rng();
        ranges.iter().map(|r| rng.random_range(r.0..=r.1)).collect()
    }
}
