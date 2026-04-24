use crate::{GeneRangesRef, Individual};
use rand::{Rng, seq::IteratorRandom};
use std::cmp::Ordering;

/// Evolves in isolation except for explicit migration/crossover.
#[derive(Debug)]
pub struct Pool<State> {
    pub number: usize,
    pub individuals: Vec<Individual<State>>,
    diversity: f32,
}

impl<State> Pool<State> {
    pub fn new(number: usize, individuals: Vec<Individual<State>>) -> Self {
        Self {
            number,
            individuals,
            diversity: f32::NAN,
        }
    }

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
    ) -> Option<&Individual<State>> {
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
    pub fn calc_diversity(&mut self, ranges: GeneRangesRef) -> f32 {
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
                let span = (max - min) as f64;

                // Skip constant genes (zero span) - they contribute nothing to diversity
                if span == 0.0 {
                    return None;
                }

                let (sum, sum_sq) = self
                    .individuals
                    .iter()
                    .fold((0_f64, 0_f64), |(s, ss), ind| {
                        debug_assert!((min..=max).contains(&ind.genome[gn_idx]));

                        let x = (ind.genome[gn_idx] - min) as f64 / span;
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

    /// Calculate mutation noise factor with optional stagnation boost for increased exploration.
    /// Returns a value in [0.0, 1.0] where higher means more aggressive mutations.
    ///
    /// `stagnation_boost` (0.0 to 1.0) increases exploration when stagnating:
    /// - 0.0 = normal noise based on diversity
    /// - 1.0 = maximum noise (1.0) to force exploration
    pub fn noise_factor(&self, stagnation_boost: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&stagnation_boost));

        let base_noise = 1.0 - self.diversity;
        base_noise + stagnation_boost * (1.0 - base_noise)
    }

    /// Returns current diversity value.
    pub fn diversity(&self) -> f32 {
        self.diversity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Context, SigmaConfig, DefaultGenerator, DefaultMutator, Generator, Lineage, Mutator, Pool};
    use itertools::Itertools;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use spectral::prelude::*;

    #[test]
    fn test_diversity() {
        let ranges = &[(0, 1_000); 100];
        let generator = DefaultGenerator::new(ranges);

        let individuals = (0..100)
            .map(|_| Individual::<()>::firstborn(0, generator.generate()))
            .collect_vec();

        let mut pool = Pool::new(0, individuals);
        let actual = pool.calc_diversity(ranges);

        // completely random items gives ~0.33
        assert_that!(actual).is_close_to(0.333, 0.02);
    }

    #[test]
    fn test_mutants() {
        let ranges = &[(0, 1000)];
        let test_cases = &[
            // sigma, tolerance
            (0.05, 1e-6), // no deviations
            (0.1, 1e-6),  // 499 - 501 - min deviations
            (2., 1e-4),   // 490 - 511
            (20., 1e-2),  // 400 - 600
            (50., 0.05),  // 250 - 750
            (100., 0.1),  // 0 - 1000 - full bell
            (1000., 1.0), // flat, but drops 2/3 populations
        ];

        for &(sigma, tolerance) in test_cases {
            let items = (0..1_000)
                .filter_map(|g| {
                    let mutator = DefaultMutator::new(
                        ranges,
                        SigmaConfig { max: sigma, min: sigma },
                    );
                    let ga_config = Config::default();
                    mutator.mutant(&[500], &Context { generation: 0, diversity: 0.5, stagnation: 0.0, config: &ga_config, state: &None::<()> })
                        .map(|genome| Individual::<()>::new(genome, Lineage::Mutant(0, g)))
                })
                .collect_vec();

            let _left = items.len();
            let mut pool = Pool::new(0, items);
            let actual = pool.calc_diversity(ranges);

            asserting(&format!("Sigma {sigma}"))
                .that(&actual)
                .is_close_to(0.0, tolerance);
        }
    }

    // #[test]
    fn _tuning() {
        let ranges = &[(0, 100)];
        let std_dev = 0.5; // 5% // 1 - 10% // 2 - 20% // 5 - 50%;
        let mutator = DefaultMutator::new(
            ranges,
            SigmaConfig { max: std_dev, min: std_dev },
        );
        let ga_cfg = Config::default();
        let ctx = Context { generation: 0, diversity: 0.5, stagnation: 0.0, config: &ga_cfg, state: &None::<()> };

        let items = (0..100_000)
            .map(|g| {
                let genome = mutator.mutant(&[50], &ctx).unwrap();
                Individual::<()>::new(genome, Lineage::Mutant(0, g))
            })
            .collect_vec();

        let left = items.len();
        let mut pool = Pool::new(0, items);
        let actual = pool.calc_diversity(ranges);

        // to tune sigma
        let genomes = pool
            .individuals
            .into_iter()
            .flat_map(|x| x.genome)
            .sorted()
            .chunk_by(|&x| x)
            .into_iter()
            .map(|x| (x.0, x.1.count()))
            .collect_vec();
        let min_max = genomes.iter().minmax_by_key(|x| x.1).into_option().unwrap();
        let avg = genomes.iter().map(|x| x.1 as f32).sum::<f32>() as usize / genomes.len();
        println!(
            "{:#?}\n{}: [{:?}, {:?}, {:?}, {:?}, {}]\n{actual}\n",
            genomes
                .iter()
                .map(|x| format!("{}: {}", x.0, x.1))
                .collect_vec(),
            left,
            genomes.first().unwrap(),
            min_max.0,
            min_max.1,
            genomes.last().unwrap(),
            avg
        );
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
            .map(|_| Individual::firstborn(0, genome.clone()))
            .collect_vec();
        let mut pool = Pool::<()>::new(0, individuals);

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
                .map(|genome| Individual::<()>::firstborn(0, genome))
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
                .map(|genome| Individual::<()>::firstborn(0, genome))
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
            Individual::<()>::firstborn(0, vec![1000, 500, 42]),
            Individual::<()>::firstborn(0, vec![1000, 500, 42]),
            Individual::<()>::firstborn(0, vec![1000, 500, 42]),
        ];

        let mut pool = Pool::new(0, individuals);
        let actual = pool.calc_diversity(ranges);

        // All constant genes should result in zero diversity
        assert!((actual - 0.0).abs() < f32::EPSILON);
    }

    #[test]

    fn test_remove_duplicates() {
        let ranges = &[(0, 1_000); 10];
        let generator = DefaultGenerator::new(ranges);
        let i1 = Individual::firstborn(0, generator.generate());
        let i2 = Individual::firstborn(0, generator.generate());
        let i3 = Individual::new(i2.genome.clone(), i2.lineage.clone());

        let mut pool = Pool::<()>::new(0, vec![i1, i2, i3]);
        pool.dedup();

        assert_that!(pool.individuals).has_length(2);
    }

    #[test]
    fn test_deduplication_keeps_with_fitness() {
        let ranges = &[(0, 1_000); 10];
        let generator = DefaultGenerator::new(ranges);
        let i1 = Individual::firstborn(0, generator.generate());
        let mut i2 = Individual::<()>::new(i1.genome.clone(), i1.lineage.clone());
        i2.fitness = 1.0;
        let mut pool = Pool::new(0, vec![i1, i2]);

        pool.dedup();

        assert_that!(pool.individuals).has_length(1);
        assert_that!(pool.individuals[0].fitness).is_equal_to(1.0);
    }

    #[test]
    fn test_noise_factor_with_stagnation() {
        let test_cases = [
            // (diversity, stagnation_boost, expected_noise)
            (0.0, 0.0, 1.0),  // low diversity, no stagnation → max noise
            (1.0, 0.0, 0.0),  // high diversity, no stagnation → min noise
            (0.5, 0.0, 0.5),  // mid diversity, no stagnation → mid noise
            (0.0, 1.0, 1.0),  // low diversity, max stagnation → max noise
            (1.0, 1.0, 1.0),  // high diversity, max stagnation → max noise (forced exploration)
            (0.5, 0.5, 0.75), // mid diversity, mid stagnation → boosted noise
            (0.8, 0.5, 0.6), // high diversity (0.2 base noise), mid stagnation → 0.2 + 0.5 * 0.8 = 0.6
        ];

        for (div, boost, expected) in test_cases {
            let individuals = vec![Individual::<()>::firstborn(0, vec![50])];
            let mut pool = Pool::new(0, individuals);
            pool.diversity = div; // manually set diversity for testing

            let noise = pool.noise_factor(boost);

            asserting(&format!("Noise for diversity={div}, boost={boost}"))
                .that(&noise)
                .is_close_to(expected, 1e-6);
        }
    }
}
