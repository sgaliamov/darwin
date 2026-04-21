use crate::{
    CallbackFn, Config, Evolver, GeneticAlgorithm, Individual, Lineage, Pool, Pools, ScoreFn,
};
use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;

impl<'a, GaState, IndState, E: Evolver> GeneticAlgorithm<'a, GaState, IndState, E>
where
    GaState: Sync,
    IndState: Send + Sync,
{
    pub fn new(config: &'a Config, evolver: E) -> Self {
        assert!(
            config.min_mutation_sigma <= config.max_mutation_sigma,
            "min_mutation_sigma must be <= mutation_sigma"
        );
        assert!(
            config.min_mutation_sigma > 0.0,
            "min_mutation_sigma must be > 0"
        );
        assert!(config.stagnation_count > 0, "stall_generations must be > 0");
        assert!((0.0..1.0).contains(&config.crossover_ratio));
        assert!(config.random_ratio >= 0.0, "random_ratio must be >= 0");
        assert!((0.0..1.0).contains(&config.mutation_ratio));
        assert!(config.pools >= 1, "Need at least two pools");
        assert!(!config.ranges.is_empty(), "At least one gene is required");
        assert!(config.population_size >= 4, "Population too small");

        let mut pools = Pools::from_vec(
            (0..config.pools)
                .map(|number| Pool::new(number, Vec::with_capacity(config.population_size)))
                .collect_vec(),
        );

        // Pre-compute counts so we do not recompute every generation.
        let mutant_count = config.mutant_count();

        let immigrant_count = (config.population_size as f32 * config.random_ratio)
            .ceil()
            .max(1.0) as usize;

        let crossover_size = (config.population_size as f32 * config.crossover_ratio)
            .ceil()
            .max(1.0) as usize;

        let ranges = config.ranges.iter().flatten().cloned().collect_vec();

        // Each seed genome goes to a pool in round-robin fashion.
        if !config.seed.is_empty() {
            for (i, genome) in config.seed.iter().cloned().enumerate() {
                assert!(
                    genome.len() == ranges.len(),
                    "seed genome length mismatch: expected {}, got {}",
                    ranges.len(),
                    genome.len()
                );
                let pool_idx = i % pools.len();
                pools[pool_idx]
                    .individuals
                    .push(Individual::firstborn(0, genome));
            }

            // Recalculate diversity for all seeded pools
            pools
                .iter_mut()
                .filter(|p| !p.individuals.is_empty())
                .for_each(|p| {
                    p.calc_diversity(&ranges);
                });
        }

        Self {
            ranges,
            score_fn: |_, _| (f64::NAN, None),
            callback_fn: |_, _, _, _| {},
            state: None,
            best: None,
            config,
            pools,
            evolver,
            crossover_size,
            stagnation_counter: 0,
            immigrant_count,
            mutant_count,
        }
    }

    pub fn set_score_fn(&mut self, score_fn: ScoreFn<GaState, IndState>) {
        self.score_fn = score_fn;
    }

    pub fn set_callback_fn(&mut self, callback_fn: CallbackFn<GaState, IndState>) {
        self.callback_fn = callback_fn;
    }

    pub fn set_state(&mut self, state: GaState) {
        self.state = Some(state);
    }

    /// Run the evolutionary loop and return a mutable reference to all pools.
    /// Callers can extract top individuals using [`Pools::top_individuals`].
    /// Pools are preserved between runs to allow reusing individuals in subsequent iterations.
    pub fn run(&mut self) -> &mut Pools<IndState> {
        self.reset();

        // tbd: [future, ga] after finding a good individual, reset all pools at the end, add it as a seed and restart evolution one more time.
        // tbd: [future, ga] identical pools should be merged to save computation time.
        for generation in 0..=self.config.max_generation {
            self.mutate(generation);
            self.recombine(generation);
            self.random(generation);
            self.evaluate_generation();
            let new_champ = self.update_champ();
            (self.callback_fn)(generation, &self.best, &self.pools, &mut self.state);

            if self.stagnation(new_champ) {
                break;
            }
        }

        // Return reference to pools; caller can extract top individuals if needed.
        &mut self.pools
    }

    /// Need to reset scores when the instance is reused,
    /// otherwise results from the previous run affects current.
    fn reset(&mut self) {
        self.best = None;
        self.pools.par_iter_mut().for_each(|pool| {
            pool.individuals
                .iter_mut()
                .for_each(|ind| ind.fitness = f64::NAN);
        });
    }

    /// Evaluate all individuals (parallel) and sort each pool descending by
    /// fitness. Truncate back to `population_size` in case parents + offspring
    /// exceeded the limit.
    fn evaluate_generation(&mut self) {
        self.pools.par_iter_mut().for_each(|pool| {
            pool.dedup();

            pool.individuals
                .par_iter_mut()
                .for_each(|ind| ind.evaluate(&self.score_fn, &self.state));

            pool.individuals.retain(|ind| ind.fitness.is_finite());

            pool.individuals
                .sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));

            pool.individuals.truncate(self.config.population_size);

            // diversity need to be updated before callback to provide actual state.
            pool.calc_diversity(&self.ranges);
        });
    }

    /// Update `global_best` and diversity for pools.
    ///
    ///  Returns `true` if a better individual is found.
    fn update_champ(&mut self) -> bool {
        let candidate = self
            .pools
            .par_iter()
            .flat_map(|pool| pool.individuals.par_iter())
            .filter(|ind| ind.fitness.is_finite())
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness));

        // tbd: [future, ga] ideally should not clone anything. it should be able to keep reference.
        match (&self.best, candidate) {
            (None, Some(champ)) => {
                let best = (champ.genome.clone(), champ.fitness);
                self.best = Some(best);
                true
            }
            (Some((_, f)), Some(champ)) if &champ.fitness > f => {
                let best = (champ.genome.clone(), champ.fitness);
                self.best = Some(best);
                true
            }
            _ => false,
        }
    }

    /// Spawn elite mutants inside every pool.
    fn mutate(&mut self, generation: usize) {
        // Calculate stagnation boost: ratio grows as we get stuck
        let stagnation_boost =
            (self.stagnation_counter as f32 / self.config.stagnation_count as f32).min(1.0);

        // Borrow evolver and mutant_count before the parallel loop so that
        // par_iter_mut can take a mutable borrow of pools independently.
        let evolver = &self.evolver;
        let mutant_count = self.mutant_count;

        self.pools.par_iter_mut().for_each(|pool| {
            if pool.individuals.is_empty() {
                return;
            }

            let m = mutant_count.min(pool.individuals.len());

            // Apply both diversity-based and stagnation-based noise.
            // When the diversity is high, we reduce mutation to allow exploitation.
            // When stagnating, we increase mutation to force exploration.
            let noise_factor = pool.noise_factor(stagnation_boost);
            debug_assert!(noise_factor.is_finite());

            let mutants = pool.individuals[..m]
                .iter()
                .filter_map(|parent| {
                    evolver
                        .mutant(&parent.genome, generation, noise_factor)
                        .map(|genome| (genome, parent.lineage.generation()))
                })
                .map(|(genome, parent)| Individual::new(genome, Lineage::Mutant(generation, parent)))
                .collect_vec();

            pool.individuals.extend(mutants);
        });
    }

    /// Recombine pools into offspring, possibly mutate, then migrate them.
    /// Short story: pair pools, breed `crossover_size` times, push kids to a chosen pool.
    fn recombine(&mut self, generation: usize) {
        let Self {
            pools,
            crossover_size,
            evolver,
            mutant_count,
            config: Config {
                tournament_size, ..
            },
            ..
        } = self;

        let crossover_size = *crossover_size;
        let tournament_size = *tournament_size;
        let mutant_count = *mutant_count;
        // Coerce to shared ref so the closure is Sync (E: Evolver: Sync).
        let evolver: &E = evolver;

        let offspring: Vec<_> = pools
            .pairs(generation)
            .par_iter()
            .map_init(
                || SmallRng::from_rng(&mut rand::rng()),
                |rng, &(ia, ib)| {
                    let pa = &pools[ia];
                    let pb = &pools[ib];
                    let mut kids = Vec::with_capacity(crossover_size * 2); // 2 as cross may produce 2 kids

                    for _ in 0..crossover_size {
                        let (Some(dad), Some(mom)) = (
                            pa.tournament_selection(tournament_size, mutant_count, rng),
                            pb.tournament_selection(tournament_size, mutant_count, rng),
                        ) else {
                            continue;
                        };

                        let (ga, gb) = (dad.lineage.generation(), mom.lineage.generation());

                        for g in evolver.cross(&dad.genome, &mom.genome, generation) {
                            kids.push(Individual::new(g, Lineage::Child(generation, ga, gb)));
                        }
                    }

                    (!kids.is_empty()).then_some((ia, ib, kids))
                },
            )
            .flatten()
            .collect();

        let mig = self.config.migration_factor;
        let mut rng = SmallRng::from_rng(&mut rand::rng());
        for (ia, ib, kids) in offspring {
            let idx = if rng.random_bool(mig) {
                ia.max(ib)
            } else {
                ia.min(ib)
            };

            self.pools[idx].individuals.extend(kids);
        }
    }

    /// Restore populations size to the original with random immigrants.
    /// May overpopulate.
    fn random(&mut self, generation: usize) {
        let quota = self.immigrant_count;
        // have to make first generation big as many individuals are not valid.
        //      ideally generation method should be delegated to a client and he could ensure,
        //      that new individuals are valid.
        let target = if generation == 0 {
            self.config.population_size * 10
        } else {
            self.config.population_size
        };

        let evolver = &self.evolver;

        self.pools.par_iter_mut().for_each(|pool| {
            let current = pool.individuals.len();
            let deficit = target.saturating_sub(current);
            let count = quota.max(deficit);

            pool.individuals.extend(
                std::iter::repeat_with(|| {
                    let genome = evolver.random();
                    Individual::firstborn(generation, genome)
                })
                .take(count),
            );
        });
    }

    /// Check for stagnation: if the global best has not improved for a
    /// number of generations equal to `stall_generations`, return `true`.
    fn stagnation(&mut self, new_best: bool) -> bool {
        if new_best {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        self.stagnation_counter >= self.config.stagnation_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Genome, GenomeRef};
    use spectral::prelude::*;
    use std::io::{BufWriter, Write};

    type State<'a> = (&'a Config, BufWriter<&'a mut Vec<u8>>);

    /// Fitness = ∑x² → 0 at the origin.
    fn sphere(genome: GenomeRef, _state: &Option<State>) -> (f64, Option<()>) {
        let score = -genome
            .iter()
            .map(|&x| x as f64 / 99.0)
            .map(|x| x.powi(2))
            .sum::<f64>();
        (score, Some(()))
    }

    fn callback_fn(
        g: usize,
        _: &Option<(Genome, f64)>,
        pools: &Pools<()>,
        state: &mut Option<State>,
    ) {
        let (config, writer) = state.as_mut().unwrap();

        for pool in pools.iter() {
            let sigma = config.sigma(g);
            let line = format!(
                "{g}_{} σ{:.2} π{:.6} total: {}\n",
                pool.number,
                sigma,
                pool.diversity(),
                pool.individuals.len()
            );

            writer.write_all(line.as_bytes()).unwrap();

            for (_, ind) in pool
                .individuals
                .iter()
                .enumerate()
                .filter(|&(i, _)| i < 5 || i >= config.population_size - 3)
            {
                let genes = ind.genome.iter().map(|x| format!("{x:>2}")).join(", ");
                let line = format!("\t{g} [{genes}]: {} {}\n", ind.fitness, ind.lineage);
                writer.write_all(line.as_bytes()).unwrap();
            }
        }

        writer.write_all("\n".as_bytes()).unwrap();
    }

    #[test]
    fn finds_origin_within_ten_runs() {
        let config = Config {
            max_generation: 1_000,
            stagnation_count: 100,
            population_size: 100,
            pools: 8,
            max_mutation_sigma: 2.0,
            min_mutation_sigma: 1.0,
            ranges: vec![vec![(0, 99); 6]],
            mutation_noise_factor: 1.0,
            tournament_size: 3,
            seed: vec![vec![0, 0, 0, 0, 0, 99]],
            ..Default::default()
        };

        let mut buffer = Vec::new();
        let writer = BufWriter::new(&mut buffer);

        // to debug
        // let file = File::create("../target/tmp.txt").unwrap();
        // let mut writer = BufWriter::new(file);
        let ok = test_run(config, writer);

        assert_that!(ok).is_true();
    }

    fn test_run(config: Config, writer: BufWriter<&mut Vec<u8>>) -> bool {
        use crate::Evolution;
        let ranges: Vec<_> = config.ranges.iter().flatten().cloned().collect();
        let groups: Vec<_> = config.ranges.iter().map(|g| g.len()).collect();
        let evolver = Evolution::new(
            &ranges,
            config.mutation_noise_factor,
            &groups,
            config.max_mutation_sigma,
            config.min_mutation_sigma,
            config.max_generation,
        );
        let mut ga = GeneticAlgorithm::new(&config, evolver);
        ga.set_score_fn(sphere);
        ga.set_callback_fn(callback_fn);
        ga.set_state((&config, writer));
        let pools = ga.run();
        pools
            .top_individuals_mut(config.bests)
            .first()
            .expect("expected some bests")
            .fitness
            <= 0.0001
    }
}
