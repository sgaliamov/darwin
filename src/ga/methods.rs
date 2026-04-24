use crate::{Gene,
    CallbackFn, Config, Context, Crossover, Generator, GeneticAlgorithm, Individual, Lineage, Mutator, Pool, Pools, ScoreFn
};
use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;

impl<'a, G, GaState, IndState, Gen, M, C> GeneticAlgorithm<'a, G, GaState, IndState, Gen, M, C>
where
    G: Gene,
    GaState: Sync,
    IndState: Send + Sync,
    Gen: Generator<G>,
    M: Mutator<G, GaState>,
    C: Crossover<G, GaState>,
{
    pub fn new(config: &'a Config<G>, generator: Gen, mutator: M, crossover: C) -> Self {
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
            flat_genome: ranges,
            score_fn: |_, _| (f64::NAN, None),
            callback_fn: |_, _, _, _| {},
            state: None,
            best: None,
            config,
            pools,
            generator,
            mutator,
            crossover,
            crossover_size,
            stagnation_counter: 0,
            immigrant_count,
            mutant_count,
        }
    }

    pub fn set_score_fn(&mut self, score_fn: ScoreFn<G, GaState, IndState>) {
        self.score_fn = score_fn;
    }

    pub fn set_callback_fn(&mut self, callback_fn: CallbackFn<G, GaState, IndState>) {
        self.callback_fn = callback_fn;
    }

    /// Optional external state can be set for use in `score_fn` and `callback_fn`.
    pub fn set_state(&mut self, state: GaState) {
        self.state = Some(state);
    }

    /// Run the evolutionary loop and return a mutable reference to all pools.
    /// Callers can extract top individuals using [`Pools::top_individuals`].
    /// Pools are preserved between runs to allow reusing individuals in subsequent iterations.
    pub fn run(&mut self) -> &mut Pools<G, IndState> {
        self.reset();

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
            pool.calc_diversity(&self.flat_genome);
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
        let evolver = &self.mutator;
        let mutant_count = self.mutant_count;
        let config = self.config;
        let state = &self.state;

        self.pools.par_iter_mut().for_each(|pool| {
            if pool.individuals.is_empty() {
                return;
            }

            let m = mutant_count.min(pool.individuals.len());

            let ctx = Context {
                generation,
                diversity: pool.diversity(),
                stagnation: stagnation_boost,
                config,
                state,
            };

            let mutants = pool.individuals[..m]
                .iter()
                .filter_map(|parent| {
                    evolver
                        .mutant(&parent.genome, &ctx)
                        .map(|genome| (genome, parent.lineage.generation()))
                })
                .map(|(genome, parent)| {
                    Individual::new(genome, Lineage::Mutant(generation, parent))
                })
                .collect_vec();

            pool.individuals.extend(mutants);
        });
    }

    /// Recombine pools into offspring, possibly mutate, then migrate them.
    /// Short story: pair pools, breed `crossover_size` times, push kids to a chosen pool.
    fn recombine(&mut self, generation: usize) {
        // todo: should use stagnation_counter directly?
        let stagnation_boost =
            (self.stagnation_counter as f32 / self.config.stagnation_count as f32).min(1.0);

        let Self {
            pools,
            crossover_size,
            crossover,
            mutant_count,
            state,
            config,
            ..
        } = self;

        let crossover_size = *crossover_size;
        let tournament_size = config.tournament_size;
        let mutant_count = *mutant_count;
        // Coerce to shared ref so the closure is Sync.
        let crossover: &C = crossover;
        let state: &Option<GaState> = state;

        let offspring: Vec<_> = pools
            .pairs(generation)
            .par_iter()
            .map_init(
                || SmallRng::from_rng(&mut rand::rng()),
                |rng, &(ia, ib)| {
                    let pa = &pools[ia];
                    let pb = &pools[ib];
                    let mut kids = Vec::with_capacity(crossover_size * 2); // 2 as cross may produce 2 kids

                    // Average diversity of the two paired pools gives a fair signal
                    // without arbitrarily favouring one partner over the other.
                    let diversity = (pa.diversity() + pb.diversity()) / 2.0;

                    for _ in 0..crossover_size {
                        let (Some(dad), Some(mom)) = (
                            pa.tournament_selection(tournament_size, mutant_count, rng),
                            pb.tournament_selection(tournament_size, mutant_count, rng),
                        ) else {
                            continue;
                        };

                        let (ga, gb) = (dad.lineage.generation(), mom.lineage.generation());

                        for g in crossover.cross(
                            &dad.genome,
                            &mom.genome,
                            &Context {
                                generation,
                                diversity,
                                stagnation: stagnation_boost,
                                config,
                                state,
                            },
                        ) {
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
        // have to make the first generation bigger, as many individuals are not valid.
        // ideally generation method should be delegated to a client and he could ensure,
        // that new individuals are valid.
        let target = if generation == 0 {
            self.config.population_size * 10
        } else {
            self.config.population_size
        };

        let evolver = &self.generator;

        self.pools.par_iter_mut().for_each(|pool| {
            let current = pool.individuals.len();
            let deficit = target.saturating_sub(current);
            let count = quota.max(deficit);

            pool.individuals.extend(
                std::iter::repeat_with(|| {
                    let genome = evolver.generate();
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

