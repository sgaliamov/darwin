use crate::{
    Callback, Config, Context, Crossover, GenInfo, Gene, Generator, GeneticAlgorithm, Individual,
    Lineage, Mutator, Pool, Pools, Evaluator,
};
use itertools::Itertools;
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

impl<'a, G, GaState, IndState, Gen, M, C, Sc, Cb>
    GeneticAlgorithm<'a, G, GaState, IndState, Gen, M, C, Sc, Cb>
where
    G: Gene,
    GaState: Sync,
    IndState: Send + Sync,
    Gen: Generator<G, GaState, IndState>,
    M: Mutator<G, GaState, IndState>,
    C: Crossover<G, GaState, IndState>,
    Sc: Evaluator<G, GaState, IndState>,
    Cb: Callback<G, GaState, IndState>,
{
    pub fn new(
        config: &'a Config<G>,
        generator: Gen,
        mutator: M,
        crossover: C,
        evaluator: Sc,
        callback: Cb,
    ) -> Self {
        assert!(config.stagnation_count > 0, "stall_generations must be > 0");
        assert!((0.0..1.0).contains(&config.crossover_ratio));
        assert!(config.random_ratio >= 0.0, "random_ratio must be >= 0");
        assert!((0.0..1.0).contains(&config.mutation_ratio));
        assert!(config.pools >= 1, "Need at least one pool");
        assert!(!config.ranges.is_empty(), "At least one gene is required");
        assert!(config.population_size >= 4, "Population too small");

        let mut pools = Pools::from_vec(
            (0..config.pools)
                .map(|number| Pool::new(number, Vec::with_capacity(config.population_size)))
                .collect_vec(),
        );

        // Pre-compute counts so we do not recompute every generation.
        let mutant_count = config.mutants_count();

        let immigrant_count = (config.population_size as f32 * config.random_ratio)
            .ceil()
            .max(1.0) as usize;

        let crossover_size = (config.population_size as f32 * config.crossover_ratio)
            .ceil()
            .max(1.0) as usize;

        let flat_genome = config.ranges.iter().flatten().cloned().collect_vec();

        // Each seed genome goes to a pool in round-robin fashion.
        if !config.seed.is_empty() {
            for (i, genome) in config.seed.iter().cloned().enumerate() {
                assert!(
                    genome.len() == flat_genome.len(),
                    "seed genome length mismatch: expected {}, got {}",
                    flat_genome.len(),
                    genome.len()
                );
                let pool_idx = i % pools.len();
                pools[pool_idx]
                    .individuals
                    .push(Individual::firstborn(pool_idx, 0, genome));
            }

            // Recalculate diversity for all seeded pools
            pools
                .iter_mut()
                .filter(|p| !p.individuals.is_empty())
                .for_each(|p| {
                    p.calc_diversity(&flat_genome);
                });
        }

        Self {
            flat_genome,
            evaluator,
            callback,
            state: None,
            best_fitness: f64::NEG_INFINITY,
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

    /// Optional external state can be set for use in `evaluator` and `callback`.
    pub fn set_state(&mut self, state: GaState) {
        self.state = Some(state);
    }

    /// Run the evolutionary loop and return a mutable reference to all pools.
    /// Callers can extract top individuals using [`Pools::top_individuals`].
    /// Pools are preserved between runs to allow reusing individuals in subsequent iterations.
    pub fn run(&mut self) -> &mut Pools<G, IndState> {
        self.reset();

        for generation in 0..=self.config.max_generation {
            let stagnation =
                (self.stagnation_counter as f32 / self.config.stagnation_count as f32).min(1.0);

            let sigma = self
                .config
                .sigma
                .get(generation, self.config.max_generation);

            let gen_info = GenInfo {
                generation,
                stagnation,
                distribution: Normal::new(0.0_f32, sigma).expect("`sigma` must be positive"),
            };

            self.mutate(&gen_info);
            self.recombine(&gen_info);
            self.random(&gen_info);
            self.evaluate_generation(&gen_info);

            let best_fitness = self.pools.best().map(|(_, f)| f);
            let improved = best_fitness.is_some_and(|f| f > self.best_fitness);

            if improved {
                self.best_fitness = best_fitness.unwrap();
            }

            let ctx = Context::new(&gen_info, &self.state, &self.pools);
            self.callback.call(&ctx);

            if self.stagnation(improved) {
                break;
            }
        }

        // Return reference to pools; caller can extract top individuals if needed.
        &mut self.pools
    }

    /// Need to reset scores when the instance is reused,
    /// otherwise results from the previous run affects current.
    fn reset(&mut self) {
        self.best_fitness = f64::NEG_INFINITY;
        self.pools.par_iter_mut().for_each(|pool| {
            pool.individuals
                .iter_mut()
                .for_each(|ind| ind.fitness = f64::NAN);
        });
    }

    /// Evaluate all individuals (parallel) and sort each pool descending by
    /// fitness. Truncate back to `population_size` in case parents + offspring
    /// exceeded the limit.
    fn evaluate_generation(&mut self, gen_info: &GenInfo) {
        // Phase 1: remove duplicates before scoring.
        self.pools.par_iter_mut().for_each(|pool| pool.dedup());

        // Phase 2: score unscored individuals.
        let evaluator = &self.evaluator;
        let ctx = Context::new(gen_info, &self.state, &self.pools);
        let scores: Vec<Vec<(usize, f64, Option<IndState>)>> = self
            .pools
            .par_iter()
            .map(|pool| {
                pool.individuals
                    .par_iter()
                    .enumerate()
                    .filter(|(_, ind)| !ind.fitness.is_finite())
                    .map(|(i, ind)| {
                        let (fitness, s) = evaluator.evaluate(ind, &ctx);
                        (i, fitness, s)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Phase 3: apply collected scores.
        for (pool_scores, pool) in scores.into_iter().zip(self.pools.iter_mut()) {
            for (idx, fitness, ind_state) in pool_scores {
                pool.individuals[idx].fitness = fitness;
                pool.individuals[idx].state = ind_state;
            }
        }

        // Phase 4: housekeeping — retain, sort, truncate, update diversity.
        self.pools.par_iter_mut().for_each(|pool| {
            pool.individuals.retain(|ind| ind.fitness.is_finite());
            pool.individuals
                .sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
            pool.individuals.truncate(self.config.population_size);
            // diversity need to be updated before callback to provide actual state.
            pool.calc_diversity(&self.flat_genome);
        });
    }

    /// Spawn elite mutants inside every pool.
    fn mutate(&mut self, gen_info: &GenInfo) {
        let mutator = &self.mutator;
        let mutant_count = self.mutant_count;
        let ctx = Context::new(gen_info, &self.state, &self.pools);
        let mutants = self
            .pools
            .par_iter()
            .enumerate()
            .filter(|(_, pool)| !pool.individuals.is_empty())
            .map(|(idx, pool)| {
                let m = mutant_count.min(pool.individuals.len());
                let new_mutants = pool.individuals[..m]
                    .iter()
                    .filter_map(|parent| {
                        mutator
                            .mutant(parent, &ctx)
                            .map(|genome| (genome, parent.lineage.generation()))
                    })
                    .map(|(genome, parent_gen)| {
                        Individual::new(
                            genome,
                            Lineage::Mutant(idx, gen_info.generation, parent_gen),
                        )
                    })
                    .collect_vec();
                (idx, new_mutants)
            })
            .collect::<Vec<_>>();

        for (idx, new_mutants) in mutants {
            self.pools[idx].individuals.extend(new_mutants);
        }
    }

    /// Recombine pools into offspring, possibly mutate, then migrate them.
    /// Short story: pair pools, breed `crossover_size` times, push kids to a chosen pool.
    fn recombine(&mut self, gen_info: &GenInfo) {
        let crossover: &C = &self.crossover;
        let crossover_size = self.crossover_size;
        let mutant_count = self.mutant_count;
        let config = self.config;
        let pools: &Pools<G, IndState> = &self.pools;
        let ctx = Context::new(gen_info, &self.state, pools);
        let offspring: Vec<_> = pools
            .pairs(gen_info.generation)
            .par_iter()
            .map_init(
                || SmallRng::from_rng(&mut rand::rng()),
                |rng, &(ia, ib)| {
                    let pa = &pools[ia];
                    let pb = &pools[ib];
                    let mut kids = Vec::with_capacity(crossover_size * 2);

                    for _ in 0..crossover_size {
                        let (Some(dad), Some(mom)) = (
                            pa.tournament_selection(config.tournament_size, mutant_count, rng),
                            pb.tournament_selection(config.tournament_size, mutant_count, rng),
                        ) else {
                            continue;
                        };

                        for g in crossover.cross(dad, mom, &ctx) {
                            kids.push(Individual::new(
                                g,
                                Lineage::Child(
                                    ia,
                                    gen_info.generation,
                                    dad.lineage.generation(),
                                    mom.lineage.generation(),
                                ),
                            ));
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
    fn random(&mut self, gen_info: &GenInfo) {
        let quota = self.immigrant_count;
        // Generation 0: overseed so the evaluator has enough valid individuals to work with.
        let target = if gen_info.generation == 0 {
            self.config.population_size * 10
        } else {
            self.config.population_size
        };

        let generator = &self.generator;
        let ctx = Context::new(gen_info, &self.state, &self.pools);
        let immigrants = self
            .pools
            .par_iter()
            .enumerate()
            .map(|(idx, pool)| {
                let current_cnt = pool.individuals.len();
                let deficit = target.saturating_sub(current_cnt);
                let count = quota.max(deficit);
                let new_individuals = std::iter::repeat_with(|| {
                    Individual::firstborn(idx, gen_info.generation, generator.generate(&ctx))
                })
                .take(count)
                .collect::<Vec<_>>();
                (idx, new_individuals)
            })
            .collect::<Vec<_>>();

        for (idx, new_individuals) in immigrants {
            self.pools[idx].individuals.extend(new_individuals);
        }
    }

    /// Check for stagnation: if the global best has not improved for a
    /// number of generations equal to `stall_generations`, return `true`.
    fn stagnation(&mut self, improved: bool) -> bool {
        if improved {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        self.stagnation_counter >= self.config.stagnation_count
    }
}
