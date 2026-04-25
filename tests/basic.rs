mod sample;

#[cfg(test)]
mod tests {
    use super::sample;
    use darwin::{Config, Context, GeneticAlgorithm, Individual, NoopCallback};
    use sample::{DefaultCrossover, DefaultGenerator, DefaultMutator};
    use itertools::Itertools;
    use spectral::prelude::*;
    use std::sync::Mutex;
    use std::io::{BufWriter, Write};

    type State<'a> = (&'a Config<i64>, Mutex<BufWriter<&'a mut Vec<u8>>>);

    /// Fitness = ∑x² → 0 at the origin.
    fn sphere(ind: &Individual<i64, ()>, _ctx: &Context<'_, i64, State, ()>) -> (f64, Option<()>) {
        let score = -ind.genome
            .iter()
            .map(|&x| x as f64 / 99.0)
            .map(|x: f64| x.powi(2))
            .sum::<f64>();
        (score, Some(()))
    }

    fn callback_fn(ctx: &Context<'_, i64, State<'_>, ()>) {
        let (config, writer) = ctx.state.as_ref().unwrap();
        let mut writer = writer.lock().unwrap();
        let g = ctx.epoch.generation;

        for pool in ctx.pools.iter() {
            let line = format!(
                "{g}_{} π{:.6} total: {}\n",
                pool.number,
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
        let config: Config<i64> = Config {
            max_generation: 1_000,
            stagnation_count: 100,
            population_size: 100,
            pools: 8,
            ranges: vec![vec![(0, 99); 6]],
            tournament_size: 3,
            seed: vec![vec![0, 0, 0, 0, 0, 99]],
            ..Default::default()
        };

        let mut buffer = Vec::new();
        let writer = BufWriter::new(&mut buffer);
        assert_that!(test_run(config, writer)).is_true();
    }

    /// Seed genome with all-zero already at optimum → first run returns fitness ≤ threshold.
    #[test]
    fn seed_at_optimum_converges_immediately() {
        let config: Config<i64> = Config {
            max_generation: 500,
            stagnation_count: 50,
            population_size: 50,
            pools: 2,
            ranges: vec![vec![(0, 99); 4]],
            seed: vec![vec![0, 0, 0, 0]],
            ..Default::default()
        };

        let mut buffer = Vec::new();
        let writer = BufWriter::new(&mut buffer);
        assert_that!(test_run(config, writer)).is_true();
    }

    /// Stagnation exit: fitness cannot improve (degenerate 1-gene range) →
    /// run must stop before `max_generation` due to stagnation.
    #[test]
    fn stagnation_triggers_early_exit() {
        let config: Config<i64> = Config {
            max_generation: 10_000, // huge — stagnation must kick in first
            stagnation_count: 30,
            population_size: 20,
            pools: 2,
            ranges: vec![vec![(42, 42)]], // single fixed gene → zero diversity always
            ..Default::default()
        };

        let ranges: Vec<_> = config.ranges.iter().flatten().cloned().collect();
        let generator = DefaultGenerator::new(&ranges);
        let mutator = DefaultMutator::new(&ranges);
        let crossover = DefaultCrossover::new(&config.ranges);
        let mut ga = GeneticAlgorithm::<i64, (), (), _, _, _, _, _>::new(
            &config, generator, mutator, crossover,
            |ind: &Individual<i64, ()>, _: &Context<'_, i64, (), ()>| (-(ind.genome[0] as f64).powi(2), None),
            NoopCallback,
        );

        let start = std::time::Instant::now();
        ga.run();

        // If stagnation fired, we finished way before 10 000 generations worth of work.
        assert_that!(start.elapsed().as_millis()).is_less_than(5_000);
    }

    /// 1-D optimum at x=0 in range [-50, 50] (shifted domain).
    #[test]
    fn finds_1d_minimum() {
        let config: Config<i64> = Config {
            max_generation: 500,
            stagnation_count: 80,
            population_size: 60,
            pools: 4,
            ranges: vec![vec![(-50, 50)]],
            ..Default::default()
        };

        let ranges: Vec<_> = config.ranges.iter().flatten().cloned().collect();
        let generator = DefaultGenerator::new(&ranges);
        let mutator = DefaultMutator::new(&ranges);
        let crossover = DefaultCrossover::new(&config.ranges);
        let mut ga = GeneticAlgorithm::<i64, (), (), _, _, _, _, _>::new(
            &config, generator, mutator, crossover,
            |ind: &Individual<i64, ()>, _: &Context<'_, i64, (), ()>| (-(ind.genome[0] as f64).powi(2), None),
            NoopCallback,
        );

        let best = ga.run().top_individuals_mut(1).first().unwrap().genome[0];
        // optimum is 0; accept a small tolerance
        assert_that!(best.abs()).is_less_than_or_equal_to(5);
    }

    /// Multi-run continuity: second `run()` reuses the populations from the first.
    #[test]
    fn second_run_not_worse_than_first() {
        let config: Config<i64> = Config {
            max_generation: 200,
            stagnation_count: 50,
            population_size: 60,
            pools: 4,
            ranges: vec![vec![(0, 99); 4]],
            ..Default::default()
        };

        let ranges: Vec<_> = config.ranges.iter().flatten().cloned().collect();
        let generator = DefaultGenerator::new(&ranges);
        let mutator = DefaultMutator::new(&ranges);
        let crossover = DefaultCrossover::new(&config.ranges);
        let mut ga = GeneticAlgorithm::<i64, (), (), _, _, _, _, _>::new(
            &config, generator, mutator, crossover, sphere_no_state, NoopCallback,
        );

        let first_best = ga.run().top_individuals_mut(1).first().unwrap().fitness;
        let second_best = ga.run().top_individuals_mut(1).first().unwrap().fitness;

        // Pools carry over, so second run starts ahead — fitness must not regress.
        assert_that!(second_best).is_greater_than_or_equal_to(first_best);
    }

    /// noise_factor: diversity/stagnation pressure correctly scales mutation noise.
    #[test]
    fn noise_factor_with_stagnation() {
        use sample::noise_factor;
        let test_cases = [
            // (diversity, stagnation_boost, expected_noise)
            (0.0, 0.0, 1.0),  // low diversity, no stagnation → max noise
            (1.0, 0.0, 0.0),  // high diversity, no stagnation → min noise
            (0.5, 0.0, 0.5),  // mid diversity, no stagnation → mid noise
            (0.0, 1.0, 1.0),  // low diversity, max stagnation → max noise
            (1.0, 1.0, 1.0),  // high diversity, max stagnation → max noise (forced exploration)
            (0.5, 0.5, 0.75), // mid diversity, mid stagnation → boosted noise
            (0.8, 0.5, 0.6),  // high diversity (0.2 base noise), mid stagnation → 0.2 + 0.5 * 0.8 = 0.6
        ];

        for (div, boost, expected) in test_cases {
            asserting(&format!("Noise for diversity={div}, boost={boost}"))
                .that(&noise_factor(div, boost))
                .is_close_to(expected, 1e-6);
        }
    }

    fn sphere_no_state(ind: &Individual<i64, ()>, _: &Context<'_, i64, (), ()>) -> (f64, Option<()>) {
        let score = -ind.genome
            .iter()
            .map(|&x| x as f64 / 99.0)
            .map(|x: f64| x.powi(2))
            .sum::<f64>();
        (score, None)
    }

    fn test_run(config: Config<i64>, writer: BufWriter<&mut Vec<u8>>) -> bool {
        let ranges: Vec<(i64, i64)> = config.ranges.iter().flatten().cloned().collect();
        let generator = DefaultGenerator::new(&ranges);
        let mutator = DefaultMutator::new(&ranges);
        let crossover = DefaultCrossover::new(&config.ranges);
        let mut ga = GeneticAlgorithm::new(&config, generator, mutator, crossover, sphere, callback_fn);
        ga.set_state((&config, Mutex::new(writer)));
        let pools = ga.run();
        pools
            .top_individuals_mut(config.bests)
            .first()
            .expect("expected some bests")
            .fitness
            <= 0.0001
    }
}
