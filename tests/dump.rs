//! Abort-dump and resume behavior.
mod sample;

#[cfg(test)]
mod tests {
    use super::sample::{DefaultGenerator, DefaultMutator};
    use darwin::{
        Config, Context, GeneticAlgorithm, Individual, NoopCallback, NoopCrossover, load_dump,
        save_dump,
    };
    use spectral::prelude::*;
    use std::path::{Path, PathBuf};

    /// Unique temp path per test; removed on drop.
    struct TempDump(PathBuf);

    impl TempDump {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("darwin-dump-{name}-{}", std::process::id())))
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDump {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Fitness = -∑x²; optimum at origin.
    fn sphere(ind: &Individual<i64, ()>, _: &Context<'_, i64, (), ()>) -> (f64, Option<()>) {
        (
            -ind.genome.iter().map(|&x| (x as f64).powi(2)).sum::<f64>(),
            Some(()),
        )
    }

    /// Evaluator fn-pointer type for the test GA.
    type Eval = fn(&Individual<i64, ()>, &Context<'_, i64, (), ()>) -> (f64, Option<()>);

    /// Test GA over `i64` genomes with a pluggable callback.
    type TestGa<Cb> = GeneticAlgorithm<
        i64,
        (),
        (),
        DefaultGenerator<i64>,
        DefaultMutator<i64>,
        NoopCrossover,
        Eval,
        Cb,
    >;

    /// Build a GA over `[0, 99]^4` with the given config tweaks applied.
    fn build_ga<Cb>(config: Config<i64>, callback: Cb) -> TestGa<Cb>
    where
        Cb: darwin::Callback<i64, (), ()>,
    {
        let ranges: Vec<_> = config.ranges.iter().flatten().cloned().collect();
        GeneticAlgorithm::new(
            config,
            DefaultGenerator::new(&ranges),
            DefaultMutator::new(&ranges),
            NoopCrossover,
            sphere,
            callback,
        )
    }

    fn config(dump: &TempDump) -> Config<i64> {
        Config {
            max_generation: 50,
            stagnation_count: 100,
            population_size: 20,
            pools: 3,
            ranges: vec![vec![(0, 99); 4]],
            dump: Some(dump.path().to_path_buf()),
            dump_ratio: 0.1,
            ..Default::default()
        }
    }

    /// Abort via callback → dump file written with top genomes.
    #[test]
    fn abort_writes_dump() {
        let dump = TempDump::new("abort");
        let mut ga = build_ga(config(&dump), |ctx: &Context<'_, i64, (), ()>| {
            ctx.generation < 3
        });

        ga.run();

        let genomes = load_dump::<i64>(dump.path(), 4).expect("dump must exist and decode");
        // 3 pools × ceil(20 × 0.1) = 6 genomes.
        assert_that!(genomes.len()).is_equal_to(6);
        assert_that!(genomes.iter().all(|g| g.len() == 4)).is_true();
    }

    /// Natural finish removes a stale dump file.
    #[test]
    fn natural_finish_removes_dump() {
        let dump = TempDump::new("cleanup");
        save_dump(dump.path(), &[vec![1i64, 2, 3, 4]]).unwrap();

        let mut cfg = config(&dump);
        cfg.max_generation = 2;
        // seed empty → seed() not called → dump untouched until run() finishes.
        let mut ga = build_ga(cfg, NoopCallback);
        ga.run();

        assert_that!(dump.path().exists()).is_false();
    }

    /// Existing dump wins over `config.seed` and lands in pools as-is.
    #[test]
    fn seed_resumes_from_dump() {
        let dump = TempDump::new("resume");
        let dumped = vec![vec![7i64, 7, 7, 7], vec![3, 3, 3, 3]];
        save_dump(dump.path(), &dumped).unwrap();

        let mut cfg = config(&dump);
        cfg.seed = vec![vec![50, 50, 50, 50]];
        cfg.seed_mutation = 5;
        let mut ga = build_ga(cfg, NoopCallback);
        ga.seed();

        let seeded: Vec<_> = ga
            .pools()
            .iter()
            .flat_map(|p| p.individuals.iter().map(|ind| ind.genome.clone()))
            .collect();

        assert_that!(seeded.len()).is_equal_to(2);
        assert_that!(seeded.contains(&dumped[0])).is_true();
        assert_that!(seeded.contains(&dumped[1])).is_true();
        assert_that!(seeded.contains(&vec![50, 50, 50, 50])).is_false();
    }

    /// Genome length mismatch → dump ignored, config.seed used.
    #[test]
    fn stale_dump_falls_back_to_seed() {
        let dump = TempDump::new("stale");
        save_dump(dump.path(), &[vec![1i64, 2]]).unwrap(); // len 2 ≠ 4

        let mut cfg = config(&dump);
        cfg.seed = vec![vec![50, 50, 50, 50]];
        let mut ga = build_ga(cfg, NoopCallback);
        ga.seed();

        let seeded: Vec<_> = ga
            .pools()
            .iter()
            .flat_map(|p| p.individuals.iter().map(|ind| ind.genome.clone()))
            .collect();

        assert_that!(seeded).is_equal_to(vec![vec![50i64, 50, 50, 50]]);
    }

    /// Dump round-trips through binary save/load.
    #[test]
    fn dump_roundtrip() {
        let dump = TempDump::new("roundtrip");
        let genomes = vec![vec![0i64, 1, 2], vec![99, 98, 97]];
        save_dump(dump.path(), &genomes).unwrap();

        assert_that!(load_dump::<i64>(dump.path(), 3)).is_equal_to(Some(genomes));
    }

    /// No dump configured → abort writes nothing.
    #[test]
    fn no_dump_path_writes_nothing() {
        let dump = TempDump::new("disabled");
        let mut cfg = config(&dump);
        cfg.dump = None;
        let mut ga = build_ga(cfg, |ctx: &Context<'_, i64, (), ()>| ctx.generation < 2);

        ga.run();

        assert_that!(dump.path().exists()).is_false();
    }
}
