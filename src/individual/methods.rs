use crate::{Context, Gene, Genome, Individual, Lineage, Scorer};

impl<G, IndState> Individual<G, IndState> {
    /// Name without genome.
    pub fn name(&self) -> String {
        format!("ω{:.6} | {:<8}", self.fitness, self.lineage.to_string())
    }

    /// Main constructor.
    pub fn new(genome: Genome<G>, lineage: Lineage) -> Self {
        Self {
            lineage,
            genome,
            fitness: f64::NAN, // to exclude from all comparisons
            state: None,
        }
    }

    pub fn firstborn(pool: usize, generation: usize, genome: Genome<G>) -> Self {
        Self::new(genome, Lineage::Firstborn(pool, generation))
    }
}

impl<G: Gene, IndState> Individual<G, IndState> {
    /// Compute and cache fitness. Idempotent.
    pub fn evaluate<GaState, Sc>(&mut self, scorer: &Sc, ctx: &Context<'_, G, GaState, IndState>)
    where
        Sc: Scorer<G, GaState, IndState>,
    {
        if !self.fitness.is_finite() {
            (self.fitness, self.state) = scorer.score(&self.genome, ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Pools};
    use spectral::prelude::*;

    fn const_score(genome: &[i64], _: &Context<'_, i64, (), ()>) -> (f64, Option<()>) {
        (genome.len() as f64, None)
    }

    /// Freshly constructed individual has NaN fitness.
    #[test]
    fn new_individual_has_nan_fitness() {
        let ind = Individual::<i64, ()>::firstborn(0, 0, vec![1, 2, 3]);
        assert_that!(ind.fitness.is_nan()).is_true();
    }

    /// `evaluate` sets fitness on first call.
    #[test]
    fn evaluate_sets_fitness() {
        let cfg = Config::<i64>::default();
        let pools = Pools::<i64, ()>::default();
        let ctx = Context::<i64, (), ()> { generation: 0, diversity: 0.0, stagnation: 0.0, sigma: cfg.sigma.get(0, cfg.max_generation), config: &cfg, state: &None, pools: &pools, __: std::marker::PhantomData };
        let mut ind = Individual::<i64, ()>::firstborn(0, 0, vec![1, 2, 3]);
        ind.evaluate(&const_score, &ctx);
        assert_that!(ind.fitness).is_equal_to(3.0);
    }

    /// `evaluate` is idempotent — second call does not re-score.
    #[test]
    fn evaluate_is_idempotent() {
        let cfg = Config::<i64>::default();
        let pools = Pools::<i64, ()>::default();
        let ctx = Context::<i64, (), ()> { generation: 0, diversity: 0.0, stagnation: 0.0, sigma: cfg.sigma.get(0, cfg.max_generation), config: &cfg, state: &None, pools: &pools, __: std::marker::PhantomData };
        let mut ind = Individual::<i64, ()>::firstborn(0, 0, vec![1, 2, 3]);
        ind.evaluate(&const_score, &ctx);
        // Override the genome to verify score is NOT recomputed.
        ind.genome = vec![9, 9, 9, 9, 9];
        ind.evaluate(&const_score, &ctx);
        assert_that!(ind.fitness).is_equal_to(3.0); // still original
    }

    /// `firstborn` wraps genome in Firstborn lineage at given generation.
    #[test]
    fn firstborn_lineage_matches_generation() {
        let ind = Individual::<i64, ()>::firstborn(0, 5, vec![0]);
        assert!(matches!(ind.lineage, Lineage::Firstborn(_, 5)));
    }
}
