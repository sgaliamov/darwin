use crate::{Genome, Individual, Lineage, ScoreFn};

impl<IndState> Individual<IndState> {
    /// Name without genome.
    pub fn name(&self) -> String {
        format!("ω{:.6} | {:<8}", self.fitness, self.lineage.to_string())
    }

    /// Main constructor.
    pub fn new(genome: Genome, lineage: Lineage) -> Self {
        Self {
            lineage,
            genome,
            fitness: f64::NAN, // to exclude from all comparisons
            state: None,
        }
    }

    pub fn firstborn(generation: usize, genome: Genome) -> Self {
        Self::new(genome, Lineage::Firstborn(generation))
    }

    /// Compute and cache fitness. Idempotent.
    pub fn evaluate<GaState>(
        &mut self,
        score: &ScoreFn<GaState, IndState>,
        state: &Option<GaState>,
    ) {
        if !self.fitness.is_finite() {
            (self.fitness, self.state) = score(&self.genome, state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral::prelude::*;

    fn const_score(genome: &[i64], _: &Option<()>) -> (f64, Option<()>) {
        (genome.len() as f64, None)
    }

    /// Freshly constructed individual has NaN fitness.
    #[test]
    fn new_individual_has_nan_fitness() {
        let ind = Individual::<()>::firstborn(0, vec![1, 2, 3]);
        assert_that!(ind.fitness.is_nan()).is_true();
    }

    /// `evaluate` sets fitness on first call.
    #[test]
    fn evaluate_sets_fitness() {
        let score: ScoreFn<(), ()> = const_score;
        let mut ind = Individual::<()>::firstborn(0, vec![1, 2, 3]);
        ind.evaluate(&score, &None);
        assert_that!(ind.fitness).is_equal_to(3.0);
    }

    /// `evaluate` is idempotent — second call does not re-score.
    #[test]
    fn evaluate_is_idempotent() {
        let score: ScoreFn<(), ()> = const_score;
        let mut ind = Individual::<()>::firstborn(0, vec![1, 2, 3]);
        ind.evaluate(&score, &None);
        // Override the genome to verify score is NOT recomputed.
        ind.genome = vec![9, 9, 9, 9, 9];
        ind.evaluate(&score, &None);
        assert_that!(ind.fitness).is_equal_to(3.0); // still original
    }

    /// `firstborn` wraps genome in Firstborn lineage at given generation.
    #[test]
    fn firstborn_lineage_matches_generation() {
        let ind = Individual::<()>::firstborn(5, vec![0]);
        assert!(matches!(ind.lineage, Lineage::Firstborn(5)));
    }
}
