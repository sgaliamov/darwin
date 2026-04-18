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
