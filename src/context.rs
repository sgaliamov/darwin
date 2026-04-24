use crate::Gene;

/// Context passed to [`super::Mutator`] and [`super::Crossover`] on each call.
/// Carries GA-level signals an evolver may use to tune its behavior.
/// All pressure values are normalized to `[0.0, 1.0]`.
#[derive(Debug)]
pub struct Context<'a, G: Gene, GaState> {
    /// Current generation number.
    pub generation: usize,

    /// Pool gene diversity: `0.0` = fully converged, `1.0` = maximally diverse.
    pub diversity: f32,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,

    /// GA configuration; gives evolvers access to e.g. `max_generation`.
    pub config: &'a crate::Config<G>,

    /// External GA state shared with the score / callback functions.
    pub state: &'a Option<GaState>,
}

impl<G: Gene, GaState> Copy for Context<'_, G, GaState> {}

impl<G: Gene, GaState> Clone for Context<'_, G, GaState> {
    fn clone(&self) -> Self {
        *self
    }
}
