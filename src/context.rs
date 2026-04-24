use crate::{Gene, Genome, Pools};
use std::marker::PhantomData;

/// Context passed to all GA operators on each call.
/// Carries GA-level signals an operator may use to tune its behavior.
/// All pressure values are normalized to `[0.0, 1.0]`.
#[derive(Debug)]
pub struct Context<'a, G: Gene, GaState, IndState> {
    /// Current generation number.
    pub generation: usize,

    /// Pool gene diversity: `0.0` = fully converged, `1.0` = maximally diverse.
    pub diversity: f32,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,

    /// GA configuration; gives operators access to e.g. `max_generation`.
    pub config: &'a crate::Config<G>,

    /// External GA state shared with all operators.
    pub state: &'a Option<GaState>,

    /// Global best genome and fitness seen so far.
    pub best: &'a Option<(Genome<G>, f64)>,

    /// All pools; allows cross-pool inspection (diversity, top individuals, etc.).
    pub pools: &'a Pools<G, IndState>,

    pub __: PhantomData<IndState>,
}

impl<G: Gene, GaState, IndState> Copy for Context<'_, G, GaState, IndState> {}

impl<G: Gene, GaState, IndState> Clone for Context<'_, G, GaState, IndState> {
    fn clone(&self) -> Self {
        *self
    }
}
