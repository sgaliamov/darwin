use crate::{Gene, Pools};
use rand_distr::Normal;
use std::marker::PhantomData;

/// Context passed to all GA operators on each call.
/// Carries GA-level signals an operator may use to tune its behavior.
/// All pressure values are normalized to `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a, G: Gene, GaState, IndState> {
    /// Current generation number.
    pub generation: usize,

    /// Pool gene diversity: `0.0` = fully converged, `1.0` = maximally diverse.
    pub diversity: f32,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,

    /// Gaussian N(0, σ) for this generation; built once, reused by all operators.
    pub normal: Normal<f32>,

    /// GA configuration; gives operators access to e.g. `max_generation`.
    pub config: &'a crate::Config<G>,

    /// External GA state shared with all operators.
    pub state: &'a Option<GaState>,

    /// All pools; allows cross-pool inspection (diversity, top individuals, etc.).
    /// Use `ctx.pools.best()` to get the current best genome and fitness.
    pub pools: &'a Pools<G, IndState>,

    pub __: PhantomData<IndState>,
}
