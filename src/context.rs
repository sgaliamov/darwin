use crate::{Epoch, Gene, Pools};
use std::marker::PhantomData;

/// Context passed to all GA operators on each call.
/// Carries GA-level signals an operator may use to tune its behavior.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a, G: Gene, GaState, IndState> {
    /// Per-generation epoch: generation index, stagnation pressure, and Gaussian N(0,σ).
    pub epoch: Epoch,

    /// External GA state shared with all operators.
    pub state: &'a Option<GaState>,

    /// All pools; allows cross-pool inspection (diversity, top individuals, etc.).
    /// Use `ctx.pools.best()` to get the current best genome and fitness.
    pub pools: &'a Pools<G, IndState>,

    pub __: PhantomData<IndState>,
}
