use crate::{Epoch, Gene, Pools};
use std::marker::PhantomData;
use std::ops::Deref;

/// Context passed to all GA operators on each call.
/// Carries GA-level signals an operator may use to tune its behavior.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a, G: Gene, GaState, IndState> {
    epoch: Epoch,

    /// External GA state shared with all operators.
    pub state: &'a Option<GaState>,

    /// All pools; allows cross-pool inspection (diversity, top individuals, etc.).
    /// Use `ctx.pools.best()` to get the current best genome and fitness.
    pub pools: &'a Pools<G, IndState>,

    pub __: PhantomData<IndState>,
}

impl<'a, G: Gene, GaState, IndState> Context<'a, G, GaState, IndState> {
    pub fn new(epoch: Epoch, state: &'a Option<GaState>, pools: &'a Pools<G, IndState>) -> Self {
        Self { epoch, state, pools, __: PhantomData }
    }
}

impl<G: Gene, GaState, IndState> Deref for Context<'_, G, GaState, IndState> {
    type Target = Epoch;
    fn deref(&self) -> &Epoch { &self.epoch }
}
