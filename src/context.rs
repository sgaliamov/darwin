use crate::{GenInfo, Gene, Pools};
use std::marker::PhantomData;
use std::ops::Deref;

/// Context passed to all GA operators on each call.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a, G: Gene, GaState, IndState> {
    info: &'a GenInfo,
    pub state: &'a Option<GaState>,    // external GA state
    pub pools: &'a Pools<G, IndState>, // all pools for cross-pool inspection
    pub __: PhantomData<IndState>,
}

impl<'a, G: Gene, GaState, IndState> Context<'a, G, GaState, IndState> {
    pub fn new(
        info: &'a GenInfo,
        state: &'a Option<GaState>,
        pools: &'a Pools<G, IndState>,
    ) -> Self {
        Self {
            info,
            state,
            pools,
            __: PhantomData,
        }
    }
}

impl<G: Gene, GaState, IndState> Deref for Context<'_, G, GaState, IndState> {
    type Target = GenInfo;
    fn deref(&self) -> &GenInfo {
        self.info
    }
}
