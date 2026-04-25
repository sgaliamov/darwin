mod lineage;
mod methods;
mod traits;

use crate::Genome;
pub use lineage::*;

/// Individual – holds DNA and its cached fitness value.
/// Cannot be cloneable to not force `State` be cloneable as well.
#[derive(Debug)]
pub struct Individual<G, State> {
    pub lineage: Lineage,

    /// Flat DNA.
    pub genome: Genome<G>,

    /// Cached fitness.
    pub fitness: f64,

    /// State returned by scorer. Can be used for memoization or to pass info from scorer to callback.
    pub state: Option<State>,
}
