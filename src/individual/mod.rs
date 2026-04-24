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
    // tbd: [ga] group genomes like in the config.
    pub genome: Genome<G>,
    pub fitness: f64,
    pub state: Option<State>,
}
