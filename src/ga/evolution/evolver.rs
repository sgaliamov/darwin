use super::super::genome::{Genome, GenomeRef};
use super::context::Context;

/// Trait for pluggable genome operation strategies.
///
/// Implementations must be `Send + Sync` so a single instance can be shared
/// across Rayon threads without cloning or locking.
pub trait Evolver<GaState>: Send + Sync {
    /// Generate a fully random genome.
    fn generate(&self) -> Genome;

    /// Return a mutated copy of `genome`, or `None` if the mutant falls outside range.
    fn mutant(&self, genome: GenomeRef, ctx: &Context<'_, GaState>) -> Option<Genome>;

    /// Produce offspring by crossing two parent genomes.
    /// Returns one or more child genomes.
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, ctx: &Context<'_, GaState>) -> Vec<Genome>;
}
