mod crossover;
mod generator;
mod mutator;

use crate::{Context, Gene, GeneRangesRef, Genome, GenomeRef, Sigma};
pub use crossover::*;
pub use generator::*;
pub use mutator::*;
use rand_distr::{Distribution, Normal};

/// Generates random genomes; must be `Send + Sync` for Rayon sharing.
pub trait Generator: Send + Sync {
    /// Produce a fully random genome within declared ranges.
    fn generate(&self) -> Genome;
}

/// Produces mutated copies of a genome; must be `Send + Sync` for Rayon sharing.
pub trait Mutator<GaState>: Send + Sync {
    /// Return a mutated copy of `genome`, or `None` if the result falls outside range.
    fn mutant(&self, genome: GenomeRef, ctx: &Context<'_, GaState>) -> Option<Genome>;
}

/// Produces offspring from two parent genomes; must be `Send + Sync` for Rayon sharing.
pub trait Crossover<GaState>: Send + Sync {
    /// Cross `dad` and `mom`, returning one or more child genomes.
    fn cross(&self, dad: GenomeRef, mom: GenomeRef, ctx: &Context<'_, GaState>) -> Vec<Genome>;
}

/// Applies Gaussian noise to `genome`.
/// Returns `None` if any mutated gene falls outside its declared range.
fn mutant_with_noise<GaState>(
    ranges: GeneRangesRef,
    config: &Sigma,
    genome: GenomeRef,
    ctx: &Context<'_, GaState>,
    noise_factor: f32,
    rng: &mut impl rand::Rng,
) -> Option<Genome> {
    // μ = 0 so shifts are symmetric around the original value.
    let normal = Normal::new(0.0_f32, config.get(ctx)).expect("`sigma` should be valid.");
    genome
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let range = ranges.get(i)?;
            if range.0 == range.1 {
                return Some(*g);
            }
            let shift = (normal.sample(rng) * noise_factor).round() as Gene;
            let new = g + shift;
            if new < range.0 || new > range.1 {
                None
            } else {
                Some(new)
            }
        })
        .collect()
}
