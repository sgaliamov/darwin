mod crossover;
mod generator;
mod mutator;
pub use crossover::*;
use darwin::{Context, Gene, GeneRangesRef, Genome, GenomeRef};
pub use generator::*;
pub use mutator::*;
use rand_distr::Distribution;
use std::ops::Add;

/// Noise scaling factor derived from GA pressure signals.
/// High diversity → exploit (low noise); high stagnation → explore (high noise).
pub fn noise_factor(diversity: f32, stagnation: f32) -> f32 {
    let base = 1.0 - diversity;
    base + stagnation * (1.0 - base)
}

/// Applies Gaussian noise to `genome` using sigma and noise derived from `ctx`.
/// Returns `None` if any mutated gene falls outside its declared range.
fn noisy_mutant<G, GaState, IndState>(
    flat_ranges: GeneRangesRef<G>,
    flat_genome: GenomeRef<G>,
    ctx: &Context<'_, G, GaState, IndState>,
    rng: &mut impl rand::Rng,
) -> Option<Genome<G>>
where
    G: Gene + Add<Output = G> + TryFrom<i64>,
    GaState: Sync,
{
    let noise = noise_factor(ctx.diversity, ctx.stagnation);

    flat_genome
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let range = flat_ranges.get(i)?;

            if range.0 == range.1 {
                return Some(*g);
            }

            let shift_i64 = (ctx.normal.sample(rng) * noise).round() as i64;

            // Attempts to convert `shift_i64` into the generic type `G`.
            // The conversion may fail if the value of `shift_i64` is out of range
            // or incompatible with the target type `G`, in which case `None` is returned.
            // Returns `None` if the conversion fails.
            let Ok(shift) = G::try_from(shift_i64) else {
                return None;
            };

            let new = *g + shift;

            if new < range.0 || new > range.1 {
                None
            } else {
                Some(new)
            }
        })
        .collect()
}
