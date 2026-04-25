mod crossover;
mod generator;
mod mutator;
use darwin::{Context, Gene, GeneRangesRef, Genome, GenomeRef, Sigma};
pub use crossover::*;
pub use generator::*;
pub use mutator::*;
use rand_distr::{Distribution, Normal};
use std::ops::Add;

/// Noise scaling factor derived from GA pressure signals.
/// High diversity → exploit (low noise); high stagnation → explore (high noise).
pub fn noise_factor(diversity: f32, stagnation: f32) -> f32 {
    let base = 1.0 - diversity;
    base + stagnation * (1.0 - base)
}

/// Applies Gaussian noise to `genome` using sigma and noise derived from `ctx`.
/// Returns `None` if any mutated gene falls outside its declared range.
pub(super) fn noisy_mutant<G, GaState, IndState>(
    ranges: GeneRangesRef<G>,
    sigma_cfg: &Sigma,
    genome: GenomeRef<G>,
    ctx: &Context<'_, G, GaState, IndState>,
    rng: &mut impl rand::Rng,
) -> Option<Genome<G>>
where
    G: Gene + Add<Output = G> + TryFrom<i64>,
    GaState: Sync,
{
    let sigma = sigma_cfg.get(ctx.generation, ctx.config.max_generation);
    let noise = noise_factor(ctx.diversity, ctx.stagnation);
    // μ = 0 so shifts are symmetric around the original value.
    let normal = Normal::new(0.0_f32, sigma).expect("`sigma` should be valid.");

    genome
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let range = ranges.get(i)?;

            if range.0 == range.1 {
                return Some(*g);
            }

            let shift_i64 = (normal.sample(rng) * noise).round() as i64;

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
