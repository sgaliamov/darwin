//! Domain agnostic genetic algorithm implementation.
mod config;
mod context;
mod genome;
mod individual;
mod methods;
mod pool;
mod pools;
mod sigma;
mod traits;

pub use config::*;
pub use context::*;
pub use genome::*;
pub use individual::*;
pub use pool::*;
pub use pools::*;
pub use sigma::*;
pub use traits::*;

/// Evolution engine with independently injectable genome operations.
pub struct GeneticAlgorithm<'a, G, GaState, IndState, Gen, M, C, Sc, Cb>
where
    G: Gene,
    Gen: Generator<G, GaState, IndState>,
    M: Mutator<G, GaState, IndState>,
    C: Crossover<G, GaState, IndState>,
    Sc: Scorer<G, GaState, IndState>,
    Cb: Callback<G, GaState, IndState>,
{
    /// GA configuration.
    config: &'a Config<G>,

    /// All individuals across multiple pools.
    /// Pools are kept internally to allow reusing evolutionary state between runs—
    /// critical for scenarios like sliding-window evolution where populations carry forward.
    pools: Pools<G, IndState>,

    /// Best fitness seen so far; used for stagnation detection.
    best_fitness: f64,

    /// External state for [`GeneticAlgorithm::score_fn`] and [`GeneticAlgorithm::callback_fn`].
    state: Option<GaState>,

    /// Fitness function.
    scorer: Sc,

    /// Callback to report progress outside each generation.
    callback: Cb,

    /// Random genome generator.
    generator: Gen,

    /// Mutation operator.
    mutator: M,

    /// Crossover operator.
    crossover: C,

    /// Flat genome ranges.
    flat_genome: GeneRanges<G>,

    // --- Cached scalars for quick access ------------------------------------
    mutant_count: usize,
    immigrant_count: usize,
    crossover_size: usize,
    stagnation_counter: usize,
}
