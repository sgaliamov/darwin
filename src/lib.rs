//! Domain agnostic genetic algorithm implementation.
mod config;
mod context;
mod gen_info;
mod genome;
mod individual;
mod methods;
mod pool;
mod pools;
mod sigma;
mod traits;

pub use config::*;
pub use context::*;
pub use gen_info::*;
pub use genome::*;
pub use individual::*;
pub use pool::*;
pub use pools::*;
pub use sigma::*;
pub use traits::*;

/// Evolution engine with independently injectable genome operations.
pub struct GeneticAlgorithm<'a, G, GaState, IndState, Gr, M, C, E, Cb>
where
    G: Gene,
    Gr: Generator<G, GaState, IndState>,
    M: Mutator<G, GaState, IndState>,
    C: Crossover<G, GaState, IndState>,
    E: Evaluator<G, GaState, IndState>,
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

    /// External state.
    state: Option<GaState>,

    /// Evaluates genomes into fitness + state.
    evaluator: E,

    /// Callback to report progress outside each generation.
    callback: Cb,

    /// Random genome generator.
    generator: Gr,

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
