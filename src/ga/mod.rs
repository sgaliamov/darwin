mod config;
mod evolution;
mod genome;
mod methods;
mod pool;
mod pools;

pub use evolution::context::*;
pub use evolution::default_evolution::*;
pub use evolution::evolver::*;
pub use evolution::evolution_config::*;

/// Static callback.
pub type CallbackFn<GaState, IndState> =
    fn(usize, &Option<(Genome, f64)>, &Pools<IndState>, &mut Option<GaState>);

//  Evolution engine
pub struct GeneticAlgorithm<'a, GaState, IndState, E: Evolver> {
    /// Flat genome ranges
    ranges: GeneRanges,

    /// GA configuration.
    config: &'a Config,

    /// All individuals across multiple pools.
    /// Pools are kept internally to allow reusing evolutionary state between runs—
    /// critical for scenarios like sliding-window evolution where populations carry forward.
    pools: Pools<IndState>,

    /// Keep genome and score only to not copy the whole [`crate::Individual`].
    // tbd: [ga] pass pool and number of the best individual,
    //      then we will be able to fetch more details about it, not only genome+fitness,
    //      and avoid cloning.
    best: Option<(Genome, f64)>,

    /// External state for [`GeneticAlgorithm::score_fn`] and [`GeneticAlgorithm::callback_fn`].
    state: Option<GaState>,

    /// Fitness function.
    score_fn: ScoreFn<GaState, IndState>,

    /// Callback to report progress outside each generation.
    callback_fn: CallbackFn<GaState, IndState>,

    /// Injected evolution strategy.
    evolver: E,

    // --- Cached scalars for quick access ------------------------------------
    //
    mutant_count: usize,
    immigrant_count: usize,
    crossover_size: usize,
    stagnation_counter: usize,
}
