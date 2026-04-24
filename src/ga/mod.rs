mod config;
mod context;
mod evolution;
mod genome;
mod methods;
mod pool;
mod pools;
mod sigma;

pub use config::*;
pub use context::*;
pub use evolution::*;
pub use genome::*;
pub use pool::*;
pub use pools::*;
pub use sigma::*;

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

/// Static score calculation function.
pub type ScoreFn<GaState, IndState> = fn(GenomeRef, &Option<GaState>) -> (f64, Option<IndState>);

/// Static callback.
pub type CallbackFn<GaState, IndState> =
    fn(usize, &Option<(Genome, f64)>, &Pools<IndState>, &mut Option<GaState>);

/// Evolution engine with independently injectable genome operations.
pub struct GeneticAlgorithm<'a, GaState, IndState, G, M, C>
where
    G: Generator,
    M: Mutator<GaState>,
    C: Crossover<GaState>,
{
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

    /// Random genome generator.
    generator: G,

    /// Mutation operator.
    mutator: M,

    /// Crossover operator.
    crossover: C,

    /// Flat genome ranges
    flat_genome: GeneRanges,

    // --- Cached scalars for quick access ------------------------------------
    mutant_count: usize,
    immigrant_count: usize,
    crossover_size: usize,
    stagnation_counter: usize,
}
