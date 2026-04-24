//! Domain agnostic genetic algorithm implementation.
mod config;
mod context;
mod genome;
mod methods;
mod pool;
mod pools;
mod sigma;
mod individual;

pub use config::*;
pub use context::*;
pub use genome::*;
pub use pool::*;
pub use pools::*;
pub use sigma::*;
pub use individual::*;

/// Generates random genomes; must be `Send + Sync` for Rayon sharing.
pub trait Generator<G: Gene>: Send + Sync {
    /// Produce a fully random genome within declared ranges.
    fn generate(&self) -> Genome<G>;
}

/// Produces mutated copies of a genome; must be `Send + Sync` for Rayon sharing.
pub trait Mutator<G: Gene, GaState>: Send + Sync {
    /// Return a mutated copy of `genome`, or `None` if the result falls outside range.
    fn mutant(&self, genome: GenomeRef<G>, ctx: &Context<'_, G, GaState>) -> Option<Genome<G>>;
}

/// Produces offspring from two parent genomes; must be `Send + Sync` for Rayon sharing.
pub trait Crossover<G: Gene, GaState>: Send + Sync {
    /// Cross `dad` and `mom`, returning one or more child genomes.
    fn cross(
        &self,
        dad: GenomeRef<G>,
        mom: GenomeRef<G>,
        ctx: &Context<'_, G, GaState>,
    ) -> Vec<Genome<G>>;
}

/// Static score calculation function.
pub type ScoreFn<G, GaState, IndState> =
    fn(GenomeRef<G>, &Option<GaState>) -> (f64, Option<IndState>);

/// Static callback.
pub type CallbackFn<G, GaState, IndState> =
    fn(usize, &Option<(Genome<G>, f64)>, &Pools<G, IndState>, &mut Option<GaState>);

/// Evolution engine with independently injectable genome operations.
pub struct GeneticAlgorithm<'a, G, GaState, IndState, Gen, M, C>
where
    G: Gene,
    Gen: Generator<G>,
    M: Mutator<G, GaState>,
    C: Crossover<G, GaState>,
{
    /// GA configuration.
    config: &'a Config<G>,

    /// All individuals across multiple pools.
    /// Pools are kept internally to allow reusing evolutionary state between runs—
    /// critical for scenarios like sliding-window evolution where populations carry forward.
    pools: Pools<G, IndState>,

    /// Keep genome and score only to not copy the whole [`crate::Individual`].
    // tbd: [ga] pass pool and number of the best individual,
    //      then we will be able to fetch more details about it, not only genome+fitness,
    //      and avoid cloning.
    best: Option<(Genome<G>, f64)>,

    /// External state for [`GeneticAlgorithm::score_fn`] and [`GeneticAlgorithm::callback_fn`].
    state: Option<GaState>,

    /// Fitness function.
    score_fn: ScoreFn<G, GaState, IndState>,

    /// Callback to report progress outside each generation.
    callback_fn: CallbackFn<G, GaState, IndState>,

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
