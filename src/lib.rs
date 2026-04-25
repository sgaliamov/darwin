//! Domain agnostic genetic algorithm implementation.
mod config;
mod context;
mod genome;
mod individual;
mod methods;
mod pool;
mod pools;
mod sigma;

pub use config::*;
pub use context::*;
pub use genome::*;
pub use individual::*;
pub use pool::*;
pub use pools::*;
pub use sigma::*;

/// Generates random genomes; must be `Send + Sync` for Rayon sharing.
pub trait Generator<G: Gene, GaState, IndState>: Send + Sync {
    /// Produce a fully random genome within declared ranges.
    fn generate(&self, ctx: &Context<'_, G, GaState, IndState>) -> Genome<G>;
}

/// Produces mutated copies of a genome; must be `Send + Sync` for Rayon sharing.
pub trait Mutator<G: Gene, GaState, IndState>: Send + Sync {
    /// Return a mutated copy of `individual.genome`, or `None` if the result falls outside range.
    fn mutant(
        &self,
        individual: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> Option<Genome<G>>;
}

/// Produces offspring from two parent genomes; must be `Send + Sync` for Rayon sharing.
pub trait Crossover<G: Gene, GaState, IndState>: Send + Sync {
    /// Cross `dad` and `mom`, returning one or more child genomes.
    fn cross(
        &self,
        dad: &Individual<G, IndState>,
        mom: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> Vec<Genome<G>>;
}

/// Computes fitness for a genome; must be `Send + Sync` for Rayon sharing.
pub trait Scorer<G: Gene, GaState, IndState>: Send + Sync {
    /// Return `(fitness, individual_state)` for `individual`.
    fn score(
        &self,
        individual: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> (f64, Option<IndState>);
}

/// Reports progress after each generation; must be `Send + Sync` for Rayon sharing.
pub trait Callback<G: Gene, GaState, IndState>: Send + Sync {
    /// Called once per generation with context and all pools.
    fn call(&self, ctx: &Context<'_, G, GaState, IndState>);
}

impl<G, GaState, IndState, F> Scorer<G, GaState, IndState> for F
where
    G: Gene,
    F: Fn(&Individual<G, IndState>, &Context<'_, G, GaState, IndState>) -> (f64, Option<IndState>)
        + Send
        + Sync,
{
    fn score(
        &self,
        individual: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> (f64, Option<IndState>) {
        self(individual, ctx)
    }
}

impl<G, GaState, IndState, F> Callback<G, GaState, IndState> for F
where
    G: Gene,
    F: Fn(&Context<'_, G, GaState, IndState>) + Send + Sync,
{
    fn call(&self, ctx: &Context<'_, G, GaState, IndState>) {
        self(ctx)
    }
}

/// No-op scorer; returns `f64::NAN` for all genomes.
pub struct NoopScorer;

impl<G: Gene, GaState, IndState> Scorer<G, GaState, IndState> for NoopScorer {
    fn score(
        &self,
        _: &Individual<G, IndState>,
        _: &Context<'_, G, GaState, IndState>,
    ) -> (f64, Option<IndState>) {
        (f64::NAN, None)
    }
}

/// No-op callback; discards all arguments.
pub struct NoopCallback;

impl<G: Gene, GaState, IndState> Callback<G, GaState, IndState> for NoopCallback {
    fn call(&self, _: &Context<'_, G, GaState, IndState>) {}
}

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
