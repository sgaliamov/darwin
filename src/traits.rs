use crate::{Context, Gene, Genome, Individual};

// ---------------------------------------------------------------------------
// Core operator traits
// ---------------------------------------------------------------------------

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
    fn evaluate(
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

// ---------------------------------------------------------------------------
// Fn blanket impls — lets closures stand in for any trait
// ---------------------------------------------------------------------------

impl<G, GaState, IndState, F> Generator<G, GaState, IndState> for F
where
    G: Gene,
    F: Fn(&Context<'_, G, GaState, IndState>) -> Genome<G> + Send + Sync,
{
    fn generate(&self, ctx: &Context<'_, G, GaState, IndState>) -> Genome<G> {
        self(ctx)
    }
}

impl<G, GaState, IndState, F> Mutator<G, GaState, IndState> for F
where
    G: Gene,
    F: Fn(&Individual<G, IndState>, &Context<'_, G, GaState, IndState>) -> Option<Genome<G>>
        + Send
        + Sync,
{
    fn mutant(
        &self,
        individual: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> Option<Genome<G>> {
        self(individual, ctx)
    }
}

impl<G, GaState, IndState, F> Crossover<G, GaState, IndState> for F
where
    G: Gene,
    F: Fn(
            &Individual<G, IndState>,
            &Individual<G, IndState>,
            &Context<'_, G, GaState, IndState>,
        ) -> Vec<Genome<G>>
        + Send
        + Sync,
{
    fn cross(
        &self,
        dad: &Individual<G, IndState>,
        mom: &Individual<G, IndState>,
        ctx: &Context<'_, G, GaState, IndState>,
    ) -> Vec<Genome<G>> {
        self(dad, mom, ctx)
    }
}

impl<G, GaState, IndState, F> Scorer<G, GaState, IndState> for F
where
    G: Gene,
    F: Fn(&Individual<G, IndState>, &Context<'_, G, GaState, IndState>) -> (f64, Option<IndState>)
        + Send
        + Sync,
{
    fn evaluate(
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

// ---------------------------------------------------------------------------
// No-op implementations
// ---------------------------------------------------------------------------

/// No-op generator; panics — placeholder only, should never be called.
pub struct NoopGenerator;

impl<G: Gene, GaState, IndState> Generator<G, GaState, IndState> for NoopGenerator {
    fn generate(&self, _: &Context<'_, G, GaState, IndState>) -> Genome<G> {
        panic!("NoopGenerator::generate called")
    }
}

/// No-op mutator; always returns `None`.
pub struct NoopMutator;

impl<G: Gene, GaState, IndState> Mutator<G, GaState, IndState> for NoopMutator {
    fn mutant(
        &self,
        _: &Individual<G, IndState>,
        _: &Context<'_, G, GaState, IndState>,
    ) -> Option<Genome<G>> {
        None
    }
}

/// No-op crossover; always returns empty vec.
pub struct NoopCrossover;

impl<G: Gene, GaState, IndState> Crossover<G, GaState, IndState> for NoopCrossover {
    fn cross(
        &self,
        _: &Individual<G, IndState>,
        _: &Individual<G, IndState>,
        _: &Context<'_, G, GaState, IndState>,
    ) -> Vec<Genome<G>> {
        vec![]
    }
}

/// No-op scorer; returns `f64::NAN` for all genomes.
pub struct NoopScorer;

impl<G: Gene, GaState, IndState> Scorer<G, GaState, IndState> for NoopScorer {
    fn evaluate(
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
