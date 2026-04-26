# darwin

Domain-agnostic, island-model genetic algorithm library in Rust.

## Features

- **Island model** — multiple pools evolve in parallel with configurable migration
- **Injectable operators** — plug your own generator, mutator, crossover, scorer, and callback via traits or closures
- **Sigma annealing** — Gaussian mutation noise decays linearly from `sigma.max` → `sigma.min`
- **Stagnation detection** — auto-stops when fitness plateaus for `stagnation_count` generations
- **Diversity tracking** — per-pool diversity metric drives adaptive noise scaling
- **Reusable state** — call `run()` multiple times; populations carry forward between runs
- **Seed genomes** — inject known-good solutions distributed round-robin across pools
- **Parallel evaluation** — scoring and pool operations run on Rayon thread pool
- **Serde support** — `Config` deserializes from JSON/TOML/etc.

## Core Concepts

### Gene

Any integer type (`i8`..`i128`, `u8`..`u128`, `isize`, `usize`) implements `Gene`. Genes are the atomic values in a genome.

### Genome

A `Vec<G>` — flat DNA sequence. Ranges define valid bounds per gene position.

### Config

```rust
use darwin::Config;

let config: Config<i64> = Config {
    ranges: vec![vec![(0, 99); 6]],   // 6 genes, each in [0, 99]
    pools: 8,                          // 8 isolated sub-populations
    population_size: 100,              // individuals per pool
    max_generation: 1_000,             // hard generation cap
    stagnation_count: 100,             // stop after 100 gens without improvement
    mutation_ratio: 0.2,               // 20% of pool size → mutants per gen
    crossover_ratio: 0.2,              // 20% of pool size → crossover offspring
    random_ratio: 0.25,                // 25% fresh random immigrants per gen
    tournament_size: 3,                // tournament selection pressure
    bests: 5,                          // top-N returned to caller
    seed: vec![vec![0, 0, 0, 0, 0, 0]], // optional known-good genomes
    sigma: Default::default(),         // Sigma { max: 3.0, min: 1.0 }
    ..Default::default()
};
```

### Ranges

Ranges define the search space as nested `Vec<Vec<(G, G)>>`. Each inner vec is a coupled subgroup; they get flattened into positional bounds for the genome.

```rust
// 3 genes in [0, 100], then 2 genes in [-50, 50]
let ranges = vec![
    vec![(0, 100); 3],
    vec![(-50, 50); 2],
];
// Flat genome length = 5
```

## Operator Traits

Implement these traits (or pass closures — blanket impls exist for `Fn`):

| Trait | Method | Purpose |
|-------|--------|---------|
| `Generator` | `generate(ctx) → Genome` | Random genome creation |
| `Mutator` | `mutant(individual, ctx) → Option<Genome>` | Mutation; `None` = out of range |
| `Crossover` | `cross(dad, mom, ctx) → Vec<Genome>` | Recombination |
| `Scorer` | `evaluate(individual, ctx) → (f64, Option<IndState>)` | Fitness function |
| `Callback` | `call(ctx)` | Per-generation reporting |

All traits require `Send + Sync` for parallel execution. No-op implementations (`NoopGenerator`, `NoopMutator`, `NoopCrossover`, `NoopScorer`, `NoopCallback`) are provided.

### Context

Every operator receives `Context<G, GaState, IndState>` which exposes:

- `generation` — current generation index
- `stagnation` — `0.0` (improving) → `1.0` (fully stagnated)
- `normal` — pre-built `Normal(0, σ)` distribution for the current generation
- `state` — optional user-provided external state
- `pools` — read-only view of all pools

## Usage

### Minimal Example (Closures)

Minimize $f(\mathbf{x}) = \sum x_i^2$ over `[0, 99]^4`:

```rust
use darwin::*;

let config: Config<i64> = Config {
    ranges: vec![vec![(0, 99); 4]],
    max_generation: 500,
    stagnation_count: 50,
    population_size: 60,
    pools: 4,
    ..Default::default()
};

// Flatten ranges for generator/mutator
let flat_ranges: Vec<(i64, i64)> = config.ranges.iter().flatten().cloned().collect();

// Generator: uniform random within ranges
let generator = |ctx: &Context<'_, i64, (), ()>| -> Genome<i64> {
    // ... produce random genome within flat_ranges
    todo!()
};

// Mutator: Gaussian perturbation
let mutator = |ind: &Individual<i64, ()>, ctx: &Context<'_, i64, (), ()>| -> Option<Genome<i64>> {
    // ... apply noise to ind.genome, return None if out of bounds
    todo!()
};

// Crossover: uniform crossover
let crossover = |dad: &Individual<i64, ()>, mom: &Individual<i64, ()>, ctx: &Context<'_, i64, (), ()>| -> Vec<Genome<i64>> {
    // ... mix genes from dad and mom
    todo!()
};

// Scorer: negative sphere (GA maximizes fitness)
let scorer = |ind: &Individual<i64, ()>, _: &Context<'_, i64, (), ()>| -> (f64, Option<()>) {
    let f = -ind.genome.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    (f, None)
};

let mut ga = GeneticAlgorithm::new(
    &config, generator, mutator, crossover, scorer, NoopCallback,
);

let pools = ga.run();
let best = pools.top_individuals_mut(1);
println!("Best fitness: {}", best[0].fitness);
println!("Best genome: {:?}", best[0].genome);
```

### With External State

Pass state to scorer/callback via `set_state`:

```rust
struct MyState {
    target: Vec<i64>,
}

// Scorer uses state
let scorer = |ind: &Individual<i64, MyState>, ctx: &Context<'_, i64, MyState, ()>| {
    let target = &ctx.state.as_ref().unwrap().target;
    let dist: f64 = ind.genome.iter()
        .zip(target)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    (-dist, None)
};

// After creating GA:
// ga.set_state(MyState { target: vec![42, 42, 42, 42] });
```

### Multi-Run (Sliding Window)

Pools persist between `run()` calls — populations carry forward:

```rust
let pools = ga.run();
let best1 = pools.top_individuals_mut(1)[0].fitness;

// Update state, re-run — starts from evolved population
ga.set_state(new_state);
let pools = ga.run();
let best2 = pools.top_individuals_mut(1)[0].fitness;
// best2 >= best1 (populations carry forward)
```

## Sigma Annealing

Mutation noise follows a linear schedule:

$$\sigma(g) = \sigma_{\max} - \frac{\sigma_{\max} - \sigma_{\min}}{G - 1} \cdot g$$

where $g$ = current generation, $G$ = `max_generation`. Actual mutation amplitude also scales with pool diversity and stagnation pressure.

## Architecture

```
┌─────────────────────────────────────┐
│         GeneticAlgorithm            │
│  ┌──────┐ ┌──────┐     ┌──────┐    │
│  │Pool 0│ │Pool 1│ ... │Pool N│    │
│  └──┬───┘ └──┬───┘     └──┬───┘    │
│     │        │             │        │
│  Each generation per pool:          │
│  1. Mutate top individuals          │
│  2. Crossover (tournament select)   │
│  3. Inject random immigrants        │
│  4. Evaluate & sort                 │
│  5. Truncate to population_size     │
│  6. Update diversity                │
│  7. Check stagnation → break?       │
└─────────────────────────────────────┘
```

## License

[MIT](LICENSE)
