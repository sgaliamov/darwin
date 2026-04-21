# Copilot Instructions — `darwin`

A domain-agnostic, island-model genetic algorithm library in Rust (edition 2024, crate v0.0.12).

---

## Persona & response style

Terse caveman. All technical substance stay. Fluff die.
Old-school pragmatic dev. Seen crap. Know what works. Know what hurts.
Wit, irony, sarcasm — keep tone sharp. No flattery. TOXIC MODE ON!

**Drop:** articles, filler (just/really/basically/actually/simply), pleasantries, hedging.
**Fragments OK.** Short synonyms. Technical terms exact. Code blocks unchanged.
Pattern: `[thing] [action] [reason]. [next step].`
Arrows for causality: X → Y. One word when one word enough. Use memes, pop culture references. Use symbols (→, ✓, ✗) where fitting.

**Auto-clarity exceptions** (write normal, resume caveman after):
- Security warnings
- Irreversible action confirmations
- Multi-step sequences where fragment order risks misread

**Code/commits/PRs/comments:** normal mode always.
**"stop caveman" / "normal mode":** revert persona until end of session.

### Coding style

Short, smart, elegant — but sane. Pattern matching, functional/fluent style where readable.
Idiomatic Rust. Meaningful names; short (`x`, `v`, `i`) in simple closures or repetitive cases.
Remove unnecessary code. Minimalistic. Every method/type gets short comment.
Don't remove existing comments unless they are wrong or misleading.

---

## Architecture

```
GeneticAlgorithm<GaState, IndState, E: Evolver>
├── Config              – all tuning knobs; deserialized from JSON (camelCase)
├── Pools<IndState>     – Vec<Pool<IndState>>, each pool is an isolated sub-population
│   └── Pool<IndState>  – Vec<Individual<IndState>> + diversity: f32
├── E: Evolver          – pluggable genome operations (random / mutant / cross)
│   └── Evolution       – built-in stateless implementation; uses thread-local RNG
├── ScoreFn             – static fn(GenomeRef, &Option<GaState>) -> (f64, Option<IndState>)
└── CallbackFn          – static fn called after every generation
```

**Core types** (`src/ga/genome.rs`):
- `Gene = i64` — integers chosen over floats to make dedup unambiguous and avoid precision issues; floating-point domains are scaled to integer ranges by the caller
- `Genome = Vec<Gene>`
- `GeneRange = (Gene, Gene)` — inclusive
- `GeneRanges = Vec<GeneRange>` — flat; groups defined by `Config::ranges: Vec<GeneRanges>` drive crossover chunk selection (genes within a group are swapped atomically to avoid meaningless mid-parameter splicing)

---

## Generation lifecycle (`src/ga/methods.rs`)

```
run()
  └─ per generation:
       mutate()            → elite mutants spawned per pool (noise scaled by diversity + stagnation)
       recombine()         → pool pairs crossed; offspring migrated via migration_factor
       random()            → immigrants injected (gen-0: 10× overseeding heuristic)
       evaluate_generation()
         ├── dedup()       → sort by genome then fitness (NaN last), dedup keeps scored copy
         ├── score unscored (fitness NaN → skip)
         ├── retain finite-fitness only
         ├── sort descending, truncate to population_size
         └── calc_diversity()
       update_champ()
       callback_fn()
       stagnation()        → early exit
```

---

## Key design decisions

- **Integers for genes**: avoids float dedup ambiguity; scales are caller-defined via `Config::ranges`. See `readme.md` for the full rationale.
- **Island model**: multiple `Pool`s run independently. Pool `i` pairs with `(i + 1 + generation % (pool_count − 1)) % pool_count`, ensuring all partner combinations are visited across generations.
- **Elite mitigation via pools**: when one individual dominates a pool, create more pools and direct crossover offspring away (via `migration_factor`) rather than back into the dominating pool.
- **Adaptive mutation noise**: `noise_factor = (1 − diversity) + stagnation_boost × diversity`. Converged pool (`diversity ≈ 0`) → noise = 1.0 (exploration); diverse pool → noise ≈ 0 (exploitation). Stagnation counter lifts the floor back toward 1.0.
- **Pluggable `Evolver` trait**: implement `random`, `mutant`, `cross` to swap in custom operators. The built-in `Evolution` is `Send + Sync` and shared across Rayon threads (no per-thread cloning); parallelism uses `rayon::par_iter_mut` at both pool and individual level.
- **`migration_factor`**: `0.0` → offspring go to lower-indexed pool, `1.0` → higher-indexed pool. **Not** dad's/mom's pool — the config comment is misleading.
- **Mutation sigma annealing**: `σ(g) = σ_max − (σ_max − σ_min) / (max_gen − 1) × g`. `Config::sigma(generation)` computes this; floor at `min_mutation_sigma` prevents freeze.
- **Pools are retained between `run()` calls**: allows sliding-window / incremental evolution with carry-forward populations.
- **Lineage**: three variants — `Firstborn(gen)`, `Mutant(gen, parent_gen)`, `Child(gen, dad_gen, mom_gen)`. Tracks generation-level ancestry; stable per-individual UIDs are not yet assigned.

---

## Implementing a fitness function

```rust
// ScoreFn signature — must be a fn pointer, not a closure
fn my_fitness(genome: GenomeRef, state: &Option<MyState>) -> (f64, Option<MyState>) {
    let score = /* higher is better */;
    (score, None) // return updated IndState or None
}
```

`GenomeRef = &[Gene]`. The return `f64` must be **finite**; `NaN`/infinite individuals are dropped each generation.

---

## Build & test

```sh
cargo build           # debug
cargo test            # all unit tests (spectral assertions used in tests)
cargo test -- --nocapture   # see println! output from callback_fn in tests
```

Tests live in `#[cfg(test)] mod tests` at the bottom of each module. The canonical integration test is `finds_origin_within_ten_runs` in `src/ga/methods.rs` (sphere function minimisation).

---

## Known issues / open items (see `docs/review.md`)

- **No explicit elitism**: global `best` is tracked but not re-injected; top individuals can be evicted when offspring flood their pool. Risk is moderate on sharp-peak landscapes.
- **`migration_factor` semantics**: config comment says "dad/mom" but routing is by pool index order — needs doc fix or rework.
- **Gen-0 10× overseeding**: workaround for the absence of a caller-supplied genome validity predicate.
- **`mutant()` rejection**: high sigma + tight range causes near-100% rejection; no clamp fallback.
- **Tournament selection skips elites**: `tournament_selection` skips the first `mutant_count` entries; since elites sort to the top, they never participate in crossover (possibly intentional).
- **No run statistics**: no per-generation `Stats` struct (mean fitness, diversity per pool, acceptance rate) — required for post-hoc diagnosis.
- **Graveyard (unimplemented idea)**: cache evaluated individuals to skip re-scoring identical genomes across `run()` calls; reset when GA is reused with new data.
