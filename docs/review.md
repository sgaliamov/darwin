# GA review findings

Date: 2026-04-14
Scope: `darwin` crate (`ga/*`, `individual/*`)

---

## Overall verdict

The implementation is strong and production-usable for single-objective tuning. It is parallelized,
modular, and has good diversity/stagnation controls. Main gaps are elitism guarantees, crossover
variety, and observability depth.

---

## Architecture overview

```
GeneticAlgorithm<GaState, IndState>
├── Config             – all tuning knobs (deserialized from JSON)
├── Pools<IndState>    – Vec<Pool<IndState>>, island model
│   └── Pool<IndState>
│       ├── individuals: Vec<Individual<IndState>>
│       └── diversity: f32
├── Evolution<'a>      – stateless genome operations (created per thread via rayon `for_each_init`)
│   ├── random()       – uniform random genome
│   ├── mutant()       – Gaussian shift, discards out-of-range results
│   └── cross()        – group-wise parent swap + optional post-mutation
├── ScoreFn            – static fn pointer; returns (f64, Option<IndState>)
└── CallbackFn         – static fn pointer; called after each generation
```

Data types: `Gene = i64`, `Genome = Vec<Gene>`, ranges are `(Gene, Gene)` inclusive tuples.
`GeneRanges = Vec<(Gene, Gene)>` groups genes; groups drive crossover chunk selection.

---

## Lifecycle of a generation

```
run()
  └─ for each generation:
       mutate()           → elite mutants spawned per pool (noise scaled by diversity + stagnation)
       recombine()        → pool pairs crossed, offspring migrated
       random()           → immigrants fill deficit (gen-0: 10× overseeding)
       evaluate_generation()
         ├── dedup()
         ├── score all unscored individuals (fitness = NaN → skip)
         ├── retain finite-fitness only
         ├── sort descending by fitness
         ├── truncate to population_size
         └── calc_diversity()
       update_champ()     → global best tracked
       callback_fn()      → user hook
       stagnation()       → early exit check
```

---

## Strengths

### Parallelism
Both pool-level evolution and individual fitness evaluation use `rayon::par_iter_mut`.
`Evolution` instances are thread-local (`for_each_init`) to avoid locking the RNG.

### Annealed mutation sigma
Linear decay: `σ(g) = σ_max – (σ_max – σ_min) / (max_gen – 1) * g`.
Keeps search aggressive early, precise late. Min floor prevents freeze.

### Adaptive mutation noise
`noise_factor = (1 – diversity) + stagnation_boost * diversity`.
When a pool is converged (`diversity ≈ 0`), noise is 1.0.
When a pool is diverse (`diversity ≈ 1`), noise ≈ 0 (exploitation phase).
Stagnation counter linearly lifts the floor back toward 1.0.

### Island model with structured pairing
Pool `i` pairs with `(i + 1 + generation % (pool_count – 1)) % pool_count`.
This visits all partner combinations across generations — no two pools are always stuck together.

### Group-aware crossover
Genes are grouped (`config.ranges` is `Vec<GeneRanges>`); each group is swapped atomically.
Prevents meaningless mid-parameter splicing for structured phenotypes.
Child is optionally mutated post-cross (`cross_noise_factor`), producing 1–2 offspring per pair.

### Deduplication
Before scoring, pool is sorted by genome then fitness (NaN last), then `dedup_by` removes clones.
Duplicate with a known fitness survives; the fresh unscored copy is discarded.

### Seeding
Seed genomes are validated for length and distributed round-robin across pools.
After seeding, diversity is computed so initial `noise_factor` is correct.

### Lineage tracking
Three variants: `Firstborn(gen)`, `Mutant(gen, parent_gen)`, `Child(gen, dad_gen, mom_gen)`.
Used for display/logging; could also feed selection weighting or age-based culling.

---

## Missing / weak areas

### No explicit elitism
After truncation, the global champion can theoretically be evicted if enough offspring flood the
same pool. There is a global `best: Option<(Genome, f64)>` for tracking, but it is not re-injected
into the population. If the pool containing the best individual has it replaced, that fitness is
lost from active evolution (though the genome is preserved in `self.best`).

**Risk:** moderate for landscapes with sharp fitness peaks.
**Fix:** reserve the first `elite_count` slots before truncation per pool, or inject global best back
after truncation.

### Single crossover operator
Only uniform group-swap crossover is implemented. For continuous-ish integer genes this tends to
plateau. Alternatives to consider:
- **Arithmetic / blend crossover** — `child[i] = α * dad[i] + (1 – α) * mom[i]`
- **SBX (Simulated Binary Crossover)** — produces children symmetrically distributed around parents
- **Uniform gene crossover** — each gene independently 50/50 rather than whole group

### Uniform mutation probability
Every gene in a genome is mutated with equal probability on every mutation event. A smaller
mutation rate (e.g. `1/genome_length`) per gene is standard in binary GAs and often better for
large genomes, since currently a 100-gene genome gets 100 Gaussian shifts per mutant attempt.

### No speciation / fitness sharing
Multiple pools mitigate this, but converged pools can all end up at the same local optimum.
Classic remedies: clearing (reset worst-performing subset of each pool periodically), fitness
sharing (penalize individuals close to others), or explicit diversity requirements.

### Static selection pressure
`tournament_size` is a fixed config value. Under prolonged stagnation, increasing it would raise
the selection pressure and force exploration of worse regions. Currently, only mutation noise
is boosted — selection remains the same.

### No catastrophic restart
When `stagnation_count` is exhausted the run simply stops. A useful extension would be to nuke
the worst 50% of pools and reinitialize them from random + global best seed, then continue.

### Lineage is generation-level only
`Mutant(gen, parent_gen)` records the parent's *generation*, not a stable UID.
Multiple individuals in generation N look identical in lineage terms, making family trees
ambiguous. Assigning monotonically increasing UIDs to individuals would fix this.

### No run statistics
No struct accumulates per-generation metrics (mean fitness, best fitness, diversity per pool,
acceptance rate of mutations). These are essential for diagnosing runs post-hoc.

### Migration factor semantics are opaque
`migration_factor = 0` → all children go to `min(ia, ib)` (lower-indexed pool).
`migration_factor = 1` → all children go to `max(ia, ib)` (higher-indexed pool).
This is not "dad's pool" vs "mom's pool"; it is an index-order preference. The comment in
`Config` says "0 = all goes to dad, 1 = all goes to mom" which is misleading — pool index
and parentage are unrelated. Default is effectively `0` (nearly always goes to lower pool).

### Generation-0 over-seeding is a leaky heuristic
`random()` overseeds gen-0 to `population_size * 10` because many random individuals may be
filtered by the caller's scoring function. This is a workaround for the fact that genome
validity is not enforced at generation time. A cleaner API would accept a generator closure
that guarantees valid genomes.

---

## Potential bugs

| Location | Issue |
|---|---|
| `evolution.rs::mutant` | High sigma + tight range → near-100% rejection; no fallback to clamp |
| `pool.rs::tournament_selection` | Skips first `mutant_count` entries; combined with elites always being first means elites never participate in crossover (intentional?) |
| `methods.rs::recombine` | `migration_factor` uses index ordering, not pool-of-origin semantics |
| `pool.rs::dedup` | Sort-then-dedup changes pool order; if called before sort-by-fitness, the scoring pass afterward handles it, but ordering guarantees across calls are implicit |

---

## Recommended next steps (priority order)

1. **Elitism**: lock top `elite_count` per pool through truncation.
2. **Migration semantics**: document or rework routing to be pool-of-origin based.
3. **Statistics hook**: add a `Stats` struct emitted per generation (best, mean, diversity, stagnation).
4. **Restart policy**: on full stagnation, reinitialize worst pools rather than halting.
5. **Adaptive tournament size**: scale with stagnation depth.
6. **Constrained genome generation**: replace gen-0 10× hack with a caller-supplied validity predicate.
7. **Lineage UIDs**: monotonic counter per individual for traceable genealogies.
8. **Mutation clamping**: optionally clamp instead of discard on range violations (configurable).

---

## Practical conclusion

Solid island-model GA, well-engineered for parallelism and diversity control. Elitism gap and
migration semantics are the most immediate correctness concerns. Observability and crossover
variety are the main capability gaps. Everything else is polish.
