use rand_distr::Normal;

/// Per-generation signals shared across all GA operators.
#[derive(Debug, Clone, Copy)]
pub struct Epoch {
    /// Current generation index.
    pub generation: usize,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,

    /// Gaussian N(0, σ) for this generation; built once, reused by all operators.
    pub normal: Normal<f32>,
}
