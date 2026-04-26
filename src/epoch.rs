use rand_distr::Normal;

/// Per-generation signals shared across all GA operators.
#[derive(Debug, Clone, Copy)]
pub struct Epoch {
    pub generation: usize,   // current generation index
    pub stagnation: f32,     // 0.0 = improving, 1.0 = fully stagnated
    pub normal: Normal<f32>, // N(0,σ) built once, reused by all operators
}
