/// Context passed to [`super::evolver::Evolver`] methods on each mutation / crossover call.
/// Carries GA-level signals an evolver may use to tune its behaviour.
/// All pressure values are normalised to `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy)]
pub struct Context {
    /// Current generation number.
    pub generation: usize,

    /// Pool gene diversity: `0.0` = fully converged, `1.0` = maximally diverse.
    pub diversity: f32,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,
}

impl Context {
    /// Noise scaling factor derived from GA signals.
    /// High diversity → exploit (low noise); high stagnation → explore (high noise).
    pub fn noise_factor(&self) -> f32 {
        (1.0 - self.diversity) + self.stagnation * self.diversity
    }
}
