/// Context passed to [`super::evolver::Mutator`] and [`super::evolver::Crossover`] on each call.
/// Carries GA-level signals an evolver may use to tune its behavior.
/// All pressure values are normalized to `[0.0, 1.0]`.
pub struct Context<'a, GaState> {
    /// Current generation number.
    pub generation: usize,

    /// Pool gene diversity: `0.0` = fully converged, `1.0` = maximally diverse.
    pub diversity: f32,

    /// Stagnation pressure: `0.0` = still improving, `1.0` = fully stagnated.
    pub stagnation: f32,

    /// External GA state shared with the score / callback functions.
    pub state: &'a Option<GaState>,
}

impl<GaState> Copy for Context<'_, GaState> {}

impl<GaState> Clone for Context<'_, GaState> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<GaState: std::fmt::Debug> std::fmt::Debug for Context<'_, GaState> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("generation", &self.generation)
            .field("diversity", &self.diversity)
            .field("stagnation", &self.stagnation)
            .field("state", &self.state)
            .finish()
    }
}

impl<GaState> Context<'_, GaState> {
    /// Noise scaling factor derived from GA signals.
    /// High diversity → exploit (low noise); high stagnation → explore (high noise).
    pub fn noise_factor(&self) -> f32 {
        (1.0 - self.diversity) + self.stagnation * self.diversity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral::prelude::*;

    /// diversity=0 → noise=1 regardless of stagnation.
    #[test]
    fn converged_pool_is_max_noise() {
        let ctx = Context { generation: 0, diversity: 0.0, stagnation: 0.0, state: &None::<()> };
        assert_that!(ctx.noise_factor()).is_close_to(1.0, 1e-6);
    }

    /// diversity=1, stagnation=0 → noise=0 (pure exploitation).
    #[test]
    fn diverse_stable_pool_is_min_noise() {
        let ctx = Context { generation: 0, diversity: 1.0, stagnation: 0.0, state: &None::<()> };
        assert_that!(ctx.noise_factor()).is_close_to(0.0, 1e-6);
    }

    /// diversity=1, stagnation=1 → noise=1 (stagnation forces exploration).
    #[test]
    fn stagnated_diverse_pool_is_max_noise() {
        let ctx = Context { generation: 0, diversity: 1.0, stagnation: 1.0, state: &None::<()> };
        assert_that!(ctx.noise_factor()).is_close_to(1.0, 1e-6);
    }

    /// diversity=0.5, stagnation=0.5 → (0.5) + (0.5 * 0.5) = 0.75.
    #[test]
    fn mid_values_blend_correctly() {
        let ctx = Context { generation: 0, diversity: 0.5, stagnation: 0.5, state: &None::<()> };
        assert_that!(ctx.noise_factor()).is_close_to(0.75, 1e-6);
    }
}
