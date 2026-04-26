#[derive(Clone, Debug)]
pub enum Lineage {
    /// Randomly generated.
    /// (pool, generation)
    Firstborn(usize, usize),

    /// Mutated from something.
    /// (pool, generation, parent)
    Mutant(usize, usize, usize),

    /// Result of crossover.
    /// (pool, generation, dad, mom)
    Child(usize, usize, usize, usize),
}

impl Lineage {
    pub fn child(pool: usize, generation: usize, dad: &Lineage, mom: &Lineage) -> Self {
        Self::Child(pool, generation, dad.generation(), mom.generation())
    }

    pub fn pool(&self) -> usize {
        match self {
            Lineage::Firstborn(pool, _) => *pool,
            Lineage::Mutant(pool, _, _) => *pool,
            Lineage::Child(pool, _, _, _) => *pool,
        }
    }

    pub fn generation(&self) -> usize {
        match self {
            Lineage::Firstborn(_, generation) => *generation,
            Lineage::Mutant(_, generation, _) => *generation,
            Lineage::Child(_, generation, _, _) => *generation,
        }
    }
}

impl Default for Lineage {
    fn default() -> Self {
        Lineage::Firstborn(0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral::prelude::*;

    /// `pool()` extracts pool number from every variant.
    #[test]
    fn pool_returns_correct_value() {
        assert_that!(Lineage::Firstborn(2, 7).pool()).is_equal_to(2);
        assert_that!(Lineage::Mutant(1, 5, 2).pool()).is_equal_to(1);
        assert_that!(Lineage::Child(3, 3, 1, 2).pool()).is_equal_to(3);
    }

    /// `generation()` extracts generation from every variant.
    #[test]
    fn generation_returns_correct_value() {
        assert_that!(Lineage::Firstborn(0, 7).generation()).is_equal_to(7);
        assert_that!(Lineage::Mutant(0, 5, 2).generation()).is_equal_to(5);
        assert_that!(Lineage::Child(0, 3, 1, 2).generation()).is_equal_to(3);
    }

    /// `child()` helper stores parent generations from their lineages.
    #[test]
    fn child_captures_parent_generations() {
        let dad = Lineage::Firstborn(0, 1);
        let mom = Lineage::Mutant(1, 4, 2);
        let child = Lineage::child(2, 10, &dad, &mom);
        assert!(matches!(child, Lineage::Child(2, 10, 1, 4)));
    }

    /// Display format matches expected string patterns.
    #[test]
    fn display_formats_correctly() {
        assert_that!(Lineage::Firstborn(0, 3).to_string()).is_equal_to("p0g03f".to_owned());
        assert_that!(Lineage::Mutant(1, 5, 2).to_string()).is_equal_to("p1g05m2".to_owned());
        assert_that!(Lineage::Child(2, 7, 1, 3).to_string()).is_equal_to("p2g07c13".to_owned());
    }
}
