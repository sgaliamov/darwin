// tbd: [ga] keep UID for each individual and use here to track parents.
#[derive(Clone, Debug)]
pub enum Lineage {
    /// Randomly generated.
    Firstborn(usize),

    /// Mutated from something.
    /// (generation, parent)
    Mutant(usize, usize),

    /// Result of crossover.
    /// (generation, dad, mom)
    Child(usize, usize, usize),
}

impl Lineage {
    pub fn child(generation: usize, dad: &Lineage, mom: &Lineage) -> Self {
        Self::Child(generation, dad.generation(), mom.generation())
    }

    pub fn generation(&self) -> usize {
        match self {
            Lineage::Firstborn(generation) => *generation,
            Lineage::Mutant(generation, _) => *generation,
            Lineage::Child(generation, _, _) => *generation,
        }
    }
}

impl Default for Lineage {
    fn default() -> Self {
        Lineage::Firstborn(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral::prelude::*;

    /// `generation()` extracts the first field from every variant.
    #[test]
    fn generation_returns_correct_value() {
        assert_that!(Lineage::Firstborn(7).generation()).is_equal_to(7);
        assert_that!(Lineage::Mutant(5, 2).generation()).is_equal_to(5);
        assert_that!(Lineage::Child(3, 1, 2).generation()).is_equal_to(3);
    }

    /// `child()` helper stores parent generations from their lineages.
    #[test]
    fn child_captures_parent_generations() {
        let dad = Lineage::Firstborn(1);
        let mom = Lineage::Mutant(4, 2);
        let child = Lineage::child(10, &dad, &mom);
        assert!(matches!(child, Lineage::Child(10, 1, 4)));
    }

    /// Display format matches expected string patterns.
    #[test]
    fn display_formats_correctly() {
        assert_that!(Lineage::Firstborn(3).to_string()).is_equal_to("03f".to_owned());
        assert_that!(Lineage::Mutant(5, 2).to_string()).is_equal_to("05m2".to_owned());
        assert_that!(Lineage::Child(7, 1, 3).to_string()).is_equal_to("07c13".to_owned());
    }
}
