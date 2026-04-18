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
