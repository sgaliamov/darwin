use crate::{Individual, Lineage};

impl<G, IndState> Default for Individual<G, IndState> {
    fn default() -> Self {
        Self {
            genome: Default::default(),
            lineage: Default::default(),
            fitness: f64::NAN, // to compare with the global best
            state: None,
        }
    }
}

// Display --------------------------------------------------------------------

impl std::fmt::Display for Lineage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lineage::Firstborn(g) => write!(f, "{g:02}f"),
            Lineage::Mutant(g, p) => write!(f, "{g:02}m{p}"),
            Lineage::Child(g, d, m) => write!(f, "{g:02}c{d}{m}"),
        }
    }
}

impl<G: std::fmt::Debug, IndState> std::fmt::Display for Individual<G, IndState> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {:?}", self.name(), self.genome)
    }
}
