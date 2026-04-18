/// One gene.
/// Signed type is used as it can be in a domain where negative genes may have sense.
pub type Gene = i64;

/// Organism definition.
pub type Genome = Vec<Gene>;

/// Organism definition as a reference.
/// Not just a reference to a vector but a reference to a slice.
pub type GenomeRef<'a> = &'a [Gene];

/// Defines min and max values for a gene.
/// Inclusive.
/// No need to validate, as it happens in `Rng::random_range`.
pub type GeneRange = (Gene, Gene);

/// Defines amount of genes and their ranges.
/// Left and right values are inclusive.
pub type GeneRanges = Vec<GeneRange>;
pub type GeneRangesRef<'a> = &'a [GeneRange];
