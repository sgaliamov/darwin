use std::fmt::Debug;
use std::ops::Sub;

/// Core constraint for gene values.
/// Integer-like types: ordered, subtractable, and castable to `f64`.
/// Arithmetic beyond subtraction belongs to the evolution implementations.
pub trait Gene: Copy + Ord + Sub<Output = Self> + Debug + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

macro_rules! impl_gene {
    ($($t:ty),*) => {
        $(impl Gene for $t { fn to_f64(self) -> f64 { self as f64 } })*
    };
}

impl_gene!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

/// Flat DNA sequence.
pub type Genome<G> = Vec<G>;

/// Borrowed DNA slice.
pub type GenomeRef<'a, G> = &'a [G];

/// Inclusive `[min, max]` range for one gene.
pub type GeneRange<G> = (G, G);

/// Per-gene ranges for a full genome.
pub type GeneRanges<G> = Vec<GeneRange<G>>;

/// Borrowed per-gene ranges.
pub type GeneRangesRef<'a, G> = &'a [GeneRange<G>];

/// Collection of genome range sets; one `GeneRanges` entry per pool.
pub type RangeSet<G> = Vec<GeneRanges<G>>;
