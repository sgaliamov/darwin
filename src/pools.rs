use super::Pool;
use crate::{Gene, Genome, Individual};
use itertools::Itertools;
use std::ops::{Deref, DerefMut};

/// Collection of genetic algorithm pools.
///
/// Wraps a vector of pools and provides transparent access to vector methods
/// through `Deref` and `DerefMut` traits.
#[derive(Debug)]
pub struct Pools<G, IndState>(Vec<Pool<G, IndState>>);

impl<G, IndState> Pools<G, IndState> {
    /// Create a new collection with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Create from an existing vector of pools.
    pub fn from_vec(pools: Vec<Pool<G, IndState>>) -> Self {
        Self(pools)
    }

    /// Get the inner vector.
    pub fn into_inner(self) -> Vec<Pool<G, IndState>> {
        self.0
    }
}

impl<G, IndState> Deref for Pools<G, IndState> {
    type Target = Vec<Pool<G, IndState>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<G, IndState> DerefMut for Pools<G, IndState> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<G, IndState> Default for Pools<G, IndState> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<G, IndState> Pools<G, IndState> {
    /// Precompute all pool pairs for this generation.
    pub fn pairs(&self, generation: usize) -> Vec<(usize, usize)> {
        let pool_count = self.len();

        self.iter()
            .enumerate()
            .filter(|(_, pool)| !pool.individuals.is_empty())
            .filter_map(|(a, _)| {
                Self::pair(pool_count, a, generation)
                    .filter(|&b| !self[b].individuals.is_empty())
                    .map(|b| (a, b))
            })
            .collect()
    }

    /// Calculate which pools should be paired in a given generation.
    fn pair(pool_count: usize, pool_index: usize, generation: usize) -> Option<usize> {
        if pool_count < 2 {
            return None;
        }

        let offset = 1 + (generation % (pool_count - 1).max(1));
        let partner = (pool_index + offset) % pool_count;
        if partner == pool_index {
            None
        } else {
            Some(partner)
        }
    }

    /// Collect all individuals from all pools, sort by fitness (best-first),
    /// and return the top `count` mutable references.
    pub fn top_individuals_mut(&mut self, count: usize) -> Vec<&mut Individual<G, IndState>> {
        self.iter_mut()
            .flat_map(|pool| pool.individuals.iter_mut())
            .sorted_by(|a, b| b.fitness.total_cmp(&a.fitness))
            .take(count)
            .collect()
    }

    /// Best individual across all pools: cloned genome + fitness.
    /// Returns `None` if all pools are empty.
    pub fn best(&self) -> Option<(&Genome<G>, f64)>
    where
        G: Gene,
    {
        self.iter()
            .flat_map(|pool| pool.individuals.iter())
            .filter(|ind| ind.fitness.is_finite())
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
            .map(|ind| (&ind.genome, ind.fitness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Individual;
    use spectral::prelude::*;

    fn make_pools(fitness_per_pool: &[&[f64]]) -> Pools<(), ()> {
        let pools = fitness_per_pool
            .iter()
            .enumerate()
            .map(|(i, fitnesses)| {
                let individuals = fitnesses
                    .iter()
                    .map(|&f| {
                        let mut ind = Individual::<(), ()>::firstborn(0, vec![]);
                        ind.fitness = f;
                        ind
                    })
                    .collect();
                Pool::new(i, individuals)
            })
            .collect();
        Pools::from_vec(pools)
    }

    /// `top_individuals_mut` collects across all pools, sorted best-first.
    #[test]
    fn top_individuals_returns_sorted_best_first() {
        let mut pools = make_pools(&[&[1.0, 3.0], &[2.0, 4.0]]);
        let top = pools.top_individuals_mut(3);
        let fitnesses: Vec<f64> = top.iter().map(|i| i.fitness).collect();
        assert_that!(fitnesses).is_equal_to(vec![4.0, 3.0, 2.0]);
    }

    /// `pairs` returns no duplicates and no self-pairs.
    #[test]
    fn pairs_no_self_pairs_no_duplicates() {
        let pools = make_pools(&[&[1.0], &[1.0], &[1.0], &[1.0]]);
        for gnr in 0..10 {
            let pairs = pools.pairs(gnr);
            for &(a, b) in &pairs {
                assert_ne!(a, b, "self-pair at gen {gnr}");
            }
        }
    }

    /// Empty pools are excluded from pairing.
    #[test]
    fn pairs_skips_empty_pools() {
        // pool 0 empty, pool 1 and 2 populated
        let pools = make_pools(&[&[], &[1.0], &[1.0]]);
        let pairs = pools.pairs(0);
        for &(a, b) in &pairs {
            assert_ne!(a, 0, "empty pool 0 paired at gen 0");
            assert_ne!(b, 0, "empty pool 0 paired at gen 0");
        }
    }

    #[test]
    fn pair_never_returns_self() {
        for g in 0..10 {
            for pool in 0..3 {
                if let Some(partner) = Pools::<(), ()>::pair(3, pool, g) {
                    assert_ne!(partner, pool);
                }
            }
        }
    }

    #[test]
    fn test_pairing() {
        // (pool_count, pool_index, generation, expected_partner)
        let test_cases = vec![
            // Normal cases - 4 pools
            (4, 0, 0, Some(1)), // gnr 0: 0->1, 1->2, 2->3, 3->0
            (4, 1, 0, Some(2)),
            (4, 2, 0, Some(3)),
            (4, 3, 0, Some(0)),
            (4, 0, 1, Some(2)), // gnr 1: 0->2, 1->3, 2->0, 3->1
            (4, 0, 2, Some(3)), // gnr 2: 0->3, 1->0, 2->1, 3->2
            (4, 0, 3, Some(1)),
            (4, 0, 4, Some(2)),
            // Edge cases
            (0, 0, 0, None),
            (1, 0, 0, None), // single pool -> no pairing possible
            // Two pools
            (2, 0, 0, Some(1)),
            (2, 0, 1, Some(1)),
            (2, 0, 2, Some(1)),
            (2, 1, 0, Some(0)),
            (2, 1, 1, Some(0)),
            (2, 1, 2, Some(0)),
        ];

        for (pools, idx, gnr, expected) in test_cases {
            asserting(&format!("pools={pools} idx={idx} gnr={gnr}"))
                .that(&Pools::<(), ()>::pair(pools, idx, gnr))
                .is_equal_to(expected);
        }
    }
}
