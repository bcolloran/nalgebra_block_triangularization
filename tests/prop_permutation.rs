// Property-based tests for the permutation module
use nalgebra::DMatrix;
use nalgebra_block_triangularization::permutation::permutation_sequence_from_order;
use proptest::prelude::*;

// Helper to apply permutation to vector
fn apply_perm_to_vec(
    perm: &nalgebra::PermutationSequence<nalgebra::Dyn>,
    v: &[usize],
) -> Vec<usize> {
    let n = v.len();
    let mut m = DMatrix::from_fn(n, 1, |i, _| v[i] as f64);
    perm.permute_rows(&mut m);
    (0..n).map(|i| m[(i, 0)] as usize).collect()
}

proptest! {
    /// Property: Result of applying permutation is a valid permutation (bijection)
    /// This ensures that permutation_sequence_from_order produces valid permutations
    /// where every element appears exactly once.
    #[test]
    fn permutation_is_bijection(
        perm_vec in Just((0..25usize).collect::<Vec<usize>>())
            .prop_shuffle()
            .prop_filter("non-empty", |v: &Vec<usize>| !v.is_empty())
    ) {
        let perm = permutation_sequence_from_order(&perm_vec);
        let n = perm_vec.len();
        let input: Vec<usize> = (0..n).collect();
        let result = apply_perm_to_vec(&perm, &input);

        // Result should contain all elements exactly once
        let mut sorted = result.clone();
        sorted.sort_unstable();
        prop_assert_eq!(sorted, input);
    }

    /// Property: Inverse permutation undoes original
    /// This tests that composing a permutation with its inverse yields the identity.
    #[test]
    fn permutation_inverse(
        perm_vec in Just((0..25usize).collect::<Vec<usize>>())
            .prop_shuffle()
            .prop_filter("non-empty", |v: &Vec<usize>| !v.is_empty())
    ) {
        let n = perm_vec.len();

        // Compute inverse
        let mut inverse = vec![0; n];
        for (i, &old) in perm_vec.iter().enumerate() {
            inverse[old] = i;
        }

        let perm = permutation_sequence_from_order(&perm_vec);
        let inv_perm = permutation_sequence_from_order(&inverse);

        let input: Vec<usize> = (0..n).map(|i| i * 10 + 7).collect();
        let permuted = apply_perm_to_vec(&perm, &input);
        let back = apply_perm_to_vec(&inv_perm, &permuted);

        prop_assert_eq!(back, input);
    }

    /// Property: Identity permutation leaves input unchanged
    /// This tests the edge case where the permutation is the identity.
    #[test]
    fn identity_permutation(n in 1..25usize) {
        let order: Vec<usize> = (0..n).collect();
        let perm = permutation_sequence_from_order(&order);

        let input: Vec<usize> = (0..n).map(|i| i * 13 + 5).collect();
        let result = apply_perm_to_vec(&perm, &input);

        prop_assert_eq!(result, input);
    }
}
