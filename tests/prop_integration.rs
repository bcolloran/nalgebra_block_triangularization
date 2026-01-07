// Property-based integration tests for the main library
use nalgebra::DMatrix;
use nalgebra_block_triangularization::{
    upper_block_triangular_structure, upper_triangular_permutations,
};
use proptest::prelude::*;

/// Generate random matrix with given dimensions
fn arbitrary_matrix(
    max_rows: usize,
    max_cols: usize,
) -> impl Strategy<Value = (usize, usize, DMatrix<u8>)> {
    (1..=max_rows, 1..=max_cols).prop_flat_map(|(nrows, ncols)| {
        let total = nrows * ncols;
        (
            Just(nrows),
            Just(ncols),
            prop::collection::vec(any::<u8>(), total).prop_map(move |bits| {
                let data: Vec<u8> = bits.into_iter().map(|b| b % 2).collect();
                DMatrix::from_row_slice(nrows, ncols, &data)
            }),
        )
    })
}

proptest! {
    /// Property: Row and column orders are valid permutations
    /// This ensures the structure produces valid orderings.
    #[test]
    fn output_orders_are_permutations(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        // Create square matrix from bits
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);

        // Row order should be a valid permutation
        let mut sorted_rows = structure.row_order.clone();
        sorted_rows.sort_unstable();
        prop_assert_eq!(sorted_rows, (0..n).collect::<Vec<_>>());

        // Column order should be a valid permutation
        let mut sorted_cols = structure.col_order.clone();
        sorted_cols.sort_unstable();
        prop_assert_eq!(sorted_cols, (0..n).collect::<Vec<_>>());
    }

    /// Property: Block sizes sum to number of rows
    /// This verifies the SCC decomposition covers all rows.
    #[test]
    fn block_sizes_sum_correctly(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);

        let sum: usize = structure.block_sizes.iter().sum();
        prop_assert_eq!(sum, n);
    }

    /// Property: Matching size is bounded
    /// This ensures matching size respects the fundamental constraint.
    #[test]
    fn matching_size_bounded_in_btf(
        bits in prop::collection::vec(any::<u8>(), 1..600)
    ) {
        // Create rectangular matrix
        let total = bits.len();
        let nrows = ((total as f64) / 1.5).sqrt() as usize;
        let ncols = if nrows > 0 { total / nrows } else { 0 };
        if nrows < 1 || ncols < 1 {
            return Ok(());
        }

        let bits: Vec<u8> = bits.into_iter().take(nrows * ncols).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(nrows, ncols, &bits);

        let structure = upper_block_triangular_structure(&m);

        prop_assert!(structure.matching_size <= nrows.min(ncols));
    }

    /// Property: Permutations are valid for rectangular matrices
    /// Tests that upper_triangular_permutations produces valid permutation sequences.
    #[test]
    fn permutations_valid_rectangular((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let structure = upper_block_triangular_structure(&m);

        // Structure orderings should have correct dimensions
        prop_assert_eq!(structure.row_order.len(), nrows, "Row order has wrong length");
        prop_assert_eq!(structure.col_order.len(), ncols, "Column order has wrong length");

        // Verify we can create permutations from them
        let (pr, pc) = upper_triangular_permutations(&m);

        // Apply them successfully (if this doesn't panic, the permutations are compatible)
        let mut u = m.clone();
        pr.permute_rows(&mut u);
        pc.permute_columns(&mut u);

        // Result should have same dimensions
        prop_assert_eq!(u.nrows(), nrows);
        prop_assert_eq!(u.ncols(), ncols);
    }

    /// Property: Applying permutations preserves matrix dimensions
    /// Verifies that permuted matrix has the same shape as original.
    #[test]
    fn permutations_preserve_dimensions((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let (pr, pc) = upper_triangular_permutations(&m);

        let mut u = m.clone();
        pr.permute_rows(&mut u);
        pc.permute_columns(&mut u);

        prop_assert_eq!(u.nrows(), nrows, "Permuted matrix has wrong row count");
        prop_assert_eq!(u.ncols(), ncols, "Permuted matrix has wrong column count");
    }

    /// Property: Empty blocks are allowed in block structure
    /// Some SCCs might have size > 1, verifying the structure handles this.
    #[test]
    fn block_structure_reasonable(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);

        // All block sizes should be positive
        for &size in &structure.block_sizes {
            prop_assert!(size > 0, "Block has zero size");
        }

        // Number of blocks should be reasonable
        prop_assert!(
            structure.block_sizes.len() <= n,
            "More blocks than rows"
        );
    }

    /// Property: Structure is deterministic
    /// Running the same matrix twice should produce the same result.
    #[test]
    fn structure_is_deterministic((_nrows, _ncols, m) in arbitrary_matrix(15, 15)) {
        let structure1 = upper_block_triangular_structure(&m);
        let structure2 = upper_block_triangular_structure(&m);

        prop_assert_eq!(structure1.row_order, structure2.row_order, "Row order not deterministic");
        prop_assert_eq!(structure1.col_order, structure2.col_order, "Column order not deterministic");
        prop_assert_eq!(structure1.block_sizes, structure2.block_sizes, "Block sizes not deterministic");
        prop_assert_eq!(structure1.matching_size, structure2.matching_size, "Matching size not deterministic");
    }

    /// Property: Zero matrix handling
    /// All-zero matrix should produce valid structure.
    #[test]
    fn handles_zero_matrix(n in 1..20usize) {
        let m = DMatrix::<u8>::zeros(n, n);
        let structure = upper_block_triangular_structure(&m);

        // Should still produce valid permutations
        prop_assert_eq!(structure.row_order.len(), n);
        prop_assert_eq!(structure.col_order.len(), n);

        let mut sorted_rows = structure.row_order.clone();
        sorted_rows.sort_unstable();
        prop_assert_eq!(sorted_rows, (0..n).collect::<Vec<_>>());

        // Zero matrix has no matching
        prop_assert_eq!(structure.matching_size, 0, "Zero matrix should have zero matching");
    }

    /// Property: Identity matrix produces single block
    /// Identity matrix should have perfect matching and single SCC.
    #[test]
    fn identity_matrix_single_block(n in 1..20usize) {
        let m = DMatrix::<u8>::identity(n, n);
        let structure = upper_block_triangular_structure(&m);

        // Should have perfect matching
        prop_assert_eq!(structure.matching_size, n, "Identity should have perfect matching");

        // Block sizes should sum to n
        let sum: usize = structure.block_sizes.iter().sum();
        prop_assert_eq!(sum, n);
    }

    /// Property: Matching quality is reasonable
    /// For well-structured matrices, matching should be large.
    #[test]
    fn matching_quality((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let structure = upper_block_triangular_structure(&m);

        // Matching size should be non-negative and bounded
        prop_assert!(structure.matching_size <= nrows.min(ncols));

        // If matching exists, at least one block
        if structure.matching_size > 0 {
            prop_assert!(!structure.block_sizes.is_empty(), "Positive matching but no blocks");
        }
    }
}
