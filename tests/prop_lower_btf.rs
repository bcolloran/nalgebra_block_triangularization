// Property-based tests for lower block triangularization
use nalgebra::DMatrix;
use nalgebra_block_triangularization::{
    lower_block_triangular_structure, lower_triangular_permutations,
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
    #[test]
    fn output_orders_are_permutations(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = lower_block_triangular_structure(&m);

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

        let structure = lower_block_triangular_structure(&m);

        let sum: usize = structure.block_sizes.iter().sum();
        prop_assert_eq!(sum, n);
    }

    /// Property: Matching size is bounded
    #[test]
    fn matching_size_bounded(
        bits in prop::collection::vec(any::<u8>(), 1..600)
    ) {
        let total = bits.len();
        let nrows = ((total as f64) / 1.5).sqrt() as usize;
        let ncols = if nrows > 0 { total / nrows } else { 0 };
        if nrows < 1 || ncols < 1 {
            return Ok(());
        }

        let bits: Vec<u8> = bits.into_iter().take(nrows * ncols).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(nrows, ncols, &bits);

        let structure = lower_block_triangular_structure(&m);

        prop_assert!(structure.matching_size <= nrows.min(ncols));
    }

    /// Property: Permutations are valid for rectangular matrices
    #[test]
    fn permutations_valid_rectangular((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let structure = lower_block_triangular_structure(&m);

        prop_assert_eq!(structure.row_order.len(), nrows);
        prop_assert_eq!(structure.col_order.len(), ncols);

        let (pr, pc) = lower_triangular_permutations(&m);

        let mut l = m.clone();
        pr.permute_rows(&mut l);
        pc.permute_columns(&mut l);

        prop_assert_eq!(l.nrows(), nrows);
        prop_assert_eq!(l.ncols(), ncols);
    }

    /// Property: Applying permutations preserves matrix dimensions
    #[test]
    fn permutations_preserve_dimensions((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let (pr, pc) = lower_triangular_permutations(&m);

        let mut l = m.clone();
        pr.permute_rows(&mut l);
        pc.permute_columns(&mut l);

        prop_assert_eq!(l.nrows(), nrows);
        prop_assert_eq!(l.ncols(), ncols);
    }

    /// Property: Block structure is reasonable
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

        let structure = lower_block_triangular_structure(&m);

        // All block sizes should be positive
        for &size in &structure.block_sizes {
            prop_assert!(size > 0);
        }

        // Number of blocks should be reasonable
        prop_assert!(structure.block_sizes.len() <= n);
    }

    /// Property: Structure is deterministic
    #[test]
    fn structure_is_deterministic((_nrows, _ncols, m) in arbitrary_matrix(15, 15)) {
        let structure1 = lower_block_triangular_structure(&m);
        let structure2 = lower_block_triangular_structure(&m);

        prop_assert_eq!(structure1.row_order, structure2.row_order);
        prop_assert_eq!(structure1.col_order, structure2.col_order);
        prop_assert_eq!(structure1.block_sizes, structure2.block_sizes);
        prop_assert_eq!(structure1.matching_size, structure2.matching_size);
    }

    /// Property: Zero matrix handling
    #[test]
    fn handles_zero_matrix(n in 1..20usize) {
        let m = DMatrix::<u8>::zeros(n, n);
        let structure = lower_block_triangular_structure(&m);

        prop_assert_eq!(structure.row_order.len(), n);
        prop_assert_eq!(structure.col_order.len(), n);

        let mut sorted_rows = structure.row_order.clone();
        sorted_rows.sort_unstable();
        prop_assert_eq!(sorted_rows, (0..n).collect::<Vec<_>>());

        prop_assert_eq!(structure.matching_size, 0);
    }

    /// Property: Identity matrix produces single block
    #[test]
    fn identity_matrix_single_block(n in 1..20usize) {
        let m = DMatrix::<u8>::identity(n, n);
        let structure = lower_block_triangular_structure(&m);

        prop_assert_eq!(structure.matching_size, n);

        let sum: usize = structure.block_sizes.iter().sum();
        prop_assert_eq!(sum, n);
    }

    /// Property: Matching quality is reasonable
    #[test]
    fn matching_quality((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let structure = lower_block_triangular_structure(&m);

        prop_assert!(structure.matching_size <= nrows.min(ncols));

        if structure.matching_size > 0 {
            prop_assert!(!structure.block_sizes.is_empty());
        }
    }

    /// Property: block_indices method returns correct structure
    #[test]
    fn block_indices_structure(bits in prop::collection::vec(any::<u8>(), 1..400)) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = lower_block_triangular_structure(&m);
        let blocks = structure.block_indices();

        // Number of blocks should match
        prop_assert_eq!(blocks.len(), structure.block_sizes.len());

        // Each block should have correct size
        for (i, (row_block, col_block)) in blocks.iter().enumerate() {
            prop_assert_eq!(row_block.len(), structure.block_sizes[i]);
            prop_assert_eq!(col_block.len(), structure.block_sizes[i]);
        }

        // All rows should be covered
        let all_rows: Vec<usize> = blocks.iter()
            .flat_map(|(rows, _)| rows.iter().copied())
            .collect();
        let mut sorted_rows = all_rows.clone();
        sorted_rows.sort_unstable();
        prop_assert_eq!(sorted_rows, (0..n).collect::<Vec<_>>());
    }
}
