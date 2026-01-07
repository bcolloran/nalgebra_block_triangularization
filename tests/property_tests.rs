// High-value property tests for the nalgebra block triangularization library
use nalgebra::DMatrix;
use nalgebra_block_triangularization::upper_block_triangular_structure;
use proptest::prelude::*;

proptest! {
    // ==================== INTEGRATION TESTS ====================

    /// Property: Row and column orders are valid permutations
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
}
