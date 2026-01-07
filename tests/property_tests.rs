// High-value property tests for the nalgebra block triangularization library
use nalgebra::DMatrix;
use nalgebra_block_triangularization::{matching::hopcroft_karp, upper_block_triangular_structure};
use proptest::prelude::*;

proptest! {
    // ==================== MATCHING TESTS ====================

    /// Property: Every matched edge must exist in the adjacency list
    #[test]
    fn matching_edges_exist(
        (n_right, adj) in (1..20usize).prop_flat_map(|n_right| {
            let n_left = 1..20usize;
            (Just(n_right), n_left.prop_flat_map(move |n_left| {
                prop::collection::vec(
                    prop::collection::vec(0..n_right, 0..15),
                    n_left,
                )
                .prop_map(move |adj_raw: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
                    adj_raw
                        .iter()
                        .map(|edges| {
                            let mut sorted = edges.clone();
                            sorted.sort_unstable();
                            sorted.dedup();
                            sorted
                        })
                        .collect()
                })
            }))
        })
    ) {
        let matching = hopcroft_karp(&adj, n_right);

        for (i, &opt_j) in matching.row_to_col.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert!(
                    adj[i].contains(&j),
                    "Row {} matched to col {} but no edge exists",
                    i,
                    j
                );
            }
        }
    }

    /// Property: Matching should be consistent (row_to_col and col_to_row agree)
    #[test]
    fn matching_consistency(
        (n_right, adj) in (1..20usize).prop_flat_map(|n_right| {
            let n_left = 1..20usize;
            (Just(n_right), n_left.prop_flat_map(move |n_left| {
                prop::collection::vec(
                    prop::collection::vec(0..n_right, 0..15),
                    n_left,
                )
                .prop_map(move |adj_raw: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
                    adj_raw
                        .iter()
                        .map(|edges| {
                            let mut sorted = edges.clone();
                            sorted.sort_unstable();
                            sorted.dedup();
                            sorted
                        })
                        .collect()
                })
            }))
        })
    ) {
        let matching = hopcroft_karp(&adj, n_right);

        // Forward consistency
        for (i, &opt_j) in matching.row_to_col.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert_eq!(matching.col_to_row[j], Some(i));
            }
        }

        // Backward consistency
        for (j, &opt_i) in matching.col_to_row.iter().enumerate() {
            if let Some(i) = opt_i {
                prop_assert_eq!(matching.row_to_col[i], Some(j));
            }
        }
    }

    /// Property: Matching size equals number of matched vertices
    #[test]
    fn matching_size_correct(
        (n_right, adj) in (1..20usize).prop_flat_map(|n_right| {
            let n_left = 1..20usize;
            (Just(n_right), n_left.prop_flat_map(move |n_left| {
                prop::collection::vec(
                    prop::collection::vec(0..n_right, 0..15),
                    n_left,
                )
                .prop_map(move |adj_raw: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
                    adj_raw
                        .iter()
                        .map(|edges| {
                            let mut sorted = edges.clone();
                            sorted.sort_unstable();
                            sorted.dedup();
                            sorted
                        })
                        .collect()
                })
            }))
        })
    ) {
        let matching = hopcroft_karp(&adj, n_right);

        let matched_rows = matching.row_to_col.iter().filter(|x| x.is_some()).count();
        let matched_cols = matching.col_to_row.iter().filter(|x| x.is_some()).count();

        prop_assert_eq!(matching.size, matched_rows);
        prop_assert_eq!(matching.size, matched_cols);
    }

    /// Property: No vertex is matched more than once
    #[test]
    fn matching_no_duplicates(
        (n_right, adj) in (1..20usize).prop_flat_map(|n_right| {
            let n_left = 1..20usize;
            (Just(n_right), n_left.prop_flat_map(move |n_left| {
                prop::collection::vec(
                    prop::collection::vec(0..n_right, 0..15),
                    n_left,
                )
                .prop_map(move |adj_raw: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
                    adj_raw
                        .iter()
                        .map(|edges| {
                            let mut sorted = edges.clone();
                            sorted.sort_unstable();
                            sorted.dedup();
                            sorted
                        })
                        .collect()
                })
            }))
        })
    ) {
        let matching = hopcroft_karp(&adj, n_right);

        // Check columns appear at most once
        let matched_cols: Vec<_> = matching.row_to_col.iter()
            .filter_map(|&x| x)
            .collect();
        let mut sorted_cols = matched_cols.clone();
        sorted_cols.sort_unstable();
        sorted_cols.dedup();
        prop_assert_eq!(matched_cols.len(), sorted_cols.len());

        // Check rows appear at most once
        let matched_rows: Vec<_> = matching.col_to_row.iter()
            .filter_map(|&x| x)
            .collect();
        let mut sorted_rows = matched_rows.clone();
        sorted_rows.sort_unstable();
        sorted_rows.dedup();
        prop_assert_eq!(matched_rows.len(), sorted_rows.len());
    }

    /// Property: Matching size is bounded by min(n_left, n_right)
    #[test]
    fn matching_size_bounded(
        (n_right, n_left, adj) in (1..20usize).prop_flat_map(|n_right| {
            let n_left_range = 1..20usize;
            n_left_range.prop_flat_map(move |n_left| {
                (
                    Just(n_right),
                    Just(n_left),
                    prop::collection::vec(
                        prop::collection::vec(0..n_right, 0..15),
                        n_left,
                    )
                    .prop_map(move |adj_raw: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
                        adj_raw
                            .iter()
                            .map(|edges| {
                                let mut sorted = edges.clone();
                                sorted.sort_unstable();
                                sorted.dedup();
                                sorted
                            })
                            .collect()
                    })
                )
            })
        })
    ) {
        let matching = hopcroft_karp(&adj, n_right);

        prop_assert!(matching.size <= n_left.min(n_right));
    }

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
