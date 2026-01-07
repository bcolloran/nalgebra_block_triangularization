// Property-based tests for the matching module (Hopcroft-Karp algorithm)
use nalgebra_block_triangularization::matching::hopcroft_karp;
use proptest::prelude::*;

proptest! {
    /// Property: Every matched edge must exist in the adjacency list
    /// This verifies the matching algorithm only selects edges that actually exist in the graph.
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
    /// This ensures the bidirectional mapping is symmetric.
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

        // Forward consistency: if row i -> col j, then col j -> row i
        for (i, &opt_j) in matching.row_to_col.iter().enumerate() {
            if let Some(j) = opt_j {
                prop_assert_eq!(matching.col_to_row[j], Some(i));
            }
        }

        // Backward consistency: if col j -> row i, then row i -> col j
        for (j, &opt_i) in matching.col_to_row.iter().enumerate() {
            if let Some(i) = opt_i {
                prop_assert_eq!(matching.row_to_col[i], Some(j));
            }
        }
    }

    /// Property: Matching size equals number of matched vertices
    /// This ensures the size field correctly counts matched vertices on both sides.
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
    /// This verifies the matching is valid (each vertex in at most one edge).
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
    /// This ensures the matching size respects the fundamental constraint of bipartite matchings.
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
}
