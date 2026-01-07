// Property-based tests for the adjacency module
use nalgebra::DMatrix;
use nalgebra_block_triangularization::adjacency::{
    build_row_adjacency, build_row_dependency_graph,
};
use nalgebra_block_triangularization::matching::hopcroft_karp;
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
    /// Property: Adjacency list should be sorted and deduplicated
    /// This ensures build_row_adjacency produces canonical adjacency lists.
    #[test]
    fn adjacency_sorted_deduped((nrows, ncols, m) in arbitrary_matrix(30, 30)) {
        let adj = build_row_adjacency(&m);

        prop_assert_eq!(adj.len(), nrows, "Adjacency list has wrong number of rows");

        for (i, row) in adj.iter().enumerate() {
            // Check sorted
            for idx in 1..row.len() {
                prop_assert!(
                    row[idx - 1] < row[idx],
                    "Row {} not sorted: {:?}",
                    i,
                    row
                );
            }

            // Check deduped
            let mut deduped = row.clone();
            deduped.dedup();
            prop_assert_eq!(
                row.len(),
                deduped.len(),
                "Row {} has duplicates: {:?}",
                i,
                row
            );

            // All column indices should be in valid range
            for &col in row {
                prop_assert!(col < ncols, "Invalid column index {} in row {}", col, i);
            }
        }
    }

    /// Property: Adjacency list correctly represents matrix nonzeros
    /// For each nonzero in the matrix, the column should appear in the adjacency list.
    #[test]
    fn adjacency_represents_nonzeros((nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let adj = build_row_adjacency(&m);

        for i in 0..nrows {
            for j in 0..ncols {
                if m[(i, j)] != 0 {
                    prop_assert!(
                        adj[i].contains(&j),
                        "Nonzero at ({}, {}) not in adjacency list",
                        i,
                        j
                    );
                }
            }
        }
    }

    /// Property: Adjacency list only contains actual nonzeros
    /// Every column in the adjacency list should correspond to a nonzero in the matrix.
    #[test]
    fn adjacency_no_false_positives((_nrows, _ncols, m) in arbitrary_matrix(20, 20)) {
        let adj = build_row_adjacency(&m);

        for (i, cols) in adj.iter().enumerate() {
            for &j in cols {
                prop_assert!(
                    m[(i, j)] != 0,
                    "Zero at ({}, {}) appears in adjacency list",
                    i,
                    j
                );
            }
        }
    }

    /// Property: Dependency graph has no self-loops
    /// A row cannot depend on itself in the dependency graph.
    #[test]
    fn dependency_no_self_loops((nrows, ncols, m) in arbitrary_matrix(30, 30)) {
        let adj = build_row_adjacency(&m);
        let matching = hopcroft_karp(&adj, ncols);
        let dep_graph = build_row_dependency_graph(&adj, &matching.col_to_row);

        prop_assert_eq!(dep_graph.len(), nrows, "Dependency graph has wrong size");

        // No self-loops
        for (i, edges) in dep_graph.iter().enumerate() {
            prop_assert!(
                !edges.contains(&i),
                "Self-loop at row {} in dependency graph: {:?}",
                i,
                edges
            );
        }
    }

    /// Property: Dependency graph edges are sorted and deduplicated
    /// The dependency graph should also maintain canonical form.
    #[test]
    fn dependency_sorted_deduped((nrows, ncols, m) in arbitrary_matrix(30, 30)) {
        let adj = build_row_adjacency(&m);
        let matching = hopcroft_karp(&adj, ncols);
        let dep_graph = build_row_dependency_graph(&adj, &matching.col_to_row);

        for (i, row) in dep_graph.iter().enumerate() {
            // Check sorted
            for idx in 1..row.len() {
                prop_assert!(
                    row[idx - 1] < row[idx],
                    "Dependency graph row {} not sorted: {:?}",
                    i,
                    row
                );
            }

            // Check deduped
            let mut deduped = row.clone();
            deduped.dedup();
            prop_assert_eq!(
                row.len(),
                deduped.len(),
                "Dependency graph row {} has duplicates: {:?}",
                i,
                row
            );

            // All row indices should be in valid range
            for &other_row in row {
                prop_assert!(
                    other_row < nrows,
                    "Invalid row index {} in dependency graph row {}",
                    other_row,
                    i
                );
            }
        }
    }

    /// Property: Dependency graph only includes matched columns
    /// If row i depends on row k, then there exists a column j matched to k that row i uses.
    #[test]
    fn dependency_respects_matching((_nrows, ncols, m) in arbitrary_matrix(20, 20)) {
        let adj = build_row_adjacency(&m);
        let matching = hopcroft_karp(&adj, ncols);
        let dep_graph = build_row_dependency_graph(&adj, &matching.col_to_row);

        for (i, deps) in dep_graph.iter().enumerate() {
            for &k in deps {
                // Row i depends on row k, so there must exist a column j such that:
                // 1. Row i has a nonzero in column j
                // 2. Column j is matched to row k
                let has_dep_edge = adj[i].iter().any(|&j| {
                    matching.col_to_row.get(j).copied().flatten() == Some(k)
                });

                prop_assert!(
                    has_dep_edge,
                    "Row {} depends on row {} but no column links them (row {} cols: {:?}, matching: {:?})",
                    i,
                    k,
                    i,
                    adj[i],
                    matching.col_to_row
                );
            }
        }
    }
}
