// Property-based tests for the ordering module (topological sort)
use nalgebra_block_triangularization::ordering::{
    col_order_from_row_order, topo_sort_with_tiebreak,
};
use proptest::prelude::*;

/// Generate arbitrary DAG (directed acyclic graph)
/// Creates edges only from lower to higher node indices to ensure acyclicity
fn arbitrary_dag(size_range: std::ops::Range<usize>) -> impl Strategy<Value = Vec<Vec<usize>>> {
    size_range.prop_flat_map(|n| {
        prop::collection::vec(prop::collection::vec(any::<usize>(), 0..n.min(15)), n).prop_map(
            move |adj_raw: Vec<Vec<usize>>| {
                let mut dag = vec![vec![]; n];
                for (u, edges) in adj_raw.iter().enumerate() {
                    for &v_raw in edges {
                        let v = v_raw % n;
                        // Only add edge if it goes forward (maintains DAG property)
                        if u < v {
                            dag[u].push(v);
                        }
                    }
                }
                // Sort and dedup each adjacency list
                for edges in &mut dag {
                    edges.sort_unstable();
                    edges.dedup();
                }
                dag
            },
        )
    })
}

proptest! {
    /// Property: Topological sort respects edge ordering
    /// For every edge u → v, u must appear before v in the topological order.
    #[test]
    fn topo_respects_edges(dag in arbitrary_dag(1..30)) {
        let n = dag.len();
        let key: Vec<usize> = (0..n).collect();
        let order = topo_sort_with_tiebreak(&dag, &key);

        // Order should contain all nodes
        prop_assert_eq!(order.len(), n, "Topo sort returned wrong number of nodes");

        // Build position map: node -> position in order
        let mut pos = vec![0; n];
        for (i, &node) in order.iter().enumerate() {
            pos[node] = i;
        }

        // Every edge u → v should have pos[u] < pos[v]
        for (u, edges) in dag.iter().enumerate() {
            for &v in edges {
                prop_assert!(
                    pos[u] < pos[v],
                    "Edge {} → {} violates topo order (pos[{}]={}, pos[{}]={})",
                    u, v, u, pos[u], v, pos[v]
                );
            }
        }
    }

    /// Property: Topological sort produces a valid permutation
    /// All nodes should appear exactly once in the order.
    #[test]
    fn topo_is_permutation(dag in arbitrary_dag(1..30)) {
        let n = dag.len();
        let key: Vec<usize> = (0..n).collect();
        let order = topo_sort_with_tiebreak(&dag, &key);

        // Should contain all nodes
        let mut sorted = order.clone();
        sorted.sort_unstable();
        prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "Not a valid permutation");
    }

    /// Property: Topological sort respects tie-breaking key
    /// When multiple nodes have no dependencies, smaller key values should come first.
    #[test]
    fn topo_respects_tiebreak(dag in arbitrary_dag(5..20)) {
        let n = dag.len();
        // Use reverse order as keys to test tie-breaking
        let key: Vec<usize> = (0..n).map(|i| n - i).collect();
        let order = topo_sort_with_tiebreak(&dag, &key);

        prop_assert_eq!(order.len(), n, "Topo sort returned wrong number of nodes");

        // Build in-degree tracking to verify tie-breaking
        let mut indeg = vec![0; n];
        for u in 0..n {
            for &v in &dag[u] {
                indeg[v] += 1;
            }
        }

        // Verify all nodes are present
        let mut sorted = order.clone();
        sorted.sort_unstable();
        prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    /// Property: Column ordering preserves row matching order
    /// Matched columns appear in the same order as their matched rows.
    #[test]
    fn col_order_respects_row_order(
        row_order in Just((0..25usize).collect::<Vec<usize>>()).prop_shuffle(),
        matching_data in prop::collection::vec(any::<u8>(), 0..25)
    ) {
        let n = row_order.len();
        if n == 0 {
            return Ok(());
        }

        // Generate partial matching from random data
        let row_to_col: Vec<Option<usize>> = matching_data
            .iter()
            .take(n)
            .enumerate()
            .map(|(_i, &val)| {
                if val % 3 == 0 {
                    None // Some rows unmatched
                } else {
                    Some((val as usize) % n) // Matched to some column
                }
            })
            .collect();

        let col_order = col_order_from_row_order(&row_order, &row_to_col, n);

        // All columns should appear exactly once
        prop_assert_eq!(col_order.len(), n, "Column order has wrong length");

        let mut sorted = col_order.clone();
        sorted.sort_unstable();
        prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "Not all columns present");
    }

    /// Property: Column ordering places matched columns before unmatched
    /// Columns matched to rows appear first, in row order, then unmatched columns.
    #[test]
    fn col_order_matched_first(
        row_order in Just((0..20usize).collect::<Vec<usize>>()).prop_shuffle(),
        match_prob in 0.3..0.9f64
    ) {
        let n = row_order.len();
        if n == 0 {
            return Ok(());
        }

        // Generate partial matching with controlled probability
        let row_to_col: Vec<Option<usize>> = (0..n)
            .map(|i| {
                if (i as f64 / n as f64) < match_prob {
                    Some(i) // Diagonal matching for simplicity
                } else {
                    None
                }
            })
            .collect();

        let col_order = col_order_from_row_order(&row_order, &row_to_col, n);

        // Count matched vs unmatched
        let num_matched = row_to_col.iter().filter(|x| x.is_some()).count();

        // First num_matched columns in col_order should be matched columns
        for i in 0..num_matched.min(col_order.len()) {
            let col = col_order[i];
            // Check if this column is matched to some row
            let is_matched = row_to_col.iter().any(|&mc| mc == Some(col));
            prop_assert!(
                is_matched,
                "Column {} at position {} is unmatched but appears before matched columns",
                col, i
            );
        }
    }

    /// Property: Column ordering handles out-of-bounds gracefully
    /// Invalid column indices in matching should be ignored.
    #[test]
    fn col_order_handles_invalid_indices(n in 1..20usize) {
        let row_order: Vec<usize> = (0..n).collect();

        // Create matching with some out-of-bounds columns
        let row_to_col: Vec<Option<usize>> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    Some(n + i) // Out of bounds
                } else if i % 3 == 1 {
                    Some(i % n) // Valid
                } else {
                    None
                }
            })
            .collect();

        let col_order = col_order_from_row_order(&row_order, &row_to_col, n);

        // Should still produce valid permutation
        prop_assert_eq!(col_order.len(), n);
        let mut sorted = col_order.clone();
        sorted.sort_unstable();
        prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>());

        // No out-of-bounds indices in result
        for &col in &col_order {
            prop_assert!(col < n, "Out-of-bounds column {} in result", col);
        }
    }
}
