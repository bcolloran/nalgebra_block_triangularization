use std::{cmp::Reverse, collections::BinaryHeap};

/// Topological sort with deterministic tie-breaking.
///
/// Uses Kahn's algorithm with a priority queue to ensure deterministic ordering
/// when multiple nodes have zero in-degree. Nodes with smaller keys are processed first.
///
/// # Arguments
///
/// * `dag` - Adjacency list of a directed acyclic graph
/// * `key` - Tie-breaking keys for each node (smaller values have priority)
///
/// # Returns
///
/// Topologically sorted order of nodes. If the graph has cycles, returns identity order as fallback.
pub fn topo_sort_with_tiebreak(dag: &[Vec<usize>], key: &[usize]) -> Vec<usize> {
    let n = dag.len();
    let mut indeg = vec![0usize; n];
    for neighbors in dag.iter().take(n) {
        for &v in neighbors {
            indeg[v] += 1;
        }
    }

    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new(); // (key, node)
    for u in 0..n {
        if indeg[u] == 0 {
            heap.push(Reverse((key[u], u)));
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(Reverse((_k, u))) = heap.pop() {
        order.push(u);
        for &v in &dag[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                heap.push(Reverse((key[v], v)));
            }
        }
    }

    // If this triggers, something is wrong (condensation should be a DAG).
    if order.len() != n {
        // Fallback: identity order (still deterministic).
        return (0..n).collect();
    }

    order
}

/// Derive column ordering from row ordering and matching.
///
/// Matched columns appear in the same order as their matched rows.
/// Unmatched columns are appended at the end in ascending order.
///
/// # Arguments
///
/// * `row_order` - Ordering of rows (new position → old row index)
/// * `row_to_col` - Matching from rows to columns
/// * `ncols` - Total number of columns
///
/// # Returns
///
/// Column ordering (new position → old column index).
pub fn col_order_from_row_order(
    row_order: &[usize],
    row_to_col: &[Option<usize>],
    ncols: usize,
) -> Vec<usize> {
    let mut used = vec![false; ncols];
    let mut col_order = Vec::with_capacity(ncols);

    for &r in row_order {
        if let Some(c) = row_to_col.get(r).copied().flatten()
            && c < ncols
            && !used[c]
        {
            used[c] = true;
            col_order.push(c);
        }
    }

    for (c, &is_used) in used.iter().enumerate().take(ncols) {
        if !is_used {
            col_order.push(c);
        }
    }

    col_order
}
