use std::{cmp::Reverse, collections::BinaryHeap};

/// Kahn topo sort with deterministic tie-break by `key[node]` (smaller first).
pub fn topo_sort_with_tiebreak(dag: &[Vec<usize>], key: &[usize]) -> Vec<usize> {
    let n = dag.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for &v in &dag[u] {
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

pub fn col_order_from_row_order(
    row_order: &[usize],
    row_to_col: &[Option<usize>],
    ncols: usize,
) -> Vec<usize> {
    let mut used = vec![false; ncols];
    let mut col_order = Vec::with_capacity(ncols);

    for &r in row_order {
        if let Some(c) = row_to_col.get(r).copied().flatten() {
            if c < ncols && !used[c] {
                used[c] = true;
                col_order.push(c);
            }
        }
    }

    for c in 0..ncols {
        if !used[c] {
            col_order.push(c);
        }
    }

    col_order
}
