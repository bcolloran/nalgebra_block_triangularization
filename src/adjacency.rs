/// Build adjacency list from rows to columns for all nonzeros (pattern only).
pub fn build_row_adjacency<T, R, C, S>(mat: &nalgebra::Matrix<T, R, C, S>) -> Vec<Vec<usize>>
where
    T: nalgebra::Scalar + PartialEq + Default,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::Storage<T, R, C>,
{
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let zero = T::default();

    let mut adj = vec![Vec::new(); nrows];
    for i in 0..nrows {
        for j in 0..ncols {
            if mat[(i, j)] != zero {
                adj[i].push(j);
            }
        }
        // Determinism helps produce repeatable matchings.
        adj[i].sort_unstable();
        adj[i].dedup();
    }
    adj
}

/// Row dependency graph used for BTF:
/// edge i -> k if row i has a nonzero in some column matched to row k.
pub fn build_row_dependency_graph(
    row_adj: &[Vec<usize>],
    col_to_row: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let nrows = row_adj.len();
    let mut g = vec![Vec::new(); nrows];

    for (i, cols) in row_adj.iter().enumerate() {
        for &j in cols {
            if let Some(k) = col_to_row.get(j).copied().flatten() {
                if k != i {
                    g[i].push(k);
                }
            }
        }
        g[i].sort_unstable();
        g[i].dedup();
    }

    g
}
