/// Build adjacency list from matrix sparsity pattern.
///
/// Creates a bipartite graph representation where each row's adjacency list
/// contains the columns with nonzero entries. Only the structural pattern is used
/// (nonzero vs. zero), not the actual values.
///
/// # Arguments
///
/// * `mat` - Input matrix (any scalar type with Default)
///
/// # Returns
///
/// Adjacency list where `result[i]` contains column indices with nonzeros in row `i`,
/// sorted and deduplicated.
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

/// Build row dependency graph for block triangularization.
///
/// Creates a directed graph where there is an edge from row `i` to row `k`
/// if row `i` has a nonzero in a column that is matched to row `k`.
/// Self-loops are excluded.
///
/// # Arguments
///
/// * `row_adj` - Row adjacency list from `build_row_adjacency`
/// * `col_to_row` - Column-to-row matching from maximum matching
///
/// # Returns
///
/// Adjacency list for the row dependency graph, sorted and deduplicated.
pub fn build_row_dependency_graph(
    row_adj: &[Vec<usize>],
    col_to_row: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let nrows = row_adj.len();
    let mut g = vec![Vec::new(); nrows];

    for (i, cols) in row_adj.iter().enumerate() {
        for &j in cols {
            if let Some(k) = col_to_row.get(j).copied().flatten()
                && k != i
            {
                g[i].push(k);
            }
        }
        g[i].sort_unstable();
        g[i].dedup();
    }

    g
}
