use nalgebra::{Dyn, Matrix, PermutationSequence, Scalar, Storage};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// Return row/column permutations P, Q (as PermutationSequence) such that:
///     U = P * mat * Q
/// is (upper) block triangular with respect to the SCC block structure induced by a
/// maximum matching.
///
/// Notes:
/// - This is purely structural: it uses mat[(i,j)] != Default::default() as "nonzero".
/// - Works best / most meaningfully for square matrices with a perfect matching.
/// - For rectangular or structurally singular patterns, it still produces a useful diagnostic
///   ordering; unmatched columns are appended at the end.
///
/// You apply these like:
///   let (pr, pc) = upper_triangular_permutations(&mat);
///   let mut u = mat.clone();
///   pr.permute_rows(&mut u);
///   pc.permute_columns(&mut u);
pub fn upper_triangular_permutations<T, R, C, S>(
    mat: &Matrix<T, R, C, S>,
) -> (PermutationSequence<Dyn>, PermutationSequence<Dyn>)
where
    T: Scalar + PartialEq + Default,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: Storage<T, R, C>,
{
    let structure = upper_block_triangular_structure(mat);

    let prow = permutation_sequence_from_order(&structure.row_order);
    let pcol = permutation_sequence_from_order(&structure.col_order);

    (prow, pcol)
}

/// Extra structure you can print for diagnostics.
#[derive(Debug, Clone)]
pub struct UpperBtfStructure {
    /// New position -> old row index
    pub row_order: Vec<usize>,
    /// New position -> old col index
    pub col_order: Vec<usize>,
    /// Sizes of diagonal SCC blocks, in order.
    pub block_sizes: Vec<usize>,
    /// Size of maximum matching.
    pub matching_size: usize,
}

/// Compute the ordering + block sizes (useful for printing block separators).
pub fn upper_block_triangular_structure<T, R, C, S>(mat: &Matrix<T, R, C, S>) -> UpperBtfStructure
where
    T: Scalar + PartialEq + Default,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: Storage<T, R, C>,
{
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    // Trivial cases.
    if nrows == 0 || ncols == 0 {
        return UpperBtfStructure {
            row_order: (0..nrows).collect(),
            col_order: (0..ncols).collect(),
            block_sizes: Vec::new(),
            matching_size: 0,
        };
    }

    let row_adj = build_row_adjacency(mat);
    let matching = hopcroft_karp(&row_adj, ncols);

    // Row dependency graph: i -> k if row i touches a column matched to row k.
    let row_graph = build_row_dependency_graph(&row_adj, &matching.col_to_row);

    // SCCs on row_graph define diagonal blocks.
    let sccs = tarjan_scc(&row_graph);

    // Condensation DAG of SCCs.
    let comp_of = scc_id_map(&sccs, nrows);
    let dag = condensation_dag(&row_graph, &comp_of, sccs.len());

    // Tie-break key per SCC for deterministic topo order: min row index inside SCC.
    let scc_key: Vec<usize> = sccs
        .iter()
        .map(|comp| comp.iter().copied().min().unwrap_or(usize::MAX))
        .collect();

    // Topologically order SCC DAG so edges go "forward" -> yields upper block triangular.
    let scc_order = topo_sort_with_tiebreak(&dag, &scc_key);

    // Build row_order from SCC order, with deterministic in-SCC ordering.
    let mut row_order = Vec::with_capacity(nrows);
    let mut block_sizes = Vec::with_capacity(sccs.len());
    for &cid in &scc_order {
        let mut comp = sccs[cid].clone();
        comp.sort_unstable();
        block_sizes.push(comp.len());
        row_order.extend(comp);
    }

    // Column order: matched columns in the same order as their rows, then unmatched columns.
    let col_order = col_order_from_row_order(&row_order, &matching.row_to_col, ncols);

    UpperBtfStructure {
        row_order,
        col_order,
        block_sizes,
        matching_size: matching.size,
    }
}

/// Build adjacency list from rows to columns for all nonzeros (pattern only).
fn build_row_adjacency<T, R, C, S>(mat: &Matrix<T, R, C, S>) -> Vec<Vec<usize>>
where
    T: Scalar + PartialEq + Default,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: Storage<T, R, C>,
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

#[derive(Debug, Clone)]
struct Matching {
    row_to_col: Vec<Option<usize>>,
    col_to_row: Vec<Option<usize>>,
    size: usize,
}

/// Hopcroft–Karp maximum bipartite matching.
/// Left side: rows (0..adj.len()).
/// Right side: columns (0..n_right).
fn hopcroft_karp(adj: &[Vec<usize>], n_right: usize) -> Matching {
    let n_left = adj.len();
    let mut row_to_col = vec![None; n_left];
    let mut col_to_row = vec![None; n_right];

    let inf = i32::MAX / 4;
    let mut dist = vec![inf; n_left];

    // BFS builds distance layers from free left nodes.
    fn bfs(
        n_left: usize,
        adj: &[Vec<usize>],
        row_to_col: &[Option<usize>],
        col_to_row: &[Option<usize>],
        dist: &mut [i32],
        inf: i32,
    ) -> bool {
        let mut q = VecDeque::new();
        for u in 0..n_left {
            if row_to_col[u].is_none() {
                dist[u] = 0;
                q.push_back(u);
            } else {
                dist[u] = inf;
            }
        }

        let mut found_augmenting = false;

        while let Some(u) = q.pop_front() {
            for &v in &adj[u] {
                if let Some(u2) = col_to_row[v] {
                    if dist[u2] == inf {
                        dist[u2] = dist[u] + 1;
                        q.push_back(u2);
                    }
                } else {
                    // We found a path to a free right node.
                    found_augmenting = true;
                }
            }
        }

        found_augmenting
    }

    // DFS tries to find augmenting paths within BFS layers.
    fn dfs(
        u: usize,
        adj: &[Vec<usize>],
        row_to_col: &mut [Option<usize>],
        col_to_row: &mut [Option<usize>],
        dist: &mut [i32],
        inf: i32,
    ) -> bool {
        for &v in &adj[u] {
            match col_to_row[v] {
                None => {
                    row_to_col[u] = Some(v);
                    col_to_row[v] = Some(u);
                    return true;
                }
                Some(u2) => {
                    if dist[u2] == dist[u] + 1 && dfs(u2, adj, row_to_col, col_to_row, dist, inf) {
                        row_to_col[u] = Some(v);
                        col_to_row[v] = Some(u);
                        return true;
                    }
                }
            }
        }
        dist[u] = inf;
        false
    }

    let mut matching_size = 0;
    while bfs(n_left, adj, &row_to_col, &col_to_row, &mut dist, inf) {
        for u in 0..n_left {
            if row_to_col[u].is_none() {
                if dfs(u, adj, &mut row_to_col, &mut col_to_row, &mut dist, inf) {
                    matching_size += 1;
                }
            }
        }
    }

    Matching {
        row_to_col,
        col_to_row,
        size: matching_size,
    }
}

/// Row dependency graph used for BTF:
/// edge i -> k if row i has a nonzero in some column matched to row k.
fn build_row_dependency_graph(
    row_adj: &[Vec<usize>],
    col_to_row: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let nrows = row_adj.len();
    let mut g = vec![Vec::new(); nrows];

    for (i, cols) in row_adj.iter().enumerate() {
        for &j in cols {
            if let Some(k) = col_to_row.get(j).and_then(|x| *x) {
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

/// Tarjan SCC on a directed graph adjacency list.
fn tarjan_scc(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = graph.len();
    let mut index = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut idx: Vec<Option<usize>> = vec![None; n];
    let mut low = vec![0usize; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();

    fn strongconnect(
        v: usize,
        graph: &[Vec<usize>],
        index: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut [bool],
        idx: &mut [Option<usize>],
        low: &mut [usize],
        comps: &mut Vec<Vec<usize>>,
    ) {
        idx[v] = Some(*index);
        low[v] = *index;
        *index += 1;

        stack.push(v);
        on_stack[v] = true;

        for &w in &graph[v] {
            if idx[w].is_none() {
                strongconnect(w, graph, index, stack, on_stack, idx, low, comps);
                low[v] = low[v].min(low[w]);
            } else if on_stack[w] {
                low[v] = low[v].min(idx[w].unwrap());
            }
        }

        // Root of SCC
        if low[v] == idx[v].unwrap() {
            let mut comp = Vec::new();
            loop {
                let w = stack.pop().expect("stack underflow");
                on_stack[w] = false;
                comp.push(w);
                if w == v {
                    break;
                }
            }
            comps.push(comp);
        }
    }

    for v in 0..n {
        if idx[v].is_none() {
            strongconnect(
                v,
                graph,
                &mut index,
                &mut stack,
                &mut on_stack,
                &mut idx,
                &mut low,
                &mut comps,
            );
        }
    }

    comps
}

fn scc_id_map(sccs: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut comp_of = vec![usize::MAX; n];
    for (cid, comp) in sccs.iter().enumerate() {
        for &v in comp {
            comp_of[v] = cid;
        }
    }
    debug_assert!(comp_of.iter().all(|&x| x != usize::MAX));
    comp_of
}

fn condensation_dag(graph: &[Vec<usize>], comp_of: &[usize], ncomp: usize) -> Vec<Vec<usize>> {
    let mut dag = vec![Vec::new(); ncomp];
    for u in 0..graph.len() {
        let cu = comp_of[u];
        for &v in &graph[u] {
            let cv = comp_of[v];
            if cu != cv {
                dag[cu].push(cv);
            }
        }
    }
    for out in &mut dag {
        out.sort_unstable();
        out.dedup();
    }
    dag
}

/// Kahn topo sort with deterministic tie-break by `key[node]` (smaller first).
fn topo_sort_with_tiebreak(dag: &[Vec<usize>], key: &[usize]) -> Vec<usize> {
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

fn col_order_from_row_order(
    row_order: &[usize],
    row_to_col: &[Option<usize>],
    ncols: usize,
) -> Vec<usize> {
    let mut used = vec![false; ncols];
    let mut col_order = Vec::with_capacity(ncols);

    for &r in row_order {
        if let Some(c) = row_to_col.get(r).and_then(|x| *x) {
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

/// Convert an explicit order (new_pos -> old_index) into a nalgebra PermutationSequence<Dyn>
/// via a minimal-ish sequence of swaps.
///
/// This generates swaps that transform [0,1,2,..] into `order`.
fn permutation_sequence_from_order(order: &[usize]) -> PermutationSequence<Dyn> {
    let n = order.len();
    let mut p = PermutationSequence::<Dyn>::identity(n); // dynamic dimension 

    // Validate it is a permutation of 0..n-1 (debug-time check).
    debug_assert!({
        let mut seen = vec![false; n];
        let mut valid = true;
        for &x in order {
            if x >= n || seen[x] {
                valid = false;
                break;
            }
            seen[x] = true;
        }
        valid && seen.iter().all(|&x| x)
    });

    let mut current: Vec<usize> = (0..n).collect(); // position -> element
    let mut pos_of: Vec<usize> = (0..n).collect(); // element -> position

    for i in 0..n {
        let desired = order[i];
        let j = pos_of[desired];
        if i != j {
            // Swap positions i and j.
            p.append_permutation(i, j);

            let a = current[i];
            let b = current[j];
            current.swap(i, j);
            pos_of[a] = j;
            pos_of[b] = i;
        }
    }

    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn apply_perms<T: Scalar + Copy>(
        mut m: DMatrix<T>,
        pr: &PermutationSequence<Dyn>,
        pc: &PermutationSequence<Dyn>,
    ) -> DMatrix<T> {
        pr.permute_rows(&mut m);
        pc.permute_columns(&mut m);
        m
    }

    fn is_upper_block_triangular_u8(m: &DMatrix<u8>, block_sizes: &[usize]) -> bool {
        let n = m.nrows();
        if n != m.ncols() {
            return false;
        }
        if block_sizes.iter().sum::<usize>() != n {
            return false;
        }

        let mut row_block = vec![0usize; n];
        let mut col_block = vec![0usize; n];

        let mut idx = 0usize;
        for (b, &sz) in block_sizes.iter().enumerate() {
            for _ in 0..sz {
                row_block[idx] = b;
                col_block[idx] = b;
                idx += 1;
            }
        }

        for i in 0..n {
            for j in 0..n {
                if m[(i, j)] != 0 && row_block[i] > col_block[j] {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn example_matrix_produces_upper_block_triangular_form() {
        // Your example:
        // 8x8 binary matrix
        let data: [[u8; 8]; 8] = [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 0, 1, 1],
        ];

        let m = DMatrix::from_fn(8, 8, |i, j| data[i][j]);
        let structure = upper_block_triangular_structure(&m);
        let (pr, pc) = upper_triangular_permutations(&m);
        let u = apply_perms(m.clone(), &pr, &pc);

        assert_eq!(structure.matching_size, 8); // perfect matching for this pattern
        assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
    }
}
