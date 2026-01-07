use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Matching {
    pub row_to_col: Vec<Option<usize>>,
    pub col_to_row: Vec<Option<usize>>,
    pub size: usize,
}

/// Hopcroft–Karp maximum bipartite matching.
/// Left side: rows (0..adj.len()).
/// Right side: columns (0..n_right).
pub fn hopcroft_karp(adj: &[Vec<usize>], n_right: usize) -> Matching {
    let n_left = adj.len();
    let mut row_to_col = vec![None; n_left];
    let mut col_to_row = vec![None; n_right];

    let inf = i32::MAX / 4;
    let mut dist = vec![inf; n_left];

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

/// BFS builds distance layers from free left nodes.
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

/// DFS tries to find augmenting paths within BFS layers.
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
