use nalgebra::DMatrix;
use nalgebra_block_triangularization::adjacency::{
    build_row_adjacency, build_row_dependency_graph,
};

#[test]
fn adjacency_empty_matrix() {
    let m: DMatrix<u8> = DMatrix::zeros(0, 0);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 0);
}

#[test]
fn adjacency_zero_rows() {
    let m: DMatrix<u8> = DMatrix::zeros(0, 5);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 0);
}

#[test]
fn adjacency_zero_cols() {
    let m: DMatrix<u8> = DMatrix::zeros(5, 0);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 5);
    for row in adj {
        assert!(row.is_empty());
    }
}

#[test]
fn adjacency_single_element_zero() {
    let m = DMatrix::from_element(1, 1, 0u8);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 1);
    assert!(adj[0].is_empty());
}

#[test]
fn adjacency_single_element_nonzero() {
    let m = DMatrix::from_element(1, 1, 1u8);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 1);
    assert_eq!(adj[0], vec![0]);
}

#[test]
fn adjacency_all_zeros() {
    let m: DMatrix<u8> = DMatrix::zeros(3, 4);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 3);
    for row in adj {
        assert!(row.is_empty());
    }
}

#[test]
fn adjacency_all_ones() {
    let m = DMatrix::from_element(3, 4, 1u8);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 3);
    for row in &adj {
        assert_eq!(row, &vec![0, 1, 2, 3]);
    }
}

#[test]
fn adjacency_identity_matrix() {
    let m: DMatrix<f64> = DMatrix::identity(5, 5);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj.len(), 5);
    for i in 0..5 {
        assert_eq!(adj[i], vec![i]);
    }
}

#[test]
fn adjacency_sparse_pattern() {
    let m = DMatrix::from_row_slice(
        4,
        5,
        &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    );
    let adj = build_row_adjacency(&m);
    assert_eq!(adj[0], vec![0, 2, 4]);
    assert_eq!(adj[1], vec![1, 3]);
    assert_eq!(adj[2], vec![0, 1]);
    assert_eq!(adj[3], vec![3, 4]);
}

#[test]
fn adjacency_sorted_and_deduped() {
    // Technically the matrix doesn't have duplicates,
    // but we test that the adjacency list is sorted
    let m = DMatrix::from_row_slice(2, 5, &[0, 0, 1, 1, 1, 1, 1, 0, 1, 0]);
    let adj = build_row_adjacency(&m);
    assert_eq!(adj[0], vec![2, 3, 4]);
    assert_eq!(adj[1], vec![0, 1, 3]);

    // Verify sorting
    for row in &adj {
        let mut sorted = row.clone();
        sorted.sort_unstable();
        assert_eq!(row, &sorted);
    }
}

#[test]
fn dependency_graph_empty() {
    let row_adj: Vec<Vec<usize>> = vec![];
    let col_to_row: Vec<Option<usize>> = vec![];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);
    assert_eq!(dep_graph.len(), 0);
}

#[test]
fn dependency_graph_no_matching() {
    let row_adj = vec![vec![0, 1], vec![2, 3]];
    let col_to_row = vec![None, None, None, None];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);
    assert_eq!(dep_graph.len(), 2);
    assert!(dep_graph[0].is_empty());
    assert!(dep_graph[1].is_empty());
}

#[test]
fn dependency_graph_perfect_matching_diagonal() {
    // Each row i matched to column i
    let row_adj = vec![vec![0], vec![1], vec![2]];
    let col_to_row = vec![Some(0), Some(1), Some(2)];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);

    // No dependencies since each row only touches its own column
    assert_eq!(dep_graph.len(), 3);
    assert!(dep_graph[0].is_empty());
    assert!(dep_graph[1].is_empty());
    assert!(dep_graph[2].is_empty());
}

#[test]
fn dependency_graph_with_dependencies() {
    // Row 0 touches cols [0, 1], col 0 -> row 0, col 1 -> row 1
    // Row 1 touches cols [1, 2], col 1 -> row 1, col 2 -> row 2
    // Row 2 touches cols [0, 2], col 0 -> row 0, col 2 -> row 2
    let row_adj = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
    let col_to_row = vec![Some(0), Some(1), Some(2)];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);

    assert_eq!(dep_graph.len(), 3);
    // Row 0 depends on row 1 (via col 1)
    assert_eq!(dep_graph[0], vec![1]);
    // Row 1 depends on row 2 (via col 2)
    assert_eq!(dep_graph[1], vec![2]);
    // Row 2 depends on row 0 (via col 0)
    assert_eq!(dep_graph[2], vec![0]);
}

#[test]
fn dependency_graph_self_loop_excluded() {
    // Row 0 touches col 0, which is matched to row 0
    // Should not create a self-loop
    let row_adj = vec![vec![0, 1]];
    let col_to_row = vec![Some(0), None];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);

    assert_eq!(dep_graph.len(), 1);
    assert!(dep_graph[0].is_empty());
}

#[test]
fn dependency_graph_multiple_paths_to_same_row() {
    // Row 0 touches cols [0, 1, 2], all matched to row 1
    let row_adj = vec![vec![0, 1, 2], vec![]];
    let col_to_row = vec![Some(1), Some(1), Some(1)];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);

    assert_eq!(dep_graph.len(), 2);
    // Should be deduped to single dependency
    assert_eq!(dep_graph[0], vec![1]);
    assert!(dep_graph[1].is_empty());
}

#[test]
fn dependency_graph_sorted() {
    let row_adj = vec![vec![0, 1, 2, 3]];
    let col_to_row = vec![Some(3), Some(1), Some(2), Some(0)];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);

    assert_eq!(dep_graph.len(), 1);
    // Dependencies should be sorted: [0, 1, 2, 3] (excluding self)
    assert_eq!(dep_graph[0], vec![1, 2, 3]);
}

#[test]
fn dependency_graph_unmatched_columns_ignored() {
    let row_adj = vec![vec![0, 1, 2], vec![1, 3]];
    // Only col 0 and 3 are matched
    let col_to_row = vec![Some(0), None, None, Some(1)];
    let dep_graph = build_row_dependency_graph(&row_adj, &col_to_row);

    assert_eq!(dep_graph.len(), 2);
    // Row 0 has no dependencies (col 0 -> row 0 self, cols 1,2 unmatched)
    assert!(dep_graph[0].is_empty());
    // Row 1 has no dependencies (col 1 unmatched, col 3 -> row 1 self)
    assert!(dep_graph[1].is_empty());
}
