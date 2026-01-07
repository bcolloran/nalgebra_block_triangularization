use nalgebra_block_triangularization::matching::hopcroft_karp;

#[test]
fn matching_empty_graph() {
    let adj: Vec<Vec<usize>> = vec![];
    let matching = hopcroft_karp(&adj, 0);
    assert_eq!(matching.size, 0);
    assert_eq!(matching.row_to_col.len(), 0);
    assert_eq!(matching.col_to_row.len(), 0);
}

#[test]
fn matching_single_node_no_edges() {
    let adj = vec![vec![]];
    let matching = hopcroft_karp(&adj, 1);
    assert_eq!(matching.size, 0);
    assert_eq!(matching.row_to_col[0], None);
    assert_eq!(matching.col_to_row[0], None);
}

#[test]
fn matching_single_edge() {
    let adj = vec![vec![0]];
    let matching = hopcroft_karp(&adj, 1);
    assert_eq!(matching.size, 1);
    assert_eq!(matching.row_to_col[0], Some(0));
    assert_eq!(matching.col_to_row[0], Some(0));
}

#[test]
fn matching_perfect_matching_diagonal() {
    // Perfect matching: row i -> col i
    let adj = vec![vec![0], vec![1], vec![2]];
    let matching = hopcroft_karp(&adj, 3);
    assert_eq!(matching.size, 3);
    assert_eq!(matching.row_to_col[0], Some(0));
    assert_eq!(matching.row_to_col[1], Some(1));
    assert_eq!(matching.row_to_col[2], Some(2));
    assert_eq!(matching.col_to_row[0], Some(0));
    assert_eq!(matching.col_to_row[1], Some(1));
    assert_eq!(matching.col_to_row[2], Some(2));
}

#[test]
fn matching_perfect_matching_shifted() {
    // Row 0 -> cols [1], Row 1 -> cols [0, 2], Row 2 -> cols [1]
    // Optimal: 0->1, 1->0, 2->2 or 0->1, 1->2, 2->... (no match for 2)
    let adj = vec![vec![1], vec![0, 2], vec![1]];
    let matching = hopcroft_karp(&adj, 3);
    // Could be size 2 or 3 depending on algorithm choices
    // Hopcroft-Karp should find maximum matching of size 2
    // Because row 0 and row 2 both want col 1
    assert_eq!(matching.size, 2);
}

#[test]
fn matching_complete_bipartite() {
    // Complete bipartite graph K_{3,3}
    let adj = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
    let matching = hopcroft_karp(&adj, 3);
    assert_eq!(matching.size, 3);

    // Verify it's a valid matching (all matched)
    for i in 0..3 {
        assert!(matching.row_to_col[i].is_some());
    }
    for j in 0..3 {
        assert!(matching.col_to_row[j].is_some());
    }

    // Verify consistency
    for i in 0..3 {
        if let Some(j) = matching.row_to_col[i] {
            assert_eq!(matching.col_to_row[j], Some(i));
        }
    }
}

#[test]
fn matching_more_rows_than_cols() {
    // 4 rows, 2 cols - max matching is 2
    let adj = vec![vec![0], vec![1], vec![0], vec![1]];
    let matching = hopcroft_karp(&adj, 2);
    assert_eq!(matching.size, 2);

    // Two rows should be matched
    let matched_rows = matching.row_to_col.iter().filter(|x| x.is_some()).count();
    assert_eq!(matched_rows, 2);
}

#[test]
fn matching_more_cols_than_rows() {
    // 2 rows, 4 cols - max matching is 2
    let adj = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
    let matching = hopcroft_karp(&adj, 4);
    assert_eq!(matching.size, 2);

    // Two cols should be matched
    let matched_cols = matching.col_to_row.iter().filter(|x| x.is_some()).count();
    assert_eq!(matched_cols, 2);
}

#[test]
fn matching_no_possible_matching() {
    // Rows that don't reach any columns
    let adj = vec![vec![], vec![], vec![]];
    let matching = hopcroft_karp(&adj, 3);
    assert_eq!(matching.size, 0);
    for i in 0..3 {
        assert_eq!(matching.row_to_col[i], None);
        assert_eq!(matching.col_to_row[i], None);
    }
}

#[test]
fn matching_disjoint_components() {
    // Two separate components: (0,0) and (1,1)
    let adj = vec![vec![0], vec![1]];
    let matching = hopcroft_karp(&adj, 2);
    assert_eq!(matching.size, 2);
    assert_eq!(matching.row_to_col[0], Some(0));
    assert_eq!(matching.row_to_col[1], Some(1));
}

#[test]
fn matching_augmenting_path() {
    // Classic augmenting path example
    // Row 0 -> [0]
    // Row 1 -> [0, 1]
    // Row 2 -> [1]
    // Optimal: 0->0, 1->1, 2 unmatched OR 0 unmatched, 1->0, 2->1
    let adj = vec![vec![0], vec![0, 1], vec![1]];
    let matching = hopcroft_karp(&adj, 2);
    assert_eq!(matching.size, 2);
}

#[test]
fn matching_consistency() {
    // Test that row_to_col and col_to_row are consistent
    let adj = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 3], vec![2, 3]];
    let matching = hopcroft_karp(&adj, 4);

    // Verify consistency
    for (i, &opt_j) in matching.row_to_col.iter().enumerate() {
        if let Some(j) = opt_j {
            assert_eq!(matching.col_to_row[j], Some(i));
        }
    }

    for (j, &opt_i) in matching.col_to_row.iter().enumerate() {
        if let Some(i) = opt_i {
            assert_eq!(matching.row_to_col[i], Some(j));
        }
    }
}

#[test]
fn matching_respects_adjacency() {
    // Verify that matched pairs are actually connected
    let adj = vec![vec![0, 2], vec![1, 3], vec![0]];
    let matching = hopcroft_karp(&adj, 4);

    for (i, &opt_j) in matching.row_to_col.iter().enumerate() {
        if let Some(j) = opt_j {
            assert!(
                adj[i].contains(&j),
                "Row {} matched to col {} but no edge exists",
                i,
                j
            );
        }
    }
}

#[test]
fn matching_large_graph() {
    // Larger test to ensure algorithm scales
    let n = 100;
    // Create a perfect matching scenario: row i -> col i
    let adj: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let matching = hopcroft_karp(&adj, n);
    assert_eq!(matching.size, n);
}

#[test]
fn matching_hall_violation() {
    // Hall's theorem violation: 3 rows all want the same 2 columns
    let adj = vec![vec![0, 1], vec![0, 1], vec![0, 1]];
    let matching = hopcroft_karp(&adj, 2);
    // Can only match 2 out of 3 rows
    assert_eq!(matching.size, 2);
}
