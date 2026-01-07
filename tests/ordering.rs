use nalgebra_block_triangularization::ordering::{topo_sort_with_tiebreak, col_order_from_row_order};

#[test]
fn topo_empty_dag() {
    let dag: Vec<Vec<usize>> = vec![];
    let key: Vec<usize> = vec![];
    let order = topo_sort_with_tiebreak(&dag, &key);
    assert_eq!(order.len(), 0);
}

#[test]
fn topo_single_node() {
    let dag = vec![vec![]];
    let key = vec![0];
    let order = topo_sort_with_tiebreak(&dag, &key);
    assert_eq!(order, vec![0]);
}

#[test]
fn topo_two_nodes_no_edges() {
    let dag = vec![vec![], vec![]];
    // Both have same in-degree; key determines order
    let key = vec![1, 0];  // Node 1 has lower key
    let order = topo_sort_with_tiebreak(&dag, &key);
    assert_eq!(order, vec![1, 0]);  // Should be sorted by key
}

#[test]
fn topo_two_nodes_with_edge() {
    // 0 -> 1
    let dag = vec![vec![1], vec![]];
    let key = vec![0, 0];  // Keys don't matter when topology constrains
    let order = topo_sort_with_tiebreak(&dag, &key);
    assert_eq!(order, vec![0, 1]);
}

#[test]
fn topo_linear_chain() {
    // 0 -> 1 -> 2 -> 3
    let dag = vec![
        vec![1],
        vec![2],
        vec![3],
        vec![],
    ];
    let key = vec![3, 2, 1, 0];  // Reverse order keys
    let order = topo_sort_with_tiebreak(&dag, &key);
    // Topology forces 0, 1, 2, 3 order regardless of keys
    assert_eq!(order, vec![0, 1, 2, 3]);
}

#[test]
fn topo_diamond() {
    // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    //   0
    //  / \
    // 1   2
    //  \ /
    //   3
    let dag = vec![
        vec![1, 2],
        vec![3],
        vec![3],
        vec![],
    ];
    let key = vec![0, 2, 1, 3];  // Node 2 has lower key than node 1
    let order = topo_sort_with_tiebreak(&dag, &key);
    // Must be 0 first, 3 last
    // Between 1 and 2, key=1 < key=2, so 2 should come before 1
    assert_eq!(order[0], 0);
    assert_eq!(order[3], 3);
    assert_eq!(order[1], 2);  // Lower key
    assert_eq!(order[2], 1);  // Higher key
}

#[test]
fn topo_parallel_branches() {
    // 0 -> 2, 1 -> 3 (two disconnected branches)
    let dag = vec![
        vec![2],
        vec![3],
        vec![],
        vec![],
    ];
    let key = vec![1, 0, 3, 2];  // 1<0, 2<3
    let order = topo_sort_with_tiebreak(&dag, &key);
    // Node 1 should come before 0 (lower key, both in-degree 0)
    // Node 3 should come after 1
    // Node 2 should come after 0
    assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 0));
    assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 2));
    assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 3));
}

#[test]
fn topo_tiebreak_by_key() {
    // All nodes have in-degree 0, so only key matters
    let dag = vec![vec![], vec![], vec![], vec![]];
    let key = vec![3, 1, 2, 0];
    let order = topo_sort_with_tiebreak(&dag, &key);
    // Should be sorted by key: [3, 1, 2, 0]
    assert_eq!(order, vec![3, 1, 2, 0]);
}

#[test]
fn topo_cycle_fallback() {
    // Contains a cycle: 0 -> 1 -> 0
    let dag = vec![vec![1], vec![0]];
    let key = vec![0, 1];
    let order = topo_sort_with_tiebreak(&dag, &key);
    // Should fall back to identity ordering
    assert_eq!(order, vec![0, 1]);
}

#[test]
fn topo_self_loop_fallback() {
    // Self-loop at node 0
    let dag = vec![vec![0], vec![]];
    let key = vec![0, 1];
    let order = topo_sort_with_tiebreak(&dag, &key);
    // Should fall back to identity ordering
    assert_eq!(order, vec![0, 1]);
}

#[test]
fn col_order_empty() {
    let row_order: Vec<usize> = vec![];
    let row_to_col: Vec<Option<usize>> = vec![];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 0);
    assert_eq!(col_order.len(), 0);
}

#[test]
fn col_order_no_matching() {
    let row_order = vec![0, 1, 2];
    let row_to_col = vec![None, None, None];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 3);
    // All columns unmatched, should be in natural order
    assert_eq!(col_order, vec![0, 1, 2]);
}

#[test]
fn col_order_perfect_matching_identity() {
    let row_order = vec![0, 1, 2];
    let row_to_col = vec![Some(0), Some(1), Some(2)];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 3);
    assert_eq!(col_order, vec![0, 1, 2]);
}

#[test]
fn col_order_perfect_matching_permuted() {
    let row_order = vec![2, 0, 1];
    let row_to_col = vec![Some(0), Some(1), Some(2)];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 3);
    // Row 2 (first) maps to col 2
    // Row 0 (second) maps to col 0
    // Row 1 (third) maps to col 1
    assert_eq!(col_order, vec![2, 0, 1]);
}

#[test]
fn col_order_partial_matching() {
    let row_order = vec![0, 1, 2];
    let row_to_col = vec![Some(1), None, Some(2)];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 4);
    // Matched: col 1 (from row 0), col 2 (from row 2)
    // Unmatched: col 0, col 3
    assert_eq!(col_order, vec![1, 2, 0, 3]);
}

#[test]
fn col_order_more_cols_than_rows() {
    let row_order = vec![0, 1];
    let row_to_col = vec![Some(0), Some(2)];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 5);
    // Matched: 0, 2
    // Unmatched: 1, 3, 4
    assert_eq!(col_order, vec![0, 2, 1, 3, 4]);
}

#[test]
fn col_order_respects_row_order() {
    // Row order determines col order for matched columns
    let row_order = vec![3, 1, 0, 2];
    let row_to_col = vec![Some(0), Some(1), Some(2), Some(3)];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 4);
    // Row 3 -> col 3, Row 1 -> col 1, Row 0 -> col 0, Row 2 -> col 2
    assert_eq!(col_order, vec![3, 1, 0, 2]);
}

#[test]
fn col_order_ignores_out_of_bounds() {
    let row_order = vec![0, 1];
    let row_to_col = vec![Some(0), Some(5)];  // Col 5 is out of bounds
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 3);
    // Only col 0 is valid, cols 1 and 2 are unmatched
    assert_eq!(col_order, vec![0, 1, 2]);
}

#[test]
fn col_order_no_duplicates() {
    // Even if row_to_col has duplicates (shouldn't happen, but test robustness)
    let row_order = vec![0, 1, 2];
    let row_to_col = vec![Some(0), Some(0), Some(1)];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 3);
    // Should not duplicate col 0
    assert_eq!(col_order.iter().filter(|&&c| c == 0).count(), 1);
    assert_eq!(col_order.len(), 3);
}

#[test]
fn col_order_all_columns_present() {
    let row_order = vec![0, 1];
    let row_to_col = vec![Some(1), None];
    let col_order = col_order_from_row_order(&row_order, &row_to_col, 3);
    
    // Verify all columns 0, 1, 2 appear exactly once
    let mut sorted = col_order.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);
}
