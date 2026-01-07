use nalgebra_block_triangularization::scc::{condensation_dag, scc_id_map, tarjan_scc};

#[test]
fn scc_empty_graph() {
    let graph: Vec<Vec<usize>> = vec![];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 0);
}

#[test]
fn scc_single_node_no_edges() {
    let graph = vec![vec![]];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0], vec![0]);
}

#[test]
fn scc_single_node_self_loop() {
    let graph = vec![vec![0]];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0], vec![0]);
}

#[test]
fn scc_two_nodes_no_edges() {
    let graph = vec![vec![], vec![]];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 2);
    // Each node is its own SCC
    let mut all_nodes: Vec<_> = sccs.iter().flatten().copied().collect();
    all_nodes.sort();
    assert_eq!(all_nodes, vec![0, 1]);
}

#[test]
fn scc_two_nodes_cycle() {
    // 0 -> 1 -> 0
    let graph = vec![vec![1], vec![0]];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 1);
    let mut scc = sccs[0].clone();
    scc.sort();
    assert_eq!(scc, vec![0, 1]);
}

#[test]
fn scc_three_nodes_cycle() {
    // 0 -> 1 -> 2 -> 0
    let graph = vec![vec![1], vec![2], vec![0]];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 1);
    let mut scc = sccs[0].clone();
    scc.sort();
    assert_eq!(scc, vec![0, 1, 2]);
}

#[test]
fn scc_dag_no_cycles() {
    // 0 -> 1 -> 2 (no cycles)
    let graph = vec![vec![1], vec![2], vec![]];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 3);

    // Each node is its own SCC
    let mut all_nodes: Vec<_> = sccs.iter().flatten().copied().collect();
    all_nodes.sort();
    assert_eq!(all_nodes, vec![0, 1, 2]);
}

#[test]
fn scc_complex_graph() {
    // Graph with 2 SCCs:
    // SCC1: 0 <-> 1
    // SCC2: 2 <-> 3
    // with edge 1 -> 2
    let graph = vec![
        vec![1],    // 0 -> 1
        vec![0, 2], // 1 -> 0, 1 -> 2
        vec![3],    // 2 -> 3
        vec![2],    // 3 -> 2
    ];
    let sccs = tarjan_scc(&graph);
    assert_eq!(sccs.len(), 2);

    // Find which SCC contains 0 and which contains 2
    let scc_with_0: Vec<_> = sccs
        .iter()
        .find(|scc| scc.contains(&0))
        .unwrap()
        .iter()
        .copied()
        .collect();
    let scc_with_2: Vec<_> = sccs
        .iter()
        .find(|scc| scc.contains(&2))
        .unwrap()
        .iter()
        .copied()
        .collect();

    let mut sorted_0 = scc_with_0.clone();
    sorted_0.sort();
    let mut sorted_2 = scc_with_2.clone();
    sorted_2.sort();

    assert_eq!(sorted_0, vec![0, 1]);
    assert_eq!(sorted_2, vec![2, 3]);
}

#[test]
fn scc_all_nodes_covered() {
    // Ensure all nodes appear exactly once across all SCCs
    let graph = vec![vec![1], vec![2], vec![0], vec![4], vec![3]];
    let sccs = tarjan_scc(&graph);

    let mut all_nodes: Vec<_> = sccs.iter().flatten().copied().collect();
    all_nodes.sort();

    assert_eq!(all_nodes, vec![0, 1, 2, 3, 4]);

    // No duplicates
    all_nodes.dedup();
    assert_eq!(all_nodes, vec![0, 1, 2, 3, 4]);
}

#[test]
fn scc_id_map_simple() {
    let sccs = vec![vec![0, 1], vec![2], vec![3, 4]];
    let comp_of = scc_id_map(&sccs, 5);

    assert_eq!(comp_of[0], 0);
    assert_eq!(comp_of[1], 0);
    assert_eq!(comp_of[2], 1);
    assert_eq!(comp_of[3], 2);
    assert_eq!(comp_of[4], 2);
}

#[test]
fn scc_id_map_empty() {
    let sccs: Vec<Vec<usize>> = vec![];
    let comp_of = scc_id_map(&sccs, 0);
    assert_eq!(comp_of.len(), 0);
}

#[test]
fn condensation_dag_empty() {
    let graph: Vec<Vec<usize>> = vec![];
    let comp_of: Vec<usize> = vec![];
    let dag = condensation_dag(&graph, &comp_of, 0);
    assert_eq!(dag.len(), 0);
}

#[test]
fn condensation_dag_single_scc() {
    // All nodes in one SCC - should have no edges in DAG
    let graph = vec![vec![1], vec![0]];
    let comp_of = vec![0, 0];
    let dag = condensation_dag(&graph, &comp_of, 1);

    assert_eq!(dag.len(), 1);
    assert!(dag[0].is_empty());
}

#[test]
fn condensation_dag_two_sccs() {
    // SCC 0: {0, 1}, SCC 1: {2}
    // 1 -> 2 creates edge SCC0 -> SCC1
    let graph = vec![vec![1], vec![0, 2], vec![]];
    let comp_of = vec![0, 0, 1];
    let dag = condensation_dag(&graph, &comp_of, 2);

    assert_eq!(dag.len(), 2);
    assert_eq!(dag[0], vec![1]);
    assert!(dag[1].is_empty());
}

#[test]
fn condensation_dag_removes_self_loops() {
    // Edges within same SCC should not appear in DAG
    let graph = vec![
        vec![0, 1, 2], // 0 -> 0 (self), 0 -> 1 (same SCC), 0 -> 2 (different SCC)
        vec![0],       // 1 -> 0 (same SCC)
        vec![],
    ];
    let comp_of = vec![0, 0, 1];
    let dag = condensation_dag(&graph, &comp_of, 2);

    assert_eq!(dag.len(), 2);
    assert_eq!(dag[0], vec![1]); // Only cross-SCC edge
    assert!(dag[1].is_empty());
}

#[test]
fn condensation_dag_deduplicates() {
    // Multiple edges between same SCCs should be deduplicated
    let graph = vec![vec![2], vec![2], vec![]];
    let comp_of = vec![0, 0, 1];
    let dag = condensation_dag(&graph, &comp_of, 2);

    assert_eq!(dag.len(), 2);
    assert_eq!(dag[0], vec![1]); // Only one edge even though both 0 and 1 -> 2
    assert!(dag[1].is_empty());
}

#[test]
fn condensation_dag_sorted() {
    // Edges should be sorted
    let graph = vec![
        vec![3, 1, 2], // Points to components 2, 0, 1 (in order)
        vec![],
        vec![],
        vec![],
    ];
    let comp_of = vec![0, 0, 1, 2];
    let dag = condensation_dag(&graph, &comp_of, 3);

    assert_eq!(dag.len(), 3);
    // Should be sorted: [1, 2]
    assert_eq!(dag[0], vec![1, 2]);
    assert!(dag[1].is_empty());
    assert!(dag[2].is_empty());
}

#[test]
fn scc_integration_test() {
    // Full workflow test
    let graph = vec![vec![1], vec![2], vec![0, 3], vec![4], vec![3]];

    let sccs = tarjan_scc(&graph);
    let comp_of = scc_id_map(&sccs, 5);
    let dag = condensation_dag(&graph, &comp_of, sccs.len());

    // Verify all nodes are accounted for
    let mut all_nodes: Vec<_> = sccs.iter().flatten().copied().collect();
    all_nodes.sort();
    assert_eq!(all_nodes, vec![0, 1, 2, 3, 4]);

    // DAG should have no self-loops
    for (i, edges) in dag.iter().enumerate() {
        assert!(!edges.contains(&i), "DAG has self-loop at component {}", i);
    }
}

#[test]
fn scc_tarjan_order_independence() {
    // Different orderings of visiting nodes should produce same SCCs
    // (though possibly in different order)
    let graph = vec![vec![1], vec![2], vec![0], vec![4], vec![3]];

    let sccs = tarjan_scc(&graph);

    // Should find SCC {0,1,2} and SCC {3,4}
    assert_eq!(sccs.len(), 2);

    let mut scc_sizes: Vec<_> = sccs.iter().map(|s| s.len()).collect();
    scc_sizes.sort();
    assert_eq!(scc_sizes, vec![2, 3]);
}
