// Property-based tests for the SCC module (Tarjan's algorithm)
use nalgebra_block_triangularization::scc::{condensation_dag, scc_id_map, tarjan_scc};
use proptest::prelude::*;

/// Generate arbitrary directed graphs
fn arbitrary_directed_graph(
    size_range: std::ops::Range<usize>,
) -> impl Strategy<Value = Vec<Vec<usize>>> {
    size_range.prop_flat_map(|n| {
        prop::collection::vec(prop::collection::vec(0..n, 0..n.min(15)), n).prop_map(
            move |adj_raw: Vec<Vec<usize>>| {
                // Sort and deduplicate adjacency lists
                adj_raw
                    .into_iter()
                    .map(|mut edges| {
                        edges.sort_unstable();
                        edges.dedup();
                        edges
                    })
                    .collect()
            },
        )
    })
}

/// Check if a graph is a DAG using DFS-based cycle detection
fn is_dag(graph: &[Vec<usize>]) -> bool {
    let n = graph.len();
    let mut color = vec![0u8; n]; // 0: white, 1: gray, 2: black

    fn dfs(u: usize, graph: &[Vec<usize>], color: &mut [u8]) -> bool {
        color[u] = 1; // gray (currently visiting)
        for &v in &graph[u] {
            if v >= color.len() {
                continue; // skip out of bounds edges
            }
            if color[v] == 1 {
                return false; // back edge = cycle
            }
            if color[v] == 0 && !dfs(v, graph, color) {
                return false;
            }
        }
        color[u] = 2; // black (done)
        true
    }

    for u in 0..n {
        if color[u] == 0 && !dfs(u, graph, &mut color) {
            return false;
        }
    }
    true
}

/// Check if node `target` is reachable from `start` in the graph
fn can_reach(graph: &[Vec<usize>], start: usize, target: usize) -> bool {
    if start == target {
        return true;
    }

    let n = graph.len();
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    visited[start] = true;

    while let Some(u) = stack.pop() {
        for &v in &graph[u] {
            if v >= n {
                continue; // skip out of bounds edges
            }
            if v == target {
                return true;
            }
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    false
}

proptest! {
    /// Property: Every node appears in exactly one SCC (partition property)
    /// This verifies that tarjan_scc correctly partitions the graph into disjoint SCCs.
    #[test]
    fn scc_partition(graph in arbitrary_directed_graph(1..30)) {
        let sccs = tarjan_scc(&graph);
        let n = graph.len();

        let mut all_nodes: Vec<_> = sccs.iter().flatten().copied().collect();
        all_nodes.sort_unstable();

        // All nodes should be covered
        prop_assert_eq!(all_nodes.len(), n, "Not all nodes are in SCCs");

        // No duplicates (each node in exactly one SCC)
        let before_dedup = all_nodes.len();
        all_nodes.dedup();
        prop_assert_eq!(all_nodes.len(), before_dedup, "Some nodes appear in multiple SCCs");

        // All values should be valid node indices
        prop_assert_eq!(all_nodes, (0..n).collect::<Vec<_>>(), "Invalid node indices in SCCs");
    }

    /// Property: SCC condensation creates a DAG (no cycles)
    /// The condensation graph (SCCs as nodes, edges between different SCCs) must be acyclic.
    #[test]
    fn condensation_is_dag(graph in arbitrary_directed_graph(1..30)) {
        let sccs = tarjan_scc(&graph);
        let comp_of = scc_id_map(&sccs, graph.len());
        let dag = condensation_dag(&graph, &comp_of, sccs.len());

        // Should be acyclic
        prop_assert!(is_dag(&dag), "Condensation graph contains cycles");
    }

    /// Property: Nodes in the same SCC are mutually reachable
    /// For any two nodes u, v in the same SCC, there should be paths u→v and v→u.
    #[test]
    fn scc_mutual_reachability(graph in arbitrary_directed_graph(2..20)) {
        let sccs = tarjan_scc(&graph);

        for scc in &sccs {
            if scc.len() > 1 {
                // Pick first two nodes in this SCC
                let u = scc[0];
                let v = scc[1];

                // Both should be able to reach each other
                prop_assert!(
                    can_reach(&graph, u, v),
                    "Node {} cannot reach {} but they're in same SCC",
                    u, v
                );
                prop_assert!(
                    can_reach(&graph, v, u),
                    "Node {} cannot reach {} but they're in same SCC",
                    v, u
                );
            }
        }
    }

    /// Property: scc_id_map creates valid mapping
    /// Every node should have a valid component ID in range [0..num_components).
    #[test]
    fn scc_id_map_valid(graph in arbitrary_directed_graph(1..30)) {
        let sccs = tarjan_scc(&graph);
        let n = graph.len();
        let comp_of = scc_id_map(&sccs, n);

        prop_assert_eq!(comp_of.len(), n, "ID map has wrong length");

        // All IDs should be in valid range
        for &cid in &comp_of {
            prop_assert!(cid < sccs.len(), "Invalid component ID: {}", cid);
        }
    }

    /// Property: Condensation DAG has no self-loops
    /// Each SCC is collapsed to a single node, so there should be no edges within the same component.
    #[test]
    fn condensation_no_self_loops(graph in arbitrary_directed_graph(1..30)) {
        let sccs = tarjan_scc(&graph);
        let comp_of = scc_id_map(&sccs, graph.len());
        let dag = condensation_dag(&graph, &comp_of, sccs.len());

        // No self-loops in condensation
        for (i, edges) in dag.iter().enumerate() {
            prop_assert!(!edges.contains(&i), "Self-loop at component {}", i);
        }
    }
}
