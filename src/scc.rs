/// Compute strongly connected components using Tarjan's algorithm.
///
/// Returns a list of SCCs, where each SCC is a vector of vertex indices.
/// The SCCs are returned in reverse topological order of the condensation DAG.
///
/// # Arguments
///
/// * `graph` - Adjacency list representation of a directed graph
///
/// # Returns
///
/// Vector of SCCs, where each SCC is a vector of vertex indices.
pub fn tarjan_scc(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = graph.len();
    let mut state = TarjanState {
        index: 0,
        stack: Vec::new(),
        on_stack: vec![false; n],
        idx: vec![None; n],
        low: vec![0; n],
        comps: Vec::new(),
    };

    for v in 0..n {
        if state.idx[v].is_none() {
            strongconnect(v, graph, &mut state);
        }
    }

    state.comps
}

struct TarjanState {
    index: usize,
    stack: Vec<usize>,
    on_stack: Vec<bool>,
    idx: Vec<Option<usize>>,
    low: Vec<usize>,
    comps: Vec<Vec<usize>>,
}

fn strongconnect(v: usize, graph: &[Vec<usize>], state: &mut TarjanState) {
    state.idx[v] = Some(state.index);
    state.low[v] = state.index;
    state.index += 1;

    state.stack.push(v);
    state.on_stack[v] = true;

    for &w in &graph[v] {
        if state.idx[w].is_none() {
            strongconnect(w, graph, state);
            state.low[v] = state.low[v].min(state.low[w]);
        } else if state.on_stack[w] {
            state.low[v] = state.low[v].min(state.idx[w].unwrap());
        }
    }

    // Root of SCC
    if state.low[v] == state.idx[v].unwrap() {
        let mut comp = Vec::new();
        loop {
            let w = state.stack.pop().expect("stack underflow");
            state.on_stack[w] = false;
            comp.push(w);
            if w == v {
                break;
            }
        }
        state.comps.push(comp);
    }
}

/// Build a mapping from vertex to SCC identifier.
///
/// # Arguments
///
/// * `sccs` - List of SCCs from `tarjan_scc`
/// * `n` - Total number of vertices
///
/// # Returns
///
/// Vector where `result[v]` is the SCC id containing vertex `v`.
pub fn scc_id_map(sccs: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut comp_of = vec![usize::MAX; n];
    for (cid, comp) in sccs.iter().enumerate() {
        for &v in comp {
            comp_of[v] = cid;
        }
    }
    debug_assert!(comp_of.iter().all(|&x| x != usize::MAX));
    comp_of
}

/// Build the condensation DAG from SCCs.
///
/// Creates a directed acyclic graph where each node represents an SCC,
/// and edges represent dependencies between SCCs.
///
/// # Arguments
///
/// * `graph` - Original graph adjacency list
/// * `comp_of` - Mapping from vertex to SCC id (from `scc_id_map`)
/// * `ncomp` - Number of SCCs
///
/// # Returns
///
/// Adjacency list for the condensation DAG.
pub fn condensation_dag(graph: &[Vec<usize>], comp_of: &[usize], ncomp: usize) -> Vec<Vec<usize>> {
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
