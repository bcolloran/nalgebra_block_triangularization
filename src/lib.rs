pub mod adjacency;
pub mod matching;
pub mod ordering;
pub mod permutation;
pub mod scc;

use nalgebra::{Dyn, Matrix, PermutationSequence, Scalar, Storage};

use adjacency::{build_row_adjacency, build_row_dependency_graph};
use matching::hopcroft_karp;
use ordering::{col_order_from_row_order, topo_sort_with_tiebreak};
use permutation::permutation_sequence_from_order;
use scc::{condensation_dag, scc_id_map, tarjan_scc};

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
