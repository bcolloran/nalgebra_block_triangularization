# nalgebra_block_triangularization

Structural decomposition of sparse matrices into block upper-triangular form using graph algorithms.

## Overview

This library computes row and column permutations that reveal the block triangular structure of a sparse matrix. Given a matrix **M**, it finds permutation matrices **P** and **Q** such that:

**U = P M Q**

is block upper-triangular, where each diagonal block corresponds to a strongly connected component (SCC) in the dependency graph induced by a maximum matching.

## Why Block Triangularization?

When solving large systems of nonlinear equations where each equation involves only a subset of unknowns, decomposing the system into smaller sub-problems can dramatically improve:

- **Solvability**: Large coupled systems may fail where sequential smaller systems succeed
- **Robustness**: Reduces dimensionality and improves conditioning
- **Diagnostics**: Reveals structural properties (over/under-determined subsystems, algebraic loops)
- **Efficiency**: Solves independent blocks separately, converts hard problems into sequences of easier ones

This is particularly valuable for:
- Parameter estimation from experimental measurements
- Physical system modeling (DAE solvers, process simulation)
- Constraint satisfaction problems
- Sparse nonlinear optimization

## Algorithm

The implementation uses a well-established graph-theoretic approach:

1. **Maximum Bipartite Matching** (Hopcroft-Karp): Treat the matrix as a bipartite graph (rows ↔ columns) and find a maximum matching
2. **Row Dependency Graph**: Build a directed graph where row *i* → row *k* if row *i* has a nonzero in a column matched to row *k*
3. **Strongly Connected Components** (Tarjan): Compute SCCs of the dependency graph—each SCC is one diagonal block
4. **Topological Ordering**: Order the SCCs topologically to achieve upper-triangular block structure
5. **Permutation Sequences**: Convert the resulting row and column orders into `nalgebra::PermutationSequence` objects

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
nalgebra = "0.32"
nalgebra_block_triangularization = "0.1"
```

### Basic Example

```rust
use nalgebra::DMatrix;
use nalgebra_block_triangularization::upper_triangular_permutations;

// Create a sparse binary matrix (0/1 pattern)
let m = DMatrix::from_row_slice(8, 8, &[
    1, 0, 1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0,
    1, 1, 0, 1, 1, 0, 0, 0,
    1, 1, 0, 1, 1, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 1, 1, 0,
    1, 1, 1, 0, 0, 1, 1, 0,
    1, 1, 0, 0, 0, 0, 1, 1,
]);

// Compute permutations
let (pr, pc) = upper_triangular_permutations(&m);

// Apply to get block upper-triangular form
let mut u = m.clone();
pr.permute_rows(&mut u);
pc.permute_columns(&mut u);

println!("Block upper-triangular form:\n{}", u);
```

### Diagnostic Information

For more detailed structural information:

```rust
use nalgebra_block_triangularization::upper_block_triangular_structure;

let structure = upper_block_triangular_structure(&m);

println!("Matching size: {}", structure.matching_size);
println!("Number of blocks: {}", structure.block_sizes.len());
println!("Block sizes: {:?}", structure.block_sizes);
println!("Row ordering: {:?}", structure.row_order);
println!("Column ordering: {:?}", structure.col_order);
```

## Interpretation

The output provides:

- **block_sizes**: Size of each diagonal SCC block
- **matching_size**: Number of matched pairs (equations-variables)
- **row_order** / **col_order**: Permutation mappings (new position → original index)

### For Equation Systems

If your matrix represents equation-variable incidence (row *i* = equation *i*, column *j* = variable *j*, nonzero means variable appears in equation):

- **Small blocks**: Simple sub-problems that can be solved independently
- **Unmatched columns**: Free variables (degrees of freedom)
- **Unmatched rows**: Over-determined constraints
- **Block ordering**: Safe computation sequence (solve earlier blocks first)

## Implementation Details

The library is organized into focused modules:

- `adjacency`: Graph construction from matrix sparsity pattern
- `matching`: Hopcroft-Karp maximum bipartite matching
- `scc`: Tarjan's strongly connected components algorithm
- `ordering`: Topological sorting with deterministic tie-breaking
- `permutation`: Conversion to nalgebra permutation sequences

All algorithms operate purely on the structural sparsity pattern (nonzero vs. zero), not on numerical values.

## Related Concepts

This implementation is closely related to several well-established techniques:

- **Dulmage-Mendelsohn (DM) decomposition**: The standard bipartite graph decomposition for structural analysis
- **Block Triangular Form (BTF)**: Matrix reordering to block structure
- **Tearing**: Breaking algebraic loops by selecting tear variables
- **Causalization**: Determining solve order in DAE systems (used in Modelica, etc.)
- **Sequential Modular Method**: Process flowsheet simulation approach

## References

- Duff, I. S., Erisman, A. M., & Reid, J. K. (1986). *Direct Methods for Sparse Matrices*
- Dulmage, A. L., & Mendelsohn, N. S. (1958). Coverings of bipartite graphs. *Canadian Journal of Mathematics*
- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM
- Baharev, A., Schichl, H., & Neumaier, A. (2016). Decomposition methods for solving sparse nonlinear systems of equations
- Tarjan, R. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Areas for future development:

- Property-based testing with larger random matrices
- Bordered block triangular form (BBTF) for tear variable identification
- Support for rectangular and structurally singular matrices
- Integration with sparse matrix formats
- Performance benchmarks and optimizations
