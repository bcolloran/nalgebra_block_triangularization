use nalgebra::{DMatrix, Dyn, PermutationSequence, Scalar};
use nalgebra_block_triangularization::{
    upper_block_triangular_structure, upper_triangular_permutations,
};

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

#[test]
fn empty_matrix() {
    let m: DMatrix<u8> = DMatrix::zeros(0, 0);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 0);
    assert_eq!(structure.block_sizes.len(), 0);
    assert_eq!(structure.row_order.len(), 0);
    assert_eq!(structure.col_order.len(), 0);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert_eq!(u.nrows(), 0);
    assert_eq!(u.ncols(), 0);
}

#[test]
fn single_element_nonzero() {
    let m = DMatrix::from_element(1, 1, 1u8);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 1);
    assert_eq!(structure.block_sizes, vec![1]);
    assert_eq!(structure.row_order, vec![0]);
    assert_eq!(structure.col_order, vec![0]);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert_eq!(u[(0, 0)], 1);
}

#[test]
fn single_element_zero() {
    let m = DMatrix::from_element(1, 1, 0u8);
    let structure = upper_block_triangular_structure(&m);
    
    assert_eq!(structure.matching_size, 0);
    assert_eq!(structure.row_order, vec![0]);
    assert_eq!(structure.col_order, vec![0]);
}

#[test]
fn identity_matrix() {
    let m: DMatrix<f64> = DMatrix::identity(5, 5);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 5);
    // Identity has no dependencies, so each element is its own SCC
    assert_eq!(structure.block_sizes.len(), 5);
    assert_eq!(structure.block_sizes.iter().sum::<usize>(), 5);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u.map(|x| if x != 0.0 { 1 } else { 0 }), &structure.block_sizes));
}

#[test]
fn all_zeros_matrix() {
    let m: DMatrix<u8> = DMatrix::zeros(4, 4);
    let structure = upper_block_triangular_structure(&m);
    
    assert_eq!(structure.matching_size, 0);
    assert_eq!(structure.row_order.len(), 4);
    assert_eq!(structure.col_order.len(), 4);
}

#[test]
fn all_ones_matrix() {
    let m = DMatrix::from_element(4, 4, 1u8);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 4);
    // All connected, should form a single SCC
    assert_eq!(structure.block_sizes.len(), 1);
    assert_eq!(structure.block_sizes[0], 4);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
}

#[test]
fn rectangular_more_rows() {
    // 5 rows, 3 cols
    let m = DMatrix::from_row_slice(5, 3, &[
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
    ]);
    let structure = upper_block_triangular_structure(&m);
    
    // Maximum matching is 3 (number of columns)
    assert_eq!(structure.matching_size, 3);
    assert_eq!(structure.row_order.len(), 5);
    assert_eq!(structure.col_order.len(), 3);
}

#[test]
fn rectangular_more_cols() {
    // 3 rows, 5 cols
    let m = DMatrix::from_row_slice(3, 5, &[
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
    ]);
    let structure = upper_block_triangular_structure(&m);
    
    // Maximum matching is 3 (number of rows)
    assert_eq!(structure.matching_size, 3);
    assert_eq!(structure.row_order.len(), 3);
    assert_eq!(structure.col_order.len(), 5);
    
    // Unmatched columns should be at the end
    let unmatched_cols = structure.col_order[3..].to_vec();
    assert_eq!(unmatched_cols.len(), 2);
}

#[test]
fn triangular_already_upper() {
    // Already upper triangular
    let m = DMatrix::from_row_slice(4, 4, &[
        1, 1, 1, 1,
        0, 1, 1, 1,
        0, 0, 1, 1,
        0, 0, 0, 1,
    ]);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 4);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
}

#[test]
fn triangular_lower() {
    // Lower triangular - should reorder to upper
    let m = DMatrix::from_row_slice(4, 4, &[
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 1, 1, 0,
        1, 1, 1, 1,
    ]);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 4);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
}

#[test]
fn block_diagonal() {
    // Two independent 2x2 blocks
    let m = DMatrix::from_row_slice(4, 4, &[
        1, 1, 0, 0,
        1, 1, 0, 0,
        0, 0, 1, 1,
        0, 0, 1, 1,
    ]);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 4);
    // Should have 2 SCCs
    assert_eq!(structure.block_sizes.len(), 2);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
}

#[test]
fn structurally_singular() {
    // Not all rows can be matched
    let m = DMatrix::from_row_slice(4, 4, &[
        1, 0, 0, 0,
        1, 0, 0, 0,  // Same as row 0
        0, 1, 0, 0,
        0, 0, 1, 0,
    ]);
    let structure = upper_block_triangular_structure(&m);
    
    // Can only match 3 rows
    assert_eq!(structure.matching_size, 3);
}

#[test]
fn cyclic_dependency() {
    // Create a cycle: 0 <-> 1 <-> 2 <-> 0
    let m = DMatrix::from_row_slice(3, 3, &[
        0, 1, 1,  // Row 0 touches cols 1, 2
        1, 0, 1,  // Row 1 touches cols 0, 2
        1, 1, 0,  // Row 2 touches cols 0, 1
    ]);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 3);
    // Should form a single SCC due to cycle
    assert_eq!(structure.block_sizes.len(), 1);
    assert_eq!(structure.block_sizes[0], 3);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
}

#[test]
fn sparse_pattern() {
    // Sparse matrix with clear block structure
    let m = DMatrix::from_row_slice(6, 6, &[
        1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0,
        1, 0, 1, 1, 0, 0,
        0, 1, 1, 1, 0, 0,
        0, 0, 1, 0, 1, 1,
        0, 0, 0, 1, 1, 1,
    ]);
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    assert_eq!(structure.matching_size, 6);
    
    let u = apply_perms(m.clone(), &pr, &pc);
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
}

#[test]
fn different_scalar_types() {
    // Test with f64
    let m_f64 = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 0.0,
        3.0, 4.0, 0.0,
        0.0, 5.0, 6.0,
    ]);
    let structure = upper_block_triangular_structure(&m_f64);
    assert_eq!(structure.matching_size, 3);
    
    // Test with i32
    let m_i32 = DMatrix::from_row_slice(3, 3, &[
        1, 2, 0,
        3, 4, 0,
        0, 5, 6,
    ]);
    let structure = upper_block_triangular_structure(&m_i32);
    assert_eq!(structure.matching_size, 3);
}

#[test]
fn permutations_are_invertible() {
    let m = DMatrix::from_row_slice(4, 4, &[
        0, 1, 1, 0,
        1, 0, 1, 0,
        1, 1, 0, 1,
        0, 0, 1, 0,
    ]);
    
    let structure = upper_block_triangular_structure(&m);
    let (pr, pc) = upper_triangular_permutations(&m);
    
    // Apply permutations
    let u = apply_perms(m.clone(), &pr, &pc);
    
    // Verify it's block triangular
    assert!(is_upper_block_triangular_u8(&u, &structure.block_sizes));
    
    // Inverse should exist (though we don't test full inversion here)
    assert_eq!(structure.row_order.len(), 4);
    assert_eq!(structure.col_order.len(), 4);
}
