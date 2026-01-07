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
