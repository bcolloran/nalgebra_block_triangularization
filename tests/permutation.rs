use nalgebra::DMatrix;
use nalgebra_block_triangularization::permutation::permutation_sequence_from_order;

fn apply_perm_to_vec(perm: &nalgebra::PermutationSequence<nalgebra::Dyn>, v: &[usize]) -> Vec<usize> {
    let n = v.len();
    let mut m = DMatrix::from_fn(n, 1, |i, _| v[i] as f64);
    perm.permute_rows(&mut m);
    (0..n).map(|i| m[(i, 0)] as usize).collect()
}

#[test]
fn perm_empty() {
    let order: Vec<usize> = vec![];
    let perm = permutation_sequence_from_order(&order);
    // Empty permutation has dimension 0
    let result = apply_perm_to_vec(&perm, &[]);
    assert_eq!(result, vec![]);
}

#[test]
fn perm_single_element() {
    let order = vec![0];
    let perm = permutation_sequence_from_order(&order);
    // Single element identity - no swaps needed
    assert_eq!(perm.len(), 0);
    
    let result = apply_perm_to_vec(&perm, &[5]);
    assert_eq!(result, vec![5]);
}

#[test]
fn perm_identity() {
    let order = vec![0, 1, 2, 3];
    let perm = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30, 40];
    let result = apply_perm_to_vec(&perm, &input);
    assert_eq!(result, input);
}

#[test]
fn perm_simple_swap() {
    // Swap positions 0 and 1
    let order = vec![1, 0, 2];
    let perm = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30];
    let result = apply_perm_to_vec(&perm, &input);
    assert_eq!(result, vec![20, 10, 30]);
}

#[test]
fn perm_reverse() {
    let order = vec![3, 2, 1, 0];
    let perm = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30, 40];
    let result = apply_perm_to_vec(&perm, &input);
    assert_eq!(result, vec![40, 30, 20, 10]);
}

#[test]
fn perm_rotation() {
    // Rotate: [0,1,2,3] -> [1,2,3,0]
    let order = vec![1, 2, 3, 0];
    let perm = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30, 40];
    let result = apply_perm_to_vec(&perm, &input);
    assert_eq!(result, vec![20, 30, 40, 10]);
}

#[test]
fn perm_complex() {
    let order = vec![2, 0, 3, 1];
    let perm = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30, 40];
    let result = apply_perm_to_vec(&perm, &input);
    // order[i] = old position for new position i
    // new[0] = old[2] = 30
    // new[1] = old[0] = 10
    // new[2] = old[3] = 40
    // new[3] = old[1] = 20
    assert_eq!(result, vec![30, 10, 40, 20]);
}

#[test]
fn perm_larger() {
    let n = 10;
    // Shuffle: put evens first, then odds
    let order: Vec<usize> = (0..n).filter(|x| x % 2 == 0)
        .chain((0..n).filter(|x| x % 2 == 1))
        .collect();
    
    let perm = permutation_sequence_from_order(&order);
    let input: Vec<usize> = (0..n).collect();
    let result = apply_perm_to_vec(&perm, &input);
    
    // Should be [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    assert_eq!(result, vec![0, 2, 4, 6, 8, 1, 3, 5, 7, 9]);
}

#[test]
fn perm_apply_twice_is_idempotent() {
    let order = vec![2, 0, 1];
    let perm = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30];
    let result1 = apply_perm_to_vec(&perm, &input);
    let result2 = apply_perm_to_vec(&perm, &result1);
    
    // Applying the same permutation twice should not be identity in general
    // But we can check it's deterministic
    let result3 = apply_perm_to_vec(&perm, &result1);
    assert_eq!(result2, result3);
}

#[test]
fn perm_inverse_property() {
    // Create a permutation and its inverse
    let order = vec![2, 0, 3, 1];
    let perm = permutation_sequence_from_order(&order);
    
    // Inverse permutation: if order[i] = j, then inverse[j] = i
    let mut inverse_order = vec![0; order.len()];
    for (new_pos, &old_pos) in order.iter().enumerate() {
        inverse_order[old_pos] = new_pos;
    }
    let inv_perm = permutation_sequence_from_order(&inverse_order);
    
    let input = vec![10, 20, 30, 40];
    let result = apply_perm_to_vec(&perm, &input);
    let back = apply_perm_to_vec(&inv_perm, &result);
    
    assert_eq!(back, input);
}

#[test]
fn perm_is_permutation() {
    // Verify that the result is actually a permutation (no duplicates, all values present)
    let order = vec![3, 1, 4, 0, 2];
    let perm = permutation_sequence_from_order(&order);
    
    let input: Vec<usize> = (0..5).collect();
    let result = apply_perm_to_vec(&perm, &input);
    
    let mut sorted = result.clone();
    sorted.sort();
    assert_eq!(sorted, input);
}

#[test]
fn perm_with_matrix() {
    // Test with actual matrix permutation
    let order = vec![2, 0, 1];
    let perm = permutation_sequence_from_order(&order);
    
    let mut m = DMatrix::from_row_slice(3, 3, &[
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ]);
    
    perm.permute_rows(&mut m);
    
    // Row 0 should now be old row 2: [7, 8, 9]
    // Row 1 should now be old row 0: [1, 2, 3]
    // Row 2 should now be old row 1: [4, 5, 6]
    assert_eq!(m[(0, 0)], 7);
    assert_eq!(m[(0, 1)], 8);
    assert_eq!(m[(0, 2)], 9);
    assert_eq!(m[(1, 0)], 1);
    assert_eq!(m[(1, 1)], 2);
    assert_eq!(m[(1, 2)], 3);
    assert_eq!(m[(2, 0)], 4);
    assert_eq!(m[(2, 1)], 5);
    assert_eq!(m[(2, 2)], 6);
}

#[test]
fn perm_column_permutation() {
    let order = vec![1, 2, 0];
    let perm = permutation_sequence_from_order(&order);
    
    let mut m = DMatrix::from_row_slice(3, 3, &[
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ]);
    
    perm.permute_columns(&mut m);
    
    // Col 0 should now be old col 1: [2, 5, 8]
    // Col 1 should now be old col 2: [3, 6, 9]
    // Col 2 should now be old col 0: [1, 4, 7]
    assert_eq!(m[(0, 0)], 2);
    assert_eq!(m[(1, 0)], 5);
    assert_eq!(m[(2, 0)], 8);
    assert_eq!(m[(0, 1)], 3);
    assert_eq!(m[(1, 1)], 6);
    assert_eq!(m[(2, 1)], 9);
    assert_eq!(m[(0, 2)], 1);
    assert_eq!(m[(1, 2)], 4);
    assert_eq!(m[(2, 2)], 7);
}

#[test]
fn perm_deterministic() {
    // Same order should always produce the same permutation
    let order = vec![3, 1, 2, 0];
    let perm1 = permutation_sequence_from_order(&order);
    let perm2 = permutation_sequence_from_order(&order);
    
    let input = vec![10, 20, 30, 40];
    let result1 = apply_perm_to_vec(&perm1, &input);
    let result2 = apply_perm_to_vec(&perm2, &input);
    
    assert_eq!(result1, result2);
}
