// Unit tests for the block_indices method
use nalgebra::DMatrix;
use nalgebra_block_triangularization::upper_block_triangular_structure;

#[test]
fn block_indices_empty_structure() {
    let m = DMatrix::<u8>::zeros(0, 0);
    let structure = upper_block_triangular_structure(&m);
    let blocks = structure.block_indices();

    // Empty matrix should have no blocks
    assert_eq!(blocks.len(), 0);
}

#[test]
fn block_indices_single_block() {
    // A cyclic pattern forms a single strongly connected component
    let m = DMatrix::from_row_slice(3, 3, &[1, 1, 0, 0, 1, 1, 1, 0, 1]);
    let structure = upper_block_triangular_structure(&m);
    let blocks = structure.block_indices();

    // Should have exactly one block (all rows in one SCC)
    assert_eq!(blocks.len(), 1);

    // Block should contain all 3 rows and 3 cols
    let (rows, cols) = &blocks[0];
    assert_eq!(rows.len(), 3);
    assert_eq!(cols.len(), 3);
}

#[test]
fn block_indices_multiple_blocks() {
    // Block diagonal matrix with 2 blocks
    let m = DMatrix::from_row_slice(4, 4, &[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]);
    let structure = upper_block_triangular_structure(&m);
    let blocks = structure.block_indices();

    // Should have 2 blocks
    assert_eq!(blocks.len(), 2);

    // Each block should have size 2
    assert_eq!(structure.block_sizes, vec![2, 2]);
    for (rows, cols) in &blocks {
        assert_eq!(rows.len(), 2);
        assert_eq!(cols.len(), 2);
    }
}

#[test]
fn block_indices_all_indices_covered() {
    let m = DMatrix::from_row_slice(
        5,
        5,
        &[
            1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
        ],
    );
    let structure = upper_block_triangular_structure(&m);
    let blocks = structure.block_indices();

    // Collect all row indices from blocks
    let mut all_row_indices: Vec<usize> =
        blocks.iter().flat_map(|(rows, _)| rows.clone()).collect();
    all_row_indices.sort_unstable();

    // Collect all col indices from blocks
    let mut all_col_indices: Vec<usize> =
        blocks.iter().flat_map(|(_, cols)| cols.clone()).collect();
    all_col_indices.sort_unstable();

    // Concatenated blocks should equal row_order and col_order
    let mut expected_row = structure.row_order.clone();
    expected_row.sort_unstable();
    let mut expected_col = structure.col_order.clone();
    expected_col.sort_unstable();

    assert_eq!(all_row_indices, expected_row);
    assert_eq!(all_col_indices, expected_col);
}

#[test]
fn block_indices_sizes_match_block_sizes() {
    let m = DMatrix::from_row_slice(
        6,
        6,
        &[
            1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
            1, 0, 0, 0, 1, 1, 1,
        ],
    );
    let structure = upper_block_triangular_structure(&m);
    let blocks = structure.block_indices();

    // Number of blocks should match block_sizes length
    assert_eq!(blocks.len(), structure.block_sizes.len());

    // Each block's size should match the corresponding entry in block_sizes
    for (i, &size) in structure.block_sizes.iter().enumerate() {
        let (rows, cols) = &blocks[i];
        assert_eq!(rows.len(), size, "Block {} row size mismatch", i);
        assert_eq!(cols.len(), size, "Block {} col size mismatch", i);
    }
}

#[test]
fn block_indices_preserves_order() {
    let m = DMatrix::from_row_slice(4, 4, &[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]);
    let structure = upper_block_triangular_structure(&m);
    let blocks = structure.block_indices();

    // Concatenate blocks to verify they match row_order and col_order in sequence
    let reconstructed_row_order: Vec<usize> =
        blocks.iter().flat_map(|(rows, _)| rows.clone()).collect();
    let reconstructed_col_order: Vec<usize> =
        blocks.iter().flat_map(|(_, cols)| cols.clone()).collect();

    assert_eq!(reconstructed_row_order, structure.row_order);
    assert_eq!(reconstructed_col_order, structure.col_order);
}
