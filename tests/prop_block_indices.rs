// Property-based tests for the block_indices method
use nalgebra::DMatrix;
use nalgebra_block_triangularization::upper_block_triangular_structure;
use proptest::prelude::*;

proptest! {
    /// Property: block_indices returns correct number of blocks
    #[test]
    fn block_indices_count_matches_block_sizes(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);
        let blocks = structure.block_indices();

        // Number of blocks should match block_sizes length
        prop_assert_eq!(blocks.len(), structure.block_sizes.len());
    }

    /// Property: Each block's size matches the corresponding entry in block_sizes
    #[test]
    fn block_indices_sizes_correct(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);
        let blocks = structure.block_indices();

        // Each block's dimensions should match block_sizes
        for (i, &expected_size) in structure.block_sizes.iter().enumerate() {
            prop_assert!(i < blocks.len(), "Missing block at index {}", i);
            let (rows, cols) = &blocks[i];
            prop_assert_eq!(rows.len(), expected_size,
                "Block {} row count mismatch", i);
            prop_assert_eq!(cols.len(), expected_size,
                "Block {} col count mismatch", i);
        }
    }

    /// Property: All row indices are covered exactly once
    #[test]
    fn block_indices_all_rows_covered_once(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);
        let blocks = structure.block_indices();

        // Collect all row indices from all blocks
        let collected_rows: Vec<usize> = blocks.iter()
            .flat_map(|(rows, _)| rows.clone())
            .collect();

        // Should equal row_order exactly (same elements in same order)
        prop_assert_eq!(collected_rows, structure.row_order);
    }

    /// Property: All column indices are covered exactly once
    #[test]
    fn block_indices_all_cols_covered_once(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);
        let blocks = structure.block_indices();

        // Collect all column indices from all blocks
        let collected_cols: Vec<usize> = blocks.iter()
            .flat_map(|(_, cols)| cols.clone())
            .collect();

        // Should equal col_order exactly (same elements in same order)
        prop_assert_eq!(collected_cols, structure.col_order);
    }

    /// Property: No duplicate indices within or across blocks
    #[test]
    fn block_indices_no_duplicates(
        bits in prop::collection::vec(any::<u8>(), 1..400)
    ) {
        let n = (bits.len() as f64).sqrt() as usize;
        if n < 1 {
            return Ok(());
        }
        let bits: Vec<u8> = bits.into_iter().take(n * n).map(|b| b % 2).collect();
        let m = DMatrix::from_row_slice(n, n, &bits);

        let structure = upper_block_triangular_structure(&m);
        let blocks = structure.block_indices();

        // Collect and check row indices for duplicates
        let mut seen_rows = std::collections::HashSet::new();
        for (rows, _) in &blocks {
            for &r in rows {
                prop_assert!(seen_rows.insert(r),
                    "Duplicate row index {} found", r);
            }
        }

        // Collect and check column indices for duplicates
        let mut seen_cols = std::collections::HashSet::new();
        for (_, cols) in &blocks {
            for &c in cols {
                prop_assert!(seen_cols.insert(c),
                    "Duplicate column index {} found", c);
            }
        }
    }
}
