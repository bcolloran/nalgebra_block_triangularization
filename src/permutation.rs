use nalgebra::{Dyn, PermutationSequence};

/// Convert an ordering to a nalgebra permutation sequence.
///
/// Takes an explicit ordering (new position → old index) and converts it
/// to a `PermutationSequence` via a sequence of transpositions.
///
/// # Arguments
///
/// * `order` - Permutation mapping where `order[i]` is the original index for position `i`
///
/// # Returns
///
/// A `PermutationSequence` that transforms the identity ordering into `order`.
///
/// # Panics
///
/// Panics in debug mode if `order` is not a valid permutation.
pub fn permutation_sequence_from_order(order: &[usize]) -> PermutationSequence<Dyn> {
    let n = order.len();
    let mut p = PermutationSequence::<Dyn>::identity(n);

    // Validate it is a permutation of 0..n-1 (debug-time check).
    debug_assert!(is_valid_permutation(order));

    let mut current: Vec<usize> = (0..n).collect(); // position -> element
    let mut pos_of: Vec<usize> = (0..n).collect(); // element -> position

    for i in 0..n {
        let desired = order[i];
        let j = pos_of[desired];
        if i != j {
            // Swap positions i and j.
            p.append_permutation(i, j);

            let a = current[i];
            let b = current[j];
            current.swap(i, j);
            pos_of[a] = j;
            pos_of[b] = i;
        }
    }

    p
}

fn is_valid_permutation(order: &[usize]) -> bool {
    let n = order.len();
    let mut seen = vec![false; n];
    for &x in order {
        if x >= n || seen[x] {
            return false;
        }
        seen[x] = true;
    }
    seen.iter().all(|&x| x)
}
