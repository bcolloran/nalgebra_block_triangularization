use nalgebra::{Dyn, PermutationSequence};

/// Convert an explicit order (new_pos -> old_index) into a nalgebra PermutationSequence<Dyn>
/// via a minimal-ish sequence of swaps.
///
/// This generates swaps that transform [0,1,2,..] into `order`.
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
