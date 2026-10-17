// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// kani-flags: -Z loop-contracts

//! Check that the initialization of a variable declared in a loop body by a function call is not
//! executed at the loop head. It used to be copied there, where it read body locals that are not
//! assigned yet and ran the call (and its side effects) before the loop guard
//! (https://github.com/model-checking/kani/issues/4949).

#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]

use std::cmp::Ordering;

fn search<F: FnMut(&u8) -> Ordering>(s: &[u8], mut f: F) -> usize {
    let len = s.len();
    if len == 0 {
        return 0;
    }
    let mut size = len;
    let mut base = 0usize;
    #[kani::loop_invariant(1 <= size && size <= len && base <= len - size)]
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        // `cmp` is borrowed by `==`, so it keeps its storage markers. Its initialization reads
        // `mid`, which the loop head does not assign.
        let cmp = f(unsafe { s.get_unchecked(mid) });
        base = if cmp == Ordering::Greater { base } else { mid };
        size -= half;
    }
    base
}

#[kani::proof]
fn binary_search() {
    let a: [u8; 8] = kani::any();
    let s: &[u8] = kani::slice::any_slice_of_array(&a);
    let _ = search(s, |_| if kani::any() { Ordering::Less } else { Ordering::Greater });
}

fn search_with_modifies<F: FnMut(&u8) -> Ordering>(s: &[u8], mut f: F) -> usize {
    let len = s.len();
    if len == 0 {
        return 0;
    }
    let mut size = len;
    let mut base = 0usize;
    #[kani::loop_invariant(1 <= size && size <= len && base <= len - size)]
    #[kani::loop_modifies(&base, &size)]
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        let cmp = f(unsafe { s.get_unchecked(mid) });
        base = if cmp == Ordering::Greater { base } else { mid };
        size -= half;
    }
    base
}

/// The loop does not write any variable declared before it other than `base` and `size`.
#[kani::proof]
fn binary_search_with_modifies() {
    let a: [u8; 8] = kani::any();
    let s: &[u8] = kani::slice::any_slice_of_array(&a);
    let _ = search_with_modifies(s, |_| if kani::any() { Ordering::Less } else { Ordering::Greater });
}

fn take(n: &mut usize) -> Option<usize> {
    if *n > 0 {
        *n -= 1;
        Some(*n)
    } else {
        None
    }
}

/// `take` has a side effect: running it once more at the loop head made the invariant false
/// before the loop (https://github.com/model-checking/kani/issues/4796).
#[kani::proof]
fn call_with_side_effect() {
    let n0: usize = kani::any();
    let mut n = n0;
    let mut i: usize = 0;
    #[kani::loop_invariant(i <= n0 && n == n0 - i)]
    while i < n0 {
        let v = take(&mut n);
        assert!(v.is_some());
        i += 1;
    }
}
