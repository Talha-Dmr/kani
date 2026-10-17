// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// kani-flags: -Z loop-contracts

//! Check that no code of a loop body runs at the head of the loop, or at the head of an enclosing
//! loop, before the loop guard is evaluated. Each harness must fail: the code used to run there
//! once more, which hid the failure (https://github.com/model-checking/kani/issues/4982).

#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]

fn bump(c: &mut u32) -> u32 {
    *c += 1;
    *c
}

/// A side effect of a call in the body ran before the loop: the invariant held at entry.
#[kani::proof]
fn head_copy_side_effect() {
    let mut c: u32 = 0;
    let mut i: u32 = 0;
    #[kani::loop_invariant(i <= 3 && c == i.wrapping_add(1))]
    while i < 3 {
        let x = bump(&mut c);
        let _ = x;
        i += 1;
    }
    assert!(c == 4);
}

fn checked_get(v: &[u32; 4], k: usize) -> u32 {
    kani::assume(k < 4);
    if k < 4 { v[k] } else { 0 }
}

/// An `assume` in a call in the body ruled out the inputs for which the loop does not run.
#[kani::proof]
fn head_copy_assume_prunes() {
    let v = [1u32, 2, 3, 4];
    let k: usize = kani::any();
    let mut i: u32 = 0;
    #[kani::loop_invariant(i <= 1)]
    while i < 1 && k < 4 {
        let e = checked_get(&v, k);
        let _ = e;
        i += 1;
    }
    assert!(k < 4);
}

fn idx(v: &[u32; 4], k: usize) -> u32 {
    v[k]
}

/// A panic in the body for inputs that never enter the loop satisfied `should_panic`.
#[kani::proof]
#[kani::should_panic]
fn head_copy_panic_as_expected() {
    let v = [1u32, 2, 3, 4];
    let k: usize = kani::any();
    let mut i: u32 = 0;
    #[kani::loop_invariant(i <= 1)]
    while i < 1 && k < 4 {
        let e = idx(&v, k);
        let _ = e;
        i += 1;
    }
}

/// The range of a nested `for` loop was computed once, from the initial value of `outer`.
#[kani::proof]
fn nested_for_dependent_range() {
    let mut count: u32 = 0;
    let mut outer: u32 = 0;
    #[kani::loop_invariant(outer <= 3 && count == 0)]
    while outer < 3 {
        #[kani::loop_invariant(count == 0)]
        for _j in 0..outer {
            count += 1;
        }
        outer += 1;
    }
    assert!(count == 0);
}

/// Same, with the bound written through a reference.
#[kani::proof]
fn nested_for_range_written_through_reference() {
    let mut n: u32 = 0;
    let k = &mut n;
    let mut count: u32 = 0;
    let mut i: u32 = 0;
    #[kani::loop_invariant(i <= 3 && count == 0)]
    while i < 3 {
        *k = (*k).wrapping_add(1);
        #[kani::loop_invariant(count == 0)]
        for _ in 0..*k {
            count = count.wrapping_add(1);
        }
        i += 1;
    }
    assert!(count == 0);
}

fn range_upto(k: usize) -> core::ops::Range<usize> {
    kani::assume(k < 4);
    0..k
}

/// An `assume` in the iterator expression of a nested `for` loop ruled out the inputs for which
/// the enclosing loop does not run.
#[kani::proof]
fn nested_for_iterator_assume() {
    let k: usize = kani::any();
    let mut i: u32 = 0;
    #[kani::loop_invariant(i <= 1)]
    while i < 1 && k < 4 {
        #[kani::loop_invariant(kani::index <= k)]
        for _ in range_upto(k) {}
        i += 1;
    }
    assert!(k < 4);
}
