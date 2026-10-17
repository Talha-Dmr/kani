// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// kani-flags: -Z loop-contracts

//! Check variables of the body of a loop with a nested loop.

#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]

#[inline(never)]
fn id(x: u32) -> u32 {
    x
}

fn bump(t: &mut u32) {
    *t = t.wrapping_add(1);
}

#[inline(never)]
fn as_mut(x: &mut u32) -> &mut u32 {
    x
}

#[inline(never)]
fn as_raw(x: *mut u32) -> *mut u32 {
    x
}

/// A variable declared in the body of the nested loop does not have to be in the
/// `loop_modifies` clause of the enclosing loop, where it cannot be named.
#[kani::proof]
fn outer_modifies_inner_body_local() {
    let mut a = [0u32; 2];
    let mut i: usize = 0;
    let mut j: usize = 0;
    #[kani::loop_invariant(i <= 2 && j <= 2)]
    #[kani::loop_modifies(&i, &j, &a)]
    while i < 2 {
        j = 0;
        #[kani::loop_invariant(j <= 2 && i < 2)]
        while j < 2 {
            let t = id(a[j]);
            a[j] = t.wrapping_add(1);
            j += 1;
        }
        i += 1;
    }
}

/// `acc` is written by the nested loop, so it is havocked at the head of the enclosing loop.
/// Its storage ends before the end of the enclosing loop body, so that havoc must not write a
/// dead object.
#[kani::proof]
fn scoped_address_taken_local() {
    let mut i: u32 = 0;
    let mut total: u32 = 0;
    #[kani::loop_invariant(i <= 2)]
    while i < 2 {
        {
            let mut acc: u32 = id(0);
            let mut j: u32 = 0;
            #[kani::loop_invariant(j <= 2)]
            while j < 2 {
                bump(&mut acc);
                j += 1;
            }
            total = total.wrapping_add(acc);
        }
        i = i.wrapping_add(1);
    }
    let _ = total;
}

/// The nested loop writes `x` only through a reference that a call in the enclosing loop body
/// returns.
#[kani::proof]
fn write_through_returned_reference() {
    let mut i: u32 = 0;
    let mut total: u32 = 0;
    #[kani::loop_invariant(i <= 2)]
    while i < 2 {
        let mut x: u32 = id(0);
        let p = as_mut(&mut x);
        let mut j: u32 = 0;
        #[kani::loop_invariant(j <= 2)]
        while j < 2 {
            *p = (*p).wrapping_add(1);
            j += 1;
        }
        total = total.wrapping_add(x);
        i += 1;
    }
    let _ = total;
}

/// The same with a raw pointer.
#[kani::proof]
fn write_through_returned_raw_pointer() {
    let mut i: u32 = 0;
    let mut total: u32 = 0;
    #[kani::loop_invariant(i <= 2)]
    while i < 2 {
        let mut x: u32 = id(0);
        let p = as_raw(&raw mut x);
        let mut j: u32 = 0;
        #[kani::loop_invariant(j <= 2)]
        while j < 2 {
            unsafe { *p = (*p).wrapping_add(1) };
            j += 1;
        }
        total = total.wrapping_add(x);
        i += 1;
    }
    let _ = total;
}
