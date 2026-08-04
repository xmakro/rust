// This test checks that a cached derive proc macro expansion is still reused when an edit only
// shifts the source position of the invocation (e.g. inserting an item above it). The cache key
// for `derive_macro_expansion` is span-agnostic, so the expansion is loaded from disk even though
// every span below the inserted item has moved.

//@ proc-macro:derive_nothing.rs
//@ revisions:rpass1 rpass2
//@ compile-flags: -Zquery-dep-graph -Zcache-proc-macros
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

#[macro_use]
extern crate derive_nothing;

// Only present in the second revision. It adds no new tokens to the `Foo` invocation below, but it
// shifts every following span (byte positions and line/column numbers).
#[cfg(rpass2)]
pub struct Padding {
    pub a: u32,
    pub b: u32,
}

#[rustc_clean(cfg = "rpass2", loaded_from_disk = "derive_macro_expansion")]
#[derive(Nothing)]
pub struct Foo;

fn main() {}
