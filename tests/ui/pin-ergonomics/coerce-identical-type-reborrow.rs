// An identical-type coercion of `Pin<&mut T>` still inserts the pin reborrow:
// `take(p)` reborrows `*p` rather than moving `p`, even when the argument type
// is spelled with the same region as the value.

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;

struct Foo;

fn take(_: Pin<&'static mut Foo>) {}

fn main() {
    let p: Pin<&'static mut Foo> = Pin::new(Box::leak(Box::new(Foo)));
    take(p);
    take(p); //~ ERROR cannot borrow `*p.pointer` as mutable more than once at a time
}
