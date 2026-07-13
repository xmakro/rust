// An identical-type coercion of a `Reborrow` ADT still inserts the reborrow:
// `method::<'a>(a)` reborrows `a` rather than moving it, even though source and
// target are spelled with the same region.

#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a mut ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a: 'a>(_: CustomMarker<'a>) {}

fn use_twice<'a>(a: CustomMarker<'a>) {
    method::<'a>(a); //~ ERROR `a` does not live long enough
    method::<'a>(a); //~ ERROR cannot borrow `a` as mutable more than once at a time
}

fn main() {
    use_twice(CustomMarker(PhantomData));
}
