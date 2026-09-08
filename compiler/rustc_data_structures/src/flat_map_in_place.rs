use std::{mem, ptr};

use smallvec::SmallVec;
use thin_vec::ThinVec;

pub trait FlatMapInPlace<T> {
    /// `f` turns each element into 0..many elements. This function will consume the existing
    /// elements in a vec-like structure and replace them with any number of new elements — fewer,
    /// more, or the same number — as efficiently as possible.
    fn flat_map_in_place<F, I>(&mut self, f: F)
    where
        F: FnMut(T) -> I,
        I: IntoIterator<Item = T>;
}

// Blanket impl for all vec-like types that impl `FlatMapInPlaceVec`.
impl<V: FlatMapInPlaceVec> FlatMapInPlace<V::Elem> for V {
    fn flat_map_in_place<F, I>(&mut self, mut f: F)
    where
        F: FnMut(V::Elem) -> I,
        I: IntoIterator<Item = V::Elem>,
    {
        struct LeakGuard<'a, V: FlatMapInPlaceVec>(&'a mut V);

        impl<'a, V: FlatMapInPlaceVec> Drop for LeakGuard<'a, V> {
            fn drop(&mut self) {
                unsafe {
                    // Leak all elements in case of panic.
                    self.0.set_len(0);
                }
            }
        }

        let guard = LeakGuard(self);

        let mut read_i = 0;
        let mut write_i = 0;
        unsafe {
            while read_i < guard.0.len() {
                // Move the read_i'th item out of the vector and map it to an iterator.
                let e = ptr::read(guard.0.as_ptr().add(read_i));
                let mut iter = f(e).into_iter();
                read_i += 1;

                while let Some(e) = iter.next() {
                    if write_i < read_i {
                        ptr::write(guard.0.as_mut_ptr().add(write_i), e);
                        write_i += 1;
                    } else {
                        // All holes are filled, so the vector is valid here. Insert the
                        // remaining output together to avoid shifting the unvisited tail
                        // separately for every element of a large expansion.
                        let old_len = guard.0.len();
                        guard.0.insert_many(write_i, std::iter::once(e).chain(iter));
                        let inserted = guard.0.len() - old_len;
                        read_i += inserted;
                        write_i += inserted;
                        break;
                    }
                }
            }

            // `write_i` tracks the number of actually written new items.
            guard.0.set_len(write_i);

            // `vec` is in a sane state again. Prevent the LeakGuard from leaking the data.
            mem::forget(guard);
        }
    }
}

/// A vec-like type must implement these operations to support `flat_map_in_place`.
///
/// # Safety
///
/// The memory safety of the unsafe block in `flat_map_in_place` relies on impls of this trait
/// implementing all the operations correctly.
pub unsafe trait FlatMapInPlaceVec {
    type Elem;

    fn len(&self) -> usize;
    unsafe fn set_len(&mut self, len: usize);
    fn as_ptr(&self) -> *const Self::Elem;
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;
    fn insert_many<I: Iterator<Item = Self::Elem>>(&mut self, idx: usize, elems: I);
}

unsafe impl<T> FlatMapInPlaceVec for Vec<T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn set_len(&mut self, len: usize) {
        unsafe {
            self.set_len(len);
        }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mut_ptr()
    }

    fn insert_many<I: Iterator<Item = T>>(&mut self, idx: usize, elems: I) {
        drop(self.splice(idx..idx, elems));
    }
}

unsafe impl<T> FlatMapInPlaceVec for ThinVec<T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn set_len(&mut self, len: usize) {
        unsafe {
            self.set_len(len);
        }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_slice().as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mut_slice().as_mut_ptr()
    }

    fn insert_many<I: Iterator<Item = T>>(&mut self, idx: usize, elems: I) {
        drop(self.splice(idx..idx, elems));
    }
}

unsafe impl<T, const N: usize> FlatMapInPlaceVec for SmallVec<[T; N]> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn set_len(&mut self, len: usize) {
        unsafe {
            self.set_len(len);
        }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mut_ptr()
    }

    fn insert_many<I: Iterator<Item = T>>(&mut self, idx: usize, elems: I) {
        self.insert_many(idx, elems);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_with_flat_map<V>()
    where
        V: FlatMapInPlace<u32> + FromIterator<u32> + AsRef<[u32]>,
    {
        for len in 0..20 {
            for phase in 0..6 {
                let expand = |x| (0..(x + phase) % 6).map(move |i| x * 100 + i);
                let expected: Vec<_> = (0..len).flat_map(expand).collect();
                let mut actual: V = (0..len).collect();
                let mut visited = Vec::new();
                actual.flat_map_in_place(|x| {
                    visited.push(x);
                    expand(x)
                });
                assert_eq!(visited, (0..len).collect::<Vec<_>>());
                assert_eq!(actual.as_ref(), expected);

                // A filter hides the exact length from bulk insertion.
                let mut actual: V = (0..len).collect();
                actual.flat_map_in_place(|x| expand(x).filter(|_| true));
                assert_eq!(actual.as_ref(), expected);
            }
        }
        // Include a large expansion before a long unvisited tail.
        let mut actual: V = (0..100).collect();
        actual.flat_map_in_place(|x| 0..if x == 0 { 1000 } else { 1 });
        assert_eq!(actual.as_ref().len(), 1099);
        assert_eq!(&actual.as_ref()[..1000], (0..1000).collect::<Vec<_>>());
        assert!(actual.as_ref()[1000..].iter().all(|&x| x == 0));
    }

    #[test]
    fn preserves_values_and_visit_order_for_all_containers() {
        compare_with_flat_map::<Vec<u32>>();
        compare_with_flat_map::<ThinVec<u32>>();
        compare_with_flat_map::<SmallVec<[u32; 4]>>();
    }
}
