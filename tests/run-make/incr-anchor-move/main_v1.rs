// Swapping `alpha` and `omega` between the two versions of this file moves both functions
// without changing the file's length or line structure: the line table and both functions'
// contents are identical across versions, only their positions swap. The `#[track_caller]`
// lines and debuginfo baked into cached object code must still be refreshed, which is what
// the `def_position` dependency exists for.
#![allow(dead_code)]
mod inner {
    #[track_caller]
    fn line_of_caller() -> u32 {
        std::panic::Location::caller().line()
    }
    pub fn alpha() -> u32 {
        line_of_caller()
    }
    pub fn omega() -> u32 {
        line_of_caller()
    }
}
fn main() {
    assert_eq!(inner::alpha(), 13);
    assert_eq!(inner::omega(), 16);
}
