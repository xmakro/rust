#[collapse_debuginfo(yes)]
macro_rules! mk_stat {
    () => {
        pub static MSTAT: u32 = 41;
    };
}
mk_stat! {}
// padxxxxx
fn main() {
    std::hint::black_box(&MSTAT);
}
