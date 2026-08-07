#[collapse_debuginfo(yes)]
macro_rules! mk_stat {
    () => {
        pub static MSTAT: u32 = 41;
    };
}
// padxxxxx
mk_stat! {}
fn main() {
    std::hint::black_box(&MSTAT);
}
