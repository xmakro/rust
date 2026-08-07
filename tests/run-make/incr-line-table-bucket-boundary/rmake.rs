//@ ignore-cross-compile

// Regression test for stale `#[track_caller]` lines when an edit on the SAME source line
// inserts a line start without moving byte offsets.
//
// `m2` begins mid-line after its sibling `m1`; the edit splits `m1` across two lines, so
// every byte offset is unchanged and `m2`'s definition merely starts one line further
// down. `def_anchor` hashes the line index of each definition's start, which is exactly
// the rendered quantity, so this edit must go red for `m2` even though `m2`'s extent,
// contents and relative spans are all untouched. The edit also sits at a 64-entry
// boundary of the cached line-length blocks (`LINE_LENGTH_BLOCK`), pinning the
// block-edge arithmetic.
//
// The `rustc_partition_*` assertions prove the invalidation is targeted: `m2`, which
// bakes the caller line, is re-codegened, while the byte-identical `m1`, which renders no
// line data, is reused.

use run_make_support::{rfs, run, rustc};

/// Generates `main.rs` with the observed call on the last line of the first line-table
/// bucket (`LINE_TABLE_BUCKET` = 64 line starts). `split` inserts the boundary line break.
fn source(split: bool) -> (String, usize) {
    let mut lines = vec![
        "#![feature(rustc_attrs)]".to_string(),
        "#![rustc_partition_reused(module = \"main-m1\", cfg = \"rpass2\")]".to_string(),
        "#![rustc_partition_codegened(module = \"main-m2\", cfg = \"rpass2\")]".to_string(),
        "#![allow(dead_code)]".to_string(),
        "#[track_caller]".to_string(),
        "pub fn line_of_caller() -> u32 {".to_string(),
        "    std::panic::Location::caller().line()".to_string(),
        "}".to_string(),
    ];
    while lines.len() < 63 {
        lines.push("// filler".to_string());
    }
    let m2 = "mod m2 { pub fn call_here() -> u32 { crate::line_of_caller() } }";
    if split {
        lines.push("mod m1 { pub fn a() -> u32 {".to_string());
        lines.push(format!("41 }} }} {m2}"));
    } else {
        lines.push(format!("mod m1 {{ pub fn a() -> u32 {{ 41 }} }} {m2}"));
    }
    let call_line = lines.len();
    lines.push("fn main() { println!(\"{}\", m2::call_here()); let _ = m1::a(); }".to_string());
    (lines.join("\n") + "\n", call_line)
}

fn main() {
    let build = |cfg: &str| {
        rustc()
            .input("main.rs")
            .incremental("incr")
            .arg("-Zquery-dep-graph")
            .cfg(cfg)
            .output("main")
            .run()
    };

    let (v1, call_v1) = source(false);
    let (v2, call_v2) = source(true);
    // The edit must insert a line start exactly at the 64-entry bucket boundary without
    // moving byte offsets; if this drifts, the test stops exercising the boundary.
    assert_eq!(v1.len(), v2.len(), "versions must have identical byte length");
    assert_eq!(v1.find("mod m2").unwrap(), v2.find("mod m2").unwrap());
    assert_eq!(call_v1, 64, "the observed call must sit on the last line of the bucket");
    assert_eq!(call_v2, 65);

    rfs::write("main.rs", &v1);
    build("rpass1");
    assert_eq!(run("main").stdout_utf8().trim(), "64");

    rfs::write("main.rs", &v2);
    build("rpass2");
    assert_eq!(run("main").stdout_utf8().trim(), "65");
}
