//@ ignore-cross-compile

// The character columns rendered into `#[track_caller]` locations are derived from the
// file's multibyte-character table, which `file_lines_prefix_hash` folds into every
// bucket. Replacing a two-byte character with two one-byte characters earlier on the call
// line changes that table without changing any byte offset or line start, so only the
// table dependency invalidates the baked column; a stale reuse reports the old column.

use run_make_support::{rfs, run, rustc};

/// Generates `main.rs`. The two versions differ in `"é"` (one char, two bytes) vs `"ee"`
/// (two chars, two bytes) before the observed call. Returns the source and the expected
/// 1-based caller column.
fn source(multibyte: bool) -> (String, usize) {
    let payload = if multibyte { "\"é\"" } else { "\"ee\"" };
    let call_line = format!("    let _s = {payload}; let c = crate::col_of_caller();");
    let column = call_line[..call_line.find("crate::").unwrap()].chars().count() + 1;
    let lines = [
        "#![allow(dead_code)]",
        "#[track_caller]",
        "pub fn col_of_caller() -> u32 {",
        "    std::panic::Location::caller().column()",
        "}",
        "fn main() {",
        &call_line,
        "    println!(\"{}\", c);",
        "}",
    ];
    (lines.join("\n") + "\n", column)
}

fn main() {
    // The command line is identical for both builds: any difference could invalidate
    // nodes on its own and mask a stale-reuse regression.
    let build = || rustc().input("main.rs").incremental("incr").output("main").run();

    let (v1, col_v1) = source(true);
    let (v2, col_v2) = source(false);
    // The edit must change the multibyte table only: identical byte offsets, identical
    // line structure, different character count before the call.
    assert_eq!(v1.len(), v2.len(), "versions must have identical byte length");
    assert_eq!(v1.find("crate::").unwrap(), v2.find("crate::").unwrap());
    assert_eq!(col_v1 + 1, col_v2);

    rfs::write("main.rs", &v1);
    build();
    assert_eq!(run("main").stdout_utf8().trim(), col_v1.to_string());

    rfs::write("main.rs", &v2);
    build();
    assert_eq!(run("main").stdout_utf8().trim(), col_v2.to_string());
}
