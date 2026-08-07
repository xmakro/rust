//@ ignore-cross-compile
// The debuginfo checks below read DWARF from the linked binary, which only works reliably
// on Linux (windows-msvc emits CodeView; macOS keeps DWARF in separate dSYM bundles).
//@ only-linux

// Regression test for stale line numbers under incremental compilation.
//
// The two generated versions of `main.rs` differ in a line break inside `pad` that moves
// without changing the file's length, so every byte offset in `inner` is identical across
// the two versions while `inner`'s line numbers shift by one. Span fingerprints only cover
// (file, offset, length), so nothing about `inner` re-fingerprints across this edit; its
// codegen must be invalidated through the explicit `def_anchor` dependency instead (the
// edit changes the hashed line index of `inner`'s start). Without that dependency the
// second build would reuse object code with a stale `#[track_caller]` caller line and
// stale debuginfo line tables baked in (the in-binary assertion and the `DW_AT_decl_line`
// checks both catch that). See issue #74890.
//
// The `rustc_partition_*` assertions prove both directions: `inner` (whose bytes are
// identical and only whose lines moved) is re-codegened, while `stable`, whose extent
// lies entirely before the edit, is reused. The latter distinguishes targeted
// invalidation from everything having gone red.

use run_make_support::{llvm_dwarfdump, rfs, run, rustc};

/// Generates `main.rs`. `split_pad` selects whether `pad` spells its constant over two
/// lines or over one line of the same total byte length; `split_body` does the same for
/// a statement *inside* `observed`'s body, before the tracked call. Returns the source,
/// the 1-based line of the `line_of_caller()` call, and the 1-based line of `fn observed`.
fn source(split_pad: bool, split_body: bool) -> (String, usize, usize) {
    let mut lines = vec![
        "#![feature(rustc_attrs)]".to_string(),
        "#![rustc_partition_reused(module = \"main-stable\", cfg = \"rpass2\")]".to_string(),
        "#![rustc_partition_codegened(module = \"main-inner\", cfg = \"rpass2\")]".to_string(),
        "#![rustc_partition_reused(module = \"main-stable\", cfg = \"rpass3\")]".to_string(),
        "#![rustc_partition_codegened(module = \"main-inner\", cfg = \"rpass3\")]".to_string(),
        "#![allow(dead_code)]".to_string(),
        "mod stable {".to_string(),
        "    #[inline(never)]".to_string(),
        "    pub fn value() -> u32 {".to_string(),
        "        7".to_string(),
        "    }".to_string(),
        "}".to_string(),
    ];
    // Push the edit site past the first 64 line starts, so the edit also crosses a
    // 64-entry cached line-length block boundary.
    while lines.len() < 70 {
        lines.push("// filler".to_string());
    }
    lines.push("mod pad {".to_string());
    if split_pad {
        lines.push("    const X: u8 =".to_string());
        lines.push("        0;".to_string());
    } else {
        lines.push("    const X: u8 = 0; //xxxxx".to_string());
    }
    lines.push("}".to_string());
    lines.push("mod inner {".to_string());
    lines.push("    #[track_caller]".to_string());
    lines.push("    fn line_of_caller() -> u32 {".to_string());
    lines.push("        std::panic::Location::caller().line()".to_string());
    lines.push("    }".to_string());
    let decl_line = lines.len() + 1;
    lines.push("    pub fn observed() -> u32 {".to_string());
    if split_body {
        lines.push("        let _y: u8 =".to_string());
        lines.push("            0;".to_string());
    } else {
        lines.push("        let _y: u8 = 0; //xxxxxxxxx".to_string());
    }
    let call_line = lines.len() + 1;
    lines.push("        line_of_caller()".to_string());
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.push("fn main() {".to_string());
    lines.push(format!("    assert_eq!(inner::observed(), {call_line});"));
    lines.push("    assert_eq!(stable::value(), 7);".to_string());
    lines.push("}".to_string());
    (lines.join("\n") + "\n", call_line, decl_line)
}

fn main() {
    let build = |cfg: &str| {
        rustc()
            .input("main.rs")
            .incremental("incr")
            .debuginfo("2")
            .arg("-Zquery-dep-graph")
            .cfg(cfg)
            .output("main")
            .run()
    };

    let (v1, call_v1, decl_v1) = source(true, true);
    let (v2, call_v2, decl_v2) = source(false, true);
    // The third version moves a line break *inside* `observed`'s body: the byte-parity
    // edit that only the extent-relative entries of the definition's own anchor cover.
    let (v3, call_v3, decl_v3) = source(false, false);
    // The edits must move line numbers without moving byte offsets; if this drifts, the
    // test stops exercising line-table invalidation and passes vacuously through span
    // fingerprints.
    assert_eq!(v1.len(), v2.len(), "versions must have identical byte length");
    assert_eq!(v2.len(), v3.len(), "versions must have identical byte length");
    assert_eq!(v1.find("mod inner").unwrap(), v2.find("mod inner").unwrap());
    assert_eq!(v2.find("mod inner").unwrap(), v3.find("mod inner").unwrap());
    assert_eq!(call_v1, call_v2 + 1);
    assert_eq!(decl_v1, decl_v2 + 1);
    assert_eq!(call_v2, call_v3 + 1);
    assert_eq!(decl_v2, decl_v3);

    let check_decl_line = |line: usize| {
        llvm_dwarfdump()
            .input("main")
            .arg("--name=observed")
            .run()
            .assert_stdout_contains_regex(format!(r"DW_AT_decl_line\s*\({line}\)"));
    };

    rfs::write("main.rs", &v1);
    build("rpass1");
    run("main");
    check_decl_line(decl_v1);

    rfs::write("main.rs", &v2);
    build("rpass2");
    run("main");
    check_decl_line(decl_v2);

    rfs::write("main.rs", &v3);
    build("rpass3");
    run("main");
    check_decl_line(decl_v3);
}
