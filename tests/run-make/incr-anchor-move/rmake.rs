//@ ignore-cross-compile
// The debuginfo checks below read DWARF from the linked binary, which only works reliably
// on Linux (windows-msvc emits CodeView; macOS keeps DWARF in separate dSYM bundles).
//@ only-linux

// Regression test for stale line numbers when a definition *moves* without any other
// observable change. `main_v1.rs` and `main_v2.rs` swap the byte-identical-length functions
// `alpha` and `omega`, so the file's line table and both functions' relative spans and
// contents are unchanged across the edit; only their absolute positions swap. The
// `source_span` anchor fingerprint deliberately ignores position, so nothing about these
// functions re-fingerprints; the `def_anchor` dependency recorded by the line-rendering
// consumers, which hashes the line index of each definition's start, is the only thing
// that invalidates their cached object code. Without it, the second build would replay
// stale `#[track_caller]` lines and stale `DW_AT_decl_line`s from the incremental cache.

use run_make_support::{llvm_dwarfdump, rfs, run, rustc};

fn main() {
    let build = || rustc().input("main.rs").incremental("incr").debuginfo("2").output("main").run();

    // `fn alpha` is declared on line 12 of `main_v1.rs` and line 15 of `main_v2.rs`.
    rfs::copy("main_v1.rs", "main.rs");
    build();
    run("main");
    llvm_dwarfdump()
        .input("main")
        .arg("--name=alpha")
        .run()
        .assert_stdout_contains_regex(r"DW_AT_decl_line\s*\(12\)");

    rfs::copy("main_v2.rs", "main.rs");
    build();
    run("main");
    llvm_dwarfdump()
        .input("main")
        .arg("--name=alpha")
        .run()
        .assert_stdout_contains_regex(r"DW_AT_decl_line\s*\(15\)");
}
