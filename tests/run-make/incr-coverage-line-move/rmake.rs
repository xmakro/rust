//@ ignore-cross-compile
//@ needs-profiler-runtime

// Coverage mappings store line/column coordinates for positions throughout a file, so
// coverage codegen records a dependency on the file's *entire* line table
// (`TyCtxt::source_file_tracked`). This test moves a line break without changing byte
// offsets: span fingerprints do not change, and only that dependency invalidates the
// coverage mapping. A stale mapping would report the executed function at its old line.

use run_make_support::{
    cwd, has_extension, has_prefix, llvm_cov, llvm_profdata, rfs, run, rustc, shallow_find_files,
};

/// Generates `main.rs`. `split_pad` moves a line break inside `pad` without changing the
/// file's byte length, shifting `covered` by one line. Returns the source and the 1-based
/// line of `fn covered`.
fn source(split_pad: bool) -> (String, usize) {
    let mut lines = vec!["mod pad {".to_string()];
    if split_pad {
        lines.push("    const X: u8 =".to_string());
        lines.push("        0;".to_string());
    } else {
        lines.push("    const X: u8 = 0; //xxxxx".to_string());
    }
    lines.push("}".to_string());
    let covered_line = lines.len() + 1;
    lines.push("fn covered() -> u32 {".to_string());
    lines.push("    41".to_string());
    lines.push("}".to_string());
    lines.push("fn main() { assert_eq!(covered(), 41); }".to_string());
    (lines.join("\n") + "\n", covered_line)
}

fn check_version(src: &str, covered_line: usize) {
    rfs::write("main.rs", src);
    // The command line is identical for both builds: any difference could invalidate
    // nodes on its own and mask a stale-reuse regression.
    rustc().input("main.rs").incremental("incr").arg("-Cinstrument-coverage").output("main").run();
    for f in shallow_find_files(cwd(), |p| has_extension(p, "profraw")) {
        rfs::remove_file(f);
    }
    run("main");
    let profraw_files =
        shallow_find_files(cwd(), |p| has_prefix(p, "default") && has_extension(p, "profraw"));
    assert!(!profraw_files.is_empty(), "no .profraw file generated");
    let mut profdata = llvm_profdata();
    profdata.merge().output("cov.profdata");
    for f in profraw_files {
        profdata.input(f);
    }
    profdata.run();
    // `llvm-cov show` prints `<line>|<count>|<source>`; the function line must carry its
    // entry count at the *current* line number.
    llvm_cov()
        .show("main")
        .instr_profile("cov.profdata")
        .run()
        .assert_stdout_contains_regex(format!(r"(?m)^\s*{covered_line}\|\s*1\|fn covered"));
}

fn main() {
    let (v1, line_v1) = source(true);
    let (v2, line_v2) = source(false);
    // The edit must move line numbers without moving byte offsets; see the header comment.
    assert_eq!(v1.len(), v2.len(), "versions must have identical byte length");
    assert_eq!(v1.find("fn covered").unwrap(), v2.find("fn covered").unwrap());
    assert_eq!(line_v1, line_v2 + 1);

    check_version(&v1, line_v1);
    check_version(&v2, line_v2);
}
