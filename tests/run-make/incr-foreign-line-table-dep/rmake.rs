//@ ignore-cross-compile

// An incremental rebuild must be able to re-execute `def_anchor` dep nodes recorded for
// *upstream* definitions. `helper` is an `#[inline]` function from another crate,
// codegened locally into this crate's CGU with `dep.rs` in its debuginfo line tables, so
// the first build records `helper`'s anchor node. During try-mark-green that node is
// forced before any span pointing into `dep.rs` has been decoded, i.e. before the file
// has been imported into the source map; decoding the recovered definition's `def_span`
// must import it as a side effect. If the provider cannot resolve the definition or its
// file, every such node counts as changed on every rebuild and all reuse is lost, which
// the `rustc_partition_reused` assertion catches.

use run_make_support::{rfs, run, rustc};

fn main() {
    rfs::write("dep.rs", "#[inline]\npub fn helper() -> u32 {\n    41\n}\n");
    rfs::write(
        "main.rs",
        "#![feature(rustc_attrs)]\n\
         #![rustc_partition_reused(module = \"main\", cfg = \"rpass2\")]\n\
         fn main() { assert_eq!(dep::helper(), 41); }\n",
    );
    rustc().input("dep.rs").crate_type("rlib").run();

    let build = |cfg: &str| {
        rustc()
            .input("main.rs")
            .incremental("incr")
            .debuginfo("2")
            .arg("-Zquery-dep-graph")
            .cfg(cfg)
            .extern_("dep", "libdep.rlib")
            .output("main")
            .run()
    };

    build("rpass1");
    run("main");
    // Unchanged rebuild: everything, including the CGU holding `helper`'s debuginfo, must
    // come back green through the upstream-file dep node.
    build("rpass2");
    run("main");
}
