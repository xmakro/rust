// The dep-graph encoder carries records forward while the previous file's index space is
// dense enough, re-encodes everything in one compacting session once deletions have left
// too much of it unoccupied, and then resumes carrying. Shrink a large crate and follow
// the carrying decision through `-Zincremental-info` across sessions.

//@ ignore-cross-compile

use run_make_support::{path, rfs, rustc};

fn main() {
    let big: String =
        (0..300).map(|i| format!("pub fn f{i}(x: u64) -> u64 {{ x + {i} }}\n")).collect();
    let tiny: String =
        (0..30).map(|i| format!("pub fn f{i}(x: u64) -> u64 {{ x + {i} }}\n")).collect();

    for threads in ["1", "2"] {
        let incr = path(format!("incr-{threads}"));
        // One (source, expected carrying) pair per session: the first session has no
        // previous graph, shrinking the crate leaves the index space over-sized so the
        // third session re-encodes densely, and the fourth carries again.
        let sessions = [(&big, false), (&tiny, true), (&tiny, false), (&tiny, true)];
        for (source, carrying) in sessions {
            rfs::write("lib.rs", source);
            rustc()
                .input("lib.rs")
                .crate_type("lib")
                .incremental(&incr)
                .arg("-Zincremental-info")
                .arg(format!("-Zthreads={threads}"))
                .run()
                .assert_stderr_contains(format!("Carrying Dep Graph: {carrying}"));
        }
    }
}
