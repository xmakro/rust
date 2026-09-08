//! cfg and check-cfg configuration
//!
//! This module contains the definition of [`Cfg`] and [`CheckCfg`]
//! as well as the logic for creating the default configuration for a
//! given [`Session`].
//!
//! It also contains the filling of the well known configs, which should
//! ALWAYS be in sync with the default_configuration.
//!
//! ## Adding a new cfg
//!
//! Adding a new feature requires two new symbols one for the cfg itself
//! and the second one for the unstable feature gate, those are defined in
//! `rustc_span::symbol`.
//!
//! As well as the following points,
//!  - Add the activation logic in [`default_configuration`]
//!  - Add the cfg to [`CheckCfg::fill_well_known`] (and related files),
//!    so that the compiler can know the cfg is expected
//!  - Add the cfg in [`disallow_cfgs`] to disallow users from setting it via `--cfg`
//!  - Add the feature gating in `compiler/rustc_feature/src/builtin_attrs.rs`

use std::hash::Hash;

use rustc_abi::Align;
use rustc_ast::ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_lint_defs::builtin::EXPLICIT_BUILTIN_CFGS_IN_FLAGS;
use rustc_span::{Symbol, sym};
use rustc_target::spec::{PanicStrategy, RelocModel, SanitizerSet, Target};

use crate::config::{CrateType, FmtDebug};
use crate::{Session, diagnostics};

/// The parsed `--cfg` options that define the compilation environment of the
/// crate, used to drive conditional compilation.
///
/// An `FxIndexSet` is used to ensure deterministic ordering of error messages
/// relating to `--cfg`.
pub type Cfg = FxIndexSet<(Symbol, Option<Symbol>)>;

/// The parsed `--check-cfg` options.
#[derive(Default)]
pub struct CheckCfg {
    /// Is well known names activated
    pub exhaustive_names: bool,
    /// Is well known values activated
    pub exhaustive_values: bool,
    /// All the expected values for a config name
    pub expecteds: FxHashMap<Symbol, ExpectedValues<Symbol>>,
    /// Well known names (only used for diagnostics purposes)
    pub well_known_names: FxHashSet<Symbol>,
}

pub enum ExpectedValues<T> {
    Some(FxHashSet<Option<T>>),
    Any,
}

impl<T: Eq + Hash> ExpectedValues<T> {
    fn insert(&mut self, value: T) -> bool {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.insert(Some(value)),
            ExpectedValues::Any => false,
        }
    }

    pub fn contains(&self, value: &Option<T>) -> bool {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.contains(value),
            ExpectedValues::Any => false,
        }
    }
}

impl<T: Eq + Hash> Extend<T> for ExpectedValues<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.extend(iter.into_iter().map(Some)),
            ExpectedValues::Any => {}
        }
    }
}

impl<'a, T: Eq + Hash + Copy + 'a> Extend<&'a T> for ExpectedValues<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.extend(iter.into_iter().map(|a| Some(*a))),
            ExpectedValues::Any => {}
        }
    }
}

/// Disallow builtin cfgs from the CLI.
pub(crate) fn disallow_cfgs(sess: &Session, user_cfgs: &Cfg) {
    let disallow = |cfg: &(Symbol, Option<Symbol>), controlled_by| {
        let cfg_name = cfg.0;
        let cfg = if let Some(value) = cfg.1 {
            format!(r#"{}="{}""#, cfg_name, value)
        } else {
            format!("{}", cfg_name)
        };
        sess.psess.opt_span_buffer_lint(
            EXPLICIT_BUILTIN_CFGS_IN_FLAGS,
            None,
            ast::CRATE_NODE_ID,
            diagnostics::UnexpectedBuiltinCfg { cfg, cfg_name, controlled_by }.into(),
        )
    };

    // We want to restrict setting builtin cfgs that will produce incoherent behavior
    // between the cfg and the rustc cli flag that sets it.
    //
    // The tests are in tests/ui/cfg/disallowed-cli-cfgs.rs.

    // By-default all builtin cfgs are disallowed, only those are allowed:
    //  - test: as it makes sense to the have the `test` cfg active without the builtin
    //          test harness. See Cargo `harness = false` config.
    //
    // Cargo `--cfg test`: https://github.com/rust-lang/cargo/blob/bc89bffa5987d4af8f71011c7557119b39e44a65/src/cargo/core/compiler/mod.rs#L1124

    for cfg in user_cfgs {
        match cfg {
            (sym::overflow_checks, None) => disallow(cfg, "-C overflow-checks"),
            (sym::debug_assertions, None) => disallow(cfg, "-C debug-assertions"),
            (sym::ub_checks, None) => disallow(cfg, "-Z ub-checks"),
            (sym::contract_checks, None) => disallow(cfg, "-Z contract-checks"),
            (sym::sanitize, None | Some(_)) => disallow(cfg, "-Z sanitizer"),
            (
                sym::sanitizer_cfi_generalize_pointers | sym::sanitizer_cfi_normalize_integers,
                None | Some(_),
            ) => disallow(cfg, "-Z sanitizer=cfi"),
            (sym::proc_macro, None) => disallow(cfg, "--crate-type proc-macro"),
            (sym::panic, Some(sym::abort | sym::unwind | sym::immediate_abort)) => {
                disallow(cfg, "-C panic")
            }
            (sym::target_feature, Some(_)) => disallow(cfg, "-C target-feature"),
            (sym::unix, None)
            | (sym::windows, None)
            | (sym::relocation_model, Some(_))
            | (sym::target_abi, None | Some(_))
            | (sym::target_arch, Some(_))
            | (sym::target_endian, Some(_))
            | (sym::target_env, None | Some(_))
            | (sym::target_family, Some(_))
            | (sym::target_object_format, Some(_))
            | (sym::target_os, Some(_))
            | (sym::target_pointer_width, Some(_))
            | (sym::target_vendor, None | Some(_))
            | (sym::target_has_atomic, Some(_))
            | (sym::target_has_threads, None | Some(_))
            | (sym::target_has_atomic_primitive_alignment, Some(_))
            | (sym::target_has_atomic_load_store, Some(_))
            | (sym::target_has_reliable_f16, None | Some(_))
            | (sym::target_has_reliable_f16_math, None | Some(_))
            | (sym::target_has_reliable_f128, None | Some(_))
            | (sym::target_has_reliable_f128_math, None | Some(_))
            | (sym::target_thread_local, None) => disallow(cfg, "--target"),
            (sym::fmt_debug, None | Some(_)) => disallow(cfg, "-Z fmt-debug"),
            _ => {}
        }
    }
}

/// Generate the default configs for a given session
pub(crate) fn default_configuration(sess: &Session) -> Cfg {
    let mut ret = Cfg::default();

    macro_rules! ins_none {
        ($key:expr) => {
            ret.insert(($key, None));
        };
    }
    macro_rules! ins_str {
        ($key:expr, $val_str:expr) => {
            ret.insert(($key, Some(Symbol::intern($val_str))));
        };
    }
    macro_rules! ins_sym {
        ($key:expr, $val_sym:expr) => {
            ret.insert(($key, Some($val_sym)));
        };
    }

    // Symbols are inserted in alphabetical order as much as possible.
    // The exceptions are where control flow forces things out of order.
    //
    // Run `rustc --print cfg` to see the configuration in practice.
    //
    // NOTE: These insertions should be kept in sync with
    // `CheckCfg::fill_well_known` below.

    if sess.opts.debug_assertions {
        ins_none!(sym::debug_assertions);
    }

    if sess.is_nightly_build() {
        match sess.opts.unstable_opts.fmt_debug {
            FmtDebug::Full => {
                ins_sym!(sym::fmt_debug, sym::full);
            }
            FmtDebug::Shallow => {
                ins_sym!(sym::fmt_debug, sym::shallow);
            }
            FmtDebug::None => {
                ins_sym!(sym::fmt_debug, sym::none);
            }
        }
    }

    if sess.overflow_checks() {
        ins_none!(sym::overflow_checks);
    }

    // We insert a cfg for the name of session's panic strategy.
    // Since the ImmediateAbort strategy is new, it also sets cfg(panic="abort"), so that code
    // which is trying to detect whether unwinding is enabled by checking for cfg(panic="abort")
    // does not need to be updated.
    ins_sym!(sym::panic, sess.panic_strategy().desc_symbol());
    if sess.panic_strategy() == PanicStrategy::ImmediateAbort {
        ins_sym!(sym::panic, PanicStrategy::Abort.desc_symbol());
    }

    // JUSTIFICATION: before wrapper fn is available
    #[allow(rustc::bad_opt_access)]
    if sess.opts.crate_types.contains(&CrateType::ProcMacro) {
        ins_none!(sym::proc_macro);
    }

    if sess.is_nightly_build() {
        ins_sym!(sym::relocation_model, sess.target.relocation_model.desc_symbol());
    }

    for mut s in sess.sanitizers() {
        // KASAN is still ASAN under the hood, so it uses the same attribute.
        if s == SanitizerSet::KERNELADDRESS {
            s = SanitizerSet::ADDRESS;
        }
        // KHWASAN is still HWASAN under the hood, so it uses the same attribute.
        if s == SanitizerSet::KERNELHWADDRESS {
            s = SanitizerSet::HWADDRESS;
        }
        ins_str!(sym::sanitize, &s.to_string());
    }

    if sess.is_sanitizer_cfi_generalize_pointers_enabled() {
        ins_none!(sym::sanitizer_cfi_generalize_pointers);
    }
    if sess.is_sanitizer_cfi_normalize_integers_enabled() {
        ins_none!(sym::sanitizer_cfi_normalize_integers);
    }

    ins_sym!(sym::target_abi, sess.target.cfg_abi.desc_symbol());
    ins_sym!(sym::target_arch, sess.target.arch.desc_symbol());
    ins_sym!(sym::target_endian, sess.target.endian.desc_symbol());
    ins_sym!(sym::target_env, sess.target.env.desc_symbol());
    ins_sym!(sym::target_object_format, sess.target.options.binary_format.desc_symbol());

    for family in sess.target.families.as_ref() {
        ins_str!(sym::target_family, family);
        if family == "windows" {
            ins_none!(sym::windows);
        } else if family == "unix" {
            ins_none!(sym::unix);
        }
    }

    // `target_has_atomic*`
    let layout = sess.target.parse_data_layout().unwrap_or_else(|err| {
        sess.dcx().emit_fatal(err);
    });
    let mut has_atomic = false;
    for (i, align) in [
        (8, layout.i8_align),
        (16, layout.i16_align),
        (32, layout.i32_align),
        (64, layout.i64_align),
        (128, layout.i128_align),
    ] {
        if i >= sess.target.min_atomic_width() && i <= sess.target.max_atomic_width() {
            if !has_atomic {
                has_atomic = true;
                if sess.is_nightly_build() {
                    if sess.target.atomic_cas {
                        ins_none!(sym::target_has_atomic);
                    }
                    ins_none!(sym::target_has_atomic_load_store);
                }
            }
            let mut insert_atomic = |sym, align: Align| {
                if sess.target.atomic_cas {
                    ins_sym!(sym::target_has_atomic, sym);
                }
                if align.bits() == i {
                    ins_sym!(sym::target_has_atomic_primitive_alignment, sym);
                }
                ins_sym!(sym::target_has_atomic_load_store, sym);
            };
            insert_atomic(sym::integer(i), align);
            if sess.target.pointer_width as u64 == i {
                insert_atomic(sym::ptr, layout.pointer_align().abi);
            }
        }
    }

    if !sess.target.singlethread(&sess.internal_target_features) {
        ins_none!(sym::target_has_threads);
    }

    ins_sym!(sym::target_os, sess.target.os.desc_symbol());
    ins_sym!(sym::target_pointer_width, sym::integer(sess.target.pointer_width));

    if sess.opts.unstable_opts.has_thread_local.unwrap_or(sess.target.has_thread_local) {
        ins_none!(sym::target_thread_local);
    }

    ins_sym!(sym::target_vendor, sess.target.vendor_symbol());

    // If the user wants a test runner, then add the test cfg.
    if sess.is_test_crate() {
        ins_none!(sym::test);
    }

    if sess.ub_checks() {
        ins_none!(sym::ub_checks);
    }

    if sess.contract_checks() {
        ins_none!(sym::contract_checks);
    }

    ret
}

impl CheckCfg {
    /// Fill the current [`CheckCfg`] with all the well known cfgs
    pub fn fill_well_known(&mut self, current_target: &Target) {
        if !self.exhaustive_values && !self.exhaustive_names {
            return;
        }

        // for `#[cfg(foo)]` (i.e. cfg value is none)
        let no_values = || {
            let mut values = FxHashSet::default();
            values.insert(None);
            ExpectedValues::Some(values)
        };

        // preparation for inserting some values
        let empty_values = || {
            let values = FxHashSet::default();
            ExpectedValues::Some(values)
        };

        macro_rules! ins {
            ($name:expr, $values:expr) => {{
                self.well_known_names.insert($name);
                self.expecteds.entry($name).or_insert_with($values)
            }};
        }

        // Symbols are inserted in alphabetical order as much as possible.
        // The exceptions are where control flow forces things out of order.
        //
        // NOTE: This should be kept in sync with `default_configuration`.
        // Note that symbols inserted conditionally in `default_configuration`
        // are inserted unconditionally here.
        //
        // One exception is the `test` cfg which is consider to be a "user-space"
        // cfg, despite being also set by in `default_configuration` above.
        // It allows the build system to "deny" using the config by not marking it
        // as expected (e.g. `lib.test = false` for Cargo).
        //
        // When adding a new config here you should also update
        // `tests/ui/check-cfg/well-known-values.rs` (in order to test the
        // expected values of the new config) and bless the all directory.
        //
        // Don't forget to update `src/doc/rustc/src/check-cfg.md`
        // in the unstable book as well!

        ins!(sym::debug_assertions, no_values);

        ins!(sym::fmt_debug, empty_values).extend(FmtDebug::all());

        // These four are never set by rustc, but we set them anyway; they
        // should not trigger the lint because `cargo clippy`, `cargo doc`,
        // `cargo test`, `cargo miri run` and `cargo fmt` (respectively)
        // can set them.
        ins!(sym::clippy, no_values);
        ins!(sym::doc, no_values);
        ins!(sym::doctest, no_values);
        ins!(sym::miri, no_values);
        ins!(sym::rust_analyzer, no_values);
        ins!(sym::rustfmt, no_values);

        ins!(sym::overflow_checks, no_values);

        ins!(sym::panic, empty_values)
            .extend(PanicStrategy::ALL.iter().map(PanicStrategy::desc_symbol));

        ins!(sym::proc_macro, no_values);

        ins!(sym::relocation_model, empty_values)
            .extend(RelocModel::ALL.iter().map(RelocModel::desc_symbol));

        let sanitize_values = SanitizerSet::all()
            .into_iter()
            .map(|sanitizer| Symbol::intern(sanitizer.as_str().unwrap()));
        ins!(sym::sanitize, empty_values).extend(sanitize_values);

        ins!(sym::sanitizer_cfi_generalize_pointers, no_values);
        ins!(sym::sanitizer_cfi_normalize_integers, no_values);

        ins!(sym::target_feature, empty_values).extend(
            rustc_target::target_features::all_rust_features()
                .filter(|(_, s)| s.in_cfg())
                .map(|(f, _s)| f)
                .chain(rustc_target::target_features::RUSTC_SPECIFIC_FEATURES.iter().cloned())
                .map(Symbol::intern),
        );

        // sym::target_*
        {
            const VALUES: [&Symbol; 9] = [
                &sym::target_abi,
                &sym::target_arch,
                &sym::target_endian,
                &sym::target_env,
                &sym::target_family,
                &sym::target_object_format,
                &sym::target_os,
                &sym::target_pointer_width,
                &sym::target_vendor,
            ];

            // Initialize (if not already initialized)
            for &e in VALUES {
                if !self.exhaustive_values {
                    ins!(e, || ExpectedValues::Any);
                } else {
                    ins!(e, empty_values);
                }
            }

            if self.exhaustive_values {
                let mut values = self
                    .expecteds
                    .get_disjoint_mut(VALUES)
                    .map(|value| value.expect("unable to get all the check-cfg values buckets"));
                for (values, known) in values.iter_mut().zip(BUILTIN_TARGET_CFG_VALUES) {
                    values.extend(known.iter().map(|value| Symbol::intern(value)));
                }
                let [
                    values_target_abi,
                    values_target_arch,
                    values_target_endian,
                    values_target_env,
                    values_target_family,
                    values_target_object_format,
                    values_target_os,
                    values_target_pointer_width,
                    values_target_vendor,
                ] = values;

                // Custom targets may contribute additional values.
                {
                    let target = current_target;
                    values_target_abi.insert(target.options.cfg_abi.desc_symbol());
                    values_target_arch.insert(target.arch.desc_symbol());
                    values_target_endian.insert(target.options.endian.desc_symbol());
                    values_target_env.insert(target.options.env.desc_symbol());
                    values_target_family.extend(
                        target.options.families.iter().map(|family| Symbol::intern(family)),
                    );
                    values_target_object_format.insert(target.options.binary_format.desc_symbol());
                    values_target_os.insert(target.options.os.desc_symbol());
                    values_target_pointer_width.insert(sym::integer(target.pointer_width));
                    values_target_vendor.insert(target.vendor_symbol());
                }
            }
        }

        let atomic_values = &[
            sym::ptr,
            sym::integer(8usize),
            sym::integer(16usize),
            sym::integer(32usize),
            sym::integer(64usize),
            sym::integer(128usize),
        ];
        for sym in [sym::target_has_atomic, sym::target_has_atomic_load_store] {
            ins!(sym, no_values).extend(atomic_values);
        }
        ins!(sym::target_has_atomic_primitive_alignment, empty_values).extend(atomic_values);

        ins!(sym::target_thread_local, no_values);
        ins!(sym::target_has_threads, no_values);

        ins!(sym::ub_checks, no_values);
        ins!(sym::contract_checks, no_values);

        ins!(sym::unix, no_values);
        ins!(sym::windows, no_values);
    }
}

// The distinct values used by built-in targets, in the same order as VALUES above.
// Avoid constructing every full target specification on each check-cfg invocation.
// `builtin_target_cfg_values_match_targets` checks exact equality, including unused
// enum variants and changes to the set of targets, so diagnostics stay unchanged.
const BUILTIN_TARGET_CFG_VALUES: [&[&str]; 9] = [
    &[
        "",
        "abi64",
        "abiv2",
        "abiv2hf",
        "eabi",
        "eabihf",
        "elfv1",
        "elfv2",
        "fortanix",
        "ilp32",
        "ilp32e",
        "llvm",
        "macabi",
        "pauthtest",
        "sim",
        "softfloat",
        "spe",
        "uwp",
        "v8plus",
        "vec-extabi",
        "x32",
    ], // abi
    &[
        "aarch64",
        "amdgpu",
        "arm",
        "arm64ec",
        "avr",
        "bpf",
        "csky",
        "hexagon",
        "loongarch32",
        "loongarch64",
        "m68k",
        "mips",
        "mips32r6",
        "mips64",
        "mips64r6",
        "msp430",
        "nvptx64",
        "powerpc",
        "powerpc64",
        "riscv32",
        "riscv64",
        "s390x",
        "sparc",
        "sparc64",
        "wasm32",
        "wasm64",
        "x86",
        "x86_64",
        "xtensa",
    ], // arch
    &["big", "little"], // target-endian
    &[
        "",
        "gnu",
        "macabi",
        "mlibc",
        "msvc",
        "musl",
        "newlib",
        "nto70",
        "nto71",
        "nto71_iosock",
        "ohos",
        "p1",
        "p2",
        "p3",
        "relibc",
        "sgx",
        "sim",
        "uclibc",
        "v5",
    ], // env
    &["unix", "wasm", "windows"], // target-family
    &["coff", "elf", "mach-o", "wasm", "xcoff"], // binary-format
    &[
        "aix",
        "amdhsa",
        "android",
        "cuda",
        "cygwin",
        "dragonfly",
        "emscripten",
        "espidf",
        "freebsd",
        "fuchsia",
        "haiku",
        "helenos",
        "hermit",
        "horizon",
        "hurd",
        "illumos",
        "ios",
        "l4re",
        "linux",
        "lynxos178",
        "macos",
        "managarm",
        "motor",
        "netbsd",
        "none",
        "nto",
        "nuttx",
        "openbsd",
        "ps3",
        "psp",
        "psx",
        "qnx",
        "qurt",
        "redox",
        "rtems",
        "solaris",
        "solid_asp3",
        "teeos",
        "trusty",
        "tvos",
        "uefi",
        "unknown",
        "vexos",
        "visionos",
        "vita",
        "vxworks",
        "wasi",
        "watchos",
        "windows",
        "xous",
        "zkvm",
    ], // os
    &["16", "32", "64"], // target-pointer-width
    &[
        "amd",
        "apple",
        "espressif",
        "fortanix",
        "ibm",
        "kmc",
        "mti",
        "nintendo",
        "nvidia",
        "openwrt",
        "pc",
        "risc0",
        "sony",
        "sun",
        "unikraft",
        "unknown",
        "uwp",
        "vex",
        "win7",
        "wrs",
    ], // vendor
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn builtin_target_cfg_values_match_targets() {
        rustc_span::create_default_session_globals_then(|| {
            let mut actual: [BTreeSet<String>; 9] = std::array::from_fn(|_| BTreeSet::new());
            for target in Target::builtins() {
                let values = [
                    vec![target.options.cfg_abi.desc_symbol().to_string()],
                    vec![target.arch.desc_symbol().to_string()],
                    vec![target.options.endian.desc_symbol().to_string()],
                    vec![target.options.env.desc_symbol().to_string()],
                    target.options.families.iter().map(|family| family.to_string()).collect(),
                    vec![target.options.binary_format.desc_symbol().to_string()],
                    vec![target.options.os.desc_symbol().to_string()],
                    vec![target.pointer_width.to_string()],
                    vec![target.vendor_symbol().to_string()],
                ];
                for (actual, values) in actual.iter_mut().zip(values) {
                    actual.extend(values);
                }
            }
            for (index, (actual, expected)) in
                actual.iter().zip(BUILTIN_TARGET_CFG_VALUES).enumerate()
            {
                assert_eq!(
                    actual.iter().map(String::as_str).collect::<Vec<_>>(),
                    expected,
                    "update BUILTIN_TARGET_CFG_VALUES entry {index} to match the built-in targets"
                );
            }
        });
    }
}
