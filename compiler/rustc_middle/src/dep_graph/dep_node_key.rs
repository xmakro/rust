use std::fmt::Debug;

use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hash::{StableHash, StableHasher};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId, LocalModDefId, ModDefId};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::{HirId, ItemLocalId, OwnerId};
use rustc_span::LocalExpnId;

use crate::dep_graph::{DepNode, KeyFingerprintStyle};
use crate::ty::TyCtxt;

/// Trait for query keys as seen by dependency-node tracking.
pub trait DepNodeKey<'tcx>: Debug + Sized {
    fn key_fingerprint_style() -> KeyFingerprintStyle;

    /// This method turns a query key into an opaque `Fingerprint` to be used
    /// in `DepNode`.
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint;

    /// This method tries to recover the query key from the given `DepNode`,
    /// something which is needed when forcing `DepNode`s during red-green
    /// evaluation. The query system will only call this method if
    /// `fingerprint_style()` is not `FingerprintStyle::Opaque`.
    /// It is always valid to return `None` here, in which case incremental
    /// compilation will treat the query as having changed instead of forcing it.
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self>;
}

// Blanket impl of `DepNodeKey`, which is specialized by other impls elsewhere.
impl<'tcx, T> DepNodeKey<'tcx> for T
where
    T: StableHash + Debug,
{
    #[inline(always)]
    default fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Opaque
    }

    #[inline(always)]
    default fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            self.stable_hash(&mut hcx, &mut hasher);
            hasher.finish()
        })
    }

    #[inline(always)]
    default fn try_recover_key(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        None
    }
}

impl<'tcx> DepNodeKey<'tcx> for () {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Unit
    }

    #[inline(always)]
    fn to_fingerprint(&self, _: TyCtxt<'tcx>) -> Fingerprint {
        Fingerprint::ZERO
    }

    #[inline(always)]
    fn try_recover_key(_: TyCtxt<'tcx>, _: &DepNode) -> Option<Self> {
        Some(())
    }
}

impl<'tcx> DepNodeKey<'tcx> for DefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        tcx.def_path_hash(*self).0
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx)
    }
}

impl<'tcx> DepNodeKey<'tcx> for LocalDefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.expect_local())
    }
}

impl<'tcx> DepNodeKey<'tcx> for OwnerId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| OwnerId { def_id: id.expect_local() })
    }
}

impl<'tcx> DepNodeKey<'tcx> for CrateNum {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let def_id = self.as_def_id();
        def_id.to_fingerprint(tcx)
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        dep_node.extract_def_id(tcx).map(|id| id.krate)
    }
}

impl<'tcx> DepNodeKey<'tcx> for (DefId, DefId) {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Opaque
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let (def_id_0, def_id_1) = *self;

        let def_path_hash_0 = tcx.def_path_hash(def_id_0);
        let def_path_hash_1 = tcx.def_path_hash(def_id_1);

        def_path_hash_0.0.combine(def_path_hash_1.0)
    }
}

impl<'tcx, 'a> DepNodeKey<'tcx> for (LocalExpnId, &'a TokenStream) {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::Opaque
    }

    /// Span-agnostic fingerprint for the `derive_macro_expansion` query key.
    ///
    /// The default blanket impl would `StableHash` the whole `(LocalExpnId, &TokenStream)`
    /// tuple, which folds every token span (and the expansion's `call_site`/`def_site`) into
    /// the fingerprint. As a result, an edit that only *shifts* source positions — e.g.
    /// inserting a comment at the top of the file — changes the fingerprint of every derive
    /// invocation and misses the on-disk cache entirely, even though the proc macro would
    /// produce identical output.
    ///
    /// Instead we hash a triple of span-agnostic, edit-stable components:
    /// - the enclosing module's `DefId` (via its `DefPathHash`);
    /// - the macro's `DefId` (via its `DefPathHash`), so distinct macros never share a node —
    ///   the proc macro crate's `crate_hash` is registered as a real dependency in
    ///   `provide_derive_macro_expansion`, which invalidates the cache when the macro's
    ///   *definition* changes;
    /// - the input token stream with span hashing disabled, so only token kinds and symbols
    ///   contribute.
    ///
    /// The `(module, macro, tokens)` triple is collision-free for all legal programs: two
    /// distinct derive invocations that hash identically would have to be the same macro applied
    /// to two items with byte-identical token trees (which include the item's name) in the same
    /// module — but duplicate item names in one module are rejected by name resolution. This
    /// matters because the dep graph panics if two distinct query keys map to one `DepNode`
    /// (see `assert_dep_node_not_yet_allocated_in_current_session`).
    ///
    /// Notably we do *not* fold in the raw `LocalExpnId` index: expansion ids are not stable
    /// across edits that insert or remove earlier expansions (e.g. a comment before other
    /// macro calls shifts every subsequent index), which would defeat the cache entirely.
    ///
    /// On a cache hit the loaded tokens carry the previous session's byte positions; hygiene
    /// (`SyntaxContext`) is still decoded correctly by the on-disk cache, so name resolution is
    /// unaffected and this remains sound — only diagnostic spans for macro-generated code may
    /// point at stale offsets.
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let (invoc_id, input) = *self;
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();

            let expn_data = invoc_id.expn_data();
            // Enclosing module and macro identity, both via span-agnostic `DefPathHash`.
            expn_data.parent_module.stable_hash(&mut hcx, &mut hasher);
            expn_data.macro_def_id.stable_hash(&mut hcx, &mut hasher);

            // The input tokens, ignoring spans.
            hcx.while_hashing_spans(false, |hcx| {
                input.stable_hash(hcx, &mut hasher);
            });

            hasher.finish()
        })
    }
}

impl<'tcx> DepNodeKey<'tcx> for HirId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::HirId
    }

    // We actually would not need to specialize the implementation of this
    // method but it's faster to combine the hashes than to instantiate a full
    // hashing context and stable-hashing state.
    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        let HirId { owner, local_id } = *self;
        let def_path_hash = tcx.def_path_hash(owner.to_def_id());
        Fingerprint::new(
            // `owner` is local, so is completely defined by the local hash
            def_path_hash.local_hash(),
            local_id.as_u32() as u64,
        )
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        if tcx.key_fingerprint_style(dep_node.kind) == KeyFingerprintStyle::HirId {
            let (local_hash, local_id) = Fingerprint::from(dep_node.key_fingerprint).split();
            let def_path_hash = DefPathHash::new(tcx.stable_crate_id(LOCAL_CRATE), local_hash);
            let def_id = tcx.def_path_hash_to_def_id(def_path_hash)?.expect_local();
            let local_id = local_id
                .as_u64()
                .try_into()
                .unwrap_or_else(|_| panic!("local id should be u32, found {local_id:?}"));
            Some(HirId { owner: OwnerId { def_id }, local_id: ItemLocalId::from_u32(local_id) })
        } else {
            None
        }
    }
}

impl<'tcx> DepNodeKey<'tcx> for ModDefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        DefId::try_recover_key(tcx, dep_node).map(ModDefId::new_unchecked)
    }
}

impl<'tcx> DepNodeKey<'tcx> for LocalModDefId {
    #[inline(always)]
    fn key_fingerprint_style() -> KeyFingerprintStyle {
        KeyFingerprintStyle::DefPathHash
    }

    #[inline(always)]
    fn to_fingerprint(&self, tcx: TyCtxt<'tcx>) -> Fingerprint {
        self.to_def_id().to_fingerprint(tcx)
    }

    #[inline(always)]
    fn try_recover_key(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> Option<Self> {
        LocalDefId::try_recover_key(tcx, dep_node).map(LocalModDefId::new_unchecked)
    }
}
