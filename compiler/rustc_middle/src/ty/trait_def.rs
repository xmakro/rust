use std::hash::{Hash, Hasher};
use std::iter;

use rustc_data_structures::fx::{FxHashMap, FxHasher, FxIndexMap};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{self as hir, find_attr};
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::Span;
use tracing::debug;

use crate::query::LocalCrate;
use crate::traits::specialization_graph;
use crate::ty::fast_reject::{self, SimplifiedType, TreatParams};
use crate::ty::print::{with_crate_prefix, with_no_trimmed_paths};
use crate::ty::{Ident, Ty, TyCtxt, TypeFlags, TypeVisitableExt};

/// A trait's definition with type information.
#[derive(StableHash, Encodable, Decodable)]
pub struct TraitDef {
    pub def_id: DefId,

    /// Restrictions on trait implementations.
    pub impl_restriction: ImplRestrictionKind,

    pub safety: hir::Safety,

    /// Whether this trait is `const`.
    pub constness: hir::Constness,

    /// If `true`, then this trait had the `#[rustc_paren_sugar]`
    /// attribute, indicating that it should be used with `Foo()`
    /// sugar. This is a temporary thing -- eventually any trait will
    /// be usable with the sugar (or without it).
    pub paren_sugar: bool,

    pub has_auto_impl: bool,

    /// If `true`, then this trait has the `#[marker]` attribute, indicating
    /// that all its associated items have defaults that cannot be overridden,
    /// and thus `impl`s of it are allowed to overlap.
    pub is_marker: bool,

    /// If `true`, then this trait has the `#[rustc_coinductive]` attribute or
    /// is an auto trait. This indicates that trait solver cycles involving an
    /// `X: ThisTrait` goal are accepted.
    ///
    /// In the future all traits should be coinductive, but we need a better
    /// formal understanding of what exactly that means and should probably
    /// also have already switched to the new trait solver.
    pub is_coinductive: bool,

    /// If `true`, then this trait has the `#[fundamental]` attribute. This
    /// affects how conherence computes whether a trait may have trait implementations
    /// added in the future.
    pub is_fundamental: bool,

    /// If `true`, then this trait has the `#[rustc_skip_during_method_dispatch(array)]`
    /// attribute, indicating that editions before 2021 should not consider this trait
    /// during method dispatch if the receiver is an array.
    pub skip_array_during_method_dispatch: bool,

    /// If `true`, then this trait has the `#[rustc_skip_during_method_dispatch(boxed_slice)]`
    /// attribute, indicating that editions before 2024 should not consider this trait
    /// during method dispatch if the receiver is a boxed slice.
    pub skip_boxed_slice_during_method_dispatch: bool,

    /// Used to determine whether the standard library is allowed to specialize
    /// on this trait.
    pub specialization_kind: TraitSpecializationKind,

    /// List of functions from `#[rustc_must_implement_one_of]` attribute one of which
    /// must be implemented.
    pub must_implement_one_of: Option<Box<[Ident]>>,

    /// Whether the trait should be considered dyn-incompatible, even if it otherwise
    /// satisfies the requirements to be dyn-compatible.
    pub force_dyn_incompatible: Option<Span>,

    /// Whether a trait is fully built-in, and any implementation is disallowed.
    /// This only applies to built-in traits, and is marked via
    /// `#[rustc_deny_explicit_impl]`.
    pub deny_explicit_impl: bool,
}

/// Whether this trait is treated specially by the standard library
/// specialization lint.
#[derive(StableHash, PartialEq, Clone, Copy, Encodable, Decodable)]
pub enum TraitSpecializationKind {
    /// The default. Specializing on this trait is not allowed.
    None,
    /// Specializing on this trait is allowed because it doesn't have any
    /// methods. For example `Sized` or `FusedIterator`.
    /// Applies to traits with the `rustc_unsafe_specialization_marker`
    /// attribute.
    Marker,
    /// Specializing on this trait is allowed because all of the impls of this
    /// trait are "always applicable". Always applicable means that if
    /// `X<'x>: T<'y>` for any lifetimes, then `for<'a, 'b> X<'a>: T<'b>`.
    /// Applies to traits with the `rustc_specialization_trait` attribute.
    AlwaysApplicable,
}

/// Whether the trait implementation is unrestricted or restricted within a specific module.
#[derive(StableHash, PartialEq, Clone, Copy, Encodable, Decodable)]
pub enum ImplRestrictionKind {
    /// The restriction does not affect this trait, and it can be implemented anywhere.
    Unrestricted,
    /// This trait can only be implemented within the specified module.
    Restricted(DefId, Span),
}

impl ImplRestrictionKind {
    /// Returns `true` if the behavior is allowed/unrestricted in the given module.
    /// A value of `false` indicates that the behavior is prohibited.
    pub fn is_allowed_in(self, module: DefId, tcx: TyCtxt<'_>) -> bool {
        match self {
            ImplRestrictionKind::Unrestricted => true,
            ImplRestrictionKind::Restricted(restricted_to, _) => {
                tcx.is_descendant_of(module, restricted_to)
            }
        }
    }

    /// Obtain the [`Span`] of the restriction. Panics if the restriction is unrestricted.
    pub fn expect_span(self) -> Span {
        match self {
            ImplRestrictionKind::Unrestricted => {
                bug!("called `expect_span` on an unrestricted item")
            }
            ImplRestrictionKind::Restricted(_, span) => span,
        }
    }

    /// Obtain the path of the restriction. If unrestricted, an empty string is returned.
    pub fn restriction_path(self, tcx: TyCtxt<'_>) -> String {
        match self {
            ImplRestrictionKind::Unrestricted => String::new(),
            ImplRestrictionKind::Restricted(restricted_to, _) => {
                if restricted_to.krate == rustc_hir::def_id::LOCAL_CRATE {
                    with_crate_prefix!(with_no_trimmed_paths!(tcx.def_path_str(restricted_to)))
                } else {
                    tcx.def_path_str(restricted_to.krate.as_mod_def_id())
                }
            }
        }
    }
}

#[derive(Default, Debug, StableHash)]
pub struct TraitImpls {
    blanket_impls: Vec<DefId>,
    /// Impls indexed by their simplified self type, for fast lookup.
    non_blanket_impls: FxIndexMap<SimplifiedType, Vec<DefId>>,
    /// A refinement of `non_blanket_impls` for the simplified self types that have enough
    /// impls for the extra indirection to pay off. Simplifying only looks at the outermost
    /// layer of a type, so a trait implemented for many instances of the same generic type
    /// puts all of them in one entry, and every lookup then has to walk all of them.
    ///
    /// See [`RefinedImpls`] for what the refinement is.
    #[stable_hash(ignore)] // Derived from `non_blanket_impls`.
    refined_impls: FxHashMap<SimplifiedType, RefinedImpls>,
}

/// The impls of one simplified self type, split by how precisely their self type is known.
///
/// A self type is *rigid* if it contains nothing that could still be inferred or normalized.
/// Such a type unifies only with a type equal to it (up to regions), so an impl for such a
/// type is relevant only to lookups of that very type. Those impls are indexed by their self
/// type; the rest have to be considered by every lookup.
///
/// Impls carry their position in the unrefined list so that lookups can yield them in the
/// original order.
#[derive(Debug, Default)]
struct RefinedImpls {
    rigid: FxHashMap<RigidSelfTy, Vec<(u32, DefId)>>,
    rest: Vec<(u32, DefId)>,
}

/// Identifies a rigid self type up to regions. See [`RefinedImpls`].
///
/// Types are interned and hash by address, so erasing the regions of two types that unify
/// gives the same hash. That makes the key cheap to build, but only meaningful within one
/// compilation session, which is why the index is not part of the stable hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RigidSelfTy(u64);

impl RigidSelfTy {
    /// Everything that could still turn a type into some other type. Regions are deliberately
    /// not in here: two types that differ only in their regions do unify, which is why the key
    /// is built from the region-erased type.
    const NOT_RIGID: TypeFlags = TypeFlags::HAS_TY_PARAM
        .union(TypeFlags::HAS_CT_PARAM)
        .union(TypeFlags::HAS_TY_INFER)
        .union(TypeFlags::HAS_CT_INFER)
        .union(TypeFlags::HAS_TY_PLACEHOLDER)
        .union(TypeFlags::HAS_CT_PLACEHOLDER)
        .union(TypeFlags::HAS_TY_FRESH)
        .union(TypeFlags::HAS_CT_FRESH)
        .union(TypeFlags::HAS_ALIAS)
        .union(TypeFlags::HAS_ERROR);

    /// Returns the key for `ty`, or `None` if `ty` is not rigid, i.e. if it could still unify
    /// with a type other than itself.
    fn new<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<RigidSelfTy> {
        if ty.has_type_flags(RigidSelfTy::NOT_RIGID) || ty.has_escaping_bound_vars() {
            return None;
        }
        // Erasing regions also anonymizes the bound ones, so types that differ only in the
        // names of their higher-ranked regions get the same key.
        let mut hasher = FxHasher::default();
        tcx.erase_and_anonymize_regions(ty).hash(&mut hasher);
        Some(RigidSelfTy(hasher.finish()))
    }
}

/// Below this many impls for one simplified self type, walking all of them is cheaper than
/// building and consulting the refined index.
const REFINE_THRESHOLD: usize = 16;

impl TraitImpls {
    pub fn is_empty(&self) -> bool {
        self.blanket_impls.is_empty() && self.non_blanket_impls.is_empty()
    }

    pub fn blanket_impls(&self) -> &[DefId] {
        self.blanket_impls.as_slice()
    }

    pub fn non_blanket_impls(&self) -> &FxIndexMap<SimplifiedType, Vec<DefId>> {
        &self.non_blanket_impls
    }
}

impl<'tcx> TraitDef {
    pub fn ancestors(
        &self,
        tcx: TyCtxt<'tcx>,
        of_impl: DefId,
    ) -> Result<specialization_graph::Ancestors<'tcx>, ErrorGuaranteed> {
        specialization_graph::ancestors(tcx, self.def_id, of_impl)
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Iterate over every impl that could possibly match the self type `self_ty`.
    ///
    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn for_each_relevant_impl(
        self,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        mut f: impl FnMut(DefId),
    ) {
        // FIXME: This depends on the set of all impls for the trait. That is
        // unfortunate wrt. incremental compilation.
        //
        // If we want to be faster, we could have separate queries for
        // blanket and non-blanket impls, and compare them separately.
        let impls = self.trait_impls_of(trait_def_id);

        for &impl_def_id in impls.blanket_impls.iter() {
            f(impl_def_id);
        }

        // This way, when searching for some impl for `T: Trait`, we do not look at any impls
        // whose outer level is not a parameter or projection. Especially for things like
        // `T: Clone` this is incredibly useful as we would otherwise look at all the impls
        // of `Clone` for `Option<T>`, `Vec<T>`, `ConcreteType` and so on.
        // Note that we're using `TreatParams::AsRigid` to query `non_blanket_impls` while using
        // `TreatParams::InstantiateWithInfer` while actually adding them.
        if let Some(simp) = fast_reject::simplify_type(self, self_ty, TreatParams::AsRigid) {
            if let Some(refined) = impls.refined_impls.get(&simp)
                && let Some(key) = RigidSelfTy::new(self, self_ty)
            {
                // Only the impls for this very self type can apply, plus the ones whose self
                // type is not pinned down. Both lists are sorted by position, so merging them
                // yields the impls in the same order as the unrefined list would.
                let rigid = refined.rigid.get(&key).map_or(&[][..], |impls| impls.as_slice());
                let mut rigid = rigid.iter().peekable();
                let mut rest = refined.rest.iter().peekable();
                loop {
                    let next = match (rigid.peek(), rest.peek()) {
                        (Some(&&(a, _)), Some(&&(b, _))) => {
                            if a < b {
                                rigid.next()
                            } else {
                                rest.next()
                            }
                        }
                        (Some(_), None) => rigid.next(),
                        (None, Some(_)) => rest.next(),
                        (None, None) => break,
                    };
                    f(next.unwrap().1);
                }
            } else if let Some(impls) = impls.non_blanket_impls.get(&simp) {
                for &impl_def_id in impls {
                    f(impl_def_id);
                }
            }
        } else {
            for &impl_def_id in impls.non_blanket_impls.values().flatten() {
                f(impl_def_id);
            }
        }
    }

    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn non_blanket_impls_for_ty(
        self,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
    ) -> impl Iterator<Item = DefId> {
        let impls = self.trait_impls_of(trait_def_id);
        if let Some(simp) =
            fast_reject::simplify_type(self, self_ty, TreatParams::InstantiateWithInfer)
        {
            if let Some(impls) = impls.non_blanket_impls.get(&simp) {
                return impls.iter().copied();
            }
        }

        [].iter().copied()
    }

    /// Returns an iterator containing all impls for `trait_def_id`.
    ///
    /// `trait_def_id` MUST BE the `DefId` of a trait.
    pub fn all_impls(self, trait_def_id: DefId) -> impl Iterator<Item = DefId> {
        let TraitImpls { blanket_impls, non_blanket_impls, .. } = self.trait_impls_of(trait_def_id);

        blanket_impls.iter().chain(non_blanket_impls.iter().flat_map(|(_, v)| v)).cloned()
    }
}

/// Query provider for `trait_impls_of`.
pub(super) fn trait_impls_of_provider(tcx: TyCtxt<'_>, trait_id: DefId) -> TraitImpls {
    let mut impls = TraitImpls::default();

    // Traits defined in the current crate can't have impls in upstream
    // crates, so we don't bother querying the cstore.
    if !trait_id.is_local() {
        for &cnum in tcx.crates(()).iter() {
            for &(impl_def_id, simplified_self_ty) in
                tcx.implementations_of_trait((cnum, trait_id)).iter()
            {
                if let Some(simplified_self_ty) = simplified_self_ty {
                    impls
                        .non_blanket_impls
                        .entry(simplified_self_ty)
                        .or_default()
                        .push(impl_def_id);
                } else {
                    impls.blanket_impls.push(impl_def_id);
                }
            }
        }
    }

    for &impl_def_id in tcx.local_trait_impls(trait_id) {
        let impl_def_id = impl_def_id.to_def_id();

        let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity().skip_norm_wip();

        if let Some(simplified_self_ty) =
            fast_reject::simplify_type(tcx, impl_self_ty, TreatParams::InstantiateWithInfer)
        {
            impls.non_blanket_impls.entry(simplified_self_ty).or_default().push(impl_def_id);
        } else {
            impls.blanket_impls.push(impl_def_id);
        }
    }

    let mut refined_impls = FxHashMap::default();
    for (&simplified_self_ty, impl_def_ids) in &impls.non_blanket_impls {
        if impl_def_ids.len() < REFINE_THRESHOLD {
            continue;
        }
        let mut refined = RefinedImpls::default();
        for (position, &impl_def_id) in impl_def_ids.iter().enumerate() {
            let position = position as u32;
            let self_ty = tcx.type_of(impl_def_id).instantiate_identity().skip_norm_wip();
            match RigidSelfTy::new(tcx, self_ty) {
                Some(key) => refined.rigid.entry(key).or_default().push((position, impl_def_id)),
                None => refined.rest.push((position, impl_def_id)),
            }
        }
        refined_impls.insert(simplified_self_ty, refined);
    }
    impls.refined_impls = refined_impls;

    impls
}

/// Query provider for `incoherent_impls`.
pub(super) fn incoherent_impls_provider(tcx: TyCtxt<'_>, simp: SimplifiedType) -> &[DefId] {
    if let Some(def_id) = simp.def()
        && !find_attr!(tcx, def_id, RustcHasIncoherentInherentImpls)
    {
        return &[];
    }

    let mut impls = Vec::new();
    for cnum in iter::once(LOCAL_CRATE).chain(tcx.crates(()).iter().copied()) {
        for &impl_def_id in tcx.crate_incoherent_impls((cnum, simp)) {
            impls.push(impl_def_id)
        }
    }
    debug!(?impls);

    tcx.arena.alloc_slice(&impls)
}

pub(super) fn traits_provider(tcx: TyCtxt<'_>, _: LocalCrate) -> &[DefId] {
    let mut traits = Vec::new();
    for id in tcx.hir_free_items() {
        if matches!(tcx.def_kind(id.owner_id), DefKind::Trait | DefKind::TraitAlias) {
            traits.push(id.owner_id.to_def_id())
        }
    }

    tcx.arena.alloc_slice(&traits)
}

pub(super) fn trait_impls_in_crate_provider(tcx: TyCtxt<'_>, _: LocalCrate) -> &[DefId] {
    let mut trait_impls = Vec::new();
    for id in tcx.hir_free_items() {
        if tcx.def_kind(id.owner_id) == (DefKind::Impl { of_trait: true }) {
            trait_impls.push(id.owner_id.to_def_id())
        }
    }

    tcx.arena.alloc_slice(&trait_impls)
}
