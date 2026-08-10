//! Filters a `ParamEnv` down to the caller bounds that are relevant to a
//! canonical query goal, so that queries about the same goal from functions
//! with different (but irrelevant) where-clauses share a canonical key.
//!
//! A caller bound can only participate in proving a goal if its head can
//! unify with some subgoal reached while proving that goal. Subgoals are
//! built from the goal's own types, from (global) impl definitions, and from
//! the right-hand sides of caller bounds that were already usable. Since
//! generic parameters are rigid during trait solving, a clause head that
//! mentions only parameters the goal (transitively) never mentions can never
//! match any subgoal. We over-approximate this with a fixpoint over bitmasks
//! of type/const parameter indices.
//!
//! Values containing type/const inference variables (or non-region bound
//! vars, e.g. embedded canonical values) unify with anything, so filtering is
//! skipped for them entirely.

use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_middle::ty::{
    self, EnvClauseClass, EnvClauseEntry, EnvClauseRelevance, Ty, TyCtxt, TypeFlags,
    TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};

/// Collects the set of type/const generic param indices as a bitmask.
/// Type and const params share the generic param index space.
#[derive(Default)]
struct ParamMaskVisitor {
    mask: u128,
    overflow: bool,
}

impl ParamMaskVisitor {
    #[inline]
    fn add_index(&mut self, index: u32) {
        if index < 128 {
            self.mask |= 1u128 << index;
        } else {
            self.overflow = true;
        }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ParamMaskVisitor {
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        if !t.has_type_flags(TypeFlags::HAS_PARAM) {
            return;
        }
        if let ty::Param(p) = t.kind() {
            self.add_index(p.index);
        }
        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) {
        if !c.has_type_flags(TypeFlags::HAS_PARAM) {
            return;
        }
        if let ty::ConstKind::Param(p) = c.kind() {
            self.add_index(p.index);
        }
        c.super_visit_with(self)
    }

    fn visit_region(&mut self, _: ty::Region<'tcx>) {}
}

fn param_mask<'tcx>(value: &impl TypeVisitable<TyCtxt<'tcx>>) -> (u128, bool) {
    let mut visitor = ParamMaskVisitor::default();
    value.visit_with(&mut visitor);
    (visitor.mask, visitor.overflow)
}

fn clause_entry<'tcx>(clause: ty::Clause<'tcx>) -> (EnvClauseEntry, bool) {
    let (all_mask, mut overflow) = param_mask(&clause);
    let kind = clause.kind().skip_binder();
    let (class, head_mask) = match kind {
        ty::ClauseKind::Trait(pred) => {
            let (m, o) = param_mask(&pred.trait_ref);
            overflow |= o;
            (EnvClauseClass::Head, m)
        }
        ty::ClauseKind::HostEffect(pred) => {
            let (m, o) = param_mask(&pred.trait_ref);
            overflow |= o;
            (EnvClauseClass::Head, m)
        }
        ty::ClauseKind::Projection(pred) => {
            let (m, o) = param_mask(&pred.projection_term);
            overflow |= o;
            (EnvClauseClass::Head, m)
        }
        ty::ClauseKind::TypeOutlives(pred) => {
            let subject = pred.0;
            let (m, o) = param_mask(&subject);
            overflow |= o;
            (
                EnvClauseClass::TypeOutlives {
                    subject_has_alias: subject.has_type_flags(TypeFlags::HAS_ALIAS),
                    subject_has_free_regions: subject.has_type_flags(TypeFlags::HAS_FREE_REGIONS),
                },
                m,
            )
        }
        // A `ConstArgHasType(N, Ty)` or `ConstEvaluatable(expr)` subgoal can
        // only mention a const param reachable from the goal.
        ty::ClauseKind::ConstArgHasType(ct, _) => {
            let (m, o) = param_mask(&ct);
            overflow |= o;
            (EnvClauseClass::Head, m)
        }
        ty::ClauseKind::ConstEvaluatable(ct) => {
            let (m, o) = param_mask(&ct);
            overflow |= o;
            (EnvClauseClass::Head, m)
        }
        // Region outlives, well-formedness, unstable-feature markers and
        // anything else: conservatively always kept.
        _ => (EnvClauseClass::Always, all_mask),
    };
    (EnvClauseEntry { class, head_mask, all_mask }, overflow)
}

fn relevance_for<'tcx>(tcx: TyCtxt<'tcx>, clauses: ty::Clauses<'tcx>) -> EnvClauseRelevance {
    if let Some(cached) = tcx.env_clause_relevance_cache.lock().get(&clauses) {
        return cached.clone();
    }
    let mut overflow = false;
    let entries: Vec<EnvClauseEntry> = clauses
        .iter()
        .map(|clause| {
            let (entry, o) = clause_entry(clause);
            overflow |= o;
            entry
        })
        .collect();
    let relevance = EnvClauseRelevance { entries: entries.into(), overflow };
    tcx.env_clause_relevance_cache.lock().insert(clauses, relevance.clone());
    relevance
}

#[inline]
fn keep(entry: &EnvClauseEntry, p: u128) -> bool {
    match entry.class {
        EnvClauseClass::Always => true,
        // A trait-like clause can match a subgoal if its head mentions a
        // reachable param, or if its head is param-free (global heads can
        // match subgoals built from global impl definitions).
        EnvClauseClass::Head => entry.head_mask == 0 || entry.head_mask & p != 0,
        // Inside canonical queries type-outlives caller bounds are only ever
        // matched against param- or alias-headed subjects; outlives goals on
        // concrete types decompose structurally without consulting the env.
        // Keep param-free subjects only if they contain aliases or free
        // regions; a fully global rigid subject (e.g. `SomeStruct: 'static`)
        // is vacuous for the query.
        EnvClauseClass::TypeOutlives { subject_has_alias, subject_has_free_regions } => {
            if entry.head_mask != 0 {
                entry.head_mask & p != 0
            } else {
                subject_has_alias || subject_has_free_regions
            }
        }
    }
}

static STAT_TOTAL: AtomicUsize = AtomicUsize::new(0);
static STAT_SKIP_WILDCARD: AtomicUsize = AtomicUsize::new(0);
static STAT_SKIP_PARAM: AtomicUsize = AtomicUsize::new(0);
static STAT_SKIP_OVERFLOW: AtomicUsize = AtomicUsize::new(0);
static STAT_CACHED: AtomicUsize = AtomicUsize::new(0);
static STAT_ALL_KEPT: AtomicUsize = AtomicUsize::new(0);
static STAT_FILTERED: AtomicUsize = AtomicUsize::new(0);
static STAT_CLAUSES_SEEN: AtomicUsize = AtomicUsize::new(0);
static STAT_CLAUSES_DROPPED: AtomicUsize = AtomicUsize::new(0);

fn stats_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUSTC_ENV_FILTER_STATS").is_some())
}

fn filter_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("RUSTC_ENV_FILTER_DISABLE").is_some())
}

fn bump(counter: &AtomicUsize) {
    if stats_enabled() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

fn stats_interval() -> usize {
    static INTERVAL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("RUSTC_ENV_FILTER_STATS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(16384)
    })
}

fn bump_total() {
    if stats_enabled() {
        let n = STAT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        if n % stats_interval() == 0 {
            eprintln!(
                "env-filter-stats: total={} wildcard={} param={} overflow={} cached={} \
                 all_kept={} filtered={} clauses_seen={} clauses_dropped={}",
                n,
                STAT_SKIP_WILDCARD.load(Ordering::Relaxed),
                STAT_SKIP_PARAM.load(Ordering::Relaxed),
                STAT_SKIP_OVERFLOW.load(Ordering::Relaxed),
                STAT_CACHED.load(Ordering::Relaxed),
                STAT_ALL_KEPT.load(Ordering::Relaxed),
                STAT_FILTERED.load(Ordering::Relaxed),
                STAT_CLAUSES_SEEN.load(Ordering::Relaxed),
                STAT_CLAUSES_DROPPED.load(Ordering::Relaxed),
            );
        }
    }
}

/// Returns the subset of `param_env`'s caller bounds that can influence
/// proving goals about `value`, or `param_env` unchanged when filtering is
/// not applicable.
pub(super) fn filter_param_env_for_goal<'tcx, V>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    value: &V,
) -> ty::ParamEnv<'tcx>
where
    V: TypeVisitable<TyCtxt<'tcx>>,
{
    let clauses = param_env.caller_bounds();
    if clauses.is_empty() || filter_disabled() {
        return param_env;
    }
    bump_total();
    // Inference variables unify with anything, so every clause is potentially
    // relevant. Non-region bound vars cover values embedding canonical
    // binders (e.g. user type annotations) which are instantiated with fresh
    // inference variables inside the query.
    if value.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_CT_INFER)
        || value.has_non_region_bound_vars()
    {
        bump(&STAT_SKIP_WILDCARD);
        return param_env;
    }

    // Only filter for param-free goals. This is the common fragmentation
    // case (identical concrete goals from functions with unrelated
    // where-clauses), and restricting to it means we never need to walk the
    // goal to collect its params: the per-call cost is a few flag checks and
    // one hash lookup. Goals mentioning generic params keep the full env.
    if value.has_type_flags(TypeFlags::HAS_PARAM) {
        bump(&STAT_SKIP_PARAM);
        return param_env;
    }
    let goal_mask = 0u128;

    if let Some(&cached) = tcx.env_filtered_cache.lock().get(&(clauses, goal_mask)) {
        bump(&STAT_CACHED);
        return cached;
    }

    let relevance = relevance_for(tcx, clauses);
    if relevance.overflow {
        bump(&STAT_SKIP_OVERFLOW);
        return param_env;
    }

    // Reachable-params fixpoint: a usable clause's right-hand side can
    // introduce further params into subgoals (e.g. projection clause RHS).
    let mut p = goal_mask;
    loop {
        let mut changed = false;
        for entry in relevance.entries.iter() {
            if keep(entry, p) && entry.all_mask & !p != 0 {
                p |= entry.all_mask;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let kept_count = relevance.entries.iter().filter(|entry| keep(entry, p)).count();
    if stats_enabled() {
        STAT_CLAUSES_SEEN.fetch_add(clauses.len(), Ordering::Relaxed);
        STAT_CLAUSES_DROPPED.fetch_add(clauses.len() - kept_count, Ordering::Relaxed);
    }
    let filtered = if kept_count == clauses.len() {
        bump(&STAT_ALL_KEPT);
        param_env
    } else {
        bump(&STAT_FILTERED);
        let kept: Vec<ty::Clause<'tcx>> = clauses
            .iter()
            .zip(relevance.entries.iter())
            .filter(|(_, entry)| keep(entry, p))
            .map(|(clause, _)| clause)
            .collect();
        ty::ParamEnv::new(tcx.mk_clauses(&kept))
    };
    tcx.env_filtered_cache.lock().insert((clauses, goal_mask), filtered);
    filtered
}
