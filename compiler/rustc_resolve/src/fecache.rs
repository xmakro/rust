//! Frontend cache (`-Zfrontend-cache`) support: recording and replaying the
//! definitions created during macro expansion.
//!
//! A def's `DefKey` (and therefore its `DefPathHash` and its raw `DefIndex`)
//! depends on the order defs are created in, because sibling disambiguators are
//! handed out by per-parent counters. The snapshot therefore records the exact
//! creation-order log, and replaying it through the same machinery reproduces
//! the def table bit-for-bit. `Resolver::create_def` then recognizes replayed
//! nodes and returns the existing def, so the def-collection walk over the
//! restored AST only redoes the session-local per-owner bookkeeping.

use rustc_ast as ast;
use rustc_hir::attrs::StrippedCfgItem;
use rustc_hir::def::DefKind;
use rustc_hir::definitions::PerParentDisambiguatorsMap;
use rustc_span::hygiene::ExpnId;
use rustc_span::{Span, Symbol};

use crate::{LocalDefId, Resolver};

/// One `Resolver::create_def` call during expansion, in creation order.
pub struct FeDefRow {
    pub node_id: ast::NodeId,
    pub parent: LocalDefId,
    pub name: Option<Symbol>,
    pub def_kind: DefKind,
    pub expn_id: ExpnId,
    pub span: Span,
    pub is_owner: bool,
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    pub fn fecache_start_recording(&mut self) {
        self.fecache_def_log = Some(Vec::new());
    }

    /// The recorded def log, or `None` if it contains defs that cannot be
    /// replayed (defs without an AST node).
    pub fn fecache_take_recorded(&mut self) -> Option<Vec<FeDefRow>> {
        let rows = self.fecache_def_log.take()?;
        if rows.iter().any(|r| r.node_id == ast::DUMMY_NODE_ID) {
            return None;
        }
        Some(rows)
    }

    /// Replays a def creation log recorded by a previous session. Must run
    /// before def collection so that `create_def` sees the replayed nodes.
    pub fn fecache_replay_defs(&mut self, rows: Vec<FeDefRow>) {
        for row in rows {
            let disambiguator = self.disambiguators.get_or_create(row.parent);
            let feed = self.tcx.create_def(row.parent, row.name, row.def_kind, None, disambiguator);
            let def_id = feed.def_id();
            if row.expn_id != ExpnId::root() {
                self.expn_that_defined.insert(def_id, row.expn_id);
            }
            let _id = self.tcx.untracked().source_span.push(row.span);
            debug_assert_eq!(_id, def_id);
            self.fecache_restored_defs.insert(row.node_id, def_id);
        }
    }

    pub fn fecache_next_node_id(&self) -> ast::NodeId {
        self.next_node_id
    }

    /// Items removed by cfg stripping, recorded during expansion for
    /// "exists but was cfg-ed out" diagnostics and crate metadata.
    pub fn fecache_stripped_cfg_items(&self) -> &[StrippedCfgItem<ast::NodeId>] {
        &self.stripped_cfg_items
    }

    pub fn fecache_set_stripped_cfg_items(&mut self, items: Vec<StrippedCfgItem<ast::NodeId>>) {
        self.stripped_cfg_items = items;
    }

    pub fn fecache_set_next_node_id(&mut self, id: ast::NodeId) {
        self.next_node_id = id;
    }

    /// The names resolved through each glob import so far. Macro path
    /// resolution during expansion records entries here, which a replayed
    /// session would otherwise miss.
    pub fn fecache_glob_map(&self) -> Vec<(LocalDefId, Vec<Symbol>)> {
        self.glob_map
            .iter()
            .map(|(def_id, names)| (*def_id, names.iter().copied().collect()))
            .collect()
    }

    pub fn fecache_extend_glob_map(&mut self, entries: Vec<(LocalDefId, Vec<Symbol>)>) {
        for (def_id, names) in entries {
            self.glob_map.entry(def_id).or_default().extend(names);
        }
    }
}
