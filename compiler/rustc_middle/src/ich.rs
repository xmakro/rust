use std::hash::Hash;
use std::sync::Arc;

use rustc_data_structures::stable_hash::{
    RawDefId, RawDefPathHash, RawSpan, StableHash, StableHashControls, StableHashCtxt, StableHasher,
};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_session::Session;
use rustc_session::cstore::Untracked;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, DUMMY_SP, Pos, SourceFile, Span};

/// This is the context state available during incr. comp. hashing. It contains
/// enough information to transform `DefId`s and `HirId`s into stable `DefPath`s (i.e.,
/// a reference to the `TyCtxt`) and it holds a few caches for speeding up various
/// things (e.g., each `DefId`/`DefPath` is only hashed once).
pub struct StableHashState<'a> {
    untracked: &'a Untracked,
    // The value of `-Z incremental-ignore-spans`.
    // This field should only be used by `unstable_opts_incremental_ignore_span`
    incremental_ignore_spans: bool,
    source_map: &'a SourceMap,
    /// One-entry cache for [`SourceMap::lookup_source_file`]; consecutive hashed spans are
    /// almost always in the same file.
    file_cache: Option<Arc<SourceFile>>,
    stable_hash_controls: StableHashControls,
}

impl<'a> StableHashState<'a> {
    #[inline]
    pub fn new(sess: &'a Session, untracked: &'a Untracked) -> Self {
        let hash_spans_initial = !sess.opts.unstable_opts.incremental_ignore_spans;

        StableHashState {
            untracked,
            incremental_ignore_spans: sess.opts.unstable_opts.incremental_ignore_spans,
            source_map: sess.source_map(),
            file_cache: None,
            stable_hash_controls: StableHashControls { hash_spans: hash_spans_initial },
        }
    }

    #[inline]
    pub fn while_hashing_spans<F: FnOnce(&mut Self)>(&mut self, hash_spans: bool, f: F) {
        let prev_hash_spans = self.stable_hash_controls.hash_spans;
        self.stable_hash_controls.hash_spans = hash_spans;
        f(self);
        self.stable_hash_controls.hash_spans = prev_hash_spans;
    }

    /// Returns the file containing `pos`, or `None` if no file contains it. Empty files are
    /// treated as containing no positions, matching the behavior of the previous
    /// line-and-column lookup (an empty file has no lines).
    #[inline]
    fn source_file_for_pos(&mut self, pos: BytePos) -> Option<&Arc<SourceFile>> {
        if !matches!(&self.file_cache, Some(file) if file.contains(pos)) {
            if self.source_map.files().is_empty() {
                return None;
            }
            // `lookup_source_file` returns the last file with `start_pos <= pos`, so the
            // `contains` check below rejects positions past the end of that file.
            let file = self.source_map.lookup_source_file(pos);
            if !file.contains(pos) || file.is_empty() {
                return None;
            }
            self.file_cache = Some(file);
        }
        self.file_cache.as_ref()
    }

    #[inline]
    fn def_span(&self, def_id: LocalDefId) -> Span {
        self.untracked.source_span.get(def_id).unwrap_or(DUMMY_SP)
    }

    #[inline]
    pub fn stable_hash_controls(&self) -> StableHashControls {
        self.stable_hash_controls
    }
}

impl<'a> StableHashCtxt for StableHashState<'a> {
    /// Hashes a span in a stable way. The raw `BytePos` fields are offsets into the `SourceMap`,
    /// which are not stable across sessions, so we hash the (file, offset within file, length)
    /// triple instead. Hashing both the start and the length keeps spans that differ only in
    /// their end position distinct (see issue #74890).
    ///
    /// This fingerprint covers a span's *position* but nothing derived from the file's line
    /// structure: an edit that moves a line break without changing byte offsets leaves it
    /// unchanged. That is sound only because every consumer that renders line/column
    /// information from a span into a cached artifact (`#[track_caller]` locations, debuginfo
    /// line tables, coverage mappings, pretty-printed closure paths in cached diagnostics)
    /// obtains the file through `TyCtxt::lookup_line_tracked` or
    /// `TyCtxt::source_file_tracked`, which record a dependency on the
    /// `file_lines_prefix_hash` query for the span's file and line. Code that derives
    /// line/column data inside a tracked context and stores the result MUST go through those
    /// lookups. (The dependency covers line indices and character columns, not display
    /// columns; see `lookup_line_tracked`.)
    ///
    /// IMPORTANT: `TAG_FULL_SPAN` in the incremental on-disk cache must encode enough to
    /// reconstruct the exact span, so that a reloaded span re-hashes to the fingerprint its
    /// containing value was stored under (see `CacheEncoder::encode_span`).
    #[inline]
    fn stable_hash_span(&mut self, raw_span: RawSpan, hasher: &mut StableHasher) {
        const TAG_VALID_SPAN: u8 = 0;
        const TAG_INVALID_SPAN: u8 = 1;
        const TAG_RELATIVE_SPAN: u8 = 2;

        if !self.stable_hash_controls().hash_spans {
            return;
        }

        let span = Span::from_raw_span(raw_span);
        let span = span.data_untracked();
        span.ctxt.stable_hash(self, hasher);
        span.parent.stable_hash(self, hasher);

        if span.is_dummy() {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        let parent = span.parent.map(|parent| self.def_span(parent).data_untracked());
        if let Some(parent) = parent
            && parent.contains(span)
        {
            // This span is enclosed in a definition: only hash the relative position. This catches
            // a subset of the cases from the `file.contains(parent.lo)`. But we can do this check
            // cheaply without any `SourceMap` lookup.
            Hash::hash(&TAG_RELATIVE_SPAN, hasher);
            (span.lo - parent.lo).to_u32().stable_hash(self, hasher);
            (span.hi - parent.lo).to_u32().stable_hash(self, hasher);
            return;
        }

        let Some(file) = self.source_file_for_pos(span.lo) else {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        };

        if span.hi > file.end_position() {
            // The span crosses a file boundary; treat it like an invalid span.
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        if let Some(parent) = parent
            && file.contains(parent.lo)
        {
            // This span is relative to another span in the same file,
            // only hash the relative position.
            Hash::hash(&TAG_RELATIVE_SPAN, hasher);
            Hash::hash(&(span.lo.0.wrapping_sub(parent.lo.0)), hasher);
            Hash::hash(&(span.hi.0.wrapping_sub(parent.lo.0)), hasher);
            return;
        }

        Hash::hash(&TAG_VALID_SPAN, hasher);
        Hash::hash(&file.stable_id, hasher);
        Hash::hash(&file.relative_position(span.lo).to_u32(), hasher);
        Hash::hash(&(span.hi - span.lo).0, hasher);
    }

    /// Hashes a definition's span used as an anchor: (file, length), deliberately excluding
    /// the offset so that a definition which merely moves keeps its fingerprint. Everything
    /// position-observable is covered elsewhere: relative spans re-anchor on decode, line
    /// renderings depend on `def_position` and `file_lines_prefix_hash`, and cross-file moves
    /// change the hashed `stable_id`. See `rustc_span::AnchorSpan`.
    fn stable_hash_anchor_span(&mut self, raw_span: RawSpan, hasher: &mut StableHasher) {
        const TAG_VALID_SPAN: u8 = 0;
        const TAG_INVALID_SPAN: u8 = 1;

        if !self.stable_hash_controls().hash_spans {
            return;
        }

        let span = Span::from_raw_span(raw_span);
        let span = span.data_untracked();
        span.ctxt.stable_hash(self, hasher);
        debug_assert_eq!(span.parent, None, "anchor spans must be absolute");

        if span.is_dummy() {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        let Some(file) = self.source_file_for_pos(span.lo) else {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        };

        if span.hi > file.end_position() {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        Hash::hash(&TAG_VALID_SPAN, hasher);
        Hash::hash(&file.stable_id, hasher);
        Hash::hash(&(span.hi - span.lo).0, hasher);
    }

    #[inline]
    fn def_path_hash(&self, raw_def_id: RawDefId) -> RawDefPathHash {
        let def_id = DefId::from_raw_def_id(raw_def_id);
        if let Some(def_id) = def_id.as_local() {
            self.untracked.definitions.read().def_path_hash(def_id)
        } else {
            self.untracked.cstore.read().def_path_hash(def_id)
        }
        .to_raw_def_path_hash()
    }

    /// Assert that the provided `StableHashCtxt` is configured with the default
    /// `StableHashControls`. We should always have bailed out before getting to here with a
    /// non-default mode. With this check in place, we can avoid the need to maintain separate
    /// versions of `ExpnData` hashes for each permutation of `StableHashControls` settings.
    #[inline]
    fn assert_default_stable_hash_controls(&self, msg: &str) {
        let stable_hash_controls = self.stable_hash_controls;
        let StableHashControls { hash_spans } = stable_hash_controls;

        // Note that we require that `hash_spans` be the inverse of the global `-Z
        // incremental-ignore-spans` option. Normally, this option is disabled, in which case
        // `hash_spans` must be true.
        //
        // Span hashing can also be disabled without `-Z incremental-ignore-spans`. This is the
        // case for instance when building a hash for name mangling. Such configuration must not be
        // used for metadata.
        assert_eq!(
            hash_spans, !self.incremental_ignore_spans,
            "Attempted hashing of {msg} with non-default StableHashControls: {stable_hash_controls:?}"
        );
    }

    #[inline]
    fn stable_hash_controls(&self) -> StableHashControls {
        self.stable_hash_controls
    }
}
