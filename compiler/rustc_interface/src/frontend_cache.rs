//! Persistent frontend cache (`-Zfrontend-cache`).
//!
//! Macro expansion is deterministic given its inputs: the source files, the
//! loaded dependency crates, the consulted environment variables and the
//! compiler configuration. Every incremental session today re-runs all of it
//! (matching, transcription, fragment parsing, proc macro execution) only to
//! produce a bit-identical expanded AST. This module snapshots the expanded
//! crate together with the session state expansion created (hygiene tables,
//! def table, source map additions, crate load order) and replays it in later
//! sessions whose frontend inputs are bit-identical.
//!
//! Everything is encoded with *raw* session-local ids (`BytePos`,
//! `SyntaxContext`, `ExpnId`, `DefIndex`, `CrateNum`, `NodeId`). This is valid
//! because the replay reconstructs the id spaces exactly: parsing reproduces
//! the source map prefix, dependency crates are re-loaded in recorded order,
//! hygiene rows are appended onto a verified table prefix, and the def
//! creation log is replayed through the regular `create_def` machinery. If any
//! validation step fails the cache is ignored and expansion runs normally.
//!
//! Known prototype limitations, on purpose for now:
//! - buffered lints from expansion are not snapshotted, so some
//!   expansion-emitted warnings disappear on cache hits (`unused_macros` is
//!   handled by skipping the check on hits);
//! - proc macros that read untracked state (files without `tracked_path`,
//!   network, time) are assumed deterministic, like the query system already
//!   assumes for incremental reuse of their downstream products;
//! - proc-macro crates themselves are not cached.

use std::path::PathBuf;
use std::sync::Arc;

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefIndex, LocalDefId};
use rustc_metadata::creader::CStore;
use rustc_middle::ty::TyCtxt;
use rustc_resolve::Resolver;
use rustc_resolve::fecache::FeDefRow;
use rustc_serialize::opaque::MemDecoder;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_session::config::CrateType;
use rustc_session::cstore::CrateDepKind;
use rustc_span::hygiene::{ExpnId, ExpnIndex, fecache as hygiene_fecache};
use rustc_span::{
    AttrId, BlobDecoder, BytePos, ByteSymbol, FileName, SourceFile, SourceFileHash, SourceFileHashAlgorithm,
    Span, SpanData, SpanDecoder, SpanEncoder, Symbol, SyntaxContext,
};
use tracing::debug;


macro_rules! fedbg {
    ($($arg:tt)*) => {
        if std::env::var_os("FECACHE_DEBUG").is_some() {
            eprintln!("fecache: {}", format_args!($($arg)*));
        }
        debug!($($arg)*);
    };
}

const MAGIC: u32 = 0x52464543; // "RFEC"
const VERSION: u32 = 1;

pub(crate) fn enabled(tcx: TyCtxt<'_>) -> bool {
    let sess = tcx.sess;
    sess.opts.incremental.is_some()
        && sess.opts.unstable_opts.frontend_cache.unwrap_or(true)
        && !tcx.crate_types().contains(&CrateType::ProcMacro)
}

fn snapshot_path(tcx: TyCtxt<'_>) -> Option<PathBuf> {
    let incr_dir = tcx.sess.opts.incremental.as_ref()?;
    let crate_name = tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE);
    let stable_id = tcx.stable_crate_id(rustc_hir::def_id::LOCAL_CRATE);
    Some(incr_dir.join(format!("fecache-{crate_name}-{:016x}.bin", stable_id.as_u64())))
}

// ------------------------------------------------------------------------
// Encoder / decoder with raw span encoding and symbol side tables

struct FeEncoder {
    data: Vec<u8>,
    syms: FxIndexSet<Symbol>,
    byte_syms: FxIndexSet<ByteSymbol>,
}

impl FeEncoder {
    fn new() -> Self {
        FeEncoder { data: Vec::new(), syms: FxIndexSet::default(), byte_syms: FxIndexSet::default() }
    }
}

macro_rules! delegate_emit {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, v: $ty) {
            self.data.extend_from_slice(&v.to_le_bytes());
        })*
    };
}

impl Encoder for FeEncoder {
    delegate_emit! {
        emit_u128(u128); emit_u64(u64); emit_u32(u32); emit_u16(u16);
        emit_i128(i128); emit_i64(i64); emit_i32(i32); emit_i16(i16);
    }

    fn emit_usize(&mut self, v: usize) {
        self.emit_u64(v as u64);
    }

    fn emit_isize(&mut self, v: isize) {
        self.emit_i64(v as i64);
    }

    fn emit_u8(&mut self, v: u8) {
        self.data.push(v);
    }

    fn emit_str(&mut self, v: &str) {
        self.emit_usize(v.len());
        self.data.extend_from_slice(v.as_bytes());
    }

    fn emit_raw_bytes(&mut self, v: &[u8]) {
        self.data.extend_from_slice(v);
    }
}

impl SpanEncoder for FeEncoder {
    fn encode_span(&mut self, span: Span) {
        let data = span.data();
        self.emit_u32(data.lo.0);
        self.emit_u32(data.hi.0);
        self.encode_syntax_context(data.ctxt);
        match data.parent {
            None => self.emit_u32(u32::MAX),
            Some(parent) => {
                assert!(parent.local_def_index.as_u32() != u32::MAX);
                self.emit_u32(parent.local_def_index.as_u32())
            }
        }
    }

    fn encode_symbol(&mut self, sym: Symbol) {
        let (idx, _) = self.syms.insert_full(sym);
        self.emit_u32(idx as u32);
    }

    fn encode_byte_symbol(&mut self, byte_sym: ByteSymbol) {
        let (idx, _) = self.byte_syms.insert_full(byte_sym);
        self.emit_u32(idx as u32);
    }

    fn encode_expn_id(&mut self, expn_id: ExpnId) {
        self.emit_u32(expn_id.krate.as_u32());
        self.emit_u32(expn_id.local_id.as_u32());
    }

    fn encode_syntax_context(&mut self, syntax_context: SyntaxContext) {
        self.emit_u32(syntax_context.as_u32());
    }

    fn encode_crate_num(&mut self, crate_num: CrateNum) {
        self.emit_u32(crate_num.as_u32());
    }

    fn encode_def_index(&mut self, def_index: DefIndex) {
        self.emit_u32(def_index.as_u32());
    }

    fn encode_def_id(&mut self, def_id: rustc_hir::def_id::DefId) {
        self.encode_crate_num(def_id.krate);
        self.encode_def_index(def_id.index);
    }
}

struct FeDecoder<'a, 'psess> {
    mem: MemDecoder<'a>,
    syms: Vec<Symbol>,
    byte_syms: Vec<ByteSymbol>,
    psess: &'psess rustc_session::parse::ParseSess,
    /// Maps the recorded session's `CrateNum`s to this session's. The replayed
    /// crate loads resolve by name and hash, so the raw numbering may differ.
    cnum_map: FxHashMap<u32, u32>,
}

impl FeDecoder<'_, '_> {
    fn map_cnum(&self, raw: u32) -> CrateNum {
        match self.cnum_map.get(&raw) {
            Some(&mapped) => CrateNum::from_u32(mapped),
            None => CrateNum::from_u32(raw),
        }
    }
}

macro_rules! delegate_read {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self) -> $ty {
            const N: usize = size_of::<$ty>();
            let bytes: [u8; N] = self.mem.read_raw_bytes(N).try_into().unwrap();
            <$ty>::from_le_bytes(bytes)
        })*
    };
}

impl<'a, 'psess> Decoder for FeDecoder<'a, 'psess> {
    delegate_read! {
        read_u128(u128); read_u64(u64); read_u32(u32); read_u16(u16);
        read_i128(i128); read_i64(i64); read_i32(i32); read_i16(i16);
    }

    fn read_usize(&mut self) -> usize {
        self.read_u64() as usize
    }
    fn read_isize(&mut self) -> isize {
        self.read_i64() as isize
    }
    fn read_u8(&mut self) -> u8 {
        self.mem.read_raw_bytes(1)[0]
    }
    fn read_str(&mut self) -> &str {
        let len = self.read_usize();
        let bytes = self.mem.read_raw_bytes(len);
        std::str::from_utf8(bytes).unwrap()
    }
    fn read_raw_bytes(&mut self, len: usize) -> &[u8] {
        self.mem.read_raw_bytes(len)
    }
    fn peek_byte(&self) -> u8 {
        self.mem.peek_byte()
    }
    fn position(&self) -> usize {
        self.mem.position()
    }
}

impl<'a, 'psess> rustc_span::BlobDecoder for FeDecoder<'a, 'psess> {
    fn decode_symbol(&mut self) -> Symbol {
        let idx = self.read_u32() as usize;
        self.syms[idx]
    }

    fn decode_byte_symbol(&mut self) -> ByteSymbol {
        let idx = self.read_u32() as usize;
        self.byte_syms[idx]
    }

    fn decode_def_index(&mut self) -> DefIndex {
        DefIndex::from_u32(self.read_u32())
    }
}

impl<'a, 'psess> SpanDecoder for FeDecoder<'a, 'psess> {
    fn decode_span(&mut self) -> Span {
        let lo = BytePos(self.read_u32());
        let hi = BytePos(self.read_u32());
        let ctxt = self.decode_syntax_context();
        let parent = match self.read_u32() {
            u32::MAX => None,
            raw => Some(LocalDefId { local_def_index: DefIndex::from_u32(raw) }),
        };
        SpanData { lo, hi, ctxt, parent }.span()
    }

    fn decode_expn_id(&mut self) -> ExpnId {
        let raw = self.read_u32();
        let krate = self.map_cnum(raw);
        let local_id = ExpnIndex::from_u32(self.read_u32());
        ExpnId { krate, local_id }
    }

    fn decode_syntax_context(&mut self) -> SyntaxContext {
        SyntaxContext::from_u32(self.read_u32())
    }

    fn decode_crate_num(&mut self) -> CrateNum {
        let raw = self.read_u32();
        self.map_cnum(raw)
    }

    fn decode_def_id(&mut self) -> rustc_hir::def_id::DefId {
        let raw = self.read_u32();
        let krate = self.map_cnum(raw);
        let index = DefIndex::from_u32(self.read_u32());
        rustc_hir::def_id::DefId { krate, index }
    }

    fn decode_attr_id(&mut self) -> AttrId {
        // Attribute ids are session-local diagnostics bookkeeping and are never
        // compared across sessions: mint fresh ones, like metadata decoding does.
        self.psess.attr_id_generator.mk_attr_id()
    }
}

// ------------------------------------------------------------------------
// Source file rows

enum FileRow {
    /// A file read from disk this session, reloadable from its path.
    Local { path: PathBuf, src_hash: SourceFileHash, start_pos: u32, len: u32 },
    /// A file imported from a dependency's metadata.
    Imported { cnum: u32, index: u32, start_pos: u32, len: u32 },
    /// A file synthesized in memory (e.g. proc macro sources); content inlined.
    Virtual { name: FileName, src: String, start_pos: u32, len: u32 },
}

fn classify_files(tcx: TyCtxt<'_>) -> Option<Vec<FileRow>> {
    let source_map = tcx.sess.source_map();
    let cstore = CStore::from_tcx(tcx);
    // Map (cnum, local start_pos) -> index in the foreign crate's source map.
    let mut import_idx: FxHashMap<(CrateNum, BytePos), u32> = FxHashMap::default();
    for (cnum, _, _, _, _) in cstore.fecache_crates() {
        for (idx, file) in cstore.fecache_imported_files(cnum) {
            import_idx.insert((cnum, file.start_pos), idx);
        }
    }

    let files = source_map.files();
    let mut rows = Vec::with_capacity(files.len());
    for file in files.iter() {
        let start_pos = file.start_pos.0;
        let len = file.normalized_source_len.0;
        if file.cnum != rustc_hir::def_id::LOCAL_CRATE {
            let &index = import_idx.get(&(file.cnum, file.start_pos))?;
            rows.push(FileRow::Imported { cnum: file.cnum.as_u32(), index, start_pos, len });
        } else if let FileName::Real(real) = &file.name
            && let Some(path) = real.local_path()
        {
            rows.push(FileRow::Local {
                path: path.to_path_buf(),
                src_hash: file.src_hash,
                start_pos,
                len,
            });
        } else if let Some(src) = &file.src {
            rows.push(FileRow::Virtual {
                name: file.name.clone(),
                src: (**src).clone(),
                start_pos,
                len,
            });
        } else {
            // A local file we can neither reload nor inline: give up on caching.
            return None;
        }
    }
    Some(rows)
}

fn encode_file_rows(enc: &mut FeEncoder, rows: &[FileRow]) {
    enc.emit_usize(rows.len());
    for row in rows {
        match row {
            FileRow::Local { path, src_hash, start_pos, len } => {
                enc.emit_u8(0);
                enc.emit_str(&path.to_string_lossy());
                src_hash.encode(enc);
                enc.emit_u32(*start_pos);
                enc.emit_u32(*len);
            }
            FileRow::Imported { cnum, index, start_pos, len } => {
                enc.emit_u8(1);
                enc.emit_u32(*cnum);
                enc.emit_u32(*index);
                enc.emit_u32(*start_pos);
                enc.emit_u32(*len);
            }
            FileRow::Virtual { name, src, start_pos, len } => {
                enc.emit_u8(2);
                name.encode(enc);
                enc.emit_str(src);
                enc.emit_u32(*start_pos);
                enc.emit_u32(*len);
            }
        }
    }
}

fn decode_file_rows(dec: &mut FeDecoder<'_, '_>) -> Vec<FileRow> {
    let n = dec.read_usize();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let row = match dec.read_u8() {
            0 => FileRow::Local {
                path: PathBuf::from(dec.read_str().to_owned()),
                src_hash: SourceFileHash::decode(dec),
                start_pos: dec.read_u32(),
                len: dec.read_u32(),
            },
            1 => FileRow::Imported {
                cnum: dec.read_u32(),
                index: dec.read_u32(),
                start_pos: dec.read_u32(),
                len: dec.read_u32(),
            },
            2 => FileRow::Virtual {
                name: FileName::decode(dec),
                src: dec.read_str().to_owned(),
                start_pos: dec.read_u32(),
                len: dec.read_u32(),
            },
            _ => unreachable!("corrupt frontend cache"),
        };
        rows.push(row);
    }
    rows
}

// ------------------------------------------------------------------------
// Snapshot writing

/// Drops the lazy token captures the parser attached for possible proc-macro
/// use: nothing consumes them after expansion, they cannot be encoded, and on
/// a cache hit the restored AST will not have them either. Stripping them on
/// the recording session too keeps both sessions on identical state.
struct TokenStripper;

impl rustc_ast::mut_visit::MutVisitor for TokenStripper {
    fn visit_item(&mut self, item: &mut ast::Item) {
        item.tokens = None;
        rustc_ast::mut_visit::walk_item(self, item);
    }
    fn visit_assoc_item(&mut self, item: &mut ast::AssocItem, ctxt: rustc_ast::visit::AssocCtxt) {
        item.tokens = None;
        rustc_ast::mut_visit::walk_assoc_item(self, item, ctxt);
    }
    fn visit_foreign_item(&mut self, item: &mut ast::ForeignItem) {
        item.tokens = None;
        rustc_ast::mut_visit::walk_item(self, item);
    }
    fn visit_expr(&mut self, expr: &mut ast::Expr) {
        expr.tokens = None;
        rustc_ast::mut_visit::walk_expr(self, expr);
    }
    fn visit_local(&mut self, local: &mut ast::Local) {
        local.tokens = None;
        rustc_ast::mut_visit::walk_local(self, local);
    }
    fn visit_attribute(&mut self, attr: &mut ast::Attribute) {
        if let ast::AttrKind::Normal(normal) = &mut attr.kind {
            normal.tokens = None;
        }
        rustc_ast::mut_visit::walk_attribute(self, attr);
    }
}

pub(crate) fn write_snapshot(
    tcx: TyCtxt<'_>,
    resolver: &mut Resolver<'_, '_>,
    krate: &mut ast::Crate,
    hygiene_mark: hygiene_fecache::HygieneMark,
) {
    let sess = tcx.sess;
    let Some(path) = snapshot_path(tcx) else { return };
    let Some(def_rows) = resolver.fecache_take_recorded() else {
        fedbg!("not writing snapshot: unreplayable defs");
        return;
    };
    if sess.dcx().has_errors_or_delayed_bugs().is_some() {
        return;
    }
    let Some(file_rows) = classify_files(tcx) else {
        fedbg!("not writing snapshot: unclassifiable source files");
        return;
    };

    rustc_ast::mut_visit::MutVisitor::visit_crate(&mut TokenStripper, krate);

    let mut enc = FeEncoder::new();

    // Validation header.
    enc.emit_str(option_env!("CFG_VERSION").unwrap_or("unknown"));
    enc.emit_u64(sess.opts.dep_tracking_hash(true).as_u64());
    enc.emit_str(tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).as_str());

    // Environment reads made by expansion.
    {
        let env_deps = sess.env_depinfo.borrow();
        enc.emit_usize(env_deps.len());
        for (key, value) in env_deps.iter() {
            enc.emit_str(key.as_str());
            match value {
                None => enc.emit_u8(0),
                Some(v) => {
                    enc.emit_u8(1);
                    enc.emit_str(v.as_str());
                }
            }
        }
    }

    // Non-source file reads (include_bytes! etc.): path plus raw content hash,
    // since the source map may only hold a lossy transcoding of these.
    {
        let file_deps: Vec<String> =
            sess.file_depinfo.borrow().iter().map(|path| path.as_str().to_owned()).collect();
        enc.emit_usize(file_deps.len());
        for path in &file_deps {
            enc.emit_str(path);
            let hash = std::fs::read(path)
                .map(|bytes| SourceFileHash::new_in_memory(SourceFileHashAlgorithm::Sha256, bytes));
            match hash {
                Ok(hash) => {
                    enc.emit_u8(1);
                    hash.encode(&mut enc);
                }
                Err(_) => enc.emit_u8(0),
            }
        }
    }

    // Dependency crates in load order.
    {
        let crates = CStore::from_tcx(tcx).fecache_crates();
        enc.emit_usize(crates.len());
        for (cnum, name, svh, dep_kind, private) in crates {
            enc.emit_u32(cnum.as_u32());
            enc.encode_symbol(name);
            enc.emit_u128(svh.as_u128());
            enc.emit_u8(match dep_kind {
                CrateDepKind::MacrosOnly => 0,
                CrateDepKind::Conditional => 1,
                CrateDepKind::Unconditional => 2,
            });
            enc.emit_bool(private);
        }
    }

    encode_file_rows(&mut enc, &file_rows);

    hygiene_fecache::encode_delta(&mut enc, hygiene_mark);

    // Metavariable spans recorded during transcription: a session-global side
    // table consulted by `Span::to`/diagnostics, so replayed sessions must see
    // the same entries or lowered spans diverge from the recording session.
    {
        let pairs = rustc_span::with_metavar_spans(|mspans| mspans.fecache_pairs());
        enc.emit_usize(pairs.len());
        for (span, var_span) in pairs {
            enc.encode_span(span);
            enc.encode_span(var_span);
        }
    }

    // Def creation log.
    enc.emit_usize(def_rows.len());
    for row in &def_rows {
        enc.emit_u32(row.node_id.as_u32());
        enc.emit_u32(row.parent.local_def_index.as_u32());
        match row.name {
            None => enc.emit_u8(0),
            Some(name) => {
                enc.emit_u8(1);
                enc.encode_symbol(name);
            }
        }
        row.def_kind.encode(&mut enc);
        enc.encode_expn_id(row.expn_id);
        enc.encode_span(row.span);
        enc.emit_bool(row.is_owner);
    }

    enc.emit_u32(resolver.fecache_next_node_id().as_u32());

    resolver.fecache_stripped_cfg_items().encode(&mut enc);

    {
        let glob_map = resolver.fecache_glob_map();
        enc.emit_usize(glob_map.len());
        for (def_id, names) in glob_map {
            enc.emit_u32(def_id.local_def_index.as_u32());
            enc.emit_usize(names.len());
            for name in names {
                enc.encode_symbol(name);
            }
        }
    }

    {
        let orders = resolver.fecache_binding_orders();
        enc.emit_usize(orders.len());
        for (module_id, keys) in orders {
            enc.emit_u32(module_id.0);
            enc.emit_u32(module_id.1);
            enc.emit_u32(module_id.2);
            enc.emit_u32(module_id.3);
            enc.emit_usize(keys.len());
            for key in keys {
                enc.encode_symbol(key.name);
                enc.encode_syntax_context(key.ctxt);
                enc.emit_u8(key.ns);
                enc.emit_u32(key.disambiguator);
            }
        }
    }

    krate.encode(&mut enc);

    // Assemble: magic, version, symbol tables, body.
    let mut out = Vec::with_capacity(enc.data.len() + 1024);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(enc.syms.len() as u32).to_le_bytes());
    for sym in &enc.syms {
        let s = sym.as_str();
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out.extend_from_slice(&(enc.byte_syms.len() as u32).to_le_bytes());
    for bsym in &enc.byte_syms {
        let b = bsym.as_byte_str();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out.extend_from_slice(&enc.data);
    out.extend_from_slice(rustc_serialize::opaque::MAGIC_END_BYTES);

    let tmp = path.with_extension("bin.tmp");
    if std::fs::write(&tmp, &out).and_then(|()| std::fs::rename(&tmp, &path)).is_err() {
        fedbg!("failed to write snapshot to {}", path.display());
        let _ = std::fs::remove_file(&tmp);
    }
}

// ------------------------------------------------------------------------
// Restore

pub(crate) enum RestoreOutcome {
    /// The snapshot was replayed; expansion will find nothing to do. The
    /// binding orders must be applied to the resolver once the reduced graph
    /// has been rebuilt over the restored crate.
    Restored(ast::Crate, Vec<(rustc_resolve::fecache::FeModuleId, Vec<rustc_resolve::fecache::FeBindingKey>)>),
    /// The inputs are unchanged but the session could not be replayed (e.g.
    /// the crate numbering diverged). Re-expand, but do not rewrite the
    /// snapshot: it would encode to the same bytes.
    InputsValid,
    /// The inputs changed or no snapshot exists: re-expand and re-record.
    Miss,
}

pub(crate) fn try_restore(tcx: TyCtxt<'_>, resolver: &mut Resolver<'_, '_>) -> RestoreOutcome {
    let sess = tcx.sess;
    fedbg!("restore: attempting");
    let Some(path) = snapshot_path(tcx) else { return RestoreOutcome::Miss };
    let Ok(buf) = std::fs::read(&path) else {
        fedbg!("miss: no snapshot at {}", path.display());
        return RestoreOutcome::Miss;
    };

    // Parse the container: magic, version, symbol tables.
    if buf.len() < 12 || u32::from_le_bytes(buf[0..4].try_into().unwrap()) != MAGIC {
        fedbg!("miss: bad magic");
        return RestoreOutcome::Miss;
    }
    if u32::from_le_bytes(buf[4..8].try_into().unwrap()) != VERSION {
        fedbg!("miss: format version");
        return RestoreOutcome::Miss;
    }
    let mut pos = 8;
    let read_u32 = |pos: &mut usize| {
        let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };
    let n_syms = read_u32(&mut pos) as usize;
    let mut syms = Vec::with_capacity(n_syms);
    for _ in 0..n_syms {
        let len = read_u32(&mut pos) as usize;
        let Ok(sym_str) = std::str::from_utf8(&buf[pos..pos + len]) else {
            return RestoreOutcome::Miss;
        };
        syms.push(Symbol::intern(sym_str));
        pos += len;
    }
    let n_byte_syms = read_u32(&mut pos) as usize;
    let mut byte_syms = Vec::with_capacity(n_byte_syms);
    for _ in 0..n_byte_syms {
        let len = read_u32(&mut pos) as usize;
        byte_syms.push(ByteSymbol::intern(&buf[pos..pos + len]));
        pos += len;
    }

    let Ok(mem) = MemDecoder::new(&buf[pos..], 0) else {
        fedbg!("miss: truncated snapshot");
        return RestoreOutcome::Miss;
    };
    let mut dec =
        FeDecoder { mem, syms, byte_syms, psess: &sess.psess, cnum_map: FxHashMap::default() };

    // --- Validation (no session mutation until all cheap checks pass) ---

    if dec.read_str() != option_env!("CFG_VERSION").unwrap_or("unknown") {
        fedbg!("miss: compiler version");
        return RestoreOutcome::Miss;
    }
    let rec_opts_hash = dec.read_u64();
    let cur_opts_hash = sess.opts.dep_tracking_hash(true).as_u64();
    if rec_opts_hash != cur_opts_hash {
        fedbg!("miss: opts hash {rec_opts_hash:x} != {cur_opts_hash:x}");
        return RestoreOutcome::Miss;
    }
    if dec.read_str() != tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).as_str() {
        fedbg!("miss: crate name");
        return RestoreOutcome::Miss;
    }

    let n_env = dec.read_usize();
    let mut env_deps = Vec::with_capacity(n_env);
    for _ in 0..n_env {
        let key = dec.read_str().to_owned();
        let recorded = match dec.read_u8() {
            0 => None,
            _ => Some(dec.read_str().to_owned()),
        };
        if std::env::var(&key).ok() != recorded {
            fedbg!("miss: env var {key} changed");
            return RestoreOutcome::Miss;
        }
        env_deps.push((key, recorded));
    }

    let n_file_deps = dec.read_usize();
    let mut file_deps = Vec::with_capacity(n_file_deps);
    for _ in 0..n_file_deps {
        let path = dec.read_str().to_owned();
        let recorded = match dec.read_u8() {
            0 => None,
            _ => Some(SourceFileHash::decode(&mut dec)),
        };
        if let Some(recorded) = recorded {
            let Ok(bytes) = std::fs::read(&path) else {
                fedbg!("miss: file dep {path} unreadable");
                return RestoreOutcome::Miss;
            };
            if SourceFileHash::new_in_memory(SourceFileHashAlgorithm::Sha256, bytes) != recorded {
                fedbg!("miss: file dep {path} changed");
                return RestoreOutcome::Miss;
            }
        } else {
            fedbg!("miss: file dep {path} had no recorded hash");
            return RestoreOutcome::Miss;
        }
        file_deps.push(path);
    }

    let n_crates = dec.read_usize();
    let mut crates = Vec::with_capacity(n_crates);
    for _ in 0..n_crates {
        let cnum = dec.read_u32();
        let name = dec.decode_symbol();
        let svh = dec.read_u128();
        let dep_kind = match dec.read_u8() {
            0 => CrateDepKind::MacrosOnly,
            1 => CrateDepKind::Conditional,
            _ => CrateDepKind::Unconditional,
        };
        let private = dec.read_bool();
        crates.push((cnum, name, svh, dep_kind, private));
    }

    let file_rows = decode_file_rows(&mut dec);

    // Validate local files against the current session before mutating
    // anything: the parse-loaded prefix must match, and files expansion would
    // load must be unchanged on disk.
    {
        let source_map = sess.source_map();
        let files = source_map.files();
        if files.len() > file_rows.len() {
            fedbg!("miss: parse loaded {} files, snapshot has {}", files.len(), file_rows.len());
            return RestoreOutcome::Miss;
        }
        for (file, row) in files.iter().zip(file_rows.iter()) {
            match row {
                FileRow::Local { path, src_hash, start_pos, len } => {
                    if file.cnum != rustc_hir::def_id::LOCAL_CRATE
                        || file.start_pos.0 != *start_pos
                        || file.normalized_source_len.0 != *len
                        || file.src_hash != *src_hash
                        || !matches!(&file.name, FileName::Real(real)
                            if real.local_path().is_some_and(|p| p == path))
                    {
                        fedbg!("miss: prefix file mismatch at {}", path.display());
                        return RestoreOutcome::Miss;
                    }
                }
                _ => {
                    fedbg!("miss: non-local file in parse prefix");
                    return RestoreOutcome::Miss;
                }
            }
        }
        for row in &file_rows[files.len()..] {
            if let FileRow::Local { path, src_hash, .. } = row {
                let Ok(src) = std::fs::read_to_string(path) else {
                    fedbg!("miss: {} unreadable", path.display());
                    return RestoreOutcome::Miss;
                };
                if !src_hash.matches(&src) {
                    fedbg!("miss: {} changed", path.display());
                    return RestoreOutcome::Miss;
                }
            }
        }
    }

    // --- Replay (mutations begin; failures below fall back to expansion) ---

    // Load the dependency crates in recorded order and verify identity.
    // Several loaded crates sharing a name (e.g. the sysroot's and the user's
    // version of the same library) cannot be re-located unambiguously, so the
    // recorded numbering will not reproduce. Detect this before touching any
    // session state: the session then expands normally, reproducing the
    // recorded session bit for bit at only the cost of this validation.
    {
        let mut names: Vec<Symbol> = crates.iter().map(|&(_, name, ..)| name).collect();
        names.sort_unstable_by_key(|name| name.as_u32());
        if names.windows(2).any(|w| w[0] == w[1]) {
            fedbg!("inputs valid, but duplicate crate names prevent replay");
            return RestoreOutcome::InputsValid;
        }
    }

    {
        let mut cstore = CStore::from_tcx_mut(tcx);
        for &(rec, name, svh, dep_kind, _) in &crates {
            match cstore.fecache_preload_crate(tcx, name, svh, dep_kind) {
                Some(got) => {
                    fedbg!("preload {name} rec={rec} got={}", got.as_u32());
                }
                None => {
                    fedbg!("miss: crate {name} failed to preload");
                    return RestoreOutcome::Miss;
                }
            }
        }
        // Match each recorded crate to a loaded one by name and hash, and
        // remap the snapshot's crate numbering onto this session's.
        let loaded = cstore.fecache_crates();
        for (rec_cnum, rec_name, rec_svh, _, private) in &crates {
            let Some((actual, ..)) = loaded
                .iter()
                .find(|(_, name, svh, _, _)| name == rec_name && svh.as_u128() == *rec_svh)
            else {
                fedbg!("miss: crate {rec_name} identity changed");
                return RestoreOutcome::Miss;
            };
            if actual.as_u32() != *rec_cnum {
                // The load order could not reproduce the recorded numbering
                // (e.g. several crates share a name). Raw crate references in
                // the snapshot could be remapped, but this session's crate
                // list would still hash differently from the recording
                // session's, going red; re-expanding is cheaper than that.
                fedbg!("inputs valid, but {rec_name} loaded as {} not {}", actual.as_u32(), rec_cnum);
                return RestoreOutcome::InputsValid;
            }
            dec.cnum_map.insert(*rec_cnum, actual.as_u32());
            cstore.fecache_set_private_dep(*actual, *private);
        }
        if loaded.len() != crates.len() {
            fedbg!("miss: crate count {} != {}", loaded.len(), crates.len());
            return RestoreOutcome::Miss;
        }
    }

    // Materialize the source map exactly as the recorded session had it.
    {
        let source_map = sess.source_map();
        let already = source_map.files().len();
        for row in &file_rows[already..] {
            let file: Arc<SourceFile> = match row {
                FileRow::Local { path, .. } => match source_map.load_file(path) {
                    Ok(file) => file,
                    Err(_) => {
                        fedbg!("miss: cannot load {}", path.display());
                        return RestoreOutcome::Miss;
                    }
                },
                FileRow::Imported { cnum, index, .. } => CStore::from_tcx(tcx)
                    .fecache_import_source_file(tcx, dec.map_cnum(*cnum), *index),
                FileRow::Virtual { name, src, .. } => {
                    source_map.new_source_file(name.clone(), src.clone())
                }
            };
            let (start_pos, len) = match row {
                FileRow::Local { start_pos, len, .. }
                | FileRow::Imported { start_pos, len, .. }
                | FileRow::Virtual { start_pos, len, .. } => (*start_pos, *len),
            };
            if file.start_pos.0 != start_pos || file.normalized_source_len.0 != len {
                fedbg!("miss: file landed at {} not {}", file.start_pos.0, start_pos);
                return RestoreOutcome::Miss;
            }
        }
    }

    if !hygiene_fecache::decode_delta(&mut dec) {
        fedbg!("miss: hygiene prefix mismatch");
        return RestoreOutcome::Miss;
    }

    {
        let n = dec.read_usize();
        rustc_span::with_metavar_spans(|mspans| {
            for _ in 0..n {
                let span = dec.decode_span();
                let var_span = dec.decode_span();
                mspans.fecache_insert(span, var_span);
            }
        });
    }

    // Replay the def creation log.
    {
        let n = dec.read_usize();
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let node_id = ast::NodeId::from_u32(dec.read_u32());
            let parent = LocalDefId { local_def_index: DefIndex::from_u32(dec.read_u32()) };
            let name = match dec.read_u8() {
                0 => None,
                _ => Some(dec.decode_symbol()),
            };
            let def_kind = DefKind::decode(&mut dec);
            let expn_id = dec.decode_expn_id();
            let span = dec.decode_span();
            let is_owner = dec.read_bool();
            rows.push(FeDefRow { node_id, parent, name, def_kind, expn_id, span, is_owner });
        }
        resolver.fecache_replay_defs(rows);
    }

    resolver.fecache_set_next_node_id(ast::NodeId::from_u32(dec.read_u32()));

    let stripped = Vec::<rustc_hir::attrs::StrippedCfgItem<ast::NodeId>>::decode(&mut dec);
    resolver.fecache_set_stripped_cfg_items(stripped);

    {
        let n = dec.read_usize();
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let def_id = LocalDefId { local_def_index: DefIndex::from_u32(dec.read_u32()) };
            let n_names = dec.read_usize();
            let names = (0..n_names).map(|_| dec.decode_symbol()).collect();
            entries.push((def_id, names));
        }
        resolver.fecache_extend_glob_map(entries);
    }

    let mut binding_orders = Vec::new();
    {
        let n = dec.read_usize();
        binding_orders.reserve(n);
        for _ in 0..n {
            let module_id = (dec.read_u32(), dec.read_u32(), dec.read_u32(), dec.read_u32());
            let n_keys = dec.read_usize();
            let keys = (0..n_keys)
                .map(|_| rustc_resolve::fecache::FeBindingKey {
                    name: dec.decode_symbol(),
                    ctxt: dec.decode_syntax_context(),
                    ns: dec.read_u8(),
                    disambiguator: dec.read_u32(),
                })
                .collect();
            binding_orders.push((module_id, keys));
        }
    }

    let krate = ast::Crate::decode(&mut dec);

    // Re-register the dependency information that expansion would have.
    {
        let mut env_map = sess.env_depinfo.borrow_mut();
        for (key, value) in env_deps {
            env_map.insert((Symbol::intern(&key), value.as_deref().map(Symbol::intern)));
        }
        drop(env_map);
        let mut file_map = sess.file_depinfo.borrow_mut();
        for path in file_deps {
            file_map.insert(Symbol::intern(&path));
        }
    }

    fedbg!("hit: restored expanded crate");
    RestoreOutcome::Restored(krate, binding_orders)
}
