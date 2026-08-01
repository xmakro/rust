
use std::mem;
use std::sync::OnceLock;

use rustc_ast::token::{
    self, Delimiter, IdentIsRaw, InvisibleOrigin, Lit, LitKind, MetaVarKind, Token, TokenKind,
};
use rustc_ast::tokenstream::{
    DelimSpacing, DelimSpan, FlatEntry, FlatSink, FlatTokenCursor, FlatTt, Spacing, TokenStream,
    TokenTree,
};
use rustc_ast::{ExprKind, StmtKind, TyKind, UnOp};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Diag, DiagCtxtHandle, PResult, listify, pluralize};
use rustc_parse::lexer::nfc_normalize;
use rustc_parse::parser::ParseNtResult;
use rustc_session::parse::ParseSess;
use rustc_span::hygiene::{LocalExpnId, Transparency};
use rustc_span::{
    BytePos, Ident, MacroRulesNormalizedIdent, Span, Symbol, SyntaxContext, kw, sym,
    with_metavar_spans,
};
use smallvec::{SmallVec, smallvec};

use crate::diagnostics::{
    ConcatInvalidIdent, CountRepetitionMisplaced, InvalidIdentReason, MacroVarStillRepeating,
    MetaVarsDifSeqMatchers, MustRepeatOnce, MveUnrecognizedVar, NoRepeatableVar,
    NoSyntaxVarsExprRepeat, VarNoTypo, VarTypoSuggestionRepeatable, VarTypoSuggestionUnrepeatable,
    VarTypoSuggestionUnrepeatableLabel,
};
use crate::mbe::macro_parser::NamedMatch;
use crate::mbe::macro_parser::NamedMatch::*;
use crate::mbe::metavar_expr::{MetaVarExprConcatElem, RAW_IDENT_ERR};
use crate::mbe::{self, KleeneOp, MetaVarExpr};

/// Context needed to perform transcription of metavariable expressions.
struct TranscrCtx<'psess, 'itp> {
    psess: &'psess ParseSess,

    /// Map from metavars to matched tokens
    interp: &'itp FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,

    /// Allow marking spans.
    marker: Marker,

    /// The stack of things yet to be completely expanded.
    ///
    /// We descend into the compiled template of the RHS, expanding things as
    /// we go. This stack contains the things we have yet to expand/are still
    /// expanding. We start the stack off with the whole template.
    stack: SmallVec<[ExecFrame<'itp>; 1]>,

    /// A stack of where we are in the repeat expansion.
    ///
    /// As we descend in the RHS, we will need to be able to match nested sequences of matchers.
    /// `repeats` keeps track of where we are in matching at each level, with the last element
    /// being the most deeply nested sequence. This is used as a stack.
    repeats: Vec<(usize, usize)>,

    /// Cache of metavar resolutions through every repetition level except the
    /// innermost, whose index is applied per call. That prefix is invariant
    /// for the lifetime of the innermost repetition frame, while the metavar
    /// occurrences inside it are resolved once per iteration, so caching it
    /// saves a map lookup and a level walk per occurrence. Cleared whenever
    /// `repeats` changes depth.
    lookup_cache: SmallVec<[(MacroRulesNormalizedIdent, Option<&'itp NamedMatch>); 4]>,

    /// The transcription result, built directly in the parser's flat form.
    /// Entering a nested `Delimited` emits an open-delimiter entry and
    /// leaving it emits the close entry, so no result stack is needed — the
    /// buffer is append-only.
    sink: FlatSink,
}

impl<'psess> TranscrCtx<'psess, '_> {
    /// Span marked with the correct expansion and transparency.
    fn visited_dspan(&mut self, dspan: DelimSpan) -> Span {
        let mut span = dspan.entire();
        self.marker.mark_span(&mut span);
        span
    }
}

/// A Marker adds the given mark to the syntax context.
struct Marker {
    expand_id: LocalExpnId,
    transparency: Transparency,
    /// One-entry cache in front of `cache`: virtually all tokens in a macro
    /// body share one syntax context, so this hits on a plain compare without
    /// hashing.
    last: Option<(SyntaxContext, SyntaxContext)>,
    cache: FxHashMap<SyntaxContext, SyntaxContext>,
}

impl Marker {
    /// Mark a span with the stored expansion ID and transparency.
    fn mark_span(&mut self, span: &mut Span) {
        // `apply_mark` is a relatively expensive operation, both due to taking hygiene lock, and
        // by itself. All tokens in a macro body typically have the same syntactic context, unless
        // it's some advanced case with macro-generated macros. So if we cache the marked version
        // of that context once, we'll typically have a 100% cache hit rate after that.
        *span = span.map_ctxt(|ctxt| {
            if let Some((from, to)) = self.last
                && from == ctxt
            {
                return to;
            }
            let to = *self
                .cache
                .entry(ctxt)
                .or_insert_with(|| ctxt.apply_mark(self.expand_id.to_expn_id(), self.transparency));
            self.last = Some((ctxt, to));
            to
        });
    }
}

/// A macro rule RHS: the parsed tree plus its transcription template,
/// compiled on first expansion so definitions that are never invoked don't
/// pay for compilation.
pub(crate) struct MacroRhs {
    pub(crate) tt: mbe::TokenTree,
    template: OnceLock<Template>,
}

impl MacroRhs {
    pub(crate) fn new(tt: mbe::TokenTree) -> MacroRhs {
        MacroRhs { tt, template: OnceLock::new() }
    }
}

/// A compiled transcription template. The RHS tree is lowered once per rule
/// into a form transcription can execute directly: runs of literal tokens are
/// pre-encoded as flat entries ready to be copied into the output buffer, and
/// metavar occurrences carry their normalization precomputed.
struct Template {
    segs: Vec<Seg>,
}

enum Seg {
    /// A run of fully literal tokens (including whole literal delimited
    /// groups), pre-encoded as flat entries with def-site spans and
    /// run-relative depths and match indices. Spliced by copy; the copied
    /// spans are then marked in place.
    Run { entries: Vec<FlatEntry>, matches: Vec<u32> },
    /// A single literal token whose kind embeds a span needing its own
    /// marking (`NtIdent`/`NtLifetime`), kept out of runs.
    Token(Token),
    /// A delimited group containing non-literal segments.
    Delimited { span: DelimSpan, spacing: DelimSpacing, delim: Delimiter, inner: Vec<Seg> },
    /// A metavar occurrence.
    MetaVar { span: Span, orig: Ident, norm: MacroRulesNormalizedIdent },
    /// A `$(...)` repetition.
    Sequence(Box<SeqSeg>),
    /// A `${...}` metavar expression.
    MetaVarExpr { dspan: DelimSpan, expr: MetaVarExpr },
}

struct SeqSeg {
    inner: Vec<Seg>,
    sep: Option<Token>,
    kleene_op: KleeneOp,
    dspan: DelimSpan,
    /// The normalized metavar occurrences in the whole subtree, in source
    /// order, for the lockstep size check.
    lockstep_vars: Vec<MacroRulesNormalizedIdent>,
    /// The original metavar idents of the subtree (excluding metavar
    /// expressions), for the no-repeatable-vars diagnostics.
    error_meta_vars: Vec<Ident>,
}

/// A frame of the transcription stack: where we are within one nesting level
/// of the compiled template.
enum ExecFrame<'itp> {
    /// The top-level template; its delimiters are not part of the result.
    Root { segs: &'itp [Seg], idx: usize },
    /// Inside a `Seg::Delimited`. The close-delimiter parts are precomputed
    /// (span already marked) for emission when the frame is popped.
    Delimited { inner: &'itp [Seg], idx: usize, delim: Delimiter, close_span: Span, close_spacing: Spacing },
    /// One iteration of a `Seg::Sequence`.
    Sequence { seq: &'itp SeqSeg, idx: usize },
}

impl<'itp> ExecFrame<'itp> {
    fn next_seg(&mut self) -> Option<&'itp Seg> {
        let (segs, idx) = match self {
            ExecFrame::Root { segs, idx } => (*segs, idx),
            ExecFrame::Delimited { inner, idx, .. } => (*inner, idx),
            ExecFrame::Sequence { seq, idx } => (&seq.inner[..], idx),
        };
        let res = segs.get(*idx);
        *idx += 1;
        res
    }
}

fn compile_template(src: &mbe::Delimited) -> Template {
    Template { segs: compile_segs(&src.tts) }
}

fn compile_segs(tts: &[mbe::TokenTree]) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut run = FlatSink::new();
    for tt in tts {
        compile_tt(tt, &mut segs, &mut run);
    }
    flush_run(&mut segs, &mut run);
    segs
}

fn flush_run(segs: &mut Vec<Seg>, run: &mut FlatSink) {
    if !run.entries.is_empty() {
        let entries = mem::take(&mut run.entries);
        let matches = mem::take(&mut run.matches);
        segs.push(Seg::Run { entries, matches });
    }
}

fn compile_tt(tt: &mbe::TokenTree, segs: &mut Vec<Seg>, run: &mut FlatSink) {
    match tt {
        mbe::TokenTree::Token(token) => {
            if matches!(token.kind, token::NtIdent(..) | token::NtLifetime(..)) {
                flush_run(segs, run);
                segs.push(Seg::Token(*token));
            } else {
                run.push_token(*token, Spacing::Alone);
            }
        }
        mbe::TokenTree::Delimited(span, spacing, delimited) => {
            if all_literal(&delimited.tts) {
                emit_literal_delimited(run, *span, *spacing, delimited);
            } else {
                flush_run(segs, run);
                segs.push(Seg::Delimited {
                    span: *span,
                    spacing: *spacing,
                    delim: delimited.delim,
                    inner: compile_segs(&delimited.tts),
                });
            }
        }
        &mbe::TokenTree::MetaVar(span, orig) => {
            flush_run(segs, run);
            segs.push(Seg::MetaVar { span, orig, norm: MacroRulesNormalizedIdent::new(orig) });
        }
        seq_tt @ mbe::TokenTree::Sequence(dspan, seq_rep) => {
            flush_run(segs, run);
            let mut lockstep_vars = Vec::new();
            collect_lockstep_vars(&seq_rep.tts, &mut lockstep_vars);
            let mut error_meta_vars = Vec::new();
            seq_tt.meta_vars(&mut error_meta_vars);
            segs.push(Seg::Sequence(Box::new(SeqSeg {
                inner: compile_segs(&seq_rep.tts),
                sep: seq_rep.separator,
                kleene_op: seq_rep.kleene.op,
                dspan: *dspan,
                lockstep_vars,
                error_meta_vars,
            })));
        }
        mbe::TokenTree::MetaVarExpr(dspan, expr) => {
            flush_run(segs, run);
            segs.push(Seg::MetaVarExpr { dspan: *dspan, expr: expr.clone() });
        }
        // There should be no meta-var declarations in a macro RHS.
        mbe::TokenTree::MetaVarDecl { .. } => panic!("unexpected `TokenTree::MetaVarDecl`"),
    }
}

/// Whether every tree is a plain token or a delimited group of plain tokens,
/// i.e. transcribes to the same entries on every expansion (modulo marking).
fn all_literal(tts: &[mbe::TokenTree]) -> bool {
    tts.iter().all(|tt| match tt {
        mbe::TokenTree::Token(token) => {
            !matches!(token.kind, token::NtIdent(..) | token::NtLifetime(..))
        }
        mbe::TokenTree::Delimited(.., delimited) => all_literal(&delimited.tts),
        _ => false,
    })
}

fn emit_literal_delimited(
    run: &mut FlatSink,
    span: DelimSpan,
    spacing: DelimSpacing,
    delimited: &mbe::Delimited,
) {
    run.open_delim(Token::new(delimited.delim.as_open_token_kind(), span.open), spacing.open);
    for tt in &delimited.tts {
        match tt {
            mbe::TokenTree::Token(token) => run.push_token(*token, Spacing::Alone),
            mbe::TokenTree::Delimited(span, spacing, delimited) => {
                emit_literal_delimited(run, *span, *spacing, delimited)
            }
            _ => unreachable!("non-literal tree in literal delimited group"),
        }
    }
    // Hack to force-insert a space after `]` in certain case.
    // See discussion of the `hex-literal` crate in #114571.
    let close_spacing =
        if delimited.delim == Delimiter::Bracket { Spacing::Alone } else { spacing.close };
    run.close_delim(Token::new(delimited.delim.as_close_token_kind(), span.close), close_spacing);
}

/// Collects the normalized idents of every metavar occurrence in the subtree,
/// in source order, mirroring the leaves the tree-walking lockstep size check
/// used to visit.
fn collect_lockstep_vars(tts: &[mbe::TokenTree], out: &mut Vec<MacroRulesNormalizedIdent>) {
    for tt in tts {
        match tt {
            mbe::TokenTree::Token(_) => {}
            mbe::TokenTree::MetaVar(_, id) | mbe::TokenTree::MetaVarDecl { name: id, .. } => {
                out.push(MacroRulesNormalizedIdent::new(*id))
            }
            mbe::TokenTree::Delimited(.., d) => collect_lockstep_vars(&d.tts, out),
            mbe::TokenTree::Sequence(_, s) => collect_lockstep_vars(&s.tts, out),
            mbe::TokenTree::MetaVarExpr(_, expr) => {
                expr.for_each_metavar((), |(), ident| {
                    out.push(MacroRulesNormalizedIdent::new(*ident))
                });
            }
        }
    }
}

/// Appends a pre-encoded literal run to the sink, rebasing the run-relative
/// depths and match indices, and marks the copied spans in place.
fn splice_run(sink: &mut FlatSink, entries: &[FlatEntry], matches: &[u32], marker: &mut Marker) {
    let dst_start = sink.entries.len();
    let depth = sink.depth();
    sink.entries.extend(entries.iter().map(|e| FlatEntry {
        token: e.token,
        spacing: e.spacing,
        depth: e.depth + depth,
    }));
    for e in &mut sink.entries[dst_start..] {
        marker.mark_span(&mut e.token.span);
    }
    let idx_delta = dst_start as u32;
    sink.matches.extend(matches.iter().map(|&m| if m == 0 { 0 } else { m + idx_delta }));
}

/// This can do Macro-By-Example transcription.
/// - `interp` is a map of meta-variables to the tokens (non-terminals) they matched in the
///   invocation. We are assuming we already know there is a match.
/// - `src` is the RHS of the MBE, that is, the "example" we are filling in.
///
/// For example,
///
/// ```rust
/// macro_rules! foo {
///     ($id:ident) => { println!("{}", stringify!($id)); }
/// }
///
/// foo!(bar);
/// ```
///
/// `interp` would contain `$id => bar` and `src` would contain `println!("{}", stringify!($id));`.
///
/// `transcribe` would return a `TokenStream` containing `println!("{}", stringify!(bar));`.
///
/// Along the way, we do some additional error checking.
pub(super) fn transcribe<'a>(
    psess: &'a ParseSess,
    interp: &FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    rhs: &MacroRhs,
    transparency: Transparency,
    expand_id: LocalExpnId,
) -> PResult<'a, FlatTokenCursor> {
    let mbe::TokenTree::Delimited(_, _, src) = &rhs.tt else {
        panic!("malformed macro rhs");
    };

    // Nothing for us to transcribe...
    if src.tts.is_empty() {
        return Ok(FlatTokenCursor::from_parts(Vec::new(), Vec::new()));
    }

    let template = rhs.template.get_or_init(|| compile_template(src));

    let mut tscx = TranscrCtx {
        psess,
        interp,
        marker: Marker { expand_id, transparency, last: None, cache: Default::default() },
        repeats: Vec::new(),
        lookup_cache: SmallVec::new(),
        stack: smallvec![ExecFrame::Root { segs: &template.segs, idx: 0 }],
        // The output typically contains at least one entry per template
        // token tree, so the template length is a cheap capacity estimate
        // that avoids the initial growth ladder of the result buffer.
        sink: FlatSink::with_capacity(src.tts.len()),
    };

    loop {
        // Look at the last frame on the stack.
        // If it still has a segment we have not looked at yet, use that segment.
        let Some(seg) = tscx.stack.last_mut().unwrap().next_seg() else {
            // This else-case never produces a value for `seg` (it `continue`s or `return`s).

            // Otherwise, if we have just reached the end of a sequence and we can keep repeating,
            // go back to the beginning of the sequence.
            let frame = tscx.stack.last_mut().unwrap();
            if let ExecFrame::Sequence { seq, idx } = frame {
                let (repeat_idx, repeat_len) = tscx.repeats.last_mut().unwrap();
                *repeat_idx += 1;
                if repeat_idx < repeat_len {
                    *idx = 0;
                    if let Some(sep) = &seq.sep {
                        tscx.sink.push_token(*sep, Spacing::Alone);
                    }
                    continue;
                }
            }

            // We are done with the top of the stack. Pop it. Depending on what it was, we do
            // different things. Note that the outermost item must be the root template.
            match tscx.stack.pop().unwrap() {
                // Done with a sequence. Pop from repeats.
                ExecFrame::Sequence { .. } => {
                    tscx.repeats.pop();
                    tscx.lookup_cache.clear();
                }

                // No results left to compute! We are back at the top-level.
                // (The RHS delimiters are not part of the result.)
                ExecFrame::Root { .. } => {
                    return Ok(tscx.sink.finish());
                }

                // We are done processing a Delimited. Emit the close delimiter
                // entry; its span was marked when the frame was entered.
                ExecFrame::Delimited { delim, close_span, close_spacing, .. } => {
                    tscx.sink
                        .close_delim(Token::new(delim.as_close_token_kind(), close_span), close_spacing);
                }
            }
            continue;
        };

        match seg {
            // Copy a pre-encoded run of literal tokens into the output and
            // mark the copied spans in place.
            Seg::Run { entries, matches } => {
                splice_run(&mut tscx.sink, entries, matches, &mut tscx.marker);
            }

            // A literal token whose kind embeds an extra span to mark.
            &Seg::Token(mut token) => {
                tscx.marker.mark_span(&mut token.span);
                if let token::NtIdent(ident, _) | token::NtLifetime(ident, _) = &mut token.kind {
                    tscx.marker.mark_span(&mut ident.span);
                }
                tscx.sink.push_token(token, Spacing::Alone);
            }

            // If we are entering a new delimiter, emit its open entry and push
            // its contents to the `stack` to be processed.
            Seg::Delimited { span, spacing, delim, inner } => {
                let mut open_span = span.open;
                let mut close_span = span.close;
                tscx.marker.mark_span(&mut open_span);
                tscx.marker.mark_span(&mut close_span);
                tscx.sink
                    .open_delim(Token::new(delim.as_open_token_kind(), open_span), spacing.open);
                // Hack to force-insert a space after `]` in certain case.
                // See discussion of the `hex-literal` crate in #114571.
                let close_spacing =
                    if *delim == Delimiter::Bracket { Spacing::Alone } else { spacing.close };
                tscx.stack.push(ExecFrame::Delimited {
                    inner: &inner[..],
                    idx: 0,
                    delim: *delim,
                    close_span,
                    close_spacing,
                });
            }

            // Replace the meta-var with the matched token tree from the invocation.
            Seg::MetaVar { span, orig, norm } => {
                transcribe_metavar(&mut tscx, *span, *orig, *norm)?;
            }

            // Replace the sequence with its expansion.
            Seg::Sequence(seq) => {
                transcribe_sequence(&mut tscx, seq)?;
            }

            // Replace meta-variable expressions with the result of their expansion.
            Seg::MetaVarExpr { dspan, expr } => {
                transcribe_metavar_expr(&mut tscx, *dspan, expr)?;
            }
        }
    }
}

/// Turn `$(...)*` sequences into tokens.
fn transcribe_sequence<'tx, 'itp>(
    tscx: &mut TranscrCtx<'tx, 'itp>,
    seq: &'itp SeqSeg,
) -> PResult<'tx, ()> {
    let dcx = tscx.psess.dcx();

    // We are descending into a sequence. We first make sure that the matchers in the RHS
    // and the matches in `interp` have the same shape. Otherwise, either the caller or the
    // macro writer has made a mistake.
    match lockstep_iter_size(seq, tscx.interp, &tscx.repeats) {
        LockstepIterSize::Unconstrained => {
            let mut repeatables = Vec::new();
            let mut non_repeatables = Vec::new();

            #[allow(rustc::potential_query_instability)]
            for (name, matcher) in tscx.interp.iter() {
                if matcher.is_repeatable() {
                    repeatables.push(name);
                } else {
                    non_repeatables.push(name);
                }
            }

            let repeatable_names: Vec<Symbol> =
                repeatables.iter().map(|&name| name.symbol()).collect();
            let non_repeatable_names: Vec<Symbol> =
                non_repeatables.iter().map(|&name| name.symbol()).collect();
            let meta_vars = &seq.error_meta_vars;
            let mut typo_repeatable = None;
            let mut typo_unrepeatable = None;
            let mut typo_unrepeatable_label = None;
            let mut var_no_typo = None;
            let mut no_repeatable_var = None;

            for &ident in meta_vars {
                if let Some(name) = rustc_span::edit_distance::find_best_match_for_name(
                    &repeatable_names[..],
                    ident.name,
                    None,
                ) {
                    typo_repeatable = Some(VarTypoSuggestionRepeatable { span: ident.span, name });
                } else if let Some(name) = rustc_span::edit_distance::find_best_match_for_name(
                    &non_repeatable_names[..],
                    ident.name,
                    None,
                ) {
                    typo_unrepeatable = Some(VarTypoSuggestionUnrepeatable { span: ident.span });
                    if let Some(&orig_ident) = non_repeatables.iter().find(|n| n.symbol() == name) {
                        typo_unrepeatable_label = Some(VarTypoSuggestionUnrepeatableLabel {
                            span: orig_ident.ident().span,
                        });
                    }
                } else {
                    if !repeatable_names.is_empty()
                        && let Some(msg) = listify(&repeatable_names, |s| format!("`${s}`"))
                    {
                        var_no_typo = Some(VarNoTypo { span: ident.span, msg });
                    } else {
                        no_repeatable_var = Some(NoRepeatableVar { span: ident.span });
                    }
                }
            }
            return Err(dcx.create_err(NoSyntaxVarsExprRepeat {
                span: seq.dspan.entire(),
                typo_unrepeatable,
                typo_repeatable,
                typo_unrepeatable_label,
                var_no_typo,
                no_repeatable_var,
            }));
        }

        LockstepIterSize::Contradiction(msg) => {
            // FIXME: this really ought to be caught at macro definition time... It
            // happens when two meta-variables are used in the same repetition in a
            // sequence, but they come from different sequence matchers and repeat
            // different amounts.
            return Err(dcx.create_err(MetaVarsDifSeqMatchers { span: seq.dspan.entire(), msg }));
        }

        LockstepIterSize::Constraint(len, _) => {
            // Is the repetition empty?
            if len == 0 {
                if seq.kleene_op == KleeneOp::OneOrMore {
                    // FIXME: this really ought to be caught at macro definition
                    // time... It happens when the Kleene operator in the matcher and
                    // the body for the same meta-variable do not match.
                    return Err(dcx.create_err(MustRepeatOnce { span: seq.dspan.entire() }));
                }
            } else {
                // 0 is the initial counter (we have done 0 repetitions so far). `len`
                // is the total number of repetitions we should generate.
                tscx.repeats.push((0, len));
                tscx.lookup_cache.clear();

                // The first time we encounter the sequence we push it to the stack. It
                // then gets reused (see the beginning of the loop) until we are done
                // repeating.
                tscx.stack.push(ExecFrame::Sequence { seq, idx: 0 });
            }
        }
    }

    Ok(())
}

/// Find the matched nonterminal from the macro invocation, and use it to replace
/// the meta-var.
///
/// We use `Spacing::Alone` everywhere here, because that's the conservative choice
/// and spacing of declarative macros is tricky. E.g. in this macro:
/// ```
/// macro_rules! idents {
///     ($($a:ident,)*) => { stringify!($($a)*) }
/// }
/// ```
/// `$a` has no whitespace after it and will be marked `JointHidden`. If you then
/// call `idents!(x,y,z,)`, each of `x`, `y`, and `z` will be marked as `Joint`. So
/// if you choose to use `$x`'s spacing or the identifier's spacing, you'll end up
/// producing "xyz", which is bad because it effectively merges tokens.
/// `Spacing::Alone` is the safer option. Fortunately, `space_between` will avoid
/// some of the unnecessary whitespace.
fn transcribe_metavar<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    mut sp: Span,
    mut original_ident: Ident,
    ident: MacroRulesNormalizedIdent,
) -> PResult<'tx, ()> {
    let dcx = tscx.psess.dcx();

    let Some(cur_matched) = lookup_cur_matched_cached(tscx, ident) else {
        // If we aren't able to match the meta-var, we push it back into the result but
        // with modified syntax context. (I believe this supports nested macros).
        tscx.marker.mark_span(&mut sp);
        tscx.marker.mark_span(&mut original_ident.span);
        tscx.sink.push_token(Token::new(token::Dollar, sp), Spacing::JointHidden);
        tscx.sink.push_token(Token::from_ast_ident(original_ident), Spacing::Alone);
        return Ok(());
    };

    let MatchedSingle(pnr) = cur_matched else {
        // We were unable to descend far enough. This is an error.
        return Err(dcx.create_err(MacroVarStillRepeating { span: sp, ident }));
    };

    transcribe_pnr(tscx, sp, pnr)
}

fn transcribe_pnr<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    mut sp: Span,
    pnr: &ParseNtResult,
) -> PResult<'tx, ()> {
    match pnr {
        ParseNtResult::Tt(ftt) => {
            // `tt`s are emitted into the output stream directly as "raw tokens",
            // without wrapping them into groups. Other variables are emitted into
            // the output stream as groups with `Delimiter::Invisible` to maintain
            // parsing priorities.
            transcribe_flat_tt(tscx, sp, ftt);
        }
        ParseNtResult::Ident(ident, is_raw) => {
            tscx.marker.mark_span(&mut sp);
            with_metavar_spans(|mspans| mspans.insert(ident.span, sp));
            let kind = token::NtIdent(*ident, *is_raw);
            tscx.sink.push_token(Token::new(kind, sp), Spacing::Alone);
        }
        ParseNtResult::Lifetime(ident, is_raw) => {
            tscx.marker.mark_span(&mut sp);
            with_metavar_spans(|mspans| mspans.insert(ident.span, sp));
            let kind = token::NtLifetime(*ident, *is_raw);
            tscx.sink.push_token(Token::new(kind, sp), Spacing::Alone);
        }
        ParseNtResult::Item(item) => {
            emit_delimited_fragment(tscx, sp, item.span, MetaVarKind::Item, TokenStream::from_ast(item))
        }
        ParseNtResult::Block(block) => {
            emit_delimited_fragment(tscx, sp, block.span, MetaVarKind::Block, TokenStream::from_ast(block))
        }
        ParseNtResult::Stmt(stmt) => {
            let stream = if let StmtKind::Empty = stmt.kind {
                // FIXME: Properly collect tokens for empty statements.
                TokenStream::token_alone(token::Semi, stmt.span)
            } else {
                TokenStream::from_ast(stmt)
            };
            emit_delimited_fragment(tscx, sp, stmt.span, MetaVarKind::Stmt, stream)
        }
        ParseNtResult::Pat(pat, pat_kind) => {
            emit_delimited_fragment(tscx, sp, pat.span, MetaVarKind::Pat(*pat_kind), TokenStream::from_ast(pat))
        }
        ParseNtResult::Expr(expr, kind) => {
            let (can_begin_literal_maybe_minus, can_begin_string_literal) = match &expr.kind {
                ExprKind::Lit(_) => (true, true),
                ExprKind::Unary(UnOp::Neg, e) if matches!(&e.kind, ExprKind::Lit(_)) => {
                    (true, false)
                }
                _ => (false, false),
            };
            emit_delimited_fragment(
                tscx,
                sp,
                expr.span,
                MetaVarKind::Expr {
                    kind: *kind,
                    can_begin_literal_maybe_minus,
                    can_begin_string_literal,
                },
                TokenStream::from_ast(expr),
            )
        }
        ParseNtResult::Literal(lit) => {
            emit_delimited_fragment(tscx, sp, lit.span, MetaVarKind::Literal, TokenStream::from_ast(lit))
        }
        ParseNtResult::Ty(ty) => {
            let is_path = matches!(&ty.kind, TyKind::Path(None, _path));
            emit_delimited_fragment(tscx, sp, ty.span, MetaVarKind::Ty { is_path }, TokenStream::from_ast(ty))
        }
        ParseNtResult::Meta(attr_item) => {
            let has_meta_form = attr_item.meta_kind().is_some();
            emit_delimited_fragment(
                tscx,
                sp,
                attr_item.span(),
                MetaVarKind::Meta { has_meta_form },
                TokenStream::from_ast(attr_item),
            )
        }
        ParseNtResult::Path(path) => {
            emit_delimited_fragment(tscx, sp, path.span, MetaVarKind::Path, TokenStream::from_ast(path))
        }
        ParseNtResult::Vis(vis) => {
            emit_delimited_fragment(tscx, sp, vis.span, MetaVarKind::Vis, TokenStream::from_ast(vis))
        }
        ParseNtResult::Guard(guard) => {
            // FIXME(macro_guard_matcher):
            // Perhaps it would be better to treat the leading `if` as part of `ast::Guard` during parsing?
            // Currently they are separate, but in macros we match and emit the leading `if` for `:guard` matchers, which creates some inconsistency.

            let leading_if_span =
                guard.span_with_leading_if.with_hi(guard.span_with_leading_if.lo() + BytePos(2));
            let mut ts =
                TokenStream::token_alone(token::Ident(kw::If, IdentIsRaw::No), leading_if_span);
            ts.push_stream(TokenStream::from_ast(&guard.cond));

            emit_delimited_fragment(tscx, sp, guard.span_with_leading_if, MetaVarKind::Guard, ts)
        }
    };

    Ok(())
}

/// Turn `${expr(...)}` metavariable expressionss into tokens.
fn transcribe_metavar_expr<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    dspan: DelimSpan,
    expr: &MetaVarExpr,
) -> PResult<'tx, ()> {
    let dcx = tscx.psess.dcx();
    let tt = match *expr {
        MetaVarExpr::Concat(ref elements) => metavar_expr_concat(tscx, dspan, elements)?,
        MetaVarExpr::Count(original_ident, depth) => {
            let matched = matched_from_ident(dcx, original_ident, tscx.interp)?;
            let count = count_repetitions(dcx, depth, matched, &tscx.repeats, &dspan)?;
            TokenTree::token_alone(
                TokenKind::lit(token::Integer, sym::integer(count), None),
                tscx.visited_dspan(dspan),
            )
        }
        MetaVarExpr::Ignore(original_ident) => {
            // Used to ensure that `original_ident` is present in the LHS
            let _ = matched_from_ident(dcx, original_ident, tscx.interp)?;
            return Ok(());
        }
        MetaVarExpr::Index(depth) => match tscx.repeats.iter().nth_back(depth) {
            Some((index, _)) => TokenTree::token_alone(
                TokenKind::lit(token::Integer, sym::integer(*index), None),
                tscx.visited_dspan(dspan),
            ),
            None => {
                return Err(out_of_bounds_err(dcx, tscx.repeats.len(), dspan.entire(), "index"));
            }
        },
        MetaVarExpr::Len(depth) => match tscx.repeats.iter().nth_back(depth) {
            Some((_, length)) => TokenTree::token_alone(
                TokenKind::lit(token::Integer, sym::integer(*length), None),
                tscx.visited_dspan(dspan),
            ),
            None => {
                return Err(out_of_bounds_err(dcx, tscx.repeats.len(), dspan.entire(), "len"));
            }
        },
    };
    let TokenTree::Token(token, spacing) = tt else {
        unreachable!("metavariable expressions produce single tokens")
    };
    tscx.sink.push_token(token, spacing);
    Ok(())
}

/// Handle the `${concat(...)}` metavariable expression.
fn metavar_expr_concat<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    dspan: DelimSpan,
    elements: &[MetaVarExprConcatElem],
) -> PResult<'tx, TokenTree> {
    let dcx = tscx.psess.dcx();
    let mut concatenated = String::new();
    for element in elements {
        let symbol = match element {
            MetaVarExprConcatElem::Ident(elem) => elem.name,
            MetaVarExprConcatElem::Literal(elem) => *elem,
            MetaVarExprConcatElem::Var(ident) => {
                let key = MacroRulesNormalizedIdent::new(*ident);
                match lookup_cur_matched(key, tscx.interp, &tscx.repeats) {
                    Some(NamedMatch::MatchedSingle(pnr)) => {
                        extract_symbol_from_pnr(dcx, pnr, ident.span)?
                    }
                    Some(NamedMatch::MatchedSeq(..)) => {
                        return Err(dcx.struct_span_err(
                            ident.span,
                            "`${concat(...)}` variable is still repeating at this depth",
                        ));
                    }
                    None => {
                        return Err(dcx.create_err(MveUnrecognizedVar { span: ident.span, key }));
                    }
                }
            }
        };
        concatenated.push_str(symbol.as_str());
    }
    let symbol = nfc_normalize(&concatenated);
    let concatenated_span = tscx.visited_dspan(dspan);
    if !rustc_lexer::is_ident(symbol.as_str()) {
        return Err(dcx.create_err(ConcatInvalidIdent {
            span: concatenated_span,
            reason: InvalidIdentReason::new(symbol),
        }));
    }
    tscx.psess.symbol_gallery.insert(symbol, concatenated_span);

    // The current implementation marks the span as coming from the macro regardless of
    // contexts of the concatenated identifiers but this behavior may change in the
    // future.
    Ok(TokenTree::Token(
        Token::from_ast_ident(Ident::new(symbol, concatenated_span)),
        Spacing::Alone,
    ))
}

/// Store the metavariable span for this original span into a side table.
/// FIXME: Try to put the metavariable span into `SpanData` instead of a side table (#118517).
/// An optimal encoding for inlined spans will need to be selected to minimize regressions.
/// The side table approach is relatively good, but not perfect due to collisions.
/// In particular, collisions happen when token is passed as an argument through several macro
/// calls, like in recursive macros.
/// The old heuristic below is used to improve spans in case of collisions, but diagnostics are
/// still degraded sometimes in those cases.
///
/// The old heuristic:
///
/// Usually metavariables `$var` produce interpolated tokens, which have an additional place for
/// keeping both the original span and the metavariable span. For `tt` metavariables that's not the
/// case however, and there's no place for keeping a second span. So we try to give the single
/// produced span a location that would be most useful in practice (the hygiene part of the span
/// must not be changed).
///
/// Different locations are useful for different purposes:
/// - The original location is useful when we need to report a diagnostic for the original token in
///   isolation, without combining it with any surrounding tokens. This case occurs, but it is not
///   very common in practice.
/// - The metavariable location is useful when we need to somehow combine the token span with spans
///   of its surrounding tokens. This is the most common way to use token spans.
///
/// So this function replaces the original location with the metavariable location in all cases
/// except these two:
/// - The metavariable is an element of undelimited sequence `$($tt)*`.
///   These are typically used for passing larger amounts of code, and tokens in that code usually
///   combine with each other and not with tokens outside of the sequence.
/// - The metavariable span comes from a different crate, then we prefer the more local span.
fn transcribe_flat_tt(tscx: &mut TranscrCtx<'_, '_>, metavar_span: Span, ftt: &FlatTt) {
    let mut metavar_span = metavar_span;
    let undelimited_seq = matches!(
        tscx.stack.last(),
        Some(ExecFrame::Sequence {
            seq: SeqSeg {
                inner,
                sep: None,
                kleene_op: KleeneOp::ZeroOrMore | KleeneOp::OneOrMore,
                ..
            },
            ..
        }) if inner.len() == 1
    );
    if undelimited_seq {
        // Do not record metavar spans for tokens from undelimited sequences, for perf reasons.
        splice_flat_tt(&mut tscx.sink, ftt);
        return;
    }

    tscx.marker.mark_span(&mut metavar_span);
    let no_collision = match ftt {
        FlatTt::Token(token, ..) => {
            with_metavar_spans(|mspans| mspans.insert(token.span, metavar_span))
        }
        FlatTt::Slice(slice) => {
            let entries = slice.entries();
            let (open, close) = (entries.first().unwrap(), entries.last().unwrap());
            let dspan = DelimSpan::from_pair(open.token.span, close.token.span);
            with_metavar_spans(|mspans| {
                mspans.insert(dspan.open, metavar_span)
                    && mspans.insert(dspan.close, metavar_span)
                    && mspans.insert(dspan.entire(), metavar_span)
            })
        }
    };
    if no_collision || tscx.psess.source_map().is_imported(metavar_span) {
        splice_flat_tt(&mut tscx.sink, ftt);
        return;
    }

    // Setting metavar spans for the heuristic spans gives better opportunities for combining them
    // with neighboring spans even despite their different syntactic contexts.
    match ftt {
        FlatTt::Token(Token { kind, span }, spacing) => {
            let span = metavar_span.with_ctxt(span.ctxt());
            with_metavar_spans(|mspans| mspans.insert(span, metavar_span));
            tscx.sink.push_token(Token { kind: *kind, span }, *spacing);
        }
        FlatTt::Slice(slice) => {
            let entries = slice.entries();
            let (open_span, close_span) =
                (entries.first().unwrap().token.span, entries.last().unwrap().token.span);
            let open = metavar_span.with_ctxt(open_span.ctxt());
            let close = metavar_span.with_ctxt(close_span.ctxt());
            with_metavar_spans(|mspans| {
                mspans.insert(open, metavar_span) && mspans.insert(close, metavar_span)
            });
            // Splice the group and rewrite the delimiter entries' spans.
            let (start, end) = tscx.sink.splice_slice(slice);
            tscx.sink.entries[start].token.span = open;
            tscx.sink.entries[end - 1].token.span = close;
        }
    }
}

/// Appends a captured `tt` fragment verbatim.
fn splice_flat_tt(sink: &mut FlatSink, ftt: &FlatTt) {
    match ftt {
        FlatTt::Slice(slice) => {
            sink.splice_slice(slice);
        }
        FlatTt::Token(token, spacing) => sink.push_token(*token, *spacing),
    }
}

/// Emits a non-`tt` metavariable fragment: its tokens wrapped in invisible
/// delimiters (unless already wrapped in invisible delimiters with the same
/// `MetaVarKind`, because some proc macros can't handle multiple layers of
/// invisible delimiters of the same `MetaVarKind`; this loses some span
/// info, though it hopefully won't matter).
fn emit_delimited_fragment(
    tscx: &mut TranscrCtx<'_, '_>,
    mut sp: Span,
    mk_span: Span,
    mv_kind: MetaVarKind,
    mut stream: TokenStream,
) {
    if stream.len() == 1 {
        let tree = stream.iter().next().unwrap();
        if let TokenTree::Delimited(_, _, delim, inner) = tree
            && let Delimiter::Invisible(InvisibleOrigin::MetaVar(mvk)) = delim
            && mv_kind == *mvk
        {
            stream = inner.clone();
        }
    }

    // Emit as tokens within `Delimiter::Invisible` to maintain parsing
    // priorities.
    tscx.marker.mark_span(&mut sp);
    with_metavar_spans(|mspans| mspans.insert(mk_span, sp));
    // Both the open delim and close delim get the same span, which covers the
    // `$foo` in the decl macro RHS.
    let delim = Delimiter::Invisible(InvisibleOrigin::MetaVar(mv_kind));
    tscx.sink.open_delim(Token::new(delim.as_open_token_kind(), sp), Spacing::Alone);
    tscx.sink.splice_stream(&stream);
    tscx.sink.close_delim(Token::new(delim.as_close_token_kind(), sp), Spacing::Alone);
}

/// Lookup the meta-var named `ident` and return the matched token tree from the invocation using
/// the set of matches `interpolations`.
///
/// See the definition of `repeats` in the `transcribe` function. `repeats` is used to descend
/// into the right place in nested matchers. If we attempt to descend too far, the macro writer has
/// made a mistake, and we return `None`.
fn lookup_cur_matched<'a>(
    ident: MacroRulesNormalizedIdent,
    interpolations: &'a FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    repeats: &[(usize, usize)],
) -> Option<&'a NamedMatch> {
    interpolations.get(&ident).map(|mut matched| {
        for &(idx, _) in repeats {
            match matched {
                MatchedSingle(_) => break,
                MatchedSeq(ads) => matched = ads.get(idx).unwrap(),
            }
        }

        matched
    })
}

/// Cached variant of [`lookup_cur_matched`] for the per-occurrence hot path.
/// Resolves through every repetition level but the innermost via
/// `TranscrCtx::lookup_cache`, then applies the innermost index, which is the
/// only part that changes between iterations of the innermost frame.
fn lookup_cur_matched_cached<'itp>(
    tscx: &mut TranscrCtx<'_, 'itp>,
    ident: MacroRulesNormalizedIdent,
) -> Option<&'itp NamedMatch> {
    let base = match tscx.lookup_cache.iter().find(|(id, _)| *id == ident) {
        Some(&(_, base)) => base,
        None => {
            let outer = &tscx.repeats[..tscx.repeats.len().saturating_sub(1)];
            let base = tscx.interp.get(&ident).map(|mut matched| {
                for &(idx, _) in outer {
                    match matched {
                        MatchedSingle(_) => break,
                        MatchedSeq(ads) => matched = ads.get(idx).unwrap(),
                    }
                }
                matched
            });
            // A pathological rule body could reference many distinct metavars
            // in one frame; cap the cache so lookups stay a short linear scan.
            if tscx.lookup_cache.len() < 8 {
                tscx.lookup_cache.push((ident, base));
            }
            base
        }
    };
    match (base, tscx.repeats.last()) {
        (Some(MatchedSeq(ads)), Some(&(idx, _))) => Some(ads.get(idx).unwrap()),
        _ => base,
    }
}

/// An accumulator over a TokenTree to be used with `fold`. During transcription, we need to make
/// sure that the size of each sequence and all of its nested sequences are the same as the sizes
/// of all the matched (nested) sequences in the macro invocation. If they don't match, somebody
/// has made a mistake (either the macro writer or caller).
#[derive(Clone)]
enum LockstepIterSize {
    /// No constraints on length of matcher. This is true for any TokenTree variants except a
    /// `MetaVar` with an actual `MatchedSeq` (as opposed to a `MatchedNonterminal`).
    Unconstrained,

    /// A `MetaVar` with an actual `MatchedSeq`. The length of the match and the name of the
    /// meta-var are returned.
    Constraint(usize, MacroRulesNormalizedIdent),

    /// Two `Constraint`s on the same sequence had different lengths. This is an error.
    Contradiction(String),
}

impl LockstepIterSize {
    /// Find incompatibilities in matcher/invocation sizes.
    /// - `Unconstrained` is compatible with everything.
    /// - `Contradiction` is incompatible with everything.
    /// - `Constraint(len)` is only compatible with other constraints of the same length.
    fn with(self, other: LockstepIterSize) -> LockstepIterSize {
        match self {
            LockstepIterSize::Unconstrained => other,
            LockstepIterSize::Contradiction(_) => self,
            LockstepIterSize::Constraint(l_len, l_id) => match other {
                LockstepIterSize::Unconstrained => self,
                LockstepIterSize::Contradiction(_) => other,
                LockstepIterSize::Constraint(r_len, _) if l_len == r_len => self,
                LockstepIterSize::Constraint(r_len, r_id) => {
                    let msg = format!(
                        "meta-variable `{}` repeats {} time{}, but `{}` repeats {} time{}",
                        l_id,
                        l_len,
                        pluralize!(l_len),
                        r_id,
                        r_len,
                        pluralize!(r_len),
                    );
                    LockstepIterSize::Contradiction(msg)
                }
            },
        }
    }
}

/// Given a sequence, make sure that all of its metavar occurrences have the same length as the
/// matches for the appropriate meta-vars in `interpolations`.
///
/// Note that if `repeats` does not match the exact correct depth of a meta-var,
/// `lookup_cur_matched` will return `None`, which is why this still works even in the presence of
/// multiple nested matcher sequences.
///
/// Example: `$($($x $y)+*);+` -- we need to make sure that `x` and `y` repeat the same amount as
/// each other at the given depth when the macro was invoked. If they don't it might mean they were
/// declared at depths which weren't equal or there was a compiler bug. For example, if we have 3 repetitions of
/// the outer sequence and 4 repetitions of the inner sequence for `x`, we should have the same for
/// `y`; otherwise, we can't transcribe them both at the given depth.
fn lockstep_iter_size(
    seq: &SeqSeg,
    interpolations: &FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    repeats: &[(usize, usize)],
) -> LockstepIterSize {
    let mut size = LockstepIterSize::Unconstrained;
    for &name in &seq.lockstep_vars {
        let var_size = match lookup_cur_matched(name, interpolations, repeats) {
            Some(MatchedSeq(ads)) => LockstepIterSize::Constraint(ads.len(), name),
            _ => LockstepIterSize::Unconstrained,
        };
        size = size.with(var_size);
    }
    size
}

/// Used solely by the `count` meta-variable expression, counts the outermost repetitions at a
/// given optional nested depth.
///
/// For example, a macro parameter of `$( { $( $foo:ident ),* } )*` called with `{ a, b } { c }`:
///
/// * `[ $( ${count(foo)} ),* ]` will return [2, 1] with a, b = 2 and c = 1
/// * `[ $( ${count(foo, 0)} ),* ]` will be the same as `[ $( ${count(foo)} ),* ]`
/// * `[ $( ${count(foo, 1)} ),* ]` will return an error because `${count(foo, 1)}` is
///   declared inside a single repetition and the index `1` implies two nested repetitions.
fn count_repetitions<'dx>(
    dcx: DiagCtxtHandle<'dx>,
    depth_user: usize,
    mut matched: &NamedMatch,
    repeats: &[(usize, usize)],
    sp: &DelimSpan,
) -> PResult<'dx, usize> {
    // Recursively count the number of matches in `matched` at given depth
    // (or at the top-level of `matched` if no depth is given).
    fn count<'a>(depth_curr: usize, depth_max: usize, matched: &NamedMatch) -> PResult<'a, usize> {
        match matched {
            MatchedSingle(_) => Ok(1),
            MatchedSeq(named_matches) => {
                if depth_curr == depth_max {
                    Ok(named_matches.len())
                } else {
                    named_matches.iter().map(|elem| count(depth_curr + 1, depth_max, elem)).sum()
                }
            }
        }
    }

    /// Maximum depth
    fn depth(counter: usize, matched: &NamedMatch) -> usize {
        match matched {
            MatchedSingle(_) => counter,
            MatchedSeq(named_matches) => {
                let rslt = counter + 1;
                if let Some(elem) = named_matches.first() { depth(rslt, elem) } else { rslt }
            }
        }
    }

    let depth_max = depth(0, matched)
        .checked_sub(1)
        .and_then(|el| el.checked_sub(repeats.len()))
        .unwrap_or_default();
    if depth_user > depth_max {
        return Err(out_of_bounds_err(dcx, depth_max + 1, sp.entire(), "count"));
    }

    // `repeats` records all of the nested levels at which we are currently
    // matching meta-variables. The meta-var-expr `count($x)` only counts
    // matches that occur in this "subtree" of the `NamedMatch` where we
    // are currently transcribing, so we need to descend to that subtree
    // before we start counting. `matched` contains the various levels of the
    // tree as we descend, and its final value is the subtree we are currently at.
    for &(idx, _) in repeats {
        if let MatchedSeq(ads) = matched {
            matched = &ads[idx];
        }
    }

    if let MatchedSingle(_) = matched {
        return Err(dcx.create_err(CountRepetitionMisplaced { span: sp.entire() }));
    }

    count(depth_user, depth_max, matched)
}

/// Returns a `NamedMatch` item declared on the LHS given an arbitrary [Ident]
fn matched_from_ident<'ctx, 'interp, 'rslt>(
    dcx: DiagCtxtHandle<'ctx>,
    ident: Ident,
    interp: &'interp FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
) -> PResult<'ctx, &'rslt NamedMatch>
where
    'interp: 'rslt,
{
    let span = ident.span;
    let key = MacroRulesNormalizedIdent::new(ident);
    interp.get(&key).ok_or_else(|| dcx.create_err(MveUnrecognizedVar { span, key }))
}

/// Used by meta-variable expressions when an user input is out of the actual declared bounds. For
/// example, index(999999) in an repetition of only three elements.
fn out_of_bounds_err<'a>(dcx: DiagCtxtHandle<'a>, max: usize, span: Span, ty: &str) -> Diag<'a> {
    let msg = if max == 0 {
        format!(
            "meta-variable expression `{ty}` with depth parameter \
             must be called inside of a macro repetition"
        )
    } else {
        format!(
            "depth parameter of meta-variable expression `{ty}` \
             must be less than {max}"
        )
    };
    dcx.struct_span_err(span, msg)
}

/// Extracts an metavariable symbol that can be an identifier, a token tree or a literal.
fn extract_symbol_from_pnr<'a>(
    dcx: DiagCtxtHandle<'a>,
    pnr: &ParseNtResult,
    span_err: Span,
) -> PResult<'a, Symbol> {
    match pnr {
        ParseNtResult::Ident(nt_ident, is_raw) => {
            if let IdentIsRaw::Yes = is_raw {
                Err(dcx.struct_span_err(span_err, RAW_IDENT_ERR))
            } else {
                Ok(nt_ident.name)
            }
        }
        ParseNtResult::Tt(FlatTt::Token(
            Token { kind: TokenKind::Ident(symbol, is_raw), .. },
            _,
        )) => {
            if let IdentIsRaw::Yes = is_raw {
                Err(dcx.struct_span_err(span_err, RAW_IDENT_ERR))
            } else {
                Ok(*symbol)
            }
        }
        ParseNtResult::Tt(FlatTt::Token(
            Token {
                kind: TokenKind::Literal(Lit { kind: LitKind::Str, symbol, suffix: None }),
                ..
            },
            _,
        )) => Ok(*symbol),
        ParseNtResult::Literal(expr)
            if let ExprKind::Lit(Lit { kind: LitKind::Str, symbol, suffix: None }) = &expr.kind =>
        {
            Ok(*symbol)
        }
        ParseNtResult::Literal(expr)
            if let ExprKind::Lit(lit @ Lit { kind: LitKind::Integer, symbol, suffix }) =
                &expr.kind =>
        {
            if lit.is_semantic_float() {
                Err(dcx
                    .struct_err("floats are not supported as metavariables of `${concat(..)}`")
                    .with_span(span_err))
            } else if suffix.is_none() {
                Ok(*symbol)
            } else {
                Err(dcx
                    .struct_err("integer metavariables of `${concat(..)}` must not be suffixed")
                    .with_span(span_err))
            }
        }
        _ => Err(dcx
            .struct_err(
                "metavariables of `${concat(..)}` must be of type `ident`, `literal` or `tt`",
            )
            .with_note("currently only string and integer literals are supported")
            .with_span(span_err)),
    }
}
