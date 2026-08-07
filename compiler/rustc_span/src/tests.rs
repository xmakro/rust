use super::*;

#[test]
fn test_lookup_line() {
    let source = "abcdefghijklm\nabcdefghij\n...".to_owned();
    let mut sf = SourceFile::new(
        FileName::Anon(Hash64::ZERO),
        source,
        SourceFileHashAlgorithm::Sha256,
        Some(SourceFileHashAlgorithm::Sha256),
    )
    .unwrap();
    sf.start_pos = BytePos(3);
    assert_eq!(sf.lines(), &[RelativeBytePos(0), RelativeBytePos(14), RelativeBytePos(25)]);

    assert_eq!(sf.lookup_line(RelativeBytePos(0)), Some(0));
    assert_eq!(sf.lookup_line(RelativeBytePos(1)), Some(0));

    assert_eq!(sf.lookup_line(RelativeBytePos(13)), Some(0));
    assert_eq!(sf.lookup_line(RelativeBytePos(14)), Some(1));
    assert_eq!(sf.lookup_line(RelativeBytePos(15)), Some(1));

    assert_eq!(sf.lookup_line(RelativeBytePos(25)), Some(2));
    assert_eq!(sf.lookup_line(RelativeBytePos(26)), Some(2));
}

#[test]
fn test_normalize_newlines() {
    fn check(before: &str, after: &str, expected_positions: &[u32]) {
        let mut actual = before.to_string();
        let mut actual_positions = vec![];
        normalize_newlines(&mut actual, &mut actual_positions);
        let actual_positions: Vec<_> = actual_positions.into_iter().map(|nc| nc.pos.0).collect();
        assert_eq!(actual.as_str(), after);
        assert_eq!(actual_positions, expected_positions);
    }
    check("", "", &[]);
    check("\n", "\n", &[]);
    check("\r", "\r", &[]);
    check("\r\r", "\r\r", &[]);
    check("\r\n", "\n", &[1]);
    check("hello world", "hello world", &[]);
    check("hello\nworld", "hello\nworld", &[]);
    check("hello\r\nworld", "hello\nworld", &[6]);
    check("\r\nhello\r\nworld\r\n", "\nhello\nworld\n", &[1, 7, 13]);
    check("\r\r\n", "\r\n", &[2]);
    check("hello\rworld", "hello\rworld", &[]);
}

#[test]
fn test_trim() {
    let span = |lo: usize, hi: usize| {
        Span::new(BytePos::from_usize(lo), BytePos::from_usize(hi), SyntaxContext::root(), None)
    };

    // Various positions, named for their relation to `start` and `end`.
    let well_before = 1;
    let before = 3;
    let start = 5;
    let mid = 7;
    let end = 9;
    let after = 11;
    let well_after = 13;

    // The resulting span's context should be that of `self`, not `other`.
    let other = span(start, end).with_ctxt(SyntaxContext::from_u32(999));

    // Test cases for `trim_end`.

    assert_eq!(span(well_before, before).trim_end(other), Some(span(well_before, before)));
    assert_eq!(span(well_before, start).trim_end(other), Some(span(well_before, start)));
    assert_eq!(span(well_before, mid).trim_end(other), Some(span(well_before, start)));
    assert_eq!(span(well_before, end).trim_end(other), Some(span(well_before, start)));
    assert_eq!(span(well_before, after).trim_end(other), Some(span(well_before, start)));

    assert_eq!(span(start, mid).trim_end(other), None);
    assert_eq!(span(start, end).trim_end(other), None);
    assert_eq!(span(start, after).trim_end(other), None);

    assert_eq!(span(mid, end).trim_end(other), None);
    assert_eq!(span(mid, after).trim_end(other), None);

    assert_eq!(span(end, after).trim_end(other), None);

    assert_eq!(span(after, well_after).trim_end(other), None);

    // Test cases for `trim_start`.

    assert_eq!(span(after, well_after).trim_start(other), Some(span(after, well_after)));
    assert_eq!(span(end, well_after).trim_start(other), Some(span(end, well_after)));
    assert_eq!(span(mid, well_after).trim_start(other), Some(span(end, well_after)));
    assert_eq!(span(start, well_after).trim_start(other), Some(span(end, well_after)));
    assert_eq!(span(before, well_after).trim_start(other), Some(span(end, well_after)));

    assert_eq!(span(mid, end).trim_start(other), None);
    assert_eq!(span(start, end).trim_start(other), None);
    assert_eq!(span(before, end).trim_start(other), None);

    assert_eq!(span(start, mid).trim_start(other), None);
    assert_eq!(span(before, mid).trim_start(other), None);

    assert_eq!(span(before, start).trim_start(other), None);

    assert_eq!(span(well_before, before).trim_start(other), None);
}

#[test]
fn test_unnormalized_source_length() {
    let source = "\u{feff}hello\r\nferries\r\n".to_owned();
    let sf = SourceFile::new(
        FileName::Anon(Hash64::ZERO),
        source,
        SourceFileHashAlgorithm::Sha256,
        Some(SourceFileHashAlgorithm::Sha256),
    )
    .unwrap();
    assert_eq!(sf.unnormalized_source_len, 19);
    assert_eq!(sf.normalized_source_len.0, 14);
}

fn file_for_extent_hash(src: &str) -> SourceFile {
    SourceFile::new(
        FileName::Anon(Hash64::ZERO),
        src.to_owned(),
        SourceFileHashAlgorithm::Sha256,
        None,
    )
    .unwrap()
}

fn extent_of(file: &SourceFile, needle: &str, len: usize) -> (RelativeBytePos, RelativeBytePos) {
    let start = file.src.as_ref().unwrap().find(needle).unwrap() as u32;
    (RelativeBytePos(start), RelativeBytePos(start + len as u32))
}

#[test]
fn line_extent_hash_ignores_edits_outside_extent() {
    // The extent covers `fn f` on one line; edits after it, and byte-shifting edits
    // before it that preserve line structure, leave the hash unchanged.
    let a = file_for_extent_hash(
        "const A: u8 = 1;
fn f() {
    g()
}
fn h() {}
",
    );
    let b = file_for_extent_hash(
        "const A: u32 = 1;
fn f() {
    g()
}
fn hh() {}
",
    );
    let (lo_a, hi_a) = extent_of(&a, "fn f", 15);
    let (lo_b, hi_b) = extent_of(&b, "fn f", 15);
    assert_ne!(lo_a, lo_b);
    assert_eq!(a.line_extent_hash(lo_a, hi_a), b.line_extent_hash(lo_b, hi_b));
}

#[test]
fn line_extent_hash_covers_anchor_line() {
    // A line added before the extent moves the anchor line and must change the hash even
    // though the extent's relative structure is identical.
    let a = file_for_extent_hash(
        "const A: u8 = 1;
fn f() {
    g()
}
",
    );
    let b = file_for_extent_hash(
        "const A: u8 =
1;
fn f() {
    g()
}
",
    );
    let (lo_a, hi_a) = extent_of(&a, "fn f", 15);
    let (lo_b, hi_b) = extent_of(&b, "fn f", 15);
    assert_ne!(a.line_extent_hash(lo_a, hi_a), b.line_extent_hash(lo_b, hi_b));
}

#[test]
fn line_extent_hash_covers_internal_line_moves() {
    // A net-zero line move inside the extent changes the hash even with identical byte
    // offsets (the #74890 class).
    let a = file_for_extent_hash(
        "fn f() {
    g();    h()
}
",
    );
    let b = file_for_extent_hash(
        "fn f() {
    g(); //
h()
}
",
    );
    assert_eq!(a.src.as_ref().unwrap().len(), b.src.as_ref().unwrap().len());
    let (lo, hi) = extent_of(&a, "fn f", a.src.as_ref().unwrap().len() - 1);
    assert_ne!(a.line_extent_hash(lo, hi), b.line_extent_hash(lo, hi));
}

#[test]
fn line_extent_hash_covers_first_line_prefix_tables() {
    // A definition starting mid-line: multibyte characters between the line start and the
    // definition affect its character columns and must be covered ("é" is two bytes).
    let a = file_for_extent_hash("const S: &str = \"é\"; fn f() { g() }\n");
    let b = file_for_extent_hash("const S: &str = \"xy\"; fn f() { g() }\n");
    let (lo_a, hi_a) = extent_of(&a, "fn f", 14);
    let (lo_b, hi_b) = extent_of(&b, "fn f", 14);
    assert_eq!(lo_a, lo_b);
    assert_ne!(a.line_extent_hash(lo_a, hi_a), b.line_extent_hash(lo_b, hi_b));
}

#[test]
fn line_extent_hash_covers_first_line_column() {
    // The definition moves within its line while every line length and table entry stays
    // the same (the sibling before it shrinks, trailing padding keeps the length):
    // rendered columns of first-line positions change, so the hash must change.
    let a = file_for_extent_hash("const AB: u8 = 1; fn f() {}\n");
    let b = file_for_extent_hash("const A: u8 = 1; fn f() {} \n");
    let (lo_a, hi_a) = extent_of(&a, "fn f", 9);
    let (lo_b, hi_b) = extent_of(&b, "fn f", 9);
    assert_ne!(lo_a, lo_b);
    assert_ne!(a.line_extent_hash(lo_a, hi_a), b.line_extent_hash(lo_b, hi_b));
}

#[test]
fn line_extent_hash_empty_file() {
    let empty = file_for_extent_hash("");
    assert_eq!(
        empty.line_extent_hash(RelativeBytePos(0), RelativeBytePos(0)),
        empty.line_extent_hash(RelativeBytePos(0), RelativeBytePos(0)),
    );
    let one = file_for_extent_hash("x");
    assert_ne!(
        empty.line_extent_hash(RelativeBytePos(0), RelativeBytePos(0)),
        one.line_extent_hash(RelativeBytePos(0), RelativeBytePos(1)),
    );
}

#[test]
fn line_extent_hash_block_path() {
    // Extents longer than one length block use the cached block hashes; the result must
    // still be shift-invariant against upstream byte edits and sensitive to internal
    // line splits.
    let body = "    x();\n".repeat(150);
    let a = file_for_extent_hash(&format!("const A: u8 = 1;\nfn f() {{\n{body}}}\n"));
    let b = file_for_extent_hash(&format!("const A: u32 = 1;\nfn f() {{\n{body}}}\n"));
    let len = "fn f() {\n".len() + body.len() + 1;
    let (lo_a, hi_a) = extent_of(&a, "fn f", len);
    let (lo_b, hi_b) = extent_of(&b, "fn f", len);
    assert_eq!(a.line_extent_hash(lo_a, hi_a), b.line_extent_hash(lo_b, hi_b));

    let split = body.replacen("    x();", "    x\n();", 1);
    let c = file_for_extent_hash(&format!("const A: u8 = 1;\nfn f() {{\n{split}}}\n"));
    let (lo_c, hi_c) = extent_of(&c, "fn f", len);
    assert_ne!(a.line_extent_hash(lo_a, hi_a), c.line_extent_hash(lo_c, hi_c));
}
