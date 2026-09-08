use rustc_span::DUMMY_SP;

use crate::token::TokenKind;
use crate::tokenstream::TokenStream;

#[test]
fn test_token_stream_iter() {
    let ts = TokenStream::token_alone(TokenKind::Eq, DUMMY_SP);
    assert_eq!(ts.len(), 1);

    let iter = ts.iter();
    assert_eq!(iter.size_hint(), (1, Some(1)));
}

#[test]
fn test_counted_token_cursor_skip() {
    use crate::token::{Delimiter, InvisibleOrigin, MetaVarKind};
    use crate::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenCursor, TokenTree};

    let leaf = TokenTree::token_alone(TokenKind::ShrEq, DUMMY_SP);
    let mut stream = TokenStream::new(vec![leaf.clone(); 120]);
    let delimiters = [
        Delimiter::Brace,
        Delimiter::Parenthesis,
        Delimiter::Bracket,
        Delimiter::Invisible(InvisibleOrigin::ProcMacro),
        Delimiter::Invisible(InvisibleOrigin::MetaVar(MetaVarKind::Block)),
    ];
    for depth in 0..25 {
        stream = TokenStream::new(vec![
            leaf.clone(),
            TokenTree::Delimited(
                DelimSpan::dummy(),
                DelimSpacing::new(Spacing::Alone, Spacing::Alone),
                delimiters[depth % delimiters.len()],
                stream,
            ),
            leaf.clone(),
        ]);
    }
    let mut cursor = TokenCursor::new(stream);
    loop {
        let token = cursor.next_and_bump().0;
        if token.kind == TokenKind::Eof {
            break;
        }
        if token.kind.open_delim().is_some() {
            let depth = cursor.depth();
            let mut stepped = cursor.clone();
            let mut skipped = cursor.clone();
            let skipped_count = skipped.bump_to_end_with_count();
            let skipped_close = skipped.next_and_bump();
            let mut stepped_count = 0;
            loop {
                let next = stepped.next_and_bump();
                stepped_count += 1;
                if stepped.depth() < depth {
                    assert_eq!(next, skipped_close);
                    break;
                }
            }
            assert_eq!(stepped_count, skipped_count + 1);
            assert_eq!(stepped.next_and_bump(), skipped.next_and_bump());
            assert_eq!(stepped.depth(), skipped.depth());
        }
    }
}
