use super::*;

fn table(region_end: u64) -> SyntaxContextTable {
    SyntaxContextTable::new(AbsoluteBytePos(region_end), FxHashMap::default())
}

fn selected_region(tables: &[SyntaxContextTable], position: u64) -> Option<u64> {
    syntax_context_table_for(tables, AbsoluteBytePos(position)).map(|table| table.region_end.0)
}

#[test]
fn syntax_context_table_selection() {
    assert_eq!(selected_region(&[], 0), None);

    // A table covers the positions up to, but excluding, its region end.
    let tables = [table(100), table(250)];
    assert_eq!(selected_region(&tables, 0), Some(100));
    assert_eq!(selected_region(&tables, 99), Some(100));
    assert_eq!(selected_region(&tables, 100), Some(250));
    assert_eq!(selected_region(&tables, 249), Some(250));
    assert_eq!(selected_region(&tables, 250), None);
}
