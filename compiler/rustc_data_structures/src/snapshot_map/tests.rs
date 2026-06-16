use super::SnapshotMap;

#[test]
fn basic() {
    let mut map = SnapshotMap::default();
    map.insert(22, "twenty-two");
    let snapshot = map.snapshot();
    map.insert(22, "thirty-three");
    assert_eq!(map[&22], "thirty-three");
    map.insert(44, "forty-four");
    assert_eq!(map[&44], "forty-four");
    assert_eq!(map.get(&33), None);
    map.rollback_to(snapshot);
    assert_eq!(map[&22], "twenty-two");
    assert_eq!(map.get(&33), None);
    assert_eq!(map.get(&44), None);
}

#[test]
#[should_panic]
fn out_of_order() {
    let mut map = SnapshotMap::default();
    map.insert(22, "twenty-two");
    let snapshot1 = map.snapshot();
    map.insert(33, "thirty-three");
    let snapshot2 = map.snapshot();
    map.insert(44, "forty-four");
    map.rollback_to(snapshot1); // bogus, but accepted
    map.rollback_to(snapshot2); // asserts
}

#[test]
fn nested_commit_then_rollback() {
    let mut map = SnapshotMap::default();
    map.insert(22, "twenty-two");
    let snapshot1 = map.snapshot();
    let snapshot2 = map.snapshot();
    map.insert(22, "thirty-three");
    map.commit(snapshot2);
    assert_eq!(map[&22], "thirty-three");
    map.rollback_to(snapshot1);
    assert_eq!(map[&22], "twenty-two");
}

#[test]
fn insert_if_absent_inserts_and_rolls_back() {
    let mut map = SnapshotMap::default();
    let snapshot = map.snapshot();
    assert_eq!(map.insert_if_absent(22, "twenty-two"), None);
    assert_eq!(map[&22], "twenty-two");
    assert_eq!(map.insert_if_absent(22, "thirty-three"), Some("twenty-two"));
    assert_eq!(map[&22], "twenty-two");
    map.rollback_to(snapshot);
    assert_eq!(map.get(&22), None);
}

#[test]
fn overwrite_unless_keeps_overwrites_and_inserts() {
    let mut map = SnapshotMap::default();
    map.insert(22, "twenty-two");
    let snapshot = map.snapshot();
    assert_eq!(map.overwrite_unless(22, "thirty-three", |v| *v == "twenty-two"), None);
    assert_eq!(map[&22], "twenty-two");
    assert_eq!(map.overwrite_unless(22, "thirty-three", |_| false), Some(false));
    assert_eq!(map[&22], "thirty-three");
    assert_eq!(map.overwrite_unless(44, "forty-four", |_| false), Some(true));
    assert_eq!(map[&44], "forty-four");
    map.rollback_to(snapshot);
    assert_eq!(map[&22], "twenty-two");
    assert_eq!(map.get(&44), None);
}
