//! The data that we will serialize and deserialize.
//!
//! Notionally, the dep-graph is a sequence of NodeInfo with the dependencies
//! specified inline. A footer stores the dead list, the per-kind counts, and the
//! total number of nodes and edges, with fixed-size positions at the very end of
//! the file so we can find them easily at decoding time.
//!
//! The serialisation is performed on-demand when each node is emitted. Using this
//! scheme, we do not need to keep the current graph in memory.
//!
//! On a warm rebuild most nodes are unchanged. Rather than re-encoding them, the
//! previous file's record region is copied into the new file wholesale and every
//! node that existed in the previous session keeps its index: a node re-verified
//! green needs no write at all (its carried record, whose edges point at other
//! kept indices, is already exactly right), a re-executed node appends a record
//! at its old index which overrides the carried one (later records win at decode
//! time), and a node this session dropped is tombstoned via the dead list in the
//! footer. Genuinely new nodes get indices above the previous index space. Dead
//! records and superseded duplicates accumulate with each carried generation, so
//! after [`MAX_CARRIED_GENERATIONS`] the file is rewritten fresh, which also
//! happens when a debugging feature needs every node to pass through the encoder.
//!
//! The deserialization is performed manually, in order to convert from the stored
//! sequence of NodeInfos to the different arrays in SerializedDepGraph. Since the
//! node and edge count are stored at the end of the file, all the arrays can be
//! pre-allocated with the right length.
//!
//! The encoding of the dep-graph is generally designed around the fact that fixed-size
//! reads of encoded data are generally faster than variable-sized reads. Ergo we adopt
//! essentially the same varint encoding scheme used in the rmeta format; the edge lists
//! for each node on the graph store a 2-bit integer which is the number of bytes per edge
//! index in that node's edge list. We effectively ignore that an edge index of 0 could be
//! encoded with 0 bytes in order to not require 3 bits to store the byte width of the edges.
//! The overhead of calculating the correct byte width for each edge is mitigated by
//! building edge lists with [`EdgesVec`] which keeps a running max of the edges in a node.
//!
//! When we decode this data, we do not immediately create [`SerializedDepNodeIndex`] and
//! instead keep the data in its denser serialized form which lets us turn our on-disk size
//! efficiency directly into a peak memory reduction. When we convert these encoded-in-memory
//! values into their fully-deserialized type, we use a fixed-size read of the encoded array
//! then mask off any errant bytes we read. The array of edge index bytes is padded to permit this.
//!
//! We also encode and decode the entire rest of each node using [`SerializedNodeHeader`]
//! to let this encoding and decoding be done in one fixed-size operation. These headers contain
//! two [`Fingerprint`]s along with the serialized [`DepKind`], and the number of edge indices
//! in the node and the number of bytes used to encode the edge indices for this node. The
//! [`DepKind`], number of edges, and bytes per edge are all bit-packed together, if they fit.
//! If the number of edges in this node does not fit in the bits available in the header, we
//! store it directly after the header with leb128.
//!
//! Dep-graph indices are bulk allocated to threads inside `LocalEncoderState`. Having threads
//! own these indices helps avoid races when they are conditionally used when marking nodes green.
//! It also reduces congestion on the shared index count.

use std::cell::RefCell;
use std::cmp::max;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::{iter, mem};

use rustc_data_structures::fingerprint::{Fingerprint, PackedFingerprint};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::outline;
use rustc_index::bit_set::DenseBitSet;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::{AtomicU64, Lock, WorkerLocal, broadcast};
use rustc_data_structures::unhash::UnhashMap;
use rustc_index::{IndexSlice, IndexVec};
use rustc_serialize::opaque::mem_encoder::MemEncoder;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder, IntEncodedWithFixedSize, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_session::Session;
use tracing::{debug, instrument};

use super::graph::{CurrentDepGraph, DepNodeColorMap, DesiredColor, TrySetColorResult};
use super::retained::RetainedDepGraph;
use super::{DepKind, DepNode, DepNodeIndex};
use crate::dep_graph::edges::EdgesVec;

// The maximum value of `SerializedDepNodeIndex` leaves the upper two bits
// unused so that we can store multiple index types in `CompressedHybridIndex`,
// and use those bits to encode which index type it contains.
rustc_index::newtype_index! {
    #[encodable]
    #[max = 0x7FFF_FFFF]
    pub struct SerializedDepNodeIndex {}
}

impl SerializedDepNodeIndex {
    /// Converts a current-session dep node index to a "serialized" index,
    /// for the purpose of serializing data to be loaded by future sessions.
    #[inline(always)]
    pub fn from_curr_for_serialization(index: DepNodeIndex) -> Self {
        SerializedDepNodeIndex::from_u32(index.as_u32())
    }
}

const DEP_NODE_SIZE: usize = size_of::<SerializedDepNodeIndex>();
/// Amount of padding we need to add to the edge list data so that we can retrieve every
/// SerializedDepNodeIndex with a fixed-size read then mask.
const DEP_NODE_PAD: usize = DEP_NODE_SIZE - 1;
/// Number of bits we need to store the number of used bytes in a SerializedDepNodeIndex.
/// Note that wherever we encode byte widths like this we actually store the number of bytes used
/// minus 1; for a 4-byte value we technically would have 5 widths to store, but using one byte to
/// store zeroes (which are relatively rare) is a decent tradeoff to save a bit in our bitfields.
const DEP_NODE_WIDTH_BITS: usize = DEP_NODE_SIZE / 2;

/// Data for use when recompiling the **current crate**.
///
/// There may be unused indices with DepKind::Null in this graph due to batch allocation of
/// indices to threads.
#[derive(Default)]
pub struct SerializedDepGraph {
    /// The set of all DepNodes in the graph
    nodes: IndexVec<SerializedDepNodeIndex, DepNode>,
    /// A value fingerprint associated with each [`DepNode`] in [`Self::nodes`],
    /// typically a hash of the value returned by the node's query in the
    /// previous incremental-compilation session.
    ///
    /// Some nodes don't have a meaningful value hash (e.g. queries with `no_hash`),
    /// so they store a dummy value here instead (e.g. [`Fingerprint::ZERO`]).
    value_fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    edge_list_indices: IndexVec<SerializedDepNodeIndex, EdgeHeader>,
    /// A flattened list of all edge targets in the graph, stored in the same
    /// varint encoding that we use on disk. Edge sources are implicit in edge_list_indices.
    edge_list_data: Vec<u8>,
    /// The lazily-built inverse of `nodes`: maps a [`DepNode`] back to its
    /// [`SerializedDepNodeIndex`] via the node's key fingerprint. See
    /// [`LazyNodeIndex`].
    reverse_index: LazyNodeIndex,
    /// The number of previous compilation sessions. This is used to generate
    /// unique anon dep nodes per session.
    session_count: u64,
    /// How many consecutive sessions have carried the record region forward without
    /// a compacting rewrite. Dead records and superseded duplicates accumulate with
    /// each carried generation, so the writer compacts once this grows too large.
    generation: u64,
    /// The memory-mapped bytes of the file this graph was decoded from, retained so
    /// that the record region can be copied into the next session's file wholesale
    /// (see [`Self::region_bytes`]). `None` for the empty default graph, which
    /// disables the carry.
    mmap: Option<Mmap>,
    /// The byte range of the record region within [`Self::mmap`]: every node record,
    /// including dead and superseded ones, and nothing else.
    records_range: std::ops::Range<usize>,
    /// Indices whose record in the region is dead: the node was dropped by an earlier
    /// session (never re-verified nor re-executed), so the record must be ignored.
    /// Cumulative across carried generations; reset by a compacting rewrite.
    dead: Vec<SerializedDepNodeIndex>,
    /// The per-`DepKind` counts of live nodes, from the file footer. Retained so the
    /// next session can compute its own footer counts as a delta.
    kind_stats: Vec<u32>,
    /// The number of live nodes, from the file footer (`nodes.len()` counts `Null`
    /// index slots too).
    live_node_count: u64,
    /// The number of edges of live nodes, from the file footer.
    live_edge_count: u64,
    /// Used to time the lazy per-`DepKind` reverse-index build. `None` only for
    /// the empty default graph, which is never looked up.
    profiler: Option<SelfProfilerRef>,
}

// `SelfProfilerRef` is not `Debug`, so we can't derive this.
impl std::fmt::Debug for SerializedDepGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SerializedDepGraph")
            .field("nodes", &self.nodes)
            .field("value_fingerprints", &self.value_fingerprints)
            .field("edge_list_indices", &self.edge_list_indices)
            .field("edge_list_data", &self.edge_list_data)
            .field("reverse_index", &self.reverse_index)
            .field("session_count", &self.session_count)
            .finish_non_exhaustive()
    }
}

/// The inverse of [`SerializedDepGraph::nodes`], built lazily per [`DepKind`].
///
/// Only few nodes are ever looked up here, and those cluster into a handful of
/// `DepKind`s. Building a map for every kind up front would be wasted work.
#[derive(Debug, Default)]
struct LazyNodeIndex {
    /// All (non-`Null`) node indices, grouped into contiguous per-`DepKind`
    /// ranges described by `kinds`. For any non-`Null` `DepKind` `k`, all values in
    /// `nodes_by_kind[kinds[k].start..][..kinds[k].len]`
    /// must be `Some` and have kind `k`.
    nodes_by_kind: Vec<Option<SerializedDepNodeIndex>>,
    /// For each `DepKind`, the range of `nodes_by_kind` holding its node indices
    /// and the lazily-built fingerprint map over that range.
    kinds: Vec<LazyKindIndex>,
}

#[derive(Debug, Default)]
struct LazyKindIndex {
    /// Offset into `LazyNodeIndex::nodes_by_kind` of this kind's first node.
    start: u32,
    /// Number of nodes of this kind.
    len: u32,
    /// `key_fingerprint -> node index`, built from this kind's range on first
    /// lookup. Empty kinds (and kinds never looked up) never build a map.
    map: OnceLock<UnhashMap<PackedFingerprint, SerializedDepNodeIndex>>,
}

impl LazyKindIndex {
    /// Returns this kind's `key_fingerprint -> node index` map.
    fn fingerprint_map(
        &self,
        kind: DepKind,
        nodes: &IndexSlice<SerializedDepNodeIndex, DepNode>,
        nodes_by_kind: &[Option<SerializedDepNodeIndex>],
        profiler: &Option<SelfProfilerRef>,
    ) -> &UnhashMap<PackedFingerprint, SerializedDepNodeIndex> {
        self.map.get_or_init(|| {
            let _prof_timer = profiler
                .as_ref()
                .map(|p| p.generic_activity("incr_comp_load_dep_graph_reverse_index"));
            let range = (self.start as usize)..(self.start as usize + self.len as usize);
            let mut map =
                UnhashMap::with_capacity_and_hasher(self.len as usize, Default::default());
            for &idx in &nodes_by_kind[range] {
                let idx = idx.expect("counting sort fills every slot of a kind's range");
                let node = nodes[idx];
                debug_assert_eq!(node.kind, kind);
                if map.insert(node.key_fingerprint, idx).is_some()
                    // Side effect nodes can legitimately share a fingerprint.
                    && node.kind != DepKind::SideEffect
                {
                    panic!(
                        "Error: A dep graph node ({kind:?}) does not have an unique index. \
                         Running a clean build on a nightly compiler with \
                         `-Z incremental-verify-ich` can help narrow down the issue for reporting. \
                         A clean build may also work around the issue.\n
                         DepNode: {node:?}"
                    )
                }
            }
            map
        })
    }
}

impl SerializedDepGraph {
    #[inline]
    pub fn edge_targets_from(
        &self,
        source: SerializedDepNodeIndex,
    ) -> impl Iterator<Item = SerializedDepNodeIndex> + Clone {
        let header = self.edge_list_indices[source];
        let mut raw = &self.edge_list_data[header.start()..];

        let bytes_per_index = header.bytes_per_index();

        // LLVM doesn't hoist EdgeHeader::mask so we do it ourselves.
        let mask = header.mask();
        (0..header.num_edges).map(move |_| {
            // Doing this slicing in this order ensures that the first bounds check suffices for
            // all the others.
            let index = &raw[..DEP_NODE_SIZE];
            raw = &raw[bytes_per_index..];
            let index = u32::from_le_bytes(index.try_into().unwrap()) & mask;
            SerializedDepNodeIndex::from_u32(index)
        })
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> &DepNode {
        &self.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode) -> Option<SerializedDepNodeIndex> {
        let kind = self.reverse_index.kinds.get(dep_node.kind.as_usize())?;
        let map = kind.fingerprint_map(
            dep_node.kind,
            &self.nodes,
            &self.reverse_index.nodes_by_kind,
            &self.profiler,
        );
        map.get(&dep_node.key_fingerprint).copied()
    }

    #[inline]
    pub fn value_fingerprint_for_index(
        &self,
        dep_node_index: SerializedDepNodeIndex,
    ) -> Fingerprint {
        self.value_fingerprints[dep_node_index]
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn session_count(&self) -> u64 {
        self.session_count
    }

    /// Whether this graph's record region can be carried into the next session's file
    /// wholesale. False for the empty default graph (no retained bytes).
    #[inline]
    fn can_carry(&self) -> bool {
        self.mmap.is_some()
    }

    /// The raw bytes of the record region, exactly as they appeared in this graph's
    /// file: every node record, including dead and superseded ones.
    ///
    /// Every node re-verified or re-executed this session keeps its previous index, so
    /// these records remain valid in the next file as-is: records of promoted green
    /// nodes are byte-for-byte what a fresh encode would produce, records superseded by
    /// a re-executed node are overridden by the appended record at the same index, and
    /// records of dropped nodes are tombstoned via the dead list in the footer.
    #[inline]
    fn region_bytes(&self) -> &[u8] {
        &self.mmap.as_ref().unwrap()[self.records_range.clone()]
    }

    /// The number of edges of the node at `index`, used for O(changed) footer accounting.
    #[inline]
    fn edge_count_for_index(&self, index: SerializedDepNodeIndex) -> usize {
        self.edge_list_indices[index].num_edges as usize
    }

    /// Whether the node at `index` has a (live or superseded) record in the region.
    /// `Null` slots come from batch index allocation and dead records of earlier
    /// generations; neither leaves a live record to tombstone.
    #[inline]
    fn index_is_occupied(&self, index: SerializedDepNodeIndex) -> bool {
        self.nodes[index].kind != DepKind::Null
    }

    /// Attaches the retained file bytes decoded by [`Self::decode`], enabling the
    /// carry of this graph's record region into the next session's file.
    pub fn attach_mmap(&mut self, mmap: Mmap) {
        self.mmap = Some(mmap);
    }
}

/// A packed representation of an edge's start index and byte width.
///
/// This is packed by stealing 2 bits from the start index, which means we only accommodate edge
/// data arrays up to a quarter of our address space. Which seems fine.
#[derive(Debug, Clone, Copy)]
struct EdgeHeader {
    repr: usize,
    num_edges: u32,
}

impl EdgeHeader {
    #[inline]
    fn start(self) -> usize {
        self.repr >> DEP_NODE_WIDTH_BITS
    }

    #[inline]
    fn bytes_per_index(self) -> usize {
        (self.repr & mask(DEP_NODE_WIDTH_BITS)) + 1
    }

    #[inline]
    fn mask(self) -> u32 {
        mask(self.bytes_per_index() * 8) as u32
    }
}

#[inline]
fn mask(bits: usize) -> usize {
    usize::MAX >> ((size_of::<usize>() * 8) - bits)
}

impl SerializedDepGraph {
    #[instrument(level = "debug", skip(d, profiler))]
    pub fn decode(d: &mut MemDecoder<'_>, profiler: &SelfProfilerRef) -> Arc<SerializedDepGraph> {
        // The last 32 bytes are the position of the dead list (which is also where the
        // record region ends), the node max, and the live node and edge counts.
        debug!("position: {:?}", d.position());

        // `node_max` is the number of indices including empty nodes while `node_count`
        // is the number of live nodes: records that are neither dead nor superseded by
        // a later record at the same index.
        let (dead_pos, node_max, node_count, edge_count) =
            d.with_position(d.len() - 4 * IntEncodedWithFixedSize::ENCODED_SIZE, |d| {
                debug!("position: {:?}", d.position());
                let dead_pos = IntEncodedWithFixedSize::decode(d).0 as usize;
                let node_max = IntEncodedWithFixedSize::decode(d).0 as usize;
                let node_count = IntEncodedWithFixedSize::decode(d).0 as usize;
                let edge_count = IntEncodedWithFixedSize::decode(d).0 as usize;
                (dead_pos, node_max, node_count, edge_count)
            });
        debug!("position: {:?}", d.position());

        debug!(?node_count, ?edge_count);

        let records_start = d.position();

        // The footer between the records and the fixed-size tail: the dead list, the
        // per-kind live counts, the session count and the carried generation count.
        // Read it up front, as decoding the records requires the dead set.
        let (dead, dead_set, kind_stats, session_count, generation) =
            d.with_position(dead_pos, |d| {
                let dead_len = d.read_u64() as usize;
                let mut dead = Vec::with_capacity(dead_len);
                let mut dead_set = DenseBitSet::new_empty(node_max);
                for _ in 0..dead_len {
                    let index = SerializedDepNodeIndex::from_u32(u32::from_le_bytes(d.read_array()));
                    dead_set.insert(index);
                    dead.push(index);
                }
                let kind_stats: Vec<u32> =
                    (0..(DepKind::MAX + 1)).map(|_| d.read_u32()).collect();
                let session_count = d.read_u64();
                let generation = d.read_u64();
                (dead, dead_set, kind_stats, session_count, generation)
            });

        // The record region may contain more than `node_count` records: dead records
        // and superseded ones (a later record at the same index overrides an earlier
        // one). This makes the capacity estimate below overshoot slightly more.
        let graph_bytes = dead_pos - records_start;

        let mut nodes = IndexVec::from_elem_n(
            DepNode {
                kind: DepKind::Null,
                key_fingerprint: PackedFingerprint::from(Fingerprint::ZERO),
            },
            node_max,
        );
        let mut value_fingerprints = IndexVec::from_elem_n(Fingerprint::ZERO, node_max);
        let mut edge_list_indices =
            IndexVec::from_elem_n(EdgeHeader { repr: 0, num_edges: 0 }, node_max);

        // This estimation assumes that all of the encoded bytes are for the edge lists or for the
        // fixed-size node headers. But that's not necessarily true; if any edge list has a length
        // that spills out of the size we can bit-pack into SerializedNodeHeader then some of the
        // total serialized size is also used by leb128-encoded edge list lengths. Neglecting that
        // contribution to graph_bytes means our estimation of the bytes needed for edge_list_data
        // slightly overshoots. But it cannot overshoot by much; consider that the worse case is
        // for a node with length 64, which means the spilled 1-byte leb128 length is 1 byte of at
        // least (34 byte header + 1 byte len + 64 bytes edge data), which is ~1%. A 2-byte leb128
        // length is about the same fractional overhead and it amortizes for yet greater lengths.
        let mut edge_list_data =
            Vec::with_capacity(graph_bytes - node_count * size_of::<SerializedNodeHeader>());

        while d.position() < dead_pos {
            // Decode the header for this edge; the header packs together as many of the fixed-size
            // fields as possible to limit the number of times we update decoder state.
            let node_header = SerializedNodeHeader { bytes: d.read_array() };

            let index = node_header.index();

            // If the length of this node's edge list is small, the length is stored in the header.
            // If it is not, we fall back to another decoder call.
            let num_edges = node_header.len().unwrap_or_else(|| d.read_u32());

            // The edges index list uses the same varint strategy as rmeta tables; we select the
            // number of byte elements per-array not per-element. This lets us read the whole edge
            // list for a node with one decoder call and also use the on-disk format in memory.
            let edges_len_bytes = node_header.bytes_per_index() * (num_edges as usize);

            // A dead record: the node was dropped by an earlier session but its bytes were
            // carried along in the region. Skip it; its slot stays `Null`.
            if dead_set.contains(index) {
                d.read_raw_bytes(edges_len_bytes);
                continue;
            }

            let node = &mut nodes[index];
            let new_node = node_header.node();
            assert!(new_node.kind != DepKind::Null);
            if node.kind != DepKind::Null {
                // A later record overrides an earlier one at the same index: the node was
                // re-executed by the session that appended it, keeping its index. The key
                // cannot change, only the value fingerprint and the edges. The exception
                // is the anon-zero-deps singleton, whose key is seeded per session.
                debug_assert!(*node == new_node || new_node.kind == DepKind::AnonZeroDeps);
            }
            *node = new_node;

            value_fingerprints[index] = node_header.value_fingerprint();

            // The in-memory structure for the edges list stores the byte width of the edges on
            // this node with the offset into the global edge data array. On an override the
            // earlier record's edge bytes are simply orphaned in `edge_list_data`.
            let edges_header = node_header.edges_header(&edge_list_data, num_edges);

            edge_list_data.extend(d.read_raw_bytes(edges_len_bytes));

            edge_list_indices[index] = edges_header;
        }

        // When we access the edge list data, we do a fixed-size read from the edge list data then
        // mask off the bytes that aren't for that edge index, so the last read may dangle off the
        // end of the array. This padding ensure it doesn't.
        edge_list_data.extend(&[0u8; DEP_NODE_PAD]);

        // Lay out the per-kind live counts (read from the footer above) as contiguous
        // ranges for the counting sort of `LazyNodeIndex`.
        let mut kinds = Vec::with_capacity(DepKind::MAX as usize + 1);
        let mut offset = 0u32;
        for &len in &kind_stats {
            kinds.push(LazyKindIndex { start: offset, len, map: OnceLock::new() });
            offset += len;
        }
        debug_assert_eq!(offset as usize, node_count);

        // Counting sort: place each node index into its kind's range. `fill[k]`
        // points at the next free slot in kind `k`'s range, so a kind's nodes end
        // up contiguous. Slots start as `None` and are each filled exactly once
        // (the counts sum to the number of non-`Null` nodes).
        let mut nodes_by_kind = vec![None; node_count];
        let mut fill: Vec<u32> = kinds.iter().map(|k| k.start).collect();
        for (idx, node) in nodes.iter_enumerated() {
            // Unused indices from batch allocation stay `Null`; they carry no
            // encoded node and are never looked up by fingerprint, so skip them.
            if node.kind == DepKind::Null {
                continue;
            }
            let k = node.kind.as_usize();
            nodes_by_kind[fill[k] as usize] = Some(idx);
            fill[k] += 1;
        }
        // Each kind's range was filled exactly to its end.
        debug_assert!(kinds.iter().zip(&fill).all(|(k, &f)| f == k.start + k.len));
        let reverse_index = LazyNodeIndex { nodes_by_kind, kinds };

        Arc::new(SerializedDepGraph {
            nodes,
            value_fingerprints,
            edge_list_indices,
            edge_list_data,
            reverse_index,
            session_count,
            generation,
            // The retained file bytes are attached by the caller via `attach_mmap`.
            mmap: None,
            records_range: records_start..dead_pos,
            dead,
            kind_stats,
            live_node_count: node_count as u64,
            live_edge_count: edge_count as u64,
            profiler: Some(profiler.clone()),
        })
    }
}

/// A packed representation of all the fixed-size fields in a `NodeInfo`.
///
/// This stores in one byte array:
/// * The value `Fingerprint` in the `NodeInfo`
/// * The key `Fingerprint` in `DepNode` that is in this `NodeInfo`
/// * The `DepKind`'s discriminant (a u16, but not all bits are used...)
/// * The byte width of the encoded edges for this node
/// * In whatever bits remain, the length of the edge list for this node, if it fits
struct SerializedNodeHeader {
    // 2 bytes for the DepNode
    // 4 bytes for the index
    // 16 for Fingerprint in DepNode
    // 16 for Fingerprint in NodeInfo
    bytes: [u8; 38],
}

// The fields of a `SerializedNodeHeader`, this struct is an implementation detail and exists only
// to make the implementation of `SerializedNodeHeader` simpler.
struct Unpacked {
    len: Option<u32>,
    bytes_per_index: usize,
    kind: DepKind,
    index: SerializedDepNodeIndex,
    key_fingerprint: PackedFingerprint,
    value_fingerprint: Fingerprint,
}

// Bit fields, where
// M: bits used to store the length of a node's edge list
// N: bits used to store the byte width of elements of the edge list
// are
// 0..M    length of the edge
// M..M+N  bytes per index
// M+N..16 kind
impl SerializedNodeHeader {
    const TOTAL_BITS: usize = size_of::<DepKind>() * 8;
    const LEN_BITS: usize = Self::TOTAL_BITS - Self::KIND_BITS - Self::WIDTH_BITS;
    const WIDTH_BITS: usize = DEP_NODE_WIDTH_BITS;
    const KIND_BITS: usize = Self::TOTAL_BITS - DepKind::MAX.leading_zeros() as usize;
    const MAX_INLINE_LEN: usize = (u16::MAX as usize >> (Self::TOTAL_BITS - Self::LEN_BITS)) - 1;

    #[inline]
    fn new(
        node: &DepNode,
        index: DepNodeIndex,
        value_fingerprint: Fingerprint,
        edge_max_index: u32,
        edge_count: usize,
    ) -> Self {
        debug_assert_eq!(Self::TOTAL_BITS, Self::LEN_BITS + Self::WIDTH_BITS + Self::KIND_BITS);

        let mut head = node.kind.as_u16();

        let free_bytes = edge_max_index.leading_zeros() as usize / 8;
        let bytes_per_index = (DEP_NODE_SIZE - free_bytes).saturating_sub(1);
        head |= (bytes_per_index as u16) << Self::KIND_BITS;

        // Encode number of edges + 1 so that we can reserve 0 to indicate that the len doesn't fit
        // in this bitfield.
        if edge_count <= Self::MAX_INLINE_LEN {
            head |= (edge_count as u16 + 1) << (Self::KIND_BITS + Self::WIDTH_BITS);
        }

        let hash: Fingerprint = node.key_fingerprint.into();

        // Using half-open ranges ensures an unconditional panic if we get the magic numbers wrong.
        let mut bytes = [0u8; 38];
        bytes[..2].copy_from_slice(&head.to_le_bytes());
        bytes[2..6].copy_from_slice(&index.as_u32().to_le_bytes());
        bytes[6..22].copy_from_slice(&hash.to_le_bytes());
        bytes[22..].copy_from_slice(&value_fingerprint.to_le_bytes());

        #[cfg(debug_assertions)]
        {
            let res = Self { bytes };
            assert_eq!(value_fingerprint, res.value_fingerprint());
            assert_eq!(*node, res.node());
            if let Some(len) = res.len() {
                assert_eq!(edge_count, len as usize);
            }
        }
        Self { bytes }
    }

    #[inline]
    fn unpack(&self) -> Unpacked {
        let head = u16::from_le_bytes(self.bytes[..2].try_into().unwrap());
        let index = u32::from_le_bytes(self.bytes[2..6].try_into().unwrap());
        let key_fingerprint = self.bytes[6..22].try_into().unwrap();
        let value_fingerprint = self.bytes[22..].try_into().unwrap();

        let kind = head & mask(Self::KIND_BITS) as u16;
        let bytes_per_index = (head >> Self::KIND_BITS) & mask(Self::WIDTH_BITS) as u16;
        let len = (head as u32) >> (Self::WIDTH_BITS + Self::KIND_BITS);

        Unpacked {
            len: len.checked_sub(1),
            bytes_per_index: bytes_per_index as usize + 1,
            kind: DepKind::from_u16(kind),
            index: SerializedDepNodeIndex::from_u32(index),
            key_fingerprint: Fingerprint::from_le_bytes(key_fingerprint).into(),
            value_fingerprint: Fingerprint::from_le_bytes(value_fingerprint),
        }
    }

    #[inline]
    fn len(&self) -> Option<u32> {
        self.unpack().len
    }

    #[inline]
    fn bytes_per_index(&self) -> usize {
        self.unpack().bytes_per_index
    }

    #[inline]
    fn index(&self) -> SerializedDepNodeIndex {
        self.unpack().index
    }

    #[inline]
    fn value_fingerprint(&self) -> Fingerprint {
        self.unpack().value_fingerprint
    }

    #[inline]
    fn node(&self) -> DepNode {
        let Unpacked { kind, key_fingerprint, .. } = self.unpack();
        DepNode { kind, key_fingerprint }
    }

    #[inline]
    fn edges_header(&self, edge_list_data: &[u8], num_edges: u32) -> EdgeHeader {
        EdgeHeader {
            repr: (edge_list_data.len() << DEP_NODE_WIDTH_BITS) | (self.bytes_per_index() - 1),
            num_edges,
        }
    }
}

#[derive(Debug)]
struct NodeInfo {
    node: DepNode,
    value_fingerprint: Fingerprint,
    edges: EdgesVec,
}

impl NodeInfo {
    fn encode(&self, e: &mut MemEncoder, index: DepNodeIndex) {
        let NodeInfo { ref node, value_fingerprint, ref edges } = *self;
        let header = SerializedNodeHeader::new(
            node,
            index,
            value_fingerprint,
            edges.max_index(),
            edges.len(),
        );
        e.write_array(header.bytes);

        if header.len().is_none() {
            // The edges are all unique and the number of unique indices is less than u32::MAX.
            e.emit_u32(edges.len().try_into().unwrap());
        }

        let bytes_per_index = header.bytes_per_index();
        for node_index in edges.iter() {
            e.write_with(|dest| {
                *dest = node_index.as_u32().to_le_bytes();
                bytes_per_index
            });
        }
    }

    /// Encode a node that was promoted from the previous graph. It reads the edges directly from
    /// the previous dep graph and expects all edges to already have a new dep node index assigned.
    /// This avoids the overhead of constructing `EdgesVec`, which would be needed to call `encode`.
    #[inline]
    fn encode_promoted(
        e: &mut MemEncoder,
        node: &DepNode,
        index: DepNodeIndex,
        value_fingerprint: Fingerprint,
        edges: &[DepNodeIndex],
    ) -> usize {
        let edge_count = edges.len();

        // Find the highest edge in the new dep node indices
        let edge_max = edges.iter().map(|x| x.as_u32()).max().unwrap_or(0);

        let header =
            SerializedNodeHeader::new(node, index, value_fingerprint, edge_max, edge_count);
        e.write_array(header.bytes);

        if header.len().is_none() {
            // The edges are all unique and the number of unique indices is less than u32::MAX.
            e.emit_u32(edge_count.try_into().unwrap());
        }

        let bytes_per_index = header.bytes_per_index();
        for edge in edges {
            let edge = edge.as_u32();
            e.write_with(|dest| {
                *dest = edge.to_le_bytes();
                bytes_per_index
            });
        }

        edge_count
    }
}

struct Stat {
    kind: DepKind,
    node_counter: u64,
    edge_counter: u64,
}

struct LocalEncoderState {
    next_node_index: u32,
    remaining_node_index: u32,
    encoder: MemEncoder,
    /// Net change to the live node count from this worker's appends. An appended
    /// record that overrides a carried one nets zero (the node was already counted
    /// by the previous footer), so only genuinely new nodes contribute.
    node_count: i64,
    /// Net change to the live edge count from this worker's appends. An override
    /// contributes the difference between its new and old edge counts.
    edge_count: i64,
    /// Indices below `first_new_index` this worker appended records for. Those appends
    /// override the carried record at the same index; anything occupied, not overridden
    /// and not marked green by the end of the session is dead.
    overridden: Vec<SerializedDepNodeIndex>,

    /// Stores the net change to the number of live nodes of each dep kind.
    /// An override nets zero here since the key (and thus the kind) cannot change.
    kind_stats: Vec<u32>,
}

struct LocalEncoderResult {
    node_max: u32,
    node_count: i64,
    edge_count: i64,
    overridden: Vec<SerializedDepNodeIndex>,

    /// Stores the net change to the number of live nodes of each dep kind.
    kind_stats: Vec<u32>,
}

struct EncoderState {
    next_node_index: AtomicU64,
    previous: Arc<SerializedDepGraph>,
    file: Lock<Option<FileEncoder<'static>>>,
    local: WorkerLocal<RefCell<LocalEncoderState>>,
    stats: Option<Lock<FxHashMap<DepKind, Stat>>>,
    /// The first dep node index handed out to genuinely new nodes this session. Nodes
    /// that existed in the previous graph keep their old indices, which all lie below
    /// this value, so new nodes never collide with them.
    first_new_index: u32,
    /// Whether this session carries the previous record region forward: the region was
    /// copied into the new file wholesale at construction, promoted green nodes write
    /// nothing, and re-executed nodes append records that override the carried ones.
    /// When false (first session, compaction, or a debugging feature retains the full
    /// graph), every live record is written out fresh.
    carrying: bool,
}

impl EncoderState {
    fn new(
        encoder: FileEncoder<'static>,
        record_stats: bool,
        previous: Arc<SerializedDepGraph>,
        carrying: bool,
    ) -> Self {
        // Indices 0 and 1 are always the two singleton nodes; carried indices fill the
        // rest of the previous index space. New nodes start above all of them.
        let first_new_index = std::cmp::max(2, previous.node_count() as u32);
        let mut encoder = encoder;
        if carrying {
            // Copy the previous record region into the new file wholesale, before any
            // appended record. Every node that survives this session keeps its index, so
            // the region stays valid: promoted green records are byte-for-byte what a
            // fresh encode would produce, re-executed nodes append overriding records at
            // their old index, and dropped nodes are tombstoned via the dead list.
            encoder.emit_raw_bytes(previous.region_bytes());
        }
        Self {
            previous,
            next_node_index: AtomicU64::new(first_new_index as u64),
            first_new_index,
            carrying,
            stats: record_stats.then(|| Lock::new(FxHashMap::default())),
            file: Lock::new(Some(encoder)),
            local: WorkerLocal::new(|_| {
                RefCell::new(LocalEncoderState {
                    next_node_index: 0,
                    remaining_node_index: 0,
                    edge_count: 0,
                    node_count: 0,
                    overridden: Vec::new(),
                    encoder: MemEncoder::new(),
                    kind_stats: iter::repeat_n(0, DepKind::MAX as usize + 1).collect(),
                })
            }),
        }
    }

    #[inline]
    fn next_index(&self, local: &mut LocalEncoderState) -> DepNodeIndex {
        if local.remaining_node_index == 0 {
            const COUNT: u32 = 256;

            // We assume that there won't be enough active threads to overflow `u64` from `u32::MAX` here.
            // This can exceed u32::MAX by at most `N` * `COUNT` where `N` is the thread pool count since
            // `try_into().unwrap()` will make threads panic when `self.next_node_index` exceeds u32::MAX.
            local.next_node_index =
                self.next_node_index.fetch_add(COUNT as u64, Ordering::Relaxed).try_into().unwrap();

            // Check that we'll stay within `u32`
            local.next_node_index.checked_add(COUNT).unwrap();

            local.remaining_node_index = COUNT;
        }

        DepNodeIndex::from_u32(local.next_node_index)
    }

    /// Marks the index previously returned by `next_index` as used. Nodes that existed
    /// in the previous graph keep their old index and don't go through here.
    #[inline]
    fn advance_index(&self, local: &mut LocalEncoderState) {
        local.remaining_node_index -= 1;
        local.next_node_index += 1;
    }

    /// Counts one written record. Appends that override a carried record are
    /// compensated afterwards by [`Self::record_override`].
    #[inline]
    fn count_node(&self, local: &mut LocalEncoderState) {
        local.node_count += 1;
    }

    #[inline]
    fn record(
        &self,
        node: &DepNode,
        index: DepNodeIndex,
        edge_count: usize,
        edges: &[DepNodeIndex],
        retained_graph: &Option<Lock<RetainedDepGraph>>,
        local: &mut LocalEncoderState,
    ) {
        local.kind_stats[node.kind.as_usize()] += 1;
        local.edge_count += edge_count as i64;

        if let Some(retained_graph) = &retained_graph {
            // Outline the build of the full dep graph as it's typically disabled and cold.
            outline(move || {
                // Block on the lock rather than using `try_lock`: under the parallel frontend
                // several threads record nodes concurrently, and dropping a node on lock
                // contention would make the retained graph nondeterministic. Readers take a
                // clone of the graph (`retained_dep_graph`) rather than holding the lock, so
                // this never deadlocks against a reentrant `record`.
                retained_graph.lock().push(index, *node, edges);
            });
        }

        if let Some(stats) = &self.stats {
            let kind = node.kind;

            // Outline the stats code as it's typically disabled and cold.
            outline(move || {
                let mut stats = stats.lock();
                let stat =
                    stats.entry(kind).or_insert(Stat { kind, node_counter: 0, edge_counter: 0 });
                stat.node_counter += 1;
                stat.edge_counter += edge_count as u64;
            });
        }
    }

    #[inline]
    fn flush_mem_encoder(&self, local: &mut LocalEncoderState) {
        let data = &mut local.encoder.data;
        if data.len() > 64 * 1024 {
            self.file.lock().as_mut().unwrap().emit_raw_bytes(&data[..]);
            data.clear();
        }
    }

    /// Encodes a node to the current graph.
    fn encode_node(
        &self,
        index: DepNodeIndex,
        node: &NodeInfo,
        retained_graph: &Option<Lock<RetainedDepGraph>>,
        local: &mut LocalEncoderState,
    ) {
        node.encode(&mut local.encoder, index);
        self.flush_mem_encoder(&mut *local);
        self.count_node(&mut *local);
        self.record(&node.node, index, node.edges.len(), &node.edges, retained_graph, &mut *local);
    }

    /// Encodes a node that was promoted from the previous graph. It reads the information directly from
    /// the previous dep graph for performance reasons.
    ///
    /// This differs from `encode_node` where you have to explicitly provide the relevant `NodeInfo`.
    ///
    /// It expects all edges to already have a new dep node index assigned.
    #[inline]
    fn encode_promoted_node(
        &self,
        index: DepNodeIndex,
        prev_index: SerializedDepNodeIndex,
        retained_graph: &Option<Lock<RetainedDepGraph>>,
        local: &mut LocalEncoderState,
        edges: &[DepNodeIndex],
    ) {
        let node = self.previous.index_to_node(prev_index);
        let value_fingerprint = self.previous.value_fingerprint_for_index(prev_index);
        let edge_count =
            NodeInfo::encode_promoted(&mut local.encoder, node, index, value_fingerprint, edges);
        self.flush_mem_encoder(&mut *local);
        self.count_node(&mut *local);
        self.record(node, index, edge_count, edges, retained_graph, &mut *local);
    }

    /// Adjusts a worker's bookkeeping after it appended a record that overrides the
    /// carried record at `prev_index`. The node was already counted by the previous
    /// footer, so the append nets zero nodes (and zero for its kind, since the key
    /// cannot change) and only the change in edge count remains.
    #[inline]
    fn record_override(
        &self,
        prev_index: SerializedDepNodeIndex,
        kind: DepKind,
        local: &mut LocalEncoderState,
    ) {
        debug_assert!(self.carrying);
        local.node_count -= 1;
        local.kind_stats[kind.as_usize()] -= 1;
        local.edge_count -= self.previous.edge_count_for_index(prev_index) as i64;
        local.overridden.push(prev_index);
    }

    fn finish(
        &self,
        profiler: &SelfProfilerRef,
        current: &CurrentDepGraph,
        colors: &DepNodeColorMap,
    ) -> FileEncodeResult {
        // Prevent more indices from being allocated.
        self.next_node_index.store(u32::MAX as u64 + 1, Ordering::SeqCst);

        let results = broadcast(|_| {
            let mut local = self.local.borrow_mut();

            // Prevent more indices from being allocated on this thread.
            local.remaining_node_index = 0;

            let data = mem::take(&mut local.encoder.data);
            self.file.lock().as_mut().unwrap().emit_raw_bytes(&data);

            LocalEncoderResult {
                kind_stats: local.kind_stats.clone(),
                node_max: local.next_node_index,
                node_count: local.node_count,
                edge_count: local.edge_count,
                overridden: mem::take(&mut local.overridden),
            }
        });

        let mut encoder = self.file.lock().take().unwrap();

        // Every count starts from the previous footer when carrying (the region already
        // holds those nodes) and from zero when writing a fresh file; the workers report
        // net changes in either case.
        let (mut kind_stats, mut node_count, mut edge_count) = if self.carrying {
            (
                self.previous.kind_stats.clone(),
                self.previous.live_node_count as i64,
                self.previous.live_edge_count as i64,
            )
        } else {
            (iter::repeat_n(0, DepKind::MAX as usize + 1).collect(), 0, 0)
        };

        let mut node_max = 0;
        let mut overridden = DenseBitSet::new_empty(self.first_new_index as usize);

        for result in results {
            node_max = max(node_max, result.node_max);
            node_count += result.node_count;
            edge_count += result.edge_count;
            for (i, stat) in result.kind_stats.iter().enumerate() {
                // The per-worker values are net changes: an override decrements the kind
                // it previously incremented, so the sum stays balanced per worker and the
                // wrapping cancels out across the base value taken from the footer.
                kind_stats[i] = kind_stats[i].wrapping_add(*stat);
            }
            for index in result.overridden {
                overridden.insert(index);
            }
        }

        // Nodes that existed in the previous graph keep their previous indices (all below
        // `first_new_index`) but don't advance any worker's `next_node_index`. If few or no
        // new nodes were encoded, the per-worker maxima can therefore understate the real
        // index space, so raise the floor to cover every carried index.
        node_max = max(node_max, self.first_new_index);

        // When carrying, tombstone every record in the region that this session dropped: a
        // node neither marked green (record still valid) nor overridden by an appended
        // record. This matches what a fresh write drops by simply not writing it. Dead
        // indices from earlier generations decode as unoccupied slots, so they are carried
        // into the new list explicitly.
        let mut dead: Vec<SerializedDepNodeIndex> = Vec::new();
        if self.carrying {
            dead.extend_from_slice(&self.previous.dead);
            for index in (0..self.previous.node_count() as u32).map(SerializedDepNodeIndex::from_u32)
            {
                if self.previous.index_is_occupied(index)
                    && !colors.is_green(index)
                    && !overridden.contains(index)
                {
                    dead.push(index);
                    let kind = self.previous.index_to_node(index).kind;
                    kind_stats[kind.as_usize()] -= 1;
                    node_count -= 1;
                    edge_count -= self.previous.edge_count_for_index(index) as i64;
                }
            }
        }

        let generation = if self.carrying { self.previous.generation + 1 } else { 0 };

        // The record region ends where the dead list begins.
        let dead_pos = encoder.position();
        encoder.emit_u64(dead.len() as u64);
        for index in &dead {
            encoder.write_array(index.as_u32().to_le_bytes());
        }

        // Encode the number of live nodes of each dep kind.
        for count in kind_stats.iter() {
            count.encode(&mut encoder);
        }

        self.previous.session_count.checked_add(1).unwrap().encode(&mut encoder);
        generation.encode(&mut encoder);

        debug!(?node_max, ?node_count, ?edge_count);
        debug!("position: {:?}", encoder.position());
        IntEncodedWithFixedSize(dead_pos.try_into().unwrap()).encode(&mut encoder);
        IntEncodedWithFixedSize(node_max.try_into().unwrap()).encode(&mut encoder);
        IntEncodedWithFixedSize(node_count.try_into().unwrap()).encode(&mut encoder);
        IntEncodedWithFixedSize(edge_count.try_into().unwrap()).encode(&mut encoder);
        debug!("position: {:?}", encoder.position());
        // Drop the encoder so that nothing is written after the counts.
        let result = encoder.finish();
        if let Ok(position) = result {
            // FIXME(rylev): we hardcode the dep graph file name so we
            // don't need a dependency on rustc_incremental just for that.
            profiler.artifact_size("dep_graph", "dep-graph.bin", position as u64);
        }

        self.print_incremental_info(current, node_count as usize, edge_count as usize);

        result
    }

    fn print_incremental_info(
        &self,
        current: &CurrentDepGraph,
        total_node_count: usize,
        total_edge_count: usize,
    ) {
        if let Some(record_stats) = &self.stats {
            let record_stats = record_stats.lock();
            // `stats` is sorted below so we can allow this lint here.
            #[allow(rustc::potential_query_instability)]
            let mut stats: Vec<_> = record_stats.values().collect();
            stats.sort_by_key(|s| -(s.node_counter as i64));

            const SEPARATOR: &str = "[incremental] --------------------------------\
                                     ----------------------------------------------\
                                     ------------";

            eprintln!("[incremental]");
            eprintln!("[incremental] DepGraph Statistics");
            eprintln!("{SEPARATOR}");
            eprintln!("[incremental]");
            eprintln!("[incremental] Total Node Count: {}", total_node_count);
            eprintln!("[incremental] Total Edge Count: {}", total_edge_count);

            if cfg!(debug_assertions) {
                let total_read_count = current.total_read_count.load(Ordering::Relaxed);
                let total_duplicate_read_count =
                    current.total_duplicate_read_count.load(Ordering::Relaxed);
                eprintln!("[incremental] Total Edge Reads: {total_read_count}");
                eprintln!("[incremental] Total Duplicate Edge Reads: {total_duplicate_read_count}");
            }

            eprintln!("[incremental]");
            eprintln!(
                "[incremental]  {:<36}| {:<17}| {:<12}| {:<17}|",
                "Node Kind", "Node Frequency", "Node Count", "Avg. Edge Count"
            );
            eprintln!("{SEPARATOR}");

            for stat in stats {
                let node_kind_ratio =
                    (100.0 * (stat.node_counter as f64)) / (total_node_count as f64);
                let node_kind_avg_edges = (stat.edge_counter as f64) / (stat.node_counter as f64);

                eprintln!(
                    "[incremental]  {:<36}|{:>16.1}% |{:>12} |{:>17.1} |",
                    format!("{:?}", stat.kind),
                    node_kind_ratio,
                    stat.node_counter,
                    node_kind_avg_edges,
                );
            }

            eprintln!("{SEPARATOR}");
            eprintln!("[incremental]");
        }
    }
}

pub(crate) struct GraphEncoder {
    profiler: SelfProfilerRef,
    status: EncoderState,
    /// In-memory copy of the dep graph; only present if `-Zquery-dep-graph` is set.
    retained_graph: Option<Lock<RetainedDepGraph>>,
}

/// After this many consecutive carried generations, write a fresh file instead. Each
/// carried generation leaves behind dead records, superseded records and their orphaned
/// index slots; a compacting rewrite reclaims all of it.
const MAX_CARRIED_GENERATIONS: u64 = 8;

impl GraphEncoder {
    pub(crate) fn new(
        sess: &Session,
        encoder: FileEncoder<'static>,
        prev_node_count: usize,
        previous: Arc<SerializedDepGraph>,
    ) -> Self {
        let retained_graph = sess
            .opts
            .unstable_opts
            .query_dep_graph
            .then(|| Lock::new(RetainedDepGraph::new(prev_node_count)));
        let record_stats = sess.opts.unstable_opts.incremental_info;
        // Carry the previous record region forward unless there is no previous file, a
        // debugging feature needs every node to pass through the encoder (the retained
        // graph and the stats both do), or enough generations accumulated that dead
        // records should be compacted away.
        let carrying = previous.can_carry()
            && retained_graph.is_none()
            && !record_stats
            && previous.generation + 1 < MAX_CARRIED_GENERATIONS;
        let status = EncoderState::new(encoder, record_stats, previous, carrying);
        GraphEncoder { status, retained_graph, profiler: sess.prof.clone() }
    }

    pub(crate) fn retained_dep_graph(&self) -> Option<RetainedDepGraph> {
        self.retained_graph.as_ref().map(|retained_graph| retained_graph.lock().clone())
    }

    /// Whether this session carries the previous record region forward. When true,
    /// promoting a node from the previous graph needs no edge list, so the marking
    /// walk can skip collecting edge indices entirely.
    #[inline]
    pub(crate) fn is_carrying(&self) -> bool {
        self.status.carrying
    }

    /// Marks a node promoted from the previous graph green without materializing its
    /// edges. Only valid when carrying: the node's record is already in the new file.
    ///
    /// Returns Some if the node is now green, or None if it had already been
    /// concurrently marked red.
    #[inline]
    pub(crate) fn send_promoted_carried(
        &self,
        prev_index: SerializedDepNodeIndex,
        colors: &DepNodeColorMap,
    ) -> Option<DepNodeIndex> {
        debug_assert!(self.status.carrying);
        let index = DepNodeIndex::from_u32(prev_index.as_u32());
        match colors.try_set_color(prev_index, DesiredColor::Green { index }) {
            TrySetColorResult::Success => Some(index),
            TrySetColorResult::AlreadyRed => None,
            TrySetColorResult::AlreadyGreen { index } => Some(index),
        }
    }

    /// Encodes a node that does not exists in the previous graph.
    pub(crate) fn send_new(
        &self,
        node: DepNode,
        value_fingerprint: Fingerprint,
        edges: EdgesVec,
    ) -> DepNodeIndex {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph");
        let node = NodeInfo { node, value_fingerprint, edges };
        let mut local = self.status.local.borrow_mut();
        let index = self.status.next_index(&mut *local);
        self.status.advance_index(&mut *local);
        self.status.encode_node(index, &node, &self.retained_graph, &mut *local);
        index
    }

    /// Encodes a node at a fixed, caller-chosen index rather than the next allocated one.
    /// Used only for the two singleton nodes, which must live at indices 0 and 1; those
    /// slots are reserved below `first_new_index`, so this cannot collide with a freshly
    /// allocated node. When carrying, the record appended here overrides the previous
    /// session's singleton record carried along in the region.
    pub(crate) fn send_new_at(
        &self,
        index: DepNodeIndex,
        node: DepNode,
        value_fingerprint: Fingerprint,
        edges: EdgesVec,
    ) -> DepNodeIndex {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph");
        let kind = node.kind;
        let node = NodeInfo { node, value_fingerprint, edges };
        let mut local = self.status.local.borrow_mut();
        self.status.encode_node(index, &node, &self.retained_graph, &mut *local);
        if self.status.carrying {
            let prev_index = SerializedDepNodeIndex::from_u32(index.as_u32());
            self.status.record_override(prev_index, kind, &mut *local);
        }
        index
    }

    /// Encodes a node that exists in the previous graph, but was re-executed.
    ///
    /// This will also ensure the dep node is colored either red or green.
    pub(crate) fn send_and_color(
        &self,
        prev_index: SerializedDepNodeIndex,
        colors: &DepNodeColorMap,
        node: DepNode,
        value_fingerprint: Fingerprint,
        edges: EdgesVec,
        is_green: bool,
    ) -> DepNodeIndex {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph");
        let kind = node.kind;
        let node = NodeInfo { node, value_fingerprint, edges };

        let mut local = self.status.local.borrow_mut();

        // A re-executed node keeps its previous index whether it came out green or red.
        // Keeping green indices stable lets records of promoted nodes, which refer to
        // their deps by index, stay valid as-is; keeping red indices stable too means
        // the appended record simply overrides the carried one, and this session's
        // edges (which may point at the red node) need no separate index space.
        let index = DepNodeIndex::from_u32(prev_index.as_u32());
        let color = if is_green { DesiredColor::Green { index } } else { DesiredColor::Red };

        // Use `try_set_color` to avoid racing when `send_promoted` is called concurrently
        // on the same index.
        match colors.try_set_color(prev_index, color) {
            TrySetColorResult::Success => {}
            TrySetColorResult::AlreadyRed => panic!("dep node {prev_index:?} is unexpectedly red"),
            TrySetColorResult::AlreadyGreen { index } => return index,
        }

        self.status.encode_node(index, &node, &self.retained_graph, &mut *local);
        if self.status.carrying {
            self.status.record_override(prev_index, kind, &mut *local);
        }
        index
    }

    /// Marks a node that was promoted from the previous graph green. It expects all edges
    /// to already have a new dep node index assigned.
    ///
    /// Tries to mark the dep node green, and returns Some if it is now green,
    /// or None if had already been concurrently marked red.
    ///
    /// A promoted node keeps its previous index; its edges (all green, all likewise kept
    /// at their previous indices) are exactly the previous edges, so its previous record
    /// remains valid. When the record region is carried forward, that record is already
    /// in the new file and marking the node green is all there is to do; otherwise the
    /// record is re-encoded into the fresh file.
    #[inline]
    pub(crate) fn send_promoted(
        &self,
        prev_index: SerializedDepNodeIndex,
        colors: &DepNodeColorMap,
        edges: &[DepNodeIndex],
    ) -> Option<DepNodeIndex> {
        let index = DepNodeIndex::from_u32(prev_index.as_u32());

        // Use `try_set_color` to avoid racing when `send_promoted` or `send_and_color`
        // is called concurrently on the same index.
        match colors.try_set_color(prev_index, DesiredColor::Green { index }) {
            TrySetColorResult::Success => {
                debug_assert!(
                    edges.iter().map(|e| e.as_u32()).eq(self
                        .status
                        .previous
                        .edge_targets_from(prev_index)
                        .map(|e| e.as_u32())),
                    "promoted green node {prev_index:?} edges diverged from the previous graph",
                );
                if !self.status.carrying {
                    let _prof_timer =
                        self.profiler.generic_activity("incr_comp_encode_dep_graph");
                    let mut local = self.status.local.borrow_mut();
                    self.status.encode_promoted_node(
                        index,
                        prev_index,
                        &self.retained_graph,
                        &mut *local,
                        edges,
                    );
                }
                Some(index)
            }
            TrySetColorResult::AlreadyRed => None,
            TrySetColorResult::AlreadyGreen { index } => Some(index),
        }
    }

    pub(crate) fn finish(
        &self,
        current: &CurrentDepGraph,
        colors: &DepNodeColorMap,
    ) -> FileEncodeResult {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph_finish");

        self.status.finish(&self.profiler, current, colors)
    }
}
