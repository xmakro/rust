use std::fs;

use rustc_data_structures::sync::par_join;
use rustc_middle::dep_graph::{CachePromotionMode, DepGraph, WorkProductMap};
use rustc_middle::query::on_disk_cache;
use rustc_middle::ty::TyCtxt;
use rustc_serialize::Encodable as RustcEncodable;
use rustc_serialize::opaque::FileEncoder;
use rustc_session::Session;
use tracing::debug;

use super::data::*;
use super::fs::*;
use super::{clean, file_format, work_product};
use crate::assert_dep_graph::assert_dep_graph;
use crate::diagnostics;

/// Saves and writes the [`DepGraph`] to the file system.
///
/// This function saves both the dep-graph and the query result cache,
/// and drops the result cache.
///
/// This function should only run after all queries have completed.
/// Trying to execute a query afterwards would attempt to read the result cache we just dropped.
pub(crate) fn save_dep_graph(tcx: TyCtxt<'_>) {
    debug!("save_dep_graph()");
    tcx.dep_graph.with_ignore(|| {
        let sess = tcx.sess;
        if sess.opts.incremental.is_none() {
            return;
        }
        // This is going to be deleted in finalize_session_directory, so let's not create it.
        if sess.dcx().has_errors_or_delayed_bugs().is_some() {
            return;
        }

        let query_cache_path = query_cache_path(sess);
        let staging_query_cache_path = staging_query_cache_path(sess);
        let dep_graph_path = dep_graph_path(sess);
        let staging_dep_graph_path = staging_dep_graph_path(sess);

        sess.time("assert_dep_graph", || assert_dep_graph(tcx));
        sess.time("check_clean", || clean::check_clean_annotations(tcx));

        par_join(
            move || {
                sess.time("incr_comp_persist_dep_graph", || {
                    if let Err(err) = fs::rename(&staging_dep_graph_path, &dep_graph_path) {
                        sess.dcx().emit_err(diagnostics::MoveDepGraph {
                            from: &staging_dep_graph_path,
                            to: &dep_graph_path,
                            err,
                        });
                    }
                });
            },
            move || {
                // We execute this after `incr_comp_persist_dep_graph` for the serial compiler
                // to catch any potential query execution writing to the dep graph.
                sess.time("incr_comp_persist_result_cache", || {
                    // The on-disk cache struct is always present in incremental mode,
                    // even if there was no previous session.
                    let on_disk_cache = tcx.query_system.on_disk_cache.as_ref().unwrap();

                    // When possible, the values of green nodes are carried forward
                    // from the previous cache file byte for byte, so its contents
                    // must stay readable while the new file is written: the mapping
                    // is kept alive until serialization is done. Carrying is
                    // periodically skipped to compact the cache (see
                    // `can_carry_data`).
                    let carry = on_disk_cache.can_carry_data(file_format::header_size(sess));

                    let carried_data = if carry {
                        // Carried values are not decoded, so the verification that
                        // loading performs would not run for them. Preserve its
                        // coverage: verify the same subset of not-loaded values
                        // that promotion (below) would have, without re-encoding.
                        tcx.dep_graph.exec_cache_promotions(tcx, CachePromotionMode::VerifyOnly);

                        on_disk_cache.take_serialized_data_mmap()
                    } else {
                        // For every green dep node that has a disk-cached value from
                        // the previous session, make sure the value is loaded into
                        // the memory cache, so that it will be serialized as part of
                        // this session.
                        //
                        // This reads data from the previous session, so it needs to
                        // happen before dropping the mmap.
                        tcx.dep_graph.exec_cache_promotions(tcx, CachePromotionMode::Promote);

                        // The mapping is not needed anymore.
                        on_disk_cache.close_serialized_data_mmap();
                        None
                    };

                    // The new file is written to a staging path and swapped in
                    // when complete: the old file stays mapped while its data
                    // region is copied into the new one, and on Windows a
                    // mapped file cannot be removed or replaced.
                    file_format::save_in(
                        sess,
                        staging_query_cache_path.clone(),
                        "query cache",
                        |encoder| {
                            tcx.sess.time("incr_comp_serialize_result_cache", || {
                                on_disk_cache::OnDiskCache::serialize(tcx, encoder, carried_data)
                            })
                        },
                    );

                    // `serialize` consumed and dropped the mapping, so the old
                    // file can be replaced now.
                    if let Err(err) = fs::rename(&staging_query_cache_path, &query_cache_path) {
                        sess.dcx().emit_err(diagnostics::MoveQueryCache {
                            from: &staging_query_cache_path,
                            to: &query_cache_path,
                            err,
                        });
                    }
                });
            },
        );
    })
}

/// Saves the work product index.
pub fn save_work_product_index(
    sess: &Session,
    dep_graph: &DepGraph,
    new_work_products: WorkProductMap,
) {
    if sess.opts.incremental.is_none() {
        return;
    }
    // This is going to be deleted in finalize_session_directory, so let's not create it
    if sess.dcx().has_errors().is_some() {
        return;
    }

    debug!("save_work_product_index()");
    dep_graph.assert_ignored();
    let path = work_products_path(sess);
    file_format::save_in(sess, path, "work product index", |mut e| {
        encode_work_product_index(&new_work_products, &mut e);
        e.finish()
    });

    // We also need to clean out old work-products, as not all of them are
    // deleted during invalidation. Some object files don't change their
    // content, they are just not needed anymore.
    let previous_work_products = dep_graph.previous_work_products();
    for (id, wp) in previous_work_products.to_sorted_stable_ord() {
        if !new_work_products.contains_key(id) {
            work_product::delete_workproduct_files(sess, wp);
            debug_assert!(
                !wp.saved_files.items().all(|(_, path)| in_incr_comp_dir_sess(sess, path).exists())
            );
        }
    }

    // Check that we did not delete one of the current work-products:
    debug_assert!({
        new_work_products.items().all(|(_, wp)| {
            wp.saved_files.items().all(|(_, path)| in_incr_comp_dir_sess(sess, path).exists())
        })
    });
}

fn encode_work_product_index(work_products: &WorkProductMap, encoder: &mut FileEncoder<'_>) {
    let serialized_products: Vec<_> = work_products
        .to_sorted_stable_ord()
        .into_iter()
        .map(|(id, work_product)| SerializedWorkProduct {
            id: *id,
            work_product: work_product.clone(),
        })
        .collect();

    serialized_products.encode(encoder)
}
