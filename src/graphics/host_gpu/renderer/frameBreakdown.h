#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/runtimeFlags.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

// KYTY_PERF_STATS=1 frame time breakdown (KYTY_PERF_STATS_VERBOSE=1 adds the heavy sub-bucket lines). Prints one line about once per second:
//   NHL27FRAME fps=... frame=... gpu_thread_busy=... wait_gpu=... readback=... faults=...
//              compile=... draw_prep=... draws=... dispatches=...
//   NHL27DRAWPREP total=... materialize=... textures=... buffers=... descriptors=... pipeline=... other=...
//   NHL27MATERIALIZE calls=... total=... flat=... eval=...(n) indirect=... spec=... rest=... reuse_hit=... reuse_miss=... reuse_fallback=...
//   NHL27BUFFERS vertex_index=... indirect=... find=... bda=... rebind=... rebind_lookup=... rebind_upload=...
//   NHL27PROTECT calls=... ms=...
//   NHL27FAULTCOST n=... gpu_r/gpu_w/guest_r/guest_w=... pages=... refault=... total=... veh=... lock=...
//   NHL27FAULTPARSE passes=... zero=... faults=... ring_wait=...ms
// Every value except fps and frames is a per-frame average (a frame is one presented flip
// interval). Header-only so probes in several translation units share one set of counters
// without touching the build files. Every probe is a no-op unless the gate is set.
namespace Libs::Graphics::FrameBreakdown {

using Clock = std::chrono::steady_clock;

// The gates are read millions of times per second; namespace-scope constants avoid the
// thread-safe-static guard check a function-local static costs on every call.
inline const bool g_enabled = [] {
	const char* value = std::getenv("KYTY_PERF_STATS");
	return value != nullptr && value[0] == '1';
}();

inline bool Enabled() {
	return g_enabled;
}

// KYTY_PERF_STATS_VERBOSE=1 (with KYTY_PERF_STATS=1): the heavy sub-timers and per-slot counters
// (NHL27TEXTURES, NHL27PIPELINE, NHL27BUFFERS2, NHL27DELTA, NHL27DELTAREUSE_SPLIT, NHL27DELTAGEN,
// NHL27SPECULATE). They run thousands of times per frame, so they are off unless asked for; when off the
// hot paths take no timestamps and touch none of those atomics.
inline const bool g_verbose = [] {
	const char* value = std::getenv("KYTY_PERF_STATS_VERBOSE");
	return g_enabled && value != nullptr && value[0] == '1';
}();

inline bool Verbose() {
	return g_verbose;
}

// Fine-grained CPU preparation timing is intentionally opt-in; it can perturb hot paths.
inline const bool g_detail_enabled = [] {
	const char* value = std::getenv("KYTY_PERF_DRAW_DETAIL");
	return g_enabled && value != nullptr && value[0] == '1';
}();

inline bool DetailEnabled() {
	return g_detail_enabled;
}

struct Counters {
	std::atomic<uint64_t> flip_ns {0};          // presented flip -> presented flip
	std::atomic<uint64_t> flip_intervals {0};   // number of those intervals (= frames)
	// Flip intervals rounded to 60 Hz vblanks: [0] = 1 vblank ... [4] = 5 or more.
	std::array<std::atomic<uint64_t>, 5> flip_vblanks {};
	std::atomic<uint64_t> gpu_process_ns {0};   // GPU thread: GuestGpu::Process wall time
	// GPU thread waiting for guest work: total and gap count by length (<1, 1-4, 4-10, 10+ ms).
	std::atomic<uint64_t> idle_ns {0};
	std::array<std::atomic<uint64_t>, 4> idle_gaps {};
	std::atomic<uint64_t> gpu_wait_ns {0};      // GPU thread: blocked in MasterSemaphore::Wait
	std::atomic<uint64_t> gpu_wait_count {0};   // GPU thread: waits that actually blocked
	std::atomic<uint64_t> readback_ns {0};      // synchronous GPU->CPU readbacks, all threads
	std::atomic<uint64_t> readback_count {0};
	std::atomic<uint64_t> fault_ns {0};         // guest page-fault handling on GPU-tracked memory
	std::atomic<uint64_t> fault_count {0};
	std::atomic<uint64_t> compile_ns {0};       // shader permutation compile (SPIR-V + module)
	std::atomic<uint64_t> compile_count {0};
	std::atomic<uint64_t> draw_prep_ns {0};     // PrepareDrawRenderState entry -> vkCmd draw
	std::atomic<uint64_t> draws {0};            // draw calls emitted (indirect = 1 call)
	std::atomic<uint64_t> dispatches {0};       // guest compute dispatches emitted
	// These are inclusive aggregates: graphics program lookup and color-target resolution
	// contain shared graphics work, while descriptor preparation also includes compute paths.
	std::atomic<uint64_t> prep_graphics_programs_ns {0};
	std::atomic<uint64_t> prep_compute_program_ns {0};
	std::atomic<uint64_t> prep_color_target_ns {0};
	std::atomic<uint64_t> prep_find_buffers_ns {0};
	std::atomic<uint64_t> prep_rebind_images_ns {0};
	std::atomic<uint64_t> prep_rebind_buffers_ns {0};
	std::atomic<uint64_t> prep_commit_graphics_ns {0};
	std::atomic<uint64_t> prep_commit_compute_ns {0};
	// NHL27DRAWPREP sub-buckets. Each is a plain Timer (gated only by KYTY_PERF_STATS). They are
	// sequential, non-overlapping scopes inside the draw prep window, so none nests in another.
	// Draws that return before their vkCmd draw still add to these buckets but not to draw_prep_ns,
	// so "other" is clamped at zero.
	std::atomic<uint64_t> dp_materialize_ns {0};  // program lookup + resource materialization
	std::atomic<uint64_t> dp_textures_ns {0};     // image / render-target / sampled-image resolution
	std::atomic<uint64_t> dp_buffers_ns {0};      // buffer lookup, BDA, storage/vertex/index/indirect buffers
	// Sub-buckets of dp_buffers_ns, each timed in addition to it. vertex_index + indirect + find_buffers +
	// bda + rebind equals dp_buffers_ns. rebind_lookup and rebind_upload are subsets of rebind.
	std::atomic<uint64_t> dp_vertex_index_ns {0};  // mesh index, AcquireVertexBuffers, PrepareIndexBuffer
	std::atomic<uint64_t> dp_indirect_ns {0};      // indirect args/count FindBuffer, ObtainBuffer and copies
	std::atomic<uint64_t> dp_find_buffers_ns {0};  // FindBuffers (descriptor -> buffer id)
	std::atomic<uint64_t> dp_bda_ns {0};           // PrepareBda (BDA dirty-range synchronization)
	std::atomic<uint64_t> dp_rebind_ns {0};        // RebindBuffers as a whole
	std::atomic<uint64_t> dp_rebind_lookup_ns {0}; // RebindBuffers: NativeStorageBuffer / ObtainBuffer loop
	std::atomic<uint64_t> dp_rebind_upload_ns {0}; // RebindBuffers: NativeUpload of SRT and shader data
	std::atomic<uint64_t> obtain_cache_hits {0};   // KYTY_BUFFER_FAST: ObtainBuffer result cache hits
	std::atomic<uint64_t> obtain_cache_misses {0}; // KYTY_BUFFER_FAST: lookups that went to ObtainBuffer
	// Why (verbose): another range in the slot, buffers registered since, buffer gone or too small, range
	// CPU-written; and misses not stored: stream buffer, registration during the lookup, still CPU-written.
	std::atomic<uint64_t> obtain_miss_slot {0}, obtain_miss_registration {0}, obtain_miss_buffer {0},
	    obtain_miss_cpu {0}, obtain_nostore_stream {0}, obtain_nostore_registration {0}, obtain_nostore_cpu {0},
	    buffer_registrations {0};
	// RebindBuffers slot memos (verbose): hits, misses by the slot's last range (none, the same range
	// whose check failed, another range in the same buffer, another buffer), by plain/written/texel.
	std::atomic<uint64_t> slot_hit[3] {}, slot_none[3] {}, slot_same[3] {}, slot_inside[3] {}, slot_other[3] {},
	    slot_global_hit[3] {};
	std::atomic<uint64_t> dp_descriptors_ns {0};  // CommitBindings (descriptor writes)
	std::atomic<uint64_t> dp_pipeline_ns {0};     // pipeline lookup, render targets, dynamic state, draw emit
	// NHL27MATERIALIZE: MaterializeResourcesImpl split. total is the whole call; flat, eval, indirect and
	// spec are sequential sub-scopes (eval includes the clean walker); rest = total - the four.
	std::atomic<uint64_t> mat_calls {0};
	std::atomic<uint64_t> mat_total_ns {0};
	std::atomic<uint64_t> mat_flat_ns {0};      // RefreshFlatBuffer
	std::atomic<uint64_t> mat_eval_ns {0};      // EvaluateDescriptor (direct sources)
	std::atomic<uint64_t> mat_indirect_ns {0};  // MaterializeIndirectDescriptor
	std::atomic<uint64_t> mat_spec_ns {0};      // BuildResourceSpecialization
	std::atomic<uint64_t> mat_evals {0};        // EvaluateDescriptor calls
	std::atomic<uint64_t> mat_flat_reads {0};     // RefreshFlatBuffer: guest-memory reader callbacks (words)
	std::atomic<uint64_t> mat_flat_read_ns {0};   // time inside those callbacks (KYTY_PERF_DRAW_DETAIL=1 only)
	std::atomic<uint64_t> mat_flat_ops {0};       // tape ops executed by RefreshFlatBuffer
	std::atomic<uint64_t> mat_flat_alloc_ns {0};  // flat.assign + trace setup (KYTY_PERF_DRAW_DETAIL=1 only)
	std::atomic<uint64_t> mat_flat_trace_hits {0};   // refreshes served by ReplayCfgTrace
	std::atomic<uint64_t> mat_flat_hit_ns {0};     // time in RefreshFlatBuffer calls served by trace replay
	std::atomic<uint64_t> mat_flat_nocfg_ns {0};   // time in RefreshFlatBuffer calls on non-CFG plans
	std::atomic<uint64_t> mat_hit_steps {0};  // trace steps replayed on hits
	std::atomic<uint64_t> mat_hit_reads {0};  // reader callbacks (RawRead execs) on hits
	std::atomic<uint64_t> mat_hit_memo {0};   // memo saves (published) + restores (Memo steps) on hits
	std::atomic<uint64_t> mat_flat_nocfg_ops {0};  // tape ops executed on non-CFG plans
	std::atomic<uint64_t> mat_flat_trace_other {0};  // refreshes that were not (miss/ineligible/non-CFG)
	std::atomic<uint64_t> mat_tr_why[16] {};  // ReplayCfgTrace outcome reasons (see NHL27TRACE print)
	std::atomic<uint64_t> mat_reuse_hits {0};   // KYTY_SRT_SOURCE_REUSE: gather output equals the previous one
	std::atomic<uint64_t> mat_reuse_misses {0}; // gather output differs / first time
	std::atomic<uint64_t> mat_reuse_fallback {0}; // sources not on the gather fast path
	// NHL27DISPATCHPREP buckets: RenderExecutor::DispatchDirect / DispatchIndirect, GPU thread.
	// Sequential, non-overlapping scopes inside dc_total_ns; other = total minus the named buckets.
	std::atomic<uint64_t> dc_total_ns {0};        // whole handler body (after the context lock)
	std::atomic<uint64_t> dc_materialize_ns {0};  // GetComputeProgram (program lookup + materialization)
	std::atomic<uint64_t> dc_textures_ns {0};     // RebindImages
	std::atomic<uint64_t> dc_buffers_ns {0};      // FindBuffers + PrepareBda + shared memory + RebindBuffers
	std::atomic<uint64_t> dc_bda_ns {0};          // PrepareBda (subset of buffers)
	std::atomic<uint64_t> dc_descriptors_ns {0};  // CommitBindings
	std::atomic<uint64_t> dc_pipeline_ns {0};     // GetComputePipeline + PrepareBindings + bindPipeline
	std::atomic<uint64_t> dc_barrier_ns {0};      // write-hazard and access barriers
	std::atomic<uint64_t> dc_record_ns {0};       // breadcrumb + vkCmdDispatch
	std::atomic<uint64_t> dc_handled {0};         // handler invocations
	std::atomic<uint64_t> dc_bind_skipped {0};    // vkCmdBindPipeline calls elided (KYTY_DISPATCH_FAST)
	std::atomic<uint64_t> dc_ahead_attempts {0};
	std::atomic<uint64_t> dc_ahead_ready {0};
	std::atomic<uint64_t> dc_ahead_hits {0};
	std::atomic<uint64_t> dc_ahead_invalid {0};
	std::atomic<uint64_t> dc_ahead_ns {0};
	std::atomic<uint64_t> dc_ahead_validate_ns {0};
	// NHL27FAULTPARSE: BDA fault-parser pass counters (FaultManager::ProcessFaultBuffer).
	std::atomic<uint64_t> fault_parse_passes {0};
	std::atomic<uint64_t> fault_parse_zero {0};   // passes whose count word was 0
	std::atomic<uint64_t> fault_parse_faults {0}; // total fault entries returned by all passes
	std::atomic<uint64_t> fault_ring_wait_ns {0}; // CPU time waiting on a reused download-ring slot
	// PageManager::Impl::Protect (host page-protection changes), all threads.
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> protect_ns {0};
	// NHL27BDA: RenderContext::PrepareBda, GPU command thread. Counts are per call or per walk.
	// Time buckets are sequential inside bda_total_ns; other = total minus lock and sync.
	std::atomic<uint64_t> bda_calls {0};
	std::atomic<uint64_t> bda_skips {0};     // early-outs through the unchanged-generation cache
	std::atomic<uint64_t> bda_scans {0};     // full walks over the mapped ranges
	std::atomic<uint64_t> bda_regions {0};   // dirty regions drained from the memory tracker queue
	std::atomic<uint64_t> bda_ranges {0};    // full walk: mapped ranges; dirty path: drained regions
	std::atomic<uint64_t> bda_pages {0};     // TRACKER_PAGE_SIZE pages covered by those ranges
	std::atomic<uint64_t> bda_total_ns {0};  // whole PrepareBda body, excluding the one-time log
	std::atomic<uint64_t> bda_lock_ns {0};   // m_mapped_ranges_mutex shared-lock acquisition
	std::atomic<uint64_t> bda_sync_ns {0};   // BufferCache sync; the dirty path includes the queue drain
	// NHL27SYNC: BufferCache::SynchronizeBuffer upload events (each ends rendering + barriers).
	std::atomic<uint64_t> sync_uploads {0};
	std::atomic<uint64_t> sync_bytes {0};
	// NHL27PORT: counters for the patch-series ports (KYTY_GUEST_SYNC, KYTY_GUEST_BARRIER_DEDUP,
	// KYTY_DISPATCH_ARGS_GUARD, KYTY_RT_DECODE_MEMO, KYTY_LONG_BUFFER_GC_AGE).
	std::atomic<uint64_t> port_auto_barriers {0};      // per-dispatch hazard/access barriers recorded
	std::atomic<uint64_t> port_auto_barriers_skipped {0}; // ... elided by KYTY_GUEST_SYNC
	std::atomic<uint64_t> port_guest_barriers {0};     // EmitGlobalBarrier barriers recorded
	std::atomic<uint64_t> port_guest_barriers_skipped {0}; // ... elided by KYTY_GUEST_BARRIER_DEDUP
	std::atomic<uint64_t> port_acquire_mem {0};        // ACQUIRE_MEM packets (barrier under GUEST_SYNC)
	std::atomic<uint64_t> port_guard_dispatches {0};   // indirect dispatches routed through the guard
	std::atomic<uint64_t> port_hint_hits {0};          // FindImage resolved by the unique by-address match
	std::atomic<uint64_t> port_hint_misses {0};        // FindImage fell back to the FindImagesInRegion walk
	std::atomic<uint64_t> port_rt_memo_hits {0};       // color-target decodes served by the memo
	std::atomic<uint64_t> port_rt_memo_misses {0};     // decodes computed and inserted
	std::atomic<uint64_t> port_buffer_gc_deletes {0};  // buffers deleted by the buffer collector
	std::atomic<uint64_t> port_buffer_creates {0};     // buffers created (churn indicator)
	// KYTY_READBACK_QUEUE / KYTY_DIRECT_EOP_INTERRUPTS / KYTY_FAULT_COALESCE / KYTY_SPARSE_BUFFER_GC
	std::atomic<uint64_t> port_rbq_used {0};           // downloads served by the second queue
	std::atomic<uint64_t> port_rbq_fallback {0};       // eligible downloads that fell back (queue unavailable/failed)
	std::atomic<uint64_t> port_rbq_wait_ns {0};        // host time blocked on the readback fence
	std::atomic<uint64_t> port_eop_direct {0};         // EOP interrupts delivered at CP time
	std::atomic<uint64_t> port_fault_windows {0};      // write faults that released a coalescing window
	std::atomic<uint64_t> port_fault_pages {0};        // extra tracker pages released by those windows
	std::atomic<uint64_t> port_gc_scans {0};           // buffer GC scans run
	std::atomic<uint64_t> port_gc_sparse_skips {0};    // buffer GC scans skipped by KYTY_SPARSE_BUFFER_GC
};
inline Counters g_counters;

// ---- BEGIN pipeline/buffer sub-bucket probes (NHL27PIPELINE, NHL27BUFFERS2; KYTY_PERF_STATS_VERBOSE) ----
// NHL27PIPELINE: split of dp_pipeline_ns (draw) and dc_pipeline_ns / dc_record_ns (dispatch). Sequential
// scopes inside the pipeline bucket (other = bucket minus the named ones). GPU thread only.
struct PipelineStats {
	// draw
	std::atomic<uint64_t> key_ns {0};        // GetGraphicsPipeline: building the GraphicsPipelineKey
	std::atomic<uint64_t> lookup_ns {0};     // GetGraphicsPipeline: hash + map lookup (or last-key compare)
	std::atomic<uint64_t> targets_ns {0};    // AcquireRenderTargets
	std::atomic<uint64_t> vcommit_ns {0};    // CommitVertexBuffers + CommitIndexBuffer / mesh push constants
	std::atomic<uint64_t> dyn_ns {0};        // SetGraphicsDynamicParams (+ feedback-loop state)
	std::atomic<uint64_t> begin_ns {0};      // BeginRendering
	std::atomic<uint64_t> bind_ns {0};       // vkCmdBindPipeline
	std::atomic<uint64_t> emit_ns {0};       // breadcrumbs + draw emit + frame-dump / debug hooks up to the draw
	std::atomic<uint64_t> lookups {0};       // GetGraphicsPipeline calls
	std::atomic<uint64_t> key_repeats {0};   // key equal to the previous draw's key
	std::atomic<uint64_t> binds {0};         // vkCmdBindPipeline issued
	std::atomic<uint64_t> binds_skipped {0}; // elided (KYTY_PIPELINE_MEMO)
	std::atomic<uint64_t> bind_same {0};     // pipeline equal to the one bound in this recording (elidable)
	std::atomic<uint64_t> dyn_calls {0};     // SetGraphicsDynamicParams calls
	std::atomic<uint64_t> dyn_cmds {0};      // individual dynamic-state vkCmd* issued
	std::atomic<uint64_t> dyn_cmds_skipped {0};
	std::atomic<uint64_t> memo_verify {0};   // sampled full lookups compared against the memo
	std::atomic<uint64_t> memo_mismatch {0}; // ... that differed (must be 0)
	// dispatch
	std::atomic<uint64_t> cs_get_ns {0};     // GetComputePipeline
	std::atomic<uint64_t> cs_prep_ns {0};    // PrepareBindings
	std::atomic<uint64_t> cs_bind_ns {0};    // ShouldBind + vkCmdBindPipeline
	std::atomic<uint64_t> cs_lookups {0};
	std::atomic<uint64_t> cs_repeats {0};    // same compute program as the previous dispatch
	std::atomic<uint64_t> cs_binds {0};
	std::atomic<uint64_t> cs_skipped {0};
};
inline PipelineStats g_pl;

// NHL27BUFFERS2: inside rebind_lookup (RebindBuffers' NativeStorageBuffer / ObtainBufferCached loop) and
// inside the buffer-cache sync path. Gated on BufScope (held by the draw / dispatch RebindBuffers lookup loop
// and by PrepareBda); other BufferCache users are not counted.
struct BufStats {
	// ObtainBuffer (rebind scope)
	std::atomic<uint64_t> ob_calls {0}, ob_written {0}, ob_stream {0}, ob_find_calls {0}, ob_syncs {0};
	std::atomic<uint64_t> ob_clean {0}; // reads of CPU-clean ranges that skipped SynchronizeBuffer
	std::atomic<uint64_t> ob_stream_check_ns {0};  // stream fast-path GPU/CPU-modified walks (tracker)
	std::atomic<uint64_t> ob_stream_copy_ns {0};   // stream Map + memcpy + Commit
	std::atomic<uint64_t> ob_find_ns {0};          // FindBuffer when the supplied id was not usable
	std::atomic<uint64_t> ob_touch_ns {0};         // TouchBuffer (LRU)
	std::atomic<uint64_t> ob_sync_ns {0};          // SynchronizeBuffer (whole)
	std::atomic<uint64_t> ob_book_ns {0};          // DropGpuFills / written bookkeeping
	std::atomic<uint64_t> ob_invalidate_ns {0};    // InvalidateMemoryFromGPU for written buffers (descriptors.cpp)
	// SynchronizeBuffer, by context: [0] = rebind (ObtainBuffer), [1] = PrepareBda
	std::atomic<uint64_t> sb_calls[2] {};
	std::atomic<uint64_t> sb_uploads[2] {};
	std::atomic<uint64_t> sb_bytes[2] {};
	std::atomic<uint64_t> sb_walk_ns[2] {};        // ForEachUploadRange tracker walk, excluding the upload callback
	std::atomic<uint64_t> sb_stage_ns[2] {};       // UploadCopies: staging Map + memcpy + Commit
	std::atomic<uint64_t> sb_end_ns[2] {};         // command.EndRendering()
	std::atomic<uint64_t> sb_rec_ns[2] {};         // barriers + copyBuffer recording
	std::atomic<uint64_t> sb_image_ns[2] {};       // SynchronizeBufferFromImage (texel buffers)
	std::atomic<uint64_t> bda_syncs {0};           // PrepareBda calls that ran a sync (not skipped)
	std::atomic<uint64_t> bda_drain_ns {0};        // sync_ns minus the nested SynchronizeBuffer time
};
inline BufStats g_buf;
inline thread_local int g_buf_scope = 0;  // 1 = rebind lookup, 2 = PrepareBda
struct BufScope {
	int  prev;
	bool on;
	explicit BufScope(int v) noexcept: prev(g_buf_scope), on(Verbose()) { if (on) g_buf_scope = v; }
	~BufScope() { if (on) g_buf_scope = prev; }
	BufScope(const BufScope&)            = delete;
	BufScope& operator=(const BufScope&) = delete;
};

inline bool BufOn() noexcept {
	return g_buf_scope > 0;  // g_buf_scope only moves when Verbose()
}
// ---- END pipeline/buffer sub-bucket probes ----

// NHL27BDABATCH (KYTY_PERF_STATS=1): safety counters for KYTY_BDA_BATCH. GPU command thread only (no atomics).
// A batch is the span between two RenderContext::BdaBatchBoundary calls. "Stale-window" events are PrepareBda
// scans after the first scan of a batch that still uploaded CPU-dirty data; with KYTY_BDA_BATCH on those
// uploads would be deferred to the next batch.
struct BdaBatchStats {
	struct Late {
		uint64_t begin, end;
		uint32_t event;
	};
	// current batch
	bool     active     = false;  // PrepareBda was called
	bool     scanned    = false;  // a real (non generation-skipped) scan already ran in this batch
	bool     collecting = false;  // inside a scan: SynchronizeBuffer reports uploaded runs
	uint32_t b_calls = 0, b_scans = 0, b_events = 0;
	std::vector<std::pair<uint64_t, uint64_t>> used;         // ranges requested through ObtainBuffer
	std::vector<std::pair<uint64_t, uint64_t>> scan_ranges;  // runs uploaded by the scan in progress
	std::vector<Late>                          late;         // runs uploaded by stale-window scans
	// interval totals (reset at each print)
	uint64_t batches = 0, calls = 0, scans = 0, extra_scans = 0, events = 0, ev_used = 0, ev_unused = 0;
	uint64_t stale_bytes = 0, trunc = 0, flag_skips = 0, flips = 0, max_calls = 0, idle_batches = 0;
	uint64_t kind[16] {};
	Clock::time_point last_print {};
};
inline BdaBatchStats g_bda_batch;

inline void BdaBatchNoteUsed(uint64_t vaddr, uint64_t size) noexcept {
	if (!Enabled()) return;
	auto& b = g_bda_batch;
	if (b.used.size() < 16384 && (b.used.empty() || b.used.back() != std::pair<uint64_t, uint64_t> {vaddr, vaddr + size})) {
		b.used.emplace_back(vaddr, vaddr + size);
	}
}

inline void BdaBatchNoteUpload(uint64_t vaddr, uint64_t size) noexcept {
	auto& b = g_bda_batch;
	if (b.collecting && b.scan_ranges.size() < 8192) {
		b.scan_ranges.emplace_back(vaddr, vaddr + size);
	}
}

// NHL27SPECULATE (Stage 0 of parallel draw prep): how much materialization could have run ahead of time.
// Seq counts materializations (draws + dispatches). Page tables are direct-mapped by hashed 4K page;
// aliasing only over-blocks. Everything is gated on Enabled() at the call sites.
struct SpecStats {
	std::atomic<uint64_t> mats {0}, spec8 {0}, spec48 {0}, blk_gpu {0}, blk_cpu8 {0}, blk_cpu48 {0};
	std::atomic<uint64_t> blk_wait8 {0}, blk_wait48 {0}, ranges {0}, noreads {0};
	std::atomic<uint64_t> blk_pw8 {0}, blk_pw48 {0};
};
inline SpecStats g_spec;

// NHL27DELTA (Stage 0 of "delta prep"): how much of each program's materialized output is unchanged since
// that same program's previous use. Keyed by the program's ResourceSnapshot address (one per SourceEntry,
// stable). GPU thread only, measurement only (KYTY_PERF_STATS_VERBOSE).
struct DeltaStats {
	std::atomic<uint64_t> calls {0}, first {0}, cmp_calls {0};      // PrepareBindings calls / first use / compared
	std::atomic<uint64_t> img_slots {0}, img_same {0}, img_reuse {0};
	std::atomic<uint64_t> smp_slots {0}, smp_same {0};
	std::atomic<uint64_t> buf_slots {0}, buf_addr_same {0}, buf_fmt_same {0}, buf_size_same {0};  // used slots only
	std::atomic<uint64_t> call_img_same {0}, call_smp_same {0}, call_shape_same {0}, call_all_same {0};
	std::atomic<uint64_t> rt_draws {0}, rt_repeat {0};
	std::atomic<uint64_t> overhead_ns {0};                         // time spent inside the counter work
	// KYTY_DRAWPREP_DELTA (Stage 1) image/sampler binding reuse, counted per image slot / sampler slot.
	std::atomic<uint64_t> reuse_hits {0};          // image slots that skipped ResolveTexture
	std::atomic<uint64_t> reuse_eligible {0};      // memo valid and T# words + resource unchanged
	std::atomic<uint64_t> reuse_rej_gen {0};       // eligible, rejected only because TextureCache::Generation() moved
	std::atomic<uint64_t> reuse_rej_state {0};     // eligible, generation ok, image unregistered / needs_rebind
	std::atomic<uint64_t> reuse_unmemoable {0};    // resolved image kind that is never memoized (null/depth/DCC/...)
	std::atomic<uint64_t> verify_checks {0};       // reuse hits re-resolved with the full ResolveTexture
	std::atomic<uint64_t> verify_mismatch {0};     // ... whose image id / view info differed (must be 0)
	std::atomic<uint64_t> smp_slots_seen {0};      // sampler slots looked up by PrepareBindings (delta on)
	std::atomic<uint64_t> smp_reuse {0};           // ... that reused the stored vk sampler
	// Same counters split for compute dispatches (the plain ones above cover all stages; draw = total - cs).
	std::atomic<uint64_t> cs_hits {0}, cs_eligible {0}, cs_rej_gen {0}, cs_rej_state {0}, cs_unmemoable {0}, cs_slots {0};
	std::atomic<uint64_t> dr_slots {0};
	// Unmemoable reasons (all stages).
	std::atomic<uint64_t> um_dcc {0}, um_depth {0}, um_other {0};
	// TextureCache::BumpGeneration call sites (a FreeImage counts in free + delete + unregister).
	std::atomic<uint64_t> gen_insert {0}, gen_free {0}, gen_unregister {0}, gen_delete {0}, gen_rebind {0};
	std::atomic<uint64_t> gen_dirty_skipped {0}, gen_untrack_skipped {0};  // no longer bump; counted for reference
	// Generation log: rejected_gen = overlap + overrun; scans = memo hits with a stale generation checked against the log.
	std::atomic<uint64_t> rej_gen_overlap {0}, rej_gen_overrun {0}, gen_log_scans {0};
};
inline DeltaStats g_delta;
struct DeltaProg {
	std::vector<ShaderRecompiler::IR::DescriptorValue> images, samplers, buffers;
	std::vector<uint32_t>                              flattened_srt, user_data;
	std::vector<uint64_t>                              img_id;    // resolved image id (index<<32|generation) per slot
	std::vector<uint8_t>                               img_same;  // this call: slot words unchanged
	bool                                               seen = false;
	bool                                               cmp  = false;  // this call was compared against a previous one
};
inline std::unordered_map<const void*, DeltaProg> g_delta_progs;
inline uint64_t DeltaNowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}
// Compare one materialized snapshot against the same program's previous one. Returns the per-program state.
inline DeltaProg* DeltaCompare(const ShaderRecompiler::IR::ResourceSnapshot& s, size_t used_buffers) {
	if (!Verbose()) return nullptr;
	used_buffers = std::min(used_buffers, s.buffers.size());
	const uint64_t t0 = DeltaNowNs();
	auto&          st = g_delta;
	auto&          p  = g_delta_progs[&s];
	st.calls.fetch_add(1, std::memory_order_relaxed);
	p.cmp = p.seen;
	if (!p.seen) {
		st.first.fetch_add(1, std::memory_order_relaxed);
		p.seen = true;
		p.img_id.assign(s.images.size(), ~0ull);
		p.img_same.assign(s.images.size(), 0);
	} else {
		uint64_t img_same = 0, smp_same = 0, addr_same = 0, fmt_same = 0, size_same = 0;
		const bool img_sz = p.images.size() == s.images.size();
		const bool smp_sz = p.samplers.size() == s.samplers.size();
		const bool buf_sz = p.buffers.size() == s.buffers.size();
		p.img_id.resize(s.images.size(), ~0ull);
		p.img_same.assign(s.images.size(), 0);
		if (img_sz) {
			for (size_t i = 0; i < s.images.size(); ++i) {
				p.img_same[i] = p.images[i] == s.images[i];
				img_same += p.img_same[i];
			}
		}
		if (smp_sz) {
			for (size_t i = 0; i < s.samplers.size(); ++i) smp_same += p.samplers[i] == s.samplers[i];
		}
		if (buf_sz) {
			for (size_t i = 0; i < used_buffers; ++i) {
				const auto& a = p.buffers[i];
				const auto& b = s.buffers[i];
				const bool  addr = a.dword_count == b.dword_count && a.dwords[0] == b.dwords[0] &&
				                  ((a.dwords[1] ^ b.dwords[1]) & 0xffffu) == 0;
				// fmt: stride/swizzle/format bits (everything except the address and num_records).
				bool fmt = a.dword_count == b.dword_count && ((a.dwords[1] ^ b.dwords[1]) & ~0xffffu) == 0;
				for (uint32_t d = 3; d < b.dword_count && d < 8; ++d) fmt &= a.dwords[d] == b.dwords[d];
				const bool size = a.dword_count == b.dword_count && (b.dword_count < 3 || a.dwords[2] == b.dwords[2]);
				addr_same += addr;
				fmt_same += fmt;
				size_same += size;
			}
		}
		const bool all_img = img_sz && img_same == s.images.size();
		const bool all_smp = smp_sz && smp_same == s.samplers.size();
		const bool all_shp = buf_sz && fmt_same == used_buffers && size_same == used_buffers;
		const bool all_adr = buf_sz && addr_same == used_buffers;
		st.cmp_calls.fetch_add(1, std::memory_order_relaxed);
		st.img_slots.fetch_add(s.images.size(), std::memory_order_relaxed);
		st.img_same.fetch_add(img_same, std::memory_order_relaxed);
		st.smp_slots.fetch_add(s.samplers.size(), std::memory_order_relaxed);
		st.smp_same.fetch_add(smp_same, std::memory_order_relaxed);
		st.buf_slots.fetch_add(buf_sz ? used_buffers : 0, std::memory_order_relaxed);
		st.buf_addr_same.fetch_add(addr_same, std::memory_order_relaxed);
		st.buf_fmt_same.fetch_add(fmt_same, std::memory_order_relaxed);
		st.buf_size_same.fetch_add(size_same, std::memory_order_relaxed);
		st.call_img_same.fetch_add(all_img, std::memory_order_relaxed);
		st.call_smp_same.fetch_add(all_smp, std::memory_order_relaxed);
		st.call_shape_same.fetch_add(all_shp, std::memory_order_relaxed);
		st.call_all_same.fetch_add(all_img && all_smp && all_shp && all_adr &&
		                               p.flattened_srt == s.flattened_srt && p.user_data == s.user_data,
		                           std::memory_order_relaxed);
	}
	p.images        = s.images;
	p.samplers      = s.samplers;
	p.buffers       = s.buffers;
	p.flattened_srt = s.flattened_srt;
	p.user_data     = s.user_data;
	st.overhead_ns.fetch_add(DeltaNowNs() - t0, std::memory_order_relaxed);
	return &p;
}
// Image slot i resolved to image_id: a binding could have been reused when the T# words and the resolved
// image id both equal this program's previous use and the image has no rebind/dirty condition (eligible).
inline void DeltaImageSlot(DeltaProg* p, size_t i, uint32_t index, uint32_t generation, bool eligible) {
	if (p == nullptr || i >= p->img_id.size()) return;
	const uint64_t t0 = DeltaNowNs();
	const uint64_t id = (static_cast<uint64_t>(index) << 32u) | generation;
	if (p->cmp && p->img_same[i] != 0 && p->img_id[i] == id && eligible) {
		g_delta.img_reuse.fetch_add(1, std::memory_order_relaxed);
	}
	p->img_id[i] = id;
	g_delta.overhead_ns.fetch_add(DeltaNowNs() - t0, std::memory_order_relaxed);
}
// Hash of the CB/DB register words feeding the draw's render targets, versus the previous draw's hash.
inline void DeltaRt(uint64_t hash) {
	static uint64_t last = 0;
	g_delta.rt_draws.fetch_add(1, std::memory_order_relaxed);
	if (hash == last) g_delta.rt_repeat.fetch_add(1, std::memory_order_relaxed);
	last = hash;
}
inline constexpr uint64_t kSpecTableBits = 18;
inline constexpr uint64_t kSpecTableSize = 1ull << kSpecTableBits;
inline std::atomic<uint64_t> g_spec_fault_seq[kSpecTableSize];  // page -> seq of last CPU write fault
inline std::atomic<uint64_t> g_spec_gpu_epoch[kSpecTableSize];  // page -> batch epoch of last GPU write
inline std::atomic<uint64_t> g_spec_pkt_seq[kSpecTableSize];    // page -> seq of last PM4 packet guest-memory write
// Measurement-only read collector for the materialization in progress on this thread (fed from the
// low-level shader memory readers; null outside NHL27SPECULATE materialization).
inline thread_local std::vector<std::pair<uint64_t, uint64_t>>* g_spec_collect = nullptr;
inline void SpecCollectRead(uint64_t addr, uint64_t size) {
	auto* c = g_spec_collect;
	if (c == nullptr) return;
	if (!c->empty() && c->back().first + c->back().second == addr) {
		c->back().second += size;
		return;
	}
	if (c->size() < 4096) c->emplace_back(addr, size);
}
inline std::atomic<uint64_t> g_spec_seq {0};
inline std::atomic<uint64_t> g_spec_wait_seq {0};
inline std::atomic<uint64_t> g_spec_epoch {1};
inline uint64_t SpecIndex(uint64_t page) { return (page ^ (page >> kSpecTableBits)) & (kSpecTableSize - 1); }
inline void SpecNoteFault(uint64_t vaddr) {
	if (!Verbose()) return;
	g_spec_fault_seq[SpecIndex(vaddr >> 12)].store(g_spec_seq.load(std::memory_order_relaxed) + 1,
	                                               std::memory_order_relaxed);
}
inline void SpecNoteGpuWrite(uint64_t vaddr, uint64_t size) {
	if (!Verbose()) return;
	const uint64_t first = vaddr >> 12;
	const uint64_t last  = (vaddr + (size != 0 ? size - 1 : 0)) >> 12;
	const uint64_t epoch = g_spec_epoch.load(std::memory_order_relaxed);
	for (uint64_t p = first; p <= last && p - first < 4096; ++p) {
		g_spec_gpu_epoch[SpecIndex(p)].store(epoch, std::memory_order_relaxed);
	}
}
inline void SpecNotePktWrite(uint64_t vaddr, uint64_t size) {
	if (!Verbose()) return;
	const uint64_t first = vaddr >> 12;
	const uint64_t last  = (vaddr + (size != 0 ? size - 1 : 0)) >> 12;
	const uint64_t at    = g_spec_seq.load(std::memory_order_relaxed) + 1;
	for (uint64_t p = first; p <= last && p - first < 4096; ++p) {
		g_spec_pkt_seq[SpecIndex(p)].store(at, std::memory_order_relaxed);
	}
}
inline void SpecNoteWait() { if (!Verbose()) return; g_spec_wait_seq.store(g_spec_seq.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed); }
inline void SpecNoteSubmit() { if (!Verbose()) return; g_spec_epoch.fetch_add(1, std::memory_order_relaxed); }
// Called once per materialization on the GPU thread with the walker's captured read ranges.
inline void SpecObserve(const std::vector<std::pair<uint64_t, uint64_t>>& reads) {
	if (!Verbose()) return;
	const uint64_t seq   = g_spec_seq.fetch_add(1, std::memory_order_relaxed) + 1;
	const uint64_t epoch = g_spec_epoch.load(std::memory_order_relaxed);
	uint64_t       newest_fault = 0, newest_pw = 0;
	bool           gpu = false;
	for (const auto& [addr, size] : reads) {
		const uint64_t first = addr >> 12;
		const uint64_t last  = (addr + (size != 0 ? size - 1 : 0)) >> 12;
		for (uint64_t p = first; p <= last && p - first < 256; ++p) {
			const uint64_t i = SpecIndex(p);
			newest_fault     = std::max(newest_fault, g_spec_fault_seq[i].load(std::memory_order_relaxed));
			newest_pw        = std::max(newest_pw, g_spec_pkt_seq[i].load(std::memory_order_relaxed));
			gpu |= g_spec_gpu_epoch[i].load(std::memory_order_relaxed) == epoch;
		}
	}
	const uint64_t wait   = g_spec_wait_seq.load(std::memory_order_relaxed);
	const auto     within = [seq](uint64_t at, uint64_t n) { return at != 0 && seq - at < n; };
	const bool cpu8 = within(newest_fault, 8), cpu48 = within(newest_fault, 48);
	const bool w8 = within(wait, 8), w48 = within(wait, 48);
	const bool pw8 = within(newest_pw, 8), pw48 = within(newest_pw, 48);
	auto& st = g_spec;
	st.mats.fetch_add(1, std::memory_order_relaxed);
	st.ranges.fetch_add(reads.size(), std::memory_order_relaxed);
	if (reads.empty()) st.noreads.fetch_add(1, std::memory_order_relaxed);
	if (gpu) st.blk_gpu.fetch_add(1, std::memory_order_relaxed);
	if (cpu8) st.blk_cpu8.fetch_add(1, std::memory_order_relaxed);
	if (cpu48) st.blk_cpu48.fetch_add(1, std::memory_order_relaxed);
	if (pw8) st.blk_pw8.fetch_add(1, std::memory_order_relaxed);
	if (pw48) st.blk_pw48.fetch_add(1, std::memory_order_relaxed);
	if (w8) st.blk_wait8.fetch_add(1, std::memory_order_relaxed);
	if (w48) st.blk_wait48.fetch_add(1, std::memory_order_relaxed);
	if (!gpu && !cpu8 && !w8 && !pw8) st.spec8.fetch_add(1, std::memory_order_relaxed);
	if (!gpu && !cpu48 && !w48 && !pw48) st.spec48.fetch_add(1, std::memory_order_relaxed);
}

// NHL27FAULTCOST: breakdown of guest page-fault handling (RenderContext::HandleFault), all threads.
// Time buckets are summed over every faulting thread (guest CPU threads run in parallel with the GPU
// thread, so the sum can exceed wall time). Nested buckets: buf_inv includes tracker lock, protect and
// (for GPU-dirty pages) readmem; tex_inv includes tex_lock; readmem includes the GPU queue wait.
struct FaultCost {
	std::atomic<uint64_t> total_ns {0};      // HandleFault body wall time (same span as faults=)
	std::atomic<uint64_t> veh_ns {0};        // VEH entry -> HandleFault entry
	std::atomic<uint64_t> lock_ns {0};       // tracker region spinlock wait + texture-cache m_lock wait
	std::atomic<uint64_t> buf_inv_ns {0};    // BufferCache::InvalidateMemory (write faults)
	std::atomic<uint64_t> tex_inv_ns {0};    // TextureCache::InvalidateMemory (write faults)
	std::atomic<uint64_t> readmem_ns {0};    // BufferCache::ReadMemory, any fault path (incl. GPU wait)
	std::atomic<uint64_t> rd_readmem_ns {0}; // ReadMemory from read faults only
	std::atomic<uint64_t> protect_ns {0};    // PageManager Protect (VirtualProtect) during a fault
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> total_gpu_ns {0};  // total_ns spent on the GPU thread (inside gpu_thread_busy)
	std::atomic<uint64_t> total_guest_ns {0};
	std::atomic<uint64_t> n[4] {};           // gpu_r, gpu_w, guest_r, guest_w
	std::atomic<uint64_t> distinct {0};      // distinct 4 KiB pages faulted, counted once per frame
	std::atomic<uint64_t> epoch {1};         // frame counter, bumped on every presented flip
	static constexpr size_t kSlots = 16384;
	std::atomic<uint64_t> slots[kSlots] {};  // epoch<<36 | page index
};
inline FaultCost g_fault_cost;
// True while this thread is inside RenderContext::HandleFault.
inline thread_local bool g_in_fault = false;

inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

// Counts the page once per frame epoch; open addressing with stale-epoch slots reusable.
inline void NoteFaultPage(uint64_t vaddr) noexcept {
	auto&          fc    = g_fault_cost;
	const uint64_t page  = (vaddr >> 12) & ((1ull << 36) - 1);
	const uint64_t epoch = fc.epoch.load(std::memory_order_relaxed);
	const uint64_t want  = (epoch << 36) | page;
	size_t         idx   = static_cast<size_t>((page * 0x9E3779B97F4A7C15ull) >> 50) % FaultCost::kSlots;
	for (int probe = 0; probe < 32; ++probe, idx = (idx + 1) % FaultCost::kSlots) {
		uint64_t cur = fc.slots[idx].load(std::memory_order_relaxed);
		if (cur == want) {
			return;
		}
		if ((cur >> 36) != epoch) {
			if (fc.slots[idx].compare_exchange_strong(cur, want, std::memory_order_relaxed)) {
				fc.distinct.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			if (cur == want) {
				return;
			}
		}
	}
	fc.distinct.fetch_add(1, std::memory_order_relaxed); // table full: count conservatively
}

// Times a nested fault bucket, only on a thread that is currently inside HandleFault.
class FaultTimer {
public:
	explicit FaultTimer(std::atomic<uint64_t>& sink) noexcept: m_sink(&sink), m_active(g_in_fault) {
		if (m_active) {
			m_start = NowNs();
		}
	}
	FaultTimer(const FaultTimer&)            = delete;
	FaultTimer& operator=(const FaultTimer&) = delete;
	~FaultTimer() {
		if (m_active) {
			m_sink->fetch_add(NowNs() - m_start, std::memory_order_relaxed);
		}
	}

private:
	std::atomic<uint64_t>* m_sink;
	uint64_t               m_start = 0;
	bool                   m_active;
};

// Adds the elapsed time to sink when the scope ends, or earlier through Stop(). An inactive timer
// writes only its flag (twenty of them run per draw).
class Timer {
public:
	explicit Timer(std::atomic<uint64_t>& sink, bool condition = true) noexcept
	    : m_active(condition && Enabled()) {
		if (m_active) {
			m_sink  = &sink;
			m_start = Clock::now().time_since_epoch().count();
		}
	}
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;
	~Timer() {
		Stop();
	}
	void Stop() noexcept {
		if (!m_active) {
			return;
		}
		m_active      = false;
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		    Clock::now() - Clock::time_point(Clock::duration(m_start)));
		m_sink->fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
	}

private:
	// Set only while active.
	std::atomic<uint64_t>* m_sink;
	Clock::rep             m_start;
	bool                   m_active;
};

// NHL27TEXTURES: breakdown of dp_textures_ns. Everything is gated on TexScope, which is held by the three
// dp_textures_ns blocks (so compute / other callers of the same functions are not counted).
// Top level (sequential, inside dp_textures_ns): resolve, bind, rebind, rt_rediscover, rt_color, rt_depth.
// Nested: rebind_resolve and view are inside rebind; refresh is inside view / rt_*; transit counts barriers.
struct TexStats {
	std::atomic<uint64_t> resolve_ns {0}, bind_ns {0}, rebind_ns {0}, rebind_resolve_ns {0}, view_ns {0};
	std::atomic<uint64_t> refresh_ns {0}, rt_rediscover_ns {0}, rt_color_ns {0}, rt_depth_ns {0};
	std::atomic<uint64_t> resolved {0}, rebind_resolved {0}, view_hit {0}, view_created {0};
	std::atomic<uint64_t> refresh_calls {0}, uploads {0}, transits {0};
};
inline TexStats g_tex;
inline thread_local int g_tex_scope = 0;
struct TexScope {
	bool on;
	TexScope() noexcept: on(Verbose()) { if (on) ++g_tex_scope; }
	~TexScope() { if (on) --g_tex_scope; }
	TexScope(const TexScope&)            = delete;
	TexScope& operator=(const TexScope&) = delete;
};
inline bool TexOn() noexcept {
	return g_tex_scope > 0;  // g_tex_scope only moves when Verbose()
}
inline void TexAdd(std::atomic<uint64_t>& counter) noexcept {
	if (TexOn()) counter.fetch_add(1, std::memory_order_relaxed);
}

inline void Add(std::atomic<uint64_t>& counter, uint64_t amount = 1) noexcept {
	if (Enabled()) {
		counter.fetch_add(amount, std::memory_order_relaxed);
	}
}

// Counter for the verbose-only stats: no atomic traffic unless KYTY_PERF_STATS_VERBOSE=1.
inline void VAdd(std::atomic<uint64_t>& counter, uint64_t amount = 1) noexcept {
	if (Verbose()) {
		counter.fetch_add(amount, std::memory_order_relaxed);
	}
}

inline void AddIdleGap(uint64_t ns) noexcept {
	if (Enabled()) {
		g_counters.idle_ns.fetch_add(ns, std::memory_order_relaxed);
		const size_t bucket = ns < 1'000'000u ? 0 : ns < 4'000'000u ? 1 : ns < 10'000'000u ? 2 : 3;
		g_counters.idle_gaps[bucket].fetch_add(1, std::memory_order_relaxed);
	}
}

inline void AddReadback(uint64_t ns) noexcept {
	if (Enabled()) {
		g_counters.readback_ns.fetch_add(ns, std::memory_order_relaxed);
		g_counters.readback_count.fetch_add(1, std::memory_order_relaxed);
	}
}

// NHL27GPUBUSY: exclusive attribution of GPU-thread busy time (Process wall time, KYTY_PERF_STATS).
// Between GpuBusyBegin and GpuBusyEnd (the same window as gpu_process_ns) every nanosecond is charged
// to exactly one bucket: the innermost active GpuBusyScope, or pm4_loop when none is active. Scopes
// nest; a nested scope's time is removed from its parent, so the buckets add up to the Process wall
// time. wait is the master-semaphore block time (gpu_thread_busy excludes it). Scopes are no-ops on
// any thread/time outside that window. draw_prep is the DrawPrepBegin -> DrawEmitted window minus
// nested scopes (its NHL27DRAWPREP total is the inclusive figure).
enum GpuBucket : uint8_t {
	GB_PM4_LOOP,         // Process/CommandProcessor loop outside packet handlers (fetch, parse, split checks)
	GB_HANDLER,          // other packet handlers (register sets, NOP/flip, indirect buffer, ...)
	GB_DRAW_HANDLER,     // draw packet handlers excluding draw_prep and nested scopes
	GB_DISPATCH_HANDLER, // dispatch packet handlers excluding nested scopes (compute prep + record)
	GB_SYNC_PKT,         // EOP/RELEASE_MEM/EVENT_WRITE/WRITE_DATA/WAIT_REG_MEM/semaphore/sync packets
	GB_DMA_PKT,          // DMA_DATA/CP_DMA/COPY_DATA packets
	GB_DRAW_PREP,        // DrawPrepBegin -> DrawEmitted, exclusive
	GB_GPU_CMDS,         // GuestGpu::ProcessCommands (commands posted from other threads)
	GB_FAULT,            // RenderContext::HandleFault on the GPU thread
	GB_GC,               // RenderContext::RunGarbageCollector
	GB_DEFERRED,         // CommandScheduler::PopPendingOperations callbacks
	GB_SUBMIT,           // CommandScheduler::Submit excluding vkQueueSubmit (End, flags refresh, trace)
	GB_QUEUE_SUBMIT,     // vkQueueSubmit incl. queue mutex
	GB_BEGIN_CMD,        // CommandScheduler::BeginCommand (pool Commit + vkBeginCommandBuffer)
	GB_RENDER_PASS,      // CommandBuffer::BeginRendering/EndRendering (actual begin/end only)
	GB_BARRIER,          // Image::Transit and shader access/hazard barriers
	GB_TEX_UPLOAD,       // Image::Upload
	GB_WAIT,             // MasterSemaphore::Wait blocked (not part of gpu_thread_busy)
	GB_COUNT
};

struct GpuBusyState {
	bool             active = false;
	uint8_t          cur    = GB_PM4_LOOP;
	uint8_t          prep_prev = GB_PM4_LOOP;
	Clock::time_point last {};
	uint64_t         acc[GB_COUNT] {};
};
inline thread_local GpuBusyState g_gpu_busy;
inline std::array<std::atomic<uint64_t>, GB_COUNT> g_gpu_busy_ns {};

inline void GpuBusySwitch(uint8_t bucket) noexcept {
	auto&      s   = g_gpu_busy;
	const auto now = Clock::now();
	s.acc[s.cur] += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - s.last).count());
	s.last = now;
	s.cur  = bucket;
}

inline void GpuBusyBegin() noexcept {
	if (!Enabled()) {
		return;
	}
	auto& s = g_gpu_busy;
	s.active = true;
	s.cur    = GB_PM4_LOOP;
	s.last   = Clock::now();
}

inline void GpuBusyEnd() noexcept {
	auto& s = g_gpu_busy;
	if (!s.active) {
		return;
	}
	GpuBusySwitch(GB_PM4_LOOP);
	for (size_t i = 0; i < GB_COUNT; ++i) {
		g_gpu_busy_ns[i].fetch_add(s.acc[i], std::memory_order_relaxed);
		s.acc[i] = 0;
	}
	s.active = false;
}

class GpuBusyScope {
public:
	explicit GpuBusyScope(GpuBucket bucket) noexcept: m_on(g_gpu_busy.active) {
		if (m_on) {
			m_prev = g_gpu_busy.cur;
			GpuBusySwitch(bucket);
		}
	}
	GpuBusyScope(const GpuBusyScope&)            = delete;
	GpuBusyScope& operator=(const GpuBusyScope&) = delete;
	~GpuBusyScope() {
		if (m_on && g_gpu_busy.active) {
			GpuBusySwitch(m_prev);
		}
	}

private:
	uint8_t m_prev = GB_PM4_LOOP;
	bool    m_on;
};

// Draw prep starts at PrepareDrawRenderState entry and ends at the vkCmd draw emit. The start is
// thread-local, so a draw skipped before its emit is simply overwritten by the next begin.
inline thread_local Clock::time_point g_draw_prep_start {};

inline void DrawPrepBegin() noexcept {
	if (Enabled()) {
		g_draw_prep_start = Clock::now();
		if (g_gpu_busy.active && g_gpu_busy.cur != GB_DRAW_PREP) {
			g_gpu_busy.prep_prev = g_gpu_busy.cur;
			GpuBusySwitch(GB_DRAW_PREP);
		}
	}
}

inline void DrawEmitted() noexcept {
	if (!Enabled()) {
		return;
	}
	// A draw skipped before its emit leaves draw_prep selected until the enclosing handler scope ends.
	if (g_gpu_busy.active && g_gpu_busy.cur == GB_DRAW_PREP) {
		GpuBusySwitch(g_gpu_busy.prep_prev);
	}
	g_counters.draws.fetch_add(1, std::memory_order_relaxed);
	if (g_draw_prep_start != Clock::time_point {}) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_draw_prep_start);
		g_counters.draw_prep_ns.fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
		g_draw_prep_start = Clock::time_point {};
	}
}

inline std::mutex         g_window_mutex;
inline Clock::time_point  g_window_start {};
inline Clock::time_point  g_last_flip {};

inline uint64_t Take(std::atomic<uint64_t>& value) noexcept {
	return value.exchange(0, std::memory_order_relaxed);
}

// Caller holds g_window_mutex. Prints and resets the window once it spans at least one second.
inline void PrintIfDueLocked(Clock::time_point now) {
	if (g_window_start == Clock::time_point {}) {
		g_window_start = now;
	}
	const double seconds = std::chrono::duration<double>(now - g_window_start).count();
	if (seconds < 1.0) {
		return;
	}
	auto& c = g_counters;
	const uint64_t frames        = Take(c.flip_intervals);
	const uint64_t frame_ns      = Take(c.flip_ns);
	const uint64_t process_ns    = Take(c.gpu_process_ns);
	const uint64_t wait_ns       = Take(c.gpu_wait_ns);
	const uint64_t wait_count    = Take(c.gpu_wait_count);
	const uint64_t readback_ns   = Take(c.readback_ns);
	const uint64_t readback_n    = Take(c.readback_count);
	const uint64_t fault_ns      = Take(c.fault_ns);
	const uint64_t fault_n       = Take(c.fault_count);
	const uint64_t compile_ns    = Take(c.compile_ns);
	const uint64_t compile_n     = Take(c.compile_count);
	const uint64_t draw_prep_ns  = Take(c.draw_prep_ns);
	const uint64_t draws         = Take(c.draws);
	const uint64_t dispatches    = Take(c.dispatches);
	const uint64_t prep_graphics_programs_ns = Take(c.prep_graphics_programs_ns);
	const uint64_t prep_compute_program_ns = Take(c.prep_compute_program_ns);
	const uint64_t prep_color_target_ns = Take(c.prep_color_target_ns);
	const uint64_t prep_find_buffers_ns = Take(c.prep_find_buffers_ns);
	const uint64_t prep_rebind_images_ns = Take(c.prep_rebind_images_ns);
	const uint64_t prep_rebind_buffers_ns = Take(c.prep_rebind_buffers_ns);
	const uint64_t prep_commit_graphics_ns = Take(c.prep_commit_graphics_ns);
	const uint64_t prep_commit_compute_ns = Take(c.prep_commit_compute_ns);
	const uint64_t dp_materialize_ns = Take(c.dp_materialize_ns);
	const uint64_t dp_textures_ns    = Take(c.dp_textures_ns);
	const uint64_t dp_buffers_ns     = Take(c.dp_buffers_ns);
	const uint64_t dp_descriptors_ns = Take(c.dp_descriptors_ns);
	const uint64_t dp_pipeline_ns    = Take(c.dp_pipeline_ns);
	const uint64_t mat_calls         = Take(c.mat_calls);
	const uint64_t mat_total_ns      = Take(c.mat_total_ns);
	const uint64_t mat_flat_ns       = Take(c.mat_flat_ns);
	const uint64_t mat_eval_ns       = Take(c.mat_eval_ns);
	const uint64_t mat_indirect_ns   = Take(c.mat_indirect_ns);
	const uint64_t mat_spec_ns       = Take(c.mat_spec_ns);
	const uint64_t mat_evals         = Take(c.mat_evals);
	const uint64_t mat_flat_reads    = Take(c.mat_flat_reads);
	const uint64_t mat_flat_read_ns  = Take(c.mat_flat_read_ns);
	const uint64_t mat_flat_ops      = Take(c.mat_flat_ops);
	const uint64_t mat_flat_alloc_ns = Take(c.mat_flat_alloc_ns);
	const uint64_t mat_flat_trace_hits  = Take(c.mat_flat_trace_hits);
	const uint64_t mat_flat_trace_other = Take(c.mat_flat_trace_other);
	const uint64_t mat_flat_hit_ns      = Take(c.mat_flat_hit_ns);
	const uint64_t mat_flat_nocfg_ns    = Take(c.mat_flat_nocfg_ns);
	const uint64_t mat_flat_nocfg_ops   = Take(c.mat_flat_nocfg_ops);
	const uint64_t mat_hit_steps        = Take(c.mat_hit_steps);
	const uint64_t mat_hit_reads        = Take(c.mat_hit_reads);
	const uint64_t mat_hit_memo         = Take(c.mat_hit_memo);
	uint64_t mat_tr_why[16];
	for (int i = 0; i < 16; ++i) mat_tr_why[i] = Take(c.mat_tr_why[i]);
	const uint64_t mat_reuse_hits    = Take(c.mat_reuse_hits);
	const uint64_t mat_reuse_misses  = Take(c.mat_reuse_misses);
	const uint64_t mat_reuse_fallback = Take(c.mat_reuse_fallback);
	const uint64_t dp_vertex_index_ns  = Take(c.dp_vertex_index_ns);
	const uint64_t dp_indirect_ns      = Take(c.dp_indirect_ns);
	const uint64_t dp_find_buffers_ns  = Take(c.dp_find_buffers_ns);
	const uint64_t dp_bda_ns           = Take(c.dp_bda_ns);
	const uint64_t dp_rebind_ns        = Take(c.dp_rebind_ns);
	const uint64_t dp_rebind_lookup_ns = Take(c.dp_rebind_lookup_ns);
	const uint64_t dp_rebind_upload_ns = Take(c.dp_rebind_upload_ns);
	const uint64_t obtain_cache_hits   = Take(c.obtain_cache_hits);
	const uint64_t obtain_cache_misses = Take(c.obtain_cache_misses);
	const uint64_t om_slot = Take(c.obtain_miss_slot), om_reg = Take(c.obtain_miss_registration);
	const uint64_t om_buf = Take(c.obtain_miss_buffer), om_cpu = Take(c.obtain_miss_cpu);
	const uint64_t on_stream = Take(c.obtain_nostore_stream), on_reg = Take(c.obtain_nostore_registration);
	const uint64_t on_cpu = Take(c.obtain_nostore_cpu), b_regs = Take(c.buffer_registrations);
	uint64_t       slot_counts[6][3] {};
	for (int k = 0; k < 3; k++) {
		slot_counts[5][k] = Take(c.slot_global_hit[k]);
		slot_counts[0][k] = Take(c.slot_hit[k]);
		slot_counts[1][k] = Take(c.slot_none[k]);
		slot_counts[2][k] = Take(c.slot_same[k]);
		slot_counts[3][k] = Take(c.slot_inside[k]);
		slot_counts[4][k] = Take(c.slot_other[k]);
	}
	const uint64_t dc_total_ns       = Take(c.dc_total_ns);
	const uint64_t dc_materialize_ns = Take(c.dc_materialize_ns);
	const uint64_t dc_textures_ns    = Take(c.dc_textures_ns);
	const uint64_t dc_buffers_ns     = Take(c.dc_buffers_ns);
	const uint64_t dc_bda_ns         = Take(c.dc_bda_ns);
	const uint64_t dc_descriptors_ns = Take(c.dc_descriptors_ns);
	const uint64_t dc_pipeline_ns    = Take(c.dc_pipeline_ns);
	const uint64_t dc_barrier_ns     = Take(c.dc_barrier_ns);
	const uint64_t dc_record_ns      = Take(c.dc_record_ns);
	const uint64_t dc_handled        = Take(c.dc_handled);
	const auto dc_ahead_attempts = Take(c.dc_ahead_attempts);
	const auto dc_ahead_ready = Take(c.dc_ahead_ready);
	const auto dc_ahead_hits = Take(c.dc_ahead_hits);
	const auto dc_ahead_invalid = Take(c.dc_ahead_invalid);
	const auto dc_ahead_ns = Take(c.dc_ahead_ns);
	const auto dc_ahead_validate_ns = Take(c.dc_ahead_validate_ns);
	const uint64_t dc_bind_skipped   = Take(c.dc_bind_skipped);
	const uint64_t fp_passes         = Take(c.fault_parse_passes);
	const uint64_t fp_zero           = Take(c.fault_parse_zero);
	const uint64_t fp_faults         = Take(c.fault_parse_faults);
	const uint64_t fp_ring_wait_ns   = Take(c.fault_ring_wait_ns);
	const uint64_t protect_calls     = Take(c.protect_calls);
	const uint64_t protect_ns        = Take(c.protect_ns);
	const uint64_t bda_calls         = Take(c.bda_calls);
	const uint64_t bda_skips         = Take(c.bda_skips);
	const uint64_t bda_scans         = Take(c.bda_scans);
	const uint64_t bda_regions       = Take(c.bda_regions);
	const uint64_t bda_ranges        = Take(c.bda_ranges);
	const uint64_t bda_pages         = Take(c.bda_pages);
	const uint64_t bda_total_ns      = Take(c.bda_total_ns);
	const uint64_t bda_lock_ns       = Take(c.bda_lock_ns);
	const uint64_t bda_sync_ns       = Take(c.bda_sync_ns);
	const uint64_t sync_uploads      = Take(c.sync_uploads);
	const uint64_t sync_bytes        = Take(c.sync_bytes);
	const uint64_t port_auto         = Take(c.port_auto_barriers);
	const uint64_t port_auto_skip    = Take(c.port_auto_barriers_skipped);
	const uint64_t port_gbar         = Take(c.port_guest_barriers);
	const uint64_t port_gbar_skip    = Take(c.port_guest_barriers_skipped);
	const uint64_t port_acq          = Take(c.port_acquire_mem);
	const uint64_t port_guard        = Take(c.port_guard_dispatches);
	const uint64_t port_hit          = Take(c.port_hint_hits);
	const uint64_t port_miss         = Take(c.port_hint_misses);
	const uint64_t port_rt_hit       = Take(c.port_rt_memo_hits);
	const uint64_t port_rt_miss      = Take(c.port_rt_memo_misses);
	const uint64_t port_gc_del       = Take(c.port_buffer_gc_deletes);
	const uint64_t port_buf_new      = Take(c.port_buffer_creates);
	const uint64_t port_rbq_used     = Take(c.port_rbq_used);
	const uint64_t port_rbq_fb       = Take(c.port_rbq_fallback);
	const uint64_t port_rbq_wait     = Take(c.port_rbq_wait_ns);
	const uint64_t port_eop          = Take(c.port_eop_direct);
	const uint64_t port_fwin         = Take(c.port_fault_windows);
	const uint64_t port_fpages       = Take(c.port_fault_pages);
	const uint64_t port_gcs          = Take(c.port_gc_scans);
	const uint64_t port_gcskip       = Take(c.port_gc_sparse_skips);
	uint64_t gb[GB_COUNT];
	for (size_t i = 0; i < GB_COUNT; ++i) {
		gb[i] = Take(g_gpu_busy_ns[i]);
	}
	g_window_start = now;
	if (frames == 0) {
		return;
	}
	const double n       = static_cast<double>(frames);
	const auto   ms      = [n](uint64_t ns) { return static_cast<double>(ns) / 1e6 / n; };
	const auto   per     = [n](uint64_t v) { return static_cast<double>(v) / n; };
	const uint64_t busy_ns = process_ns > wait_ns ? process_ns - wait_ns : 0;
	const double draw_us = draws != 0 ? static_cast<double>(draw_prep_ns) / 1e3 / static_cast<double>(draws) : 0.0;
	std::printf("NHL27FRAME fps=%.1f frame=%.2fms frames=%" PRIu64
	            " gpu_thread_busy=%.2fms wait_gpu=%.2fms(%.2f) readback=%.2fms(%.2f) "
	            "faults=%.2fms(%.2f) compile=%.2fms(%.3f) draw_prep=%.2fms(%.1fus/draw) "
	            "draws=%.1f dispatches=%.1f vblanks=%llu,%llu,%llu,%llu,%llu idle=%.2fms gaps=%.1f,%.1f,%.1f,%.1f\n",
	            static_cast<double>(frames) / seconds, ms(frame_ns), frames, ms(busy_ns),
	            ms(wait_ns), per(wait_count), ms(readback_ns), per(readback_n), ms(fault_ns),
	            per(fault_n), ms(compile_ns), per(compile_n), ms(draw_prep_ns), draw_us,
	            per(draws), per(dispatches), static_cast<unsigned long long>(Take(c.flip_vblanks[0])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[1])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[2])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[3])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[4])), ms(Take(c.idle_ns)),
	            per(Take(c.idle_gaps[0])), per(Take(c.idle_gaps[1])), per(Take(c.idle_gaps[2])),
	            per(Take(c.idle_gaps[3])));
	// Flush every line: when stdout is a pipe (the launcher pipes into
	// Tee-Object) libc uses a block buffer, so without this the window lines
	// arrive minutes late and an automated collector cannot attribute them.
	std::fflush(stdout);
	// Exclusive buckets (see GpuBucket): they sum to Process wall time, so busy = sum - wait.
	// unattributed = busy - sum(non-wait buckets); it is only timer skew / uncovered gaps.
	{
		uint64_t attributed = 0;
		for (size_t i = 0; i < GB_COUNT; ++i) {
			if (i != GB_WAIT) {
				attributed += gb[i];
			}
		}
		const double unattr_ms = ms(busy_ns) - ms(attributed);
		std::printf("NHL27GPUBUSY busy=%.2fms pm4_loop=%.2fms handler=%.2fms draw_handler=%.2fms "
		            "dispatch_handler=%.2fms sync_pkt=%.2fms dma_pkt=%.2fms draw_prep_excl=%.2fms "
		            "gpu_cmds=%.2fms fault=%.2fms gc=%.2fms deferred_ops=%.2fms submit=%.2fms "
		            "queue_submit=%.2fms begin_cmd=%.2fms render_pass=%.2fms barrier=%.2fms "
		            "tex_upload=%.2fms wait=%.2fms unattributed=%.2fms "
		            "(draw_prep_incl=%.2fms)\n",
		            ms(busy_ns), ms(gb[GB_PM4_LOOP]), ms(gb[GB_HANDLER]), ms(gb[GB_DRAW_HANDLER]),
		            ms(gb[GB_DISPATCH_HANDLER]), ms(gb[GB_SYNC_PKT]), ms(gb[GB_DMA_PKT]),
		            ms(gb[GB_DRAW_PREP]), ms(gb[GB_GPU_CMDS]), ms(gb[GB_FAULT]), ms(gb[GB_GC]),
		            ms(gb[GB_DEFERRED]), ms(gb[GB_SUBMIT]), ms(gb[GB_QUEUE_SUBMIT]),
		            ms(gb[GB_BEGIN_CMD]), ms(gb[GB_RENDER_PASS]), ms(gb[GB_BARRIER]),
		            ms(gb[GB_TEX_UPLOAD]), ms(gb[GB_WAIT]), unattr_ms, ms(draw_prep_ns));
		std::fflush(stdout);
	}
	std::printf("NHL27PORT per_frame: auto_barriers=%.1f(skipped %.1f) guest_barriers=%.1f(deduped %.1f) "
	            "acquire_mem=%.1f guard_dispatches=%.1f image_by_addr=%.1f(region_scans %.1f) "
	            "rt_memo_hit=%.1f(miss %.1f) buffer_creates=%.1f buffer_gc_deletes=%.1f\n",
	            per(port_auto), per(port_auto_skip), per(port_gbar), per(port_gbar_skip), per(port_acq),
	            per(port_guard), per(port_hit), per(port_miss), per(port_rt_hit), per(port_rt_miss),
	            per(port_buf_new), per(port_gc_del));
	std::printf("NHL27PORT2 per_frame: readback_queue_used=%.2f(fallbacks %.2f wait=%.3fms) eop_direct=%.2f "
	            "fault_coalesce_windows=%.2f(pages %.2f) sparse_gc_scans=%.2f(skipped %.2f)\n",
	            per(port_rbq_used), per(port_rbq_fb), ms(port_rbq_wait),
	            per(port_eop), per(port_fwin), per(port_fpages), per(port_gcs), per(port_gcskip));
	std::fflush(stdout);
	if (DetailEnabled()) {
		std::printf("NHL27PREP frames=%" PRIu64
		            " graphics_programs=%.3fms compute_program=%.3fms color_target=%.3fms"
		            " find_buffers=%.3fms rebind_images=%.3fms rebind_buffers=%.3fms"
		            " commit_graphics=%.3fms commit_compute=%.3fms\n",
		            frames, ms(prep_graphics_programs_ns), ms(prep_compute_program_ns),
		            ms(prep_color_target_ns), ms(prep_find_buffers_ns), ms(prep_rebind_images_ns),
		            ms(prep_rebind_buffers_ns), ms(prep_commit_graphics_ns), ms(prep_commit_compute_ns));
		std::fflush(stdout);
	}
	// Same window as NHL27FRAME. other = draw_prep minus the named buckets (clamped at zero).
	{
		const double named_ms = ms(dp_materialize_ns) + ms(dp_textures_ns) + ms(dp_buffers_ns) +
		                        ms(dp_descriptors_ns) + ms(dp_pipeline_ns);
		const double other_ms = std::max(0.0, ms(draw_prep_ns) - named_ms);
		std::printf("NHL27DRAWPREP total=%.2fms materialize=%.2fms textures=%.2fms buffers=%.2fms "
		            "descriptors=%.2fms pipeline=%.2fms other=%.2fms\n",
		            ms(draw_prep_ns), ms(dp_materialize_ns), ms(dp_textures_ns), ms(dp_buffers_ns),
		            ms(dp_descriptors_ns), ms(dp_pipeline_ns), other_ms);
		if (Verbose()) {
			auto&          tx = g_tex;
			const uint64_t t_res = Take(tx.resolve_ns), t_bind = Take(tx.bind_ns), t_reb = Take(tx.rebind_ns);
			const uint64_t t_rres = Take(tx.rebind_resolve_ns), t_view = Take(tx.view_ns);
			const uint64_t t_ref = Take(tx.refresh_ns), t_rtre = Take(tx.rt_rediscover_ns);
			const uint64_t t_rtc = Take(tx.rt_color_ns), t_rtd = Take(tx.rt_depth_ns);
			const uint64_t n_res = Take(tx.resolved), n_rres = Take(tx.rebind_resolved);
			const uint64_t n_vh = Take(tx.view_hit), n_vc = Take(tx.view_created);
			const uint64_t n_ref = Take(tx.refresh_calls), n_up = Take(tx.uploads), n_tr = Take(tx.transits);
			const double   t_other = std::max(0.0, ms(dp_textures_ns) - ms(t_res) - ms(t_bind) - ms(t_reb) -
			                                           ms(t_rtre) - ms(t_rtc) - ms(t_rtd));
			std::printf("NHL27TEXTURES total=%.2fms resolve=%.2fms bind=%.2fms rebind=%.2fms(resolve=%.2f view=%.2f) "
			            "rt_rediscover=%.2fms rt_color=%.2fms rt_depth=%.2fms other=%.2fms refresh=%.2fms(nested) "
			            "resolved=%.1f rebind_resolved=%.1f views_hit=%.1f views_created=%.1f refresh_calls=%.1f "
			            "uploads=%.1f transits=%.1f\n",
			            ms(dp_textures_ns), ms(t_res), ms(t_bind), ms(t_reb), ms(t_rres), ms(t_view), ms(t_rtre),
			            ms(t_rtc), ms(t_rtd), t_other, ms(t_ref), per(n_res), per(n_rres), per(n_vh), per(n_vc),
			            per(n_ref), per(n_up), per(n_tr));
		}
		{
			const uint64_t mat_named = mat_flat_ns + mat_eval_ns + mat_indirect_ns + mat_spec_ns;
			const uint64_t mat_rest  = mat_total_ns > mat_named ? mat_total_ns - mat_named : 0;
			std::printf("NHL27MATERIALIZE calls=%.1f total=%.2fms flat=%.2fms eval=%.2fms(%.1f) "
			            "indirect=%.2fms spec=%.2fms rest=%.2fms reuse_hit=%.1f reuse_miss=%.1f "
			            "reuse_fallback=%.1f\n",
			            per(mat_calls), ms(mat_total_ns), ms(mat_flat_ns), ms(mat_eval_ns), per(mat_evals),
			            ms(mat_indirect_ns), ms(mat_spec_ns), ms(mat_rest), per(mat_reuse_hits),
			            per(mat_reuse_misses), per(mat_reuse_fallback));
			std::printf("NHL27FLAT words=%.1f read=%.2fms(detail) ops=%.1f alloc=%.2fms(detail) "
			            "trace_hit=%.1f trace_other=%.1f hit_t=%.2fms(%.2fus/call) nocfg_t=%.2fms(%.2fus/call) "
			            "nocfg_ops_per_call=%.1f\n",
			            per(mat_flat_reads), ms(mat_flat_read_ns), per(mat_flat_ops), ms(mat_flat_alloc_ns),
			            per(mat_flat_trace_hits), per(mat_flat_trace_other), ms(mat_flat_hit_ns),
			            mat_flat_trace_hits > 0 ? ms(mat_flat_hit_ns) * 1000.0 / per(mat_flat_trace_hits) : 0.0,
			            ms(mat_flat_nocfg_ns),
			            mat_tr_why[0] > 0 ? ms(mat_flat_nocfg_ns) * 1000.0 / per(mat_tr_why[0]) : 0.0,
			            mat_tr_why[0] > 0 ? per(mat_flat_nocfg_ops) / per(mat_tr_why[0]) : 0.0);
			std::printf("NHL27HIT steps_per_call=%.1f reads_per_call=%.1f memo_per_call=%.1f\n",
			            mat_flat_trace_hits > 0 ? per(mat_hit_steps) / per(mat_flat_trace_hits) : 0.0,
			            mat_flat_trace_hits > 0 ? per(mat_hit_reads) / per(mat_flat_trace_hits) : 0.0,
			            mat_flat_trace_hits > 0 ? per(mat_hit_memo) / per(mat_flat_trace_hits) : 0.0);
			if (Verbose()) {
				auto&          sp  = g_spec;
				const uint64_t sm  = Take(sp.mats);
				const uint64_t srg = Take(sp.ranges);
				std::printf("NHL27SPECULATE mats=%.1f spec8=%.1f spec48=%.1f blocked_pkt_write8=%.1f "
				            "blocked_pkt_write48=%.1f blocked_wait8=%.1f blocked_wait48=%.1f "
				            "blocked_cpu_fault8=%.1f blocked_cpu_fault48=%.1f blocked_gpu_write=%.1f "
				            "no_reads=%.1f avg_ranges=%.2f\n",
				            per(sm), per(Take(sp.spec8)), per(Take(sp.spec48)), per(Take(sp.blk_pw8)),
				            per(Take(sp.blk_pw48)), per(Take(sp.blk_wait8)), per(Take(sp.blk_wait48)),
				            per(Take(sp.blk_cpu8)), per(Take(sp.blk_cpu48)), per(Take(sp.blk_gpu)),
				            per(Take(sp.noreads)),
				            sm != 0 ? static_cast<double>(srg) / static_cast<double>(sm) : 0.0);
			}
			std::printf("NHL27TRACE nocfg=%.1f flag_off=%.1f inelig_mode=%.1f empty=%.1f disabled=%.1f tape_nr=%.1f "
			            "strict=%.1f m_memo=%.1f m_exec=%.1f m_memoslot=%.1f m_cond=%.1f m_eval=%.1f m_condeval=%.1f "
			            "rec_abort=%.1f merge_refused=%.1f merge_new=%.1f\n",
			            per(mat_tr_why[0]), per(mat_tr_why[1]), per(mat_tr_why[2]), per(mat_tr_why[3]),
			            per(mat_tr_why[4]), per(mat_tr_why[5]), per(mat_tr_why[6]), per(mat_tr_why[7]),
			            per(mat_tr_why[8]), per(mat_tr_why[9]), per(mat_tr_why[10]), per(mat_tr_why[11]),
			            per(mat_tr_why[12]), per(mat_tr_why[13]), per(mat_tr_why[14]), per(mat_tr_why[15]));
			const double dc_named = ms(dc_materialize_ns) + ms(dc_textures_ns) + ms(dc_buffers_ns) +
			                        ms(dc_descriptors_ns) + ms(dc_pipeline_ns) + ms(dc_barrier_ns) +
			                        ms(dc_record_ns);
			std::printf("NHL27DISPATCHPREP total=%.2fms materialize=%.2fms textures=%.2fms buffers=%.2fms "
			            "bda=%.2fms descriptors=%.2fms pipeline=%.2fms barrier=%.2fms record=%.2fms "
			            "other=%.2fms count=%.1f bind_skipped=%.1f\n",
			            ms(dc_total_ns), ms(dc_materialize_ns), ms(dc_textures_ns), ms(dc_buffers_ns),
			            ms(dc_bda_ns), ms(dc_descriptors_ns), ms(dc_pipeline_ns), ms(dc_barrier_ns),
			            ms(dc_record_ns), std::max(0.0, ms(dc_total_ns) - dc_named), per(dc_handled),
			            per(dc_bind_skipped));
			if (dc_ahead_attempts || dc_ahead_hits || dc_ahead_invalid) {
				std::printf("NHL27COMPUTEAHEAD attempts=%.1f ready=%.1f hits=%.1f invalid=%.1f "
				            "resolve=%.2fms validate=%.2fms\n", per(dc_ahead_attempts), per(dc_ahead_ready),
				            per(dc_ahead_hits), per(dc_ahead_invalid), ms(dc_ahead_ns), ms(dc_ahead_validate_ns));
			}
			{
				auto&          dl   = g_delta;
				const uint64_t d_calls = Take(dl.calls), d_first = Take(dl.first), d_cmp = Take(dl.cmp_calls);
				const uint64_t d_is = Take(dl.img_slots), d_isame = Take(dl.img_same), d_ireuse = Take(dl.img_reuse);
				const uint64_t d_ss = Take(dl.smp_slots), d_ssame = Take(dl.smp_same);
				const uint64_t d_bs = Take(dl.buf_slots), d_basame = Take(dl.buf_addr_same);
				const uint64_t d_bfmt = Take(dl.buf_fmt_same), d_bsize = Take(dl.buf_size_same), d_ci = Take(dl.call_img_same);
				const uint64_t d_cs = Take(dl.call_smp_same), d_csh = Take(dl.call_shape_same);
				const uint64_t d_call = Take(dl.call_all_same), d_rd = Take(dl.rt_draws);
				const uint64_t d_rr = Take(dl.rt_repeat), d_ov = Take(dl.overhead_ns);
				const auto     pct = [](uint64_t a, uint64_t b) {
					return b != 0 ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0;
				};
				if (Verbose()) std::printf("NHL27DELTA calls=%.1f img_slots=%.1f img_same=%.1f%% img_reuse=%.1f%% "
				            "smp_same=%.1f%% buf_addr_same=%.1f%% buf_fmt_same=%.1f%% buf_size_same=%.1f%% call_img_same=%.1f%% "
				            "call_shape_same=%.1f%% call_all_same=%.1f%% rt_repeat=%.1f%% overhead=%.2fms "
				            "first=%.1f smp_slots=%.1f buf_slots=%.1f call_smp_same=%.1f%%\n",
				            per(d_calls), per(d_is), pct(d_isame, d_is), pct(d_ireuse, d_is), pct(d_ssame, d_ss),
				            pct(d_basame, d_bs), pct(d_bfmt, d_bs), pct(d_bsize, d_bs), pct(d_ci, d_cmp), pct(d_csh, d_cmp),
				            pct(d_call, d_cmp), pct(d_rr, d_rd), ms(d_ov), per(d_first), per(d_ss), per(d_bs),
				            pct(d_cs, d_cmp));
				const uint64_t r_hit = Take(dl.reuse_hits), r_el = Take(dl.reuse_eligible);
				const uint64_t r_gen = Take(dl.reuse_rej_gen), r_st = Take(dl.reuse_rej_state);
				const uint64_t r_um = Take(dl.reuse_unmemoable), r_vc = Take(dl.verify_checks);
				const uint64_t r_vm = Take(dl.verify_mismatch), r_ss = Take(dl.smp_slots_seen);
				const uint64_t r_sr = Take(dl.smp_reuse);
				std::printf("NHL27DELTAREUSE reuse_hits=%.1f reuse_eligible=%.1f rejected_gen=%.1f rejected_state=%.1f "
				            "unmemoable=%.1f verify_checks=%.1f verify_mismatch=%.1f smp_slots=%.1f smp_reuse=%.1f\n",
				            per(r_hit), per(r_el), per(r_gen), per(r_st), per(r_um), per(r_vc), per(r_vm), per(r_ss),
				            per(r_sr));
					const uint64_t c_hit = Take(dl.cs_hits), c_el = Take(dl.cs_eligible), c_gen = Take(dl.cs_rej_gen);
					const uint64_t c_st = Take(dl.cs_rej_state), c_um = Take(dl.cs_unmemoable), c_sl = Take(dl.cs_slots);
					const uint64_t d_sl = Take(dl.dr_slots), u_dcc = Take(dl.um_dcc), u_dep = Take(dl.um_depth);
					const uint64_t u_oth = Take(dl.um_other);
					if (Verbose()) {
					std::printf("NHL27DELTAREUSE_SPLIT draw: slots=%.1f eligible=%.1f hits=%.1f rejected_gen=%.1f "
					            "rejected_state=%.1f unmemoable=%.1f | dispatch: slots=%.1f eligible=%.1f hits=%.1f "
					            "rejected_gen=%.1f rejected_state=%.1f unmemoable=%.1f | unmemoable_by: dcc_cmask=%.1f "
					            "depth=%.1f other=%.1f\n",
					            per(d_sl), per(r_el - c_el), per(r_hit - c_hit), per(r_gen - c_gen), per(r_st - c_st),
					            per(r_um - c_um), per(c_sl), per(c_el), per(c_hit), per(c_gen), per(c_st), per(c_um),
					            per(u_dcc), per(u_dep), per(u_oth));
					std::printf("NHL27DELTAGEN gen_insert=%.1f gen_free=%.1f gen_unregister=%.1f gen_delete=%.1f "
					            "gen_rebind=%.1f gen_dirty_skipped=%.1f gen_untrack_skipped=%.1f "
						            "rejected_gen_overlap=%.1f rejected_gen_overrun=%.1f gen_log_scans=%.1f\n",
					            per(Take(dl.gen_insert)), per(Take(dl.gen_free)), per(Take(dl.gen_unregister)),
					            per(Take(dl.gen_delete)), per(Take(dl.gen_rebind)), per(Take(dl.gen_dirty_skipped)),
					            per(Take(dl.gen_untrack_skipped)), per(Take(dl.rej_gen_overlap)),
						            per(Take(dl.rej_gen_overrun)), per(Take(dl.gen_log_scans)));
					}
				if (Common::RuntimeFlags::Current().srt_compiled) {
					static ShaderRecompiler::IR::SrtCompiledStats last_cs {};
					const auto cs = ShaderRecompiler::IR::GetSrtCompiledStats();
					const auto d  = [&](uint64_t now, uint64_t before) { return per(now - before); };
					uint64_t fb_now = 0, fb_before = 0;
					for (size_t i = 0; i < cs.fallback.size(); ++i) { fb_now += cs.fallback[i]; fb_before += last_cs.fallback[i]; }
					const double served = static_cast<double>(cs.compiled - last_cs.compiled);
					const double total  = served + static_cast<double>(fb_now - fb_before);
					const double mc     = static_cast<double>(cs.mask_calls - last_cs.mask_calls - (cs.mask_first - last_cs.mask_first));
					const auto   mp     = [&](uint64_t now, uint64_t before) { return mc > 0 ? 100.0 * static_cast<double>(now - before) / mc : 0.0; };
					std::printf("NHL27COMPILED coverage=%.1f%% compiled=%.1f nocfg=%.1f cfg=%.1f failed=%.1f fb_ineligible=%.1f "
					            "fb_nocfg_unsup=%.1f fb_cfg_no_trace=%.1f fb_cfg_eval=%.1f fb_cfg_miss=%.1f builds=%.1f "
					            "reads=%.1f words=%.1f groups_hit=%.1f groups_miss=%.1f spec_recompute=%.1f spec_reuse=%.1f "
					            "verify_checks=%.1f verify_mismatch=%.1f mask_img=%.1f%% mask_smp=%.1f%% mask_fmt=%.1f%% "
					            "mask_addr=%.1f%% mask_size=%.1f%% mask_flat=%.1f%% mask_ud=%.1f%% mask_all_same=%.1f%% "
					            "desc_reuse=%.1f desc_eval=%.1f memo_skipped=%.1f\n",
					            total > 0 ? 100.0 * served / total : 0.0, d(cs.compiled, last_cs.compiled), d(cs.nocfg, last_cs.nocfg),
					            d(cs.cfg, last_cs.cfg), d(cs.failed, last_cs.failed), d(cs.fallback[0], last_cs.fallback[0]),
					            d(cs.fallback[1], last_cs.fallback[1]), d(cs.fallback[2], last_cs.fallback[2]),
					            d(cs.fallback[3], last_cs.fallback[3]), d(cs.fallback[4], last_cs.fallback[4]),
					            d(cs.builds, last_cs.builds), d(cs.reads, last_cs.reads), d(cs.read_words, last_cs.read_words),
					            d(cs.groups_hit, last_cs.groups_hit), d(cs.groups_miss, last_cs.groups_miss),
					            d(cs.spec_recomputed, last_cs.spec_recomputed), d(cs.spec_reused, last_cs.spec_reused),
					            d(cs.verify_checks, last_cs.verify_checks), d(cs.verify_mismatch, last_cs.verify_mismatch),
					            mp(cs.mask_images, last_cs.mask_images), mp(cs.mask_samplers, last_cs.mask_samplers),
					            mp(cs.mask_buffer_fmt, last_cs.mask_buffer_fmt), mp(cs.mask_buffer_addr, last_cs.mask_buffer_addr),
					            mp(cs.mask_buffer_size, last_cs.mask_buffer_size), mp(cs.mask_flat, last_cs.mask_flat),
					            mp(cs.mask_user_data, last_cs.mask_user_data), mp(cs.mask_all_same, last_cs.mask_all_same),
					            d(cs.desc_reuse, last_cs.desc_reuse), d(cs.desc_eval, last_cs.desc_eval),
					            d(cs.memo_skipped, last_cs.memo_skipped));
					last_cs = cs;
				}
			}
		}
		std::printf("NHL27BUFFERS vertex_index=%.2fms indirect=%.2fms find=%.2fms bda=%.2fms "
		            "rebind=%.2fms rebind_lookup=%.2fms rebind_upload=%.2fms\n",
		            ms(dp_vertex_index_ns), ms(dp_indirect_ns), ms(dp_find_buffers_ns), ms(dp_bda_ns),
		            ms(dp_rebind_ns), ms(dp_rebind_lookup_ns), ms(dp_rebind_upload_ns));
			// drain_ms is always 0: the queue drain runs inside BufferCache::SynchronizeBuffersInRanges,
			// so its time is counted in sync_ms. pagetable_ms is always 0: PrepareBda never writes the
			// BDA page table (BufferCache::ChangeRegister does, at buffer registration).
			const double bda_lock_ms  = ms(bda_lock_ns);
			const double bda_sync_ms  = ms(bda_sync_ns);
			const double bda_other_ms = std::max(0.0, ms(bda_total_ns) - bda_lock_ms - bda_sync_ms);
			std::printf("NHL27BDA calls=%.2f skips=%.2f scans=%.2f regions=%.2f ranges=%.2f pages=%.2f "
			            "lock_ms=%.3f drain_ms=%.3f sync_ms=%.3f pagetable_ms=%.3f other_ms=%.3f\n",
			            per(bda_calls), per(bda_skips), per(bda_scans), per(bda_regions), per(bda_ranges),
			            per(bda_pages), bda_lock_ms, 0.0, bda_sync_ms, 0.0, bda_other_ms);
			std::printf("NHL27SYNC uploads=%.2f kb=%.2f\n", per(sync_uploads), per(sync_bytes) / 1024.0);
		if (Verbose()) {
			// pipeline / buffer sub-bucket lines (see the PipelineStats / BufStats blocks)
			auto& pl = g_pl;
			const uint64_t p_key = Take(pl.key_ns), p_lk = Take(pl.lookup_ns), p_tg = Take(pl.targets_ns);
			const uint64_t p_vc = Take(pl.vcommit_ns), p_dy = Take(pl.dyn_ns), p_bg = Take(pl.begin_ns);
			const uint64_t p_bd = Take(pl.bind_ns), p_em = Take(pl.emit_ns);
			const uint64_t p_n = Take(pl.lookups), p_rep = Take(pl.key_repeats), p_b = Take(pl.binds);
			const uint64_t p_bs = Take(pl.binds_skipped), p_same = Take(pl.bind_same), p_dc = Take(pl.dyn_calls);
			const uint64_t p_dm = Take(pl.dyn_cmds), p_ds = Take(pl.dyn_cmds_skipped);
			const uint64_t p_vf = Take(pl.memo_verify), p_vm = Take(pl.memo_mismatch);
			const uint64_t c_get = Take(pl.cs_get_ns), c_prep = Take(pl.cs_prep_ns), c_bind = Take(pl.cs_bind_ns);
			const uint64_t c_n = Take(pl.cs_lookups), c_rep = Take(pl.cs_repeats), c_b = Take(pl.cs_binds);
			const uint64_t c_sk = Take(pl.cs_skipped);
			const double   p_named = ms(p_key) + ms(p_lk) + ms(p_tg) + ms(p_vc) + ms(p_dy) + ms(p_bg) + ms(p_bd) + ms(p_em);
			std::printf("NHL27PIPELINE draw: bucket=%.2fms key=%.2fms lookup=%.2fms targets=%.2fms vcommit=%.2fms "
			            "dynamic=%.2fms begin=%.2fms bind=%.2fms emit=%.2fms other=%.2fms | lookups=%.1f "
			            "key_repeat=%.1f binds=%.1f bind_skipped=%.1f bind_same=%.1f dyn_calls=%.1f dyn_cmds=%.1f "
			            "dyn_skipped=%.1f memo_verify=%.2f memo_mismatch=%.2f | dispatch: bucket=%.2fms get=%.2fms "
			            "prep=%.2fms bind=%.2fms lookups=%.1f repeat=%.1f binds=%.1f skipped=%.1f\n",
			            ms(dp_pipeline_ns), ms(p_key), ms(p_lk), ms(p_tg), ms(p_vc), ms(p_dy), ms(p_bg), ms(p_bd),
			            ms(p_em), std::max(0.0, ms(dp_pipeline_ns) - p_named), per(p_n), per(p_rep), per(p_b), per(p_bs),
			            per(p_same), per(p_dc), per(p_dm), per(p_ds), per(p_vf), per(p_vm), ms(dc_pipeline_ns),
			            ms(c_get), ms(c_prep), ms(c_bind), per(c_n), per(c_rep), per(c_b), per(c_sk));
			auto& bf = g_buf;
			const uint64_t o_calls = Take(bf.ob_calls), o_wr = Take(bf.ob_written), o_st = Take(bf.ob_stream);
			const uint64_t o_fc = Take(bf.ob_find_calls), o_sy = Take(bf.ob_syncs), o_cl = Take(bf.ob_clean);
			const uint64_t o_sck = Take(bf.ob_stream_check_ns), o_scp = Take(bf.ob_stream_copy_ns);
			const uint64_t o_fn = Take(bf.ob_find_ns), o_tn = Take(bf.ob_touch_ns), o_sn = Take(bf.ob_sync_ns);
			const uint64_t o_bk = Take(bf.ob_book_ns), o_inv = Take(bf.ob_invalidate_ns);
			std::printf("NHL27BUFFERS2 rebind_lookup=%.2fms obtain: calls=%.1f written=%.1f stream_path=%.1f "
			            "find_calls=%.1f syncs=%.1f clean=%.1f stream_check=%.2fms stream_copy=%.2fms find=%.2fms "
			            "touch=%.2fms sync=%.2fms book=%.2fms invalidate_gpu=%.2fms\n",
			            ms(dp_rebind_lookup_ns), per(o_calls), per(o_wr), per(o_st), per(o_fc), per(o_sy), per(o_cl), ms(o_sck),
			            ms(o_scp), ms(o_fn), ms(o_tn), ms(o_sn), ms(o_bk), ms(o_inv));
			std::printf("NHL27BUFFERS2 obtain_cache: hits=%.1f misses=%.1f (slot=%.1f registration=%.1f buffer=%.1f "
			            "cpu=%.1f; not stored: stream=%.1f registration=%.1f cpu=%.1f) registrations=%.1f\n",
			            per(obtain_cache_hits), per(obtain_cache_misses), per(om_slot), per(om_reg), per(om_buf),
			            per(om_cpu), per(on_stream), per(on_reg), per(on_cpu), per(b_regs));
			static const char* const slot_kind[3] = {"plain", "written", "texel"};
			for (int k = 0; k < 3; k++) {
				std::printf("NHL27SLOTMEMO %s: hit=%.1f by-range=%.1f; missed: none=%.1f same=%.1f inside=%.1f "
				            "other=%.1f\n",
				            slot_kind[k], per(slot_counts[0][k]), per(slot_counts[5][k]), per(slot_counts[1][k]),
				            per(slot_counts[2][k]), per(slot_counts[3][k]), per(slot_counts[4][k]));
			}
			static const char* const sb_name[2] = {"rebind", "bda"};
			const uint64_t bda_syncs = Take(bf.bda_syncs);
			(void)Take(bf.bda_drain_ns);
			for (int k = 0; k < 2; ++k) {
				const uint64_t s_c = Take(bf.sb_calls[k]), s_u = Take(bf.sb_uploads[k]), s_b = Take(bf.sb_bytes[k]);
				const uint64_t s_w = Take(bf.sb_walk_ns[k]), s_s = Take(bf.sb_stage_ns[k]), s_e = Take(bf.sb_end_ns[k]);
				const uint64_t s_r = Take(bf.sb_rec_ns[k]), s_i = Take(bf.sb_image_ns[k]);
				// sb_walk_ns holds the whole ForEachUploadRange call (tracker walk + the staging callback).
				std::printf("NHL27BUFFERS2 sync[%s]: calls=%.1f uploads=%.2f kb=%.1f walk=%.3fms "
				            "stage_memcpy=%.3fms end_rendering=%.3fms barriers_copy=%.3fms from_image=%.3fms%s",
				            sb_name[k], per(s_c), per(s_u), per(s_b) / 1024.0, ms(s_w > s_s ? s_w - s_s : 0), ms(s_s),
				            ms(s_e), ms(s_r), ms(s_i), k == 1 ? "" : "\n");
				if (k == 1) {
					const uint64_t inner = s_w + s_e + s_r + s_i;
					std::printf(" bda_syncs=%.2f drain_and_range_walk=%.3fms (bda_sync_total=%.3fms)\n", per(bda_syncs),
					            ms(bda_sync_ns > inner ? bda_sync_ns - inner : 0), ms(bda_sync_ns));
				}
			}
		}
		std::printf("NHL27PROTECT calls=%.2f ms=%.3f\n", per(protect_calls), ms(protect_ns));
		{
			auto&          fc  = g_fault_cost;
			const uint64_t gr  = Take(fc.n[0]);
			const uint64_t gw  = Take(fc.n[1]);
			const uint64_t cr  = Take(fc.n[2]);
			const uint64_t cw  = Take(fc.n[3]);
			const uint64_t all = gr + gw + cr + cw;
			const uint64_t dis = Take(fc.distinct);
			const uint64_t tot = Take(fc.total_ns);
			const uint64_t buf = Take(fc.buf_inv_ns);
			const uint64_t tex = Take(fc.tex_inv_ns);
			const uint64_t rdm = Take(fc.rd_readmem_ns);
			const uint64_t rem = tot > buf + tex + rdm ? tot - buf - tex - rdm : 0;
			const uint64_t pn  = Take(fc.protect_calls);
			std::printf("NHL27FAULTCOST n=%.1f gpu_r=%.1f gpu_w=%.1f guest_r=%.1f guest_w=%.1f "
			            "pages=%.1f refault=%.2f total=%.2fms(gpu=%.2f guest=%.2f) us/fault=%.1f "
			            "veh=%.2fms lock=%.2fms buf_inv=%.2fms tex_inv=%.2fms readmem=%.2fms "
			            "rd_readmem=%.2fms protect=%.2fms(%.1f) other=%.2fms\n",
			            per(all), per(gr), per(gw), per(cr), per(cw), per(dis),
			            dis != 0 ? static_cast<double>(all) / static_cast<double>(dis) : 0.0, ms(tot),
			            ms(Take(fc.total_gpu_ns)), ms(Take(fc.total_guest_ns)),
			            all != 0 ? static_cast<double>(tot) / 1e3 / static_cast<double>(all) : 0.0,
			            ms(Take(fc.veh_ns)), ms(Take(fc.lock_ns)), ms(buf), ms(tex),
			            ms(Take(fc.readmem_ns)), ms(rdm), ms(Take(fc.protect_ns)), per(pn), ms(rem));
		}
		std::printf("NHL27FAULTPARSE passes=%.2f zero=%.2f faults=%.2f ring_wait=%.2fms\n",
		            per(fp_passes), per(fp_zero), per(fp_faults), ms(fp_ring_wait_ns));
		std::fflush(stdout);
	}
}

// Called on every presented flip (guest PM4 EOP flip and CPU flip both end here).
inline void OnPresentedFlip() {
	// Refresh runtime experiment flags at the frame boundary. This is a safe point,
	// and it keeps the published snapshot live even when no probe site on this
	// frame's path runs, so a collector always sees the NHL27CFG tag it waits for.
	// Throttled internally: a timestamp compare per flip.
	Common::RuntimeFlags::Refresh();

	if (!Enabled()) {
		return;
	}
	g_fault_cost.epoch.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(g_window_mutex);
	const auto now = Clock::now();
	if (g_last_flip != Clock::time_point {}) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_last_flip);
		g_counters.flip_ns.fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
		g_counters.flip_intervals.fetch_add(1, std::memory_order_relaxed);
		const auto vblanks = (static_cast<uint64_t>(ns.count()) + 8'333'333u) / 16'666'667u;
		g_counters.flip_vblanks[std::clamp<uint64_t>(vblanks, 1u, 5u) - 1u].fetch_add(
		    1, std::memory_order_relaxed);
	}
	g_last_flip = now;
	PrintIfDueLocked(now);
}

// Called from the GPU thread after each Process slice so the line still prints if flips pause.
inline void MaybePrint() {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_window_mutex);
	PrintIfDueLocked(Clock::now());
}

} // namespace Libs::Graphics::FrameBreakdown
