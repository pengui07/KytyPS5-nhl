#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EXECREGIONS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EXECREGIONS_H_

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

// KYTY_DBG_EXEC_REGIONS=1: the execution thread's time by region of the draw and dispatch paths,
// from the time stamp counter (a few cycles per region), every 5 s as ms per second (NHL27REGIONS).
namespace Libs::Graphics::ExecRegions {

enum Region : uint32_t {
	PrePending,      // DrawIndex: pending operations.
	PreLock,         // DrawIndex: debug info and the context lock.
	PreSpecial,      // DrawIndex: metadata, depth copy and resolve draws.
	EntryPre,        // DrawIndex up to AcquireDrawRenderState: special draws, topology, indices.
	EntryAcquire,    // AcquireDrawRenderState: the resolution and its guest read check.
	StateMasks,      // PrepareDrawRenderState: output masks (and shaders when not resolved).
	EntryState,      // PrepareDrawRenderState: shaders, render target memo or lookup.
	DrawEntry,       // From there up to the stages' PrepareBindings.
	StagesMemo,      // PrepareBindings: the program memos.
	StagesImages,    // PrepareBindings: direct images.
	StagesResolve,   // PrepareBindings: ResolveTexture of direct images.
	StagesMap,       // PrepareBindings: description lookups of direct images.
	StagesTables,    // PrepareBindings: image tables.
	StagesSamplers,  // PrepareBindings: samplers.
	DrawStages,      // PrepareBindings: user data, and between the stages.
	DrawFindBuffers, // PrepareGraphicsBindings: FindBuffers.
	DrawDma,         // PrepareGraphicsBindings: DMA bases.
	DrawBda,         // PrepareGraphicsBindings: the BDA synchronization (PrepareBda).
	DrawHeap,        // PrepareGraphicsBindings: image heap rebind.
	DrawImages,      // PrepareGraphicsBindings: RebindImages, render target rediscovery.
	BuffersMemo,     // RebindBuffers: slots whose memo held.
	BuffersObtain,   // RebindBuffers: slots looked up.
	DrawBuffers,     // PrepareGraphicsBindings: RebindBuffers (the rest: setup, uploads).
	DrawVertex,      // Vertex and index buffers, indirect arguments.
	DrawPipeline,    // GetGraphicsPipeline.
	DrawTargets,     // AcquireRenderTargets.
	CommitStart,     // CommitBindings: up to the per-stage counts.
	CommitCounts,    // CommitBindings: the per-stage counts.
	CommitSetup,     // CommitBindings: scalar read check, image heap.
	CommitTransit,   // CommitBindings: image layout transitions.
	CommitWrites,    // CommitBindings: descriptor writes and image infos.
	DrawDescriptors, // CommitBindings: push constants and descriptors.
	DrawRecord,      // Dynamic state, barriers, rendering, the draw.
	DrawExit,        // After ExecutePreparedDraw.
	DispProgram,     // DispatchDirect: up to the program (resolved ahead or looked up).
	DispFill,        // DispatchDirect: fill, clear, pattern and copy recognition.
	DispPipeline,    // DispatchDirect: GetComputePipeline.
	DispStages,      // DispatchDirect: PrepareBindings.
	DispFind,        // DispatchDirect: FindBuffers, indirect write targets, DMA.
	DispImages,      // DispatchDirect: RebindImages.
	DBufPlain,       // RebindBuffers in a dispatch: plain-read slot memo hits.
	DBufWritten,     // ... written slot memo hits.
	DBufTexel,       // ... texel-read slot memo hits.
	DBufGlobal,      // ... by-range memo hits.
	DBufObtain,      // ... slots looked up.
	DispBuffers,     // DispatchDirect: shared memory, RebindBuffers (the rest).
	DispCommit,      // DispatchDirect: CommitBindings.
	Dispatch,        // DispatchDirect / DispatchIndirect: the rest.
	Count
};

inline const char* const Names[Count] = {"pending", "lock", "special", "pre", "acquire", "masks", "state", "entry", "memo", "simages", "sresolve", "smap",
                                         "tables", "samplers", "stages", "find", "dma", "bda", "heap", "images", "bmemo", "bobtain", "buffers", "vertex",
                                         "pipeline", "targets", "cstart", "ccounts", "csetup", "ctransit", "cwrites", "descriptors", "record", "exit",
                                         "dprogram", "dfill", "dpipeline", "dstages", "dfind", "dimages", "dbplain",
                                         "dbwritten", "dbtexel", "dbglobal", "dbobtain", "dbuffers",
                                         "dcommit", "dispatch"};

// Built only with the CMake option KYTY_EXEC_REGIONS: otherwise every mark compiles to nothing.
#if defined(KYTY_EXEC_REGIONS) && KYTY_EXEC_REGIONS
[[nodiscard]] inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_EXEC_REGIONS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
#else
[[nodiscard]] constexpr bool Enabled() {
	return false;
}
#endif

inline uint64_t g_ticks[Count] {};
inline uint64_t g_counts[Count] {};
inline bool     g_in_draw = false;
// Draws that repeat the draw before them (same programs, T#s, S#s and targets) are charged apart.
inline bool     g_lean = false;
inline uint64_t g_lean_ticks[Count] {};
inline uint64_t g_lean_counts[Count] {};
inline uint64_t g_lean_draws = 0;
// Event counts (per second in the line).
enum Event : uint32_t {
	ImageSlots,
	ImageMemoDesc,
	ImageMapLookups,
	ImageReused,
	ImageResolved,
	FindSlots,
	FindSame,
	FindEpochs,
	ResolveWritten,
	ResolveDynamic,
	ResolveNoDesc,
	ResolveInvalid,
	ReuseNoImage,
	ReuseRebind,
	ReuseState,
	ReuseEpoch,
	HeapRebinds,
	HeapCommits,
	HeapEpochs,
	HeapAway,
	HeapActive,
	BufferSlots,
	BufferMemoHits,
	BufferGlobalHits,
	BufferObtains,
	SameContext,
	SamePrograms,
	SameTextures,
	SameBufferVs,
	SameUserData,
	SameAllButData,
	TargetsReused,
	LeanStages,
	LeanSlots,
	// Buffer slots that missed both memos (counted for draws and dispatches apart, NoteOp()): no
	// memo kind (formatted raw reads, null ranges), a by-range memo of the range whose check failed
	// (plain, written, texel), no by-range memo of the range, and lookups whose result is not kept.
	ObtainNoKind,
	ObtainFailedPlain,
	ObtainFailedWritten,
	ObtainFailedTexel,
	ObtainNoMemo,
	ObtainNotKept,
	ObtainStream,
	// Of the no-memo lookups: the range is CPU-modified (an upload is due), the range was bound
	// before but its by-range memo slot holds another range now, the slot memo had the same range.
	ObtainNoMemoDirty,
	ObtainNoMemoEvicted,
	ObtainNoMemoSlotSame,
	// Stages that do not keep their bindings (see PrepareBindings()): other T#s than the program's
	// last draw, the memo held by another stage, the last draw's bindings not held to its end, the
	// texture epoch moved since (T#s the same).
	LeanOtherImages,
	LeanOtherHolder,
	LeanNotHeld,
	LeanEpochMoved,
	DmaStages,
	DmaMemoHits,
	EventCount
};
inline const char* const EventNames[EventCount] = {"img_slots", "img_memo",  "img_map",   "img_reused",
                                                   "img_resolved", "find_slots", "find_same", "find_epochs",
                                                   "r_written", "r_dynamic", "r_nodesc", "r_invalid",
                                                   "u_noimage", "u_rebind", "u_state", "u_epoch",
                                                   "h_rebinds", "h_commits", "h_epochs", "h_away", "h_active",
                                                   "b_slots", "b_memo", "b_global", "b_obtain",
                                                   "same_ctx", "same_prog", "same_tex", "same_vs", "same_user",
                                                   "same_but_data", "rt_reused", "lean_stages", "lean_slots",
                                                   "o_nokind", "o_plain", "o_written", "o_texel", "o_nomemo",
                                                   "o_notkept", "o_stream", "o_dirty", "o_evicted", "o_slotsame",
                                                   "l_images", "l_holder", "l_notheld", "l_epoch", "dma_stages", "dma_hits"};
// Ranges bound before (diagnostics of the by-range memo's evictions): a direct-mapped set of 1M keys.
inline bool SeenRange(uint64_t address, uint64_t size, uint8_t kind) {
	static auto* keys = new uint64_t[1u << 20u] {};
	const uint64_t key  = (address * 0x9E3779B97F4A7C15ull) ^ (size << 40u) ^ kind ^ 1u;
	auto&          slot = keys[(key >> 20u) & ((1u << 20u) - 1u)];
	const bool     seen = slot == key;
	slot                = key;
	return seen;
}
inline uint64_t          g_events[EventCount] {};
inline uint64_t          g_dispatch_events[EventCount] {};
inline void              Note(Event event) {
	if (Enabled() && g_in_draw) g_events[event]++;
}
// An event of a draw or of a dispatch (printed per dispatch in NHL27REGIONSDISP).
inline void NoteOp(Event event) {
	if (Enabled()) (g_in_draw ? g_events : g_dispatch_events)[event]++;
}
inline uint64_t g_draws      = 0;
inline uint64_t g_dispatches = 0;
// KYTY_DBG_WARM=1: an operation timed twice in a row (first call, then again with its data in the
// cache): how much of its cost is waiting for memory. Per kind: [0] first ticks, [1] second ticks,
// [2] count.
enum WarmKind : uint32_t { WarmReuse, WarmFindSlot, WarmTransit, WarmPlain, WarmCount };
inline const char* const WarmNames[WarmCount] = {"reuse", "findslot", "transit", "plain"};
inline uint64_t          g_warm[WarmCount][3] {};
[[nodiscard]] inline bool WarmEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_WARM");
		return Enabled() && value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// `check()` (a test without lasting effects when repeated); when it holds and KYTY_DBG_WARM is on,
// runs it again and charges both runs. Without the region timers it is just `check()`.
template <typename Check>
[[nodiscard]] inline bool TimedTwice(WarmKind kind, Check&& check) {
	if (!(Enabled() && WarmEnabled())) {
		return check();
	}
	const auto first = __rdtsc();
	if (!check()) {
		return false;
	}
	const auto second = __rdtsc();
	(void)check();
	const auto end = __rdtsc();
	g_warm[kind][0] += second - first;
	g_warm[kind][1] += end - second;
	g_warm[kind][2]++;
	return true;
}

inline void Print() {
	static uint64_t start_tsc = __rdtsc();
	static auto     start     = std::chrono::steady_clock::now();
	const auto      now       = std::chrono::steady_clock::now();
	if (now - start < std::chrono::seconds(5)) {
		return;
	}
	const auto   tsc = __rdtsc();
	const double ns  = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
	// ms of the region per second of wall time.
	const double scale = 1000.0 / static_cast<double>(tsc - start_tsc);
	char         line[4096];
	size_t       at     = 0;
	const auto   append = [&](const char* format, auto... values) {
		if (at < sizeof(line)) {
			const int written = std::snprintf(line + at, sizeof(line) - at, format, values...);
			at += written > 0 ? static_cast<size_t>(written) : 0;
		}
	};
	append("NHL27REGIONS draws/s=%.0f dispatches/s=%.0f", static_cast<double>(g_draws) * 1e9 / ns,
	       static_cast<double>(g_dispatches) * 1e9 / ns);
	double total = 0;
	for (uint32_t i = 0; i < Count; i++) {
		const double ms = static_cast<double>(g_ticks[i]) * scale;
		total += ms;
		append(" %s=%.0f/%.0fk", Names[i], ms, static_cast<double>(g_counts[i]) * 1e6 / ns);
		g_ticks[i] = g_counts[i] = 0;
	}
	for (uint32_t i = 0; i < EventCount; i++) {
		append(" %s=%.0fk", EventNames[i], static_cast<double>(g_events[i]) * 1e6 / ns);
		g_events[i] = 0;
	}
	std::printf("%s total=%.0f (ms/s)\n", line, total);
	at = 0;
	append("NHL27REGIONSLEAN draws/s=%.0f", static_cast<double>(g_lean_draws) * 1e9 / ns);
	total = 0;
	for (uint32_t i = 0; i < Count; i++) {
		const double ms = static_cast<double>(g_lean_ticks[i]) * scale;
		total += ms;
		append(" %s=%.0f/%.0fk", Names[i], ms, static_cast<double>(g_lean_counts[i]) * 1e6 / ns);
		g_lean_ticks[i] = g_lean_counts[i] = 0;
	}
	std::printf("%s total=%.0f (ms/s)\n", line, total);
	at = 0;
	append("NHL27REGIONSDISP dispatches/s=%.0f", static_cast<double>(g_dispatches) * 1e9 / ns);
	for (uint32_t i = 0; i < EventCount; i++) {
		if (g_dispatch_events[i] != 0) {
			append(" %s=%.1fk", EventNames[i], static_cast<double>(g_dispatch_events[i]) * 1e6 / ns);
		}
		g_dispatch_events[i] = 0;
	}
	std::printf("%s\n", line);
	if (WarmEnabled()) {
		at = 0;
		append("NHL27WARM");
		for (uint32_t i = 0; i < WarmCount; i++) {
			const double count = static_cast<double>(g_warm[i][2]);
			if (count > 0) {
				const double ns_per_tick = ns / static_cast<double>(tsc - start_tsc);
				append(" %s: first=%.1fns second=%.1fns n=%.0fk", WarmNames[i],
				       static_cast<double>(g_warm[i][0]) * ns_per_tick / count,
				       static_cast<double>(g_warm[i][1]) * ns_per_tick / count, count / 1000.0);
			}
			g_warm[i][0] = g_warm[i][1] = g_warm[i][2] = 0;
		}
		std::printf("%s\n", line);
	}
	std::fflush(stdout);
	g_draws = g_dispatches = g_lean_draws = 0;
	start_tsc = tsc;
	start     = now;
}

// The execution thread alone charges regions: Begin() starts a draw or dispatch, To() charges the
// time since the last mark to a region.
inline uint64_t g_last = 0;

inline void Begin() {
	if (Enabled()) g_last = __rdtsc();
}

inline void To(Region region) {
	if (!Enabled()) return;
	const auto now = __rdtsc();
	(g_lean ? g_lean_ticks : g_ticks)[region] += now - g_last;
	(g_lean ? g_lean_counts : g_counts)[region]++;
	g_last = now;
}

// Marks inside functions that dispatches share.
inline void InDraw(Region region) {
	if (Enabled() && g_in_draw) To(region);
}

// A mark with one region for draws and one for dispatches.
inline void InOp(Region draw, Region dispatch) {
	if (Enabled()) To(g_in_draw ? draw : dispatch);
}

// A draw or dispatch: the time after its last mark goes to `last`.
class Scope {
public:
	explicit Scope(Region last): m_last(last) {
		if (!Enabled()) return;
		Begin();
		g_in_draw = last != Dispatch;
	}
	~Scope() {
		if (!Enabled()) return;
		g_in_draw = false;
		To(m_last);
		(m_last == Dispatch ? g_dispatches : g_lean ? g_lean_draws : g_draws)++;
		g_lean = false;
		Print();
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Region m_last;
};

} // namespace Libs::Graphics::ExecRegions

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EXECREGIONS_H_ */
