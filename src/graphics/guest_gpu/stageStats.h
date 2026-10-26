#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_STAGESTATS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_STAGESTATS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

// KYTY_DBG_STAGE_STATS=1: where the GPU pipeline's execution and resolve threads wait, measured
// only when they actually wait (an uncontended lock or a ready operation costs one branch).
// Printed every 5 s as NHL27STALLS by the execution thread. With the frame rate capped (menus,
// a paused match), execution-thread busy time per frame still shows what a change saves.
namespace Libs::Graphics::StageStats {

enum Counter : uint32_t {
	ExecIdle,        // execution thread waiting for operations
	ExecProgramLock, // execution thread waiting for the program lock
	ResolveProgramLock,
	ResolveQueueFull, // resolve thread waiting for room in the execution queue
	ResolveIdle,      // resolve thread waiting for operations from the command processor
	// Execution-thread idle time by what the stage before it was doing when the queue ran dry:
	ExecIdleResolve,  // the resolve thread was resolving the next operation
	ExecIdleCp,       // the command processor was processing packets
	ExecIdleNoWork,   // no guest submission was queued
	ExecIdleBlocked,  // every queue waited on a WAIT_REG_MEM the guest had not satisfied
	ExecIdleDecision, // the command processor waited for a write it decides on (clock, label)
	ExecIdleOther,    // the command processor waited for the execution thread otherwise
	// Execution-thread waits for the draw (dispatch) resolver's slot of the next operation, and the
	// part of all its waits within the first 256 operations after a flip.
	ExecIdleDrawSlot,
	ExecIdleDispatchSlot,
	ExecIdleEarly,
	CounterCount,
};

enum class Role : uint8_t { Other, Execution, Resolve };

// What the command processor thread is doing (stats only).
enum class CpState : uint8_t { Processing, NoWork, Blocked, WaitDecision, WaitOther };
inline std::atomic<CpState> g_cp_state {CpState::Processing};
inline std::atomic<bool>    g_resolve_working {false};

inline std::atomic<uint64_t> g_ns[CounterCount] {};
// Resolved-read checks (GuestReadLog::StillValid) by kind (0 draws, 1 dispatches): calls, entries, bytes, pages.
inline std::atomic<uint64_t> g_check_calls[2] {}, g_check_entries[2] {}, g_check_bytes[2] {}, g_check_pages[2] {};
inline std::atomic<uint64_t> g_check_skipped[2] {}; // reads skipped (see GuestReadLog::StillValid)
// Reads checked because their bytes may have been GPU-owned when read, or a later operation writes their page.
inline std::atomic<uint64_t> g_check_unsafe[2] {}, g_check_later[2] {};
// Buffer bindings the execution thread looks up, those its stage slot had the draw before, written.
inline std::atomic<uint64_t> g_bind_total {0}, g_bind_same {0}, g_bind_written {0};
inline std::atomic<uint64_t> g_bda_syncs {0}; // BDA synchronizations that scanned for CPU-dirty pages
inline std::atomic<uint64_t> g_ops {0}; // operations the execution thread ran (written by it only)
// Execution-thread time and count by operation kind (GpuOperationKind; execution thread only).
inline std::atomic<uint64_t> g_kind_ns[8] {};
inline std::atomic<uint64_t> g_kind_count[8] {};
// Callbacks by GpuCallbackTag (execution thread only).
inline std::atomic<uint64_t> g_callback_ns[16] {};
inline std::atomic<uint64_t> g_callback_count[16] {};
// HostCopyQueue::Drain before operations, by callback tag (15: draws and other non-callbacks):
// drains, those that had to wait for the copy thread, and the wait time.
inline std::atomic<uint64_t> g_drain_count[16] {}, g_drain_waits[16] {}, g_drain_ns[16] {};
inline thread_local Role     t_role = Role::Other;

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_STAGE_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline void Add(Counter counter, std::chrono::steady_clock::time_point start) {
	g_ns[counter].fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                                  std::chrono::steady_clock::now() - start)
	                                                  .count()),
	                        std::memory_order_relaxed);
}

// Sets the command processor state for a scope (stats only).
class CpStateScope {
public:
	explicit CpStateScope(CpState state) {
		if (Enabled()) {
			m_previous = g_cp_state.exchange(state, std::memory_order_relaxed);
			m_active   = true;
		}
	}
	~CpStateScope() {
		if (m_active) g_cp_state.store(m_previous, std::memory_order_relaxed);
	}
	CpStateScope(const CpStateScope&)            = delete;
	CpStateScope& operator=(const CpStateScope&) = delete;

private:
	CpState m_previous = CpState::Processing;
	bool    m_active   = false;
};

// The execution-idle counter for what the earlier stages are doing now.
[[nodiscard]] inline Counter ExecIdleCause() {
	if (g_resolve_working.load(std::memory_order_relaxed)) return ExecIdleResolve;
	switch (g_cp_state.load(std::memory_order_relaxed)) {
		case CpState::Processing: return ExecIdleCp;
		case CpState::NoWork: return ExecIdleNoWork;
		case CpState::Blocked: return ExecIdleBlocked;
		case CpState::WaitDecision: return ExecIdleDecision;
		default: return ExecIdleOther;
	}
}

// The program-lock counter of the calling stage thread, or CounterCount for other threads.
[[nodiscard]] inline Counter ProgramLockCounter() {
	switch (t_role) {
		case Role::Execution: return ExecProgramLock;
		case Role::Resolve: return ResolveProgramLock;
		default: return CounterCount;
	}
}

} // namespace Libs::Graphics::StageStats

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_STAGESTATS_H_ */
