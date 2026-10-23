#include "common/experiments.h"

#include "common/emulatorConfig.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <cinttypes>

namespace Common::Experiments {

namespace {

// "1" enables, "0" disables, anything else (including unset) keeps the default.
bool EnvSwitch(const char* name, bool on_by_default) {
	const char* value = std::getenv(name);
	if (value != nullptr && (value[0] == '0' || value[0] == '1') && value[1] == '\0') {
		return value[0] == '1';
	}
	return on_by_default;
}

uint64_t EnvironmentMask() {
	uint64_t mask = 0;
	const auto set = [&mask](const char* name, Bit bit, bool on_by_default) {
		if (EnvSwitch(name, on_by_default)) mask |= bit;
	};
	// The NHL 27 recommended set is on unless its variable is set to 0.
	// Off now: building its key cost the execution thread more than the pushes it saved (the
	// recording thread has room for them); two boots, draw cost -2..-5%.
	set("KYTY_SKIP_REDUNDANT_BINDS", SkipRedundantBinds, false);
	set("KYTY_SRT_BATCH_READS", SrtBatchReads, true);
	set("KYTY_DIRECT_SRT_READS", DirectSrtReads, true);
	set("KYTY_NATIVE_MINMAX", NativeMinMax, true);
	set("KYTY_IMAGE_LOOKUP_HINT", ImageLookupHint, true);
	set("KYTY_WIDE_FAULT_WINDOWS", WideFaultWindows, true);
	set("KYTY_CLEAN_FROM_DIRECT", CleanFromDirect, true);
	set("KYTY_LDS_LANE_SLOTS", LdsLaneSlots, true);
	// Not part of the recommended set: opt in with 1.
	set("KYTY_CPU_COPY_FORWARD", CpuCopyForward, false);
	set("KYTY_WIDE_WRITE_DOWNLOADS", WideWriteDownloads, true);
	set("KYTY_ASYNC_SUBMIT", AsyncSubmit, false);
	set("KYTY_LAZY_SHARED_WRITES", LazySharedWrites, false);
	set("KYTY_NATIVE_FP16", NativeFp16, false);
	set("KYTY_FAULT_PAGE_READS", FaultPageReads, false);
	set("KYTY_EXEC_SELECTS", ExecSelects, false);
	set("KYTY_EXEC_BRANCHES", ExecBranches, true);
	set("KYTY_EXEC_BRANCH_MEMORY", ExecBranchMemory, true);
	set("KYTY_EXEC_BRANCH_CONTINUE", ExecBranchContinue, true);
	set("KYTY_F16_LANE_VIEWS", F16LaneViews, true);
	set("KYTY_LAZY_GUEST_BARRIERS", LazyGuestBarriers, false);
	set("KYTY_EARLY_LATE_FRAGMENT_TESTS", EarlyLateFragmentTests, true);
	set("KYTY_BDA_LOAD_GROUPS", BdaLoadGroups, true);
	// On since 2026-10-22: with the stage cores kept for the stages, the resolve thread limited the
	// heavier scenes; sharing its core with the game, the second resolver took 60 Hz play 57.4 ->
	// 59.4 fps (paired rounds 413 / 447). Measured on 16 logical CPUs (its default CPU is 14): with
	// fewer it would take time from the game's threads, so it stays off there unless asked for.
	const unsigned cpus = std::thread::hardware_concurrency();
	set("KYTY_DRAW_RESOLVER", DrawResolver, cpus >= 16);
	set("KYTY_GPU_COLOR_CLEAR", GpuColorClear, false);
	set("KYTY_POOLED_BDA_BUFFERS", PooledBdaBuffers, false);
	set("KYTY_ASYNC_GUEST_READS", AsyncGuestReads, false);
	set("KYTY_BDA_DIRTY_REGIONS", BdaDirtyRegions, false);
	// On again: the match-start hang it was suspected of was the label readback race
	// (WriteDataBuffer), and it cuts the execution thread's readback waits to ~20 ms/s.
	set("KYTY_CP_BACKING_READS", CpBackingReads, true);
	set("KYTY_EARLY_SUBMIT_WRITERS", EarlySubmitWriters, false);
	set("KYTY_READBACK_COMPUTE", ReadbackCompute, true);
	set("KYTY_ROBUST_LOADS", RobustLoads, true);
	set("KYTY_FUSED_MAD", FusedMad, false);
	set("KYTY_FAULT_PAGE_DIRECT", FaultPageDirect, true);
	set("KYTY_BACKING_READ_MRU", BackingReadMru, true);
	set("KYTY_DISPATCH_AHEAD", DispatchAhead, true);
	set("KYTY_DISPATCH_RESOLVER", DispatchResolver, false);
	set("KYTY_GPU_CONDITIONS_BOTH_PATHS", GpuConditionsBothPaths, true);
	set("KYTY_COMPUTE_UNLOCKED", ComputeUnlocked, true);
	set("KYTY_AHEAD_UNLOCKED", AheadUnlocked, true);
	set("KYTY_BACKING_LOOKUP_CACHE", BackingLookupCache, true);
	set("KYTY_PROTECT_SHARED_LOCK", ProtectSharedLock, false);
	set("KYTY_CLOCK_WAIT_PREDICT", ClockWaitPredict, false);
	// On: the command processor's and guest threads' reads of bytes the GPU never wrote, on pages
	// protected for other GPU-written bytes, use the backing (exact, no speculation): the execution
	// thread's readback waits for them 65 -> 24 ms/s in play.
	set("KYTY_FALSE_SHARE_ANY_THREAD", FalseShareAnyThread, true);
	set("KYTY_SPECULATE_DIRTY_READS", SpeculateDirtyReads, false);
	set("KYTY_CP_CLEAN_REG_READS", CpCleanRegReads, false);
	set("KYTY_CP_SKIP_IDLE_SPIN", CpSkipIdleSpin, false);
	// Two boots: execution-thread cost per draw -4..-7%, per dispatch -18..-20% (8 cores, 16 logical
	// CPUs). With fewer than 12 logical CPUs the game would keep too few.
	set("KYTY_GUEST_OFF_STAGE_CORES", GuestOffStageCores, cpus >= 12);
	set("KYTY_GUEST_OFF_COPY_CORE", GuestOffCopyCore, false);
	set("KYTY_GUEST_OFF_CP_CORES", GuestOffCpCores, false);
	return mask;
}

// Bits GuestOffStageCores / GuestOffCopyCore / GuestOffCpCores: the process default CPU sets, which every thread
// without its own CPU selection follows; the pinned GPU stage threads keep their affinity.
void ApplyCpuSets(uint64_t mask) {
#if defined(_WIN32)
	const int wanted = (mask & GuestOffStageCores) != 0 ? 2 : 0;
	uint64_t  excluded = 0;
	const auto exclude = [&](Config::GpuStageThread stage) {
		const int cpu = Config::GpuStageCpu(stage);
		if (wanted == 0 || cpu < 0 || cpu >= 64) {
			return;
		}
		// SMT siblings are adjacent logical CPUs.
		excluded |= uint64_t {1} << static_cast<uint32_t>(cpu ^ 1);
		if (wanted == 2) {
			excluded |= uint64_t {1} << static_cast<uint32_t>(cpu);
		}
	};
	exclude(Config::GpuStageThread::Execution);
	exclude(Config::GpuStageThread::Resolve);
	if ((mask & GuestOffCopyCore) != 0) {
		exclude(Config::GpuStageThread::HostCopy);
	}
	if ((mask & GuestOffCpCores) != 0) {
		exclude(Config::GpuStageThread::CommandProcessor);
		exclude(Config::GpuStageThread::Recording);
	}
	// The extra resolvers (bits 41 and 43) share their cores: keeping the game off them as well
	// (one core fewer for it) or putting them on the stages' siblings measured slower.
	static uint64_t applied = 0;
	if (excluded == applied) {
		return;
	}
	applied = excluded;
	if (excluded == 0) {
		(void)::SetProcessDefaultCpuSets(::GetCurrentProcess(), nullptr, 0);
		std::printf("Experiments: default CPU sets cleared\n");
		std::fflush(stdout);
		return;
	}
	ULONG length = 0;
	(void)::GetSystemCpuSetInformation(nullptr, 0, &length, ::GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(length);
	if (length == 0 || !::GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
	                                                 length, &length, ::GetCurrentProcess(), 0)) {
		return;
	}
	std::vector<ULONG> ids;
	for (ULONG at = 0; at < length;) {
		const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + at);
		if (info->Size == 0) {
			break;
		}
		if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 &&
		    info->CpuSet.LogicalProcessorIndex < 64 &&
		    ((excluded >> info->CpuSet.LogicalProcessorIndex) & 1u) == 0) {
			ids.push_back(info->CpuSet.Id);
		}
		at += info->Size;
	}
	const bool ok = !ids.empty() &&
	                ::SetProcessDefaultCpuSets(::GetCurrentProcess(), ids.data(), static_cast<ULONG>(ids.size())) != 0;
	std::printf("Experiments: threads without a CPU of their own avoid CPUs 0x%llx (%zu CPU sets, %s)\n",
	            static_cast<unsigned long long>(excluded), ids.size(), ok ? "ok" : "failed");
	std::fflush(stdout);
#else
	(void)mask;
#endif
}

struct State {
	std::atomic<uint64_t>& mask = Detail::g_mask;

	State() {
		const uint64_t defaults = EnvironmentMask();
		mask.store(defaults, std::memory_order_relaxed);
		ApplyCpuSets(defaults);
		const char* path_env = std::getenv("KYTY_EXPERIMENT_FILE");
		if (path_env == nullptr || path_env[0] == '\0') {
			return;
		}
		std::thread([this, path = std::string(path_env), defaults] {
			uint64_t last = defaults;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				uint64_t value = defaults;
				if (std::FILE* f = std::fopen(path.c_str(), "rb"); f != nullptr) {
					uint64_t parsed = 0;
					if (std::fscanf(f, "%" SCNu64, &parsed) == 1) {
						value = parsed;
					}
					std::fclose(f);
				}
				if (value != last) {
					mask.store(value, std::memory_order_relaxed);
					ApplyCpuSets(value);
					std::printf("Experiments: mask %" PRIu64 " -> %" PRIu64 "\n", last, value);
					std::fflush(stdout);
					last = value;
				}
			}
		}).detach();
	}
};

State& GetState() {
	static State state;
	return state;
}

} // namespace

bool FalseShareReads() noexcept {
	static const bool enabled = EnvSwitch("KYTY_FALSE_SHARE_READS", false);
	return enabled;
}

namespace Detail {
constinit std::atomic<uint64_t> g_mask {Uninitialized};

uint64_t InitMask() noexcept {
	(void)GetState();
	return g_mask.load(std::memory_order_relaxed);
}
} // namespace Detail

} // namespace Common::Experiments

namespace Common::Timeline {

namespace {

struct Event {
	int64_t     ns    = 0;
	uint32_t    tid   = 0;
	const char* what  = nullptr;
	uint64_t    value = 0;
};

struct Recorder {
	static constexpr size_t Capacity = 1u << 19u;
	std::unique_ptr<Event[]> events = std::make_unique<Event[]>(Capacity);
	std::atomic<size_t>      next {0};
	std::atomic<bool>        recording {false};
	bool                     enabled = false;
	std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();

	Recorder() {
		const char* value = std::getenv("KYTY_DBG_TIMELINE");
		if (value == nullptr || value[0] == '\0') return;
		enabled = true;
		// A number is a delay in seconds; anything else names a file whose appearance starts it.
		const bool        by_file = value[0] < '0' || value[0] > '9';
		const auto        delay   = by_file ? 0 : std::atoi(value);
		const std::string trigger = by_file ? value : "";
		std::thread([this, delay, trigger] {
			if (trigger.empty()) {
				std::this_thread::sleep_for(std::chrono::seconds(delay));
			} else {
				for (;;) {
					if (std::FILE* f = std::fopen(trigger.c_str(), "rb"); f != nullptr) {
						std::fclose(f);
						break;
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
				}
			}
			origin = std::chrono::steady_clock::now();
			recording.store(true, std::memory_order_release);
			// KYTY_DBG_TIMELINE_MS: capture length (default 1500 ms).
			const char* length = std::getenv("KYTY_DBG_TIMELINE_MS");
			std::this_thread::sleep_for(std::chrono::milliseconds(length != nullptr ? std::atoi(length) : 1500));
			recording.store(false, std::memory_order_release);
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			const auto count = std::min(next.load(), Capacity);
#if defined(_WIN32)
			// Return addresses in marks are absolute; this base turns them into image offsets.
			std::printf("NHL27TL: image base 0x%llx\n",
			            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr))));
#endif
			std::vector<Event> sorted(events.get(), events.get() + count);
			std::sort(sorted.begin(), sorted.end(),
			          [](const Event& a, const Event& b) { return a.ns < b.ns; });
			for (const auto& e: sorted) {
				std::printf("NHL27TL: %9.3f ms tid=%u %s %llu\n", static_cast<double>(e.ns) / 1e6, e.tid,
				            e.what, static_cast<unsigned long long>(e.value));
			}
			std::fflush(stdout);
		}).detach();
	}
};

Recorder& GetRecorder() {
	static Recorder recorder;
	return recorder;
}

uint32_t ThreadTag() {
	static std::atomic<uint32_t> counter {0};
	thread_local const uint32_t tag = counter.fetch_add(1) + 1;
	return tag;
}

} // namespace

bool Enabled() noexcept {
	return GetRecorder().enabled;
}

void MarkStack() noexcept {
	auto& recorder = GetRecorder();
	if (!recorder.enabled || !recorder.recording.load(std::memory_order_acquire)) return;
#if defined(_WIN32)
	void*      frames[10] {};
	const auto count = RtlCaptureStackBackTrace(1, 10, frames, nullptr);
	for (USHORT i = 0; i < count; i++) {
		Mark("ra", reinterpret_cast<uint64_t>(frames[i]));
	}
#endif
}

void Mark(const char* what, uint64_t value) noexcept {
	auto& recorder = GetRecorder();
	if (!recorder.enabled || !recorder.recording.load(std::memory_order_acquire)) return;
	const auto index = recorder.next.fetch_add(1, std::memory_order_relaxed);
	if (index >= Recorder::Capacity) return;
	recorder.events[index] = {
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
	                                                         recorder.origin).count(),
	    ThreadTag(), what, value};
}

} // namespace Common::Timeline
