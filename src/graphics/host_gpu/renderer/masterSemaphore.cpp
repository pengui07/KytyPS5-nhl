#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/experiments.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// NHL27LONGWAIT (KYTY_PERF_STATS): host-GPU waits of 3 ms or more, grouped by host call stack, the
// top entries by time every 5 s. The exe is linked at a fixed base: resolve the addresses with the
// linker map (address - 0x140000000 = RVA).
static void NoteLongWait(uint64_t ns, bool on_gpu_thread) {
	struct Stat {
		uint64_t                count = 0;
		uint64_t                ns    = 0;
		uint64_t                max   = 0;
		bool                    gpu   = false;
		std::array<uint64_t, 12> frames {};
		int                     depth = 0;
	};
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, Stat> stats;
	static auto                               last = std::chrono::steady_clock::now();

	std::array<uint64_t, 12> frames {};
	const int depth = LibKernel::Memory::CaptureHostStack(frames.data(), static_cast<int>(frames.size()));
	uint64_t  key   = 1469598103934665603ull;
	for (int i = 0; i < depth; i++) {
		key = (key ^ frames[static_cast<size_t>(i)]) * 1099511628211ull;
	}
	std::scoped_lock lock(mutex);
	auto& stat = stats[key];
	stat.count++;
	stat.ns += ns;
	stat.max = std::max(stat.max, ns);
	stat.gpu = on_gpu_thread;
	stat.frames = frames;
	stat.depth  = depth;
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	last = now;
	std::vector<const Stat*> top;
	for (const auto& [k, s]: stats) {
		top.push_back(&s);
	}
	std::sort(top.begin(), top.end(), [](const Stat* a, const Stat* b) { return a->ns > b->ns; });
	for (size_t i = 0; i < top.size() && i < 6; i++) {
		const auto& s = *top[i];
		std::printf("NHL27LONGWAIT: count=%" PRIu64 " ms=%.1f max_ms=%.1f gpu_thread=%d stack=", s.count,
		            static_cast<double>(s.ns) / 1e6, static_cast<double>(s.max) / 1e6, s.gpu ? 1 : 0);
		for (int f = 0; f < s.depth; f++) {
			std::printf("%s0x%" PRIx64, f == 0 ? "" : ",", s.frames[static_cast<size_t>(f)]);
		}
		std::printf("\n");
	}
	std::fflush(stdout);
	stats.clear();
}

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	// KYTY_PERF_STATS: time the GPU thread spends blocked on host GPU progress.
	const bool on_gpu_thread = GuestGpu::IsGpuThread();
	FrameBreakdown::Timer wait_timer(FrameBreakdown::g_counters.gpu_wait_ns, on_gpu_thread);
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_WAIT);
	if (on_gpu_thread) {
		FrameBreakdown::Add(FrameBreakdown::g_counters.gpu_wait_count);
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	const bool timed  = BufferCache::PerfTimersEnabled() && GuestGpu::IsGpuThread();
	const bool stats  = FrameBreakdown::Enabled();
	const auto start  = timed || stats ? std::chrono::steady_clock::now()
	                                   : std::chrono::steady_clock::time_point {};
	Common::Timeline::Mark("host-wait", tick);
	Common::Timeline::MarkStack();
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	Common::Timeline::Mark("host-done", tick);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (stats) {
		const auto ns = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
		if (ns >= 3000000u) {
			NoteLongWait(ns, on_gpu_thread);
		}
	}
	if (timed) {
		BufferCache::PerfAddTime(BufferCache::PerfTimer::GpuWait,
		                         static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                             std::chrono::steady_clock::now() - start).count()));
	}
	Refresh();
}

} // namespace Libs::Graphics
