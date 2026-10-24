#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/pageWriteSeq.h"
#include "common/experiments.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/frameDump.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

bool EnvFlag(const char* name) {
	const char* setting = std::getenv(name);
	return setting != nullptr && setting[0] == '1';
}

// KYTY_PERF_STATS=1: count synchronous GPU->CPU readbacks (ReadMemory -> DownloadBufferMemory).
constexpr size_t PerfBuckets = 4; // color clear, fault, invalidate (CPU write), other
struct PerfStats {
	std::atomic<uint64_t> waits[PerfBuckets] {};
	std::atomic<uint64_t> wait_ns[PerfBuckets] {};
	std::atomic<uint64_t> events[static_cast<size_t>(BufferCache::PerfEvent::Count)] {};
	std::atomic<uint32_t> pm4_op {0};
	std::atomic<uint64_t> created_bytes {0};
	std::atomic<uint64_t> memory_used {0};
	std::atomic<uint64_t> texture_trigger {0};
	std::atomic<uint64_t> buffer_trigger {0};
	std::atomic<uint64_t> timer_count[static_cast<size_t>(BufferCache::PerfTimer::Count)] {};
	std::atomic<uint64_t> timer_ns[static_cast<size_t>(BufferCache::PerfTimer::Count)] {};
	// Only touched from the GPU thread.
	uint64_t                              last_waits[PerfBuckets] {};
	uint64_t                              last_wait_ns[PerfBuckets] {};
	uint64_t last_events[static_cast<size_t>(BufferCache::PerfEvent::Count)] {};
	std::chrono::steady_clock::time_point last_print {};
};
PerfStats g_perf;

// Per-4KiB-page statistics of fault-triggered downloads; GPU thread only, reset every print.
struct FaultPageStat {
	uint64_t    count       = 0;
	uint64_t    bytes       = 0;
	uint64_t    exact       = 0; // faulting byte itself was GPU-dirty
	uint64_t    tid         = 0;
	bool        gpu_thread  = false;
	const char* writer      = "?";
	uint64_t    writer_hash = 0;
	double      ms          = 0;
	uint32_t    pm4_op      = 0; // PM4 opcode being processed when the GPU thread itself faulted
	const char* via         = "fault";
};
std::unordered_map<uint64_t, FaultPageStat> g_fault_pages;

// Namespace-scope constants: these gates run on every buffer operation, and a function-local
// static costs a thread-safe-static guard check per call.
const bool g_perf_stats_enabled = EnvFlag("KYTY_PERF_STATS");
const bool g_perf_no_sync_readback = EnvFlag("KYTY_PERF_NO_SYNC_READBACK");

bool PerfStatsEnabled() {
	return g_perf_stats_enabled;
}

} // namespace

bool BufferCache::GuestSyncOnly() {
	return Common::RuntimeFlags::Current().guest_sync;
}

bool BufferCache::PerfNoSyncReadback() {
	return g_perf_no_sync_readback;
}

void BufferCache::PerfSetPm4Op(uint32_t opcode) {
	if (PerfStatsEnabled()) {
		g_perf.pm4_op.store(opcode, std::memory_order_relaxed);
	}
}

bool BufferCache::PerfTimersEnabled() {
	return PerfStatsEnabled();
}

void BufferCache::PerfAddTime(PerfTimer timer, uint64_t ns) {
	const auto index = static_cast<size_t>(timer);
	g_perf.timer_count[index].fetch_add(1, std::memory_order_relaxed);
	g_perf.timer_ns[index].fetch_add(ns, std::memory_order_relaxed);
}

void BufferCache::PerfSetMemoryUsage(uint64_t used, uint64_t texture_trigger, uint64_t buffer_trigger) {
	if (!PerfStatsEnabled()) return;
	g_perf.memory_used.store(used, std::memory_order_relaxed);
	if (texture_trigger != 0) g_perf.texture_trigger.store(texture_trigger, std::memory_order_relaxed);
	if (buffer_trigger != 0) g_perf.buffer_trigger.store(buffer_trigger, std::memory_order_relaxed);
}

uint64_t BufferCache::PerfFlipCount() {
	return g_perf.events[static_cast<size_t>(PerfEvent::Flip)].load(std::memory_order_relaxed);
}

void BufferCache::PerfCount(PerfEvent event) {
	if (PerfStatsEnabled()) {
		g_perf.events[static_cast<size_t>(event)].fetch_add(1, std::memory_order_relaxed);
	}
}

void BufferCache::PerfMaybePrint() {
	if (!PerfStatsEnabled()) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (g_perf.last_print == std::chrono::steady_clock::time_point {}) {
		g_perf.last_print = now;
		return;
	}
	const double seconds = std::chrono::duration<double>(now - g_perf.last_print).count();
	if (seconds < 5.0) {
		return;
	}
	g_perf.last_print = now;
	uint64_t waits[PerfBuckets] {};
	double   ms[PerfBuckets] {};
	uint64_t total_waits = 0;
	double   total_ms    = 0;
	for (size_t i = 0; i < PerfBuckets; ++i) {
		const auto count_now = g_perf.waits[i].load(std::memory_order_relaxed);
		const auto ns_now    = g_perf.wait_ns[i].load(std::memory_order_relaxed);
		waits[i]             = count_now - g_perf.last_waits[i];
		ms[i]                = static_cast<double>(ns_now - g_perf.last_wait_ns[i]) / 1e6;
		g_perf.last_waits[i]   = count_now;
		g_perf.last_wait_ns[i] = ns_now;
		total_waits += waits[i];
		total_ms += ms[i];
	}
	constexpr size_t EventCount = static_cast<size_t>(PerfEvent::Count);
	uint64_t         events[EventCount] {};
	for (size_t i = 0; i < EventCount; ++i) {
		const auto now_count = g_perf.events[i].load(std::memory_order_relaxed);
		events[i]            = now_count - g_perf.last_events[i];
		g_perf.last_events[i] = now_count;
	}
	std::printf("NHL27PERF: %.1fs sync_waits=%llu wait_ms=%.1f | color_clear=%llu (%.1f ms) "
	            "fault=%llu (%.1f ms) invalidate=%llu (%.1f ms) other=%llu (%.1f ms) | "
	            "avoided: color_clear_shadow=%llu wait_reg_mem_backing=%llu | no_sync=%d\n",
	            seconds, static_cast<unsigned long long>(total_waits), total_ms,
	            static_cast<unsigned long long>(waits[0]), ms[0],
	            static_cast<unsigned long long>(waits[1]), ms[1],
	            static_cast<unsigned long long>(waits[2]), ms[2],
	            static_cast<unsigned long long>(waits[3]), ms[3],
	            static_cast<unsigned long long>(events[0]),
	            static_cast<unsigned long long>(events[1]), PerfNoSyncReadback() ? 1 : 0);
	std::printf("NHL27PERF: work dispatches=%llu draws=%llu auto_dispatch_barriers=%llu "
	            "guest_barriers=%llu (skipped %llu) acquire_mem=%llu render_passes=%llu experiments=%llu\n",
	            static_cast<unsigned long long>(events[2]),
	            static_cast<unsigned long long>(events[3]),
	            static_cast<unsigned long long>(events[4]),
	            static_cast<unsigned long long>(events[5]),
	            static_cast<unsigned long long>(
	                events[static_cast<size_t>(PerfEvent::BarrierSkipped)]),
	            static_cast<unsigned long long>(events[6]),
	            static_cast<unsigned long long>(events[7]), static_cast<unsigned long long>(Common::Experiments::Mask()));
	{
		constexpr size_t TimerCount = static_cast<size_t>(PerfTimer::Count);
		static uint64_t  last_count[TimerCount] {};
		static uint64_t  last_ns[TimerCount] {};
		uint64_t         count[TimerCount] {};
		double           timer_ms[TimerCount] {};
		for (size_t i = 0; i < TimerCount; ++i) {
			const auto c  = g_perf.timer_count[i].load(std::memory_order_relaxed);
			const auto ns = g_perf.timer_ns[i].load(std::memory_order_relaxed);
			count[i]      = c - last_count[i];
			timer_ms[i]   = static_cast<double>(ns - last_ns[i]) / 1e6;
			last_count[i] = c;
			last_ns[i]    = ns;
		}
		std::printf("NHL27PERF: cmd begins=%llu (%.1f ms) submits=%llu (%.1f ms) "
		            "gpu_waits=%llu (%.1f ms) false_share_reads=%llu pm4_shadow=%llu\n",
		            static_cast<unsigned long long>(count[0]), timer_ms[0],
		            static_cast<unsigned long long>(count[1]), timer_ms[1],
		            static_cast<unsigned long long>(count[2]), timer_ms[2],
		            static_cast<unsigned long long>(events[static_cast<size_t>(PerfEvent::FalseShareRead)]),
		            static_cast<unsigned long long>(events[static_cast<size_t>(PerfEvent::Pm4ShadowPacket)]));
		const auto flips = events[static_cast<size_t>(PerfEvent::Flip)];
		static uint64_t last_created_bytes = 0;
		const uint64_t  created_now        = g_perf.created_bytes.load(std::memory_order_relaxed);
		const uint64_t  created_bytes      = created_now - last_created_bytes;
		last_created_bytes                 = created_now;
		const auto per_frame = [&](PerfTimer timer) {
			return flips != 0 ? timer_ms[static_cast<size_t>(timer)] / static_cast<double>(flips) : 0.0;
		};
		std::printf("NHL27PHASE: ms/frame pm4=%.1f draw_programs=%.1f draw_targets=%.1f "
		            "draw_execute=%.1f (draws %llu) dispatch=%.1f dispatch_program=%.1f "
		            "(dispatches %llu) gpu_waits=%.1f submits=%.1f\n",
		            per_frame(PerfTimer::Pm4), per_frame(PerfTimer::DrawPrograms),
		            per_frame(PerfTimer::DrawTargets), per_frame(PerfTimer::DrawExecute),
		            static_cast<unsigned long long>(count[static_cast<size_t>(PerfTimer::DrawExecute)]),
		            per_frame(PerfTimer::Dispatch), per_frame(PerfTimer::DispatchProgram),
		            static_cast<unsigned long long>(count[static_cast<size_t>(PerfTimer::Dispatch)]),
		            per_frame(PerfTimer::GpuWait), per_frame(PerfTimer::Submit));
		std::printf("NHL27DRAW: ms/frame bind_prepare=%.1f bind_resolve=%.1f vertex=%.1f "
		            "pipeline=%.1f attachments=%.1f commit=%.1f record=%.1f | per frame: "
		            "buffer_creates=%.1f (%.1f MiB) buffer_deletes=%.1f (gc %.1f) buffer_uploads=%.1f "
		            "image_inserts=%.1f image_gc_frees=%.1f image_uploads=%.1f | memory %.0f MiB, gc triggers "
		            "texture %.0f buffer %.0f MiB\n",
		            per_frame(PerfTimer::DrawBindPrepare), per_frame(PerfTimer::DrawBindResolve),
		            per_frame(PerfTimer::DrawVertex), per_frame(PerfTimer::DrawPipeline),
		            per_frame(PerfTimer::DrawAttachments), per_frame(PerfTimer::DrawCommit),
		            per_frame(PerfTimer::DrawRecord),
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::BufferCreate)]) / flips : 0.0,
		            flips != 0 ? static_cast<double>(created_bytes) / (1024.0 * 1024.0) / flips : 0.0,
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::BufferDelete)]) / flips : 0.0,
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::BufferGcDelete)]) / flips : 0.0,
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::BufferUpload)]) / flips : 0.0,
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::ImageInsert)]) / flips : 0.0,
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::ImageGcFree)]) / flips : 0.0,
		            flips != 0 ? static_cast<double>(events[static_cast<size_t>(PerfEvent::ImageUpload)]) / flips : 0.0,
		            static_cast<double>(g_perf.memory_used.load(std::memory_order_relaxed)) / (1024.0 * 1024.0),
		            static_cast<double>(g_perf.texture_trigger.load(std::memory_order_relaxed)) / (1024.0 * 1024.0),
		            static_cast<double>(g_perf.buffer_trigger.load(std::memory_order_relaxed)) / (1024.0 * 1024.0));
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	{
		// GPU-thread CPU time per presented frame: the frame cost of emulation work without the
		// time spent blocked on the GPU or on guest submissions.
		static uint64_t last_cpu_100ns = 0;
		FILETIME        created {}, exited {}, kernel {}, user {};
		if (GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user) != 0) {
			const auto as_u64 = [](const FILETIME& t) {
				return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) | t.dwLowDateTime;
			};
			const uint64_t cpu   = as_u64(kernel) + as_u64(user);
			const double   cpu_ms = static_cast<double>(cpu - last_cpu_100ns) / 1e4;
			last_cpu_100ns        = cpu;
			const auto flips      = events[static_cast<size_t>(PerfEvent::Flip)];
			std::printf("NHL27PERF: frame flips=%llu wall_ms/frame=%.1f gpu_thread_cpu_ms/frame=%.1f "
			            "(busy %.0f%%)\n",
			            static_cast<unsigned long long>(flips),
			            flips != 0 ? seconds * 1000.0 / static_cast<double>(flips) : 0.0,
			            flips != 0 ? cpu_ms / static_cast<double>(flips) : 0.0,
			            100.0 * cpu_ms / (seconds * 1000.0));
		}
	}
#endif
	{
		static uint64_t last[5] {};
		uint64_t        now[5] {};
		ProtectStats::Read(now[0], now[1], now[2], now[3], now[4]);
		std::printf("NHL27PERF: protect calls=%llu (gpu-thread %llu) MiB=%.1f | tracking faults "
		            "read=%llu write=%llu\n",
		            static_cast<unsigned long long>(now[0] - last[0]),
		            static_cast<unsigned long long>(now[2] - last[2]),
		            static_cast<double>(now[1] - last[1]) / (1024.0 * 1024.0),
		            static_cast<unsigned long long>(now[3] - last[3]),
		            static_cast<unsigned long long>(now[4] - last[4]));
		std::copy(std::begin(now), std::end(now), std::begin(last));
	}
	std::vector<std::pair<uint64_t, FaultPageStat>> pages(g_fault_pages.begin(), g_fault_pages.end());
	std::sort(pages.begin(), pages.end(),
	          [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	for (size_t i = 0; i < pages.size() && i < 8; ++i) {
		const auto& [page, stat] = pages[i];
		std::printf("NHL27PERF: fault page=0x%llx count=%llu bytes=%llu exact=%llu ms=%.1f "
		            "thread=%s tid=%llu pm4_op=0x%x writer=%s hash=%016llx via=%s\n",
		            static_cast<unsigned long long>(page),
		            static_cast<unsigned long long>(stat.count),
		            static_cast<unsigned long long>(stat.bytes),
		            static_cast<unsigned long long>(stat.exact), stat.ms,
		            stat.gpu_thread ? "gpu-cp" : "guest-cpu",
		            static_cast<unsigned long long>(stat.tid), stat.pm4_op, stat.writer,
		            static_cast<unsigned long long>(stat.writer_hash), stat.via);
	}
	if (!pages.empty()) {
		uint64_t gpu_count = 0, cpu_count = 0, unique_pages = pages.size();
		double   gpu_ms = 0, cpu_ms = 0;
		for (const auto& [page, stat]: pages) {
			(stat.gpu_thread ? gpu_count : cpu_count) += stat.count;
			(stat.gpu_thread ? gpu_ms : cpu_ms) += stat.ms;
		}
		std::printf("NHL27PERF: fault totals pages=%llu gpu-cp=%llu (%.1f ms) guest-cpu=%llu (%.1f ms)\n",
		            static_cast<unsigned long long>(unique_pages),
		            static_cast<unsigned long long>(gpu_count), gpu_ms,
		            static_cast<unsigned long long>(cpu_count), cpu_ms);
	}
	g_fault_pages.clear();
	std::fflush(stdout);
}

void BufferCache::LogGpuWrite(uint64_t vaddr, uint64_t size) {
	FrameBreakdown::SpecNoteGpuWrite(vaddr, size);
	constexpr size_t MaxRecords = 512;
	const WriterRecord record {vaddr, vaddr + size, m_writer_hash, m_writer_kind};
	if (m_writer_log.size() < MaxRecords) {
		m_writer_log.push_back(record);
	} else {
		m_writer_log[m_writer_head] = record;
		m_writer_head               = (m_writer_head + 1) % MaxRecords;
	}
}

namespace {
constexpr uint32_t GpuFillChunkShift = 16;

// Sets (or tests) the m_gpu_fill_filter bits of the 64 KiB chunks [begin, end) touches.
void MarkGpuFillChunks(std::array<uint64_t, 4>& filter, uint64_t begin, uint64_t end) {
	const uint64_t first = begin >> GpuFillChunkShift;
	const uint64_t last  = (end - 1) >> GpuFillChunkShift;
	if (last - first >= 255u) {
		filter.fill(~uint64_t {0});
		return;
	}
	for (uint64_t chunk = first; chunk <= last; chunk++) {
		filter[(chunk >> 6u) & 3u] |= uint64_t {1} << (chunk & 63u);
	}
}

// Whether any m_gpu_fill_filter bit of the 64 KiB chunks [begin, end) touches is set.
bool TestGpuFillChunks(const std::array<uint64_t, 4>& filter, uint64_t begin, uint64_t end) {
	const uint64_t first = begin >> GpuFillChunkShift;
	const uint64_t last  = (end - 1) >> GpuFillChunkShift;
	if (last - first >= 255u) {
		return (filter[0] | filter[1] | filter[2] | filter[3]) != 0;
	}
	for (uint64_t chunk = first; chunk <= last;) {
		const uint64_t bit   = chunk & 63u;
		const uint64_t count = std::min<uint64_t>(64u - bit, last - chunk + 1u);
		const uint64_t mask  = (count == 64u ? ~uint64_t {0} : (uint64_t {1} << count) - 1u) << bit;
		if ((filter[(chunk >> 6u) & 3u] & mask) != 0) {
			return true;
		}
		chunk += count;
	}
	return false;
}
} // namespace

void BufferCache::DropGpuFills(uint64_t vaddr, uint64_t size) {
	if (m_gpu_fill_count.load(std::memory_order_acquire) == 0) {
		return;
	}
	std::scoped_lock lock(m_gpu_fills_mutex);
	const auto end = vaddr + size;
	// Written buffers (bound by many draws, often megabytes) test the filter a word at a time.
	if (size != 0 && !TestGpuFillChunks(m_gpu_fill_filter, vaddr, end)) {
		return;
	}
	const auto before = m_gpu_fills.size();
	std::erase_if(m_gpu_fills, [&](const GpuFill& fill) { return fill.begin < end && vaddr < fill.end; });
	if (m_gpu_fills.size() != before) {
		m_gpu_fill_filter = {};
		for (const auto& fill: m_gpu_fills) {
			MarkGpuFillChunks(m_gpu_fill_filter, fill.begin, fill.end);
		}
	}
	m_gpu_fill_count.store(m_gpu_fills.size(), std::memory_order_release);
}

void BufferCache::RecordGpuFill(uint64_t vaddr, uint64_t size, uint32_t value) {
	if (!PerfNoSyncReadback() || (vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 ||
	    size > UINT64_MAX - vaddr) {
		return;
	}
	const auto       end = vaddr + size;
	std::scoped_lock lock(m_gpu_fills_mutex);
	std::erase_if(m_gpu_fills, [&](const GpuFill& fill) { return fill.begin < end && vaddr < fill.end; });
	constexpr size_t MaxFills = 64;
	if (m_gpu_fills.size() >= MaxFills) {
		m_gpu_fills.erase(m_gpu_fills.begin());
	}
	m_gpu_fills.push_back({vaddr, vaddr + size, value});
	MarkGpuFillChunks(m_gpu_fill_filter, vaddr, vaddr + size);
	m_gpu_fill_count.store(m_gpu_fills.size(), std::memory_order_release);
}

bool BufferCache::TryGetGpuFill(uint64_t vaddr, uint64_t size, uint32_t& value) const {
	if (size == 0 || size > UINT64_MAX - vaddr) {
		return false;
	}
	if (m_gpu_fill_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	std::scoped_lock lock(m_gpu_fills_mutex);
	for (const auto& fill: m_gpu_fills) {
		if (vaddr >= fill.begin && vaddr + size <= fill.end && ((vaddr - fill.begin) & 3u) == 0) {
			value = fill.value;
			return true;
		}
	}
	return false;
}

void BufferCache::DiscardGpuModified(uint64_t vaddr, uint64_t size) {
	// Without whole pages the tracker could drop ownership of unrelated bytes. Leave the range
	// dirty instead; the caller's invalidation then performs the ordinary download.
	if (size == 0 || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 || (size & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	DropGpuFills(vaddr, size);
	{
		std::unique_lock ranges_lock(m_gpu_ranges_mutex);
		m_gpu_modified_ranges.Subtract(vaddr, size);
		PublishGpuRangesVersion();
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
}

void BufferCache::ForgetGpuWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	DropGpuFills(vaddr, size);
	{
		std::unique_lock ranges_lock(m_gpu_ranges_mutex);
		m_gpu_modified_ranges.Subtract(vaddr, size);
		PublishGpuRangesVersion();
	}
	const auto first = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto last  = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
		if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
		}
	}
}

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	// A readback on the second queue waits for the buffer's last writer only: without this, one
	// that followed a label write (TryWriteLabel) copied the buffer before the label's copy ran
	// and published the old value over the label in guest memory (match-start hangs on a
	// counter one short).
	buffer.last_write_tick = m_scheduler.CurrentTick();
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	++m_registration_generation;
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_memory_tracker.QueueCpuDirtyRange(buffer.CpuAddress(), buffer.Size());
		m_buffer_set_epoch++;
		m_total_used_memory += buffer.Size();
		buffer.lru_id   = m_lru_cache.Insert(id, m_gc_tick);
		buffer.lru_tick = m_gc_tick;
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		m_buffer_set_epoch++;
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	// The LRU item lives in another allocation; skip it when this tick already touched it.
	if (!buffer.is_deleted && buffer.lru_tick != m_gc_tick) {
		buffer.lru_tick = m_gc_tick;
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	PerfCount(PerfEvent::BufferDelete);
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferHostOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

namespace {
// KYTY_DBG_CP_READS=1: what the synchronous downloads wait for, printed as NHL27RBWAIT every 2 s.
// lag = submissions between the last write of the buffer and the command buffer being recorded
// (0: the write is in it, so it must be submitted first); done = the write had finished on the
// GPU when the download began; from = who read (gpu = the execution thread itself, cp = the
// command processor, guest = a guest thread, ? = another path); exact = the bytes read were
// GPU-written themselves (else they only share a page with such bytes); via = the ReadMemory
// caller (fault, clear = fast-clear metadata, other = an explicit read such as a strict resource
// table read).
thread_local uint32_t g_readback_origin = 3;
thread_local uint32_t g_readback_exact  = 2;
thread_local uint32_t g_readback_caller = 3;
thread_local uint64_t g_readback_shader = 0;
struct ReadbackWaitNote {
	bool                                  active = false;
	bool                                  queue  = false;
	uint32_t                              key    = 0;
	uint64_t                              shader = 0;
	std::chrono::steady_clock::time_point start;

	void Begin(CommandScheduler& scheduler, uint64_t write_tick) {
		const auto current = scheduler.CurrentTick();
		const auto lag  = static_cast<uint32_t>(std::min<uint64_t>(current - std::min(write_tick, current), 3));
		const bool done = scheduler.GetMasterSemaphore().KnownGpuTick() >= write_tick;
		active          = true;
		key = g_readback_caller * 10000 + g_readback_origin * 1000 + g_readback_exact * 100 + lag * 10 + (done ? 1 : 0);
		// Strict resource reads of GPU-written bytes (caller other, exact): by shader.
		shader = g_readback_caller == 0 && g_readback_exact == 1 ? g_readback_shader : 0;
		start             = std::chrono::steady_clock::now();
		g_readback_origin = 3;
		g_readback_exact  = 2;
		g_readback_caller = 3;
	}
	~ReadbackWaitNote() {
		if (!active) {
			return;
		}
		struct Stat {
			uint64_t count = 0;
			uint64_t ns    = 0;
		};
		static std::map<uint32_t, Stat>                   stats;
		static std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
		const auto                                        now = std::chrono::steady_clock::now();
		auto& stat = stats[key + (queue ? 100000u : 0u)];
		stat.count++;
		const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
		stat.ns += ns;
		static std::map<uint64_t, Stat> shaders;
		if (shader != 0) {
			shaders[shader].count++;
			shaders[shader].ns += ns;
		}
		if (now - last < std::chrono::seconds(2)) {
			return;
		}
		last = now;
		for (const auto& [k, s]: stats) {
			static const char* const origins[] = {"gpu", "cp", "guest", "?"};
			// Writes: own = the stored bytes are GPU-written, many = next to more than 256 GPU-written
			// bytes of the page, lazy-off = next to a few (LazySharedWrites off).
			static const char* const exacts[]  = {"no", "yes", "?", "own", "many", "lazy-off"};
			static const char* const callers[] = {"other", "clear", "fault", "?"};
			std::printf("NHL27RBWAIT: %s via=%s from=%s exact=%s lag=%u done=%u count=%llu ms=%.1f\n",
			            k >= 100000 ? "queue " : "inline", callers[(k / 10000) % 10], origins[(k / 1000) % 10],
			            exacts[(k / 100) % 10], (k / 10) % 10, k % 10, static_cast<unsigned long long>(s.count),
			            static_cast<double>(s.ns) / 1e6);
		}
		std::vector<std::pair<uint64_t, Stat>> by_time(shaders.begin(), shaders.end());
		std::sort(by_time.begin(), by_time.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (size_t i = 0; i < by_time.size() && i < 6; i++) {
			std::printf("NHL27RBWAIT: strict GPU-written reads of shader %016llx count=%llu ms=%.1f\n",
			            static_cast<unsigned long long>(by_time[i].first),
			            static_cast<unsigned long long>(by_time[i].second.count),
			            static_cast<double>(by_time[i].second.ns) / 1e6);
		}
		std::fflush(stdout);
		stats.clear();
		shaders.clear();
	}
};
} // namespace

template <bool async, bool merge>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	static_assert(async || !merge);
	DropGpuFills(vaddr, size);
	// The publish writes only what is still mapped by then (unmaps no longer wait for it).
	const auto unmap_sequence = m_scheduler.Context().UnmapSequence();
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    std::unique_lock ranges_lock(m_gpu_ranges_mutex);
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    PublishGpuRangesVersion();
	    });
	m_last_download_bytes = 0;
	if (copies.empty()) {
		return false;
	}
	for (const auto& copy: copies) {
		m_last_download_bytes += copy.size;
	}

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}
	// Experiment LazySharedWrites: the guest bytes as they are before the faulting CPU write runs.
	// Asynchronous guest reads publish the same way: bytes the CPU stored while the download was
	// pending (through a mapping or a path that did not fault) are newer than the GPU's.
	std::vector<uint8_t> snapshot;
	if constexpr (async) {
		snapshot.resize(m_last_download_bytes);
		uint64_t at = 0;
		for (const auto& copy: copies) {
			if (!Libs::LibKernel::Memory::TryReadBacking(buffer_address + copy.srcOffset,
			                                             snapshot.data() + at, copy.size)) {
				snapshot.clear();
				break;
			}
			at += copy.size;
		}
	}
	auto publish = [this, mapped, offset, total_size, buffer_address, copies = copies, unmap_sequence,
	                owner = std::move(temporary), snapshot = std::move(snapshot)]() mutable {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		uint64_t at = 0;
		for (const auto& copy: copies) {
			const auto* gpu = mapped + (copy.dstOffset - offset);
			if constexpr (!merge) {
				// Diagnostics: a pending guest read whose bytes changed before their publish (the old
				// publish wrote the GPU's bytes over them).
				if (!snapshot.empty()) {
					std::vector<uint8_t> now(copy.size);
					if (Libs::LibKernel::Memory::TryReadBacking(buffer_address + copy.srcOffset, now.data(), copy.size) &&
					    std::memcmp(now.data(), snapshot.data() + at, copy.size) != 0) {
						uint64_t changed = 0;
						for (uint64_t i = 0; i < copy.size; i++) {
							changed += now[i] != snapshot[at + i] ? 1u : 0u;
						}
						static std::atomic<uint32_t> reported {0};
						if (reported.fetch_add(1, std::memory_order_relaxed) < 64) {
							std::printf("NHL27ASYNCKEEP: guest read download 0x%016llx+0x%llx: %llu bytes changed while "
							            "pending, kept\n",
							            static_cast<unsigned long long>(buffer_address + copy.srcOffset),
							            static_cast<unsigned long long>(copy.size),
							            static_cast<unsigned long long>(changed));
							std::fflush(stdout);
						}
					}
				}
			}
			const auto copy_address = buffer_address + copy.srcOffset;
			const auto* expected     = snapshot.empty() ? nullptr : snapshot.data() + at;
			m_scheduler.Context().WriteUnlessUnmapped(
			    unmap_sequence, copy_address, copy.size, [&](uint64_t address, uint64_t bytes) {
				    const auto skip = address - copy_address;
				    if (expected != nullptr) {
					    // Bytes the CPU stored since the download was queued are newer than the GPU's.
					    if (!Libs::LibKernel::Memory::TryMergeBacking(address, gpu + skip, expected + skip, bytes)) {
						    EXIT("BufferCache: merge of a shared-page download failed\n");
					    }
				    } else {
					    Libs::LibKernel::Memory::WriteBacking(address, gpu + skip, bytes);
				    }
			    });
			at += copy.size;
		}
	};
	ReadbackWaitNote wait_note;
	if constexpr (!async) {
		const uint64_t wait_tick = buffer.last_write_tick;
		if (CpReadDebugEnabled()) {
			wait_note.Begin(m_scheduler, wait_tick);
		}
		// KYTY_READBACK_QUEUE: a buffer last written by an already submitted command buffer is
		// copied on the second queue once that submission finishes, not after everything queued.
		if (wait_tick < m_scheduler.CurrentTick() &&
		    Common::RuntimeFlags::Current().readback_queue) {
			const auto rb_start = std::chrono::steady_clock::now();
			const bool ok = ReadbackOnSecondQueue(buffer, download, copies, wait_tick);
			if (FrameBreakdown::Enabled()) {
				FrameBreakdown::Add(ok ? FrameBreakdown::g_counters.port_rbq_used
				                       : FrameBreakdown::g_counters.port_rbq_fallback);
				FrameBreakdown::Add(FrameBreakdown::g_counters.port_rbq_wait_ns,
				                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                        std::chrono::steady_clock::now() - rb_start).count()));
			}
			if (ok) {
				// Older asynchronous downloads of these bytes publish first, as on the normal path.
				m_scheduler.WaitPriorityOperations(wait_tick);
				publish();
				wait_note.queue = true;
				return true;
			}
		}
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	uint64_t written_begin = UINT64_MAX;
	uint64_t written_end   = 0;
	for (const auto& copy: copies) {
		written_begin = std::min(written_begin, buffer_address + copy.srcOffset);
		written_end   = std::max(written_end, buffer_address + copy.srcOffset + copy.size);
	}
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish), written_begin,
		                                   written_end - written_begin);
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

bool BufferCache::ReadbackOnSecondQueue(const Buffer& source, const Buffer& destination,
                                        std::span<const vk::BufferCopy> copies,
                                        uint64_t                        wait_tick) {
	auto& graphics = m_graphics;
	// Experiments ReadbackTransfer / ReadbackCompute: a queue of another family copies beside the
	// graphics work instead of after the work running on the graphics engine.
	uint32_t kind = 0;
	// KYTY_READBACK_TRANSFER=1 (env only; -2% +- 7.5 as experiment bit 8): the transfer-only family.
	static const bool readback_transfer = [] {
		const char* value = std::getenv("KYTY_READBACK_TRANSFER");
		return value != nullptr && value[0] == '1';
	}();
	if (readback_transfer && graphics.readback_queues[2] != nullptr) {
		kind = 2;
	} else if (Common::Experiments::Enabled(Common::Experiments::ReadbackCompute) &&
	           graphics.readback_queues[1] != nullptr) {
		kind = 1;
	}
	const auto queue = graphics.readback_queues[kind];
	if (queue == nullptr || copies.empty()) {
		return false;
	}
	auto& readback_buffer = m_readback_buffer[kind];
	auto& readback_fence  = m_readback_fence[kind];
	if (readback_buffer == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = graphics.readback_families[kind];
		if (graphics.device.createCommandPool(&pool_info, nullptr, &m_readback_pool[kind]) !=
		    vk::Result::eSuccess) {
			return false;
		}
		vk::CommandBufferAllocateInfo alloc {};
		alloc.commandPool        = m_readback_pool[kind];
		alloc.level              = vk::CommandBufferLevel::ePrimary;
		alloc.commandBufferCount = 1;
		vk::FenceCreateInfo fence_info {};
		if (graphics.device.allocateCommandBuffers(&alloc, &readback_buffer) != vk::Result::eSuccess ||
		    graphics.device.createFence(&fence_info, nullptr, &readback_fence) != vk::Result::eSuccess) {
			readback_buffer = nullptr;
			return false;
		}
	}
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	if (readback_buffer.begin(&begin) != vk::Result::eSuccess) {
		return false;
	}
	readback_buffer.copyBuffer(source.Handle(), destination.Handle(),
	                             static_cast<uint32_t>(copies.size()), copies.data());
	vk::MemoryBarrier host {};
	host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	readback_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                  vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0, nullptr, 0,
	                                  nullptr);
	if (readback_buffer.end() != vk::Result::eSuccess) {
		return false;
	}
	const auto                      semaphore = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    stage     = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &wait_tick;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &readback_buffer;
	// A wait for a tick the recording thread has not submitted yet (wait before signal) lost the
	// device on NHL 27 with the threaded GPU pipeline.
	m_scheduler.WaitSubmitted(wait_tick);
	// The compute queue's own wait for the writer timed the GPU out (TDR, device lost) at one
	// point of NHL 27's match load in about one boot of four (none in five boots without the
	// second queue): the CPU waits for the writer first, so the queue's wait (kept for its memory
	// dependency) is already satisfied.
	m_scheduler.GetMasterSemaphore().Wait(wait_tick);
	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		result = queue.submit(1, &submit, readback_fence);
	}
	if (result != vk::Result::eSuccess) {
		// The ordinary path (record into the current command buffer, wait for it) still works.
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("NHL27READBACK: second-queue submission failed (%s, tick %llu, gpu thread %d); "
			            "using the ordinary readback\n",
			            vk::to_string(result).c_str(), static_cast<unsigned long long>(wait_tick),
			            GuestGpu::IsGpuThread() ? 1 : 0);
			std::fflush(stdout);
		}
		(void)readback_buffer.reset({});
		return false;
	}
	Common::Timeline::Mark("readback-queue", wait_tick);
	// KYTY_READBACK_SPIN=1: most copies of finished data take tens of microseconds, less than
	// waking a thread that sleeps on the fence (measured within noise as experiment bit 9).
	static const bool readback_spin = [] {
		const char* value = std::getenv("KYTY_READBACK_SPIN");
		return value != nullptr && value[0] == '1';
	}();
	if (readback_spin) {
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
		while (graphics.device.getFenceStatus(readback_fence) == vk::Result::eNotReady &&
		       std::chrono::steady_clock::now() < deadline) {
			std::this_thread::yield();
		}
	}
	EXIT_NOT_IMPLEMENTED(graphics.device.waitForFences(1, &readback_fence, VK_TRUE, UINT64_MAX) !=
	                     vk::Result::eSuccess);
	Common::Timeline::Mark("readback-done", wait_tick);
	(void)graphics.device.resetFences(1, &readback_fence);
	(void)readback_buffer.reset({});
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

// NHL27LABEL (KYTY_PERF_STATS=1): how direct label writes resolve. A failed label write falls back to
// PrepareGpuWrite, which downloads GPU-dirty bytes on the page and waits for queued GPU work. Every 5 s.
namespace {
enum LabelOutcome : size_t {
	LabelOk,
	LabelUncached,
	LabelBadSize,
	LabelNoBuffer,
	LabelNotCovered,
	LabelOwnerCovers,
	LabelBacking,
	LabelCount
};
std::atomic<uint64_t> g_label_outcomes[LabelCount] {};
void NoteLabel(LabelOutcome outcome, uint64_t vaddr) {
	if (!PerfStatsEnabled()) {
		return;
	}
	g_label_outcomes[outcome].fetch_add(1, std::memory_order_relaxed);
	static std::atomic<int64_t>  last_ns {0};
	static std::atomic<uint64_t> last_vaddr {0};
	if (outcome != LabelOk && outcome != LabelUncached) {
		last_vaddr.store(vaddr, std::memory_order_relaxed);
	}
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	int64_t prev = last_ns.load(std::memory_order_relaxed);
	if (now - prev < 5'000'000'000 || !last_ns.compare_exchange_strong(prev, now)) {
		return;
	}
	uint64_t n[LabelCount] {};
	for (size_t i = 0; i < LabelCount; i++) {
		n[i] = g_label_outcomes[i].exchange(0, std::memory_order_relaxed);
	}
	std::printf("NHL27LABEL: ok=%llu uncached_ok=%llu bad_size=%llu no_buffer=%llu not_covered=%llu "
	            "(page_owner_covers=%llu) backing_fail=%llu last_fail_vaddr=0x%016llx\n",
	            static_cast<unsigned long long>(n[LabelOk]), static_cast<unsigned long long>(n[LabelUncached]),
	            static_cast<unsigned long long>(n[LabelBadSize]), static_cast<unsigned long long>(n[LabelNoBuffer]),
	            static_cast<unsigned long long>(n[LabelNotCovered]),
	            static_cast<unsigned long long>(n[LabelOwnerCovers]), static_cast<unsigned long long>(n[LabelBacking]),
	            static_cast<unsigned long long>(last_vaddr.load(std::memory_order_relaxed)));
	std::fflush(stdout);
}
} // namespace

namespace {
// Labels the host copy thread wrote (DeferredLabel slots are reused after it did).
std::atomic<uint64_t> g_deferred_labels_written {0};

void WriteDeferredLabel(void* argument) noexcept {
	const auto& label = *static_cast<const BufferCache::DeferredLabel*>(argument);
	if (!Libs::LibKernel::Memory::TryWriteBacking(label.vaddr, label.bytes.data(), label.size)) {
		static std::atomic_flag reported = ATOMIC_FLAG_INIT;
		if (!reported.test_and_set(std::memory_order_relaxed)) {
			std::printf("NHL27LABEL: deferred label write at 0x%016llx has no backing\n",
			            static_cast<unsigned long long>(label.vaddr));
			std::fflush(stdout);
		}
	}
	g_deferred_labels_written.fetch_add(1, std::memory_order_release);
}
} // namespace

bool BufferCache::TryWriteLabel(uint64_t vaddr, const void* data, uint64_t size, HostCopyQueue* deferred) {
	if (size == 0 || size > 256 || (size & 3u) != 0 || (vaddr & 3u) != 0 || data == nullptr ||
	    !GuestRange {vaddr, size}.Valid()) {
		NoteLabel(LabelBadSize, vaddr);
		return false;
	}
	Libs::LibKernel::Memory::NoteHostWrite(vaddr, size, Libs::LibKernel::Memory::HostWriteSource::Label,
	                                       data);
	// A label too long for a deferred slot is written now: the copies queued before it first.
	if (deferred != nullptr && size > sizeof(DeferredLabel::bytes)) {
		deferred->Drain();
		deferred = nullptr;
	}
	// KYTY_LABEL_UNCACHED_DIRECT=1 (experiment, off by default): a label no buffer covers has no GPU copy
	// to keep coherent, so write the backing directly. Its page can still be protected for other
	// GPU-written bytes; preparing it would download those and wait for queued GPU work (about 19 ms per
	// label in the NHL 27 menus). Labels inside an image, or on GPU-dirty bytes, keep the old path.
	static const bool uncached_direct = [] {
		const char* value = std::getenv("KYTY_LABEL_UNCACHED_DIRECT");
		return value != nullptr && value[0] == '1';
	}();
	// The copy thread writes the bytes into guest memory (TryWriteLabel with a queue).
	const auto queue_backing = [&] {
		// A slot is reused once the copy thread wrote it (the queue holds fewer labels than slots
		// before every submission drains it).
		if (m_deferred_labels_queued - g_deferred_labels_written.load(std::memory_order_acquire) >=
		    DeferredLabelSlots) {
			deferred->Drain();
		}
		auto& label = m_deferred_labels[m_deferred_labels_queued++ % DeferredLabelSlots];
		label.vaddr = vaddr;
		label.size  = size;
		std::memcpy(label.bytes.data(), data, size);
		deferred->Call(&WriteDeferredLabel, &label);
		(void)deferred->Flush();
	};
	// A deferred label needs no GPU copy either: the copy thread writes the backing (most of NHL 27's
	// end-of-pipe labels; the old path drained the copy queue and prepared the page first).
	const auto write_uncached = [&] {
		if ((!uncached_direct && deferred == nullptr) || HasGpuDirtyBytes(vaddr, size) ||
		    m_texture_cache.HasImagesInRegion(vaddr, size)) {
			return false;
		}
		if (deferred != nullptr) {
			queue_backing();
		} else if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
			return false;
		}
		DropGpuFills(vaddr, size);
		return true;
	};
	// Labels fit within one cache page. Do not create a buffer or change page ownership.
	const auto found = m_buffers.upper_bound(vaddr);
	if (found == m_buffers.begin()) {
		const bool written = write_uncached();
		NoteLabel(written ? LabelUncached : LabelNoBuffer, vaddr);
		return written;
	}
	auto& buffer = m_slot_buffers[std::prev(found)->second];
	if (vaddr < buffer.CpuAddress() || vaddr - buffer.CpuAddress() > buffer.Size() ||
	    size > buffer.Size() - (vaddr - buffer.CpuAddress())) {
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		const bool owner_covers = owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size);
		if (owner_covers) {
			NoteLabel(LabelOwnerCovers, vaddr);
			return false;
		}
		const bool written = write_uncached();
		NoteLabel(written ? LabelUncached : LabelNotCovered, vaddr);
		return written;
	}
	if (deferred != nullptr) {
		queue_backing();
	} else if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		NoteLabel(LabelBacking, vaddr);
		return false;
	}
	NoteLabel(LabelOk, vaddr);
	DropGpuFills(vaddr, size);
	// Keep the GPU copy coherent so a later download cannot restore an old label.
	// Other GPU-owned bytes on the page are untouched and remain tracked as dirty.
	WriteDataBuffer(buffer, vaddr, data, size);
	TouchBuffer(buffer);
	return true;
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	DropGpuFills(vaddr, size);
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

static bool SharedWritesDebug() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_SHARED_WRITES");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool BufferCache::LazySharedWriteEligible(uint64_t vaddr, uint64_t size) {
	// The stored bytes must not be GPU-written themselves, and the GPU must own only a few bytes
	// of each page involved.
	if (size == 0 || HasGpuDirtyBytes(vaddr, size)) {
		return false;
	}
	const auto first = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto last  = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	uint64_t   dirty = 0;
	m_gpu_modified_ranges.ForEachInRange(first, last - first,
	                                     [&](uint64_t start, uint64_t end) { dirty += end - start; });
	return dirty != 0 && dirty <= 256;
}

bool BufferCache::OverlapsPendingGuestBytes(uint64_t vaddr, uint64_t size) const {
	for (const auto& read: m_pending_guest_reads) {
		if (!(read.vaddr < vaddr + size && vaddr < read.vaddr + read.size)) continue;
		for (const auto& [begin, end]: read.bytes) {
			if (begin < vaddr + size && vaddr < end) return true;
		}
	}
	return false;
}

bool BufferCache::OverlapsPendingGuestRead(uint64_t vaddr, uint64_t size) const {
	for (const auto& read: m_pending_guest_reads) {
		if (read.vaddr < vaddr + size && vaddr < read.vaddr + read.size) return true;
	}
	return false;
}

void BufferCache::RebuildPendingReadRanges() {
	std::unique_lock ranges_lock(m_gpu_ranges_mutex);
	m_pending_read_ranges.Clear();
	for (const auto& read: m_pending_guest_reads) {
		m_pending_read_ranges.Add(read.vaddr, read.size);
	}
	PublishGpuRangesVersion();
}

void BufferCache::FinishPendingGuestReads(uint64_t vaddr, uint64_t size) {
	// Once their bytes are in guest memory, finished reads leave the shared set.
	struct Rebuild {
		BufferCache* cache;
		bool         finished = false;
		~Rebuild() {
			if (finished) cache->RebuildPendingReadRanges();
		}
	} rebuild {this};
	for (size_t i = 0; i < m_pending_guest_reads.size();) {
		const auto read = m_pending_guest_reads[i];
		if (!(read.vaddr < vaddr + size && vaddr < read.vaddr + read.size)) {
			++i;
			continue;
		}
		m_pending_guest_reads.erase(m_pending_guest_reads.begin() + static_cast<ptrdiff_t>(i));
		// The publish wrote the downloaded bytes to guest memory. Pages the GPU wrote again since
		// keep their protection, like ForgetGpuWrite.
		m_scheduler.WaitPriorityOperations(read.tick);
		m_pending_guest_read_count.fetch_sub(1, std::memory_order_release);
		rebuild.finished = true;
		const auto first = read.vaddr & ~(TRACKER_PAGE_SIZE - 1);
		const auto last  = (read.vaddr + read.size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
		for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
			if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
			    !OverlapsPendingGuestRead(page, TRACKER_PAGE_SIZE)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
			}
		}
	}
}

static thread_local uint64_t g_fault_rip = 0;
static thread_local BufferCache::WriteTag g_write_tag = BufferCache::WriteTag::None;

BufferCache::WriteTagScope::WriteTagScope(WriteTag tag) noexcept: m_saved(g_write_tag) {
	g_write_tag = tag;
}

BufferCache::WriteTagScope::~WriteTagScope() {
	g_write_tag = m_saved;
}

// Index Count: a guest thread's invalidation outside a write fault (file I/O, unmap).
static const char* WriteTagName(uint64_t rip) {
	static constexpr const char* names[] = {"gpu-untagged", "eop-label", "write-data",    "texture-keys",
	                                        "unmap",        "fill",      "guest-no-fault"};
	static_assert(std::size(names) == static_cast<size_t>(BufferCache::WriteTag::Count) + 1);
	return rip < std::size(names) ? names[rip] : nullptr;
}

void BufferCache::NoteReadbackShader(uint64_t hash) noexcept {
	g_readback_shader = hash;
}

void BufferCache::NoteFaultRip(uint64_t rip) {
	g_fault_rip = rip;
}

// NHL27WRITEBACK: CPU-write readbacks (a write to a GPU-written page) by faulting guest instruction,
// cumulative, the top entries by wait time every 5 s. Writes with no faulting guest instruction (rip=0:
// GPU-thread writes such as CP labels, unmaps, metadata and fills) are split by thread and invalidation
// size, so their sources can be told apart: a CP label write invalidates 1 byte, an unmap its whole range.
struct WriteBackStat {
	uint64_t rip    = 0;
	uint64_t count  = 0;
	uint64_t ns     = 0;
	uint64_t vaddr  = 0;
	uint64_t size   = 0;
	uint64_t bytes  = 0;
	bool     on_gpu = false;
};
static std::mutex                                g_writeback_mutex;
static std::unordered_map<uint64_t, WriteBackStat> g_writeback;

static void WriteBackRecord(uint64_t rip, bool on_gpu, uint64_t vaddr, uint64_t size, uint64_t bytes,
                            uint64_t ns) {
	std::scoped_lock lock {g_writeback_mutex};
	const uint64_t   key = rip != 0 ? rip
	                                : (uint64_t {1} << 63u) | (on_gpu ? uint64_t {1} << 62u : 0u) |
	                                      (size & ((uint64_t {1} << 62u) - 1u));
	auto& stat = g_writeback[key];
	stat.rip    = rip;
	stat.on_gpu = on_gpu;
	stat.size   = size;
	stat.count++;
	stat.ns += ns;
	stat.bytes += bytes;
	stat.vaddr = vaddr;
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	last = now;
	std::vector<std::pair<uint64_t, WriteBackStat>> top(g_writeback.begin(), g_writeback.end());
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	for (size_t i = 0; i < top.size() && i < 12; i++) {
		const auto& stat = top[i].second;
		char rip[32];
		if (const char* name = WriteTagName(stat.rip)) {
			std::snprintf(rip, sizeof(rip), "%s", name);
		} else {
			std::snprintf(rip, sizeof(rip), "0x%016" PRIx64, stat.rip);
		}
		std::printf("NHL27WRITEBACK: rip=%s thread=%s size=0x%" PRIx64 " count=%" PRIu64
		            " ms=%.1f downloaded_KiB=%.1f last_vaddr=0x%016" PRIx64 "\n",
		            rip, stat.on_gpu ? "gpu" : "guest", stat.size, stat.count,
		            static_cast<double>(stat.ns) / 1e6, static_cast<double>(stat.bytes) / 1024.0, stat.vaddr);
	}
	std::fflush(stdout);
}

void BufferCache::NoteDirtyRead(uint64_t vaddr, uint64_t size) {
	if (!Common::Experiments::Enabled(Common::Experiments::EarlySubmitWriters)) {
		return;
	}
	constexpr size_t MaxWatched = 64;
	for (const auto& [address, bytes]: m_watched_writes) {
		if (address <= vaddr && vaddr + size <= address + bytes) {
			return;
		}
	}
	if (m_watched_writes.size() < MaxWatched) {
		m_watched_writes.emplace_back(vaddr, size);
	} else {
		m_watched_writes[m_watched_next] = {vaddr, size};
		m_watched_next                   = (m_watched_next + 1) % MaxWatched;
	}
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write, ReadCaller caller) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	FrameBreakdown::FaultTimer fault_readmem_timer(FrameBreakdown::g_fault_cost.readmem_ns);
	// The pipelined command processor counts as the GPU: its faults (PM4 and register data next to
	// GPU-written bytes) are served after the operations it emitted, and the sync download waits
	// only for the writer's submission (KYTY_READBACK_QUEUE). The asynchronous guest-read path
	// below would wait for every queued GPU operation instead: 34% of its time in a match, and
	// -10% fps in an in-match A/B (8 rounds).
	const bool     on_gpu  = GuestGpu::IsGpuThread() || GuestGpu::IsCommandProcessorThread();
	const uint32_t origin  = GuestGpu::IsGpuThread() ? 0u : GuestGpu::IsCommandProcessorThread() ? 1u : 2u;
	const uint64_t tid   = static_cast<uint64_t>(std::hash<std::thread::id> {}(std::this_thread::get_id()));
	// Experiment AsyncGuestReads: a guest thread's read is queued and submitted by the GPU
	// thread, which then goes on recording; the guest thread waits for the copy itself.
	uint64_t       async_tick = 0;
	const uint64_t fault_rip = on_gpu ? static_cast<uint64_t>(g_write_tag)
	                                : g_fault_rip != 0 ? g_fault_rip : static_cast<uint64_t>(WriteTag::Count);
	// A guest thread's download whose writer the GPU has not finished comes back with the writer's
	// tick instead: the guest thread waits for it here while the execution thread goes on, then
	// asks again (AbPending3: the execution thread waits). Waits for unfinished writers were 7 ms
	// each, 4% of the execution thread in play.
	for (uint32_t attempt = 0;; ++attempt) {
		uint64_t retry_tick = 0;
		m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size, is_write, caller, on_gpu, tid,
		                                                      fault_rip, origin, &async_tick, attempt, &retry_tick] {
			if (!m_pending_guest_reads.empty()) {
				FinishPendingGuestReads(vaddr, size);
			}
			if (is_write && !IsRegionRegistered(vaddr, size)) {
				return;
			}
			auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
			if (!is_write && !on_gpu && Common::Experiments::Enabled(Common::Experiments::AsyncGuestReads)) {
				// The download covers whole tracker pages; so does the pending record. The bytes it
				// carries are the GPU-written ones of those pages.
				const auto first = vaddr & ~(TRACKER_PAGE_SIZE - 1);
				const auto last  = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
				std::vector<std::pair<uint64_t, uint64_t>> bytes;
				m_gpu_modified_ranges.ForEachInRange(first, last - first,
				                                     [&](uint64_t begin, uint64_t end) { bytes.emplace_back(begin, end); });
				// Shared queries stay conservative while downloads move bytes out of the ordinary
				// dirty ranges. Publish before DownloadBufferMemory removes those ranges.
				m_pending_guest_read_count.fetch_add(1, std::memory_order_release);
				{
					std::unique_lock ranges_lock(m_gpu_ranges_mutex);
					m_pending_read_ranges.Add(first, last - first);
					PublishGpuRangesVersion();
				}
				if (DownloadBufferMemory<true>(buffer, vaddr, size)) {
					async_tick       = m_scheduler.CurrentTick();
					m_pending_guest_reads.push_back({first, last - first, async_tick, std::move(bytes)});
					g_scheduler_flush_sites[2].fetch_add(1, std::memory_order_relaxed);
					m_scheduler.Flush();
					Common::Timeline::Mark("async-guest-read", vaddr);
				} else {
					m_pending_guest_read_count.fetch_sub(1, std::memory_order_release);
					RebuildPendingReadRanges();
				}
				return;
			}

			const bool timed = PerfStatsEnabled();
			const auto start = timed ? std::chrono::steady_clock::now()
			                         : std::chrono::steady_clock::time_point {};
			const bool exact_dirty =
			    timed && caller == ReadCaller::Fault && !is_write && HasGpuDirtyBytes(vaddr, 1);
			// Experiment LazySharedWrites: a guest thread's store next to a few GPU-written bytes
			// (NHL 27 builds command buffers in pages a compute shader wrote 16 bytes of) need not wait
			// for that shader. The download is queued and merged later, keeping any bytes the CPU
			// changes meanwhile, which are newer.
			const bool lazy = is_write && !on_gpu &&
			                  Common::Experiments::Enabled(Common::Experiments::LazySharedWrites) &&
			                  LazySharedWriteEligible(vaddr, size);
			if (CpReadDebugEnabled()) {
				g_readback_origin = origin;
				g_readback_exact  = !is_write                          ? (HasGpuDirtyBytes(vaddr, size) ? 1u : 0u)
				                    : HasGpuDirtyBytes(vaddr, size)    ? 3u
				                    : !LazySharedWriteEligible(vaddr, size) ? 4u
				                                                       : 5u;
				g_readback_caller = static_cast<uint32_t>(caller);
				// A guest store over GPU-written bytes: the GPU-written ranges of its page (NHL27OWNW,
				// at most 24 lines per 5 s), to see whether the store would cover them all.
				if (is_write && !on_gpu && g_readback_exact == 3u) {
					static auto     window = std::chrono::steady_clock::now();
					static uint32_t lines  = 0;
					const auto      now    = std::chrono::steady_clock::now();
					if (now - window >= std::chrono::seconds(5)) {
						window = now;
						lines  = 0;
					}
					if (lines++ < 24) {
						const auto  page = vaddr & ~(TRACKER_PAGE_SIZE - 1);
						std::string ranges;
						uint64_t    bytes = 0;
						m_gpu_modified_ranges.ForEachInRange(page, TRACKER_PAGE_SIZE, [&](uint64_t start, uint64_t end) {
							bytes += end - start;
							if (ranges.size() < 160) {
								char text[48];
								std::snprintf(text, sizeof(text), " +0x%llx..0x%llx",
								              static_cast<unsigned long long>(start - page),
								              static_cast<unsigned long long>(end - page));
								ranges += text;
							}
						});
						std::printf("NHL27OWNW: vaddr=0x%016" PRIx64 " rip=0x%016" PRIx64 " page_gpu_bytes=%llu last_write=%llu "
						            "current=%llu ranges:%s\n",
						            vaddr, fault_rip, static_cast<unsigned long long>(bytes),
						            static_cast<unsigned long long>(buffer.last_write_tick),
						            static_cast<unsigned long long>(m_scheduler.CurrentTick()), ranges.c_str());
					}
				}
			}
			// Experiment WideWriteDownloads: NHL 27 fills whole GPU-written pages with memcpy-style
			// stores (vmovntdq/vmovdqa), each page one fault and one readback (~90 us). The GPU-written
			// bytes of the store's 64 KiB window (the window a CPU write fault releases) come down at
			// once, so the next pages of the fill fault only for CPU-write tracking.
			uint64_t download_vaddr = vaddr;
			uint64_t download_size  = size;
			if (is_write && !on_gpu && !lazy &&
			    Common::Experiments::Enabled(Common::Experiments::WideWriteDownloads)) {
				constexpr uint64_t Window = 16 * TRACKER_PAGE_SIZE;
				const auto begin = std::max(vaddr & ~(Window - 1), buffer.CpuAddress());
				const auto end   = std::min((vaddr + size + Window - 1) & ~(Window - 1),
				                            buffer.CpuAddress() + buffer.Size());
				if (begin <= vaddr && vaddr + size <= end) {
					download_vaddr = begin;
					download_size  = end - begin;
				}
			}
			if (!lazy && !on_gpu && attempt < 4 && buffer.last_write_tick != 0 &&
			    !Common::Experiments::Enabled(Common::Experiments::AbPending3) &&
			    !m_scheduler.IsFree(buffer.last_write_tick) &&
			    m_gpu_modified_ranges.Intersects(download_vaddr, download_size)) {
				if (buffer.last_write_tick >= m_scheduler.CurrentTick()) {
					// The writer is in the open command buffer.
					m_scheduler.Flush();
				}
				retry_tick = buffer.last_write_tick;
				return;
			}
			// KYTY_DBG_CP_READS: the GPU thread's own writes into pages the GPU wrote, whose downloads
			// wait for the writer (NHL27GPUWRITEWAIT, at most 200 lines): the write's tag, address
			// and how long it waited.
			const bool gpu_write_debug = CpReadDebugEnabled() && is_write && on_gpu && !lazy;
			const auto gpu_write_start = gpu_write_debug ? std::chrono::steady_clock::now()
			                                             : std::chrono::steady_clock::time_point {};
			const auto gpu_write_tick  = buffer.last_write_tick;
			const bool downloaded = lazy ? DownloadBufferMemory<true, true>(buffer, vaddr, size)
			                             : DownloadBufferMemory<false>(buffer, download_vaddr, download_size);
			if (gpu_write_debug && downloaded) {
				const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
				                                                           gpu_write_start).count();
				static std::atomic<uint32_t> lines {0};
				if (ms >= 1.0 && lines.fetch_add(1, std::memory_order_relaxed) < 200) {
					const char* name = WriteTagName(fault_rip);
					std::printf("NHL27GPUWRITEWAIT: tag=%s vaddr=0x%016" PRIx64 " size=%" PRIu64 " downloaded=%" PRIu64
					            " writer_tick=%" PRIu64 " current=%" PRIu64 " ms=%.1f\n",
					            name != nullptr ? name : "?", vaddr, size, m_last_download_bytes, gpu_write_tick,
					            m_scheduler.CurrentTick(), ms);
					std::fflush(stdout);
				}
			}
			if (lazy) {
				Common::Timeline::Mark("lazy-shared-write", vaddr);
			}
			if (timed) {
				if (downloaded && !is_write) {
					const auto page = vaddr & ~(TRACKER_PAGE_SIZE - 1);
					auto&      stat = g_fault_pages[page];
					stat.via        = caller == ReadCaller::Fault        ? "fault"
					                  : caller == ReadCaller::ColorClear ? "clear"
					                                                     : "other";
					stat.count++;
					stat.bytes += m_last_download_bytes;
					stat.exact += exact_dirty ? 1 : 0;
					stat.tid        = tid;
					stat.gpu_thread = on_gpu;
					stat.pm4_op     = on_gpu ? g_perf.pm4_op.load(std::memory_order_relaxed) : 0;
					stat.ms += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					               std::chrono::steady_clock::now() - start).count()) / 1e6;
					stat.writer = "unknown";
					stat.writer_hash = 0;
					for (size_t n = 0; n < m_writer_log.size(); ++n) {
						const auto& rec = m_writer_log[(m_writer_head + m_writer_log.size() - 1 - n) %
						                               m_writer_log.size()];
						if (rec.begin < page + TRACKER_PAGE_SIZE && page < rec.end) {
							stat.writer      = rec.kind;
							stat.writer_hash = rec.hash;
							break;
						}
					}
				}
				if (downloaded) {
					const auto bucket = is_write                         ? size_t {2}
					                    : caller == ReadCaller::ColorClear ? size_t {0}
					                    : caller == ReadCaller::Fault      ? size_t {1}
					                                                       : size_t {3};
					// Buckets: 0 color clear, 1 fault, 2 invalidate (CPU write), 3 other.
					const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
					                    std::chrono::steady_clock::now() - start)
					                    .count();
					FrameBreakdown::AddReadback(static_cast<uint64_t>(ns));
					if (is_write) {
											WriteBackRecord(fault_rip, on_gpu, vaddr, size, m_last_download_bytes, static_cast<uint64_t>(ns));
					}
					g_perf.waits[bucket].fetch_add(1, std::memory_order_relaxed);
					g_perf.wait_ns[bucket].fetch_add(static_cast<uint64_t>(ns),
					                                 std::memory_order_relaxed);
				}
				PerfMaybePrint();
			}
			if (downloaded && is_write && SharedWritesDebug()) {
				static uint32_t lines = 0;
				if (lines++ < 200) {
					const auto page = vaddr & ~(TRACKER_PAGE_SIZE - 1);
					const char* kind = "unknown";
					uint64_t    hash = 0, wbegin = 0, wend = 0;
					for (size_t n = 0; n < m_writer_log.size(); ++n) {
						const auto& rec = m_writer_log[(m_writer_head + m_writer_log.size() - 1 - n) %
						                               m_writer_log.size()];
						if (rec.begin < page + TRACKER_PAGE_SIZE && page < rec.end) {
							kind = rec.kind; hash = rec.hash; wbegin = rec.begin; wend = rec.end;
							break;
						}
					}
					std::printf("NHL27SHD: fault=0x%llx bytes=%llu buffer=0x%llx+0x%llx last_write=%llu "
					            "current=%llu writer=%s %016llx [0x%llx,0x%llx) tid=%llx\n",
					            static_cast<unsigned long long>(vaddr),
					            static_cast<unsigned long long>(m_last_download_bytes),
					            static_cast<unsigned long long>(buffer.CpuAddress()),
					            static_cast<unsigned long long>(buffer.Size()),
					            static_cast<unsigned long long>(buffer.last_write_tick),
					            static_cast<unsigned long long>(m_scheduler.CurrentTick()), kind,
					            static_cast<unsigned long long>(hash), static_cast<unsigned long long>(wbegin),
					            static_cast<unsigned long long>(wend), static_cast<unsigned long long>(tid));
				}
			}
			if (downloaded) {
				m_memory_tracker.UnmarkRegionAsGpuModified(download_vaddr, download_size);
			}
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
		});
		if (retry_tick == 0) {
			break;
		}
		m_scheduler.WaitSubmitted(retry_tick);
		m_scheduler.GetMasterSemaphore().Wait(retry_tick);
	}
	if (async_tick != 0) {
		m_scheduler.WaitPriorityOperations(async_tick);
		m_scheduler.Context().GetGpu().SendUrgentCommandSync(
		    [this, vaddr, size] { FinishPendingGuestReads(vaddr, size); });
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);
	PerfCount(PerfEvent::BufferCreate);
	if (PerfStatsEnabled()) {
		g_perf.created_bytes.fetch_add(overlap.end - overlap.begin, std::memory_order_relaxed);
	}
	FrameBreakdown::Add(FrameBreakdown::g_counters.port_buffer_creates);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	// NHL27BUFFERS2 (KYTY_PERF_STATS): [0] = rebind lookup scope, [1] = PrepareBda scope.
	const bool bon = FrameBreakdown::BufOn();
	const int  bi  = bon ? FrameBreakdown::g_buf_scope - 1 : 0;
	auto&      bs  = FrameBreakdown::g_buf;
	if (bon) {
		bs.sb_calls[bi].fetch_add(1, std::memory_order_relaxed);
	}
	FrameBreakdown::Timer walk_timer(bs.sb_walk_ns[bi], bon);
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
		    FrameBreakdown::BdaBatchNoteUpload(address, bytes);
	    },
	    [&]() noexcept {
		    FrameBreakdown::Timer stage_timer(bs.sb_stage_ns[bi], bon);
		    source = UploadCopies(buffer, copies, total_size);
	    });
	walk_timer.Stop();
	if (source) {
		PerfCount(PerfEvent::BufferUpload);
		FrameBreakdown::Add(FrameBreakdown::g_counters.sync_uploads);
		FrameBreakdown::Add(FrameBreakdown::g_counters.sync_bytes, total_size);
		if (bon) {
			bs.sb_uploads[bi].fetch_add(1, std::memory_order_relaxed);
			bs.sb_bytes[bi].fetch_add(total_size, std::memory_order_relaxed);
		}
		auto& command = m_scheduler.Current();
		{
			FrameBreakdown::Timer end_timer(bs.sb_end_ns[bi], bon);
			command.EndRendering();
		}
		FrameBreakdown::Timer rec_timer(bs.sb_rec_ns[bi], bon);
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		FrameBreakdown::Timer image_timer(bs.sb_image_ns[bi], bon);
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		// The copies into coherent staging memory go to the host copy thread like the stream
		// buffer's (StreamGuestData); the submission waits for them.
		const bool async = m_staging_buffer.IsCoherent();
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (async) {
				StreamGuestData(mapped + copy.srcOffset, address, copy.size);
			} else {
				std::memcpy(mapped + copy.srcOffset, reinterpret_cast<const void*>(address), copy.size);
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	// Copies made here must come after the protections the host copy thread still applies.
	if (PageManager::DeferringProtection()) {
		m_scheduler.HostCopies()->Drain();
	}
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		std::memcpy(temporary->Mapped().data() + copy.srcOffset,
		            reinterpret_cast<const void*>(address), copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferHostOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

void BufferCache::StreamGuestData(uint8_t* destination, uint64_t vaddr, uint64_t size) {
	// The execution thread hands larger copies to the host copy thread. It reads the backing
	// store, which holds the same bytes as the guest range and is never protected, so a copy that
	// runs after a later draw protected the range for GPU writes does not fault.
	constexpr uint64_t MinHostCopy = 256;
	auto*              copies      = m_scheduler.HostCopies();
	// Callers write coherent memory (the stream buffer, coherent staging).
	// While protection is deferred, every copy goes there too (after the protections queued
	// before it), or waits for them.
	const bool deferring = PageManager::DeferringProtection();
	if (copies != nullptr && (size >= MinHostCopy || deferring) && GuestGpu::IsGpuThread()) {
		if (const void* source = Libs::LibKernel::Memory::FindBackingPointer(vaddr, size)) {
			copies->Copy(destination, source, size);
			return;
		}
	}
	if (deferring && copies != nullptr) {
		copies->Drain();
	}
	std::memcpy(destination, reinterpret_cast<const void*>(vaddr), size);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	FrameBreakdown::BdaBatchNoteUsed(vaddr, size);
	// NHL27BUFFERS2: counted only inside the RebindBuffers lookup scope.
	const bool bon = FrameBreakdown::BufOn() && FrameBreakdown::g_buf_scope == 1;
	auto&      bs  = FrameBreakdown::g_buf;
	if (bon) {
		bs.ob_calls.fetch_add(1, std::memory_order_relaxed);
		if (is_written) bs.ob_written.fetch_add(1, std::memory_order_relaxed);
	}
	// Draws bind the same read-only ranges over and over. While the range stays in the same
	// buffer and none of its pages became CPU dirty since a lookup found it clean (or uploaded
	// it), the lookup finds the same buffer and has nothing to upload. The epoch is read before
	// the dirty check: a page dirtied after it advances it.
	const bool  plain_read = !is_written && !is_texel_buffer;
	// Texel (formatted) reads too, while no image starts at the address: only such an image is
	// copied into the buffer (SynchronizeBufferFromImage), and they are most of the bindings.
	// The memo keeps the image registration version its answer was found at: asking the texture
	// cache (a lock and a hash lookup) on every bind was 3.6% of the execution thread.
	const bool  texel_candidate = !is_written && is_texel_buffer;
	bool        texel_read      = false;
	uint64_t    image_version   = 0;
	ObtainMemo* memo            = nullptr;
	uint64_t    cpu_epoch       = 0;
	if (plain_read || texel_candidate) {
		const auto hash = (vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull;
		memo            = &(*m_obtain_memo)[(hash >> 32u) % ObtainMemoSlots];
		if (texel_candidate) {
			// Only registrations of images at addresses sharing the filter bit change the answer (the
			// version of every registration made the memo miss most texel binds).
			image_version = m_texture_cache.AddressBucketVersion(vaddr);
			texel_read    = (memo->vaddr == vaddr && memo->texel && memo->image_version == image_version) ||
			             !m_texture_cache.HasImageAt(vaddr);
		}
	}
	if (plain_read || texel_read) {
		cpu_epoch = CpuDirtyEpoch(vaddr, size);
		// The epochs move whenever the game writes other bytes of the 64 KiB blocks: a range with
		// no CPU-dirty page has nothing to upload either (about 4,000 synchronizations a frame in
		// play found nothing).
		if (memo->vaddr == vaddr && memo->size == size && memo->texel == texel_read &&
		    memo->buffer_epoch == m_buffer_set_epoch &&
		    (memo->cpu_epoch == cpu_epoch || !m_memory_tracker.IsRegionCpuModified(vaddr, size))) {
			memo->cpu_epoch = cpu_epoch;
			if (texel_read) memo->image_version = image_version;
			auto& buffer = m_slot_buffers[memo->id];
			TouchBuffer(buffer);
			return {&buffer, memo->offset};
		}
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// Draws stream the same small buffers (frame and pass constants) again and again. Within
		// a draw window the guest cannot have written them since the last copy, so draws share it
		// while it is in the stream buffer and no work wrote cached buffers.
		const auto window = m_scheduler.Context().GetRenderExecutor().DrawWindow();
		auto& stream = (*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) %
		                                StreamMemoSlots];
		if (stream.vaddr == vaddr && stream.size == size && stream.window == window &&
		    stream.lap == m_stream_buffer.Lap() &&
		    stream.write_generation == m_gpu_write_generation) {
			return {&m_stream_buffer, stream.offset};
		}
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (m_stream_buffer.IsCoherent()) {
				StreamGuestData(mapped, vaddr, size);
			} else {
				std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			}
			m_stream_buffer.Commit();
			if (bon) bs.ob_stream.fetch_add(1, std::memory_order_relaxed);
			stream = {.vaddr            = vaddr,
			          .size             = size,
			          .window           = window,
			          .lap              = m_stream_buffer.Lap(),
			          .write_generation = m_gpu_write_generation,
			          .offset           = offset};
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		FrameBreakdown::Timer find_timer(bs.ob_find_ns, bon);
		if (bon) bs.ob_find_calls.fetch_add(1, std::memory_order_relaxed);
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	{
		FrameBreakdown::Timer touch_timer(bs.ob_touch_ns, bon);
		TouchBuffer(buffer);
	}
	bool synced_from_image = false;
	// A read of a range without CPU-dirty pages has nothing to upload, and a texel read with no
	// image at the address nothing to copy from one: SynchronizeBuffer would walk the region's
	// pages under its lock and find nothing (about 5,000 of the 5,200 synchronizations a frame of
	// binds made in play). The check reads the bit words without the lock, as the memo above does.
	if ((plain_read || texel_read) && !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		if (bon) bs.ob_clean.fetch_add(1, std::memory_order_relaxed);
	} else {
		FrameBreakdown::Timer sync_timer(bs.ob_sync_ns, bon);
		if (bon) bs.ob_syncs.fetch_add(1, std::memory_order_relaxed);
		synced_from_image = SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	}
	FrameBreakdown::Timer book_timer(bs.ob_book_ns, bon);
	if (plain_read || (texel_read && !synced_from_image)) {
		*memo = {.vaddr        = vaddr,
		         .size         = size,
		         .buffer_epoch = m_buffer_set_epoch,
		         .cpu_epoch    = cpu_epoch,
		         .offset       = buffer.Offset(vaddr),
		         .id           = id,
		         .texel        = texel_read,
		         .image_version = image_version};
	}
	// A texel read changes the buffer only when an image was copied into it (known fills of its
	// memory then no longer hold): dropping them for every texel bind took a lock and a scan of
	// the fills (1.7% of the execution thread). AbPending7: for every texel bind.
	if (is_written || (is_texel_buffer && (synced_from_image ||
	                                       Common::Experiments::Enabled(Common::Experiments::AbPending7)))) {
		DropGpuFills(vaddr, size);
	}
	if (is_written) {
		buffer.last_write_tick = m_scheduler.CurrentTick();
		if (PerfStatsEnabled()) {
			LogGpuWrite(vaddr, size);
		}
		if (FrameDump::TraceEnabled()) {
			FrameDump::TraceWrite("buffer-write", vaddr, vaddr + size, m_writer_kind, m_writer_hash);
		}
		m_gpu_write_generation++;
		{
			std::unique_lock ranges_lock(m_gpu_ranges_mutex);
			m_gpu_modified_ranges.Add(vaddr, size);
			PublishGpuRangesVersion();
			PageWriteSeq::RaiseCurrent(vaddr, size);
		}
		++m_gpu_dirty_generation;
		if (!m_watched_writes.empty() && !m_submit_requested &&
		    Common::Experiments::Enabled(Common::Experiments::EarlySubmitWriters)) {
			for (const auto& [address, bytes]: m_watched_writes) {
				if (address < vaddr + size && vaddr < address + bytes) {
					m_submit_requested = true;
					break;
				}
			}
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferCached(uint64_t vaddr, uint64_t size,
                                                             bool is_written, bool is_texel_buffer,
                                                             BufferId id) {
	if (is_written || is_texel_buffer || !Common::RuntimeFlags::Current().buffer_fast) {
		return ObtainBuffer(vaddr, size, is_written, is_texel_buffer, id);
	}
	// The slot comes from the high bits of the product: its low bits only depend on the low bits of
	// the address, and with 256-byte aligned ranges the old index (the low bits, 512 slots) used
	// one slot in sixteen.
	const auto slot =
	    static_cast<size_t>(((vaddr ^ (size << 40u)) * 0x9E3779B97F4A7C15ull) >> 32u) % kObtainCacheSize;
	auto&      entry = m_obtain_cache[slot];
	const auto cpu   = m_memory_tracker.CpuModificationGeneration();
	if (entry.vaddr == vaddr && entry.size == size && entry.registration == m_registration_generation &&
	    !IsBufferInvalid(entry.id) && m_slot_buffers[entry.id].IsInBounds(vaddr, size) &&
	    (entry.cpu_generation == cpu || !m_memory_tracker.IsRegionCpuModified(vaddr, size))) {
		// Registration unchanged: the buffer set, page table and offsets are the same. The range was
		// CPU-clean when stored and is still CPU-clean, so SynchronizeBuffer would upload nothing.
		entry.cpu_generation = cpu;
		FrameBreakdown::BdaBatchNoteUsed(vaddr, size);
		auto& buffer         = m_slot_buffers[entry.id];
		TouchBuffer(buffer);
		FrameBreakdown::VAdd(FrameBreakdown::g_counters.obtain_cache_hits);
		return {&buffer, entry.offset};
	}
	FrameBreakdown::VAdd(FrameBreakdown::g_counters.obtain_cache_misses);
	// Stamp before the lookup so a CPU write racing with it forces a recheck on the next hit.
	const auto registration = m_registration_generation;
	const auto result       = ObtainBuffer(vaddr, size, false, false, id);
	if (result.first != &m_stream_buffer && registration == m_registration_generation &&
	    !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto found = FindBuffer(vaddr, size);
		if (!IsBufferInvalid(found) && &m_slot_buffers[found] == result.first) {
			entry = {vaddr, size, registration, cpu, found, result.second};
		}
	}
	return result;
}

bool BufferCache::PlainReadHolds(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& cpu_generation,
                                 uint64_t registration) {
	if (&buffer == &m_stream_buffer || registration != m_registration_generation) {
		return false;
	}
	const auto cpu = m_memory_tracker.CpuModificationGeneration();
	if (cpu != cpu_generation && m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		return false;
	}
	cpu_generation = cpu;
	FrameBreakdown::BdaBatchNoteUsed(vaddr, size);
	TouchBuffer(buffer);
	return true;
}

bool BufferCache::WrittenBindHolds(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& ranges_version,
                                   uint64_t registration) {
	if (registration != m_registration_generation) {
		return false;
	}
	if (const auto version = GpuRangesVersion(); version != ranges_version) {
		// Other ranges change all the time (new written ranges, downloads): the range still holds
		// while it is wholly GPU-written, as its pages are then GPU-owned with nothing to upload.
		std::shared_lock ranges_lock(m_gpu_ranges_mutex);
		if (!m_gpu_modified_ranges.Contains(vaddr, size) || m_pending_read_ranges.Intersects(vaddr, size)) {
			return false;
		}
		ranges_version = version;
	}
	FrameBreakdown::BdaBatchNoteUsed(vaddr, size);
	TouchBuffer(buffer);
	// As ObtainBuffer for a written binding, after SynchronizeBuffer (which found the pages GPU-owned).
	DropGpuFills(vaddr, size);
	buffer.last_write_tick = m_scheduler.CurrentTick();
	if (PerfStatsEnabled()) {
		LogGpuWrite(vaddr, size);
	}
	if (FrameDump::TraceEnabled()) {
		FrameDump::TraceWrite("buffer-write", vaddr, vaddr + size, m_writer_kind, m_writer_hash);
	}
	m_gpu_write_generation++;
	PageWriteSeq::RaiseCurrent(vaddr, size);
	++m_gpu_dirty_generation;
	if (!m_watched_writes.empty() && !m_submit_requested &&
	    Common::Experiments::Enabled(Common::Experiments::EarlySubmitWriters)) {
		for (const auto& [address, bytes]: m_watched_writes) {
			if (address < vaddr + size && vaddr < address + bytes) {
				m_submit_requested = true;
				break;
			}
		}
	}
	return true;
}

bool BufferCache::TexelReadHolds(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& cpu_generation,
                                 uint64_t registration, uint64_t image_version) {
	return image_version == m_texture_cache.AddressBucketVersion(vaddr) &&
	       PlainReadHolds(buffer, vaddr, size, cpu_generation, registration);
}

bool BufferCache::TexelReadKeepable(uint64_t vaddr) {
	return !m_texture_cache.HasImageAt(vaddr);
}

uint64_t BufferCache::TexelImageVersion(uint64_t vaddr) const {
	return m_texture_cache.AddressBucketVersion(vaddr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr || !Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	m_gpu_write_generation++;
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Invalidate cached buffers and images for the whole range up front, as the write
		// faults would page by page, so the fill below does not fault on every tracked page.
		WriteTagScope tag(WriteTag::Fill);
		InvalidateMemory(vaddr, size);
		m_texture_cache.InvalidateMemory(vaddr, size);
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	SetWriterTag("fill", 0);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	// The GPU now holds one repeated dword here; color-clear decoding can use it without a download.
	RecordGpuFill(vaddr, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	m_gpu_write_generation++;
	if (!dst_gds) {
		Libs::LibKernel::Memory::NoteHostWrite(dst_vaddr, size,
		                                       Libs::LibKernel::Memory::HostWriteSource::DmaCopy, nullptr);
	}
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	SetWriterTag("copy", 0);
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// A pending guest read makes only the bytes it carries dirty: the execution thread's resource
	// checks on the other bytes of its pages read the backing instead of faulting and waiting for
	// that download (2.7% of the thread in a match).
	if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
		return true;
	}
	if (m_pending_guest_reads.empty()) {
		return false;
	}
	return OverlapsPendingGuestBytes(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size) const {
	// The resolve thread asks per page for each read log: while neither set changed, a page's
	// answer holds (written buffers are added again on every bind, which changes nothing).
	struct PageAnswer {
		uint64_t page    = ~uint64_t {0};
		uint64_t version = 0;
		bool     dirty   = false;
	};
	static thread_local std::array<PageAnswer, 64> answers {};
	const bool page_query = (vaddr & 0xfffu) == 0 && size == 0x1000 &&
	                        !Common::Experiments::Enabled(Common::Experiments::AbLegacy);
	if (page_query) {
		const auto& answer = answers[(vaddr >> 12u) & 63u];
		if (answer.page == vaddr && answer.version == m_gpu_ranges_version.load(std::memory_order_acquire)) {
			return answer.dirty;
		}
	}
	std::shared_lock ranges_lock(m_gpu_ranges_mutex);
	if (page_query) {
		auto& answer   = answers[(vaddr >> 12u) & 63u];
		answer.page    = vaddr;
		answer.version = m_gpu_modified_ranges.Version() + m_pending_read_ranges.Version();
		answer.dirty   = m_pending_read_ranges.Intersects(vaddr, size) || m_gpu_modified_ranges.Intersects(vaddr, size);
		return answer.dirty;
	}
	// Pending records are owned by the execution thread. Other threads see the pages of pending
	// asynchronous downloads as dirty until their bytes are in guest memory (AbLegacy: every range
	// while any download is pending).
	if (Common::Experiments::Enabled(Common::Experiments::AbLegacy)) {
		return m_pending_guest_read_count.load(std::memory_order_acquire) != 0 ||
		       m_gpu_modified_ranges.Intersects(vaddr, size);
	}
	return m_pending_read_ranges.Intersects(vaddr, size) || m_gpu_modified_ranges.Intersects(vaddr, size);
}

uint64_t BufferCache::GpuDirtyByteCount(uint64_t vaddr, uint64_t size) const {
	std::shared_lock ranges_lock(m_gpu_ranges_mutex);
	uint64_t         dirty = 0;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) { dirty += end - start; });
	return dirty;
}

namespace {

// Experiment SpeculateDirtyReads bookkeeping, keyed by read address and size. Counts are totals
// since start.
struct SpecReadEntry {
	uint64_t vaddr       = 0;
	uint64_t size        = 0;
	uint32_t stable      = 0; // consecutive downloads that found the bytes unchanged
	uint64_t downloads   = 0;
	uint64_t changed     = 0;
	uint64_t speculated  = 0;
	uint64_t confirmed   = 0;
	uint64_t mispredicts = 0;
	uint64_t wait_ns     = 0;
	uint64_t context     = 0; // shader being materialized
	uint64_t exec_op     = 0;
	uint8_t  value[16] {};
	uint64_t stack[10] {};
	int      depth = 0;
};
std::mutex                                  g_spec_mutex;
std::unordered_map<uint64_t, SpecReadEntry> g_spec_reads;
std::chrono::steady_clock::time_point       g_spec_last_print {};
uint64_t                                    g_spec_miss_lines = 0;
constexpr uint32_t                          SpecStableNeeded  = 4;
constexpr uint64_t                          SpecMaxBytes      = 16;

uint64_t SpecKey(uint64_t vaddr, uint64_t size) {
	return vaddr ^ (size << 56u);
}

std::string SpecHex(const uint8_t* bytes, uint64_t size) {
	std::string text;
	char        digits[3] {};
	for (uint64_t i = 0; i < size; ++i) {
		std::snprintf(digits, sizeof(digits), "%02x", bytes[i]);
		text += digits;
	}
	return text;
}

// Caller holds g_spec_mutex. KYTY_PERF_STATS: the most-read addresses every 5 s (NHL27SPECREAD).
void SpecMaybePrint() {
	if (!PerfStatsEnabled()) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_spec_last_print < std::chrono::seconds(5)) {
		return;
	}
	g_spec_last_print = now;
	std::vector<const SpecReadEntry*> top;
	top.reserve(g_spec_reads.size());
	for (const auto& entry: g_spec_reads) {
		top.push_back(&entry.second);
	}
	std::sort(top.begin(), top.end(), [](const SpecReadEntry* a, const SpecReadEntry* b) {
		return a->downloads + a->speculated > b->downloads + b->speculated;
	});
	for (size_t i = 0; i < top.size() && i < 6; ++i) {
		const auto& e = *top[i];
		std::string stack;
		for (int f = 0; f < e.depth; ++f) {
			char frame[24] {};
			std::snprintf(frame, sizeof(frame), "%s0x%" PRIx64, f == 0 ? "" : ",", e.stack[f]);
			stack += frame;
		}
		std::printf("NHL27SPECREAD: vaddr=0x%" PRIx64 " size=%" PRIu64 " value=%s downloads=%" PRIu64
		            " changed=%" PRIu64 " speculated=%" PRIu64 " confirmed=%" PRIu64 " mispredicts=%" PRIu64
		            " stable=%u wait_ms=%.1f shader=0x%016" PRIx64 " op=0x%" PRIx64 " stack=%s\n",
		            e.vaddr, e.size, SpecHex(e.value, e.size).c_str(), e.downloads, e.changed, e.speculated,
		            e.confirmed, e.mispredicts, e.stable, static_cast<double>(e.wait_ns) / 1e6, e.context,
		            e.exec_op, stack.c_str());
	}
	std::fflush(stdout);
}

} // namespace

void BufferCache::NoteDirtyRead(uint64_t vaddr, uint64_t size, bool changed, const void* value,
                                uint64_t wait_ns) {
	if (size == 0 || size > SpecMaxBytes || value == nullptr) {
		return;
	}
	std::scoped_lock lock(g_spec_mutex);
	auto&            e = g_spec_reads[SpecKey(vaddr, size)];
	if (e.downloads == 0 && e.speculated == 0) {
		e.vaddr = vaddr;
		e.size  = size;
		e.depth = Libs::LibKernel::Memory::CaptureHostStack(e.stack, 10);
	}
	e.downloads++;
	e.wait_ns += wait_ns;
	e.changed += changed ? 1 : 0;
	e.stable = changed ? 0 : e.stable + 1;
	std::memcpy(e.value, value, size);
	e.context = Libs::LibKernel::Memory::ReadContext();
	e.exec_op = GuestGpu::CurrentExecOp();
	SpecMaybePrint();
}

bool BufferCache::TrySpeculativeRead(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0 || size > SpecMaxBytes || data == nullptr || !GuestGpu::IsGpuThread()) {
		return false;
	}
	const auto key = SpecKey(vaddr, size);
	{
		std::scoped_lock lock(g_spec_mutex);
		const auto       it = g_spec_reads.find(key);
		if (it == g_spec_reads.end() || it->second.stable < SpecStableNeeded) {
			return false;
		}
	}
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !Libs::LibKernel::Memory::TryReadBacking(vaddr, data, size)) {
		return false;
	}
	auto [mapped, offset] = m_download_buffer.Map(SpecMaxBytes, 64);
	if (mapped == nullptr) {
		return false;
	}
	m_download_buffer.Commit();
	// Copy the GPU's bytes back after the work recorded so far, without changing what is tracked
	// as GPU-dirty: later reads and faults still download as before.
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {buffer.Offset(vaddr), offset, size};
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0,
	                       nullptr, 1, &after, 0, nullptr);
	std::array<uint8_t, SpecMaxBytes> expected {};
	std::memcpy(expected.data(), data, size);
	m_scheduler.DeferHostOperation([this, mapped, offset, size, key, expected] {
		m_download_buffer.Invalidate(offset, size);
		const bool       same = std::memcmp(mapped, expected.data(), size) == 0;
		std::scoped_lock lock(g_spec_mutex);
		auto&            e = g_spec_reads[key];
		if (same) {
			e.confirmed++;
			return;
		}
		e.mispredicts++;
		e.stable = 0;
		if (g_spec_miss_lines++ < 64) {
			std::printf("NHL27SPECMISS: vaddr=0x%" PRIx64 " size=%" PRIu64 " guessed=%s gpu=%s shader=0x%016" PRIx64 "\n",
			            e.vaddr, size, SpecHex(expected.data(), size).c_str(), SpecHex(mapped, size).c_str(),
			            e.context);
			std::fflush(stdout);
		}
	});
	std::scoped_lock lock(g_spec_mutex);
	g_spec_reads[key].speculated++;
	SpecMaybePrint();
	return true;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	PerfMaybePrint();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	PerfSetMemoryUsage(m_total_used_memory, 0, m_trigger_gc_memory);
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	// The scan revisits every idle GPU-modified buffer, which it cannot collect, after each guest
	// submission. KYTY_SPARSE_BUFFER_GC scans every 16th one unless memory is critical.
	if (!aggressive && (tick & 15u) != 0 && Common::RuntimeFlags::Current().sparse_buffer_gc) {
		FrameBreakdown::Add(FrameBreakdown::g_counters.port_gc_sparse_skips);
		return;
	}
	FrameBreakdown::Add(FrameBreakdown::g_counters.port_gc_scans);
	// The collector runs after every guest submission, so 160 ticks is only a few frames: buffers for
	// transient data revisited every few frames are deleted and recreated each frame. KYTY_LONG_BUFFER_GC_AGE
	// keeps idle buffers 10x longer, except under critical memory pressure (the aggressive age is unchanged).
	const uint64_t normal_age = Common::RuntimeFlags::Current().long_buffer_gc_age ? 1600 : 160;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : normal_age, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			PerfCount(PerfEvent::BufferGcDelete);
			FrameBreakdown::Add(FrameBreakdown::g_counters.port_buffer_gc_deletes);
			if (m_record_released) {
				m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
			}
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		if (m_record_released) {
			m_released_ranges.Add(buffer.CpuAddress(), buffer.Size());
		}
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::NoteGpuWrites(uint64_t vaddr, uint64_t size) {
	m_gpu_write_generation++;
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid GPU write range\n");
	}
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start >= finish) {
			continue;
		}
		TouchBuffer(buffer);
		m_memory_tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
		    start, finish - start, [&](uint64_t page, uint64_t bytes) {
			    const auto first = std::max(page, start);
			    const auto last  = std::min(page + bytes, finish);
			    std::unique_lock ranges_lock(m_gpu_ranges_mutex);
			    m_gpu_modified_ranges.Add(first, last - first);
			    PublishGpuRangesVersion();
			    PageWriteSeq::RaiseCurrent(first, last - first);
		    });
	}
}

uint64_t BufferCache::SynchronizeBuffersInRanges(const RangeSet& ranges, bool full) {
	// Drain before a full scan so a CPU write during that scan remains queued.
	const auto visited = m_memory_tracker.DrainCpuDirtyRegions([&](uint64_t address, uint64_t size) {
		if (!full) {
			ranges.ForEachInRange(address, size, [&](uint64_t begin, uint64_t end) {
				SynchronizeBuffersInRange(begin, end - begin);
			});
		}
	});
	if (full) {
		ranges.ForEach([&](uint64_t begin, uint64_t end) {
			SynchronizeBuffersInRange(begin, end - begin);
		});
	}
	return visited;
}

bool BufferCache::SynchronizeBuffersInRangesBefore(const RangeSet& ranges, const RangeSet* full_ranges,
                                                   uint32_t limit, uint32_t& first_left) {
	// Pages uploaded here are protected again by the host copy thread, before the uploads' copies
	// (which it makes as well: the staging memory is coherent).
	auto* const host_copies = m_scheduler.HostCopies();
	const bool  defer       = host_copies != nullptr && m_staging_buffer.IsCoherent() && GuestGpu::IsGpuThread();
	const PageManager::DeferredProtectionScope deferred(defer ? host_copies : nullptr);
	bool       left = false;
	const auto sync = [&](uint64_t vaddr, uint64_t size) {
		left |= m_memory_tracker.ForEachCpuDirtySpanBefore(
		    vaddr, size, limit, first_left, [&](uint64_t span, uint64_t span_size) {
			    const auto end = span + span_size;
			    auto       it  = m_buffers.upper_bound(span);
			    if (it != m_buffers.begin()) {
				    --it;
			    }
			    for (; it != m_buffers.end() && it->first < end; ++it) {
				    auto&      buffer = m_slot_buffers[it->second];
				    const auto start  = std::max(buffer.CpuAddress(), span);
				    const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
				    if (start < finish) {
					    (void)SynchronizeBuffer(buffer, start, finish - start, false, false);
				    }
			    }
		    });
	};
	// Drain before a full scan so a CPU write during that scan remains queued.
	const bool all = full_ranges == &ranges;
	m_memory_tracker.DrainCpuDirtyRegions([&](uint64_t address, uint64_t size) {
		if (!all) {
			ranges.ForEachInRange(address, size, [&](uint64_t begin, uint64_t end) { sync(begin, end - begin); });
		}
	});
	if (full_ranges != nullptr) {
		full_ranges->ForEach([&](uint64_t begin, uint64_t end) { sync(begin, end - begin); });
	}
	return left;
}

void BufferCache::SynchronizeDirtyRegionsInRanges(const RangeSet& ranges, bool all) {
	// Read each region's epoch before scanning it: pages dirtied during or after the scan advance
	// it again, and the next call visits the region again.
	m_sync_regions.clear();
	ranges.ForEach([&](uint64_t start, uint64_t end) {
		for (auto index = start / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < end; index++) {
			if (!m_sync_regions.empty() && m_sync_regions.back().first == index) {
				continue;
			}
			const auto epoch = m_memory_tracker.RegionCpuDirtyEpoch(index);
			if (!all) {
				// Synchronizing a buffer tracks its pages, so an untracked region holds no buffer
				// since the last full pass.
				if (epoch == 0) {
					continue;
				}
				const auto synced = m_synced_region_epochs.find(index);
				if (synced != m_synced_region_epochs.end() && synced->second == epoch) {
					continue;
				}
			}
			m_sync_regions.emplace_back(index, epoch);
		}
	});
	if (all) {
		m_synced_region_epochs.clear();
		ranges.ForEach([this](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start);
		});
	} else {
		for (const auto& [index, epoch]: m_sync_regions) {
			ranges.ForEachInRange(index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE,
			                      [this](uint64_t start, uint64_t end) {
				                      SynchronizeBuffersInRange(start, end - start);
			                      });
		}
	}
	for (const auto& [index, epoch]: m_sync_regions) {
		m_synced_region_epochs[index] = epoch;
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	// Only tracking regions with pending CPU writes can have anything to upload.
	m_memory_tracker.ForEachCpuDirtySpan(vaddr, size, [&](uint64_t span, uint64_t span_size) {
		const auto end = span + span_size;
		auto       it  = m_buffers.upper_bound(span);
		if (it != m_buffers.begin()) {
			--it;
		}
		for (; it != m_buffers.end() && it->first < end; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto start  = std::max(buffer.CpuAddress(), span);
			const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
			if (start < finish) {
				(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
			}
		}
	});
}

} // namespace Libs::Graphics
