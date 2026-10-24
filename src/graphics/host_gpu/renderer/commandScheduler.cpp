#include "common/eventTrace.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/emulatorConfig.h"
#include "common/experiments.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fmt/format.h>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

struct BreadcrumbEntry {
	const char* kind   = nullptr;
	uint64_t    hash_a = 0;
	uint64_t    hash_b = 0;
	uint64_t    info   = 0;
};

struct BreadcrumbState {
	std::mutex                              mutex;
	bool                                    initialized = false;
	bool                                    available   = false;
	VkBuffer                                buffer      = VK_NULL_HANDLE;
	VmaAllocation                           allocation  = nullptr;
	volatile uint32_t*                      mapped      = nullptr;
	VkBuffer                                args_buffer = VK_NULL_HANDLE;
	VmaAllocation                           args_allocation = nullptr;
	volatile uint32_t*                      args_mapped = nullptr;
	uint32_t                                next_id     = 1;
	std::array<BreadcrumbEntry, 1u << 16u>  entries {};
};

BreadcrumbState g_breadcrumbs;

bool BreadcrumbInit(GraphicContext& graphics) {
	if (g_breadcrumbs.initialized) {
		return g_breadcrumbs.available;
	}
	g_breadcrumbs.initialized = true;
	const char* enabled       = std::getenv("KYTY_DBG_BREADCRUMBS");
	if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0') {
		return false;
	}
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteBufferMarkerAMD == nullptr) {
		std::printf("NHL26CRUMB: VK_AMD_buffer_marker not available\n");
		return false;
	}
	VkBufferCreateInfo buffer_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer_info.size  = 64;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
	                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
	allocation_info.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VmaAllocationInfo result_info {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info,
	                    &g_breadcrumbs.buffer, &g_breadcrumbs.allocation, &result_info) !=
	        VK_SUCCESS ||
	    result_info.pMappedData == nullptr) {
		std::printf("NHL26CRUMB: breadcrumb buffer allocation failed\n");
		return false;
	}
	g_breadcrumbs.mapped    = static_cast<volatile uint32_t*>(result_info.pMappedData);
	g_breadcrumbs.mapped[0] = 0;
	g_breadcrumbs.mapped[1] = 0;
	// One 16-byte slot per breadcrumb entry for captured indirect arguments.
	buffer_info.size  = g_breadcrumbs.entries.size() * 16u;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationInfo args_info {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info,
	                    &g_breadcrumbs.args_buffer, &g_breadcrumbs.args_allocation, &args_info) ==
	        VK_SUCCESS &&
	    args_info.pMappedData != nullptr) {
		g_breadcrumbs.args_mapped = static_cast<volatile uint32_t*>(args_info.pMappedData);
		for (size_t i = 0; i < g_breadcrumbs.entries.size() * 4u; ++i) {
			g_breadcrumbs.args_mapped[i] = 0xdeadbeefu;
		}
	} else {
		g_breadcrumbs.args_buffer = VK_NULL_HANDLE;
	}
	g_breadcrumbs.available = true;
	return true;
}

void DumpBreadcrumbs() {
	std::lock_guard lock(g_breadcrumbs.mutex);
	if (!g_breadcrumbs.available) {
		return;
	}
	const uint32_t started  = g_breadcrumbs.mapped[0];
	const uint32_t finished = g_breadcrumbs.mapped[1];
	const uint32_t recorded = g_breadcrumbs.next_id - 1u;
	std::printf("NHL26CRUMB: last started=%u last finished=%u last recorded=%u\n", started,
	            finished, recorded);
	const uint32_t first = finished > 4u ? finished - 4u : 1u;
	const uint32_t last  = std::min(recorded, std::max(started, finished) + 4u);
	for (uint32_t id = first; id <= last && id != 0; id++) {
		const auto& e = g_breadcrumbs.entries[id & (g_breadcrumbs.entries.size() - 1u)];
		const char* tag = id <= finished ? "done   " : (id <= started ? "RUNNING" : "pending");
		std::printf("NHL26CRUMB:   [%s] id=%u %s a=0x%016llx b=0x%016llx info=0x%llx\n", tag, id,
		            e.kind != nullptr ? e.kind : "?", static_cast<unsigned long long>(e.hash_a),
		            static_cast<unsigned long long>(e.hash_b),
		            static_cast<unsigned long long>(e.info));
		if (g_breadcrumbs.args_mapped != nullptr && e.kind != nullptr &&
		    std::strstr(e.kind, "indirect") != nullptr) {
			const auto* args = g_breadcrumbs.args_mapped + (id & (g_breadcrumbs.entries.size() - 1u)) * 4u;
			std::printf("NHL26CRUMB:     args = %u %u %u (0x%08x 0x%08x 0x%08x)\n", args[0], args[1],
			            args[2], args[0], args[1], args[2]);
		}
	}
	std::fflush(stdout);
}

// Always on, CPU side only (no GPU markers, so no timing change): the last draws and dispatches
// recorded, with the scheduler tick of their command buffer. After a device loss, the ones in ticks
// the GPU had not finished name the work it hung on.
struct CpuCrumb {
	const char* kind = nullptr;
	uint64_t    hash_a = 0, hash_b = 0, info = 0, tick = 0;
};
// About a frame of draws and dispatches, small enough to stay in the L2 cache (16384 entries, 640
// KiB, missed it: 0.9% of the execution thread). The execution thread records them all.
std::array<CpuCrumb, 4096> g_cpu_crumbs;
std::atomic<uint64_t>      g_cpu_crumb_next {0};
const MasterSemaphore*     g_crumb_master = nullptr;

void NoteCpuCrumb(const char* kind, uint64_t hash_a, uint64_t hash_b, uint64_t info) {
	if (g_crumb_master == nullptr) {
		return;
	}
	const auto n = g_cpu_crumb_next.load(std::memory_order_relaxed);
	g_cpu_crumb_next.store(n + 1, std::memory_order_relaxed);
	g_cpu_crumbs[n % g_cpu_crumbs.size()] = {kind, hash_a, hash_b, info, g_crumb_master->CurrentTick()};
}

void DumpCpuCrumbs() {
	if (g_crumb_master == nullptr) {
		return;
	}
	const auto known = g_crumb_master->KnownGpuTick();
	const auto next  = g_cpu_crumb_next.load(std::memory_order_relaxed);
	const auto first = next > g_cpu_crumbs.size() ? next - g_cpu_crumbs.size() : 0;
	std::printf("NHL27HANG: GPU finished tick %llu, recording tick %llu\n",
	            static_cast<unsigned long long>(known),
	            static_cast<unsigned long long>(g_crumb_master->CurrentTick()));
	// The first unfinished commands and the last ones (the GPU hangs somewhere in between, most
	// likely early, but the last ones show what was queued behind).
	uint64_t unfinished = 0;
	for (auto n = first; n < next; n++) {
		const auto& c = g_cpu_crumbs[n % g_cpu_crumbs.size()];
		unfinished += (c.kind != nullptr && c.tick > known) ? 1u : 0u;
	}
	std::printf("NHL27HANG: %llu unfinished commands\n", static_cast<unsigned long long>(unfinished));
	uint64_t index = 0;
	for (auto n = first; n < next; n++) {
		const auto& c = g_cpu_crumbs[n % g_cpu_crumbs.size()];
		if (c.kind == nullptr || c.tick <= known) {
			continue;
		}
		if (index < 32 || index + 32 >= unfinished) {
			std::printf("NHL27HANG:   #%llu tick %llu %s a=0x%016llx b=0x%016llx info=0x%llx\n",
			            static_cast<unsigned long long>(index), static_cast<unsigned long long>(c.tick), c.kind,
			            static_cast<unsigned long long>(c.hash_a), static_cast<unsigned long long>(c.hash_b),
			            static_cast<unsigned long long>(c.info));
		}
		index++;
	}
	std::fflush(stdout);
}

// NHL26 debugging: after a device loss, print what VK_EXT_device_fault reports (faulting GPU
// addresses and vendor fault codes). No-op when the extension is not enabled.
void DumpDeviceFault(vk::Device device) {
	DumpCpuCrumbs();
	DumpBreadcrumbs();
	const auto get_fault_info = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, "vkGetDeviceFaultInfoEXT"));
	if (get_fault_info == nullptr) {
		std::printf("NHL26FAULT: VK_EXT_device_fault not available\n");
		return;
	}
	VkDeviceFaultCountsEXT counts {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
	if (get_fault_info(device, &counts, nullptr) < 0) {
		std::printf("NHL26FAULT: vkGetDeviceFaultInfoEXT (counts) failed\n");
		return;
	}
	std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	VkDeviceFaultInfoEXT info {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
	info.pAddressInfos     = addresses.data();
	info.pVendorInfos      = vendors.data();
	counts.vendorBinarySize = 0;
	const auto result      = get_fault_info(device, &counts, &info);
	std::printf("NHL26FAULT: result=%d description=\"%s\" addresses=%u vendor_infos=%u\n",
	            static_cast<int>(result), info.description, counts.addressInfoCount,
	            counts.vendorInfoCount);
	static const char* const kTypes[] = {"none",       "read-invalid",  "write-invalid",
	                                     "exec-invalid", "ip-unknown",  "ip-invalid",
	                                     "ip-fault"};
	for (const auto& a: addresses) {
		const auto type = static_cast<uint32_t>(a.addressType);
		std::printf("NHL26FAULT:   address type=%s addr=0x%016llx precision=0x%llx\n",
		            type < std::size(kTypes) ? kTypes[type] : "?",
		            static_cast<unsigned long long>(a.reportedAddress),
		            static_cast<unsigned long long>(a.addressPrecision));
	}
	for (const auto& v: vendors) {
		std::printf("NHL26FAULT:   vendor \"%s\" code=0x%llx data=0x%llx\n", v.description,
		            static_cast<unsigned long long>(v.vendorFaultCode),
		            static_cast<unsigned long long>(v.vendorFaultData));
	}
	std::fflush(stdout);
}

// NHL27 profiling (KYTY_PERF_GPU_TIMING=1): per-draw / per-dispatch GPU timing. Timestamps are
// written around each command into a ring of (begin, end) query pairs; completed pairs are read
// back without waiting after every Submit and accumulated per (kind, hash_a, hash_b). Every few
// seconds the top keys by total GPU time are printed. The begin/end timestamps are
// top-of-pipe/bottom-of-pipe, so back-to-back commands that overlap on the GPU are each charged for
// the overlap; the sum can exceed the real GPU-busy time.
constexpr uint32_t GPU_TIMING_PAIRS      = 16384;
constexpr uint32_t GPU_TIMING_READ_CHUNK = 256;
constexpr double   GPU_TIMING_WINDOW_SEC = 5.0;
constexpr size_t   GPU_TIMING_TOP        = 20;

struct GpuTimingSlot {
	const char* kind   = nullptr;
	uint64_t    hash_a = 0;
	uint64_t    hash_b = 0;
	uint64_t    info   = 0;
};

struct GpuTimingAccum {
	double   total_ms = 0.0;
	uint64_t count    = 0;
	uint64_t info     = 0;
};

using GpuTimingKey = std::tuple<std::string, uint64_t, uint64_t>;

struct GpuTimingState {
	std::mutex                            mutex;
	bool                                  initialized = false;
	bool                                  available   = false;
	vk::Device                            device      = nullptr;
	vk::QueryPool                         pool        = nullptr;
	double                                period_ns   = 0.0;
	uint64_t                              valid_mask  = ~0ull;
	uint64_t                              head        = 0;
	uint64_t                              tail        = 0;
	uint64_t                              dropped     = 0;
	uint64_t                              samples     = 0;
	double                                total_ms    = 0.0;
	std::chrono::steady_clock::time_point window_start;
	std::vector<GpuTimingSlot>            slots;
	std::vector<uint64_t>                 results;
	std::map<GpuTimingKey, GpuTimingAccum> accum;
};

GpuTimingState g_timing;

bool GpuTimingInit(GraphicContext& graphics) {
	if (g_timing.initialized) {
		return g_timing.available;
	}
	g_timing.initialized = true;
	const char* enabled  = std::getenv("KYTY_PERF_GPU_TIMING");
	if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0') {
		return false;
	}
	if (!graphics.host_query_reset_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkResetQueryPool == nullptr) {
		std::printf("NHL27GPU: hostQueryReset not available, GPU timing disabled\n");
		return false;
	}
	const auto families = graphics.physical_device.getQueueFamilyProperties();
	if (graphics.queue_family >= families.size()) {
		return false;
	}
	const uint32_t valid_bits = families[graphics.queue_family].timestampValidBits;
	if (valid_bits == 0 || graphics.physical_device_properties.limits.timestampPeriod <= 0.0f) {
		return false;
	}
	g_timing.valid_mask = valid_bits >= 64 ? ~0ull : ((1ull << valid_bits) - 1ull);
	g_timing.period_ns  = graphics.physical_device_properties.limits.timestampPeriod;

	vk::QueryPoolCreateInfo create {};
	create.queryType  = vk::QueryType::eTimestamp;
	create.queryCount = GPU_TIMING_PAIRS * 2u;
	if (graphics.device.createQueryPool(&create, nullptr, &g_timing.pool) != vk::Result::eSuccess) {
		std::printf("NHL27GPU: query pool creation failed\n");
		return false;
	}
	g_timing.device = graphics.device;
	g_timing.slots.resize(GPU_TIMING_PAIRS);
	g_timing.results.resize(static_cast<size_t>(GPU_TIMING_READ_CHUNK) * 4u);
	g_timing.window_start = std::chrono::steady_clock::now();
	g_timing.available    = true;
	std::printf("NHL27GPU: GPU timing enabled (%u slots, %.3f ns/tick, %u valid bits)\n",
	            GPU_TIMING_PAIRS, g_timing.period_ns, valid_bits);
	std::fflush(stdout);
	return true;
}

void GpuTimingReport() {
	const auto   now    = std::chrono::steady_clock::now();
	const double window = std::chrono::duration<double>(now - g_timing.window_start).count();
	std::vector<std::pair<const GpuTimingKey*, const GpuTimingAccum*>> sorted;
	for (const auto& [key, acc]: g_timing.accum) {
		sorted.emplace_back(&key, &acc);
	}
	std::sort(sorted.begin(), sorted.end(),
	          [](const auto& l, const auto& r) { return l.second->total_ms > r.second->total_ms; });
	std::printf("NHL27GPU: ---- window=%.2f s measured=%.1f ms (%.1f ms/s) samples=%llu "
	            "dropped=%llu keys=%zu\n",
	            window, g_timing.total_ms, window > 0.0 ? g_timing.total_ms / window : 0.0,
	            static_cast<unsigned long long>(g_timing.samples),
	            static_cast<unsigned long long>(g_timing.dropped), sorted.size());
	{
		// Every key by kind (draw, dispatch...): operations, time, and those under 10 us each.
		std::map<std::string, std::array<double, 4>> kinds;
		for (const auto& [key, acc]: sorted) {
			auto&      sums = kinds[std::get<0>(*key)];
			const auto avg  = acc->count != 0 ? acc->total_ms * 1000.0 / static_cast<double>(acc->count) : 0.0;
			sums[0] += static_cast<double>(acc->count);
			sums[1] += acc->total_ms;
			if (avg < 10.0) {
				sums[2] += static_cast<double>(acc->count);
				sums[3] += acc->total_ms;
			}
		}
		std::string line = "NHL27GPUKINDS:";
		for (const auto& [kind, sums]: kinds) {
			line += fmt::format(" {}: n={:.0f} {:.1f} ms (under 10 us: n={:.0f} {:.1f} ms)", kind, sums[0], sums[1],
			                    sums[2], sums[3]);
		}
		std::printf("%s\n", line.c_str());
	}
	for (size_t i = 0; i < sorted.size() && i < GPU_TIMING_TOP; i++) {
		const GpuTimingKey&   key = *sorted[i].first;
		const GpuTimingAccum& acc = *sorted[i].second;
		std::printf("NHL27GPU: %.2f ms %.1f%% n=%llu avg=%.1f us %s a=0x%llx b=0x%llx info=0x%llx\n",
		            acc.total_ms,
		            g_timing.total_ms > 0.0 ? 100.0 * acc.total_ms / g_timing.total_ms : 0.0,
		            static_cast<unsigned long long>(acc.count),
		            acc.count != 0 ? acc.total_ms * 1000.0 / static_cast<double>(acc.count) : 0.0,
		            std::get<0>(key).c_str(), static_cast<unsigned long long>(std::get<1>(key)),
		            static_cast<unsigned long long>(std::get<2>(key)),
		            static_cast<unsigned long long>(acc.info));
	}
	std::fflush(stdout);
	g_timing.accum.clear();
	g_timing.total_ms     = 0.0;
	g_timing.samples      = 0;
	g_timing.dropped      = 0;
	g_timing.window_start = now;
}

// Reads back completed (begin, end) pairs in recording order, without waiting. Caller holds the mutex.
void GpuTimingPollLocked() {
	while (g_timing.tail < g_timing.head) {
		const uint32_t first = static_cast<uint32_t>(g_timing.tail % GPU_TIMING_PAIRS);
		const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(
		    {g_timing.head - g_timing.tail, GPU_TIMING_READ_CHUNK, GPU_TIMING_PAIRS - first}));
		// 64-bit results, each followed by its availability value: 4 uint64 per pair.
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
		    static_cast<VkDevice>(g_timing.device), static_cast<VkQueryPool>(g_timing.pool),
		    first * 2u, count * 2u, g_timing.results.size() * sizeof(uint64_t),
		    g_timing.results.data(), 2u * sizeof(uint64_t),
		    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
		uint32_t done = 0;
		for (; done < count; done++) {
			const uint64_t* r = &g_timing.results[static_cast<size_t>(done) * 4u];
			if (r[1] == 0 || r[3] == 0) {
				break;
			}
			const auto&    slot = g_timing.slots[first + done];
			const uint64_t diff = (r[2] - r[0]) & g_timing.valid_mask;
			const double   ms   = static_cast<double>(diff) * g_timing.period_ns * 1e-6;
			auto&          acc  = g_timing.accum[GpuTimingKey {
                slot.kind != nullptr ? slot.kind : "?", slot.hash_a, slot.hash_b}];
			acc.total_ms += ms;
			acc.count++;
			acc.info = slot.info;
			g_timing.total_ms += ms;
			g_timing.samples++;
		}
		g_timing.tail += done;
		if (done < count) {
			break;
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (std::chrono::duration<double>(now - g_timing.window_start).count() >=
	    GPU_TIMING_WINDOW_SEC) {
		GpuTimingReport();
	}
}

// Returns the slot index + 1, or 0 when timing is off or the sample is dropped.
uint32_t GpuTimingBegin(GraphicContext& graphics, const CommandRecorder& cmd, const char* kind,
                        uint64_t hash_a, uint64_t hash_b, uint64_t info) {
	if (g_timing.initialized && !g_timing.available) {
		return 0;
	}
	std::lock_guard lock(g_timing.mutex);
	if (!GpuTimingInit(graphics)) {
		return 0;
	}
	if (g_timing.head - g_timing.tail >= GPU_TIMING_PAIRS) {
		GpuTimingPollLocked();
		if (g_timing.head - g_timing.tail >= GPU_TIMING_PAIRS) {
			g_timing.dropped++; // Ring still full: drop the sample rather than stall.
			return 0;
		}
	}
	const uint32_t slot = static_cast<uint32_t>(g_timing.head % GPU_TIMING_PAIRS);
	g_timing.head++;
	g_timing.slots[slot] = {kind, hash_a, hash_b, info};
	// The previous use of this pair has been read back (tail passed it), so the host reset is safe.
	VULKAN_HPP_DEFAULT_DISPATCHER.vkResetQueryPool(static_cast<VkDevice>(g_timing.device),
	                                               static_cast<VkQueryPool>(g_timing.pool), slot * 2u,
	                                               2u);
	// KYTY_PERF_GPU_TIMING=2 starts each interval when all earlier work completed, so intervals
	// do not overlap and add up to GPU busy time (incremental cost per operation).
	static const bool incremental = [] {
		const char* value = std::getenv("KYTY_PERF_GPU_TIMING");
		return value != nullptr && value[0] == '2';
	}();
	cmd.writeTimestamp(incremental ? vk::PipelineStageFlagBits::eBottomOfPipe
	                               : vk::PipelineStageFlagBits::eTopOfPipe,
	                   g_timing.pool, slot * 2u);
	return slot + 1u;
}

void GpuTimingEnd(const CommandRecorder& cmd, uint32_t slot_plus_one) {
	if (slot_plus_one == 0) {
		return;
	}
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, g_timing.pool,
	                   (slot_plus_one - 1u) * 2u + 1u);
}

// KYTY_IDLE_TRACE per-command timestamps (defined with the rest of the GPU trace code below).
uint32_t GpuTraceCmdBegin(GraphicContext& graphics, const CommandRecorder& cmd, const char* kind,
                          uint64_t hash_a, uint64_t hash_b, uint64_t info, uint64_t aux);
void     GpuTraceCmdEnd(const CommandRecorder& cmd, uint32_t slot_plus_one);

} // namespace

uint64_t BreadcrumbBegin(GraphicContext& graphics, const CommandRecorder& cmd, const char* kind,
                         uint64_t hash_a, uint64_t hash_b, uint64_t info, uint64_t aux) {
	// Bits 0-31: breadcrumb id (0 = none). Bits 32-46: KYTY_PERF_GPU_TIMING slot + 1 (0 = none).
	// Bits 47-63: KYTY_IDLE_TRACE per-command slot + 1 (0 = none).
	NoteCpuCrumb(kind, hash_a, hash_b, info);
	uint32_t id = 0;
	if (!(g_breadcrumbs.initialized && !g_breadcrumbs.available)) {
		std::lock_guard lock(g_breadcrumbs.mutex);
		if (BreadcrumbInit(graphics)) {
			id = g_breadcrumbs.next_id++;
			g_breadcrumbs.entries[id & (g_breadcrumbs.entries.size() - 1u)] = {kind, hash_a,
			                                                                   hash_b, info};
			cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_breadcrumbs.buffer, 0,
			                         id);
		}
	}
	const uint32_t timing = GpuTimingBegin(graphics, cmd, kind, hash_a, hash_b, info);
	const uint32_t trace = GpuTraceCmdBegin(graphics, cmd, kind, hash_a, hash_b, info, aux);
	return (static_cast<uint64_t>(trace) << 47u) | (static_cast<uint64_t>(timing & 0x7fffu) << 32u) | id;
}

void ReportDeviceLost(GraphicContext& graphics) {
	DumpDeviceFault(graphics.device);
}

void BreadcrumbCaptureArgs(const CommandRecorder& cmd, uint64_t id, vk::Buffer source, uint64_t offset,
                           uint32_t bytes) {
	const auto crumb = static_cast<uint32_t>(id);
	if (crumb == 0 || !g_breadcrumbs.available || g_breadcrumbs.args_buffer == VK_NULL_HANDLE ||
	    bytes == 0 || bytes > 16u) {
		return;
	}
	vk::MemoryBarrier written {};
	written.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	written.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                    {}, 1, &written, 0, nullptr, 0, nullptr);
	const vk::BufferCopy copy {offset, (crumb & (g_breadcrumbs.entries.size() - 1u)) * 16u, bytes};
	cmd.copyBuffer(source, vk::Buffer(g_breadcrumbs.args_buffer), 1, &copy);
}

void BreadcrumbEnd(GraphicContext& graphics, const CommandRecorder& cmd, uint64_t id) {
	(void)graphics;
	GpuTraceCmdEnd(cmd, static_cast<uint32_t>(id >> 47u));
	GpuTimingEnd(cmd, static_cast<uint32_t>((id >> 32u) & 0x7fffu));
	const auto crumb = static_cast<uint32_t>(id);
	if (crumb == 0 || !g_breadcrumbs.available) {
		return;
	}
	cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_breadcrumbs.buffer, 4,
	                         crumb);
}

void GpuTimingPoll() {
	if (!g_timing.available) {
		return;
	}
	std::lock_guard lock(g_timing.mutex);
	GpuTimingPollLocked();
}

// KYTY_IDLE_TRACE GPU timeline. While the event trace is capturing, every native command buffer gets a
// top-of-pipe timestamp when it begins and a bottom-of-pipe timestamp just before it is ended; after
// completion the pair is converted to the trace clock (Common::EventTrace::NowNs == steady_clock ns)
// and emitted as GPU_BEGIN / GPU_END (a = tick). Independent of KYTY_PERF_GPU_TIMING.
namespace {

constexpr uint32_t GPU_TRACE_PAIRS = 4096; // 8192 queries
// Per draw / dispatch (begin, end) pairs: 131072 queries.
constexpr uint32_t GPU_TRACE_CMD_PAIRS = 65536;

struct GpuTraceCmd {
	uint32_t slot;
	uint32_t index; // Command ordinal within its command buffer (counts skipped ones too).
	bool     dispatch;
	uint64_t identity; // Shader hash reported in the GPU_DRAW/GPU_DISPATCH event.
	uint64_t size;     // Saturated vertex (draw) or thread-group (dispatch) count.
};

struct GpuTraceCmdPending {
	uint64_t    tick;
	GpuTraceCmd cmd;
};

// Side file row key: (kind, vs, ps/cs, aux). Identity = ps != 0 ? ps : vs for draws, cs for dispatch.
using GpuTraceShaderKey = std::tuple<std::string, uint64_t, uint64_t, uint64_t>;

struct GpuTracePending {
	uint64_t tick;
	uint32_t slot;
};

struct GpuTraceState {
	std::mutex                   mutex;
	bool                         initialized = false;
	bool                         available   = false;
	vk::Device                   device      = nullptr;
	vk::QueryPool                pool        = nullptr;
	double                       period_ns   = 0.0;
	uint64_t                     valid_mask  = ~0ull;
	unsigned                     valid_bits  = 64;
	bool                         can_calibrate = false;
	uint32_t                     time_domain = 0; // VkTimeDomainEXT of the CPU side
	std::vector<uint32_t>        free_slots;
	std::vector<GpuTracePending> pending; // in submit order
	std::vector<uint64_t>        results;
	// Per-capture clock mapping (keyed by g_start_ns).
	int64_t  mapped_start_ns = -1;
	bool     calibrated      = false; // false -> aligned fallback
	bool     have_anchor     = false;
	uint64_t anchor_gpu      = 0;
	int64_t  anchor_cpu_ns   = 0;
	bool     end_check_due   = false;
	// Per-command timestamps.
	vk::QueryPool                         cmd_pool = nullptr;
	std::vector<uint32_t>                 cmd_free;
	std::vector<GpuTraceCmd>              cmd_open; // Recorded into the command buffer being built.
	uint32_t                              cmd_open_count = 0;
	std::vector<GpuTraceCmdPending>       cmd_pending; // Submitted, in submit order.
	std::vector<uint64_t>                 cmd_results;
	uint64_t                              cmd_skipped = 0;
	std::map<GpuTraceShaderKey, uint64_t> shader_rows; // -> draw count
};

GpuTraceState g_gtrace;

int64_t GpuTraceSignedDelta(uint64_t now, uint64_t ref) {
	uint64_t d = (now - ref) & g_gtrace.valid_mask;
	if (g_gtrace.valid_bits < 64 && (d & (1ull << (g_gtrace.valid_bits - 1u))) != 0) {
		d |= ~g_gtrace.valid_mask;
	}
	return static_cast<int64_t>(d);
}

// Samples (GPU ticks, CPU time in steady_clock ns) at the same instant. On Windows steady_clock is
// QueryPerformanceCounter-based: ns = qpc * 1e9 / freq (the STL does exactly this, with epoch = QPC
// zero), so QPC ticks convert to steady_clock ns directly. On other platforms CLOCK_MONOTONIC is
// already steady_clock ns.
bool GpuTraceCalibrate(uint64_t& gpu_ticks, int64_t& cpu_ns) {
	if (!g_gtrace.can_calibrate) {
		return false;
	}
	VkCalibratedTimestampInfoEXT infos[2] {};
	infos[0].sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
	infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
	infos[1].sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
	infos[1].timeDomain = static_cast<VkTimeDomainEXT>(g_gtrace.time_domain);
	uint64_t ts[2]      = {0, 0};
	uint64_t deviation  = 0;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsEXT(
	        static_cast<VkDevice>(g_gtrace.device), 2, infos, ts, &deviation) != VK_SUCCESS) {
		return false;
	}
	gpu_ticks = ts[0] & g_gtrace.valid_mask;
#if defined(_WIN32)
	static const uint64_t freq = [] {
		LARGE_INTEGER f {};
		QueryPerformanceFrequency(&f);
		return static_cast<uint64_t>(f.QuadPart);
	}();
	if (freq == 0) {
		return false;
	}
	cpu_ns = static_cast<int64_t>((ts[1] / freq) * 1'000'000'000ull +
	                              (ts[1] % freq) * 1'000'000'000ull / freq);
#else
	cpu_ns = static_cast<int64_t>(ts[1]);
#endif
	return true;
}

// Writes idle-trace-<unix>-shaders.csv (same unix time as the event CSV) and clears the rows.
void GpuTraceWriteShaderFile(long long unix_s) {
	std::lock_guard lock(g_gtrace.mutex);
	if (!g_gtrace.shader_rows.empty()) {
		char name[96];
		std::snprintf(name, sizeof(name), "idle-trace-%lld-shaders.csv", unix_s);
		if (FILE* f = std::fopen(name, "w")) {
			std::fprintf(f, "identity,kind,vs,ps_or_cs,rt_format,rt_width,rt_height,draws\n");
			for (const auto& [key, count]: g_gtrace.shader_rows) {
				const uint64_t vs = std::get<1>(key), ps = std::get<2>(key), aux = std::get<3>(key);
				const bool     dispatch = std::get<0>(key).rfind("dispatch", 0) == 0;
				const uint64_t identity = dispatch ? vs : (ps != 0 ? ps : vs);
				const auto     fmt      = static_cast<uint32_t>(aux >> 32u);
				std::fprintf(f, "0x%016llx,%s,0x%016llx,0x%016llx,%s,%u,%u,%llu\n",
				             static_cast<unsigned long long>(identity), std::get<0>(key).c_str(),
				             static_cast<unsigned long long>(dispatch ? 0 : vs),
				             static_cast<unsigned long long>(dispatch ? vs : ps),
				             fmt != 0 ? vk::to_string(static_cast<vk::Format>(fmt)).c_str() : "none",
				             static_cast<uint32_t>((aux >> 16u) & 0xffffu),
				             static_cast<uint32_t>(aux & 0xffffu),
				             static_cast<unsigned long long>(count));
			}
			std::fclose(f);
			std::printf("NHL27IDLETRACE: wrote %s (%zu rows, %llu commands without a timestamp slot)\n",
			            name, g_gtrace.shader_rows.size(),
			            static_cast<unsigned long long>(g_gtrace.cmd_skipped));
			std::fflush(stdout);
		}
	}
	g_gtrace.shader_rows.clear();
	g_gtrace.cmd_skipped = 0;
}

bool GpuTraceInitLocked(GraphicContext& graphics) {
	if (g_gtrace.initialized) {
		return g_gtrace.available;
	}
	g_gtrace.initialized = true;
	if (!graphics.host_query_reset_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkResetQueryPool == nullptr) {
		std::printf("NHL27IDLETRACE: hostQueryReset not available, GPU timeline disabled\n");
		return false;
	}
	const auto families = graphics.physical_device.getQueueFamilyProperties();
	if (graphics.queue_family >= families.size()) {
		return false;
	}
	const uint32_t valid_bits = families[graphics.queue_family].timestampValidBits;
	if (valid_bits == 0 || graphics.physical_device_properties.limits.timestampPeriod <= 0.0f) {
		std::printf("NHL27IDLETRACE: timestamps unsupported, GPU timeline disabled\n");
		return false;
	}
	g_gtrace.valid_bits = valid_bits >= 64 ? 64u : valid_bits;
	g_gtrace.valid_mask = valid_bits >= 64 ? ~0ull : ((1ull << valid_bits) - 1ull);
	g_gtrace.period_ns  = graphics.physical_device_properties.limits.timestampPeriod;

	vk::QueryPoolCreateInfo create {};
	create.queryType  = vk::QueryType::eTimestamp;
	create.queryCount = GPU_TRACE_PAIRS * 2u;
	if (graphics.device.createQueryPool(&create, nullptr, &g_gtrace.pool) != vk::Result::eSuccess) {
		std::printf("NHL27IDLETRACE: query pool creation failed\n");
		return false;
	}
	g_gtrace.device = graphics.device;
	{
		vk::QueryPoolCreateInfo cmd_create {};
		cmd_create.queryType  = vk::QueryType::eTimestamp;
		cmd_create.queryCount = GPU_TRACE_CMD_PAIRS * 2u;
		if (graphics.device.createQueryPool(&cmd_create, nullptr, &g_gtrace.cmd_pool) !=
		    vk::Result::eSuccess) {
			std::printf("NHL27IDLETRACE: per-command query pool creation failed\n");
			g_gtrace.cmd_pool = nullptr;
		} else {
			g_gtrace.cmd_results.resize(4);
			g_gtrace.cmd_free.reserve(GPU_TRACE_CMD_PAIRS);
			for (uint32_t i = GPU_TRACE_CMD_PAIRS; i > 0; i--) {
				g_gtrace.cmd_free.push_back(i - 1u);
			}
		}
	}
	Common::EventTrace::g_csv_written_hook = &GpuTraceWriteShaderFile;
	g_gtrace.results.resize(4);
	g_gtrace.free_slots.reserve(GPU_TRACE_PAIRS);
	for (uint32_t i = GPU_TRACE_PAIRS; i > 0; i--) {
		g_gtrace.free_slots.push_back(i - 1u);
	}

	if (graphics.calibrated_timestamps_enabled &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsEXT != nullptr &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT != nullptr) {
#if defined(_WIN32)
		const VkTimeDomainEXT wanted = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
		const VkTimeDomainEXT wanted = VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
#endif
		uint32_t count = 0;
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
		    static_cast<VkPhysicalDevice>(graphics.physical_device), &count, nullptr);
		std::vector<VkTimeDomainEXT> domains(count);
		if (count != 0) {
			VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
			    static_cast<VkPhysicalDevice>(graphics.physical_device), &count, domains.data());
		}
		bool has_device = false, has_cpu = false;
		for (uint32_t i = 0; i < count; i++) {
			has_device |= domains[i] == VK_TIME_DOMAIN_DEVICE_EXT;
			has_cpu |= domains[i] == wanted;
		}
		if (has_device && has_cpu) {
			g_gtrace.can_calibrate = true;
			g_gtrace.time_domain   = static_cast<uint32_t>(wanted);
		}
	}
	g_gtrace.available = true;
	return true;
}

// Establishes the GPU->trace clock mapping for the current capture (once per capture).
void GpuTraceMapCaptureLocked() {
	const int64_t start = Common::EventTrace::g_start_ns;
	if (g_gtrace.mapped_start_ns == start) {
		return;
	}
	g_gtrace.mapped_start_ns = start;
	g_gtrace.have_anchor     = false;
	g_gtrace.end_check_due   = false;
	uint64_t gpu = 0;
	int64_t  cpu = 0;
	if (GpuTraceCalibrate(gpu, cpu)) {
		g_gtrace.calibrated    = true;
		g_gtrace.have_anchor   = true;
		g_gtrace.anchor_gpu    = gpu;
		g_gtrace.anchor_cpu_ns = cpu;
		g_gtrace.end_check_due = true;
		std::printf("NHL27IDLETRACE: gpu_time_domain=calibrated\n");
	} else {
		g_gtrace.calibrated = false;
		std::printf("NHL27IDLETRACE: gpu_time_domain=aligned\n");
		std::printf("NHL27IDLETRACE: warning: calibrated timestamps unavailable; GPU times are "
		            "aligned to the first observed completion (offset error up to the polling latency)\n");
	}
	std::fflush(stdout);
}

int64_t GpuTraceToCpuNs(uint64_t gpu) {
	return g_gtrace.anchor_cpu_ns +
	       static_cast<int64_t>(static_cast<double>(GpuTraceSignedDelta(gpu, g_gtrace.anchor_gpu)) *
	                            g_gtrace.period_ns);
}

// Reads back completed draw / dispatch pairs in submit order. Event encoding:
//   GPU_DRAW / GPU_DISPATCH          at begin time: a = tick, b = shader identity
//   GPU_DRAW_END / GPU_DISPATCH_END  at end time:   a = tick, b = (command index in tick << 32) | size
// size = saturated vertex count * instance count (draw) or x*y*z thread groups (dispatch, 0 if indirect).
void GpuTraceCmdPollLocked() {
	size_t consumed = 0;
	for (; consumed < g_gtrace.cmd_pending.size(); consumed++) {
		const auto& p = g_gtrace.cmd_pending[consumed];
		const auto  r = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
            static_cast<VkDevice>(g_gtrace.device), static_cast<VkQueryPool>(g_gtrace.cmd_pool),
            p.cmd.slot * 2u, 2u, g_gtrace.cmd_results.size() * sizeof(uint64_t),
            g_gtrace.cmd_results.data(), 2u * sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
		const uint64_t* v = g_gtrace.cmd_results.data();
		if ((r != VK_SUCCESS && r != VK_NOT_READY) || v[1] == 0 || v[3] == 0) {
			break;
		}
		g_gtrace.cmd_free.push_back(p.cmd.slot);
		if (!Common::EventTrace::g_capturing.load(std::memory_order_relaxed) ||
		    g_gtrace.mapped_start_ns != Common::EventTrace::g_start_ns) {
			continue;
		}
		const uint64_t begin = v[0] & g_gtrace.valid_mask;
		const uint64_t end   = v[2] & g_gtrace.valid_mask;
		if (!g_gtrace.have_anchor) {
			g_gtrace.have_anchor   = true;
			g_gtrace.anchor_gpu    = end;
			g_gtrace.anchor_cpu_ns = Common::EventTrace::NowNs();
		}
		namespace ET = Common::EventTrace;
		ET::RecordAt(p.cmd.dispatch ? ET::GPU_DISPATCH : ET::GPU_DRAW, GpuTraceToCpuNs(begin), p.tick,
		             p.cmd.identity);
		ET::RecordAt(p.cmd.dispatch ? ET::GPU_DISPATCH_END : ET::GPU_DRAW_END, GpuTraceToCpuNs(end),
		             p.tick, (static_cast<uint64_t>(p.cmd.index) << 32u) | p.cmd.size);
	}
	if (consumed != 0) {
		g_gtrace.cmd_pending.erase(g_gtrace.cmd_pending.begin(),
		                           g_gtrace.cmd_pending.begin() + static_cast<std::ptrdiff_t>(consumed));
	}
}

// Reads back completed command buffers in submit order, emits their events and recycles the slots.
void GpuTracePollLocked() {
	if (g_gtrace.end_check_due && !Common::EventTrace::g_capturing.load(std::memory_order_relaxed)) {
		g_gtrace.end_check_due = false;
		uint64_t gpu = 0;
		int64_t  cpu = 0;
		if (GpuTraceCalibrate(gpu, cpu)) {
			const int64_t predicted = GpuTraceToCpuNs(gpu);
			std::printf("NHL27IDLETRACE: gpu clock drift over capture = %.1f us\n",
			            static_cast<double>(predicted - cpu) / 1000.0);
			std::fflush(stdout);
		}
	}
	size_t consumed = 0;
	for (; consumed < g_gtrace.pending.size(); consumed++) {
		const auto& p = g_gtrace.pending[consumed];
		// 64-bit results, each followed by its availability value.
		const auto r = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
		    static_cast<VkDevice>(g_gtrace.device), static_cast<VkQueryPool>(g_gtrace.pool),
		    p.slot * 2u, 2u, g_gtrace.results.size() * sizeof(uint64_t), g_gtrace.results.data(),
		    2u * sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
		const uint64_t* v = g_gtrace.results.data();
		if ((r != VK_SUCCESS && r != VK_NOT_READY) || v[1] == 0 || v[3] == 0) {
			break;
		}
		g_gtrace.free_slots.push_back(p.slot);
		if (!Common::EventTrace::g_capturing.load(std::memory_order_relaxed) ||
		    g_gtrace.mapped_start_ns != Common::EventTrace::g_start_ns) {
			continue; // Capture over: just recycle.
		}
		const uint64_t begin = v[0] & g_gtrace.valid_mask;
		const uint64_t end   = v[2] & g_gtrace.valid_mask;
		if (!g_gtrace.have_anchor) {
			// Aligned fallback: first completed GPU_END == the moment we observe it.
			g_gtrace.have_anchor   = true;
			g_gtrace.anchor_gpu    = end;
			g_gtrace.anchor_cpu_ns = Common::EventTrace::NowNs();
		}
		Common::EventTrace::RecordAt(Common::EventTrace::GPU_BEGIN, GpuTraceToCpuNs(begin), p.tick, 0);
		Common::EventTrace::RecordAt(Common::EventTrace::GPU_END, GpuTraceToCpuNs(end), p.tick, 0);
	}
	if (consumed != 0) {
		g_gtrace.pending.erase(g_gtrace.pending.begin(),
		                       g_gtrace.pending.begin() + static_cast<std::ptrdiff_t>(consumed));
	}
	if (g_gtrace.cmd_pool != nullptr) {
		GpuTraceCmdPollLocked();
	}
}

// Called right after vkBeginCommandBuffer. Returns slot + 1, or 0 when not tracing.
uint32_t GpuTraceBegin(GraphicContext& graphics, const CommandRecorder& cmd) {
	if (!Common::EventTrace::g_capturing.load(std::memory_order_relaxed)) {
		return 0;
	}
	std::lock_guard lock(g_gtrace.mutex);
	if (!GpuTraceInitLocked(graphics)) {
		return 0;
	}
	GpuTraceMapCaptureLocked();
	if (g_gtrace.free_slots.empty()) {
		GpuTracePollLocked();
		if (g_gtrace.free_slots.empty()) {
			return 0;
		}
	}
	const uint32_t slot = g_gtrace.free_slots.back();
	g_gtrace.free_slots.pop_back();
	// The slot was read back (or never used), so no GPU work references it: host reset is safe.
	// Resets happen here, outside any render pass; the timestamps themselves are legal anywhere.
	VULKAN_HPP_DEFAULT_DISPATCHER.vkResetQueryPool(static_cast<VkDevice>(g_gtrace.device),
	                                               static_cast<VkQueryPool>(g_gtrace.pool), slot * 2u, 2u);
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, g_gtrace.pool, slot * 2u);
	return slot + 1u;
}

void GpuTraceEnd(const CommandRecorder& cmd, uint32_t slot_plus_one) {
	if (slot_plus_one != 0) {
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, g_gtrace.pool,
		                   (slot_plus_one - 1u) * 2u + 1u);
	}
}

// After vkQueueSubmit succeeded: remember which tick owns the slot.
void GpuTraceSubmitted(uint32_t slot_plus_one, uint64_t tick) {
	if (!g_gtrace.available) {
		return;
	}
	std::lock_guard lock(g_gtrace.mutex);
	for (const auto& cmd: g_gtrace.cmd_open) {
		g_gtrace.cmd_pending.push_back({tick, cmd});
	}
	g_gtrace.cmd_open.clear();
	g_gtrace.cmd_open_count = 0;
	if (slot_plus_one != 0) {
		g_gtrace.pending.push_back({tick, slot_plus_one - 1u});
	}
}

// Cheap when idle: one relaxed-load check plus a mutex only if something is pending.
void GpuTracePoll() {
	if (!g_gtrace.available) {
		return;
	}
	std::lock_guard lock(g_gtrace.mutex);
	if (!g_gtrace.pending.empty() || !g_gtrace.cmd_pending.empty() || g_gtrace.end_check_due) {
		GpuTracePollLocked();
	}
}

uint32_t GpuTraceCmdBegin(GraphicContext& graphics, const CommandRecorder& cmd, const char* kind,
                          uint64_t hash_a, uint64_t hash_b, uint64_t info, uint64_t aux) {
	// KYTY_IDLE_TRACE_NO_CMDS=1: command buffer timestamps only, so the GPU busy time per flip is
	// measured without the cost of a timestamp pair around every draw and dispatch.
	static const bool no_cmds = std::getenv("KYTY_IDLE_TRACE_NO_CMDS") != nullptr;
	if (no_cmds || !Common::EventTrace::g_capturing.load(std::memory_order_relaxed)) {
		return 0;
	}
	std::lock_guard lock(g_gtrace.mutex);
	if (!GpuTraceInitLocked(graphics) || g_gtrace.cmd_pool == nullptr) {
		return 0;
	}
	GpuTraceMapCaptureLocked();
	const bool dispatch = std::strncmp(kind, "dispatch", 8) == 0;
	uint64_t   size     = 0;
	if (!dispatch) {
		size = (info >> 32u) * (info & 0xffffffffu);
	} else if (std::strcmp(kind, "dispatch") == 0) {
		size = ((info >> 40u) & 0xffffffu) * ((info >> 20u) & 0xfffffu) * (info & 0xfffffu);
	}
	size = std::min<uint64_t>(size, 0xffffffffu);
	g_gtrace.shader_rows[GpuTraceShaderKey {kind, hash_a, dispatch ? 0 : hash_b, dispatch ? 0 : aux}]++;
	const uint32_t index = g_gtrace.cmd_open_count++;
	if (g_gtrace.cmd_free.empty()) {
		GpuTracePollLocked();
		if (g_gtrace.cmd_free.empty()) {
			g_gtrace.cmd_skipped++;
			return 0;
		}
	}
	const uint32_t slot = g_gtrace.cmd_free.back();
	g_gtrace.cmd_free.pop_back();
	VULKAN_HPP_DEFAULT_DISPATCHER.vkResetQueryPool(static_cast<VkDevice>(g_gtrace.device),
	                                               static_cast<VkQueryPool>(g_gtrace.cmd_pool),
	                                               slot * 2u, 2u);
	g_gtrace.cmd_open.push_back(
	    {slot, index, dispatch, dispatch ? hash_a : (hash_b != 0 ? hash_b : hash_a), size});
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, g_gtrace.cmd_pool, slot * 2u);
	return slot + 1u;
}

void GpuTraceCmdEnd(const CommandRecorder& cmd, uint32_t slot_plus_one) {
	if (slot_plus_one != 0) {
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, g_gtrace.cmd_pool,
		                   (slot_plus_one - 1u) * 2u + 1u);
	}
}

} // namespace

uint32_t HostTraceBegin(const CommandRecorder& cmd, const char* kind, uint64_t identity) {
	static const bool no_cmds = std::getenv("KYTY_IDLE_TRACE_NO_CMDS") != nullptr;
	if (no_cmds || !Common::EventTrace::g_capturing.load(std::memory_order_relaxed)) {
		return 0;
	}
	std::lock_guard lock(g_gtrace.mutex);
	// Only once a guest command initialized the trace (it needs the graphics context).
	if (!g_gtrace.initialized || !g_gtrace.available || g_gtrace.cmd_pool == nullptr) {
		return 0;
	}
	GpuTraceMapCaptureLocked();
	g_gtrace.shader_rows[GpuTraceShaderKey {kind, identity, 0, 0}]++;
	const uint32_t index = g_gtrace.cmd_open_count++;
	if (g_gtrace.cmd_free.empty()) {
		GpuTracePollLocked();
		if (g_gtrace.cmd_free.empty()) {
			g_gtrace.cmd_skipped++;
			return 0;
		}
	}
	const uint32_t slot = g_gtrace.cmd_free.back();
	g_gtrace.cmd_free.pop_back();
	VULKAN_HPP_DEFAULT_DISPATCHER.vkResetQueryPool(static_cast<VkDevice>(g_gtrace.device),
	                                               static_cast<VkQueryPool>(g_gtrace.cmd_pool),
	                                               slot * 2u, 2u);
	g_gtrace.cmd_open.push_back({slot, index, false, identity, 0});
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, g_gtrace.cmd_pool, slot * 2u);
	return slot + 1u;
}

void HostTraceEnd(const CommandRecorder& cmd, uint32_t slot_plus_one) {
	GpuTraceCmdEnd(cmd, slot_plus_one);
}

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	return m_buffers[Acquire(m_master.CurrentTick())];
}

std::pair<vk::CommandBuffer, size_t> CommandScheduler::CommandPool::CommitDeferred() {
	const auto index = Acquire(std::numeric_limits<uint64_t>::max());
	return {m_buffers[index], index};
}

void CommandScheduler::CommandPool::Retire(size_t index, uint64_t tick) {
	EXIT_IF(index >= m_ticks.size() || m_ticks[index] != std::numeric_limits<uint64_t>::max());
	m_ticks[index] = tick;
}

size_t CommandScheduler::CommandPool::Acquire(uint64_t busy_until) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick, busy_until](size_t begin,
	                                                  size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = busy_until;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = busy_until;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return *found;
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics,
                                   bool deferred_recording)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_deferred(deferred_recording),
      m_stream(&m_chunk_queue), m_command(*this, deferred_recording ? &m_stream : nullptr),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	// The CPU crumbs count in the render scheduler's ticks; the presenter has a scheduler of its
	// own, made later, whose ticks made every recorded command look unfinished in hang reports.
	if (&context.GetCommandScheduler() == this) {
		g_crumb_master = &m_master;
	}
	if (m_deferred) {
		m_host_copies = std::make_unique<HostCopyQueue>(
		    [] { Config::ConfigureGpuStageThread(Config::GpuStageThread::HostCopy); });
		m_recording_thread = std::jthread([this] { RecordingThread(); });
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	if (m_recording_thread.joinable()) {
		// Every recorded submission has completed.
		m_chunk_queue.Stop();
		m_recording_thread.join();
	}
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(const HW::Context& registers, const HW::UserConfig& user_config,
                             const HW::Shader& shaders) {
	// Every GPU operation binds its register state. A shutdown submits the open command buffer
	// before it closes the scheduler, so an open command buffer means an open scheduler.
	if (!m_command.IsInvalid()) {
		m_command.Bind(registers, user_config, shaders);
		return;
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);
	BeginNext();
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	g_scheduler_flushes.fetch_add(1, std::memory_order_relaxed);
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Every draw and dispatch comes here: only query the timeline semaphore (a driver call) when
	// the oldest operation is not already known to be free.
	bool refreshed = false;
	// Without the lock: nothing pending, or the oldest operation is not done and the timeline
	// was queried a few calls ago. Pending operations only free resources and complete host
	// work, so finding them some microseconds later costs nothing, while a query per draw is a
	// driver call per draw.
	if (const auto front = m_pending_front_tick.load(std::memory_order_acquire);
	    front == UINT64_MAX) {
		return;
	} else if (!m_master.IsFree(front)) {
		constexpr uint32_t RefreshInterval = 8;
		// Draws come from the execution thread: no locked instruction needed (a racing caller at
		// worst queries once more or once less).
		if (const auto skips = m_pending_skips.load(std::memory_order_relaxed) + 1;
		    skips < RefreshInterval) {
			m_pending_skips.store(skips, std::memory_order_relaxed);
			return;
		}
		m_pending_skips.store(0, std::memory_order_relaxed);
		m_master.Refresh();
		refreshed = true;
		if (!m_master.IsFree(front)) {
			return;
		}
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			if (!m_master.IsFree(m_pending_operations.front().tick)) {
				if (refreshed) {
					return;
				}
				m_master.Refresh();
				refreshed = true;
				if (!m_master.IsFree(m_pending_operations.front().tick)) {
					return;
				}
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
			m_pending_front_tick.store(m_pending_operations.empty()
			                               ? UINT64_MAX
			                               : m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		WaitPriorityOperations(operation.tick);
		Common::EventTrace::Record(Common::EventTrace::NATIVE_DONE, operation.tick);
		GpuTracePoll();
		FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_DEFERRED);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	QueueOperation(TrackGuestOperation(std::move(operation), address, size));
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                              uint64_t address, uint64_t size) {
	QueuePriorityOperation(TrackGuestOperation(std::move(operation), address, size));
}

void CommandScheduler::DeferHostOperation(Common::UniqueFunction<void>&& operation,
                                          bool                           priority) {
	if (priority) {
		QueuePriorityOperation(std::move(operation));
	} else {
		QueueOperation(std::move(operation));
	}
}

Common::UniqueFunction<void>
CommandScheduler::TrackGuestOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	EXIT_IF(!operation);
	uint64_t id = 0;
	{
		std::lock_guard lock(m_guest_writes_mutex);
		id = ++m_next_guest_write;
		m_guest_writes.push_back(
		    {id, address, size > UINT64_MAX - address ? UINT64_MAX : address + size});
	}
	return [this, id, operation = std::move(operation)]() mutable {
		operation();
		std::lock_guard lock(m_guest_writes_mutex);
		// Operations mostly complete in the order they were queued.
		const auto it = std::find_if(m_guest_writes.begin(), m_guest_writes.end(),
		                             [id](const GuestWrite& write) { return write.id == id; });
		EXIT_IF(it == m_guest_writes.end());
		m_guest_writes.erase(it);
	};
}

bool CommandScheduler::HasPendingGuestOperations(uint64_t address, uint64_t size) {
	const auto      end = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	std::lock_guard lock(m_guest_writes_mutex);
	return std::any_of(m_guest_writes.begin(), m_guest_writes.end(), [&](const GuestWrite& write) {
		return write.begin < end && address < write.end;
	});
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		if (m_pending_operations.size() == 1) {
			m_pending_front_tick.store(m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::QueuePriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		Common::EventTrace::Record(Common::EventTrace::NATIVE_DONE, operation.tick);
		GpuTracePoll();
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}


void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

void CommandScheduler::WaitSubmitted(uint64_t tick) {
	while (m_submitted_tick.load(std::memory_order_acquire) < tick) {
		std::this_thread::yield();
	}
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_BEGIN_CMD);
	if (m_deferred) {
		m_command.m_open = true;
	} else {
		m_command.m_buffer = m_command_pool.Commit();
	}
	if (BufferCache::PerfTimersEnabled()) {
		const auto start = std::chrono::steady_clock::now();
		m_command.Begin();
		BufferCache::PerfAddTime(BufferCache::PerfTimer::CommandBegin,
		                         static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                             std::chrono::steady_clock::now() - start).count()));
	} else {
		m_command.Begin();
	}
	m_dynamic_state.Invalidate();
	m_trace_slot = GpuTraceBegin(m_graphics, m_command.Handle());
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	if (FrameBreakdown::Enabled()) {
		FrameBreakdown::SpecNoteSubmit();
	}
	EXIT_IF(m_command.IsInvalid());
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_SUBMIT);
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	const bool timed       = BufferCache::PerfTimersEnabled();
	const auto submit_start = timed ? std::chrono::steady_clock::now()
	                                : std::chrono::steady_clock::time_point {};
	if (m_trace_slot != 0) {
		m_command.EndRendering(); // Idempotent; keeps the end timestamp after the last pass.
		GpuTraceEnd(m_command.Handle(), m_trace_slot);
	}
	if (Common::Experiments::DirectGpuReadsMemory()) {
		// Experiment DirectGpuReads: this submission's writes become visible to the CPU once its
		// tick is reached, so the buffer cache can read finished results from mapped game buffers.
		m_command.EndRendering();
		vk::MemoryBarrier host {};
		host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		host.dstAccessMask = vk::AccessFlagBits::eHostRead;
		m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                                   vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0, nullptr, 0,
		                                   nullptr);
	}
	m_command.End();

	// Safe point for runtime experiment flags: the command buffer is complete and
	// nothing is submitted yet, so the snapshot can be replaced without racing a
	// reader mid-submission. Throttled internally.
	Common::RuntimeFlags::Refresh();
	Common::EventTrace::Poll();

	// Only this scheduler's owner allocates its ticks, in submission order.
	const auto     tick = m_master.NextTick();
	RecordedSubmit debug {};
	debug.debug_op        = m_command.m_debug_op;
	debug.debug_submit_id = m_command.m_debug_submit_id;
	debug.debug_arg0      = m_command.m_debug_arg0;
	debug.debug_arg1      = m_command.m_debug_arg1;
	debug.debug_arg2      = m_command.m_debug_arg2;
	debug.debug_arg3      = m_command.m_debug_arg3;
	debug.debug_arg4      = m_command.m_debug_arg4;
	if (m_deferred) {
		// Waiting for the tick waits for the recording thread too: the timeline semaphore only
		// reaches it after the thread submitted the commands recorded until now.
		auto* recorded = m_stream.RecordSpecial<RecordedSubmit>(CommandStream::RecordKind::Submit);
		*recorded      = debug;
		recorded->info = submit;
		recorded->tick = tick;
		recorded->host_copies = m_host_copies != nullptr ? m_host_copies->Flush() : 0;
		m_stream.Publish();
		m_command.m_open = false;
	} else {
		QueueSubmit(m_command.m_buffer, submit, tick, debug);
		m_command.m_buffer = nullptr;
	}
	if (timed) {
		BufferCache::PerfAddTime(BufferCache::PerfTimer::Submit,
		                         static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                             std::chrono::steady_clock::now() - submit_start).count()));
	}
	Common::EventTrace::Record(Common::EventTrace::NATIVE_SUBMIT, tick);
	GpuTraceSubmitted(m_trace_slot, tick);
	m_trace_slot = 0;
	GpuTracePoll();
	GpuTimingPoll();
	return tick;
}

void CommandScheduler::QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick,
                                   const RecordedSubmit& debug) {
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);
	submit.AddSignal(m_master.Handle(), tick);

	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = buffer != nullptr ? 1 : 0;
	submit_info.pCommandBuffers      = buffer != nullptr ? &buffer : nullptr;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		FrameBreakdown::GpuBusyScope queue_scope(FrameBreakdown::GB_QUEUE_SUBMIT);
		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	m_submitted_tick.store(tick, std::memory_order_release);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceFault(graphics.device);
		// NHL26 debugging: GDS is host-visible; print its nonzero dwords as of the hang.
		const auto gds = m_context.GetBufferCache().GetGdsBuffer()->Mapped();
		if (!gds.empty()) {
			const auto* words = reinterpret_cast<const uint32_t*>(gds.data());
			uint32_t    shown = 0;
			for (size_t i = 0; i < gds.size() / 4 && shown < 64; i++) {
				if (words[i] != 0) {
					std::printf("NHL26GDS: [0x%04zx] = 0x%08x (%u)\n", i * 4, words[i], words[i]);
					shown++;
				}
			}
			std::printf("NHL26GDS: %u nonzero dwords shown\n", shown);
			std::fflush(stdout);
		}
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, debug.debug_op, debug.debug_submit_id,
		                  debug.debug_arg0, debug.debug_arg1, debug.debug_arg2, debug.debug_arg3,
		                  debug.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::RecordingThread() {
	KYTY_PROFILER_THREAD("Thread_GpuRecord");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Recording);
	vk::CommandBuffer buffer       = nullptr;
	size_t            buffer_index = 0;
	const auto        acquire      = [this, &buffer, &buffer_index] {
		if (buffer == nullptr) {
			std::tie(buffer, buffer_index) = m_command_pool.CommitDeferred();
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(buffer.begin(&begin_info) != vk::Result::eSuccess);
		}
		return buffer;
	};
	for (;;) {
		auto chunk = m_chunk_queue.Pop();
		if (chunk == nullptr) {
			break;
		}
		CommandStream::Replay(*chunk, acquire, [&](const CommandStream::Entry& entry) {
			EXIT_IF(entry.kind != CommandStream::RecordKind::Submit);
			SubmitRecorded(buffer, buffer_index,
			               *static_cast<const RecordedSubmit*>(entry.Payload()));
		});
		m_chunk_queue.Recycle(std::move(chunk));
	}
	// A scheduler only stops after its last submission.
	EXIT_IF(buffer != nullptr);
}

void CommandScheduler::SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
                                      const RecordedSubmit& recorded) {
	if (buffer != nullptr) {
		EXIT_NOT_IMPLEMENTED(buffer.end() != vk::Result::eSuccess);
	}
	// A submission without commands still signals its tick.
	// The command buffer reads the stream data the execution thread's copies write.
	if (m_host_copies != nullptr) {
		m_host_copies->WaitFor(recorded.host_copies);
	}
	auto submit = recorded.info;
	QueueSubmit(buffer, submit, recorded.tick, recorded);
	if (buffer != nullptr) {
		m_command_pool.Retire(buffer_index, recorded.tick);
		buffer = nullptr;
	}
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
