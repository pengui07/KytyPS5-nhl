#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/experiments.h"
#include "common/hostException.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/runtimeFlags.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/stageStats.h"
#include "graphics/host_gpu/pageWriteSeq.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstdio>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this),
      m_command_scheduler(*this, graphics,
                          (Config::GpuPipelineStages() & Config::GPU_PIPELINE_RECORDING) != 0),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Experiment FaultPageReads: resource-table reads of a page the GPU thread took a read fault on
	// (the GPU wrote other bytes of it) read clean bytes through the backing instead of faulting
	// again; GPU-written bytes still fault and download.
	ShaderRecompiler::IR::SetFaultPageReader([](uint64_t address, void* data, uint64_t size) {
		return Common::Experiments::Enabled(Common::Experiments::FaultPageReads) &&
		       LibKernel::Memory::IsReadFaultPage(address, size) &&
		       LibKernel::Memory::TryReadCleanBufferBacking(address, data, size);
	});
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	// A page another thread's fault released but has not protected yet (PageManager::ReleaseScope).
	if (m_page_manager.ApplyPending(fault_vaddr)) {
		return true;
	}
	if (!IsMapped(fault_vaddr, fault_size)) {
		// Cache tracking may still protect a page whose range left the GPU-mapped set; that
		// fault is ours to resolve, not a guest access violation.
		if (!GuestRange {fault_vaddr, fault_size}.Valid() || !m_page_manager.IsWatched(fault_vaddr)) {
			return false;
		}
	}
	ProtectStats::CountFault(access == PageFaultAccess::Write);
	if (access == PageFaultAccess::Write && CpReadDebugEnabled()) {
		ProtectStats::NoteWriteFaultPage(fault_vaddr);
	}
	Common::Timeline::Mark(access == PageFaultAccess::Write ? "fault-write" : "fault-read", fault_vaddr);
	// KYTY_PERF_STATS: fault time includes any synchronous readback it triggers (overlaps readback).
	FrameBreakdown::Timer fault_timer(FrameBreakdown::g_counters.fault_ns);
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_FAULT);
	FrameBreakdown::Add(FrameBreakdown::g_counters.fault_count);
	// NHL27FAULTCOST accounting (KYTY_PERF_STATS only).
	struct FaultCostScope {
		bool     active = FrameBreakdown::Enabled();
		bool     gpu    = false;
		uint64_t start  = 0;
		~FaultCostScope() {
			if (!active) {
				return;
			}
			auto&          fc = FrameBreakdown::g_fault_cost;
			const uint64_t ns = FrameBreakdown::NowNs() - start;
			fc.total_ns.fetch_add(ns, std::memory_order_relaxed);
			(gpu ? fc.total_gpu_ns : fc.total_guest_ns).fetch_add(ns, std::memory_order_relaxed);
			FrameBreakdown::g_in_fault = false;
		}
	} cost_scope;
	if (cost_scope.active) {
		auto&          fc    = FrameBreakdown::g_fault_cost;
		const uint64_t entry = Common::HostException::LastEntryTimeNs();
		cost_scope.start     = FrameBreakdown::NowNs();
		if (entry != 0 && cost_scope.start > entry) {
			fc.veh_ns.fetch_add(cost_scope.start - entry, std::memory_order_relaxed);
		}
		cost_scope.gpu = GuestGpu::IsGpuThread();
		fc.n[(cost_scope.gpu ? 0 : 2) + (access == PageFaultAccess::Write ? 1 : 0)].fetch_add(
		    1, std::memory_order_relaxed);
		FrameBreakdown::NoteFaultPage(fault_vaddr);
		FrameBreakdown::g_in_fault = true;
	}
	if (access == PageFaultAccess::Write) {
		// A guest thread's releases are protected when this ends, not under the tracking locks
		// (in-boot A/B: execution 3.76 vs 3.95 us per draw, the guest threads' faults contend
		// with its synchronizations). The GPU threads' faults can download into released pages.
		const PageManager::ReleaseScope release_scope(!GuestGpu::IsGpuThread() &&
		                                              !GuestGpu::IsCommandProcessorThread());
		if (FrameBreakdown::Enabled()) {
			FrameBreakdown::SpecNoteFault(fault_vaddr);
		}
		// KYTY_FAULT_COALESCE: release the clean neighbors of the faulting page's aligned window
		// (KYTY_FAULT_COALESCE_PAGES tracker pages, default 16) in one protection change. Pages that
		// are GPU-modified or already CPU-dirty are skipped; marking an unwritten page CPU-dirty
		// only costs an upload. The window is released before the faulting page, with it when it
		// is clean: one protection change instead of three (the page, then the neighbors on each
		// side of it; in-play A/B: protect calls -41%, those of the faulting threads -56%).
		const auto release_window = [&] {
			if (!Common::RuntimeFlags::Current().fault_coalesce) {
				return;
			}
			static const uint64_t wide_window = [] {
				const char* value = std::getenv("KYTY_FAULT_COALESCE_PAGES");
				const auto  pages = value != nullptr ? std::strtoull(value, nullptr, 10) : 16ull;
				return std::bit_ceil(std::clamp<uint64_t>(pages, 1, 256)) * TRACKER_PAGE_SIZE;
			}();
			// Experiment WideFaultWindows off: 4-page windows. NHL 27 faults ~200 times per frame and
			// the 16-page windows upload ~11 MB per frame, mostly unwritten neighbors.
			const uint64_t window = Common::Experiments::Enabled(Common::Experiments::WideFaultWindows)
			                            ? wide_window
			                            : std::min<uint64_t>(wide_window, 4 * TRACKER_PAGE_SIZE);
			const uint64_t begin = fault_vaddr & ~(window - 1);
			if (window > TRACKER_PAGE_SIZE && IsMapped(begin, window)) {
				const uint64_t marked = m_buffer_cache.MarkCpuDirtyAhead(begin, window);
				if (marked != 0) {
					FrameBreakdown::Add(FrameBreakdown::g_counters.port_fault_windows);
					FrameBreakdown::Add(FrameBreakdown::g_counters.port_fault_pages, marked);
				}
			}
		};
		release_window();
		{
			FrameBreakdown::FaultTimer t(FrameBreakdown::g_fault_cost.buf_inv_ns);
			m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		}
		{
			FrameBreakdown::FaultTimer t(FrameBreakdown::g_fault_cost.tex_inv_ns);
			m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		}
		// KYTY_FAULT_AHEAD: streaming writers (ring buffers) fault on every page they touch
		// each frame, and each page is re-protected after its upload. Unprotect the next
		// pages of the buffer tracker up front so one fault covers the run.
		if (Common::RuntimeFlags::Current().fault_ahead) {
			constexpr uint64_t kAheadBytes = 15 * TRACKER_PAGE_SIZE;
			const uint64_t next = (fault_vaddr & ~(TRACKER_PAGE_SIZE - 1)) + TRACKER_PAGE_SIZE;
			if (IsMapped(next, kAheadBytes)) {
				m_buffer_cache.MarkCpuDirtyAhead(next, kAheadBytes);
			}
		}
	} else {
		FrameBreakdown::FaultTimer t(FrameBreakdown::g_fault_cost.rd_readmem_ns);
		if (GuestGpu::IsGpuThread()) {
			LibKernel::Memory::NoteReadFaultPage(fault_vaddr);
		}
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size, false, BufferCache::ReadCaller::Fault);
	}
	return true;
}

void RenderContext::PrepareGpuWrite(const void* destination, uint64_t size) noexcept {
	const auto vaddr     = reinterpret_cast<uint64_t>(destination);
	const auto page_size = m_page_manager.GetPageSize();
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// A watched page is protected: the write would fault into HandleFault().
	for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
		if (m_page_manager.IsWatched(page)) {
			(void)HandleFault(PageFaultAccess::Write, std::max(page, vaddr));
		}
	}
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

uint64_t RenderContext::UnmapSequence() const noexcept {
	std::shared_lock lock(m_unmap_log_mutex);
	return m_unmap_sequence;
}

bool RenderContext::UnmappedSince(uint64_t sequence, uint64_t vaddr, uint64_t size) const noexcept {
	std::shared_lock lock(m_unmap_log_mutex);
	const auto       first = m_unmap_sequence > m_unmap_log.size() ? m_unmap_sequence - m_unmap_log.size() : 0;
	for (auto s = std::max(sequence, first); s < m_unmap_sequence; s++) {
		const auto& record = m_unmap_log[s % m_unmap_log.size()];
		if (record.begin < vaddr + size && vaddr < record.end) {
			return true;
		}
	}
	return false;
}

bool RenderContext::IsReadProtected(uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	const auto page_size = m_page_manager.GetPageSize();
	for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
		if (m_page_manager.IsReadWatched(page)) {
			return true;
		}
	}
	return false;
}

uint64_t RenderContext::MappedRangesVersion() const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges_version;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	Common::Timeline::Mark("map", size);
	PageWriteSeq::ForceChecks();
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_bda_new_ranges.Add(vaddr, size);
	++m_mapping_generation;
	m_mapped_ranges_version++;
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	Common::Timeline::Mark("unmap", size);
	PageWriteSeq::ForceChecks();
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Finishing the GPU protects GPU writes to the range. Queued GPU work only reads cached
		// copies, host-only completions (cache bookkeeping, flips, interrupts) never touch the
		// range, and completions writing guest memory (downloads) skip what was unmapped since
		// they were queued (WriteUnlessUnmapped), so without GPU-owned pages or images in the range
		// the unmap needs no drain (AbPending: it drains for those completions too).
		if (m_command_scheduler.Active()) {
			m_command_scheduler.PopPendingOperations();
		}
		const bool active  = m_command_scheduler.Active();
		const bool gpu     = active && m_buffer_cache.IsRegionGpuModified(vaddr, size);
		const bool images  = active && !gpu && m_texture_cache.HasImagesInRegion(vaddr, size);
		const bool pending = active && !gpu && !images && m_command_scheduler.HasPendingGuestOperations(vaddr, size);
		const bool drain   = gpu || images ||
		                   (pending && Common::Experiments::Enabled(Common::Experiments::AbPending));
		const auto start   = CpReadDebugEnabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		if (drain) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		if (CpReadDebugEnabled()) {
			// KYTY_DBG_CP_READS: unmaps by why they drained the GPU, NHL27UNMAP every 2 s (execution thread).
			static uint64_t counts[4] {}, ns[4] {}, bytes = 0;
			static auto     last  = std::chrono::steady_clock::now();
			const auto      now   = std::chrono::steady_clock::now();
			const size_t    why   = gpu ? 1 : images ? 2 : pending ? 3 : 0;
			counts[why]++;
			ns[why] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
			bytes += size;
			if (now - last >= std::chrono::seconds(2)) {
				last = now;
				std::printf("NHL27UNMAP: none=%llu gpu-modified=%llu (%.1f ms) images=%llu (%.1f ms) pending=%llu (%.1f ms) MiB=%.1f\n",
				            static_cast<unsigned long long>(counts[0]), static_cast<unsigned long long>(counts[1]),
				            static_cast<double>(ns[1]) / 1e6, static_cast<unsigned long long>(counts[2]),
				            static_cast<double>(ns[2]) / 1e6, static_cast<unsigned long long>(counts[3]),
				            static_cast<double>(ns[3]) / 1e6, static_cast<double>(bytes) / (1024.0 * 1024.0));
				std::fflush(stdout);
				std::fill(std::begin(counts), std::end(counts), 0);
				std::fill(std::begin(ns), std::end(ns), 0);
				bytes = 0;
			}
		}
		BufferCache::WriteTagScope tag(BufferCache::WriteTag::Unmap);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		{
			std::lock_guard log_lock(m_unmap_log_mutex);
			m_unmap_log[m_unmap_sequence % m_unmap_log.size()] = {m_unmap_sequence, vaddr, vaddr + size};
			m_unmap_sequence++;
		}
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_bda_new_ranges.Subtract(vaddr, size);
		++m_mapping_generation;
		m_mapped_ranges_version++;
		PageWriteSeq::ForceChecks();
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	// The guest unmaps memory once the GPU work that used it executed: it waits on labels the
	// execution thread writes after the operations before them. The operations queued since do not
	// use the range, so the unmap runs before them instead of making the guest wait for them;
	// streaming games unmap many times per frame, holding their own locks.
	m_gpu->SendUrgentCommandSync(unmap);
}

void RenderContext::CacheDmaBases(const ShaderStageRuntime& runtime) {
	const auto&      program   = *runtime.program;
	const auto&      user_data = runtime.resources->user_data;
	std::shared_lock lock(m_mapped_ranges_mutex);
	for (const auto reg: program.info.dma_base_registers) {
		const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
		if (reg < program.user_data_base || index + 1u >= user_data.size()) {
			continue;
		}
		const auto base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
		// The registers of an access that never runs may hold stale data. Mapped memory never
		// reaches the top of the address space, so a window that would overflow is skipped.
		if (base > UINT64_MAX - BufferCache::CACHING_PAGESIZE) {
			continue;
		}
		// DMA reaches only memory with a cached buffer; any other access records a fault and
		// reads zero, and the buffer arrives after the shader has run. That loses data the guest
		// writes for a single dispatch, such as the glyph bitmaps GTA V copies into its font atlas.
		m_mapped_ranges.ForEachInRange(base, BufferCache::CACHING_PAGESIZE,
		                               [this](uint64_t start, uint64_t end) {
			                               (void)m_buffer_cache.FindBuffer(start, end - start);
		                               });
	}
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	m_buffer_cache.PrepareFaultBuffer();
	static const bool cache_enabled = [] {
		const char* value = std::getenv("KYTY_BDA_DIRTY_CACHE");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	static const bool stats_enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && value[0] == '1';
	}();
	// KYTY_BDA_DIRTY_REGIONS (experiment bit, switchable while running: the dirty-region queue
	// fills either way, and a full scan covers whatever it holds).
	const bool dirty_regions = Common::Experiments::Enabled(Common::Experiments::BdaDirtyRegions);
	FrameBreakdown::Add(FrameBreakdown::g_counters.bda_calls);
	FrameBreakdown::Timer total_timer(FrameBreakdown::g_counters.bda_total_ns);
	std::shared_lock<decltype(m_mapped_ranges_mutex)> lock;
	{
		FrameBreakdown::Timer lock_timer(FrameBreakdown::g_counters.bda_lock_ns);
		lock = std::shared_lock(m_mapped_ranges_mutex);
	}
	auto& batch = FrameBreakdown::g_bda_batch;
	if (stats_enabled) {
		batch.active = true;
		++batch.b_calls;
	}
	// KYTY_BDA_BATCH: only the first call of a batch synchronizes. A mapping change (unmap/map from the
	// GPU command lane or a guest thread) always forces a sync. Writes that land after the batch's sync
	// become visible at the next batch boundary (see NHL27BDABATCH for how often that window matters).
	if (m_bda_batch_synced && Common::RuntimeFlags::g_published.load(std::memory_order_relaxed) >> Common::RuntimeFlags::kBitBdaBatch & 1u &&
	    m_mapping_generation == m_bda_mapping_generation) {
		if (stats_enabled) {
			++batch.flag_skips;
		}
		return;
	}
	const auto cpu = m_buffer_cache.CpuModificationGeneration();
	const auto registrations = m_buffer_cache.RegistrationGeneration();
	// Pages left by the last synchronization that this operation needs (their epoch passed).
	const bool left_due = m_bda_pages_left &&
	                      (!m_operation_epoch_valid ||
	                       RegionManager::EpochBefore(m_bda_first_left, m_operation_write_epoch));
	const bool unchanged = cache_enabled && !left_due && cpu == m_bda_cpu_generation &&
	                       registrations == m_bda_registration_generation &&
	                       m_mapping_generation == m_bda_mapping_generation;
	if (unchanged) {
		++m_bda_stats.reused;
		FrameBreakdown::Add(FrameBreakdown::g_counters.bda_skips);
	} else {
		++m_bda_stats.scans;
		const bool extra_scan = batch.scanned;
		if (stats_enabled) {
			batch.collecting = true;
			batch.scan_ranges.clear();
		}
		if (cache_enabled && dirty_regions) {
			// Drained regions outside the mapped ranges are dropped, so only memory mapped since the
			// last synchronization can hold CPU-dirty pages no drain will bring back (pages left for
			// later epochs are queued again); unmaps leave nothing to upload. The full scan covers
			// just that memory (AbPending: all mapped ranges after any mapping change).
			const bool mapping_changed = m_mapping_generation != m_bda_mapping_generation;
			const bool new_only        = m_operation_epoch_valid &&
			                      !Common::Experiments::Enabled(Common::Experiments::AbPending);
			const bool full = new_only ? !m_bda_new_ranges.Empty() : mapping_changed;
			m_bda_stats.full_scans += full;
			if (full) {
				FrameBreakdown::Add(FrameBreakdown::g_counters.bda_scans);
			}
			uint64_t drained = 0;
			{
				FrameBreakdown::Timer sync_timer(FrameBreakdown::g_counters.bda_sync_ns);
				FrameBreakdown::BufScope buf_scope(2);
				FrameBreakdown::VAdd(FrameBreakdown::g_buf.bda_syncs);
				// Only the pages written before this operation was emitted: pages the game is still
				// filling for later submissions are not uploaded and protected again just to fault
				// once more (in-play A/B: protect calls and write faults -20%).
				if (StageStats::Enabled()) {
					StageStats::g_bda_syncs.fetch_add(1, std::memory_order_relaxed);
				}
				if (m_operation_epoch_valid) {
					uint32_t first_left = 0;
					const RangeSet* full_ranges = !full ? nullptr : new_only ? &m_bda_new_ranges : &m_mapped_ranges;
					m_bda_pages_left            = m_buffer_cache.SynchronizeBuffersInRangesBefore(
					    m_mapped_ranges, full_ranges, m_operation_write_epoch, first_left);
					m_bda_first_left = first_left;
				} else {
					drained          = m_buffer_cache.SynchronizeBuffersInRanges(m_mapped_ranges, full);
					m_bda_pages_left = false;
				}
			}
		} else {
			// Nothing to upload when no page became CPU dirty, no buffer was registered and no
			// range was mapped since the last synchronization started. Otherwise only the tracking
			// regions whose pages became CPU dirty since are visited, unless a new buffer or mapped
			// range may cover CPU-dirty pages that earlier passes skipped.
			const std::array<uint64_t, 3> epochs {RegionManager::CpuDirtyEpoch(), registrations,
			                                      m_mapping_generation};
			if (epochs != m_bda_sync_epochs) {
				const bool all = epochs[1] != m_bda_sync_epochs[1] || epochs[2] != m_bda_sync_epochs[2];
				m_bda_stats.full_scans += all;
				FrameBreakdown::Add(FrameBreakdown::g_counters.bda_scans);
				FrameBreakdown::Timer sync_timer(FrameBreakdown::g_counters.bda_sync_ns);
				FrameBreakdown::BufScope buf_scope(2);
				FrameBreakdown::VAdd(FrameBreakdown::g_buf.bda_syncs);
				m_buffer_cache.SynchronizeDirtyRegionsInRanges(m_mapped_ranges, all);
				m_bda_sync_epochs = epochs;
			} else {
				FrameBreakdown::Add(FrameBreakdown::g_counters.bda_skips);
			}
		}
		// Save the pre-scan stamps: a CPU write concurrent with the scan must force
		// another scan, even if it targets a range we already visited.
		m_bda_cpu_generation = cpu;
		m_bda_registration_generation = registrations;
		if (m_bda_mapping_generation != m_mapping_generation) {
			// Every path above synchronized the newly mapped memory.
			m_bda_new_ranges.Clear();
		}
		m_bda_mapping_generation = m_mapping_generation;
		if (stats_enabled) {
			batch.collecting = false;
			++batch.b_scans;
			batch.scanned = true;
			if (extra_scan && !batch.scan_ranges.empty()) {
				const uint32_t event = batch.b_events++;
				for (const auto& r: batch.scan_ranges) {
					batch.stale_bytes += r.second - r.first;
					if (batch.late.size() < 8192) {
						batch.late.push_back({r.first, r.second, event});
					} else {
						++batch.trunc;
					}
				}
			}
		}
	}
	m_bda_batch_synced = true;
	total_timer.Stop();
	// Lazy BDA faults still need processing even when uploads were unchanged.
	m_fault_process_pending = true;
	if (stats_enabled && ((m_bda_stats.scans + m_bda_stats.reused) & 4095u) == 0) {
		static auto last = std::chrono::steady_clock::now();
		const auto now = std::chrono::steady_clock::now();
		if (now - last >= std::chrono::seconds(5)) {
			std::printf("NHL27BDA: cache=%d regions=%d scans=%llu reused=%llu full_scans=%llu dirty_regions=%llu\n", cache_enabled, dirty_regions,
			            static_cast<unsigned long long>(m_bda_stats.scans),
			            static_cast<unsigned long long>(m_bda_stats.reused),
			            static_cast<unsigned long long>(m_bda_stats.full_scans),
			            static_cast<unsigned long long>(m_bda_stats.dirty_regions));
			last = now;
		}
	}
}

static const char* const kBdaBoundaryNames[] = {"submit", "wait", "event", "acquire", "dma", "sync", "ib", "flip", "other"};

void RenderContext::BdaBatchBoundary(BdaBoundary kind) {
	m_bda_batch_synced = false;
	if (!FrameBreakdown::Enabled()) {
		return;
	}
	auto& b = FrameBreakdown::g_bda_batch;
	++b.kind[static_cast<size_t>(kind)];
	if (kind == BdaBoundary::Flip) {
		++b.flips;
	}
	if (!b.active) {
		b.used.clear();
		b.late.clear();
		b.b_events = 0;
		return;
	}
	// Close the batch: classify stale-window events against the buffers requested during the batch.
	++b.batches;
	b.calls += b.b_calls;
	b.scans += b.b_scans;
	b.extra_scans += b.b_scans > 0 ? b.b_scans - 1 : 0;
	b.max_calls = std::max<uint64_t>(b.max_calls, b.b_calls);
	if (b.b_events != 0) {
		std::sort(b.used.begin(), b.used.end());
		std::vector<std::pair<uint64_t, uint64_t>> merged;
		for (const auto& r: b.used) {
			if (!merged.empty() && r.first <= merged.back().second) {
				merged.back().second = std::max(merged.back().second, r.second);
			} else {
				merged.push_back(r);
			}
		}
		std::vector<char> hit(b.b_events, 0);
		for (const auto& l: b.late) {
			auto it = std::upper_bound(merged.begin(), merged.end(), l.begin,
			                           [](uint64_t v, const std::pair<uint64_t, uint64_t>& m) { return v < m.second; });
			if (it != merged.end() && it->first < l.end) {
				hit[l.event] = 1;
			}
		}
		for (const char h: hit) {
			++b.events;
			(h ? b.ev_used : b.ev_unused) += 1;
		}
	}
	b.active  = false;
	b.scanned = false;
	b.b_calls = b.b_scans = b.b_events = 0;
	b.used.clear();
	b.late.clear();
	const auto now = std::chrono::steady_clock::now();
	if (b.last_print == std::chrono::steady_clock::time_point {}) {
		b.last_print = now;
	}
	if (now - b.last_print >= std::chrono::seconds(5)) {
		const double flips   = static_cast<double>(std::max<uint64_t>(b.flips, 1));
		const double batches = static_cast<double>(std::max<uint64_t>(b.batches, 1));
		std::printf("NHL27BDABATCH flag=%d flips=%llu batches/flip=%.1f calls/batch=%.2f scans/batch=%.2f max_calls=%llu "
		            "avoidable_syncs/flip=%.1f flag_skips/flip=%.1f stale_events/flip=%.2f (a_overlaps_used=%llu "
		            "b_unused=%llu) stale_kb/flip=%.1f trunc=%llu bnd/flip[",
		            static_cast<int>(Common::RuntimeFlags::g_published.load(std::memory_order_relaxed) >> Common::RuntimeFlags::kBitBdaBatch & 1u),
		            static_cast<unsigned long long>(b.flips), static_cast<double>(b.batches) / flips,
		            static_cast<double>(b.calls) / batches, static_cast<double>(b.scans) / batches,
		            static_cast<unsigned long long>(b.max_calls), static_cast<double>(b.extra_scans) / flips,
		            static_cast<double>(b.flag_skips) / flips, static_cast<double>(b.events) / flips,
		            static_cast<unsigned long long>(b.ev_used), static_cast<unsigned long long>(b.ev_unused),
		            static_cast<double>(b.stale_bytes) / 1024.0 / flips, static_cast<unsigned long long>(b.trunc));
		for (size_t i = 0; i < static_cast<size_t>(BdaBoundary::Count); ++i) {
			std::printf("%s%s=%.1f", i == 0 ? "" : " ", kBdaBoundaryNames[i], static_cast<double>(b.kind[i]) / flips);
		}
		std::printf("]\n");
		std::fflush(stdout);
		b.batches = b.calls = b.scans = b.extra_scans = b.events = b.ev_used = b.ev_unused = 0;
		b.stale_bytes = b.trunc = b.flag_skips = b.flips = b.max_calls = 0;
		std::fill(std::begin(b.kind), std::end(b.kind), 0);
		b.last_print = now;
	}
}

void RenderContext::RunGarbageCollector() {
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_GC);
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	if (m_indirect_write_tables.TakePendingWrites()) {
		m_buffer_cache.ProcessWriteBuffer([this, sequence = UnmapSequence()](std::span<const uint64_t> pages) {
			m_indirect_write_tables.NoteWrites(pages, sequence);
		});
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
