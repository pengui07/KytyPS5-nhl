#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/pageWriteSeq.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Whether a page of the tracking regions the range touches may be GPU dirty, without taking
	// their locks: false means the range can be read directly.
	[[nodiscard]] bool MayBeGpuModified(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || size > TRACKER_ADDRESS_SIZE || vaddr > TRACKER_ADDRESS_SIZE - size) {
			return true;
		}
		const auto last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = vaddr / TRACKER_REGION_SIZE; index <= last && index < REGION_COUNT;
		     index++) {
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager != nullptr && manager->HasGpuDirty()) {
				return true;
			}
		}
		return false;
	}
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// CPU writes/invalidations and newly published initially-dirty regions invalidate BDA preparation.
	[[nodiscard]] uint64_t CpuModificationGeneration() const noexcept {
		return m_cpu_generation.load(std::memory_order_acquire);
	}
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::unique_lock<TrackingSpinLock> lock(manager->lock, std::defer_lock);
				{
					FrameBreakdown::FaultTimer lock_timer(FrameBreakdown::g_fault_cost.lock_ns);
					lock.lock();
				}
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				QueueCpuDirty(manager);
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				m_cpu_generation.fetch_add(1, std::memory_order_release);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif
	// Marks the CPU-clean pages of a range CPU-modified without flushing anything:
	// GPU-modified pages are skipped. Used to unprotect pages a sequential CPU writer
	// is about to reach, so one write fault covers several pages. Marking a page
	// CPU-modified is always conservative (it is re-uploaded before GPU use).
	// Returns the number of tracker pages newly marked.
	uint64_t MarkCpuDirtyAhead(uint64_t vaddr, uint64_t size) noexcept {
		CheckNotInUploadCallback();
		uint64_t marked_pages = 0;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			uint64_t run_begin = 0;
			uint64_t run_bytes = 0;
			bool     changed   = false;
			const auto flush_run = [&] {
				if (run_bytes == 0) return;
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + run_begin, run_bytes);
				marked_pages += run_bytes / TRACKER_PAGE_SIZE;
				run_bytes = 0;
				changed   = true;
			};
			for (uint64_t page = offset; page < offset + bytes; page += TRACKER_PAGE_SIZE) {
				const bool eligible = !manager->IsModified<DirtySource::Gpu>(page, TRACKER_PAGE_SIZE) &&
				                      !manager->IsModified<DirtySource::Cpu>(page, TRACKER_PAGE_SIZE);
				if (!eligible) {
					flush_run();
					continue;
				}
				if (run_bytes == 0) run_begin = page;
				run_bytes += TRACKER_PAGE_SIZE;
			}
			flush_run();
			if (changed) {
				QueueCpuDirty(manager);
				m_cpu_generation.fetch_add(1, std::memory_order_release);
			}
		});
		return marked_pages;
	}

	// Single GPU-thread consumer. Read the next link before releasing this node to
	// producers; writes racing with its upload can then enqueue it for the next pass.
	template <typename Func>
	uint64_t DrainCpuDirtyRegions(Func&& func) {
		CheckNotInUploadCallback();
		auto* manager = m_cpu_dirty_head.exchange(nullptr, std::memory_order_acquire);
		uint64_t count = 0;
		while (manager != nullptr) {
			auto* next = manager->cpu_dirty_next;
			manager->cpu_dirty_queued.store(false, std::memory_order_release);
			func(manager->GetCpuAddr(), TRACKER_REGION_SIZE);
			manager = next;
			++count;
		}
		return count;
	}

	// Newly registered buffers must revisit initially dirty pages even when their
	// containing tracking region was already drained before the buffer existed.
	void QueueCpuDirtyRange(uint64_t vaddr, uint64_t size) {
		CheckNotInUploadCallback();
		Iterate<true>(vaddr, size,
		    [this](RegionManager* manager, uint64_t, uint64_t) { QueueCpuDirty(manager); });
	}

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	// The CPU-dirty epoch of the tracking region `index` (address / TRACKER_REGION_SIZE), which
	// advances whenever its pages become CPU dirty; 0 while the region is untracked.
	[[nodiscard]] uint64_t RegionCpuDirtyEpoch(uint64_t index) const {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		return manager != nullptr ? manager->RegionCpuDirtyEpoch() : 0;
	}
	// The sum of the CPU-dirty epochs of the 64 KiB blocks of [vaddr, vaddr + size) in the tracking
	// region `index` (0 while it is untracked); see RegionManager::BlockCpuDirtyEpoch.
	[[nodiscard]] uint64_t BlockCpuDirtyEpochs(uint64_t index, uint64_t vaddr, uint64_t size) const {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return 0;
		}
		const uint64_t region_begin = index * TRACKER_REGION_SIZE;
		const uint64_t begin        = std::max(vaddr, region_begin) - region_begin;
		const uint64_t end          = std::min(vaddr + size, region_begin + TRACKER_REGION_SIZE) - region_begin;
		constexpr uint64_t BlockBytes = RegionManager::DirtyBlockPages * TRACKER_PAGE_SIZE;
		uint64_t           epoch      = 0;
		for (uint64_t block = begin / BlockBytes; block * BlockBytes < end; block++) {
			epoch += manager->BlockCpuDirtyEpoch(static_cast<size_t>(block));
		}
		return epoch;
	}

	// Calls func(address, size) for maximal runs of pages that may hold CPU writes not yet
	// uploaded, after releasing the region locks. Untracked regions count as dirty, as a new
	// region starts dirty.
	template <typename Func>
	void ForEachCpuDirtySpan(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		CheckNotInUploadCallback();
		const auto                                  end = vaddr + size;
		std::vector<std::pair<uint64_t, uint64_t>> spans; // [begin, end)
		const auto add = [&](uint64_t begin, uint64_t finish) {
			begin  = std::max(begin, vaddr);
			finish = std::min(finish, end);
			if (begin >= finish) {
				return;
			}
			if (!spans.empty() && spans.back().second == begin) {
				spans.back().second = finish;
			} else {
				spans.emplace_back(begin, finish);
			}
		};
		for (uint64_t index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < end;
		     index++) {
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			const auto start   = std::max(index * TRACKER_REGION_SIZE, vaddr);
			const auto finish  = std::min((index + 1) * TRACKER_REGION_SIZE, end);
			if (manager == nullptr) {
				add(start, finish);
				continue;
			}
			if (!manager->HasCpuDirty()) {
				continue;
			}
			std::scoped_lock lock(manager->lock);
			manager->ForEachModifiedRange<DirtySource::Cpu, false>(
			    start, finish - start,
			    [&](uint64_t address, uint64_t bytes) { add(address, address + bytes); });
		}
		for (const auto& [begin, finish]: spans) {
			func(begin, finish - begin);
		}
	}

	// ForEachCpuDirtySpan for the pages dirtied before write epoch `limit` (see RegionManager). A
	// region whose later pages were left out is queued again for the next drain; returns whether
	// any were, and lowers `first_left` to the earliest of their epochs.
	template <typename Func>
	bool ForEachCpuDirtySpanBefore(uint64_t vaddr, uint64_t size, uint32_t limit, uint32_t& first_left,
	                               Func&& func) {
		ValidateRange(vaddr, size);
		CheckNotInUploadCallback();
		const auto end = vaddr + size;
		// [begin, end). The outermost call of a thread reuses one vector (an allocation per call was
		// 0.4% of the execution thread); func may call back in, which then takes its own.
		thread_local std::vector<std::pair<uint64_t, uint64_t>> t_spans;
		thread_local uint32_t                                   t_depth = 0;
		std::vector<std::pair<uint64_t, uint64_t>>              local;
		const bool reuse = t_depth == 0;
		auto&      spans = reuse ? t_spans : local;
		spans.clear();
		struct Depth {
			uint32_t& depth;
			explicit Depth(uint32_t& d): depth(d) { depth++; }
			~Depth() { depth--; }
		} const nesting(t_depth);
		const auto add = [&](uint64_t begin, uint64_t finish) {
			begin  = std::max(begin, vaddr);
			finish = std::min(finish, end);
			if (begin >= finish) {
				return;
			}
			if (!spans.empty() && spans.back().second == begin) {
				spans.back().second = finish;
			} else {
				spans.emplace_back(begin, finish);
			}
		};
		bool left = false;
		for (uint64_t index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < end;
		     index++) {
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			const auto start   = std::max(index * TRACKER_REGION_SIZE, vaddr);
			const auto finish  = std::min((index + 1) * TRACKER_REGION_SIZE, end);
			if (manager == nullptr) {
				add(start, finish);
				continue;
			}
			if (!manager->HasCpuDirty()) {
				continue;
			}
			bool region_left = false;
			{
				std::scoped_lock lock(manager->lock);
				region_left = manager->ForEachCpuDirtyRangeBefore(
				    start, finish - start, limit, first_left,
				    [&](uint64_t address, uint64_t bytes) { add(address, address + bytes); });
			}
			if (region_left) {
				QueueCpuDirty(manager);
				left = true;
			}
		}
		for (const auto& [begin, finish]: spans) {
			func(begin, finish - begin);
		}
		return left;
	}

	// Calls func(address, size) for the runs of pages of [vaddr, vaddr + size) that the CPU may
	// have written since the previous call over them: pages that became CPU dirty since then, or
	// are still CPU dirty. Untracked pages start tracking as written. A single consumer owns the
	// written state.
	template <typename Func>
	void ConsumeCpuWrites(uint64_t vaddr, uint64_t size, Func&& func) {
		CheckNotInUploadCallback();
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->ConsumeCpuWritten(manager->GetCpuAddr() + offset, bytes,
			                           [&](uint64_t address, uint64_t length) {
				                           if (!runs.empty() &&
				                               runs.back().first + runs.back().second == address) {
					                           runs.back().second += length;
				                           } else {
					                           runs.emplace_back(address, length);
				                           }
			                           });
		});
		const auto end = vaddr + size;
		for (const auto& [address, length]: runs) {
			const auto begin  = std::max(address, vaddr);
			const auto finish = std::min(address + length, end);
			if (begin < finish) {
				func(begin, finish - begin);
			}
		}
	}

	// Marks the pages of [vaddr, vaddr + size) GPU modified, except those that are CPU dirty, and
	// calls func(address, size) for the page runs it marked.
	template <typename Func>
	void MarkRegionAsGpuModifiedUnlessCpuDirty(uint64_t vaddr, uint64_t size, Func&& func) {
		CheckNotInUploadCallback();
		PageWriteSeq::RaiseCurrent(vaddr, size);
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->MarkGpuDirtyUnlessCpuDirty(
			    manager->GetCpuAddr() + offset, bytes,
			    [&](uint64_t address, uint64_t length) { runs.emplace_back(address, length); });
		});
		for (const auto& [address, length]: runs) {
			func(address, length);
		}
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			// A read-only use of a region without CPU-dirty pages has nothing to upload (a CPU
			// write racing with this check is one that lands after the synchronization).
			if (!is_written && !manager->HasCpuDirty()) {
				return;
			}
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			PageWriteSeq::RaiseCurrent(vaddr, size);
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);
	void QueueCpuDirty(RegionManager* manager) noexcept {
		if (manager->cpu_dirty_queued.exchange(true, std::memory_order_acq_rel)) return;
		auto* head = m_cpu_dirty_head.load(std::memory_order_relaxed);
		do {
			manager->cpu_dirty_next = head;
		} while (!m_cpu_dirty_head.compare_exchange_weak(head, manager,
		    std::memory_order_release, std::memory_order_relaxed));
	}

	std::atomic<RegionManager*> m_cpu_dirty_head {nullptr};
	std::atomic_uint64_t m_cpu_generation {1};
	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
