#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/hostCopyQueue.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <unordered_map>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {
std::atomic<uint64_t> g_protect_calls {0};
std::atomic<uint64_t> g_protect_bytes {0};
std::atomic<uint64_t> g_protect_gpu_calls {0};
std::atomic<uint64_t> g_read_faults {0};
std::atomic<uint64_t> g_write_faults {0};
} // namespace

void ProtectStats::Count(uint64_t bytes) noexcept {
	g_protect_calls.fetch_add(1, std::memory_order_relaxed);
	g_protect_bytes.fetch_add(bytes, std::memory_order_relaxed);
	if (GuestGpu::IsGpuThread()) {
		g_protect_gpu_calls.fetch_add(1, std::memory_order_relaxed);
	}
}

void ProtectStats::CountFault(bool write) noexcept {
	(write ? g_write_faults : g_read_faults).fetch_add(1, std::memory_order_relaxed);
}

namespace {
std::mutex                             g_fault_pages_mutex;
std::unordered_map<uint64_t, uint32_t> g_fault_pages;
} // namespace

void ProtectStats::NoteWriteFaultPage(uint64_t vaddr) noexcept {
	std::lock_guard lock(g_fault_pages_mutex);
	++g_fault_pages[vaddr & ~uint64_t {0xfff}];
}

void ProtectStats::PrintFaultPages() noexcept {
	std::vector<std::pair<uint64_t, uint32_t>> pages;
	{
		std::lock_guard lock(g_fault_pages_mutex);
		pages.assign(g_fault_pages.begin(), g_fault_pages.end());
		g_fault_pages.clear();
	}
	std::array<uint64_t, 5> buckets {}; // 1, 2-9, 10-49, 50-99, 100+
	std::array<uint64_t, 5> faults {};
	uint64_t                total = 0;
	for (const auto& [page, count]: pages) {
		const size_t b = count == 1 ? 0 : count < 10 ? 1 : count < 50 ? 2 : count < 100 ? 3 : 4;
		++buckets[b];
		faults[b] += count;
		total += count;
	}
	std::sort(pages.begin(), pages.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
	std::printf("NHL27FAULTPAGES: faults=%llu pages=%zu | pages(faults) 1:%llu(%llu) 2-9:%llu(%llu) "
	            "10-49:%llu(%llu) 50-99:%llu(%llu) 100+:%llu(%llu) | top",
	            static_cast<unsigned long long>(total), pages.size(),
	            static_cast<unsigned long long>(buckets[0]), static_cast<unsigned long long>(faults[0]),
	            static_cast<unsigned long long>(buckets[1]), static_cast<unsigned long long>(faults[1]),
	            static_cast<unsigned long long>(buckets[2]), static_cast<unsigned long long>(faults[2]),
	            static_cast<unsigned long long>(buckets[3]), static_cast<unsigned long long>(faults[3]),
	            static_cast<unsigned long long>(buckets[4]), static_cast<unsigned long long>(faults[4]));
	for (size_t i = 0; i < pages.size() && i < 6; ++i) {
		std::printf(" %llx:%u", static_cast<unsigned long long>(pages[i].first), pages[i].second);
	}
	std::printf("\n");
}

void ProtectStats::Read(uint64_t& calls, uint64_t& bytes, uint64_t& gpu_calls,
                        uint64_t& read_faults, uint64_t& write_faults) noexcept {
	calls        = g_protect_calls.load(std::memory_order_relaxed);
	bytes        = g_protect_bytes.load(std::memory_order_relaxed);
	gpu_calls    = g_protect_gpu_calls.load(std::memory_order_relaxed);
	read_faults  = g_read_faults.load(std::memory_order_relaxed);
	write_faults = g_write_faults.load(std::memory_order_relaxed);
}

namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	// Many small images can share one page (NHL 27's jersey selection registers more than 127
	// on a page), so write watchers use 15 bits.
	struct PageState {
		uint16_t write_watchers  : 15 = 0;
		uint16_t access_watchers : 1  = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7fff) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 2);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		// Pages whose protection still has to be applied (from their watchers then), and whether
		// the host copy thread has an apply call queued (deferred protection).
		std::array<uint64_t, (REGION_PAGES + 63) / 64> stale {};
		std::atomic<bool>                               apply_queued {false};
		// Held across an apply, protection calls included, so applies of a region run one at a
		// time and the last one sees the latest watchers. The spin lock is held only to read them.
		std::mutex apply_mutex;
		uint64_t   base = 0;
	};
	inline static thread_local HostCopyQueue* t_deferred_queue = nullptr;
	// ReleaseScope: the regions whose released pages the scope applies when it ends (zeroed:
	// thread storage).
	struct ReleaseList {
		std::array<Region*, 8> regions;
		size_t                 count;
		bool                   active;
	};
	inline static thread_local ReleaseList t_release;

	// Host copy thread: protects the region's stale pages as their watchers want now.
	static void ApplyStale(void* argument) noexcept {
		auto& region = *static_cast<Region*>(argument);
		region.apply_queued.store(false, std::memory_order_release);
		ApplyStaleNow(region);
	}

	// Protects the region's stale pages as their watchers want now: runs of pages with the same
	// protection are collected under the spin lock and protected after it is released.
	static void ApplyStaleNow(Region& region) noexcept {
		struct Run {
			size_t                      first = 0;
			size_t                      count = 0;
			Common::VirtualMemory::Mode perms = Common::VirtualMemory::Mode::NoAccess;
		};
		std::lock_guard     apply_lock(region.apply_mutex);
		std::array<Run, 16> runs;
		for (size_t page = 0; page < REGION_PAGES;) {
			size_t count = 0;
			{
				SpinGuard lock(region.lock);
				while (page < REGION_PAGES && count < runs.size()) {
					const auto word = region.stale[page / 64] >> (page % 64);
					if (word == 0) {
						page = (page / 64 + 1) * 64;
						continue;
					}
					page += static_cast<size_t>(std::countr_zero(word));
					const auto perms = region.pages[page].Perms();
					size_t     end   = page;
					while (end < REGION_PAGES && (region.stale[end / 64] >> (end % 64) & 1u) != 0 &&
					       region.pages[end].Perms() == perms) {
						region.stale[end / 64] &= ~(uint64_t {1} << (end % 64));
						end++;
					}
					runs[count++] = {page, end - page, perms};
					page          = end;
				}
			}
			for (size_t i = 0; i < count; i++) {
				Protect(region.base + runs[i].first * PAGE_SIZE, runs[i].count * PAGE_SIZE,
				        runs[i].perms);
			}
		}
	}

	// Whether the calling thread's ReleaseScope takes the region (false: apply it now).
	static bool DeferRelease(Region& region) noexcept {
		auto& list = t_release;
		if (!list.active) {
			return false;
		}
		for (size_t i = 0; i < list.count; i++) {
			if (list.regions[i] == &region) {
				return true;
			}
		}
		if (list.count == list.regions.size()) {
			return false;
		}
		list.regions[list.count++] = &region;
		return true;
	}

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		region->base = index * REGION_SIZE;
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	static void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		ProtectStats::Count(size);
		FrameBreakdown::Add(FrameBreakdown::g_counters.protect_calls);
		FrameBreakdown::Timer protect_timer(FrameBreakdown::g_counters.protect_ns);
		FrameBreakdown::FaultTimer fault_protect_timer(FrameBreakdown::g_fault_cost.protect_ns);
		if (FrameBreakdown::g_in_fault) {
			FrameBreakdown::g_fault_cost.protect_calls.fetch_add(1, std::memory_order_relaxed);
		}
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		bool apply = false;
		UpdateRegionWatchersLocked<track, is_read, masked>(region, base_addr, first, last, mask,
		                                                     apply);
		if (!apply) {
			return;
		}
		// Deferred protection: only pages gaining write watchers (see DeferredProtectionScope).
		// Queued after the lock is released: the copy thread takes it to apply.
		if constexpr (track && !is_read) {
			if (auto* const deferred = t_deferred_queue; deferred != nullptr) {
				if (!region.apply_queued.exchange(true, std::memory_order_acq_rel)) {
					deferred->Call(&ApplyStale, &region);
				}
				return;
			}
		}
		if constexpr (!track) {
			if (DeferRelease(region)) {
				return;
			}
		}
		ApplyStaleNow(region);
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchersLocked(Region& region, uint64_t base_addr, size_t first, size_t last,
	                                const RegionBits* mask, bool& apply) {
		SpinGuard lock(region.lock);
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;
		bool      read_changed          = false;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				for (auto page = range_begin; page < range_begin + range_bytes / PAGE_SIZE; page++) {
					region.stale[page / 64] |= uint64_t {1} << (page % 64);
				}
				apply                 = true;
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if constexpr (is_read) {
				read_changed = read_changed || watcher_edge;
			}
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
		if constexpr (is_read) {
			if (read_changed) {
				read_watch_version.fetch_add(1, std::memory_order_release);
			}
		}
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	// Advances after any page gains or loses its read watcher (IsReadWatched caches per thread).
	std::atomic<uint64_t> read_watch_version {0};
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::DeferredProtectionScope::DeferredProtectionScope(HostCopyQueue* queue) noexcept
    : m_previous(Impl::t_deferred_queue) {
	Impl::t_deferred_queue = queue;
}

PageManager::DeferredProtectionScope::~DeferredProtectionScope() {
	Impl::t_deferred_queue = m_previous;
}

bool PageManager::DeferringProtection() noexcept {
	return Impl::t_deferred_queue != nullptr;
}

PageManager::ReleaseScope::ReleaseScope(bool enable) noexcept
    : m_owner(enable && !Impl::t_release.active) {
	if (m_owner) {
		Impl::t_release.active = true;
	}
}

PageManager::ReleaseScope::~ReleaseScope() {
	if (!m_owner) {
		return;
	}
	auto& list  = Impl::t_release;
	list.active = false;
	for (size_t i = 0; i < list.count; i++) {
		Impl::ApplyStaleNow(*list.regions[i]);
	}
	list.count = 0;
}

bool PageManager::ApplyPending(uint64_t vaddr) {
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto page = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	{
		SpinGuard lock(region->lock);
		if ((region->stale[page / 64] >> (page % 64) & 1u) == 0) {
			return false;
		}
	}
	Impl::ApplyStaleNow(*region);
	return true;
}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

bool PageManager::IsWatched(uint64_t vaddr) const {
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto page = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	SpinGuard  lock(region->lock);
	return region->pages[page].write_watchers != 0 || region->pages[page].access_watchers != 0;
}

bool PageManager::IsReadWatched(uint64_t vaddr) const {
	// Read checks ask about the same pages draw after draw, while read watchers change rarely.
	struct Answer {
		uint64_t page    = ~uint64_t {0};
		uint64_t version = 0;
		bool     watched = false;
	};
	static thread_local std::array<Answer, 64> answers {};
	const auto page_address = vaddr & ~(PAGE_SIZE - 1);
	auto&      answer       = answers[(page_address / PAGE_SIZE) & 63u];
	const auto version      = m_impl->read_watch_version.load(std::memory_order_acquire);
	if (answer.page == page_address && answer.version == version) {
		return answer.watched;
	}
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto page = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	SpinGuard  lock(region->lock);
	const bool watched = region->pages[page].access_watchers != 0;
	answer = {page_address, version, watched};
	return watched;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
