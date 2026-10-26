// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

static constexpr uint64_t HOST_SYSTEM_MANAGED_MIN  = 0x0000040000ull;
static constexpr uint64_t HOST_SYSTEM_MANAGED_MAX  = 0x07ffffbfffull;
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fbfffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x7000000000ull;
#else
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fffffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x1000000000ull;
#endif
static constexpr uint64_t HOST_USER_MAX = 0xfbffffffffull;

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	if (UINT64_MAX - base < range_size || candidate < base) {
		return false;
	}
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return size != 0 && VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}
#endif

template <typename Regions, typename Region, typename Gap>
static void ForEachSparseRange(const Regions& regions, uint64_t vaddr, uint64_t size,
                               Region&& region, Gap&& gap) {
	const auto end     = vaddr + size;
	auto       mapping = regions.lower_bound(vaddr);
	if (mapping != regions.begin()) {
		auto previous = std::prev(mapping);
		if (previous->second.vaddr + previous->second.size > vaddr) {
			mapping = previous;
		}
	}

	auto cursor = vaddr;
	for (; mapping != regions.end() && mapping->second.vaddr < end; ++mapping) {
		const auto& entry = mapping->second;
		const auto  start = std::max(vaddr, entry.vaddr);
		const auto  bytes = std::min(end, entry.vaddr + entry.size) - start;
		if (cursor < start) {
			gap(cursor, start - cursor);
		}
		region(entry, start, bytes);
		cursor = start + bytes;
	}
	if (cursor < end) {
		gap(cursor, end - cursor);
	}
}

template <typename Regions, typename Source, typename Gap>
static void CopySparseRange(const Regions& regions, uint64_t vaddr, void* data, uint64_t size,
                            Source&& source, Gap&& gap) {
	auto* destination = static_cast<uint8_t*>(data);
	ForEachSparseRange(
	    regions, vaddr, size,
	    [&](const auto& entry, uint64_t start, uint64_t bytes) {
		    std::memcpy(destination + (start - vaddr), source(entry, start), bytes);
	    },
	    [&](uint64_t start, uint64_t bytes) { gap(start, destination + (start - vaddr), bytes); });
}

class GuestBackingStore {
public:
	enum class FailureReason {
		None,
		BackingUnavailable,
		PhysicalRangeOutOfBounds,
		FixedMapViewFailed,
		PlaceholderMapFailed,
		ProtectFailed,
		WrongHostAddress,
		AddressRangeUnavailable,
	};

	explicit GuestBackingStore(uint64_t size): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		m_handle = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
		if (m_handle == nullptr) {
			LOGF_COLOR(Log::Color::Red,
			           "\t direct-memory backing: CreateFileMapping failed: 0x%08" PRIx32 "\n",
			           static_cast<uint32_t>(GetLastError()));
			return;
		}
		m_backing_base = static_cast<uint8_t*>(
		    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));
		if (m_backing_base == nullptr) {
			LOGF_COLOR(Log::Color::Red,
			           "\t direct-memory backing: writable alias failed: 0x%08" PRIx32 "\n",
			           static_cast<uint32_t>(GetLastError()));
			CloseHandle(m_handle);
			m_handle = nullptr;
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#if defined(__APPLE__)
		{
			char shm_name[64];
			snprintf(shm_name, sizeof(shm_name), "/kyty-dm-%d-%llx", static_cast<int>(getpid()),
			         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
			m_fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
			if (m_fd >= 0) {
				shm_unlink(shm_name);
			}
		}
#else
		m_fd = static_cast<int>(syscall(SYS_memfd_create, "KytyDirectMemory", 0));
#endif
		if (m_fd < 0) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: shared fd creation failed: %d\n",
			           errno);
			return;
		}
		if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: ftruncate failed: %d\n", errno);
			close(m_fd);
			m_fd = -1;
			return;
		}
		auto* backing = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
		if (backing == MAP_FAILED) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: writable alias failed: %d\n",
			           errno);
			close(m_fd);
			m_fd = -1;
			return;
		}
		m_backing_base = static_cast<uint8_t*>(backing);
#endif
	}

	~GuestBackingStore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [_, entry]: m_maps) {
			UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size);
		}
		m_maps.clear();
		BumpMapGeneration();
		MapsChangedUnlocked();
		if (m_backing_base != nullptr) {
			UnmapViewOfFile(m_backing_base);
		}
		if (m_handle != nullptr) {
			CloseHandle(m_handle);
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		for (const auto& [_, entry]: m_maps) {
			EXIT_IF(
			    !UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size));
		}
		m_maps.clear();
		BumpMapGeneration();
		MapsChangedUnlocked();
		if (m_backing_base != nullptr) {
			munmap(m_backing_base, m_size);
		}
		if (m_fd >= 0) {
			close(m_fd);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestBackingStore);

	static const char* GetFailureReasonName(FailureReason reason) {
		switch (reason) {
			case FailureReason::None: return "n/a";
			case FailureReason::BackingUnavailable: return "backing-unavailable";
			case FailureReason::PhysicalRangeOutOfBounds: return "physical-range-out-of-bounds";
			case FailureReason::FixedMapViewFailed: return "fixed-map-view-failed";
			case FailureReason::PlaceholderMapFailed: return "placeholder-map-failed";
			case FailureReason::ProtectFailed: return "protect-failed";
			case FailureReason::WrongHostAddress: return "wrong-host-address";
			case FailureReason::AddressRangeUnavailable: return "address-range-unavailable";
			default: return "unknown";
		}
	}

	[[nodiscard]] bool IsAvailable() const {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return m_handle != nullptr && m_backing_base != nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		return m_fd >= 0 && m_backing_base != nullptr;
#else
		return false;
#endif
	}

	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }

	bool ZeroRange(uint64_t backing_offset, uint64_t size) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		std::memset(m_backing_base + backing_offset, 0, size);
		return true;
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		const auto* source = static_cast<const uint8_t*>(data);
		if (TryCachedTransfer(vaddr, size, [source](uint8_t* backing, size_t bytes) {
			    std::memcpy(backing, source, bytes);
		    })) {
			return true;
		}

		// Skip unbacked gaps, mirroring sparse reads that zero-fill them.
		std::lock_guard lock(m_mutex);
		const auto      end   = vaddr + size;
		auto            first = m_maps.upper_bound(vaddr);
		if (first != m_maps.begin()) {
			RememberMapping(std::prev(first)->second, vaddr, end);
		}
		ForEachSparseRange(
		    m_maps, vaddr, size,
		    [&](const auto& entry, uint64_t start, uint64_t bytes) {
			    std::memcpy(m_backing_base + (entry.backing_offset + start - entry.vaddr),
			                source + (start - vaddr), bytes);
		    },
		    [](uint64_t, uint64_t) {});
		return true;
	}

	// Writes each byte of `data` whose backing byte still equals `expected` (aligned dwords with one
	// compare-exchange each, so a concurrent CPU store into the dword wins).
	bool TryMergeBacking(uint64_t vaddr, const void* data, const void* expected, uint64_t size) {
		if (data == nullptr || expected == nullptr) {
			return false;
		}
		const auto* source = static_cast<const uint8_t*>(data);
		const auto* old    = static_cast<const uint8_t*>(expected);
		return TryTransferBacking(vaddr, size, [source, old](uint8_t* backing, uint64_t data_offset, size_t bytes) {
			for (size_t i = 0; i < bytes;) {
				const auto at = data_offset + i;
				if ((reinterpret_cast<uintptr_t>(backing + i) & 3u) == 0 && bytes - i >= 4) {
					uint32_t want = 0;
					uint32_t seen = 0;
					std::memcpy(&want, source + at, 4);
					std::memcpy(&seen, old + at, 4);
					std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t*>(backing + i))
					    .compare_exchange_strong(seen, want);
					i += 4;
				} else {
					if (backing[i] == old[at]) backing[i] = source[at];
					++i;
				}
			}
		});
	}

	// fast: enables the lock-free single-mapping read cache (KYTY_SRT_SOURCE_REUSE).
	// Resource reads are mostly a descriptor or a few dwords: those are copied without a call (the
	// calls were 7% of the resolve thread).
	static void CopyGuestBytes(uint8_t* destination, const uint8_t* source, uint64_t size) {
		switch (size) {
			case 4: std::memcpy(destination, source, 4); return;
			case 8: std::memcpy(destination, source, 8); return;
			case 16: std::memcpy(destination, source, 16); return;
			case 32: std::memcpy(destination, source, 32); return;
			default: std::memcpy(destination, source, size); return;
		}
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size, bool fast = false) {
		if (data == nullptr) {
			return false;
		}
		auto* destination = static_cast<uint8_t*>(data);
		if (fast) {
			// Lock-free hit on this thread's last single-mapping read. Every m_maps mutation
			// bumps m_map_gen (release, under m_mutex) right after mutating; the entry is
			// used only if the generation is unchanged before AND after the copy, so a
			// concurrent unmap/remap is either ordered after this read or forces the slow path.
			auto&      c  = t_read_cache;
			const auto g0 = m_map_gen.load(std::memory_order_acquire);
			if (c.owner == this && c.gen == g0 && size != 0 && vaddr >= c.begin && vaddr - c.begin < c.length &&
			    size <= c.length - (vaddr - c.begin)) {
				CopyGuestBytes(destination, c.host + (vaddr - c.begin), size);
				std::atomic_thread_fence(std::memory_order_acquire);
				if (m_map_gen.load(std::memory_order_relaxed) == g0) return true;
			}
			// Experiment BackingReadMru: resource reads alternate between a few mappings (tables
			// and the buffers they point to); the one before the last also skips the map search
			// (5% of the resolve thread), and becomes the last again.
			if (Common::Experiments::Enabled(Common::Experiments::BackingReadMru)) {
				auto& p = t_read_cache_previous;
				if (p.owner == this && p.gen == g0 && size != 0 && vaddr >= p.begin && vaddr - p.begin < p.length &&
				    size <= p.length - (vaddr - p.begin)) {
					CopyGuestBytes(destination, p.host + (vaddr - p.begin), size);
					std::atomic_thread_fence(std::memory_order_acquire);
					if (m_map_gen.load(std::memory_order_relaxed) == g0) {
						std::swap(c, p);
						return true;
					}
				}
			}
			// Experiment BackingLookupCache: the per-block lookup cache before the map search.
			if (Common::Experiments::Enabled(Common::Experiments::BackingLookupCache) &&
			    TryCachedTransfer(vaddr, size, [destination](uint8_t* backing, size_t bytes) {
				    CopyGuestBytes(destination, backing, bytes);
			    })) {
				return true;
			}
			return TryTransferBacking(
			    vaddr, size,
			    [destination](uint8_t* backing, uint64_t data_offset, size_t bytes) {
				    std::memcpy(destination + data_offset, backing, bytes);
			    },
			    true);
		}
		if (TryCachedTransfer(vaddr, size, [destination](uint8_t* backing, size_t bytes) {
			    CopyGuestBytes(destination, backing, bytes);
		    })) {
			return true;
		}
		return TryTransferBacking(
		    vaddr, size, [destination](uint8_t* backing, uint64_t data_offset, size_t bytes) {
			    std::memcpy(destination + data_offset, backing, bytes);
		    });
	}

	// Compares backing bytes with `expected` without copying them. Same mapping validation as TryReadBacking:
	// false (equal untouched) when the span is unavailable or crosses a gap.
	bool TryCompareBacking(uint64_t vaddr, const void* expected, uint64_t size, bool& equal) {
		if (expected == nullptr) {
			return false;
		}
		const auto* source = static_cast<const uint8_t*>(expected);
		bool        same   = true;
		{
			// Lock-free hit on this thread's last single-mapping access (see TryReadBacking): the
			// compare is valid only if the map generation is unchanged before and after it.
			auto&      c  = t_read_cache;
			const auto g0 = m_map_gen.load(std::memory_order_acquire);
			if (c.owner == this && c.gen == g0 && size != 0 && vaddr >= c.begin && vaddr - c.begin < c.length &&
			    size <= c.length - (vaddr - c.begin)) {
				const bool match = std::memcmp(c.host + (vaddr - c.begin), source, size) == 0;
				std::atomic_thread_fence(std::memory_order_acquire);
				if (m_map_gen.load(std::memory_order_relaxed) == g0) {
					equal = match;
					return true;
				}
			}
			// As TryReadBacking's fast path: the mapping before the last, then the per-block
			// lookup cache, before the map search (the search was a third of the execution
			// thread's compares of read-log entries).
			auto& p = t_read_cache_previous;
			if (Common::Experiments::Enabled(Common::Experiments::BackingReadMru) && p.owner == this &&
			    p.gen == g0 && size != 0 && vaddr >= p.begin && vaddr - p.begin < p.length &&
			    size <= p.length - (vaddr - p.begin)) {
				const bool match = std::memcmp(p.host + (vaddr - p.begin), source, size) == 0;
				std::atomic_thread_fence(std::memory_order_acquire);
				if (m_map_gen.load(std::memory_order_relaxed) == g0) {
					std::swap(c, p);
					equal = match;
					return true;
				}
			}
		}
		if (Common::Experiments::Enabled(Common::Experiments::BackingLookupCache) &&
		    TryCachedTransfer(vaddr, size, [source, &same](uint8_t* backing, size_t bytes) {
			    same = std::memcmp(backing, source, bytes) == 0;
		    })) {
			equal = same;
			return true;
		}
		const bool  ok     = TryTransferBacking(
		    vaddr, size,
		    [source, &same](uint8_t* backing, uint64_t data_offset, size_t bytes) {
			    same = same && std::memcmp(backing, source + data_offset, bytes) == 0;
		    },
		    true);
		if (ok) {
			equal = same;
		}
		return ok;
	}

	// The backing alias of a range that one mapping holds, or null. The alias stays mapped for
	// the store's lifetime, so the pointer can be read later, even after the guest range changed.
	const uint8_t* FindBackingPointer(uint64_t vaddr, uint64_t size) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return nullptr;
		}
		const uint8_t* result = nullptr;
		if (TryCachedTransfer(vaddr, size, [&](uint8_t* backing, size_t) { result = backing; })) {
			return result;
		}

		std::shared_lock lock(m_mutex);
		const auto       end     = vaddr + size;
		auto             mapping = m_maps.upper_bound(vaddr);
		if (mapping == m_maps.begin()) {
			return nullptr;
		}
		const auto& entry = std::prev(mapping)->second;
		if (vaddr < entry.vaddr || end > entry.vaddr + entry.size ||
		    !RangeOk(entry.backing_offset + vaddr - entry.vaddr, size)) {
			return nullptr;
		}
		RememberMapping(entry, vaddr, end);
		return m_backing_base + entry.backing_offset + (vaddr - entry.vaddr);
	}

	template <typename Gap>
	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size, Gap&& gap) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::shared_lock lock(m_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		CopySparseRange(m_maps, vaddr, data, size,
		                [this](const auto& entry, uint64_t start) {
			                return m_backing_base + (entry.backing_offset + start - entry.vaddr);
		                },
		                gap);
		return true;
	}

	bool MapFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	              VirtualMemory::Mode mode, FailureReason* failure_reason = nullptr) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			SetFailure(failure_reason, !IsAvailable() ? FailureReason::BackingUnavailable
			                                          : FailureReason::PhysicalRangeOutOfBounds);
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		void* mapped = ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			if (mapped != nullptr) {
				UnmapViewPreservePlaceholder(mapped, size);
				SetFailure(failure_reason, FailureReason::WrongHostAddress);
			}
			return false;
		}

		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(!m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode}).second);
			BumpMapGeneration();
			MapsChangedUnlocked();
		}
		BumpBackingEpoch();
		SetFailure(failure_reason, FailureReason::None);
		return true;
#else
		std::lock_guard lock(m_mutex);
		void* mapped = MapView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			return false;
		}
		RemoveMappingsUnlocked(vaddr, size);
		EXIT_IF(!m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode}).second);
		BumpMapGeneration();
		MapsChangedUnlocked();
		BumpBackingEpoch();
		SetFailure(failure_reason, FailureReason::None);
		return true;
#endif
	}

	bool Unmap(uint64_t vaddr, uint64_t size, bool* placeholder_preserved = nullptr) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		BumpBackingEpoch();

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = false;
		}

		const auto            end = vaddr + size;
		std::vector<MapEntry> targets;
		{
			std::lock_guard lock(m_mutex);
			auto            current = vaddr;
			while (current < end) {
				MapEntry entry {};
				if (!FindContainingUnlocked(current, 1, &entry)) {
					return false;
				}
				const auto part_size = std::min(end, entry.vaddr + entry.size) - current;
				targets.push_back(
				    {current, part_size, entry.backing_offset + current - entry.vaddr, entry.mode});
				current += part_size;
			}
		}

		std::vector<MapEntry> removed;
		removed.reserve(targets.size());
		for (const auto& entry: targets) {
			bool part_placeholder_preserved = false;
			if (!UnmapOne(entry.vaddr, entry.size, &part_placeholder_preserved)) {
				for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
					EXIT_IF(!MapFixed(it->vaddr, it->size, it->backing_offset, it->mode));
				}
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = false;
				}
				return false;
			}
			EXIT_IF(!part_placeholder_preserved);
			removed.push_back(entry);
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Each unmapped file view becomes a separate Windows placeholder. The address-space owner
		// records this contiguous guest span as one free range, so normalize fragmented backing
		// views into one placeholder before publishing it there.
		if (removed.size() > 1 && !CoalescePlaceholders(vaddr, size)) {
			for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
				EXIT_IF(!MapFixed(it->vaddr, it->size, it->backing_offset, it->mode));
			}
			return false;
		}
#endif

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
	}

	bool UnmapOne(uint64_t vaddr, uint64_t size, bool* placeholder_preserved = nullptr) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_store_unmaps_before_failure == 0) {
			g_test_backing_store_unmaps_before_failure = UINT32_MAX;
			return false;
		}
		if (g_test_backing_store_unmaps_before_failure != UINT32_MAX) {
			g_test_backing_store_unmaps_before_failure--;
		}
#endif

		MapEntry old {};
		{
			std::lock_guard lock(m_mutex);
			auto            next = m_maps.upper_bound(vaddr);
			if (next != m_maps.begin()) {
				auto entry = std::prev(next);
				if (vaddr + size <= entry->second.vaddr + entry->second.size) {
					old = entry->second;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					m_maps.erase(entry);
					BumpMapGeneration();
					MapsChangedUnlocked();
#endif
				}
			}
		}

		if (old.size == 0) {
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const uint64_t left_size  = vaddr - old.vaddr;
		const uint64_t right_addr = vaddr + size;
		const uint64_t right_size = old.vaddr + old.size - right_addr;
		const bool exact_unmap = (vaddr == old.vaddr && size == old.size);
		if (exact_unmap) {
			const bool unmapped =
			    UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size);
			if (!unmapped) {
				std::lock_guard lock(m_mutex);
				m_maps.emplace(old.vaddr, old);
				BumpMapGeneration();
				MapsChangedUnlocked();
				return false;
			}
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}

		if (UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size)) {
			bool ok = true;
			if (left_size != 0 && left_size != old.size) {
				ok = SplitPlaceholder(old.vaddr, left_size) && ok;
			}
			if (right_size != 0) {
				ok = SplitPlaceholder(vaddr, size) && ok;
			}
			if (left_size != 0) {
				ok = MapFixed(old.vaddr, left_size, old.backing_offset, old.mode) && ok;
			}
			if (right_size != 0) {
				ok = MapFixed(right_addr, right_size,
				              old.backing_offset + (right_addr - old.vaddr), old.mode) &&
				     ok;
			}
			if (ok) {
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = true;
				}
				return true;
			}

			if (left_size != 0 && Contains(old.vaddr, left_size)) {
				EXIT_IF(!UnmapOne(old.vaddr, left_size));
			}
			if (right_size != 0 && Contains(right_addr, right_size)) {
				EXIT_IF(!UnmapOne(right_addr, right_size));
			}
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(old.vaddr), old.size,
			                    MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) == 0);
			EXIT_IF(!MapFixed(old.vaddr, old.size, old.backing_offset, old.mode));
			return false;
		}
		std::lock_guard lock(m_mutex);
		m_maps.emplace(old.vaddr, old);
		BumpMapGeneration();
		MapsChangedUnlocked();
		return false;
#else
		// Replace only the requested range so concurrent accesses and GPU watch
		// protections on the surviving mappings remain intact.
		if (!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size)) {
			return false;
		}

		std::lock_guard lock(m_mutex);
		RemoveMappingsUnlocked(vaddr, size);
		MapsChangedUnlocked();

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
#endif
	}

	bool Contains(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::shared_lock lock(m_mutex);
		const auto      end     = vaddr + size;
		auto            current = vaddr;

		while (current < end) {
			MapEntry entry {};
			if (!FindContainingUnlocked(current, 1, &entry)) {
				return false;
			}
			const auto entry_end = entry.vaddr + entry.size;
			if (entry_end <= current) {
				return false;
			}
			current = std::min(end, entry_end);
		}

		return true;
	}

private:
	struct ReadCache {
		// No member initializers: a thread_local static is zero-initialized.
		const void* owner;
		uint64_t    gen;
		uint64_t    begin;
		uint64_t    length;
		uint8_t*    host;
	};
	static inline thread_local ReadCache t_read_cache;
	// The mapping t_read_cache held before (experiment BackingReadMru).
	static inline thread_local ReadCache t_read_cache_previous;
	static inline std::atomic<uint64_t> s_gen_epoch {0};
	std::atomic<uint64_t>               m_map_gen {s_gen_epoch.fetch_add(uint64_t {1} << 32) + 1};
	// Called with m_mutex held, after mutating m_maps.
	void BumpMapGeneration() { m_map_gen.fetch_add(1, std::memory_order_release); }

	template <typename Transfer>
	bool TryTransferBacking(uint64_t vaddr, uint64_t size, Transfer transfer, bool fill_cache = false) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		// Copies out and compares only: concurrent readers share the lock.
		std::shared_lock lock(m_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		const auto end = vaddr + size;
		{
			// Single-mapping fast path (the common one-word SRT read): one lookup instead of
			// a validation pass plus a transfer pass. Same checks, same result, same lock.
			MapEntry first {};
			if (FindContainingUnlocked(vaddr, 1, &first) && first.vaddr + first.size >= end &&
			    RangeOk(first.backing_offset + vaddr - first.vaddr, size)) {
				transfer(m_backing_base + first.backing_offset + (vaddr - first.vaddr), 0,
				         static_cast<size_t>(size));
				RememberMapping(first, vaddr, end);
				if (fill_cache) {
					// m_mutex is held, so m_map_gen is stable and describes this entry.
					const auto ok_len = std::min<uint64_t>(first.size, m_size - first.backing_offset);
					t_read_cache_previous = t_read_cache;
					t_read_cache = {this, m_map_gen.load(std::memory_order_relaxed), first.vaddr, ok_len,
					                m_backing_base + first.backing_offset};
				}
				return true;
			}
		}
		auto            current = vaddr;
		while (current < end) {
			MapEntry entry {};
			if (!FindContainingUnlocked(current, 1, &entry)) {
				return false;
			}
			const auto bytes = std::min(end, entry.vaddr + entry.size) - current;
			if (!RangeOk(entry.backing_offset + current - entry.vaddr, bytes)) {
				return false;
			}
			current += bytes;
		}

		current = vaddr;
		while (current < end) {
			MapEntry entry {};
			EXIT_IF(!FindContainingUnlocked(current, 1, &entry));
			const auto bytes  = std::min(end, entry.vaddr + entry.size) - current;
			const auto offset = entry.backing_offset + current - entry.vaddr;
			transfer(m_backing_base + offset, current - vaddr, static_cast<size_t>(bytes));
			current += bytes;
		}
		return true;
	}

	struct MapEntry {
		uint64_t            vaddr          = 0;
		uint64_t            size           = 0;
		uint64_t            backing_offset = 0;
		VirtualMemory::Mode mode           = VirtualMemory::Mode::NoAccess;
	};

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	void RemoveMappingsUnlocked(uint64_t vaddr, uint64_t size) {
		BumpMapGeneration();
		const auto end = vaddr + size;
		auto       it = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) {
			--it;
		}
		while (it != m_maps.end() && it->first < end) {
			const auto old     = it->second;
			const auto old_end = old.vaddr + old.size;
			if (old_end <= vaddr) {
				++it;
				continue;
			}
			it = m_maps.erase(it);
			if (old.vaddr < vaddr) {
				m_maps.emplace(old.vaddr,
				               MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode});
			}
			if (end < old_end) {
				m_maps.emplace(end, MapEntry {end, old_end - end,
				                             old.backing_offset + end - old.vaddr, old.mode});
			}
		}
	}
#endif
	// One mapping per 64 KiB block of guest addresses (direct mapped): resource reads of one draw
	// touch more mappings than a few recent ones (the resolve thread spent 5% in the map lookup).
	static constexpr uint32_t LookupSlots     = 64;
	static constexpr uint32_t LookupSlotShift = 16;
	struct LookupCache {
		const GuestBackingStore*          owner      = nullptr;
		uint64_t                          generation = 0;
		std::array<MapEntry, LookupSlots> entries {};
	};
	static constexpr uint32_t LookupSlot(uint64_t vaddr) noexcept {
		return static_cast<uint32_t>(vaddr >> LookupSlotShift) % LookupSlots;
	}

	static LookupCache& ThreadLookupCache() noexcept {
		thread_local LookupCache cache;
		return cache;
	}

	// Called after every change of m_maps, under m_mutex. The generation is shared by all stores,
	// so a store recreated at the address of a destroyed one never matches a stale lookup.
	static void MapsChangedUnlocked() noexcept {
		s_maps_generation.fetch_add(1, std::memory_order_release);
	}

	inline static std::atomic<uint64_t> s_maps_generation {0};

	// The GPU thread streams thousands of small buffers per frame from a few mappings: reuse the
	// lookups it made last, without the lock, while no mapping changed. The entries and the
	// generation were read together under m_mutex, and every change of m_maps advances the
	// generation under m_mutex, so observing it unchanged (acquire) means no change completed
	// since. A change racing with this transfer orders as if it followed it: the backing alias
	// stays mapped for the store's lifetime, and host views are never unmapped under m_mutex
	// either.
	template <typename Transfer>
	bool TryCachedTransfer(uint64_t vaddr, uint64_t size, Transfer transfer) {
		const auto& cached = ThreadLookupCache();
		if (cached.owner != this ||
		    cached.generation != s_maps_generation.load(std::memory_order_acquire)) {
			return false;
		}
		const auto  end   = vaddr + size;
		const auto& entry = cached.entries[LookupSlot(vaddr)];
		if (vaddr >= entry.vaddr && end <= entry.vaddr + entry.size) {
			transfer(m_backing_base + entry.backing_offset + (vaddr - entry.vaddr), static_cast<size_t>(size));
			return true;
		}
		return false;
	}

	// Called under m_mutex (shared or exclusive) after a lookup: remembers a mapping that holds
	// the whole transfer [entry.vaddr <= vaddr, end].
	void RememberMapping(const MapEntry& entry, uint64_t vaddr, uint64_t end) {
		if (vaddr < entry.vaddr || end > entry.vaddr + entry.size || !RangeOk(entry.backing_offset, entry.size)) {
			return;
		}
		auto&      cached     = ThreadLookupCache();
		const auto generation = s_maps_generation.load(std::memory_order_relaxed);
		if (cached.owner != this || cached.generation != generation) {
			cached = {this, generation};
		}
		cached.entries[LookupSlot(vaddr)] = entry;
	}

	bool FindContainingUnlocked(uint64_t vaddr, uint64_t size, MapEntry* out) const {
		EXIT_IF(out == nullptr);

		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto next = m_maps.upper_bound(vaddr);
		if (next == m_maps.begin()) {
			return false;
		}
		const auto& entry = std::prev(next)->second;
		if (vaddr + size <= entry.vaddr + entry.size) {
			*out = entry;
			return true;
		}

		return false;
	}

	[[nodiscard]] bool RangeOk(uint64_t backing_offset, uint64_t size) const {
		return size != 0 && backing_offset < m_size && size <= m_size - backing_offset;
	}

	static void SetFailure(FailureReason* out, FailureReason reason) {
		if (out != nullptr) {
			*out = reason;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}

	static bool ApplyProtection(void* ptr, uint64_t size, VirtualMemory::Mode mode) {
		DWORD old_protect = 0;
		return VirtualProtect(ptr, size, GetProtect(mode), &old_protect) != 0;
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	void* ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                             VirtualMemory::Mode mode, FailureReason* failure_reason) {
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr =
		    MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr),
		                   backing_offset, size, MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return nullptr;
		}

		if (mode == VirtualMemory::Mode::NoAccess && !ApplyProtection(ptr, size, mode)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return nullptr;
		}
		return ptr;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t /*size*/) {
		return UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	HANDLE m_handle = nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	[[nodiscard]] static uint64_t GetMappableTestSize() { return 0x4000; }

	static int GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PROT_READ;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::Execute: return PROT_EXEC;
			case VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PROT_NONE;
		}
	}

	void* MapView(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	              FailureReason* failure_reason) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (const auto callback = g_test_before_backing_map) {
			g_test_before_backing_map = nullptr;
			callback(vaddr, size);
		}
#endif
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, GetProtect(mode),
		                 MAP_SHARED | MAP_FIXED, m_fd, static_cast<off_t>(backing_offset));
		if (ptr == MAP_FAILED) {
			SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
			return nullptr;
		}
		return ptr;
	}

	// Replace the file view atomically instead of exposing an unmapped hole in the
	// process-lifetime guest reservation.
	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		void* ptr = mmap(vaddr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		return ptr != MAP_FAILED && ptr == vaddr;
	}

	int m_fd = -1;
#endif

	uint8_t*                     m_backing_base = nullptr;
	uint64_t                     m_size         = 0;
	std::shared_mutex            m_mutex;
	std::map<uint64_t, MapEntry> m_maps;
};

class GuestAddressSpace {
public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size);
		if (!m_backing->IsAvailable()) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory. Windows commits\n"
			     "this up front, so the paging file is usually what needs to be larger.\n",
			     backing_size / (1024U * 1024U));
#else
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory\n",
			     backing_size / (1024U * 1024U));
#endif
		}
	}

	~GuestAddressSpace() {
		delete m_backing;
		m_backing = nullptr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [start, size]: m_owned) {
			const auto end     = start + size;
			auto       current = start;
			while (current < end) {
				MEMORY_BASIC_INFORMATION info {};
				EXIT_IF(VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) ==
				        0);
				const auto region_start = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end   = region_start + static_cast<uint64_t>(info.RegionSize);
				if (info.State == MEM_COMMIT) {
					EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_start), info.RegionSize,
					                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
				if (info.State == MEM_RESERVE) {
					EXIT_IF(VirtualFree(info.AllocationBase, 0, MEM_RELEASE) == 0);
				}
				current = region_end;
			}
		}
#else
		for (const auto& [start, size]: m_owned) {
			munmap(reinterpret_cast<void*>(start), size);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestAddressSpace);

	[[nodiscard]] uint64_t GetGranularity() const { return m_granularity; }
	[[nodiscard]] uint64_t GetBackingBase() const { return m_backing->GetBackingBase(); }
	[[nodiscard]] uint64_t GetBackingSize() const { return m_backing->GetBackingSize(); }

	bool ZeroBacking(uint64_t backing_offset, uint64_t size) {
		return m_backing->ZeroRange(backing_offset, size);
	}

	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return VirtualRangesOverlap(vaddr, size, range.first, range.second);
		});
	}

	// Writes the backed bytes of a range that holds no private mappings.
	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return !OverlapsPrivateUnlocked(vaddr, size) &&
		       m_backing->TryWriteBacking(vaddr, data, size);
	}

	bool TryMergeBacking(uint64_t vaddr, const void* data, const void* expected, uint64_t size) {
		return m_backing->TryMergeBacking(vaddr, data, expected, size);
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size, bool fast = false) {
		return m_backing->TryReadBacking(vaddr, data, size, fast);
	}

	bool TryCompareBacking(uint64_t vaddr, const void* expected, uint64_t size, bool& equal) {
		return m_backing->TryCompareBacking(vaddr, expected, size, equal);
	}

	const uint8_t* FindBackingPointer(uint64_t vaddr, uint64_t size) {
		return m_backing->FindBackingPointer(vaddr, size);
	}

	template <bool copy_private>
	bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
		if (data == nullptr || !Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		if constexpr (!copy_private) {
			if (OverlapsPrivateUnlocked(vaddr, size)) {
				return false;
			}
		}
		return m_backing->TryReadSparseBacking(
		    vaddr, data, size, [&](uint64_t start, uint8_t* destination, uint64_t bytes) {
			    const auto zero = [](uint64_t, uint8_t* output, uint64_t length) {
				    std::memset(output, 0, length);
			    };
			    if constexpr (copy_private) {
				    // Backing gaps can contain private mappings or reservations only.
				    CopySparseRange(m_mapped, start, destination, bytes,
				                    [](const auto& region, uint64_t address) {
					                    EXIT_IF(region.type != MappingType::Private);
					                    return reinterpret_cast<const void*>(address);
				                    },
				                    zero);
			    } else {
				    zero(start, destination, bytes);
			    }
		    });
	}

	bool BackingContains(uint64_t vaddr, uint64_t size) { return m_backing->Contains(vaddr, size); }

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		// A shared lock here (the PageManager region lock orders protections of one page) measured as
		// noise twice and was removed (it was experiment bit 32, ProtectSharedLock).
		std::lock_guard lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
		std::lock_guard lock(m_mutex);
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		const bool replacing = ContainsMappedUnlocked(vaddr, size, MappingType::Backing);
#else
		constexpr bool replacing = false;
#endif
		if (!replacing && !ConsumeUnlocked(vaddr, size)) {
			if (failure_reason != nullptr) {
				*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
			}
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;
		if (m_backing->MapFixed(vaddr, size, backing_offset, host_mode, failure_reason)) {
			if (!replacing) {
				EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
			}
			return true;
		}
		if (!replacing) {
			AddFreeUnlocked(vaddr, size);
		}
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Backing)) {
			return false;
		}
		bool placeholder_preserved = false;
		if (!m_backing->Unmap(vaddr, size, &placeholder_preserved)) {
			return false;
		}
		if (!placeholder_preserved) {
			return false;
		}
		EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Backing));
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	uint64_t ReserveAligned(uint64_t search_addr, uint64_t size, uint64_t alignment) {
		if (size == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment != 0 ? alignment : PageSize(), PageSize());
		std::lock_guard lock(m_mutex);
		if (search_addr == 0) {
			auto addr = FindFreeAlignedUnlocked(HOST_SYSTEM_MANAGED_MIN,
			                                    HOST_SYSTEM_MANAGED_MAX + 1u, size, alignment);
			return addr != 0 ? addr
			                 : FindFreeAlignedUnlocked(HOST_USER_MIN, HOST_USER_MAX + 1u, size,
			                                           alignment);
		}
		return FindFreeAlignedUnlocked(search_addr, HOST_USER_MAX + 1u, size, alignment);
	}

	uint64_t FindFreeAligned(uint64_t search_start, uint64_t search_end, uint64_t size,
	                         uint64_t alignment) {
		if (size == 0 || alignment == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment, PageSize());
		std::lock_guard lock(m_mutex);
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ReserveFixed(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	bool TestContainsFree(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}
#endif

	bool ReleaseFree(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::lock_guard lock(m_mutex);
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto* ptr = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
		                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
		                          GetProtect(host_mode), nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				VirtualFree(ptr, 0, MEM_RELEASE);
			}
			AddFreeUnlocked(vaddr, size);
			return false;
		}
		EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Private));
		return true;
#else
		if (VirtualMemory::Protect(vaddr, size, host_mode)) {
			EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Private));
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
#endif
	}

	bool ReleaseCommitted(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Private)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		while (current < end) {
			const auto it       = FindMappedContainingUnlocked(current);
			const auto part_end = std::min(end, it->second.vaddr + it->second.size);
			regions.emplace_back(current, part_end - current);
			current = part_end;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [region_vaddr, region_size]: regions) {
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_vaddr), region_size,
			                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
		}
		// ReleaseCommitted also publishes the aggregate span as one logical free range.
		EXIT_IF(regions.size() > 1 && !CoalescePlaceholders(vaddr, size));
#else
		for (const auto& [region_vaddr, region_size]: regions) {
			RestorePlaceholder(region_vaddr, region_size);
		}
#endif
		EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Private));
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	bool SelfTest() {
		const auto addr = ReserveAligned(0, PageSize(), PageSize());
		if (addr == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: 16 KiB range unavailable\n");
			return false;
		}
		if (!Commit(addr, PageSize(), VirtualMemory::Mode::ReadWrite)) {
			LOGF_COLOR(Log::Color::Red, "\t guest address-space self-test: commit failed\n");
			return false;
		}

		auto* ptr = reinterpret_cast<uint64_t*>(addr);
		*ptr      = 0x4b59545950484f4cull; // "KYTYPHOL"

		if (!ReleaseCommitted(addr, PageSize())) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: release-committed failed\n");
			return false;
		}

		LOGF_COLOR(Log::Color::Green, "\t guest address-space self-test: ok\n");
		return true;
	}

private:
	enum class MappingType { Backing, Private };

	struct MappedRegion {
		uint64_t    vaddr = 0;
		uint64_t    size  = 0;
		MappingType type  = MappingType::Private;
	};

	static constexpr uint64_t PageSize() { return 0x4000; }

	static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
		if (alignment == 0) {
			return 0;
		}
		const auto remainder = value % alignment;
		const auto increment = (remainder != 0 ? alignment - remainder : 0);
		return increment <= UINT64_MAX - value ? value + increment : 0;
	}

	static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
		return value & ~(alignment - 1u);
	}

	[[nodiscard]] static bool IsPageAligned(uint64_t value) {
		return (value & (PageSize() - 1u)) == 0;
	}

	[[nodiscard]] bool IsGranularityAligned(uint64_t value) const {
		return (value & (m_granularity - 1u)) == 0;
	}

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (range_start < range_end &&
			    !VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
	}

	bool ConsumeUnlocked(uint64_t vaddr, uint64_t size) {
		auto it = FindContaining(vaddr, size);
		if (it == m_free.end() || !SplitUnlocked(it, vaddr, size)) {
			return false;
		}
		it = m_free.find(vaddr);
		if (it == m_free.end() || it->second != size) {
			return false;
		}
		m_free.erase(it);
		return true;
	}

	std::map<uint64_t, MappedRegion>::iterator FindMappedContainingUnlocked(uint64_t vaddr) {
		auto next = m_mapped.upper_bound(vaddr);
		if (next == m_mapped.begin()) {
			return m_mapped.end();
		}
		auto it = std::prev(next);
		return vaddr < it->second.vaddr + it->second.size ? it : m_mapped.end();
	}

	bool ContainsMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it = FindMappedContainingUnlocked(current);
			if (it == m_mapped.end() || it->second.type != type) {
				return false;
			}
			current = std::min(end, it->second.vaddr + it->second.size);
		}
		return true;
	}

	[[nodiscard]] bool OverlapsPrivateUnlocked(uint64_t vaddr, uint64_t size) const {
		auto mapping = m_mapped.upper_bound(vaddr);
		if (mapping != m_mapped.begin()) {
			--mapping;
		}
		for (; mapping != m_mapped.end() && mapping->second.vaddr < vaddr + size; ++mapping) {
			const auto& region = mapping->second;
			if (region.type == MappingType::Private && region.vaddr + region.size > vaddr) {
				return true;
			}
		}
		return false;
	}

	bool AddMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		const auto end  = vaddr + size;
		auto       next = m_mapped.lower_bound(vaddr);
		if (next != m_mapped.end() && next->second.vaddr < end) {
			return false;
		}
		if (next != m_mapped.begin()) {
			const auto& previous = std::prev(next)->second;
			if (previous.vaddr + previous.size > vaddr) {
				return false;
			}
		}
		return m_mapped.emplace(vaddr, MappedRegion {vaddr, size, type}).second;
	}

	bool RemoveMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (!ContainsMappedUnlocked(vaddr, size, type)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			auto       it       = FindMappedContainingUnlocked(current);
			const auto original = it->second;
			const auto part_end = std::min(end, original.vaddr + original.size);
			m_mapped.erase(it);
			if (original.vaddr < current) {
				const auto left_size = current - original.vaddr;
				EXIT_IF(!AddMappedUnlocked(original.vaddr, left_size, original.type));
			}
			if (part_end < original.vaddr + original.size) {
				const auto right_size = original.vaddr + original.size - part_end;
				EXIT_IF(!AddMappedUnlocked(part_end, right_size, original.type));
			}
			current = part_end;
		}
		return true;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}
#endif

	void ReserveGuestRegions() {
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, HOST_USER_MAX + 1u},
		    {kExtendedMemoryBase, kExtendedMemoryBase + kExtendedMemorySize},
		}};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		for (const auto& [guest_start, guest_end]: regions) {
			auto next_addr = guest_start;
			while (next_addr < guest_end) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(next_addr), &info, sizeof(info)) == 0) {
					EXIT("failed to inspect guest address space at 0x%016" PRIx64 "\n", next_addr);
				}

				const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end = std::min<uint64_t>(
				    guest_end, region_base + static_cast<uint64_t>(info.RegionSize));
				const auto reserve_start =
				    AlignUp(std::max(region_base, guest_start), m_granularity);
				const auto reserve_end = AlignDown(region_end, m_granularity);

				if (info.State == MEM_FREE && reserve_end > reserve_start &&
				    reserve_end - reserve_start > MinimumRegionSize) {
					auto* ptr = VirtualAlloc2(
					    process, reinterpret_cast<void*>(reserve_start), reserve_end - reserve_start,
					    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
					if (reinterpret_cast<uint64_t>(ptr) != reserve_start) {
						EXIT("failed to reserve guest address space at 0x%016" PRIx64
						     ", size 0x%016" PRIx64 "\n",
						     reserve_start, reserve_end - reserve_start);
					}
					AddFreeUnlocked(reserve_start, reserve_end - reserve_start);
					m_owned.emplace_back(reserve_start, reserve_end - reserve_start);
				}

				if (region_end <= next_addr) {
					EXIT("guest address-space scan did not advance at 0x%016" PRIx64 "\n", next_addr);
				}
				next_addr = region_end;
			}
		}
#else
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
		std::vector<std::pair<uint64_t, uint64_t>> occupied;
		FILE*                                      maps = fopen("/proc/self/maps", "r");
		EXIT_IF(maps == nullptr);

		char line[512];
		while (fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long mapping_start = 0;
			unsigned long long mapping_end   = 0;
			if (sscanf(line, "%llx-%llx", &mapping_start, &mapping_end) == 2) {
				occupied.emplace_back(static_cast<uint64_t>(mapping_start),
				                      static_cast<uint64_t>(mapping_end));
			}
		}
		fclose(maps);

		auto reserve_range = [this](uint64_t start, uint64_t end) {
			start = AlignUp(start, PageSize());
			end   = AlignDown(end, PageSize());
			if (start == 0 || end <= start) {
				return;
			}

			const auto size = end - start;
			int        flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif

			void* ptr = mmap(reinterpret_cast<void*>(start), size, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, size);
				}
				return;
			}

			AddFreeUnlocked(start, size);
			m_owned.emplace_back(start, size);
		};

		for (const auto& [region_start, region_end]: regions) {
			auto current = region_start;

			for (const auto& [mapping_start, mapping_end]: occupied) {
				if (mapping_end <= current) {
					continue;
				}
				if (mapping_start >= region_end) {
					break;
				}

				if (mapping_start > current) {
					reserve_range(current, std::min(mapping_start, region_end));
				}

				current = std::max(current, mapping_end);
				if (current >= region_end) {
					break;
				}
			}

			if (current < region_end) {
				reserve_range(current, region_end);
			}
		}
#else
		for (const auto& [start, end]: regions) {
			int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif
			void* ptr = mmap(reinterpret_cast<void*>(start), end - start, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, end - start);
				}
				EXIT("failed to reserve guest address space at 0x%016" PRIx64
				     ", size 0x%016" PRIx64 "\n",
				     start, end - start);
			}
			AddFreeUnlocked(start, end - start);
			m_owned.emplace_back(start, end - start);
		}
#endif
#endif
	}

	uint64_t FindFreeAlignedUnlocked(uint64_t search_start, uint64_t search_end, uint64_t size,
	                                 uint64_t alignment) {
		if (search_start >= search_end || size > search_end - search_start) {
			return 0;
		}
		for (const auto& [base, range_size]: m_free) {
			const auto candidate = AlignUp(std::max(base, search_start), alignment);
			if (candidate < search_end && size <= search_end - candidate &&
			    GuestFreeRangeContains(base, range_size, candidate, size)) {
				return candidate;
			}
		}
		return 0;
	}

	std::map<uint64_t, uint64_t>::iterator FindContaining(uint64_t vaddr, uint64_t size) {
		auto next = m_free.upper_bound(vaddr);
		if (next == m_free.begin()) {
			return m_free.end();
		}
		auto it  = std::prev(next);
		auto end = it->first + it->second;
		return (vaddr >= it->first && vaddr + size <= end ? it : m_free.end());
	}

	bool SplitUnlocked(std::map<uint64_t, uint64_t>::iterator it, uint64_t vaddr, uint64_t size) {
		const auto base = it->first;
		const auto end  = it->first + it->second;
		if (base == vaddr && end == vaddr + size) {
			return true;
		}

		if (base != vaddr) {
			const auto left_size = vaddr - base;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(base), left_size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           base, end - base, vaddr, size, static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			it->second = left_size;
			it         = m_free.emplace(vaddr, end - vaddr).first;
		}

		if (it->second != size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(it->first), size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           it->first, it->second, vaddr, size,
				           static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			const auto right_start = it->first + size;
			const auto right_size  = it->second - size;
			it->second             = size;
			m_free.emplace(right_start, right_size);
		}

		return true;
	}

	void AddFreeUnlocked(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0) {
			return;
		}
		auto [it, inserted] = m_free.emplace(vaddr, size);
		if (!inserted) {
			it->second = size;
		}

		auto base = it->first;
		auto end  = it->first + it->second;

		auto first = it;
		while (first != m_free.begin()) {
			auto prev = std::prev(first);
			if (prev->first + prev->second != base) {
				break;
			}
			base  = prev->first;
			first = prev;
		}

		auto last = std::next(it);
		while (last != m_free.end() && end == last->first) {
			end  = last->first + last->second;
			last = std::next(last);
		}

		if (base != it->first || end != it->first + it->second) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (CoalescePlaceholders(base, end - base)) {
				m_free.erase(first, last);
				m_free.emplace(base, end - base);
			}
#else
			m_free.erase(first, last);
			m_free.emplace(base, end - base);
#endif
		}
	}

	void RestorePlaceholder(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// The address remains owned for the process lifetime; MAP_FIXED only replaces one of our
		// mappings and never acquires a new guest address.
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, PROT_NONE,
		                 MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		EXIT_IF(ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != vaddr);
#else
		(void)vaddr;
		(void)size;
#endif
	}

	std::shared_mutex                          m_mutex;
	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};
