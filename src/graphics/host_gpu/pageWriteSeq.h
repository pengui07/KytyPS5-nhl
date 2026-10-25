#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEWRITESEQ_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEWRITESEQ_H_

#include <atomic>
#include <cstdint>
#include <memory>

// Per guest page (4 KiB), the sequence of the last GPU operation that writes it from the emulator's
// side: writes the command processor notes for an operation (labels, DMA, clocks), and what the
// execution thread does while executing one (marking buffer ranges GPU-written, writing GPU data
// back into guest memory). The resolve thread records the sequence the execution thread had
// finished when it read guest memory ahead; a read on a page no later operation writes still holds
// when its operation executes (see GuestReadLog::StillValid).
namespace Libs::Graphics::PageWriteSeq {

namespace Detail {
constexpr uint32_t AddressBits = 44;

// A two-level table of sequences for units of 2^UnitBits bytes.
template <uint32_t UnitBits>
struct Table {
	static constexpr uint32_t LeafBits = 16;
	static constexpr uint32_t TopBits  = AddressBits - UnitBits - LeafBits;
	static constexpr uint64_t LeafMask = (uint64_t {1} << LeafBits) - 1;
	std::atomic<std::atomic<uint64_t>*> top[size_t {1} << TopBits] {};

	std::atomic<uint64_t>* Leaf(uint64_t unit, bool create) {
		auto& slot = top[(unit >> LeafBits) & ((uint64_t {1} << TopBits) - 1)];
		auto* leaf = slot.load(std::memory_order_acquire);
		if (leaf != nullptr || !create) {
			return leaf;
		}
		auto* fresh = new std::atomic<uint64_t>[size_t {1} << LeafBits] {};
		if (slot.compare_exchange_strong(leaf, fresh, std::memory_order_acq_rel)) {
			return fresh;
		}
		delete[] fresh;
		return leaf;
	}
	void Raise(uint64_t first, uint64_t last, uint64_t sequence) {
		for (uint64_t unit = first; unit <= last; unit++) {
			auto& cell = Leaf(unit, true)[unit & LeafMask];
			auto  seen = cell.load(std::memory_order_relaxed);
			while (seen < sequence && !cell.compare_exchange_weak(seen, sequence, std::memory_order_release)) {
			}
		}
	}
	uint64_t Last(uint64_t first, uint64_t last) {
		uint64_t result = 0;
		for (uint64_t unit = first; unit <= last; unit++) {
			const auto* leaf = Leaf(unit, false);
			if (leaf != nullptr) {
				const auto value = leaf[unit & LeafMask].load(std::memory_order_acquire);
				result           = value > result ? value : result;
			}
		}
		return result;
	}
};

constexpr uint32_t PageBits  = 12;
constexpr uint32_t ChunkBits = 20;
// Small writes per page; writes of more than 16 pages (GPU-written buffers, every bind) per 1 MiB
// chunk, which costs one entry per MiB instead of one per page.
inline Table<PageBits>  g_pages;
inline Table<ChunkBits> g_chunks;
} // namespace Detail

// The operation the calling thread executes (the execution thread sets it per operation).
inline thread_local uint64_t t_operation = 0;
inline std::atomic<uint64_t> g_operation {0};

inline void SetOperation(uint64_t sequence) {
	t_operation = sequence;
	// Stored before the operation's completion is published (GuestGpu's executed counter).
	g_operation.store(sequence, std::memory_order_release);
}

[[nodiscard]] inline uint64_t CurrentOperation() {
	// Writes of other threads (labels the copy thread writes, download publishes) belong after the
	// operation the execution thread runs now: a resolve that read the bytes before them had seen at
	// most the completion of an operation before that one. (Raising them to "always check" kept
	// every read ahead on their pages from ever being skipped: all of a dispatch's reads and five of
	// a draw's 35.)
	return t_operation != 0 ? t_operation : g_operation.load(std::memory_order_acquire) + 1;
}

// Records that operation `sequence` writes [vaddr, vaddr + size).
inline void Raise(uint64_t vaddr, uint64_t size, uint64_t sequence) {
	if (size == 0 || vaddr >= (uint64_t {1} << Detail::AddressBits)) {
		return;
	}
	const uint64_t end = vaddr + size - 1;
	if ((end >> Detail::PageBits) - (vaddr >> Detail::PageBits) < 16) {
		Detail::g_pages.Raise(vaddr >> Detail::PageBits, end >> Detail::PageBits, sequence);
	} else {
		Detail::g_chunks.Raise(vaddr >> Detail::ChunkBits, end >> Detail::ChunkBits, sequence);
	}
}

inline void RaiseCurrent(uint64_t vaddr, uint64_t size) {
	Raise(vaddr, size, CurrentOperation());
}

// Guest memory changed in a way no page records (mapping changes): reads resolved before the
// operations after the current one are checked in full.
inline std::atomic<uint64_t> g_floor {0};
inline void ForceChecks() {
	const auto floor = g_operation.load(std::memory_order_relaxed) + 1;
	auto       seen  = g_floor.load(std::memory_order_relaxed);
	while (seen < floor && !g_floor.compare_exchange_weak(seen, floor, std::memory_order_release)) {
	}
}

// The last operation recorded writing any page of [vaddr, vaddr + size).
[[nodiscard]] inline uint64_t Last(uint64_t vaddr, uint64_t size) {
	if (size == 0 || vaddr >= (uint64_t {1} << Detail::AddressBits)) {
		return ~uint64_t {0};
	}
	const uint64_t end    = vaddr + size - 1;
	const uint64_t pages  = Detail::g_pages.Last(vaddr >> Detail::PageBits, end >> Detail::PageBits);
	const uint64_t chunks = Detail::g_chunks.Last(vaddr >> Detail::ChunkBits, end >> Detail::ChunkBits);
	return pages > chunks ? pages : chunks;
}

} // namespace Libs::Graphics::PageWriteSeq

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEWRITESEQ_H_
