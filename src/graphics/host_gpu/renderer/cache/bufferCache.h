#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/experiments.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <chrono>

#include <map>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class HostCopyQueue;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// The fault buffer holds two bitmaps of CACHING_NUMPAGES bits: pages that device-address
	// accesses found without a cached buffer, then pages that V#-table stores wrote.
	static constexpr uint64_t WRITE_BITMAP_OFFSET = CACHING_NUMPAGES / 8;
	static constexpr uint64_t FAULT_BUFFER_SIZE   = 2 * WRITE_BITMAP_OFFSET;
	// Word index of the fault-parse gate's dirty word, which follows both bitmaps.
	static constexpr uint64_t FAULT_DIRTY_WORD    = FAULT_BUFFER_SIZE / sizeof(uint32_t);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	// Writes a small CPU-visible label into an existing buffer without a page readback. With
	// `deferred`, a label of up to 16 bytes reaches guest memory from the host copy thread, after the
	// copies queued before it (the guest cannot reuse what earlier draws read before they are done),
	// instead of the caller draining that queue first.
	bool TryWriteLabel(uint64_t vaddr, const void* data, uint64_t size, HostCopyQueue* deferred = nullptr);
	// A label the host copy thread writes into guest memory (TryWriteLabel with a queue).
	struct DeferredLabel {
		uint64_t                vaddr = 0;
		uint64_t                size  = 0;
		std::array<uint8_t, 16> bytes {};
	};
	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// See MemoryTracker::MarkCpuDirtyAhead.
	uint64_t MarkCpuDirtyAhead(uint64_t vaddr, uint64_t size) noexcept {
		return m_memory_tracker.MarkCpuDirtyAhead(vaddr, size);
	}
	// Only used to attribute synchronous readbacks in the KYTY_PERF_STATS output.
	enum class ReadCaller : uint8_t { Other, ColorClear, Fault };
	enum class PerfEvent : uint8_t {
		ColorClearShadowHit,
		WaitRegMemBackingRead,
		Dispatch,
		Draw,
		AutoDispatchBarrier,
		GuestBarrier,
		AcquireMem,
		RenderPassBegin,
		Flip,
		BarrierSkipped,
		FalseShareRead,
		Pm4ShadowPacket,
		BufferCreate,
		BufferUpload,
		BufferDelete,
		BufferGcDelete,
		ImageInsert,
		ImageGcFree,
		ImageUpload,
		Count
	};
	// KYTY_GUEST_SYNC (runtime flag, default on): compute dispatches rely on the guest's own partial-flush,
	// cache and ACQUIRE_MEM packets for ordering instead of a full barrier around each dispatch.
	[[nodiscard]] static bool GuestSyncOnly();
	// KYTY_PERF_NO_SYNC_READBACK=1 enables the readback-avoidance paths (default: off).
	[[nodiscard]] static bool PerfNoSyncReadback();
	static void               PerfCount(PerfEvent event);
	// KYTY_PERF_STATS=1: the faulting guest instruction of the calling thread, so CPU-write
	// readbacks can be attributed to the code that caused them (NHL27WRITEBACK).
	static void               NoteFaultRip(uint64_t rip);
	// KYTY_DBG_CP_READS: the shader whose resources the calling thread materializes, named by
	// NHL27RBWAIT for the downloads of GPU-written bytes it waits for.
	static void               NoteReadbackShader(uint64_t hash) noexcept;
	// NHL27WRITEBACK: the emulator path behind a GPU-thread CPU-write readback, which has no guest
	// instruction. Logged in place of the rip (rip=0 is a GPU-thread write nobody tagged).
	enum class WriteTag : uint64_t { None = 0, EopLabel, WriteData, TextureKeys, Unmap, Fill, Count };
	class WriteTagScope {
	public:
		explicit WriteTagScope(WriteTag tag) noexcept;
		~WriteTagScope();
		WriteTagScope(const WriteTagScope&)            = delete;
		WriteTagScope& operator=(const WriteTagScope&) = delete;

	private:
		WriteTag m_saved;
	};
	[[nodiscard]] static uint64_t PerfFlipCount();
	// KYTY_PERF_STATS: device memory use the texture/buffer collectors last saw.
	static void PerfSetMemoryUsage(uint64_t used, uint64_t texture_trigger, uint64_t buffer_trigger);
	static void               PerfSetPm4Op(uint32_t opcode);
	// KYTY_PERF_STATS: host time spent beginning command buffers, submitting them and waiting
	// for the GPU on the GPU thread.
	enum class PerfTimer : uint8_t {
		CommandBegin,
		Submit,
		GpuWait,
		// GPU-thread phases (inclusive): PM4 processing, then within it draw program lookup and
		// resource materialization, render-target resolution, draw execution, whole dispatches
		// and their program lookup.
		Pm4,
		DrawPrograms,
		DrawTargets,
		DrawExecute,
		Dispatch,
		DispatchProgram,
		// Sections of draw execution.
		DrawBindPrepare,
		DrawBindResolve,
		DrawVertex,
		DrawPipeline,
		DrawAttachments,
		DrawCommit,
		DrawRecord,
		Count
	};
	[[nodiscard]] static bool PerfTimersEnabled();
	static void               PerfAddTime(PerfTimer timer, uint64_t ns);
	class PerfTimerScope {
	public:
		explicit PerfTimerScope(PerfTimer timer): m_timer(timer), m_active(PerfTimersEnabled()) {
			if (m_active) m_start = std::chrono::steady_clock::now();
		}
		~PerfTimerScope() { Stop(); }
		KYTY_CLASS_NO_COPY(PerfTimerScope);
		void Stop() {
			if (!m_active) return;
			m_active = false;
			PerfAddTime(m_timer, static_cast<uint64_t>(
			                         std::chrono::duration_cast<std::chrono::nanoseconds>(
			                             std::chrono::steady_clock::now() - m_start)
			                             .count()));
		}

	private:
		PerfTimer                             m_timer;
		bool                                  m_active;
		std::chrono::steady_clock::time_point m_start {};
	};
	// Labels the next GPU writes (KYTY_PERF_STATS fault diagnostics only).
	void SetWriterTag(const char* kind, uint64_t hash) {
		m_writer_kind = kind;
		m_writer_hash = hash;
	}
	void ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false,
	                ReadCaller caller = ReadCaller::Other);
	// Remembers that the GPU filled [vaddr, vaddr + size) with one repeated dword. The record
	// lives only until any other write, download or invalidation touches the range.
	void RecordGpuFill(uint64_t vaddr, uint64_t size, uint32_t value);
	// Experiment FillPatterns: the GPU wrote `words` at the start of every `stride` bytes of
	// [vaddr, vaddr + size) (the bytes between are unknown).
	void RecordGpuFillPattern(uint64_t vaddr, uint64_t size, uint32_t stride,
	                          std::span<const uint32_t> words);
	// Whether one recorded pattern covers [vaddr, vaddr + size), the range holds at least one whole
	// written element, and that element's bytes are not all equal: the range cannot be one repeated
	// byte (a uniform metadata clear).
	[[nodiscard]] bool GpuFillCoversNonUniform(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool TryGetGpuFill(uint64_t vaddr, uint64_t size, uint32_t& value) const;
	// Drops GPU ownership of whole pages without a download. The caller must be about to
	// overwrite the CPU backing of the range and mark it CPU-modified.
	void DiscardGpuModified(uint64_t vaddr, uint64_t size);
	// The bytes were published to guest memory by the CPU with the same values the GPU writes,
	// so they no longer need a readback. Pages keep GPU ownership while other bytes on them do.
	void ForgetGpuWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	// KYTY_BUFFER_FAST: ObtainBuffer for plain reads, reusing the previous (buffer, offset) while the
	// registration generation is unchanged and the range is not CPU-modified. Other requests
	// (written or texel buffers) always go through ObtainBuffer.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferCached(uint64_t vaddr, uint64_t size,
	                                                              bool is_written, bool is_texel_buffer,
	                                                              BufferId id = {});
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// A plain read ObtainBufferCached served from `buffer` when the registration generation was
	// `registration` and the CPU modification generation `cpu_generation`: whether the same buffer
	// and offset still hold (registration unchanged, range not CPU-modified since); touches it and
	// advances `cpu_generation`. The stream buffer never holds. `touched` is the LRU tick the caller's
	// memo last touched the buffer at: a memo that already touched it this tick does not read the
	// buffer object again (a cold line per binding).
	[[nodiscard]] bool PlainReadHolds(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& cpu_generation,
	                                  uint64_t registration, uint64_t& touched);
	[[nodiscard]] bool IsStreamBuffer(const Buffer* buffer) const noexcept { return buffer == &m_stream_buffer; }
	// The same for a texel (formatted) read found while no image started at the address, at image
	// version `image_version` (TextureCache::AddressBucketVersion()).
	[[nodiscard]] bool TexelReadHolds(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& cpu_generation,
	                                  uint64_t registration, uint64_t image_version, uint64_t& touched);
	// Whether a texel read of `vaddr` may be kept as TexelReadHolds() checks it: no image starts there.
	[[nodiscard]] bool     TexelReadKeepable(uint64_t vaddr);
	[[nodiscard]] uint64_t TexelImageVersion(uint64_t vaddr) const;
	// A written binding ObtainBuffer made of [vaddr, vaddr + size) in `buffer`, after which the GPU
	// ranges had version `ranges_version`: while neither they nor the buffer set changed, the range
	// is still GPU-owned with nothing to upload, so binding it again only needs the per-operation
	// bookkeeping of a GPU write, done here.
	[[nodiscard]] bool WrittenBindHolds(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& ranges_version,
	                                    uint64_t registration);
	[[nodiscard]] uint64_t GpuRangesVersion() const noexcept {
		return m_gpu_ranges_version.load(std::memory_order_acquire);
	}
	// Copies guest memory into coherent mapped memory (the stream buffer, staging), possibly on the
	// host copy thread (see CommandScheduler::HostCopies()).
	void StreamGuestData(uint8_t* destination, uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Experiment EarlySubmitWriters (GPU execution thread): a strict resource read had to download
	// these GPU-written bytes; later writes to them ask for an early submit (TakeSubmitRequest).
	void NoteDirtyRead(uint64_t vaddr, uint64_t size);
	// True once after a draw or dispatch wrote bytes a strict resource read downloaded before: the
	// caller ends the host command buffer, so the next read waits for that work only.
	[[nodiscard]] bool TakeSubmitRequest() {
		const bool requested = m_submit_requested;
		m_submit_requested   = false;
		return requested;
	}
	// Any thread: HasGpuDirtyBytes under a shared lock (the GPU thread changes the ranges under an
	// exclusive one). Used by experiment FalseShareAnyThread.
	[[nodiscard]] bool HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size) const;
	// Any thread: how many bytes of the range are GPU-dirty (diagnostics, KYTY_PERF_GPU_FAULT_STACKS).
	[[nodiscard]] uint64_t GpuDirtyByteCount(uint64_t vaddr, uint64_t size) const;
	// Experiment SpeculateDirtyReads (GPU thread): a small read of GPU-dirty bytes whose recent
	// downloads all found the bytes the backing already held is served from the backing without
	// waiting. A copy of the GPU's bytes is compared once the GPU gets there; a mismatch ends the
	// guess for that address until its downloads are steady again.
	[[nodiscard]] bool TrySpeculativeRead(uint64_t vaddr, void* data, uint64_t size);
	// True for bytes strict guessing refuses (seen changing twice or guessed wrong): their reads
	// always download. KYTY_SRT_UNION_DIRTY skips only these.
	[[nodiscard]] bool IsStrictRefused(uint64_t vaddr, uint64_t size);
	// After a real download for such a read: whether the GPU changed the bytes, and the wait.
	void NoteDirtyRead(uint64_t vaddr, uint64_t size, bool changed, const void* value, uint64_t wait_ns);
	// Changes whenever new GPU-dirty bytes are recorded (GPU thread).
	[[nodiscard]] uint64_t GpuDirtyGeneration() const noexcept { return m_gpu_dirty_generation; }
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               PrepareFaultBuffer() { m_fault_manager.PrepareFaultBuffer(); }
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;

	void               ProcessFaultBuffer();
	void ProcessWriteBuffer(FaultManager::PagesHandler&& handler) {
		m_fault_manager.ProcessWriteBuffer(std::move(handler));
	}
	// Records stores that a shader made through device addresses, known once it completed: the
	// bytes of [vaddr, vaddr + size) that a cached buffer holds become GPU modified, except on
	// pages the CPU dirtied since, whose upload replaces them.
	void NoteGpuWrites(uint64_t vaddr, uint64_t size);
	// Calls func(address, size) for the runs of [vaddr, vaddr + size) the CPU may have written
	// since the previous call over them (see MemoryTracker::ConsumeCpuWrites).
	template <typename Func>
	void ConsumeCpuWrites(uint64_t vaddr, uint64_t size, Func&& func) {
		m_memory_tracker.ConsumeCpuWrites(vaddr, size, std::forward<Func>(func));
	}
	// Calls func(address, size) for the ranges whose buffers the garbage collector released since
	// the previous call, and forgets them; the first call starts recording them. Merging buffers
	// keeps every address cached.
	template <typename Func>
	void ConsumeReleasedRanges(Func&& func) {
		m_released_ranges.ForEach(func);
		m_released_ranges.Clear();
		m_record_released = true;
	}
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	uint64_t           SynchronizeBuffersInRanges(const RangeSet& ranges, bool full);
	// Synchronizes only the pages dirtied before write epoch `limit` (see RegionManager); returns
	// whether later ones were left (first_left: the earliest of their epochs). The queued dirty
	// regions are synchronized within `ranges`, and all of `full_ranges` (if set) is scanned.
	bool               SynchronizeBuffersInRangesBefore(const RangeSet& ranges, const RangeSet* full_ranges,
	                                                    uint32_t limit, uint32_t& first_left);
	[[nodiscard]] uint64_t CpuModificationGeneration() const noexcept {
		return m_memory_tracker.CpuModificationGeneration();
	}
	[[nodiscard]] uint64_t RegistrationGeneration() const noexcept { return m_registration_generation; }
	// Uploads the pending CPU writes of the buffers within `ranges`. Unless `all`, visits only the
	// tracking regions whose pages became CPU dirty since the previous call: that call left no
	// pending write under a buffer anywhere else in `ranges`, provided no buffer was registered
	// and `ranges` did not change since. Callers pass `all` otherwise.
	void SynchronizeDirtyRegionsInRanges(const RangeSet& ranges, bool all);
	// Advances whenever a buffer is registered: a new buffer may cover CPU-dirty pages that no
	// earlier synchronization uploaded.
	[[nodiscard]] uint64_t RegisterEpoch() const { return m_registration_generation; }
	// Advances whenever a buffer is registered or unregistered.
	[[nodiscard]] uint64_t BufferSetEpoch() const { return m_buffer_set_epoch; }
	// Advances whenever recorded work may write cached buffers (written bindings, fills, copies,
	// device-address stores).
	[[nodiscard]] uint64_t GpuWriteGeneration() const noexcept { return m_gpu_write_generation; }
	// Advances whenever a page of the range becomes CPU dirty (see
	// MemoryTracker::RegionCpuDirtyEpoch); pages that stay CPU dirty do not advance it.
	[[nodiscard]] uint64_t CpuDirtyEpoch(uint64_t vaddr, uint64_t size) const {
		uint64_t epoch = 0;
		for (auto index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < vaddr + size;
		     index++) {
			// Per 64 KiB block where the range covers part of a region (the region's epoch moves
			// whenever the game writes anywhere in its 4 MiB: the read memo missed almost always).
			const bool whole = vaddr <= index * TRACKER_REGION_SIZE &&
			                   vaddr + size >= (index + 1) * TRACKER_REGION_SIZE;
			epoch += whole ? m_memory_tracker.RegionCpuDirtyEpoch(index)
			             : m_memory_tracker.BlockCpuDirtyEpochs(index, vaddr, size);
		}
		return epoch;
	}
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void DropGpuFills(uint64_t vaddr, uint64_t size);
	void LogGpuWrite(uint64_t vaddr, uint64_t size);
	void PerfMaybePrint();
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	// merge (async only): the publish keeps bytes the CPU changed since the download was queued.
	template <bool async, bool merge = false>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool LazySharedWriteEligible(uint64_t vaddr, uint64_t size);
	// Experiment AsyncGuestReads (GPU thread): waits for the asynchronous guest reads overlapping
	// the range and releases their pages that hold no newer GPU-written bytes.
	void FinishPendingGuestReads(uint64_t vaddr, uint64_t size);
	// m_pending_read_ranges from m_pending_guest_reads (under m_gpu_ranges_mutex).
	void RebuildPendingReadRanges();
	[[nodiscard]] bool OverlapsPendingGuestRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool OverlapsPendingGuestBytes(uint64_t vaddr, uint64_t size) const;
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);


	// Experiment EarlySubmitWriters: GPU-written ranges strict resource reads downloaded (oldest
	// replaced first), and whether a write to one of them asks for a submit.
	std::vector<std::pair<uint64_t, uint64_t>>        m_watched_writes;
	size_t                                            m_watched_next     = 0;
	bool                                              m_submit_requested = false;
	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// Guards m_gpu_modified_ranges and m_pending_read_ranges against readers on other threads
	// (HasGpuDirtyBytesShared).
	mutable std::shared_mutex                         m_gpu_ranges_mutex;
	// The pages of pending guest reads (their bytes reach guest memory when they finish), from
	// before their bytes leave m_gpu_modified_ranges until they are in guest memory.
	RangeSet                                          m_pending_read_ranges;
	// The versions of both sets added (under m_gpu_ranges_mutex, after each change):
	// HasGpuDirtyBytesShared answers a page again from its thread's cache while it holds.
	std::atomic<uint64_t> m_gpu_ranges_version {0};
	void                  PublishGpuRangesVersion() {
		m_gpu_ranges_version.store(m_gpu_modified_ranges.Version() + m_pending_read_ranges.Version(),
		                           std::memory_order_release);
	}
	uint64_t                                          m_gpu_dirty_generation = 0;
	RangeSet                                          m_released_ranges;
	bool                                              m_record_released = false;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;

	MemoryTracker                                     m_memory_tracker;
	uint64_t                                           m_gpu_write_generation = 0;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	// KYTY_READBACK_QUEUE: copies downloads on GraphicContext::readback_queue.
	vk::CommandPool                                   m_readback_pool[3] {};
	vk::CommandBuffer                                 m_readback_buffer[3] {};
	vk::Fence                                         m_readback_fence[3] {};
	bool ReadbackOnSecondQueue(const Buffer& source, const Buffer& destination,
	                           std::span<const vk::BufferCopy> copies, uint64_t wait_tick);
	// Experiment AsyncReadback (KYTY_ASYNC_READBACK, measured slower, not wired since the Oct 18 sync with
	// nhl27-eadb): the bookkeeping of downloads waited for by the requesting thread.
	struct AsyncReadback;
	struct AsyncSlot {
		vk::CommandBuffer command = nullptr;
		vk::Fence         fence   = nullptr;
	};
	void NoteAsyncReadbackWrite(uint64_t vaddr, uint64_t size);
	void WaitAsyncReadbacksOf(BufferId id);
	std::mutex                                        m_async_mutex;
	std::vector<AsyncReadback*>                       m_async_readbacks;
	std::atomic<uint32_t>                             m_async_count {0};
	vk::CommandPool                                   m_async_pool = nullptr;
	std::vector<AsyncSlot>                            m_async_free;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	struct GpuFill {
		uint64_t                begin;
		uint64_t                end;
		uint32_t                value;
		uint32_t                words  = 1; // pattern: the written words of each element
		uint32_t                stride = 0; // pattern: element stride (0 = uniform dword fill)
		bool                    non_uniform = false; // pattern bytes are not all equal
		std::array<uint32_t, 4> pattern {};
	};
	// The GPU thread records fills while guest write faults drop them, so the list is locked;
	// m_gpu_fill_count lets the common no-fill case skip the lock.
	std::vector<GpuFill>                              m_gpu_fills;
	// Experiment AsyncGuestReads: downloads queued for a guest thread's read fault. Until the
	// guest thread finishes one, its pages stay protected but m_gpu_modified_ranges no longer
	// lists their bytes, so HasGpuDirtyBytes reports them and readers finish the read first.
	// `bytes` are the GPU-written ranges the download carries: other bytes of the pages are
	// current in guest memory already (HasGpuDirtyBytes looks at these).
	struct PendingGuestRead {
		uint64_t                                     vaddr;
		uint64_t                                     size;
		uint64_t                                     tick;
		std::vector<std::pair<uint64_t, uint64_t>> bytes;
	};
	std::vector<PendingGuestRead>                     m_pending_guest_reads;
	std::atomic<size_t>                               m_pending_guest_read_count {0};
	// One bit per (64 KiB chunk % 256) that a recorded fill touches (a superset). DropGpuFills runs
	// for every written buffer binding and nearly always misses; the filter skips the fill scan.
	std::array<uint64_t, 4>                           m_gpu_fill_filter {};
	mutable std::mutex                                m_gpu_fills_mutex;
	std::atomic<size_t>                               m_gpu_fill_count {0};
	struct WriterRecord {
		uint64_t    begin;
		uint64_t    end;
		uint64_t    hash;
		const char* kind;
	};
	std::vector<WriterRecord>                         m_writer_log;
	size_t                                            m_writer_head       = 0;
	const char*                                       m_writer_kind       = "storage";
	uint64_t                                          m_writer_hash       = 0;
	uint64_t                                          m_last_download_bytes = 0;
	uint64_t                                          m_total_used_memory  = 0;
	// Advances whenever a buffer is registered or unregistered.
	uint64_t m_buffer_set_epoch = 0;
	// The buffer a read-only binding of a range found, while no buffer was registered or
	// unregistered and no page of the range became CPU dirty since (see ObtainBuffer()).
	struct ObtainMemo {
		uint64_t vaddr        = 0;
		uint64_t size         = 0;
		uint64_t buffer_epoch = 0;
		uint64_t cpu_epoch    = 0;
		uint64_t offset       = 0;
		BufferId id;
		bool     texel = false; // a formatted read, while no image started at the address
		// TextureCache::AddressIndexVersion() when no image started at the address (texel).
		uint64_t image_version = 0;
	};
	// The stream copy a range got within the current draw window (see ObtainBuffer()).
	struct StreamMemo {
		uint64_t vaddr            = 0;
		uint64_t size             = 0;
		uint64_t window           = UINT64_MAX;
		uint64_t lap              = 0;
		uint64_t write_generation = 0;
		uint64_t offset           = 0;
	};
	static constexpr size_t                                  StreamMemoSlots = 1024;
	std::unique_ptr<std::array<StreamMemo, StreamMemoSlots>> m_stream_memo =
	    std::make_unique<std::array<StreamMemo, StreamMemoSlots>>();
	static constexpr size_t                                  ObtainMemoSlots = 4096;
	std::unique_ptr<std::array<ObtainMemo, ObtainMemoSlots>> m_obtain_memo =
	    std::make_unique<std::array<ObtainMemo, ObtainMemoSlots>>();
	static constexpr size_t                    DeferredLabelSlots = 4096;
	std::unique_ptr<DeferredLabel[]>           m_deferred_labels = std::make_unique<DeferredLabel[]>(DeferredLabelSlots);
	uint64_t                                   m_deferred_labels_queued = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint64_t m_registration_generation = 1;

	struct ObtainCacheEntry {
		uint64_t   vaddr          = 0;
		uint64_t   size           = 0;
		uint64_t   registration   = 0;
		uint64_t   cpu_generation = 0;
		BufferId   id {};
		uint64_t   offset         = 0;
	};
	static constexpr size_t kObtainCacheSize = 2048;
	std::array<ObtainCacheEntry, kObtainCacheSize> m_obtain_cache {};
	// CPU-dirty epoch of each tracking region as SynchronizeBuffersInRanges last visited it.
	std::unordered_map<uint64_t, uint64_t>     m_synced_region_epochs;
	std::vector<std::pair<uint64_t, uint64_t>> m_sync_regions;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
