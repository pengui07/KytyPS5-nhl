#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/indirectWriteTables.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <algorithm>
#include <array>
#include <memory>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	ImageHeap&          GetImageHeap() { return m_image_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	IndirectWriteTables& GetIndirectWriteTables() { return m_indirect_write_tables; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// Before the emulated GPU writes guest memory (labels, WRITE_DATA): resolves a tracked page
	// as its write fault would, without raising the exception.
	void PrepareGpuWrite(const void* destination, uint64_t size) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// Unmaps are numbered: whether [vaddr, vaddr + size) was unmapped after UnmapSequence()
	// returned `sequence` (false once the log of the last unmaps no longer reaches back that far).
	[[nodiscard]] uint64_t UnmapSequence() const noexcept;
	[[nodiscard]] bool     UnmappedSince(uint64_t sequence, uint64_t vaddr, uint64_t size) const noexcept;
	// Calls write(address, bytes) for the parts of [vaddr, vaddr + size) not unmapped since
	// UnmapSequence() returned `sequence`, holding off unmaps meanwhile: a completion queued before
	// an unmap (a download) must not write memory the guest gave up, or mapped again since. Writes
	// nothing once the log no longer reaches back that far.
	template <typename Write>
	void WriteUnlessUnmapped(uint64_t sequence, uint64_t vaddr, uint64_t size, Write&& write) const {
		std::shared_lock lock(m_unmap_log_mutex);
		if (sequence == m_unmap_sequence) {
			write(vaddr, size);
			return;
		}
		const auto first = m_unmap_sequence > m_unmap_log.size() ? m_unmap_sequence - m_unmap_log.size() : 0;
		if (sequence < first) {
			return;
		}
		const auto end = vaddr + size;
		std::vector<std::pair<uint64_t, uint64_t>> holes;
		for (auto s = sequence; s < m_unmap_sequence; s++) {
			const auto& record = m_unmap_log[s % m_unmap_log.size()];
			if (record.begin < end && vaddr < record.end) {
				holes.emplace_back(std::max(record.begin, vaddr), std::min(record.end, end));
			}
		}
		std::sort(holes.begin(), holes.end());
		auto at = vaddr;
		for (const auto& [begin, hole_end]: holes) {
			if (begin > at) {
				write(at, begin - at);
			}
			at = std::max(at, hole_end);
		}
		if (at < end) {
			write(at, end - at);
		}
	}
	// Whether a read of the range would fault on a page holding GPU-written bytes. Any thread.
	[[nodiscard]] bool IsReadProtected(uint64_t vaddr, uint64_t size) const noexcept;
	// Advances whenever a range is mapped or unmapped.
	[[nodiscard]] uint64_t MappedRangesVersion() const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               CacheDmaBases(const ShaderStageRuntime& runtime);
	void               PrepareBda();
	// KYTY_BDA_BATCH: a point where guest memory can become ordered for the GPU or CPU. The next PrepareBda
	// syncs again. GPU command thread only.
	enum class BdaBoundary : uint8_t { Submit, Wait, Event, Acquire, Dma, Sync, Ib, Flip, Other, Count };
	void               BdaBatchBoundary(BdaBoundary kind);
	// The write epoch of the operation being executed (BDA synchronization uploads the guest
	// writes of earlier epochs).
	void               SetOperationWriteEpoch(uint32_t epoch) {
		m_operation_write_epoch = epoch;
		m_operation_epoch_valid = true;
	}
	struct BdaPreparationStats {
		uint64_t scans = 0; uint64_t reused = 0;
		uint64_t full_scans = 0; uint64_t dirty_regions = 0;
	};
	// Access only on the serialized GPU command lane.
	[[nodiscard]] BdaPreparationStats GetBdaPreparationStats() const noexcept { return m_bda_stats; }
	void               RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	IndirectWriteTables       m_indirect_write_tables {*this};
	ImageHeap                 m_image_heap;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	// The parts of m_mapped_ranges mapped since the last BDA synchronization (it clears them under
	// the shared lock: it is their only reader, and mapping changes take the lock exclusively).
	RangeSet                  m_bda_new_ranges;
	struct UnmapRecord {
		uint64_t sequence = 0;
		uint64_t begin    = 0;
		uint64_t end      = 0;
	};
	// Its own lock: deferred downloads hold it while they write guest memory.
	mutable std::shared_mutex     m_unmap_log_mutex;
	std::array<UnmapRecord, 8192> m_unmap_log {};
	uint64_t                      m_unmap_sequence = 0;
	uint64_t                  m_mapped_ranges_version = 0;
	// Epochs (CPU dirty, buffer registrations, mapped ranges) read before the last full BDA
	// synchronization.
	std::array<uint64_t, 3> m_bda_sync_epochs {UINT64_MAX, UINT64_MAX, UINT64_MAX};
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_bda_logged = false;
	uint64_t                  m_mapping_generation = 1;

public:
	// Advances whenever guest memory is mapped or unmapped (execution thread).
	[[nodiscard]] uint64_t MappingGeneration() const { return m_mapping_generation; }

private:
	uint64_t                  m_bda_cpu_generation = 0;
	uint64_t                  m_bda_registration_generation = 0;
	uint64_t                  m_bda_mapping_generation = 0;
	bool                      m_bda_batch_synced = false;  // KYTY_BDA_BATCH: this batch already synchronized
	// Write epochs (RegionManager): the operation being executed, and whether the last BDA
	// synchronization left pages of later epochs (and the earliest of those).
	uint32_t                  m_operation_write_epoch = 0;
	bool                      m_operation_epoch_valid = false;
	bool                      m_bda_pages_left        = false;
	uint32_t                  m_bda_first_left        = 0;
	BdaPreparationStats       m_bda_stats;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
