#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/hostCopyQueue.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// KYTY_DBG_STAGE_STATS: command buffers submitted by Flush() (NHL27EXECIDLE flushes/s).
inline std::atomic<uint64_t> g_scheduler_flushes {0};
// The flushes of: 0 flush callbacks, 1 batch-ready callbacks, 2 asynchronous guest reads, 3 flips.
inline std::array<std::atomic<uint64_t>, 4> g_scheduler_flush_sites {};

struct GraphicContext;

// NHL26 debugging: GPU breadcrumbs (VK_AMD_buffer_marker). BreadcrumbBegin writes the command's
// id at top-of-pipe before it, BreadcrumbEnd writes it at bottom-of-pipe after it; on a device
// loss the ids between the two markers are the commands that never finished.
//
// The same hooks also drive the KYTY_PERF_GPU_TIMING=1 per-draw/dispatch GPU timing profiler
// (timestamp queries, summarised every few seconds as NHL27GPU lines). The returned handle packs the
// breadcrumb id and the timing slots; pass it unchanged to BreadcrumbEnd. While KYTY_IDLE_TRACE is
// capturing they also write per-command trace timestamps. aux = render target (format << 32 | width << 16 | height).
uint64_t BreadcrumbBegin(GraphicContext& graphics, const CommandRecorder& cmd, const char* kind,
                         uint64_t hash_a, uint64_t hash_b, uint64_t info = 0, uint64_t aux = 0);
void     BreadcrumbEnd(GraphicContext& graphics, const CommandRecorder& cmd, uint64_t id);
// With breadcrumbs on, copies up to 16 bytes the GPU is about to consume for breadcrumb `id`
// (an indirect dispatch's group counts) into a host-visible ring that the device-loss dump prints.
void     BreadcrumbCaptureArgs(const CommandRecorder& cmd, uint64_t id, vk::Buffer source,
                               uint64_t offset, uint32_t bytes);
// Reads back finished GPU timing samples without waiting; called after every queue submit.
void     GpuTimingPoll();
// Prints breadcrumbs and VK_EXT_device_fault info; call when any Vulkan call reports device loss.
void     ReportDeviceLost(GraphicContext& graphics);

class CommandScheduler {
public:
	// With deferred_recording, command buffers are recorded into a CommandStream and a thread of
	// the scheduler replays them into Vulkan command buffers and submits them in order.
	CommandScheduler(RenderContext& context, GraphicContext& graphics,
	                 bool deferred_recording = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(const HW::Context& registers, const HW::UserConfig& user_config,
	                     const HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	// Runs the pending operations whose ticks completed. Each waits for the priority operations of
	// its tick or earlier; with `wait_priority` false (draws and dispatches) one that would wait
	// stays queued for a later call instead.
	void                      PopPendingOperations(bool wait_priority = true);
	void                      DrainPriorityOperations();
	// Whether a deferred or priority operation that may write guest memory in
	// [address, address + size) is queued or running.
	[[nodiscard]] bool HasPendingGuestOperations(uint64_t address, uint64_t size);
	void               WaitPriorityOperations(uint64_t tick);
	// Completions that may act on guest memory: an unmap of memory one may write drains the GPU
	// while it is pending. They write at most [address, address + size), by default anywhere.
	void DeferOperation(Common::UniqueFunction<void>&& operation, uint64_t address = 0,
	                    uint64_t size = UINT64_MAX);
	void DeferPriorityOperation(Common::UniqueFunction<void>&& operation, uint64_t address = 0,
	                            uint64_t size = UINT64_MAX);
	// Completions that only act on host state (cache bookkeeping, releases, flips, interrupts).
	void DeferHostOperation(Common::UniqueFunction<void>&& operation, bool priority = false);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	// Returns once the submission that signals `tick` was handed to the queue: with deferred
	// recording the recording thread submits after the tick was allocated. Work on another queue
	// that waits for the tick must not be submitted before (KYTY_READBACK_QUEUE).
	void                           WaitSubmitted(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Copies of guest data into the stream buffer that the execution thread hands off, with
	// deferred recording (null otherwise). Submissions wait for the copies recorded before them;
	// the owner of the scheduler drains them before any guest-visible effect.
	[[nodiscard]] HostCopyQueue* HostCopies() noexcept { return m_host_copies.get(); }

private:
	void QueueOperation(Common::UniqueFunction<void>&& operation);
	void QueuePriorityOperation(Common::UniqueFunction<void>&& operation);
	Common::UniqueFunction<void> TrackGuestOperation(Common::UniqueFunction<void>&& operation,
	                                                 uint64_t address, uint64_t size);

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();
		// A command buffer for the recording thread, busy until Retire() gives it the tick of
		// its submission.
		[[nodiscard]] std::pair<vk::CommandBuffer, size_t> CommitDeferred();
		void                                               Retire(size_t index, uint64_t tick);

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();
		// A command buffer the GPU finished with, marked busy until `busy_until`.
		size_t Acquire(uint64_t busy_until);

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	// A submission recorded into the stream: the recording thread submits it in order.
	struct RecordedSubmit {
		SubmitInfo info;
		uint64_t   tick            = 0;
		// Host copies queued before the submission.
		uint64_t   host_copies     = 0;
		uint64_t   debug_submit_id = 0;
		uint64_t   debug_arg4      = 0;
		uint32_t   debug_op        = 0;
		uint32_t   debug_arg0      = 0;
		uint32_t   debug_arg1      = 0;
		uint32_t   debug_arg2      = 0;
		uint32_t   debug_arg3      = 0;
	};

	void BeginNext();
	void RecordingThread();
	void SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
	                    const RecordedSubmit& recorded);
	void QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick,
	                 const RecordedSubmit& debug);
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	const bool                   m_deferred;
	CommandChunkQueue            m_chunk_queue;
	CommandStream                m_stream;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	// The tick of the oldest pending operation (UINT64_MAX without any), written under
	// m_operation_mutex: draws check it without the lock.
	std::atomic<uint64_t>        m_pending_front_tick {UINT64_MAX};
	// Last tick whose submission reached vkQueueSubmit (submissions are made in tick order).
	std::atomic<uint64_t>        m_submitted_tick {0};
	// Calls since draws last queried the GPU timeline for pending operations.
	std::atomic<uint32_t>        m_pending_skips {0};
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	// Guest memory that queued or running guest operations may write, [begin, end) by id.
	struct GuestWrite {
		uint64_t id    = 0;
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	std::mutex                   m_guest_writes_mutex;
	std::vector<GuestWrite>      m_guest_writes;
	uint64_t                     m_next_guest_write     = 0;
	OperationState               m_operation_state      = OperationState::Open;
	uint32_t                     m_trace_slot           = 0; // KYTY_IDLE_TRACE GPU timeline slot + 1
	std::unique_ptr<HostCopyQueue> m_host_copies;
	// Replays m_stream when recording is deferred; declared last so it stops first.
	std::jthread m_recording_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
