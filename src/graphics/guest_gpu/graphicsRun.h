#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/spscQueue.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class RenderContext;

// Bounded performance diagnostics only; does not resolve or change fault handling.
void NotePm4MemoryFault(uint64_t fault_vaddr, uint32_t access) noexcept;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	// Runs a host command on the execution thread before the operations queued so far, and
	// waits for it. Only for work that no queued operation depends on (memory the guest unmaps
	// after the operations that used it executed).
	void SendUrgentCommandSync(Common::UniqueFunction<void>&& command);
	// Retries blocked submissions now instead of after the 100 ms stall timeout.
	void               Wake();


	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> draw_commands,
	            std::span<const uint32_t> constant_commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	// The thread that owns the renderer: the execution thread of a pipelined GPU.
	[[nodiscard]] static bool IsGpuThread() noexcept;
	// The pipelined command processor's thread (it parses PM4; the execution thread owns caches).
	[[nodiscard]] static bool IsCommandProcessorThread() noexcept;
	// The guest queue the GPU thread is processing (0 = graphics ring, 1.. = compute rings).
	[[nodiscard]] static uint32_t CurrentQueue() noexcept;
	// Execution thread, while it waits inside an operation (a shader compile worker): runs the
	// urgent commands other threads queued, such as the readback of a page a compile worker
	// faulted on, which would otherwise wait for this thread forever. No-op elsewhere.
	static void ServiceUrgentCommands();
	// Execution thread: the operation it is executing (GpuOperationKind | GpuOperation::trace_op << 8),
	// or 0 between operations. Diagnostics only (KYTY_PERF_GPU_FAULT_STACKS).
	[[nodiscard]] static uint64_t CurrentExecOp() noexcept;

	// A pipelined GPU parses command buffers on its command processor thread and executes the
	// operations they produce, in order, on its execution thread.
	[[nodiscard]] bool Pipelined() const noexcept { return m_pipelined; }
	// Command processor thread.
	[[nodiscard]] uint64_t NextSequence() const noexcept { return m_emitted + 1; }
	// Returns the sequence of the operation.
	uint64_t               EmitOperation(GpuOperation&& operation);
	// What a pending write stores when its value is not known at emit time (NHL27DECISION).
	static constexpr uint8_t WriteOriginOther = 0;
	static constexpr uint8_t WriteOriginClock = 1; // the 64-bit reference clock, read when it executes
	static constexpr uint8_t WriteOriginDma   = 2;
	// The guest memory an emitted operation writes, and the value it writes when that is known
	// when the operation is emitted (`value` null otherwise, or larger than 8 bytes).
	// `kind` names the writer for KYTY_DBG_CP_READS (a static string).
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value,
	                    const char* kind = "write", uint8_t origin = WriteOriginOther);
	// Experiment ClockWaitPredict: when the newest pending write of exactly [address, address + size)
	// stores the 64-bit reference clock, returns true with a value the write will not be below (the
	// clock read now, while the write is still pending). The clock only grows.
	[[nodiscard]] bool TryPredictClockWrite(const volatile void* address, size_t size, uint64_t& lower_bound);
	// Reads guest memory the command processor decides on as a serial GPU would at this point
	// of the stream: from the value of a pending write when it is known, otherwise from memory
	// once the operations that write it executed.
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size);
	// Command processor thread, experiment CpBackingReads: reads guest memory on a page whose reads
	// fault (the GPU wrote other bytes of it) through the execution thread, in order after the
	// operations emitted so far, without downloading the page unless the bytes read are
	// GPU-written themselves. False when the range is not read-protected or the backing cannot
	// serve it: the caller then reads it directly.
	[[nodiscard]] bool TryReadProtectedForCp(uint64_t address, void* value, size_t size);
	// Command processor thread: whether CpBackingReads is on (and no check found it wrong).
	[[nodiscard]] bool CpBackingReadsOn() const;
	// Command processor thread: `cp` used `size` bytes at `address` read from the backing of a
	// read-protected page; the execution thread checks in order that no earlier operation wrote
	// them (when one did, it compares them with what it wrote; a difference turns the reads off).
	void CheckCpBackingRead(CommandProcessor& cp, uint64_t address, const void* bytes, size_t size);
	[[nodiscard]] size_t PendingWriteCount() const { return m_pending_count; }
	// Execution thread: an end-of-pipe clock write ran with this value. Command processor: takes the
	// newest such value covering the range (once).
	void NoteExecutedWrite(uint64_t address, const void* value, uint64_t size);
	bool TakeExecutedWrite(uint64_t address, void* value, uint64_t size);
	// True when the newest pending write over [address, address + size) is an end-of-pipe clock
	// write of the current guest queue covering it (its value is the reference clock when it
	// executes).
	[[nodiscard]] bool PendingClockWrite(uint64_t address, uint64_t size);
	void                   WaitForExecution(uint64_t sequence) { m_executed.WaitFor(sequence); }
	// Waits until every operation emitted so far executed.
	void WaitForExecution() { m_executed.WaitFor(m_emitted); }
	[[nodiscard]] const GpuRegisterState& NeutralState() const noexcept { return m_neutral_state; }

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		uint64_t                  flip_request_id   = 0;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	void              ExecutionThread();
	void              ResolveThread();
	void              DispatchResolveThread();
	// Experiment DispatchResolver (resolve thread): hands the dispatch to the dispatch resolver.
	void              QueueDispatchResolution(GpuOperation& operation, bool indirect);
	// Experiment DrawResolver: the second draw resolver, and the resolve thread handing it a draw.
	void              DrawResolveThread(uint32_t which);
	void              QueueDrawResolution(GpuOperation& operation, uint32_t which);
	static uint32_t   DrawResolverCount();
	// The experiment is on, or the dispatch resolver still has queued dispatches.
	[[nodiscard]] bool DispatchResolverActive() const {
		return Common::Experiments::Enabled(Common::Experiments::DispatchResolver) ||
		       m_dispatch_jobs_done.load(std::memory_order_acquire) != m_dispatch_jobs_queued;
	}
	void              StopExecution();
	// Command processor thread of a pipelined GPU: runs a command on the execution thread and
	// waits for it.
	void              ExecuteSync(Common::UniqueFunction<void>&& command);
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_wake_pending      = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;

	// Pipelined execution. m_emitted is only used by the command processor thread.
	struct ExecutedWrite {
		uint64_t address  = 0;
		uint64_t size     = 0;
		uint8_t  value[8] = {};
	};
	std::mutex                     m_executed_writes_mutex;
	std::array<ExecutedWrite, 16> m_executed_writes {};
	size_t                         m_executed_writes_next = 0;

	struct GuestWrite {
		uint64_t sequence = 0;
		uint64_t address  = 0;
		uint64_t size     = 0;
		bool        known    = false;
		uint8_t     value[8] = {};
		const char* kind     = nullptr;
		uint32_t    queue    = 0; // the guest queue that wrote it
		uint8_t     origin   = 0;
	};

	const bool                      m_pipelined;
	// With a resolve stage, the command processor queues operations for the resolve thread,
	// which forwards them to the execution thread.
	const bool                      m_resolving;
	Common::SpscQueue<GpuOperation> m_operations;
	Common::SpscQueue<GpuOperation> m_resolved;
	Common::ProgressCounter         m_executed;
	uint64_t                        m_emitted = 0;
	// Command processor thread: a ring of the writes of operations that may not have executed
	// yet, oldest first. It holds more writes than operations can be queued.
	static constexpr size_t PendingWriteCapacity = 16384;
	std::unique_ptr<GuestWrite[]> m_pending_writes;
	size_t                        m_pending_first = 0;
	size_t                        m_pending_count = 0;
	void                          ForgetExecutedWrites();
	// Commands that run before the next operation the execution thread takes.
	void                                      RunUrgentCommands();
	std::mutex                                m_urgent_mutex;
	std::vector<Common::UniqueFunction<void>> m_urgent_commands;
	std::atomic_uint32_t                      m_urgent_pending {0};
	// Game and host threads bump it when they queue work, under m_queue_mutex; the command
	// processor thread spins on it before it sleeps.
	std::atomic_uint64_t m_work_epoch {0};
	HW::Context                     m_neutral_context;
	HW::UserConfig                  m_neutral_user_config;
	HW::Shader                      m_neutral_shaders;
	GpuRegisterState                m_neutral_state;
	std::jthread                    m_executor;
	std::jthread                    m_resolver;
	// Experiment DispatchResolver: dispatches the resolve thread passes on to a second one.
	struct DispatchJob {
		GpuDispatchSlot*   slot     = nullptr;
		const HW::Context* context  = nullptr;
		const HW::Shader*  shaders  = nullptr;
		uint32_t           x        = 0;
		uint32_t           y        = 0;
		uint32_t           z        = 0;
		uint32_t           mode     = 0;
		bool               async    = false;
		bool               indirect = false;
	};
	static constexpr size_t           DispatchSlotCount = 1024;
	std::unique_ptr<GpuDispatchSlot[]> m_dispatch_slots;
	size_t                            m_dispatch_slot_next = 0;
	Common::SpscQueue<DispatchJob>    m_dispatch_jobs;
	// Queued (resolve thread) and resolved (dispatch resolver) jobs: the resolve thread resolves
	// dispatches itself again only once the dispatch resolver is idle (they share a ring).
	uint64_t                          m_dispatch_jobs_queued = 0;
	std::atomic<uint64_t>             m_dispatch_jobs_done {0};
	std::jthread                      m_dispatch_resolver;
	// Experiment DrawResolver: draws the resolve thread passes on to a second draw resolver (by
	// their shader addresses: the draws of a program pair stay on one thread).
	struct DrawJob {
		GpuDrawSlot*          slot        = nullptr;
		const HW::Context*    context     = nullptr;
		const HW::UserConfig* user_config = nullptr;
		const HW::Shader*     shaders     = nullptr;
	};
	static constexpr size_t        DrawSlotCount = 1024;
	// Per draw resolver: [0] the second (experiment DrawResolver), [1] a third (DrawResolverCount).
	struct DrawResolver {
		std::unique_ptr<GpuDrawSlot[]> slots;
		size_t                         slot_next = 0;
		Common::SpscQueue<DrawJob>     jobs {DrawSlotCount};
		std::jthread                   thread;
	};
	DrawResolver                   m_draw_resolvers[2];

	std::jthread m_thread;

	friend class CommandProcessor;
};

// The kind of pending guest writes of end-of-pipe clock values (GuestGpu::PendingClockWrite).
extern const char kEopClockWrite[];

// KYTY_DBG_CP_READS=1: guest memory reads of the command processor (indirect register pairs,
// WAIT_REG_MEM values), timed so the ones that fault and wait for the GPU show up. Printed as
// NHL27CPREAD every 2 s, by time spent. One thread at a time (the command processor).
[[nodiscard]] bool CpReadDebugEnabled() noexcept;
void               NoteCpRead(const char* kind, uint64_t address, const uint32_t* pairs, uint32_t pair_count,
                              uint64_t ns);
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
