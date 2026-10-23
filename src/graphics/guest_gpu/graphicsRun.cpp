#include "graphics/guest_gpu/stageStats.h"
#include "graphics/host_gpu/pageWriteSeq.h"
#include "common/eventTrace.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/bindEpoch.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/experiments.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
// Execution thread: the operation being executed (see GuestGpu::CurrentExecOp).
static thread_local uint64_t          g_exec_op           = 0;
// The command processor thread of a pipelined GPU.
static thread_local bool              g_cp_thread         = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;
// The GuestGpu whose execution thread this is (ServiceUrgentCommands).
static thread_local GuestGpu*         g_execution_gpu     = nullptr;

struct Pm4FaultScope {
	uint64_t packet_addr = 0;
	uint32_t packet_dw   = 0;
	uint32_t header      = 0;
	bool     executing   = false;
};
static thread_local Pm4FaultScope g_pm4_fault_scope;

void NotePm4MemoryFault(uint64_t fault_vaddr, uint32_t access) noexcept {
	static const bool enabled = [] {
		const char* setting = std::getenv("KYTY_PERF_STATS");
		return (setting != nullptr && setting[0] == '1') || std::getenv("KYTY_DBG_CP_READS") != nullptr;
	}();
	if (!enabled || (!GuestGpu::IsGpuThread() && !g_cp_thread) || g_pm4_fault_scope.packet_addr == 0) return;
	// Limit log traffic without exhausting the diagnostic during startup before training.
	static thread_local auto interval = std::chrono::steady_clock::now();
	static thread_local uint32_t lines = 0;
	const auto now = std::chrono::steady_clock::now();
	if (now - interval >= std::chrono::seconds(5)) {
		interval = now;
		lines = 0;
	}
	if (lines++ >= 8) return;
	const auto& scope = g_pm4_fault_scope;
	const uint64_t bytes = static_cast<uint64_t>(scope.packet_dw) * sizeof(uint32_t);
	const bool in_packet = fault_vaddr >= scope.packet_addr &&
	                       fault_vaddr - scope.packet_addr < bytes;
	std::printf("NHL27PM4FAULT: fault=0x%llx access=%u packet=0x%llx packet_dw=%u "
	            "header=0x%08x op=0x%x custom=0x%x phase=%s in_packet=%u\n",
	            static_cast<unsigned long long>(fault_vaddr), access,
	            static_cast<unsigned long long>(scope.packet_addr), scope.packet_dw,
	            scope.header, scope.executing ? ((scope.header >> 8u) & 0xffu) : UINT32_MAX,
	            scope.executing ? KYTY_PM4_R(scope.header) : UINT32_MAX,
	            scope.executing ? "handler" : "fetch", static_cast<unsigned>(in_packet));
}
// Operations a pipelined command processor may queue ahead of their execution.
constexpr size_t OperationQueueSize = 4096;
// Resolved operations queued for the execution thread; see RenderExecutor::ResolvedDrawRing.
constexpr size_t ResolvedQueueSize = 256;
// How long a command processor whose queues all wait spins before it retries them.
constexpr auto BlockedRetryInterval = std::chrono::microseconds(20);

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

// Reads memory other threads may write, with the widest aligned accesses.
static void CopyFromGuest(void* destination, const volatile void* source, size_t size) {
	auto*      out     = static_cast<uint8_t*>(destination);
	const auto address = reinterpret_cast<uint64_t>(source);
	if ((address & 7u) == 0 && (size & 7u) == 0) {
		const auto* words = static_cast<const volatile uint64_t*>(source);
		for (size_t i = 0; i < size / 8u; i++) {
			const uint64_t word = words[i];
			std::memcpy(out + i * 8u, &word, 8u);
		}
		return;
	}
	if ((address & 3u) == 0 && (size & 3u) == 0) {
		const auto* words = static_cast<const volatile uint32_t*>(source);
		for (size_t i = 0; i < size / 4u; i++) {
			const uint32_t word = words[i];
			std::memcpy(out + i * 4u, &word, 4u);
		}
		return;
	}
	const auto* bytes = static_cast<const volatile uint8_t*>(source);
	for (size_t i = 0; i < size; i++) {
		out[i] = bytes[i];
	}
}

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer)
    : m_renderer(renderer),
      m_pipelined((Config::GpuPipelineStages() & Config::GPU_PIPELINE_COMMAND_PROCESSOR) != 0),
      m_resolving(m_pipelined && (Config::GpuPipelineStages() & Config::GPU_PIPELINE_RESOLVE) != 0),
      m_operations(OperationQueueSize), m_resolved(ResolvedQueueSize), m_dispatch_jobs(DispatchSlotCount),
      m_draw_jobs(DrawSlotCount),
      m_neutral_state {&m_neutral_context, &m_neutral_user_config, &m_neutral_shaders} {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	if (m_pipelined) {
		m_pending_writes = std::make_unique<GuestWrite[]>(PendingWriteCapacity);
		m_gfx_cp->AttachPipeline(this);
		m_executor = std::jthread([this] { ExecutionThread(); });
		if (m_resolving) {
			m_dispatch_slots    = std::make_unique<GpuDispatchSlot[]>(DispatchSlotCount);
			m_dispatch_resolver = std::jthread([this] { DispatchResolveThread(); });
			m_draw_slots        = std::make_unique<GpuDrawSlot[]>(DrawSlotCount);
			m_draw_resolver     = std::jthread([this] { DrawResolveThread(); });
			m_resolver          = std::jthread([this] { ResolveThread(); });
		}
	}
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_epoch.fetch_add(1, std::memory_order_release);
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	if (g_cp_thread) {
		m_gfx_cp->EmitHostCommand(std::move(command));
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread() && !g_cp_thread);
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		if (m_pipelined) {
			// Host commands run on the execution thread, after the operations emitted so far.
			m_gfx_cp->BdaBoundary(RenderContext::BdaBoundary::Other);
			m_gfx_cp->EmitHostCommand(std::move(command));
		} else {
			FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_GPU_CMDS);
			m_renderer.BdaBatchBoundary(RenderContext::BdaBoundary::Other);
			command();
		}
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	if (g_cp_thread) {
		// A readback for a page the command processor faulted on: the operations emitted so far
		// may write that memory.
		ExecuteSync(std::move(command));
		return;
	}
	std::binary_semaphore done {0};
	Common::Timeline::Mark("sync-cmd", reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	SendCommand([operation = std::move(command), &done]() mutable {
		Common::Timeline::Mark("sync-cmd-run");
		operation();
		done.release();
	});
	done.acquire();
	Common::Timeline::Mark("sync-cmd-done");
}

void GuestGpu::SendUrgentCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (!m_pipelined || IsGpuThread() || g_cp_thread) {
		SendCommandSync(std::move(command));
		return;
	}
	std::binary_semaphore done {0};
	{
		std::lock_guard lock(m_urgent_mutex);
		m_urgent_commands.push_back([operation = std::move(command), &done]() mutable {
			operation();
			done.release();
		});
		m_urgent_pending.fetch_add(1, std::memory_order_release);
	}
	// A busy execution thread runs the command before its next operation; an idle one waits for
	// operations, so queue one that runs it too.
	SendCommand([this] { RunUrgentCommands(); });
	done.acquire();
}

void GuestGpu::RunUrgentCommands() {
	if (m_urgent_pending.load(std::memory_order_acquire) == 0) {
		return;
	}
	m_renderer.GetRenderExecutor().InvalidateRenderTargetMemo();
	// Unmaps free memory that copies of executed draws may still read.
	if (auto* copies = m_renderer.GetCommandScheduler().HostCopies(); copies != nullptr) {
		copies->Drain();
	}
	std::vector<Common::UniqueFunction<void>> commands;
	{
		std::lock_guard lock(m_urgent_mutex);
		commands.swap(m_urgent_commands);
		m_urgent_pending.store(0, std::memory_order_relaxed);
	}
	for (auto& command: commands) {
		command();
	}
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Submission submission;
	Common::EventTrace::Record(Common::EventTrace::GUEST_SUBMIT, 0, draw_commands.size());
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Common::Timeline::Mark("submit-gfx", draw_commands.size());
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Common::EventTrace::Record(Common::EventTrace::GUEST_SUBMIT, 1 + compute_queue, commands.size());
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Common::Timeline::Mark("submit-cs", commands.size());
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Common::Timeline::Mark("submit-flip", request_id);
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	Common::Timeline::Mark("suspend-wait");
	Common::EventTrace::Record(Common::EventTrace::SUSPEND_ACQ_BEGIN);
	m_suspend_point_ready->acquire();
	Common::EventTrace::Record(Common::EventTrace::SUSPEND_ACQ_END);
	Common::Timeline::Mark("suspend-go");
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
		if (m_pipelined) {
			processor->AttachPipeline(this);
		}
	}
	return *processor;
}

uint64_t GuestGpu::EmitOperation(GpuOperation&& operation) {
	const auto sequence = ++m_emitted;
	operation.sequence  = sequence;
	m_operations.Push(std::move(operation));
	return sequence;
}

void GuestGpu::ForgetExecutedWrites() {
	// Executed writes are in guest memory now.
	const auto executed = m_executed.Load();
	while (m_pending_count != 0 && m_pending_writes[m_pending_first].sequence <= executed) {
		m_pending_first = (m_pending_first + 1) % PendingWriteCapacity;
		m_pending_count--;
	}
}

void GuestGpu::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
                              const void* value, const char* kind, uint8_t origin) {
	PageWriteSeq::Raise(address, size, sequence);
	ForgetExecutedWrites();
	if (sequence <= m_executed.Load()) {
		return;
	}
	if (m_pending_count == PendingWriteCapacity) {
		WaitForExecution(m_pending_writes[m_pending_first].sequence);
		ForgetExecutedWrites();
	}
	auto& write = m_pending_writes[(m_pending_first + m_pending_count) % PendingWriteCapacity];
	write       = {.sequence = sequence, .address = address, .size = size, .kind = kind, .queue = CurrentQueue(), .origin = origin};
	if (value != nullptr && size <= sizeof(write.value)) {
		write.known = true;
		std::memcpy(write.value, value, size);
	}
	m_pending_count++;
}

const char kEopClockWrite[] = "wait-eop-clock";

bool GuestGpu::PendingClockWrite(uint64_t address, uint64_t size) {
	ForgetExecutedWrites();
	const auto end = address + size;
	for (size_t i = m_pending_count; i-- > 0;) {
		const auto* write = &m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write->address >= end || write->address + write->size <= address) {
			continue;
		}
		return write->kind == kEopClockWrite && write->address <= address && end <= write->address + write->size &&
		       write->queue == CurrentQueue();
	}
	return false;
}
// NHL27DECISION (KYTY_PERF_STATS=1): what the command processor waits for when it decides on guest
// memory (WAIT_REG_MEM, COND_EXEC, predication): pending writes whose value is only known once the
// execution thread ran them (by write size), direct reads, and the WAIT_REG_MEM handler itself.
namespace {
struct DecisionStats {
	std::atomic<uint64_t> reads {0};
	std::atomic<uint64_t> known {0};
	std::atomic<uint64_t> waits[3] {}; // by the pending write's origin: other, clock, DMA
	std::atomic<uint64_t> wait_ns[3] {};
	std::atomic<uint64_t> predicted {0};
	std::atomic<uint32_t> slow_logged {0};
	std::atomic<uint64_t> copies {0};
	std::atomic<uint64_t> copy_ns {0};
	std::atomic<uint64_t> wrm_calls {0};
	std::atomic<uint64_t> wrm_suspends {0};
	std::atomic<uint64_t> wrm_ns {0};
	std::atomic<uint64_t> wrm_read_ns {0};
	std::atomic<uint64_t> wrm_barrier_ns {0};
	std::atomic<int64_t>  last_print {0};
};
DecisionStats g_decision;

uint64_t DecisionNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void DecisionMaybePrint() {
	const auto now  = static_cast<int64_t>(DecisionNowNs() / 1000000u);
	auto       last = g_decision.last_print.load(std::memory_order_relaxed);
	if (last == 0) {
		g_decision.last_print.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 5000 || !g_decision.last_print.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto take = [](std::atomic<uint64_t>& v) { return v.exchange(0, std::memory_order_relaxed); };
	const auto ms   = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	const auto reads = take(g_decision.reads);
	const auto known = take(g_decision.known);
	uint64_t   w[3];
	uint64_t   wn[3];
	for (int i = 0; i < 3; i++) {
		w[i]  = take(g_decision.waits[i]);
		wn[i] = take(g_decision.wait_ns[i]);
	}
	const auto predicted = take(g_decision.predicted);
	const auto copies    = take(g_decision.copies);
	const auto copy_ns   = take(g_decision.copy_ns);
	const auto calls     = take(g_decision.wrm_calls);
	const auto suspends  = take(g_decision.wrm_suspends);
	const auto wrm_ns    = take(g_decision.wrm_ns);
	const auto read_ns   = take(g_decision.wrm_read_ns);
	const auto barrier_ns = take(g_decision.wrm_barrier_ns);
	std::printf("NHL27DECISION: window=%.1fs reads=%llu known=%llu wait_other=%llu/%.1fms wait_clock=%llu/%.1fms "
	            "wait_dma=%llu/%.1fms copies=%llu/%.1fms predicted=%llu | wait_reg_mem calls=%llu suspends=%llu "
	            "total=%.1fms read=%.1fms barrier=%.1fms\n",
	            static_cast<double>(now - last) / 1000.0, static_cast<unsigned long long>(reads),
	            static_cast<unsigned long long>(known), static_cast<unsigned long long>(w[0]), ms(wn[0]),
	            static_cast<unsigned long long>(w[1]), ms(wn[1]), static_cast<unsigned long long>(w[2]),
	            ms(wn[2]), static_cast<unsigned long long>(copies), ms(copy_ns),
	            static_cast<unsigned long long>(predicted),
	            static_cast<unsigned long long>(calls), static_cast<unsigned long long>(suspends), ms(wrm_ns),
	            ms(read_ns), ms(barrier_ns));
	std::fflush(stdout);
}
} // namespace

void GuestGpu::ReadDecisionMemory(const volatile void* address, void* value, size_t size) {
	const bool timed = FrameBreakdown::Enabled();
	if (timed) {
		g_decision.reads.fetch_add(1, std::memory_order_relaxed);
	}
	ForgetExecutedWrites();
	const auto begin = reinterpret_cast<uint64_t>(address);
	const auto end   = begin + size;
	// The newest pending write that overlaps the range decides; older ones execute before it.
	for (size_t i = m_pending_count; i-- > 0;) {
		const auto* write = &m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write->address >= end || write->address + write->size <= begin) {
			continue;
		}
		if (write->known && write->address <= begin && end <= write->address + write->size) {
			if (timed) {
				g_decision.known.fetch_add(1, std::memory_order_relaxed);
			}
			std::memcpy(value, write->value + (begin - write->address), size);
			return;
		}
		// The value is only known once the write executed; waiting acquires it.
		const auto write_size    = write->size;
		const auto write_address = write->address;
		const auto write_origin  = write->origin;
		const auto wait_start    = (timed || CpReadDebugEnabled()) ? DecisionNowNs() : 0;
		{
			StageStats::CpStateScope state(StageStats::CpState::WaitDecision);
			WaitForExecution(write->sequence);
		}
		if (CpReadDebugEnabled()) NoteCpRead(write->kind, begin, nullptr, 0,
		    DecisionNowNs() - wait_start);
		if (timed) {
			const auto waited = DecisionNowNs() - wait_start;
			const int  bucket = write_origin < 3 ? write_origin : 0;
			g_decision.waits[bucket].fetch_add(1, std::memory_order_relaxed);
			g_decision.wait_ns[bucket].fetch_add(waited, std::memory_order_relaxed);
			if (waited > 1000000u && g_decision.slow_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::printf("NHL27DECISION: slow wait %.2fms read=0x%016llx+%zu pending write=0x%016llx+%llu origin=%u\n",
				            static_cast<double>(waited) / 1e6, static_cast<unsigned long long>(begin), size,
				            static_cast<unsigned long long>(write_address),
				            static_cast<unsigned long long>(write_size), static_cast<unsigned>(write_origin));
				std::fflush(stdout);
			}
		}
		// An end-of-pipe clock: the value it wrote, not what memory holds now. The interrupt after
		// it wakes the game, which may reset the label before this read (the menus then hung on a
		// WAIT_REG_MEM for a nonzero clock that read 0 forever).
		if (write->address <= begin && end <= write->address + write->size &&
		    TakeExecutedWrite(begin, value, size)) {
			return;
		}
		break;
	}
	const bool debug_reads = CpReadDebugEnabled();
	const auto copy_start = (timed || debug_reads) ? DecisionNowNs() : 0;
	if (!TryReadProtectedForCp(begin, value, size)) {
		CopyFromGuest(value, address, size);
	}
	if (timed || debug_reads) {
		const auto elapsed = DecisionNowNs() - copy_start;
		if (debug_reads) NoteCpRead("wait-reg-mem", begin, nullptr, 0, elapsed);
		if (timed) {
			g_decision.copies.fetch_add(1, std::memory_order_relaxed);
			g_decision.copy_ns.fetch_add(elapsed, std::memory_order_relaxed);
		}
	}
}

namespace {
// Experiment CpBackingReads: reads served from the backing, and what their in-order checks found.
struct CpBackingReadStats {
	std::atomic<uint64_t> reads {0};
	uint64_t              checked = 0, dirty = 0, mismatches = 0;
	std::atomic<bool>     disabled {false};
	std::chrono::steady_clock::time_point last {};
};
CpBackingReadStats g_cp_backing_reads;

uint64_t HashBytes(const void* data, size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	uint64_t    hash  = 0xcbf29ce484222325ull;
	for (size_t i = 0; i < size; i++) {
		hash = (hash ^ bytes[i]) * 0x100000001b3ull;
	}
	return hash;
}
} // namespace

bool GuestGpu::CpBackingReadsOn() const {
	return g_cp_thread && Common::Experiments::Enabled(Common::Experiments::CpBackingReads) &&
	       !g_cp_backing_reads.disabled.load(std::memory_order_relaxed);
}

void GuestGpu::CheckCpBackingRead(CommandProcessor& cp, uint64_t address, const void* bytes, size_t size) {
	g_cp_backing_reads.reads.fetch_add(1, std::memory_order_relaxed);
	cp.EmitHostCommand([this, address, size, hash = HashBytes(bytes, size)] {
		auto& stats = g_cp_backing_reads;
		stats.checked++;
		if (m_renderer.GetBufferCache().HasGpuDirtyBytes(address, size)) {
			// An earlier operation wrote bytes the CP already used: compare with what it wrote.
			stats.dirty++;
			std::vector<uint8_t> actual(size);
			if (!LibKernel::Memory::TryReadBufferBacking(address, actual.data(), actual.size()) ||
			    HashBytes(actual.data(), actual.size()) != hash) {
				stats.mismatches++;
				stats.disabled.store(true, std::memory_order_relaxed);
				std::printf("NHL27CPSPEC: MISMATCH addr=0x%016" PRIx64 " size=%zu: GPU-written bytes the command "
				            "processor read early; CP backing reads now off\n",
				            address, size);
				std::fflush(stdout);
			}
		}
		const auto now = std::chrono::steady_clock::now();
		if (CpReadDebugEnabled() && now - stats.last >= std::chrono::seconds(5)) {
			stats.last = now;
			std::printf("NHL27CPSPEC: reads=%llu checked=%llu dirty=%llu mismatches=%llu\n",
			            static_cast<unsigned long long>(stats.reads.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(stats.checked), static_cast<unsigned long long>(stats.dirty),
			            static_cast<unsigned long long>(stats.mismatches));
			std::fflush(stdout);
		}
	});
}

bool GuestGpu::TryReadProtectedForCp(uint64_t address, void* value, size_t size) {
	// The page holds GPU-written bytes, in NHL 27 nearly always other ones than the CP reads
	// (register tables and WAIT_REG_MEM values next to compute-written bytes). A plain read
	// faulted, and its download waited for the writer (~6 ms each while it still ran); reading
	// through the execution thread instead waits for it to catch up (~0.25 ms each, and the CP
	// stops running ahead). The CP reads the backing now, as it reads an unprotected page, and the
	// execution thread checks in order that no earlier operation wrote these bytes.
	if (size == 0 || !CpBackingReadsOn() || !m_renderer.IsReadProtected(address, size) ||
	    !LibKernel::Memory::TryReadBacking(address, value, size)) {
		return false;
	}
	CheckCpBackingRead(*m_gfx_cp, address, value, size);
	return true;
}

// CommandProcessor::BufferFlush() requests by site (KYTY_DBG_STAGE_STATS).
static std::array<std::atomic<uint64_t>, static_cast<size_t>(CommandProcessor::FlushSite::Count)> g_buffer_flush_emits {};

// KYTY_DBG_STAGE_STATS: every 5 s (checked when the execution thread runs out of operations).
static void PrintStageStats() {
	static auto     last       = std::chrono::steady_clock::now();
	static uint64_t last_items = 0;
	static uint64_t last_ns[StageStats::CounterCount] {};
	const auto      now        = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - last).count();
	last                 = now;
	double ms[StageStats::CounterCount];
	for (uint32_t i = 0; i < StageStats::CounterCount; i++) {
		const auto value = StageStats::g_ns[i].load(std::memory_order_relaxed);
		ms[i]            = static_cast<double>(value - last_ns[i]) / 1e6 / seconds;
		last_ns[i]       = value;
	}
	const uint64_t ops   = StageStats::g_ops.load(std::memory_order_relaxed);
	const uint64_t items = ops - last_items;
	last_items           = ops;
	const double   busy  = 1000.0 - ms[StageStats::ExecIdle];
	std::printf("NHL27STALLS mask=%llu ops/s=%.0f exec busy=%.1f idle=%.1f lock=%.1f ms/s busy/op=%.2f us"
	            " (%.2f us without lock waits) | resolve lock=%.1f queue-full=%.1f idle=%.1f busy=%.1f ms/s\n",
	            static_cast<unsigned long long>(Common::Experiments::Mask()), static_cast<double>(items) / seconds, busy,
	            ms[StageStats::ExecIdle], ms[StageStats::ExecProgramLock],
	            items != 0 ? busy * 1e3 * seconds / static_cast<double>(items) : 0.0,
	            items != 0 ? (busy - ms[StageStats::ExecProgramLock]) * 1e3 * seconds / static_cast<double>(items) : 0.0,
	            ms[StageStats::ResolveProgramLock], ms[StageStats::ResolveQueueFull], ms[StageStats::ResolveIdle],
	            1000.0 - ms[StageStats::ResolveQueueFull] - ms[StageStats::ResolveIdle]);
	std::printf("NHL27EXECIDLE ms/s: resolve=%.1f cp-busy=%.1f no-submission=%.1f blocked=%.1f "
	            "cp-decision-wait=%.1f cp-other-wait=%.1f | bda syncs/s=%.0f flushes/s=%.0f (callbacks %.0f batch %.0f guest-reads %.0f flips %.0f)"
	            " | asked by other %.0f submission %.0f interrupt %.0f suspend %.0f split %.0f\n",
	            ms[StageStats::ExecIdleResolve], ms[StageStats::ExecIdleCp], ms[StageStats::ExecIdleNoWork],
	            ms[StageStats::ExecIdleBlocked], ms[StageStats::ExecIdleDecision], ms[StageStats::ExecIdleOther],
	            static_cast<double>(StageStats::g_bda_syncs.exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_scheduler_flushes.exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_scheduler_flush_sites[0].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_scheduler_flush_sites[1].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_scheduler_flush_sites[2].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_scheduler_flush_sites[3].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_buffer_flush_emits[0].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_buffer_flush_emits[1].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_buffer_flush_emits[2].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_buffer_flush_emits[3].exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_buffer_flush_emits[4].exchange(0, std::memory_order_relaxed)) / seconds);
	for (int kind = 0; kind < 2; kind++) {
		const auto calls = StageStats::g_check_calls[kind].exchange(0, std::memory_order_relaxed);
		const auto entries = StageStats::g_check_entries[kind].exchange(0, std::memory_order_relaxed);
		const auto bytes   = StageStats::g_check_bytes[kind].exchange(0, std::memory_order_relaxed);
		const auto pages   = StageStats::g_check_pages[kind].exchange(0, std::memory_order_relaxed);
		const auto skipped = StageStats::g_check_skipped[kind].exchange(0, std::memory_order_relaxed);
		const auto unsafe  = StageStats::g_check_unsafe[kind].exchange(0, std::memory_order_relaxed);
		const auto later   = StageStats::g_check_later[kind].exchange(0, std::memory_order_relaxed);
		if (calls == 0) continue;
		const auto per = [calls](uint64_t value) { return static_cast<double>(value) / static_cast<double>(calls); };
		std::printf("NHL27CHECKS %s: calls/s=%.0f entries/call=%.1f bytes/call=%.0f pages/call=%.1f skipped/call=%.1f"
		            " checked: gpu-owned/call=%.1f written-later/call=%.1f\n",
		            kind == 0 ? "draws" : "dispatches", static_cast<double>(calls) / seconds, per(entries), per(bytes),
		            per(pages), per(skipped), per(unsafe), per(later));
	}
	{
		const auto total = StageStats::g_bind_total.exchange(0, std::memory_order_relaxed);
		const auto same  = StageStats::g_bind_same.exchange(0, std::memory_order_relaxed);
		const auto wr    = StageStats::g_bind_written.exchange(0, std::memory_order_relaxed);
		std::printf("NHL27BINDS: buffer bindings/s=%.0f same-as-last=%.0f written=%.0f\n", static_cast<double>(total) / seconds,
		            static_cast<double>(same) / seconds, static_cast<double>(wr) / seconds);
	}
	// By operation kind: ms/s and us per operation.
	static uint64_t last_kind_ns[8] {}, last_kind_count[8] {};
	static const char* const kinds[] = {"draw", "draw-auto", "draw-indirect", "dispatch", "dispatch-indirect", "callback", "?", "?"};
	std::string line = "NHL27OPS:";
	for (size_t k = 0; k < 6; k++) {
		const auto total_ns    = StageStats::g_kind_ns[k].load(std::memory_order_relaxed);
		const auto total_count = StageStats::g_kind_count[k].load(std::memory_order_relaxed);
		const auto ns          = total_ns - last_kind_ns[k];
		const auto count       = total_count - last_kind_count[k];
		last_kind_ns[k]        = total_ns;
		last_kind_count[k]     = total_count;
		if (count == 0) continue;
		line += fmt::format(" {}={:.0f}/s {:.1f}ms/s {:.2f}us", kinds[k], static_cast<double>(count) / seconds,
		                    static_cast<double>(ns) / 1e6 / seconds, static_cast<double>(ns) / 1e3 / static_cast<double>(count));
	}
	std::printf("%s\n", line.c_str());
	// Callbacks by what they do (GpuCallbackTag).
	static uint64_t    last_cb_ns[16] {}, last_cb_count[16] {};
	static const char* const tags[] = {"other",   "bda",        "flush", "finish", "flip",
	                                   "eop",     "event",      "barrier", "write-data", "dma",
	                                   "clock",   "host",       "gc",    "flip-wait"};
	{
		// Waits for the host copy thread before operations (HostCopyQueue::Drain), by callback tag.
		static uint64_t last_drains[16] {}, last_waits[16] {}, last_drain_ns[16] {};
		std::string drains = "NHL27DRAINS:";
		for (size_t k = 0; k < 16; k++) {
			const auto count = StageStats::g_drain_count[k].load(std::memory_order_relaxed);
			const auto waits = StageStats::g_drain_waits[k].load(std::memory_order_relaxed);
			const auto ns    = StageStats::g_drain_ns[k].load(std::memory_order_relaxed);
			if (count != last_drains[k]) {
				drains += fmt::format(" {}={:.0f}/s waits={:.0f}/s {:.1f}ms/s", k < std::size(tags) ? tags[k] : "op",
				                      static_cast<double>(count - last_drains[k]) / seconds,
				                      static_cast<double>(waits - last_waits[k]) / seconds,
				                      static_cast<double>(ns - last_drain_ns[k]) / 1e6 / seconds);
			}
			last_drains[k]   = count;
			last_waits[k]    = waits;
			last_drain_ns[k] = ns;
		}
		std::printf("%s\n", drains.c_str());
	}
	line = "NHL27CALLBACKS:";
	for (size_t k = 0; k < std::size(tags); k++) {
		const auto total_ns    = StageStats::g_callback_ns[k].load(std::memory_order_relaxed);
		const auto total_count = StageStats::g_callback_count[k].load(std::memory_order_relaxed);
		const auto ns          = total_ns - last_cb_ns[k];
		const auto count       = total_count - last_cb_count[k];
		last_cb_ns[k]          = total_ns;
		last_cb_count[k]       = total_count;
		if (count == 0) continue;
		line += fmt::format(" {}={:.0f}/s {:.1f}ms/s {:.2f}us", tags[k], static_cast<double>(count) / seconds,
		                    static_cast<double>(ns) / 1e6 / seconds, static_cast<double>(ns) / 1e3 / static_cast<double>(count));
	}
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}

void GuestGpu::NoteExecutedWrite(uint64_t address, const void* value, uint64_t size) {
	if (size > sizeof(ExecutedWrite::value)) {
		return;
	}
	std::lock_guard lock(m_executed_writes_mutex);
	auto& entry   = m_executed_writes[m_executed_writes_next++ % m_executed_writes.size()];
	entry.address = address;
	entry.size    = size;
	std::memcpy(entry.value, value, size);
}

bool GuestGpu::TakeExecutedWrite(uint64_t address, void* value, uint64_t size) {
	std::lock_guard lock(m_executed_writes_mutex);
	// The newest write covering the range.
	for (size_t i = 0; i < m_executed_writes.size(); i++) {
		auto& entry = m_executed_writes[(m_executed_writes_next + m_executed_writes.size() - 1 - i) %
		                                m_executed_writes.size()];
		if (entry.size != 0 && entry.address <= address && address + size <= entry.address + entry.size) {
			std::memcpy(value, entry.value + (address - entry.address), size);
			entry.size = 0;
			return true;
		}
	}
	return false;
}

bool GuestGpu::TryPredictClockWrite(const volatile void* address, size_t size, uint64_t& lower_bound) {
	ForgetExecutedWrites();
	const auto begin = reinterpret_cast<uint64_t>(address);
	const auto end   = begin + size;
	for (size_t i = m_pending_count; i-- > 0;) {
		const auto* write = &m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write->address >= end || write->address + write->size <= begin) {
			continue;
		}
		if (write->origin != WriteOriginClock || write->known || write->address != begin || write->size != size) {
			return false;
		}
		const auto sequence = write->sequence;
		lower_bound         = Sync::ReadReferenceClock();
		// The write reads the clock after this only while it is still pending.
		return m_executed.Load() < sequence;
	}
	return false;
}

void GuestGpu::ExecuteSync(Common::UniqueFunction<void>&& command) {
	m_gfx_cp->EmitHostCommand(std::move(command));
	WaitForExecution();
}

// KYTY_DBG_LONG_OPS=<us>: every operation the execution thread spends that long on prints where the
// time went (NHL27LONGOP: waiting for its resolution, then the parts of executing it; the parts need
// KYTY_PERF_STATS=1, and readback, fault and protect counts include other threads).
static const int64_t g_long_op_ns = [] {
	const char* value = std::getenv("KYTY_DBG_LONG_OPS");
	return value != nullptr ? std::strtoll(value, nullptr, 10) * 1000 : int64_t {0};
}();

namespace {
struct LongOpCounters {
	static constexpr size_t Count = 17;
	std::array<uint64_t, Count> values {};

	static LongOpCounters Take() {
		auto&          c = FrameBreakdown::g_counters;
		LongOpCounters s;
		const std::array<const std::atomic<uint64_t>*, Count> sources {
		    &c.dp_materialize_ns, &c.dp_textures_ns, &c.dp_buffers_ns, &c.dp_bda_ns, &c.dp_rebind_ns,
		    &c.dp_descriptors_ns, &c.dp_pipeline_ns, &c.dc_total_ns,  &c.dc_bda_ns, &c.readback_ns,
		    &c.fault_ns,          &c.compile_ns,     &c.sync_uploads, &c.sync_bytes, &c.protect_calls,
		    &c.protect_ns,        &c.gpu_wait_ns};
		for (size_t i = 0; i < Count; i++) {
			s.values[i] = sources[i]->load(std::memory_order_relaxed);
		}
		return s;
	}
};

void PrintLongOp(uint64_t op_kind, int64_t start, int64_t resolved, int64_t end, const LongOpCounters& before) {
	const auto after = LongOpCounters::Take();
	std::array<double, LongOpCounters::Count> d {};
	for (size_t i = 0; i < d.size(); i++) {
		d[i] = static_cast<double>(after.values[i] - before.values[i]);
	}
	const auto us = [](double ns) { return ns / 1000.0; };
	std::printf("NHL27LONGOP kind=%u op=0x%x total=%.0fus resolve_wait=%.0fus exec=%.0fus | draw: mat=%.0f tex=%.0f "
	            "buf=%.0f bda=%.0f rebind=%.0f desc=%.0f pipe=%.0f | dispatch=%.0f bda=%.0f | readback=%.0f "
	            "fault=%.0f compile=%.0f gpu_wait=%.0f | uploads=%.0f kb=%.0f protects=%.0f (%.0fus)\n",
	            static_cast<uint32_t>(op_kind & 0xffu), static_cast<uint32_t>(op_kind >> 8u),
	            us(static_cast<double>(end - start)), us(static_cast<double>(resolved - start)),
	            us(static_cast<double>(end - resolved)), us(d[0]), us(d[1]), us(d[2]), us(d[3]), us(d[4]), us(d[5]),
	            us(d[6]), us(d[7]), us(d[8]), us(d[9]), us(d[10]), us(d[11]), us(d[16]), d[12], d[13] / 1024.0, d[14],
	            us(d[15]));
}
} // namespace

void GuestGpu::ExecutionThread() {
	KYTY_PROFILER_THREAD("Thread_GpuExecute");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Execution);
	StageStats::t_role = StageStats::Role::Execution;
	g_gpu_thread    = true;
	g_execution_gpu = this;
	RegionManager::MarkGpuOrderedThread();
	// Waking the command processor costs a fence: publish every operation, but only look for a
	// waiter now and then and before sleeping.
	constexpr uint32_t NotifyInterval = 16;
	uint32_t           unnotified     = 0;
	auto&              operations     = m_resolving ? m_resolved : m_operations;
	for (;;) {
		if (m_urgent_pending.load(std::memory_order_relaxed) != 0) {
			RunUrgentCommands();
		}
		auto operation = operations.TryPop();
		if (!operation) {
			m_executed.Notify();
			unnotified = 0;
			Common::EventTrace::Record(Common::EventTrace::EXEC_WAIT_BEGIN);
			if (StageStats::Enabled()) {
				const auto start = std::chrono::steady_clock::now();
				const auto cause = StageStats::ExecIdleCause();
				operation        = operations.Pop();
				StageStats::Add(StageStats::ExecIdle, start);
				StageStats::Add(cause, start);
				PrintStageStats();
			} else {
				operation = operations.Pop();
			}
			Common::EventTrace::Record(Common::EventTrace::EXEC_WAIT_END);
			if (!operation) {
				break;
			}
		}
		// KYTY_IDLE_TRACE: operations taking 20 us or more, recorded when they end
		// (a = GpuOperationKind | GpuOperation::trace_op << 8, b = microseconds).
		const bool traced   = Common::EventTrace::g_capturing.load(std::memory_order_relaxed);
		const auto op_kind  = static_cast<uint64_t>(operation->kind) | (uint64_t {operation->trace_op} << 8u);
		const auto op_start = traced ? Common::EventTrace::NowNs() : int64_t {0};
		g_exec_op           = op_kind;
		const bool long_ops   = g_long_op_ns != 0;
		const auto long_start = long_ops ? Common::EventTrace::NowNs() : int64_t {0};
		const auto long_before = long_ops ? std::optional(LongOpCounters::Take()) : std::nullopt;
		if (auto* slot = operation->dispatch_slot; slot != nullptr) {
			if (slot->state.load(std::memory_order_acquire) != 2) {
				const auto start = StageStats::Enabled() ? std::chrono::steady_clock::now()
				                                         : std::chrono::steady_clock::time_point {};
				for (Common::SpinWait wait; slot->state.load(std::memory_order_acquire) != 2;) {
					if (!wait.Spin()) {
						std::this_thread::yield();
					}
				}
				if (StageStats::Enabled()) {
					StageStats::Add(StageStats::ExecIdle, start);
					StageStats::Add(StageStats::ExecIdleResolve, start);
				}
			}
			operation->resolved_dispatch = slot->result;
			slot->state.store(0, std::memory_order_release);
			operation->dispatch_slot = nullptr;
		}
		if (auto* slot = operation->draw_slot; slot != nullptr) {
			if (slot->state.load(std::memory_order_acquire) != 2) {
				const auto start = StageStats::Enabled() ? std::chrono::steady_clock::now()
				                                         : std::chrono::steady_clock::time_point {};
				for (Common::SpinWait wait; slot->state.load(std::memory_order_acquire) != 2;) {
					if (!wait.Spin()) {
						std::this_thread::yield();
					}
				}
				if (StageStats::Enabled()) {
					StageStats::Add(StageStats::ExecIdle, start);
					StageStats::Add(StageStats::ExecIdleResolve, start);
				}
			}
			operation->resolved = slot->result;
			slot->state.store(0, std::memory_order_release);
			operation->draw_slot = nullptr;
		}
		PageWriteSeq::SetOperation(operation->sequence);
		const auto long_resolved = long_ops ? Common::EventTrace::NowNs() : int64_t {0};
		if (StageStats::Enabled()) {
			const auto kind  = static_cast<size_t>(operation->kind) & 7u;
			const auto start = std::chrono::steady_clock::now();
			operation->processor->Execute(*operation);
			const auto ns    = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
			StageStats::g_kind_ns[kind].store(StageStats::g_kind_ns[kind].load(std::memory_order_relaxed) + ns, std::memory_order_relaxed);
			StageStats::g_kind_count[kind].store(StageStats::g_kind_count[kind].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
			if (operation->kind == GpuOperationKind::Callback) {
				const auto tag = static_cast<size_t>(operation->callback_tag) & 15u;
				StageStats::g_callback_ns[tag].store(StageStats::g_callback_ns[tag].load(std::memory_order_relaxed) + ns,
				                                     std::memory_order_relaxed);
				StageStats::g_callback_count[tag].store(
				    StageStats::g_callback_count[tag].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
			}
			StageStats::g_ops.store(StageStats::g_ops.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		} else {
			operation->processor->Execute(*operation);
		}
		g_exec_op = 0;
		if (long_ops) {
			const auto long_end = Common::EventTrace::NowNs();
			if (long_end - long_start >= g_long_op_ns) {
				PrintLongOp(op_kind, long_start, long_resolved, long_end, *long_before);
			}
		}
		if (traced) {
			const auto op_ns = Common::EventTrace::NowNs() - op_start;
			if (op_ns >= 20000) {
				Common::EventTrace::Record(Common::EventTrace::EXEC_OP, op_kind, static_cast<uint64_t>(op_ns / 1000));
			}
		}
		const auto sequence = operation->sequence;
		// Destroy the callback before the command processor may reuse what it references.
		operation.reset();
		m_executed.Publish(sequence);
		// Work between operations (urgent commands) writes after this one.
		PageWriteSeq::SetOperation(sequence + 1);
		if (++unnotified >= NotifyInterval) {
			m_executed.Notify();
			unnotified = 0;
		}
	}
	m_executed.Notify();
	g_gpu_thread = false;
}

void GuestGpu::ResolveThread() {
	KYTY_PROFILER_THREAD("Thread_GpuResolve");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Resolve);
	StageStats::t_role = StageStats::Role::Resolve;
	auto& executor = m_renderer.GetRenderExecutor();
	for (;;) {
		auto operation = m_operations.TryPop();
		if (!operation) {
			const auto start = StageStats::Enabled() ? std::chrono::steady_clock::now()
			                                         : std::chrono::steady_clock::time_point {};
			operation = m_operations.Pop();
			if (StageStats::Enabled()) {
				StageStats::Add(StageStats::ResolveIdle, start);
			}
			if (!operation) {
				break;
			}
		}
		if (StageStats::Enabled()) StageStats::g_resolve_working.store(true, std::memory_order_relaxed);
		GuestReadLog::SetResolveBase(m_executed.Load());
		if ((operation->kind == GpuOperationKind::DrawIndex ||
		     operation->kind == GpuOperationKind::DrawAuto) &&
		    Common::Experiments::Enabled(Common::Experiments::DrawResolver)) {
			// Experiment DrawResolver: by the vertex and pixel shader addresses, about half of the
			// draws go to the second draw resolver; the draws of a program pair stay on one thread,
			// so the two rarely materialize the same program at once (the second one then leaves
			// the draw to the execution thread).
			const auto& shaders = *operation->state.shaders;
			const auto  key     = shaders.GetPs().ps_regs.data_addr ^ shaders.GetVs().gs_regs.data_addr ^
			                 shaders.GetVs().es_regs.data_addr;
			if (((key >> 8u) * 0x9e3779b97f4a7c15ull) >> 63u) {
				QueueDrawResolution(*operation);
			} else {
				operation->resolved =
				    executor.ResolveDrawAhead(*operation->state.context, *operation->state.user_config,
				                              *operation->state.shaders);
			}
		} else if (operation->kind == GpuOperationKind::DrawIndex ||
		    operation->kind == GpuOperationKind::DrawAuto) {
			// The register snapshots stay valid until the operation executed.
			operation->resolved =
			    executor.ResolveDrawAhead(*operation->state.context, *operation->state.user_config,
			                              *operation->state.shaders);
		} else if (operation->kind == GpuOperationKind::DispatchDirect &&
		           (Common::Experiments::Enabled(Common::Experiments::DispatchAhead) ||
		            Common::RuntimeFlags::Current().compute_resolve_ahead) &&
		           DispatchResolverActive()) {
			QueueDispatchResolution(*operation, false);
		} else if (operation->kind == GpuOperationKind::DispatchIndirect &&
		           (operation->dispatch.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0 &&
		           (Common::Experiments::Enabled(Common::Experiments::DispatchAhead) ||
		            Common::RuntimeFlags::Current().compute_resolve_ahead) &&
		           DispatchResolverActive()) {
			QueueDispatchResolution(*operation, true);
		} else if (operation->kind == GpuOperationKind::DispatchDirect &&
		           (Common::Experiments::Enabled(Common::Experiments::DispatchAhead) ||
		            Common::RuntimeFlags::Current().compute_resolve_ahead)) {
			const auto& dispatch = operation->dispatch;
			operation->resolved_dispatch = executor.ResolveDispatchAhead(
			    *operation->state.context, *operation->state.shaders, dispatch.thread_group_x,
			    dispatch.thread_group_y, dispatch.thread_group_z, dispatch.mode,
			    operation->processor->IsAsyncComputeQueue());
		} else if (operation->kind == GpuOperationKind::DispatchIndirect &&
		           (operation->dispatch.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0 &&
		           (Common::Experiments::Enabled(Common::Experiments::DispatchAhead) ||
		            Common::RuntimeFlags::Current().compute_resolve_ahead)) {
			// Indirect dispatches materialized on the execution thread: 3.5% of it in a match
			// (in-boot A/B: 24.9 -> 11.5 us of the execution thread per indirect dispatch).
			operation->resolved_dispatch = executor.ResolveDispatchIndirectAhead(
			    *operation->state.context, *operation->state.shaders, operation->processor->IsAsyncComputeQueue());
		}
		if (StageStats::Enabled()) StageStats::g_resolve_working.store(false, std::memory_order_relaxed);
		if (StageStats::Enabled() && !m_resolved.TryPush(*operation)) {
			const auto start = std::chrono::steady_clock::now();
			m_resolved.Push(std::move(*operation));
			StageStats::Add(StageStats::ResolveQueueFull, start);
		} else if (!StageStats::Enabled()) {
			m_resolved.Push(std::move(*operation));
		}
	}
	m_resolved.Stop();
}

void GuestGpu::QueueDispatchResolution(GpuOperation& operation, bool indirect) {
	auto& slot = m_dispatch_slots[m_dispatch_slot_next];
	m_dispatch_slot_next = (m_dispatch_slot_next + 1) % DispatchSlotCount;
	// The execution thread frees the slots in order.
	for (Common::SpinWait wait; slot.state.load(std::memory_order_acquire) != 0;) {
		if (!wait.Spin()) {
			std::this_thread::yield();
		}
	}
	slot.result = nullptr;
	slot.state.store(1, std::memory_order_relaxed);
	const auto& dispatch = operation.dispatch;
	DispatchJob job {.slot     = &slot,
	                 .context  = operation.state.context,
	                 .shaders  = operation.state.shaders,
	                 .x        = dispatch.thread_group_x,
	                 .y        = dispatch.thread_group_y,
	                 .z        = dispatch.thread_group_z,
	                 .mode     = dispatch.mode,
	                 .async    = operation.processor->IsAsyncComputeQueue(),
	                 .indirect = indirect};
	m_dispatch_jobs.Push(std::move(job));
	m_dispatch_jobs_queued++;
	operation.dispatch_slot = &slot;
}

void GuestGpu::QueueDrawResolution(GpuOperation& operation) {
	auto& slot       = m_draw_slots[m_draw_slot_next];
	m_draw_slot_next = (m_draw_slot_next + 1) % DrawSlotCount;
	// The execution thread frees the slots in order.
	for (Common::SpinWait wait; slot.state.load(std::memory_order_acquire) != 0;) {
		if (!wait.Spin()) {
			std::this_thread::yield();
		}
	}
	slot.result = nullptr;
	slot.state.store(1, std::memory_order_relaxed);
	m_draw_jobs.Push(DrawJob {.slot        = &slot,
	                          .context     = operation.state.context,
	                          .user_config = operation.state.user_config,
	                          .shaders     = operation.state.shaders});
	operation.draw_slot = &slot;
}

// Experiment DrawResolver: resolves the draws the resolve thread passes on, in order, into a
// resolution ring of its own.
void GuestGpu::DrawResolveThread() {
	KYTY_PROFILER_THREAD("Thread_GpuResolveDraw");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::DrawResolve);
	auto& executor = m_renderer.GetRenderExecutor();
	for (;;) {
		auto job = m_draw_jobs.Pop();
		if (!job) {
			break;
		}
		GuestReadLog::SetResolveBase(m_executed.Load());
		job->slot->result = executor.ResolveDrawAhead(*job->context, *job->user_config, *job->shaders, 1);
		job->slot->state.store(2, std::memory_order_release);
	}
}

// Experiment DispatchResolver: resolves the dispatches the resolve thread passes on, in order,
// while it resolves the draws. Compute programs are never draw stages, so the two threads do not
// materialize the same resource plans.
void GuestGpu::DispatchResolveThread() {
	KYTY_PROFILER_THREAD("Thread_GpuResolveDispatch");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::DispatchResolve);
	auto& executor = m_renderer.GetRenderExecutor();
	for (;;) {
		auto job = m_dispatch_jobs.Pop();
		if (!job) {
			break;
		}
		GuestReadLog::SetResolveBase(m_executed.Load());
		auto* result = job->indirect
		                   ? executor.ResolveDispatchIndirectAhead(*job->context, *job->shaders, job->async)
		                   : executor.ResolveDispatchAhead(*job->context, *job->shaders, job->x, job->y, job->z,
		                                                   job->mode, job->async);
		job->slot->result = result;
		job->slot->state.store(2, std::memory_order_release);
		m_dispatch_jobs_done.fetch_add(1, std::memory_order_release);
	}
}

void GuestGpu::StopExecution() {
	WaitForExecution();
	m_operations.Stop();
	if (m_resolver.joinable()) {
		m_resolver.join();
	}
	// After the resolve thread: it queues no more dispatches; the queued ones are still resolved.
	m_dispatch_jobs.Stop();
	if (m_dispatch_resolver.joinable()) {
		m_dispatch_resolver.join();
	}
	m_draw_jobs.Stop();
	if (m_draw_resolver.joinable()) {
		m_draw_resolver.join();
	}
	if (m_executor.joinable()) {
		m_executor.join();
	}
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_ctx_dirty    = true;
	m_ucfg_dirty   = true;
	m_sh_ctx_dirty = true;
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances.value              = 1;
	m_num_instances.sequence++;
	m_pending_shader_offsets.clear();
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	m_ctx_dirty = true;
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

// KYTY_IDLE_TRACE: the PM4 packet the command processor thread is handling (GpuOperation::trace_op).
static thread_local uint16_t g_trace_op = 0;

GpuOperation CommandProcessor::MakeOperation(GpuOperationKind kind) {
	GpuOperation operation;
	operation.kind          = kind;
	operation.trace_op      = g_trace_op;
	operation.processor     = this;
	operation.submit_id     = m_submit_id;
	operation.num_instances = m_num_instances;
	operation.after_wait    = std::exchange(m_wait_since_operation, false);
	operation.write_epoch   = RegionManager::CurrentWriteEpoch();
	if (m_pipeline == nullptr) {
		operation.state = {&m_ctx, &m_ucfg, &m_sh_ctx};
	} else if (kind == GpuOperationKind::Callback) {
		// Only draws and dispatches read registers.
		operation.state = m_pipeline->NeutralState();
	} else {
		operation.state = SnapshotState(m_pipeline->NextSequence());
	}
	return operation;
}

template <typename T>
const T* CommandProcessor::Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty,
                                    uint64_t sequence, uint64_t* id) {
	static_assert(std::is_trivially_copyable_v<T>);
	if (dirty || ring.current == SnapshotRing<T>::Size) {
		if (ring.slots == nullptr) {
			ring.slots = std::make_unique<typename SnapshotRing<T>::Slot[]>(SnapshotRing<T>::Size);
			ring.last_use = std::make_unique<uint64_t[]>(SnapshotRing<T>::Size);
		}
		auto& slot = ring.slots[ring.next];
		// The execution thread publishes the sequence of an operation after it stops reading the
		// operation's snapshots; the wait acquires that.
		{
			StageStats::CpStateScope state(StageStats::CpState::WaitOther);
			m_pipeline->WaitForExecution(ring.last_use[ring.next]);
		}
		slot.value   = live;
		slot.id      = s_snapshot_ids.fetch_add(1, std::memory_order_relaxed) + 1;
		ring.current = ring.next;
		ring.next    = (ring.next + 1) % SnapshotRing<T>::Size;
		dirty        = false;
	}
	// The operation built from this state is the next one the command processor emits.
	ring.last_use[ring.current] = sequence;
	if (id != nullptr) {
		*id = ring.slots[ring.current].id;
	}
	return &ring.slots[ring.current].value;
}

GpuRegisterState CommandProcessor::SnapshotState(uint64_t sequence) {
	GpuRegisterState state;
	state.context     = Snapshot(m_ctx_snapshots, m_ctx, m_ctx_dirty, sequence, &state.context_id);
	state.user_config = Snapshot(m_ucfg_snapshots, m_ucfg, m_ucfg_dirty, sequence);
	state.shaders     = Snapshot(m_sh_ctx_snapshots, m_sh_ctx, m_sh_ctx_dirty, sequence);
	return state;
}

// Draws and dispatches emitted by every command processor (pipelined), and their count at the
// last BDA boundary emitted: one command processor thread emits them all.
static uint64_t g_work_emitted     = 0;
static uint64_t g_bda_work_emitted = ~uint64_t {0};

uint64_t CommandProcessor::Emit(GpuOperation&& operation) {
	if (m_pipeline != nullptr) {
		EXIT_IF(operation.kind != GpuOperationKind::Callback &&
		        m_pipeline->NextSequence() !=
		            m_ctx_snapshots.last_use[m_ctx_snapshots.current]);
		g_work_emitted += operation.kind != GpuOperationKind::Callback ? 1u : 0u;
		return m_pipeline->EmitOperation(std::move(operation));
	}
	Execute(operation);
	return 0;
}

void CommandProcessor::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
                                      const void* value, const char* kind, uint8_t origin) {
	LibKernel::Memory::NoteHostWrite(address, size, LibKernel::Memory::HostWriteSource::CommandProcessor, value);
	if (m_pipeline != nullptr && size != 0) {
		m_pipeline->NoteGuestWrite(sequence, address, size, value, kind, origin);
	}
}

void CommandProcessor::EmitHostCommand(Common::UniqueFunction<void>&& command) {
	EmitCallback(std::move(command), GpuCallbackTag::Host);
}

void CommandProcessor::ReadDecisionMemory(const volatile void* address, void* value,
                                          size_t size) {
	if (m_pipeline != nullptr) {
		m_pipeline->ReadDecisionMemory(address, value, size);
		return;
	}
	if (!TryPeekGuest(address, value, size)) {
		CopyFromGuest(value, address, size);
	}
}

void CommandProcessor::BdaBoundary(RenderContext::BdaBoundary kind) {
	// Guest writes from now on belong to the operations after this boundary.
	RegionManager::AdvanceWriteEpoch();
	if (m_pipeline == nullptr) {
		m_renderer.BdaBatchBoundary(kind);
		return;
	}
	// The boundary only makes the next draw or dispatch synchronize again: with none since the
	// last boundary, that one already does (80,000 boundaries a second were mostly such repeats,
	// each an operation through the pipeline). Statistics (KYTY_PERF_STATS) count them all.
	if (g_bda_work_emitted == g_work_emitted && !FrameBreakdown::Enabled()) {
		return;
	}
	g_bda_work_emitted = g_work_emitted;
	// KYTY_BDA_BATCH orders boundaries with the draws and dispatches that execute around them.
	const auto saved_trace_op = g_trace_op;
	g_trace_op                = static_cast<uint16_t>(g_trace_op | 0x4000u);
	EmitCallback([this, kind] { m_renderer.BdaBatchBoundary(kind); }, GpuCallbackTag::Bda);
	g_trace_op = saved_trace_op;
}

uint64_t CommandProcessor::EmitCallback(GpuCallback&& callback, GpuCallbackTag tag) {
	auto operation         = MakeOperation(GpuOperationKind::Callback);
	operation.callback     = std::move(callback);
	operation.callback_tag = tag;
	return Emit(std::move(operation));
}

void CommandProcessor::Execute(GpuOperation& operation) {
	EXIT_IF(operation.processor != this);
	m_renderer.SetOperationWriteEpoch(operation.write_epoch);
	BindState(operation.state);
	auto& executor = m_renderer.GetRenderExecutor();
	executor.UseContextId(operation.state.context_id);
	if (operation.after_wait) {
		executor.InvalidateRenderTargetMemo();
	}
	if (operation.kind != GpuOperationKind::DrawIndex &&
	    operation.kind != GpuOperationKind::DrawAuto &&
	    operation.kind != GpuOperationKind::DrawIndirect) {
		executor.InvalidateRenderTargetMemo();
		// Guest-visible effects (labels, interrupts, flips, host commands) let the guest overwrite
		// what earlier draws read: their copies complete first. A dispatch has none (a submission
		// waits for the copies before it; with CpuCopyForward a dispatch also writes guest memory),
		// and waiting before each of them was 7.6% of this thread.
		const bool dispatch = operation.kind == GpuOperationKind::DispatchDirect ||
		                      operation.kind == GpuOperationKind::DispatchIndirect;
		// Neither have BDA boundaries, barriers and cache events (the occlusion dump is tagged
		// Other): two thirds of the callbacks in play, and the waits before them were 4.5% of this
		// thread.
		const bool invisible =
		    operation.kind == GpuOperationKind::Callback &&
		    (operation.callback_tag == GpuCallbackTag::Bda || operation.callback_tag == GpuCallbackTag::Barrier ||
		     operation.callback_tag == GpuCallbackTag::Event);
		if (auto* copies = GetScheduler().HostCopies();
		    copies != nullptr && !invisible &&
		    (!dispatch || Common::Experiments::Enabled(Common::Experiments::CpuCopyForward))) {
			if (StageStats::Enabled()) {
				const auto tag   = operation.kind == GpuOperationKind::Callback
				                       ? static_cast<size_t>(operation.callback_tag) & 15u
				                       : size_t {15};
				const auto start = std::chrono::steady_clock::now();
				const bool waits = copies->Pending();
				copies->Drain();
				StageStats::g_drain_count[tag].fetch_add(1, std::memory_order_relaxed);
				if (waits) {
					StageStats::g_drain_waits[tag].fetch_add(1, std::memory_order_relaxed);
					StageStats::g_drain_ns[tag].fetch_add(
					    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                              std::chrono::steady_clock::now() - start)
					                              .count()),
					    std::memory_order_relaxed);
				}
			} else {
				copies->Drain();
			}
		}
	}
	switch (operation.kind) {
		case GpuOperationKind::DrawIndex:
		case GpuOperationKind::DrawAuto: {
			executor.UseResolvedDraw(operation.resolved);
			if (operation.kind == GpuOperationKind::DrawIndex) {
				ExecuteDrawIndex(operation, operation.draw_index);
			} else {
				ExecuteDrawAuto(operation, operation.draw_auto);
			}
			executor.UseResolvedDraw(nullptr);
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		}
		case GpuOperationKind::DrawIndirect: ExecuteDrawIndirect(operation); break;
		case GpuOperationKind::DispatchDirect: {
			const auto& dispatch = operation.dispatch;
			ExecuteDispatchDirect(operation, dispatch.thread_group_x, dispatch.thread_group_y,
			                      dispatch.thread_group_z, dispatch.mode);
			break;
		}
		case GpuOperationKind::DispatchIndirect: ExecuteDispatchIndirect(operation); break;
		case GpuOperationKind::Callback:
			if (operation.callback) {
				operation.callback();
			}
			break;
	}
	if (operation.resolved_dispatch != nullptr) {
		RenderExecutor::ReleaseResolvedDispatch(operation.resolved_dispatch);
	}
	// Starts the copies of the operation while the next ones are prepared.
	if (auto* copies = GetScheduler().HostCopies(); copies != nullptr) {
		(void)copies->Flush();
	}
	if (m_renderer.GetBufferCache().TakeSubmitRequest()) {
		// Experiment EarlySubmitWriters: this operation wrote bytes a later resource read will
		// download; submitting now lets that read wait for this work only (second queue).
		GetScheduler().Flush();
		Common::Timeline::Mark("early-submit", static_cast<uint64_t>(operation.kind));
	}
}

void CommandProcessor::BindState(const GpuRegisterState& state) {
	GetScheduler().Begin(*state.context, *state.user_config, *state.shaders);
}

void CommandProcessor::BufferInit() {
	// Executing any operation binds its register state and begins a command buffer.
	if (m_pipeline == nullptr) {
		Emit(MakeOperation(GpuOperationKind::Callback));
	}
}

void CommandProcessor::BufferFlush(FlushSite site) {
	if (StageStats::Enabled()) {
		g_buffer_flush_emits[static_cast<size_t>(site)].fetch_add(1, std::memory_order_relaxed);
	}
	EmitCallback([this] {
		g_scheduler_flush_sites[0].fetch_add(1, std::memory_order_relaxed);
		GetScheduler().Flush();
	}, GpuCallbackTag::Flush);
}

void CommandProcessor::NoteRecordedWork() {
	const auto tick = GetScheduler().CurrentTick();
	if (tick != m_batch_tick) {
		m_batch_tick = tick;
		m_batch_work = 0;
	}
	m_batch_work++;
}

void CommandProcessor::BufferFlushIfBatchReady() {
	// Label writes reach guest memory when the packet is executed, and resources are captured
	// when a draw is recorded, so submitting here only starts the GPU earlier. A submission per
	// label costs a queue submit and a new command buffer, so wait for a batch of work.
	EmitCallback([this] { FlushIfBatchReady(); }, GpuCallbackTag::Flush);
}

void CommandProcessor::FlushIfBatchReady() {
	constexpr uint32_t MinBatchWork = 64;
	if (GetScheduler().CurrentTick() == m_batch_tick && m_batch_work >= MinBatchWork) {
		g_scheduler_flush_sites[1].fetch_add(1, std::memory_order_relaxed);
		GetScheduler().Flush();
	}
}

void CommandProcessor::BufferWait() {
	EmitCallback([this] { GetScheduler().Finish(); }, GpuCallbackTag::Finish);
}

void CommandProcessor::RunGarbageCollector() {
	EmitCallback([this] { m_renderer.RunGarbageCollector(); }, GpuCallbackTag::Gc);
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	if (FrameBreakdown::Enabled()) FrameBreakdown::SpecNotePktWrite(reinterpret_cast<uint64_t>(dst), static_cast<uint64_t>(dw_num) * 4);
	// The constant engine keeps writing its RAM; the dump carries the words it has now.
	std::vector<uint32_t> words(m_const_ram + offset / 4, m_const_ram + offset / 4 + dw_num);
	const auto            sequence = EmitCallback([dst, words = std::move(words)] {
		memcpy(dst, words.data(), words.size() * sizeof(uint32_t));
	}, GpuCallbackTag::WriteData);
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), uint64_t {dw_num} * 4u,
	               m_const_ram + offset / 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

bool CommandProcessor::TryPeekCleanBackingBytes(const volatile void* addr, void* out, uint64_t size) {
	// Read current bytes without a GPU->CPU download when they are not GPU-dirty. The
	// backing copy is authoritative for CPU-written bytes: a guest computation writes
	// its command stream, submits it, and only then does the CP consume it, so the CP
	// cannot be reading bytes a concurrent GPU write is still producing. Any exact-range
	// GPU write still refuses the peek, so the fault path keeps the final say. This is
	// the property that makes a false-sharing read cheap: a dirty neighbour on the same
	// 4 KiB page must not drag a whole-page download and queue drain behind it.
	const auto vaddr = reinterpret_cast<uint64_t>(addr);
	if (!GuestRange {vaddr, size}.Valid() || m_renderer.GetBufferCache().HasGpuDirtyBytes(vaddr, size) ||
	    !LibKernel::Memory::TryReadBacking(vaddr, out, size)) {
		return false;
	}
	return true;
}

bool CommandProcessor::TryReadProtected(const volatile void* address, void* out, uint64_t size) {
	return m_pipeline != nullptr &&
	       m_pipeline->TryReadProtectedForCp(reinterpret_cast<uint64_t>(address), out, size);
}

bool CommandProcessor::TryPeekGuest(const volatile void* addr, void* out, uint64_t size) {
	if (!BufferCache::PerfNoSyncReadback() || !GuestGpu::IsGpuThread()) {
		return false;
	}
	if (!TryPeekCleanBackingBytes(addr, out, size)) {
		return false;
	}
	BufferCache::PerfCount(BufferCache::PerfEvent::WaitRegMemBackingRead);
	return true;
}

// A WAIT_REG_MEM that keeps failing for the same memory: NHL 27 menus sometimes hang this way
// (the game then calls Coredump). Printed as NHL27STUCKWAIT after 2 s and every 5 s while stuck.
void CommandProcessor::NoteStuckWait(uint64_t address, uint64_t value, uint64_t ref, uint64_t mask,
                                     uint32_t func, uint32_t size) {
	// Per queue (command processor): queues that all wait take turns on the same thread.
	auto& stuck = m_stuck_wait;
	if (stuck.address != address || stuck.ref != ref || stuck.func != func) {
		stuck = {.address = address, .ref = ref, .func = func, .count = 0, .since = std::chrono::steady_clock::now(),
		         .printed = {}};
	}
	if ((++stuck.count & 255u) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - stuck.since < std::chrono::seconds(2) || now - stuck.printed < std::chrono::seconds(5)) {
		return;
	}
	stuck.printed      = now;
	const bool gpu_dirty = m_renderer.GetBufferCache().HasGpuDirtyBytes(address, size);
	std::printf("NHL27STUCKWAIT: addr=0x%llx size=%u func=%u ref=0x%llx mask=0x%llx value=0x%llx polls=%llu "
	            "for %.1f s queue=%u async=%d gpu_dirty=%d pending_writes=%llu\n",
	            static_cast<unsigned long long>(address), size, func, static_cast<unsigned long long>(ref),
	            static_cast<unsigned long long>(mask), static_cast<unsigned long long>(value),
	            static_cast<unsigned long long>(stuck.count),
	            std::chrono::duration<double>(now - stuck.since).count(), static_cast<unsigned>(m_interrupt_event_id), IsAsyncComputeQueue() ? 1 : 0,
	            gpu_dirty ? 1 : 0,
	            static_cast<unsigned long long>(m_pipeline != nullptr ? m_pipeline->PendingWriteCount() : 0));
	std::fflush(stdout);
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	m_wait_since_operation = true;
	// Experiment ClockWaitAhead: NHL 27 writes the GPU clock at end of pipe and waits for it to
	// be written. The value written is the reference clock when the write executes, so at least
	// the clock now; a test that holds for the clock now and for every later value (nonzero, or
	// at least / more than the reference) needs no wait for the execution.
	if constexpr (sizeof(T) == sizeof(uint64_t)) {
		const bool monotone = mask == ~T {0} && (func == 5 || func == 6 || (func == 4 && ref == 0));
		static const bool clock_wait_ahead = [] {
			const char* value = std::getenv("KYTY_CLOCK_WAIT_AHEAD");
			return value != nullptr && value[0] == '1';
		}();
		if (m_pipeline != nullptr && monotone && clock_wait_ahead &&
		    m_pipeline->PendingClockWrite(reinterpret_cast<uint64_t>(addr), sizeof(T)) &&
		    TestWaitRegMemValue(Sync::ReadReferenceClock(), ref, mask, func)) {
			if (BufferCache::GuestSyncOnly()) {
				EmitGlobalBarrier();
			}
			return;
		}
	}
	const bool timed       = FrameBreakdown::Enabled();
	const auto start       = timed ? DecisionNowNs() : 0;
	// Experiment ClockWaitPredict: a wait on a 64-bit reference-clock write that is still pending
	// (EOP timestamp) would stop the command processor until the execution thread catches up. The
	// clock only grows, so when the clock now already satisfies a wait for "at least" / "above" /
	// "not equal", the value written later satisfies it too.
	bool predicted = false;
	if constexpr (sizeof(T) == sizeof(uint64_t)) {
		uint64_t lower = 0;
		if (m_pipeline != nullptr && mask == ~T {0} &&
		    Common::Experiments::Enabled(Common::Experiments::ClockWaitPredict) &&
		    m_pipeline->TryPredictClockWrite(addr, sizeof(T), lower)) {
			predicted = func == 0 || (func == 5 && lower >= ref) || ((func == 6 || func == 4) && lower > ref);
		}
	}
	if (predicted && timed) {
		g_decision.predicted.fetch_add(1, std::memory_order_relaxed);
	}
	const auto value       = predicted ? T {0} : ReadDecisionValue<T>(addr);
	const auto read_done   = timed ? DecisionNowNs() : 0;
	const bool satisfied   = predicted || TestWaitRegMemValue(value, ref, mask, func);
	if (!satisfied) {
		NoteStuckWait(reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(value), ref, mask, func, sizeof(T));
		SuspendPm4();
	}
	// KYTY_GUEST_SYNC: labels are written at CP time, so a guest RELEASE_MEM + WAIT_REG_MEM pair
	// no longer orders GPU work by itself. Without the automatic dispatch barriers the wait must.
	if (BufferCache::GuestSyncOnly()) {
		EmitGlobalBarrier();
	}
	if (timed) {
		const auto done = DecisionNowNs();
		g_decision.wrm_calls.fetch_add(1, std::memory_order_relaxed);
		g_decision.wrm_suspends.fetch_add(satisfied ? 0 : 1, std::memory_order_relaxed);
		g_decision.wrm_ns.fetch_add(done - start, std::memory_order_relaxed);
		g_decision.wrm_read_ns.fetch_add(read_done - start, std::memory_order_relaxed);
		g_decision.wrm_barrier_ns.fetch_add(done - read_done, std::memory_order_relaxed);
		DecisionMaybePrint();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}
	if (FrameBreakdown::Enabled()) {
		FrameBreakdown::SpecNotePktWrite(reinterpret_cast<uint64_t>(dst),
		                                 write_one_address ? sizeof(uint32_t)
		                                                   : static_cast<uint64_t>(dw_num) * sizeof(uint32_t));
	}

	// Labels commonly share a protected page with GPU-written data. Update both
	// copies through the existing exact-range label path, without downloading or
	// unprotecting unrelated bytes. Read each source before its corresponding
	// repeated-address write: source and destination may alias.
	if (Common::RuntimeFlags::Current().pm4_write_data_fast && GuestGpu::IsGpuThread()) {
		auto& cache = m_renderer.GetBufferCache();
		if (write_one_address) {
			for (uint32_t i = 0; i < dw_num; i++) {
				const uint32_t value = src[i];
				if (!cache.TryWriteLabel(reinterpret_cast<uint64_t>(dst), &value, sizeof(value))) {
					dst[0] = value;
				}
			}
			return;
		}
		if (dw_num <= 64) {
			std::array<uint32_t, 64> values{};
			std::memcpy(values.data(), src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
			if (cache.TryWriteLabel(reinterpret_cast<uint64_t>(dst), values.data(),
			                        static_cast<uint64_t>(dw_num) * sizeof(uint32_t))) {
				return;
			}
		}
	}

	// The source dwords are in the command buffer, which stays valid until the guest sees a
	// later label.
	const auto sequence = EmitCallback([this, dst, src, dw_num, write_one_address] {
		BufferCache::WriteTagScope tag(BufferCache::WriteTag::WriteData);
		// The pipelined command processor's writes reach here: they take the label path as well
		// (both copies updated, nothing downloaded), not PrepareGpuWrite, which downloads the
		// page's GPU-written bytes and waited for their writer, 6-23 ms about once a second in
		// play (AbPending4: always PrepareGpuWrite).
		if (Common::RuntimeFlags::Current().pm4_write_data_fast &&
		    !Common::Experiments::Enabled(Common::Experiments::AbPending4)) {
			auto& cache = m_renderer.GetBufferCache();
			if (write_one_address) {
				uint32_t written = 0;
				while (written < dw_num) {
					const uint32_t value = src[written];
					if (!cache.TryWriteLabel(reinterpret_cast<uint64_t>(dst), &value, sizeof(value))) {
						break;
					}
					written++;
				}
				if (written == dw_num) {
					return;
				}
				m_renderer.PrepareGpuWrite(dst, sizeof(uint32_t));
				for (uint32_t i = written; i < dw_num; i++) {
					dst[0] = src[i];
				}
				return;
			}
			if (dw_num <= 64) {
				std::array<uint32_t, 64> values {};
				std::memcpy(values.data(), src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
				if (cache.TryWriteLabel(reinterpret_cast<uint64_t>(dst), values.data(),
				                        static_cast<uint64_t>(dw_num) * sizeof(uint32_t))) {
					return;
				}
			}
		}
		m_renderer.PrepareGpuWrite(dst, uint64_t {write_one_address ? 1u : dw_num} * sizeof(uint32_t));
		if (write_one_address) {
			for (uint32_t i = 0; i < dw_num; i++) {
				dst[0] = src[i];
			}
		} else {
			memcpy(dst, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
		}
	}, GpuCallbackTag::WriteData);
	if (write_one_address) {
		NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), sizeof(uint32_t),
		               src + dw_num - 1);
	} else {
		NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst),
		               uint64_t {dw_num} * sizeof(uint32_t), src);
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	if (FrameBreakdown::Enabled()) FrameBreakdown::SpecNotePktWrite(dst_address, num_bytes);
	const auto sequence = EmitCallback([dst_address, num_bytes] {
		const auto value = Sync::ReadReferenceClock();
		std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
		static std::atomic<uint32_t> clock_log_count {0};
		if (clock_log_count.fetch_add(1) < 64) {
			LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
			     " size=%u\n",
			     dst_address, value, num_bytes);
		}
	}, GpuCallbackTag::Clock);
	NoteGuestWrite(sequence, dst_address, num_bytes, nullptr, "wait-clock-copy", num_bytes == sizeof(uint64_t) ? GuestGpu::WriteOriginClock : GuestGpu::WriteOriginOther);
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	bool src_gds = false;
	if (src_sel != 2) {
		if (!decode_gds(src_sel, src_gds)) {
			EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
		}
		if (src_gds && dst_gds) {
			EXIT("unsupported dmaData GDS-to-GDS copy\n");
		}
	}
	const auto sequence = EmitCallback([=, this] {
		ExecuteDmaData(dst_gds, dst_address_or_offset, src_sel, src_gds,
		               src_address_or_offset_or_immediate, num_bytes);
	}, GpuCallbackTag::Dma);
	if (!dst_gds) {
		NoteGuestWrite(sequence, dst_address_or_offset, num_bytes, nullptr, "wait-dma-data", GuestGpu::WriteOriginDma);
	}
}

void CommandProcessor::ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset,
                                      uint8_t src_sel, bool src_gds,
                                      uint64_t src_address_or_offset_or_immediate,
                                      uint32_t num_bytes) {
	if (!dst_gds && FrameBreakdown::Enabled()) {
		FrameBreakdown::SpecNotePktWrite(dst_address_or_offset, num_bytes);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	{
		// Preserve early packet order, then report newly seen GDS regions throughout the run.
		static std::mutex gds_log_mutex;
		static uint64_t gds_packets = 0;
		static std::vector<std::array<uint64_t, 5>> gds_regions;
		if (dst_gds || src_sel == 1) {
			std::lock_guard lock(gds_log_mutex);
			const std::array<uint64_t, 5> region {
			    dst_gds ? 1u : 0u, src_sel, dst_gds ? dst_address_or_offset : 0,
			    src_sel == 1 || src_sel == 2 ? src_address_or_offset_or_immediate : 0,
			    num_bytes};
			const bool unseen = std::find(gds_regions.begin(), gds_regions.end(), region) ==
			                    gds_regions.end();
			const bool report_new = unseen && gds_regions.size() < 256;
			if (report_new) gds_regions.push_back(region);
			const auto packet = ++gds_packets;
			// KYTY_DBG_GDS_TRACE=1: also log every DMA touching GDS bytes 0..15 (rate-limited).
			static const bool gds_trace = [] {
				const char* v = std::getenv("KYTY_DBG_GDS_TRACE");
				return v != nullptr && v[0] == '1';
			}();
			static uint64_t gds_low_packets = 0;
			bool            low_gds = false;
			if (gds_trace && dst_gds && dst_address_or_offset < 16) {
				const auto n = ++gds_low_packets;
				low_gds      = n <= 400 || n % 50 == 0;
			}
			if (packet <= 48 || report_new || low_gds) {
				std::printf("NHL26GDSW: dma dst_gds=%u src_sel=%u dst=0x%llx src=0x%llx bytes=%u\n",
				            dst_gds ? 1u : 0u, static_cast<uint32_t>(src_sel),
				            static_cast<unsigned long long>(dst_address_or_offset),
				            static_cast<unsigned long long>(src_address_or_offset_or_immediate),
				            static_cast<uint32_t>(num_bytes));
				std::fflush(stdout);
			}
		}
	}
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

namespace {
// Draws and dispatches the GPU thread has processed (KYTY_DBG_QUEUE_TIME reports time per item).
uint64_t g_work_items = 0;
// The guest queue of the submission being processed (GPU thread only).
uint32_t g_current_queue = 0;
// Guest barriers emitted, and how many of them ended a render pass (KYTY_DBG_QUEUE_TIME).
uint64_t g_guest_barriers = 0, g_guest_barrier_breaks = 0;

// KYTY_DBG_QUEUE_TIME=1: GPU-thread time spent processing each guest queue (0 = graphics,
// 1.. = compute rings), printed every 5 s as NHL27QUEUES.
struct QueueTimeScope {
	static bool Enabled() {
		static const bool enabled = std::getenv("KYTY_DBG_QUEUE_TIME") != nullptr;
		return enabled;
	}
	explicit QueueTimeScope(uint32_t queue): queue(queue) {
		if (Enabled()) start = std::chrono::steady_clock::now();
	}
	~QueueTimeScope() {
		if (!Enabled()) return;
		static std::array<uint64_t, 64> ns {};
		static std::array<uint64_t, 64> slices {};
		if (queue >= ns.size()) return;
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		ns[queue] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
		slices[queue]++;
		if (now - last < std::chrono::seconds(5)) return;
		const double seconds = std::chrono::duration<double>(now - last).count();
		last = now;
		uint64_t total_ns = 0;
		for (const auto value: ns) total_ns += value;
		static uint64_t last_items = 0;
		const uint64_t  items      = g_work_items - last_items;
		last_items                 = g_work_items;
		static uint64_t last_barriers = 0, last_breaks = 0;
		std::string line = fmt::format("NHL27QUEUES mask={} items/s={:.0f} us/item={:.2f} barriers/s={:.0f} pass_breaks/s={:.0f} ms/s:",
		                               static_cast<uint64_t>(Common::Experiments::Mask()),
		                               static_cast<double>(items) / seconds,
		                               items != 0 ? static_cast<double>(total_ns) / 1e3 / static_cast<double>(items) : 0.0,
		                               static_cast<double>(g_guest_barriers - last_barriers) / seconds,
		                               static_cast<double>(g_guest_barrier_breaks - last_breaks) / seconds);
		last_barriers = g_guest_barriers;
		last_breaks   = g_guest_barrier_breaks;
		for (uint32_t i = 0; i < ns.size(); i++) {
			if (slices[i] == 0) continue;
			line += fmt::format(" q{}={:.1f}({:.0f})", i, static_cast<double>(ns[i]) / 1e6 / seconds,
			                    static_cast<double>(slices[i]) / seconds);
		}
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
		ns     = {};
		slices = {};
	}
	uint32_t                              queue;
	std::chrono::steady_clock::time_point start {};
};
} // namespace

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = !gpu->m_pipelined;
	g_cp_thread  = gpu->m_pipelined;
	RegionManager::MarkGpuOrderedThread();
	g_gpu_state  = gpu;
	if (gpu->m_pipelined) {
		Config::ConfigureGpuStageThread(Config::GpuStageThread::CommandProcessor);
	}

	// When every queue waits (WAIT_REG_MEM), a pipelined command processor retries without
	// sleeping until the spin budget is spent.
	std::optional<std::chrono::steady_clock::time_point> blocked_since;
	bool                                                  spin_blocked = false;

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		if (spin_blocked) {
			StageStats::CpStateScope state(StageStats::CpState::Blocked);
			spin_blocked      = false;
			const auto epoch  = gpu->m_work_epoch.load(std::memory_order_acquire);
			const auto target = std::chrono::steady_clock::now() + BlockedRetryInterval;
			while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch &&
			       std::chrono::steady_clock::now() < target) {
				Common::SpinPause();
			}
		}
		if (gpu->m_pipelined) {
			// The GPU is idle once the execution thread ran every operation too. Only this
			// thread emits operations, so none can follow the wait until it finds work.
			bool     no_work = false;
			uint64_t epoch   = 0;
			{
				Common::LockGuard lock(gpu->m_queue_mutex);
				no_work = gpu->m_commands.empty() && gpu->m_submission_count == 0 &&
				          !gpu->m_stopping;
			}
			if (no_work) {
				StageStats::CpStateScope state(StageStats::CpState::NoWork);
				gpu->WaitForExecution();
				bool arrived = false;
				{
					Common::LockGuard lock(gpu->m_queue_mutex);
					epoch = gpu->m_work_epoch.load(std::memory_order_acquire);
					if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
						gpu->m_processing = false;
						gpu->m_idle.SignalAll();
					} else {
						arrived = true;
					}
				}
				// Spin for the next submission; sleep below only when none came. Work queued
				// during WaitForExecution already moved the epoch read above, so spinning for
				// the epoch to move again would wait out the whole spin budget (2 ms) although
				// the work is waiting. Experiment CpSkipIdleSpin: start that work at once.
				const bool skip = arrived && Common::Experiments::Enabled(Common::Experiments::CpSkipIdleSpin);
				const auto spin_start = arrived ? std::chrono::steady_clock::now()
				                                : std::chrono::steady_clock::time_point {};
				if (!skip) {
					Common::SpinWait spin;
					while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch && spin.Spin()) {
					}
				}
				if (arrived && FrameBreakdown::Enabled()) {
					// NHL27CPSPIN: how often work was already queued after the wait, and the spin
					// time spent on it (none when the experiment is on).
					static std::atomic<uint64_t> s_arrived {0};
					static std::atomic<uint64_t> s_spin_ns {0};
					static std::atomic<int64_t>  s_last_print {0};
					s_arrived.fetch_add(1, std::memory_order_relaxed);
					s_spin_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                        std::chrono::steady_clock::now() - spin_start).count()),
					                    std::memory_order_relaxed);
					const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					                        std::chrono::steady_clock::now().time_since_epoch()).count();
					const auto last = s_last_print.load(std::memory_order_relaxed);
					if (last == 0) {
						s_last_print.store(now_ms, std::memory_order_relaxed);
					} else if (now_ms - last >= 5000) {
						s_last_print.store(now_ms, std::memory_order_relaxed);
						std::printf("NHL27CPSPIN: window=%.1fs queued_after_wait=%llu spin_ms=%.1f skip=%d\n",
						            static_cast<double>(now_ms - last) / 1000.0,
						            static_cast<unsigned long long>(s_arrived.exchange(0, std::memory_order_relaxed)),
						            static_cast<double>(s_spin_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
						            skip ? 1 : 0);
						std::fflush(stdout);
					}
				}
			}
		}
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			StageStats::CpStateScope idle_state(StageStats::CpState::NoWork);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				Common::Timeline::Mark("gpu-idle");
				const auto idle_start = std::chrono::steady_clock::now();
				Common::EventTrace::Record(Common::EventTrace::GPU_WAIT_BEGIN);
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				Common::EventTrace::Record(Common::EventTrace::GPU_WAIT_END,
				                           gpu->m_commands.size() + gpu->m_submission_count);
				FrameBreakdown::AddIdleGap(static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - idle_start)
				        .count()));
				Common::Timeline::Mark("gpu-wake");
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					const auto now    = std::chrono::steady_clock::now();
					if (gpu->m_pipelined && !blocked_since) {
						blocked_since = now;
					}
					if (gpu->m_pipelined && now - *blocked_since < Common::SpinBudget()) {
						spin_blocked = true;
					} else {
						Common::Timeline::Mark("gpu-blocked");
						StageStats::CpStateScope state(StageStats::CpState::Blocked);
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
						Common::Timeline::Mark("gpu-unblocked");
					}
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			if (gpu->m_pipelined) {
				gpu->StopExecution();
			}
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			g_cp_thread  = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_BEGIN, 1, 0);
			if (gpu->m_pipelined) {
				// Stays processing until the command executed; see the idle wait above.
				gpu->m_gfx_cp->BdaBoundary(RenderContext::BdaBoundary::Other);
				gpu->m_gfx_cp->EmitHostCommand(std::move(command));
				continue;
			}
			command();
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_END, 1, 0);

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		Common::Timeline::Mark("slice-begin", submission.queue_id * 10u + static_cast<uint32_t>(submission.type));
		bool complete = false;
		{
			// KYTY_PERF_STATS: GPU thread busy = Process wall time minus its master-semaphore waits.
			FrameBreakdown::Timer process_timer(FrameBreakdown::g_counters.gpu_process_ns);
			// NHL27GPUBUSY: exclusive bucket attribution over the same window.
			struct GpuBusyWindow {
				GpuBusyWindow() { FrameBreakdown::GpuBusyBegin(); }
				~GpuBusyWindow() { FrameBreakdown::GpuBusyEnd(); }
			} gpu_busy_window;
			// kind: 2 + SubmissionType (graphics/compute/flip prep/suspend point), +0x100 when sliced/blocked
			const uint64_t kind = 2 + static_cast<uint64_t>(submission.type);
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_BEGIN, kind, gpu->m_submit_id);
			QueueTimeScope queue_time(submission.queue_id);
			g_current_queue = submission.queue_id;
			complete = gpu->Process(submission);
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_END, kind | (complete ? 0u : 0x100u),
			                           gpu->m_submit_id);
		}
		Common::Timeline::Mark(complete ? "slice-done" : "slice-blocked", submission.queue_id);
		FrameBreakdown::MaybePrint();
		if (complete) {
			blocked_since.reset();
		}

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		if (gpu->m_pipelined) {
			// Stays processing until the operations executed; see the idle wait above.
			continue;
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

// KYTY_SUBMIT_SPLIT: while recording one large guest command buffer, submit the native command
// buffer after enough draws/dispatches or recording time so the host GPU can start earlier.
// GPU-thread only. Policy (kept simple): split at packet boundaries right after a draw/dispatch
// packet completes, regardless of render-pass state. Submit() ends dynamic rendering and the next
// draw re-begins it, exactly like the existing Flip()/WaitFlipDone()/slice-end flushes.
struct SubmitSplitState {
	bool                                  enabled = false;
	uint32_t                              work    = 0;
	std::chrono::steady_clock::time_point since {};
};
static thread_local SubmitSplitState g_submit_split;
static std::atomic<uint64_t>         g_submit_split_count {0};

// Split thresholds: env KYTY_SUBMIT_SPLIT_DRAWS / KYTY_SUBMIT_SPLIT_US set the defaults; a
// "split.ctl" file in the working directory (lines "draws=N", "us=T") overrides them and is
// re-read every 500 ms so the thresholds can be tuned while the game runs.
static std::atomic<uint32_t> g_split_draws {0};
static std::atomic<int64_t>  g_split_us {0};

static void SubmitSplitRefresh() {
	// Measured best on an RX 9070 XT: 768 draws / 12 ms when each split costs this thread a
	// vkQueueSubmit, 48 draws / 0.75 ms with experiment AsyncSubmit (+4.5% paused match). With the
	// recording thread submitting and guest command buffers batched, 192 draws / 3 ms (in-boot A/B
	// through split.ctl, 48 / 192 / 768 draws: 433 / 455 / 451 fps in paired rounds).
	const bool            async     = Common::Experiments::Enabled(Common::Experiments::AsyncSubmit);
	static const uint32_t env_draws = [] {
		const char* s = std::getenv("KYTY_SUBMIT_SPLIT_DRAWS");
		const long  n = s != nullptr ? std::atol(s) : 0;
		return n > 0 ? static_cast<uint32_t>(n) : 0u;
	}();
	static const int64_t env_us = [] {
		const char* s = std::getenv("KYTY_SUBMIT_SPLIT_US");
		const long  n = s != nullptr ? std::atol(s) : 0;
		return n > 0 ? static_cast<int64_t>(n) : int64_t {0};
	}();
	static auto last = std::chrono::steady_clock::time_point {};
	const auto  now  = std::chrono::steady_clock::now();
	if (last != std::chrono::steady_clock::time_point {} && now - last < std::chrono::milliseconds(500)) {
		return;
	}
	last           = now;
	uint32_t draws = env_draws != 0 ? env_draws : async ? 192u : 768u;
	int64_t  us    = env_us != 0 ? env_us : async ? int64_t {3000} : int64_t {12000};
	if (FILE* file = std::fopen("split.ctl", "r"); file != nullptr) {
		char line[64];
		while (std::fgets(line, sizeof(line), file) != nullptr) {
			long value = 0;
			if (std::sscanf(line, "draws=%ld", &value) == 1 && value > 0) {
				draws = static_cast<uint32_t>(value);
			} else if (std::sscanf(line, "us=%ld", &value) == 1 && value > 0) {
				us = value;
			}
		}
		std::fclose(file);
	}
	g_split_draws.store(draws, std::memory_order_relaxed);
	g_split_us.store(us, std::memory_order_relaxed);
}

static uint32_t SubmitSplitDraws() {
	return g_split_draws.load(std::memory_order_relaxed);
}

static int64_t SubmitSplitUs() {
	return g_split_us.load(std::memory_order_relaxed);
}

static bool SubmitSplitIsWorkOpcode(uint32_t op) {
	switch (op) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

// NHL27GPUBUSY: which exclusive bucket a PM4 packet handler is charged to.
static FrameBreakdown::GpuBucket PacketBusyBucket(uint32_t op) {
	switch (op) {
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW: return FrameBreakdown::GB_DRAW_HANDLER;
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return FrameBreakdown::GB_DISPATCH_HANDLER;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_MEM_SEMAPHORE:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_SURFACE_SYNC:
		case Pm4::IT_ACQUIRE_MEM: return FrameBreakdown::GB_SYNC_PKT;
		case Pm4::IT_DMA_DATA:
		case Pm4::IT_CP_DMA:
		case Pm4::IT_COPY_DATA: return FrameBreakdown::GB_DMA_PKT;
		default: return FrameBreakdown::GB_HANDLER;
	}
}

// NHL27PM4OPS (KYTY_PERF_STATS=1): time spent inside each PM4 packet handler, by opcode (IT_NOP by
// its custom R code), so the sync_pkt bucket of NHL27GPUBUSY can be broken down. Totals per 5 s.
namespace {
struct Pm4OpStat {
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> count {0};
};
constexpr size_t kPm4OpStatSlots = 256 + Pm4::R_NUM; // opcodes, then IT_NOP custom R codes
Pm4OpStat        g_pm4_op_stats[kPm4OpStatSlots];
std::atomic<int64_t> g_pm4_op_last_print {0};

void Pm4OpStatsAdd(uint32_t opcode, uint32_t header, uint64_t ns) {
	const size_t slot = opcode == Pm4::IT_NOP ? 256u + KYTY_PM4_R(header) : opcode;
	g_pm4_op_stats[slot].ns.fetch_add(ns, std::memory_order_relaxed);
	g_pm4_op_stats[slot].count.fetch_add(1, std::memory_order_relaxed);
	const auto now  = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
	auto       last = g_pm4_op_last_print.load(std::memory_order_relaxed);
	if (last == 0) {
		g_pm4_op_last_print.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 5000 || !g_pm4_op_last_print.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	struct Entry {
		uint64_t ns;
		uint64_t count;
		size_t   slot;
	};
	std::vector<Entry> top;
	for (size_t i = 0; i < kPm4OpStatSlots; i++) {
		const auto t = g_pm4_op_stats[i].ns.exchange(0, std::memory_order_relaxed);
		const auto c = g_pm4_op_stats[i].count.exchange(0, std::memory_order_relaxed);
		if (c != 0) {
			top.push_back({t, c, i});
		}
	}
	std::sort(top.begin(), top.end(), [](const Entry& a, const Entry& b) { return a.ns > b.ns; });
	std::printf("NHL27PM4OPS: window=%.1fs", static_cast<double>(now - last) / 1000.0);
	for (size_t i = 0; i < top.size() && i < 12; i++) {
		const auto t    = top[i].ns;
		const auto c    = top[i].count;
		const auto slot = top[i].slot;
		if (slot >= 256) {
			std::printf(" | nop_r=0x%02zx n=%llu ms=%.1f", slot - 256, static_cast<unsigned long long>(c),
			            static_cast<double>(t) / 1e6);
		} else {
			std::printf(" | op=0x%02zx n=%llu ms=%.1f", slot, static_cast<unsigned long long>(c),
			            static_cast<double>(t) / 1e6);
		}
	}
	std::printf("\n");
	std::fflush(stdout);
}
} // namespace

static void SubmitSplitMaybePrint() {
	static const bool stats = [] {
		const char* s = std::getenv("KYTY_PERF_STATS");
		return s != nullptr && s[0] == '1';
	}();
	if (!stats) return;
	static auto     last = std::chrono::steady_clock::now();
	static uint64_t last_count = 0;
	const auto      now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	const auto count = g_submit_split_count.load(std::memory_order_relaxed);
	std::printf("NHL27SPLIT: splits=%llu delta=%llu draws_thr=%u us_thr=%lld\n",
	            static_cast<unsigned long long>(count),
	            static_cast<unsigned long long>(count - last_count), SubmitSplitDraws(),
	            static_cast<long long>(SubmitSplitUs()));
	std::fflush(stdout);
	last = now;
	last_count = count;
}

// KYTY_BDA_BATCH: PM4 packets after which guest memory may be ordered for the GPU or CPU. WRITE_DATA, COPY_DATA
// and draws/dispatches are deliberately not boundaries: they do not order CPU-written memory for the GPU.
static bool BdaBoundaryKind(uint32_t opcode, uint32_t header, RenderContext::BdaBoundary& kind) {
	using B = RenderContext::BdaBoundary;
	switch (opcode) {
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_WAIT_ON_CE_COUNTER:
		case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF: kind = B::Wait; return true;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS: kind = B::Event; return true;
		case Pm4::IT_ACQUIRE_MEM: kind = B::Acquire; return true;
		case Pm4::IT_DMA_DATA: kind = B::Dma; return true;
		case Pm4::IT_PFP_SYNC_ME: kind = B::Sync; return true;
		case Pm4::IT_INDIRECT_BUFFER: kind = B::Ib; return true;
		case Pm4::IT_NOP:
			switch (KYTY_PM4_R(header)) {
				case Pm4::R_WAIT_FLIP_DONE: kind = B::Wait; return true;
				case Pm4::R_ACQUIRE_MEM: kind = B::Acquire; return true;
				case Pm4::R_RELEASE_MEM: kind = B::Event; return true;
				case Pm4::R_FLIP: kind = B::Flip; return true;
				default: return false;
			}
		default: return false;
	}
}

// The end of a guest command buffer submits the host one only once it holds a batch of work
// (about 1,400 guest submissions a second); frame ends, interrupts and command-buffer splits still
// submit, and a readback of the open command buffer submits it.
static void FlushGuestSubmission(CommandProcessor& cp) {
	cp.BufferFlushIfBatchReady();
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);
	// Start of each PM4 submission slice (graphics, compute queue, flip preparation, suspend point).
	cp.BdaBoundary(submission.type == SubmissionType::FlipPreparation ? RenderContext::BdaBoundary::Flip
	                                                                   : RenderContext::BdaBoundary::Submit);

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	g_submit_split.enabled = Common::RuntimeFlags::Current().submit_split;
	if (g_submit_split.enabled) {
		SubmitSplitRefresh();
	}
	g_submit_split.work    = 0;
	g_submit_split.since   = {};
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				FlushGuestSubmission(cp);
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				FlushGuestSubmission(cp);
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			// Age caches at the frame boundary, not once per command list.
			cp.RunGarbageCollector();
			cp.EmitGlobalBarrier();
			// Registered at the operation's place in the stream: a pipelined command processor
			// runs ahead of the recorded commands. Releasing the slot only touches host state.
			cp.EmitHostCommand([this, ready = m_suspend_point_ready] {
				m_renderer.GetCommandScheduler().DeferHostOperation([ready] {
					Common::EventTrace::Record(Common::EventTrace::SUSPEND_RELEASE);
					ready->release();
				}, true);
			});
			cp.BufferFlush(CommandProcessor::FlushSite::Suspend);
			cp.Reset();
			break;
	}

	g_submit_split.enabled = false;
	SubmitSplitMaybePrint();
	cp.BdaBoundary(RenderContext::BdaBoundary::Submit);
	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	BufferCache::PerfTimerScope phase_timer(BufferCache::PerfTimer::Pm4);
	ProcessPm4(execution);
	phase_timer.Stop();
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	if (FrameBreakdown::Enabled()) FrameBreakdown::SpecNoteWait();
	g_current_execution->m_suspended = true;
}

// Experiment FalseShareReads: a compute dispatch can write into data embedded in the command
// buffer itself, which protects the whole page. Reading the following packets on that page
// would then fault and wait for all queued GPU work. When the page holds GPU-dirty bytes but the
// packet's own bytes are clean, parse a copy taken from the backing instead.
// Experiment CpBackingReads: the pipelined command processor's packets on a read-protected page
// (NHL 27 embeds labels the GPU writes in its command buffers) are parsed from a copy of the
// backing, checked in order like GuestGpu::TryReadProtectedForCp, instead of faulting into a
// download that waits for the label's writer.
const uint32_t* CommandProcessor::ShadowProtectedPm4Packet(const uint32_t* packet, uint32_t remaining_dw) {
	if (m_pipeline == nullptr || remaining_dw == 0 || !m_pipeline->CpBackingReadsOn()) {
		return packet;
	}
	const auto address = reinterpret_cast<uint64_t>(packet);
	const auto page    = address & ~uint64_t {0xfff};
	// The execution thread protects a page the moment it records the GPU write next to the
	// packets the CP is parsing, so only "protected" is remembered for the page (a page that stops
	// being protected is still read correctly from the backing; one that becomes protected after
	// the check faults as before).
	if (page != m_cp_shadow_page || !m_cp_shadow_protected) {
		m_cp_shadow_page      = page;
		m_cp_shadow_protected = m_renderer.IsReadProtected(address, 8);
	}
	if (!m_cp_shadow_protected) {
		return packet;
	}
	uint32_t header[2] {};
	if (!LibKernel::Memory::TryReadBacking(address, header, std::min<uint32_t>(2u, remaining_dw) * 4u)) {
		return packet;
	}
	uint32_t length = header[0] == 0x80000000u
	                      ? 1u
	                      : std::min<uint32_t>(KYTY_PM4_LEN(header[0]), remaining_dw);
	// A plain NOP wraps embedded data, often the very bytes the GPU writes. Its handler reads
	// only the first payload dword unless the NOP carries a marker string.
	if (((header[0] >> 8u) & 0xffu) == Pm4::IT_NOP && KYTY_PM4_R(header[0]) == Pm4::R_ZERO &&
	    length > 2u && (header[1] & 0xffff0000u) != 0x68750000u) {
		length = 2u;
	}
	m_pm4_shadow.resize(length);
	if (!LibKernel::Memory::TryReadBacking(address, m_pm4_shadow.data(), length * 4u)) {
		return packet;
	}
	m_pipeline->CheckCpBackingRead(*this, address, m_pm4_shadow.data(), length * 4u);
	return m_pm4_shadow.data();
}

const uint32_t* CommandProcessor::ShadowPm4Packet(const uint32_t* packet, uint32_t remaining_dw) {
	if (!GuestGpu::IsGpuThread() || remaining_dw == 0) {
		return packet;
	}
	auto&          buffers    = m_renderer.GetBufferCache();
	const auto     address    = reinterpret_cast<uint64_t>(packet);
	const auto     page       = address & ~uint64_t {0xfff};
	const uint64_t generation = buffers.GpuDirtyGeneration();
	if (page != m_pm4_page || generation != m_pm4_page_generation) {
		m_pm4_page            = page;
		m_pm4_page_generation = generation;
		m_pm4_page_dirty = GuestRange {page, 0x1000}.Valid() && buffers.HasGpuDirtyBytes(page, 0x1000);
	}
	if (!m_pm4_page_dirty) {
		return packet;
	}
	const auto read = [&](uint64_t offset_dw, uint32_t* out, uint64_t count_dw) {
		const auto at    = address + offset_dw * 4u;
		const auto bytes = count_dw * 4u;
		return !buffers.HasGpuDirtyBytes(at, bytes) &&
		       LibKernel::Memory::TryReadBacking(at, out, bytes);
	};
	uint32_t header[2] {};
	if (!read(0, header, std::min<uint32_t>(2u, remaining_dw))) {
		return packet;
	}
	uint32_t length = header[0] == 0x80000000u
	                      ? 1u
	                      : std::min<uint32_t>(KYTY_PM4_LEN(header[0]), remaining_dw);
	// A plain NOP wraps embedded data, often the very bytes the GPU writes. Its handler reads
	// only the first payload dword unless the NOP carries a marker string.
	if (((header[0] >> 8u) & 0xffu) == Pm4::IT_NOP && KYTY_PM4_R(header[0]) == Pm4::R_ZERO &&
	    length > 2u && (header[1] & 0xffff0000u) != 0x68750000u) {
		length = 2u;
	}
	m_pm4_shadow.resize(length);
	if (!read(0, m_pm4_shadow.data(), length)) {
		return packet;
	}
	BufferCache::PerfCount(BufferCache::PerfEvent::Pm4ShadowPacket);
	return m_pm4_shadow.data();
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	static const bool clean_backing = [] {
		const char* setting = std::getenv("KYTY_PM4_CLEAN_BACKING");
		return setting != nullptr && setting[0] == '1';
	}();
	// Only bypass exact-clean header reads. Some register handlers fuse adjacent packets,
	// so their payload pointer and available stream length must remain the original ones.
	struct FaultScopeRestore {
		Pm4FaultScope previous = g_pm4_fault_scope;
		~FaultScopeRestore() { g_pm4_fault_scope = previous; }
	} fault_scope_restore;
	while (!execution.m_buffer_stack.empty()) {
		g_pm4_fault_scope = {};
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			if (execution.m_buffer_stack.size() > 1) {
				BdaBoundary(RenderContext::BdaBoundary::Ib);  // return edge of an indirect buffer call
			}
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const guest_packet  = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		// KYTY_FALSE_SHARE_READS shadows packets on the GPU thread only (the unpipelined design).
		const auto* const shadow =
		    Common::Experiments::FalseShareReads() ? ShadowPm4Packet(guest_packet, remaining_dw) : guest_packet;
		const auto* const packet =
		    shadow != guest_packet ? shadow : ShadowProtectedPm4Packet(guest_packet, remaining_dw);
		g_pm4_fault_scope = {reinterpret_cast<uint64_t>(packet), 1, 0, false};
		uint32_t packet_header = 0;
		if (!clean_backing || !TryPeekGuest(packet, &packet_header, sizeof(packet_header))) {
			packet_header = packet[0];
		}
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		g_pm4_fault_scope = {reinterpret_cast<uint64_t>(packet),
		                     std::min<uint32_t>(KYTY_PM4_LEN(packet_header), remaining_dw),
		                     packet_header, true};
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);
		const auto declared_dw = KYTY_PM4_LEN(packet_header);
		EXIT_NOT_IMPLEMENTED(declared_dw > remaining_dw);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(guest_packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(guest_packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = guest_packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		BufferCache::PerfSetPm4Op(opcode);
		g_trace_op = static_cast<uint16_t>(opcode | (opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet_header) << 8u : 0u));
		uint32_t packet_dw;
		{
			FrameBreakdown::GpuBusyScope busy_scope(PacketBusyBucket(opcode));
			const bool op_timed = FrameBreakdown::Enabled();
			const auto op_start = op_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			if (op_timed) {
				Pm4OpStatsAdd(opcode, packet_header, static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - op_start).count()));
			}
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (RenderContext::BdaBoundary bda_kind; BdaBoundaryKind(opcode, packet_header, bda_kind)) {
			BdaBoundary(bda_kind);  // after the handler, also when the packet suspended on a wait
		}
		g_trace_op = 0;
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (g_submit_split.enabled && SubmitSplitIsWorkOpcode(opcode)) {
			// Packet fully handled and cursor advanced: a flush here cannot separate a
			// draw/dispatch from any later EOP/label packet in the stream.
			const auto now = std::chrono::steady_clock::now();
			if (g_submit_split.work++ == 0) {
				g_submit_split.since = now;
			} else if (g_submit_split.work >= SubmitSplitDraws() ||
			           std::chrono::duration_cast<std::chrono::microseconds>(now - g_submit_split.since)
			                   .count() >= SubmitSplitUs()) {
				BufferFlush(FlushSite::Split);
				g_submit_split.work = 0;
				g_submit_split_count.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances.value = num_instances;
	m_num_instances.sequence++;
}

bool CommandProcessor::HasPendingInstances(const GpuOperation& operation) const {
	return !m_pending_instances.empty() &&
	       m_pending_instances_sequence == operation.num_instances.sequence;
}

uint32_t CommandProcessor::ResolveNumInstances(const GpuOperation& operation) {
	m_last_instance_args_addr = 0;
	if (HasPendingInstances(operation)) {
		ResolvePendingInstances();
	} else {
		// A NUM_INSTANCES packet followed the pending draws.
		m_pending_instances.clear();
	}
	// An indirect draw executed after the last NUM_INSTANCES packet of the operation's stream
	// position set the persistent count.
	if (m_has_indirect_instances &&
	    m_indirect_instances_sequence == operation.num_instances.sequence) {
		return m_indirect_instances;
	}
	return operation.num_instances.value;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t      ready_bit = 1ull << 63u;
			std::array<uint64_t, 32> results {};
			ReadDecisionMemory(address, results.data(), sizeof(results));
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = ReadDecisionValue<uint64_t>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

// KYTY_NATIVE_INDIRECT, execution side: the persistent instance count is the one of the newest
// pending native indirect draw that drew anything, read from the argument snapshots its draws
// captured. Waits for the GPU.
void CommandProcessor::ResolvePendingInstances() {
	m_last_instance_args_addr = 0;
	if (m_pending_instances.empty()) return;
	// Read immutable bytes captured when each indirect packet executed. Guest arguments may
	// already have been overwritten, so reading their current addresses would be incorrect.
	m_renderer.GetCommandScheduler().Finish();
	for (auto it = m_pending_instances.rbegin(); it != m_pending_instances.rend(); ++it) {
		it->snapshot->Invalidate(0, it->snapshot->Size());
		const auto bytes = it->snapshot->Mapped();
		uint32_t count = 0;
		std::memcpy(&count, bytes.data() + it->ArgsSize(), sizeof(count));
		count = std::min(count, it->max_count);
		if (count == 0) continue;
		uint32_t instances = 0;
		std::memcpy(&instances, bytes.data() + static_cast<uint64_t>(count - 1) *
		            it->stride + offsetof(vk::DrawIndirectCommand, instanceCount), sizeof(uint32_t));
		m_last_instance_args_addr = it->args_addr;
		const bool zero = instances == 0;
		const bool fixed = zero && InstanceZeroDiag::FixEnabled();
		if (fixed) {
			// Mirror SetNumInstances(): a zero instance count is clamped to 1.
			instances = 1;
			InstanceZeroDiag::NoteFixApplied();
		}
		if (zero && InstanceZeroDiag::Enabled() && InstanceZeroDiag::TakeLine()) {
			std::printf("NHL27INST: resolve_zero args=0x%llx count_word=%u max=%u stride=%u "
			            "indexed=%d fixed=%d\n",
			            static_cast<unsigned long long>(it->args_addr), count, it->max_count,
			            it->stride, it->indexed ? 1 : 0, fixed ? 1 : 0);
		}
		m_indirect_instances          = instances;
		m_indirect_instances_sequence = m_pending_instances_sequence;
		m_has_indirect_instances      = true;
		break;
	}
	m_pending_instances.clear();
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	++g_work_items;
	ResolveNativeShaderRegisters();
	args.index_type_and_size = m_index_type_and_size;
	auto operation       = MakeOperation(GpuOperationKind::DrawIndex);
	operation.draw_index = args;
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args) {
	NoteRecordedWork();
	if (args.instance_count == 0) {
		const bool from_pending = HasPendingInstances(operation);
		args.instance_count     = ResolveNumInstances(operation);
		args.instance_source    = from_pending ? InstanceZeroDiag::kSourcePending
		                                       : InstanceZeroDiag::kSourcePersist;
		args.instance_args_addr = m_last_instance_args_addr;
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(operation.submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	++g_work_items;
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

// Writes indirect draw offsets to the live SH registers (KYTY_NATIVE_INDIRECT write-back).
static void WriteIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
}

void CommandProcessor::CancelNativeShaderRegister(uint32_t location) {
	for (auto& source: m_pending_shader_offsets) {
		for (auto& reg: source.shader_regs) if (reg == location) reg = Pm4::SH_NOP;
	}
	std::erase_if(m_pending_shader_offsets, [](const NativeIndirectDraw& source) {
		return std::ranges::all_of(source.shader_regs, [](uint32_t reg) { return reg == Pm4::SH_NOP; });
	});
}

void CommandProcessor::ResolveNativeShaderRegisters() {
	if (m_pending_shader_offsets.empty()) return;
	// Move first: the register handlers call GetShCtx(), which must not recurse.
	auto pending = std::move(m_pending_shader_offsets);
	m_pending_shader_offsets.clear();
	// Pending entries exist only without a pipeline: the renderer runs on this thread.
	m_renderer.GetCommandScheduler().Finish();
	for (auto& source: pending) {
		source.snapshot->Invalidate(0, source.snapshot->Size());
		const auto bytes = source.snapshot->Mapped();
		uint32_t count = 0;
		std::memcpy(&count, bytes.data() + source.ArgsSize(), sizeof(count));
		count = std::min(count, source.max_count);
		if (count == 0) continue;
		const auto* record = bytes.data() + static_cast<uint64_t>(count - 1u) * source.stride;
		uint32_t vertex = 0, instance = 0, index = 0;
		std::memcpy(&vertex, record + (source.indexed ? 12 : 8), 4);
		std::memcpy(&instance, record + (source.indexed ? 16 : 12), 4);
		if (source.indexed) std::memcpy(&index, record + 8, 4);
		const auto saved_marker = m_user_data_marker;
		m_user_data_marker = static_cast<HW::UserSgprType>(source.shader_marker);
		WriteIndirectDrawOffsets(*this, {source.shader_regs[0], source.shader_regs[1], source.shader_regs[2]}, vertex, instance, index);
		m_user_data_marker = saved_marker;
	}
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	++g_work_items;
	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	++g_work_items;
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	// Report packet shapes without reading GPU-written argument or count memory.
	// These SH destinations determine which shader inputs must become GPU draw parameters.
	static const bool audit_indirect = [] {
		const char* setting = std::getenv("KYTY_PERF_STATS");
		return setting != nullptr && setting[0] == '1';
	}();
	if (audit_indirect) {
		struct PacketShape {
			uint32_t vertex, instance, index, stride, maximum;
			bool indexed, counted;
			bool operator==(const PacketShape&) const = default;
		};
		static thread_local std::vector<PacketShape> reported;
		const PacketShape shape {registers.vertex_offset, registers.instance_offset,
		                         registers.index_offset, stride_in_bytes, max_count_or_count,
		                         indexed, count_addr != nullptr};
		if (reported.size() < 32 && std::find(reported.begin(), reported.end(), shape) == reported.end()) {
			reported.push_back(shape);
			std::printf("NHL27INDIRECT: indexed=%u counted=%u max=%u stride=%u "
			            "vertex_sh=0x%x instance_sh=0x%x index_sh=0x%x\n",
			            static_cast<unsigned>(indexed), static_cast<unsigned>(count_addr != nullptr),
			            max_count_or_count, stride_in_bytes, registers.vertex_offset,
			            registers.instance_offset, registers.index_offset);
		}
	}

	auto operation          = MakeOperation(GpuOperationKind::DrawIndirect);
	operation.draw_indirect = {.args_base         = m_draw_indirect_args_base_addr,
	                           .data_offset       = data_offset,
	                           .max_count         = max_count_or_count,
	                           .count_addr        = count_addr,
	                           .stride            = stride_in_bytes,
	                           .draw_initiator    = draw_initiator,
	                           .index_type        = m_index_type_and_size,
	                           .index_buffer_size = m_index_buffer_size,
	                           .index_base        = m_index_base_addr,
	                           .indexed           = indexed,
	                           .multi             = true,
	                           .registers         = registers,
	                           .user_data_marker  = static_cast<uint32_t>(m_user_data_marker)};
	Emit(std::move(operation));
}

static bool PatchUserSgpr(HW::Shader& shaders, uint32_t location, uint32_t value) {
	const auto in  = [location](uint32_t base) { return location >= base && location < base + 32u; };
	const auto set = [value](const HW::UserSgprInfo& sgprs, uint32_t id, auto&& setter) {
		if (id < sgprs.count && sgprs.value[id] == value) {
			return false;
		}
		setter(id, value, sgprs.type[id]);
		return true;
	};
	if (in(Pm4::SPI_SHADER_USER_DATA_GS_0)) {
		return set(shaders.GetVs().gs_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_GS_0,
		           [&](uint32_t id, uint32_t v, HW::UserSgprType t) {
			           shaders.SetGsUserSgpr(id, v, t);
		           });
	}
	if (in(Pm4::SPI_SHADER_USER_DATA_HS_0)) {
		return set(shaders.GetVs().hs_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_HS_0,
		           [&](uint32_t id, uint32_t v, HW::UserSgprType t) {
			           shaders.SetHsUserSgpr(id, v, t);
		           });
	}
	if (in(Pm4::SPI_SHADER_USER_DATA_PS_0)) {
		return set(shaders.GetPs().ps_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_PS_0,
		           [&](uint32_t id, uint32_t v, HW::UserSgprType t) {
			           shaders.SetPsUserSgpr(id, v, t);
		           });
	}
	EXIT("unsupported indirect draw offset register 0x%" PRIx32 "\n", location);
	return false;
}

// Writes the offsets of one draw of an indirect packet to the user SGPRs the packet names, as
// the CP does before the draw. Without a pipeline the registers are the live ones. A pipelined
// operation carries a register snapshot shared with other operations: a copy is patched and
// bound for this draw, and the ahead resolution (made from the unpatched registers) is dropped
// when a value changed. Returns whether the packet names any register.
bool CommandProcessor::PatchIndirectDrawOffsets(const GpuOperation& operation,
                                                uint32_t vertex_offset, uint32_t instance_offset,
                                                uint32_t first_index) {
	const auto& registers = operation.draw_indirect.registers;
	if (registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP) {
		return false;
	}
	const std::pair<uint32_t, uint32_t> writes[] = {{registers.vertex_offset, vertex_offset},
	                                                {registers.instance_offset, instance_offset},
	                                                {registers.index_offset, first_index}};
	if (m_pipeline == nullptr) {
		for (const auto& [location, value]: writes) {
			if (location == Pm4::SH_NOP) {
				continue;
			}
			EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
			                     g_hw_sh_indirect_func[location] == nullptr);
			g_hw_sh_indirect_func[location](*this, location, value);
		}
		return true;
	}
	if (m_indirect_shaders == nullptr) {
		m_indirect_shaders = std::make_unique<HW::Shader>();
	}
	*m_indirect_shaders = *operation.state.shaders;
	bool changed        = false;
	for (const auto& [location, value]: writes) {
		if (location != Pm4::SH_NOP) {
			changed |= PatchUserSgpr(*m_indirect_shaders, location, value);
		}
	}
	if (changed) {
		m_renderer.GetRenderExecutor().UseResolvedDraw(nullptr);
	}
	GetScheduler().Begin(*operation.state.context, *operation.state.user_config,
	                     *m_indirect_shaders);
	return true;
}

// The arguments are read when the draw executes: earlier GPU work may produce them.
// KYTY_NATIVE_INDIRECT: records the packet as a GPU indirect draw, so its arguments and count
// are never read back; the instance count stays pending for later draws. User SGPR offsets need
// a shader variant and are written back to the live registers when the command processor reads
// them, so a pipelined GPU, whose command processor runs ahead, takes only packets without them.
bool CommandProcessor::TryExecuteNativeIndirect(const GpuOperation& operation) {
	static const bool native_indirect = [] {
		const char* setting = std::getenv("KYTY_NATIVE_INDIRECT");
		return setting != nullptr && setting[0] == '1';
	}();
	const auto& indirect = operation.draw_indirect;
	if (!native_indirect || indirect.max_count == 0) {
		return false;
	}
	const auto& registers = indirect.registers;
	const std::array<uint32_t, 3> shader_regs {registers.vertex_offset, registers.instance_offset,
	                                           registers.index_offset};
	const bool has_patch = std::ranges::any_of(
	    shader_regs, [](uint32_t location) { return location != Pm4::SH_NOP; });
	if (has_patch && m_pipeline != nullptr) {
		return false;
	}
	if (std::ranges::any_of(m_pending_shader_offsets, [&](const NativeIndirectDraw& old) {
		    return std::ranges::any_of(old.shader_regs, [&](uint32_t reg) {
			    return reg != Pm4::SH_NOP && std::ranges::find(shader_regs, reg) == shader_regs.end();
		    });
	    })) {
		ResolveNativeShaderRegisters();
	}
	const bool patch_supported = std::ranges::all_of(shader_regs, [](uint32_t location) {
		return location == Pm4::SH_NOP || (location >= Pm4::SPI_SHADER_USER_DATA_GS_0 &&
		                                   location <= Pm4::SPI_SHADER_USER_DATA_GS_31);
	});
	const auto marker = static_cast<HW::UserSgprType>(indirect.user_data_marker);
	if (!patch_supported ||
	    (has_patch && indirect.count_addr != nullptr && marker != HW::UserSgprType::Unknown) ||
	    (has_patch && !m_renderer.GetGraphics().native_draw_parameters_enabled)) {
		return false;
	}
	if (m_pending_shader_offsets.size() >= 64) {
		ResolveNativeShaderRegisters();
	}
	if (m_pending_instances.size() >= 64) {
		ResolvePendingInstances();
	}
	NativeIndirectDraw source {.args_addr         = indirect.args_base + indirect.data_offset,
	                           .count_addr        = reinterpret_cast<uint64_t>(indirect.count_addr),
	                           .max_count         = indirect.max_count,
	                           .stride            = indirect.stride,
	                           .indexed           = indirect.indexed,
	                           .index_base        = indirect.index_base,
	                           .index_count_limit = indirect.index_buffer_size,
	                           .index_type        = indirect.index_type,
	                           .shader_regs       = shader_regs,
	                           .shader_marker     = indirect.user_data_marker};
	if (!m_renderer.GetRenderExecutor().DrawAutoIndirect(operation.submit_id, CurrentBuffer(),
	                                                     source)) {
		return false;
	}
	NoteRecordedWork();
	if (source.count_addr == 0 ||
	    m_pending_instances_sequence != operation.num_instances.sequence) {
		m_pending_instances.clear();
	}
	if (has_patch) {
		// Without a pipeline this runs on the command processor thread.
		m_pending_shader_offsets.push_back(source);
		if (indirect.count_addr == nullptr) {
			m_user_data_marker = HW::UserSgprType::Unknown;
		}
	}
	m_pending_instances.push_back(std::move(source));
	m_pending_instances_sequence = operation.num_instances.sequence;
	return true;
}

void CommandProcessor::ExecuteDrawIndirect(const GpuOperation& operation) {
	const auto& indirect = operation.draw_indirect;
	if (TryExecuteNativeIndirect(operation)) {
		return;
	}
	ResolveNativeShaderRegisters();

	uint32_t draw_count = indirect.max_count;
	if (indirect.count_addr != nullptr) {
		draw_count = PeekGuest(indirect.count_addr);
		if (draw_count > indirect.max_count) {
			draw_count = indirect.max_count;
		}
	}
	if (draw_count == 0) {
		return;
	}
	m_pending_instances.clear();

	const auto args_size =
	    indirect.indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(indirect.multi && indirect.stride < args_size);

	uint64_t index_size = 0;
	if (indirect.indexed) {
		switch (indirect.index_type) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", indirect.index_type);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto* args_addr = reinterpret_cast<const void*>(
		    indirect.args_base + indirect.data_offset + static_cast<uint64_t>(i) * indirect.stride);

		if (!indirect.indexed) {
			DrawIndirectArgs args {};
			std::memcpy(&args, args_addr, sizeof(args));
			m_indirect_instances          = args.instance_count;
			m_indirect_instances_sequence = operation.num_instances.sequence;
			m_has_indirect_instances      = true;
			PatchIndirectDrawOffsets(operation, args.start_vertex_location,
			                         args.start_instance_location, 0);
			ExecuteDrawAuto(operation, {.vertex_count   = args.vertex_count_per_instance,
			                            .instance_count = args.instance_count,
			                            .first_vertex   = args.start_vertex_location,
			                            .first_instance = args.start_instance_location,
			                            .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		DrawIndexedIndirectArgs args {};
		std::memcpy(&args, args_addr, sizeof(args));
		PatchIndirectDrawOffsets(operation, args.base_vertex_location,
		                         args.start_instance_location, args.start_index_location);

		auto* index_addr = reinterpret_cast<const void*>(
		    indirect.index_base + static_cast<uint64_t>(args.start_index_location) * index_size);

		const uint32_t index_count =
		    (indirect.index_buffer_size != 0
		         ? std::min(args.index_count_per_instance, indirect.index_buffer_size)
		         : args.index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args.index_count_per_instance, index_count);
			}
		}

		m_indirect_instances          = args.instance_count;
		m_indirect_instances_sequence = operation.num_instances.sequence;
		m_has_indirect_instances      = true;
		ExecuteDrawIndex(operation, {.index_count         = index_count,
		                             .index_addr          = index_addr,
		                             .instance_count      = args.instance_count,
		                             .index_type_and_size = indirect.index_type,
		                             .base_vertex    = static_cast<int32_t>(args.base_vertex_location),
		                             .first_instance = args.start_instance_location,
		                             .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	++g_work_items;
	GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	auto operation     = MakeOperation(GpuOperationKind::DispatchDirect);
	operation.dispatch = {.thread_group_x = thread_group_x,
	                      .thread_group_y = thread_group_y,
	                      .thread_group_z = thread_group_z,
	                      .mode           = mode};
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x,
                                             uint32_t thread_group_y, uint32_t thread_group_z,
                                             uint32_t mode) {
	NoteRecordedWork();

	if (GraphicsRunDebugDumpEnabled()) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
			const auto  frame_num = m_renderer.GetGpu().GetFrameNum();
			const auto& ucfg      = *operation.state.user_config;
			const auto& cs        = operation.state.shaders->GetCs().cs_regs;
			const auto& oa        = ucfg.GetGdsOaCounter(ucfg.GetGdsOaState().GetIndex());
			LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
			     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
			     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32 "\n",
			     frame_num, operation.submit_id, thread_group_x, thread_group_y, thread_group_z,
			     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
			     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
			     cs.data_addr, ucfg.GetGdsOaState().GetIndex(),
			     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
			     oa.GetSpaceAvailable());
		}
	}

	m_renderer.GetRenderExecutor().DispatchDirect(operation.submit_id, CurrentBuffer(),
	                                              thread_group_x, thread_group_y, thread_group_z,
	                                              mode, IsAsyncComputeQueue(), operation.resolved_dispatch);
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	++g_work_items;
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	auto operation     = MakeOperation(GpuOperationKind::DispatchIndirect);
	operation.dispatch = {.mode = mode, .args_addr = args_addr};
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDispatchIndirect(const GpuOperation& operation) {
	const auto& dispatch = operation.dispatch;
	NoteRecordedWork();
	if ((dispatch.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		// The arguments are read when the dispatch executes: earlier GPU work may produce them.
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(dispatch.args_addr);
		ExecuteDispatchDirect(operation, args->x, args->y, args->z, dispatch.mode);
		return;
	}
	m_renderer.GetRenderExecutor().DispatchIndirect(operation.submit_id, CurrentBuffer(),
	                                                dispatch.args_addr, dispatch.mode,
	                                                IsAsyncComputeQueue(), operation.resolved_dispatch);
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	++g_work_items;
	ResolveNativeShaderRegisters();
	auto operation      = MakeOperation(GpuOperationKind::DrawAuto);
	operation.draw_auto = args;
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDrawAuto(const GpuOperation& operation, DrawAutoArgs args) {
	NoteRecordedWork();
	if (args.instance_count == 0) {
		const bool from_pending = HasPendingInstances(operation);
		args.instance_count     = ResolveNumInstances(operation);
		args.instance_source    = from_pending ? InstanceZeroDiag::kSourcePending
		                                       : InstanceZeroDiag::kSourcePersist;
		args.instance_args_addr = m_last_instance_args_addr;
	}
	m_renderer.GetRenderExecutor().DrawAuto(operation.submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	EmitCallback([this, video_out_handle, display_buffer_index] {
		GetScheduler().Flush();
		m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
		                                      static_cast<int>(display_buffer_index));
	}, GpuCallbackTag::FlipWait);
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id, bool flush_if_batch_ready) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	const auto sequence = EmitCallback([=, this, submit_id = m_submit_id] {
		ExecuteWriteAtEndOfPipe(submit_id, eop_event_type, cache_action, event_index,
		                        event_write_source, dst_gpu_addr, value, interrupt_selector,
		                        interrupt_context_id);
		if (flush_if_batch_ready) {
			FlushIfBatchReady();
		}
	}, GpuCallbackTag::Eop);
	// What ExecuteWriteAtEndOfPipe() writes: nothing for a graphics end-of-pipe interrupt, GDS
	// dwords or the clock (values known only then), the low dword of a 64-bit value from source
	// 1, or the value.
	const auto address = reinterpret_cast<uint64_t>(dst_gpu_addr);
	if (interrupt_selector == 0x01 && !IsAsyncComputeQueue()) {
		return;
	}
	if constexpr (sizeof(T) == sizeof(uint32_t)) {
		if (event_write_source == 0x01) {
			NoteGuestWrite(sequence, address, uint64_t {value >> 16u} * sizeof(uint32_t), nullptr,
			               "wait-eop-gds");
			return;
		}
		NoteGuestWrite(sequence, address, sizeof(value), &value);
	} else {
		if (event_write_source == 0x04) {
			NoteGuestWrite(sequence, address, sizeof(value), nullptr, kEopClockWrite, GuestGpu::WriteOriginClock);
			return;
		}
		const auto low = static_cast<uint32_t>(value);
		if (event_write_source == 0x01) {
			NoteGuestWrite(sequence, address, sizeof(low), &low);
			return;
		}
		NoteGuestWrite(sequence, address, sizeof(value), &value);
	}
}

template <typename T>
void CommandProcessor::ExecuteWriteAtEndOfPipe(uint64_t submit_id, uint32_t eop_event_type,
                                               uint32_t cache_action, uint32_t event_index,
                                               uint32_t event_write_source, void* dst_gpu_addr,
                                               T value, uint32_t interrupt_selector,
                                               uint32_t interrupt_context_id) {
	auto& command = CurrentBuffer();

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				DeferEopInterrupt(interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	// Opt-in label path for NHL26 menu synchronization profiling.
	static const bool direct_labels = [] {
		const char* setting = std::getenv("KYTY_EOP_DIRECT_LABELS");
		return setting != nullptr && setting[0] == '1';
	}();
	auto write_label = [&](void* destination, const void* data, uint64_t bytes) {
		Common::EventTrace::Record(Common::EventTrace::LABEL_WRITE,
		                           reinterpret_cast<uint64_t>(destination) & 0xffffffffu,
		                           m_renderer.GetCommandScheduler().CurrentTick());
		// The exact-range label write goes first: preparing the page for a host store downloads
		// and unprotects it, waiting for all queued GPU work.
		if (!direct_labels || !m_renderer.GetBufferCache().TryWriteLabel(
		        reinterpret_cast<uint64_t>(destination), data, bytes)) {
			// A store into a watched page would fault: download its GPU-written bytes first. A
			// direct label write goes through the backing and the label's GPU copy instead, so the
			// other GPU-written bytes of the page stay where they are and nothing waits for the
			// GPU (15% of the execution thread in a match, before this).
			BufferCache::WriteTagScope tag(BufferCache::WriteTag::EopLabel);
			m_renderer.PrepareGpuWrite(destination, bytes);
			std::memcpy(destination, data, static_cast<size_t>(bytes));
		} else {
			static std::atomic_flag reported = ATOMIC_FLAG_INIT;
			if (!reported.test_and_set(std::memory_order_relaxed)) {
				std::printf("NHL26MENU: direct EOP label write active width=%llu\n",
				            static_cast<unsigned long long>(bytes));
				std::fflush(stdout);
			}
		}
	};
	auto write32 = [&] {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		EXIT_IF(dst == nullptr);
		(void)command.Handle();
		write_label(dst, &data, sizeof(data));

		if (with_interrupt) {
			DeferEopInterrupt(interrupt_context_id);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					GetScheduler().Finish();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32();
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00:
						case 0x38: write32(); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&] {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					EXIT_IF(dst == nullptr);
					(void)command.Handle();
					write_label(dst, &value, sizeof(value));
					// The command processor may be waiting for this clock (WAIT_REG_MEM): it gets the
					// value written, even if the game resets the label once the interrupt wakes it.
					if (event_write_source == 0x04 && m_pipeline != nullptr) {
						m_pipeline->NoteExecutedWrite(reinterpret_cast<uint64_t>(dst), &value, sizeof(value));
					}

					if (with_interrupt) {
						DeferEopInterrupt(interrupt_context_id);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64();
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64();
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64();
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64();
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64();
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64();
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64();
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id, bool flush_if_batch_ready) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id, flush_if_batch_ready);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id, false);
}

void CommandProcessor::EmitGlobalBarrier() {
	EmitCallback([this] { ExecuteGlobalBarrier(); }, GpuCallbackTag::Barrier);
}

void CommandProcessor::ExecuteGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());
	// KYTY_GUEST_BARRIER_DEDUP: a global barrier with no draw, dispatch or host helper pass (tiler, fault
	// manager, blit, indirect prep: they bump the pipeline epochs) recorded since the previous one in the
	// same command-buffer recording orders nothing new. A recording is identified by the scheduler tick.
	static uint64_t last_work = UINT64_MAX, last_compute = UINT64_MAX, last_graphics = UINT64_MAX;
	static uint64_t last_tick = UINT64_MAX;
	if (Common::RuntimeFlags::Current().guest_barrier_dedup) {
		const uint64_t work     = g_work_epoch.load(std::memory_order_relaxed);
		const uint64_t compute  = g_compute_pipeline_epoch.load(std::memory_order_relaxed);
		const uint64_t graphics = g_graphics_pipeline_epoch.load(std::memory_order_relaxed);
		const uint64_t tick     = GetScheduler().CurrentTick();
		const bool     redundant = work == last_work && compute == last_compute &&
		                           graphics == last_graphics && tick == last_tick;
		last_work = work; last_compute = compute; last_graphics = graphics; last_tick = tick;
		if (redundant) {
			FrameBreakdown::Add(FrameBreakdown::g_counters.port_guest_barriers_skipped);
			return;
		}
	} else {
		last_tick = UINT64_MAX;
	}
	FrameBreakdown::Add(FrameBreakdown::g_counters.port_guest_barriers);
	++g_guest_barriers;
	if (CurrentBuffer().IsRendering()) {
		// Dropping these (keeping the render pass) measured no GPU gain: about 40 per frame.
		++g_guest_barrier_breaks;
	}

	vk::MemoryBarrier2 barrier {};
	// The device-side accesses by name: with MEMORY_READ/WRITE the driver may also write back and
	// invalidate for the host (paused GPU A/B: 13.75-13.82 -> 13.65 ms per flip).
	barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
	                        vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eTransferWrite;
	barrier.dstAccessMask =
	    vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eIndexRead |
	    vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eUniformRead |
	    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite |
	    vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	if (m_renderer.GetGraphics().conditional_rendering_enabled) {
		barrier.dstAccessMask |= vk::AccessFlagBits2::eConditionalRenderingReadEXT;
	}

	// Experiment LazyGuestBarriers: owed until later work needs it (GuestBarriers::CheckWork).
	if (GuestBarriers::Lazy() && GuestBarriers::Owe(CurrentBuffer(), barrier)) {
		return;
	}
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().Handle().pipelineBarrier2(dependency);
	GuestBarriers::Recorded();
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	EmitCallback([this, interrupt_context_id] { DeferEopInterrupt(interrupt_context_id); },
	             GpuCallbackTag::Eop);
}

void CommandProcessor::DeferEopInterrupt(uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();
	// KYTY_DIRECT_EOP_INTERRUPTS: end-of-pipe labels are already written when the CP reaches them
	// and recorded work holds its own copies of guest memory, so deliver the interrupt with the
	// label instead of after host completion. CPU reads of GPU-written memory still wait for the
	// host GPU through the page tracking.
	if (Common::RuntimeFlags::Current().direct_eop_interrupts) {
		FrameBreakdown::Add(FrameBreakdown::g_counters.port_eop_direct);
		Common::EventTrace::Record(Common::EventTrace::EOP_TRIGGER,
		                           static_cast<uint64_t>(m_interrupt_event_id),
		                           GetScheduler().CurrentTick());
		m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
		return;
	}
	GetScheduler().DeferHostOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id,
	     tick = GetScheduler().CurrentTick()] {
		    Common::EventTrace::Record(Common::EventTrace::EOP_TRIGGER, static_cast<uint64_t>(event_id),
		                               tick);
		    renderer.TriggerInterrupt(event_id, interrupt_context_id);
	    },
	    true);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	// The occlusion counter dump writes guest memory (see CommandProcessor::Execute()).
	const auto sequence = EmitCallback([this, event_type, event_index, event_address] {
		ExecuteTriggerEvent(event_type, event_index, event_address);
	}, event_type == 0x00000039 ? GpuCallbackTag::Other : GpuCallbackTag::Event);
	if (event_type == 0x00000039) {
		// Occlusion counter pairs of 16 DBs.
		NoteGuestWrite(sequence, event_address, 16u * 2u * sizeof(uint64_t), nullptr, "wait-occlusion");
	}
}

void CommandProcessor::ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index,
                                           uint64_t event_address) {
	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: ExecuteGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			ExecuteGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			ExecuteGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::WriteLodStats(void* dst, uint32_t size) {
	const auto sequence = EmitCallback([dst, size] {
		memset(dst, 0, size);
		// Hack?
		if (size >= sizeof(uint32_t)) {
			auto* label = static_cast<uint32_t*>(dst);
			*label      = 1;
		}
	}, GpuCallbackTag::Other);
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), size, nullptr, "wait-lod-stats");
}

void CommandProcessor::Flip() {
	Common::Timeline::Mark("cp-flip");
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	EmitCallback([this, flip = m_flip] {
		auto& command = CurrentBuffer();
		auto  request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
		                                          flip.flip_mode, flip.flip_arg);
		GetScheduler().DeferHostOperation(
		    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); },
		    true);
		g_scheduler_flush_sites[3].fetch_add(1, std::memory_order_relaxed);
		GetScheduler().Flush();
	}, GpuCallbackTag::Flip);
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	EXIT_IF(dst_gpu_addr == nullptr);
	const auto sequence = EmitCallback([this, dst_gpu_addr, value, flip = m_flip] {
		auto& command = CurrentBuffer();
		std::memcpy(dst_gpu_addr, &value, sizeof(value));
		auto request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
		                                         flip.flip_mode, flip.flip_arg);
		GetScheduler().DeferHostOperation(
		    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); },
		    true);
		g_scheduler_flush_sites[3].fetch_add(1, std::memory_order_relaxed);
		GetScheduler().Flush();
	}, GpuCallbackTag::Flip);
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value), &value);
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	EXIT_IF(dst_gpu_addr == nullptr);
	const auto sequence = EmitCallback([this, dst_gpu_addr, value, flip = m_flip] {
		auto& command = CurrentBuffer();
		std::memcpy(dst_gpu_addr, &value, sizeof(value));
		auto request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
		                                         flip.flip_mode, flip.flip_arg);
		GetScheduler().DeferHostOperation(
		    [&renderer = m_renderer, event_id = m_interrupt_event_id, request,
		     tick = GetScheduler().CurrentTick()] {
			    renderer.GetVideoOut().CompleteFlip(request);
			    Common::EventTrace::Record(Common::EventTrace::EOP_TRIGGER,
			                               static_cast<uint64_t>(event_id), tick);
			    renderer.TriggerInterrupt(event_id, 0);
		    },
		    true);
		g_scheduler_flush_sites[3].fetch_add(1, std::memory_order_relaxed);
		GetScheduler().Flush();
	}, GpuCallbackTag::Flip);
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value), &value);
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	EmitCallback([this, request_id] {
		struct ProcessorScope {
			explicit ProcessorScope(CommandProcessor& processor) {
				g_current_processor = &processor;
			}
			~ProcessorScope() { g_current_processor = nullptr; }
		};
		ProcessorScope processor_scope(*this);

		m_renderer.GetVideoOut().PrepareFlip(request_id, CurrentBuffer());
		GetScheduler().DeferHostOperation(
		    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); }, true);
		g_scheduler_flush_sites[3].fetch_add(1, std::memory_order_relaxed);
		GetScheduler().Flush();
	}, GpuCallbackTag::Flip);
}

void CommandProcessor::SynchronizeGpu() {
	EmitCallback([this] { GetScheduler().Finish(); }, GpuCallbackTag::Finish);
}

uint32_t GuestGpu::CurrentQueue() noexcept {
	return g_current_queue;
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

bool GuestGpu::IsCommandProcessorThread() noexcept {
	return g_cp_thread;
}

bool CpReadDebugEnabled() noexcept {
	static const bool enabled = std::getenv("KYTY_DBG_CP_READS") != nullptr;
	return enabled;
}

void NoteCpRead(const char* kind, uint64_t address, const uint32_t* pairs, uint32_t pair_count, uint64_t ns) {
	struct Stat {
		const char*             kind    = nullptr;
		uint64_t                address = 0;
		uint32_t                pairs   = 0;
		std::array<uint32_t, 4> regs {};
		uint64_t                count = 0;
		uint64_t                slow  = 0;
		uint64_t                ns    = 0;
	};
	static std::unordered_map<uint64_t, Stat>      stats;
	static std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	// Register loads are keyed by their register offsets (the address changes per draw), waits by
	// their address.
	uint64_t                key = std::hash<const char*> {}(kind);
	std::array<uint32_t, 4> regs {};
	for (uint32_t i = 0; i < pair_count && i < regs.size(); ++i) {
		regs[i] = pairs[i * 2];
		key     = key * 0x9e3779b97f4a7c15ull + regs[i];
	}
	key = pairs != nullptr ? key * 31u + pair_count : key ^ address;
	auto& stat = stats[key];
	if (stat.count == 0) {
		stat.kind  = kind;
		stat.pairs = pair_count;
		stat.regs  = regs;
	}
	stat.address = address;
	stat.count++;
	stat.slow += ns > 20000 ? 1u : 0u;
	stat.ns += ns;
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(2)) {
		return;
	}
	last = now;
	std::vector<const Stat*> sorted;
	sorted.reserve(stats.size());
	uint64_t total_ns = 0;
	for (const auto& [k, s]: stats) {
		sorted.push_back(&s);
		total_ns += s.ns;
	}
	std::sort(sorted.begin(), sorted.end(), [](const Stat* a, const Stat* b) { return a->ns > b->ns; });
	std::printf("NHL27CPREAD: total %.1f ms in 2 s over %zu kinds\n", static_cast<double>(total_ns) / 1e6,
	            sorted.size());
	{
		// Page protection changes and tracking faults in the same 2 s (all threads).
		static uint64_t previous[5] {};
		uint64_t        now_stats[5] {};
		ProtectStats::Read(now_stats[0], now_stats[1], now_stats[2], now_stats[3], now_stats[4]);
		std::printf("NHL27PROTECT: protect calls=%llu (execution %llu) MiB=%.1f | faults read=%llu write=%llu\n",
		            static_cast<unsigned long long>(now_stats[0] - previous[0]),
		            static_cast<unsigned long long>(now_stats[2] - previous[2]),
		            static_cast<double>(now_stats[1] - previous[1]) / (1024.0 * 1024.0),
		            static_cast<unsigned long long>(now_stats[3] - previous[3]),
		            static_cast<unsigned long long>(now_stats[4] - previous[4]));
		std::copy(std::begin(now_stats), std::end(now_stats), std::begin(previous));
		ProtectStats::PrintFaultPages();
	}
	for (size_t i = 0; i < sorted.size() && i < 10; ++i) {
		const auto& s = *sorted[i];
		std::printf("NHL27CPREAD: %s ms=%.1f count=%llu slow=%llu last=0x%llx pairs=%u regs=%x,%x,%x,%x\n", s.kind,
		            static_cast<double>(s.ns) / 1e6, static_cast<unsigned long long>(s.count),
		            static_cast<unsigned long long>(s.slow), static_cast<unsigned long long>(s.address), s.pairs,
		            s.regs[0], s.regs[1], s.regs[2], s.regs[3]);
	}
	std::fflush(stdout);
	stats.clear();
}

void GuestGpu::ServiceUrgentCommands() {
	if (g_execution_gpu != nullptr &&
	    g_execution_gpu->m_urgent_pending.load(std::memory_order_acquire) != 0) {
		g_execution_gpu->RunUrgentCommands();
	}
}

uint64_t GuestGpu::CurrentExecOp() noexcept {
	return g_exec_op;
}

} // namespace Libs::Graphics
