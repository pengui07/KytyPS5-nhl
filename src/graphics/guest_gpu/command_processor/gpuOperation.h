#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H

#include "common/inlineFunction.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

class CommandProcessor;

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

// The register state a draw or a dispatch reads when it executes.
struct GpuRegisterState {
	const HW::Context*    context     = nullptr;
	const HW::UserConfig* user_config = nullptr;
	const HW::Shader*     shaders     = nullptr;
	// Unique per snapshot of the context registers (0: live registers, not a snapshot):
	// operations with the same id see the same context registers.
	uint64_t context_id = 0;
};

// NUM_INSTANCES as the command processor last set it. Indirect draws also set the persistent
// instance count, from arguments read when they execute; `sequence` counts NUM_INSTANCES
// packets so an execution can tell which of the two was written last.
struct GpuNumInstances {
	uint32_t value    = 1;
	uint64_t sequence = 0;
};

// The SH registers (user SGPRs) a draw-indirect packet writes its offsets to, or Pm4::SH_NOP.
struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
};

struct GpuDrawIndirectArgs {
	uint64_t                 args_base         = 0;
	uint32_t                 data_offset       = 0;
	uint32_t                 max_count         = 1;
	const volatile uint32_t* count_addr        = nullptr;
	uint32_t                 stride            = 0;
	uint32_t                 draw_initiator    = 0;
	uint32_t                 index_type        = 0;
	uint32_t                 index_buffer_size = 0;
	uint64_t                 index_base        = 0;
	bool                     indexed           = false;
	bool                     multi             = false;
	IndirectDrawRegisters    registers {};
	// HW::UserSgprType of the user data the draw follows (KYTY_NATIVE_INDIRECT).
	uint32_t user_data_marker = 0;
};

struct GpuDispatchArgs {
	uint32_t thread_group_x = 0;
	uint32_t thread_group_y = 0;
	uint32_t thread_group_z = 0;
	uint32_t mode           = 0;
	uint64_t args_addr      = 0; // Indirect dispatches.
};

// Callbacks of queued operations live in the operation itself; the largest ones capture about
// ten words.
using GpuCallback = Common::InlineFunction<96>;

enum class GpuOperationKind : uint8_t {
	DrawIndex,
	DrawAuto,
	DrawIndirect,
	DispatchDirect,
	DispatchIndirect,
	// Any other guest-visible effect or renderer work, in command order.
	Callback,
};

enum class GpuCallbackTag : uint8_t {
	Other,
	Bda,
	Flush,
	Finish,
	Flip,
	Eop,
	Event,
	Barrier,
	WriteData,
	Dma,
	Clock,
	Host,
	Gc,
	FlipWait,
	Count
};

// Experiment DispatchResolver: where the second resolve thread leaves a dispatch's resolution
// (state 0 free, 1 queued, 2 resolved); the execution thread waits for it and frees it.
struct GpuDispatchSlot {
	std::atomic<uint32_t>             state {0};
	RenderExecutor::ResolvedDispatch* result = nullptr;
};

// Experiment DrawResolver: the same for a draw the second draw resolver resolves.
struct GpuDrawSlot {
	std::atomic<uint32_t>         state {0};
	RenderExecutor::ResolvedDraw* result = nullptr;
};

// One unit of emulated GPU work produced by a command processor. Operations execute in the
// order the command processors produce them; every guest-visible effect of a command buffer
// (draws, dispatches, labels, interrupts, flips, DMA) is an operation.
struct GpuOperation {
	GpuOperationKind  kind      = GpuOperationKind::Callback;
	CommandProcessor* processor = nullptr;
	uint64_t          submit_id = 0;
	// Position in the stream of a pipelined GPU; see GuestGpu::EmitOperation().
	uint64_t          sequence  = 0;
	GpuRegisterState  state;
	GpuNumInstances   num_instances;
	// Draws: the shader programs the resolve thread resolved ahead, if any.
	RenderExecutor::ResolvedDraw* resolved = nullptr;
	// Direct dispatches: the compute program the resolve thread resolved ahead, if any.
	RenderExecutor::ResolvedDispatch* resolved_dispatch = nullptr;
	// Experiment DispatchResolver: the resolution of resolved_dispatch is still being made.
	GpuDispatchSlot* dispatch_slot = nullptr;
	// Experiment DrawResolver: the resolution of `resolved` is still being made.
	GpuDrawSlot* draw_slot = nullptr;
	// The command processor processed a WAIT_REG_MEM since the previous operation: the guest may
	// have written memory the operations before it read.
	bool after_wait = false;
	// A KYTY_BDA_BATCH boundary was passed since the previous draw or dispatch (it is executed
	// before this one instead of as an operation of its own).
	bool bda_boundary = false;
	// The write epoch when the command processor emitted the operation (see RegionManager).
	uint32_t write_epoch = 0;
	// KYTY_IDLE_TRACE: the PM4 packet that produced the operation (opcode | IT_NOP R code << 8,
	// 0x4000 = the KYTY_BDA_BATCH boundary after the packet, 0 = outside a packet).
	uint16_t trace_op = 0;
	// Callbacks: what they do (KYTY_DBG_STAGE_STATS, NHL27CALLBACKS).
	GpuCallbackTag callback_tag = GpuCallbackTag::Other;
	union {
		DrawIndexArgs       draw_index;
		DrawAutoArgs        draw_auto;
		GpuDrawIndirectArgs draw_indirect;
		GpuDispatchArgs     dispatch;
	};
	GpuCallback callback;

	GpuOperation(): draw_index {} {}
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H
