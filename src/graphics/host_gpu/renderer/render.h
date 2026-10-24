#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
class NativeIndirectPrep;
class DispatchArgsGuard;
struct RenderExecutorTestAccess;

// An image description derived from a T# and a shader's view of it (see ResolveTexture).
struct TextureDescDerivation {
	TextureCache::ImageDesc desc;
	bool                    shader_conversion = false;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	// The image the last lookup of a sampled texture with this description found, the
	// description it was found with and the view acquired for it. Reused while no image on the
	// pages of the range was registered or unregistered since and the image keeps the state the
	// acquisition left it in.
	struct Found {
		ImageId                 id;
		TextureCache::ImageDesc desc;
		vk::ImageView           view            = nullptr;
		uint64_t                image_set_epoch = 0;
		ImageLookupState        state;
		bool                    valid = false;
		// Unique per recording (TextureBinding::desc_serial): a binding that already holds
		// this `desc` need not copy it again.
		uint64_t serial = 0;
	};
	Found found;
};

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// DIAGNOSTIC helpers for KYTY_DBG_ZERO_INSTANCE (instance-zero draw logging) and
// KYTY_INSTANCE_ZERO_FIX (clamp a resolved zero instance count to 1). Both off by default.
namespace InstanceZeroDiag {

constexpr uint32_t kMaxLines = 3000;

// Where DrawIndexArgs/DrawAutoArgs::instance_count came from when it was resolved in
// CommandProcessor (0 = passed in explicitly by the caller).
constexpr uint8_t kSourceExplicit = 0;
constexpr uint8_t kSourcePending  = 2; // from a pending indirect snapshot
constexpr uint8_t kSourcePersist  = 3; // persisted m_num_instances (SET_NUM_INSTANCES or default)

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_ZERO_INSTANCE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline bool FixEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_INSTANCE_ZERO_FIX");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Returns true while the shared line budget is not exhausted.
inline bool TakeLine() {
	static std::atomic<uint32_t> lines {0};
	return lines.fetch_add(1) < kMaxLines;
}

// Counts zero_fix applications; logs the first one and then at most every 5 seconds.
inline void NoteFixApplied() {
	static std::atomic<uint64_t> applied {0};
	static std::atomic<int64_t>  last_log_ns {0};
	const uint64_t               total = applied.fetch_add(1) + 1;
	const auto   now_tp = std::chrono::steady_clock::now().time_since_epoch();
	const int64_t now   = std::chrono::duration_cast<std::chrono::nanoseconds>(now_tp).count();
	const int64_t last  = last_log_ns.load();
	if ((total == 1 || now - last >= 5'000'000'000LL) && TakeLine()) {
		last_log_ns.store(now);
		std::printf("NHL27INST: zero_fix applied total=%llu\n",
		            static_cast<unsigned long long>(total));
	}
}

} // namespace InstanceZeroDiag

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Diagnostics only: see InstanceZeroDiag.
	uint8_t  instance_source     = InstanceZeroDiag::kSourceExplicit;
	uint64_t instance_args_addr  = 0;
};

class Buffer;

// Frozen argument bytes also retain the NUM_INSTANCES value for later inherited draws.
struct NativeIndirectDraw {
	uint64_t args_addr = 0;
	uint64_t count_addr = 0;
	uint32_t max_count = 0;
	uint32_t stride = 0;
	bool indexed = false;
	uint64_t index_base = 0;
	uint32_t index_count_limit = 0;
	uint32_t index_type = 0;
	std::array<uint32_t, 3> shader_regs {0x280, 0x280, 0x280};
	uint32_t shader_marker = 0;
	[[nodiscard]] uint32_t RecordSize() const {
		return indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	}
	std::shared_ptr<Buffer> snapshot;
	[[nodiscard]] uint64_t ArgsSize() const {
		return static_cast<uint64_t>(max_count - 1u) * stride + RecordSize();
	}
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Diagnostics only: see InstanceZeroDiag.
	uint8_t  instance_source    = InstanceZeroDiag::kSourceExplicit;
	uint64_t instance_args_addr = 0;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool IsRendering() const { return m_rendering; }

	[[nodiscard]] CommandRecorder Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] const HW::Context&    GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] const HW::UserConfig& GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] const HW::Shader&     GetShaders() const noexcept { return *m_shaders; }

	// Pipelines recorded outside the draws (the presentation overlay) replace the dynamic state
	// the draws set.
	void InvalidateDynamicState() const;

private:
	CommandBuffer(CommandScheduler& scheduler, CommandStream* stream);
	void Bind(const HW::Context& registers, const HW::UserConfig& user_config,
	          const HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	CommandScheduler&   m_scheduler;
	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	// Direct recording writes m_buffer; deferred recording appends to m_stream while m_open.
	vk::CommandBuffer   m_buffer          = nullptr;
	CommandStream*      m_stream          = nullptr;
	bool                m_open            = false;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState   m_render_state;
	mutable bool          m_rendering = false;
	const HW::Context*    m_registers   = nullptr;
	const HW::UserConfig* m_user_config = nullptr;
	const HW::Shader*     m_shaders     = nullptr;

	friend class CommandScheduler;
class NativeIndirectPrep;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	struct ResolvedDispatch;
	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute,
	                    ResolvedDispatch* resolved = nullptr);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute, ResolvedDispatch* resolved = nullptr);
	// Shader programs and resources of a draw resolved ahead of its execution.
	struct ResolvedDraw;
	// Resolve thread: resolves the shader programs of a draw with the given registers ahead of
	// its execution; null when the draw resolves them itself (tessellation, programs that are not
	// compiled yet, guest memory that cannot be read ahead). Waits while every resolution is in
	// use: the execution thread releases them in order. Each resolving thread has its own ring
	// (ring 1: experiment DrawResolver).
	// The execution thread pulls an upcoming resolution into its cache, in two steps: step 0 the
	// snapshot and log headers and the draw state it reads first (two operations ahead), step 1 the
	// arrays they point to (one ahead), so neither step waits for a miss.
	static void PrefetchResolvedDraw(const ResolvedDraw* resolved, int step);
	static void PrefetchResolvedDispatch(const ResolvedDispatch* resolved, int step);
	[[nodiscard]] ResolvedDraw* ResolveDrawAhead(const HW::Context&    context,
	                                             const HW::UserConfig& user_config,
	                                             const HW::Shader& shaders, uint32_t ring_index = 0);
	// Execution thread: the next draw uses `resolved` when the guest memory it read still holds
	// the same bytes.
	void UseResolvedDraw(ResolvedDraw* resolved) noexcept { m_resolved_draw = resolved; }
	// Execution thread: the operation that carried `resolved` executed.
	static void ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept;
	// Direct and ordinary indirect dispatches resolved ahead of execution.
	[[nodiscard]] ResolvedDispatch* ResolveDispatchAhead(const HW::Context& context, const HW::Shader& shaders,
	                                                     uint32_t thread_group_x, uint32_t thread_group_y,
	                                                     uint32_t thread_group_z, uint32_t mode, bool async_compute,
	                                                     bool indirect = false);
	// Indirect dispatches: their program does not depend on the group counts.
	[[nodiscard]] ResolvedDispatch* ResolveDispatchIndirectAhead(const HW::Context& context, const HW::Shader& shaders,
	                                                             bool async_compute);
	static void ReleaseResolvedDispatch(ResolvedDispatch* resolved) noexcept;
	// Execution thread: the context register snapshot of the operation about to execute (0 for
	// live registers); operations with the same id see the same context registers.
	void UseContextId(uint64_t context_id) noexcept { m_context_id = context_id; }
	// Execution thread: an operation other than a draw is about to execute (see
	// RenderTargetMemo).
	void InvalidateRenderTargetMemo() noexcept;
	// Advances whenever an operation other than a draw executes (see
	// InvalidateRenderTargetMemo()): within one value only draws executed, so no guest-visible
	// effect let the guest write the memory they read.
	[[nodiscard]] uint64_t DrawWindow() const noexcept { return m_draw_window; }

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	bool DrawAutoIndirect(uint64_t submit_id, CommandBuffer& buffer, NativeIndirectDraw& source);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// Bindless table elements resolved for one root image, by T#. Most of a table is unchanged
	// from dispatch to dispatch, and resolving an element repeats the same texture-cache lookup
	// while no image in its range was registered or unregistered and the guest backing did not
	// change. Elements with DCC metadata are resolved again on every use: their lookup
	// materializes fast clears.
	struct TableElementResolution {
		TextureBinding texture;
		GuestRange     range; // Guest memory whose images decide the lookup, if any.
		uint64_t       image_epoch   = 0;
		uint64_t       backing_epoch = 0;
		// The global image-set epoch at which the element was last known valid: while it holds,
		// no image anywhere changed and the range needs no check.
		uint64_t       checked_epoch = 0;
		bool           permanent     = false; // Null for the root's view, whatever the memory.
		bool           volatile_dcc  = false;
	};
	struct TableDescriptorHash {
		size_t operator()(const std::array<uint32_t, 8>& dwords) const noexcept;
	};
	struct TableResolution {
		ShaderRecompiler::IR::ImageResource root;
		uint64_t                            last_use = 0;
		std::unordered_map<std::array<uint32_t, 8>, TableElementResolution, TableDescriptorHash>
		    elements;
	};

	[[nodiscard]] const TextureBinding&
	ResolveTableElement(TableResolution& table, const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TableResolution& FindTableResolution(const ShaderRecompiler::IR::ImageResource& root);

	// ResolveTexture's image descriptions by T# and shader view (see DeriveTextureDesc).
	struct TextureDescKey {
		std::array<uint32_t, 8> dwords {};
		std::array<uint32_t, 5> view {};
		// One memcmp of the 52 bytes (no padding): the defaulted compare went dword by dword
		// (1% of the execution thread, in the description lookups of every texture bind).
		bool operator==(const TextureDescKey& other) const noexcept {
			return std::memcmp(this, &other, sizeof(*this)) == 0;
		}
	};
	struct TextureDescKeyHash {
		size_t operator()(const TextureDescKey& key) const noexcept;
	};
	[[nodiscard]] static TextureDescKey
	MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
	                   const ShaderRecompiler::IR::DescriptorValue& value);
	// Writes the image and view the last lookup of a description found into `binding`, when
	// they still hold (see TextureDescDerivation::Found).
	[[nodiscard]] bool ReuseTextureLookup(TextureDescDerivation& derived, TextureBinding& binding);
	static constexpr size_t MaxTextureDescs = size_t {1} << 16u;
	std::unordered_map<TextureDescKey, TextureDescDerivation, TextureDescKeyHash> m_texture_descs;
	// Bindings point into m_texture_descs until their views are acquired: a full map is cleared
	// once the bindings of the operation are reset.
	bool m_clear_texture_descs = false;
	// Advances when m_texture_descs is cleared (bindings keep pointers into it).
	uint64_t m_texture_descs_generation = 1;
	// Per program (compiled programs live as long as the cache): the T#s its last draw bound
	// and the description each had, so draws that alternate programs on a stage skip the
	// description lookups of unchanged T#s (see PrepareBindings()).
	std::unordered_map<const void*, ProgramImageMemo> m_program_image_memos;
	// Per program too: its buffer slots, so an unchanged V# finds what it bound the program's last
	// draw (see PrepareBindings()).
	std::unordered_map<const void*, ProgramBufferMemo> m_program_buffer_memos;
	// The two memos of a program by its address, direct-mapped in front of the maps (their lookups
	// were 0.8% of the execution thread). The maps never erase, and their elements stay in place.
	struct ProgramMemos {
		const void*        program = nullptr;
		ProgramImageMemo*  image   = nullptr;
		ProgramBufferMemo* buffer  = nullptr;
	};
	std::array<ProgramMemos, 256> m_program_memo_cache {};
	ProgramMemos& ProgramMemosOf(const void* program) {
		auto& entry = m_program_memo_cache[(reinterpret_cast<uintptr_t>(program) >> 6u) & 255u];
		if (entry.program != program) {
			entry = {program, &m_program_image_memos[program], &m_program_buffer_memos[program]};
		}
		return entry;
	}
	// The last TextureDescDerivation::Found::serial given out.
	uint64_t m_found_serial = 0;

	// The host sampler for a sampler binding; see NativeSampler().
	[[nodiscard]] vk::Sampler NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                        uint32_t                                        index,
	                                        const ShaderRecompiler::IR::DescriptorValue&    value);
	struct SamplerMemoEntry {
		std::array<uint32_t, 4> fields {};
		bool                    integer_border = false;
		vk::Sampler             sampler        = nullptr;
	};
	std::array<SamplerMemoEntry, 256> m_sampler_memo {};

	// table_candidate rejects (EXIT) a texture whose guest memory cannot be read, for tables
	// that keep T#s of freed textures, and declines depth/color conversions. examined receives
	// the guest range before the lookup depends on the texture cache: a failure without it
	// depends only on the T# and the guest backing.
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            bool        table_candidate = false,
	                                            GuestRange* examined        = nullptr);
	[[nodiscard]] TextureBinding
	ResolveTableTexture(const ShaderRecompiler::IR::ImageResource&   root,
	                    const ShaderRecompiler::IR::DescriptorValue& value,
	                    GuestRange*                                  examined = nullptr);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	struct DrawRenderStorage;
	struct ColorTargetMemo;
	struct DepthTargetMemo;
	struct RenderTargetMemo;
	struct ResolvedDrawRing;
	struct ResolvedDispatchRing;
	// The input info DispatchDirect builds for a dispatch (group counts from thread counts).
	static void PrepareComputeInputInfo(const HW::Shader& shaders, uint32_t thread_group_x, uint32_t thread_group_y,
	                                    uint32_t thread_group_z, uint32_t mode, bool async_compute,
	                                    ShaderComputeInputInfo& input_info);
	// See RenderTargetMemo.
	void               CaptureRenderTargetKey(const HW::Context& registers, uint32_t mrt_mask,
	                                          uint32_t slice_offset);
	[[nodiscard]] bool RenderTargetMemoCurrent();
	[[nodiscard]] bool TryReuseRenderTargets(const HW::Context& registers, uint32_t mrt_mask,
	                                         uint32_t slice_offset, DrawRenderState& state);
	void StoreRenderTargetMemo(const RenderColorInfo* colors, uint32_t color_count,
	                           const RenderDepthInfo& depth, const vk::ImageView* views,
	                           vk::ImageView depth_view, ImageId stencil);
	// The draw state of the executing draw: the one its resolution prepared when that is still
	// valid (shaders_resolved), otherwise the reused state.
	[[nodiscard]] DrawRenderState& AcquireDrawRenderState(bool  tessellation,
	                                                      bool& shaders_resolved);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, bool shaders_resolved,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState&    state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	std::shared_ptr<NativeIndirectPrep> m_native_indirect_prep;
	std::shared_ptr<DispatchArgsGuard>   m_dispatch_guard;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	// ResolveTexture: memoized pure descriptor -> ImageDesc decode (direct-mapped).
	struct TextureDescMemo {
		bool                                valid = false;
		uint32_t                            fields[8] {};
		ShaderRecompiler::IR::ImageResource resource;
		TextureCache::ImageDesc             desc;
		bool                                shader_conversion = false;
		vk::Format                          pixel_format      = vk::Format::eUndefined;
		vk::Format                          view_format       = vk::Format::eUndefined;
		uint64_t                            size              = 0;
		ImageId                             last_image {}; // ImageLookupHint: exact image of the last lookup
	};
	std::vector<TextureDescMemo> m_texture_desc_memo = std::vector<TextureDescMemo>(1024);
	// KYTY_DRAWPREP_DELTA: per-program (keyed by the stage's ResourceSnapshot address) memo of the last
	// PrepareBindings outputs per image / sampler slot. Every entry is revalidated against the slot's
	// descriptor words, the slot's resource and TextureCache::Generation() before it is reused.
	struct ImageSlotMemo {
		bool                                         valid = false;
		ShaderRecompiler::IR::DescriptorValue        words;
		uint64_t                                     resource_key = 0;
		uint64_t                                     generation   = 0;
		// Guest range [range_begin, range_end): union of the T# range and the resolved image's range.
		uint64_t                                     range_begin  = 0;
		uint64_t                                     range_end    = 0;
		ImageId                                      image_id;
		TextureCache::ImageDesc                      desc;
	};
	struct SamplerSlotMemo {
		bool                                         valid = false;
		ShaderRecompiler::IR::DescriptorValue        words;
		uint32_t                                     key = 0;
		vk::Sampler                                  sampler = nullptr;
	};
	struct StageBindingMemo {
		std::vector<ImageSlotMemo>   images;
		std::vector<SamplerSlotMemo> samplers;
	};
	std::unordered_map<const void*, StageBindingMemo> m_binding_memos;
	// CommitBindings scratch: ranges written by bound images / shader buffers.
	std::vector<std::pair<uint64_t, uint64_t>> m_written_image_ranges;
	std::vector<std::pair<uint64_t, uint64_t>> m_written_buffer_ranges;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	// KYTY_SKIP_REDUNDANT_BINDS: last push-descriptor/push-constant contents per bind point
	// (0 graphics, 1 compute) within the current command buffer.
	struct RedundantBindState {
		uint64_t              tick   = UINT64_MAX;
		uint64_t              epoch  = UINT64_MAX;
		const void*           buffer = nullptr; // CommandRecorder::Key()
		vk::PipelineLayout    layout = nullptr;
		std::vector<uint64_t> descriptors;
		bool                  descriptors_valid = false;
		std::vector<uint32_t> push;
		vk::ShaderStageFlags  push_stages {};
		bool                  push_valid = false;
	};
	RedundantBindState    m_redundant[2];
	std::vector<uint64_t> m_descriptor_key;
	std::vector<TableResolution>          m_table_resolutions;
	uint64_t                              m_table_resolution_tick = 0;
	// Reused by every draw; see AcquireDrawRenderState().
	std::unique_ptr<DrawRenderStorage, void (*)(DrawRenderStorage*)> m_draw_state {nullptr,
	                                                                               nullptr};
	// Resolutions ahead of execution; see ResolveDrawAhead().
	std::unique_ptr<ResolvedDrawRing, void (*)(ResolvedDrawRing*)> m_resolved_ring {nullptr,
	                                                                                nullptr};
	std::unique_ptr<ResolvedDrawRing, void (*)(ResolvedDrawRing*)> m_resolved_ring_second {nullptr,
	                                                                                       nullptr};
	ResolvedDraw*                                                  m_resolved_draw = nullptr;
	std::unique_ptr<ResolvedDispatchRing, void (*)(ResolvedDispatchRing*)> m_resolved_dispatch_ring {nullptr,
	                                                                                                nullptr};
	uint64_t                                                       m_context_id    = 0;
	// The context snapshot whose registers hw_check() last accepted.
	uint64_t m_checked_context_id = 0;
	// The last color-target descriptions per slot; see ResolveRenderColorTarget().
	std::unique_ptr<ColorTargetMemo, void (*)(ColorTargetMemo*)> m_color_target_memo {nullptr,
	                                                                                  nullptr};
	// The last depth-target descriptions; see ResolveRenderDepthTarget().
	std::unique_ptr<DepthTargetMemo, void (*)(DepthTargetMemo*)> m_depth_target_memo {nullptr,
	                                                                                  nullptr};
	// The render targets of the last draw; see RenderTargetMemo.
	std::unique_ptr<RenderTargetMemo, void (*)(RenderTargetMemo*)> m_render_target_memo {nullptr,
	                                                                                     nullptr};
	// Whether the executing draw reused the render targets of the draw before.
	bool     m_render_targets_reused = false;
	uint64_t m_draw_window           = 0;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
