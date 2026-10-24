#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <atomic>
#include <filesystem>
#include <memory>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// Guest memory that a resolution ahead of execution read. The execution thread repeats the reads
// before it uses the resolution: guest memory may have changed in between (operations that
// execute before the draw write it), and ahead of execution GPU-owned bytes cannot be read.
class GuestReadLog {
public:
	void Clear() {
		m_entries.clear();
		m_run_head = NoRun;
		m_bytes.clear();
		m_failed     = false;
		m_base       = t_resolve_base;
		m_base_valid = t_resolve_base_valid;
		m_unsafe_pages.fill(~uint64_t {0});
	}
	// Resolve thread, before each resolution: the last operation the execution thread finished.
	// Reads of pages no operation after it writes (PageWriteSeq) need no check.
	static void SetResolveBase(uint64_t executed) {
		t_resolve_base       = executed;
		t_resolve_base_valid = true;
	}
	// Reads the backing store without faulting. Strict reads are checked the way specialization
	// memory is read (GPU-owned bytes fail), the others as plain loads.
	bool               Read(uint64_t address, void* data, uint64_t size, bool strict);
	void               Fail() { m_failed = true; }
	[[nodiscard]] bool Failed() const { return m_failed; }
	// Execution thread: whether every read still gives the recorded bytes.
	// without_faults: bytes the GPU may have written since fail the check instead of being read back
	// (a plain load of their page would fault and wait for all queued GPU work).
	[[nodiscard]] bool StillValid(bool without_faults = false) const;
	// Execution thread, before the operation that checks it: pulls its arrays into the cache.
	void Prefetch() const;

private:
	static constexpr uint64_t StrictCheckBytes = 64;
	struct Entry {
		uint64_t address = 0;
		uint32_t size    = 0;
		uint32_t offset  = 0;
		bool     strict  = false;
		// The bytes may have been GPU-dirty when read (read-protected page, pending download):
		// always checked.
		bool unsafe = false;
		// The first entry of a clean run (consecutive single-page entries on one page, neither
		// strict nor possibly GPU-owned): the run's length; 0 for the other entries.
		uint32_t run_count = 0;
	};
	std::vector<Entry>   m_entries;
	// The clean run the last entry extends (StillValid checks a clean run with one lookup).
	static constexpr uint32_t NoRun = UINT32_MAX;
	uint32_t                  m_run_head = NoRun;
	[[nodiscard]] static bool RunClean(const Entry& entry) {
		return !entry.strict && !entry.unsafe && (entry.address >> 12u) == ((entry.address + entry.size - 1) >> 12u);
	}
	std::vector<uint8_t> m_bytes;
	bool                 m_failed     = false;
	uint64_t             m_base       = 0;
	bool                 m_base_valid = false;
	// The last pages Read() looked up and whether their bytes may be GPU-owned.
	std::array<uint64_t, 8> m_unsafe_pages {~uint64_t {0}, ~uint64_t {0}, ~uint64_t {0}, ~uint64_t {0},
	                                        ~uint64_t {0}, ~uint64_t {0}, ~uint64_t {0}, ~uint64_t {0}};
	std::array<bool, 8>     m_unsafe_values {};
	size_t                  m_unsafe_next = 0;
	inline static thread_local uint64_t t_resolve_base       = 0;
	inline static thread_local bool     t_resolve_base_valid = false;
};

// Storage of the shader programs a draw resolved ahead of its execution: the resource snapshots
// and specializations its stages point to, and the guest memory they read.
struct ProgramResolution {
	static constexpr uint32_t MaxStages = 4;

	std::array<ShaderRecompiler::IR::ResourceSnapshot, MaxStages>       resources;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, MaxStages> specializations;
	GuestReadLog                                                        reads;
	uint32_t                                                            stages = 0;

	void Reset() {
		reads.Clear();
		stages = 0;
	}
	// Execution thread, before the operation that uses it: step 0 pulls this structure (the
	// snapshot and log headers) into the cache, step 1 the arrays they point to.
	void Prefetch(int step) const;
};

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 116);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	// The entries past the counts are never set (and the key hash leaves them out).
	bool operator==(const PipelineVertexInputState& other) const {
		return binding_count == other.binding_count && attribute_count == other.attribute_count &&
		       std::equal(bindings.begin(), bindings.begin() + binding_count, other.bindings.begin()) &&
		       std::equal(attributes.begin(), attributes.begin() + attribute_count,
		                  other.attributes.begin());
	}
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// The owning renderer serializes access, including saves while the GPU is running.
namespace PipelineRecord { class RecordFile; struct Records; }

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, const NativeDrawShaderInfo& native_draw = {}, ProgramResolution* ahead = nullptr);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                ProgramResolution*          ahead = nullptr);
	// Resolve thread: the program of a dispatch ahead of its execution, reading guest memory
	// through `ahead`'s log; empty when it is not compiled yet or cannot be resolved ahead.
	ShaderProgram GetComputeProgramAhead(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
	                                     ShaderComputeInputInfo& input_info, ProgramResolution& ahead);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	friend struct PipelineCacheTestAccess;
	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	// Draws resolve their programs on the resolve thread and, when a resolution is not usable,
	// on the execution thread, which also resolves dispatches.
	std::atomic_flag              m_program_lock;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	// Pipelines created since the driver cache was last written; a background thread writes
	// the cache periodically so a crash does not lose the pipelines compiled until then.
	std::atomic_uint64_t          m_unsaved_pipelines {0};
	Common::Mutex                 m_write_mutex;
	std::jthread                  m_save_thread;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	// KYTY_PIPELINE_MEMO / NHL27PIPELINE: key and result of the previous GetGraphicsPipeline call. Pipelines are
	// never erased before destruction, so the pointer stays valid.
	GraphicsPipelineKey m_last_key {};
	Pipeline*           m_last_pipeline = nullptr;
	uint64_t            m_last_compute_id = 0;
	// The pipeline the last lookup found; pipelines are never removed while the cache lives.
	GraphicsPipelineKey m_last_graphics_key {};
	Pipeline*           m_last_graphics_pipeline = nullptr;

	void InitializeDriverCache();
	std::unique_ptr<PipelineRecord::RecordFile> m_record;
	void InitializeShaderRecord();
	void WarmUp(PipelineRecord::Records& records);
	void RecordGraphicsPipeline(const GraphicsPipelineKey& key,
	                            std::span<const ShaderVertexInputInfo> vertex_info,
	                            const ShaderPixelInputInfo* ps_input_info);
	// Writes the driver cache data to m_driver_cache_path; the cache must stay alive meanwhile.
	void WriteDriverCache();
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
