#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <array>

#include <bit>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <limits>
#include <span>
#include <vector>

namespace Libs::Graphics {

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static bool logged_113 = false;
		if (!logged_113) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
			logged_113 = true;
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static bool logged_121 = false;
		if (!logged_121) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
			logged_121 = true;
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind),
		     program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

static void CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
	uint64_t descriptor_count = 0;
	uint32_t sampled_images   = 0;
	uint32_t stage_count      = 0;
	uint64_t storage_buffers = 0;
	uint64_t samplers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {}, stage_samplers {};

	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		if (binding.descriptorType != vk::DescriptorType::eSampledImage) goto check_storage;
		sampled_images += binding.descriptorCount;
		uint32_t stage_stage = 0;
		for (const auto& other: bindings) {
			if (other.descriptorType == vk::DescriptorType::eSampledImage &&
			    other.stageFlags == binding.stageFlags) {
				stage_stage += other.descriptorCount;
			}
		}
		stage_count = std::max(stage_count, stage_stage);
	check_storage:;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer;
		const bool sampler = binding.descriptorType == vk::DescriptorType::eSampler;
		if (storage) storage_buffers += binding.descriptorCount;
		if (sampler) samplers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (sampler) stage_samplers[stage] += binding.descriptorCount;
			else
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources ||
			    stage_samplers[stage] > limits.maxPerStageDescriptorSamplers) {
				EXIT("native descriptor stage %u exceeds device limits: buffers=%" PRIu64
				     "/%u resources=%" PRIu64 "/%u samplers=%" PRIu64 "/%u\n", stage, stage_buffers[stage],
				     limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage], limits.maxPerStageResources,
				     stage_samplers[stage], limits.maxPerStageDescriptorSamplers);
			}
		}
	}
	// Bindless image tables can need more sampled images than the device or a descriptor pool
	// provides: fail clearly instead of creating an invalid layout.
	if (sampled_images > limits.maxDescriptorSetSampledImages ||
	    stage_count > limits.maxPerStageDescriptorSampledImages ||
	    sampled_images > DescriptorHeap::MaxSampledImages) {
		EXIT("pipeline needs %u sampled images (%u in one stage); the device allows %u per set "
		     "and %u per stage, a descriptor pool holds %u\n",
		     sampled_images, stage_count, limits.maxDescriptorSetSampledImages,
		     limits.maxPerStageDescriptorSampledImages, DescriptorHeap::MaxSampledImages);
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		EXIT("native storage-buffer descriptors exceed device limit: %" PRIu64 "/%u\n",
		     storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	if (samplers > limits.maxDescriptorSetSamplers) {
		EXIT("native sampler descriptors exceed device limit: %" PRIu64 "/%u\n",
		     samplers, limits.maxDescriptorSetSamplers);
	}
	pipeline.uses_push_descriptors = descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &create, nullptr, &pipeline.descriptor_set_layout) != vk::Result::eSuccess);
}

// Debugging: KYTY_DBG_PIPELINE_STATS=<hex hash>[,...] (or "all") captures driver statistics (VGPRs,
// spills, instruction counts) and internal representations (ISA) of pipelines that use those
// shaders. Statistics print as NHL27PSTAT lines; each representation goes to _Shaders.
static bool PipelineStatsWanted(const GraphicContext& graphics, std::initializer_list<uint64_t> hashes) {
	if (!graphics.pipeline_stats_enabled) {
		return false;
	}
	static const std::pair<bool, std::vector<uint64_t>> wanted = [] {
		std::vector<uint64_t> list;
		const char*           v = std::getenv("KYTY_DBG_PIPELINE_STATS");
		if (v == nullptr) return std::pair {false, list};
		if (std::strcmp(v, "all") == 0) return std::pair {true, list};
		for (const char* p = v; *p != '\0';) {
			char*      end   = nullptr;
			const auto value = std::strtoull(p, &end, 16);
			if (end == p) break;
			list.push_back(value);
			p = (*end == ',') ? end + 1 : end;
		}
		return std::pair {false, list};
	}();
	if (wanted.first) return true;
	return std::any_of(hashes.begin(), hashes.end(), [](uint64_t hash) {
		return hash != 0 && std::find(wanted.second.begin(), wanted.second.end(), hash) != wanted.second.end();
	});
}

static void ReportPipelineStats(GraphicContext& graphics, vk::Pipeline pipeline, const char* kind,
                                uint64_t hash_a, uint64_t hash_b) {
	auto* const device       = static_cast<VkDevice>(graphics.device);
	const auto  get_props    = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
	const auto  get_stats    = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
	const auto  get_internal = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device,
	                                                      "vkGetPipelineExecutableInternalRepresentationsKHR"));
	if (get_props == nullptr || get_stats == nullptr) {
		return;
	}
	VkPipelineInfoKHR info {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
	info.pipeline   = static_cast<VkPipeline>(pipeline);
	uint32_t count  = 0;
	get_props(device, &info, &count, nullptr);
	std::vector<VkPipelineExecutablePropertiesKHR> props(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
	get_props(device, &info, &count, props.data());
	static std::atomic<uint32_t> serial {0};
	for (uint32_t e = 0; e < count; ++e) {
		VkPipelineExecutableInfoKHR exec {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
		exec.pipeline        = info.pipeline;
		exec.executableIndex = e;
		uint32_t stat_count  = 0;
		get_stats(device, &exec, &stat_count, nullptr);
		std::vector<VkPipelineExecutableStatisticKHR> stats(stat_count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
		get_stats(device, &exec, &stat_count, stats.data());
		std::string line;
		for (const auto& stat: stats) {
			char value[64];
			switch (stat.format) {
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::snprintf(value, sizeof(value), "%u", stat.value.b32); break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::snprintf(value, sizeof(value), "%lld", static_cast<long long>(stat.value.i64)); break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::snprintf(value, sizeof(value), "%llu", static_cast<unsigned long long>(stat.value.u64)); break;
				default: std::snprintf(value, sizeof(value), "%g", stat.value.f64); break;
			}
			line += std::string(" ") + stat.name + "=" + value;
		}
		std::printf("NHL27PSTAT: %s a=%016llx b=%016llx exec=%u stage=%s subgroup=%u%s\n", kind,
		            static_cast<unsigned long long>(hash_a), static_cast<unsigned long long>(hash_b), e,
		            props[e].name, props[e].subgroupSize, line.c_str());
		if (get_internal == nullptr) continue;
		uint32_t rep_count = 0;
		get_internal(device, &exec, &rep_count, nullptr);
		std::vector<VkPipelineExecutableInternalRepresentationKHR> reps(rep_count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
		get_internal(device, &exec, &rep_count, reps.data());
		std::vector<std::vector<char>> storage(rep_count);
		for (uint32_t r = 0; r < rep_count; ++r) {
			storage[r].resize(reps[r].dataSize);
			reps[r].pData = storage[r].data();
		}
		get_internal(device, &exec, &rep_count, reps.data());
		for (uint32_t r = 0; r < rep_count; ++r) {
			const auto path = Config::GetShaderLogFolder() /
			                  fmt::format("pstat_{:04d}_{}_{:016x}_{:016x}_{}_{}.txt", serial++, kind, hash_a,
			                              hash_b, e, reps[r].name);
			Common::File::CreateDirectories(path.parent_path());
			if (FILE* file = std::fopen(Common::PathToString(path).c_str(), "wb"); file != nullptr) {
				std::fwrite(storage[r].data(), 1, reps[r].dataSize, file);
				std::fclose(file);
			}
		}
	}
	std::fflush(stdout);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(tess_control_shader_module));
		}

		tess_eval_shader_module = CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(tess_eval_shader_module));
		}
	}

	EXIT_NOT_IMPLEMENTED(
	    rect_list && (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr));

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	uint32_t                          shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           NativeShaderStage(vertex_info[i].logical_stage),
		                                       .module = programs.vertex[i].module,
		                                       .pName  = "main"};
	}
	if (rect_list) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationControl,
		                                       .module = tess_control_shader_module,
		                                       .pName  = "main"};
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                       .module = tess_eval_shader_module,
		                                       .pName  = "main"};
	}
	if (ps_active) {
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                       .module = pixel_program.module,
		                                       .pName  = "main"};
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, input_attr[index].binding, input_attr[index].offset,
				     input_desc[input_attr[index].binding].stride,
				     static_cast<int>(input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static bool logged = false;
			if (!logged) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
				logged = true;
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::Bool32 color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		color_write_enable[i] = VK_TRUE;
	}

	vk::PipelineColorWriteCreateInfoEXT color_write {};
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = color_write_enable;

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		pipeline.uses_image_heap |= stage.stage.program->info.uses_image_heap;
		graphics_stages |= native_stage;
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_image_heap |= ps_input_info->stage.program->info.uses_image_heap;
	}
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {graphics_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	const std::array set_layouts {pipeline.descriptor_set_layout, graphics.image_heap_layout};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = pipeline.uses_image_heap ? 2u : 1u;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout begin VS=%" PRIu64 " PS=%" PRIu64
		     " set_layouts=1 push_constants=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0, 1u);
	}
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                   &pipeline.pipeline_layout);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout done result=%s layout=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
	dynamic_states.push_back(vk::DynamicState::eDepthBounds);
	if (rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	pipeline_info.pTessellationState = (rect_list || tessellation) ? &tessellation_state : nullptr;
	pipeline_info.pViewportState          = &viewport_state;
	pipeline_info.pRasterizationState     = &rasterizer;
	pipeline_info.pMultisampleState       = &multisampling;
	pipeline_info.pDepthStencilState      = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState        = &color_blending;
	pipeline_info.pDynamicState           = &dynamic_state;
	pipeline_info.layout                  = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex       = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     dynamic_state.dynamicStateCount);
	}
	const uint64_t stats_vs = vs_input_info.stage.program->shader_hash;
	const uint64_t stats_ps = ps_active ? ps_input_info->stage.program->shader_hash : 0;
	const bool     stats    = PipelineStatsWanted(graphics, {stats_vs, stats_ps});
	if (stats) {
		pipeline_info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR |
		                       vk::PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR;
	}
	result = graphics.device.createGraphicsPipelines(stats ? nullptr : driver_cache, 1, &pipeline_info,
	                                                 nullptr, &pipeline.pipeline);
	if (stats && result == vk::Result::eSuccess) {
		ReportPipelineStats(graphics, pipeline.pipeline, "gfx", stats_vs, stats_ps);
	}
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	if (result != vk::Result::eSuccess) {
		std::printf("vkCreateGraphicsPipelines failed: %s\n", vk::to_string(result).c_str());
		std::fflush(stdout);
		if (result == vk::Result::eErrorDeviceLost) {
			ReportDeviceLost(graphics);
		}
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);

	if (tess_control_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	}
	if (tess_eval_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_image_heap = input_info.stage.program->info.uses_image_heap;
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	const std::array set_layouts {pipeline.descriptor_set_layout, graphics.image_heap_layout};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = pipeline.uses_image_heap ? 2u : 1u;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	if (input_info.stage.program->info.uses_gds) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	const uint64_t stats_cs = input_info.stage.program->shader_hash;
	const bool     stats    = PipelineStatsWanted(graphics, {stats_cs});
	if (stats) {
		info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR |
		              vk::PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR;
	}
	result = graphics.device.createComputePipelines(stats ? nullptr : driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	if (stats && result == vk::Result::eSuccess) {
		ReportPipelineStats(graphics, pipeline.pipeline, "cs", stats_cs, 0);
	}
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	if (result != vk::Result::eSuccess) {
		std::printf("vkCreateComputePipelines failed: %s (CS hash 0x%016llx)\n",
		            vk::to_string(result).c_str(),
		            static_cast<unsigned long long>(input_info.stage.program->shader_hash));
		std::fflush(stdout);
		if (result == vk::Result::eErrorDeviceLost) {
			ReportDeviceLost(graphics);
		}
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
