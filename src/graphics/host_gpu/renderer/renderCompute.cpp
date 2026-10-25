#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/experiments.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bindEpoch.h"
#include "graphics/host_gpu/renderer/depthAliasLog.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/frameDump.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/nativeIndirect.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/fsr/fsr2Constants.h"
#include "graphics/presentation/fsr/nhl27Fsr3.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <tuple>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <tuple>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

// Stores through V#s selected from a descriptor table write guest memory by device address.
// Cache every target of the tables so the BDA page table covers them. Returns whether any table
// names a target.
static bool PrepareIndirectWriteTargets(RenderContext& context, const PreparedBindings& bindings) {
	const auto& program = *bindings.runtime->program;
	bool        prepared = false;
	for (const auto& resource: program.info.buffers) {
		// Buffer sources follow the bound resources, which leave out buffers the shader never
		// accesses, not the shader's buffer list.
		if (!resource.indirect_write_table || resource.descriptor_index == UINT32_MAX ||
		    resource.descriptor_index >= bindings.buffer_sources.size()) {
			continue;
		}
		const auto& table = bindings.buffer_sources[resource.descriptor_index];
		if (table.address == 0 || table.size <= resource.indirect_table_offset) {
			continue;
		}
		prepared |= context.GetIndirectWriteTables().Prepare(
		    table.address + resource.indirect_table_offset,
		    table.size - resource.indirect_table_offset);
	}
	return prepared;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

// Experiment FillPatterns: a proven fill (resources.uniform_pattern) whose element words differ.
// Formatted stores write only the components of the V#'s format, at the V#'s stride: the written
// bytes are the first min(words, components) words at the start of every record. Returns the
// written range, its stride and the written words.
static bool ResolveComputeBufferPattern(const ShaderComputeInputInfo& input, uint32_t group_x,
                                        uint32_t group_y, uint32_t group_z, uint32_t mode,
                                        uint64_t& address, uint64_t& size, uint32_t& stride,
                                        uint32_t& written_words, const char*& why) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_pattern;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer || fill.words < 2 ||
	    fill.words > 4 || fill.resource >= resources.buffers.size()) {
		why = "no-pattern";
		return false;
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	const auto format = std::ranges::find(formats, descriptor.Format());
	if (format == formats.end()) {
		why = "format";
		return false;
	}
	written_words = std::min<uint32_t>(fill.words, static_cast<uint32_t>(format - formats.begin()) + 1u);
	if (descriptor.Stride() < written_words * 4u || (descriptor.Stride() & 3u) != 0 ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0 || (descriptor.Base48() & 3u) != 0) {
		why = "descriptor-shape";
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		why = "dispatch-shape";
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const uint64_t bytes = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || bytes == 0 || bytes > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		why = "invocations-vs-records";
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), bytes}, fill.resource)) {
		why = "sources-overlap";
		return false;
	}
	address = descriptor.Base48();
	size    = bytes;
	stride  = descriptor.Stride();
	return true;
}

// why (optional, KYTY_DBG_FILL_DIAG): the first check that failed.
bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size, const char** why) {
	const auto fail = [why](const char* reason) {
		if (why != nullptr) *why = reason;
		return false;
	};
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return fail(fill.kind == ShaderRecompiler::IR::UniformFillKind::Image ? "image-fill"
		                                                                       : "no-fill-pattern");
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return fail("descriptor-shape");
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return fail("dispatch-shape");
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return fail("invocations-vs-records");
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return fail("sources-overlap");
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// KYTY_DBG_FILL_DIAG=1: why a compute dispatch that writes memory is not taken as a uniform fill.
// NHL 27 clears DCC/CMASK metadata with compute shaders; color-clear decoding then reads the
// metadata back (a full GPU drain) unless the fill was recognized. NHL27FILLDIAG, once per shader,
// path and verdict (at most 512 lines).
static bool FillDiagEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_FILL_DIAG");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

static void FillDiag(const ShaderComputeInputInfo& input, const char* path, uint32_t group_x,
                     uint32_t group_y, uint32_t group_z, uint32_t mode, const char* verdict) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	uint32_t    written_buffers = 0;
	uint32_t    written_images  = 0;
	for (const auto& buffer: program.info.buffers) written_buffers += buffer.written ? 1u : 0u;
	for (const auto& image: program.info.images) written_images += image.written ? 1u : 0u;
	if (written_buffers == 0 && written_images == 0) {
		return;
	}
	static std::mutex                      mutex;
	static std::unordered_set<std::string> seen;
	char                                   key[160];
	std::snprintf(key, sizeof(key), "%016llx %s %s", static_cast<unsigned long long>(program.shader_hash),
	              path, verdict);
	{
		std::scoped_lock lock {mutex};
		if (seen.size() >= 512 || !seen.insert(key).second) {
			return;
		}
	}
	std::printf("NHL27FILLDIAG: shader=0x%016llx path=%s verdict=%s fill_kind=%d words=%u spec_reads=%zu "
	            "buffers=%zu written_buffers=%u images=%zu written_images=%u groups=%ux%ux%u "
	            "threads=%ux%ux%u mode=0x%x\n",
	            static_cast<unsigned long long>(program.shader_hash), path, verdict,
	            static_cast<int>(resources.uniform_fill.kind), resources.uniform_fill.words,
	            resources.specialization_reads.size(), program.info.buffers.size(), written_buffers,
	            program.info.images.size(), written_images, group_x, group_y, group_z, input.threads_num[0],
	            input.threads_num[1], input.threads_num[2], mode);
	std::fflush(stdout);
}

// KYTY_CPU_COPY_FORWARD=1: a proven dst[i] = src[i] copy dispatch (UniformCopy) over contiguous,
// in-bounds, disjoint ranges. The dispatch still runs on the GPU; the emulator also copies the
// bytes in guest memory so CPU readers of the destination need no GPU readback.
static bool CpuCopyForwardEnabled() {
	return Common::Experiments::Enabled(Common::Experiments::CpuCopyForward);
}

static bool ResolveComputeBufferCopy(const ShaderComputeInputInfo& input, uint32_t group_x,
                                     uint32_t group_y, uint32_t group_z, uint32_t mode,
                                     uint64_t& source, uint64_t& destination, uint64_t& size) {
	const auto& resources = *input.stage.resources;
	const auto& copy      = resources.uniform_copy;
	if (!copy.valid || copy.words == 0 || copy.words > 4 ||
	    copy.source >= resources.buffers.size() || copy.destination >= resources.buffers.size()) {
		return false;
	}
	const uint32_t element = copy.words * sizeof(uint32_t);
	const auto     src = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[copy.source]);
	const auto     dst =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[copy.destination]);
	for (const auto* d: {&src, &dst}) {
		if (d->Stride() != element || d->SwizzleEnabled() || d->IndexStride() != 0 || d->AddTid() ||
		    d->Base48() == 0) {
			return false;
		}
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != copy.group_stride ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	if (input.dispatch_thread_dimensions &&
	    (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	     input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1)) {
		return false;
	}
	// Every invocation must be in bounds for both buffers, so each one copies exactly one element.
	if (invocations == 0 || invocations > src.NumRecords() || invocations > dst.NumRecords()) {
		return false;
	}
	size        = invocations * element;
	source      = src.Base48();
	destination = dst.Base48();
	if (size > (uint64_t {256} << 20u) ||
	    !(source + size <= destination || destination + size <= source)) {
		return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		// Larger fills stay on the GPU: a CPU fill would also have to be uploaded again.
		constexpr uint64_t MaxBufferFill = uint64_t {1} << 20u;
		if ((descriptor.Base48() & 3u) != 0 || size > MaxBufferFill ||
		    !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.program->shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.program->bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// KYTY_LDS_NO_READBACK=1 (experiment): an indirect dispatch does not read its GPU-written
	// arguments back to size the LDS buffer. Default OFF keeps the readback below.
	static const bool no_readback = [] {
		const char* v = std::getenv("KYTY_LDS_NO_READBACK");
		return v != nullptr && v[0] == '1';
	}();
	const bool skip_readback = no_readback && indirect_args != 0;
	if (indirect_args != 0 && !skip_readback) {
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (skip_readback) {
		// The shader addresses LDS with its own gl_WorkGroupID and gl_NumWorkGroups (GPU builtins),
		// so the CPU only has to bind the largest whole number of LDS slots that fits.
		const uint64_t stride = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(stride == 0 || stride > limit);
		size = limit / stride * stride;
		const uint64_t max_groups = size / stride;
		static std::atomic<bool> announced {false};
		if (!announced.exchange(true, std::memory_order_relaxed)) {
			std::printf("NHL27PERF: LDS no-readback active (lds_bytes=%llu buffer=%llu "
			            "max_groups=%llu)\n",
			            static_cast<unsigned long long>(stride),
			            static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(max_groups));
			std::fflush(stdout);
		}
		static std::atomic<bool> warned {false};
		if (max_groups < 4096 && !warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("NHL27PERF: WARNING LDS no-readback buffer holds only %llu workgroups "
			            "(lds_bytes=%llu); larger indirect dispatches overrun it\n",
			            static_cast<unsigned long long>(max_groups),
			            static_cast<unsigned long long>(stride));
			std::fflush(stdout);
		}
	} else if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

// KYTY_DISPATCH_FAST: vkCmdBindPipeline for a compute pipeline is elided when the very same pipeline
// is already the bound compute pipeline of the same command-buffer recording. A recording is
// identified by (VkCommandBuffer, scheduler tick): the tick advances on every submit, so a recycled
// VkCommandBuffer handle never matches. Every other compute bind (tiler, fault manager, native
// indirect prep) bumps g_compute_pipeline_epoch, which invalidates the cache. Called with the
// context mutex held, so the statics are not racy.
static bool ShouldBindComputePipeline(RenderContext& context, const CommandRecorder& recorder,
                                      vk::Pipeline pipeline) {
	const void*        cmd = recorder.Key();
	static const void* last_cmd {};
	static uint64_t          last_tick  = 0;
	static uint64_t          last_epoch = 0;
	static vk::Pipeline      last_pipeline {};
	const uint64_t tick  = context.GetCommandScheduler().CurrentTick();
	const uint64_t epoch = g_compute_pipeline_epoch.load(std::memory_order_relaxed);
	const auto     flags = Common::RuntimeFlags::Current();
	const bool     same  = (flags.dispatch_fast || flags.pipeline_memo) && last_cmd == cmd &&
	                       last_tick == tick && last_epoch == epoch && last_pipeline == pipeline &&
	                       pipeline != vk::Pipeline {};
	last_cmd      = cmd;
	last_tick     = tick;
	last_epoch    = epoch;
	last_pipeline = pipeline;
	if (same) {
		FrameBreakdown::Add(FrameBreakdown::g_counters.dc_bind_skipped);
		FrameBreakdown::VAdd(FrameBreakdown::g_pl.cs_skipped);
	}
	return !same;
}

struct RenderExecutor::ResolvedDispatch {
	ProgramResolution      programs;
	ShaderComputeInputInfo input_info;
	ShaderProgram          program;
	uint64_t               flags = 0;
	std::atomic_bool       busy {false};
};

void RenderExecutor::PrefetchResolvedDispatch(const ResolvedDispatch* resolved, int step) {
	resolved->programs.Prefetch(step);
}

// Diagnostic only (KYTY_FSR2_CAPTURE=1): finds the dispatches of NHL's AMD FSR2 passes by
// comparing their bound buffers with the guest's own copy of cbFSR2 (see Fsr2Capture), then
// prints the raw block, decoded fields, shader and every bound image. The emulator's GPU thread
// may run behind the guest CPU, so frame-varying fields (frame index, jitter, exposure, delta
// time) are only required to be close; the sizes and sequence length must be equal.
namespace {

struct Fsr2CaptureState {
	bool     checked  = false;
	bool     valid    = false;
	uint64_t base     = 0;
	uint64_t dispatches = 0;
	uint64_t matches  = 0;
	uint64_t blob_resident = 0;
	uint64_t printed  = 0;
	std::vector<std::pair<uint64_t, uint64_t>> per_hash; // shader hash -> matches
};

// The eboot image is mapped at its guest address but has no guest-memory backing, so
// TryReadBacking refuses it; fall back to the mapping itself once it is known to be readable.
bool Fsr2ReadEboot(uint64_t address, void* out, uint64_t size) {
	if (LibKernel::Memory::TryReadBacking(address, out, size)) return true;
	if (!Common::VirtualMemory::IsReadable(address, size)) return false;
	std::memcpy(out, reinterpret_cast<const void*>(address), size);
	return true;
}

bool Fsr2CaptureEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FSR2_CAPTURE");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

const char* Fsr2RoleGuess(const ImageInfo& image, bool written, const Fsr2ConstantsCandidate& c) {
	const uint32_t w = image.extent.width, h = image.extent.height;
	const bool     render  = w == c.words[0] && h == c.words[1];
	const bool     display = w == c.words[4] && h == c.words[5];
	const bool     input   = w == c.words[6] && h == c.words[7];
	switch (image.pixel_format) {
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD16Unorm: return "depth?";
		default: break;
	}
	if (w == 1 && h == 1) return "exposure?";
	if (written && display) return "output?";
	if (image.pixel_format == vk::Format::eR16G16Sfloat && (render || display || input)) return "motion-vectors?";
	if (image.pixel_format == vk::Format::eR32Sfloat && (render || input)) return written ? "dilated-depth?" : "depth?";
	if (image.pixel_format == vk::Format::eR8Unorm && (render || input)) return "reactive/transparency-mask?";
	if (!written && (render || input)) return "color?";
	return "-";
}

template <typename Program>
void Fsr2CaptureDispatch(const Program& program, const PreparedBindings& bindings, uint64_t cs_addr,
                         uint32_t gx, uint32_t gy, uint32_t gz) {
	static Fsr2CaptureState s;
	if (!s.checked) {
		s.base = Fsr2Capture::g_eboot_base.load(std::memory_order_acquire);
		if (s.base == 0) return;
		// The name is in the eboot's read-only segment, which may have no backing to read; only a
		// successful read that differs rules the build out. The size word is in .data and must
		// read back as 31; until it can be read at all, retry on later dispatches.
		char       name[7] {};
		uint32_t   size      = 0;
		const bool name_read = Fsr2ReadEboot(s.base + Fsr2Capture::CbNameOffset, name, sizeof(name));
		const bool name_ok   = !name_read || std::memcmp(name, "cbFSR2", sizeof(name)) == 0;
		const bool size_read = Fsr2ReadEboot(s.base + Fsr2Capture::GlobalConstantsOffset, &size, 4);
		if (name_ok && !size_read && ++s.dispatches < 65536) return;
		s.checked    = true;
		s.valid      = name_ok && size_read && size == 31;
		s.dispatches = 0;
		std::printf("NHL27FSR2CAP %s eboot=0x%" PRIx64 " global=0x%" PRIx64 " name_read=%d size_read=%d size=%u\n",
		            s.valid ? "enabled" : "disabled: eboot signature mismatch", s.base,
		            s.base + Fsr2Capture::GlobalConstantsOffset, name_read ? 1 : 0, size_read ? 1 : 0, size);
		std::fflush(stdout);
	}
	if (!s.valid) return;
	++s.dispatches;

	// Guest copy: { uint32Size, data[31] } -> decoder words 0..30 (word 31 is padding).
	std::array<uint32_t, 32> global_raw {};
	Fsr2ConstantsCandidate   guest;
	if (Fsr2ReadEboot(s.base + Fsr2Capture::GlobalConstantsOffset, global_raw.data(), 128)) {
		std::copy(global_raw.begin() + 1, global_raw.end(), guest.words.begin());
	}
	const bool guest_active = guest.words[0] != 0 && guest.words[1] != 0;

	const bool blob_shader = cs_addr >= s.base + Fsr2Capture::BlobRangeBegin && cs_addr < s.base + Fsr2Capture::BlobRangeEnd;
	if (blob_shader) ++s.blob_resident;

	if ((s.dispatches % 8192) == 0) {
		std::printf("NHL27FSR2CAP status dispatches=%" PRIu64 " matches=%" PRIu64 " blob_resident=%" PRIu64
		            " guest_active=%d guest_frame=%u guest_render=%ux%u guest_display=%ux%u\n",
		            s.dispatches, s.matches, s.blob_resident, guest_active ? 1 : 0, guest.words[11], guest.words[0],
		            guest.words[1], guest.words[4], guest.words[5]);
		std::fflush(stdout);
	}
	if (!guest_active && !blob_shader) return;

	auto close = [](float a, float b, float tolerance) { return std::abs(a - b) <= tolerance; };
	for (size_t slot = 0; slot < bindings.buffer_sources.size(); ++slot) {
		const auto& source = bindings.buffer_sources[slot];
		if (!source.address || source.size < 124) continue;
		Fsr2ConstantsCandidate gpu;
		const uint64_t         read_size = std::min<uint64_t>(source.size, 128);
		const char*            backing   = "clean";
		if (!LibKernel::Memory::TryReadCleanBufferBacking(source.address, gpu.words.data(), read_size)) {
			if (!LibKernel::Memory::TryReadBacking(source.address, gpu.words.data(), read_size)) continue;
			backing = "dirty-backing";
		}
		bool match = guest_active;
		for (unsigned i = 0; match && i < 11; ++i) match = gpu.words[i] == guest.words[i];
		match = match && gpu.words[27] == guest.words[27] &&
		        (gpu.words[11] - guest.words[11] + 8u) <= 16u && close(gpu.Float(104), guest.Float(104), 1e-3f);
		// Without a guest match, a blob-resident shader still gets its first cb-sized buffer logged.
		if (!match && !(blob_shader && !guest_active && gpu.Plausible())) continue;

		++s.matches;
		auto it = std::find_if(s.per_hash.begin(), s.per_hash.end(), [&](const auto& p) { return p.first == program.shader_hash; });
		if (it == s.per_hash.end()) it = s.per_hash.insert(s.per_hash.end(), {program.shader_hash, 0});
		const uint64_t n = ++it->second;
		if (!(n <= 3 || (n % 600) == 0) || s.printed >= 2000) break;
		++s.printed;

		const auto& c = gpu;
		std::printf("NHL27FSR2CAP dispatch=%" PRIu64 " match=%" PRIu64 " hash_seen=%" PRIu64 " cs_hash=%016" PRIx64
		            " cs_addr=0x%" PRIx64 "%s groups=%ux%ux%u cb_slot=%zu cb_addr=0x%" PRIx64 " cb_size=%" PRIu64
		            " read=%s guest_frame=%u gpu_frame=%u exact_guest=%d\n",
		            s.dispatches, s.matches, n, program.shader_hash, cs_addr, blob_shader ? " (eboot FSR2 blob)" : "",
		            gx, gy, gz, slot, source.address, source.size, backing, guest.words[11], c.words[11],
		            std::memcmp(c.words.data(), guest.words.data(), 124) == 0 ? 1 : 0);
		std::printf("NHL27FSR2CAP raw:");
		for (size_t i = 0; i < read_size / 4; ++i) std::printf(" %08x", c.words[i]);
		std::printf("\n");
		std::printf("NHL27FSR2CAP sizes frame=%u render=%ux%u max_render=%ux%u display=%ux%u input_color=%ux%u"
		            " luma_mip=%ux%u luma_mip_level=%u\n",
		            c.words[11], c.words[0], c.words[1], c.words[2], c.words[3], c.words[4], c.words[5], c.words[6],
		            c.words[7], c.words[8], c.words[9], c.words[10]);
		std::printf("NHL27FSR2CAP temporal jitter=%.9g,%.9g motion_vector_scale=%.9g,%.9g downscale=%.9g,%.9g"
		            " jitter_cancellation=%.9g,%.9g jitter_sequence_length=%.9g dynamic_res_change=%.9g\n",
		            c.Float(64), c.Float(68), c.Float(72), c.Float(76), c.Float(80), c.Float(84), c.Float(88),
		            c.Float(92), c.Float(108), c.Float(116));
		std::printf("NHL27FSR2CAP camera device_to_view_depth=%.9g,%.9g,%.9g,%.9g tan_half_fov_horizontal=%.9g"
		            " pre_exposure=%.9g previous_pre_exposure=%.9g delta_time_s=%.9g view_space_to_meters=%.9g\n",
		            c.Float(48), c.Float(52), c.Float(56), c.Float(60), c.Float(104), c.Float(96), c.Float(100),
		            c.Float(112), c.Float(120));
		for (size_t b = 0; b < bindings.buffer_sources.size(); ++b) {
			const auto& other = bindings.buffer_sources[b];
			const bool  w     = b < program.info.buffers.size() && program.info.buffers[b].written;
			std::printf("NHL27FSR2CAP buffer slot=%zu addr=0x%" PRIx64 " size=%" PRIu64 " %s\n", b, other.address,
			            other.size, w ? "rw" : "r");
		}
		for (size_t i = 0; i < bindings.images.size(); ++i) {
			const auto& image   = bindings.images[i].desc.info;
			const bool  has_res = i < program.info.images.size();
			const bool  written = has_res && program.info.images[i].written;
			const bool  storage = has_res && program.info.images[i].resource_class ==
			                                     ShaderRecompiler::IR::ImageResourceClass::Storage;
			std::printf("NHL27FSR2CAP image slot=%zu src=%u desc=%u %s%s addr=0x%" PRIx64 " size=0x%" PRIx64
			            " dims=%ux%ux%u mips=%u format=%s role=%s\n",
			            i, has_res ? program.info.images[i].source : 0u,
			            has_res ? program.info.images[i].descriptor_index : 0u, storage ? "storage" : "sampled",
			            written ? "+written" : "", image.data.address, image.data.size, image.extent.width,
			            image.extent.height, image.extent.depth, image.resources.levels,
			            vk::to_string(image.pixel_format).c_str(), Fsr2RoleGuess(image, written, c));
		}
		if (!bindings.table_images.empty()) {
			std::printf("NHL27FSR2CAP bindless_images=%zu (not listed)\n", bindings.table_images.size());
		}
		std::fflush(stdout);
		break;
	}
}

// KYTY_FSR3_CAPTURE=1: NHL 27 upscales with Frostbite's own Fsr3Upsampler (shader programs
// PrepareInputs, PrepareReactivity, LumaChange, Accumulation; IDA: names at eboot+0x2775700),
// not the linked but unreached AMD FSR2 SDK, so there is no guest copy of the constants to match.
// Discovery instead: a dispatch that reads a color image and writes a larger image of the same
// aspect is an upscale candidate and teaches the render/display sizes; every dispatch whose
// small constant buffer holds both size pairs is then logged with its raw words and images.
bool Fsr3CaptureEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FSR3_CAPTURE");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

struct Fsr3SizePair {
	uint32_t render_w, render_h, display_w, display_h;
};

struct Fsr3CaptureState {
	// Every upscale-shaped pair seen (bloom upsamples qualify too); the constant buffer filter
	// below is what separates the upscaler.
	std::vector<Fsr3SizePair> pairs;
	uint64_t                  dispatches = 0, matches = 0, printed = 0;
	std::vector<std::pair<uint64_t, uint64_t>> per_hash;
};

template <typename Program>
void Fsr3CaptureDispatch(const Program& program, const PreparedBindings& bindings, uint64_t cs_addr,
                         uint32_t gx, uint32_t gy, uint32_t gz) {
	static Fsr3CaptureState s;
	++s.dispatches;
	if ((s.dispatches % 16384) == 0) {
		std::printf("NHL27FSR3CAP status dispatches=%" PRIu64 " size_pairs=%zu matches=%" PRIu64 "\n", s.dispatches,
		            s.pairs.size(), s.matches);
		std::fflush(stdout);
	}

	// Learn sizes from an upscale-shaped dispatch: written image larger than a read 2D image by the
	// same factor (1.05..4) on both axes.
	const size_t image_count = bindings.images.size();
	for (size_t wi = 0; wi < image_count; ++wi) {
		if (wi >= program.info.images.size() || !program.info.images[wi].written) continue;
		const auto& out = bindings.images[wi].desc.info.extent;
		if (out.width < 640 || out.height < 360 || out.depth != 1) continue;
		for (size_t ri = 0; ri < image_count; ++ri) {
			if (ri == wi || (ri < program.info.images.size() && program.info.images[ri].written)) continue;
			const auto& in = bindings.images[ri].desc.info.extent;
			if (in.width < 320 || in.height < 180 || in.depth != 1) continue;
			const float fx = float(out.width) / float(in.width), fy = float(out.height) / float(in.height);
			if (fx < 1.05f || fx > 4.0f || std::abs(fx - fy) > 0.03f * fx) continue;
			const bool known = std::any_of(s.pairs.begin(), s.pairs.end(), [&](const Fsr3SizePair& p) {
				return p.render_w == in.width && p.render_h == in.height && p.display_w == out.width &&
				       p.display_h == out.height;
			});
			if (!known && s.pairs.size() < 32) {
				s.pairs.push_back({in.width, in.height, out.width, out.height});
				std::printf("NHL27FSR3CAP sizes cs_hash=%016" PRIx64 " render=%ux%u display=%ux%u (image slots r%zu w%zu)\n",
				            program.shader_hash, in.width, in.height, out.width, out.height, ri, wi);
				std::fflush(stdout);
			}
			wi = image_count;
			break;
		}
	}
	if (s.pairs.empty()) return;

	auto find_pair = [](const uint32_t* w, size_t n, uint32_t a, uint32_t b) -> int {
		for (size_t i = 0; i + 1 < n; ++i)
			if (w[i] == a && w[i + 1] == b) return int(i);
		return -1;
	};
	for (size_t slot = 0; slot < bindings.buffer_sources.size(); ++slot) {
		const auto& source = bindings.buffer_sources[slot];
		if (!source.address || source.size < 16 || source.size > 4096) continue;
		std::array<uint32_t, 128> words {};
		const uint64_t read_size = std::min<uint64_t>(source.size, sizeof(words)) & ~uint64_t {3};
		const char*    backing   = "clean";
		if (!LibKernel::Memory::TryReadCleanBufferBacking(source.address, words.data(), read_size)) {
			if (!LibKernel::Memory::TryReadBacking(source.address, words.data(), read_size)) continue;
			backing = "dirty-backing";
		}
		const size_t n  = read_size / 4;
		int                 ro = -1, display_off = -1;
		const Fsr3SizePair* pair = nullptr;
		for (const auto& p : s.pairs) {
			ro          = find_pair(words.data(), n, p.render_w, p.render_h);
			display_off = ro < 0 ? -1 : find_pair(words.data(), n, p.display_w, p.display_h);
			if (ro >= 0 && display_off >= 0) {
				pair = &p;
				break;
			}
		}
		if (pair == nullptr) continue;

		++s.matches;
		auto it = std::find_if(s.per_hash.begin(), s.per_hash.end(), [&](const auto& p) { return p.first == program.shader_hash; });
		if (it == s.per_hash.end()) it = s.per_hash.insert(s.per_hash.end(), {program.shader_hash, 0});
		const uint64_t hits = ++it->second;
		if (!(hits <= 3 || (hits % 600) == 0) || s.printed >= 2000) break;
		++s.printed;

		std::printf("NHL27FSR3CAP dispatch=%" PRIu64 " hash_seen=%" PRIu64 " cs_hash=%016" PRIx64 " cs_addr=0x%" PRIx64
		            " groups=%ux%ux%u cb_slot=%zu cb_addr=0x%" PRIx64 " cb_size=%" PRIu64 " read=%s render=%ux%u@w%d"
		            " display=%ux%u@w%d\n",
		            s.dispatches, hits, program.shader_hash, cs_addr, gx, gy, gz, slot, source.address, source.size,
		            backing, pair->render_w, pair->render_h, ro, pair->display_w, pair->display_h, display_off);
		std::printf("NHL27FSR3CAP raw:");
		for (size_t i = 0; i < n; ++i) std::printf(" %08x", words[i]);
		std::printf("\n");
		// Layout read off the 208-byte cb shared by all four Fsr3Upsampler passes (Oct 18 run, two
		// consecutive frames): FSR-style fDeviceToViewDepth (w3 = tan(fovY/2), w2 = aspect * w3),
		// int sizes, float sizes and reciprocals, then per-frame values. Jitter is in render pixels,
		// Halton(2,3) at frame_index + 1 with x negated. w40..w48 are tuning constants whose names
		// (Fsr3UpsamplerRenderSettings: ShadingChange*, LumaInstabilityMask*, Lock*) are not mapped.
		if (source.size == 208 && ro == 4 && display_off == 8) {
			const auto f = [&](unsigned i) { return std::bit_cast<float>(words[i]); };
			std::printf("NHL27FSR3CAP fields frame=%u render=%ux%u max_render=%ux%u display=%ux%u max_display=%ux%u"
			            " jitter=%.9g,%.9g prev_jitter=%.9g,%.9g motion_vector_scale=%.9g,%.9g downscale=%.9g,%.9g"
			            " device_to_view_depth=%.9g,%.9g,%.9g,%.9g tan_half_fov_y=%.9g delta_time_s=%.9g"
			            " pre_exposure=%.9g previous_pre_exposure=%.9g tuning=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
			            words[39], words[4], words[5], words[6], words[7], words[8], words[9], words[10], words[11],
			            f(28), f(29), f(30), f(31), f(32), f(33), f(34), f(35), f(0), f(1), f(2), f(3), f(3), f(36),
			            f(37), f(38), f(40), f(41), f(42), f(43), f(44), f(45), f(46), f(47), f(48));
		}
		// Word index, hex and float view, eight per line, for reading the layout off the log.
		for (size_t i = 0; i < n; i += 8) {
			std::printf("NHL27FSR3CAP w%03zu:", i);
			for (size_t j = i; j < std::min(n, i + 8); ++j) std::printf(" %u/%.6g", words[j], std::bit_cast<float>(words[j]));
			std::printf("\n");
		}
		for (size_t b = 0; b < bindings.buffer_sources.size(); ++b) {
			const auto& other = bindings.buffer_sources[b];
			const bool  w     = b < program.info.buffers.size() && program.info.buffers[b].written;
			std::printf("NHL27FSR3CAP buffer slot=%zu addr=0x%" PRIx64 " size=%" PRIu64 " %s\n", b, other.address,
			            other.size, w ? "rw" : "r");
		}
		for (size_t i = 0; i < image_count; ++i) {
			const auto& image   = bindings.images[i].desc.info;
			const bool  has_res = i < program.info.images.size();
			const bool  written = has_res && program.info.images[i].written;
			const bool  storage = has_res && program.info.images[i].resource_class ==
			                                     ShaderRecompiler::IR::ImageResourceClass::Storage;
			std::printf("NHL27FSR3CAP image slot=%zu src=%u desc=%u %s%s addr=0x%" PRIx64 " size=0x%" PRIx64
			            " dims=%ux%ux%u mips=%u format=%s\n",
			            i, has_res ? program.info.images[i].source : 0u,
			            has_res ? program.info.images[i].descriptor_index : 0u, storage ? "storage" : "sampled",
			            written ? "+written" : "", image.data.address, image.data.size, image.extent.width,
			            image.extent.height, image.extent.depth, image.resources.levels,
			            vk::to_string(image.pixel_format).c_str());
		}
		if (!bindings.table_images.empty()) {
			std::printf("NHL27FSR3CAP bindless_images=%zu (not listed)\n", bindings.table_images.size());
		}
		std::fflush(stdout);
		break;
	}
}

} // namespace

// As for draws: more resolutions than operations can wait between the two threads.
struct RenderExecutor::ResolvedDispatchRing {
	static constexpr size_t Size = 320;

	std::array<ResolvedDispatch, Size> dispatches;
	size_t                             next = 0;
};

void RenderExecutor::ReleaseResolvedDispatch(ResolvedDispatch* resolved) noexcept {
	resolved->busy.store(false, std::memory_order_release);
}

void RenderExecutor::PrepareComputeInputInfo(const HW::Shader& shaders, uint32_t thread_group_x,
                                             uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
                                             bool async_compute, ShaderComputeInputInfo& input_info) {
	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	const auto&        cs_regs                                  = shaders.GetCs();
	input_info.async_compute                                    = async_compute;
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
}

// Dispatches the resolve thread could not resolve ahead: [0] no program yet (or busy), [1] a
// guest-memory read it could not make (GPU-written bytes, for example).
static std::atomic<uint64_t> g_dispatch_ahead_misses[2] {};

RenderExecutor::ResolvedDispatch* RenderExecutor::ResolveDispatchIndirectAhead(const HW::Context& context,
                                                                               const HW::Shader& shaders,
                                                                               bool async_compute) {
	return ResolveDispatchAhead(context, shaders, 1, 1, 1, 0, async_compute, true);
}

RenderExecutor::ResolvedDispatch* RenderExecutor::ResolveDispatchAhead(const HW::Context& context,
                                                                       const HW::Shader& shaders,
                                                                       uint32_t thread_group_x,
                                                                       uint32_t thread_group_y,
                                                                       uint32_t thread_group_z, uint32_t mode,
                                                                       bool async_compute, bool indirect) {
	if ((!indirect && (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0)) ||
	    shaders.GetCs().cs_regs.data_addr == 0 ||
	    (indirect && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) ||
	    // Resolving ahead must not reach a shader the guest has not registered yet; the
	    // execution thread looks it up in order.
	    !ShaderMapContains({shaders.GetCs().cs_regs.data_addr})) {
		return nullptr;
	}
	Common::RuntimeFlags::EvaluationScope flags_scope;
	FrameBreakdown::Add(FrameBreakdown::g_counters.dc_ahead_attempts);
	FrameBreakdown::Timer ahead_timer(FrameBreakdown::g_counters.dc_ahead_ns);
	if (m_resolved_dispatch_ring == nullptr) {
		m_resolved_dispatch_ring = {new ResolvedDispatchRing {}, [](ResolvedDispatchRing* ring) { delete ring; }};
	}
	auto& ring     = *m_resolved_dispatch_ring;
	auto& dispatch = ring.dispatches[ring.next];
	// The execution thread releases resolutions in order.
	for (Common::SpinWait wait; dispatch.busy.load(std::memory_order_acquire);) {
		if (!wait.Spin()) {
			std::this_thread::yield();
		}
	}
	ring.next = (ring.next + 1) % ResolvedDispatchRing::Size;
	dispatch.programs.Reset();
	dispatch.input_info = {};
	if (indirect) {
		// As DispatchIndirect() builds it: the group counts are only known when it executes.
		dispatch.input_info.async_compute = async_compute;
	} else {
		PrepareComputeInputInfo(shaders, thread_group_x, thread_group_y, thread_group_z, mode, async_compute,
		                        dispatch.input_info);
	}
	dispatch.flags = Common::RuntimeFlags::g_evaluation_bits;
	dispatch.program = m_context.GetPipelineCache().GetComputeProgramAhead(
	    shaders.GetCs(), context.GetShaderRegisters(), dispatch.input_info, dispatch.programs);
	if (!dispatch.program || dispatch.programs.reads.Failed()) {
		// KYTY_DBG_CP_READS: why (printed with NHL27AHEAD by the execution thread).
		g_dispatch_ahead_misses[dispatch.programs.reads.Failed() ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	FrameBreakdown::Add(FrameBreakdown::g_counters.dc_ahead_ready);
	// Published to the execution thread with the operation that carries it.
	dispatch.busy.store(true, std::memory_order_relaxed);
	return &dispatch;
}

static bool ValidateResolvedDispatch(RenderExecutor::ResolvedDispatch* resolved) {
	if (resolved == nullptr) {
		return false;
	}
	FrameBreakdown::Timer timer(FrameBreakdown::g_counters.dc_ahead_validate_ns);
	const auto& flags = Common::RuntimeFlags::Current();
	const bool enabled = flags.compute_resolve_ahead ||
	                     Common::Experiments::Enabled(Common::Experiments::DispatchAhead);
	const bool valid = enabled && Common::RuntimeFlags::Bits(flags) == resolved->flags &&
	                   resolved->programs.reads.StillValid(true);
	FrameBreakdown::Add(valid ? FrameBreakdown::g_counters.dc_ahead_hits : FrameBreakdown::g_counters.dc_ahead_invalid);
	return valid;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute,
                                    ResolvedDispatch* resolved) {
	BufferCache::PerfTimerScope phase_timer(BufferCache::PerfTimer::Dispatch);
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	FrameBreakdown::Timer total_timer(FrameBreakdown::g_counters.dc_total_ns);
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	PrepareComputeInputInfo(sh_ctx, thread_group_x, thread_group_y, thread_group_z, mode, async_compute, input_info);
	const bool use_thread_dimensions = input_info.dispatch_thread_dimensions;
	BufferCache::PerfTimerScope program_timer(BufferCache::PerfTimer::DispatchProgram);
	m_context.GetImageHeap().BeginDraw();
	// Experiment DispatchAhead: the resolve thread may have resolved the program already; it
	// holds when the guest memory it read still has the same bytes.
	const auto compute_program = [&] {
		const bool ahead = ValidateResolvedDispatch(resolved);
		// KYTY_DBG_CP_READS=1: how many dispatches use the resolve thread's read-ahead (2 s).
		if (Libs::Graphics::CpReadDebugEnabled() && Common::Experiments::Enabled(Common::Experiments::DispatchAhead)) {
			static uint64_t counts[3] {};
			static auto     last = std::chrono::steady_clock::now();
			counts[resolved == nullptr ? 0 : ahead ? 1 : 2]++;
			if ((counts[0] + counts[1] + counts[2]) % 512 == 0 &&
			    std::chrono::steady_clock::now() - last > std::chrono::seconds(2)) {
				last = std::chrono::steady_clock::now();
				std::printf("NHL27AHEAD: dispatches resolved ahead=%llu stale=%llu not resolved=%llu "
				            "(no program %llu, failed read %llu)\n",
				            static_cast<unsigned long long>(counts[1]), static_cast<unsigned long long>(counts[2]),
				            static_cast<unsigned long long>(counts[0]),
				            static_cast<unsigned long long>(g_dispatch_ahead_misses[0].exchange(0)),
				            static_cast<unsigned long long>(g_dispatch_ahead_misses[1].exchange(0)));
				std::fflush(stdout);
				counts[0] = counts[1] = counts[2] = 0;
			}
		}
		if (ahead) {
			input_info = resolved->input_info;
			return resolved->program;
		}
		FrameBreakdown::Timer materialize_timer(FrameBreakdown::g_counters.dc_materialize_ns);
		return m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}();
	program_timer.Stop();
	if (!compute_program) {
		// NHL debugging: the shader failed to build (KYTY_DBG_SKIP_BAD_SHADERS=1); skip the dispatch.
		ResetBindings();
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (FillDiagEnabled()) {
		const char*          why = "fill";
		ShaderBufferResource diag_descriptor;
		uint32_t             diag_value = 0;
		uint64_t             diag_size  = 0;
		if (!resources.specialization_reads.empty()) {
			why = "specialization-reads";
		} else {
			(void)ResolveComputeBufferFill(input_info, thread_group_x, thread_group_y, thread_group_z, mode,
			                               diag_descriptor, diag_value, diag_size, &why);
		}
		FillDiag(input_info, "direct", thread_group_x, thread_group_y, thread_group_z, mode, why);
	}
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
	                           thread_group_z, mode))) {
		ResetBindings();
		return;
	}
	// KYTY_PERF_NO_SYNC_READBACK: a proven uniform dword fill that still executes (e.g. a DCC or
	// CMASK clear) is remembered so color-clear decoding need not read the metadata back.
	uint64_t known_fill_address = 0;
	uint64_t known_fill_size    = 0;
	uint32_t known_fill_value   = 0;
	if (BufferCache::PerfNoSyncReadback() && resources.specialization_reads.empty()) {
		ShaderBufferResource fill_descriptor;
		if (ResolveComputeBufferFill(input_info, thread_group_x, thread_group_y, thread_group_z,
		                             mode, fill_descriptor, known_fill_value, known_fill_size)) {
			known_fill_address = fill_descriptor.Base48();
		}
	}
	uint64_t known_pattern_address = 0;
	uint64_t known_pattern_size    = 0;
	uint32_t known_pattern_stride  = 0;
	uint32_t known_pattern_words   = 0;
	if (known_fill_size == 0 && BufferCache::PerfNoSyncReadback() &&
	    resources.specialization_reads.empty() &&
	    resources.uniform_pattern.kind == ShaderRecompiler::IR::UniformFillKind::Buffer &&
	    Common::Experiments::EnvEnabled(Common::Experiments::EnvBit::FillPatterns)) {
		const char* pattern_why = "ok";
		if (ResolveComputeBufferPattern(input_info, thread_group_x, thread_group_y, thread_group_z,
		                                mode, known_pattern_address, known_pattern_size,
		                                known_pattern_stride, known_pattern_words, pattern_why)) {
		} else {
			known_pattern_size = 0;
			if (FrameBreakdown::Enabled()) {
				// NHL27PATTERN reject: once per shader and reason (at most 64 lines).
				static std::mutex                      reject_mutex;
				static std::unordered_set<std::string> seen;
				char key[128];
				std::snprintf(key, sizeof(key), "%016llx %s",
				              static_cast<unsigned long long>(program.shader_hash), pattern_why);
				bool first = false;
				{
					std::scoped_lock lock {reject_mutex};
					first = seen.size() < 64 && seen.insert(key).second;
				}
				if (first) {
					const auto d = resources.uniform_pattern.resource < resources.buffers.size()
					                   ? DecodeNativeDescriptor<ShaderBufferResource>(
					                         resources.buffers[resources.uniform_pattern.resource])
					                   : ShaderBufferResource {};
					std::printf("NHL27PATTERN: reject shader=0x%016llx why=%s groups=%ux%ux%u threads=%ux%ux%u mode=0x%x "
					            "words=%u stride=%u records=%u format=%u\n",
					            static_cast<unsigned long long>(program.shader_hash), pattern_why,
					            thread_group_x, thread_group_y, thread_group_z, input_info.threads_num[0],
					            input_info.threads_num[1], input_info.threads_num[2], mode,
					            resources.uniform_pattern.words, d.Stride(), d.NumRecords(),
					            static_cast<unsigned>(d.Format()));
					std::fflush(stdout);
				}
			}
		}
	}
	uint64_t copy_source      = 0;
	uint64_t copy_destination = 0;
	uint64_t copy_size        = 0;
	if (CpuCopyForwardEnabled() && resources.specialization_reads.empty() &&
	    !ResolveComputeBufferCopy(input_info, thread_group_x, thread_group_y, thread_group_z, mode,
	                              copy_source, copy_destination, copy_size)) {
		copy_size = 0;
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	// KYTY_DBG_GDS_TRACE=1: log GDS[0..3] around dispatches that bind GDS. Debug-only: waits for
	// the GPU before recording (so "before" is exact) and again after the dispatch.
	static const bool gds_trace = [] {
		const char* v = std::getenv("KYTY_DBG_GDS_TRACE");
		return v != nullptr && v[0] == '1';
	}();
	bool     gds_trace_now = false;
	uint32_t gds_before[4] {};
	const auto read_gds = [&](uint32_t (&out)[4]) {
		const auto gds = m_context.GetBufferCache().GetGdsBuffer()->Mapped();
		if (gds.size() >= sizeof(out)) std::memcpy(out, gds.data(), sizeof(out));
	};
	if (gds_trace &&
	    program.bindings.descriptor_counts[static_cast<size_t>(
	        ShaderRecompiler::IR::DescriptorBindingKind::Gds)] != 0) {
		static uint64_t gds_dispatches = 0;
		const auto      n              = ++gds_dispatches;
		if (n <= 200 || n % 100 == 0) {
			gds_trace_now = true;
			m_context.GetCommandScheduler().Wait(m_context.GetCommandScheduler().CurrentTick());
			read_gds(gds_before);
		}
	}

	buffer.EndRendering();
	FrameBreakdown::Timer pipeline_timer(FrameBreakdown::g_counters.dc_pipeline_ns);
	FrameBreakdown::Timer cs_get_timer(FrameBreakdown::g_pl.cs_get_ns, FrameBreakdown::Verbose());
	auto& pipeline =
	    m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	cs_get_timer.Stop();
	auto& bindings = m_compute_bindings;
	FrameBreakdown::Timer cs_prep_timer(FrameBreakdown::g_pl.cs_prep_ns, FrameBreakdown::Verbose());
	PrepareBindings(input_info.stage, bindings);
	cs_prep_timer.Stop();
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	pipeline_timer.Stop();
	PreparedBindings* descriptor_stage = &bindings;
	bool indirect_writes = false;
	{
		FrameBreakdown::Timer buffers_timer(FrameBreakdown::g_counters.dc_buffers_ns);
		FindBuffers(std::span {&descriptor_stage, 1u});
		// New target buffers must exist before PrepareBda uploads the CPU writes of cached buffers.
		indirect_writes = PrepareIndirectWriteTargets(m_context, bindings);
		if (program.info.uses_dma) {
			FrameBreakdown::Timer bda_timer(FrameBreakdown::g_counters.dc_bda_ns);
			m_context.CacheDmaBases(input_info.stage);
			m_context.PrepareBda();
		}
	}
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	{
		FrameBreakdown::Timer textures_timer(FrameBreakdown::g_counters.dc_textures_ns);
		RebindImages(bindings);
	}
	{
		FrameBreakdown::Timer buffers_timer(FrameBreakdown::g_counters.dc_buffers_ns);
		BindSharedMemory(m_context, input_info, bindings);
		RebindBuffers(bindings);
	}

	auto              vk_buffer        = buffer.Handle();
	{
		FrameBreakdown::Timer descriptors_timer(FrameBreakdown::g_counters.dc_descriptors_ns);
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
	}
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) || indirect_writes;
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	BufferCache::PerfCount(BufferCache::PerfEvent::Dispatch);
	NoteShaderWork();
	if (has_storage_writes && !BufferCache::GuestSyncOnly()) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		BufferCache::PerfCount(BufferCache::PerfEvent::AutoDispatchBarrier);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	// NHL26 debugging: KYTY_DBG_SKIP_CS=<hex hash>[,<hex hash>...] skips those compute shaders.
	static const std::vector<uint64_t> skip_cs = [] {
		std::vector<uint64_t> list;
		const char*           v = std::getenv("KYTY_DBG_SKIP_CS");
		while (v != nullptr && *v != '\0') {
			char*      end  = nullptr;
			const auto hash = std::strtoull(v, &end, 16);
			if (end == v) {
				break;
			}
			list.push_back(hash);
			v = (*end == ',') ? end + 1 : end;
		}
		return list;
	}();
	if (const auto skip = std::find(skip_cs.begin(), skip_cs.end(), program.shader_hash);
	    skip != skip_cs.end()) {
		// Dispatch holds the context mutex; give each skipped shader its own log budget.
		static std::vector<uint32_t> skipped(skip_cs.size(), 0);
		if (skipped[static_cast<size_t>(skip - skip_cs.begin())]++ < 8) {
			std::printf("NHL26GDSCS: skipping CS 0x%016llx groups=%ux%ux%u "
			            "threads=%ux%ux%u wave=%u host_subgroup=%u lds_dwords=%u\n",
			            static_cast<unsigned long long>(program.shader_hash),
			            thread_group_x, thread_group_y, thread_group_z,
			            input_info.threads_num[0], input_info.threads_num[1],
			            input_info.threads_num[2], program.wave_size,
			            input_info.host_subgroup_size, input_info.lds_size_dwords);
			std::fflush(stdout);
		}
		ResetBindings();
		return;
	}
	if (DepthAliasLog::Enabled() && DepthAliasLog::Filtered()) {
		// DIAGNOSTIC (KYTY_DBG_DEPTH_ALIAS_ADDR): compute image bindings overlapping the window.
		const auto seq = DepthAliasLog::NextSeq();
		for (size_t i = 0; i < bindings.images.size() && i < program.info.images.size(); i++) {
			const auto& info = bindings.images[i].desc.info;
			if (!DepthAliasLog::Want(info.data.address, info.data.size) ||
			    !DepthAliasLog::TakeLine()) {
				continue;
			}
			const auto& resource = program.info.images[i];
			const bool  sampled  = resource.resource_class ==
			                      ShaderRecompiler::IR::ImageResourceClass::Sampled;
			std::printf("NHL27DEPTH: bind #%llu cs hash=0x%016llx addr=0x%llx size=0x%llx "
			            "fmt=%s extent=%ux%u usage=%s written=%d\n",
			            static_cast<unsigned long long>(seq),
			            static_cast<unsigned long long>(program.shader_hash),
			            static_cast<unsigned long long>(info.data.address),
			            static_cast<unsigned long long>(info.data.size),
			            vk::to_string(info.pixel_format).c_str(), info.extent.width,
			            info.extent.height, sampled ? "sampled" : "storage",
			            resource.written ? 1 : 0);
		}
	}
	FrameBreakdown::Timer record_timer(FrameBreakdown::g_counters.dc_record_ns);
	{
		FrameBreakdown::Timer cs_bind_timer(FrameBreakdown::g_pl.cs_bind_ns, FrameBreakdown::Verbose());
		if (ShouldBindComputePipeline(m_context, vk_buffer, pipeline.pipeline)) {
			FrameBreakdown::VAdd(FrameBreakdown::g_pl.cs_binds);
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		}
	}
	// GPU timing tells dispatches from the guest's compute rings apart ("dispatch-acq").
	const auto crumb = BreadcrumbBegin(m_context.GetGraphics(), vk_buffer,
	                                   GuestGpu::CurrentQueue() != 0 ? "dispatch-acq" : "dispatch",
	                                   program.shader_hash,
	                                   (static_cast<uint64_t>(thread_group_x) << 40u) |
	                                       (static_cast<uint64_t>(thread_group_y) << 20u) |
	                                       thread_group_z);
	FrameBreakdown::Add(FrameBreakdown::g_counters.dispatches);
	FrameBreakdown::Add(FrameBreakdown::g_counters.dc_handled);
	// The NHL27 FSR3 present path overwrites the game's accumulation output, so it can drop it.
	if (!Nhl27Fsr3::Enabled() || !m_nhl27_fsr3 ||
	    !m_nhl27_fsr3->ReplacesDispatch(m_context, buffer, program.shader_hash, bindings)) {
		constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
		const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
		                                   std::max(input_info.threads_num[1], 1u) *
		                                   std::max(input_info.threads_num[2], 1u);
		if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
		    group_threads <= input_info.wave_size && !input_info.lds_storage) {
			// A single guest wave per group can run in launch order without a device-wide
			// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
			for (uint32_t z = 0; z < thread_group_z; ++z) {
				for (uint32_t y = 0; y < thread_group_y; ++y) {
					for (uint32_t x = 0; x < thread_group_x; ++x) {
						if ((x | y | z) != 0u) {
							ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
						}
						vk_buffer.RecordExternal([x, y, z](vk::CommandBuffer c) {
							c.dispatchBase(x, y, z, 1, 1, 1);
						});
					}
				}
			}
		} else {
			vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
		}
	}
	BreadcrumbEnd(m_context.GetGraphics(), vk_buffer, crumb);
	m_context.GetTextureCache().FlushStorageShadows();
	record_timer.Stop();
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);


	// The removed host fence also ordered read-only dispatches before later writers.
	if (!BufferCache::GuestSyncOnly()) {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		BufferCache::PerfCount(BufferCache::PerfEvent::AutoDispatchBarrier);
	}
	// Diagnostic only: inspect already-clean constant-buffer backing without GPU waits or
	// protected guest reads. A plausible cbFSR2 match never enables reconstruction by itself.
	static const bool fsr2_probe = [] {
		const char* value = std::getenv("KYTY_FSR2_PROBE");
		return value && std::strcmp(value, "1") == 0;
	}();
	if (Fsr2CaptureEnabled()) {
		Fsr2CaptureDispatch(program, bindings, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		                    thread_group_z);
	}
	if (Fsr3CaptureEnabled()) {
		Fsr3CaptureDispatch(program, bindings, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		                    thread_group_z);
	}
	if (fsr2_probe) {
		static uint64_t dispatches = 0, clean = 0, refused = 0, candidates = 0;
		++dispatches;
		for (size_t slot = 0; slot < bindings.buffer_sources.size(); ++slot) {
			const auto& source = bindings.buffer_sources[slot];
			if (!source.address || source.size < 128) continue;
			Fsr2ConstantsCandidate constants;
			if (!LibKernel::Memory::TryReadCleanBufferBacking(source.address, constants.words.data(), 128)) {
				++refused;
				continue;
			}
			++clean;
			if (!constants.Plausible()) continue;
			if (++candidates > 64) continue;
			std::printf("NHL27FSR2 candidate=%" PRIu64 " cs=%016" PRIx64 " buf=%zu addr=%" PRIx64
			            " render=%ux%u display=%ux%u frame=%u jitter=%g,%g motion_scale=%g,%g tan_half_horizontal_fov=%g delta_seconds=%g\n",
			            candidates, program.shader_hash, slot, source.address, constants.words[0], constants.words[1],
			            constants.words[4], constants.words[5], constants.words[11], constants.Float(64), constants.Float(68),
			            constants.Float(72), constants.Float(76), constants.Float(104), constants.Float(112));
			std::printf("NHL27FSR2 words:");
			for (auto word: constants.words) std::printf(" %08x", word);
			std::printf("\n");
			for (size_t i = 0; i < bindings.images.size(); ++i) {
				const auto& image = bindings.images[i].desc.info;
				std::printf("NHL27FSR2 img=%zu addr=%" PRIx64 " size=%ux%u format=%s written=%d\n",
				            i, image.data.address, image.extent.width, image.extent.height,
				            vk::to_string(image.pixel_format).c_str(),
				            i < program.info.images.size() && program.info.images[i].written);
			}
			std::fflush(stdout);
		}
		if ((dispatches % 4096) == 0) {
			std::printf("NHL27FSR2 probe dispatches=%" PRIu64 " clean=%" PRIu64 " refused=%" PRIu64 " candidates=%" PRIu64 "\n",
			            dispatches, clean, refused, candidates);
			std::fflush(stdout);
		}
	}
	if (FrameDump::TraceEnabled()) {
		for (size_t i = 0; i < bindings.images.size() && i < program.info.images.size(); i++) {
			if (!program.info.images[i].written) continue;
			const auto& d = bindings.images[i].desc.info.data;
			FrameDump::TraceWrite("cs-image", d.address, d.address + d.size, "dispatch", program.shader_hash);
		}
	}
	if (FrameDump::TraceCsEnabled(program.shader_hash)) {
		std::vector<uint8_t> written(bindings.images.size(), 0);
		for (size_t i = 0; i < written.size() && i < program.info.images.size(); i++) {
			written[i] = program.info.images[i].written ? 1 : 0;
		}
		FrameDump::TraceDispatch(program.shader_hash, thread_group_x, thread_group_y, thread_group_z, bindings,
		                         written);
	}
	if (FrameDump::Active()) {
		std::vector<FrameDump::StorageUse> uses(bindings.images.size());
		for (size_t i = 0; i < uses.size() && i < program.info.images.size(); i++) {
			uses[i].binding = &bindings.images[i];
			uses[i].written = program.info.images[i].written;
			uses[i].storage = program.info.images[i].resource_class ==
			                  ShaderRecompiler::IR::ImageResourceClass::Storage;
		}
		FrameDump::OnDispatch(m_context, buffer, program.shader_hash, thread_group_x, thread_group_y,
		                      thread_group_z, false, uses, &bindings);
	}
	if (known_fill_size != 0) {
		// Recorded after the bindings marked the range GPU-written, which clears older records.
		m_context.GetBufferCache().RecordGpuFill(known_fill_address, known_fill_size,
		                                         known_fill_value);
	}
	if (known_pattern_size != 0) {
		m_context.GetBufferCache().RecordGpuFillPattern(
		    known_pattern_address, known_pattern_size, known_pattern_stride,
		    std::span(resources.uniform_pattern_values).first(known_pattern_words));
	}
	if (copy_size != 0) {
		auto& cache = m_context.GetBufferCache();
		// Only when the source bytes in guest memory are what the GPU reads: no pending GPU
		// writes to the source. Its CPU writes were uploaded when the source was bound above.
		if (!cache.HasGpuDirtyBytes(copy_source, copy_size)) {
			std::vector<uint8_t> bytes(static_cast<size_t>(copy_size));
			if (LibKernel::Memory::TryReadBacking(copy_source, bytes.data(), bytes.size())) {
				LibKernel::Memory::WriteBacking(copy_destination, bytes.data(), bytes.size());
				cache.ForgetGpuWrite(copy_destination, copy_size);
				static std::atomic<uint64_t> forwarded {0};
				const auto n = forwarded.fetch_add(1, std::memory_order_relaxed) + 1;
				if (n <= 4 || n % 5000 == 0) {
					std::printf("NHL27COPY: forwarded #%llu cs=%016llx 0x%llx -> 0x%llx (%llu bytes)\n",
					            static_cast<unsigned long long>(n),
					            static_cast<unsigned long long>(program.shader_hash),
					            static_cast<unsigned long long>(copy_source),
					            static_cast<unsigned long long>(copy_destination),
					            static_cast<unsigned long long>(copy_size));
				}
			}
		}
	}
	if (gds_trace_now) {
		uint32_t gds_after[4] {};
		m_context.GetCommandScheduler().Wait(m_context.GetCommandScheduler().CurrentTick());
		read_gds(gds_after);
		std::printf("NHL27GDS: cs=0x%016llx groups=%ux%ux%u gds[0..3] before=%u,%u,%u,%u "
		            "after=%u,%u,%u,%u\n",
		            static_cast<unsigned long long>(program.shader_hash), thread_group_x,
		            thread_group_y, thread_group_z, gds_before[0], gds_before[1], gds_before[2],
		            gds_before[3], gds_after[0], gds_after[1], gds_after[2], gds_after[3]);
		std::fflush(stdout);
	}
	if (Nhl27Fsr3::Enabled()) {
		if (!m_nhl27_fsr3)
			m_nhl27_fsr3 = std::make_shared<Nhl27Fsr3>(m_context.GetGraphics());
		m_nhl27_fsr3->OnDispatch(m_context, buffer, program.shader_hash, bindings);
	}
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool async_compute,
                                      ResolvedDispatch* resolved) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	BufferCache::PerfTimerScope phase_timer(BufferCache::PerfTimer::Dispatch);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	FrameBreakdown::Timer total_timer(FrameBreakdown::g_counters.dc_total_ns);
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}

	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	BufferCache::PerfTimerScope program_timer(BufferCache::PerfTimer::DispatchProgram);
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	m_context.GetImageHeap().BeginDraw();
	// As for direct dispatches: the resolve thread may have resolved the program already (it
	// holds while the guest memory it read has the same bytes).
	const auto compute_program = [&] {
		if (ValidateResolvedDispatch(resolved)) {
			input_info = resolved->input_info;
			return resolved->program;
		}
		FrameBreakdown::Timer materialize_timer(FrameBreakdown::g_counters.dc_materialize_ns);
		return m_context.GetPipelineCache().GetComputeProgram(cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	}();
	program_timer.Stop();
	if (!compute_program) {
		// NHL debugging: the shader failed to build (KYTY_DBG_SKIP_BAD_SHADERS=1); skip the dispatch.
		ResetBindings();
		return;
	}
	if (FillDiagEnabled()) {
		// Indirect dispatches are never taken as fills: their group counts are GPU data.
		FillDiag(input_info, "indirect", 0, 0, 0, mode,
		         input_info.stage.resources->uniform_fill.kind == ShaderRecompiler::IR::UniformFillKind::None
		             ? "no-fill-pattern"
		             : "fill-pattern-but-indirect");
	}
	buffer.EndRendering();
	FrameBreakdown::Timer pipeline_timer(FrameBreakdown::g_counters.dc_pipeline_ns);
	FrameBreakdown::Timer cs_get_timer(FrameBreakdown::g_pl.cs_get_ns, FrameBreakdown::Verbose());
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	cs_get_timer.Stop();
	auto& bindings = m_compute_bindings;
	FrameBreakdown::Timer cs_prep_timer(FrameBreakdown::g_pl.cs_prep_ns, FrameBreakdown::Verbose());
	PrepareBindings(input_info.stage, bindings);
	cs_prep_timer.Stop();
	pipeline_timer.Stop();
	PreparedBindings* descriptor_stage = &bindings;
	const auto& program = *input_info.stage.program;
	bool indirect_writes = false;
	{
		FrameBreakdown::Timer buffers_timer(FrameBreakdown::g_counters.dc_buffers_ns);
		FindBuffers(std::span {&descriptor_stage, 1u});
		// New target buffers must exist before PrepareBda uploads the CPU writes of cached buffers.
		indirect_writes = PrepareIndirectWriteTargets(m_context, bindings);
		if (program.info.uses_dma) {
			FrameBreakdown::Timer bda_timer(FrameBreakdown::g_counters.dc_bda_ns);
			m_context.CacheDmaBases(input_info.stage);
			m_context.PrepareBda();
		}
		BindSharedMemory(m_context, input_info, bindings, args_addr);
	}
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	{
		FrameBreakdown::Timer textures_timer(FrameBreakdown::g_counters.dc_textures_ns);
		RebindImages(bindings);
	}

	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	FrameBreakdown::Timer args_timer(FrameBreakdown::g_counters.dc_buffers_ns);
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	// KYTY_DISPATCH_ARGS_GUARD (runtime flag, default on; =0 disables) is the GPU-side check of the group counts.
	// Arguments the CPU wrote come from the stream buffer, which has no device address: the guard
	// is for counts the GPU wrote (a fatal error during an NHL 27 match load).
	const bool guard_args = Common::RuntimeFlags::Current().dispatch_args_guard && args_buffer->HasDeviceAddress();
	vk::Buffer indirect_buffer = args_buffer->Handle();
	uint64_t   indirect_offset = args_offset;
	if (guard_args) {
		if (!m_dispatch_guard) {
			m_dispatch_guard = std::make_shared<DispatchArgsGuard>(m_context.GetGraphics(),
			                                                       m_context.GetCommandScheduler());
		}
		std::tie(indirect_buffer, indirect_offset) =
		    m_dispatch_guard->Record(buffer, *args_buffer, args_offset, args_addr);
	}
	RebindBuffers(bindings);
	args_timer.Stop();
	{
		FrameBreakdown::Timer descriptors_timer(FrameBreakdown::g_counters.dc_descriptors_ns);
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
	}
	const auto vk_buffer = buffer.Handle();
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) || indirect_writes ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	FrameBreakdown::Timer barrier_timer(FrameBreakdown::g_counters.dc_barrier_ns);
	BufferCache::PerfCount(BufferCache::PerfEvent::Dispatch);
	NoteShaderWork();
	if (has_storage_writes && !BufferCache::GuestSyncOnly()) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		BufferCache::PerfCount(BufferCache::PerfEvent::AutoDispatchBarrier);
	}
	// Kept under KYTY_GUEST_SYNC: it makes earlier writes of the argument record visible to the
	// indirect read, and the guest cannot be relied on to order the host-side argument staging.
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {},
	                          1, &barrier, 0, nullptr, 0, nullptr);
	barrier_timer.Stop();
	FrameBreakdown::Timer record_timer(FrameBreakdown::g_counters.dc_record_ns);
	{
		FrameBreakdown::Timer cs_bind_timer(FrameBreakdown::g_pl.cs_bind_ns, FrameBreakdown::Verbose());
		if (ShouldBindComputePipeline(m_context, vk_buffer, pipeline.pipeline)) {
			FrameBreakdown::VAdd(FrameBreakdown::g_pl.cs_binds);
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		}
	}
	const auto crumb = BreadcrumbBegin(m_context.GetGraphics(), vk_buffer, "dispatch-indirect",
	                                   program.shader_hash, args_addr);
	FrameBreakdown::Add(FrameBreakdown::g_counters.dc_handled);
	BreadcrumbCaptureArgs(vk_buffer, crumb, args_buffer->Handle(), args_offset,
	                      sizeof(vk::DispatchIndirectCommand));
	vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
	BreadcrumbEnd(m_context.GetGraphics(), vk_buffer, crumb);
	m_context.GetTextureCache().FlushStorageShadows();
	record_timer.Stop();
	if (!BufferCache::GuestSyncOnly()) {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		BufferCache::PerfCount(BufferCache::PerfEvent::AutoDispatchBarrier);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);

	ResetBindings();
}

} // namespace Libs::Graphics
