#include "graphics/host_gpu/renderer/image/image.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <iterator>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

[[nodiscard]] vk::ImageType HostImageType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D: return vk::ImageType::e1D;
		case Prospero::ImageType::kColor3D: return vk::ImageType::e3D;
		case Prospero::ImageType::kColor2D: return vk::ImageType::e2D;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(type));
	}
}

[[nodiscard]] vk::ImageCreateFlags
ImageCreateFlags(const GraphicContext& graphics, const ImageInfo& info, vk::ImageUsageFlags usage) {
	vk::ImageCreateFlags flags {};
	if (DepthAspectTransferFormat(info.pixel_format) == vk::Format::eUndefined) {
		flags |= vk::ImageCreateFlagBits::eMutableFormat;
		if (info.IsBlock() && graphics.supports_block_texel_view) {
			flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
		}
	}
	if (usage & vk::ImageUsageFlagBits::eStorage) {
		flags |= vk::ImageCreateFlagBits::eExtendedUsage;
	}
	if (info.IsVolume()) {
		flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	return flags;
}

[[nodiscard]] bool HasFormatFeature(vk::FormatProperties      properties,
                                    vk::FormatFeatureFlagBits feature) {
	return static_cast<bool>(properties.optimalTilingFeatures & feature);
}

[[nodiscard]] vk::ImageUsageFlags ImageUsageFlags(GraphicContext& graphics, const ImageInfo& info) {
	auto usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	if (info.IsBlock()) {
		usage |= vk::ImageUsageFlagBits::eSampled;
		if (graphics.supports_block_texel_view) {
			const auto storage = usage | vk::ImageUsageFlagBits::eStorage;
			if (graphics.GetImageFormatProperties(info.pixel_format, HostImageType(info.type),
			                                      vk::ImageTiling::eOptimal, storage,
			                                      ImageCreateFlags(graphics, info, storage),
			                                      nullptr) == vk::Result::eSuccess) {
				usage = storage;
			} else {
				static std::atomic_flag warned = ATOMIC_FLAG_INIT;
				if (!warned.test_and_set(std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Warning: format {} does not support storage access; compatible "
					    "uncompressed writes will use storage shadows.\n",
					    vk::to_string(info.pixel_format)));
				}
				if (std::getenv("KYTY_DEBUG_BLOCK_STORAGE") != nullptr) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "BlockStorage: no storage for {} type={} {}x{}x{} levels={} layers={} "
					    "addr=0x{:x} flags=0x{:x}\n",
					    vk::to_string(info.pixel_format), static_cast<int>(info.type),
					    info.extent.width, info.extent.height, info.extent.depth,
					    info.resources.levels, info.resources.layers, info.data.address,
					    static_cast<uint32_t>(ImageCreateFlags(graphics, info, storage))));
				}
			}
		}
		return usage;
	}
	const auto properties = graphics.GetFormatProperties(info.pixel_format);
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eSampledImage)) {
		usage |= vk::ImageUsageFlagBits::eSampled;
	}
	if (DepthAspectTransferFormat(info.pixel_format) != vk::Format::eUndefined) {
		usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
		if (graphics.attachment_feedback_loop_enabled && (usage & vk::ImageUsageFlagBits::eSampled)) {
			usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
		}
		return usage;
	}
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eColorAttachment)) {
		usage |= vk::ImageUsageFlagBits::eColorAttachment;
	}
	if (graphics.GetPhysicalDeviceProperties().limits.storageImageSampleCounts &
	    static_cast<vk::SampleCountFlagBits>(info.samples)) {
		usage |= vk::ImageUsageFlagBits::eStorage;
	}
	return usage;
}

void ValidateOptionalRange(GuestRange range, const char* name) {
	if (!range.ValidOrEmpty()) {
		EXIT("invalid %s image range: address=0x%016llx size=0x%016llx\n", name,
		     static_cast<unsigned long long>(range.address),
		     static_cast<unsigned long long>(range.size));
	}
}

} // namespace

vk::ImageAspectFlags Image::FullAspectMask(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD32Sfloat: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

Image::Barriers Image::GetBarriers(vk::ImageLayout                      destination_layout,
                                   vk::AccessFlags2                     destination_access,
                                   vk::PipelineStageFlags2              destination_stage,
                                   std::optional<ImageSubresourceRange> range) {
	auto& state              = backing.state;
	auto& subresource_states = backing.subresource_states;
	if (range && info.IsVolume()) {
		range->base_layer  = 0;
		range->layer_count = 1;
	}
	// Guest levels past the host chain alias its last (1x1) level.
	const uint32_t host_levels = info.HostLevels();
	if (range && range->base_level < info.resources.levels) {
		range->base_level  = std::min(range->base_level, host_levels - 1);
		range->level_count = std::min(range->level_count, host_levels - range->base_level);
	}

	const bool partial =
	    range && (range->base_level != 0 || range->level_count != host_levels ||
	              range->base_layer != 0 || range->layer_count != info.resources.layers);
	const bool has_subresource_states = !subresource_states.empty();

	Barriers barriers;
	if (partial || has_subresource_states) {
		if (!has_subresource_states) {
			subresource_states.resize(host_levels * info.resources.layers, state);
		}

		const uint32_t base_level  = partial ? range->base_level : 0;
		const uint32_t level_count = partial ? range->level_count : host_levels;
		const uint32_t base_layer  = partial ? range->base_layer : 0;
		const uint32_t layer_count = partial ? range->layer_count : info.resources.layers;
		for (uint32_t level = base_level; level < base_level + level_count; level++) {
			for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
				const auto index = level * info.resources.layers + layer;
				EXIT_IF(index >= subresource_states.size());
				auto& subresource_state = subresource_states[index];

				constexpr auto write_access = vk::AccessFlagBits2::eTransferWrite |
				                              vk::AccessFlagBits2::eShaderWrite |
				                              vk::AccessFlagBits2::eMemoryWrite;
				const bool     repeated_write =
				    static_cast<bool>(subresource_state.access_mask & write_access);
				if (subresource_state.layout != destination_layout ||
				    subresource_state.access_mask != destination_access || repeated_write) {
					vk::ImageMemoryBarrier2 barrier {};
					barrier.srcStageMask                    = subresource_state.pl_stage;
					barrier.srcAccessMask                   = subresource_state.access_mask;
					barrier.dstStageMask                    = destination_stage;
					barrier.dstAccessMask                   = destination_access;
					barrier.oldLayout                       = subresource_state.layout;
					barrier.newLayout                       = destination_layout;
					barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.image                           = backing.image;
					barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
					barrier.subresourceRange.baseMipLevel   = level;
					barrier.subresourceRange.levelCount     = 1;
					barrier.subresourceRange.baseArrayLayer = layer;
					barrier.subresourceRange.layerCount     = 1;
					barriers.push_back(barrier);
					subresource_state = {destination_stage, destination_access, destination_layout};
				}
			}
		}

		if (!partial) {
			subresource_states.clear();
		}
	} else {
		constexpr auto write_access   = vk::AccessFlagBits2::eTransferWrite |
		                                vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eMemoryWrite;
		const bool     repeated_write = static_cast<bool>(state.access_mask & write_access);
		if (state.layout == destination_layout && state.access_mask == destination_access &&
		    !repeated_write) {
			return {};
		}

		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask                    = state.pl_stage;
		barrier.srcAccessMask                   = state.access_mask;
		barrier.dstStageMask                    = destination_stage;
		barrier.dstAccessMask                   = destination_access;
		barrier.oldLayout                       = state.layout;
		barrier.newLayout                       = destination_layout;
		barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                           = backing.image;
		barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
		barrier.subresourceRange.baseMipLevel   = 0;
		barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
		barriers.push_back(barrier);
	}

	// Resident heaps sample their images in the read state; only leaving it invalidates them.
	if (std::ranges::any_of(barriers, [&](const vk::ImageMemoryBarrier2& barrier) {
		    return barrier.oldLayout == ReadLayout() &&
		           barrier.srcAccessMask == vk::AccessFlagBits2::eShaderRead;
	    })) {
		HeapChanged();
	}
	// A barrier between writes that keeps the layout and access (storage images, written every
	// dispatch) changes nothing texture bindings depend on.
	if (has_subresource_states != !subresource_states.empty() ||
	    std::ranges::any_of(barriers, [](const vk::ImageMemoryBarrier2& barrier) {
		    return barrier.oldLayout != barrier.newLayout || barrier.srcAccessMask != barrier.dstAccessMask;
	    })) {
		TextureStateChanged();
	}
	state = {destination_stage, destination_access, destination_layout};
	return barriers;
}

void Image::Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
                    std::optional<ImageSubresourceRange> range, CommandRecorder command_buffer) {
	const auto transfer_access =
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	vk::PipelineStageFlags2 destination_stage {};
	if (static_cast<bool>(destination_access & transfer_access)) {
		destination_stage |= vk::PipelineStageFlagBits2::eTransfer;
	}
	if (!destination_access ||
	    static_cast<bool>(destination_access & ~vk::AccessFlags2 {transfer_access})) {
		destination_stage |=
		    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
	}
	const auto barriers =
	    GetBarriers(destination_layout, destination_access, destination_stage, range);
	if (barriers.empty()) {
		return;
	}
	FrameBreakdown::TexAdd(FrameBreakdown::g_tex.transits);
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_BARRIER);
	m_scheduler.EndRendering();
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command_buffer.pipelineBarrier2(dependency);
}

void Image::Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
                   uint64_t size) {
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_TEX_UPLOAD);
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	// Guest levels past the host chain have no host storage; leave them in guest memory.
	std::vector<vk::BufferImageCopy> host_copies;
	if (std::ranges::any_of(copies, [&](const auto& copy) {
		    return copy.imageSubresource.mipLevel >= backing.mip_levels;
	    })) {
		std::ranges::copy_if(copies, std::back_inserter(host_copies), [&](const auto& copy) {
			return copy.imageSubresource.mipLevel < backing.mip_levels;
		});
		copies = host_copies;
		if (copies.empty()) {
			return;
		}
	}
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyBufferToImage(buffer, backing.image, vk::ImageLayout::eTransferDstOptimal,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                     uint64_t offset, uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	// Guest levels past the host chain are 1x1 like the last host level; read that one back.
	std::vector<vk::BufferImageCopy> host_copies;
	if (std::ranges::any_of(copies, [&](const auto& copy) {
		    return copy.imageSubresource.mipLevel >= backing.mip_levels;
	    })) {
		const auto last = backing.mip_levels - 1;
		host_copies.assign(copies.begin(), copies.end());
		for (auto& copy: host_copies) {
			if (copy.imageSubresource.mipLevel > last) {
				copy.imageSubresource.mipLevel = last;
				copy.imageOffset.x             = 0;
				copy.imageOffset.y             = 0;
				copy.imageExtent.width  = std::min(copy.imageExtent.width,
				                                   std::max(backing.extent.width >> last, 1u));
				copy.imageExtent.height = std::min(copy.imageExtent.height,
				                                   std::max(backing.extent.height >> last, 1u));
				if (backing.image_type == vk::ImageType::e3D) {
					copy.imageOffset.z     = 0;
					copy.imageExtent.depth = std::min(copy.imageExtent.depth,
					                                  std::max(backing.extent.depth >> last, 1u));
				}
			}
		}
		copies = host_copies;
	}
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyImageToBuffer(backing.image, vk::ImageLayout::eTransferSrcOptimal, buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
}

std::pair<uint32_t, uint32_t> Image::SanitizeCopyLayers(const Image& source,
                                                        const Image& destination, uint32_t depth) {
	const auto source_type        = source.backing.image_type;
	const auto destination_type   = destination.backing.image_type;
	uint32_t   source_layers      = source.backing.layers;
	uint32_t   destination_layers = destination.backing.layers;
	if (source_type == vk::ImageType::e3D) {
		source_layers = 1;
	}
	if (destination_type == vk::ImageType::e3D) {
		destination_layers = 1;
	}
	if (source_type == destination_type) {
		source_layers = destination_layers = std::min(source_layers, destination_layers);
	} else if (source_type == vk::ImageType::e2D && destination_type == vk::ImageType::e3D) {
		source_layers = depth;
	} else if (source_type == vk::ImageType::e3D && destination_type == vk::ImageType::e2D) {
		destination_layers = depth;
	}
	return {source_layers, destination_layers};
}

void Image::CopyImage(Image& source) {
	EXIT_IF(source.backing.samples != backing.samples);
	m_scheduler.EndRendering();
	const uint32_t levels     = std::min(source.backing.mip_levels, backing.mip_levels);
	const uint32_t base_depth = source.backing.image_type == backing.image_type
	                                ? std::min(source.backing.extent.depth, backing.extent.depth)
	                            : backing.image_type == vk::ImageType::e3D
	                                ? backing.extent.depth
	                                : source.backing.extent.depth;
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	// Extents are in source texels. Clamp in texel blocks so neither image's mip is overrun.
	const auto source_block      = vk::blockExtent(source.backing.format);
	const auto destination_block = vk::blockExtent(backing.format);
	const auto fit               = [](uint32_t source_size, uint32_t destination_size,
                                uint32_t source_unit, uint32_t destination_unit) {
		const auto blocks = std::min((source_size + source_unit - 1) / source_unit,
		                             (destination_size + destination_unit - 1) / destination_unit);
		return std::min(source_size, blocks * source_unit);
	};
	std::vector<vk::ImageCopy> copies;
	copies.reserve(levels);
	for (uint32_t level = 0; level < levels; level++) {
		// Overlapping cache images can differ in size; copying the source extent into a smaller
		// destination writes out of bounds and loses the device. Clamp in texel blocks so
		// neither image's mip is overrun.
		const auto width = std::min(
		    fit(std::max(source.backing.extent.width >> level, 1u),
		        std::max(backing.extent.width >> level, 1u), source_block[0], destination_block[0]),
		    std::max(backing.extent.width >> level, 1u));
		const auto height = std::min(
		    fit(std::max(source.backing.extent.height >> level, 1u),
		        std::max(backing.extent.height >> level, 1u), source_block[1], destination_block[1]),
		    std::max(backing.extent.height >> level, 1u));
		const auto depth  = std::max(base_depth >> level, 1u);
		auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
		// Mixed 2D/3D copies map the 3D side's slices to array layers; the 2D side may have fewer.
		source_layers      = std::min(source_layers, source.backing.layers);
		destination_layers = std::min(destination_layers, backing.layers);
		vk::ImageCopy copy {};
		copy.srcSubresource = {source_aspect, level, 0, 1};
		copy.dstSubresource = {destination_aspect, level, 0, 1};
		if (source.backing.image_type == backing.image_type) {
			if (source.backing.image_type == vk::ImageType::e3D) {
				copy.extent = {width, height, depth};
			} else {
				copy.srcSubresource.layerCount = std::min(source_layers, destination_layers);
				copy.dstSubresource.layerCount = copy.srcSubresource.layerCount;
				copy.extent                    = {width, height, 1};
			}
		} else if (source.backing.image_type == vk::ImageType::e2D) {
			copy.srcSubresource.layerCount = source_layers;
			copy.extent                    = {width, height, source_layers};
		} else {
			copy.dstSubresource.layerCount = destination_layers;
			copy.extent                    = {width, height, destination_layers};
		}
		copies.push_back(copy);
	}
	if (copies.empty()) {
		return;
	}
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, static_cast<uint32_t>(copies.size()),
	                  copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Resolve(Image& source, const ImageSubresourceRange& source_range,
                    const ImageSubresourceRange& destination_range) {
	EXIT_IF(backing.samples != 1 || source.backing.image_type != vk::ImageType::e2D ||
	        backing.image_type != vk::ImageType::e2D || source_range.level_count != 1 ||
	        destination_range.level_count != 1 ||
	        source_range.base_level >= source.backing.mip_levels ||
	        destination_range.base_level >= backing.mip_levels ||
	        source_range.base_layer >= source.backing.layers ||
	        destination_range.base_layer >= backing.layers);
	const auto layers       = std::min({source_range.layer_count, destination_range.layer_count,
	                                    source.backing.layers - source_range.base_layer,
	                                    backing.layers - destination_range.base_layer});
	const auto source_width = std::max(source.backing.extent.width >> source_range.base_level, 1u);
	const auto source_height =
	    std::max(source.backing.extent.height >> source_range.base_level, 1u);
	const auto destination_width =
	    std::max(backing.extent.width >> destination_range.base_level, 1u);
	const auto destination_height =
	    std::max(backing.extent.height >> destination_range.base_level, 1u);
	const bool copy = source.backing.samples == 1;
	EXIT_IF(layers == 0 || info.extent.width > source_width || info.extent.height > source_height ||
	        info.extent.width > destination_width || info.extent.height > destination_height ||
	        (copy ? !ImageViewOps::FormatsCompatible(source.backing.format, backing.format)
	              : source.backing.format != backing.format));
	auto resolved_source_range             = source_range;
	auto resolved_destination_range        = destination_range;
	resolved_source_range.layer_count      = layers;
	resolved_destination_range.layer_count = layers;
	const vk::Extent3D resolve_extent {info.extent.width, info.extent.height, 1};

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               resolved_source_range, command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	        resolved_destination_range, command);
	if (copy) {
		vk::ImageCopy region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, region);
	} else {
		vk::ImageResolve region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.resolveImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                     backing.image, vk::ImageLayout::eTransferDstOptimal, region);
	}
}

uint32_t Image::CopyRows(uint64_t row_size, uint32_t rows, uint64_t capacity) noexcept {
	if (row_size == 0 || rows == 0 || row_size > capacity) {
		return 0;
	}
	return static_cast<uint32_t>(std::min<uint64_t>(rows, capacity / row_size));
}

void Image::CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler) {
	EXIT_IF(buffer.Handle() == nullptr || source.backing.samples != 1 || backing.samples != 1);
	m_scheduler.EndRendering();
	const uint32_t levels = std::min(source.backing.mip_levels, backing.mip_levels);
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto     source_bytes      = DepthAspectTransferBytes(source.backing.format) != 0
	                                       ? DepthAspectTransferBytes(source.backing.format)
	                                       : source.info.bytes_per_block;
	const auto     destination_bytes = DepthAspectTransferBytes(backing.format) != 0
	                                       ? DepthAspectTransferBytes(backing.format)
	                                       : info.bytes_per_block;
	const uint32_t source_block      = source.info.IsBlock() ? 4u : 1u;
	const uint32_t destination_block = info.IsBlock() ? 4u : 1u;
	EXIT_IF(levels == 0 || source_bytes == 0 || source_bytes != destination_bytes ||
	        source_block != destination_block);
	const auto source_transform = source.info.GetColorTransform();
	const auto target_transform = info.GetColorTransform();

	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eTransferRead;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer.Handle();
	barrier.offset              = 0;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	auto command                        = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	for (uint32_t level = 0; level < levels; level++) {
		// Copy the blocks both mips hold; each side's extent stops at its own edge.
		const auto source_width      = std::max(source.backing.extent.width >> level, 1u);
		const auto source_height     = std::max(source.backing.extent.height >> level, 1u);
		const auto target_width      = std::max(backing.extent.width >> level, 1u);
		const auto target_height     = std::max(backing.extent.height >> level, 1u);
		const auto blocks_x          = std::min((source_width + source_block - 1) / source_block,
		                                        (target_width + source_block - 1) / source_block);
		const auto block_rows        = std::min((source_height + source_block - 1) / source_block,
		                                        (target_height + source_block - 1) / source_block);
		const auto width             = std::min(source_width, blocks_x * source_block);
		const auto target_copy_width = std::min(target_width, blocks_x * source_block);
		const auto source_depth      = source.backing.image_type == vk::ImageType::e3D
		                                   ? std::max(source.backing.extent.depth >> level, 1u)
		                                   : source.backing.layers;
		const auto destination_depth = backing.image_type == vk::ImageType::e3D
		                                   ? std::max(backing.extent.depth >> level, 1u)
		                                   : backing.layers;
		const auto slices            = std::min(source_depth, destination_depth);
		const auto row_size          = static_cast<uint64_t>(blocks_x) * source_bytes;
		const auto rows_per_copy = CopyRows(row_size, block_rows, buffer.Size());
		EXIT_IF(slices == 0 || rows_per_copy == 0);
		for (uint32_t slice = 0; slice < slices; slice++) {
			for (uint32_t block_row = 0; block_row < block_rows; block_row += rows_per_copy) {
				const auto          copy_rows   = std::min(rows_per_copy, block_rows - block_row);
				const auto          y           = block_row * source_block;
				const auto          copy_height = std::min(copy_rows * source_block, source_height - y);
				const auto          copy_size   = row_size * copy_rows;
				vk::BufferImageCopy source_copy {};
				source_copy.imageSubresource = {
				    source_aspect, level,
				    source.backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				source_copy.imageOffset           = {0, static_cast<int32_t>(y),
				                                     source.backing.image_type == vk::ImageType::e3D
				                                         ? static_cast<int32_t>(slice)
				                                         : 0};
				source_copy.imageExtent           = {width, copy_height, 1};
				auto destination_copy             = source_copy;
				destination_copy.imageSubresource = {
				    destination_aspect, level,
				    backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				destination_copy.imageOffset.z =
				    backing.image_type == vk::ImageType::e3D ? static_cast<int32_t>(slice) : 0;
				destination_copy.imageExtent = {
				    target_copy_width, std::min(copy_rows * source_block, target_height - y), 1};
				barrier.size          = copy_size;
				barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
				barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
				command.pipelineBarrier2(dependency);
				command.copyImageToBuffer(source.backing.image,
				                          vk::ImageLayout::eTransferSrcOptimal, buffer.Handle(),
				                          source_copy);
				if (source_transform != target_transform) {
					const TileManager::Result bytes {buffer.Handle(), 0, copy_size};
					if (source_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, source_transform, false);
					}
					if (target_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, target_transform, true);
					}
				} else {
					barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
					barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
					command.pipelineBarrier2(dependency);
				}
				command.copyBufferToImage(buffer.Handle(), backing.image,
				                          vk::ImageLayout::eTransferDstOptimal, destination_copy);
			}
		}
	}
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::CopyMip(Image& source, uint32_t mip, uint32_t layer) {
	if (mip >= backing.mip_levels && mip < info.resources.levels) {
		return; // Guest level past the host chain: no host storage to fill.
	}
	EXIT_IF(source.backing.samples != backing.samples || mip >= backing.mip_levels ||
	        layer >= backing.layers);
	m_scheduler.EndRendering();
	const auto width  = std::max(backing.extent.width >> mip, 1u);
	const auto height = std::max(backing.extent.height >> mip, 1u);
	const auto depth  = std::max(backing.extent.depth >> mip, 1u);
	EXIT_IF(width != source.backing.extent.width || height != source.backing.extent.height);
	const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
	const auto aspects                             = FullAspectMask(source.backing.format);
	EXIT_IF(aspects != FullAspectMask(backing.format));
	std::array<vk::ImageCopy, 2> copies {};
	uint32_t                     copy_count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth,
	                         vk::ImageAspectFlagBits::eStencil}) {
		if (!static_cast<bool>(aspects & aspect)) {
			continue;
		}
		auto& copy          = copies[copy_count++];
		copy.srcSubresource = {aspect, 0, 0, source_layers};
		copy.dstSubresource = {aspect, mip, layer, destination_layers};
		copy.extent         = {width, height, depth};
	}
	auto command = m_scheduler.Current().Handle();
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy_count, copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

namespace ImageOps {

void Validate(const ImageInfo& info) {
	ValidateOptionalRange(info.data, "data");
	ValidateOptionalRange(info.stencil, "stencil");

	if (info.pixel_format == vk::Format::eUndefined) {
		const bool metadata_empty =
		    info.metadata.range.Empty() && info.metadata.kind == ImageMetadataKind::None &&
		    info.metadata.control == 0 &&
		    info.metadata.compression == VideoOutCompression::Uncompressed &&
		    !info.metadata.stencil_compressed;
		if (info.data.Empty() || info.HasStencil() || !metadata_empty || info.extent.width == 0 ||
		    info.extent.height == 0 || info.extent.depth == 0 || info.resources.levels != 1 ||
		    info.resources.layers != 1 || info.samples != 1 || info.pitch != 0 ||
		    info.bytes_per_block != 0) {
			EXIT("invalid stencil association image\n");
		}
		return;
	}

	if (info.extent.width == 0 || info.extent.height == 0 || info.extent.depth == 0 ||
	    info.resources.levels == 0 || info.resources.levels > info.mip_layout.size() ||
	    info.resources.layers == 0 || info.samples == 0 ||
	    vulkan_sample_count(info.samples) == vk::SampleCountFlagBits {} ||
	    info.bytes_per_block == 0 || (info.data.address != 0 && info.pitch == 0)) {
		EXIT("invalid image geometry or format\n");
	}

	switch (info.type) {
		case Prospero::ImageType::kColor1D:
			if (info.extent.height != 1 || info.extent.depth != 1) {
				EXIT("invalid 1D image shape\n");
			}
			break;
		case Prospero::ImageType::kColor3D:
			if (info.resources.layers != 1) {
				EXIT("3D images cannot have array layers\n");
			}
			break;
		case Prospero::ImageType::kColor2D:
			if (info.extent.depth != 1) {
				EXIT("invalid 2D image shape\n");
			}
			break;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(info.type));
	}
	if (info.samples > 1 && info.resources.levels != 1) {
		EXIT("multisampled images cannot have mip levels\n");
	}

	if (info.metadata.stencil_compressed && !info.HasStencil()) {
		EXIT("compressed stencil metadata requires a stencil plane\n");
	}
	switch (info.metadata.kind) {
		case ImageMetadataKind::None:
			if (!info.metadata.range.Empty() || info.metadata.control != 0 ||
			    info.metadata.compression != VideoOutCompression::Uncompressed ||
			    info.metadata.stencil_compressed) {
				EXIT("metadata-free image has metadata state\n");
			}
			break;
		case ImageMetadataKind::Htile:
			if (!info.metadata.range.Valid() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("invalid HTILE metadata\n");
			}
			break;
		case ImageMetadataKind::Dcc:
		case ImageMetadataKind::Cmask:
			if (!GuestRange {info.metadata.range.address,
			                 std::max<uint64_t>(info.metadata.range.size, 1)}.Valid() ||
			    info.metadata.compression == VideoOutCompression::Unsupported) {
				EXIT("invalid color metadata\n");
			}
			break;
	}
}

Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element) {
	switch (bytes_per_element) {
		case 1: return Prospero::BufferFormat::k8UNorm;
		case 2: return Prospero::BufferFormat::k16UNorm;
		case 4: return Prospero::BufferFormat::k32Float;
		case 8: return Prospero::BufferFormat::k16_16_16_16Float;
		case 16: return Prospero::BufferFormat::k32_32_32_32Float;
		default: EXIT("unsupported render-target element size: %u\n", bytes_per_element);
	}
}

} // namespace ImageOps

Image::Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& image_info)
    : info(image_info),
      stencil_subresources {0, image_info.resources.levels, 0, image_info.resources.layers},
      m_graphics(graphics), m_scheduler(scheduler) {
	KYTY_PROFILER_FUNCTION();
	ImageOps::Validate(info);
	TextureStateChanged();
	m_cpu_dirty =
	    !info.data.Empty() && info.metadata.compression == VideoOutCompression::Uncompressed;
	if (info.pixel_format == vk::Format::eUndefined) {
		return;
	}

	vk::ImageCreateInfo create {};
	create.usage         = ImageUsageFlags(graphics, info);
	create.flags         = ImageCreateFlags(graphics, info, create.usage);
	create.imageType     = HostImageType(info.type);
	create.extent        = info.extent;
	create.mipLevels     = info.HostLevels();
	create.arrayLayers   = info.IsVolume() ? 1u : info.resources.layers;
	create.format        = info.pixel_format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.samples       = vulkan_sample_count(info.samples);

	vk::ImageFormatProperties properties {};
	if (graphics.GetImageFormatProperties(create.format, create.imageType, create.tiling,
	                                      create.usage, create.flags,
	                                      &properties) != vk::Result::eSuccess ||
	    !static_cast<bool>(properties.sampleCounts & create.samples)) {
		EXIT("image format does not support required usage: format=%d type=%d usage=0x%x "
		     "flags=0x%x samples=%u\n",
		     static_cast<int>(create.format), static_cast<int>(create.imageType),
		     static_cast<vk::ImageUsageFlags::MaskType>(create.usage),
		     static_cast<vk::ImageCreateFlags::MaskType>(create.flags), info.samples);
	}

	if (!graphics.CreateImage(create, backing)) {
		EXIT("failed to create image: extent=%ux%ux%u format=%d layers=%u levels=%u\n",
		     create.extent.width, create.extent.height, create.extent.depth,
		     static_cast<int>(create.format), create.arrayLayers, create.mipLevels);
	}
	SetVulkanObjectNameF(
	    graphics.device, backing.image,
	    "Kyty.Image[guest=0x{:016x} size=0x{:x} extent={}x{}x{} format={} mips={} layers={} samples={}]",
	    info.data.address, info.data.size, info.extent.width, info.extent.height, info.extent.depth,
	    static_cast<uint32_t>(info.pixel_format), info.resources.levels, info.resources.layers,
	    info.samples);
}

uint64_t Image::HashGuestEdges() const {
	std::array<uint8_t, TRACKER_PAGE_SIZE * 2> bytes {};
	const auto                                 range = info.data;
	const uint64_t head_end =
	    std::min(range.End(), Common::AlignUp(range.address, TRACKER_PAGE_SIZE));
	const uint64_t tail_begin =
	    std::max(range.address, Common::AlignDown(range.End(), TRACKER_PAGE_SIZE));
	const uint64_t head_size    = head_end - range.address;
	const uint64_t tail_address = tail_begin < head_end ? head_end : tail_begin;
	const uint64_t tail_size    = range.End() - tail_address;
	if ((head_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(range.address, bytes.data(), head_size)) ||
	    (tail_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(tail_address, bytes.data() + head_size, tail_size))) {
		EXIT("Image: failed to hash guest backing\n");
	}
	return XXH3_64bits(bytes.data(), static_cast<size_t>(head_size + tail_size));
}

Image::ShadowLevel& Image::ShadowLevelAt(uint32_t level) {
	EXIT_IF(level >= backing.mip_levels);
	if (m_shadow == nullptr) {
		m_shadow = std::make_unique<StorageShadow>();
		m_shadow->levels.resize(backing.mip_levels);
	}
	auto& slot = m_shadow->levels[level];
	if (slot != nullptr) {
		return *slot;
	}
	EXIT_IF(info.bytes_per_block != 16 && info.bytes_per_block != 8);
	const bool          volume = backing.image_type == vk::ImageType::e3D;
	const auto          blocks = [level](uint32_t texels) {
		return (std::max(texels >> level, 1u) + 3u) / 4u;
	};
	vk::ImageCreateInfo create {};
	create.flags = vk::ImageCreateFlagBits::eMutableFormat;
	if (volume) {
		create.flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	create.imageType     = backing.image_type;
	create.extent        = {blocks(backing.extent.width), blocks(backing.extent.height),
	                        volume ? std::max(backing.extent.depth >> level, 1u) : 1u};
	create.mipLevels     = 1;
	create.arrayLayers   = backing.layers;
	create.format        = info.bytes_per_block == 16 ? vk::Format::eR32G32B32A32Uint
	                                                  : vk::Format::eR32G32Uint;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
	               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.samples = vk::SampleCountFlagBits::e1;
	if (std::getenv("KYTY_DEBUG_BLOCK_STORAGE") != nullptr) {
		std::printf("BlockStorage: shadow for format %d level %u %ux%ux%u layers=%u addr=0x%llx\n",
		            static_cast<int>(info.pixel_format), level, create.extent.width,
		            create.extent.height, create.extent.depth, backing.layers,
		            static_cast<unsigned long long>(info.data.address));
	}
	slot = std::make_unique<ShadowLevel>();
	if (!m_graphics.CreateImage(create, slot->image)) {
		EXIT("failed to create block storage shadow: extent=%ux%ux%u layers=%u level=%u\n",
		     create.extent.width, create.extent.height, create.extent.depth, create.arrayLayers,
		     level);
	}
	SetVulkanObjectNameF(m_graphics.device, slot->image.image,
	                     "Kyty.BlockStorageShadow[guest=0x{:016x} format={} level={}]",
	                     info.data.address, static_cast<uint32_t>(info.pixel_format), level);
	return *slot;
}

void Image::ShadowBarrier(ShadowLevel& shadow, const CommandRecorder& command,
                          vk::PipelineStageFlags2 destination_stage,
                          vk::AccessFlags2        destination_access) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask        = destination_stage;
	barrier.dstAccessMask       = destination_access;
	barrier.oldLayout           = shadow.initialized ? vk::ImageLayout::eGeneral
	                                                 : vk::ImageLayout::eUndefined;
	barrier.newLayout           = vk::ImageLayout::eGeneral;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = shadow.image.image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0,
	                               VK_REMAINING_ARRAY_LAYERS};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	shadow.initialized = true;
}

void Image::CopyStorageShadow(const ImageSubresourceRange& range, bool to_shadow,
                              const CommandRecorder& command) {
	const bool volume = backing.image_type == vk::ImageType::e3D;
	for (uint32_t level = range.base_level;
	     level < range.base_level + range.level_count && level < backing.mip_levels; level++) {
		auto&                       shadow      = ShadowLevelAt(level);
		const ImageSubresourceRange image_range = volume
		    ? ImageSubresourceRange {level, 1, 0, 1}
		    : ImageSubresourceRange {level, 1, range.base_layer, range.layer_count};
		Transit(to_shadow ? vk::ImageLayout::eTransferSrcOptimal
		                  : vk::ImageLayout::eTransferDstOptimal,
		        to_shadow ? vk::AccessFlagBits2::eTransferRead : vk::AccessFlagBits2::eTransferWrite,
		        image_range, command);
		ShadowBarrier(shadow, command, vk::PipelineStageFlagBits2::eTransfer,
		              to_shadow ? vk::AccessFlagBits2::eTransferWrite
		                        : vk::AccessFlagBits2::eTransferRead);
		const auto  width  = std::max(backing.extent.width >> level, 1u);
		const auto  height = std::max(backing.extent.height >> level, 1u);
		const auto  depth  = volume ? std::max(backing.extent.depth >> level, 1u) : 1u;
		vk::ImageCopy copy {};
		copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, level, image_range.base_layer,
		                       image_range.layer_count};
		copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, image_range.base_layer,
		                       image_range.layer_count};
		// Extents are in source texels: the whole compressed level, or its block grid.
		if (to_shadow) {
			copy.extent = {width, height, depth};
			command.copyImage(backing.image, vk::ImageLayout::eTransferSrcOptimal,
			                  shadow.image.image, vk::ImageLayout::eGeneral, 1, &copy);
		} else {
			std::swap(copy.srcSubresource, copy.dstSubresource);
			copy.extent = {(width + 3u) / 4u, (height + 3u) / 4u, depth};
			command.copyImage(shadow.image.image, vk::ImageLayout::eGeneral, backing.image,
			                  vk::ImageLayout::eTransferDstOptimal, 1, &copy);
		}
	}
}

vk::ImageView Image::PrepareShadowStorage(const ImageViewInfo& requested_view) {
	EXIT_IF(!NeedsStorageShadow() || requested_view.level_count != 1);
	auto view_info = requested_view;
	if (view_info.base_level < info.resources.levels) {
		// Guest levels past the host chain alias its last (1x1) level.
		view_info.base_level = std::min(view_info.base_level, backing.mip_levels - 1);
	}
	m_scheduler.EndRendering();
	const auto command = m_scheduler.Current().Handle();
	const ImageSubresourceRange range {view_info.base_level, 1, view_info.base_layer,
	                                   view_info.layer_count};
	if (std::getenv("KYTY_DEBUG_BLOCK_STORAGE") != nullptr) {
		std::printf("BlockStorage: shadow write format %d type %d level %u layer %u+%u\n",
		            static_cast<int>(view_info.format), static_cast<int>(view_info.type),
		            view_info.base_level, view_info.base_layer, view_info.layer_count);
	}
	auto& shadow = ShadowLevelAt(view_info.base_level);
	if (std::find(m_shadow->pending.begin(), m_shadow->pending.end(), range) ==
	    m_shadow->pending.end()) {
		// Writes may cover part of the range; start from the image's current contents.
		CopyStorageShadow(range, true, command);
		m_shadow->pending.push_back(range);
	}
	ShadowBarrier(shadow, command,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllGraphics,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);

	auto normalized       = view_info;
	normalized.usage      = vk::ImageUsageFlagBits::eStorage;
	normalized.base_level = 0;
	for (const auto& cached: shadow.views) {
		if (cached.info == normalized) {
			return cached.view;
		}
	}
	vk::ImageViewUsageCreateInfo usage {};
	usage.usage = vk::ImageUsageFlagBits::eStorage;
	vk::ImageViewCreateInfo create {};
	create.pNext                           = &usage;
	create.image                           = shadow.image.image;
	create.viewType                        = normalized.type;
	create.format                          = normalized.format;
	create.components                      = normalized.mapping;
	create.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	create.subresourceRange.baseMipLevel   = 0;
	create.subresourceRange.levelCount     = 1;
	create.subresourceRange.baseArrayLayer = normalized.base_layer;
	create.subresourceRange.layerCount     = normalized.layer_count;
	vk::ImageView view;
	if (m_graphics.device.createImageView(&create, nullptr, &view) != vk::Result::eSuccess) {
		EXIT("failed to create block storage shadow view: format=%d\n",
		     static_cast<int>(normalized.format));
	}
	shadow.views.push_back({normalized, view});
	return view;
}

void Image::FlushShadowStorage() {
	if (m_shadow == nullptr || m_shadow->pending.empty()) {
		return;
	}
	m_scheduler.EndRendering();
	const auto command = m_scheduler.Current().Handle();
	for (const auto& range: m_shadow->pending) {
		CopyStorageShadow(range, false, command);
	}
	m_shadow->pending.clear();
}

Image::~Image() {
	KYTY_PROFILER_FUNCTION();
	TextureStateChanged();
	for (const auto& cached: views) {
		if (cached.view != nullptr) {
			m_graphics.device.destroyImageView(cached.view, nullptr);
		}
	}
	if (m_shadow != nullptr) {
		for (const auto& level: m_shadow->levels) {
			if (level == nullptr) {
				continue;
			}
			for (const auto& cached: level->views) {
				m_graphics.device.destroyImageView(cached.view, nullptr);
			}
			m_graphics.DeleteImage(level->image);
		}
	}
	if (backing.image != nullptr) {
		m_graphics.DeleteImage(backing);
	}
}

} // namespace Libs::Graphics
