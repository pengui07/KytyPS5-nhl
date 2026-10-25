#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "graphics/host_gpu/renderer/execRegions.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/experiments.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/stageStats.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/bindEpoch.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/frameDump.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <fmt/format.h>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

struct UnmappedBufferKey {
	uint64_t shader_hash;
	uint64_t address;
	uint64_t size;
	uint32_t resource;
	ShaderType stage;
	bool operator==(const UnmappedBufferKey&) const = default;
};

struct UnmappedBufferKeyHash {
	size_t operator()(const UnmappedBufferKey& key) const {
		size_t hash = std::hash<uint64_t> {}(key.shader_hash);
		const auto mix = [&](size_t value) {
			hash ^= value + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
		};
		mix(std::hash<uint64_t> {}(key.address));
		mix(std::hash<uint64_t> {}(key.size));
		mix(std::hash<uint32_t> {}(key.resource));
		mix(std::hash<unsigned> {}(static_cast<unsigned>(key.stage)));
		return hash;
	}
};

bool ShouldLogUnmappedBuffer(const UnmappedBufferKey& key) {
	thread_local std::unordered_set<UnmappedBufferKey, UnmappedBufferKeyHash> logged;
	return logged.insert(key).second;
}

bool IsSrtTapeVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_TAPE_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

// Visits each bindless image table range once, with the first root image that samples it.
template <typename Fn>
void ForEachImageTable(const std::vector<ShaderRecompiler::IR::ImageResource>& images, Fn&& fn) {
	for (uint32_t i = 0; i < images.size(); i++) {
		const auto& image = images[i];
		if (image.table_capacity == 0u) {
			continue;
		}
		const auto kind  = *ShaderRecompiler::IR::DescriptorBindingForImage(image);
		bool       first = true;
		for (uint32_t j = 0; j < i && first; j++) {
			first = images[j].table_capacity == 0u ||
			        images[j].table_descriptor_base != image.table_descriptor_base ||
			        ShaderRecompiler::IR::DescriptorBindingForImage(images[j]) != kind;
		}
		if (first) {
			fn(i, kind);
		}
	}
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset,
                    Buffer** obtained = nullptr) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBufferCached(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	if (obtained != nullptr) {
		*obtained = buffer;
	}
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written) {
		FrameBreakdown::Timer invalidate_timer(FrameBreakdown::g_buf.ob_invalidate_ns, FrameBreakdown::BufOn());
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
	    descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	// Depth-compare samples need a depth view even for a null slot; sampling an R32F color
	// view with a comparison is invalid and hangs AMD GPUs (NHL 26 shadow-map arrays).
	if (resource.depth_compare && binding == TextureCache::BindingType::Texture &&
	    resource.numeric_class == Prospero::TextureNumericClass::Float) {
		desc.info.pixel_format = vk::Format::eD32Sfloat;
		desc.view_info.format  = vk::Format::eD32Sfloat;
		desc.view_info.aspect  = vk::ImageAspectFlagBits::eDepth;
	}
	desc.type                 = binding;
	return desc;
}


static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical        = info;
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

// What ResolveTexture derives from a T# and the shader's view of it before any texture-cache
// lookup: a pure function of both (EXITs on unsupported descriptors).
static TextureDescDerivation DeriveTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               const ShaderTextureResource& descriptor) {
	const bool storage = resource.written;

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : raw_sint_format != Prospero::BufferFormat::kInvalid
	                                     ? VulkanFormat(raw_sint_format)
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           !resource.r128 && descriptor.MetaCompress(), view_levels,
		                           view_base)) {
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	} else {
		desc.info.UpdateSize();
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	const auto data_size = desc.info.data.size;
	return {.desc              = std::move(desc),
	        .shader_conversion = shader_conversion,
	        .pixel_format      = pixel_format,
	        .view_format       = view_format,
	        .size              = data_size};
}

size_t RenderExecutor::TextureDescKeyHash::operator()(const TextureDescKey& key) const noexcept {
	// Every draw hashes the key of each texture: independent multiplies of 64-bit lanes instead
	// of a dependent chain per dword.
	std::array<uint64_t, 7> lanes {};
	static_assert(sizeof(key.dwords) + sizeof(key.view) <= sizeof(lanes));
	std::memcpy(lanes.data(), key.dwords.data(), sizeof(key.dwords));
	std::memcpy(reinterpret_cast<uint8_t*>(lanes.data()) + sizeof(key.dwords), key.view.data(),
	            sizeof(key.view));
	constexpr std::array<uint64_t, 7> Multipliers {
	    0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0xd6e8feb86659fd93ull,
	    0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull, 0x94d049bb133111ebull};
	uint64_t hash = 0;
	for (size_t i = 0; i < lanes.size(); i++) {
		const auto mixed = (lanes[i] ^ (lanes[i] >> 31u)) * Multipliers[i];
		hash ^= (mixed << (i * 9u % 64u)) | (mixed >> ((64u - i * 9u % 64u) % 64u));
	}
	hash ^= hash >> 32u;
	hash *= 0x9e3779b97f4a7c15ull;
	return static_cast<size_t>(hash ^ (hash >> 29u));
}

RenderExecutor::TextureDescKey
RenderExecutor::MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
                                   const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureDescKey key {};
	key.dwords  = value.dwords;
	key.view[0] = static_cast<uint32_t>(resource.resource_class) |
	              (static_cast<uint32_t>(resource.numeric_class) << 8u) |
	              (static_cast<uint32_t>(resource.dimension) << 16u) |
	              (static_cast<uint32_t>(resource.mip_mode) << 24u);
	key.view[1] = resource.mip_count;
	key.view[2] = static_cast<uint32_t>(resource.conversion_format);
	key.view[3] = resource.shader_swizzle;
	key.view[4] =
	    static_cast<uint32_t>(resource.read) | (static_cast<uint32_t>(resource.written) << 1u) |
	    (static_cast<uint32_t>(resource.atomic) << 2u) |
	    (static_cast<uint32_t>(resource.depth_compare) << 3u) |
	    (static_cast<uint32_t>(resource.cube) << 4u) | (static_cast<uint32_t>(resource.r128) << 5u) |
	    (static_cast<uint32_t>(resource.atomic64) << 6u);
	return key;
}

bool RenderExecutor::ReuseTextureLookup(TextureDescDerivation& derived, TextureBinding& binding) {
	auto&       texture_cache = m_context.GetTextureCache();
	auto&       found         = derived.found;
	auto*       image         = texture_cache.m_slot_images.try_get(found.id);
	const auto& range         = found.desc.info.data;
	// Null images (no guest range) are never registered and depend on no guest pages.
	if (image == nullptr || (!image->registered && !range.Empty()) || image->binding.needs_rebind ||
	    ImageLookupState::Of(found.id, *image) != found.state) {
		ExecRegions::Note(image == nullptr || (!image->registered && !range.Empty()) ? ExecRegions::ReuseNoImage
		                  : image->binding.needs_rebind                              ? ExecRegions::ReuseRebind
		                                                                             : ExecRegions::ReuseState);
		found.valid = false;
		return false;
	}
	if (const auto set_epoch = texture_cache.ImageSetEpoch(); found.image_set_epoch != set_epoch && !range.Empty()) {
		if (texture_cache.ImageEpochInRegion(range.address, range.size) > found.image_set_epoch) {
			ExecRegions::Note(ExecRegions::ReuseEpoch);
			found.valid = false;
			return false;
		}
		// No image of the range changed until now: later draws compare with now instead of
		// scanning the pages of the range again.
		found.image_set_epoch = set_epoch;
	}
	// As the lookup and the acquisition would: the image stays in use.
	image->tick_accessed_last = m_context.GetCommandScheduler().CurrentTick();
	texture_cache.TouchImage(*image);
	binding.image_id   = found.id;
	binding.image_view = found.view;
	// The binding of a stage slot usually reuses the same lookup draw after draw: the description
	// (a few hundred bytes) is copied once per recording.
	if (binding.desc_serial != found.serial) {
		binding.desc        = found.desc;
		binding.desc_serial = found.serial;
	}
	binding.layout     = vk::ImageLayout::eUndefined;
	binding.mip_views.clear();
	binding.lookup = &derived;
	binding.reused = true;
	return true;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              bool table_candidate, GuestRange* examined) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto       descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage    = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		// Sampled null T#s go through the description memo like the others, so later draws reuse
		// the lookup and its view (each draw built the null description, looked up its image and
		// then its view again: about 1% of the execution thread in play).
		if (!storage && !table_candidate && examined == nullptr &&
		    resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			const auto key    = MakeTextureDescKey(resource, value);
			auto       cached = m_texture_descs.find(key);
			if (cached == m_texture_descs.end()) {
				TextureDescDerivation derived;
				derived.desc = NullTextureDesc(resource, TextureCache::BindingType::Texture);
				if (m_texture_descs.size() >= MaxTextureDescs) {
					m_clear_texture_descs = true;
				}
				cached = m_texture_descs.emplace(key, std::move(derived)).first;
			}
			auto& derived = cached->second;
			if (derived.found.valid) {
				TextureBinding binding;
				if (ReuseTextureLookup(derived, binding)) {
					return binding;
				}
			}
			TextureBinding binding {.image_id = texture_cache.FindImage(derived.desc), .desc = derived.desc};
			binding.lookup = &derived;
			return binding;
		}
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {.image_id = id, .desc = std::move(desc)};
	}

	// The description is computed once per T# and view of it; images are looked up every time.
	const auto key    = MakeTextureDescKey(resource, value);
	auto       cached = m_texture_descs.find(key);
	if (cached == m_texture_descs.end()) {
		// Derived before insertion: a derivation that exits (softly, for table candidates) leaves
		// nothing behind.
		auto derived = DeriveTextureDesc(resource, descriptor);
		if (m_texture_descs.size() >= MaxTextureDescs) {
			m_clear_texture_descs = true;
		}
		cached = m_texture_descs.emplace(key, std::move(derived)).first;
	}
	auto& derived = cached->second;
	// Image heaps keep the descriptors of unmapped textures, which PS5 never samples.
	if (resource.heap &&
	    !m_context.IsMapped(derived.desc.info.data.address, derived.desc.info.data.size)) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	// Sampled textures without metadata or stencil: the image and view the last lookup of this
	// description found, when nothing they depend on changed (see TextureDescDerivation::Found).
	// Lookups with metadata materialize fast clears and run every time.
	const bool reusable = !table_candidate && !storage && examined == nullptr &&
	                      resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic &&
	                      derived.desc.info.metadata.kind == ImageMetadataKind::None &&
	                      !derived.desc.info.HasStencil();
	if (reusable && derived.found.valid) {
		TextureBinding binding;
		if (ReuseTextureLookup(derived, binding)) {
			return binding;
		}
	}
	auto        desc              = derived.desc;
	const bool  shader_conversion = derived.shader_conversion;
	const auto  pixel_format      = derived.pixel_format;
	const auto  view_format       = derived.view_format;

	if (examined != nullptr) {
		*examined = desc.info.data;
	}
	if (table_candidate &&
	    ((!texture_cache.IsRegionGpuModified(desc.info.data.address, desc.info.data.size) &&
	      !Libs::LibKernel::Memory::IsBackingReadable(desc.info.data.address,
	                                                  desc.info.data.size)) ||
	     (desc.info.metadata.kind != ImageMetadataKind::None &&
	      !Libs::LibKernel::Memory::IsBackingReadable(desc.info.metadata.range.address,
	                                                  desc.info.metadata.range.size)))) {
		EXIT("texture 0x%016" PRIx64 " size 0x%" PRIx64 " has no guest backing\n",
		     desc.info.data.address, desc.info.data.size);
	}
	ImageId id {};
	if (table_candidate) {
		id = texture_cache.FindImageSpeculative(desc, shader_conversion);
		if (!id) {
			// Only speculative lookups decline: the table binds this element as a null image.
			return {.image_id = id, .desc = std::move(desc)};
		}
	} else {
		id = texture_cache.FindImage(desc, shader_conversion);
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		// Block-compressed images whose format has no storage support on this device (e.g.
		// BC6H on AMD) are created without STORAGE usage; a storage view on them is invalid and
		// hangs the GPU. Drop the guest's writes into a null storage image instead.
		if (image->backing.image != nullptr &&
		    !(image->backing.usage & vk::ImageUsageFlagBits::eStorage)) {
			static std::atomic_flag warned = ATOMIC_FLAG_INIT;
			if (!warned.test_and_set(std::memory_order_relaxed)) {
				std::printf("Warning: dropping storage writes to %s image without storage usage\n",
				            vk::to_string(image->info.pixel_format).c_str());
				std::fflush(stdout);
			}
			auto       null_desc = NullTextureDesc(resource, TextureCache::BindingType::Storage);
			const auto null_id   = texture_cache.FindImage(null_desc);
			return {.image_id = null_id, .desc = std::move(null_desc)};
		}
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	TextureBinding binding {.image_id = id, .desc = std::move(desc)};
	// Recorded once the view is acquired (AcquireImageViews()). Stencil associations and images
	// with stencil refresh another image as well: they are looked up every time.
	if (reusable && !stencil_association && !image->info.HasStencil()) {
		binding.lookup = &derived;
	}
	return binding;
}

// A bindless table holds every texture the game has loaded; bind a T# the shader's view cannot
// read (other class, dimension or layout) as a null image, which the shader never selects for it.
TextureBinding RenderExecutor::ResolveTableTexture(
    const ShaderRecompiler::IR::ImageResource&   root,
    const ShaderRecompiler::IR::DescriptorValue& value, GuestRange* examined) {
	if (ShaderRecompiler::IR::ImageTableSlotCompatible(root, value)) {
		try {
			Common::SoftExitScope soft_exit(true);
			auto texture = ResolveTexture(root, value, true, examined);
			if (texture.image_id) {
				return texture;
			}
		} catch (const Common::SoftExitError& error) {
			static std::atomic_uint reported = 0;
			if (reported.fetch_add(1) < 8u) {
				LOGF("bindless table texture bound as null: %s\n", error.what());
			}
		}
	}
	return ResolveTexture(root, ShaderRecompiler::IR::DescriptorValue {.dword_count = 8u});
}

size_t RenderExecutor::TableDescriptorHash::operator()(
    const std::array<uint32_t, 8>& dwords) const noexcept {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto dword: dwords) {
		hash = (hash ^ dword) * 0x100000001b3ull;
		hash ^= hash >> 29u;
	}
	return static_cast<size_t>(hash);
}

RenderExecutor::TableResolution&
RenderExecutor::FindTableResolution(const ShaderRecompiler::IR::ImageResource& root) {
	constexpr size_t MaxTables   = 8;
	constexpr size_t MaxElements = size_t {1} << 16u;
	const auto       tick        = ++m_table_resolution_tick;
	// A shader samples several roots from the same tables. Where a root reads its table and where
	// the shader uses it do not affect how a T# resolves, so such roots share their elements.
	auto view                 = root;
	view.source               = 0;
	view.first_use_pc         = 0;
	view.table                = ShaderRecompiler::IR::ImageResource::NoImageTable;
	view.table_capacity       = 0;
	view.table_mapping_offset = 0;
	view.table_entry_mask     = 0;
	auto found = std::ranges::find_if(m_table_resolutions,
	                                  [&](const TableResolution& entry) { return entry.root == view; });
	if (found == m_table_resolutions.end()) {
		if (m_table_resolutions.size() < MaxTables) {
			found = m_table_resolutions.emplace(m_table_resolutions.end());
		} else {
			found = std::ranges::min_element(m_table_resolutions, {}, &TableResolution::last_use);
		}
		found->root = std::move(view);
		found->elements.clear();
	} else if (found->elements.size() > MaxElements) {
		found->elements.clear();
	}
	found->last_use = tick;
	return *found;
}

const TextureBinding&
RenderExecutor::ResolveTableElement(TableResolution&                             table,
                                    const ShaderRecompiler::IR::DescriptorValue& value) {
	auto&      texture_cache = m_context.GetTextureCache();
	const auto backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
	const auto [entry, inserted] = table.elements.try_emplace(value.dwords);
	auto& element                = entry->second;
	const auto set_epoch = texture_cache.ImageSetEpoch();
	if (!inserted && !element.volatile_dcc) {
		if (element.permanent ||
		    (element.checked_epoch == set_epoch && element.backing_epoch == backing_epoch)) {
			return element.texture;
		}
		const auto image_epoch =
		    element.range.Valid()
		        ? texture_cache.ImageEpochInRegion(element.range.address, element.range.size)
		        : 0;
		if (image_epoch <= element.image_epoch && element.backing_epoch == backing_epoch) {
			element.checked_epoch = set_epoch;
			return element.texture;
		}
	}
	// Record the epoch before resolving: an image the lookup itself inserts makes the next use
	// resolve again, after which the element is stable.
	element               = {};
	element.image_epoch   = set_epoch;
	element.checked_epoch = set_epoch;
	element.backing_epoch = backing_epoch;
	GuestRange examined {};
	element.texture  = ResolveTableTexture(table.root, value, &examined);
	const auto& data = element.texture.desc.info.data;
	if (!data.Empty()) {
		element.range        = data;
		element.volatile_dcc = element.texture.desc.info.metadata.kind == ImageMetadataKind::Dcc;
	} else if (!ShaderRecompiler::IR::ImageTableSlotCompatible(table.root, value)) {
		element.permanent = true;
	} else {
		// Declined by the texture cache lookup, or rejected before it from the T# and the
		// backing alone (no range examined: only a backing change can alter the outcome).
		element.range = examined;
	}
	return element.texture;
}

vk::Sampler RenderExecutor::NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                          uint32_t                                        index,
                                          const ShaderRecompiler::IR::DescriptorValue&    value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler    = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	} else {
		// DIAGNOSTIC ONLY: KYTY_DBG_FORCE_SHADOW_COMPARE=always|0..7 (S# compare func; 7=always
		// lit, 1=less, 3=lequal, 4=greater, 6=gequal) overrides every depth-compare sampler.
		static const int force_func = [] {
			const char* e = std::getenv("KYTY_DBG_FORCE_SHADOW_COMPARE");
			if (e == nullptr || *e == 0) return -1;
			if (std::string_view(e) == "always") return 7;
			return std::atoi(e) & 7;
		}();
		static const bool log_shadow = std::getenv("KYTY_DBG_LOG_SHADOW") != nullptr;
		if (log_shadow) {
			static std::mutex            mtx;
			static std::vector<uint32_t> seen;
			const uint32_t               key = descriptor.fields[0] ^ (descriptor.fields[1] * 31u);
			std::lock_guard<std::mutex>  lock(mtx);
			if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
				seen.push_back(key);
				printf("NHL27SHADOW: sampler s0=%08x s1=%08x func=%u clamp=%u,%u compareEnable=%d\n",
				       descriptor.fields[0], descriptor.fields[1], descriptor.DepthCompareFunc(),
				       descriptor.ClampX(), descriptor.ClampY(),
				       descriptor.DepthCompareFunc() != 0 ? 1 : 0);
			}
		}
		if (force_func >= 0) {
			descriptor.fields[0] = (descriptor.fields[0] & ~(0x7u << 12u)) |
			                       (static_cast<uint32_t>(force_func) << 12u);
		}
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	// Samplers live as long as the sampler cache: remember recent ones without its lock.
	uint64_t hash = sampler.integer_border ? 0x9e3779b97f4a7c15ull : 0;
	for (const auto field: descriptor.fields) {
		hash = (hash ^ field) * 0x100000001b3ull;
	}
	auto& entry = m_sampler_memo[(hash ^ (hash >> 32u)) % m_sampler_memo.size()];
	if (entry.sampler != nullptr && entry.integer_border == sampler.integer_border &&
	    std::memcmp(entry.fields.data(), descriptor.fields, sizeof(entry.fields)) == 0) {
		return entry.sampler;
	}
	const auto native = m_context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
	std::memcpy(entry.fields.data(), descriptor.fields, sizeof(entry.fields));
	entry.integer_border = sampler.integer_border;
	entry.sampler        = native;
	return native;
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

// A stage's flattened SRT and shader data in one stream reservation: one map and one commit
// instead of two (the uploads were 1.8% of the execution thread).
static std::pair<vk::DescriptorBufferInfo, vk::DescriptorBufferInfo>
NativeUploadPair(RenderContext& context, std::span<const uint32_t> first, std::span<const uint32_t> second) {
	EXIT_IF(first.empty() || second.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&            buffer       = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	constexpr uint64_t Alignment  = 256;
	const uint64_t   second_start = (first.size_bytes() + Alignment - 1) & ~(Alignment - 1);
	const auto [data, offset]     = buffer.Map(second_start + second.size_bytes(), Alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, first.data(), first.size_bytes());
	std::memcpy(data + second_start, second.data(), second.size_bytes());
	buffer.Commit();
	return {vk::DescriptorBufferInfo {buffer.Handle(), offset, first.size_bytes()},
	        vk::DescriptorBufferInfo {buffer.Handle(), offset + second_start, second.size_bytes()}};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_storage |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
	m_bound_storage = false;
	if (m_clear_texture_descs) {
		m_clear_texture_descs = false;
		m_texture_descs.clear();
		m_texture_descs_generation++;
	}
}

namespace {

// KYTY_DRAWPREP_DELTA: hash of every ImageResource field that ResolveTexture reads.
uint64_t ImageResourceKey(const ShaderRecompiler::IR::ImageResource& r) {
	uint64_t h = 0xcbf29ce484222325ull;
	const auto mix = [&h](uint64_t v) { h = (h ^ v) * 0x100000001b3ull; };
	mix(r.source);
	mix(r.first_use_pc);
	mix(static_cast<uint64_t>(r.resource_class));
	mix(static_cast<uint64_t>(r.numeric_class));
	mix(static_cast<uint64_t>(r.dimension));
	mix(static_cast<uint64_t>(r.mip_mode));
	mix(r.mip_count);
	mix(static_cast<uint64_t>(r.conversion_format));
	mix(r.shader_swizzle);
	mix(static_cast<uint64_t>(r.read) | (static_cast<uint64_t>(r.written) << 1u) |
	    (static_cast<uint64_t>(r.atomic) << 2u) | (static_cast<uint64_t>(r.atomic64) << 3u) |
	    (static_cast<uint64_t>(r.depth_compare) << 4u) | (static_cast<uint64_t>(r.cube) << 5u) |
	    (static_cast<uint64_t>(r.r128) << 6u));
	mix(r.indirect_root);
	mix(r.indirect_mapping_offset);
	mix(r.indirect_search_iterations);
	mix(r.indirect_resources.size());
	return h;
}

bool DrawPrepDeltaVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAWPREP_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// A draw binding the T#s of its program's last draw keeps that draw's texture bindings while no
// image changed (see PrepareBindings()); KYTY_LEAN_DRAWS=0 turns it off. KYTY_LEAN_VERIFY=1
// checks every kept binding against a full lookup, and runs the skipped rebind and transitions to
// check that they change nothing.
bool LeanDrawsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LEAN_DRAWS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool LeanVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LEAN_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// DrawRepeats: per program, the resources of its last use.
struct RepeatLast {
	std::vector<ShaderRecompiler::IR::DescriptorValue> buffers, images, samplers;
	std::vector<uint32_t>                              user_data, flattened_srt;
};
struct RepeatCounters {
	uint64_t stages = 0, first = 0, images = 0, samplers = 0, buffers = 0, user = 0, srt = 0, all = 0;
	uint64_t buffer_slots = 0, buffer_slots_changed = 0, image_slots = 0, image_slots_changed = 0;
	uint64_t draws = 0, draws_all = 0, draws_but_buffers = 0, draws_but_buffers_user = 0;
};
std::unordered_map<const void*, RepeatLast> g_repeat_last;
RepeatCounters                              g_repeat;
uint32_t                                    g_repeat_draw_flags = 0;
bool                                        g_repeat_in_draw    = false;

// Bits: 1 images, 2 samplers, 4 buffers, 8 user data, 16 flattened SRT the same as last time.
void RepeatStage(const void* program, const ShaderRecompiler::IR::ResourceSnapshot& snapshot) {
	auto& c     = g_repeat;
	auto  found = g_repeat_last.find(program);
	c.stages++;
	uint32_t flags = 0;
	if (found == g_repeat_last.end()) {
		c.first++;
		found = g_repeat_last.emplace(program, RepeatLast {}).first;
	} else {
		const auto& last = found->second;
		flags |= last.images == snapshot.images ? 1u : 0u;
		flags |= last.samplers == snapshot.samplers ? 2u : 0u;
		flags |= last.buffers == snapshot.buffers ? 4u : 0u;
		flags |= last.user_data == snapshot.user_data ? 8u : 0u;
		flags |= last.flattened_srt == snapshot.flattened_srt ? 16u : 0u;
		c.images += flags & 1u;
		c.samplers += (flags >> 1u) & 1u;
		c.buffers += (flags >> 2u) & 1u;
		c.user += (flags >> 3u) & 1u;
		c.srt += (flags >> 4u) & 1u;
		c.all += flags == 31u ? 1u : 0u;
		if (last.buffers.size() == snapshot.buffers.size()) {
			for (size_t i = 0; i < snapshot.buffers.size(); i++) {
				c.buffer_slots++;
				c.buffer_slots_changed += last.buffers[i] == snapshot.buffers[i] ? 0u : 1u;
			}
		}
		if (last.images.size() == snapshot.images.size()) {
			for (size_t i = 0; i < snapshot.images.size(); i++) {
				c.image_slots++;
				c.image_slots_changed += last.images[i] == snapshot.images[i] ? 0u : 1u;
			}
		}
	}
	if (g_repeat_in_draw) g_repeat_draw_flags &= flags;
	auto& last         = found->second;
	last.buffers       = snapshot.buffers;
	last.images        = snapshot.images;
	last.samplers      = snapshot.samplers;
	last.user_data     = snapshot.user_data;
	last.flattened_srt = snapshot.flattened_srt;
}

void RepeatPrint() {
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	last          = now;
	const auto& c = g_repeat;
	const auto  pct = [](uint64_t part, uint64_t whole) { return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0; };
	const auto  seen = c.stages - c.first;
	std::printf("NHL27REPEATS stages=%llu first=%llu same: images=%.1f%% samplers=%.1f%% buffers=%.1f%% user=%.1f%% srt=%.1f%% "
	            "all=%.1f%% | buffer slots changed %.1f%% of %llu, image slots changed %.1f%% of %llu | draws=%llu all-same=%.1f%% "
	            "all-but-buffers=%.1f%% all-but-buffers-and-user=%.1f%%\n",
	            static_cast<unsigned long long>(c.stages), static_cast<unsigned long long>(c.first), pct(c.images, seen),
	            pct(c.samplers, seen), pct(c.buffers, seen), pct(c.user, seen), pct(c.srt, seen), pct(c.all, seen),
	            pct(c.buffer_slots_changed, c.buffer_slots), static_cast<unsigned long long>(c.buffer_slots),
	            pct(c.image_slots_changed, c.image_slots), static_cast<unsigned long long>(c.image_slots),
	            static_cast<unsigned long long>(c.draws), pct(c.draws_all, c.draws), pct(c.draws_but_buffers, c.draws),
	            pct(c.draws_but_buffers_user, c.draws));
	std::fflush(stdout);
	g_repeat = {};
}

} // namespace

bool DrawRepeats::Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_DRAW_REPEATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void DrawRepeats::DrawBegin() {
	g_repeat_in_draw    = true;
	g_repeat_draw_flags = 31u;
}

void DrawRepeats::DrawEnd() {
	g_repeat_in_draw = false;
	auto& c          = g_repeat;
	c.draws++;
	c.draws_all += g_repeat_draw_flags == 31u ? 1u : 0u;
	c.draws_but_buffers += (g_repeat_draw_flags | 4u) == 31u ? 1u : 0u;
	c.draws_but_buffers_user += (g_repeat_draw_flags | 4u | 8u | 16u) == 31u ? 1u : 0u;
	RepeatPrint();
}

// Descriptor value arrays compared as bytes (DescriptorValue has no padding): the element-wise
// compares of a stage's T#s and S#s against its last draw's were 1.4% of the execution thread.
static bool SameDescriptorValues(const std::vector<ShaderRecompiler::IR::DescriptorValue>& a,
                                 const std::vector<ShaderRecompiler::IR::DescriptorValue>& b) {
	return a.size() == b.size() &&
	       (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0);
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (DrawRepeats::Enabled()) {
		RepeatStage(&program, snapshot);
	}
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.push_dwords_valid = false;
	prepared.shader_data.clear();
	const bool same_program = prepared.last_program == &program &&
	                          prepared.last_descs_generation == m_texture_descs_generation;
	// The T#s and descriptions of the program's last draw: a stage alternates between programs,
	// and a memo of only the stage's last program missed (the description lookups were 3% of
	// the execution thread).
	auto* memo = prepared.last_program == &program ? prepared.image_memo : nullptr;
	if (memo == nullptr) {
		memo = ProgramMemosOf(&program).image;
	}
	// The program's own image bindings come back with it: a slot reusing the lookup its last draw
	// made already holds the description (ReuseTextureLookup copied one for most slots of
	// alternating programs, 1.2% of the execution thread).
	if (prepared.image_memo != memo && (memo->holder == nullptr || memo->holder == &prepared)) {
		if (prepared.image_memo != nullptr && prepared.image_memo->holder == &prepared) {
			prepared.images.swap(prepared.image_memo->bindings);
			prepared.image_memo->holder = nullptr;
		}
		prepared.images.swap(memo->bindings);
		memo->holder = &prepared;
	}
	prepared.image_memo    = memo;
	prepared.images.resize(program.info.images.size());
	ExecRegions::InDraw(ExecRegions::DrawStages);
	// The program's buffer slots come back with it as well: FindBuffers and RebindBuffers compare a
	// slot with what it bound the last time, and with programs alternating on a stage that was
	// mostly another program's draw (half the buffer bindings looked up again; 40% of the slots had
	// a new V# for their program).
	auto* buffer_memo = prepared.last_program == &program ? prepared.buffer_memo : nullptr;
	if (buffer_memo == nullptr) {
		buffer_memo = ProgramMemosOf(&program).buffer;
	}
	if (prepared.buffer_memo != buffer_memo &&
	    (buffer_memo->holder == nullptr || buffer_memo->holder == &prepared)) {
		if (prepared.buffer_memo != nullptr && prepared.buffer_memo->holder == &prepared) {
			prepared.buffer_memo->Swap(prepared);
			prepared.buffer_memo->holder = nullptr;
		}
		buffer_memo->Swap(prepared);
		buffer_memo->holder = &prepared;
	}
	prepared.buffer_memo = buffer_memo;
	auto&      last_images = memo->images;
	auto&      image_descs = memo->image_descs;
	const bool same_descs  = memo->descs_generation == m_texture_descs_generation;
	const bool same_images = same_descs && SameDescriptorValues(last_images, snapshot.images);
	// Draws of a program change some T#s and keep the others: a slot whose T# is unchanged keeps
	// its description instead of looking it up again.
	const bool slot_reuse = same_descs && last_images.size() == snapshot.images.size() &&
	                        image_descs.size() == program.info.images.size();
	if (!same_images && !slot_reuse) {
		image_descs.assign(program.info.images.size(), nullptr);
	}
	ExecRegions::InDraw(ExecRegions::StagesMemo);
	// A program binding the T#s of its last draw keeps that draw's bindings while none of their images
	// and not the image set changed since its lookups last held (ProgramImageMemo::lean_epoch,
	// KeptImagesHold()): each slot would reuse its description's lookup (ReuseTextureLookup) and find
	// the same image and view. Only the per-draw bookkeeping of a reuse (access tick, binding state) is
	// redone, and RebindImages() and the layout transitions are skipped while they hold (see
	// CommitBindings()).
	prepared.lookup_set_epoch = g_image_set_epoch.load(std::memory_order_relaxed);
	prepared.lookup_epoch     = g_texture_epoch.load(std::memory_order_relaxed);
	prepared.lean             = LeanDrawsEnabled() && same_images && memo->holder == &prepared &&
	                memo->lean_epoch != UINT64_MAX && image_descs.size() == program.info.images.size() &&
	                KeptImagesHold(prepared, memo->lean_epoch, memo->lean_set_epoch);
	if (!prepared.lean) {
		if (ExecRegions::Enabled()) {
			ExecRegions::Note(!same_images                    ? ExecRegions::LeanOtherImages
			                  : memo->holder != &prepared     ? ExecRegions::LeanOtherHolder
			                  : memo->lean_epoch == UINT64_MAX ? ExecRegions::LeanNotHeld
			                                                   : ExecRegions::LeanEpochMoved);
		}
		// Set again by FinishLeanBindings() once the bindings this draw makes have held to its end.
		memo->lean_epoch = UINT64_MAX;
	} else {
		ExecRegions::Note(ExecRegions::LeanStages);
		auto&      texture_cache = m_context.GetTextureCache();
		const auto tick          = m_context.GetCommandScheduler().CurrentTick();
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			if (program.info.images[i].heap) {
				continue;
			}
			ExecRegions::Note(ExecRegions::LeanSlots);
			auto& binding = prepared.images[i];
			if (LeanVerifyEnabled()) {
				TextureBinding check;
				auto*          derived = image_descs[i];
				if (derived == nullptr || !derived->found.valid || !ReuseTextureLookup(*derived, check) ||
				    check.image_id != binding.image_id || check.image_view != binding.image_view) {
					EXIT("lean draw: image slot %u of shader 0x%016" PRIx64 " no longer reuses its lookup\n",
					     i, program.shader_hash);
				}
				// RebindImages() then treats the binding as the reuse would have left it.
				binding.lookup = derived;
				binding.reused = true;
				prepared.lean_layouts.resize(program.info.images.size());
				prepared.lean_layouts[i] = binding.layout;
			}
			auto& image              = texture_cache.GetImage(binding.image_id);
			image.tick_accessed_last = tick;
			BindImage(binding.image_id, false);
		}
	}
	for (uint32_t i = 0; i < program.info.images.size() && !prepared.lean; i++) {
		// Sampled textures whose last lookup still holds are written in place.
		const auto& resource = program.info.images[i];
		// Bindless heap images resolve through the image heap, not the texture cache.
		if (resource.heap) {
			prepared.images[i] = {};
			continue;
		}
		ExecRegions::Note(ExecRegions::ImageSlots);
		if (!resource.written &&
		    resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			auto* derived = image_descs[i];
			if (!same_images && !(slot_reuse && last_images[i] == snapshot.images[i])) {
				ExecRegions::Note(ExecRegions::ImageMapLookups);
				ExecRegions::InDraw(ExecRegions::StagesImages);
				const auto cached =
				    m_texture_descs.find(MakeTextureDescKey(resource, snapshot.images[i]));
				derived = cached != m_texture_descs.end() ? &cached->second : nullptr;
				ExecRegions::InDraw(ExecRegions::StagesMap);
			} else {
				ExecRegions::Note(ExecRegions::ImageMemoDesc);
			}
			if (derived != nullptr && derived->found.valid &&
			    ExecRegions::TimedTwice(ExecRegions::WarmReuse,
			                            [&] { return ReuseTextureLookup(*derived, prepared.images[i]); })) {
				ExecRegions::Note(ExecRegions::ImageReused);
				image_descs[i] = derived;
				BindImage(prepared.images[i].image_id, false);
				continue;
			}
			ExecRegions::Note(derived == nullptr ? ExecRegions::ResolveNoDesc : ExecRegions::ResolveInvalid);
		} else {
			ExecRegions::Note(resource.written ? ExecRegions::ResolveWritten : ExecRegions::ResolveDynamic);
		}
		ExecRegions::Note(ExecRegions::ImageResolved);
		ExecRegions::InDraw(ExecRegions::StagesImages);
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		ExecRegions::InDraw(ExecRegions::StagesResolve);
		// The description whose lookups later draws reuse (see ResolveTexture()).
		image_descs[i] = binding.lookup;
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	if (!same_images) {
		last_images = snapshot.images;
	}
	memo->descs_generation = m_texture_descs_generation;
	ExecRegions::InDraw(ExecRegions::StagesImages);
	prepared.table_images.clear();
	prepared.table_sources.clear();
	prepared.table_roots.clear();
	ForEachImageTable(program.info.images, [&](uint32_t root_index, auto /*kind*/) {
		{
			const auto& root  = program.info.images.at(root_index);
			const auto& table = snapshot.image_tables.at(root.table);
			const ShaderRecompiler::IR::DescriptorValue null_value {.dword_count = 8u};
			auto&       resolution   = FindTableResolution(root);
			const auto& null_texture = ResolveTableElement(resolution, null_value);
			BindImage(null_texture.image_id, false);
			for (uint32_t slot = 0; slot < root.table_capacity; slot++) {
				const bool real = slot != 0u && slot < table.slots.size();
				if (real) {
					const auto& texture = ResolveTableElement(resolution, table.slots[slot]);
					BindImage(texture.image_id, false);
					prepared.table_images.push_back(texture);
				} else {
					prepared.table_images.push_back(null_texture);
				}
				prepared.table_sources.push_back(real ? table.slots[slot] : null_value);
				prepared.table_roots.push_back(root_index);
			}
		}
	});
	if (!prepared.lean) {
		prepared.lean_eligible = prepared.table_images.empty();
		for (uint32_t i = 0; i < program.info.images.size() && prepared.lean_eligible; i++) {
			prepared.lean_eligible = program.info.images[i].heap || image_descs[i] != nullptr;
		}
	}
	ExecRegions::InDraw(ExecRegions::StagesTables);
	// Samplers live as long as the sampler cache: the same S#s of the same program give the same
	// samplers.
	if (!same_program || prepared.samplers.size() != program.info.samplers.size() ||
	    !SameDescriptorValues(prepared.last_samplers, snapshot.samplers)) {
		prepared.samplers.clear();
		prepared.samplers.reserve(program.info.samplers.size());
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			prepared.samplers.push_back(NativeSampler(
			    program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
		}
		prepared.last_samplers = snapshot.samplers;
	}
	ExecRegions::InDraw(ExecRegions::StagesSamplers);
	prepared.last_program          = &program;
	prepared.last_descs_generation = m_texture_descs_generation;
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_find_buffers_ns,
	                                 FrameBreakdown::DetailEnabled());
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto& prepared = *stage;
		EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		prepared.buffer_keys.resize(prepared.buffer_sources.size());
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		// Half the slots bind the range they had the draw before: unchanged V#s keep their source
		// (the clamp and the buffer lookup) while no buffer was registered or unregistered and no
		// memory was mapped or unmapped.
		const uint64_t generation = cache.BufferSetEpoch() * 0x9e3779b97f4a7c15ull ^ m_context.MappingGeneration();
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			auto& key    = prepared.buffer_keys[metadata.descriptor_index];
			ExecRegions::Note(ExecRegions::FindSlots);
			if (ExecRegions::Enabled() && key.program == &program && key.generation != generation &&
			    key.value == snapshot.buffers[resource]) {
				ExecRegions::Note(ExecRegions::FindEpochs);
			}
			if (ExecRegions::TimedTwice(ExecRegions::WarmFindSlot, [&] {
				    return key.program == &program && key.generation == generation &&
				           key.value == snapshot.buffers[resource];
			    })) {
				ExecRegions::Note(ExecRegions::FindSame);
				if (StageStats::Enabled()) {
					StageStats::g_bind_total.fetch_add(1, std::memory_order_relaxed);
					StageStats::g_bind_same.fetch_add(1, std::memory_order_relaxed);
				}
				continue;
			}
			key = {snapshot.buffers[resource], &program, generation};
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				// OOB_SELECT=2 only tests NumRecords != 0; a zero-stride fetch
				// still needs its full payload when NumRecords is smaller than it.
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address == 0 || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					key.program = nullptr;
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				//  PPSA15246
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
			}
			// Finite GPU descriptors can span sparse PRT holes; clamp only oversized/sentinel ranges.
			if (size >= (uint64_t {1} << 30u) || descriptor.NumRecords() == UINT32_MAX) {
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			}
			if (IsSrtTapeVerifyEnabled() && !m_context.IsMapped(address, size) &&
			    ShouldLogUnmappedBuffer({program.shader_hash, address, size, resource, program.stage})) {
				const auto& raw = snapshot.buffers[resource];
				std::fprintf(stderr, "GPU buffer descriptor range not fully mapped stage=%u shader=0x%016" PRIx64
				     " resource=%u addr=0x%016" PRIx64 " size=0x%016" PRIx64
				     " records=%u dwords=%08x,%08x,%08x,%08x\n",
				     static_cast<unsigned>(program.stage), program.shader_hash, resource, address, size,
				     descriptor.NumRecords(), raw.dwords[0], raw.dwords[1], raw.dwords[2], raw.dwords[3]);
				std::fflush(stderr);
			}
			// A descriptor wholly outside guest memory (NHL 27 menus, base 0x11ffffffff once in
			// ~50 boots) is bound as a null buffer instead of ending the emulator.
			size = Libs::LibKernel::Memory::ClampRangeSizeOrZero(address, size);
			if (size == 0) {
				source = {};
				continue;
			}
			// KYTY_DBG_STAGE_STATS: bindings, those the slot had the draw before, written ones (NHL27BINDS).
			if (StageStats::Enabled()) {
				StageStats::g_bind_total.fetch_add(1, std::memory_order_relaxed);
				if (source.address == address && source.size == size) {
					StageStats::g_bind_same.fetch_add(1, std::memory_order_relaxed);
				}
				if (metadata.written) {
					StageStats::g_bind_written.fetch_add(1, std::memory_order_relaxed);
				}
			}

			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

// FindBuffer only creates what is missing: while the base registers, the buffer set and the mappings
// are unchanged, looking the bases' windows up again (a mapped-range walk under the mapping lock and
// a page-table search per piece) finds the same buffers. That was 2.9% of the execution thread in
// play (46,000 stages a second, 540 ns each).
void RenderExecutor::CacheDmaBases(PreparedBindings& prepared) {
	const auto& runtime   = *prepared.runtime;
	const auto& program   = *runtime.program;
	const auto& user_data = runtime.resources->user_data;
	auto&       cache     = m_context.GetBufferCache();
	const auto  dword     = [&](uint32_t reg, uint32_t half) -> uint32_t {
		const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
		// As RenderContext::CacheDmaBases(): registers outside the user data are skipped.
		return reg < program.user_data_base || index + 1u >= user_data.size() ? 0u : user_data[index + half];
	};
	const auto generation = [&] {
		return cache.BufferSetEpoch() * 0x9e3779b97f4a7c15ull ^ m_context.MappingGeneration();
	};
	const auto& registers = program.info.dma_base_registers;
	bool        same      = prepared.dma_program == &program && prepared.dma_generation == generation() &&
	              prepared.dma_bases.size() == registers.size() * 2u;
	for (size_t i = 0; same && i < registers.size(); i++) {
		same = prepared.dma_bases[i * 2u] == dword(registers[i], 0) &&
		       prepared.dma_bases[i * 2u + 1u] == dword(registers[i], 1);
	}
	if (same) {
		return;
	}
	m_context.CacheDmaBases(runtime);
	// After the lookups, which may have created buffers.
	prepared.dma_program    = &program;
	prepared.dma_generation = generation();
	prepared.dma_bases.clear();
	for (const auto reg: registers) {
		prepared.dma_bases.push_back(dword(reg, 0));
		prepared.dma_bases.push_back(dword(reg, 1));
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_rebind_buffers_ns,
	                                 FrameBreakdown::DetailEnabled());
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);

	m_context.GetBufferCache().SetWriterTag(
	    program.stage == ShaderType::Compute ? "cs-storage" : "gfx-storage", program.shader_hash);
	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	FrameBreakdown::Timer lookup_timer(FrameBreakdown::g_counters.dp_rebind_lookup_ns);
	FrameBreakdown::BufScope buf_scope(1);
	auto& cache = m_context.GetBufferCache();
	prepared.buffer_memos.resize(prepared.buffers.size());
	// Plain reads (most slots) whose range found the same buffer the draw before skip the lookup
	// while no buffer was registered and the range stayed CPU-clean.
	const bool slot_memo = Common::RuntimeFlags::Current().buffer_fast;
	ExecRegions::InOp(ExecRegions::DrawBuffers, ExecRegions::DispBuffers);
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		ExecRegions::NoteOp(ExecRegions::BufferSlots);
		uint32_t    buffer_offset = 0;
		const auto& source        = prepared.buffer_sources[index];
		auto&       memo          = prepared.buffer_memos[index];
		const bool  plain = slot_memo && !resource.written && !resource.formatted &&
		                   !(resource.read && Config::SyncRawImageBuffersEnabled()) && source.address != 0 &&
		                   source.size != 0;
		// Written slots too, while the GPU ranges and the buffer set are unchanged.
		const bool  written = slot_memo && resource.written && source.address != 0 && source.size != 0;
		// And texel reads while no image starts at the address.
		const bool  texel = slot_memo && !resource.written && resource.formatted &&
		                   !(resource.read && Config::SyncRawImageBuffersEnabled()) && source.address != 0 &&
		                   source.size != 0;
		if (plain && memo.buffer != nullptr && !memo.written && memo.image_version == 0 &&
		    memo.address == source.address && memo.size == source.size &&
		    cache.PlainReadHolds(*memo.buffer, source.address, source.size, memo.cpu_generation,
		                         memo.registration)) {
			prepared.buffers[index] = memo.info;
			pack_memory_offset(index, memo.buffer_offset);
			FrameBreakdown::VAdd(FrameBreakdown::g_counters.slot_hit[0]);
			ExecRegions::NoteOp(ExecRegions::BufferMemoHits);
			ExecRegions::InOp(ExecRegions::BuffersMemo, ExecRegions::DBufPlain);
			continue;
		}
		if (written && memo.buffer != nullptr && memo.written && memo.address == source.address &&
		    memo.size == source.size &&
		    cache.WrittenBindHolds(*memo.buffer, source.address, source.size, memo.ranges_version,
		                           memo.registration)) {
			m_context.GetTextureCache().InvalidateMemoryFromGPU(source.address, source.size);
			prepared.buffers[index] = memo.info;
			pack_memory_offset(index, memo.buffer_offset);
			FrameBreakdown::VAdd(FrameBreakdown::g_counters.slot_hit[1]);
			ExecRegions::NoteOp(ExecRegions::BufferMemoHits);
			ExecRegions::InOp(ExecRegions::BuffersMemo, ExecRegions::DBufWritten);
			continue;
		}
		if (texel && memo.buffer != nullptr && memo.image_version != 0 && memo.address == source.address &&
		    memo.size == source.size &&
		    cache.TexelReadHolds(*memo.buffer, source.address, source.size, memo.cpu_generation,
		                         memo.registration, memo.image_version)) {
			prepared.buffers[index] = memo.info;
			pack_memory_offset(index, memo.buffer_offset);
			FrameBreakdown::VAdd(FrameBreakdown::g_counters.slot_hit[2]);
			ExecRegions::NoteOp(ExecRegions::BufferMemoHits);
			ExecRegions::InOp(ExecRegions::BuffersMemo, ExecRegions::DBufTexel);
			continue;
		}
		// The same range bound by another slot or draw before (see m_bind_memo).
		const uint8_t kind      = plain ? 1 : written ? 2 : texel ? 3 : 0;
		auto&         bind_memo = BindMemoOf(source.address, source.size, kind);
		if (kind != 0 && bind_memo.kind == kind && bind_memo.slot.address == source.address &&
		    bind_memo.slot.size == source.size && bind_memo.slot.buffer != nullptr) {
			auto&      global = bind_memo.slot;
			const bool holds =
			    kind == 1   ? cache.PlainReadHolds(*global.buffer, source.address, source.size, global.cpu_generation,
			                                       global.registration)
			    : kind == 2 ? cache.WrittenBindHolds(*global.buffer, source.address, source.size,
			                                         global.ranges_version, global.registration)
			                : cache.TexelReadHolds(*global.buffer, source.address, source.size, global.cpu_generation,
			                                       global.registration, global.image_version);
			if (holds) {
				if (kind == 2) m_context.GetTextureCache().InvalidateMemoryFromGPU(source.address, source.size);
				prepared.buffers[index] = global.info;
				pack_memory_offset(index, global.buffer_offset);
				memo = global;
				FrameBreakdown::VAdd(FrameBreakdown::g_counters.slot_global_hit[kind - 1]);
				ExecRegions::NoteOp(ExecRegions::BufferGlobalHits);
				ExecRegions::InOp(ExecRegions::BuffersMemo, ExecRegions::DBufGlobal);
				continue;
			}
		}
		if (ExecRegions::Enabled()) {
			const bool range_memo = kind != 0 && bind_memo.kind == kind && bind_memo.slot.address == source.address &&
			                        bind_memo.slot.size == source.size && bind_memo.slot.buffer != nullptr;
			ExecRegions::NoteOp(kind == 0     ? ExecRegions::ObtainNoKind
			                    : !range_memo ? ExecRegions::ObtainNoMemo
			                    : kind == 1   ? ExecRegions::ObtainFailedPlain
			                    : kind == 2   ? ExecRegions::ObtainFailedWritten
			                                  : ExecRegions::ObtainFailedTexel);
			if (kind != 0 && !range_memo) {
				if (cache.IsRegionCpuModified(source.address, source.size)) ExecRegions::NoteOp(ExecRegions::ObtainNoMemoDirty);
				if (ExecRegions::SeenRange(source.address, source.size, kind)) ExecRegions::NoteOp(ExecRegions::ObtainNoMemoEvicted);
				if (memo.buffer != nullptr && memo.address == source.address && memo.size == source.size) {
					ExecRegions::NoteOp(ExecRegions::ObtainNoMemoSlotSame);
				}
			}
		}
		if (FrameBreakdown::Verbose() && (plain || written || texel)) {
			auto&      c    = FrameBreakdown::g_counters;
			const int  kind = written ? 1 : texel ? 2 : 0;
			auto&      slot = memo.buffer == nullptr                                         ? c.slot_none[kind]
			                  : memo.address == source.address && memo.size == source.size ? c.slot_same[kind]
			                  : memo.buffer->IsInBounds(source.address, source.size)        ? c.slot_inside[kind]
			                                                                                : c.slot_other[kind];
			FrameBreakdown::VAdd(slot);
		}
		const auto cpu           = cache.CpuModificationGeneration();
		const auto registration  = cache.RegistrationGeneration();
		const auto image_version = texel ? cache.TexelImageVersion(source.address) : 0;
		Buffer*    obtained      = nullptr;
		prepared.buffers[index]  = NativeStorageBuffer(m_context, source, resource, buffer_offset, &obtained);
		pack_memory_offset(index, buffer_offset);
		if ((plain || written || texel) && obtained != nullptr && !cache.IsStreamBuffer(obtained) &&
		    registration == cache.RegistrationGeneration() &&
		    (!texel || (image_version == cache.TexelImageVersion(source.address) &&
		                cache.TexelReadKeepable(source.address)))) {
			memo = {.buffer         = obtained,
			        .address        = source.address,
			        .size           = source.size,
			        .cpu_generation = cpu,
			        .registration   = registration,
			        .ranges_version = cache.GpuRangesVersion(),
			        .written        = written,
			        .image_version  = image_version,
			        .info           = prepared.buffers[index],
			        .buffer_offset  = buffer_offset};
			bind_memo = {memo, kind};
		} else {
			memo.buffer = nullptr;
			if (ExecRegions::Enabled() && (plain || written || texel)) {
				ExecRegions::NoteOp(obtained != nullptr && cache.IsStreamBuffer(obtained) ? ExecRegions::ObtainStream
				                                                                          : ExecRegions::ObtainNotKept);
			}
		}
		ExecRegions::NoteOp(ExecRegions::BufferObtains);
		ExecRegions::InOp(ExecRegions::BuffersObtain, ExecRegions::DBufObtain);
	}
	lookup_timer.Stop();
	FrameBreakdown::Timer upload_timer(FrameBreakdown::g_counters.dp_rebind_upload_ns);
	const bool srt  = layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0;
	const bool data = layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0;
	if (srt && data && !snapshot.flattened_srt.empty() && !prepared.shader_data.empty()) {
		std::tie(prepared.flattened_srt, prepared.shader_data_buffer) =
		    NativeUploadPair(m_context, snapshot.flattened_srt, prepared.shader_data);
	} else {
		if (srt) {
			prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
		}
		if (data) {
			prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
		}
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_rebind_images_ns,
	                                 FrameBreakdown::DetailEnabled());
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Kept bindings (see PrepareBindings()) whose images nothing changed since.
	const bool kept = prepared.lean && KeptImagesHold(prepared, prepared.lookup_epoch, prepared.lookup_set_epoch);
	if (kept && !LeanVerifyEnabled()) {
		return;
	}
	std::array<vk::ImageView, 64> kept_views {};
	for (uint32_t i = 0; kept && i < images.size() && i < kept_views.size(); i++) {
		kept_views[i] = images[i].image_view;
	}
	const auto stale = [&](const TextureBinding& binding) {
		const auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
		return image == nullptr || (!image->registered && !image->info.data.Empty()) ||
		       image->binding.needs_rebind;
	};
	// Table elements may share images with each other and with the direct images: decide which
	// are stale before any rebind clears an image's binding state.
	std::vector<bool> stale_tables(prepared.table_images.size());
	for (size_t k = 0; k < prepared.table_images.size(); k++) {
		stale_tables[k] = stale(prepared.table_images[k]);
	}
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		// Null images (no guest range) are never registered: as for the table elements below,
		// they need no rediscovery (each draw looked every null slot up again).
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			FrameBreakdown::Timer rres_timer(FrameBreakdown::g_tex.rebind_resolve_ns, FrameBreakdown::TexOn());
			FrameBreakdown::TexAdd(FrameBreakdown::g_tex.rebind_resolved);
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		FrameBreakdown::Timer view_timer(FrameBreakdown::g_tex.view_ns, FrameBreakdown::TexOn());
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (!binding.reused) {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		view_timer.Stop();
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		if (storage ? !image.usage.storage : !image.usage.texture) image.StateChanged();
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
		if (binding.lookup != nullptr && !binding.reused) {
			auto& found           = binding.lookup->found;
			found.id              = binding.image_id;
			found.desc            = binding.desc;
			found.view            = binding.image_view;
			found.image_set_epoch = texture_cache.ImageSetEpoch();
			found.state           = ImageLookupState::Of(binding.image_id, image);
			found.valid           = true;
			found.serial          = ++m_found_serial;
		}
		// The binding no longer refers to the description, which may be cleared.
		binding.lookup = nullptr;
		binding.reused = false;
	}
	for (size_t k = 0; k < prepared.table_images.size(); k++) {
		auto& texture = prepared.table_images[k];
		// A rediscovery above may have replaced (expanded, merged or recreated) the image this
		// element resolved to.
		if (stale_tables[k] || stale(texture)) {
			if (auto* old_image = texture_cache.m_slot_images.try_get(texture.image_id)) {
				old_image->binding = {};
			}
			texture = ResolveTableTexture(program.info.images.at(prepared.table_roots[k]),
			                              prepared.table_sources[k]);
			BindImage(texture.image_id, false);
		}
		texture.image_view = texture_cache.FindTexture(texture.image_id, texture.desc);
		auto& table_image  = texture_cache.GetImage(texture.image_id);
		if (!table_image.usage.texture) table_image.StateChanged();
		table_image.usage.texture = true;

	}
	// (The epoch itself may move meanwhile: other threads change images too, the presentation
	// thread transitions the images it shows.)
	if (kept) {
		uint32_t slot = 0;
		while (slot < images.size() && slot < kept_views.size() && images[slot].image_view == kept_views[slot]) {
			slot++;
		}
		if (slot < std::min(images.size(), kept_views.size())) {
			EXIT("lean draw: rebinding the kept images of shader 0x%016" PRIx64 " changed view %u of %zu\n",
			     program.shader_hash, slot, images.size());
		}
	}
}

bool RenderExecutor::KeptImagesHold(const PreparedBindings& prepared, uint64_t epoch, uint64_t set_epoch) {
	if (g_texture_epoch.load(std::memory_order_relaxed) == epoch) {
		return true;
	}
	if (g_image_set_epoch.load(std::memory_order_relaxed) != set_epoch) {
		return false;
	}
	// Other images changed (layouts, guest writes, targets of other passes): the bindings hold while
	// none of their own did.
	const auto& resources = prepared.runtime->program->info.images;
	if (prepared.images.size() != resources.size()) {
		return false;
	}
	auto& texture_cache = m_context.GetTextureCache();
	for (uint32_t i = 0; i < resources.size(); i++) {
		if (resources[i].heap) {
			continue;
		}
		const auto* image = texture_cache.m_slot_images.try_get(prepared.images[i].image_id);
		if (image == nullptr || image->binding_epoch.load(std::memory_order_relaxed) > epoch) {
			return false;
		}
	}
	return true;
}

void RenderExecutor::FinishLeanBindings(std::span<PreparedBindings* const> stages) {
	for (auto* stage: stages) {
		auto* memo = stage->image_memo;
		if (memo == nullptr) {
			continue;
		}
		// From when the lookups were made: a change of one of the images after then has a later stamp.
		const bool held = stage->lean_eligible && memo->holder == stage &&
		                  KeptImagesHold(*stage, stage->lookup_epoch, stage->lookup_set_epoch);
		memo->lean_epoch     = held ? stage->lookup_epoch : UINT64_MAX;
		memo->lean_set_epoch = stage->lookup_set_epoch;
		memo->lean_targets   = m_lean_targets;
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	// Draw-prep buckets (NHL27DRAWPREP) are sequential blocks here, in the original call order.
	bool uses_dma        = false;
	bool uses_image_heap = false;

	FindBuffers(stages);
	ExecRegions::To(ExecRegions::DrawFindBuffers);
	for (auto* stage: stages) {
		if (stage->runtime->program->info.uses_dma) {
			ExecRegions::Note(ExecRegions::DmaStages);
			CacheDmaBases(*stage);
			uses_dma = true;
		}
		uses_image_heap |= stage->runtime->program->info.uses_image_heap;
	}
	if (uses_dma) m_context.PrepareBda();
	ExecRegions::To(ExecRegions::DrawDma);
	if (uses_image_heap) {
		ExecRegions::Note(ExecRegions::HeapRebinds);
		m_context.GetImageHeap().Rebind(*this);
	}
	ExecRegions::To(ExecRegions::DrawHeap);
	{
		FrameBreakdown::Timer textures_timer(FrameBreakdown::g_counters.dp_textures_ns);
		FrameBreakdown::TexScope tex_scope;
		{
			FrameBreakdown::Timer rebind_timer(FrameBreakdown::g_tex.rebind_ns, FrameBreakdown::TexOn());
			for (auto* stage: stages) {
				RebindImages(*stage);
			}
		}
		auto& cache = m_context.GetTextureCache();
		FrameBreakdown::Timer rediscover_timer(FrameBreakdown::g_tex.rt_rediscover_ns, FrameBreakdown::TexOn());
		for (auto& target: colors) {
			EXIT_IF(!target.image_id);
			const auto old_image = cache.m_slot_images.try_get(target.image_id);
			if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
			    old_image->binding.needs_rebind) {
				if (old_image != nullptr) {
					old_image->binding = {};
				}
				InvalidateRenderTargetMemo();
				target.desc.view_info.base_level = target.guest_mip_level;
				target.desc.view_info.base_layer = target.guest_array_layer;
				target.image_id = cache.FindImage(target.desc);
				BindRenderTarget(target.image_id);
			}
		}
	}
	ExecRegions::To(ExecRegions::DrawImages);
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	{
		FrameBreakdown::Timer buffers_timer(FrameBreakdown::g_counters.dp_buffers_ns);
		FrameBreakdown::Timer rebind_timer(FrameBreakdown::g_counters.dp_rebind_ns);
		for (auto* stage: stages) {
			RebindBuffers(*stage);
		}
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	FrameBreakdown::Timer prep_graphics_timer(
	    FrameBreakdown::g_counters.prep_commit_graphics_ns,
	    pipeline_bind_point == vk::PipelineBindPoint::eGraphics && FrameBreakdown::DetailEnabled());
	FrameBreakdown::Timer prep_compute_timer(
	    FrameBreakdown::g_counters.prep_commit_compute_ns,
	    pipeline_bind_point == vk::PipelineBindPoint::eCompute && FrameBreakdown::DetailEnabled());
	KYTY_PROFILER_FUNCTION();
	auto   vk_buffer        = buffer.Handle();
	size_t image_descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += std::popcount(program.bindings.descriptor_mask);
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// The written ranges do not depend on the read, so collect them once per commit
	// (bound images can number in the hundreds; slot lookups per read dominated here).
	bool     written_ready = false;
	uint64_t written_lo    = UINT64_MAX;
	uint64_t written_hi    = 0;
	const auto check_read = [&](uint64_t address, uint64_t size) {
		if (address >= written_hi || address + size <= written_lo) return;
		for (const auto& [written_address, written_size]: m_written_image_ranges) {
			if (ImageRangeOverlaps(address, size, written_address, written_size)) {
				EXIT("scalar resource reads overlap an image or attachment write\n");
			}
		}
		for (const auto& [written_address, written_size]: m_written_buffer_ranges) {
			if (ImageRangeOverlaps(address, size, written_address, written_size)) {
				EXIT("scalar resource reads overlap a shader buffer write\n");
			}
		}
	};
	// KYTY_DBG_CHECK_SCALAR_READS=1: proves per draw that no scalar resource read overlaps a write
	// of the same draw (it never fired in NHL 27; about 1% of the execution thread).
	static const bool check_scalar_reads = std::getenv("KYTY_DBG_CHECK_SCALAR_READS") != nullptr;
	for (const auto* reader: prepared_bindings) {
		if (!check_scalar_reads) break;
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		if (!written_ready) {
			written_ready = true;
			for (const auto* writer: prepared_bindings) {
				if (writer->runtime->program->has_address_writes) {
					EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
				}
			}
			m_written_image_ranges.clear();
			m_written_buffer_ranges.clear();
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0) m_written_image_ranges.emplace_back(written.address, written.size);
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				for (const auto& resource: program.info.buffers) {
					if (!resource.written || resource.descriptor_index == UINT32_MAX) continue;
					const auto& written = writer->buffer_sources[resource.descriptor_index];
					if (written.size != 0) m_written_buffer_ranges.emplace_back(written.address, written.size);
				}
			}
			for (const auto& ranges: {std::cref(m_written_image_ranges), std::cref(m_written_buffer_ranges)}) {
				for (const auto& [written_address, written_size]: ranges.get()) {
					written_lo = std::min(written_lo, written_address);
					written_hi = std::max(written_hi, written_address + written_size);
				}
			}
		}
		// Reads are hundreds per draw, mostly adjacent and in address order: check each run of
		// touching reads once, and skip runs outside the span of all writes.
		uint64_t run_begin = 0;
		uint64_t run_end   = 0;
		for (const auto [address, size]: reads) {
			if (size == 0) continue;
			if (run_end != 0 && address >= run_begin && address <= run_end) {
				run_end = std::max(run_end, address + size);
				continue;
			}
			if (run_end != 0) check_read(run_begin, run_end - run_begin);
			run_begin = address;
			run_end   = address + size;
		}
		if (run_end != 0) check_read(run_begin, run_end - run_begin);
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
	}
	ExecRegions::InDraw(ExecRegions::CommitSetup);
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	m_descriptor_writes.reserve(write_count);
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		const auto transit = [&](TextureBinding& binding) {
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		};
		ExecRegions::InDraw(ExecRegions::CommitWrites);
		// Kept bindings (see PrepareBindings()) are in the layouts the draw that made them left
		// while no image changed since.
		const bool keep_layouts =
		    descriptors.lean && KeptImagesHold(descriptors, descriptors.lookup_epoch, descriptors.lookup_set_epoch) &&
		    pipeline_bind_point == vk::PipelineBindPoint::eGraphics && descriptors.image_memo != nullptr &&
		    descriptors.image_memo->lean_targets == m_lean_targets;
		if (!keep_layouts || LeanVerifyEnabled()) {
			// The layouts the kept bindings had when PrepareBindings() kept them (RebindImages()
			// looks up the null images again, which resets theirs).
			const auto& kept_layouts = descriptors.lean_layouts;
			for (auto& binding: descriptors.images) {
				transit(binding);
			}
			for (auto& binding: descriptors.table_images) {
				transit(binding);
			}
			for (size_t i = 0; keep_layouts && i < descriptors.images.size() && i < kept_layouts.size(); i++) {
				const auto& binding = descriptors.images[i];
				if (!binding.image_id || binding.layout == kept_layouts[i]) continue;
				const auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
				EXIT("lean draw: kept image %zu of shader 0x%016" PRIx64 " needed a transition %s -> %s "
				     "(target=%d general=%d write=%d depth=%d empty=%d storage=%d)\n",
				     i, program.shader_hash, vk::to_string(kept_layouts[i]).c_str(), vk::to_string(binding.layout).c_str(),
				     image.binding.is_target, image.binding.force_general, image.binding.shader_write,
				     image.info.IsDepth(), image.info.data.Empty(),
				     binding.desc.type == TextureCache::BindingType::Storage);
			}
		}
		ExecRegions::InDraw(ExecRegions::CommitTransit);

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != write.descriptorCount);
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
						}
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							EXIT_IF(sampler == nullptr);
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			if (resource.heap) continue;
			EXIT_IF(texture.image_id &&
			        resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()));
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}
		size_t table_element = 0;
		ForEachImageTable(program.info.images, [&](uint32_t root_index, BindingKind kind) {
			const auto& root  = program.info.images[root_index];
			const auto  start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   root.table_descriptor_base;
			for (uint32_t slot = 0; slot < root.table_capacity; slot++) {
				m_descriptor_images[start + slot] =
				    MakeImageInfo(descriptors.table_images.at(table_element++), 0u);
			}
		});

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	ExecRegions::InDraw(ExecRegions::CommitWrites);
	// Keep this draw's push-constant block on each stage for the frame dump (FrameDump::OnDraw).
	for (auto* prepared: prepared_bindings) {
		if (!FrameDump::Enabled()) break;
		prepared->push_dwords       = push_data.dwords;
		prepared->push_dwords_valid = has_push_data;
	}

	// KYTY_SKIP_REDUNDANT_BINDS=1: within one command buffer and pipeline layout, push
	// descriptors and push constants stay bound, so identical consecutive pushes can be skipped.
	// Objects referenced by the current command buffer are not destroyed while it records.
	const bool skip_redundant =
	    Common::Experiments::Enabled(Common::Experiments::SkipRedundantBinds);
	auto& last = m_redundant[pipeline_bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0];
	if (skip_redundant) {
		const auto tick  = m_context.GetCommandScheduler().CurrentTick();
		const auto epoch = BindEpoch();
		if (last.tick != tick || last.epoch != epoch || last.buffer != vk_buffer.Key() ||
		    last.layout != pipeline.pipeline_layout) {
			last.tick              = tick;
			last.epoch             = epoch;
			last.buffer            = vk_buffer.Key();
			last.layout            = pipeline.pipeline_layout;
			last.descriptors_valid = false;
			last.push_valid        = false;
		}
	} else {
		// The experiment can be switched at run time: pushes made while it is off are not
		// remembered, so forget the last state.
		last.descriptors_valid = false;
		last.push_valid        = false;
	}

	if (has_push_data) {
		const bool same = skip_redundant && last.push_valid && last.push_stages == push_stages &&
		                  std::equal(last.push.begin(), last.push.end(), push_data.dwords.begin(),
		                             push_data.dwords.end());
		if (!same) {
			vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
			                        push_data.dwords.data());
			if (skip_redundant) {
				last.push.assign(push_data.dwords.begin(), push_data.dwords.end());
				last.push_stages = push_stages;
				last.push_valid  = true;
			}
		}
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			bool same = false;
			if (skip_redundant) {
				auto& key = m_descriptor_key;
				key.clear();
				for (const auto& write: m_descriptor_writes) {
					key.push_back((uint64_t {write.dstBinding} << 32u) |
					              (uint64_t {static_cast<uint32_t>(write.descriptorType)} << 16u) |
					              write.descriptorCount);
					for (uint32_t i = 0; i < write.descriptorCount; i++) {
						if (write.pBufferInfo != nullptr) {
							const auto& info = write.pBufferInfo[i];
							key.push_back(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer)));
							key.push_back(info.offset);
							key.push_back(info.range);
						} else if (write.pImageInfo != nullptr) {
							const auto& info = write.pImageInfo[i];
							key.push_back(reinterpret_cast<uint64_t>(static_cast<VkSampler>(info.sampler)));
							key.push_back(reinterpret_cast<uint64_t>(static_cast<VkImageView>(info.imageView)));
							key.push_back(static_cast<uint64_t>(info.imageLayout));
						} else if (write.pTexelBufferView != nullptr) {
							key.push_back(reinterpret_cast<uint64_t>(
							    static_cast<VkBufferView>(write.pTexelBufferView[i])));
						} else {
							key.push_back(UINT64_MAX);
						}
					}
				}
				same = last.descriptors_valid && last.descriptors == key;
				if (!same) {
					last.descriptors.assign(key.begin(), key.end());
					last.descriptors_valid = true;
				}
			}
			if (!same) {
				vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
				                               static_cast<uint32_t>(m_descriptor_writes.size()),
				                               m_descriptor_writes.data());
			}
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			// On the recording thread, before the bind (vkUpdateDescriptorSets was 2.9% of the
			// execution thread in play).
			vk_buffer.updateDescriptorSets(m_context.GetGraphics().device,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
			InvalidateRedundantBinds();
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
}

} // namespace Libs::Graphics
