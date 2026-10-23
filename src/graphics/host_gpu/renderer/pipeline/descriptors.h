#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureDescDerivation;

// The state of a cached image that its lookups and acquisitions depend on. While an image keeps
// it (and the image set around its range is unchanged), looking it up again and acquiring its
// views finds the same image and views and changes nothing.
struct ImageLookupState {
	ImageId  id;
	ImageId  depth_id;
	uint64_t track_addr      = 0;
	uint64_t track_addr_end  = 0;
	bool     cpu_dirty       = false;
	bool     buffer_modified = false;
	bool     gpu_modified    = false;
	bool     texture         = false;
	bool     storage         = false;
	bool     render_target   = false;
	bool     depth_target    = false;

	bool operator==(const ImageLookupState&) const = default;

	[[nodiscard]] static ImageLookupState Of(ImageId id, const Image& image) {
		return {.id              = id,
		        .depth_id        = image.depth_id,
		        .track_addr      = image.track_addr,
		        .track_addr_end  = image.track_addr_end,
		        .cpu_dirty       = image.IsCpuDirty(),
		        .buffer_modified = image.IsBufferModified(),
		        .gpu_modified    = image.IsGpuModified(),
		        .texture         = image.usage.texture,
		        .storage         = image.usage.storage,
		        .render_target   = image.usage.render_target,
		        .depth_target    = image.usage.depth_target};
	}
};

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
	// The description whose last lookup this binding reuses (image_view is set) or records once
	// its view is acquired; see RenderExecutor::ResolveTexture().
	TextureDescDerivation* lookup = nullptr;
	bool                   reused = false;
	// The Found::serial `desc` was copied from by RenderExecutor::ReuseTextureLookup (0: none).
	uint64_t desc_serial = 0;
};

// The T#s the last draw of a program bound and the description each had (null when its lookups
// are not reused), valid while the description map keeps `descs_generation`.
struct ProgramImageMemo {
	uint64_t                                           descs_generation = 0;
	std::vector<ShaderRecompiler::IR::DescriptorValue> images;
	std::vector<TextureDescDerivation*>                image_descs;
	// The program's image bindings while another program uses its stage (swapped with
	// PreparedBindings::images; `holder` has them meanwhile).
	std::vector<TextureBinding> bindings;
	const void*                 holder = nullptr;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	// FindBuffers: the V# each buffer slot was found for, with its program and the buffer-set and
	// mapping generations then: a slot whose V# is unchanged keeps its source.
	struct BufferSlotKey {
		ShaderRecompiler::IR::DescriptorValue value;
		const void*                           program    = nullptr;
		uint64_t                              generation = 0;
	};
	std::vector<BufferSlotKey> buffer_keys;
	std::vector<vk::DescriptorBufferInfo> buffers;
	// RebindBuffers: what each plain-read slot bound last (see BufferCache::PlainReadHolds()).
	struct BufferSlotMemo {
		Buffer*                  buffer         = nullptr;
		uint64_t                 address        = 0;
		uint64_t                 size           = 0;
		uint64_t                 cpu_generation = 0;
		uint64_t                 registration   = 0;
		// Written slots (WrittenBindHolds()): the GPU ranges version after the binding.
		uint64_t                 ranges_version = 0;
		bool                     written        = false;
		uint64_t                 image_version  = 0; // texel reads (0: not one)
		vk::DescriptorBufferInfo info;
		uint32_t                 buffer_offset = 0;
	};
	std::vector<BufferSlotMemo> buffer_memos;
	std::vector<TextureBinding>           images;
	// Bindless table descriptors in binding and table-range order, with the T# and the root
	// image each element was resolved for.
	std::vector<TextureBinding>                         table_images;
	std::vector<ShaderRecompiler::IR::DescriptorValue> table_sources;
	std::vector<uint32_t>                               table_roots;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
	// Push-constant block committed for the draw (frame dump only; valid when push_dwords_valid).
	std::array<uint32_t, ShaderRecompiler::IR::PushData::DwordCount> push_dwords {};
	bool                                                              push_dwords_valid = false;
	// What the previous draw of this stage bound: a stage that binds the same program with the
	// same S#s again keeps its samplers.
	const void*                                        last_program          = nullptr;
	uint64_t                                           last_descs_generation = 0;
	std::vector<ShaderRecompiler::IR::DescriptorValue> last_samplers;
	// The image memo of `last_program` (RenderExecutor::m_program_image_memos).
	ProgramImageMemo* image_memo = nullptr;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

// KYTY_DBG_DRAW_REPEATS=1: how each draw's stage resources compare with the last use of the same
// program (NHL27REPEATS every 5 s). Begin/End bracket the stages of one draw.
namespace DrawRepeats {
[[nodiscard]] bool Enabled();
void DrawBegin();
void DrawEnd();
} // namespace DrawRepeats

class CommandBuffer;

// Experiment LazyGuestBarriers (bit 52, KYTY_LAZY_GUEST_BARRIERS): a guest global barrier is owed to the
// recording target instead of recorded, and recorded before the first later draw or dispatch that reads
// a range written before it, writes a range read or written before it, or accesses memory by address,
// and before any other work that accesses memory there (CommandRecorder::GuestWork). The ranges are the
// bound buffers, the images (data, stencil, metadata), the attachments and the index, vertex and indirect
// ranges. KYTY_DBG_BARRIER_HAZARDS=1 prints how the barriers went (NHL27HAZARDS every 5 s).
namespace GuestBarriers {
using Range = std::pair<uint64_t, uint64_t>; // address, size
[[nodiscard]] bool Lazy();
[[nodiscard]] bool StatsEnabled();
// ExecuteGlobalBarrier with the experiment on: owes `barrier` to `buffer`'s recording target. False:
// record it now (not all the work since the last recorded barrier was tracked, or too many ranges).
[[nodiscard]] bool Owe(CommandBuffer& buffer, const vk::MemoryBarrier2& barrier);
// ExecuteGlobalBarrier recorded the barrier: nothing is owed and tracking starts over.
void Recorded();
// A draw or dispatch is checked, or a barrier is owed: CheckWork has something to do.
[[nodiscard]] bool Active();
// Until it ends, the checked draw's or dispatch's commands do not record an owed barrier.
struct CheckedScope {
	CheckedScope() = default;
	~CheckedScope();
	CheckedScope(const CheckedScope&)            = delete;
	CheckedScope& operator=(const CheckedScope&) = delete;
};
// A draw or dispatch about to record its commands, with its ranges besides its stages' bindings:
// records an owed barrier first when the work overlaps the work before it.
void CheckWork(CommandBuffer& buffer, std::span<PreparedBindings* const> stages, std::span<const Range> reads,
               std::span<const Range> writes, bool unknown);
} // namespace GuestBarriers

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
