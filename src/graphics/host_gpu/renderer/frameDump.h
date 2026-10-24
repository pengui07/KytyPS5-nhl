#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_

// "Poor man's RenderDoc": KYTY_DBG_FRAME_DUMP=1 enables a frame dump that is triggered with F2 in the
// game window. KYTY_DBG_FRAME_DUMP_CS_CONSTANTS=1 additionally surveys all compute bindings and
// copies bound GPU buffers up to 4096 bytes, without reading protected guest backing.
// The next complete guest frame (flip to flip) is captured: every draw/dispatch is
// logged to <captures>/frame_<n>/frame.txt and every render pass's color/depth targets and every
// dispatch's written storage images are read back and saved as PNG. Zero cost when the flag is off
// (one relaxed atomic load per draw/dispatch/end-of-render-pass).

#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

struct TextureBinding;
struct PreparedBindings;

namespace FrameDump {

extern std::atomic<bool> g_active;

[[nodiscard]] inline bool Active() noexcept {
	return g_active.load(std::memory_order_relaxed);
}

// True when KYTY_DBG_FRAME_DUMP is set (cached).
[[nodiscard]] bool Enabled() noexcept;

// F2 in the game window: capture the next complete guest frame.
void Request();

// Writes 8-bit RGB pixels (rows top to bottom) to a PNG file.
bool WriteRgbPng(const std::string& path, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgb);

// Called when the guest submits a flip (command processor thread). Starts or finishes a capture.
void OnGuestFlip(CommandBuffer& buffer);

struct TargetRef {
	vk::ImageView                   view = nullptr;
	ImageId                         image_id;
	uint32_t                        slot = 0;
	vk::ImageLayout                 layout = vk::ImageLayout::eUndefined;
	const TextureCache::ImageDesc*  desc   = nullptr;
	uint32_t                        mip    = 0;
	uint32_t                        layer  = 0;
	vk::Extent2D                    extent {};
};

struct DrawLog {
	uint64_t                         vs_hash        = 0;
	uint64_t                         ps_hash        = 0;
	uint32_t                         count          = 0;
	uint32_t                         instances      = 0;
	bool                             indexed        = false;
	bool                             mesh           = false;
	bool                             indirect       = false;
	bool                             ps_active      = false;
	bool                             depth_test     = false;
	bool                             depth_write    = false;
	vk::CompareOp                    depth_compare  = vk::CompareOp::eNever;
	std::span<const TargetRef>       colors;
	const TargetRef*                 depth = nullptr;
	const std::vector<TextureBinding>* vs_images[3] {};
	const std::vector<TextureBinding>* ps_images = nullptr;
	// Resource tables of the draw (for the lighting-buffer content dump, see FrameDump::OnDraw).
	const PreparedBindings*            ps_bindings = nullptr;
	std::string                       state_text;
	std::shared_ptr<Buffer>            indirect_snapshot;
	uint64_t                          indirect_args_size = 0;
	uint64_t                          indirect_guest_addr = 0;
	uint32_t                          indirect_stride = 0;
	uint32_t                          indirect_max_count = 0;
	bool                              skipped = false;
};

// Registers the draw's render targets (so the end of the render pass can read them back) and logs it.
void OnDraw(RenderContext& context, const DrawLog& draw);

// Called from CommandBuffer::EndRendering after vkCmdEndRendering, with the pass that just ended.
void OnEndRendering(const CommandBuffer& buffer, const RenderState& pass);

struct StorageUse {
	const TextureBinding* binding = nullptr;
	bool                  written = false;
	bool                  storage = false;
};

// Called after a dispatch was recorded; logs it and reads back the storage images it wrote.
void OnDispatch(RenderContext& context, CommandBuffer& buffer, uint64_t cs_hash, uint32_t gx,
                uint32_t gy, uint32_t gz, bool indirect, std::span<const StorageUse> images,
                const PreparedBindings* bindings = nullptr);

// KYTY_DBG_TRACE_WRITES=<hexaddr>[,<hexaddr>...]: print a line whenever a GPU write (buffer cache write
// log, color/depth target of a draw, storage image of a dispatch) overlaps one of the addresses. Works
// without KYTY_DBG_FRAME_DUMP. One relaxed bool test when unset.
[[nodiscard]] bool TraceEnabled() noexcept;
void TraceWrite(const char* what, uint64_t begin, uint64_t end, const char* kind, uint64_t hash);

// KYTY_DBG_TRACE_CS=<hex cs hash>[,...]: for every dispatch of those compute shaders (first 24, then every
// 64th) print groups, user-data, shader_data, every image (addr/size/format/type/mips/layers/view, WRITTEN)
// and every buffer source (guest addr/size, nonzero dwords + first 8 dwords of the CPU view).
[[nodiscard]] bool TraceCsEnabled(uint64_t cs_hash) noexcept;
void TraceDispatch(uint64_t cs_hash, uint32_t gx, uint32_t gy, uint32_t gz, const PreparedBindings& bindings,
                   std::span<const uint8_t> image_written);

} // namespace FrameDump
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_
