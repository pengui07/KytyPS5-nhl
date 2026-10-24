#include "graphics/presentation/fsr/nhl27Fsr3.h"
#include "graphics/presentation/fsr/nhl27Fsr3Constants.h"
#include "graphics/presentation/fsr/fsr3.h"
#include "graphics/presentation/fsr/fsr4Bridge.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "gpu_tiler_shaders/gpu_tiler_fsr_color_clean_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_fsr_ghost_guard_spv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {
namespace {
constexpr std::array<uint64_t, 4> kPass {
	0x4618098710b83f8fULL, 0xdb7e537e65c00dd1ULL,
	0xbbb674ad1182ed8cULL, 0xa36f0b4faa20493aULL};
using S = vk::PipelineStageFlagBits2;
using A = vk::AccessFlagBits2;

void MemoryBarrier(CommandRecorder command) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask = barrier.dstStageMask = S::eAllCommands;
	barrier.srcAccessMask = barrier.dstAccessMask = A::eMemoryRead | A::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers = &barrier;
	command.pipelineBarrier2(dependency);
}
void ImageBarrier(CommandRecorder command, vk::Image image, vk::ImageLayout from,
				  vk::ImageLayout to, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask = barrier.dstStageMask = S::eAllCommands;
	barrier.srcAccessMask = from == vk::ImageLayout::eUndefined
		? vk::AccessFlags2 {} : A::eMemoryRead | A::eMemoryWrite;
	barrier.dstAccessMask = A::eMemoryRead | A::eMemoryWrite;
	barrier.oldLayout = from;
	barrier.newLayout = to;
	barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	command.pipelineBarrier2(dependency);
}
}
struct Nhl27Fsr3::Impl {
	struct Owned {
		Fsr3Image resource;
		VmaAllocation allocation = nullptr;
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	};
	struct Role {
		uint64_t address = 0;
		vk::Format format {};
		vk::Extent3D extent {};
		bool Matches(const TextureBinding& other) const {
			return address == other.desc.info.data.address &&
				format == other.desc.info.pixel_format && extent == other.desc.info.extent;
		}
		static Role Of(const TextureBinding& binding) {
			return {binding.desc.info.data.address, binding.desc.info.pixel_format,
					binding.desc.info.extent};
		}
	};
	GraphicContext& graphics;
	std::array<Owned, 7> images {}; // color, depth, raw motion, output, SDK depth/motion/previous
	std::unique_ptr<Fsr3> fsr;
	vk::Buffer constants = nullptr, transfer = nullptr;
	VmaAllocation constants_alloc = nullptr, transfer_alloc = nullptr;
	void* mapped = nullptr;
	uint64_t last_tick = 0, serial = 0, rejected = 0, recorded = 0;
	unsigned stage = 0;
	std::atomic<bool> last_recorded {false}, failed {false};
	Nhl27Fsr3Constants captured;
	std::optional<Nhl27Fsr3Constants> end;
	std::array<Role, 8> prepare {}, reactive {};
	Role luma;
	Nhl27HistoryState last_state {};
	bool have_last = false, missed = true;
	uint64_t resets = 0, present_skips = 0, dirty_reads = 0, clean_mismatches = 0, replaced = 0;
	uint64_t dumps = 0;
	// KYTY_FSR3_CLEAN_COLOR=1: broken and lone over-bright colour pixels are repaired before FSR
	// (gpu_tiler_fsr_color_clean.comp). `clean` receives the repaired copy of `transfer`.
	vk::Buffer clean = nullptr;
	VmaAllocation clean_alloc = nullptr;
	vk::PipelineLayout clean_layout = nullptr;
	vk::Pipeline clean_pipeline = nullptr;
	uint64_t cleaned = 0;
	// KYTY_FSR4=1: FSR 4 through the D3D12 bridge in place of FSR 3.1 (falls back to 3.1 when unusable).
	std::unique_ptr<Fsr4Bridge> bridge;
	uint64_t fsr4_frames = 0, fsr4_failed = 0;
	// KYTY_FSR4_GUARD=1: reactive mask for FSR 4 (gpu_tiler_fsr_ghost_guard.comp), marking pixels whose
	// last-frame colour no longer fits the current picture, so FSR 4 drops its history there.
	struct GuardBuffer {
		vk::Buffer buffer = nullptr;
		VmaAllocation allocation = nullptr;
	};
	std::array<GuardBuffer, 2> guard_color {}; // this frame / last frame, alternating
	GuardBuffer guard_motion, guard_raw, guard_eroded, guard_packed, guard_show, guard_motion_out;
	vk::PipelineLayout guard_layout = nullptr;
	vk::Pipeline guard_pipeline = nullptr;
	bool guard_failed = false, guard_have_last = false, reactive_ready = false;
	unsigned guard_flip = 0;
	uint32_t guard_last_frame = 0;
	uint64_t guarded = 0;
	unsigned logged_bindings = 0;
	std::array<uint64_t, 8> reset_reasons {}; // FSR history resets by Nhl27ResetReason
	uint64_t soft_skipped = 0;                 // resets KYTY_FSR4_SOFT_RESET left out for FSR 4
	std::array<uint32_t, 52> last_words {};    // the last constant block read (for rejection details)
	bool last_words_valid = false;
	uint64_t prepare_rejects = 0;
	uint64_t chain_skipped = 0; // game FSR 2 passes 1-3 dropped (KYTY_FSR3_NHL27_SKIP_CHAIN=1)
	float last_delta_ms = 33.3333f;
	// CPU time spent on FSR work (stats line, microseconds per frame): all of OnDispatch, the two
	// queue submissions FSR 4 adds, and the D3D12 side (FSR 4 recording + submit).
	uint64_t cpu_ns_total = 0, cpu_ns_flush = 0, cpu_ns_d3d = 0;
	static uint64_t Now() {
		return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count());
	}
	std::atomic<uint64_t> presented {0};
	explicit Impl(GraphicContext& gfx) : graphics(gfx) {
		if (!gfx.physical_device.getFeatures().shaderInt16) {
			failed.store(true, std::memory_order_release);
			std::printf("NHL27FSR3GAME status=unsupported reason=shaderInt16\n");
			std::fflush(stdout);
		}
	}
	~Impl() {
		// RenderContext shuts down/drains the scheduler before RenderExecutor's owner dies.
		bridge.reset();
		ReleaseImages();
		if (constants) vmaDestroyBuffer(graphics.allocator, constants, constants_alloc);
		if (clean_pipeline) graphics.device.destroyPipeline(clean_pipeline, nullptr);
		if (clean_layout) graphics.device.destroyPipelineLayout(clean_layout, nullptr);
		if (guard_pipeline) graphics.device.destroyPipeline(guard_pipeline, nullptr);
		if (guard_layout) graphics.device.destroyPipelineLayout(guard_layout, nullptr);
	}
	void Reject(const char* why) {
		stage = 0;
		missed = true; // next dispatch must not trust history
		if (++rejected <= 8 || rejected % 600 == 0) {
			std::printf("NHL27FSR3GAME status=rejected reason=%s count=%llu\n", why,
						static_cast<unsigned long long>(rejected));
			std::fflush(stdout);
		}
	}
	void ReleaseImages() {
		fsr.reset();
		for (auto& image : images) {
			if (image.resource.image)
				vmaDestroyImage(graphics.allocator, image.resource.image, image.allocation);
			image = {};
		}
		if (transfer) vmaDestroyBuffer(graphics.allocator, transfer, transfer_alloc);
		transfer = nullptr;
		transfer_alloc = nullptr;
		if (clean) vmaDestroyBuffer(graphics.allocator, clean, clean_alloc);
		clean = nullptr;
		clean_alloc = nullptr;
		ReleaseGuard();
	}
	void ReleaseGuard() {
		for (auto* g : {&guard_color[0], &guard_color[1], &guard_motion, &guard_raw, &guard_eroded, &guard_packed,
						&guard_show, &guard_motion_out}) {
			if (g->buffer) vmaDestroyBuffer(graphics.allocator, g->buffer, g->allocation);
			*g = {};
		}
		guard_have_last = false;
	}
	// The order the game binds images to the four FSR passes, printed once per pass (for later work).
	void LogBindings(unsigned pass, const std::vector<TextureBinding>& b) {
		if ((logged_bindings & (1u << pass)) != 0) return;
		logged_bindings |= 1u << pass;
		for (size_t i = 0; i < b.size(); ++i) {
			std::printf("NHL27FSR3 binding pass=%u slot=%zu format=%d size=%ux%u\n", pass, i,
						static_cast<int>(b[i].desc.info.pixel_format), b[i].desc.info.extent.width,
						b[i].desc.info.extent.height);
		}
		std::fflush(stdout);
	}
	bool AllocateBuffer(uint64_t bytes, bool host, vk::Buffer& buffer,
						VmaAllocation& allocation, void** pointer = nullptr) {
		vk::BufferCreateInfo info {};
		info.size = bytes;
		info.usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		VmaAllocationCreateInfo alloc {};
		alloc.usage = VMA_MEMORY_USAGE_AUTO;
		if (host) alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer native = VK_NULL_HANDLE;
		VmaAllocationInfo result {};
		if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&info),
							&alloc, &native, &allocation, &result) != VK_SUCCESS) return false;
		buffer = native;
		if (pointer) *pointer = result.pMappedData;
		return true;
	}
	std::optional<Nhl27Fsr3Constants> ReadConstants(RenderContext& context, CommandBuffer& command,
													 const PreparedBindings& bindings) {
		if (bindings.buffer_sources.empty() || bindings.buffers.empty() ||
			bindings.buffer_sources[0].size != 208 || bindings.buffers[0].range < 208 ||
			!bindings.buffers[0].buffer) return std::nullopt;
		// The game writes the block from the CPU into a ring, so its backing is normally clean
		// and equals what the dispatch uploaded. Verify mode also reads the GPU bytes and compares.
		std::array<uint32_t, 52> clean;
		const bool have_clean = LibKernel::Memory::TryReadCleanBufferBacking(
			bindings.buffer_sources[0].address, clean.data(), 208);
		if (have_clean) {
			last_words = clean;
			last_words_valid = true;
		}
		if (have_clean && !VerifyEnabled()) return Nhl27Fsr3Constants::Decode(clean);
		if (!constants && !AllocateBuffer(208, true, constants, constants_alloc, &mapped))
			return std::nullopt;
		auto recorder = command.Handle();
		MemoryBarrier(recorder);
		vk::BufferCopy copy {bindings.buffers[0].offset, 0, 208};
		recorder.copyBuffer(bindings.buffers[0].buffer, constants, 1, &copy);
		vk::MemoryBarrier2 host {};
		host.srcStageMask = S::eTransfer; host.srcAccessMask = A::eTransferWrite;
		host.dstStageMask = S::eHost; host.dstAccessMask = A::eHostRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &host;
		recorder.pipelineBarrier2(dependency);
		// Synchronous: decode the actual shader-visible GPU bytes when the backing is dirty or
		// in verify mode. The wrapper stays stable across the new submission.
		context.GetCommandScheduler().Wait(context.GetCommandScheduler().CurrentTick());
		InvalidateRedundantBinds();
		if (vmaInvalidateAllocation(graphics.allocator, constants_alloc, 0, 208) != VK_SUCCESS)
			return std::nullopt;
		std::array<uint32_t, 52> words;
		std::memcpy(words.data(), mapped, 208);
		last_words = words;
		last_words_valid = true;
		if (!have_clean) ++dirty_reads;
		if (have_clean && clean != words && ++clean_mismatches <= 8) {
			std::printf("NHL27FSR3GAME status=verify-mismatch source=constants count=%llu\n",
						static_cast<unsigned long long>(clean_mismatches));
			std::fflush(stdout);
		}
		return Nhl27Fsr3Constants::Decode(words);
	}
	void CheckOutput(RenderContext& context, CommandBuffer& command, uint32_t guest_frame, bool history) {
		auto recorder = command.Handle();
		auto& output = images[3].resource;
		ImageBarrier(recorder, output.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
		vk::BufferImageCopy copy {};
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageOffset = {int32_t(output.extent.width / 2), int32_t(output.extent.height / 2), 0};
		copy.imageExtent = {2, 2, 1};
		recorder.copyImageToBuffer(output.image, vk::ImageLayout::eTransferSrcOptimal, constants, 1, &copy);
		ImageBarrier(recorder, output.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
		vk::MemoryBarrier2 host {};
		host.srcStageMask = S::eTransfer; host.srcAccessMask = A::eTransferWrite;
		host.dstStageMask = S::eHost; host.dstAccessMask = A::eHostRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &host;
		recorder.pipelineBarrier2(dependency);
		context.GetCommandScheduler().Wait(last_tick);
		InvalidateRedundantBinds();
		if (!last_recorded.load(std::memory_order_acquire) ||
			vmaInvalidateAllocation(graphics.allocator, constants_alloc, 0, 32) != VK_SUCCESS) {
			failed.store(true, std::memory_order_release);
			Reject("shadow-dispatch-or-readback");
			return;
		}
		std::array<uint16_t, 16> pixels;
		std::memcpy(pixels.data(), mapped, 32);
		std::array<float, 16> decoded {};
		bool finite = true;
		for (size_t i = 0; i < pixels.size(); ++i) {
			const unsigned exponent = (pixels[i] >> 10) & 31, mantissa = pixels[i] & 1023;
			if (exponent == 31) { finite = false; continue; }
			decoded[i] = exponent ? std::ldexp(1 + float(mantissa) / 1024, int(exponent) - 15)
				: std::ldexp(float(mantissa), -24);
			if (pixels[i] & 0x8000) decoded[i] = -decoded[i];
		}
		std::printf("NHL27FSR3GAME status=readback mode=%s guest_frame=%u "
			"pixels=4 finite=%d rgba=%.9g,%.9g,%.9g,%.9g\n",
			history ? "history-shadow" : "reset-shadow", guest_frame, int(finite),
			decoded[0], decoded[1], decoded[2], decoded[3]);
		std::fflush(stdout);
		if (!finite) { failed.store(true, std::memory_order_release); Reject("nonfinite-shadow-output"); }
	}
	// KYTY_FSR3_NHL27_DUMP=N (debug, waits for the GPU): every N FSR frames, up to 4 times, read back
	// FSR's colour input, motion input and output, print what they hold (NHL27FSR3DUMP lines: broken
	// values, very bright pixels, isolated bright dots) and write the raw images next to the emulator
	// (fsr_dump_<frame>_color.bin = B10G11R11, _motion.bin = RG16F, _output.bin = RGBA16F).
	static uint64_t DumpEvery() {
		static const uint64_t every = [] {
			const char* value = std::getenv("KYTY_FSR3_NHL27_DUMP");
			return value != nullptr ? static_cast<uint64_t>(std::strtoull(value, nullptr, 10)) : uint64_t {0};
		}();
		return every;
	}
	static float Half(uint16_t bits) {
		const unsigned exponent = (bits >> 10) & 31, mantissa = bits & 1023;
		float value = exponent == 31 ? (mantissa ? NAN : INFINITY)
			: exponent ? std::ldexp(1 + float(mantissa) / 1024, int(exponent) - 15)
			: std::ldexp(float(mantissa), -24);
		return (bits & 0x8000) ? -value : value;
	}
	static float SmallFloat(uint32_t bits, unsigned mantissa_bits) {
		const unsigned exponent = (bits >> mantissa_bits) & 31, mantissa = bits & ((1u << mantissa_bits) - 1);
		if (exponent == 31) return mantissa ? NAN : INFINITY;
		const float scale = float(1u << mantissa_bits);
		return exponent ? std::ldexp(1 + float(mantissa) / scale, int(exponent) - 15)
			: std::ldexp(float(mantissa) / scale, -14);
	}
	// rgb per pixel -> stats line; isolated = brighter than 1.0 and than 8x the brightest of its 4 neighbours.
	static void DumpStats(const char* what, uint32_t frame, uint32_t width, uint32_t height,
						  const std::vector<float>& luma, uint64_t nonfinite) {
		uint64_t over16 = 0, over256 = 0, over4096 = 0, isolated = 0;
		float    peak = 0;
		double   sum = 0;
		for (uint32_t y = 0; y < height; ++y) {
			for (uint32_t x = 0; x < width; ++x) {
				const float v = luma[size_t(y) * width + x];
				if (!std::isfinite(v)) continue;
				sum += v;
				peak = std::max(peak, v);
				over16 += v > 16 ? 1 : 0;
				over256 += v > 256 ? 1 : 0;
				over4096 += v > 4096 ? 1 : 0;
				if (v > 1 && x > 0 && y > 0 && x + 1 < width && y + 1 < height) {
					float around = 0;
					for (const auto [dx, dy] : {std::pair {-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
						const float n = luma[size_t(y + dy) * width + (x + dx)];
						around = std::max(around, std::isfinite(n) ? n : 0.0f);
					}
					isolated += v > around * 8 ? 1 : 0;
				}
			}
		}
		std::printf("NHL27FSR3DUMP %s frame=%u size=%ux%u nonfinite=%llu over16=%llu over256=%llu "
					"over4096=%llu isolated=%llu max=%g mean=%g\n", what, frame, width, height,
					static_cast<unsigned long long>(nonfinite), static_cast<unsigned long long>(over16),
					static_cast<unsigned long long>(over256), static_cast<unsigned long long>(over4096),
					static_cast<unsigned long long>(isolated), double(peak),
					sum / (double(width) * height));
		std::fflush(stdout);
	}
	void DumpFrame(RenderContext& context, CommandBuffer& command, uint32_t guest_frame) {
		struct Target {
			Owned*        image;
			uint32_t      bytes_per_pixel;
			const char*   name;
			vk::Buffer    buffer = nullptr;
			VmaAllocation allocation = nullptr;
			void*         data = nullptr;
		};
		std::array<Target, 3> targets {{{&images[0], 4, "color"}, {&images[2], 4, "motion"}, {&images[3], 8, "output"}}};
		for (auto& t : targets) {
			const auto& e = t.image->resource.extent;
			if (!AllocateBuffer(uint64_t(e.width) * e.height * t.bytes_per_pixel, true, t.buffer, t.allocation, &t.data)) {
				for (auto& f : targets) if (f.buffer) vmaDestroyBuffer(graphics.allocator, f.buffer, f.allocation);
				return;
			}
		}
		auto recorder = command.Handle();
		for (auto& t : targets) {
			auto& owned = *t.image;
			ImageBarrier(recorder, owned.resource.image, owned.layout, vk::ImageLayout::eTransferSrcOptimal);
			vk::BufferImageCopy copy {};
			copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.imageExtent = {owned.resource.extent.width, owned.resource.extent.height, 1};
			recorder.copyImageToBuffer(owned.resource.image, vk::ImageLayout::eTransferSrcOptimal, t.buffer, 1, &copy);
			ImageBarrier(recorder, owned.resource.image, vk::ImageLayout::eTransferSrcOptimal, owned.layout);
		}
		vk::MemoryBarrier2 host {};
		host.srcStageMask = S::eTransfer; host.srcAccessMask = A::eTransferWrite;
		host.dstStageMask = S::eHost; host.dstAccessMask = A::eHostRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &host;
		recorder.pipelineBarrier2(dependency);
		context.GetCommandScheduler().Wait(last_tick);
		InvalidateRedundantBinds();
		const bool recorded_ok = last_recorded.load(std::memory_order_acquire);
		for (auto& t : targets) {
			const auto& e = t.image->resource.extent;
			const uint64_t pixels = uint64_t(e.width) * e.height;
			if (vmaInvalidateAllocation(graphics.allocator, t.allocation, 0, VK_WHOLE_SIZE) == VK_SUCCESS) {
				std::vector<float> luma(pixels);
				uint64_t nonfinite = 0;
				for (uint64_t i = 0; i < pixels; ++i) {
					float r = 0, g = 0, b = 0;
					if (t.bytes_per_pixel == 8) {
						const auto* h = static_cast<const uint16_t*>(t.data) + i * 4;
						r = Half(h[0]); g = Half(h[1]); b = Half(h[2]);
					} else if (t.image == &images[0]) {
						const uint32_t w = static_cast<const uint32_t*>(t.data)[i];
						r = SmallFloat(w & 0x7ff, 6); g = SmallFloat((w >> 11) & 0x7ff, 6); b = SmallFloat(w >> 22, 5);
					} else {
						const auto* h = static_cast<const uint16_t*>(t.data) + i * 2;
						r = std::fabs(Half(h[0])); g = std::fabs(Half(h[1]));
					}
					if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) ++nonfinite;
					luma[i] = t.image == &images[2] ? std::sqrt(r * r + g * g) : 0.2126f * r + 0.7152f * g + 0.0722f * b;
				}
				DumpStats(t.name, guest_frame, e.width, e.height, luma, nonfinite);
				char path[96];
				std::snprintf(path, sizeof(path), "fsr_dump_%u_%s.bin", guest_frame, t.name);
				if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
					std::fwrite(t.data, t.bytes_per_pixel, pixels, f);
					std::fclose(f);
				}
			}
			vmaDestroyBuffer(graphics.allocator, t.buffer, t.allocation);
		}
		std::printf("NHL27FSR3DUMP written frame=%u fsr_recorded=%d\n", guest_frame, recorded_ok ? 1 : 0);
		std::fflush(stdout);
	}
	static bool Fsr4Enabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_FSR4");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}
	static bool SoftResetEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_FSR4_SOFT_RESET");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}
	// Shared (bridge) image barrier with an optional queue-family ownership transfer.
	static void SharedBarrier(CommandRecorder recorder, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
							  uint32_t src_family, uint32_t dst_family) {
		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask = barrier.dstStageMask = S::eAllCommands;
		barrier.srcAccessMask = from == vk::ImageLayout::eUndefined ? vk::AccessFlags2 {} : A::eMemoryRead | A::eMemoryWrite;
		barrier.dstAccessMask = A::eMemoryRead | A::eMemoryWrite;
		barrier.oldLayout = from;
		barrier.newLayout = to;
		barrier.srcQueueFamilyIndex = src_family;
		barrier.dstQueueFamilyIndex = dst_family;
		barrier.image = image;
		barrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers = &barrier;
		recorder.pipelineBarrier2(dependency);
	}
	struct GuardSettings {
		bool on = false, show = false, erode = true, kick = true;
		float threshold = 0.04f, gain = 6.0f;
	};
	// KYTY_FSR4_GUARD=1 on; KYTY_FSR4_GUARD_SHOW=1 tints marked pixels red (debug);
	// KYTY_FSR4_GUARD_THRESHOLD / _GAIN shape the mask; KYTY_FSR4_GUARD_ERODE=0 keeps thin marks;
	// KYTY_FSR4_GUARD_KICK=0 leaves the motion vectors alone (mask only, which FSR 4.1 ignores).
	static const GuardSettings& Guard() {
		static const GuardSettings settings = [] {
			GuardSettings g;
			const char* on = std::getenv("KYTY_FSR4_GUARD");
			const char* show = std::getenv("KYTY_FSR4_GUARD_SHOW");
			const char* erode = std::getenv("KYTY_FSR4_GUARD_ERODE");
			const char* kick = std::getenv("KYTY_FSR4_GUARD_KICK");
			g.kick = kick == nullptr || kick[0] != '0';
			g.on = on != nullptr && on[0] == '1';
			g.show = g.on && show != nullptr && show[0] == '1';
			g.erode = erode == nullptr || erode[0] != '0';
			g.threshold = EnvFloat("KYTY_FSR4_GUARD_THRESHOLD", 0.04f);
			g.gain = EnvFloat("KYTY_FSR4_GUARD_GAIN", 6.0f);
			if (g.on) {
				std::printf("NHL27FSR4 guard on threshold=%g gain=%g erode=%d show=%d kick=%d\n", double(g.threshold),
							double(g.gain), g.erode ? 1 : 0, g.show ? 1 : 0, g.kick ? 1 : 0);
				std::fflush(stdout);
			}
			return g;
		}();
		return settings;
	}
	bool EnsureGuard(vk::Extent2D extent) {
		if (guard_failed) return false;
		if (!guard_pipeline) {
			const vk::PushConstantRange range {vk::ShaderStageFlagBits::eCompute, 0, 88};
			vk::PipelineLayoutCreateInfo layout {};
			layout.pushConstantRangeCount = 1;
			layout.pPushConstantRanges = &range;
			if (graphics.device.createPipelineLayout(&layout, nullptr, &guard_layout) != vk::Result::eSuccess) {
				guard_layout = nullptr;
				guard_failed = true;
				return false;
			}
			const auto module = CompileSPV(GPU_TILER_FSR_GHOST_GUARD_SPV, graphics.device);
			vk::ComputePipelineCreateInfo info {};
			info.layout = guard_layout;
			info.stage.stage = vk::ShaderStageFlagBits::eCompute;
			info.stage.module = module;
			info.stage.pName = "main";
			const auto result = graphics.device.createComputePipelines(nullptr, 1, &info, nullptr, &guard_pipeline);
			graphics.device.destroyShaderModule(module, nullptr);
			if (result != vk::Result::eSuccess) {
				guard_pipeline = nullptr;
				guard_failed = true;
				std::printf("NHL27FSR4 guard failed reason=pipeline\n");
				std::fflush(stdout);
				return false;
			}
		}
		if (guard_packed.buffer) return true;
		const uint64_t pixels = uint64_t(extent.width) * extent.height;
		bool ok = true;
		for (auto* g : {&guard_color[0], &guard_color[1], &guard_motion, &guard_raw, &guard_eroded})
			ok = ok && AllocateBuffer(pixels * 4, false, g->buffer, g->allocation);
		ok = ok && AllocateBuffer((pixels + 3) / 4 * 4, false, guard_packed.buffer, guard_packed.allocation);
		if (ok && Guard().show) ok = AllocateBuffer(pixels * 4, false, guard_show.buffer, guard_show.allocation);
		if (ok && Guard().kick)
			ok = AllocateBuffer(pixels * 4, false, guard_motion_out.buffer, guard_motion_out.allocation);
		if (!ok) {
			ReleaseGuard();
			guard_failed = true;
			std::printf("NHL27FSR4 guard failed reason=allocation\n");
			std::fflush(stdout);
			return false;
		}
		return true;
	}
	// Builds this frame's mask in guard_packed (R8, render size). False = no mask this frame.
	bool RecordGuard(CommandRecorder recorder, const Fsr3Frame& frame) {
		const vk::Extent2D extent = frame.render_size;
		if (!EnsureGuard(extent)) return false;
		const auto& settings = Guard();
		const uint32_t guest_frame = captured.Frame();
		const bool have_prev = guard_have_last && !frame.reset && guest_frame == guard_last_frame + 1;
		auto& cur = guard_color[guard_flip];
		auto& prev = guard_color[guard_flip ^ 1];
		const std::array<std::pair<Owned*, vk::Buffer>, 2> copies {{{&images[0], cur.buffer},
																	{&images[2], guard_motion.buffer}}};
		for (const auto& [owned, buffer] : copies) {
			ImageBarrier(recorder, owned->resource.image, owned->layout, vk::ImageLayout::eTransferSrcOptimal);
			vk::BufferImageCopy copy {};
			copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.imageExtent = vk::Extent3D {extent.width, extent.height, 1};
			recorder.copyImageToBuffer(owned->resource.image, vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &copy);
			ImageBarrier(recorder, owned->resource.image, vk::ImageLayout::eTransferSrcOptimal, owned->layout);
		}
		MemoryBarrier(recorder);
		struct Params {
			uint64_t cur, prev, motion, src, dst, show, motion_out;
			uint32_t width, height;
			float    mv_x, mv_y, threshold, gain;
			uint32_t mode, flags;
		};
		static_assert(sizeof(Params) == 88);
		const auto address = [this](vk::Buffer buffer) -> uint64_t {
			if (!buffer) return 0;
			vk::BufferDeviceAddressInfo info {};
			info.buffer = buffer;
			return graphics.device.getBufferAddress(info);
		};
		Params params {address(cur.buffer), address(prev.buffer), address(guard_motion.buffer), 0, 0,
					   address(guard_show.buffer), address(guard_motion_out.buffer), extent.width, extent.height,
					   frame.motion_scale_x, frame.motion_scale_y, settings.threshold, settings.gain, 0,
					   (have_prev ? 1u : 0u) | (settings.erode ? 2u : 0u) | (settings.show ? 4u : 0u) |
						   (settings.kick ? 8u : 0u)};
		g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		recorder.bindPipeline(vk::PipelineBindPoint::eCompute, guard_pipeline);
		InvalidateRedundantBinds();
		const uint32_t groups_x = (extent.width + 7) / 8, groups_y = (extent.height + 7) / 8;
		const auto pass = [&](uint32_t mode, vk::Buffer src, vk::Buffer dst, uint32_t x) {
			params.mode = mode;
			params.src = address(src);
			params.dst = address(dst);
			recorder.pushConstants(guard_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), &params);
			recorder.dispatch(x, groups_y, 1);
			MemoryBarrier(recorder);
		};
		pass(0, nullptr, guard_raw.buffer, groups_x);
		pass(1, guard_raw.buffer, guard_eroded.buffer, groups_x);
		pass(2, guard_eroded.buffer, guard_packed.buffer, (extent.width + 31) / 32); // 4 pixels per thread
		guard_flip ^= 1;
		guard_have_last = true;
		guard_last_frame = guest_frame;
		++guarded;
		return true;
	}
	// FSR 4 for this frame through the D3D12 bridge. Records: inputs -> shared images, then submits
	// (Vulkan signals v1), queues D3D12 (waits v1, FSR 4, signals v2), records the output copy into the
	// guest image and submits again waiting for v2. False = FSR 4 did not take the frame (use FSR 3.1).
	bool RecordFsr4(RenderContext& context, CommandBuffer& command, const TextureBinding& target,
					const Fsr3Frame& frame) {
		auto& scheduler = context.GetCommandScheduler();
		if (!bridge) {
			bridge = std::make_unique<Fsr4Bridge>(graphics);
			std::printf("NHL27FSR4 status=%s\n", bridge->Status().c_str());
			std::fflush(stdout);
		}
		if (!bridge->Usable()) return false;
		const vk::Extent2D render = frame.render_size, display = frame.output_size;
		if (bridge->NeedsRecreate(render, display)) {
			// Old shared images may still be used by queued Vulkan work.
			scheduler.Wait(scheduler.CurrentTick());
			InvalidateRedundantBinds();
		}
		if (!bridge->Ensure(render, display)) return false;
		const bool fresh = bridge->Fresh();
		bridge->ClearFresh();
		if (fresh) reactive_ready = false;
		const uint32_t family = graphics.queue_family;
		constexpr uint32_t ignored = VK_QUEUE_FAMILY_IGNORED, external = VK_QUEUE_FAMILY_EXTERNAL;
		auto recorder = command.Handle();
		MemoryBarrier(recorder);
		const bool guarded_frame = Guard().on && RecordGuard(recorder, frame);
		const bool show_mask = guarded_frame && Guard().show && guard_show.buffer;
		const bool kick_motion = guarded_frame && Guard().kick && guard_motion_out.buffer;
		for (int i = 0; i < 3; ++i) {
			auto& owned = images[i];
			const vk::Image shared = bridge->Image(i);
			ImageBarrier(recorder, owned.resource.image, owned.layout, vk::ImageLayout::eTransferSrcOptimal);
			SharedBarrier(recorder, shared, fresh ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral,
						  vk::ImageLayout::eGeneral, fresh ? ignored : external, fresh ? ignored : family);
			if ((i == Fsr4Bridge::Color && show_mask) || (i == Fsr4Bridge::Motion && kick_motion)) {
				vk::BufferImageCopy copy {};
				copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
				copy.imageExtent = vk::Extent3D {render.width, render.height, 1};
				recorder.copyBufferToImage(i == Fsr4Bridge::Color ? guard_show.buffer : guard_motion_out.buffer, shared,
										   vk::ImageLayout::eGeneral, 1, &copy);
			} else {
				vk::ImageCopy copy {};
				copy.srcSubresource = copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
				copy.extent = vk::Extent3D {render.width, render.height, 1};
				recorder.copyImage(owned.resource.image, vk::ImageLayout::eTransferSrcOptimal, shared,
								   vk::ImageLayout::eGeneral, 1, &copy);
			}
			SharedBarrier(recorder, shared, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, family, external);
			ImageBarrier(recorder, owned.resource.image, vk::ImageLayout::eTransferSrcOptimal, owned.layout);
		}
		if (guarded_frame) {
			const vk::Image shared = bridge->Image(Fsr4Bridge::Reactive);
			SharedBarrier(recorder, shared, reactive_ready ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
						  vk::ImageLayout::eGeneral, reactive_ready ? external : ignored,
						  reactive_ready ? family : ignored);
			vk::BufferImageCopy copy {};
			copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.imageExtent = vk::Extent3D {render.width, render.height, 1};
			recorder.copyBufferToImage(guard_packed.buffer, shared, vk::ImageLayout::eGeneral, 1, &copy);
			SharedBarrier(recorder, shared, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, family, external);
			reactive_ready = true;
		}
		if (fresh) {
			// The output belongs to D3D12 between frames.
			const vk::Image output = bridge->Image(Fsr4Bridge::Output);
			SharedBarrier(recorder, output, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, ignored, ignored);
			SharedBarrier(recorder, output, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, family, external);
		}
		const auto [inputs_ready, upscaled] = bridge->NextValues();
		SubmitInfo signal {};
		signal.AddSignal(bridge->Semaphore(), inputs_ready);
		const uint64_t flush_start = Now();
		scheduler.Flush(signal);
		cpu_ns_flush += Now() - flush_start;
		InvalidateRedundantBinds();
		Fsr4Params params {};
		params.jitter_x = frame.jitter_x;
		params.jitter_y = frame.jitter_y;
		params.motion_scale_x = frame.motion_scale_x;
		params.motion_scale_y = frame.motion_scale_y;
		params.near_plane = frame.near_plane;
		params.far_plane = frame.far_plane;
		params.vertical_fov = frame.vertical_fov;
		params.delta_ms = frame.delta_ms;
		params.pre_exposure = frame.pre_exposure;
		params.reset = frame.reset;
		params.reactive = guarded_frame;
		const uint64_t d3d_start = Now();
		const auto result = bridge->Dispatch(params, inputs_ready, upscaled);
		cpu_ns_d3d += Now() - d3d_start;
		if (result == Fsr4Bridge::Result::NotQueued) {
			++fsr4_failed;
			return false;
		}
		const bool ok = result == Fsr4Bridge::Result::Upscaled;
		last_recorded.store(ok, std::memory_order_release);
		auto after = command.Handle();
		if (ok && PresentEnabled() && PresentTarget(context, target)) {
			const auto& guest = context.GetTextureCache().GetImage(target.image_id);
			const vk::Image output = bridge->Image(Fsr4Bridge::Output);
			SharedBarrier(after, output, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, external, family);
			SharedBarrier(after, output, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, ignored,
						  ignored);
			ImageBarrier(after, guest.backing.image, target.layout, vk::ImageLayout::eTransferDstOptimal);
			vk::ImageCopy copy {};
			copy.srcSubresource = copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.extent = vk::Extent3D {display.width, display.height, 1};
			after.copyImage(output, vk::ImageLayout::eTransferSrcOptimal, guest.backing.image,
							vk::ImageLayout::eTransferDstOptimal, 1, &copy);
			ImageBarrier(after, guest.backing.image, vk::ImageLayout::eTransferDstOptimal, target.layout);
			SharedBarrier(after, output, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral, ignored,
						  ignored);
			SharedBarrier(after, output, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, family, external);
			const auto count = presented.fetch_add(1, std::memory_order_relaxed) + 1;
			if (count <= 3 || count % 600 == 0) {
				std::printf("NHL27FSR4 status=presented count=%llu\n", static_cast<unsigned long long>(count));
				std::fflush(stdout);
			}
		} else if (ok && PresentEnabled() && (++present_skips <= 8 || present_skips % 600 == 0)) {
			std::printf("NHL27FSR4 status=present-skip reason=guest-output-layout count=%llu\n",
						static_cast<unsigned long long>(present_skips));
			std::fflush(stdout);
		}
		SubmitInfo wait {};
		wait.AddWait(bridge->Semaphore(), upscaled);
		const uint64_t wait_flush_start = Now();
		scheduler.Flush(wait);
		cpu_ns_flush += Now() - wait_flush_start;
		InvalidateRedundantBinds();
		if (!ok) ++fsr4_failed;
		++fsr4_frames;
		return true;
	}
	bool AccumulationEdges(const std::vector<TextureBinding>& b) const {
		return stage == 3 && b.size() == 8 && reactive[6].Matches(b[0]) && prepare[4].Matches(b[1]) &&
			prepare[3].Matches(b[4]) && reactive[7].Matches(b[5]) &&
			b[7].desc.info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
			b[7].desc.info.extent == vk::Extent3D {captured.OutputWidth(), captured.OutputHeight(), 1};
	}
	bool PresentTarget(RenderContext& context, const TextureBinding& target) const {
		const auto& guest = context.GetTextureCache().GetImage(target.image_id);
		return target.image_id && target.desc.view_info.base_level == 0 &&
			target.desc.view_info.base_layer == 0 && guest.backing.image &&
			guest.backing.samples == 1 && guest.backing.format == vk::Format::eR16G16B16A16Sfloat &&
			guest.backing.extent == vk::Extent3D {captured.OutputWidth(), captured.OutputHeight(), 1} &&
			(guest.backing.usage & vk::ImageUsageFlagBits::eTransferDst) &&
			target.layout != vk::ImageLayout::eUndefined;
	}
	bool EnsureImages(RenderContext& context, const Nhl27Fsr3Constants& c) {
		const vk::Extent2D input {c.Width(), c.Height()}, output {c.OutputWidth(), c.OutputHeight()};
		if (images[0].resource.image && images[0].resource.extent == input &&
			images[3].resource.extent == output) return true;
		// Previous FSR work and its SDK callbacks must finish before the images go.
		if (images[0].resource.image) context.GetCommandScheduler().Wait(context.GetCommandScheduler().CurrentTick());
		ReleaseImages();
		constexpr std::array<vk::Format, 7> formats {
			vk::Format::eB10G11R11UfloatPack32, vk::Format::eR32Sfloat,
			vk::Format::eR16G16Sfloat, vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR32Sfloat, vk::Format::eR16G16Sfloat, vk::Format::eR32Uint};
		for (size_t i = 0; i < images.size(); ++i) {
			auto& owned = images[i];
			owned.resource.format = formats[i];
			owned.resource.extent = i == 3 ? output : input;
			vk::ImageCreateInfo info {};
			info.imageType = vk::ImageType::e2D;
			info.format = formats[i];
			info.extent = {owned.resource.extent.width, owned.resource.extent.height, 1};
			info.mipLevels = info.arrayLayers = 1;
			info.samples = vk::SampleCountFlagBits::e1;
			info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
			VmaAllocationCreateInfo alloc {};
			alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			VkImage native = VK_NULL_HANDLE;
			if (vmaCreateImage(graphics.allocator, reinterpret_cast<const VkImageCreateInfo*>(&info),
							   &alloc, &native, &owned.allocation, nullptr) != VK_SUCCESS) {
				ReleaseImages();
				return false;
			}
			owned.resource.image = native;
		}
		if (!AllocateBuffer(uint64_t(input.width) * input.height * 4, false, transfer, transfer_alloc)) {
			ReleaseImages();
			return false;
		}
		if (CleanColorEnabled() && EnsureCleanPipeline() &&
			!AllocateBuffer(uint64_t(input.width) * input.height * 4, false, clean, clean_alloc)) {
			clean = nullptr;
			clean_alloc = nullptr;
		}
		fsr = std::make_unique<Fsr3>(graphics.physical_device, graphics.device);
		return true;
	}
	static bool CleanColorEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_FSR3_CLEAN_COLOR");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}
	static float EnvFloat(const char* name, float fallback) {
		const char* value = std::getenv(name);
		return value != nullptr && value[0] != '\0' ? std::strtof(value, nullptr) : fallback;
	}
	bool EnsureCleanPipeline() {
		if (clean_pipeline) return true;
		const vk::PushConstantRange range {vk::ShaderStageFlagBits::eCompute, 0, 40};
		vk::PipelineLayoutCreateInfo layout {};
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges = &range;
		if (graphics.device.createPipelineLayout(&layout, nullptr, &clean_layout) != vk::Result::eSuccess) {
			clean_layout = nullptr;
			return false;
		}
		const auto module = CompileSPV(GPU_TILER_FSR_COLOR_CLEAN_SPV, graphics.device);
		vk::ComputePipelineCreateInfo info {};
		info.layout = clean_layout;
		info.stage.stage = vk::ShaderStageFlagBits::eCompute;
		info.stage.module = module;
		info.stage.pName = "main";
		const auto result = graphics.device.createComputePipelines(nullptr, 1, &info, nullptr, &clean_pipeline);
		graphics.device.destroyShaderModule(module, nullptr);
		if (result != vk::Result::eSuccess) {
			clean_pipeline = nullptr;
			return false;
		}
		std::printf("NHL27FSR3 clean-color on ratio=%g floor=%g cap=%g\n",
					double(EnvFloat("KYTY_FSR3_CLEAN_RATIO", 8.0f)), double(EnvFloat("KYTY_FSR3_CLEAN_FLOOR", 1.0f)),
					double(EnvFloat("KYTY_FSR3_CLEAN_CAP", 1024.0f)));
		std::fflush(stdout);
		return true;
	}
	// transfer (as copied from the guest colour) -> clean, one thread per pixel.
	void RecordClean(CommandRecorder recorder, vk::Extent2D extent) {
		struct Params {
			uint64_t src, dst;
			uint32_t width, height;
			float    ratio, floor_luma, cap;
			uint32_t reserved;
		};
		static_assert(sizeof(Params) == 40);
		static const float ratio = EnvFloat("KYTY_FSR3_CLEAN_RATIO", 8.0f);
		static const float floor_luma = EnvFloat("KYTY_FSR3_CLEAN_FLOOR", 1.0f);
		static const float cap = EnvFloat("KYTY_FSR3_CLEAN_CAP", 1024.0f);
		vk::BufferDeviceAddressInfo source_info {}, clean_info {};
		source_info.buffer = transfer;
		clean_info.buffer = clean;
		const Params params {graphics.device.getBufferAddress(source_info), graphics.device.getBufferAddress(clean_info),
							 extent.width, extent.height, ratio, floor_luma, cap, 0};
		g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		recorder.bindPipeline(vk::PipelineBindPoint::eCompute, clean_pipeline);
		InvalidateRedundantBinds();
		recorder.pushConstants(clean_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), &params);
		recorder.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
		++cleaned;
	}
	bool Snapshot(RenderContext& context, CommandBuffer& command, const TextureBinding& binding,
				  Owned& owned, bool clean_color = false) {
		if (!binding.image_id || binding.desc.view_info.base_level != 0 ||
			binding.desc.view_info.base_layer != 0) return false;
		auto& source = context.GetTextureCache().GetImage(binding.image_id);
		if (!source.backing.image || source.backing.samples != 1 ||
			!(source.backing.usage & vk::ImageUsageFlagBits::eTransferSrc) ||
			binding.layout == vk::ImageLayout::eUndefined ||
			source.backing.extent != vk::Extent3D {owned.resource.extent.width,
				owned.resource.extent.height, 1}) return false;
		auto full = vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor), aspect = full;
		const auto format = source.backing.format;
		if (format == vk::Format::eD32SfloatS8Uint) {
			full = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
			aspect = vk::ImageAspectFlagBits::eDepth;
		} else if (format == vk::Format::eD32Sfloat) {
			full = aspect = vk::ImageAspectFlagBits::eDepth;
		} else if (format != owned.resource.format &&
			!(owned.resource.format == vk::Format::eR32Sfloat && format == vk::Format::eR32Uint)) {
			return false;
		}
		if (aspect != vk::ImageAspectFlagBits::eColor && owned.resource.format != vk::Format::eR32Sfloat)
			return false;
		auto recorder = command.Handle();
		MemoryBarrier(recorder);
		ImageBarrier(recorder, source.backing.image, binding.layout,
					 vk::ImageLayout::eTransferSrcOptimal, full);
		ImageBarrier(recorder, owned.resource.image, owned.layout, vk::ImageLayout::eTransferDstOptimal);
		vk::BufferImageCopy copy {};
		copy.imageExtent = source.backing.extent;
		copy.imageSubresource = {aspect, 0, 0, 1};
		recorder.copyImageToBuffer(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
								   transfer, 1, &copy);
		MemoryBarrier(recorder);
		vk::Buffer upload = transfer;
		if (clean_color && clean && clean_pipeline && format == vk::Format::eB10G11R11UfloatPack32) {
			RecordClean(recorder, owned.resource.extent);
			MemoryBarrier(recorder);
			upload = clean;
		}
		copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		recorder.copyBufferToImage(upload, owned.resource.image, vk::ImageLayout::eTransferDstOptimal,
								   1, &copy);
		ImageBarrier(recorder, source.backing.image, vk::ImageLayout::eTransferSrcOptimal, binding.layout, full);
		ImageBarrier(recorder, owned.resource.image, vk::ImageLayout::eTransferDstOptimal,
					 vk::ImageLayout::eShaderReadOnlyOptimal);
		owned.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
		MemoryBarrier(recorder);
		return true;
	}
};
Nhl27Fsr3::Nhl27Fsr3(GraphicContext& graphics) : m(std::make_unique<Impl>(graphics)) {}
Nhl27Fsr3::~Nhl27Fsr3() = default;
bool Nhl27Fsr3::HistoryEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FSR3_NHL27_HISTORY");
		return !value || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}
bool Nhl27Fsr3::PresentEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FSR3_NHL27_PRESENT");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}
bool Nhl27Fsr3::VerifyEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FSR3_NHL27_VERIFY");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}
bool Nhl27Fsr3::SkipGuestEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FSR3_NHL27_SKIP_GUEST");
		return PresentEnabled() && (!value || std::strcmp(value, "0") != 0);
	}();
	return enabled;
}
bool Nhl27Fsr3::ReplacesDispatch(RenderContext& context, CommandBuffer& command, uint64_t hash,
								 const PreparedBindings& bindings) {
	// Only the accumulation output reaches later passes, and the present copy overwrites it, so
	// the guest dispatch is skipped once FSR has recorded a frame. Its own history goes stale,
	// which only matters if FSR stops; the copy then stops too and the guest restarts cold.
	// KYTY_FSR3_NHL27_SKIP_CHAIN=1 (test): also drop the game's three earlier FSR 2 passes (prepare,
	// luma, reactivity). Their outputs only feed the accumulation pass, which FSR replaces; FSR reads
	// the prepare pass's inputs (colour, depth, motion), not its outputs.
	static const bool skip_chain = [] {
		const char* value = std::getenv("KYTY_FSR3_NHL27_SKIP_CHAIN");
		return value != nullptr && value[0] == '1';
	}();
	if (skip_chain && (hash == kPass[0] || hash == kPass[1] || hash == kPass[2]) && Enabled() &&
		SkipGuestEnabled() && PresentEnabled() && !m->failed.load(std::memory_order_acquire) &&
		m->last_recorded.load(std::memory_order_acquire)) {
		// The later passes are dropped only while the chain is on track (OnDispatch advances stage).
		const unsigned expected = hash == kPass[0] ? 0u : hash == kPass[1] ? 1u : 2u;
		if (hash == kPass[0] || m->stage == expected) {
			++m->chain_skipped;
			return true;
		}
	}
	if (hash != kPass[3] || !Enabled() || !SkipGuestEnabled() ||
		m->failed.load(std::memory_order_acquire) || !m->last_recorded.load(std::memory_order_acquire) ||
		!m->AccumulationEdges(bindings.images) || !m->PresentTarget(context, bindings.images[7])) return false;
	// OnDispatch must not reject after this, so its constant check runs now.
	m->end = m->ReadConstants(context, command, bindings);
	if (!m->end || !m->captured.SameFrame(*m->end)) return false;
	++m->replaced;
	return true;
}
bool Nhl27Fsr3::Enabled() {
#if KYTY_ENABLE_FSR3
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FSR3_NHL27_SHADOW");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
#else
	return false;
#endif
}
void Nhl27Fsr3::OnDispatch(RenderContext& context, CommandBuffer& command, uint64_t hash,
						  const PreparedBindings& bindings) {
	if (!Enabled() || m->failed.load(std::memory_order_acquire)) return;
	struct Timer {
		uint64_t& total;
		uint64_t start = Impl::Now();
		~Timer() { total += Impl::Now() - start; }
	} timer {m->cpu_ns_total};
	const auto& b = bindings.images;
	const auto count = b.size();
	if (hash == kPass[0]) {
		m->stage = 0;
		// One queue keeps the previous FSR work ahead of these snapshots; verify mode also waits
		// so a failed Record() is seen before the next frame.
		if (VerifyEnabled() && m->last_tick) context.GetCommandScheduler().Wait(m->last_tick);
		if (m->failed.load(std::memory_order_acquire)) return;
		m->last_words_valid = false;
		const auto constants = m->ReadConstants(context, command, bindings);
		if (!constants || count != 8) {
			// Why the game's frame was not taken (first 40 times): which check failed, and the values.
			if (++m->prepare_rejects <= 40) {
				const auto& w = m->last_words;
				const char* why = !m->last_words_valid ? "no-constants"
					: constants ? "slot-count" : Nhl27Fsr3Constants::RejectReason(w);
				const auto f = [&w](size_t i) { return double(std::bit_cast<float>(w[i])); };
				std::printf("NHL27FSR3GAME prepare-reject n=%llu why=%s slots=%zu size=%ux%u out=%ux%u prev=%ux%u "
							"depth=%g,%g,%g,%g jitter=%g,%g prev_jitter=%g,%g mv=%g,%g down=%g,%g dt=%g exp=%g w38=%g "
							"frame=%u buffers=%zu first_buffer_bytes=%llu\n",
							static_cast<unsigned long long>(m->prepare_rejects), why ? why : "accepted?", count, w[4],
							w[5], w[8], w[9], w[6], w[7], f(0), f(1), f(2), f(3), f(28), f(29), f(30), f(31), f(32),
							f(33), f(34), f(35), f(36), f(37), f(38), w[39], bindings.buffer_sources.size(),
							static_cast<unsigned long long>(bindings.buffer_sources.empty() ? 0 : bindings.buffer_sources[0].size));
				std::fflush(stdout);
			}
			m->Reject("prepare-constants-or-slots");
			return;
		}
		const auto& c = *constants;
		const vk::Extent3D extent {c.Width(), c.Height(), 1};
		if (b[0].desc.info.pixel_format != vk::Format::eR32Sfloat ||
			b[1].desc.info.pixel_format != vk::Format::eR16G16Sfloat ||
			b[3].desc.info.pixel_format != vk::Format::eB10G11R11UfloatPack32 ||
			b[0].desc.info.extent != extent || b[1].desc.info.extent != extent ||
			b[3].desc.info.extent != extent || !m->EnsureImages(context, c)) {
			m->Reject("prepare-layout-or-allocation"); return;
		}
		if (!m->Snapshot(context, command, b[3], m->images[0], true) ||
			!m->Snapshot(context, command, b[0], m->images[1]) ||
			!m->Snapshot(context, command, b[1], m->images[2])) {
			m->Reject("snapshot-layout"); return;
		}
		m->captured = c;
		for (size_t i = 0; i < 8; ++i) m->prepare[i] = Impl::Role::Of(b[i]);
		m->stage = 1;
		m->LogBindings(0, b);
	} else if (hash == kPass[1]) {
		if (m->stage != 1 || count != 2 || !m->prepare[7].Matches(b[0])) {
			m->Reject("luma-edge"); return;
		}
		m->luma = Impl::Role::Of(b[1]);
		m->stage = 2;
		m->LogBindings(1, b);
	} else if (hash == kPass[2]) {
		if (m->stage != 2 || count != 8 || !m->prepare[5].Matches(b[0]) ||
			!m->prepare[4].Matches(b[1]) || !m->prepare[2].Matches(b[2]) || !m->luma.Matches(b[3])) {
			m->Reject("reactivity-edges"); return;
		}
		for (size_t i = 0; i < 8; ++i) m->reactive[i] = Impl::Role::Of(b[i]);
		m->stage = 3;
		m->LogBindings(2, b);
	} else if (hash == kPass[3]) {
		if (!m->AccumulationEdges(b)) { m->Reject("accumulation-edges"); return; }
		m->stage = 0;
		m->LogBindings(3, b);
		// ReplacesDispatch already read them when it skipped the guest dispatch.
		const auto end = m->end ? m->end : m->ReadConstants(context, command, bindings);
		m->end.reset();
		if (!end || !m->captured.SameFrame(*end)) { m->Reject("constant-frame-mismatch"); return; }
		const auto& c = m->captured;
		Fsr3Frame frame {};
		frame.color = m->images[0].resource;
		frame.depth = m->images[1].resource;
		frame.motion = m->images[2].resource;
		frame.output = m->images[3].resource;
		frame.dilated_depth = m->images[4].resource;
		frame.dilated_motion = m->images[5].resource;
		frame.previous_depth = m->images[6].resource;
		frame.render_size = {c.Width(), c.Height()};
		frame.output_size = {c.OutputWidth(), c.OutputHeight()};
		frame.frame_id = ++m->serial;
		frame.jitter_x = c.Float(28); frame.jitter_y = c.Float(29);
		frame.motion_scale_x = c.Float(32) * c.Width();
		frame.motion_scale_y = c.Float(33) * c.Height();
		frame.near_plane = c.Near(); frame.far_plane = c.Far();
		frame.vertical_fov = c.FovY();
		// The game reports a frame time of 0 on some frames; FSR needs a real one.
		if (c.Float(36) > 0) m->last_delta_ms = c.Float(36) * 1000;
		frame.delta_ms = m->last_delta_ms;
		frame.pre_exposure = c.Float(37);
		frame.sharpness = 0;
		frame.hdr = frame.inverted_depth = true;
		// Actual same-frame GPU resources/coefficients are verified above. Inputs were verified
		// live (jitter-free MVs, SDK jitter sign, scale, camera); history continuity is inferred
		// from the guest frame counter because the cb has no reset flag.
		const bool history = HistoryEnabled();
		const auto current = Nhl27HistoryState::From(c);
		const auto reason = ShouldResetHistory(m->have_last ? &m->last_state : nullptr, current,
											   m->missed, history);
		// A small lens change (the broadcast camera zooming a little every frame) is not a cut: the
		// motion vectors already describe it, so FSR keeps its history (Oct 19 match: ~270 needless
		// resets in a few minutes). KYTY_FSR_STRICT_CUTS=1 restores resetting on any change.
		// KYTY_FSR4_SOFT_RESET=1 also lets FSR 4 keep its history through a skipped frame or a jitter hiccup.
		static const bool strict_cuts = [] {
			const char* value = std::getenv("KYTY_FSR_STRICT_CUTS");
			return value != nullptr && value[0] == '1';
		}();
		bool fsr4_keep_history = false, small_camera_change = false;
		if (m->have_last) {
			const auto& last = m->last_state;
			const auto ratio = [](float a, float b) {
				a = std::abs(a);
				b = std::abs(b);
				return std::max(a, b) / std::max(std::min(a, b), 1e-6f);
			};
			const bool big_cut = ratio(current.fov, last.fov) > 1.15f ||
				ratio(current.near_plane, last.near_plane) > 2.0f || ratio(current.far_plane, last.far_plane) > 2.0f;
			small_camera_change = !strict_cuts && reason == Nhl27ResetReason::CameraCut && !big_cut;
			fsr4_keep_history = Impl::SoftResetEnabled() &&
				(reason == Nhl27ResetReason::MissedPass || reason == Nhl27ResetReason::JitterMismatch);
		}
		m->last_state = current;
		m->have_last = true;
		m->missed = false;
		frame.verified = true;
		frame.reset = reason != Nhl27ResetReason::None && !small_camera_change;
		if (small_camera_change) ++m->soft_skipped;
		if (frame.reset) ++m->reset_reasons[std::min<size_t>(size_t(reason), m->reset_reasons.size() - 1)];
		const char* reason_name = Nhl27ResetReasonName(reason);
		if (frame.reset && history && (++m->resets <= 16 || m->resets % 600 == 0)) {
			std::printf("NHL27FSR3GAME status=history-reset reason=%s guest_frame=%u resets=%llu\n",
						reason_name, c.Frame(), static_cast<unsigned long long>(m->resets));
			std::fflush(stdout);
		}
		const auto guest_frame = c.Frame();
		auto* state = m.get();
		state->last_recorded.store(false, std::memory_order_release);
		bool fsr4_done = false;
		if (Impl::Fsr4Enabled()) {
			// KYTY_FSR4_JITTER_PREV=1 (test): hand FSR 4 the block's previous-frame jitter (w30/w31).
			static const bool jitter_prev = [] {
				const char* value = std::getenv("KYTY_FSR4_JITTER_PREV");
				return value != nullptr && value[0] == '1';
			}();
			Fsr3Frame fsr4_frame = frame;
			if (fsr4_frame.reset && fsr4_keep_history) {
				fsr4_frame.reset = false;
				++m->soft_skipped;
			}
			if (jitter_prev) {
				fsr4_frame.jitter_x = c.Float(30);
				fsr4_frame.jitter_y = c.Float(31);
			}
			fsr4_done = m->RecordFsr4(context, command, b[7], fsr4_frame);
		}
		if (!fsr4_done) {
		auto recorder = command.Handle();
		for (size_t i = 3; i < m->images.size(); ++i) {
			auto& owned = m->images[i];
			ImageBarrier(recorder, owned.resource.image, owned.layout, vk::ImageLayout::eGeneral);
			owned.layout = vk::ImageLayout::eGeneral;
		}
		recorder.RecordExternal([state, frame, guest_frame, history, reason_name](vk::CommandBuffer native) {
			const bool ok = state->fsr->Record(native, frame);
			state->last_recorded.store(ok, std::memory_order_release);
			if (!ok) state->failed.store(true, std::memory_order_release);
			if (++state->recorded <= 3 || state->recorded % 600 == 0 || !ok) {
				std::printf("NHL27FSR3GAME status=%s mode=%s guest_frame=%u reset=%s reason=%s "
					"render=%ux%u display=%ux%u detail=%s\n", ok ? "recorded" : "failed",
					history ? "history-shadow" : "reset-shadow", guest_frame,
					frame.reset ? "yes" : "no", reason_name, frame.render_size.width, frame.render_size.height,
					frame.output_size.width, frame.output_size.height, state->fsr->Status().c_str());
				std::fflush(stdout);
			}
		});
		if (PresentEnabled()) {
			// The guest accumulation dispatch (unless skipped) is already recorded; the next guest draw reads this
			// image and the one after reuses its memory as depth, so the copy must land here.
			const auto& target = b[7];
			const auto& guest = context.GetTextureCache().GetImage(target.image_id);
			const vk::Extent3D extent {frame.output_size.width, frame.output_size.height, 1};
			if (m->PresentTarget(context, target)) {
				const vk::Image source = m->images[3].resource.image, destination = guest.backing.image;
				const vk::ImageLayout layout = target.layout;
				recorder.RecordExternal([state, source, destination, layout, extent](vk::CommandBuffer native) {
					// Record() failing leaves the output undefined; keep the guest's own image then.
					if (!state->last_recorded.load(std::memory_order_acquire)) return;
					const auto barrier = [native](vk::Image image, vk::ImageLayout from, vk::ImageLayout to) {
						vk::ImageMemoryBarrier2 b {};
						b.srcStageMask = b.dstStageMask = S::eAllCommands;
						b.srcAccessMask = b.dstAccessMask = A::eMemoryRead | A::eMemoryWrite;
						b.oldLayout = from;
						b.newLayout = to;
						b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
						b.image = image;
						b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
						vk::DependencyInfo dependency {};
						dependency.imageMemoryBarrierCount = 1;
						dependency.pImageMemoryBarriers = &b;
						native.pipelineBarrier2(dependency);
					};
					barrier(source, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
					barrier(destination, layout, vk::ImageLayout::eTransferDstOptimal);
					vk::ImageCopy copy {};
					copy.srcSubresource = copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
					copy.extent = extent;
					native.copyImage(source, vk::ImageLayout::eTransferSrcOptimal, destination,
									 vk::ImageLayout::eTransferDstOptimal, 1, &copy);
					barrier(source, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
					barrier(destination, vk::ImageLayout::eTransferDstOptimal, layout);
					const auto count = state->presented.fetch_add(1, std::memory_order_relaxed) + 1;
					if (count <= 3 || count % 600 == 0) {
						std::printf("NHL27FSR3GAME status=presented count=%llu\n",
									static_cast<unsigned long long>(count));
						std::fflush(stdout);
					}
				});
			} else if (++m->present_skips <= 8 || m->present_skips % 600 == 0) {
				std::printf("NHL27FSR3GAME status=present-skip reason=guest-output-layout count=%llu\n",
							static_cast<unsigned long long>(m->present_skips));
				std::fflush(stdout);
			}
		}
		} // !fsr4_done
		InvalidateRedundantBinds();
		g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		NoteShaderWork();
		m->last_tick = context.GetCommandScheduler().CurrentTick();
		if (VerifyEnabled() && (m->serial <= 3 || m->serial % 600 == 0))
			m->CheckOutput(context, command, guest_frame, history);
		if (const auto every = Impl::DumpEvery(); every != 0 && m->serial % every == 0 && m->dumps < 4) {
			++m->dumps;
			m->DumpFrame(context, command, guest_frame);
		}
		if (m->serial % 600 == 0) {
			std::printf("NHL27FSR3GAME status=stats frames=%llu dirty_reads=%llu mismatches=%llu "
						"replaced=%llu cleaned=%llu fsr4_frames=%llu fsr4_failed=%llu guarded=%llu "
						"resets=%llu cut=%llu missed=%llu counter=%llu jitter=%llu soft_skipped=%llu cpu_us_per_frame=%.0f "
						"flush_us=%.0f d3d_us=%.0f chain_skipped=%llu\n", static_cast<unsigned long long>(m->serial),
						static_cast<unsigned long long>(m->dirty_reads),
						static_cast<unsigned long long>(m->clean_mismatches),
						static_cast<unsigned long long>(m->replaced),
						static_cast<unsigned long long>(m->cleaned),
						static_cast<unsigned long long>(m->fsr4_frames),
						static_cast<unsigned long long>(m->fsr4_failed),
						static_cast<unsigned long long>(m->guarded), static_cast<unsigned long long>(m->resets),
						static_cast<unsigned long long>(m->reset_reasons[size_t(Nhl27ResetReason::CameraCut)]),
						static_cast<unsigned long long>(m->reset_reasons[size_t(Nhl27ResetReason::MissedPass)]),
						static_cast<unsigned long long>(m->reset_reasons[size_t(Nhl27ResetReason::FrameDiscontinuity)]),
						static_cast<unsigned long long>(m->reset_reasons[size_t(Nhl27ResetReason::JitterMismatch)]),
						static_cast<unsigned long long>(m->soft_skipped), double(m->cpu_ns_total) / 600e3,
						double(m->cpu_ns_flush) / 600e3, double(m->cpu_ns_d3d) / 600e3,
						static_cast<unsigned long long>(m->chain_skipped));
			m->cpu_ns_total = m->cpu_ns_flush = m->cpu_ns_d3d = 0;
			std::fflush(stdout);
		}
	}
}
} // namespace Libs::Graphics
