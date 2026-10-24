#include "graphics/presentation/fsr/fsr3.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#if KYTY_ENABLE_FSR3
#include "graphics/presentation/fsr/fsr3Spirv.h"
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>
#endif

namespace Libs::Graphics {
#if KYTY_ENABLE_FSR3
namespace {
VKAPI_ATTR VkResult VKAPI_CALL CreateFsrShaderModule(VkDevice device,
													 const VkShaderModuleCreateInfo* info,
													 const VkAllocationCallbacks* allocation,
													 VkShaderModule* module) {
	auto patched = FixFsr3LumaHistorySpirv({info->pCode, info->codeSize / sizeof(uint32_t)});
	if (patched.empty())
		return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule(device, info, allocation, module);
	auto create = *info;
	create.pCode = patched.data();
	create.codeSize = patched.size() * sizeof(uint32_t);
	std::printf("NHL27FSR3 fix=luma-history-format rgba8-to-rgba16f\n");
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule(device, &create, allocation, module);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL FsrDeviceProcAddr(VkDevice device, const char* name) {
	if (std::strcmp(name, "vkCreateShaderModule") == 0)
		return reinterpret_cast<PFN_vkVoidFunction>(CreateFsrShaderModule);
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, name);
}
// KYTY_FSR3_TUNE="vf=0,rs=1,sc=1,ap=0.333,md=-0.333" overrides the FSR 3.1.4 tuning constants
// (velocity factor, reactiveness scale, shading change scale, accumulation added per frame,
// min disocclusion accumulation). Unlisted keys keep the SDK defaults.
void ApplyTuning(FfxFsr3UpscalerContext& context) {
	const char* env = std::getenv("KYTY_FSR3_TUNE");
	if (env == nullptr) return;
	constexpr std::array<std::pair<const char*, FfxFsr3UpscalerConfigureKey>, 5> keys {{
		{"vf", FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR},
		{"rs", FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FREACTIVENESSSCALE},
		{"sc", FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE},
		{"ap", FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FACCUMULATIONADDEDPERFRAME},
		{"md", FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION}}};
	for (const char* p = env; *p;) {
		const char* eq = std::strchr(p, '=');
		if (!eq) break;
		char* end = nullptr;
		float value = std::strtof(eq + 1, &end);
		for (const auto& [name, key] : keys)
			if (size_t(eq - p) == std::strlen(name) && std::strncmp(p, name, eq - p) == 0) {
				const auto result = ffxFsr3UpscalerSetConstant(&context, key, &value);
				std::printf("NHL27FSR3 tune %s=%g result=%d\n", name, value, int(result));
			}
		p = (end && *end == ',') ? end + 1 : (end ? end : eq + 1);
		if (end && *end != ',' ) break;
	}
	std::fflush(stdout);
}
} // namespace
#endif
struct Fsr3::Impl {
	vk::PhysicalDevice physical;
	vk::Device device;
	std::string status = "FSR 3.1 waiting for verified temporal inputs";
	bool history = false, created = false, failed = false;
	uint64_t last_frame = 0;
	vk::Extent2D input {}, output {};
	uint32_t flags = 0;
#if KYTY_ENABLE_FSR3
	FfxFsr3UpscalerContext context {};
	std::vector<uint8_t> scratch;
	bool Check(FfxErrorCode result, const char* operation) {
		if (result == FFX_OK)
			return true;
		status = fmt::format("FSR 3.1 {} failed ({})", operation, int(result));
		failed = true;
		history = false;
		return false;
	}
	static FfxResource Resource(const Fsr3Image& image, bool writable) {
		FfxResourceDescription desc {};
		desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
		desc.format = ffxGetSurfaceFormatVK(static_cast<VkFormat>(image.format));
		desc.width = image.extent.width;
		desc.height = image.extent.height;
		desc.depth = desc.mipCount = 1;
		desc.usage = writable ? FFX_RESOURCE_USAGE_UAV : FFX_RESOURCE_USAGE_READ_ONLY;
		return ffxGetResourceVK(
			reinterpret_cast<void*>(static_cast<VkImage>(image.image)), desc, L"Kyty FSR3",
			writable ? FFX_RESOURCE_STATE_UNORDERED_ACCESS : FFX_RESOURCE_STATE_COMPUTE_READ);
	}
#endif
};

Fsr3::Fsr3(vk::PhysicalDevice physical, vk::Device device) : m(std::make_unique<Impl>()) {
	m->physical = physical;
	m->device = device;
}
Fsr3::~Fsr3() {
#if KYTY_ENABLE_FSR3
	if (m->created)
		(void)ffxFsr3UpscalerContextDestroy(&m->context);
#endif
}
void Fsr3::ResetHistory() { m->history = false; }
const std::string& Fsr3::Status() const { return m->status; }

bool Fsr3::Record(vk::CommandBuffer command, const Fsr3Frame& frame) {
#if KYTY_ENABLE_FSR3
	if (m->failed)
		return false;
	auto reject = [&](const char* why) {
		m->status = std::string("FSR 3.1 rejected: ") + why;
		m->history = false;
		return false;
	};
	if (!frame.verified || !command || !frame.frame_id)
		return reject("unverified frame");
	const auto input = frame.render_size, output = frame.output_size;
	if (!input.width || !input.height || input.width > output.width || input.height > output.height)
		return reject("invalid extents");
	const std::array<const Fsr3Image*, 7> images {
		&frame.color,		  &frame.depth,			 &frame.motion,		   &frame.output,
		&frame.dilated_depth, &frame.dilated_motion, &frame.previous_depth};
	for (size_t i = 0; i < images.size(); ++i) {
		if (!images[i]->image || images[i]->extent != (i == 3 ? output : input))
			return reject("missing or mismatched resource");
		for (size_t j = 0; j < i; ++j)
			if (images[i]->image == images[j]->image)
				return reject("aliased resource");
	}
	if (frame.depth.format != vk::Format::eR32Sfloat ||
		frame.motion.format != vk::Format::eR16G16Sfloat ||
		frame.dilated_depth.format != vk::Format::eR32Sfloat ||
		frame.dilated_motion.format != vk::Format::eR16G16Sfloat ||
		frame.previous_depth.format != vk::Format::eR32Uint)
		return reject("unsupported depth or motion format");
	if (frame.hdr) {
		const char* enabled = std::getenv("KYTY_FSR3_HDR");
		if (!enabled || std::strcmp(enabled, "1") != 0)
			return reject("HDR requires experimental KYTY_FSR3_HDR=1");
		if (frame.output.format != vk::Format::eR16G16B16A16Sfloat ||
			(frame.color.format != vk::Format::eB10G11R11UfloatPack32 &&
			 frame.color.format != vk::Format::eR16G16B16A16Sfloat))
			return reject("HDR requires float color and FP16 output");
	} else if (frame.output.format != vk::Format::eR8G8B8A8Unorm ||
			   (frame.color.format != vk::Format::eR8G8B8A8Unorm &&
				frame.color.format != vk::Format::eB8G8R8A8Unorm)) {
		return reject("unsupported SDR format");
	}
	const std::array values {frame.jitter_x,	   frame.jitter_y,	 frame.motion_scale_x,
							 frame.motion_scale_y, frame.near_plane, frame.far_plane,
							 frame.vertical_fov,   frame.delta_ms,	 frame.pre_exposure,
							 frame.sharpness};
	for (const auto value : values)
		if (!std::isfinite(value))
			return reject("nonfinite camera");
	if (frame.motion_scale_x == 0 || frame.motion_scale_y == 0 || frame.near_plane <= 0 ||
		(!frame.infinite_depth && frame.far_plane <= frame.near_plane) || frame.vertical_fov <= 0 ||
		frame.vertical_fov >= 3.141593f || frame.delta_ms <= 0 || frame.pre_exposure <= 0 ||
		frame.sharpness < 0 || frame.sharpness > 1)
		return reject("invalid camera or motion scale");
	if (m->created && frame.frame_id <= m->last_frame)
		return reject("repeated or out-of-order frame");
	const uint32_t flags =
		FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE |
		(frame.hdr ? FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE : 0) |
		(frame.inverted_depth ? FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED : 0) |
		(frame.infinite_depth ? FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE : 0) |
		(frame.jittered_motion ? FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0);
	if (m->created && (m->input != input || m->output != output || m->flags != flags))
		return reject("configuration changed; recreate after GPU completion");
	if (!m->created) {
		if (ffxFsr3UpscalerGetEffectVersion() != FFX_SDK_MAKE_VERSION(3, 1, 4))
			return reject("SDK version mismatch; expected upscaler 3.1.4");
		const char* fix = std::getenv("KYTY_FSR3_LUMA_FORMAT_FIX");
		if (!fix || std::strcmp(fix, "1") != 0)
			return reject("SDK 1.1.4 requires experimental KYTY_FSR3_LUMA_FORMAT_FIX=1");
		VkDeviceContext device {m->device, m->physical,
								VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr};
		device.vkDeviceProcAddr = FsrDeviceProcAddr;
		m->scratch.resize(ffxGetScratchMemorySizeVK(m->physical, FFX_FSR3UPSCALER_CONTEXT_COUNT));
		FfxFsr3UpscalerContextDescription description {};
		description.flags = flags;
		description.maxRenderSize = {input.width, input.height};
		description.maxUpscaleSize = {output.width, output.height};
		if (!m->Check(ffxGetInterfaceVK(&description.backendInterface, ffxGetDeviceVK(&device),
										m->scratch.data(), m->scratch.size(),
										FFX_FSR3UPSCALER_CONTEXT_COUNT),
					  "backend"))
			return false;
		if (!m->Check(ffxFsr3UpscalerContextCreate(&m->context, &description), "create"))
			return false;
		m->created = true;
		ApplyTuning(m->context);
		m->input = input;
		m->output = output;
		m->flags = flags;
	}
	FfxFsr3UpscalerDispatchDescription dispatch {};
	dispatch.commandList = ffxGetCommandListVK(command);
	dispatch.color = Impl::Resource(frame.color, false);
	dispatch.depth = Impl::Resource(frame.depth, false);
	dispatch.motionVectors = Impl::Resource(frame.motion, false);
	dispatch.output = Impl::Resource(frame.output, true);
	dispatch.dilatedDepth = Impl::Resource(frame.dilated_depth, true);
	dispatch.dilatedMotionVectors = Impl::Resource(frame.dilated_motion, true);
	dispatch.reconstructedPrevNearestDepth = Impl::Resource(frame.previous_depth, true);
	dispatch.jitterOffset = {frame.jitter_x, frame.jitter_y};
	dispatch.motionVectorScale = {frame.motion_scale_x, frame.motion_scale_y};
	dispatch.renderSize = {input.width, input.height};
	dispatch.upscaleSize = {output.width, output.height};
	dispatch.enableSharpening = frame.sharpness > 0;
	dispatch.sharpness = frame.sharpness;
	dispatch.frameTimeDelta = frame.delta_ms;
	dispatch.preExposure = frame.pre_exposure;
	dispatch.reset = frame.reset || !m->history || frame.frame_id != m->last_frame + 1;
	const float far_plane =
		frame.infinite_depth ? std::numeric_limits<float>::max() : frame.far_plane;
	dispatch.cameraNear = frame.inverted_depth ? far_plane : frame.near_plane;
	dispatch.cameraFar = frame.inverted_depth ? frame.near_plane : far_plane;
	dispatch.cameraFovAngleVertical = frame.vertical_fov;
	dispatch.viewSpaceToMetersFactor = 1.0f;
	if (!m->Check(ffxFsr3UpscalerContextDispatch(&m->context, &dispatch), "dispatch"))
		return false;
	m->history = true;
	m->last_frame = frame.frame_id;
	m->status = frame.hdr ? "FSR 3.1.4 Super Resolution (HDR)" : "FSR 3.1.4 Super Resolution";
	return true;
#else
	m->status = "FSR 3.1 unavailable: build with KYTY_ENABLE_FSR3";
	return false;
#endif
}
} // namespace Libs::Graphics
