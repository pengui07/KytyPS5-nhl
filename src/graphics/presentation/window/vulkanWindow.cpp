#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "common/experiments.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/fsr/nhl27Fsr3.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/controller.h"
#include "loader/systemContent.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

#define KYTY_ENABLE_DEBUG_PRINTF

namespace Libs::Graphics {

struct VulkanExtensions {
	bool enable_validation_layers = false;

	std::vector<const char*>             required_extensions;
	std::vector<vk::ExtensionProperties> available_extensions;
	std::vector<const char*>             required_layers;
	std::vector<vk::LayerProperties>     available_layers;
};

vk::PhysicalDeviceVulkan11Features WindowContext::RequiredVulkan11Features() noexcept {
	vk::PhysicalDeviceVulkan11Features features {};
	features.storageBuffer16BitAccess = VK_TRUE;
	return features;
}

vk::PhysicalDeviceVulkan12Features WindowContext::RequiredVulkan12Features() noexcept {
	vk::PhysicalDeviceVulkan12Features features {};
	features.samplerMirrorClampToEdge  = VK_TRUE;
	features.timelineSemaphore         = VK_TRUE;
	features.shaderOutputLayer         = VK_TRUE;
	features.shaderOutputViewportIndex = VK_TRUE;
	features.bufferDeviceAddress       = VK_TRUE;
	features.shaderBufferInt64Atomics  = VK_TRUE;
	features.storageBuffer8BitAccess   = VK_TRUE;
	features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	features.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
	features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
	features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
	features.descriptorBindingUpdateUnusedWhilePending    = VK_TRUE;
	features.descriptorBindingPartiallyBound              = VK_TRUE;
	return features;
}

vk::PhysicalDeviceVulkan13Features WindowContext::RequiredVulkan13Features() noexcept {
	vk::PhysicalDeviceVulkan13Features features {};
	features.dynamicRendering = VK_TRUE;
	features.synchronization2 = VK_TRUE;
	return features;
}

// RenderDoc compatibility: KYTY_RENDERDOC_COMPAT=1 forces it on, =0 forces off; otherwise it is
// auto-enabled when RenderDoc's Vulkan layer is requested for this process.
// KYTY_RDC_SKIP_EXT (comma list of extension names) overrides the default skip list.
static bool RenderDocCompat() {
	static const bool v = [] {
		const char* e = std::getenv("KYTY_RENDERDOC_COMPAT");
		if (e != nullptr && e[0] != '\0') return e[0] != '0';
		const char* l = std::getenv("VK_INSTANCE_LAYERS");
		return (l != nullptr && std::strstr(l, "RENDERDOC") != nullptr) ||
		       std::getenv("ENABLE_VULKAN_RENDERDOC_CAPTURE") != nullptr;
	}();
	return v;
}

static bool RenderDocSkipsExtension(const char* name) {
	if (!RenderDocCompat()) return false;
	const char* list = std::getenv("KYTY_RDC_SKIP_EXT");
	if (list == nullptr) {
		list = "VK_AMD_buffer_marker,VK_EXT_device_fault";
	}
	return std::strstr(list, name) != nullptr;
}

static bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const auto& ext) { return strcmp(ext.extensionName, name) == 0; });
}

static bool HasExtension(const std::vector<const char*>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const char* ext) { return strcmp(ext, name) == 0; });
}

static bool HasLayer(const std::vector<vk::LayerProperties>& layers, const char* name) {
	return std::any_of(layers.begin(), layers.end(),
	                   [name](const auto& layer) { return strcmp(layer.layerName, name) == 0; });
}

static void GetSurfaceCapabilities(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
                                   SurfaceCapabilities& r) {
	RequireVulkanSuccess(physical_device.getSurfaceCapabilitiesKHR(surface, &r.capabilities),
	                     "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

	r.formats = EnumerateVulkan<vk::SurfaceFormatKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfaceFormatsKHR", [&](uint32_t* count, vk::SurfaceFormatKHR* values) {
		    return physical_device.getSurfaceFormatsKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.formats.empty());

	r.present_modes = EnumerateVulkan<vk::PresentModeKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfacePresentModesKHR",
	    [&](uint32_t* count, vk::PresentModeKHR* values) {
		    return physical_device.getSurfacePresentModesKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.present_modes.empty());
}

static bool CheckFormat(vk::PhysicalDevice device, vk::Format format, bool tile,
                        vk::FormatFeatureFlags features) {
	vk::FormatProperties format_props {};
	device.getFormatProperties(format, &format_props);

	const auto supported_features =
	    (tile ? format_props.optimalTilingFeatures : format_props.linearTilingFeatures);
	return (supported_features & features) == features;
}

static uint32_t VulkanFindQueueFamily(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
	EXIT_IF(device == nullptr);
	EXIT_IF(surface == nullptr);

	uint32_t queue_family_count = 0;
	device.getQueueFamilyProperties(&queue_family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
	device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

	const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
	for (uint32_t family = 0; family < queue_family_count; family++) {
		const auto& properties             = queue_families[family];
		vk::Bool32  presentation_supported = VK_FALSE;
		RequireVulkanSuccess(device.getSurfaceSupportKHR(family, surface, &presentation_supported),
		                     "vkGetPhysicalDeviceSurfaceSupportKHR");

		LOGF("\tqueue family: %s [count = %u], [present = %s]\n",
		     vk::to_string(properties.queueFlags).c_str(), properties.queueCount,
		     (presentation_supported == VK_TRUE ? "true" : "false"));
		if (properties.queueCount != 0 && (properties.queueFlags & required) == required &&
		    presentation_supported == VK_TRUE) {
			LOGF("\tselected universal queue family %u\n", family);
			return family;
		}
	}
	return static_cast<uint32_t>(-1);
}

// On failure out_device is null and out_rejections says why each device was skipped.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void VulkanFindPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface,
                                     const std::vector<const char*>& device_extensions,
                                     SurfaceCapabilities&            out_capabilities,
                                     vk::PhysicalDevice& out_device, uint32_t& out_queue_family,
                                     std::string& out_rejections) {
	EXIT_IF(instance == nullptr);
	EXIT_IF(surface == nullptr);

	auto devices = EnumerateVulkan<vk::PhysicalDevice>(
	    "vkEnumeratePhysicalDevices", [&](uint32_t* count, vk::PhysicalDevice* values) {
		    return instance.enumeratePhysicalDevices(count, values);
	    });
	EXIT_NOT_IMPLEMENTED(devices.empty());

	if (Config::GetGpuIndex() >= 0) {
		if (static_cast<size_t>(Config::GetGpuIndex()) < devices.size()) {
			devices = {devices[Config::GetGpuIndex()]};
		} else {
			LOGF("Vulkan GPU index %d is unavailable; selecting automatically\n",
			     Config::GetGpuIndex());
		}
	} else if (vk::PhysicalDevice headset; Vr::OpenXrPhysicalDevice(instance, headset)) {
		devices = {headset}; // The headset runtime renders on this GPU.
	}

	vk::PhysicalDevice  best_device       = nullptr;
	uint32_t            best_queue_family = static_cast<uint32_t>(-1);
	SurfaceCapabilities best_capabilities;

	for (const auto& device: devices) {
		bool skip_device = false;

		vk::PhysicalDeviceProperties device_properties {};
		device.getProperties(&device_properties);

		const auto reject = [&](const std::string& reason) {
			if (Config::RenderDocEnabled()) {
				LOGF("RenderDoc: device rejected: %s: %s\n", device_properties.deviceName.data(),
				     reason.c_str());
			}
			LOGF("%s\n", reason.c_str());
			out_rejections +=
			    skip_device ? "; " : fmt::format("\n  {}: ", device_properties.deviceName.data());
			out_rejections += reason;
			skip_device = true;
		};

		LOGF("Vulkan device: %s\n", device_properties.deviceName.data());
		if (device_properties.apiVersion < VULKAN_TARGET_API_VERSION) {
			reject(fmt::format("Vulkan {}.{} is required, but device supports only {}.{}.{}",
			                   VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MAJOR(device_properties.apiVersion),
			                   VK_VERSION_MINOR(device_properties.apiVersion),
			                   VK_VERSION_PATCH(device_properties.apiVersion)));
			continue;
		}

		vk::PhysicalDeviceFeatures2 device_features2 {};

		vk::PhysicalDeviceVulkan13Features features13 {};

		vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_ext {};
		vk::PhysicalDeviceImageViewMinLodFeaturesEXT  image_view_min_lod {};
		color_write_ext.pNext = &image_view_min_lod;

		vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
		depth_clip_enable.pNext = &color_write_ext;

		vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
		depth_clip_control.pNext = &depth_clip_enable;

		vk::PhysicalDeviceVulkan12Features features12 {};
		vk::PhysicalDeviceVulkan11Features features11 {};
		vk::PhysicalDeviceRobustness2FeaturesEXT robustness2 {};
#if defined(__APPLE__)
		features12.pNext = &depth_clip_control;
#else
		vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_barycentric {};
		fragment_barycentric.pNext = &depth_clip_control;
		features12.pNext           = &fragment_barycentric;
#endif
		robustness2.pNext      = features12.pNext;
		features11.pNext       = &robustness2;
		features12.pNext       = &features11;
		features13.pNext       = &features12;
		device_features2.pNext = &features13;

		device.getFeatures2(&device_features2);
		if (Config::RenderDocEnabled()) {
			LOGF("RenderDoc: GPU=%s; vendor=0x%x; device=0x%x; driver(raw)=%u; Vulkan=%u.%u.%u; "
			     "BDA=%u; BDA captureReplay=%u; shaderInt64=%u\n",
			     device_properties.deviceName.data(), device_properties.vendorID,
			     device_properties.deviceID, device_properties.driverVersion,
			     VK_VERSION_MAJOR(device_properties.apiVersion),
			     VK_VERSION_MINOR(device_properties.apiVersion),
			     VK_VERSION_PATCH(device_properties.apiVersion), features12.bufferDeviceAddress,
			     features12.bufferDeviceAddressCaptureReplay,
			     device_features2.features.shaderInt64);
			const auto extensions = EnumerateVulkan<vk::ExtensionProperties>(
			    "vkEnumerateDeviceExtensionProperties",
			    [&](uint32_t* count, vk::ExtensionProperties* values) {
				    return device.enumerateDeviceExtensionProperties(nullptr, count, values);
			    });
			for (const auto& ext: extensions) {
				LOGF("RenderDoc: available device extension %s (version %u)\n",
				     ext.extensionName.data(), ext.specVersion);
			}
			for (const auto* ext: device_extensions) {
				LOGF("RenderDoc: required device extension %s; available=%u\n", ext,
				     HasExtension(extensions, ext) ? 1u : 0u);
			}
		}
		const auto required_features12 = WindowContext::RequiredVulkan12Features();
		const auto required_features13 = WindowContext::RequiredVulkan13Features();

		const auto queue_family = VulkanFindQueueFamily(device, surface);
		if (queue_family == static_cast<uint32_t>(-1)) {
			reject("No universal graphics, compute, and presentation queue");
		}

		const auto check_feature = [&](vk::Bool32 supported, const char* name,
		                               vk::Bool32 required = VK_TRUE) {
			if (Config::RenderDocEnabled()) {
				LOGF("RenderDoc: device feature %s; required=%u; available=%u\n", name, required,
				     supported);
			}
			if (required == VK_TRUE && supported != VK_TRUE) {
				reject(fmt::format("{} is not supported", name));
			}
		};

#if defined(__APPLE__)
		if (color_write_ext.colorWriteEnable != VK_TRUE) {
			LOGF("colorWriteEnable is not supported\n");
		}
#else
		check_feature(color_write_ext.colorWriteEnable, "colorWriteEnable");
#endif
		check_feature(image_view_min_lod.minLod, "image view minLod");

		check_feature(depth_clip_control.depthClipControl, "depthClipControl");
#if defined(__APPLE__)
		if (depth_clip_enable.depthClipEnable != VK_TRUE) {
			LOGF("depthClipEnable is not supported\n");
		}
#else
		check_feature(depth_clip_enable.depthClipEnable, "depthClipEnable");
#endif
#if !defined(__APPLE__)
		check_feature(device_features2.features.depthClamp, "depthClamp");
#endif

		check_feature(features12.samplerMirrorClampToEdge, "samplerMirrorClampToEdge",
		              required_features12.samplerMirrorClampToEdge);
		check_feature(features11.storageBuffer16BitAccess, "storageBuffer16BitAccess",
		              WindowContext::RequiredVulkan11Features().storageBuffer16BitAccess);
		check_feature(features12.storageBuffer8BitAccess, "storageBuffer8BitAccess",
		              required_features12.storageBuffer8BitAccess);
		check_feature(features12.timelineSemaphore, "timelineSemaphore",
		              required_features12.timelineSemaphore);
		check_feature(features12.shaderOutputLayer, "shaderOutputLayer",
		              required_features12.shaderOutputLayer);
		check_feature(features12.shaderOutputViewportIndex, "shaderOutputViewportIndex",
		              required_features12.shaderOutputViewportIndex);
		check_feature(features12.bufferDeviceAddress, "bufferDeviceAddress",
		              required_features12.bufferDeviceAddress);
		check_feature(features12.shaderBufferInt64Atomics, "shaderBufferInt64Atomics",
		              required_features12.shaderBufferInt64Atomics);
		check_feature(features12.shaderSampledImageArrayNonUniformIndexing,
		              "shaderSampledImageArrayNonUniformIndexing",
		              required_features12.shaderSampledImageArrayNonUniformIndexing);
		check_feature(features12.shaderStorageImageArrayNonUniformIndexing,
		              "shaderStorageImageArrayNonUniformIndexing",
		              required_features12.shaderStorageImageArrayNonUniformIndexing);
		check_feature(features12.shaderStorageBufferArrayNonUniformIndexing,
		              "shaderStorageBufferArrayNonUniformIndexing",
		              required_features12.shaderStorageBufferArrayNonUniformIndexing);
		check_feature(features12.descriptorBindingSampledImageUpdateAfterBind,
		              "descriptorBindingSampledImageUpdateAfterBind",
		              required_features12.descriptorBindingSampledImageUpdateAfterBind);
		check_feature(features12.descriptorBindingUpdateUnusedWhilePending,
		              "descriptorBindingUpdateUnusedWhilePending",
		              required_features12.descriptorBindingUpdateUnusedWhilePending);
		check_feature(features12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound",
		              required_features12.descriptorBindingPartiallyBound);
		check_feature(features13.robustImageAccess, "robustImageAccess");
		check_feature(robustness2.nullDescriptor, "nullDescriptor");
		check_feature(features13.dynamicRendering, "dynamicRendering",
		              required_features13.dynamicRendering);
		check_feature(features13.synchronization2, "synchronization2",
		              required_features13.synchronization2);
		check_feature(device_features2.features.sampleRateShading, "sampleRateShading");
		check_feature(device_features2.features.depthBiasClamp, "depthBiasClamp");
		check_feature(device_features2.features.shaderClipDistance, "shaderClipDistance");
		check_feature(device_features2.features.shaderCullDistance, "shaderCullDistance");
		check_feature(device_features2.features.largePoints, "largePoints");
		check_feature(device_features2.features.multiViewport, "multiViewport");
		check_feature(device_features2.features.fillModeNonSolid, "fillModeNonSolid");
		check_feature(device_features2.features.fragmentStoresAndAtomics,
		              "fragmentStoresAndAtomics");

		check_feature(device_features2.features.samplerAnisotropy, "samplerAnisotropy");
		check_feature(device_features2.features.robustBufferAccess, "robustBufferAccess");
#if defined(__APPLE__)
		if (device_features2.features.depthBounds != VK_TRUE) {
			LOGF("depthBounds is not supported\n");
		}
#else
		check_feature(device_features2.features.depthBounds, "depthBounds");
#endif
		check_feature(device_features2.features.shaderStorageImageWriteWithoutFormat,
		              "shaderStorageImageWriteWithoutFormat");

		check_feature(device_features2.features.shaderImageGatherExtended,
		              "shaderImageGatherExtended");

		check_feature(device_features2.features.independentBlend, "independentBlend");
		check_feature(device_features2.features.tessellationShader, "tessellationShader");
		check_feature(device_features2.features.drawIndirectFirstInstance,
		              "drawIndirectFirstInstance");
		check_feature(device_features2.features.multiDrawIndirect, "multiDrawIndirect");

		if (!skip_device) {
			auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
			    "vkEnumerateDeviceExtensionProperties",
			    [&](uint32_t* count, vk::ExtensionProperties* values) {
				    return device.enumerateDeviceExtensionProperties(nullptr, count, values);
			    });
			EXIT_NOT_IMPLEMENTED(available_extensions.empty());

			for (const char* ext: device_extensions) {
				if (!HasExtension(available_extensions, ext)) {
					reject(fmt::format("{} is not supported", ext));
				}
			}

			if (skip_device) {
				for (const auto& ext: available_extensions) {
					LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
					     ext.specVersion);
				}
			}
		}

		SurfaceCapabilities candidate_capabilities;
		if (!skip_device) {
			GetSurfaceCapabilities(device, surface, candidate_capabilities);

			if (!(candidate_capabilities.capabilities.supportedUsageFlags &
			      vk::ImageUsageFlagBits::eTransferDst)) {
				reject("Surface cannot be destination of blit");
			}
		}

		const auto check_format = [&](vk::Format format, vk::FormatFeatureFlags features,
		                              const char* usage) {
			if (!skip_device && !CheckFormat(device, format, true, features)) {
				reject(fmt::format("Format vk::Format::e{} cannot be used as {}",
				                   vk::to_string(format), usage));
			}
		};

		const auto depth_features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
		const auto texture_features =
		    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
		const auto storage_features =
		    vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferDst;

		check_format(vk::Format::eD32Sfloat, depth_features, "depth buffer");
		check_format(vk::Format::eD32SfloatS8Uint, depth_features, "depth buffer");
		check_format(vk::Format::eD16Unorm, depth_features, "depth buffer");
		check_format(vk::Format::eBc3SrgbBlock, texture_features, "texture");
		check_format(vk::Format::eR8G8B8A8Srgb, texture_features, "texture");
		check_format(vk::Format::eR8Unorm, texture_features, "texture");
		check_format(vk::Format::eR8G8Unorm, texture_features, "texture");

		if (!skip_device &&
		    !CheckFormat(device, vk::Format::eR8G8B8A8Srgb, true, storage_features)) {
			LOGF("Format vk::Format::eR8G8B8A8Srgb cannot be used as storage image\n");
			check_format(vk::Format::eR8G8B8A8Unorm, storage_features, "storage image");
		}

		if (!skip_device &&
		    !CheckFormat(device, vk::Format::eB8G8R8A8Srgb, true, storage_features)) {
			LOGF("Format vk::Format::eB8G8R8A8Srgb cannot be used as storage image\n");
			check_format(vk::Format::eB8G8R8A8Unorm, storage_features, "storage image");
		}

		if (!skip_device && device_properties.limits.maxSamplerAnisotropy < 16.0f) {
			reject("maxSamplerAnisotropy < 16.0f");
		}

		if (skip_device) {
			continue;
		}

		if (best_device == nullptr ||
		    device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
			best_device       = device;
			best_queue_family = queue_family;
			best_capabilities = std::move(candidate_capabilities);
		}
	}

	out_device       = best_device;
	out_queue_family = best_queue_family;
	if (best_device != nullptr) {
		out_capabilities = std::move(best_capabilities);
	}
}

static vk::Device VulkanCreateDevice(GraphicContext& graphics,
	                                 const std::vector<const char*>& device_extensions) {
	const auto physical_device = graphics.physical_device;
	const auto queue_family    = graphics.queue_family;
	EXIT_IF(physical_device == nullptr);
	EXIT_IF(queue_family == static_cast<uint32_t>(-1));

	const float               queue_priority[2] = {1.0f, 1.0f};
	vk::DeviceQueueCreateInfo queue_create_info {};
	queue_create_info.queueFamilyIndex = queue_family;
	queue_create_info.queueCount       = 1;
	queue_create_info.pQueuePriorities = queue_priority;
	std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
	{
		const auto families = physical_device.getQueueFamilyProperties();
		if (queue_family < families.size() && families[queue_family].queueCount >= 2) {
			queue_create_info.queueCount = 2;
		}
		queue_create_infos.push_back(queue_create_info);
		graphics.buffer_families = {queue_family};
		// Readback queues beside the graphics engine (experiments ReadbackCompute/Transfer).
		const char* extra = std::getenv("KYTY_READBACK_FAMILIES");
		if (extra == nullptr || extra[0] != '0') {
			const auto find = [&](vk::QueueFlags want, vk::QueueFlags reject) {
				for (uint32_t i = 0; i < families.size(); i++) {
					if (i != queue_family && families[i].queueCount != 0 && (families[i].queueFlags & want) == want &&
					    !(families[i].queueFlags & reject)) {
						return i;
					}
				}
				return static_cast<uint32_t>(-1);
			};
			const uint32_t compute  = find(vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics);
			const uint32_t transfer = find(vk::QueueFlagBits::eTransfer,
			                               vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
			for (const auto family: {compute, transfer}) {
				if (family == static_cast<uint32_t>(-1)) {
					continue;
				}
				vk::DeviceQueueCreateInfo info {};
				info.queueFamilyIndex = family;
				info.queueCount       = 1;
				info.pQueuePriorities = queue_priority;
				queue_create_infos.push_back(info);
				graphics.buffer_families.push_back(family);
			}
			graphics.readback_families[1] = compute;
			graphics.readback_families[2] = transfer;
		} else {
			graphics.readback_families[1] = graphics.readback_families[2] = static_cast<uint32_t>(-1);
		}
		graphics.readback_families[0] = queue_family;
	}

	vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_ext {};
	color_write_ext.colorWriteEnable = VK_TRUE;

	vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
	depth_clip_enable.pNext = &color_write_ext;
	depth_clip_enable.depthClipEnable = VK_TRUE;

	vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
	vk::PhysicalDeviceImageViewMinLodFeaturesEXT  image_view_min_lod {};
	image_view_min_lod.minLod = VK_TRUE;
	depth_clip_control.pNext  = &image_view_min_lod;
	// MoltenVK lacks VK_EXT_depth_clip_enable and VK_EXT_color_write_enable, so drop those
	// feature structs from the chain on macOS (the renderer falls back to default depth
	// clipping and static color-write masks).
#if !defined(__APPLE__)
	image_view_min_lod.pNext = &depth_clip_enable;
#endif
	depth_clip_control.depthClipControl = VK_TRUE;

	const bool workgroup_layout_extension =
	    HasExtension(device_extensions, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR supported_workgroup_layout {};
	vk::PhysicalDeviceVulkan12Features supported_features12 {};
	supported_features12.pNext = workgroup_layout_extension ? &supported_workgroup_layout : nullptr;
	vk::PhysicalDeviceVulkan13Features supported_features13 {};
	supported_features13.pNext = &supported_features12;

	const auto robustness2_ext_enabled =
	    HasExtension(device_extensions, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);

	vk::PhysicalDeviceRobustness2FeaturesEXT supported_robustness2 {};
	if (robustness2_ext_enabled) {
		supported_robustness2.pNext = supported_features13.pNext;
		supported_features13.pNext = &supported_robustness2;
	}

	const bool mesh_extension = HasExtension(device_extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
	vk::PhysicalDeviceMeshShaderFeaturesEXT supported_mesh {};
	supported_mesh.pNext = &supported_features13;
	vk::PhysicalDeviceFeatures2 supported_features2 {};
	supported_features2.pNext = mesh_extension ? static_cast<void*>(&supported_mesh)
	                                           : static_cast<void*>(&supported_features13);
	const bool feedback_extensions =
	    HasExtension(device_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
	    HasExtension(device_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
	vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT feedback_layout {};
	vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT feedback_dynamic {};
	if (feedback_extensions) {
		feedback_dynamic.pNext = supported_features2.pNext;
		feedback_layout.pNext  = &feedback_dynamic;
		supported_features2.pNext = &feedback_layout;
	}
	const bool provoking_extension =
	    HasExtension(device_extensions, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
	vk::PhysicalDeviceProvokingVertexFeaturesEXT provoking_vertex {};
	if (provoking_extension) {
		provoking_vertex.pNext = supported_features2.pNext;
		supported_features2.pNext = &provoking_vertex;
	}
	const bool maintenance8_extension =
	    HasExtension(device_extensions, VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
	vk::PhysicalDeviceMaintenance8FeaturesKHR maintenance8 {};
	if (maintenance8_extension) {
		maintenance8.pNext = supported_features2.pNext;
		supported_features2.pNext = &maintenance8;
	}
	const bool image_atomic_int64_extension =
	    HasExtension(device_extensions, VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
	vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic_int64 {};
	if (image_atomic_int64_extension) {
		image_atomic_int64.pNext = supported_features2.pNext;
		supported_features2.pNext = &image_atomic_int64;
	}
	const bool early_late_extension =
	    HasExtension(device_extensions, VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_EXTENSION_NAME);
	vk::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD early_late {};
	if (early_late_extension) {
		early_late.pNext          = supported_features2.pNext;
		supported_features2.pNext = &early_late;
	}
	const bool conditional_rendering_extension =
	    HasExtension(device_extensions, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
	vk::PhysicalDeviceConditionalRenderingFeaturesEXT conditional_rendering {};
	if (conditional_rendering_extension) {
		conditional_rendering.pNext = supported_features2.pNext;
		supported_features2.pNext   = &conditional_rendering;
	}
	const bool barycentric_extension =
	    HasExtension(device_extensions, VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR supported_barycentric {};
	if (barycentric_extension) {
		supported_barycentric.pNext = supported_features2.pNext;
		supported_features2.pNext   = &supported_barycentric;
	}
	const bool maximal_reconvergence_extension =
	    HasExtension(device_extensions, VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
	vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR maximal_reconvergence {};
	if (maximal_reconvergence_extension) {
		maximal_reconvergence.pNext = supported_features2.pNext;
		supported_features2.pNext   = &maximal_reconvergence;
	}
	physical_device.getFeatures2(&supported_features2);
	const bool maximal_reconvergence_enabled =
	    maximal_reconvergence_extension && maximal_reconvergence.shaderMaximalReconvergence == VK_TRUE;
	g_shader_maximal_reconvergence.store(maximal_reconvergence_enabled, std::memory_order_relaxed);
	if (MaximalReconvergenceRequested()) {
		std::printf("KYTY_MAXIMAL_RECONVERGENCE: maximal reconvergence %s\n",
		            maximal_reconvergence_enabled ? "enabled" : "unavailable");
	}
	const bool barycentric_supported =
	    barycentric_extension && supported_barycentric.fragmentShaderBarycentric == VK_TRUE;
	if (!barycentric_supported) {
		Log::WriteToConsoleAndLog(
		    "WARNING: Your GPU is not fully supported (fragmentShaderBarycentric is unavailable). "
		    "Some games may render incorrectly or crash.\n");
	}
	graphics.shader_image_int64_atomics_enabled = image_atomic_int64.shaderImageInt64Atomics;
	const bool early_late_enabled = early_late_extension && early_late.shaderEarlyAndLateFragmentTests == VK_TRUE;
	g_shader_early_late_tests.store(early_late_enabled, std::memory_order_relaxed);

	auto features12 = WindowContext::RequiredVulkan12Features();
	features12.shaderSharedInt64Atomics = supported_features12.shaderSharedInt64Atomics;
	features12.hostQueryReset           = supported_features12.hostQueryReset;
	features12.drawIndirectCount = supported_features12.drawIndirectCount;
	graphics.native_indirect_enabled = supported_features12.drawIndirectCount &&
	    supported_features2.features.multiDrawIndirect &&
	    supported_features2.features.drawIndirectFirstInstance;
	graphics.host_query_reset_enabled   = supported_features12.hostQueryReset != VK_FALSE;
	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout {};
	workgroup_layout.workgroupMemoryExplicitLayout =
	    supported_workgroup_layout.workgroupMemoryExplicitLayout;
	workgroup_layout.pNext = &depth_clip_control;
	features12.pNext = workgroup_layout_extension ? static_cast<void*>(&workgroup_layout)
	                                             : static_cast<void*>(&depth_clip_control);
	vk::PhysicalDeviceVulkan11Features native_supported11 {};
	vk::PhysicalDeviceFeatures2 native_supported2 {};
	native_supported2.pNext = &native_supported11;
	physical_device.getFeatures2(&native_supported2);
	auto features11 = WindowContext::RequiredVulkan11Features();
	features11.shaderDrawParameters = native_supported11.shaderDrawParameters;
	graphics.native_draw_parameters_enabled = native_supported11.shaderDrawParameters != VK_FALSE;
	features11.pNext = features12.pNext;
	features12.pNext = &features11;
	if (!features12.shaderSharedInt64Atomics || !workgroup_layout.workgroupMemoryExplicitLayout) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: Native 64-bit LDS atomics are unavailable: shaderSharedInt64Atomics={}, "
		    "workgroupMemoryExplicitLayout={}. Continuing with the selected Vulkan device.\n",
		    features12.shaderSharedInt64Atomics != VK_FALSE,
		    workgroup_layout.workgroupMemoryExplicitLayout != VK_FALSE));
	}
	graphics.mesh_shader_enabled = mesh_extension && supported_mesh.meshShader;
	// Bindless image tables index sampled image arrays per lane.
	features12.shaderSampledImageArrayNonUniformIndexing =
	    supported_features12.shaderSampledImageArrayNonUniformIndexing;
	if (supported_features12.shaderSampledImageArrayNonUniformIndexing != VK_TRUE ||
	    supported_features2.features.shaderSampledImageArrayDynamicIndexing != VK_TRUE) {
		LOGF("WARNING: non-uniform sampled image indexing is not supported; bindless image "
		     "tables will not work\n");
	}

	vk::PhysicalDeviceSubgroupSizeControlProperties subgroup_size_control {};

	vk::PhysicalDeviceVulkan11Properties properties11 {};
	properties11.pNext = &subgroup_size_control;

	vk::PhysicalDeviceFloatControlsProperties float_controls {};
	float_controls.pNext = &properties11;
	vk::PhysicalDeviceProperties2 properties2 {};
	properties2.pNext = &float_controls;

	if (graphics.mesh_shader_enabled) {
		subgroup_size_control.pNext = &graphics.mesh_shader_properties;
	}
	// KYTY_SKIP_BOUNDS_CHECKS needs the storage-buffer robustness alignment.
	vk::PhysicalDeviceRobustness2PropertiesEXT robustness2_properties {};
	if (robustness2_ext_enabled) {
		robustness2_properties.pNext = properties2.pNext;
		properties2.pNext            = &robustness2_properties;
	}
	physical_device.getProperties2(&properties2);
	EXIT_NOT_IMPLEMENTED(!(properties11.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote));

	graphics.subgroup_size                 = properties11.subgroupSize;
	graphics.pixel_quad_supported =
	    bool(properties11.subgroupSupportedStages & vk::ShaderStageFlagBits::eFragment) &&
	    bool(properties11.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eQuad);
	graphics.min_subgroup_size             = subgroup_size_control.minSubgroupSize;
	graphics.max_subgroup_size             = subgroup_size_control.maxSubgroupSize;
	graphics.required_subgroup_size_stages = subgroup_size_control.requiredSubgroupSizeStages;
	graphics.compute_subgroup_size_control_enabled =
	    supported_features13.subgroupSizeControl == VK_TRUE &&
	    (graphics.required_subgroup_size_stages & vk::ShaderStageFlagBits::eCompute) &&
	    subgroup_size_control.minSubgroupSize <= 64 &&
	    subgroup_size_control.maxSubgroupSize >= 64;

	LOGF("Vulkan subgroup: default=%u min=%u max=%u stages=0x%08x size_control=%s wave64=%s\n",
	     graphics.subgroup_size, graphics.min_subgroup_size, graphics.max_subgroup_size,
	     static_cast<vk::ShaderStageFlags::MaskType>(graphics.required_subgroup_size_stages),
	     graphics.compute_subgroup_size_control_enabled ? "true" : "false",
	     graphics.SupportsComputeWave64() ? "true" : "false");
	graphics.provoking_vertex_last_enabled = provoking_extension && provoking_vertex.provokingVertexLast;
	graphics.attachment_feedback_loop_enabled =
	    feedback_extensions && feedback_layout.attachmentFeedbackLoopLayout &&
	    feedback_dynamic.attachmentFeedbackLoopDynamicState;
	graphics.maintenance8_enabled = maintenance8_extension && maintenance8.maintenance8;
	graphics.conditional_rendering_enabled =
	    conditional_rendering_extension && conditional_rendering.conditionalRendering;
	LOGF("Vulkan maintenance8 support: %s\n",
	     graphics.maintenance8_enabled ? "true" : "false");
	LOGF("Vulkan depth feedback support: %s\n",
	     graphics.attachment_feedback_loop_enabled ? "true" : "false");
	if (graphics.mesh_shader_enabled) {
		LOGF("Vulkan MeshEXT: invocations=%u vertices=%u primitives=%u shared=%u\n",
		     graphics.mesh_shader_properties.maxMeshWorkGroupInvocations,
		     graphics.mesh_shader_properties.maxMeshOutputVertices,
		     graphics.mesh_shader_properties.maxMeshOutputPrimitives,
		     graphics.mesh_shader_properties.maxMeshSharedMemorySize);
	}
	// VulkanFindPhysicalDevice already checked the required creation features. These
	// requirements are specific to this creation path and are not part of device selection.
	if (Config::RenderDocEnabled()) {
		LOGF("RenderDoc: creation-only required features: shaderInt64=%u; "
		     "vertexPipelineStoresAndAtomics=%u; dualSrcBlend=%u (all require 1)\n",
		     supported_features2.features.shaderInt64,
		     supported_features2.features.vertexPipelineStoresAndAtomics,
		     supported_features2.features.dualSrcBlend);
	}
	EXIT_NOT_IMPLEMENTED(supported_features2.features.shaderInt64 != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_features2.features.vertexPipelineStoresAndAtomics != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_features2.features.dualSrcBlend != VK_TRUE);
	vk::PhysicalDeviceFeatures device_features {};
	if (Nhl27Fsr3::Enabled()) {
		device_features.shaderInt16 = supported_features2.features.shaderInt16;
		std::printf("NHL27FSR3GAME feature=shaderInt16 enabled=%u\n",
			unsigned(device_features.shaderInt16));
		std::fflush(stdout);
	}
	device_features.fragmentStoresAndAtomics = VK_TRUE;
	device_features.samplerAnisotropy        = VK_TRUE;
	device_features.multiDrawIndirect = supported_features2.features.multiDrawIndirect;
	device_features.drawIndirectFirstInstance = supported_features2.features.drawIndirectFirstInstance;
	device_features.robustBufferAccess       = VK_TRUE;
#if !defined(__APPLE__)
	device_features.depthBounds = VK_TRUE; // unsupported by MoltenVK
	device_features.depthClamp  = VK_TRUE;
#endif
	device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	device_features.shaderSampledImageArrayDynamicIndexing =
	    supported_features2.features.shaderSampledImageArrayDynamicIndexing;
	device_features.shaderStorageImageMultisample        =
	    supported_features2.features.shaderStorageImageMultisample;
	device_features.shaderImageGatherExtended            = VK_TRUE;
	device_features.independentBlend                     = VK_TRUE;
	device_features.dualSrcBlend                         = VK_TRUE;
	device_features.tessellationShader                   = VK_TRUE;
	device_features.drawIndirectFirstInstance            = VK_TRUE;
	device_features.multiDrawIndirect                    = VK_TRUE;
	device_features.sampleRateShading                    = VK_TRUE;
	device_features.depthBiasClamp                       = VK_TRUE;
	device_features.shaderClipDistance                   = VK_TRUE;
	device_features.shaderCullDistance                   = VK_TRUE;
	device_features.largePoints                          = VK_TRUE;
	device_features.multiViewport                        = VK_TRUE;
	device_features.fillModeNonSolid                      = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics       = VK_TRUE;
	// Storage image sample counts beyond 1x require the feature.
	if (!device_features.shaderStorageImageMultisample) {
		graphics.physical_device_properties.limits.storageImageSampleCounts =
		    vk::SampleCountFlagBits::e1;
	}
	graphics.sample_rate_shading_enabled                 = true;
	device_features.shaderInt64 = VK_TRUE;
	g_shader_ftz32.store(float_controls.shaderDenormFlushToZeroFloat32 == VK_TRUE &&
	                         float_controls.denormBehaviorIndependence !=
	                             vk::ShaderFloatControlsIndependence::eNone,
	                     std::memory_order_relaxed);
	g_shader_storage_align4.store(properties2.properties.limits.minStorageBufferOffsetAlignment <= 4u,
	                              std::memory_order_relaxed);
	std::printf("NHL27CODEGEN storage_offset_align=%llu\n",
	            static_cast<unsigned long long>(
	                properties2.properties.limits.minStorageBufferOffsetAlignment));
	std::printf("NHL27CODEGEN ftz32=%d (flush32 %d independence %d)\n", g_shader_ftz32.load() ? 1 : 0,
	            float_controls.shaderDenormFlushToZeroFloat32 == VK_TRUE ? 1 : 0,
	            static_cast<int>(float_controls.denormBehaviorIndependence));
	device_features.shaderFloat64 =
	    supported_features2.features.shaderFloat64 &&
	    float_controls.shaderSignedZeroInfNanPreserveFloat64;
	// if (device_features.shaderFloat64 && !float_controls.shaderDenormPreserveFloat64) {
	// 	Log::WriteToConsoleAndLog(
	// 	    "WARNING: Vulkan device does not guarantee FP64 denormal preservation; "
	// 	    "continuing with native FP64 arithmetic. Very small values may be flushed to zero.\n");
	// }

	vk::PhysicalDeviceRobustness2FeaturesEXT robustness2 {};
#if defined(__APPLE__)
	robustness2.pNext = &features12;
#else
	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_barycentric {};
	if (barycentric_supported) {
		fragment_barycentric.pNext                     = &features12;
		fragment_barycentric.fragmentShaderBarycentric = VK_TRUE;
		robustness2.pNext                              = &fragment_barycentric;
	} else {
		robustness2.pNext = &features12;
	}
	if (robustness2_ext_enabled) {
		g_shader_robust_buffer_access2.store(supported_robustness2.robustBufferAccess2 == VK_TRUE,
		                                     std::memory_order_relaxed);
	}
#endif
	robustness2.robustBufferAccess2 = supported_robustness2.robustBufferAccess2;
	robustness2.robustImageAccess2  = supported_robustness2.robustImageAccess2;
	robustness2.nullDescriptor      = VK_TRUE;

	auto features13 = WindowContext::RequiredVulkan13Features();
	features13.pNext = &robustness2;
	features13.robustImageAccess   = supported_features13.robustImageAccess;
	features13.subgroupSizeControl =
	    graphics.compute_subgroup_size_control_enabled ? VK_TRUE : VK_FALSE;

	{
		// Native f16 arithmetic for plain V_PK_{ADD,MUL,FMA,FMAC}_F16: the feature is enabled whenever
		// the device has it; experiment NativeFp16 (KYTY_NATIVE_FP16=1) selects the codegen.
		const bool  fp16_gate  = Common::Experiments::Enabled(Common::Experiments::NativeFp16);
		const bool  fp16_ok    = supported_features12.shaderFloat16 == VK_TRUE;
		uint32_t    fp16_modes = 0;
		if (fp16_ok) {
			features12.shaderFloat16 = VK_TRUE;
			if (float_controls.shaderDenormPreserveFloat16) {
				fp16_modes |= kNativeFp16DenormPreserve;
			}
			if (float_controls.shaderSignedZeroInfNanPreserveFloat16) {
				fp16_modes |= kNativeFp16SignedZeroInfNan;
			}
			if (float_controls.shaderRoundingModeRTEFloat16) {
				fp16_modes |= kNativeFp16RoundTiesEven;
			}
			// Guest needs RNE + denorm preserve; log (but still allow) when not guaranteed.
			if ((fp16_modes & (kNativeFp16DenormPreserve | kNativeFp16RoundTiesEven)) !=
			    (kNativeFp16DenormPreserve | kNativeFp16RoundTiesEven)) {
				LOGF("WARNING: native FP16 without full float-controls support (modes=0x%x): "
				     "f16 denorm/rounding may not match the guest\n",
				     fp16_modes);
			}
		}
		g_shader_native_fp16_modes.store(fp16_modes, std::memory_order_relaxed);
		g_shader_native_pkrtz.store(properties2.properties.vendorID == 0x1002u, std::memory_order_relaxed);
		g_shader_native_fp16.store(fp16_ok, std::memory_order_relaxed);
		LOGF("Vulkan shaderFloat16=%d gate=%d native_fp16=%d denormPreserve16=%d szinp16=%d rte16=%d\n",
		     supported_features12.shaderFloat16 == VK_TRUE, fp16_gate, fp16_ok,
		     float_controls.shaderDenormPreserveFloat16 == VK_TRUE,
		     float_controls.shaderSignedZeroInfNanPreserveFloat16 == VK_TRUE,
		     float_controls.shaderRoundingModeRTEFloat16 == VK_TRUE);
		std::printf("NHL27CODEGEN native_fp16=%d (device %d)\n", fp16_ok && fp16_gate ? 1 : 0,
		            fp16_ok ? 1 : 0);
		std::fflush(stdout);
	}
	LOGF("Vulkan robustness: robustImageAccess=%s robustImageAccess2=%s\n",
	     features13.robustImageAccess == VK_TRUE ? "true" : "false",
	     robustness2.robustImageAccess2 == VK_TRUE ? "true" : "false");
	{
		// KYTY_SKIP_BOUNDS_CHECKS=1 relies on robustBufferAccess2 zeroing out-of-range dword reads.
		const bool rba2  = robustness2_ext_enabled && robustness2.robustBufferAccess2 == VK_TRUE;
		const auto align = static_cast<unsigned long long>(
		    robustness2_properties.robustStorageBufferAccessSizeAlignment);
		g_shader_robust_buffer2.store(rba2 && align <= 4u, std::memory_order_relaxed);
		std::printf("NHL27CODEGEN robustBufferAccess2=%d storage_align=%llu skip_bounds_checks=%d\n",
		            rba2 ? 1 : 0, align, ShaderSkipBoundsChecks() ? 1 : 0);
		std::fflush(stdout);
	}

	vk::DeviceCreateInfo create_info {};
	vk::PhysicalDeviceMeshShaderFeaturesEXT mesh_features {};
	mesh_features.pNext                 = &features13;
	mesh_features.meshShader            = graphics.mesh_shader_enabled;
	feedback_dynamic.pNext =
	    mesh_extension ? static_cast<void*>(&mesh_features) : static_cast<void*>(&features13);
	create_info.pNext = graphics.attachment_feedback_loop_enabled
	                        ? static_cast<void*>(&feedback_layout)
	                        : feedback_dynamic.pNext;
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.pNext = const_cast<void*>(create_info.pNext);
		provoking_vertex.transformFeedbackPreservesProvokingVertex = VK_FALSE;
		create_info.pNext = &provoking_vertex;
	}
	if (graphics.shader_image_int64_atomics_enabled) {
		image_atomic_int64.pNext = const_cast<void*>(create_info.pNext);
		image_atomic_int64.sparseImageInt64Atomics = VK_FALSE;
		create_info.pNext = &image_atomic_int64;
	}
	vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipeline_executable {};
	if (graphics.pipeline_stats_enabled) {
		pipeline_executable.pNext                  = const_cast<void*>(create_info.pNext);
		pipeline_executable.pipelineExecutableInfo = VK_TRUE;
		create_info.pNext                          = &pipeline_executable;
	}
	vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR maximal_reconvergence_enable {};
	if (maximal_reconvergence_enabled) {
		maximal_reconvergence_enable.pNext                      = const_cast<void*>(create_info.pNext);
		maximal_reconvergence_enable.shaderMaximalReconvergence = VK_TRUE;
		create_info.pNext                                       = &maximal_reconvergence_enable;
	}
	vk::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD early_late_enable {};
	if (early_late_enabled) {
		early_late_enable.pNext                           = const_cast<void*>(create_info.pNext);
		early_late_enable.shaderEarlyAndLateFragmentTests = VK_TRUE;
		create_info.pNext                                 = &early_late_enable;
	}
	// VK_EXT_device_fault: lets a device loss report the faulting GPU address.
	vk::PhysicalDeviceFaultFeaturesEXT device_fault {};
	if (HasExtension(device_extensions, VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
		device_fault.pNext       = const_cast<void*>(create_info.pNext);
		device_fault.deviceFault = VK_TRUE;
		create_info.pNext        = &device_fault;
	}
	if (graphics.maintenance8_enabled) {
		maintenance8.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &maintenance8;
	}
	if (graphics.conditional_rendering_enabled) {
		conditional_rendering.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &conditional_rendering;
	}
	create_info.pQueueCreateInfos       = queue_create_infos.data();
	create_info.queueCreateInfoCount    = static_cast<uint32_t>(queue_create_infos.size());
	create_info.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
	create_info.ppEnabledExtensionNames = device_extensions.data();
	create_info.pEnabledFeatures        = &device_features;

	vk::Device device = nullptr;

	if (Config::RenderDocEnabled()) {
		for (const auto* ext: device_extensions) {
			LOGF("RenderDoc: enabling device extension %s\n", ext);
		}
		LOGF("RenderDoc: vkCreateDevice begin; BDA=%u; BDA captureReplay requested by app=%u; "
		     "shaderInt64=%u\n",
		     features12.bufferDeviceAddress, features12.bufferDeviceAddressCaptureReplay,
		     device_features.shaderInt64);
		LOGF("RenderDoc: compatibility mode=%u; meshShader=%u; workgroupMemoryExplicitLayout=%u; "
		     "shaderSharedInt64Atomics=%u; deviceFault=%u\n",
		     RenderDocCompat() ? 1u : 0u, mesh_features.meshShader,
		     workgroup_layout.workgroupMemoryExplicitLayout, features12.shaderSharedInt64Atomics,
		     device_fault.deviceFault);
		Log::Flush();
	}
	auto result = physical_device.createDevice(&create_info, nullptr, &device);
	if (Config::RenderDocEnabled()) {
		LOGF("RenderDoc: vkCreateDevice result=%s\n", vk::to_string(result).c_str());
	}
	if (result != vk::Result::eSuccess) {
		LOGF("vkCreateDevice failed: %s\n", vk::to_string(result).c_str());
		return nullptr;
	}

	return device;
}

static void VulkanGetExtensions(VulkanExtensions& r) {
	uint32_t required_extensions_count = 0;

	const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&required_extensions_count);
	EXIT_NOT_IMPLEMENTED(extensions == nullptr);
	EXIT_NOT_IMPLEMENTED(required_extensions_count == 0);
	r.required_extensions.assign(extensions, extensions + required_extensions_count);

	r.available_extensions =
	    EnumerateVulkan<vk::ExtensionProperties>( // @suppress("Ambiguous problem")
	        "vkEnumerateInstanceExtensionProperties",
	        [](uint32_t* count, vk::ExtensionProperties* values) {
		        return vk::enumerateInstanceExtensionProperties(nullptr, count, values);
	        });

	r.enable_validation_layers = Config::VulkanValidationEnabled();
	if (Config::HdrEnabled() &&
	    HasExtension(r.available_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) &&
	    !HasExtension(r.required_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
	}

	if (HasExtension(r.available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	} else {
		r.enable_validation_layers = false;
	}

	for (const char* ext: r.required_extensions) {
		LOGF("Vulkan required extension: %s\n", ext);
	}

	for (const auto& ext: r.available_extensions) {
		LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
		     ext.specVersion);
	}

	r.available_layers = EnumerateVulkan<vk::LayerProperties>( // @suppress("Ambiguous problem")
	    "vkEnumerateInstanceLayerProperties", [](uint32_t* count, vk::LayerProperties* values) {
		    return vk::enumerateInstanceLayerProperties(count, values);
	    });

	for (const auto& l: r.available_layers) {
		LOGF("Vulkan available layer: %s, specVersion = %u, implVersion = %u, %s\n",
		     l.layerName.data(), l.specVersion, l.implementationVersion, l.description.data());
	}

	r.required_layers = {"VK_LAYER_KHRONOS_validation"};

	if (r.enable_validation_layers) {
		for (const char* l: r.required_layers) {
			if (!HasLayer(r.available_layers, l)) {
				LOGF("no validation layer: %s\n", l);
				r.enable_validation_layers = false;
				break;
			}
		}
	}

	if (r.enable_validation_layers) {
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateInstanceExtensionProperties",
		    [](uint32_t* count, vk::ExtensionProperties* values) {
			    return vk::enumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation",
			                                                    count, values);
		    });

		for (const auto& ext: available_extensions) {
			LOGF("VK_LAYER_KHRONOS_validation available extension: %s, version = %u\n",
			     ext.extensionName.data(), ext.specVersion);
		}

		if (HasExtension(available_extensions, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		} else {
			r.enable_validation_layers = false;
		}
	}
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDebugMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT      message_severity,
    vk::DebugUtilsMessageTypeFlagsEXT             message_types,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, void* /*user_data*/) {
	EXIT_IF(callback_data == nullptr);
	EXIT_IF(callback_data->pMessage == nullptr);

	const char*     severity_str   = nullptr;
	fmt::text_style severity_style = Log::Color::Default;
	bool            skip           = false;
	bool            error          = false;
	bool            debug_printf   = false;
	switch (message_severity) {
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
			severity_str   = "V";
			severity_style = Log::Color::BrightWhite;
			skip           = true;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
			if ((message_types & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) &&
			    Config::SpirvDebugPrintfEnabled() && callback_data->pMessageIdName != nullptr &&
			    strcmp(callback_data->pMessageIdName, "UNASSIGNED-DEBUG-PRINTF") == 0) {
				debug_printf   = true;
				severity_style = Log::Color::BrightYellow;
				skip           = true;
			} else {
				severity_str   = "I";
				severity_style = Log::Color::Default;
				skip           = true;
			}
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
			severity_str   = "W";
			severity_style = Log::Color::Red;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
			severity_str   = "E";
			severity_style = Log::Color::BrightRed;
			// Only validation errors are fatal; GENERAL-type errors can come
			// from unrelated loader/layer issues (e.g. a broken overlay).
			error = static_cast<bool>(message_types &
			                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
			break;
		default: severity_str = "?";
	}

	// NHL26 debugging: KYTY_DBG_VALIDATION_NONFATAL=1 logs validation errors (once per message
	// id) to stdout instead of exiting, so one run collects everything up to a device loss.
	static const bool nonfatal = [] {
		const char* v = std::getenv("KYTY_DBG_VALIDATION_NONFATAL");
		return v != nullptr && v[0] != '\0' && v[0] != '0';
	}();
	if (error && nonfatal) {
		static std::mutex                      seen_mutex;
		static std::unordered_set<std::string> seen;
		const std::string id = callback_data->pMessageIdName != nullptr
		                           ? callback_data->pMessageIdName
		                           : std::string(callback_data->pMessage).substr(0, 120);
		bool first = false;
		{
			std::lock_guard lock(seen_mutex);
			first = seen.insert(id).second;
		}
		if (first) {
			std::printf("NHL26VAL: [%s] %s\n", id.c_str(), callback_data->pMessage);
			std::fflush(stdout);
		}
		error = false;
		skip  = true;
	}
	if (error) {
		EXIT_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (!skip) {
		LOGF_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (debug_printf) {
		auto strs = Common::Split(std::string(callback_data->pMessage), '|');
		if (!strs.empty()) {
			LOGF_COLOR(severity_style, "%s\n", strs[strs.size() - 1].c_str());
		}
	}

	return VK_FALSE;
}

static VKAPI_ATTR vk::Result VKAPI_CALL VulkanCreateDebugUtilsMessengerEXT(
    vk::Instance instance, const vk::DebugUtilsMessengerCreateInfoEXT* create_info,
    const vk::AllocationCallbacks* allocator, vk::DebugUtilsMessengerEXT* messenger) {
	EXIT_IF(instance == nullptr);

	if (auto func = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDebugUtilsMessengerEXT; func != nullptr) {
		return instance.createDebugUtilsMessengerEXT(create_info, allocator, messenger);
	}
	return vk::Result::eErrorExtensionNotPresent;
}

static void VulkanCheckInstanceVersion() {
	uint32_t version = VK_API_VERSION_1_0;

	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion != nullptr) {
		auto result = vk::enumerateInstanceVersion(&version);
		if (result != vk::Result::eSuccess) {
			EXIT("Could not query Vulkan loader version: %s\n", vk::to_string(result).c_str());
		}
	}

	LOGF("Vulkan loader version: %u.%u.%u\n", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
	     VK_VERSION_PATCH(version));
	if (version < VULKAN_TARGET_API_VERSION) {
		EXIT("Vulkan %u.%u is required, but loader supports only %u.%u.%u\n",
		     VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
		     VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION), VK_VERSION_MAJOR(version),
		     VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
	}
}

void WindowContext::CreateVulkan() {
	EXIT_IF(window == nullptr);
	EXIT_IF(graphic_ctx.instance != nullptr);
	EXIT_IF(graphic_ctx.physical_device != nullptr);
	EXIT_IF(graphic_ctx.device != nullptr);
	EXIT_IF(surface != nullptr);

	auto get_instance_proc_addr =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
	if (get_instance_proc_addr == nullptr) {
		EXIT("Could not load Vulkan: %s\n", SDL_GetError());
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);

	Vr::OpenXrInit();

	VulkanExtensions r;
	VulkanGetExtensions(r);
	Vr::OpenXrAppendExtensions(r.required_extensions, r.available_extensions, false);
	VulkanCheckInstanceVersion();

	vk::ApplicationInfo app_info {};
	app_info.pApplicationName   = "Kyty";
	app_info.applicationVersion = 1;
	app_info.pEngineName        = "Kyty";
	app_info.engineVersion      = 1;
	app_info.apiVersion         = VULKAN_TARGET_API_VERSION; // NOLINT

	if (Config::SpirvDebugPrintfEnabled() && Config::GpuAssistedValidationEnabled()) {
		EXIT("--spirv-debug-printf and --gpu-assisted-validation are mutually exclusive\n");
	}

	vk::ValidationFeatureEnableEXT enabled_features[3]    = {};
	uint32_t                       enabled_features_count = 0;
#ifdef KYTY_ENABLE_BEST_PRACTICES
	enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eBestPractices;
#endif
#ifdef KYTY_ENABLE_DEBUG_PRINTF
	if (Config::SpirvDebugPrintfEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eDebugPrintf;
	}
#endif
	if (Config::GpuAssistedValidationEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eGpuAssisted;
		LOGF("Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
		std::printf("Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
		std::fflush(stdout);
	}

	vk::ValidationFeaturesEXT validation_features {};
	validation_features.enabledValidationFeatureCount  = enabled_features_count;
	validation_features.pEnabledValidationFeatures     = enabled_features;

	vk::DebugUtilsMessengerCreateInfoEXT dbg_create_info {};
	dbg_create_info.pNext           = &validation_features;
	dbg_create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	dbg_create_info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
	dbg_create_info.pfnUserCallback = VulkanDebugMessengerCallback;

	vk::InstanceCreateInfo inst_info {};
	inst_info.pNext                   = (r.enable_validation_layers ? &dbg_create_info : nullptr);
#if defined(__APPLE__)
	// MoltenVK requires VK_KHR_portability_enumeration + flag to surface
	// portability devices. Without this, enumeratePhysicalDevices hides the
	// MoltenVK adapter on some driver versions.
	if (HasExtension(r.available_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
		if (!HasExtension(r.required_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		}
		inst_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
		LOGF("Vulkan instance: enabled %s\n", VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	}
#endif
	inst_info.pApplicationInfo        = &app_info;
	inst_info.enabledExtensionCount   = static_cast<uint32_t>(r.required_extensions.size());
	inst_info.ppEnabledExtensionNames = r.required_extensions.data();
	inst_info.enabledLayerCount =
	    (r.enable_validation_layers ? static_cast<uint32_t>(r.required_layers.size()) : 0);
	inst_info.ppEnabledLayerNames =
	    (r.enable_validation_layers ? r.required_layers.data() : nullptr);

	if (Config::RenderDocEnabled()) {
		LOGF("RenderDoc: vkCreateInstance begin\n");
		Log::Flush();
	}
	const vk::Result result = vk::createInstance(&inst_info, nullptr, &graphic_ctx.instance);
	if (Config::RenderDocEnabled()) {
		LOGF("RenderDoc: vkCreateInstance result=%s\n", vk::to_string(result).c_str());
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eErrorIncompatibleDriver:
			EXIT("Unable to find a compatible Vulkan Driver");
		default: EXIT("Could not create a Vulkan instance (for unknown reasons)");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.instance);

	if (r.enable_validation_layers) {
		dbg_create_info.pNext = nullptr;
		if (VulkanCreateDebugUtilsMessengerEXT(graphic_ctx.instance, &dbg_create_info, nullptr,
		                                       &graphic_ctx.debug_messenger) !=
		    vk::Result::eSuccess) {
			EXIT("Could not create debug messenger");
		}
	}

	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not create a Vulkan surface");
	}
	surface = native_surface;

	std::vector<const char*> device_extensions = {
	    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	    VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	    VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, "VK_KHR_maintenance1"};
#if KYTY_ENABLE_FSR3
	// SDK 1.1.4 resolves these KHR aliases even on Vulkan 1.3 devices.
	device_extensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	device_extensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
#endif

#if defined(__APPLE__)
	// MoltenVK lacks VK_EXT_depth_clip_enable and VK_EXT_color_write_enable; the renderer
	// falls back to default depth clipping and static color-write masks on macOS. It also
	// requires VK_KHR_portability_subset per the Vulkan portability spec.
	device_extensions.push_back("VK_KHR_portability_subset");
#else
	device_extensions.push_back(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
	device_extensions.push_back(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME);
#endif

#ifdef KYTY_ENABLE_DEBUG_PRINTF
	if (Config::SpirvDebugPrintfEnabled()) {
		device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
	}
#endif

	std::string rejected_devices;
	VulkanFindPhysicalDevice(graphic_ctx.instance, surface, device_extensions, surface_capabilities,
	                         graphic_ctx.physical_device, graphic_ctx.queue_family,
	                         rejected_devices);

	if (graphic_ctx.physical_device == nullptr) {
		EXIT("Could not find suitable device:%s", rejected_devices.c_str());
	}

	vk::PhysicalDevicePushDescriptorProperties push_descriptor_properties {};
	vk::PhysicalDeviceProperties2              physical_device_properties {};
	physical_device_properties.pNext = &push_descriptor_properties;
	graphic_ctx.physical_device.getProperties2(&physical_device_properties);
	graphic_ctx.physical_device_properties = physical_device_properties.properties;
	graphic_ctx.max_push_descriptors       = push_descriptor_properties.maxPushDescriptors;
	graphic_ctx.physical_device.getMemoryProperties(&graphic_ctx.physical_device_memory_properties);
	const auto& device_properties = graphic_ctx.GetPhysicalDeviceProperties();

	LOGF("Select device: %s\n", device_properties.deviceName.data());

	const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info {
	    .format = vk::Format::eBc1RgbaUnormBlock,
	    .type = vk::ImageType::e2D,
	    .tiling = vk::ImageTiling::eOptimal,
	    .usage = vk::ImageUsageFlagBits::eSampled,
	    .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
	};
	const auto block_texel_view_props =
	    graphic_ctx.physical_device.getImageFormatProperties2(block_texel_view_info);
	graphic_ctx.supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
	LOGF("Block Texel View support: %s\n", graphic_ctx.supports_block_texel_view ? "Yes" : "No");

	{
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateDeviceExtensionProperties",
		    [&](uint32_t* count, vk::ExtensionProperties* values) {
			    return graphic_ctx.physical_device.enumerateDeviceExtensionProperties(
			        nullptr, count, values);
		    });

		if (HasExtension(available_extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
			graphic_ctx.memory_budget_ext_enabled = true;
		}
		for (const auto* extension: {VK_KHR_MAINTENANCE_8_EXTENSION_NAME,
		                             VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,
		                             VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME,
		                             VK_EXT_MESH_SHADER_EXTENSION_NAME,
		                             VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
		                             VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME,
		                             VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME,
		                             VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME,
		                             VK_EXT_DEVICE_FAULT_EXTENSION_NAME,
		                             VK_AMD_BUFFER_MARKER_EXTENSION_NAME,
		                             VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_EXTENSION_NAME}) {
			if (RenderDocSkipsExtension(extension)) {
				LOGF("RenderDoc compat: skipping %s\n", extension);
				continue;
			}
			if (HasExtension(available_extensions, extension)) {
				device_extensions.push_back(extension);
			}
		}
		// KYTY_MAXIMAL_RECONVERGENCE=1 (shader.h): compute shaders declare MaximallyReconvergesKHR when the
		// device has it.
		if (MaximalReconvergenceRequested() &&
		    HasExtension(available_extensions, VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
			device_extensions.push_back(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME);
		}
		if (HasExtension(available_extensions, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
			graphic_ctx.conditional_rendering_enabled = true;
		}
#if defined(_WIN32)
		// FSR 4 bridge (KYTY_FSR4=1): D3D12 images and fence shared with Vulkan.
		for (const auto* extension: {"VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32"}) {
			if (HasExtension(available_extensions, extension)) {
				device_extensions.push_back(extension);
			}
		}
#endif
		// KYTY_DBG_PIPELINE_STATS=<hex hash>[,...]: driver statistics and ISA of those shaders' pipelines.
		if (std::getenv("KYTY_DBG_PIPELINE_STATS") != nullptr &&
		    HasExtension(available_extensions, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
			graphic_ctx.pipeline_stats_enabled = true;
		}
		// Used only by the KYTY_IDLE_TRACE GPU timeline (GPU -> CPU clock conversion).
		if (HasExtension(available_extensions, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
			graphic_ctx.calibrated_timestamps_enabled = true;
		}
		if (HasExtension(available_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
			device_extensions.push_back(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
		}
		Vr::OpenXrAppendExtensions(device_extensions, available_extensions, true);
	}

	graphic_ctx.device = VulkanCreateDevice(graphic_ctx, device_extensions);
	if (graphic_ctx.device == nullptr) {
		EXIT("Could not create device");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.device);
	graphic_ctx.device.getQueue(graphic_ctx.queue_family, 0, &graphic_ctx.queue);
	EXIT_IF(graphic_ctx.queue == nullptr);
	{
		const auto families = graphic_ctx.physical_device.getQueueFamilyProperties();
		if (graphic_ctx.queue_family < families.size() &&
		    families[graphic_ctx.queue_family].queueCount >= 2) {
			graphic_ctx.device.getQueue(graphic_ctx.queue_family, 1, &graphic_ctx.readback_queue);
		}
		graphic_ctx.readback_queues[0] = graphic_ctx.readback_queue;
		for (uint32_t kind = 1; kind < 3; kind++) {
			if (graphic_ctx.readback_families[kind] != static_cast<uint32_t>(-1)) {
				graphic_ctx.device.getQueue(graphic_ctx.readback_families[kind], 0,
				                            &graphic_ctx.readback_queues[kind]);
			}
		}
		std::printf("NHL27READBACK: queue families graphics=%u compute=%d transfer=%d\n", graphic_ctx.queue_family,
		            static_cast<int>(graphic_ctx.readback_families[1]), static_cast<int>(graphic_ctx.readback_families[2]));
	}

	if (!graphic_ctx.CreateAllocator()) {
		EXIT("Could not create Vulkan memory allocator");
	}

	render_context = std::make_unique<RenderContext>(graphic_ctx);
	LibKernel::Memory::InstallGpuResources(render_context.get());
	presenter = std::make_unique<Presenter>(*this);
	Vr::OpenXrStartSession(graphic_ctx);
}

void WindowContext::RefreshSurfaceCapabilities() {
	EXIT_IF(graphic_ctx.physical_device == nullptr || surface == nullptr);
	GetSurfaceCapabilities(graphic_ctx.physical_device, surface, surface_capabilities);
}

void WindowContext::RecreateSurface() {
	EXIT_IF(window == nullptr || graphic_ctx.instance == nullptr);
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not recreate the Vulkan surface: %s\n", SDL_GetError());
	}
	surface = native_surface;
}

WindowContext::~WindowContext() {
	Vr::OpenXrShutdown();
	ShutdownSystemOverlayInput();
	presenter.reset();
	LibKernel::Memory::InstallGpuResources(nullptr);
	render_context.reset();

	if (graphic_ctx.device != nullptr) {
		RequireVulkanSuccess(graphic_ctx.device.waitIdle(), "wait for Vulkan device shutdown");
		graphic_ctx.DestroyAllocator();
		graphic_ctx.device.destroy(nullptr);
		graphic_ctx.device = nullptr;
		graphic_ctx.queue  = nullptr;
	}
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	if (graphic_ctx.debug_messenger != nullptr) {
		graphic_ctx.instance.destroyDebugUtilsMessengerEXT(graphic_ctx.debug_messenger, nullptr);
		graphic_ctx.debug_messenger = nullptr;
	}
	if (graphic_ctx.instance != nullptr) {
		graphic_ctx.instance.destroy(nullptr);
		graphic_ctx.instance        = nullptr;
		graphic_ctx.physical_device = nullptr;
	}
	if (window != nullptr) {
		SDL_DestroyWindow(window);
		window = nullptr;
	}
	SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
}

} // namespace Libs::Graphics
