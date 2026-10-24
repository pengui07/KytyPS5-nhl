#include "graphics/presentation/fsr/fsr4Bridge.h"

#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if KYTY_ENABLE_FSR4 && defined(_WIN32)
#define KYTY_FSR4_BRIDGE 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vulkan/vulkan_win32.h>
// ffx_api_loader.h only defines ffxLoadFunctions for these platform macros.
#ifndef _WINDOWS
#define _WINDOWS 1
#endif
#include "dx12/ffx_api_dx12.h"
#include "ffx_api.h"
#include "ffx_api_loader.h"
#include "ffx_upscale.h"
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#else
#define KYTY_FSR4_BRIDGE 0
#endif

namespace Libs::Graphics {

#if KYTY_FSR4_BRIDGE
namespace {
using Microsoft::WRL::ComPtr;

void Log(const char* format, const std::string& detail) {
	std::printf(format, detail.c_str());
	std::fflush(stdout);
}

void FfxMessage(uint32_t type, const wchar_t* message) {
	char text[512] {};
	if (message != nullptr) {
		WideCharToMultiByte(CP_UTF8, 0, message, -1, text, sizeof(text) - 1, nullptr, nullptr);
	}
	std::printf("NHL27FSR4 ffx-%s: %s\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", text);
	std::fflush(stdout);
}

// Test switches (read once):
//   KYTY_FSR4_VERSION=<text>    use the upscaler version whose name contains it (e.g. 3.1.5), else the newest
//   KYTY_FSR4_JITTER=<x>,<y>    multiply the jitter (default 1,1; -1,1 flips x; 0,0 turns it off)
//   KYTY_FSR4_PREEXPOSURE=1     pass pre-exposure 1 instead of the game's value
//   KYTY_FSR4_DEBUG_VIEW=1      FSR's own debug overlay (motion vectors, reactivity) on the output
//   KYTY_FSR4_DEPTH_NOSWAP=1    pass camera near/far as they are instead of swapped for inverted depth
//   KYTY_FSR4_MV=<x>,<y>        multiply the motion-vector scale (default 1,1)
//   KYTY_FSR4_DEBUG_CHECK=1     FSR checks its inputs and prints warnings (NHL27FSR4 ffx-warning lines)
//   KYTY_FSR4_AUTOEXPOSURE=0    let FSR use exposure 1 instead of computing its own
//   KYTY_FSR4_METERS=<f>        view-space units to metres (default 1)
struct Tuning {
	std::string version;
	float       jitter_x = 1, jitter_y = 1;
	bool        unit_pre_exposure = false;
	bool        debug_view        = false;
	bool        depth_noswap      = false;
	float       mv_x = 1, mv_y = 1;
	bool        debug_check       = false;
	bool        auto_exposure     = true;
	float       meters            = 1;
};
const Tuning& GetTuning() {
	static const Tuning tuning = [] {
		Tuning t;
		if (const char* v = std::getenv("KYTY_FSR4_VERSION")) t.version = v;
		if (const char* v = std::getenv("KYTY_FSR4_JITTER")) {
			if (std::sscanf(v, "%f,%f", &t.jitter_x, &t.jitter_y) != 2) t.jitter_x = t.jitter_y = 1;
		}
		if (const char* v = std::getenv("KYTY_FSR4_PREEXPOSURE")) t.unit_pre_exposure = v[0] == '1';
		if (const char* v = std::getenv("KYTY_FSR4_DEBUG_VIEW")) t.debug_view = v[0] == '1';
		if (const char* v = std::getenv("KYTY_FSR4_DEPTH_NOSWAP")) t.depth_noswap = v[0] == '1';
		if (const char* v = std::getenv("KYTY_FSR4_DEBUG_CHECK")) t.debug_check = v[0] == '1';
		if (const char* v = std::getenv("KYTY_FSR4_AUTOEXPOSURE")) t.auto_exposure = v[0] != '0';
		if (const char* v = std::getenv("KYTY_FSR4_METERS")) {
			const float meters = std::strtof(v, nullptr);
			if (meters > 0) t.meters = meters;
		}
		if (const char* v = std::getenv("KYTY_FSR4_MV")) {
			if (std::sscanf(v, "%f,%f", &t.mv_x, &t.mv_y) != 2) t.mv_x = t.mv_y = 1;
		}
		std::printf("NHL27FSR4 tuning version=%s jitter=%g,%g pre_exposure=%s debug_view=%d depth_noswap=%d mv=%g,%g "
		            "debug_check=%d auto_exposure=%d meters=%g\n",
		            t.version.empty() ? "newest" : t.version.c_str(), double(t.jitter_x), double(t.jitter_y),
		            t.unit_pre_exposure ? "1" : "game", t.debug_view ? 1 : 0, t.depth_noswap ? 1 : 0,
		            double(t.mv_x), double(t.mv_y), t.debug_check ? 1 : 0, t.auto_exposure ? 1 : 0,
		            double(t.meters));
		std::fflush(stdout);
		return t;
	}();
	return tuning;
}

struct Format {
	DXGI_FORMAT dxgi;
	VkFormat    vk;
};
constexpr std::array<Format, Fsr4Bridge::Count> kFormats {{
    {DXGI_FORMAT_R11G11B10_FLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32},
    {DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT},
    {DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
    {DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM},
}};
} // namespace
#endif

struct Fsr4Bridge::Impl {
	GraphicContext& graphics;
	std::string     status = "FSR 4 not initialized";
	bool            failed = false;
	bool            fresh  = false;
	vk::Extent2D    render {}, display {};
	uint64_t        value = 0;

	explicit Impl(GraphicContext& gfx): graphics(gfx) {}

	bool Fail(const std::string& why) {
		failed = true;
		status = "FSR 4 unavailable: " + why;
		std::printf("NHL27FSR4 status=failed reason=%s\n", why.c_str());
		std::fflush(stdout);
		return false;
	}

#if KYTY_FSR4_BRIDGE
	static constexpr int Ring = 3;
	struct Shared {
		ComPtr<ID3D12Resource> resource;
		VkImage                image  = VK_NULL_HANDLE;
		VkDeviceMemory         memory = VK_NULL_HANDLE;
	};

	HMODULE                                          loader = nullptr;
	ffxFunctions                                     fns {};
	ComPtr<ID3D12Device>                             device;
	ComPtr<ID3D12CommandQueue>                       queue;
	std::array<ComPtr<ID3D12CommandAllocator>, Ring> allocators;
	std::array<uint64_t, Ring>                       allocator_value {};
	ComPtr<ID3D12GraphicsCommandList>                list;
	int                                              ring = 0;
	ComPtr<ID3D12Fence>                              fence;
	HANDLE                                           event = nullptr;
	VkSemaphore                                      semaphore = VK_NULL_HANDLE;
	std::array<Shared, Count>                        shared {};
	ffxContext                                       context = nullptr;
	uint64_t                                         last_signal = 0;
	uint64_t                                         dispatched  = 0;
	uint64_t                                         override_id = 0;
	PFN_vkGetMemoryWin32HandlePropertiesKHR          get_handle_properties = nullptr;
	PFN_vkImportSemaphoreWin32HandleKHR              import_semaphore      = nullptr;

	[[nodiscard]] VkDevice Vk() const { return static_cast<VkDevice>(graphics.device); }

	bool Init() {
		const VkDevice vk_device = Vk();
		get_handle_properties = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(vk_device, "vkGetMemoryWin32HandlePropertiesKHR"));
		import_semaphore = reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(vk_device, "vkImportSemaphoreWin32HandleKHR"));
		if (get_handle_properties == nullptr || import_semaphore == nullptr) {
			return Fail("VK_KHR_external_memory_win32 / VK_KHR_external_semaphore_win32 not enabled");
		}

		loader = LoadLibraryW(L"amd_fidelityfx_loader_dx12.dll");
		if (loader == nullptr) {
			return Fail("amd_fidelityfx_loader_dx12.dll not found next to the emulator");
		}
		ffxLoadFunctions(&fns, loader);
		if (fns.CreateContext == nullptr || fns.DestroyContext == nullptr || fns.Dispatch == nullptr ||
		    fns.Query == nullptr) {
			return Fail("amd_fidelityfx_loader_dx12.dll has no ffx API exports");
		}

		// The D3D12 device must be on the same graphics card as the Vulkan device.
		VkPhysicalDeviceIDProperties id {};
		id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
		VkPhysicalDeviceProperties2 properties {};
		properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		properties.pNext = &id;
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(
		    static_cast<VkPhysicalDevice>(graphics.physical_device), &properties);
		if (id.deviceLUIDValid != VK_TRUE) {
			return Fail("Vulkan device has no LUID");
		}
		LUID luid {};
		std::memcpy(&luid, id.deviceLUID, sizeof(luid));
		ComPtr<IDXGIFactory4> factory;
		if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
			return Fail("CreateDXGIFactory1");
		}
		ComPtr<IDXGIAdapter1> adapter;
		if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
			return Fail("no DXGI adapter with the Vulkan device's LUID");
		}
		if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
			return Fail("D3D12CreateDevice");
		}
		D3D12_COMMAND_QUEUE_DESC queue_desc {};
		queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)))) {
			return Fail("CreateCommandQueue");
		}
		for (auto& allocator: allocators) {
			if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)))) {
				return Fail("CreateCommandAllocator");
			}
		}
		if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0].Get(), nullptr,
		                                     IID_PPV_ARGS(&list))) ||
		    FAILED(list->Close())) {
			return Fail("CreateCommandList");
		}
		event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

		// Shared fence -> Vulkan timeline semaphore.
		if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)))) {
			return Fail("CreateFence");
		}
		HANDLE fence_handle = nullptr;
		if (FAILED(device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &fence_handle))) {
			return Fail("CreateSharedHandle(fence)");
		}
		VkSemaphoreTypeCreateInfo timeline {};
		timeline.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
		timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		timeline.initialValue  = 0;
		VkSemaphoreCreateInfo semaphore_info {};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphore_info.pNext = &timeline;
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &semaphore) !=
		    VK_SUCCESS) {
			CloseHandle(fence_handle);
			return Fail("vkCreateSemaphore");
		}
		VkImportSemaphoreWin32HandleInfoKHR import {};
		import.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
		import.semaphore  = semaphore;
		import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
		import.handle     = fence_handle;
		const VkResult imported = import_semaphore(vk_device, &import);
		CloseHandle(fence_handle);
		if (imported != VK_SUCCESS) {
			return Fail("vkImportSemaphoreWin32HandleKHR");
		}

		// Which upscaler versions the signed DLLs offer on this card (FSR 4 needs RDNA 3/4).
		uint64_t count = 0;
		ffxQueryDescGetVersions versions {};
		versions.header.type    = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
		versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		versions.device         = device.Get();
		versions.outputCount    = &count;
		fns.Query(nullptr, &versions.header);
		std::vector<uint64_t>    ids(count);
		std::vector<const char*> names(count);
		versions.versionIds   = ids.data();
		versions.versionNames = names.data();
		if (count != 0) fns.Query(nullptr, &versions.header);
		for (uint64_t i = 0; i < count; ++i) {
			std::printf("NHL27FSR4 available-version %llu: %s\n", static_cast<unsigned long long>(i),
			            names[i] != nullptr ? names[i] : "?");
			const auto& wanted = GetTuning().version;
			if (!wanted.empty() && override_id == 0 && names[i] != nullptr &&
			    std::strstr(names[i], wanted.c_str()) != nullptr) {
				override_id = ids[i];
			}
		}
		std::fflush(stdout);
		status = "FSR 4 ready";
		return true;
	}

	void WaitIdle() {
		if (fence && last_signal != 0 && fence->GetCompletedValue() < last_signal) {
			fence->SetEventOnCompletion(last_signal, event);
			WaitForSingleObject(event, 5000);
		}
	}

	void Release() {
		WaitIdle();
		if (context != nullptr) {
			fns.DestroyContext(&context, nullptr);
			context = nullptr;
		}
		const VkDevice vk_device = Vk();
		for (auto& s: shared) {
			if (s.image != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(vk_device, s.image, nullptr);
			if (s.memory != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(vk_device, s.memory, nullptr);
			s = {};
		}
		render  = {};
		display = {};
	}

	bool CreateShared(int index, vk::Extent2D extent) {
		auto& s = shared[index];
		D3D12_RESOURCE_DESC desc {};
		desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		desc.Width            = extent.width;
		desc.Height           = extent.height;
		desc.DepthOrArraySize = 1;
		desc.MipLevels        = 1;
		desc.Format           = kFormats[index].dxgi;
		desc.SampleDesc.Count = 1;
		desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		desc.Flags            = index == Output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
		D3D12_HEAP_PROPERTIES heap {};
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;
		if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
		                                           nullptr, IID_PPV_ARGS(&s.resource)))) {
			return Fail("CreateCommittedResource(shared)");
		}
		HANDLE handle = nullptr;
		if (FAILED(device->CreateSharedHandle(s.resource.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) {
			return Fail("CreateSharedHandle(image)");
		}
		const VkDevice vk_device = Vk();
		VkExternalMemoryImageCreateInfo external {};
		external.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
		VkImageCreateInfo info {};
		info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		info.pNext         = &external;
		info.imageType     = VK_IMAGE_TYPE_2D;
		info.format        = kFormats[index].vk;
		info.extent        = {extent.width, extent.height, 1};
		info.mipLevels     = 1;
		info.arrayLayers   = 1;
		info.samples       = VK_SAMPLE_COUNT_1_BIT;
		info.tiling        = VK_IMAGE_TILING_OPTIMAL;
		info.usage         = index == Output ? (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT)
		                                     : VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage(vk_device, &info, nullptr, &s.image) != VK_SUCCESS) {
			CloseHandle(handle);
			return Fail("vkCreateImage(external)");
		}
		VkMemoryRequirements requirements {};
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements(vk_device, s.image, &requirements);
		VkMemoryWin32HandlePropertiesKHR handle_properties {};
		handle_properties.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
		uint32_t type_bits      = requirements.memoryTypeBits;
		if (get_handle_properties(vk_device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, handle,
		                          &handle_properties) == VK_SUCCESS &&
		    (handle_properties.memoryTypeBits & type_bits) != 0) {
			type_bits &= handle_properties.memoryTypeBits;
		}
		VkPhysicalDeviceMemoryProperties memory_properties {};
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties(
		    static_cast<VkPhysicalDevice>(graphics.physical_device), &memory_properties);
		uint32_t type = UINT32_MAX;
		for (uint32_t i = 0; i < memory_properties.memoryTypeCount && type == UINT32_MAX; ++i) {
			if ((type_bits & (1u << i)) != 0 &&
			    (memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
				type = i;
			}
		}
		for (uint32_t i = 0; i < memory_properties.memoryTypeCount && type == UINT32_MAX; ++i) {
			if ((type_bits & (1u << i)) != 0) type = i;
		}
		if (type == UINT32_MAX) {
			CloseHandle(handle);
			return Fail("no memory type for the shared image");
		}
		VkMemoryDedicatedAllocateInfo dedicated {};
		dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
		dedicated.image = s.image;
		VkImportMemoryWin32HandleInfoKHR import {};
		import.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		import.pNext      = &dedicated;
		import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
		import.handle     = handle;
		VkMemoryAllocateInfo allocate {};
		allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate.pNext           = &import;
		allocate.allocationSize  = requirements.size;
		allocate.memoryTypeIndex = type;
		const VkResult allocated =
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(vk_device, &allocate, nullptr, &s.memory);
		CloseHandle(handle);
		if (allocated != VK_SUCCESS) {
			return Fail("vkAllocateMemory(import D3D12 resource)");
		}
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory(vk_device, s.image, s.memory, 0) != VK_SUCCESS) {
			return Fail("vkBindImageMemory(shared)");
		}
		return true;
	}

	bool Ensure(vk::Extent2D new_render, vk::Extent2D new_display) {
		if (failed) return false;
		if (render == new_render && display == new_display && context != nullptr) return true;
		Release();
		for (int i = 0; i < Count; ++i) {
			if (!CreateShared(i, i == Output ? new_display : new_render)) return false;
		}
		ffxCreateBackendDX12Desc backend {};
		backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
		backend.device      = device.Get();
		ffxCreateContextDescUpscale desc {};
		desc.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		desc.header.pNext   = &backend.header;
		ffxOverrideVersion version_override {};
		if (override_id != 0) {
			version_override.header.type  = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
			version_override.header.pNext = &backend.header;
			version_override.versionId    = override_id;
			desc.header.pNext             = &version_override.header;
		}
		desc.flags          = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
		             (GetTuning().auto_exposure ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0u) |
		             (GetTuning().debug_view ? FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION : 0u) |
		             (GetTuning().debug_check ? FFX_UPSCALE_ENABLE_DEBUG_CHECKING : 0u);
		desc.maxRenderSize  = {new_render.width, new_render.height};
		desc.maxUpscaleSize = {new_display.width, new_display.height};
		desc.fpMessage      = FfxMessage;
		const auto created  = fns.CreateContext(&context, &desc.header, nullptr);
		if (created != FFX_API_RETURN_OK || context == nullptr) {
			context = nullptr;
			return Fail("ffxCreateContext(upscale) returned " + std::to_string(created));
		}
		if (GetTuning().debug_check && fns.Configure != nullptr) {
			ffxConfigureDescGlobalDebug1 debug {};
			debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
			debug.fpMessage   = FfxMessage;
			debug.debugLevel  = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;
			fns.Configure(&context, &debug.header);
		}
		ffxQueryGetProviderVersion provider {};
		provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
		fns.Query(&context, &provider.header);
		status = std::string("FSR 4 bridge: ") + (provider.versionName != nullptr ? provider.versionName : "?");
		std::printf("NHL27FSR4 status=created provider=%s render=%ux%u display=%ux%u\n",
		            provider.versionName != nullptr ? provider.versionName : "?", new_render.width,
		            new_render.height, new_display.width, new_display.height);
		// Which inputs this upscaler needs / uses when given (bit 4 = reactive mask, 5 = transparency).
		ffxQueryDescUpscaleGetResourceRequirements needs {};
		needs.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS;
		if (fns.Query(&context, &needs.header) == FFX_API_RETURN_OK) {
			std::printf("NHL27FSR4 resources required=0x%llx optional=0x%llx\n",
			            static_cast<unsigned long long>(needs.required_resources),
			            static_cast<unsigned long long>(needs.optional_resources));
		}
		std::fflush(stdout);
		render  = new_render;
		display = new_display;
		fresh   = true;
		return true;
	}

	Result Dispatch(const Fsr4Params& p, uint64_t wait_value, uint64_t signal_value) {
		if (failed || context == nullptr) return Result::NotQueued;
		auto& allocator = allocators[ring];
		if (allocator_value[ring] != 0 && fence->GetCompletedValue() < allocator_value[ring]) {
			fence->SetEventOnCompletion(allocator_value[ring], event);
			WaitForSingleObject(event, 5000);
		}
		if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.Get(), nullptr))) {
			Fail("command list reset");
			return Result::NotQueued;
		}
		// FSR 4 expects its inputs in NON_PIXEL_SHADER_RESOURCE and the output as a UAV; between frames
		// the shared images stay in COMMON, which Vulkan sees as VK_IMAGE_LAYOUT_GENERAL.
		const auto transition = [this](bool to_ffx) {
			std::array<D3D12_RESOURCE_BARRIER, Count> barriers {};
			for (int i = 0; i < Count; ++i) {
				const auto used = i == Output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
				                              : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
				auto& b                  = barriers[i];
				b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
				b.Transition.pResource   = shared[i].resource.Get();
				b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
				b.Transition.StateBefore = to_ffx ? D3D12_RESOURCE_STATE_COMMON : used;
				b.Transition.StateAfter  = to_ffx ? used : D3D12_RESOURCE_STATE_COMMON;
			}
			list->ResourceBarrier(Count, barriers.data());
		};
		transition(true);
		const auto& tuning  = GetTuning();
		ffxDispatchDescUpscale d {};
		d.header.type       = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
		d.commandList       = list.Get();
		d.color             = ffxApiGetResourceDX12(shared[Color].resource.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
		d.depth             = ffxApiGetResourceDX12(shared[Depth].resource.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
		d.motionVectors     = ffxApiGetResourceDX12(shared[Motion].resource.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
		if (p.reactive) {
			d.reactive = ffxApiGetResourceDX12(shared[Reactive].resource.Get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
		}
		d.output            = ffxApiGetResourceDX12(shared[Output].resource.Get(),
		                                            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
		d.jitterOffset      = {p.jitter_x * tuning.jitter_x, p.jitter_y * tuning.jitter_y};
		d.motionVectorScale = {p.motion_scale_x * tuning.mv_x, p.motion_scale_y * tuning.mv_y};
		d.renderSize        = {render.width, render.height};
		d.upscaleSize       = {display.width, display.height};
		d.enableSharpening  = false;
		d.sharpness         = 0;
		d.frameTimeDelta    = p.delta_ms;
		d.preExposure       = tuning.unit_pre_exposure ? 1.0f : p.pre_exposure;
		d.reset             = p.reset;
		// Inverted depth: near and far swap, as for FSR 3.1.
		d.cameraNear              = tuning.depth_noswap ? p.near_plane : p.far_plane;
		d.cameraFar               = tuning.depth_noswap ? p.far_plane : p.near_plane;
		d.cameraFovAngleVertical  = p.vertical_fov;
		d.viewSpaceToMetersFactor = tuning.meters;
		d.flags                   = tuning.debug_view ? FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW : 0u;
		const auto result         = fns.Dispatch(&context, &d.header);
		transition(false);
		if (FAILED(list->Close())) {
			Fail("command list close");
			return Result::NotQueued;
		}
		// Even when FSR 4 refused the frame, the Vulkan side waits for signal_value: always signal it.
		queue->Wait(fence.Get(), wait_value);
		if (result == FFX_API_RETURN_OK) {
			ID3D12CommandList* lists[] = {list.Get()};
			queue->ExecuteCommandLists(1, lists);
		}
		queue->Signal(fence.Get(), signal_value);
		allocator_value[ring] = signal_value;
		ring                  = (ring + 1) % Ring;
		last_signal           = signal_value;
		if (result != FFX_API_RETURN_OK) {
			Fail("ffxDispatch(upscale) returned " + std::to_string(result));
			return Result::Failed;
		}
		if (++dispatched <= 24 || dispatched % 600 == 0) {
			std::printf("NHL27FSR4 status=dispatched count=%llu wait=%llu signal=%llu reset=%d jitter=%.4f,%.4f "
			            "mv_scale=%.2f,%.2f near=%g far=%g fov=%.4f dt_ms=%.3f pre_exposure=%g reactive=%d\n",
			            static_cast<unsigned long long>(dispatched), static_cast<unsigned long long>(wait_value),
			            static_cast<unsigned long long>(signal_value), p.reset ? 1 : 0, double(d.jitterOffset.x),
			            double(d.jitterOffset.y), double(d.motionVectorScale.x), double(d.motionVectorScale.y),
			            double(d.cameraNear), double(d.cameraFar), double(d.cameraFovAngleVertical),
			            double(d.frameTimeDelta), double(d.preExposure), p.reactive ? 1 : 0);
			std::fflush(stdout);
		}
		return Result::Upscaled;
	}

	void Shutdown() {
		Release();
		if (semaphore != VK_NULL_HANDLE) {
			VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroySemaphore(Vk(), semaphore, nullptr);
			semaphore = VK_NULL_HANDLE;
		}
		if (event != nullptr) CloseHandle(event);
		event = nullptr;
		list.Reset();
		for (auto& allocator: allocators) allocator.Reset();
		queue.Reset();
		fence.Reset();
		device.Reset();
		// The loader DLL stays loaded: FSR's DLLs may keep threads or callbacks around.
	}
#endif
};

Fsr4Bridge::Fsr4Bridge(GraphicContext& graphics): m(std::make_unique<Impl>(graphics)) {
#if KYTY_FSR4_BRIDGE
	m->Init();
#else
	m->Fail("build without KYTY_ENABLE_FSR4 (Windows only)");
#endif
}

Fsr4Bridge::~Fsr4Bridge() {
#if KYTY_FSR4_BRIDGE
	m->Shutdown();
#endif
}

bool Fsr4Bridge::Usable() const {
	return !m->failed;
}

const std::string& Fsr4Bridge::Status() const {
	return m->status;
}

bool Fsr4Bridge::NeedsRecreate(vk::Extent2D render, vk::Extent2D display) const {
#if KYTY_FSR4_BRIDGE
	return m->context == nullptr || m->render != render || m->display != display;
#else
	(void)render;
	(void)display;
	return false;
#endif
}

bool Fsr4Bridge::Ensure(vk::Extent2D render, vk::Extent2D display) {
#if KYTY_FSR4_BRIDGE
	return m->Ensure(render, display);
#else
	(void)render;
	(void)display;
	return false;
#endif
}

bool Fsr4Bridge::Fresh() const {
	return m->fresh;
}

void Fsr4Bridge::ClearFresh() {
	m->fresh = false;
}

vk::Image Fsr4Bridge::Image(int index) const {
#if KYTY_FSR4_BRIDGE
	return index >= 0 && index < Count ? vk::Image(m->shared[index].image) : vk::Image {};
#else
	(void)index;
	return {};
#endif
}

vk::Semaphore Fsr4Bridge::Semaphore() const {
#if KYTY_FSR4_BRIDGE
	return vk::Semaphore(m->semaphore);
#else
	return {};
#endif
}

std::pair<uint64_t, uint64_t> Fsr4Bridge::NextValues() {
	m->value += 2;
	return {m->value - 1, m->value};
}

Fsr4Bridge::Result Fsr4Bridge::Dispatch(const Fsr4Params& params, uint64_t wait_value, uint64_t signal_value) {
#if KYTY_FSR4_BRIDGE
	return m->Dispatch(params, wait_value, signal_value);
#else
	(void)params;
	(void)wait_value;
	(void)signal_value;
	return Result::NotQueued;
#endif
}

} // namespace Libs::Graphics
