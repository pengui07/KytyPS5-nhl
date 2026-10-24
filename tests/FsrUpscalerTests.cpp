// Exercises the production FSR implementations on the active Vulkan GPU. Synthetic guides are
// confined to this test; passing this does not certify any game's motion or camera capture.
#include "common/runtimeFlags.h"
#include "graphics/presentation/fsr/fsr1.h"
#include "graphics/presentation/fsr/fsr2Constants.h"
#include "graphics/presentation/fsr/fsr3.h"
#include "graphics/presentation/fsr/nhl27Fsr3Constants.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
using namespace Libs::Graphics;

static void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FSR test FAILED: %s\n", message);
		std::exit(1);
	}
}
static void Ok(vk::Result result) {
	Check(result == vk::Result::eSuccess, vk::to_string(result).c_str());
}

int main() {
	{
		// Captured Free Skate frame 62, not the embedded public FSR2 layout.
		const std::array<uint32_t, 52> words {0xbcf0f0f1u,0x41a4b4b5u,0x3e898ec1u,0x3e1ac098u,0x0000092cu,0x00000528u,0x0000092cu,0x00000528u,0x00000dc0u,0x000007bcu,0x00000dc0u,0x000007bcu,0x4512c000u,0x44a50000u,0x4512c000u,0x44a50000u,0x455c0000u,0x44f78000u,0x455c0000u,0x44f78000u,0x39df4a93u,0x3a46980cu,0x39df4a93u,0x3a46980cu,0x3994f209u,0x3a04655eu,0x3994f209u,0x3a04655eu,0x3ea00000u,0xbeb425ecu,0xbea00000u,0x3e5097b4u,0xbf000000u,0x3f000000u,0x3f2ac37eu,0x3f2aaaabu,0x3d1ee852u,0x3f800000u,0x3f800000u,0x0000003eu,0x00000000u,0x41200000u,0xbdcccccdu,0x40000000u,0x00000000u,0x3d23d70au,0xbe4cccccu,0x3d4ccccdu,0x3ecccccdu,0xffffffffu,0xffffffffu,0xffffffffu};
		const auto captured = Nhl27Fsr3Constants::Decode(words);
		Check(captured && captured->Frame() == 62 && captured->Width() == 2348 &&
			captured->OutputWidth() == 3520, "decode actual Frostbite GPU constant fixture");
		Check(std::abs(captured->Near() - 20) < 0.001f &&
			std::abs(captured->Far() - 700) < 0.01f, "captured depth coefficient conversion");
		auto changed = *captured;
		changed.words[49] ^= 1;
		Check(captured->SameFrame(changed), "ignore pass-specific constant tail");
		changed.words[39]++;
		Check(!captured->SameFrame(changed), "reject mismatched temporal frame");
		changed = *captured;
		changed.words[3] = 0x7fc00000;
		Check(!Nhl27Fsr3Constants::Decode(changed.words), "reject nonfinite captured camera");
		changed = *captured;
		changed.words[10]++;
		Check(!Nhl27Fsr3Constants::Decode(changed.words), "reject unobserved dynamic resource layout");
		for (size_t i = 20; i < 28; ++i) {
			changed = *captured;
			changed.words[i] = 0x7fc00000;
			Check(!Nhl27Fsr3Constants::Decode(changed.words), "reject nonfinite reciprocal extent");
		}
		{
			const auto prev = Nhl27HistoryState::From(*captured);
			auto next = *captured;
			next.words[39]++;
			next.words[30] = captured->words[28]; next.words[31] = captured->words[29];
			auto cur = Nhl27HistoryState::From(next);
			using R = Nhl27ResetReason;
			Check(ShouldResetHistory(&prev, cur, false) == R::None, "history continues on consecutive frame");
			Check(ShouldResetHistory(nullptr, cur, false) == R::First, "reset on first frame");
			Check(ShouldResetHistory(&prev, cur, true) == R::MissedPass, "reset after missed pass");
			Check(ShouldResetHistory(&prev, cur, false, false) == R::Disabled, "history switch off resets");
			auto bad = cur; bad.frame += 1;
			Check(ShouldResetHistory(&prev, bad, false) == R::FrameDiscontinuity, "reset on frame skip");
			bad = cur; bad.frame = prev.frame;
			Check(ShouldResetHistory(&prev, bad, false) == R::FrameDiscontinuity, "reset on repeated frame");
			bad = cur; bad.frame = prev.frame - 5;
			Check(ShouldResetHistory(&prev, bad, false) == R::FrameDiscontinuity, "reset on backwards frame");
			bad = cur; bad.output_width += 2;
			Check(ShouldResetHistory(&prev, bad, false) == R::SizeChange, "reset on output size change");
			bad = cur; bad.height -= 8;
			Check(ShouldResetHistory(&prev, bad, false) == R::SizeChange, "reset on render size change");
			bad = cur; bad.far_plane *= 1.5f;
			Check(ShouldResetHistory(&prev, bad, false) == R::CameraCut, "reset on far change");
			bad = cur; bad.near_plane *= 2;
			Check(ShouldResetHistory(&prev, bad, false) == R::CameraCut, "reset on near change");
			bad = cur; bad.fov *= 1.1f;
			Check(ShouldResetHistory(&prev, bad, false) == R::CameraCut, "reset on fov change");
			bad = cur; bad.fov *= 1.001f;
			Check(ShouldResetHistory(&prev, bad, false) == R::None, "tolerate tiny fov drift");
			bad = cur; bad.prev_jitter_x += 0.01f;
			Check(ShouldResetHistory(&prev, bad, false) == R::JitterMismatch, "reset on previous jitter mismatch");
			bad = cur; bad.prev_jitter_y = std::nanf("");
			Check(ShouldResetHistory(&prev, bad, false) == R::JitterMismatch, "reset on nonfinite jitter");
		}
		std::puts("NHL27 FSR3 constants PASS: captured GPU fixture and fail-closed checks");
	}
	{
		int observed = 0;
		auto* destination = &observed;
		CommandRecorder direct {vk::CommandBuffer {}};
		direct.RecordExternal([destination](vk::CommandBuffer) { *destination = 1; });
		Check(observed == 1, "external callback records immediately in direct mode");
		CommandStream stream;
		CommandRecorder deferred {&stream};
		int captured_value = 7;
		deferred.RecordExternal([destination, captured_value](vk::CommandBuffer) {
			*destination += captured_value;
		});
		captured_value = 99;
		Check(observed == 1, "external callback waits for replay in deferred mode");
		auto chunks = stream.TakeChunks();
		for (const auto& chunk : chunks)
			CommandStream::Replay(*chunk, [] { return vk::CommandBuffer {}; },
				[](const CommandStream::Entry&) {});
		Check(observed == 8, "external callback retains copied payload until replay");
		std::puts("External GPU callback PASS: direct/deferred order and payload lifetime");
	}
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	{
		Fsr2ConstantsCandidate cb;
		for (unsigned i = 0; i < 8; ++i)
			cb.words[i] = i == 4 || i == 5 ? 64 : 32;
		auto set = [&](unsigned offset, float value) {
			cb.words[offset / 4] = std::bit_cast<uint32_t>(value);
		};
		set(72, -1);
		set(76, -1);
		set(80, 0.5f);
		set(84, 0.5f);
		set(96, 1);
		set(100, 1);
		set(104, 1);
		set(108, 8);
		set(112, 0.016f);
		set(120, 1);
		Check(cb.Plausible(), "FSR2 reflected-layout candidate");
		set(80, 0.75f);
		Check(!cb.Plausible(), "FSR2 reject inconsistent downscale");
		set(80, 0.5f);
		cb.words[4] = 0;
		Check(!cb.Plausible(), "FSR2 reject zero display size");
		std::puts(
			"FSR2 probe PASS: reflected-layout decoder and rejection checks (heuristic only)");
	}
	{
		using namespace Common::RuntimeFlags;
		Check(!FlagSet {}.fsr1, "FSR1 is off by default");
		FlagSet flags {};
		auto* field = FieldFor(flags, "KYTY_FSR1");
		Check(field != nullptr, "FSR1 runtime parser mapping");
		*field = true;
		Check((Bits(flags) ^ Bits(FlagSet {})) == (uint64_t {1} << 36), "FSR1 uses bit 36 only");
		const auto previous = g_published.load();
		g_published.store(Bits(flags) | (uint64_t {7} << kRevisionShift));
		Check(Current().fsr1 && Revision() == 7, "FSR1 decode preserves revision");
		g_published.store(previous);
	}
	vk::detail::DynamicLoader loader;
	VULKAN_HPP_DEFAULT_DISPATCHER.init(
		loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo app {};
	app.pApplicationName = "Kyty FSR backend tests";
	app.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info {};
	instance_info.pApplicationInfo = &app;
	const char* instance_extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
	instance_info.enabledExtensionCount = 1;
	instance_info.ppEnabledExtensionNames = instance_extensions;
	vk::Instance instance;
	Ok(vk::createInstance(&instance_info, nullptr, &instance));
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
	uint32_t count = 0;
	Ok(instance.enumeratePhysicalDevices(&count, nullptr));
	Check(count > 0, "no GPU");
	std::vector<vk::PhysicalDevice> devices(count);
	Ok(instance.enumeratePhysicalDevices(&count, devices.data()));
	auto physical = devices[0];
	for (auto candidate : devices)
		if (candidate.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
			physical = candidate;
			break;
		}
	std::printf("GPU: %s\n", physical.getProperties().deviceName.data());
	const auto families = physical.getQueueFamilyProperties();
	uint32_t family = 0;
	for (; family < families.size(); ++family)
		if ((families[family].queueFlags &
			 (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) ==
			(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))
			break;
	Check(family < families.size(), "no graphics/compute queue");
	const float priority = 1;
	vk::DeviceQueueCreateInfo queue_info {};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount = 1;
	queue_info.pQueuePriorities = &priority;
	auto features =
		physical.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
							  vk::PhysicalDeviceVulkan13Features>();
	auto f12 = features.get<vk::PhysicalDeviceVulkan12Features>();
	auto f13 = features.get<vk::PhysicalDeviceVulkan13Features>();
	Check(f13.dynamicRendering, "dynamic rendering unavailable");
	f12.pNext = &f13;
	f13.pNext = nullptr;
	const char* extensions[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
								VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
								VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME};
	vk::DeviceCreateInfo device_info {};
	device_info.pNext = &f12;
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos = &queue_info;
	device_info.enabledExtensionCount = 3;
	device_info.ppEnabledExtensionNames = extensions;
	device_info.pEnabledFeatures = &features.get<vk::PhysicalDeviceFeatures2>().features;
	vk::Device device;
	Ok(physical.createDevice(&device_info, nullptr, &device));
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
	const auto queue = device.getQueue(family, 0);
	const auto memory_properties = physical.getMemoryProperties();
	auto allocate = [&](vk::MemoryRequirements requirements, vk::MemoryPropertyFlags flags) {
		uint32_t index = 0;
		for (; index < memory_properties.memoryTypeCount; ++index)
			if ((requirements.memoryTypeBits & (1u << index)) &&
				(memory_properties.memoryTypes[index].propertyFlags & flags) == flags)
				break;
		Check(index < memory_properties.memoryTypeCount, "memory type");
		vk::MemoryAllocateInfo info {};
		info.allocationSize = requirements.size;
		info.memoryTypeIndex = index;
		vk::DeviceMemory memory;
		Ok(device.allocateMemory(&info, nullptr, &memory));
		return memory;
	};
	struct Image {
		Fsr3Image resource;
		vk::ImageView view;
		vk::DeviceMemory memory;
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	};
	std::vector<Image> images;
	auto image = [&](vk::Format format, vk::Extent2D extent) {
		Image result {};
		result.resource.format = format;
		result.resource.extent = extent;
		vk::ImageCreateInfo info {};
		info.imageType = vk::ImageType::e2D;
		info.format = format;
		info.extent = {extent.width, extent.height, 1};
		info.mipLevels = info.arrayLayers = 1;
		info.samples = vk::SampleCountFlagBits::e1;
		info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
					 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
		Ok(device.createImage(&info, nullptr, &result.resource.image));
		result.memory = allocate(device.getImageMemoryRequirements(result.resource.image),
								 vk::MemoryPropertyFlagBits::eDeviceLocal);
		Ok(device.bindImageMemory(result.resource.image, result.memory, 0));
		vk::ImageViewCreateInfo view {};
		view.image = result.resource.image;
		view.viewType = vk::ImageViewType::e2D;
		view.format = format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		Ok(device.createImageView(&view, nullptr, &result.view));
		images.push_back(result);
		return result;
	};
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = family;
	vk::CommandPool pool;
	Ok(device.createCommandPool(&pool_info, nullptr, &pool));
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool = pool;
	command_info.level = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command;
	Ok(device.allocateCommandBuffers(&command_info, &command));
	auto begin = [&] {
		Ok(command.reset());
		vk::CommandBufferBeginInfo info {};
		Ok(command.begin(&info));
	};
	auto submit = [&] {
		Ok(command.end());
		vk::SubmitInfo info {};
		info.commandBufferCount = 1;
		info.pCommandBuffers = &command;
		Ok(queue.submit(1, &info, nullptr));
		Ok(queue.waitIdle());
	};
	auto barrier = [&](vk::Image handle, vk::ImageLayout before, vk::ImageLayout after) {
		vk::ImageMemoryBarrier b {};
		b.srcAccessMask = before == vk::ImageLayout::eUndefined
							  ? vk::AccessFlags {}
							  : vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		b.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		b.oldLayout = before;
		b.newLayout = after;
		b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image = handle;
		b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
								vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 0, nullptr,
								1, &b);
	};
	auto clear = [&](Image& im, std::array<float, 4> value) {
		barrier(im.resource.image, im.layout, vk::ImageLayout::eTransferDstOptimal);
		const vk::ClearColorValue color {value};
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(im.resource.image, vk::ImageLayout::eTransferDstOptimal, &color, 1,
								&range);
		barrier(im.resource.image, vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::eShaderReadOnlyOptimal);
		im.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	};
	const char* captured_size = std::getenv("KYTY_FSR3_TEST_CAPTURE_SIZE");
	const bool large = captured_size && std::strcmp(captured_size, "1") == 0;
	const vk::Extent2D input = large ? vk::Extent2D {2348, 1320} : vk::Extent2D {32, 24};
	const vk::Extent2D output = large ? vk::Extent2D {3520, 1980} : vk::Extent2D {64, 48};
	std::printf("FSR test sizes: %ux%u -> %ux%u\n", input.width, input.height,
		output.width, output.height);
	vk::BufferCreateInfo read_info {};
	read_info.size = std::max<uint64_t>(128 * 96 * 4, uint64_t(output.width) * output.height * 8);
	read_info.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
	vk::Buffer readback;
	Ok(device.createBuffer(&read_info, nullptr, &readback));
	const auto read_memory = allocate(device.getBufferMemoryRequirements(readback),
									  vk::MemoryPropertyFlagBits::eHostVisible |
										  vk::MemoryPropertyFlagBits::eHostCoherent);
	Ok(device.bindBufferMemory(readback, read_memory, 0));
	auto verify = [&](vk::Image output, vk::Extent2D extent, vk::ImageLayout layout,
					  bool quadrants = false) {
		barrier(output, layout, vk::ImageLayout::eTransferSrcOptimal);
		vk::BufferImageCopy copy {};
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent = {extent.width, extent.height, 1};
		command.copyImageToBuffer(output, vk::ImageLayout::eTransferSrcOptimal, readback, 1, &copy);
		barrier(output, vk::ImageLayout::eTransferSrcOptimal, layout);
		submit();
		void* mapped = nullptr;
		Ok(device.mapMemory(read_memory, 0, VK_WHOLE_SIZE, {}, &mapped));
		const auto* pixels = static_cast<const uint8_t*>(mapped);
		for (uint32_t i = 0; i < extent.width * extent.height; ++i) {
			const auto x = i % extent.width, y = i / extent.width;
			if (quadrants && (std::abs(int(x) - int(extent.width / 2)) < 5 ||
							  std::abs(int(y) - int(extent.height / 2)) < 5))
				continue;
			const unsigned quadrant =
				(x >= extent.width / 2 ? 1 : 0) | (y >= extent.height / 2 ? 2 : 0);
			const std::array<std::array<int, 3>, 4> corners {
				{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}}};
			for (uint32_t c = 0; c < 3; ++c) {
				const auto expected =
					quadrants ? corners[quadrant][c] : std::array {64, 128, 191}[c];
				Check(std::abs(int(pixels[i * 4 + c]) - expected) <= 4,
					  "color reconstruction and orientation");
			}
		}
		device.unmapMemory(read_memory);
	};
	auto color = image(vk::Format::eR8G8B8A8Unorm, input);
	begin();
	clear(color, {0.25f, 0.5f, 0.75f, 1});
	submit();
	{
		Fsr1 fsr(physical, device);
		for (int i = 0; i < 3; ++i) {
			begin();
			const auto result = fsr.Record(command, color.view, input, output, 0.2f);
			Check(bool(result), "FSR1 EASU+RCAS");
			verify(result, output, vk::ImageLayout::eTransferSrcOptimal);
		}
		begin();
		Check(!fsr.Record(command, color.view, output, input, 0.2f), "reject downscale");
		const auto native = fsr.Record(command, color.view, input, input, 0.2f);
		Check(bool(native), "native RCAS and resize");
		verify(native, input, vk::ImageLayout::eTransferSrcOptimal);
		void* mapped = nullptr;
		Ok(device.mapMemory(read_memory, 0, VK_WHOLE_SIZE, {}, &mapped));
		auto* pixels = static_cast<uint8_t*>(mapped);
		for (uint32_t y = 0; y < input.height; ++y)
			for (uint32_t x = 0; x < input.width; ++x) {
				const unsigned quadrant =
					(x >= input.width / 2 ? 1 : 0) | (y >= input.height / 2 ? 2 : 0);
				const std::array<std::array<uint8_t, 4>, 4> corners {
					{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 255, 255}}};
				std::memcpy(pixels + (y * input.width + x) * 4, corners[quadrant].data(), 4);
			}
		device.unmapMemory(read_memory);
		begin();
		barrier(color.resource.image, color.layout, vk::ImageLayout::eTransferDstOptimal);
		vk::BufferImageCopy upload {};
		upload.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		upload.imageExtent = {input.width, input.height, 1};
		command.copyBufferToImage(readback, color.resource.image,
								  vk::ImageLayout::eTransferDstOptimal, 1, &upload);
		barrier(color.resource.image, vk::ImageLayout::eTransferDstOptimal, color.layout);
		submit();
		begin();
		const auto patterned = fsr.Record(command, color.view, input, output, 0.0f);
		Check(bool(patterned), "FSR1 patterned upscaling, zero sharpening");
		verify(patterned, output, vk::ImageLayout::eTransferSrcOptimal, true);
	}
	begin();
	clear(color, {0.25f, 0.5f, 0.75f, 1});
	submit();
	std::puts("FSR1 PASS: EASU+RCAS, reuse, native RCAS, resize, downscale rejection, quadrant "
			  "orientation and zero sharpening");
#if KYTY_ENABLE_FSR3
	auto depth = image(vk::Format::eR32Sfloat, input);
	auto motion = image(vk::Format::eR16G16Sfloat, input);
	auto upscaled = image(vk::Format::eR8G8B8A8Unorm, output);
	auto dilated_depth = image(vk::Format::eR32Sfloat, input);
	auto dilated_motion = image(vk::Format::eR16G16Sfloat, input);
	auto previous_depth = image(vk::Format::eR32Uint, input);
	begin();
	clear(depth, {0.5f, 0, 0, 0});
	clear(motion, {0, 0, 0, 0});
	for (auto* im : {&upscaled, &dilated_depth, &dilated_motion, &previous_depth})
		barrier(im->resource.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
	submit();
	{
		Fsr3 fsr(physical, device);
		Fsr3Frame frame {};
		frame.color = color.resource;
		frame.depth = depth.resource;
		frame.motion = motion.resource;
		frame.output = upscaled.resource;
		frame.dilated_depth = dilated_depth.resource;
		frame.dilated_motion = dilated_motion.resource;
		frame.previous_depth = previous_depth.resource;
		frame.render_size = input;
		frame.output_size = output;
		frame.near_plane = 0.1f;
		frame.far_plane = 1000;
		frame.vertical_fov = 1;
		frame.delta_ms = 16.6667f;
		frame.motion_scale_x = input.width;
		frame.motion_scale_y = input.height;
		begin();
		Check(!fsr.Record(command, frame), "reject unverified inputs");
		submit();
		frame.verified = true;
		for (uint64_t id = 1; id <= 3; ++id) {
			frame.frame_id = id;
			begin();
			const bool recorded = fsr.Record(command, frame);
			std::printf("%s\n", fsr.Status().c_str());
			Check(recorded, "FSR3 dispatch");
			verify(upscaled.resource.image, output, vk::ImageLayout::eGeneral);
		}
		begin();
		Check(!fsr.Record(command, frame), "reject repeated frame");
		frame.frame_id = 4;
		frame.motion.extent.width = 1;
		Check(!fsr.Record(command, frame), "reject mismatched motion");
		submit();
		frame.motion = motion.resource;
		frame.frame_id = 5;
		begin();
		Check(fsr.Record(command, frame), "recover with reset after missing/rejected frame");
		verify(upscaled.resource.image, output, vk::ImageLayout::eGeneral);
	}
	std::puts(
		"FSR3 PASS: three history frames plus recovery, readback checked; invalid inputs rejected");
	// Game captures use packed floating-point color and FP16 temporal output. Keep
	// this synthetic test separate from the game's still-unverified temporal provider.
	auto hdr_output = image(vk::Format::eR16G16B16A16Sfloat, output);
	begin();
	barrier(hdr_output.resource.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
	submit();
	auto set_hdr = [](const char* value) {
#if defined(_WIN32)
		Check(_putenv_s("KYTY_FSR3_HDR", value ? value : "") == 0, "set test HDR flag");
#else
		Check((value ? setenv("KYTY_FSR3_HDR", value, 1) : unsetenv("KYTY_FSR3_HDR")) == 0,
			  "set test HDR flag");
#endif
	};
	const char* old_hdr_value = std::getenv("KYTY_FSR3_HDR");
	const std::string old_hdr = old_hdr_value ? old_hdr_value : "";
	const bool had_hdr = old_hdr_value != nullptr;
	auto verify_hdr = [&] {
		barrier(hdr_output.resource.image, vk::ImageLayout::eGeneral,
				vk::ImageLayout::eTransferSrcOptimal);
		vk::BufferImageCopy copy {};
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent = {output.width, output.height, 1};
		command.copyImageToBuffer(hdr_output.resource.image, vk::ImageLayout::eTransferSrcOptimal,
								  readback, 1, &copy);
		barrier(hdr_output.resource.image, vk::ImageLayout::eTransferSrcOptimal,
				vk::ImageLayout::eGeneral);
		submit();
		void* mapped = nullptr;
		Ok(device.mapMemory(read_memory, 0, VK_WHOLE_SIZE, {}, &mapped));
		const auto* pixels = static_cast<const uint16_t*>(mapped);
		for (uint32_t i = 0; i < output.width * output.height; ++i) {
			for (uint32_t c = 0; c < 3; ++c) {
				const uint16_t half = pixels[i * 4 + c];
				const unsigned exponent = (half >> 10) & 31, mantissa = half & 1023;
				Check(exponent != 31 && !(half & 0x8000), "finite positive HDR reconstruction");
				const float value = exponent
					? std::ldexp(1.0f + float(mantissa) / 1024.0f, int(exponent) - 15)
					: std::ldexp(float(mantissa), -24);
				const float expected = std::array {2.0f, 4.0f, 8.0f}[c];
				if (std::abs(value - expected) > expected * 0.03f)
					std::printf("HDR mismatch pixel=%u channel=%u got=%g expected=%g\n",
								i, c, value, expected);
				Check(std::abs(value - expected) <= expected * 0.03f,
					  "HDR values above one survive reconstruction");
			}
		}
		device.unmapMemory(read_memory);
	};
	for (const auto format : {vk::Format::eB10G11R11UfloatPack32,
							  vk::Format::eR16G16B16A16Sfloat}) {
		auto hdr_color = image(format, input);
		begin();
		clear(hdr_color, {2, 4, 8, 1});
		submit();
		Fsr3 fsr(physical, device);
		Fsr3Frame frame {};
		frame.color = hdr_color.resource;
		frame.depth = depth.resource;
		frame.motion = motion.resource;
		frame.output = hdr_output.resource;
		frame.dilated_depth = dilated_depth.resource;
		frame.dilated_motion = dilated_motion.resource;
		frame.previous_depth = previous_depth.resource;
		frame.render_size = input;
		frame.output_size = output;
		frame.near_plane = 0.1f;
		frame.far_plane = 1000;
		frame.vertical_fov = 1;
		frame.delta_ms = 16.6667f;
		frame.motion_scale_x = -0.5f * input.width;
		frame.motion_scale_y = 0.5f * input.height;
		frame.hdr = true;
		frame.verified = true;
		frame.frame_id = 1;
		set_hdr(nullptr);
		begin();
		Check(!fsr.Record(command, frame) && fsr.Status().find("KYTY_FSR3_HDR") != std::string::npos,
			  "HDR requires explicit opt-in");
		submit();
		set_hdr("1");
		for (uint64_t id = 1; id <= 3; ++id) {
			frame.frame_id = id;
			begin();
			const bool recorded = fsr.Record(command, frame);
			std::printf("%s input=%s\n", fsr.Status().c_str(), vk::to_string(format).c_str());
			Check(recorded, "HDR FSR3 dispatch");
			verify_hdr();
		}
	}
	set_hdr(had_hdr ? old_hdr.c_str() : nullptr);
	std::puts("FSR3 HDR PASS: opt-in enforced, packed-float and FP16 inputs preserve 2/4/8 "
			  "through three history frames each into FP16 output");
#endif
	Ok(device.waitIdle());
	device.destroyBuffer(readback);
	device.freeMemory(read_memory);
	for (const auto& im : images) {
		device.destroyImageView(im.view);
		device.destroyImage(im.resource.image);
		device.freeMemory(im.memory);
	}
	device.destroyCommandPool(pool);
	device.destroy();
	instance.destroy();
	return 0;
}
