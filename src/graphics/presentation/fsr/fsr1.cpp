#include "graphics/presentation/fsr/fsr1.h"

#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_fsr1_easu_spv.h"
#include "gpu_blit_shaders/gpu_fsr1_rcas_spv.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace Libs::Graphics {
namespace {
struct Settings {
	vk::Extent2D input, output;
	float sharpness;
};
static_assert(sizeof(Settings) == 20);
constexpr auto FORMAT = vk::Format::eR8G8B8A8Unorm;
} // namespace

struct Fsr1::Impl {
	vk::PhysicalDevice physical;
	vk::Device device;
	vk::Extent2D extent {};
	vk::DescriptorSetLayout descriptors;
	vk::PipelineLayout layout;
	vk::Sampler sampler;
	std::array<vk::Pipeline, 2> pipelines {};
	struct Image {
		vk::Image handle;
		vk::DeviceMemory memory;
		vk::ImageView view;
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	};
	std::array<Image, 2> images;

	void DestroyImages() {
		for (auto& image : images) {
			device.destroyImageView(image.view);
			device.destroyImage(image.handle);
			device.freeMemory(image.memory);
			image = {};
		}
		extent = {};
	}
	bool Create(vk::Extent2D target) {
		const auto required = vk::FormatFeatureFlagBits::eColorAttachment |
							  vk::FormatFeatureFlagBits::eSampledImage |
							  vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
							  vk::FormatFeatureFlagBits::eBlitSrc;
		if ((physical.getFormatProperties(FORMAT).optimalTilingFeatures & required) != required)
			return false;
		if (extent != target) {
			DestroyImages();
			vk::ImageCreateInfo info {};
			info.imageType = vk::ImageType::e2D;
			info.format = FORMAT;
			info.extent = {target.width, target.height, 1};
			info.mipLevels = info.arrayLayers = 1;
			info.samples = vk::SampleCountFlagBits::e1;
			info.usage = vk::ImageUsageFlagBits::eColorAttachment |
						 vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;
			for (auto& image : images) {
				if (device.createImage(&info, nullptr, &image.handle) != vk::Result::eSuccess)
					return false;
				const auto requirements = device.getImageMemoryRequirements(image.handle);
				const auto properties = physical.getMemoryProperties();
				uint32_t type = 0;
				for (; type < properties.memoryTypeCount; ++type)
					if ((requirements.memoryTypeBits & (1u << type)) &&
						(properties.memoryTypes[type].propertyFlags &
						 vk::MemoryPropertyFlagBits::eDeviceLocal))
						break;
				if (type == properties.memoryTypeCount)
					return false;
				vk::MemoryAllocateInfo allocation {};
				allocation.allocationSize = requirements.size;
				allocation.memoryTypeIndex = type;
				if (device.allocateMemory(&allocation, nullptr, &image.memory) !=
						vk::Result::eSuccess ||
					device.bindImageMemory(image.handle, image.memory, 0) != vk::Result::eSuccess)
					return false;
				vk::ImageViewCreateInfo view {};
				view.image = image.handle;
				view.viewType = vk::ImageViewType::e2D;
				view.format = FORMAT;
				view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
				if (device.createImageView(&view, nullptr, &image.view) != vk::Result::eSuccess)
					return false;
			}
			extent = target;
		}
		if (pipelines[1])
			return true;
		if (!layout) {
			const vk::DescriptorSetLayoutBinding binding {0,
														  vk::DescriptorType::eCombinedImageSampler,
														  1, vk::ShaderStageFlagBits::eFragment};
			vk::DescriptorSetLayoutCreateInfo desc {};
			desc.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
			desc.bindingCount = 1;
			desc.pBindings = &binding;
			if (device.createDescriptorSetLayout(&desc, nullptr, &descriptors) !=
				vk::Result::eSuccess)
				return false;
			const vk::PushConstantRange constants {vk::ShaderStageFlagBits::eFragment, 0,
												   sizeof(Settings)};
			vk::PipelineLayoutCreateInfo pipeline_layout {};
			pipeline_layout.setLayoutCount = pipeline_layout.pushConstantRangeCount = 1;
			pipeline_layout.pSetLayouts = &descriptors;
			pipeline_layout.pPushConstantRanges = &constants;
			if (device.createPipelineLayout(&pipeline_layout, nullptr, &layout) !=
				vk::Result::eSuccess)
				return false;
			vk::SamplerCreateInfo sample {};
			sample.magFilter = sample.minFilter = vk::Filter::eLinear;
			sample.addressModeU = sample.addressModeV = sample.addressModeW =
				vk::SamplerAddressMode::eClampToEdge;
			if (device.createSampler(&sample, nullptr, &sampler) != vk::Result::eSuccess)
				return false;
		}
		for (uint32_t i = 0; i < pipelines.size(); ++i) {
			if (pipelines[i])
				continue;
			auto module = [&](const auto& code, vk::ShaderModule& out) {
				vk::ShaderModuleCreateInfo create {};
				create.codeSize = sizeof(code);
				create.pCode = code;
				return device.createShaderModule(&create, nullptr, &out) == vk::Result::eSuccess;
			};
			vk::ShaderModule vertex, fragment;
			if (!module(GPU_BLIT_FS_TRIANGLE_SPV, vertex))
				return false;
			const bool compiled =
				i == 0 ? module(GPU_FSR1_EASU_SPV, fragment) : module(GPU_FSR1_RCAS_SPV, fragment);
			if (!compiled) {
				device.destroyShaderModule(vertex);
				return false;
			}
			std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
			stages[0].stage = vk::ShaderStageFlagBits::eVertex;
			stages[0].module = vertex;
			stages[0].pName = stages[1].pName = "main";
			stages[1].stage = vk::ShaderStageFlagBits::eFragment;
			stages[1].module = fragment;
			vk::PipelineVertexInputStateCreateInfo input {};
			vk::PipelineInputAssemblyStateCreateInfo assembly {};
			assembly.topology = vk::PrimitiveTopology::eTriangleList;
			vk::PipelineViewportStateCreateInfo viewport {};
			viewport.viewportCount = viewport.scissorCount = 1;
			vk::PipelineRasterizationStateCreateInfo raster {};
			raster.lineWidth = 1.0f;
			vk::PipelineMultisampleStateCreateInfo samples {};
			samples.rasterizationSamples = vk::SampleCountFlagBits::e1;
			vk::PipelineColorBlendAttachmentState attachment {};
			attachment.colorWriteMask =
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
			vk::PipelineColorBlendStateCreateInfo blend {};
			blend.attachmentCount = 1;
			blend.pAttachments = &attachment;
			const std::array states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
			vk::PipelineDynamicStateCreateInfo dynamic {};
			dynamic.dynamicStateCount = static_cast<uint32_t>(states.size());
			dynamic.pDynamicStates = states.data();
			vk::PipelineRenderingCreateInfo rendering {};
			rendering.colorAttachmentCount = 1;
			rendering.pColorAttachmentFormats = &FORMAT;
			vk::GraphicsPipelineCreateInfo create {};
			create.pNext = &rendering;
			create.stageCount = static_cast<uint32_t>(stages.size());
			create.pStages = stages.data();
			create.pVertexInputState = &input;
			create.pInputAssemblyState = &assembly;
			create.pViewportState = &viewport;
			create.pRasterizationState = &raster;
			create.pMultisampleState = &samples;
			create.pColorBlendState = &blend;
			create.pDynamicState = &dynamic;
			create.layout = layout;
			const auto result =
				device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipelines[i]);
			device.destroyShaderModule(fragment);
			device.destroyShaderModule(vertex);
			if (result != vk::Result::eSuccess)
				return false;
		}
		return true;
	}
	void Transit(vk::CommandBuffer command, Image& image, vk::ImageLayout next) {
		vk::ImageMemoryBarrier barrier {};
		barrier.srcAccessMask =
			image.layout == vk::ImageLayout::eUndefined
				? vk::AccessFlags {}
				: vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		barrier.oldLayout = image.layout;
		barrier.newLayout = next;
		barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image.handle;
		barrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
								vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 0, nullptr,
								1, &barrier);
		image.layout = next;
	}
	void Pass(vk::CommandBuffer command, vk::ImageView source, uint32_t index,
			  const Settings& settings) {
		auto& image = images[index];
		Transit(command, image, vk::ImageLayout::eColorAttachmentOptimal);
		const vk::DescriptorImageInfo descriptor {sampler, source,
												  vk::ImageLayout::eShaderReadOnlyOptimal};
		vk::WriteDescriptorSet write {};
		write.descriptorCount = 1;
		write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
		write.pImageInfo = &descriptor;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &write);
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines[index]);
		command.pushConstants(layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(Settings),
							  &settings);
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView = image.view;
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp = vk::AttachmentLoadOp::eDontCare;
		attachment.storeOp = vk::AttachmentStoreOp::eStore;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = extent;
		rendering.layerCount = rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments = &attachment;
		command.beginRendering(&rendering);
		const vk::Viewport viewport {0, 0, float(extent.width), float(extent.height), 0, 1};
		const vk::Rect2D scissor {{0, 0}, extent};
		command.setViewport(0, 1, &viewport);
		command.setScissor(0, 1, &scissor);
		command.draw(3, 1, 0, 0);
		command.endRendering();
	}
};

Fsr1::Fsr1(vk::PhysicalDevice physical, vk::Device device) : m(std::make_unique<Impl>()) {
	m->physical = physical;
	m->device = device;
}
Fsr1::~Fsr1() {
	m->DestroyImages();
	for (auto pipeline : m->pipelines)
		m->device.destroyPipeline(pipeline);
	m->device.destroySampler(m->sampler);
	m->device.destroyPipelineLayout(m->layout);
	m->device.destroyDescriptorSetLayout(m->descriptors);
}
vk::Image Fsr1::Record(vk::CommandBuffer command, vk::ImageView source, vk::Extent2D input,
					   vk::Extent2D output, float sharpness) {
	if (!source || !command || !std::isfinite(sharpness) || !input.width || !input.height ||
		output.width < input.width || output.height < input.height || !m->Create(output))
		return nullptr;
	const Settings settings {input, output, std::clamp(sharpness, 0.0f, 1.0f)};
	if (input != output) {
		m->Pass(command, source, 0, settings);
		m->Transit(command, m->images[0], vk::ImageLayout::eShaderReadOnlyOptimal);
		source = m->images[0].view;
	}
	m->Pass(command, source, 1, settings);
	m->Transit(command, m->images[1], vk::ImageLayout::eTransferSrcOptimal);
	return m->images[1].handle;
}
} // namespace Libs::Graphics
