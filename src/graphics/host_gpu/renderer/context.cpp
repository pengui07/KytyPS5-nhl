#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/frameDump.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler, CommandStream* stream)
    : m_scheduler(scheduler), m_context(scheduler.Context()), m_graphics(scheduler.Graphics()),
      m_stream(stream) {}

void CommandBuffer::InvalidateDynamicState() const {
	m_scheduler.GetDynamicState().Invalidate();
}

bool CommandBuffer::IsInvalid() const {
	return m_stream != nullptr ? !m_open : m_buffer == nullptr;
}

CommandRecorder CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_stream != nullptr ? CommandRecorder(m_stream) : CommandRecorder(m_buffer);
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	if (m_stream != nullptr) {
		// The recording thread begins its Vulkan command buffer when it replays commands.
		return;
	}
	auto buffer = m_buffer;

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	if (m_stream != nullptr) {
		return;
	}
	auto buffer = m_buffer;

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

// Only the attachments in use: the defaulted compare went through all eight color attachments.
static bool SameRenderState(const RenderState& a, const RenderState& b) {
	if (a.num_color_attachments != b.num_color_attachments || a.width != b.width || a.height != b.height ||
	    a.num_layers != b.num_layers || !(a.depth_stencil_attachment == b.depth_stencil_attachment)) {
		return false;
	}
	for (uint32_t i = 0; i < a.num_color_attachments && i < a.color_attachments.size(); i++) {
		if (!(a.color_attachments[i] == b.color_attachments[i])) {
			return false;
		}
	}
	return true;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && SameRenderState(m_render_state, state)) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_RENDER_PASS);
	EndRendering();
	BufferCache::PerfCount(BufferCache::PerfEvent::RenderPassBegin);

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state = state;
	for (auto& color: m_render_state.color_attachments) {
		color.is_clear = false;
	}
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_RENDER_PASS);
	Handle().endRendering();
	if (FrameDump::Active()) {
		FrameDump::OnEndRendering(*this, m_render_state);
	}
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
