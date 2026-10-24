#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <algorithm>
#include <cstdlib>

namespace Libs::Graphics {

namespace {
// KYTY_IDLE_TRACE timestamps around one of the emulator's own commands.
struct HostTraceScope {
	HostTraceScope(const CommandRecorder& cmd, const char* kind, uint64_t code)
	    : cmd(cmd), slot(HostTraceBegin(cmd, kind, 0x4057000000000000ull | code)) {}
	~HostTraceScope() { HostTraceEnd(cmd, slot); }
	HostTraceScope(const HostTraceScope&)            = delete;
	HostTraceScope& operator=(const HostTraceScope&) = delete;
	const CommandRecorder& cmd;
	uint32_t               slot;
};
} // namespace

CommandStream::Chunk::Chunk(size_t capacity): capacity(capacity), arrays(capacity) {
	data = static_cast<std::byte*>(::operator new(capacity, std::align_val_t {Alignment}));
}

CommandStream::Chunk::~Chunk() {
	::operator delete(data, std::align_val_t {Alignment});
}

void CommandStream::NextChunk(size_t bytes) {
	if (m_current != nullptr && m_output != nullptr) {
		m_output->Push(std::move(m_chunks.back()));
		m_chunks.clear();
	}
	std::unique_ptr<Chunk> chunk;
	if (bytes <= ChunkCapacity) {
		if (m_output != nullptr) {
			chunk = m_output->TakeRecycled();
		} else if (!m_free.empty()) {
			chunk = std::move(m_free.back());
			m_free.pop_back();
		}
	}
	if (chunk == nullptr) {
		chunk = std::make_unique<Chunk>(std::max(ChunkCapacity, RoundUp(bytes)));
	}
	chunk->used   = 0;
	chunk->arrays = chunk->capacity;
	m_current     = chunk.get();
	m_chunks.push_back(std::move(chunk));
}

void CommandStream::Publish() {
	EXIT_IF(m_output == nullptr);
	if (m_current != nullptr) {
		m_output->Push(std::move(m_chunks.back()));
		m_chunks.clear();
		m_current = nullptr;
	}
}

std::vector<std::unique_ptr<CommandStream::Chunk>> CommandStream::TakeChunks() {
	EXIT_IF(m_output != nullptr);
	std::vector<std::unique_ptr<Chunk>> chunks;
	chunks.swap(m_chunks);
	m_current = nullptr;
	return chunks;
}

void CommandStream::Recycle(std::unique_ptr<Chunk> chunk) {
	// Oversized chunks hold one large command; do not keep them around.
	if (chunk->capacity == ChunkCapacity) {
		m_free.push_back(std::move(chunk));
	}
}

void CommandChunkQueue::Recycle(ChunkPtr chunk) {
	// Oversized chunks hold one large command; do not keep them around, nor more than the ring
	// holds.
	if (chunk->capacity != CommandStream::ChunkCapacity) {
		return;
	}
	(void)m_recycled.TryPush(chunk);
}

void CommandRecorder::pipelineBarrier(vk::PipelineStageFlags src_stages,
                                      vk::PipelineStageFlags dst_stages,
                                      vk::DependencyFlags dependency, uint32_t memory_count,
                                      const vk::MemoryBarrier* memory, uint32_t buffer_count,
                                      const vk::BufferMemoryBarrier* buffers,
                                      uint32_t image_count, const vk::ImageMemoryBarrier* images) const {
	const HostTraceScope trace(*this, "host-barrier", 1);
	if (m_stream == nullptr) {
		m_direct.pipelineBarrier(src_stages, dst_stages, dependency, memory_count, memory,
		                         buffer_count, buffers, image_count, images);
		return;
	}
	EXIT_IF((memory_count != 0 && memory->pNext != nullptr) ||
	        (buffer_count != 0 && buffers->pNext != nullptr) ||
	        (image_count != 0 && images->pNext != nullptr));
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::MemoryBarrier) * memory_count) +
	                  CommandStream::RoundUp(sizeof(vk::BufferMemoryBarrier) * buffer_count) +
	                  CommandStream::RoundUp(sizeof(vk::ImageMemoryBarrier) * image_count) + 128);
	const auto* memory_copy  = m_stream->Copy(memory, memory_count);
	const auto* buffer_copy  = m_stream->Copy(buffers, buffer_count);
	const auto* image_copy   = m_stream->Copy(images, image_count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.pipelineBarrier(src_stages, dst_stages, dependency, memory_count, memory_copy,
		                  buffer_count, buffer_copy, image_count, image_copy);
	});
}

void CommandRecorder::pipelineBarrier2(const vk::DependencyInfo& dependency) const {
	const HostTraceScope trace(*this, "host-barrier2", 2);
	if (m_stream == nullptr) {
		m_direct.pipelineBarrier2(dependency);
		return;
	}
	EXIT_IF(dependency.pNext != nullptr);
	m_stream->Reserve(
	    CommandStream::RoundUp(sizeof(vk::DependencyInfo)) +
	    CommandStream::RoundUp(sizeof(vk::MemoryBarrier2) * dependency.memoryBarrierCount) +
	    CommandStream::RoundUp(sizeof(vk::BufferMemoryBarrier2) *
	                           dependency.bufferMemoryBarrierCount) +
	    CommandStream::RoundUp(sizeof(vk::ImageMemoryBarrier2) * dependency.imageMemoryBarrierCount) +
	    128);
	auto* copy                     = m_stream->Copy(&dependency, 1);
	copy->pMemoryBarriers          = m_stream->Copy(dependency.pMemoryBarriers,
	                                                dependency.memoryBarrierCount);
	copy->pBufferMemoryBarriers    = m_stream->Copy(dependency.pBufferMemoryBarriers,
	                                                dependency.bufferMemoryBarrierCount);
	copy->pImageMemoryBarriers     = m_stream->Copy(dependency.pImageMemoryBarriers,
	                                                dependency.imageMemoryBarrierCount);
	const vk::DependencyInfo* info = copy;
	m_stream->Record([=](vk::CommandBuffer c) { c.pipelineBarrier2(info); });
}

void CommandRecorder::bindDescriptorSets(vk::PipelineBindPoint bind_point,
                                         vk::PipelineLayout layout, uint32_t first_set,
                                         uint32_t set_count, const vk::DescriptorSet* sets,
                                         uint32_t        dynamic_count,
                                         const uint32_t* dynamic_offsets) const {
	if (m_stream == nullptr) {
		m_direct.bindDescriptorSets(bind_point, layout, first_set, set_count, sets, dynamic_count,
		                            dynamic_offsets);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::DescriptorSet) * set_count) +
	                  CommandStream::RoundUp(sizeof(uint32_t) * dynamic_count) + 128);
	const auto* sets_copy    = m_stream->Copy(sets, set_count);
	const auto* offsets_copy = m_stream->Copy(dynamic_offsets, dynamic_count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.bindDescriptorSets(bind_point, layout, first_set, set_count, sets_copy, dynamic_count,
		                     offsets_copy);
	});
}

// Descriptor writes and the arrays they point to, copied into the stream (one chunk).
static const vk::WriteDescriptorSet* CopyWrites(CommandStream* stream, uint32_t write_count,
                                                const vk::WriteDescriptorSet* writes) {
	size_t bytes = CommandStream::RoundUp(sizeof(vk::WriteDescriptorSet) * write_count) + 128;
	for (uint32_t i = 0; i < write_count; i++) {
		const auto& write = writes[i];
		EXIT_IF(write.pNext != nullptr);
		if (write.pBufferInfo != nullptr) {
			bytes += CommandStream::RoundUp(sizeof(vk::DescriptorBufferInfo) * write.descriptorCount);
		}
		if (write.pImageInfo != nullptr) {
			bytes += CommandStream::RoundUp(sizeof(vk::DescriptorImageInfo) * write.descriptorCount);
		}
		if (write.pTexelBufferView != nullptr) {
			bytes += CommandStream::RoundUp(sizeof(vk::BufferView) * write.descriptorCount);
		}
	}
	stream->Reserve(bytes);
	auto* copy = stream->Copy(writes, write_count);
	for (uint32_t i = 0; i < write_count; i++) {
		auto& write = copy[i];
		if (write.pBufferInfo != nullptr) {
			write.pBufferInfo = stream->Copy(write.pBufferInfo, write.descriptorCount);
		}
		if (write.pImageInfo != nullptr) {
			write.pImageInfo = stream->Copy(write.pImageInfo, write.descriptorCount);
		}
		if (write.pTexelBufferView != nullptr) {
			write.pTexelBufferView = stream->Copy(write.pTexelBufferView, write.descriptorCount);
		}
	}
	return copy;
}

void CommandRecorder::pushDescriptorSetKHR(vk::PipelineBindPoint bind_point,
                                           vk::PipelineLayout layout, uint32_t set,
                                           uint32_t                      write_count,
                                           const vk::WriteDescriptorSet* writes) const {
	if (m_stream == nullptr) {
		m_direct.pushDescriptorSetKHR(bind_point, layout, set, write_count, writes);
		return;
	}
	const auto* writes_copy = CopyWrites(m_stream, write_count, writes);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.pushDescriptorSetKHR(bind_point, layout, set, write_count, writes_copy);
	});
}

void CommandRecorder::updateDescriptorSets(vk::Device device, uint32_t write_count,
                                           const vk::WriteDescriptorSet* writes) const {
	if (m_stream == nullptr) {
		device.updateDescriptorSets(write_count, writes, 0, nullptr);
		return;
	}
	const auto* writes_copy = CopyWrites(m_stream, write_count, writes);
	m_stream->Record([=](vk::CommandBuffer /*c*/) {
		device.updateDescriptorSets(write_count, writes_copy, 0, nullptr);
	});
}

void CommandRecorder::updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                                   const void* data) const {
	const HostTraceScope trace(*this, "host-update", 3);
	if (m_stream == nullptr) {
		m_direct.updateBuffer(buffer, offset, size, data);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(size) + 128);
	const auto* data_copy = m_stream->Copy(static_cast<const std::byte*>(data), size);
	m_stream->Record(
	    [=](vk::CommandBuffer c) { c.updateBuffer(buffer, offset, size, data_copy); });
}

void CommandRecorder::pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages,
                                    uint32_t offset, uint32_t size, const void* values) const {
	if (m_stream == nullptr) {
		m_direct.pushConstants(layout, stages, offset, size, values);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(size) + 128);
	const auto* values_copy = m_stream->Copy(static_cast<const std::byte*>(values), size);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.pushConstants(layout, stages, offset, size, values_copy);
	});
}

void CommandRecorder::beginRendering(const vk::RenderingInfo& rendering) const {
	if (m_stream == nullptr) {
		m_direct.beginRendering(rendering);
		return;
	}
	EXIT_IF(rendering.pNext != nullptr);
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::RenderingInfo)) +
	                  CommandStream::RoundUp(sizeof(vk::RenderingAttachmentInfo) *
	                                         rendering.colorAttachmentCount) +
	                  2 * CommandStream::RoundUp(sizeof(vk::RenderingAttachmentInfo)) + 128);
	auto* copy               = m_stream->Copy(&rendering, 1);
	copy->pColorAttachments  = m_stream->Copy(rendering.pColorAttachments,
	                                          rendering.colorAttachmentCount);
	copy->pDepthAttachment   = m_stream->Copy(rendering.pDepthAttachment, 1);
	copy->pStencilAttachment = m_stream->Copy(rendering.pStencilAttachment, 1);
	const vk::RenderingInfo* info = copy;
	m_stream->Record([=](vk::CommandBuffer c) { c.beginRendering(info); });
}

void CommandRecorder::bindVertexBuffers2(uint32_t first, uint32_t count,
                                         const vk::Buffer* buffers, const vk::DeviceSize* offsets,
                                         const vk::DeviceSize* sizes,
                                         const vk::DeviceSize* strides) const {
	if (m_stream == nullptr) {
		m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::Buffer) * count) +
	                  3 * CommandStream::RoundUp(sizeof(vk::DeviceSize) * count) + 128);
	const auto* buffers_copy = m_stream->Copy(buffers, count);
	const auto* offsets_copy = m_stream->Copy(offsets, count);
	const auto* sizes_copy   = m_stream->Copy(sizes, count);
	const auto* strides_copy = m_stream->Copy(strides, count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.bindVertexBuffers2(first, count, buffers_copy, offsets_copy, sizes_copy, strides_copy);
	});
}

void CommandRecorder::copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
                                 const vk::BufferCopy* regions) const {
	const HostTraceScope trace(*this, "host-copy-buffer", 4);
	if (m_stream == nullptr) {
		m_direct.copyBuffer(source, destination, count, regions);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::BufferCopy) * count) + 128);
	const auto* copy = m_stream->Copy(regions, count);
	m_stream->Record(
	    [=](vk::CommandBuffer c) { c.copyBuffer(source, destination, count, copy); });
}

void CommandRecorder::copyImage(vk::Image source, vk::ImageLayout source_layout,
                                vk::Image destination, vk::ImageLayout destination_layout,
                                uint32_t count, const vk::ImageCopy* regions) const {
	const HostTraceScope trace(*this, "host-copy-image", 5);
	if (m_stream == nullptr) {
		m_direct.copyImage(source, source_layout, destination, destination_layout, count,
		                   regions);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ImageCopy) * count) + 128);
	const auto* copy = m_stream->Copy(regions, count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.copyImage(source, source_layout, destination, destination_layout, count, copy);
	});
}

void CommandRecorder::resolveImage(vk::Image source, vk::ImageLayout source_layout,
                                   vk::Image destination, vk::ImageLayout destination_layout,
                                   uint32_t count, const vk::ImageResolve* regions) const {
	const HostTraceScope trace(*this, "host-resolve", 6);
	if (m_stream == nullptr) {
		m_direct.resolveImage(source, source_layout, destination, destination_layout, count,
		                      regions);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ImageResolve) * count) + 128);
	const auto* copy = m_stream->Copy(regions, count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.resolveImage(source, source_layout, destination, destination_layout, count, copy);
	});
}

void CommandRecorder::blitImage(vk::Image source, vk::ImageLayout source_layout,
                                vk::Image destination, vk::ImageLayout destination_layout,
                                uint32_t count, const vk::ImageBlit* regions,
                                vk::Filter filter) const {
	const HostTraceScope trace(*this, "host-blit", 7);
	if (m_stream == nullptr) {
		m_direct.blitImage(source, source_layout, destination, destination_layout, count, regions,
		                   filter);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ImageBlit) * count) + 128);
	const auto* copy = m_stream->Copy(regions, count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.blitImage(source, source_layout, destination, destination_layout, count, copy, filter);
	});
}

void CommandRecorder::copyImageToBuffer(vk::Image source, vk::ImageLayout source_layout,
                                        vk::Buffer destination, uint32_t count,
                                        const vk::BufferImageCopy* regions) const {
	const HostTraceScope trace(*this, "host-image-to-buffer", 8);
	if (m_stream == nullptr) {
		m_direct.copyImageToBuffer(source, source_layout, destination, count, regions);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::BufferImageCopy) * count) + 128);
	const auto* copy = m_stream->Copy(regions, count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.copyImageToBuffer(source, source_layout, destination, count, copy);
	});
}

void CommandRecorder::copyBufferToImage(vk::Buffer source, vk::Image destination,
                                        vk::ImageLayout destination_layout, uint32_t count,
                                        const vk::BufferImageCopy* regions) const {
	const HostTraceScope trace(*this, "host-buffer-to-image", 9);
	if (m_stream == nullptr) {
		m_direct.copyBufferToImage(source, destination, destination_layout, count, regions);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::BufferImageCopy) * count) + 128);
	const auto* copy = m_stream->Copy(regions, count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.copyBufferToImage(source, destination, destination_layout, count, copy);
	});
}

void CommandRecorder::clearColorImage(vk::Image image, vk::ImageLayout layout,
                                      const vk::ClearColorValue* color, uint32_t range_count,
                                      const vk::ImageSubresourceRange* ranges) const {
	const HostTraceScope trace(*this, "host-clear-color", 10);
	if (m_stream == nullptr) {
		m_direct.clearColorImage(image, layout, color, range_count, ranges);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ClearColorValue)) +
	                  CommandStream::RoundUp(sizeof(vk::ImageSubresourceRange) * range_count) + 128);
	const auto* color_copy  = m_stream->Copy(color, 1);
	const auto* ranges_copy = m_stream->Copy(ranges, range_count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.clearColorImage(image, layout, color_copy, range_count, ranges_copy);
	});
}

void CommandRecorder::clearAttachments(uint32_t attachment_count,
                                       const vk::ClearAttachment* attachments,
                                       uint32_t rect_count, const vk::ClearRect* rects) const {
	const HostTraceScope trace(*this, "host-clear-attachments", 11);
	if (m_stream == nullptr) {
		m_direct.clearAttachments(attachment_count, attachments, rect_count, rects);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ClearAttachment) * attachment_count) +
	                  CommandStream::RoundUp(sizeof(vk::ClearRect) * rect_count) + 128);
	const auto* attachments_copy = m_stream->Copy(attachments, attachment_count);
	const auto* rects_copy       = m_stream->Copy(rects, rect_count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.clearAttachments(attachment_count, attachments_copy, rect_count, rects_copy);
	});
}

void CommandRecorder::clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
                                             const vk::ClearDepthStencilValue* value,
                                             uint32_t                          range_count,
                                             const vk::ImageSubresourceRange*  ranges) const {
	const HostTraceScope trace(*this, "host-clear-depth", 12);
	if (m_stream == nullptr) {
		m_direct.clearDepthStencilImage(image, layout, value, range_count, ranges);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ClearDepthStencilValue)) +
	                  CommandStream::RoundUp(sizeof(vk::ImageSubresourceRange) * range_count) + 128);
	const auto* value_copy  = m_stream->Copy(value, 1);
	const auto* ranges_copy = m_stream->Copy(ranges, range_count);
	m_stream->Record([=](vk::CommandBuffer c) {
		c.clearDepthStencilImage(image, layout, value_copy, range_count, ranges_copy);
	});
}

void CommandRecorder::setViewport(uint32_t first, uint32_t count,
                                  const vk::Viewport* viewports) const {
	if (m_stream == nullptr) {
		m_direct.setViewport(first, count, viewports);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::Viewport) * count) + 128);
	const auto* copy = m_stream->Copy(viewports, count);
	m_stream->Record([=](vk::CommandBuffer c) { c.setViewport(first, count, copy); });
}

void CommandRecorder::setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
	if (m_stream == nullptr) {
		m_direct.setScissor(first, count, scissors);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::Rect2D) * count) + 128);
	const auto* copy = m_stream->Copy(scissors, count);
	m_stream->Record([=](vk::CommandBuffer c) { c.setScissor(first, count, copy); });
}

void CommandRecorder::setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
	if (m_stream == nullptr) {
		m_direct.setViewportWithCount(count, viewports);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::Viewport) * count) + 128);
	const auto* copy = m_stream->Copy(viewports, count);
	m_stream->Record([=](vk::CommandBuffer c) { c.setViewportWithCount(count, copy); });
}

void CommandRecorder::setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
	if (m_stream == nullptr) {
		m_direct.setScissorWithCount(count, scissors);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::Rect2D) * count) + 128);
	const auto* copy = m_stream->Copy(scissors, count);
	m_stream->Record([=](vk::CommandBuffer c) { c.setScissorWithCount(count, copy); });
}

void CommandRecorder::setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
	if (m_stream == nullptr) {
		m_direct.setColorWriteEnableEXT(count, enables);
		return;
	}
	m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::Bool32) * count) + 128);
	const auto* copy = m_stream->Copy(enables, count);
	m_stream->Record([=](vk::CommandBuffer c) { c.setColorWriteEnableEXT(count, copy); });
}

} // namespace Libs::Graphics
