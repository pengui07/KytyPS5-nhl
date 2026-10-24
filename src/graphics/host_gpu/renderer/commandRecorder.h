#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "common/common.h"
#include "common/spscQueue.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// A linear buffer of Vulkan commands recorded on one thread and replayed into a command buffer
// later, possibly on another thread. Every command is a trivially copyable callable stored in a
// chunk together with copies of the arrays it points to, so replay only calls the callable.
//
// Records grow from the start of a chunk and the arrays they reference grow down from its end,
// so replay walks the records alone. A command and its arrays always live in the same chunk
// (Reserve() rolls over to a new chunk before any of them is written), so a chunk can be
// recycled as soon as it has been replayed.
class CommandChunkQueue;

class CommandStream {
public:
	struct Chunk {
		explicit Chunk(size_t capacity);
		~Chunk();
		KYTY_CLASS_NO_COPY(Chunk);

		std::byte* data     = nullptr;
		size_t     capacity = 0;
		size_t     used     = 0; // End of the records.
		size_t     arrays   = 0; // Start of the arrays.
	};

	// The kinds of records; commands are replayed through their callable, the others are
	// interpreted by the consumer.
	enum class RecordKind : uint32_t { Command, Submit };

	struct Entry {
		using Fn = void (*)(vk::CommandBuffer command, const void* payload);
		Fn         fn   = nullptr;
		uint32_t   size = 0; // Bytes from this record to the next one.
		RecordKind kind = RecordKind::Command;

		[[nodiscard]] const void* Payload() const noexcept {
			return reinterpret_cast<const std::byte*>(this) + sizeof(Entry);
		}
	};
	static_assert(sizeof(Entry) == 16);

	// Small enough that the recording thread starts on a chunk soon after the stream began it.
	static constexpr size_t ChunkCapacity = 64u * 1024u;
	static constexpr size_t Alignment     = 16;

	CommandStream() = default;
	// Full chunks go to `output` as soon as the stream moves past them.
	explicit CommandStream(CommandChunkQueue* output): m_output(output) {}
	~CommandStream() = default;
	KYTY_CLASS_NO_COPY(CommandStream);

	// Makes sure that the next `bytes` bytes of records and arrays fit in the current chunk.
	void Reserve(size_t bytes) {
		if (m_current == nullptr || m_current->arrays - m_current->used < bytes) {
			NextChunk(bytes);
		}
	}

	template <typename T>
	[[nodiscard]] T* Copy(const T* source, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (count == 0 || source == nullptr) {
			return nullptr;
		}
		auto* target = static_cast<T*>(AllocateArray(sizeof(T) * count));
		std::memcpy(static_cast<void*>(target), source, sizeof(T) * count);
		return target;
	}

	template <typename F>
	void Record(F&& function) {
		using Callable = std::decay_t<F>;
		static_assert(std::is_trivially_copyable_v<Callable> &&
		              std::is_trivially_destructible_v<Callable>);
		static_assert(alignof(Callable) <= Alignment);
		constexpr auto size = RoundUp(sizeof(Entry) + sizeof(Callable));
		auto*          record = static_cast<Entry*>(Allocate(size));
		record->fn            = [](vk::CommandBuffer command, const void* payload) {
			(*static_cast<const Callable*>(payload))(command);
		};
		record->size = static_cast<uint32_t>(size);
		record->kind = RecordKind::Command;
		::new (const_cast<void*>(record->Payload())) Callable(std::forward<F>(function));
		m_commands++;
	}

	// Appends a record with a trivially copyable payload for the consumer.
	template <typename T>
	[[nodiscard]] T* RecordSpecial(RecordKind kind) {
		static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= Alignment);
		constexpr auto size = RoundUp(sizeof(Entry) + sizeof(T));
		auto*          record = static_cast<Entry*>(Allocate(size));
		record->fn            = nullptr;
		record->size          = static_cast<uint32_t>(size);
		record->kind          = kind;
		return ::new (const_cast<void*>(record->Payload())) T {};
	}

	// Sends the current chunk to the output queue.
	void Publish();
	// Without an output queue: takes the chunks recorded so far.
	[[nodiscard]] std::vector<std::unique_ptr<Chunk>> TakeChunks();
	// Without an output queue: gives replayed chunks back for reuse.
	void Recycle(std::unique_ptr<Chunk> chunk);

	[[nodiscard]] uint64_t CommandCount() const noexcept { return m_commands; }

	// Calls every command of a chunk in the command buffer `acquire()` returns; `special`
	// receives the other records in order.
	template <typename Acquire, typename Special>
	static void Replay(const Chunk& chunk, Acquire&& acquire, Special&& special) {
		size_t offset = 0;
		while (offset < chunk.used) {
			const auto* record = reinterpret_cast<const Entry*>(chunk.data + offset);
			if (record->kind == RecordKind::Command) {
				record->fn(acquire(), record->Payload());
			} else {
				special(*record);
			}
			offset += record->size;
		}
	}

	static constexpr size_t RoundUp(size_t size) {
		return (size + Alignment - 1) & ~(Alignment - 1);
	}

private:
	void* Allocate(size_t bytes) {
		bytes = RoundUp(bytes);
		Reserve(bytes);
		auto* result = m_current->data + m_current->used;
		m_current->used += bytes;
		return result;
	}
	void* AllocateArray(size_t bytes) {
		bytes = RoundUp(bytes);
		Reserve(bytes);
		m_current->arrays -= bytes;
		return m_current->data + m_current->arrays;
	}
	void NextChunk(size_t bytes);

	CommandChunkQueue*                  m_output = nullptr;
	std::vector<std::unique_ptr<Chunk>> m_chunks;
	Chunk*                              m_current = nullptr;
	std::vector<std::unique_ptr<Chunk>> m_free;
	uint64_t                            m_commands = 0;
};

// Hands recorded chunks from the recording thread to the replaying thread, and replayed chunks
// back, through two lock-free single-producer single-consumer rings. The release/acquire pair
// of a ring orders every write to a chunk before its replay and every replay before the chunk is
// written again.
class CommandChunkQueue {
public:
	using ChunkPtr = std::unique_ptr<CommandStream::Chunk>;

	CommandChunkQueue(): m_chunks(MaxQueued), m_recycled(MaxRecycled) {}
	~CommandChunkQueue() = default;
	KYTY_CLASS_NO_COPY(CommandChunkQueue);

	// Recording side: waits while MaxQueued chunks wait for the consumer.
	void Push(ChunkPtr chunk) { m_chunks.Push(std::move(chunk)); }
	[[nodiscard]] ChunkPtr TakeRecycled() {
		auto chunk = m_recycled.TryPop();
		return chunk ? std::move(*chunk) : nullptr;
	}

	// Replaying side: waits until a chunk is available; returns null once stopped and empty.
	[[nodiscard]] ChunkPtr Pop() {
		auto chunk = m_chunks.Pop();
		return chunk ? std::move(*chunk) : nullptr;
	}
	void Recycle(ChunkPtr chunk);

	// Any thread.
	void Stop() { m_chunks.Stop(); }

private:
	// Chunks a producer may queue before it waits for the consumer.
	static constexpr size_t MaxQueued   = 256;
	static constexpr size_t MaxRecycled = 256;

	Common::SpscQueue<ChunkPtr> m_chunks;
	Common::SpscQueue<ChunkPtr> m_recycled;
};

// The vkCmd* subset used by the renderer. A recorder either calls the Vulkan command buffer
// directly or appends the command to a CommandStream; the method names and signatures follow
// vk::CommandBuffer so recording code does not depend on the mode.
class CommandRecorder;
// KYTY_IDLE_TRACE: GPU timestamps around the emulator's own commands (copies, clears, blits,
// barriers), reported like draws with identity 0x4057'0000'0000'00xx; 0 when not capturing.
uint32_t HostTraceBegin(const CommandRecorder& cmd, const char* kind, uint64_t identity);
void     HostTraceEnd(const CommandRecorder& cmd, uint32_t slot_plus_one);

class CommandRecorder {
public:
	CommandRecorder() = default;
	explicit CommandRecorder(vk::CommandBuffer direct) noexcept: m_direct(direct) {}
	explicit CommandRecorder(CommandStream* stream) noexcept: m_stream(stream) {}

	[[nodiscard]] bool IsDirect() const noexcept { return m_stream == nullptr; }
	[[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
		return m_stream == nullptr && m_direct == nullptr;
	}
	// Identifies the recording target (a command buffer or a scheduler's stream) for memo keys.
	[[nodiscard]] const void* Key() const noexcept {
		return m_stream != nullptr ? static_cast<const void*>(m_stream)
		                           : static_cast<const void*>(static_cast<VkCommandBuffer>(m_direct));
	}
	// The Vulkan command buffer of a direct recorder (presentation paths that are never deferred).
	[[nodiscard]] vk::CommandBuffer Direct() const {
		EXIT_IF(m_stream != nullptr || m_direct == nullptr);
		return m_direct;
	}

	void pipelineBarrier(vk::PipelineStageFlags src_stages, vk::PipelineStageFlags dst_stages,
	                     vk::DependencyFlags dependency, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const;
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const;
	void pipelineBarrier2(const vk::DependencyInfo* dependency) const {
		pipelineBarrier2(*dependency);
	}

	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		if (m_stream == nullptr) {
			m_direct.bindPipeline(bind_point, pipeline);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.bindPipeline(bind_point, pipeline); });
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count,
	                        const vk::DescriptorSet* sets, uint32_t dynamic_count,
	                        const uint32_t* dynamic_offsets) const;
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t write_count,
	                          const vk::WriteDescriptorSet* writes) const;
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, writes.size(), writes.data());
	}
	// Not a command: vkUpdateDescriptorSets on a set no recorded command used yet, done at its place
	// in the stream (on the recording thread) instead of by the thread that records.
	void updateDescriptorSets(vk::Device device, uint32_t write_count,
	                          const vk::WriteDescriptorSet* writes) const;
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const;

	void beginRendering(const vk::RenderingInfo& rendering) const;
	void beginRendering(const vk::RenderingInfo* rendering) const { beginRendering(*rendering); }
	void endRendering() const {
		if (m_stream == nullptr) {
			m_direct.endRendering();
			return;
		}
		m_stream->Record([](vk::CommandBuffer c) { c.endRendering(); });
	}

	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		if (m_stream == nullptr) {
			m_direct.dispatch(x, y, z);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.dispatch(x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		if (m_stream == nullptr) {
			m_direct.dispatchIndirect(buffer, offset);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.dispatchIndirect(buffer, offset); });
	}
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance) const {
		if (m_stream == nullptr) {
			m_direct.draw(vertex_count, instance_count, first_vertex, first_instance);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.draw(vertex_count, instance_count, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		if (m_stream == nullptr) {
			m_direct.drawIndexed(index_count, instance_count, first_index, vertex_offset,
			                     first_instance);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
		});
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		if (m_stream == nullptr) {
			m_direct.drawMeshTasksEXT(x, y, z);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.drawMeshTasksEXT(x, y, z); });
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                  uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndirect(buffer, offset, count, stride);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.drawIndirect(buffer, offset, count, stride); });
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                         uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndexedIndirect(buffer, offset, count, stride);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.drawIndexedIndirect(buffer, offset, count, stride); });
	}
	void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                       vk::DeviceSize count_offset, uint32_t max_count, uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_count, stride);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_count, stride);
		});
	}
	void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset,
	                              vk::Buffer count_buffer, vk::DeviceSize count_offset,
	                              uint32_t max_count, uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_count,
			                                  stride);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_count, stride);
		});
	}

	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const;
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		if (m_stream == nullptr) {
			m_direct.bindIndexBuffer(buffer, offset, type);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.bindIndexBuffer(buffer, offset, type); });
	}

	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const;
	void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                  const void* data) const;
	void writeTimestamp(vk::PipelineStageFlagBits stage, vk::QueryPool pool, uint32_t query) const {
		if (m_stream == nullptr) {
			m_direct.writeTimestamp(stage, pool, query);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.writeTimestamp(stage, pool, query); });
	}
	void writeBufferMarkerAMD(vk::PipelineStageFlagBits stage, vk::Buffer buffer,
	                          vk::DeviceSize offset, uint32_t marker) const {
		if (m_stream == nullptr) {
			m_direct.writeBufferMarkerAMD(stage, buffer, offset, marker);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.writeBufferMarkerAMD(stage, buffer, offset, marker); });
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const;
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout,
	               vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
		copyImage(source, source_layout, destination, destination_layout, regions.size(),
		          regions.data());
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, uint32_t count,
	                  const vk::ImageResolve* regions) const;
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout,
	                  vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
		resolveImage(source, source_layout, destination, destination_layout, regions.size(),
		             regions.data());
	}
	void blitImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageBlit* regions, vk::Filter filter) const;
	void copyImageToBuffer(vk::Image source, vk::ImageLayout source_layout,
	                       vk::Buffer destination, uint32_t count,
	                       const vk::BufferImageCopy* regions) const;
	void copyImageToBuffer(vk::Image source, vk::ImageLayout source_layout,
	                       vk::Buffer                                      destination,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyImageToBuffer(source, source_layout, destination, regions.size(), regions.data());
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination,
	                       vk::ImageLayout destination_layout, uint32_t count,
	                       const vk::BufferImageCopy* regions) const;
	void copyBufferToImage(vk::Buffer source, vk::Image destination,
	                       vk::ImageLayout                                  destination_layout,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyBufferToImage(source, destination, destination_layout, regions.size(), regions.data());
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t value) const {
		if (m_stream == nullptr) {
			m_direct.fillBuffer(buffer, offset, size, value);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.fillBuffer(buffer, offset, size, value); });
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t range_count, const vk::ImageSubresourceRange* ranges) const;
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t range_count,
	                            const vk::ImageSubresourceRange* ranges) const;
	// vk::CommandBuffer's single-range forms.
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
	                     const vk::ImageSubresourceRange& range) const {
		clearColorImage(image, layout, &color, 1, &range);
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue& value,
	                            const vk::ImageSubresourceRange& range) const {
		clearDepthStencilImage(image, layout, &value, 1, &range);
	}

	// Experiment GpuColorClear: predicated clears of a render pass's attachments.
	void clearAttachments(uint32_t attachment_count, const vk::ClearAttachment* attachments,
	                      uint32_t rect_count, const vk::ClearRect* rects) const;
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* begin) const {
		if (m_stream == nullptr) {
			m_direct.beginConditionalRenderingEXT(begin);
			return;
		}
		const auto info = *begin;
		m_stream->Record([=](vk::CommandBuffer c) { c.beginConditionalRenderingEXT(&info); });
	}
	void endConditionalRenderingEXT() const {
		if (m_stream == nullptr) {
			m_direct.endConditionalRenderingEXT();
			return;
		}
		m_stream->Record([](vk::CommandBuffer c) { c.endConditionalRenderingEXT(); });
	}

	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const;
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const;
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const;
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const;
	void setLineWidth(float width) const {
		if (m_stream == nullptr) {
			m_direct.setLineWidth(width);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setLineWidth(width); });
	}
	void setBlendConstants(const float constants[4]) const {
		if (m_stream == nullptr) {
			m_direct.setBlendConstants(constants);
			return;
		}
		struct Values {
			float v[4];
		};
		Values values {};
		std::memcpy(values.v, constants, sizeof(values.v));
		m_stream->Record([=](vk::CommandBuffer c) { c.setBlendConstants(values.v); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthTestEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthWriteEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		if (m_stream == nullptr) {
			m_direct.setDepthCompareOp(op);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthCompareOp(op); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBiasEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBiasEnable(enable); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBias(constant, clamp, slope);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBias(constant, clamp, slope); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBoundsTestEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBounds(min_bounds, max_bounds);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBounds(min_bounds, max_bounds); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setStencilTestEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		if (m_stream == nullptr) {
			m_direct.setStencilOp(faces, fail, pass, depth_fail, compare);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.setStencilOp(faces, fail, pass, depth_fail, compare);
		});
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		if (m_stream == nullptr) {
			m_direct.setStencilCompareMask(faces, mask);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilCompareMask(faces, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		if (m_stream == nullptr) {
			m_direct.setStencilWriteMask(faces, mask);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilWriteMask(faces, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		if (m_stream == nullptr) {
			m_direct.setStencilReference(faces, reference);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilReference(faces, reference); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const;
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		if (m_stream == nullptr) {
			m_direct.setAttachmentFeedbackLoopEnableEXT(aspects);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.setAttachmentFeedbackLoopEnableEXT(aspects); });
	}

private:
	vk::CommandBuffer m_direct = nullptr;
	CommandStream*    m_stream = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
