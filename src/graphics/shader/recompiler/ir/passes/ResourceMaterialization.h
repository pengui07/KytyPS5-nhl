#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Ordinary (non-strict) resource-table reads dereference guest memory. When set, this reader is
// tried first; it returns false to keep the plain read (experiment FaultPageReads).
using FaultPageReader = bool (*)(uint64_t address, void* data, uint64_t size);
void SetFaultPageReader(FaultPageReader reader);
// Experiment GpuConditionsBothPaths: set while a direct resource walk evaluates a branch condition.
// Strict reads then decline GPU-written bytes instead of waiting for the GPU to write them back;
// the walk takes both successors (the union of the resources of both paths).
[[nodiscard]] bool ConditionReadsNonBlocking() noexcept;
void               SetConditionReadsNonBlocking(bool value) noexcept;
// The reader, when set; false when the plain read must be used.
[[nodiscard]] bool TryFaultPageRead(uint64_t address, void* data, uint64_t size);


// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		uint32_t                      table                      = ImageResource::NoImageTable;
		uint32_t                      table_capacity             = 0;
		uint32_t                      table_mapping_offset       = 0;
		uint32_t                      table_entry_mask           = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          heap                       = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Retains direct descriptor decoding independently of changing addresses/payloads.
// The owner must reset this cache if its immutable plan is replaced or modified.
struct ResourceSpecializationCache {
	struct BufferEntry {
		std::array<uint32_t, 2> key {};
		ResourceSpecialization::Buffer value;
		bool valid = false;
	};
	struct ImageEntry {
		std::array<uint32_t, 7> key {};
		ResourceSpecialization::Image value;
		bool null = false;
		bool valid = false;
	};
	const ResourcePlan* owner = nullptr;
	bool active = false; // A nested materialization falls back instead of sharing mutable entries.
	std::vector<BufferEntry> buffers;
	std::vector<ImageEntry> images;
	void Reset() { owner = nullptr; active = false; buffers.clear(); images.clear(); }
};

struct ResourceSpecializationCacheStats {
	uint64_t calls = 0;
	uint64_t buffer_hits = 0, buffer_misses = 0;
	uint64_t image_hits = 0, image_misses = 0;
	uint64_t indirect_uncached = 0;
};
// Per-thread counters: diagnostics do not introduce contended atomics in draw prep.
ResourceSpecializationCacheStats GetResourceSpecializationCacheStats();

// KYTY_SRT_COMPILED (part A): the previous outputs of one program (owned by its plan entry) and the
// per-part change mask of the most recent call against them. Keyed on the program's own last use,
// not on the previous draw. Compare-then-copy keeps the stored copy current without extra passes.
struct ResourceChangeMask {
	bool valid             = false; // a previous successful call existed to compare against
	bool images            = false;
	bool samplers          = false;
	bool buffer_fmt        = false; // stride/format/swizzle words (everything but address and num_records)
	bool buffer_addr       = false; // dword0 plus dword1[15:0]
	bool buffer_size       = false; // num_records
	bool flat_changed      = false;
	bool user_data_changed = false;
};

struct ResourceDelta {
	std::vector<DescriptorValue> images, samplers, buffers;
	std::vector<uint32_t>        flat, user_data;
	bool                         have = false;
	ResourceChangeMask           mask;
	// Specialization built from the image words below; reused while those words are unchanged.
	const ResourcePlan*                         owner = nullptr;
	std::vector<DescriptorValue>                spec_images_in;
	std::vector<ResourceSpecialization::Image> spec_images_out;
	bool                                        spec_valid = false;
	uint64_t                                    verify_ticker = 0;
	// Input-keyed descriptor reuse (SrtWalker::EvaluateDescriptor): per descriptor source, the exact memo
	// values the last successful recursive evaluation consumed and the descriptor it produced.
	struct SourceReuse {
		bool                  valid = false;
		uint8_t               mode  = 0;
		uint64_t              plan_serial = 0;
		std::vector<uint64_t> inputs;
		DescriptorValue       output;
	};
	std::vector<SourceReuse>                    sources;
	void Reset() {
		have = spec_valid = false;
		mask = {};
		owner = nullptr;
		for (auto& entry: sources) entry.valid = false;
	}
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// Whether the host can bind a bindless table T# in the descriptor array of a specialized root.
bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
