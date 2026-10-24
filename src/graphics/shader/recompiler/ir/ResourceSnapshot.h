#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	// Defaulted: nine dwords without padding, so the type is trivially equality-comparable and
	// whole arrays of them compare as one memcmp (the element-wise compares of a stage's T#s and
	// S#s against its last draw's were 1.4% of the execution thread).
	bool operator==(const DescriptorValue& other) const = default;
};
static_assert(sizeof(DescriptorValue) == 9 * sizeof(uint32_t));

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;

	bool operator==(const UniformFill&) const = default;
};

// A compute shader whose only effect is dst[i] = src[i] for every invocation i (same index,
// same element size), e.g. NHL 27's dword copy 5f193bbc12f9c6d3. The emulator can perform the
// copy on guest memory as well, so later CPU readers need no GPU readback.
struct UniformCopy {
	bool     valid        = false;
	uint32_t source       = 0;
	uint32_t destination  = 0;
	uint32_t words        = 0;
	uint32_t group_stride = 0;

	bool operator==(const UniformCopy&) const = default;
};

// The distinct T#s of a bindless image table; slot 0 is null.
struct ImageTableSnapshot {
	uint64_t                     base         = 0;
	uint64_t                     size         = 0;
	uint32_t                     offset       = 0;
	uint32_t                     entry_mask   = 0;
	std::vector<uint32_t>        raw;     // The table words the slots were built from.
	std::vector<DescriptorValue> slots;
	std::vector<uint32_t>        mapping; // Entry count, then the slot of each entry.
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<ImageTableSnapshot> image_tables;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	UniformCopy                 uniform_copy;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
