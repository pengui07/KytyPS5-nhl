#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace Xbyak {
class CodeGenerator;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	IndirectBuffer,
	Flat,
	FlatLocal,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::FlatLocal || kind == ResourceKind::Global ||
	       kind == ResourceKind::Scratch;
}

struct MemoryInfo {
	ResourceKind            kind                     = ResourceKind::None;
	uint32_t                resource                 = 0;
	uint32_t                sampler                  = 0;
	uint32_t                offset                   = 0;
	uint32_t                secondary_offset         = 0;
	uint32_t                dmask                    = 0;
	uint32_t                data_dwords              = 1;
	uint32_t                data_bits                = 32;
	uint32_t                component_index          = 0;
	uint32_t                component_count          = 1;
	uint32_t                data_format              = 0;
	uint32_t                number_format            = 0;
	uint32_t                image_sample_flags       = 0;
	Decoder::ImageDimension image_dimension          = Decoder::ImageDimension::Unknown;
	uint32_t                image_address_components = 0;
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		return !typed && data_bits == 32u &&
		       (formatted ? opcode == ValueOpcode::LoadBufferU32
		                  : opcode == ValueOpcode::ReadConstBuffer ||
		                        opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4);
	}
	// Raw DWORD stores through a V# selected at runtime from a descriptor table.
	[[nodiscard]] bool SupportsIndirectBufferStore(ValueOpcode opcode) const {
		return !formatted && !typed && data_bits == 32u &&
		       (opcode == ValueOpcode::StoreBufferU32 || opcode == ValueOpcode::StoreBufferU32x2 ||
		        opcode == ValueOpcode::StoreBufferU32x3 || opcode == ValueOpcode::StoreBufferU32x4);
	}
	[[nodiscard]] bool SupportsIndirectBufferAccess(ValueOpcode opcode) const {
		return SupportsIndirectBufferLoad(opcode) || SupportsIndirectBufferStore(opcode);
	}

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind   = ExportTargetKind::Unknown;
	uint32_t         target = 0;
	uint32_t         index  = 0;
	uint32_t         en     = 0;
	bool             done   = false;
	bool             compr  = false;
	bool             vm     = false;

	bool operator==(const ExportInfo& other) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               descriptor_index   = UINT32_MAX;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;
	// This buffer is a table of V#s (16-byte entries starting at indirect_table_offset) that
	// selects the targets of IndirectBuffer stores; the host caches every entry it names.
	bool                   indirect_write_table  = false;
	uint32_t               indirect_table_offset = 0;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;
	static constexpr uint32_t NoImageTable    = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      descriptor_index  = UINT32_MAX;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          atomic64          = false;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	std::vector<uint32_t>         indirect_resources;
	// A root sampling a GPU-indexed T# table (bindless): the key selects table entry
	// key & table_entry_mask, flattened_srt[table_mapping_offset] holds the entry count and
	// then one slot per entry, and the slot indexes table_capacity descriptors of this image's
	// binding that the host fills from the table (slot 0 is a null image).
	uint32_t                      table                = NoImageTable;
	uint32_t                      table_capacity       = 0;
	uint32_t                      table_mapping_offset = 0;
	uint32_t                      table_entry_mask     = 0;
	// First element of the table's descriptors in this image's binding, after the binding's
	// resources; roots of one table and binding share them.
	uint32_t                      table_descriptor_base = UINT32_MAX;

	bool operator==(const ImageResource& other) const = default;
};

struct SamplerResource {
	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	std::vector<uint32_t> indirect_resources;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;

	bool operator==(const SamplerResource& other) const = default;
};

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	Factor
};

enum class StageInputKind {
	DrawIndex,
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothSample,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	NumWorkgroups,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	Depth,
	SampleMask,
	PointSize,
	ClipDistance,
	CullDistance,
	Layer,
	ViewportIndex
};

struct PositionExportComponent {
	uint32_t clip_distance = UINT32_MAX;
	uint32_t cull_distance = UINT32_MAX;
	bool     point_size     = false;
	bool     layer          = false;
	bool     viewport       = false;
};

inline PositionExportComponent DecodePositionExportComponent(uint32_t control,
	                                                           uint32_t pos_index,
	                                                           uint32_t component) {
	PositionExportComponent result;
	if (pos_index == 0 || component >= 4) {
		return result;
	}

	uint32_t slot   = pos_index - 1;
	uint32_t vector = 3;
	for (uint32_t i = 0; i < 3; i++) {
		if ((control & (1u << (21u + i))) != 0) {
			if (slot == 0) {
				vector = i;
				break;
			}
			slot--;
		}
	}
	if (vector == 3) {
		return result;
	}

	if (vector == 0) {
		result.point_size = component == 0 && (control & (1u << 16u)) != 0;
		result.layer      = component == 2 && (control & (1u << 18u)) != 0;
		result.viewport   = component == 2 && (control & (1u << 19u)) != 0;
		return result;
	}

	const auto scalar = (vector - 1) * 4 + component;
	const auto lower  = (1u << scalar) - 1u;
	const auto clip   = control & 0xffu;
	const auto cull   = (control >> 8u) & 0xffu;
	if ((clip & (1u << scalar)) != 0) {
		result.clip_distance = std::popcount(clip & lower);
	}
	if ((cull & (1u << scalar)) != 0) {
		result.cull_distance = std::popcount(cull & lower);
	}
	return result;
}

struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
	StageOutputKind kind     = StageOutputKind::Parameter;
	uint32_t        index    = 0;
	uint32_t        location = 0;
	std::string     debug_name;

	bool operator==(const StageOutput& other) const = default;
};

inline constexpr uint32_t FirstImageBinding           = 1u;
inline constexpr uint32_t FirstComparisonImageBinding = 22u;
inline constexpr uint32_t FirstStorageImageBinding    = 29u;
inline constexpr uint32_t ImageBindingCount           = 48u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	SharedMemory,
	Count,
};

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 49u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 56u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	[[nodiscard]] static constexpr bool CanFit(uint32_t start, uint32_t size) {
		return size != 0 && start <= DwordCount && size <= DwordCount - start;
	}
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		return CanFit(cursor, size) ? cursor : NoStart;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	return static_cast<uint32_t>(kind) +
	       group * static_cast<uint32_t>(DescriptorBindingKind::Count);
}

[[nodiscard]] constexpr ImageResourceClass ImageBindingResourceClass(DescriptorBindingKind kind) {
	const auto value = static_cast<uint32_t>(kind);
	if (value >= FirstImageBinding && value < FirstStorageImageBinding) {
		return ImageResourceClass::Sampled;
	}
	if (value >= FirstStorageImageBinding &&
	    value < static_cast<uint32_t>(DescriptorBindingKind::Samplers)) {
		return ImageResourceClass::Storage;
	}
	return ImageResourceClass::None;
}

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 5u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 5u;

	uint32_t base    = 0;
	bool     sampled = false;
	if (image.resource_class == ImageResourceClass::Sampled) {
		if (image.atomic) {
			return std::nullopt;
		}
		sampled = true;
		switch (image.numeric_class) {
			case Prospero::TextureNumericClass::Float:
				base = image.depth_compare ? FirstComparisonImageBinding : SampledFloatBinding;
				break;
			case Prospero::TextureNumericClass::Uint: base = SampledUintBinding; break;
			case Prospero::TextureNumericClass::Sint: base = SampledSintBinding; break;
			case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
			default: return std::nullopt;
		}
		if (image.depth_compare && image.numeric_class != Prospero::TextureNumericClass::Float) {
			return std::nullopt;
		}
	} else if (image.resource_class == ImageResourceClass::Storage) {
		if (image.atomic) {
			if (image.numeric_class != Prospero::TextureNumericClass::Uint) {
				return std::nullopt;
			}
			base = AtomicUintBinding + (image.atomic64 ? 5u : 0u);
		} else {
			switch (image.numeric_class) {
				case Prospero::TextureNumericClass::Float: base = StorageFloatBinding; break;
				case Prospero::TextureNumericClass::Uint: base = StorageUintBinding; break;
				case Prospero::TextureNumericClass::Sint:
				case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
				default: return std::nullopt;
			}
		}
	} else {
		return std::nullopt;
	}

	uint32_t dimension = 0;
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D: break;
		case Decoder::ImageDimension::Dim1DArray: dimension = 1u; break;
		case Decoder::ImageDimension::Dim2D: dimension = 2u; break;
		case Decoder::ImageDimension::Dim2DArray: dimension = 3u; break;
		case Decoder::ImageDimension::Dim2DMsaa:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 4u;
			break;
		case Decoder::ImageDimension::Dim2DMsaaArray:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 5u;
			break;
		case Decoder::ImageDimension::Dim3D: dimension = sampled ? 6u : 4u; break;
		case Decoder::ImageDimension::Unknown: return std::nullopt;
		default: return std::nullopt;
	}
	return static_cast<DescriptorBindingKind>(base + dimension);
}

// The first table descriptor of each image (UINT32_MAX without a bindless table): the
// descriptors of a table follow the resources of its roots' binding, whose count grows by the
// table capacity; roots of one table and binding share them.
[[nodiscard]] inline std::vector<uint32_t> AllocateImageTableDescriptors(
    const std::vector<ImageResource>& images,
    std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)>& counts) {
	std::vector<uint32_t> bases(images.size(), UINT32_MAX);
	for (size_t i = 0; i < images.size(); i++) {
		const auto& image = images[i];
		if (image.table_capacity == 0u) {
			continue;
		}
		const auto kind = *DescriptorBindingForImage(image);
		for (size_t j = 0; j < i && bases[i] == UINT32_MAX; j++) {
			if (bases[j] != UINT32_MAX && images[j].table == image.table &&
			    DescriptorBindingForImage(images[j]) == kind) {
				if (images[j].table_capacity != image.table_capacity) {
					EXIT("shader binding layout failed: image table %u has two capacities",
					     image.table);
				}
				bases[i] = bases[j];
			}
		}
		if (bases[i] == UINT32_MAX) {
			auto& count = counts[static_cast<size_t>(kind)];
			bases[i]    = count;
			count += image.table_capacity;
		}
	}
	return bases;
}

struct Bindings {
	std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)> descriptor_counts {};
	uint64_t descriptor_mask = 0;
	uint32_t push_data_start_dword = PushData::NoStart;
	uint32_t dispatch_thread_dword = PushData::NoStart;
	// Native vertex draw parameters (base vertex, base instance, draw id): 3 push-data dwords.
	uint32_t native_draw_dword = PushData::NoStart;
	uint32_t memory_offset_dword = 0;

	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return memory_offset_dword +
		       (descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] + 3u) / 4u;
	}
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const Bindings& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t MaxBuffers      = 64;
	static constexpr uint32_t MaxImages       = 64;
	static constexpr uint32_t MaxSamplers     = 32;
	static constexpr uint32_t MaxSampledPairs = 64;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::vector<uint32_t>            user_data_registers;
	uint64_t                        live_buffers = 0;
	// User-data registers holding the low dword of a 64-bit base address that DMA (FLAT/global)
	// accesses offset from; the next register holds the high dword.
	std::vector<uint32_t>            dma_base_registers;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                            has_bitwise_xor               = false;
	bool                            uses_dma                      = false;
	bool                            uses_swizzle                  = false;
	bool                            uses_lds                      = false;
	bool                            uses_gds                      = false;
	bool                            uses_flattened_srt            = false;
	bool                            uses_dispatch_threads         = false;
	bool                            bvh                           = false;
	bool                            subgroup_ballot               = false;
	bool                            subgroup_barrier              = false;
	bool                            subgroup_shuffle              = false;
	bool                            subgroup_local_invocation_id  = false;
	bool                            compute_derivatives           = false;
	bool                            image_gather_extended         = false;
	bool                            function_lds                  = false;
	// Every LDS access of a non-compute stage is LaneId * 4 plus constants (guest VGPR spills to
	// per-lane LDS slots; experiment LdsLaneSlots). The per-invocation array is then indexed by
	// the constants alone: a shift by LaneId * 4 for every access, which keeps all aliasing,
	// makes the array small (function_lds_dwords) and lets the host compiler keep it in registers.
	bool                            function_lds_lane_slots       = false;
	uint32_t                        function_lds_dwords           = 0;
	bool                            function_scratch              = false;
	// Experiment MovRelArrays: the register arrays (RegArrayLoad/Store immediates) the program uses.
	std::vector<uint32_t>           reg_arrays;
	bool                            pixel_valid_mask              = false;
	bool                            buffer_int64_atomics          = false;
	bool                            buffer_u8                     = false;
	bool                            buffer_u16                    = false;
	bool                            shared_int64_atomics          = false;
	bool                            coherent_buffers              = false;
	bool                            float64                       = false;
	bool                            native_draw                   = false;
	bool                            subgroup_arithmetic           = false;
	bool                            float16                       = false;
	// KYTY_WORKGROUP_BUFFER_SYNC: barriers also order and publish buffer memory.
	bool                            workgroup_buffer_sync         = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct DescriptorSource {
	struct IndirectDescriptor {
		struct SelectorRead {
			uint32_t source = UINT32_MAX;
			uint32_t stride = 0;
			uint32_t offset = 0;

			bool operator==(const SelectorRead& other) const = default;
		};
		std::optional<SelectorRead> selector;
		uint32_t table_source    = 0;
		uint32_t table_offset    = 0;
		uint32_t table_immediate = 0;
		uint32_t table_stride    = 0;
		uint32_t table_record_bytes = 0;
		bool     table_scalar = false;
		uint32_t workgroup_axis  = UINT32_MAX;
		// The table entry is at ((key * table_stride) & table_mask) + table_offset.
		uint32_t table_mask      = UINT32_MAX;
		// A bindless T# table: every table entry is a candidate.
		bool     table_array     = false;
		Value    key_count;
		Value                 selector_first;
		Value    selector_mask;
		std::vector<uint32_t> sources;

		bool operator==(const IndirectDescriptor& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectDescriptor> indirect_descriptor;

	bool operator==(const DescriptorSource& other) const = default;
};

struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
	std::vector<uint32_t> srt_reads;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          has_address_writes  = false;
	ShaderInfo                    info;
	Bindings                      bindings;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
};

// Resource analysis retained by the shader cache. It owns immutable descriptor/SRT,
// condition and fill values without translated blocks, plus reusable evaluation scratch.
struct ResourcePlan {
	// Compiled descriptor gathers avoid interpreting literal/user-data leaves on
	// every refresh. Register numbers stay absolute so rebasing remains valid.
	struct DescriptorGather {
		struct Word {
			enum class Kind { Literal, UserData, Flat };
			uint32_t value = 0;
			Kind kind = Kind::Literal;
			// Derived at plan compilation/load; deliberately excluded from the fixture archive.
			uint32_t read_slot = UINT32_MAX;
		};
		std::array<Word, 8> words {};
		bool valid = false;
		bool needs_flat = false;
	};
	struct EvaluationContext {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};

		std::vector<Entry> values;
		uint64_t           generation = 0;
		uint64_t           cfg_gather_epoch = 0;
		std::vector<uint64_t> cfg_gather_slot_epochs;
		// Keep the vector identity, not its data pointer: indirect materialization
		// can append mappings and reallocate its storage after a successful refresh.
		const std::vector<uint32_t>* refreshed_flat = nullptr;
		// Reused demand-tape scratch; scoped to an evaluator session, never a draw cache.
		std::vector<uint64_t> tape_values;
		std::vector<uint8_t> tape_marks;
		std::vector<std::pair<uint32_t, uint8_t>> tape_pending;
	};

	ResourcePlan();
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>                     value_storage;
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<DescriptorGather>       descriptor_gathers;
	std::vector<ResourceBlock>          control_flow;
	std::vector<SrtRead>                srt_reads;
	// Contiguous scalar reads eligible for one backing read, indexed by first slot.
	std::vector<uint32_t>               srt_batch_counts;
	std::vector<uint8_t>                clean_flat_slots;
	bool                                requires_specialization_memory = false;
	std::vector<std::pair<uint64_t, uint64_t>> source_reads;
	bool                                capture_specialization_reads = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
	UniformCopy                         uniform_copy;
	std::unique_ptr<Xbyak::CodeGenerator> srt_code;
	// GPU-thread scratch for nested clean/EXEC memos, activity and material keys.
	mutable std::deque<EvaluationContext> evaluation_contexts;
	mutable uint32_t                       evaluation_value_count = 0;
	mutable uint32_t                       evaluation_depth       = 0;
	mutable std::vector<uint8_t>            active_sources;
	mutable std::vector<uint8_t>            visited_blocks;
	mutable std::vector<uint32_t>           pending_blocks;
	mutable std::vector<uint32_t>           material_keys;
	// Dense evaluation index of each srt_reads[slot].value, filled on first batched refresh.
	mutable std::vector<uint32_t>           srt_evaluation_indices;
	// When every SRT read is a raw dword load from a user-data pointer pair plus an immediate
	// offset, RefreshFlatBuffer computes the addresses directly instead of walking the IR.
	struct DirectSrtRead {
		uint32_t low_register     = 0; // user-data index of the pointer's low half
		uint32_t high_register    = 0;
		int64_t  relative         = 0; // dword-aligned immediate offset, as RawReadAddress applies it
		uint32_t evaluation_index = 0;
		uint32_t run              = 1; // contiguous slots starting here (0 inside a run)
		bool     clean            = false;
	};
	enum class DirectSrtState : uint8_t { Unknown, Direct, Unsupported };
	mutable DirectSrtState             direct_srt_state = DirectSrtState::Unknown;
	mutable std::vector<DirectSrtRead> direct_srt_reads;
	// Per descriptor source, 8 entries: the flat offset a dword copies (ReadConst of one slot),
	// or -1 when the dword needs the IR walk. Built with direct_srt_reads.
	mutable std::vector<int32_t>       direct_descriptor_offsets;
	// Flat slots a direct refresh filled (plans with control flow skip unreached blocks).
	mutable std::vector<uint8_t>       direct_filled;
	// Direct plans with control flow: each block's read runs (direct_cf_run_begin has one entry
	// per block plus an end), built for one batching mode, and the sources active on entry.
	struct DirectCfRun {
		uint32_t slot  = 0;
		uint32_t count = 0;
	};
	mutable std::vector<DirectCfRun> direct_cf_runs;
	mutable std::vector<uint32_t>    direct_cf_run_begin;
	mutable std::vector<uint8_t>     direct_cf_active;
	mutable int8_t                   direct_cf_batch = -1; // -1 = not built
	mutable bool                     direct_cf_valid = false;
	// Per block: the inputs its branch condition depends on (user data, flat words, shader base)
	// when it depends on nothing else, and the result for the inputs seen last.
	struct DirectCfCondition {
		enum class Kind : uint8_t { UserData, Flat, ShaderBase };
		struct Leaf {
			Kind     kind  = Kind::UserData;
			uint32_t index = 0;
		};
		static constexpr size_t MaxLeaves = 16;
		std::vector<Leaf>     leaves;
		std::vector<uint64_t> last_values;
		uint32_t              last_result = 0;
		bool                  cacheable   = false;
		bool                  has_last    = false;
	};
	mutable std::vector<DirectCfCondition> direct_cf_conditions;
	// Linearized SRT read expressions (KYTY_SRT_TAPE=1). Built lazily on the GPU thread,
	// one per variant: [0] = no clean evaluator, [1] = with clean evaluator. Ops are in the
	// exact order the recursive walker would evaluate them, so captured reads and failure
	// points match. Plans with conditional/lazy operators stay on the walker.
	struct SrtTape {
		enum class State : uint8_t { Unbuilt, Ready, Unsupported };
		enum class Kind : uint8_t { Immediate, UserData, ShaderBase, Inst, RawRead };
		struct Op {
			ValueOpcode             opcode {};
			const Inst*             source = nullptr;
			Kind                    kind       = Kind::Immediate;
			uint8_t                 clean      = 0;
			uint32_t                aux        = 0;
			uint32_t                memo_index = UINT32_MAX;
			uint8_t                 dependency_count = 0;
			std::array<uint32_t, 5> args {};
			uint64_t                imm = 0;
		};
		std::vector<Op>       ops;
		std::vector<uint32_t> slot_ops;
		std::vector<uint32_t> condition_ops;
		// Per condition: its op yields 0/1, or 2 where the walker's predicate fails.
		std::vector<uint8_t> condition_tri;
		// Cached per-demand-root DFS programs. Enter skips the whole subtree on
		// a memo hit; exit executes the op after its ordered dependencies.
		struct DemandStep { uint32_t op = 0; uint32_t next = UINT32_MAX; };
		struct DemandSchedule {
			std::vector<DemandStep> steps;
			State state = State::Unbuilt;
		};
		std::vector<DemandSchedule> demand_schedules;
		bool                  clean_root = false;
		State                 state      = State::Unbuilt;
	};
	mutable std::array<SrtTape, 2> srt_tapes;
	mutable std::array<SrtTape, 2> cfg_srt_tapes;
	mutable std::vector<uint64_t>  srt_tape_values;
	// Recorded CFG tape refreshes (KYTY_SRT_CFG_TRACE=1), one per cfg_srt_tapes variant.
	// A trace replays the exact op/memo/flat/activity sequence of a complete tape-only
	// refresh. Cond steps check the recorded predicate outcome and branch to other
	// observed paths, so the steps form a trie; an unseen outcome falls back to the DFS.
	struct CfgTrace {
		// Eval/CondEval re-run the recursive walker for roots the tape does not cover.
		// Gather marks a descriptor source CFG-gather ready (EvaluateDescriptor still verifies it).
		// CondTri: a tri-state condition op (0, 1, or 2 = predicate failed).
		enum class Kind : uint8_t { Exec, Memo, Flat, Activate, Cond, Eval, CondEval, Gather, End, CondTri };
		struct Step {
			Kind     kind = Kind::End;
			uint32_t a    = 0; // op (Exec/Memo/Flat/Cond), slot (Eval), block (Activate/CondEval), source (Gather)
			uint32_t b    = 0; // Flat/Eval: flat offset; Cond/CondEval (recording): outcome
			// Cond/CondEval: next step per outcome (0 = zero, 1 = non-zero, 2 = evaluation failed).
			std::array<uint32_t, 3> next {UINT32_MAX, UINT32_MAX, UINT32_MAX};
		};
		std::vector<Step>    steps;
		std::vector<uint8_t> base_active;
		uint32_t             paths    = 0;
		uint32_t             hits     = 0;
		uint32_t             misses   = 0;
		bool                 disabled = false;
		bool                 strict   = false; // recorded with a strict reader (conditions evaluated)
	};
	mutable std::array<CfgTrace, 2> cfg_traces;
	mutable std::vector<CfgTrace::Step> cfg_trace_scratch;
	// KYTY_SRT_COMPILED: a flat step program compiled from the straight-line tape (non-CFG plans) or from
	// the recorded CFG trace trie. It re-executes every operation, user-data read and guest read on every
	// call; nothing observed in a previous call is trusted. Same indexing as the tape: `out`/`args` address
	// the evaluator value array, immediates are folded into `consts`.
	struct CompiledSrt {
		enum class State : uint8_t { Unbuilt, Ready, Unsupported };
		enum class Code : uint8_t { UserData, ShaderBase, Alu, Read, Flat, Activate, Cond, CondTri, Memo, Gather, End };
		struct Step {
			Code                    code   = Code::End;
			uint8_t                 clean  = 0;
			// Dead-memo elimination: 0 = nothing can read this step's memo slot after it ran, so the
			// value/generation store and the rollback bookkeeping are skipped (see ClassifyMemoPublication).
			uint8_t                 publish = 1;
			// Read: number of adjacent same-pointer scalar loads starting here that may be fetched in
			// one reader call when their addresses are contiguous (1 = single read).
			uint8_t                 group  = 1;
			ValueOpcode             opcode {};
			uint32_t                out    = 0;           // value index written (Exec), read (Flat/Cond/Memo)
			uint32_t                memo   = UINT32_MAX;  // evaluator memo slot (UINT32_MAX = none)
			uint32_t                aux    = 0;           // UserData: register; Read: memory_info; Flat: flat offset;
			                                              // Activate: block; Gather: descriptor source
			std::array<uint32_t, 5> args {};
			uint64_t                imm    = 0;
			std::array<uint32_t, 3> next {UINT32_MAX, UINT32_MAX, UINT32_MAX};
		};
		std::vector<Step>                         steps;
		std::vector<std::pair<uint32_t, uint64_t>> consts;
		State    state        = State::Unbuilt;
		uint8_t  reason       = 0;   // why Unsupported (see SrtCompiledReasonName)
		uint32_t value_count  = 0;
		uint32_t source_steps = 0;  // CFG: trace size this program was compiled from
		uint32_t source_paths = 0;
		uint32_t read_steps   = 0;  // statistics: Read steps / merged groups
		uint32_t group_steps  = 0;
		uint32_t dead_memo_steps = 0; // steps whose memo publication was eliminated
		// CFG: per evaluator role (0 = ordinary, 1 = clean), the Gather step id whose memo/flat agreement
		// EvaluateDescriptor already verified per source. A trie step has one fixed predecessor path, so
		// the agreement is a property of the step, not of the values. Cleared whenever the program is rebuilt.
		std::array<std::vector<uint32_t>, 2> verified_gather;
	};
	mutable std::array<CompiledSrt, 2> compiled_nocfg;
	mutable std::array<CompiledSrt, 2> compiled_cfg;
	// KYTY_SRT_COMPILED input-keyed descriptor reuse: for one (evaluator mode, descriptor source) the memo
	// leaves its recursive evaluation can reach. Built lazily from a purity walk of the source's dwords.
	struct SourceReusePlan {
		enum class State : uint8_t { Unbuilt, Eligible, Ineligible };
		struct Leaf {
			const Inst* inst = nullptr;
			uint8_t     role = 0;     // 0 = the evaluating walker's own context, 1 = its clean partner's
			uint32_t    memo_index = 0;
		};
		State             state = State::Unbuilt;
		uint64_t          serial = 0;       // changes whenever the plan is rebuilt
		const void*       tape  = nullptr;  // tape the stop set was derived from
		size_t            tape_ops = 0;
		std::vector<Leaf> leaves;
	};
	mutable std::array<std::vector<SourceReusePlan>, 3> source_reuse_plans;
	// KYTY_SRT_SOURCE_REUSE: previous gather-path output per descriptor source (GPU thread only).
	mutable std::vector<DescriptorValue> source_reuse_last;
	mutable std::vector<uint8_t>         source_reuse_valid;
};

struct Program: ResourcePlan {
	Program() = default;
	~Program();

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
	Program(Program&&) noexcept         = default;
	Program& operator=(Program&& other) noexcept;
	CompiledShaderInfo TakeCompiledInfo() &&;

	std::vector<std::unique_ptr<Block>> block_storage;
	BlockList                           blocks;
	uint32_t                      wave_size      = 64;
	uint32_t                      scratch_dwords = 0;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	struct ScalarWrite { uint32_t pc; ScalarReg reg; };
	std::vector<ScalarWrite>      scalar_writes;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo>       export_info;
	bool                          has_address_writes = false;
	bool                          shader_info_complete = false;
	Bindings                      bindings;

};

uint32_t StorageBufferElementBits(const Program& program, const MemoryInfo& memory);

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);

void  ValidateProgram(const Program& program, bool require_ssa);
void  ResolveControlFlowIdentities(Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
