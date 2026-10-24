#include "common/experiments.h"
#include "common/runtimeFlags.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtProf.h"
#include "graphics/shader/recompiler/ir/passes/ResourcePlanFixture.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {
std::atomic<FaultPageReader> g_fault_page_reader_slot {nullptr};
thread_local bool            t_condition_reads_nonblocking = false;
} // namespace

bool ConditionReadsNonBlocking() noexcept {
	return t_condition_reads_nonblocking;
}

void SetConditionReadsNonBlocking(bool value) noexcept {
	t_condition_reads_nonblocking = value;
}

namespace {
}

void SetFaultPageReader(FaultPageReader reader) {
	g_fault_page_reader_slot.store(reader, std::memory_order_relaxed);
}

bool TryFaultPageRead(uint64_t address, void* data, uint64_t size) {
	const auto reader = g_fault_page_reader_slot.load(std::memory_order_relaxed);
	return reader != nullptr && reader(address, data, size);
}

namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectDescriptorProbes = 65536u;
thread_local ResourceSpecializationCacheStats g_specialization_cache_stats;

class MaterializationTimer {
public:
	explicit MaterializationTimer(MaterializationTimings* timings): m_timings(timings) {
		if (m_timings != nullptr) {
			++m_timings->calls;
			m_begin = std::chrono::steady_clock::now();
		}
	}
	~MaterializationTimer() { Step(m_phase); }
	void Step(MaterializationTimings::Phase phase) {
		if (m_timings == nullptr) return;
		const auto now = std::chrono::steady_clock::now();
		m_timings->ns[static_cast<size_t>(m_phase)] +=
		    std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_begin).count();
		m_phase = phase;
		m_begin = now;
	}

private:
	MaterializationTimings*               m_timings;
	MaterializationTimings::Phase         m_phase = MaterializationTimings::Phase::Setup;
	std::chrono::steady_clock::time_point m_begin {};
};

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto base_level = (descriptor.dwords[3] >> 12u) & 0xfu;
		const auto fragments  = (descriptor.dwords[3] >> 16u) & 0xfu;
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && fragments >= 1 && fragments <= 3 &&
		       (r128 || max_mip == fragments);
	}
	return true;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

enum class SamplerClass : uint8_t { Float, Integer, PointInteger };

SamplerClass ClassifySampler(const ImageResource& image) {
	if (image.numeric_class == Prospero::TextureNumericClass::Sint ||
	    (image.numeric_class == Prospero::TextureNumericClass::Uint &&
	     image.conversion_format != Prospero::BufferFormat::kInvalid)) {
		return SamplerClass::PointInteger;
	}
	return image.numeric_class == Prospero::TextureNumericClass::Uint ? SamplerClass::Integer
	                                                               : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return base <= last ? last - base + 1u : 0u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

struct ReadCapture {
	SrtRuntime                                  source;
	std::vector<std::pair<uint64_t, uint64_t>>& ranges;
	bool                                        coalesce = false;
};

bool CaptureStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_specialization_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureStrictSpan(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (capture.source.try_read_specialization_span == nullptr ||
	    !capture.source.try_read_specialization_span(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (capture.source.read_memory != nullptr) {
		if (!capture.source.read_memory(capture.source.userdata, address, values)) return false;
	} else {
		const auto reader = g_fault_page_reader_slot.load(std::memory_order_relaxed);
		if (reader == nullptr || !reader(address, values.data(), values.size_bytes())) {
			std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
		}
	}
	// Under KYTY_SRT_SOURCE_REUSE, extend the previous range when this read is exactly
	// adjacent (the union of covered bytes is unchanged).
	if (capture.coalesce && !capture.ranges.empty() &&
	    capture.ranges.back().first + capture.ranges.back().second == address) {
		capture.ranges.back().second += values.size_bytes();
		return true;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureOrdinarySpan(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (capture.source.try_read_memory_span == nullptr ||
	    !capture.source.try_read_memory_span(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value>       pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto value = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < slots.size()) {
				slots[slot.U32()] = 1u;
				pending.push_back(program.srt_reads[slot.U32()].value);
			}
			continue;
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t dynamic_offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto offset = dynamic_offset & ~uint64_t {3};
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	base &= AddressMask & ~uint64_t {3};
	if (offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

bool NormalizeIndirectStoreBuffer(DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    descriptor.Base48() == 0u || descriptor.GetSize() == 0u) {
		value.dwords.fill(0u);
	} else if (descriptor.Stride() != 1u || descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	           descriptor.OutOfBounds() != 3u) {
		return SpecializationFail("indirect buffer stores require linear byte bounds");
	}
	return true;
}

bool IsBoundedDescriptorTable(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect) {
	const auto* table = Source(program, indirect.table_source);
	return indirect.sources.empty() && !indirect.selector &&
	       indirect.workgroup_axis == UINT32_MAX && table != nullptr && table->dword_count == 4u;
}

template <typename Specialization, typename Normalize, typename Walker>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   const DescriptorSource::IndirectDescriptor& indirect,
                                   uint32_t resource_index, uint32_t dword_count,
                                   const SrtRuntime& runtime, Walker& clean,
                                   ResourceSnapshot&             snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   uint32_t maximum_resources, Normalize&& normalize) {
	const auto  descriptor_bytes = dword_count * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	const auto* selector = indirect.selector ? &*indirect.selector : nullptr;
	auto& keys = program.material_keys;
	const auto children_begin = descriptors.size();
	const auto root_resource = specializations[resource_index];
	const auto intern_candidate = [&](const DescriptorValue& candidate, uint32_t& ordinal) {
		ordinal = 0u;
		if (candidate == descriptors[resource_index]) return true;
		const auto found = std::find(descriptors.begin() + children_begin, descriptors.end(), candidate);
		ordinal = static_cast<uint32_t>(found - descriptors.begin() - children_begin + 1u);
		if (found == descriptors.end()) {
			if (descriptors.size() >= maximum_resources) return false;
			descriptors.push_back(candidate);
			auto child = root_resource;
			child.indirect_root = resource_index;
			specializations.push_back(child);
		}
		return true;
	};
	const auto read_keys = [&](const ShaderBufferResource& material, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (step == 4u) {
			return ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, keys);
		}
		for (auto& key: keys) {
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, {&key, 1}))
				return false;
			first += step;
		}
		return true;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	if (sources.empty()) {
		keys.clear();
		DescriptorValue material_value;
		DescriptorValue table_value;
		if ((selector != nullptr &&
		     !clean.EvaluateDescriptor(selector->source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return false;
		}
		ShaderBufferResource table;
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return false;
		}
		if (IsBoundedDescriptorTable(program, indirect)) {
			if (!indirect.table_scalar && table.Format() == Prospero::BufferFormat::kInvalid) {
				descriptors[resource_index] = {.dword_count = dword_count};
				return true;
			}
			if (table_size > UINT32_MAX ||
			    (indirect.table_record_bytes != 0u &&
			     (table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			      table.OutOfBounds() != 0u || (table_base & 3u) != 0u ||
			      (table.Stride() & 3u) != 0u || table.Stride() < indirect.table_record_bytes)))
				return false;
			// Descriptor bytes are refreshed together. The GPU selects the DWORD address;
			// no material selector or unused table field is interpreted on the CPU.
			keys.resize(table_size / 4u);
			if (!ReadScalarTable(table_base, table_size, 0u, runtime, keys)) return false;
			const auto mapping_offset = snapshot.flattened_srt.size();
			descriptors[resource_index] = {.dword_count = dword_count};
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			for (uint32_t word = 0; word < keys.size(); ++word) {
				DescriptorValue candidate {.dword_count = dword_count};
				std::copy_n(keys.begin() + word, std::min<size_t>(dword_count, keys.size() - word),
				            candidate.dwords.begin());
				if (!normalize(candidate)) return false;
				uint32_t ordinal;
				if (!intern_candidate(candidate, ordinal)) return false;
				if (ordinal == 0u) continue;
				snapshot.flattened_srt.resize(mapping_offset + 2u + word, 0u);
				snapshot.flattened_srt[mapping_offset + 1u + word] = ordinal;
			}
			if (descriptors.size() == children_begin) {
				snapshot.flattened_srt.resize(mapping_offset);
			} else {
				snapshot.flattened_srt[mapping_offset] =
				    static_cast<uint32_t>(snapshot.flattened_srt.size() - mapping_offset - 1u);
				auto& root = specializations[resource_index];
				root.indirect_root = resource_index;
				root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
				root.indirect_search_iterations = 0u;
			}
			return true;
		}
		if (selector == nullptr) {
			uint32_t key_count = 0;
			if (indirect.workgroup_axis != UINT32_MAX) {
				if (indirect.workgroup_axis >= runtime.workgroup_counts.size()) return false;
				key_count = runtime.workgroup_counts[indirect.workgroup_axis];
				if (key_count == 0u) return false;
			} else {
				if (table_value.dword_count != 2u || !clean.Evaluate(indirect.key_count, key_count))
					return false;
				if (std::bit_cast<int32_t>(key_count) <= 0) key_count = 0;
			}
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (key_count != 0u && uint64_t {indirect.table_offset} + indirect.table_immediate +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return false;
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), 0u);
		} else if (!indirect.selector_first.IsEmpty()) {
			ShaderBufferResource material;
			uint32_t             first = 0, count = 0;
			if (!DecodeBufferDescriptor(material_value, material) ||
			    !clean.Evaluate(indirect.selector_first, first) ||
			    !clean.Evaluate(indirect.key_count, count) || count > MaxIndirectDescriptorProbes)
				return false;
			// SMEM uses its instruction's byte stride; V# stride only supplies its bounds.
			const auto step = selector->stride != 0u ? selector->stride : material.Stride();
			if (selector->stride == 0u &&
			    (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
			     material.OutOfBounds() != 0u || uint64_t {selector->offset} + 4u > step ||
			     uint64_t {first} + count > material.NumRecords())) return false;
			const auto end = count != 0u
			    ? (uint64_t {first} + count - 1u) * step + selector->offset + 4u : 0u;
			if (end > uint64_t {UINT32_MAX} + 1u || end > material.GetSize())
				return false;
			if (!read_keys(material, uint64_t {first} * step + selector->offset, step, count))
				return false;
		} else if (!indirect.selector_mask.IsEmpty()) {
			uint32_t mask = 0;
			uint32_t count = 0;
			if (material_value.dword_count != 2u || table_value.dword_count != 2u ||
			    !clean.Evaluate(indirect.selector_mask, mask) ||
			    !clean.Evaluate(indirect.key_count, count) || count == 0u || count > 32u) {
				return false;
			}
			if (count < 32u) mask &= (1u << count) - 1u;
			const auto material_base =
			    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
			keys.reserve(std::popcount(mask));
			while (mask != 0u) {
				const auto index = std::countr_zero(mask);
				const auto offset = static_cast<uint64_t>(selector->offset) +
				                    static_cast<uint64_t>(index) * selector->stride;
				if (offset > UINT32_MAX) return false;
				uint32_t key = 0;
				if (!ReadScalarTable(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
				                     runtime, {&key, 1})) return false;
				keys.push_back(key);
				mask &= mask - 1u;
			}
		} else return false;
		if (selector != nullptr) {
			std::ranges::sort(keys);
			keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		}
	}

	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto key_count = sources.empty() ? keys.size() : sources.size();
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_count * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? keys[entry] : entry;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			const uint64_t table_offset =
			    uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
			    (indirect.table_immediate & ~3u);
			if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(dword_count))) {
				return false;
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return false;
		}
		if (!normalize(candidate)) return false;
		uint32_t ordinal = 0;
		if (entry == 0) {
			descriptors[resource_index] = candidate;
		} else if (!intern_candidate(candidate, ordinal)) {
			return false;
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_count);
	}
	return true;
}

// Exemplar that specializes a table root: a T# with the root's dimension and a native format,
// preferring the float textures that sampled material tables mostly hold.
bool TableExemplar(const ImageResource& root, const DescriptorValue& descriptor, bool float_only) {
	if (NullImageDescriptor(descriptor) || DescriptorIsCube(descriptor) != root.cube) {
		return false;
	}
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	return DescriptorDimension(descriptor, root.dimension) == root.dimension &&
	       numeric_class != Prospero::TextureNumericClass::Unsupported &&
	       (!float_only || numeric_class == Prospero::TextureNumericClass::Float) &&
	       ImageConversionFormat(format) == Prospero::BufferFormat::kInvalid &&
	       !Prospero::IsFmaskTextureFormat(format);
}

uint32_t ImageTableCapacity(size_t slots) {
	return std::bit_ceil(std::max<uint32_t>(static_cast<uint32_t>(slots), 256u));
}

// A bindless table: the key the shader computes from GPU data selects entry key & entry_mask,
// so every entry the table holds is a candidate. Distinct valid T#s become descriptor slots and
// the per-entry slot mapping is appended to the flattened SRT.
bool MaterializeImageTable(const ResourcePlan& program,
                           const DescriptorSource::IndirectDescriptor& indirect,
                           const DescriptorValue& table_value, uint32_t image_index,
                           const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                           ResourceSpecialization& specialization,
                           std::vector<ImageTableSnapshot>& previous,
                           std::vector<uint32_t>& mapping_offsets) {
	ShaderBufferResource table;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, table) ||
	    (indirect.table_offset & 31u) != 0u) {
		return false;
	}
	const auto base       = table.Base48();
	const auto size       = table.GetSize();
	const auto entry_mask = indirect.table_mask >> 5u;
	const auto entries    = std::min<uint64_t>(
	    {size > indirect.table_offset ? (size - indirect.table_offset) / 32u : 0u,
	     uint64_t {entry_mask} + 1u, MaxIndirectDescriptorProbes});
	const auto& root = program.info.images[image_index];
	const auto same_table = [&](const ImageTableSnapshot& candidate) {
		return candidate.base == base && candidate.size == size &&
		       candidate.offset == indirect.table_offset && candidate.entry_mask == entry_mask;
	};
	auto found = std::ranges::find_if(snapshot.image_tables, same_table);
	if (found == snapshot.image_tables.end()) {
		ImageTableSnapshot next {.base = base, .size = size, .offset = indirect.table_offset,
		                         .entry_mask = entry_mask};
		// Tables span megabytes and rarely change: read into a reused buffer and keep the
		// previous snapshot's words when they are equal.
		static thread_local std::vector<uint32_t> words;
		words.resize(static_cast<size_t>(entries) * 8u);
		if (!words.empty() && !ReadScalarTable(base, size, indirect.table_offset, runtime, words)) {
			return false;
		}
		const auto reusable = std::ranges::find_if(previous, [&](const ImageTableSnapshot& old) {
			return same_table(old) && old.raw == words && !old.slots.empty();
		});
		if (reusable != previous.end()) {
			next.raw     = std::move(reusable->raw);
			next.slots   = std::move(reusable->slots);
			next.mapping = std::move(reusable->mapping);
		} else {
			next.raw.swap(words);
			next.slots.push_back(DescriptorValue {.dword_count = 8u});
			next.mapping.assign(static_cast<size_t>(entries) + 1u, 0u);
			next.mapping[0] = static_cast<uint32_t>(entries);
			std::unordered_map<std::string_view, uint32_t> slot_of;
			for (uint32_t entry = 0; entry < entries; entry++) {
				DescriptorValue candidate {.dword_count = 8u};
				std::copy_n(next.raw.begin() + entry * 8u, 8u, candidate.dwords.begin());
				if (NullImageDescriptor(candidate) || !ValidImageDescriptor(candidate, root.r128)) {
					continue;
				}
				const std::string_view key(
				    reinterpret_cast<const char*>(next.raw.data() + entry * 8u),
				    8u * sizeof(uint32_t));
				const auto [slot, inserted] =
				    slot_of.try_emplace(key, static_cast<uint32_t>(next.slots.size()));
				if (inserted) {
					next.slots.push_back(candidate);
				}
				next.mapping[entry + 1u] = slot->second;
			}
		}
		snapshot.image_tables.push_back(std::move(next));
		mapping_offsets.push_back(static_cast<uint32_t>(snapshot.flattened_srt.size()));
		const auto& mapping = snapshot.image_tables.back().mapping;
		snapshot.flattened_srt.insert(snapshot.flattened_srt.end(), mapping.begin(), mapping.end());
		found = snapshot.image_tables.end() - 1;
	}
	const auto table_index     = static_cast<uint32_t>(found - snapshot.image_tables.begin());
	auto&      image           = specialization.images[image_index];
	image.table                = table_index;
	image.table_capacity       = ImageTableCapacity(found->slots.size());
	image.table_mapping_offset = mapping_offsets[table_index];
	image.table_entry_mask     = entry_mask;
	snapshot.images[image_index] = {.dword_count = 8u};
	for (const bool float_only: {true, false}) {
		const auto exemplar = std::ranges::find_if(found->slots, [&](const DescriptorValue& slot) {
			return TableExemplar(root, slot, float_only);
		});
		if (exemplar != found->slots.end()) {
			snapshot.images[image_index] = *exemplar;
			break;
		}
	}
	return true;
}

} // namespace

bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor) {
	if (NullImageDescriptor(descriptor) || !ValidImageDescriptor(descriptor, root.r128) ||
	    DescriptorIsCube(descriptor) != root.cube) {
		return false;
	}
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto dimension = DescriptorDimension(descriptor, root.dimension);
	const auto is_2d     = [](Decoder::ImageDimension value) {
		return value == Decoder::ImageDimension::Dim2D ||
		       value == Decoder::ImageDimension::Dim2DArray;
	};
	return (dimension == root.dimension || (is_2d(dimension) && is_2d(root.dimension))) &&
	       Prospero::SampledTextureNumericClass(format) == root.numeric_class &&
	       ImageConversionFormat(format) == root.conversion_format &&
	       !Prospero::IsFmaskTextureFormat(format);
}

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::array<std::array<uint32_t, 3>, ShaderInfo::MaxSamplers> mapping;
	std::array<Binding, ShaderInfo::MaxSamplers>                bindings;
	uint32_t                                                  sampler_count = 0;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization,
                                        ResourceSpecializationCache* cache) {
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		ResourceSpecializationCache::ImageEntry* cached = nullptr;
		std::array<uint32_t, 7> cache_key {};
		const bool null = NullImageDescriptor(descriptor);
		if (cache != nullptr) {
			const auto* source = Source(program, base.source);
			if (i < program.info.images.size() &&
			    image.indirect_root == ImageResource::NoIndirectImage && source != nullptr &&
			    !source->indirect_descriptor.has_value()) {
				cached = &cache->images[i];
				std::copy_n(descriptor.dwords.begin() + 1, cache_key.size(), cache_key.begin());
				// Address word 0 changes independently of shader typing. Nullness is separate.
				if (cached->valid && cached->key == cache_key && cached->null == null) {
					image = cached->value;
					++g_specialization_cache_stats.image_hits;
					continue;
				}
				++g_specialization_cache_stats.image_misses;
			} else {
				++g_specialization_cache_stats.indirect_uncached;
			}
		}
		const auto retain = [&] {
			if (cached != nullptr) {
				*cached = {.key = cache_key, .value = image, .null = null, .valid = true};
			}
		};
		image.mip_count = ImageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(
			    fmt::format("image descriptor {} has an invalid mip range", i));
		}
		if (null) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			retain();
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const bool raw_sint_storage = storage && format == Prospero::BufferFormat::k32SInt &&
		                              base.written && !base.read && !base.atomic;
		image.numeric_class         = Prospero::SampledTextureNumericClass(format);
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		retain();
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (key_count == 0u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * (root.indirect_search_iterations == 0u ? 1u : 2u) >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			if (image.numeric_class != image_class.numeric_class ||
			    (!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension))) ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	if (base.samplers.size() > plan.mapping.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
		for (const auto candidate: base.samplers[pair.sampler].indirect_resources) {
			if (candidate >= base.samplers.size()) return false;
			usage[candidate] |= usage[pair.sampler];
		}
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : plan.sampler_count++;
			if (target >= ShaderInfo::MaxSamplers) return false;
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	// The renderer checks captured scalar reads against final buffer and image write ranges.
	if (program.has_address_writes) {
		return {};
	}
	std::unordered_map<const Block*, uint32_t> indices;
	for (uint32_t i = 0; i < program.blocks.size(); i++) {
		if (!indices.emplace(program.blocks[i], i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&             info       = *program.blocks[i];
		const auto&           terminator = info.terminator;
		const std::array        targets {terminator.true_block, terminator.false_block};
		std::span<Block* const> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors = std::span(targets).first(1); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors      = targets;
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch: successors = info.ImmSuccessors(); break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	for (auto& block: blocks) block.condition = predicate(block.condition);
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard,
                                                      uint32_t depth = 0) {
	value = ResolveActiveU32(value, guard);
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, guard, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, guard, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

// One buffer store whose data is a load of the same element from another buffer at the same
// invocation index: dst[i] = src[i]. Both accesses are raw, index-addressed and unconditional
// for the whole dispatch, so the copied range follows from the descriptors and dispatch size.
static UniformCopy AnalyzeUniformCopy(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    program.info.uses_dma ||
	    !program.info.samplers.empty() || !program.info.images.empty() ||
	    program.info.buffers.size() != 2) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const Block*                     block = program.blocks[0];
	const Inst*                      store = nullptr;
	for (;;) {
		if (!visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write) return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch || term.true_block == nullptr) return {};
		block = term.true_block;
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
	                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
	constexpr std::array loads {ValueOpcode::LoadBufferU32, ValueOpcode::LoadBufferU32x2,
	                            ValueOpcode::LoadBufferU32x3, ValueOpcode::LoadBufferU32x4};
	const auto store_op = std::ranges::find(stores, store->GetOpcode());
	if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
	    store->Arg(3).Resolve() != Value(0u))
		return {};
	const auto words = static_cast<uint32_t>(store_op - stores.begin() + 1);
	const auto guard = store->Arg(5).Resolve();
	if (!IsFullDispatchPredicate(guard)) return {};
	const auto* load = store->Arg(4).Resolve().TryInstruction();
	if (load == nullptr || load->GetOpcode() != loads[words - 1] ||
	    load->Arg(2).Resolve() != Value(0u) || load->Arg(3).Resolve() != Value(0u))
		return {};
	const auto load_guard = load->Arg(4).Resolve();
	if (load_guard != Value(true) && load_guard != guard) return {};
	const auto& dst = program.memory_info.at(store->Flags<MemoryFlags>().index);
	const auto& src = program.memory_info.at(load->Flags<MemoryFlags>().index);
	const auto  raw = [words](const MemoryInfo& memory) {
		return !memory.formatted && !memory.typed && memory.idxen && !memory.offen &&
		       memory.offset == 0 && memory.data_bits == 32 && memory.data_dwords == words;
	};
	if (!raw(dst) || !raw(src) || dst.resource == src.resource ||
	    dst.resource >= program.info.buffers.size() || src.resource >= program.info.buffers.size())
		return {};
	const auto& src_buffer = program.info.buffers[src.resource];
	const auto& dst_buffer = program.info.buffers[dst.resource];
	if (!src_buffer.read || src_buffer.written || dst_buffer.read || !dst_buffer.written) return {};
	// The load and the store must address the same element.
	if (ResolveActiveU32(load->Arg(1), guard) != ResolveActiveU32(store->Arg(1), guard)) return {};
	const auto address = FillIndex(store->Arg(1), 0, guard);
	if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
	UniformCopy result;
	result.valid        = true;
	result.source       = src.resource;
	result.destination  = dst.resource;
	result.words        = words;
	result.group_stride = static_cast<uint32_t>((*address)[2]);
	return result;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const auto*                      block = program.blocks.front();
	const Inst*                      store = nullptr;
	for (;;) {
		if (block == nullptr || !visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		block = term.true_block;
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true)) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.source_reads               = program.source_reads;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		if (target.indirect_descriptor.has_value()) {
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	plan.descriptor_gathers.resize(plan.descriptor_sources.size());
	for (size_t source_index = 0; source_index < plan.descriptor_sources.size(); ++source_index) {
		const auto& source = plan.descriptor_sources[source_index];
		auto& gather = plan.descriptor_gathers[source_index];
		if (source.indirect_descriptor.has_value() || source.dword_count > gather.words.size()) continue;
		gather.valid = true;
		for (uint32_t word = 0; word < source.dword_count; ++word) {
			const auto value = source.dwords[word].Resolve();
			auto& target = gather.words[word];
			if (value.IsImmediate()) {
				switch (value.GetType()) {
					case Type::U1: target.value = value.U1(); break;
					case Type::U8: target.value = value.U8(); break;
					case Type::U16: target.value = value.U16(); break;
					case Type::U32: target.value = value.U32(); break;
					case Type::U64: target.value = static_cast<uint32_t>(value.U64()); break;
					case Type::F32: target.value = std::bit_cast<uint32_t>(value.F32Value()); break;
					default: gather.valid = false; break;
				}
			} else if (const auto* inst = value.TryInstruction();
			           inst != nullptr && inst->GetOpcode() == ValueOpcode::GetUserData) {
				target.kind = ResourcePlan::DescriptorGather::Word::Kind::UserData;
				target.value = RegIndex(inst->Arg(0).ScalarRegister());
			} else if (const auto* inst = value.TryInstruction();
			           inst != nullptr && inst->GetOpcode() == ValueOpcode::ReadConst) {
				const auto slot = inst->Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= program.srt_reads.size()) {
					gather.valid = false;
				} else {
					target.kind = ResourcePlan::DescriptorGather::Word::Kind::Flat;
					target.value = program.srt_reads[slot.U32()].flat_offset;
					target.read_slot = slot.U32();
					// Each cached slot must name one read, not a slot overwritten by
					// a different read during the refresh.
					gather.valid = target.value < program.srt_reads.size() &&
					    std::ranges::count_if(program.srt_reads, [&](const auto& read) {
						    return read.flat_offset == target.value;
					    }) == 1;
					gather.needs_flat = true;
				}
			} else {
				gather.valid = false;
			}
			if (!gather.valid) break;
		}
	}
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// A proven uniform factor can decide a branch even when its other lanes are unknown.
	// Keep only that Boolean structure, never the varying shader dependency graph.
	Value unknown;
	// Predicates share subterms; memoize per instruction so the walk stays linear in DAG size.
	std::unordered_map<const Inst*, Value> cloned_predicates;
	std::function<Value(Value)> ClonePredicateImpl;
	bool capture_control_reads = false;
	std::unordered_map<const Inst*, Value> predicates;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		const auto* key = value.IsEmpty() ? nullptr : value.TryInstruction();
		if (key == nullptr) return ClonePredicateImpl(value);
		if (const auto it = cloned_predicates.find(key); it != cloned_predicates.end()) {
			return it->second;
		}
		auto result = ClonePredicateImpl(value);
		cloned_predicates.emplace(key, result);
		return result;
	};
	ClonePredicateImpl = [&](Value value) -> Value {
		if (value.IsEmpty()) return {};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr)
			return ValidateRuntimeValue(program, value, RuntimeValueType::Integer) ? value : Value {};
		// Preserve shared EXEC predicates, including subgraphs with no uniform factor.
		const auto [found, inserted] = predicates.try_emplace(inst);
		if (!inserted) return found->second;
		auto& result = found->second;
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			capture_control_reads |=
			    !ValidateRuntimeValue(program, value, RuntimeValueType::ImmutableInteger);
			return result = Clone(value);
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef || op == ValueOpcode::LogicalNot) {
			const auto operand = ClonePredicate(inst->Arg(0));
			if (operand.IsEmpty() || op == ValueOpcode::ConditionRef) return result = operand;
			auto& node = plan.value_storage.emplace_back(op);
			node.SetArg(0, operand);
			return result = Value(&node);
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr) return {};
		auto left = ClonePredicate(inst->Arg(0));
		auto right = ClonePredicate(inst->Arg(1));
		if (left.IsEmpty() && right.IsEmpty()) return {};
		if (left.IsEmpty() || right.IsEmpty()) {
			if (unknown.IsEmpty()) unknown = Value(&plan.value_storage.emplace_back(ValueOpcode::UndefU1));
			if (left.IsEmpty()) left = unknown;
			if (right.IsEmpty()) right = unknown;
		}
		auto& node = plan.value_storage.emplace_back(op);
		node.SetArg(0, left);
		node.SetArg(1, right);
		return result = Value(&node);
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	plan.uniform_fill = AnalyzeUniformFill(program);
	plan.uniform_copy = AnalyzeUniformCopy(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	bool capture_indirect_reads = false;
	for (const auto& source: plan.descriptor_sources) {
		if (!source.indirect_descriptor.has_value()) continue;
		const auto& indirect                = *source.indirect_descriptor;
		plan.requires_specialization_memory = true;
		capture_indirect_reads = true;
		if (indirect.selector)
			MarkCleanFlatSlots(plan, Source(plan, indirect.selector->source), plan.clean_flat_slots);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.selector_mask);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.selector_first);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.key_count);
		if (indirect.sources.empty()) {
			MarkCleanFlatSlots(plan, Source(plan, indirect.table_source), plan.clean_flat_slots);
		} else {
			for (const auto candidate: indirect.sources) {
				MarkCleanFlatSlots(plan, Source(plan, candidate), plan.clean_flat_slots);
			}
		}
	}
	plan.capture_specialization_reads |= capture_indirect_reads || capture_control_reads || !plan.source_reads.empty();
	if (plan.capture_specialization_reads) {
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.written) MarkCleanFlatSlots(plan, Source(plan, buffer.source), plan.clean_flat_slots);
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanFlatSlots(plan, Source(plan, image.source), plan.clean_flat_slots);
		}
	}
	if (capture_indirect_reads || !plan.source_reads.empty())
		plan.resource_tracking_complete &= !program.has_address_writes;
	// Runtime traversal limits these candidates to consecutive reads in a reached block.
	// Descriptor/offset expressions, reader semantics and bytes must all match.
	{
		plan.srt_batch_counts.assign(plan.srt_reads.size(), 1u);
		const auto raw = [&](uint32_t slot) -> const Inst* {
			const auto* inst = plan.srt_reads[slot].value.Resolve().TryInstruction();
			if (inst == nullptr ||
			    (inst->GetOpcode() != ValueOpcode::LoadAddressU32 &&
			     inst->GetOpcode() != ValueOpcode::ReadConstBuffer)) return nullptr;
			if (inst->NumArgs() != (inst->GetOpcode() == ValueOpcode::LoadAddressU32 ? 4u : 2u)) return nullptr;
			const auto index = inst->Flags<MemoryFlags>().index;
			if (index >= plan.memory_info.size()) return nullptr;
			const auto kind = inst->GetOpcode() == ValueOpcode::LoadAddressU32
			                      ? ResourceKind::ScalarAddress : ResourceKind::ScalarBuffer;
			return plan.memory_info[index].kind == kind ? inst : nullptr;
		};
		for (uint32_t first = 0; first < plan.srt_reads.size();) {
			const auto* head = raw(first);
			uint32_t count = 1;
			if (head != nullptr && plan.srt_reads[first].flat_offset < plan.clean_flat_slots.size()) {
				const auto start_offset = static_cast<int64_t>(static_cast<int32_t>(
				    plan.memory_info[head->Flags<MemoryFlags>().index].offset)) & ~int64_t{3};
				const auto flat_offset = plan.srt_reads[first].flat_offset;
				while (count < 64 && first + count < plan.srt_reads.size()) {
					const auto* next = raw(first + count);
					if (next == nullptr || next->GetOpcode() != head->GetOpcode() ||
					    next->Arg(0).Resolve() != head->Arg(0).Resolve() ||
					    next->Arg(1).Resolve() != head->Arg(1).Resolve() ||
					    plan.srt_reads[first + count].flat_offset != flat_offset + count ||
					    flat_offset + count >= plan.clean_flat_slots.size() ||
					    plan.clean_flat_slots[flat_offset + count] != plan.clean_flat_slots[flat_offset]) break;
					const auto offset = static_cast<int64_t>(static_cast<int32_t>(
					    plan.memory_info[next->Flags<MemoryFlags>().index].offset)) & ~int64_t{3};
					if (offset != start_offset + count * 4u) break;
					++count;
				}
			}
			for (uint32_t i = 0; i < count; ++i) plan.srt_batch_counts[first + i] = count - i;
			first += count;
		}
	}
	CompileSrtWalker(plan);
	return plan;
}

static std::atomic<uint64_t> g_compiled_verify_ticker {0};

// Image typing depends only on image words 1..7 and the null flag when no table is indirect.
static bool SpecializationReusable(const ResourcePlan& program, const ResourceSpecialization& specialization) {
	if (specialization.images.size() != program.info.images.size()) return false;
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto* source = Source(program, program.info.images[i].source);
		if (source == nullptr || source->indirect_descriptor.has_value() ||
		    specialization.images[i].indirect_root != ImageResource::NoIndirectImage)
			return false;
	}
	return true;
}

static bool TryReuseImageSpecialization(const ResourcePlan& program, ResourceDelta& delta,
                                        ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	if (!delta.spec_valid || delta.owner != &program || delta.spec_images_in.size() != snapshot.images.size() ||
	    delta.spec_images_out.size() != snapshot.images.size() ||
	    !SpecializationReusable(program, specialization))
		return false;
	for (size_t i = 0; i < snapshot.images.size(); ++i) {
		const auto& before = delta.spec_images_in[i];
		const auto& now    = snapshot.images[i];
		if (before.dword_count != now.dword_count || NullImageDescriptor(before) != NullImageDescriptor(now)) return false;
		for (uint32_t d = 1; d < 8; ++d) {
			if (before.dwords[d] != now.dwords[d]) return false;
		}
	}
	specialization.images = delta.spec_images_out;
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

static bool MaterializeResourcesCore(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	SRT_PROF_SCOPE(SrtProf::MatTotal);
	const Common::RuntimeFlags::EvaluationScope flags_scope;
	FrameBreakdown::g_counters.mat_calls.fetch_add(FrameBreakdown::Enabled() ? 1u : 0u, std::memory_order_relaxed);
	FrameBreakdown::Timer mat_total_timer(FrameBreakdown::g_counters.mat_total_ns);
	// Measurement only: the low-level readers feed g_spec_collect, so straight-line (tape) reads
	// that never reach snapshot.specialization_reads are still observed.
	struct SpecScope {
		const ResourceSnapshot&                    snap;
		std::vector<std::pair<uint64_t, uint64_t>> collected;
		std::vector<std::pair<uint64_t, uint64_t>>* previous = nullptr;
		bool                                       active   = FrameBreakdown::Verbose();
		SpecScope(const ResourceSnapshot& s) : snap(s) {
			if (active) {
				previous                      = FrameBreakdown::g_spec_collect;
				FrameBreakdown::g_spec_collect = &collected;
			}
		}
		~SpecScope() {
			if (!active) return;
			FrameBreakdown::g_spec_collect = previous;
			FrameBreakdown::SpecObserve(collected.empty() ? snap.specialization_reads : collected);
		}
	} spec_scope {snapshot};
	auto* cache = runtime.specialization_cache;
	if (cache != nullptr && cache->active) cache = nullptr;
	if (cache != nullptr) {
		if (cache->owner != &program || cache->buffers.size() != program.info.buffers.size() ||
		    cache->images.size() != program.info.images.size()) {
			cache->Reset();
			cache->owner = &program;
			cache->buffers.resize(program.info.buffers.size());
			cache->images.resize(program.info.images.size());
		}
		cache->active = true;
		++g_specialization_cache_stats.calls;
	}
	struct CacheAttempt {
		ResourceSpecializationCache* cache;
		bool success = false;
		~CacheAttempt() {
			if (cache != nullptr) {
				cache->active = false;
				if (!success) cache->Reset();
			}
		}
	} cache_attempt {cache};
	if (!program.resource_tracking_complete ||
	    ((program.requires_specialization_memory || !program.source_reads.empty()) &&
	     runtime.read_specialization_memory == nullptr)) {
		return false;
	}
	MaterializationTimer timer(runtime.materialization_timings);
	const bool capture_reads = program.capture_specialization_reads || !program.source_reads.empty();
	auto& reads = snapshot.specialization_reads;
	reads = program.source_reads;
	ReadCapture capture {runtime, reads, Common::RuntimeFlags::Current().srt_source_reuse};
	SrtRuntime observed = runtime;
	if (capture_reads) {
		observed.userdata = &capture;
		observed.read_specialization_memory = runtime.read_specialization_memory != nullptr
		                                         ? CaptureStrictRead : nullptr;
		observed.read_memory = CaptureOrdinaryRead;
		observed.try_read_specialization_span = runtime.try_read_specialization_span != nullptr
		                                           ? CaptureStrictSpan : nullptr;
		observed.try_read_memory_span = runtime.try_read_memory_span != nullptr
		                                   ? CaptureOrdinarySpan : nullptr;
	}
	SrtWalker clean(program, CleanRuntime(observed));
	SrtWalker walker(program, observed, program.clean_flat_slots,
	                 capture_reads || program.requires_specialization_memory ? &clean : nullptr);
	timer.Step(MaterializationTimings::Phase::Flat);
	bool flat_ok;
	{
		FrameBreakdown::Timer timer(FrameBreakdown::g_counters.mat_flat_ns);
		flat_ok = walker.RefreshFlatBuffer(snapshot.flattened_srt);
	}
	if (!flat_ok) {
		return false;
	}
	timer.Step(MaterializationTimings::Phase::Fill);
	const auto active = std::span<const uint8_t>(program.active_sources);
	snapshot.uniform_fill = {};
	snapshot.uniform_copy = program.uniform_copy;
	const auto& fill = program.uniform_fill;
	const auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	uint32_t fill_stop = 0;
	bool     fill_eval_failed = false;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		fill_stop        = i;
		fill_eval_failed = !clean.Evaluate(fill.values[i], stored[i]);
		uniform_fill     = !fill_eval_failed && stored[i] == stored[0];
	}
	snapshot.uniform_pattern        = {};
	snapshot.uniform_pattern_values = {};
	// Computed when KYTY_FILL_PATTERNS=1 was set at startup, so cached snapshots already hold the
	// pattern when an A/B switches experiment bit 9 on; the consumers check the bit.
	static const bool compute_patterns = [] {
		const char* value = std::getenv("KYTY_FILL_PATTERNS");
		return value != nullptr && value[0] == '1';
	}();
	if (words > 1 && !uniform_fill && !fill_eval_failed && compute_patterns) {
		// The words differ: evaluate the rest; a proven element pattern is still known memory.
		bool pattern = true;
		for (uint32_t i = fill_stop + 1; i < words && pattern; ++i) {
			pattern = clean.Evaluate(fill.values[i], stored[i]);
		}
		if (pattern) {
			snapshot.uniform_pattern        = fill.fill;
			snapshot.uniform_pattern_values = stored;
		}
	}
	if (words != 0 && !uniform_fill) {
		// KYTY_DBG_FILL_DIAG=1: why a planned uniform fill was not taken (NHL27FILLVALUE, once per
		// shader and outcome, at most 256 lines).
		static const bool diag = [] {
			const char* value = std::getenv("KYTY_DBG_FILL_DIAG");
			return value != nullptr && value[0] == '1';
		}();
		if (diag) {
			static std::mutex                   diag_mutex;
			static std::unordered_set<uint64_t> seen;
			const uint64_t key = program.shader_hash ^ (uint64_t {fill_stop} << 60u) ^
			                     (fill_eval_failed ? (uint64_t {1} << 59u) : 0u) ^
			                     (fill_eval_failed ? 0u : (uint64_t {stored[fill_stop]} << 20u));
			bool first = false;
			{
				std::scoped_lock lock {diag_mutex};
				first = seen.size() < 256 && seen.insert(key).second;
			}
			if (first) {
				std::printf("NHL27FILLVALUE: shader=0x%016llx kind=%d words=%u stop_word=%u %s values=%08x %08x %08x %08x\n",
				            static_cast<unsigned long long>(program.shader_hash),
				            static_cast<int>(fill.fill.kind), words, fill_stop,
				            fill_eval_failed ? "evaluate_failed" : "words_differ", stored[0], stored[1],
				            stored[2], stored[3]);
				std::fflush(stdout);
			}
		}
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			FrameBreakdown::Timer timer(FrameBreakdown::g_counters.mat_eval_ns);
			FrameBreakdown::g_counters.mat_evals.fetch_add(FrameBreakdown::Enabled() ? 1u : 0u, std::memory_order_relaxed);
			return (capture_reads && written ? clean : walker).EvaluateDescriptor(source, value);
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	timer.Step(MaterializationTimings::Phase::Buffers);
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return false;
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return false;
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (active.empty() || active[base.source]) {
					FrameBreakdown::Timer indirect_timer(FrameBreakdown::g_counters.mat_indirect_ns);
					if (!MaterializeIndirectDescriptor(
					        program, *source->indirect_descriptor, i, 4u, observed, clean, snapshot,
					        snapshot.buffers, specialization.buffers, ShaderInfo::MaxBufferRoots,
					        NormalizeIndirectStoreBuffer))
						return false;
				}
			} else if (!evaluate(base.source, snapshot.buffers[i], base.written)) {
				return false;
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		ResourceSpecializationCache::BufferEntry* cached = nullptr;
		const std::array<uint32_t, 2> cache_key {descriptor.fields[1] & 0xffff0000u,
		                                        descriptor.fields[3]};
		if (cache != nullptr) {
			const auto* source = Source(program, base.source);
			if (i < program.info.buffers.size() && source != nullptr &&
			    !source->indirect_descriptor.has_value()) {
				cached = &cache->buffers[i];
				// NumRecords and the 48-bit address stay live in snapshot, not specialization.
				if (cached->valid && cached->key == cache_key) {
					specialization.buffers[i] = cached->value;
					++g_specialization_cache_stats.buffer_hits;
					continue;
				}
				++g_specialization_cache_stats.buffer_misses;
			} else {
				++g_specialization_cache_stats.indirect_uncached;
			}
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		if (cached != nullptr) *cached = {.key = cache_key, .value = buffer, .valid = true};
	}
	timer.Step(MaterializationTimings::Phase::Images);
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	auto previous_tables = std::move(snapshot.image_tables);
	snapshot.image_tables.clear();
	std::vector<uint32_t> table_mappings;
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .table = image.table,
		    .table_capacity = image.table_capacity,
		    .table_mapping_offset = image.table_mapping_offset,
		    .table_entry_mask = image.table_entry_mask,
		    .cube = image.cube,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return false;
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			if (indirect.table_array) {
				DescriptorValue table;
				if (!clean.EvaluateDescriptor(indirect.table_source, table) ||
				    !MaterializeImageTable(program, indirect, table, i, observed, snapshot,
				                           specialization, previous_tables, table_mappings)) {
					return false;
				}
				continue;
			}
			const bool bounded_table = IsBoundedDescriptorTable(program, indirect);
			FrameBreakdown::Timer indirect_timer(FrameBreakdown::g_counters.mat_indirect_ns);
			if (!MaterializeIndirectDescriptor(
			        program, indirect, i, 8u, observed, clean, snapshot, snapshot.images,
			        specialization.images, UINT32_MAX, [&](DescriptorValue& value) {
				        // A broad heap also contains resources for other typed image operations.
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128) ||
				            (bounded_table && DescriptorDimension(value, image.dimension) != image.dimension))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				return false;
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return false;
			}
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	timer.Step(MaterializationTimings::Phase::Samplers);
	snapshot.samplers.resize(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
		auto source = program.info.samplers[i].source;
		if (source >= program.descriptor_sources.size()) return false;
		const auto& descriptor = program.descriptor_sources[source];
		if (descriptor.indirect_descriptor) {
			if (descriptor.indirect_descriptor->sources.empty()) return false;
			source = descriptor.indirect_descriptor->sources[0];
		}
		if (!evaluate(source, snapshot.samplers[i])) {
			return false;
		}
		if (program.info.samplers[i].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			// MipNone always selects the base level. Explicit point gathers currently require
			// encoded-zero primary and secondary bias; linear primary-mip selection is unsupported.
			if (filter > 1u || (filter == 1u && (control & 0xfffffu) != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires mip filtering None or Point with zero LOD biases");
			}
		}
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	timer.Step(MaterializationTimings::Phase::Specialization);
	{
		FrameBreakdown::Timer timer(FrameBreakdown::g_counters.mat_spec_ns);
		auto* delta = runtime.delta;
		if (delta != nullptr && TryReuseImageSpecialization(program, *delta, snapshot, specialization)) {
			// Same image words as the last successful build: the typing is the same function of them.
			cache_attempt.success = true;
			SrtCompiledCountSpec(false);
		} else {
			if (delta != nullptr) {
				delta->spec_images_in = snapshot.images;
				SrtCompiledCountSpec(true);
			}
			cache_attempt.success = BuildResourceSpecialization(program, snapshot, specialization, cache);
			if (delta != nullptr) {
				delta->owner      = &program;
				delta->spec_valid = cache_attempt.success && SpecializationReusable(program, specialization);
				if (delta->spec_valid) delta->spec_images_out = specialization.images;
			}
		}
	}
	return cache_attempt.success;
}

namespace {

// The ChangeMask of this call against the same program's previous successful call. Compare-then-copy:
// unchanged parts cost one pass and no copy.
template <typename T>
bool UpdateStored(std::vector<T>& stored, const std::vector<T>& current) {
	if (stored == current) return false;
	stored = current;
	return true;
}

void UpdateDelta(ResourceDelta& delta, const ResourceSnapshot& snapshot) {
	ResourceChangeMask mask;
	mask.valid = delta.have;
	if (delta.have && delta.buffers.size() == snapshot.buffers.size()) {
		for (size_t i = 0; i < snapshot.buffers.size(); ++i) {
			const auto& a = delta.buffers[i];
			const auto& b = snapshot.buffers[i];
			mask.buffer_addr |= a.dwords[0] != b.dwords[0] || ((a.dwords[1] ^ b.dwords[1]) & 0xffffu) != 0u;
			mask.buffer_size |= a.dwords[2] != b.dwords[2];
			bool fmt = a.dword_count != b.dword_count || ((a.dwords[1] ^ b.dwords[1]) & ~0xffffu) != 0u;
			for (uint32_t d = 3; d < b.dword_count && d < 8; ++d) fmt |= a.dwords[d] != b.dwords[d];
			mask.buffer_fmt |= fmt;
		}
	} else if (delta.have) {
		mask.buffer_addr = mask.buffer_size = mask.buffer_fmt = true;
	}
	const bool images   = UpdateStored(delta.images, snapshot.images);
	const bool samplers = UpdateStored(delta.samplers, snapshot.samplers);
	const bool flat     = UpdateStored(delta.flat, snapshot.flattened_srt);
	const bool user     = UpdateStored(delta.user_data, snapshot.user_data);
	mask.images            = delta.have && images;
	mask.samplers          = delta.have && samplers;
	mask.flat_changed      = delta.have && flat;
	mask.user_data_changed = delta.have && user;
	if (!delta.have || mask.buffer_addr || mask.buffer_size || mask.buffer_fmt) delta.buffers = snapshot.buffers;
	delta.mask = mask;
	SrtCompiledCountMask(!delta.have, mask.images, mask.samplers, mask.buffer_fmt, mask.buffer_addr,
	                     mask.buffer_size, mask.flat_changed, mask.user_data_changed);
	delta.have = true;
}

bool SameRangeUnion(const std::vector<std::pair<uint64_t, uint64_t>>& a,
                    const std::vector<std::pair<uint64_t, uint64_t>>& b) {
	const auto normalize = [](const auto& ranges, std::vector<std::pair<uint64_t, uint64_t>>& result) {
		for (const auto& [address, size]: ranges) {
			if (size != 0u) result.emplace_back(address, address + size);
		}
		std::sort(result.begin(), result.end());
		size_t write = 0;
		for (size_t i = 0; i < result.size(); ++i) {
			if (write != 0u && result[i].first <= result[write - 1].second)
				result[write - 1].second = std::max(result[write - 1].second, result[i].second);
			else
				result[write++] = result[i];
		}
		result.resize(write);
	};
	std::vector<std::pair<uint64_t, uint64_t>> left, right;
	normalize(a, left);
	normalize(b, right);
	return left == right;
}

// Names the first differing output, or nullptr when the two evaluations are bit-identical.
const char* FirstDifference(const ResourceSnapshot& a, const ResourceSpecialization& sa,
                            const ResourceSnapshot& b, const ResourceSpecialization& sb) {
	if (a.buffers != b.buffers) return "buffers";
	if (a.images != b.images) return "images";
	if (a.samplers != b.samplers) return "samplers";
	if (a.flattened_srt != b.flattened_srt) return "flattened_srt";
	if (a.user_data != b.user_data) return "user_data";
	if (!(a.uniform_fill == b.uniform_fill)) return "uniform_fill";
	if (!(sa == sb)) return "specialization";
	if (!SameRangeUnion(a.specialization_reads, b.specialization_reads)) return "specialization_reads";
	return nullptr;
}

bool CompiledVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAWPREP_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

} // namespace

// KYTY_SRT_COMPILED wrapper: the core evaluation, the per-program ChangeMask, and the sampled
// KYTY_DRAWPREP_VERIFY check that re-runs the existing (non-compiled) path and compares everything.
static bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	if (!runtime.allow_compiled || !Common::RuntimeFlags::Current().srt_compiled)
		return MaterializeResourcesCore(program, runtime, snapshot, specialization);
	const bool ok = MaterializeResourcesCore(program, runtime, snapshot, specialization);
	if (runtime.delta != nullptr) {
		if (ok) UpdateDelta(*runtime.delta, snapshot);
		else runtime.delta->Reset();
	}
	if (CompiledVerifyEnabled() && (g_compiled_verify_ticker.fetch_add(1, std::memory_order_relaxed) & 63u) == 0u) {
		ResourceSnapshot       reference;
		ResourceSpecialization reference_spec;
		SrtRuntime             old_path = runtime;
		old_path.allow_compiled         = false;
		old_path.delta                  = nullptr;
		old_path.specialization_cache   = nullptr;
		const auto saved_active = program.active_sources;
		const bool reference_ok = MaterializeResourcesCore(program, old_path, reference, reference_spec);
		const char* difference  = nullptr;
		if (reference_ok != ok) difference = "success";
		else if (ok) difference = FirstDifference(snapshot, specialization, reference, reference_spec);
		if (difference == nullptr && ok && program.active_sources != saved_active) difference = "active_sources";
		SrtCompiledCountVerify(difference != nullptr);
		if (difference != nullptr) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
				std::fprintf(stderr, "SRT compiled verify MISMATCH shader=0x%016llx stage=%u field=%s ok=%d ref_ok=%d\n",
				             static_cast<unsigned long long>(program.shader_hash),
				             static_cast<unsigned>(program.stage), difference, ok ? 1 : 0, reference_ok ? 1 : 0);
			}
			// The existing path is the reference: adopt its outputs.
			if (reference_ok) {
				snapshot       = std::move(reference);
				specialization = std::move(reference_spec);
			}
			if (runtime.delta != nullptr) runtime.delta->Reset();
			return reference_ok;
		}
	}
	return ok;
}

ResourceSpecializationCacheStats GetResourceSpecializationCacheStats() {
	return g_specialization_cache_stats;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
 // Separate opt-in from the legacy process-wide .arb recorder: those files lack
 // plan identity and combine changing memory from many evaluations.
 static const std::string directory = [] {
  const char* p = std::getenv("KYTY_RESOURCE_PLAN_RECORD");
  return std::string(p ? p : "");
 }();
 if (directory.empty() || program.srt_reads.empty())
  return MaterializeResourcesImpl(program,runtime,snapshot,specialization);
 // Optional scene gate. Touch this file after reaching the intended scene;
 // no navigation, restart or capture of startup plans is required.
 static const std::string trigger = [] {
  const char* p = std::getenv("KYTY_RESOURCE_PLAN_TRIGGER");
  return std::string(p ? p : "");
 }();
 if (!trigger.empty()) {
  static thread_local auto next_poll = std::chrono::steady_clock::time_point{};
  static thread_local bool armed = false;
  if (!armed) {
   const auto now = std::chrono::steady_clock::now();
   if (now >= next_poll) {
    next_poll = now + std::chrono::milliseconds(250);
    std::error_code ec;
    armed = std::filesystem::exists(trigger,ec) && !ec;
   }
   if (!armed) return MaterializeResourcesImpl(program,runtime,snapshot,specialization);
  }
 }
 static std::atomic<bool> exhausted{false};
 if (exhausted.load(std::memory_order_relaxed)) return MaterializeResourcesImpl(program,runtime,snapshot,specialization);
 static std::mutex mutex;
 static std::set<std::pair<uint64_t,uint32_t>> recorded;
 static const uint64_t selected_hash = [] {
  const char* p = std::getenv("KYTY_RESOURCE_PLAN_HASH");
  return p ? std::strtoull(p,nullptr,16) : uint64_t{0};
 }();
 if (selected_hash && program.shader_hash != selected_hash)
  return MaterializeResourcesImpl(program,runtime,snapshot,specialization);
 bool capture = false;
 {
  std::lock_guard<std::mutex> lock(mutex);
  if (recorded.size() < 32)
   capture = recorded.emplace(program.shader_hash,static_cast<uint32_t>(program.stage)).second;
  if (recorded.size()>=32) exhausted.store(true,std::memory_order_relaxed);
 }
 if (!capture) return MaterializeResourcesImpl(program,runtime,snapshot,specialization);
 std::error_code ec;
 std::filesystem::create_directories(directory,ec);
 if (ec) return MaterializeResourcesImpl(program,runtime,snapshot,specialization);
 const auto name = fmt::format("{:016x}-{}.rpf",program.shader_hash,static_cast<uint32_t>(program.stage));
 return CaptureResourcePlan(program,runtime,snapshot,specialization,
  (std::filesystem::path(directory)/name).string(),MaterializeResourcesImpl);
}
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.table                      = source.table;
		image.table_capacity             = source.table_capacity;
		image.table_mapping_offset       = source.table_mapping_offset;
		image.table_entry_mask           = source.table_entry_mask;
		image.cube                       = source.cube;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto original_sampler_count = samplers.size();
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::PointInteger;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		for (auto& candidate: samplers[index].indirect_resources) {
			candidate = sampler_plan.mapping[candidate][static_cast<uint32_t>(binding.type)];
			EXIT_IF(candidate == UINT32_MAX);
		}
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
		for (const auto candidate: samplers[pair.sampler].indirect_resources)
			samplers[candidate].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				if (memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds regardless of its address.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						for (uint32_t component = 0; component < count; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							const auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(one), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			EXIT_IF(image.indirect_root == memory.resource &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageGatherRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageRead);
			EXIT_IF(image.table_capacity != 0u && inst.GetOpcode() != ValueOpcode::ImageSampleRaw);
		}
	}
	for (auto& memory: memory_info) {
		if (memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap[buffer.image_alias];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
}

uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value) {
	// Mirrors BuildResourceSpecialization for sampled roots: entries it would reject, and
	// entries that need shader-side conversion, read as null instead of failing the draw.
	const auto format = static_cast<Prospero::BufferFormat>((value.dwords[1] >> 20u) & 0x1ffu);
	const auto tile = static_cast<Prospero::TileMode>((value.dwords[3] >> 20u) & 0x1fu);
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (NullImageDescriptor(value) || !ValidImageDescriptor(value, root.r128) ||
	    DescriptorDimension(value, root.dimension) != root.dimension ||
	    Prospero::IsFmaskTextureFormat(format) || format == Prospero::BufferFormat::k11_11_10UNorm ||
	    ImageConversionFormat(format) != Prospero::BufferFormat::kInvalid ||
	    (root.unfiltered_access_only && format == Prospero::BufferFormat::k16UInt &&
	     tile == Prospero::TileMode::kDepth) ||
	    numeric_class == Prospero::TextureNumericClass::Unsupported ||
	    (root.depth_compare && numeric_class != Prospero::TextureNumericClass::Float)) {
		value.dwords.fill(0u);
		return 0u;
	}
	return 1u + static_cast<uint32_t>(numeric_class) * 2u + (DescriptorIsCube(value) ? 1u : 0u);
}

ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind) {
	ImageResource resource;
	resource.resource_class         = ImageResourceClass::Sampled;
	resource.read                   = true;
	resource.heap                   = true;
	resource.dimension              = root.dimension;
	resource.unfiltered_access_only = root.unfiltered_access_only;
	resource.depth_compare          = root.depth_compare;
	resource.r128                   = root.r128;
	if (kind != 0u) {
		resource.numeric_class = static_cast<Prospero::TextureNumericClass>((kind - 1u) / 2u);
		resource.cube          = ((kind - 1u) & 1u) != 0u;
	}
	return resource;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
