#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/experiments.h"
#include "common/runtimeFlags.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceFixture.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtProf.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}
SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.try_read_memory_span = runtime.try_read_specialization_span;
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

// ConvertU32F32 as the guest's V_CVT_U32_F32: NaN and values at or below zero give 0, values at
// or above 2^32 give UINT32_MAX, the rest truncate.
[[nodiscard]] uint32_t SaturateF32ToU32(float value) {
	if (!(value > 0.0f)) {
		return 0;
	}
	if (value >= 4294967296.0f) {
		return UINT32_MAX;
	}
	return static_cast<uint32_t>(value);
}


// --- resource fixture recording ------------------------------------------------
// Opt-in recording of the memory access trace the SRT walker performs, so a real
// plan's workload can be replayed offline with no emulator run. Off unless
// KYTY_SRT_FIXTURE_RECORD names an output path.
//
// This wraps the runtime readers and calls straight through to the original reader,
// so enabling it cannot change evaluation results. The trace is the memory access
// stream; which reader issued a given read is a property of the plan, not of the
// bytes, so replay is keyed per address rather than per reader.
namespace FixtureRecord {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* path = std::getenv("KYTY_SRT_FIXTURE_RECORD");
		return path != nullptr && path[0] != '\0';
	}();
	return enabled;
}

inline std::string& Path() {
	static std::string path = [] {
		const char* value = std::getenv("KYTY_SRT_FIXTURE_RECORD");
		return std::string(value != nullptr ? value : "");
	}();
	return path;
}

// Flush after this many newly accumulated reads. A flush appends only the new
// records, so its cost is proportional to the batch, not to the whole fixture.
// An earlier version rewrote the entire file every 20k reads while holding a global
// mutex on every read; at ~26 MB per rewrite that stalled the walk path and the
// emulator stopped consuming CPU entirely.
inline constexpr uint64_t kFlushEveryReads = 20000;

inline std::mutex& FlushMutex() {
	static std::mutex mutex;
	return mutex;
}

inline ResourceFixture& Fixture() {
	static ResourceFixture fixture;
	return fixture;
}

// Records accumulated on walk threads and not yet written. Thread-local so the walk
// path takes no lock at all: readers only append to their own vector.
struct PendingReads {
	std::vector<FixtureRead> reads;
	std::vector<uint64_t>    registered;
};

inline std::vector<PendingReads*>& RegisteredPending() {
	static std::vector<PendingReads*> pending;
	return pending;
}

inline PendingReads& Pending() {
	static thread_local PendingReads pending;
	static thread_local bool        registered = false;
	if (!registered) {
		registered = true;
		std::lock_guard<std::mutex> lock(FlushMutex());
		RegisteredPending().push_back(&pending);
	}
	return pending;
}

inline uint64_t& FlushCount() {
	static uint64_t flushes = 0;
	return flushes;
}

// Writes everything accumulated on all threads into the shared accumulator and then
// writes the whole fixture out atomically. Runs inside the flush lock; walk threads are
// not blocked by it because they never take that lock.
//
// The accumulator is word-deduplicated, so its size is bounded by distinct addresses
// touched rather than by read count, which is what makes a whole-file write acceptable
// here. An incremental append path was tried and removed: delta tracking across merges
// was wrong three times in a row and silently dropped words.
inline void MergeAndFlush() {
	std::lock_guard<std::mutex> lock(FlushMutex());
	auto&                       fixture = Fixture();

	for (PendingReads* pending : RegisteredPending()) {
		for (auto& read : pending->reads) {
			fixture.AddRead(std::move(read));
		}
		pending->reads.clear();
	}
	fixture.Rebuild();

	if (!fixture.SaveAtomic(Path())) {
		return;
	}
	++FlushCount();
	std::printf("NHL27FIXTURE: reads=%llu records=%zu words_added=%llu words_kept=%llu "
	            "coalesced=%llu flush=%llu -> %s\n",
	            static_cast<unsigned long long>(fixture.TotalReads()), fixture.DistinctReads(),
	            static_cast<unsigned long long>(fixture.WordsAdded()),
	            static_cast<unsigned long long>(fixture.WordsRetained()),
	            static_cast<unsigned long long>(fixture.CoalescedRecords()),
	            static_cast<unsigned long long>(FlushCount()), Path().c_str());
	std::fflush(stdout);
}

inline void NoteRead(uint64_t address, std::span<const uint32_t> values, bool valid) {
	// No lock here on purpose: this is the per-read walk path.
	PendingReads& pending = Pending();
	FixtureRead   read;
	read.address = address;
	read.valid   = valid;
	read.values.assign(values.begin(), values.end());
	pending.reads.push_back(std::move(read));
	if (pending.reads.size() >= kFlushEveryReads) {
		MergeAndFlush();
	}
}

inline bool RecordRead(void* userdata, uint64_t address, std::span<uint32_t> values,
                       SrtMemoryReader reader) {
	// Pass through first, exactly as the unwrapped runtime would.
	const bool valid = reader != nullptr && reader(userdata, address, values);
	NoteRead(address, values, valid);
	return valid;
}

} // namespace FixtureRecord

bool RecordOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	const auto* state = static_cast<const SrtRuntime*>(userdata);
	return FixtureRecord::RecordRead(state->userdata, address, values, state->read_memory);
}

bool RecordSpecializationRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	const auto* state = static_cast<const SrtRuntime*>(userdata);
	return FixtureRecord::RecordRead(state->userdata, address, values,
	                                 state->read_specialization_memory);
}

// Wraps both readers in place. The runtime passed here is the walker's own copy, so
// the wrapper is correctly scoped to this evaluation.
//
// The original reader and its userdata are kept in a per-call holder rather than one
// shared thread_local: SRT walkers nest (a walker can own a clean evaluator), and a
// shared holder would redirect an outer walker's live readers to the inner one.
void AttachFixtureRecording(SrtRuntime& runtime) {
	if (!FixtureRecord::Enabled()) {
		return;
	}
	{
		std::lock_guard<std::mutex> lock(FixtureRecord::FlushMutex());
		auto&                       fixture = FixtureRecord::Fixture();
		fixture.SetUserData(runtime.user_data);
		fixture.SetShaderBase(runtime.shader_base);
	}
	// Heap-allocated so the address stays stable, and never freed: a walker's readers
	// may be used after the attaching frame returns. One small allocation per walker
	// is irrelevant next to the recording itself, and growing this linearly would
	// need a matching detach to be safe.
	auto* holder = new SrtRuntime(runtime);
	if (runtime.read_memory != nullptr) {
		holder->read_memory = runtime.read_memory;
		runtime.read_memory = RecordOrdinaryRead;
	}
	if (runtime.read_specialization_memory != nullptr) {
		holder->read_specialization_memory = runtime.read_specialization_memory;
		runtime.read_specialization_memory = RecordSpecializationRead;
	}
	runtime.userdata = holder;
}

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool SrtTapeVerifyEnabled() {
	static const bool enabled = [] {
		const char* val = std::getenv("KYTY_SRT_TAPE_VERIFY");
		return val != nullptr && val[0] == '1';
	}();
	return enabled;
}

struct TapeReadRecord {
	uint64_t address = 0;
	bool strict = false;
	bool valid = false;
	std::vector<uint32_t> values;
};

struct TapeReadCapture {
	SrtRuntime source;
	std::vector<TapeReadRecord> reads;
	SrtRuntime clean_source;
};

template <bool specialization>
bool CaptureCleanTapeRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<TapeReadCapture*>(userdata);
	const auto& source = capture.clean_source;
	TapeReadRecord read {.address = address, .strict = true};
	read.values.resize(values.size());
	const auto reader = specialization ? source.read_specialization_memory : source.read_memory;
	read.valid = reader != nullptr && reader(source.userdata, address, values);
	std::ranges::copy(values, read.values.begin());
	capture.reads.push_back(std::move(read));
	return capture.reads.back().valid;
}

template <bool strict>
bool CaptureTapeRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<TapeReadCapture*>(userdata);
	const auto reader = strict ? capture.source.read_specialization_memory : capture.source.read_memory;
	TapeReadRecord read {.address = address, .strict = strict};
	read.values.resize(values.size());
	read.valid = reader != nullptr && reader(capture.source.userdata, address, values);
	std::ranges::copy(values, read.values.begin());
	capture.reads.push_back(std::move(read));
	return capture.reads.back().valid;
}

struct TapeReadReplay {
	const std::vector<TapeReadRecord>& reads;
	bool inconclusive = false;
	bool has_unmatched = false;
	bool unmatched_strict = false;
	uint64_t unmatched_address = 0;
	size_t unmatched_words = 0;
};

void MarkTapeReplayUnmatched(TapeReadReplay& replay, bool strict, uint64_t address, size_t words) {
	if (!replay.has_unmatched) {
		replay.has_unmatched = true;
		replay.unmatched_strict = strict;
		replay.unmatched_address = address;
		replay.unmatched_words = words;
	}
	replay.inconclusive = true;
}

struct TapeWordKey {
	uint64_t address;
	bool strict;
	bool operator==(const TapeWordKey&) const = default;
};

struct TapeWordKeyHash {
	size_t operator()(const TapeWordKey& key) const {
		const auto address_hash = std::hash<uint64_t> {}(key.address);
		return address_hash ^ (static_cast<size_t>(key.strict) + 0x9e3779b9u + (address_hash << 6u) +
		                       (address_hash >> 2u));
	}
};

struct TapeInstKey {
	const Inst* inst;
	bool clean;
	bool operator==(const TapeInstKey&) const = default;
};

struct TapeInstKeyHash {
	size_t operator()(const TapeInstKey& key) const {
		const auto inst_hash = std::hash<const Inst*> {}(key.inst);
		return inst_hash ^ (static_cast<size_t>(key.clean) + 0x9e3779b9u + (inst_hash << 6u) +
		                    (inst_hash >> 2u));
	}
};

struct GlobalTapeStats {
	std::atomic<uint64_t> runs {0}, failed {0}, walker {0}, built {0}, unsupported {0}, ops {0};
	std::atomic<uint64_t> cfg_runs {0}, cfg_fallback {0};
	std::atomic<uint64_t> cfg_scheduled {0};
	std::atomic<uint64_t> cfg_checked {0}, cfg_matched {0}, cfg_mismatch {0}, cfg_inconclusive {0};
	std::atomic<uint64_t> cfg_trace_hits {0}, cfg_trace_misses {0}, cfg_trace_recorded {0};
	std::array<std::atomic<uint64_t>, 8> reject {};
} g_tape_stats;

enum class TapeVerifyLogKind : uint8_t {
	UnstableReads,
	UnmatchedReplay,
	Mismatch,
	OpUnmatchedReplay,
	OpMismatch
};

struct TapeVerifyLogKey {
	const ResourcePlan* plan;
	TapeVerifyLogKind kind;
	bool operator==(const TapeVerifyLogKey&) const = default;
};

struct TapeVerifyLogKeyHash {
	size_t operator()(const TapeVerifyLogKey& key) const {
		const auto plan_hash = std::hash<const ResourcePlan*> {}(key.plan);
		return plan_hash ^ (static_cast<size_t>(key.kind) + 0x9e3779b9u + (plan_hash << 6u) +
		                    (plan_hash >> 2u));
	}
};

bool ShouldLogTapeVerify(const ResourcePlan& plan, TapeVerifyLogKind kind) {
	thread_local std::unordered_set<TapeVerifyLogKey, TapeVerifyLogKeyHash> logged;
	return logged.insert({&plan, kind}).second;
}

bool HasStableTapeReads(const std::vector<TapeReadRecord>& reads) {
	std::unordered_map<TapeWordKey, std::pair<bool, uint32_t>, TapeWordKeyHash> words;
	for (const auto& read: reads) {
		if (read.values.size() > (UINT64_MAX - read.address) / sizeof(uint32_t)) return false;
		for (size_t i = 0; i < read.values.size(); ++i) {
			const TapeWordKey key {read.address + i * sizeof(uint32_t), read.strict};
			const std::pair value {read.valid, read.valid ? read.values[i] : 0u};
			const auto [it, inserted] = words.emplace(key, value);
			if (!inserted && it->second != value) return false;
		}
	}
	return true;
}

void TransferCurrentEvaluationValues(const ResourcePlan::EvaluationContext& source,
	                                 ResourcePlan::EvaluationContext& destination) {
	if (destination.values.size() < source.values.size()) destination.values.resize(source.values.size());
	for (size_t i = 0; i < source.values.size(); ++i) {
		const auto& input = source.values[i];
		auto& output = destination.values[i];
		if (input.generation == source.generation) {
			output = {.value = input.value, .generation = destination.generation};
		} else if (input.generation == (source.generation | 2u)) {
			output = {.value = input.value, .generation = destination.generation | 2u};
		}
	}
}

template <bool strict>
bool ReplayTapeRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& replay = *static_cast<TapeReadReplay*>(userdata);
	if (values.size() > (UINT64_MAX - address) / sizeof(uint32_t)) {
		MarkTapeReplayUnmatched(replay, strict, address, values.size());
		return false;
	}
	const auto end = address + values.size_bytes();
	for (size_t i = 0; i < replay.reads.size(); ++i) {
		const auto& read = replay.reads[i];
		if (read.strict != strict || address < read.address ||
		    values.size() > (UINT64_MAX - read.address) / sizeof(uint32_t) ||
		    end > read.address + read.values.size() * sizeof(uint32_t)) {
			continue;
		}
		const auto offset = static_cast<size_t>((address - read.address) / sizeof(uint32_t));
		if (((address - read.address) & 3u) != 0u) continue;
		std::ranges::copy(std::span(read.values).subspan(offset, values.size()), values.begin());
		return read.valid;
	}
	MarkTapeReplayUnmatched(replay, strict, address, values.size());
	return false;
}

bool CanReplayTape(const ResourcePlan::SrtTape& tape, const SrtRuntime& runtime) {
	for (const auto& op: tape.ops) {
		if (op.kind != ResourcePlan::SrtTape::Kind::RawRead) continue;
		const bool strict = op.clean != 0u || op.opcode == ValueOpcode::LoadBufferU32;
		if ((strict ? runtime.read_specialization_memory : runtime.read_memory) == nullptr) return false;
	}
	return true;
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		// Uniform checks depend on the active EXEC mask; reuse a success under the same mask.
		// Without this, shared subexpressions are revalidated once per path (exponential).
		if (require_uniform) {
			if (const auto it = m_validated_uniform.find(inst); it != m_validated_uniform.end() &&
			    std::find(it->second.begin(), it->second.end(), m_active_mask) != it->second.end()) {
				return true;
			}
		}
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			if (valid && require_uniform) m_validated_uniform[inst].push_back(m_active_mask);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    source->GetOpcode() != ValueOpcode::CompositeConstructU32x2) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
	std::unordered_map<const Inst*, std::vector<Value>> m_validated_uniform;
};


} // namespace

void CompileSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.capture_specialization_reads || !program.control_flow.empty() ||
	    program.srt_reads.empty()) return;

	using Reads = std::map<int32_t, const SrtRead*>;
	std::map<uint32_t, Reads> roots;
	std::unordered_map<const Inst*, Reads> pointer_uses;
	std::unordered_map<const Inst*, int32_t> offsets;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    inst->NumArgs() < 2 || !IsRawRead(program, *inst) ||
		    read.flat_offset >= program.srt_reads.size()) return;
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return;
		const auto& memory = program.memory_info[inst->Flags<MemoryFlags>().index];
		const auto relative =
		    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
		    static_cast<int64_t>(offset.U32() & ~3u);
		if (relative < INT32_MIN || relative > INT32_MAX ||
		    !offsets.emplace(inst, static_cast<int32_t>(relative)).second) return;
	}
	uint32_t user_data_count = 0;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) return;
		const auto* low = handle->Arg(0).Resolve().TryInstruction();
		const auto* high = handle->Arg(1).Resolve().TryInstruction();
		if (low == nullptr || high == nullptr) return;
		Reads* uses = nullptr;
		if (low->GetOpcode() == ValueOpcode::GetUserData &&
		    high->GetOpcode() == ValueOpcode::GetUserData && low->NumArgs() == 1 &&
		    high->NumArgs() == 1 && low->Arg(0).GetType() == Type::ScalarReg &&
		    high->Arg(0).GetType() == Type::ScalarReg) {
			const auto reg = RegIndex(low->Arg(0).ScalarRegister());
			if (reg < program.user_data_base ||
			    RegIndex(high->Arg(0).ScalarRegister()) != reg + 1u) return;
			const auto base = reg - program.user_data_base;
			user_data_count = std::max(user_data_count, base + 2u);
			uses = &roots[base];
		} else {
			const auto lo = offsets.find(low);
			const auto hi = offsets.find(high);
			if (lo == offsets.end() || hi == offsets.end() ||
			    static_cast<int64_t>(lo->second) + 4 != hi->second ||
			    !EquivalentValue(program, low->Arg(0), high->Arg(0))) return;
			uses = &pointer_uses[low];
		}
		if (!uses->emplace(offsets.at(inst), &read).second) return;
	}
	std::unordered_set<const Inst*> visited;
	const auto reachable = [&](auto&& self, const Reads& reads) -> bool {
		for (const auto& [offset, read]: reads) {
			const auto* inst = read->value.ResolveInstruction();
			const auto child = pointer_uses.find(inst);
			if (child != pointer_uses.end() &&
			    (!visited.insert(inst).second || !self(self, child->second))) return false;
		}
		return true;
	};
	for (const auto& [reg, reads]: roots) {
		if (!reachable(reachable, reads)) return;
	}
	if (visited.size() != pointer_uses.size()) return;

	program.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code = *program.srt_code;
	Xbyak::Label failed, finished;
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.cmp(code.r8, user_data_count);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	code.mov(code.r9, AddressMask & ~uint64_t {3});
	// Follow shadPS4's VisitPointer: copy one table level in offset order, then
	// descend through its child pointers. Seed the descriptor evaluator's memo.
	const auto visit = [&](auto&& self, int32_t pointer_offset, const Reads& reads) -> void {
		code.push(code.rdi);
		code.mov(code.rdi, code.qword[code.rdi + pointer_offset]);
		code.and_(code.rdi, code.r9);
		const auto first = reads.begin()->first;
		const auto last = reads.rbegin()->first;
		if (first < 0) {
			code.mov(code.r10, -static_cast<int64_t>(first));
			code.cmp(code.rdi, code.r10);
			code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		if (last > 0) {
			code.mov(code.r10, AddressMask - last);
			code.cmp(code.rdi, code.r10);
			code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		for (const auto& [offset, read]: reads) {
			const auto index = read->value.ResolveInstruction()->EvaluationIndex(program.evaluation_value_count);
			using Entry = ResourcePlan::EvaluationContext::Entry;
			const auto slot = code.rdx + index * sizeof(Entry);
			code.mov(code.r10d, code.dword[code.rdi + offset]);
			code.mov(code.dword[code.rsi + read->flat_offset * sizeof(uint32_t)], code.r10d);
			code.mov(code.qword[slot + offsetof(Entry, value)], code.r10);
			code.mov(code.qword[slot + offsetof(Entry, generation)], code.rcx);
		}
		for (const auto& [offset, read]: reads) {
			const auto child = pointer_uses.find(read->value.ResolveInstruction());
			if (child == pointer_uses.end()) continue;
			self(self, offset, child->second);
		}
		code.pop(code.rdi);
	};
	for (const auto& [reg, reads]: roots) {
		visit(visit, reg * sizeof(uint32_t), reads);
	}
	code.mov(code.eax, 1);
	code.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
	code.L(failed);
	code.xor_(code.eax, code.eax);
	code.L(finished);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}
SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask, bool skip_tape)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_skip_tape(skip_tape), m_context(AcquireContext(program)) {
	// Shader memory reads can submit work and publish new runtime flags. Latch
	// evaluator choices before preparing any scratch, so a root cannot switch to
	// an unprepared CFG tape midway through this walker. The ordinary evaluator
	// inherits its clean partner's choices for the complete materialization.
	const auto flags = Common::RuntimeFlags::Current();
	m_cfg_tape_enabled = clean_evaluator != nullptr ? clean_evaluator->m_cfg_tape_enabled : flags.srt_cfg_tape;
	m_cfg_schedule_enabled = clean_evaluator != nullptr ? clean_evaluator->m_cfg_schedule_enabled : flags.srt_cfg_schedule;
	m_scalar_batch_enabled = clean_evaluator != nullptr ? clean_evaluator->m_scalar_batch_enabled : flags.srt_scalar_batch;
	m_cfg_gather_enabled = clean_evaluator != nullptr ? clean_evaluator->m_cfg_gather_enabled : flags.srt_cfg_gather;
	m_cfg_trace_enabled = clean_evaluator != nullptr ? clean_evaluator->m_cfg_trace_enabled : flags.srt_cfg_trace;
	m_source_reuse_enabled = clean_evaluator != nullptr ? clean_evaluator->m_source_reuse_enabled : flags.srt_source_reuse;
	m_compiled_enabled = clean_evaluator != nullptr ? clean_evaluator->m_compiled_enabled
	                                                : (flags.srt_compiled && runtime.allow_compiled);
	// Opt-in: record the memory access trace this walker performs so the workload can
	// be replayed offline. Wraps readers on the walker's own copy only, and calls
	// straight through, so results are unchanged when it is off or on.
	AttachFixtureRecording(m_runtime);
}

SrtWalker::~SrtWalker() {
	if (m_desc_reuse != 0u || m_desc_eval != 0u || m_memo_skipped != 0u)
		SrtCompiledCountDesc(m_desc_reuse, m_desc_eval, m_memo_skipped);
	--m_program.evaluation_depth;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 4;
	context.refreshed_flat = nullptr;
	return context;
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low bits distinguish in-progress and failed evaluations in this session.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		++m_cycle_count;
		return false;
	}
	if (m_context.values[index].generation == (m_context.generation | 2u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const auto cycles_before = m_cycle_count;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		// Cyclic dependencies may become evaluable after another node completes.
		// Only retain failures independent of an in-progress evaluation.
		memo.generation = cycles_before == m_cycle_count ? m_context.generation | 2u : 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	return false;
}

size_t SrtWalker::TryScalarReadBatch(std::span<const Value> values, bool strict,
                                    std::span<uint32_t> output) {
	// Existing diagnostic readers replace userdata. Until their optional-span
	// wrappers can capture/replay the same observations, retain scalar execution.
	if (!m_scalar_batch_enabled || FixtureRecord::Enabled() ||
	    SrtTapeVerifyEnabled() || !m_active_mask.IsEmpty() || values.size() < 2u) return 0;
	const auto peek = strict ? m_runtime.try_read_specialization_span : m_runtime.try_read_memory_span;
	if (peek == nullptr || (strict && m_clean_evaluator == nullptr)) return 0;
	auto& evaluator = strict ? *m_clean_evaluator : *this;
	if (!evaluator.m_active_mask.IsEmpty()) return 0;
	std::array<const Inst*, 16> reads {};
	std::array<uint32_t, 16> memo_indices {};
	const Inst* handle = nullptr;
	Value offset;
	uint32_t first_offset = 0, first_component = 0, pc = 0;
	size_t count = 0;
	for (; count < std::min({values.size(), output.size(), reads.size()}); ++count) {
		const auto* inst = values[count].ResolveInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    inst->NumArgs() != 4u || !IsRawRead(m_program, *inst)) break;
		const auto guard = inst->Arg(3).Resolve();
		if (!guard.IsImmediate() || guard.GetType() != Type::U1 || !guard.U1()) break;
		const auto flags = inst->Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) break;
		const auto& memory = m_program.memory_info[flags.index];
		if (memory.kind != ResourceKind::ScalarAddress) break;
		const auto* candidate_handle = inst->Arg(0).ResolveInstruction();
		if (candidate_handle == nullptr || candidate_handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    candidate_handle->NumArgs() != 2u) break;
		if (count == 0) {
			handle = candidate_handle; offset = inst->Arg(1).Resolve();
			first_offset = memory.offset; first_component = memory.component_index; pc = flags.pc;
		} else if (candidate_handle != handle || inst->Arg(1).Resolve() != offset || flags.pc != pc ||
		           uint64_t(memory.offset) != uint64_t(first_offset) + count * sizeof(uint32_t) ||
		           uint64_t(memory.component_index) != uint64_t(first_component) + count) break;
		const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
		if (index < evaluator.m_context.values.size()) {
			const auto generation = evaluator.m_context.values[index].generation;
			if ((generation & ~uint64_t{3}) == evaluator.m_context.generation) break;
		}
		reads[count] = inst; memo_indices[count] = index;
	}
	if (count < 2u) return 0;
	// Only evaluate the first word's shared dependencies, in its normal order.
	// Later words introduce no dependency, guard, or bounds evaluation at all.
	uint64_t low = 0, high = 0, byte_offset = 0;
	if (!evaluator.Arg(*handle, 0, low) || !evaluator.Arg(*handle, 1, high) ||
	    !evaluator.EvaluateWide(offset, byte_offset)) return 0;
	const auto base = (((high << 32u) | static_cast<uint32_t>(low)) & AddressMask) & ~uint64_t{3};
	const auto relative = (int64_t(static_cast<int32_t>(first_offset)) & ~int64_t{3}) +
	                      int64_t(static_cast<uint32_t>(byte_offset) & ~3u);
	uint64_t address = 0;
	if (!AddSignedAddress(base, relative, address) ||
	    count * sizeof(uint32_t) > UINT64_MAX - address) return 0;
	// A negative signed offset may cross the signed boundary between components.
	// Check exact scalar address equivalence before asking the capability to read.
	for (size_t i = 1; i < count; ++i) {
		const auto scalar_offset = m_program.memory_info[reads[i]->Flags<MemoryFlags>().index].offset;
		const auto scalar_relative = (int64_t(static_cast<int32_t>(scalar_offset)) & ~int64_t{3}) +
		                             int64_t(static_cast<uint32_t>(byte_offset) & ~3u);
		uint64_t scalar_address = 0;
		if (!AddSignedAddress(base, scalar_relative, scalar_address) || scalar_address != address + i * 4u) return 0;
	}
	if (!peek(m_runtime.userdata, address, output.first(count))) return 0;
	evaluator.m_context.values.resize(m_program.evaluation_value_count);
	for (size_t i = 0; i < count; ++i) {
		auto& memo = evaluator.m_context.values[memo_indices[i]];
		memo.value = output[i]; memo.generation = evaluator.m_context.generation;
	}
	if (m_runtime.evaluation_counts) m_runtime.evaluation_counts->scalar_batch_words += count;
	return count;
}

size_t SrtWalker::TryFlatScalarBatch(std::span<const uint32_t> slots, uint32_t sequential_slot,
                                    std::vector<uint32_t>& flat) {
	if (!m_scalar_batch_enabled) return 0;
	std::array<Value, 16> values {};
	std::array<uint32_t, 16> offsets {};
	std::array<uint32_t, 16> words {};
	size_t count = 0;
	bool strict = false;
	const size_t available = slots.empty() ? m_program.srt_reads.size() - sequential_slot : slots.size();
	for (; count < std::min(available, values.size()); ++count) {
		const auto slot = slots.empty() ? sequential_slot + static_cast<uint32_t>(count) : slots[count];
		if (slot >= m_program.srt_reads.size()) break;
		const auto& read = m_program.srt_reads[slot];
		if (read.flat_offset >= flat.size()) break;
		const bool clean = read.flat_offset < m_clean_flat_slots.size() && m_clean_flat_slots[read.flat_offset] != 0u;
		if (count == 0) strict = clean;
		else if (strict != clean) break;
		values[count] = read.value; offsets[count] = read.flat_offset;
	}
	count = TryScalarReadBatch(std::span(values).first(count), strict, words);
	for (size_t i = 0; i < count; ++i) flat[offsets[i]] = words[i];
	return count;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector) return false;
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	if (m_runtime.evaluation_counts) ++m_runtime.evaluation_counts->recursive_ops;
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			const auto index = reg - m_program.user_data_base;
			if (m_runtime.observe_user_data != nullptr) {
				m_runtime.observe_user_data(m_runtime.user_data_observer_userdata, index);
			}
			result = m_runtime.user_data[index];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (m_direct_flat != nullptr) {
				// A direct refresh already fetched this word (and captured the read).
				const auto offset = m_program.srt_reads[slot.U32()].flat_offset;
				if (offset < m_direct_flat->size() &&
				    (m_direct_filled == nullptr ||
				     (offset < m_direct_filled->size() && (*m_direct_filled)[offset] != 0u))) {
					result = (*m_direct_flat)[offset];
					return true;
				}
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (inst.GetOpcode() == ValueOpcode::LoadBufferU32 && m_clean_evaluator != nullptr &&
				    m_clean_evaluator->m_active_mask == m_active_mask) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
			if (binary()) {
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				// Saturating, as V_CVT_U32_F32 (NaN and negatives give 0).
				result = SaturateF32ToU32(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU32: return false;
		default: break;
	}
	return false;
}
// --- input-keyed descriptor reuse -----------------------------------------------------------------
// The recursive evaluation of a descriptor source after a compiled refresh is a function of the memo
// entries it hits. This walk finds those entries ("leaves") for one (evaluator mode, source): it stops at
// any node the served tape publishes in the evaluator's context and descends only through operations that
// are pure functions of their operands. Anything else (guest reads, user data, shader base, selects, phi)
// makes the source ineligible, so a reused result can never skip a read, an observer call or a decision.
namespace {

constexpr size_t kMaxReuseLeaves = 24;

// Diagnostic switch for A/B measurement: KYTY_SRT_NO_DESC_REUSE=1 restores the full per-call evaluation.
bool DescriptorReuseEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_NO_DESC_REUSE");
		return value == nullptr || value[0] != '1';
	}();
	return enabled;
}

bool IsPureReuseOpcode(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::BitFieldInsert: return true;
		default: return false;
	}
}

void BuildSourceReusePlan(const ResourcePlan& program, const ResourcePlan::SrtTape& tape, int mode,
                          std::span<const uint8_t> clean_flat_slots, uint32_t source,
                          ResourcePlan::SourceReusePlan& plan) {
	static std::atomic<uint64_t> serial {1};
	plan.leaves.clear();
	plan.serial   = serial.fetch_add(1, std::memory_order_relaxed);
	plan.tape     = &tape;
	plan.tape_ops = tape.ops.size();
	plan.state    = ResourcePlan::SourceReusePlan::State::Ineligible;
	if (mode < 0 || mode > 2 || source >= program.descriptor_sources.size()) return;
	const auto& descriptor = program.descriptor_sources[source];
	if (descriptor.indirect_descriptor.has_value()) return;
	// Stop set: which (instruction, role) pairs the served tape publishes into the evaluating
	// walker's context. Mode 0: everything (no clean partner). Mode 1: ordinary ops for the walker's own
	// context, clean ops for its partner's. Mode 2 (the clean partner itself): clean ops only.
	std::unordered_map<const Inst*, uint32_t> stops; // bit 0 = own context, bit 1 = clean partner context
	for (const auto& op: tape.ops) {
		if (op.source == nullptr || op.memo_index == UINT32_MAX) continue;
		uint32_t bit = 0;
		if (mode == 0) bit = 1u;
		else if (mode == 1) bit = op.clean != 0u ? 2u : 1u;
		else if (op.clean != 0u) bit = 1u;
		if (bit != 0u) stops[op.source] |= bit;
	}
	struct Item { Value value; uint8_t role; };
	std::vector<Item> stack;
	std::unordered_set<uint64_t> visited;
	for (uint32_t i = 0; i < descriptor.dword_count && i < descriptor.dwords.size(); ++i)
		stack.push_back({descriptor.dwords[i], 0u});
	while (!stack.empty()) {
		auto item = stack.back();
		stack.pop_back();
		const auto value = item.value.Resolve();
		if (value.IsEmpty()) return;
		if (value.IsImmediate()) continue;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return;
		if (!visited.insert((reinterpret_cast<uint64_t>(inst) << 1u) | item.role).second) continue;
		if (visited.size() > 256u) return;
		if (const auto found = stops.find(inst); found != stops.end() && (found->second & (1u << item.role)) != 0u) {
			if (plan.leaves.size() >= kMaxReuseLeaves) return;
			uint32_t memo_index = UINT32_MAX;
			for (const auto& op: tape.ops) {
				if (op.source == inst && op.memo_index != UINT32_MAX) { memo_index = op.memo_index; break; }
			}
			if (memo_index == UINT32_MAX) return;
			plan.leaves.push_back({inst, item.role, memo_index});
			continue;
		}
		const auto opcode = inst->GetOpcode();
		if (opcode == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 || slot.U32() >= program.srt_reads.size()) return;
			uint8_t role = item.role;
			if (mode == 1 && slot.U32() < clean_flat_slots.size() && clean_flat_slots[slot.U32()] != 0u) role = 1u;
			stack.push_back({program.srt_reads[slot.U32()].value, role});
			continue;
		}
		if (!IsPureReuseOpcode(opcode)) return;
		for (size_t i = 0; i < inst->NumArgs(); ++i) stack.push_back({inst->Arg(i), item.role});
	}
	plan.state = ResourcePlan::SourceReusePlan::State::Eligible;
}

} // namespace

bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	SRT_PROF_SCOPE(SrtProf::EvalDesc);
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	if (m_direct_flat != nullptr && m_program.direct_descriptor_offsets.size() >= (source + 1u) * 8u) {
		const int32_t* direct = m_program.direct_descriptor_offsets.data() + source * 8u;
		for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
			if (index < 8u && direct[index] >= 0 &&
			    static_cast<size_t>(direct[index]) < m_direct_flat->size() &&
			    (m_direct_filled == nullptr ||
			     (static_cast<size_t>(direct[index]) < m_direct_filled->size() &&
			      (*m_direct_filled)[static_cast<size_t>(direct[index])] != 0u))) {
				result.dwords[index] = (*m_direct_flat)[static_cast<size_t>(direct[index])];
				continue;
			}
			if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
				return false;
			}
		}
		return true;
	}
	if (source < m_program.descriptor_gathers.size()) {
		const auto& gather = m_program.descriptor_gathers[source];
		const auto* flat = m_context.refreshed_flat;
		bool ready = gather.valid && m_active_mask.IsEmpty() &&
		             (!gather.needs_flat || (flat != nullptr &&
	              (m_program.control_flow.empty() ||
	               (m_cfg_gather_enabled && source < m_cfg_gather_ready_sources.size() &&
	                m_cfg_gather_ready_sources[source] != 0u))));
		if (ready && gather.needs_flat) {
			for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
				const auto& word = gather.words[index];
				if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::Flat &&
				    word.value >= flat->size()) ready = false;
			}
		}
		if (ready) {
			bool cfg_gather = gather.needs_flat && !m_program.control_flow.empty();
			// A compiled Gather step has one fixed predecessor path in the trie, so whether the memo of every
			// Flat word's srt_read agrees with the flat buffer is decided by the step, not by the values:
			// the memo slot and the flat word are both written from the same op result. Verify once per
			// (program build, Gather step, evaluator role), then answer without re-reading the memo.
			uint32_t* verified = nullptr;
			bool      verified_skip = false;
			if (cfg_gather && m_served_cfg != nullptr && DescriptorReuseEnabled() && source < m_cfg_gather_ready_sources.size() &&
			    m_cfg_gather_ready_sources[source] >= 2u) {
				auto& table = m_served_cfg->verified_gather[m_served_mode == 2 ? 1 : 0];
				if (table.size() <= source) table.resize(m_program.descriptor_sources.size(), 0u);
				verified      = &table[source];
				verified_skip = *verified == m_cfg_gather_ready_sources[source];
			}
			if (cfg_gather && !verified_skip) {
				// The CFG gather is only a shortcut for recursive lookup of values that
				// this exact evaluator already computed successfully in this generation.
				for (uint32_t index = 0; index < descriptor.dword_count && cfg_gather; ++index) {
					const auto& word = gather.words[index];
					if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::UserData) {
						cfg_gather = false; break; // preserve the observer callback sequence
					}
					if (word.kind != ResourcePlan::DescriptorGather::Word::Kind::Flat) continue;
					const auto slot = word.read_slot;
					if (slot >= m_program.srt_reads.size() ||
					    m_program.srt_reads[slot].flat_offset != word.value) { cfg_gather = false; break; }
					const auto& read = m_program.srt_reads[slot];
					const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
					                   m_clean_flat_slots[read.flat_offset] != 0u;
					auto* evaluator = clean ? m_clean_evaluator : this;
					if (evaluator == nullptr || !evaluator->m_active_mask.IsEmpty() ||
					    slot >= m_program.srt_reads.size()) { cfg_gather = false; break; }
					auto value = read.value.Resolve();
					auto* inst = value.TryInstruction();
					if (inst == nullptr) { cfg_gather = false; break; }
					const auto memo_index = inst->EvaluationIndex(m_program.evaluation_value_count);
					if (memo_index >= evaluator->m_context.values.size() ||
					    evaluator->m_context.values[memo_index].generation != evaluator->m_context.generation ||
					    word.value >= flat->size() ||
					    static_cast<uint32_t>(evaluator->m_context.values[memo_index].value) != (*flat)[word.value])
						cfg_gather = false;
				}
				if (!cfg_gather) ready = false;
				else if (verified != nullptr) *verified = m_cfg_gather_ready_sources[source];
			}
			if (ready && cfg_gather && m_runtime.evaluation_counts != nullptr) {
				++m_runtime.evaluation_counts->cfg_gather_eligible;
				++m_runtime.evaluation_counts->cfg_gather_hits;
			}
			if (ready) {
				for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
					const auto& word = gather.words[index];
					if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::UserData) {
						if (word.value < m_program.user_data_base ||
						    word.value - m_program.user_data_base >= m_runtime.user_data.size()) return false;
						const auto user_data_index = word.value - m_program.user_data_base;
						if (m_runtime.observe_user_data != nullptr) {
							m_runtime.observe_user_data(m_runtime.user_data_observer_userdata, user_data_index);
						}
						result.dwords[index] = m_runtime.user_data[user_data_index];
					} else if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::Flat) {
						result.dwords[index] = (*flat)[word.value];
					} else {
						result.dwords[index] = word.value;
					}
				}
				if (m_source_reuse_enabled && FrameBreakdown::Enabled()) {
					// Measurement only: the gather output is a copy of its Literal/UserData/Flat
					// inputs, so "reuse" cannot beat this copy. Count how often it equals the last one.
					auto& last  = m_program.source_reuse_last;
					auto& valid = m_program.source_reuse_valid;
					if (last.size() != m_program.descriptor_sources.size()) {
						last.assign(m_program.descriptor_sources.size(), {});
						valid.assign(m_program.descriptor_sources.size(), 0u);
					}
					auto& counters = FrameBreakdown::g_counters;
					if (valid[source] != 0u && last[source] == result) {
						counters.mat_reuse_hits.fetch_add(1, std::memory_order_relaxed);
					} else {
						counters.mat_reuse_misses.fetch_add(1, std::memory_order_relaxed);
						last[source]  = result;
						valid[source] = 1u;
					}
				}
				SRT_PROF_COUNT(eval_gather, 1);
				if (verified_skip) ++m_desc_reuse;
				else ++m_desc_eval;
				return true;
			}
		}
	}
	SRT_PROF_COUNT(eval_recursive, 1);
	if (m_source_reuse_enabled && FrameBreakdown::Enabled()) {
		FrameBreakdown::g_counters.mat_reuse_fallback.fetch_add(1, std::memory_order_relaxed);
	}
	// Input-keyed reuse. After a compiled refresh, a recursive evaluation of a source whose value graph is
	// pure arithmetic over published memo values is a function of exactly those values. If they are all
	// published and bit-identical to the ones behind the stored result, the result is that stored result.
	// Anything impure (guest reads, user data, shader base, lazy ops) makes the source ineligible.
	ResourceDelta::SourceReuse* entry = nullptr;
	std::array<uint64_t, kMaxReuseLeaves> inputs;
	size_t input_count = 0;
	if (m_served_mode >= 0 && m_served_tape != nullptr && m_runtime.delta != nullptr && DescriptorReuseEnabled() &&
	    m_active_mask.IsEmpty() && m_program.control_flow.empty() == (m_served_cfg == nullptr)) {
		auto& plans = m_program.source_reuse_plans[static_cast<size_t>(m_served_mode)];
		if (plans.size() != m_program.descriptor_sources.size()) plans.assign(m_program.descriptor_sources.size(), {});
		auto& plan = plans[source];
		if (plan.state == ResourcePlan::SourceReusePlan::State::Unbuilt || plan.tape != m_served_tape ||
		    plan.tape_ops != m_served_tape->ops.size())
			BuildSourceReusePlan(m_program, *m_served_tape, m_served_mode, m_clean_flat_slots, source, plan);
		if (plan.state == ResourcePlan::SourceReusePlan::State::Eligible) {
			bool all = true;
			for (const auto& leaf: plan.leaves) {
				const SrtWalker* evaluator = leaf.role != 0u ? m_clean_evaluator : this;
				if (evaluator == nullptr) { all = false; break; }
				const auto& context = evaluator->m_context;
				if (leaf.memo_index >= context.values.size() ||
				    context.values[leaf.memo_index].generation != context.generation) { all = false; break; }
				inputs[input_count++] = context.values[leaf.memo_index].value;
			}
			if (all) {
				auto& sources = m_runtime.delta->sources;
				if (sources.size() != m_program.descriptor_sources.size()) sources.assign(m_program.descriptor_sources.size(), {});
				entry = &sources[source];
				if (entry->valid && entry->mode == static_cast<uint8_t>(m_served_mode) &&
				    entry->plan_serial == plan.serial && entry->inputs.size() == input_count &&
				    std::equal(inputs.begin(), inputs.begin() + static_cast<ptrdiff_t>(input_count), entry->inputs.begin())) {
					result = entry->output;
					++m_desc_reuse;
					SRT_PROF_COUNT(eval_reused, 1);
					return true;
				}
				entry->plan_serial = plan.serial;
			}
		}
	}
	++m_desc_eval;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			if (entry != nullptr) entry->valid = false;
			return false;
		}
	}
	if (entry != nullptr) {
		entry->valid  = true;
		entry->mode   = static_cast<uint8_t>(m_served_mode);
		entry->inputs.assign(inputs.begin(), inputs.begin() + static_cast<ptrdiff_t>(input_count));
		entry->output = result;
	}
	return true;
}

bool SrtWalker::VerifyCfgRefresh(std::vector<uint32_t>& flat) {
	g_tape_stats.cfg_checked.fetch_add(1, std::memory_order_relaxed);
	const auto disable = [&](const char* reason, bool fallback) {
		(fallback ? g_tape_stats.cfg_mismatch : g_tape_stats.cfg_inconclusive).fetch_add(1, std::memory_order_relaxed);
		for (auto& tape: m_program.cfg_srt_tapes) tape.state = ResourcePlan::SrtTape::State::Unsupported;
		std::fprintf(stderr,
		             "SRT CFG verify disabled shader=0x%016llx stage=%u reason=%s recursive_fallback=%u\n",
		             static_cast<unsigned long long>(m_program.shader_hash),
		             static_cast<unsigned>(m_program.stage), reason, fallback ? 1u : 0u);
		std::fflush(stderr);
	};
	// Op.clean names the strict domain of this runtime. A caller-supplied clean
	// evaluator with different inputs/readers cannot share that capture domain.
	// Reject it rather than treating one callback's values as another's evidence.
	if (m_clean_evaluator != nullptr) {
		const auto expected = CleanRuntime(m_runtime);
		const auto& actual = m_clean_evaluator->m_runtime;
		if (actual.userdata != expected.userdata || actual.read_memory != expected.read_memory ||
		    actual.read_specialization_memory != expected.read_specialization_memory ||
		    actual.observe_user_data != expected.observe_user_data ||
		    actual.user_data_observer_userdata != expected.user_data_observer_userdata ||
		    actual.shader_base != expected.shader_base ||
		    actual.user_data.data() != expected.user_data.data() || actual.user_data.size() != expected.user_data.size() ||
		    actual.workgroup_counts.data() != expected.workgroup_counts.data() ||
		    actual.workgroup_counts.size() != expected.workgroup_counts.size()) {
			disable("distinct_clean_runtime", false);
			return false;
		}
	}
	ResourcePlan::EvaluationContext initial;
	initial.values = m_context.values;
	initial.generation = m_context.generation;
	initial.refreshed_flat = m_context.refreshed_flat;
	ResourcePlan::EvaluationContext initial_clean;
	if (m_clean_evaluator != nullptr) {
		initial_clean.values = m_clean_evaluator->m_context.values;
		initial_clean.generation = m_clean_evaluator->m_context.generation;
		initial_clean.refreshed_flat = m_clean_evaluator->m_context.refreshed_flat;
	}
	const auto original_flat = flat;
	const auto original_active = m_program.active_sources;
	const auto original_cycles = m_cycle_count;
	const auto original_clean_cycles = m_clean_evaluator != nullptr ? m_clean_evaluator->m_cycle_count : 0;
	const auto restore_inputs = [&] {
		m_context.values = initial.values;
		m_context.refreshed_flat = initial.refreshed_flat;
		m_cycle_count = original_cycles;
		if (m_clean_evaluator != nullptr) {
			m_clean_evaluator->m_context.values = initial_clean.values;
			m_clean_evaluator->m_context.refreshed_flat = initial_clean.refreshed_flat;
			m_clean_evaluator->m_cycle_count = original_clean_cycles;
		}
	};
	TapeReadCapture capture {.source = m_runtime};
	auto observed = m_runtime;
	observed.userdata = &capture;
	observed.read_memory = CaptureTapeRead<false>;
	observed.read_specialization_memory = m_runtime.read_specialization_memory != nullptr
	                                         ? CaptureTapeRead<true> : nullptr;
	const auto original_runtime = m_runtime;
	SrtRuntime original_clean_runtime;
	if (m_clean_evaluator != nullptr) {
		original_clean_runtime = m_clean_evaluator->m_runtime;
		capture.clean_source = original_clean_runtime;
		auto observed_clean = original_clean_runtime;
		observed_clean.userdata = &capture;
		observed_clean.read_memory = CaptureCleanTapeRead<false>;
		observed_clean.read_specialization_memory = original_clean_runtime.read_specialization_memory != nullptr
		                                              ? CaptureCleanTapeRead<true> : nullptr;
		m_clean_evaluator->m_runtime = observed_clean;
	}
	m_runtime = observed;
	m_cfg_verifying = true;
	const bool tape_ok = RefreshFlatBuffer(flat);
	m_cfg_verifying = false;
	m_runtime = original_runtime;
	if (m_clean_evaluator != nullptr) m_clean_evaluator->m_runtime = original_clean_runtime;
	const auto tape_active = m_program.active_sources;
	// Observed readers are always callbacks, even when the original reader is
	// absent, so capture can never fall through to direct-pointer memory. Pure
	// arithmetic CFGs still verify without a reader; attempted opaque reads fail.
	if (original_runtime.read_memory == nullptr &&
	    std::ranges::any_of(capture.reads, [](const auto& read) { return !read.strict; })) {
		restore_inputs();
		flat = original_flat;
		m_program.active_sources = original_active;
		disable("missing_capture_reader", false);
		return false;
	}
	if (!HasStableTapeReads(capture.reads)) {
		restore_inputs();
		flat = original_flat;
		m_program.active_sources = original_active;
		disable("unstable_capture", false);
		return false;
	}

	TapeReadReplay replay {.reads = capture.reads};
	auto replay_runtime = original_runtime;
	replay_runtime.userdata = &replay;
	// Even an absent reader is represented by a rejecting replay callback: there
	// must never be a direct-pointer fallback inside this verifier.
	replay_runtime.read_memory = ReplayTapeRead<false>;
	replay_runtime.read_specialization_memory = original_runtime.read_specialization_memory != nullptr
	                                            ? ReplayTapeRead<true> : nullptr;
	SrtWalker replay_clean(m_program, CleanRuntime(replay_runtime), {}, nullptr,
	                       m_clean_evaluator != nullptr ? m_clean_evaluator->m_active_mask : Value{}, true);
	SrtWalker replay_walker(m_program, replay_runtime, m_clean_flat_slots,
	                        m_clean_evaluator != nullptr ? &replay_clean : nullptr, m_active_mask, true);
	// Values evaluated before RefreshFlatBuffer are inputs, not evidence newly
	// produced by the tape. Seed their exact domains into fresh replay sessions.
	TransferCurrentEvaluationValues(initial, replay_walker.m_context);
	if (m_clean_evaluator != nullptr) TransferCurrentEvaluationValues(initial_clean, replay_clean.m_context);
	const auto seed_in_progress = [](const auto& source, auto& destination) {
		if (destination.values.size() < source.values.size()) destination.values.resize(source.values.size());
		for (size_t index = 0; index < source.values.size(); ++index) {
			if (source.values[index].generation == (source.generation | 1u)) {
				destination.values[index] = {.value = source.values[index].value,
				                             .generation = destination.generation | 1u};
			}
		}
	};
	seed_in_progress(initial, replay_walker.m_context);
	if (m_clean_evaluator != nullptr) seed_in_progress(initial_clean, replay_clean.m_context);
	std::vector<uint32_t> recursive_flat;
	const bool recursive_ok = replay_walker.RefreshFlatBuffer(recursive_flat);
	const auto recursive_active = m_program.active_sources;
	bool mismatch = tape_ok != recursive_ok || flat != recursive_flat || tape_active != recursive_active;
	const auto& tape = m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0];
	if (!replay.inconclusive && tape_ok && recursive_ok) {
		for (size_t index = 0; index < tape.ops.size(); ++index) {
			const auto& op = tape.ops[index];
			if (op.source == nullptr || op.memo_index == UINT32_MAX) continue;
			const auto& context = op.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
			if (op.memo_index >= context.values.size() ||
			    context.values[op.memo_index].generation != context.generation) continue;
			auto& evaluator = op.clean && m_clean_evaluator != nullptr ? replay_clean : replay_walker;
			uint64_t recursive_value = 0;
			const bool evaluated = evaluator.EvaluateWide(Value(const_cast<Inst*>(op.source)), recursive_value);
			if (!evaluated || recursive_value != context.values[op.memo_index].value) {
				mismatch = true;
				std::fprintf(stderr,
				             "SRT CFG verify op shader=0x%016llx index=%zu opcode=%u clean=%u tape=0x%016llx recursive=%s0x%016llx\n",
				             static_cast<unsigned long long>(m_program.shader_hash), index,
				             static_cast<unsigned>(op.opcode), op.clean,
				             static_cast<unsigned long long>(context.values[op.memo_index].value),
				             evaluated ? "" : "failed/", static_cast<unsigned long long>(recursive_value));
				break;
			}
		}
	}
	if (replay.inconclusive) {
		restore_inputs();
		flat = original_flat;
		m_program.active_sources = original_active;
		disable("unmatched_replay", false);
		return false;
	}
	if (mismatch) {
		restore_inputs();
		TransferCurrentEvaluationValues(replay_walker.m_context, m_context);
		if (m_clean_evaluator != nullptr) TransferCurrentEvaluationValues(replay_clean.m_context, m_clean_evaluator->m_context);
		flat = std::move(recursive_flat);
		m_context.refreshed_flat = nullptr;
		if (m_clean_evaluator != nullptr) m_clean_evaluator->m_context.refreshed_flat = nullptr;
		m_program.active_sources = recursive_active;
		disable("value_or_activation_mismatch", true);
		return recursive_ok;
	}
	m_program.active_sources = tape_active;
	g_tape_stats.cfg_matched.fetch_add(1, std::memory_order_relaxed);
	return tape_ok;
}

namespace {

std::atomic<int> g_direct_srt_override {-1};

bool DirectSrtReadsEnabled() {
	const int forced = g_direct_srt_override.load(std::memory_order_relaxed);
	return forced >= 0 ? forced != 0 : Common::Experiments::Enabled(Common::Experiments::DirectSrtReads);
}

// Decides once per plan whether every SRT read is LoadAddressU32(GetAddressResource(
// GetUserData(lo), GetUserData(hi)), imm, ...) on a scalar address, and records its operands.
// KYTY_DBG_DIRECT_SRT=1: why plans fall back to the interpreter and how often they refresh.
bool DirectSrtDebug() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_DIRECT_SRT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

std::string DirectShape(Value value, int depth) {
	value = value.Resolve();
	if (value.IsImmediate()) return "imm";
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return "?";
	std::string result(ValueOpcodeName(inst->GetOpcode()));
	if (inst->GetOpcode() == ValueOpcode::ReadConst || inst->GetOpcode() == ValueOpcode::GetUserData)
		return result;
	if (depth >= 3) return result + "(...)";
	result += "(";
	for (size_t i = 0; i < inst->NumArgs(); ++i) {
		if (i != 0) result += ",";
		result += DirectShape(inst->Arg(i), depth + 1);
	}
	return result + ")";
}

struct DirectDebugState {
	std::mutex                                      lock;
	std::unordered_map<const ResourcePlan*, std::string> reasons;
	std::unordered_map<const ResourcePlan*, uint64_t>    refreshes;
	uint64_t                                        counts[3] {};
};

DirectDebugState& DirectDebug() {
	static DirectDebugState state;
	return state;
}

bool RejectDirect(const ResourcePlan& program, const char* reason, Value value) {
	if (DirectSrtDebug()) {
		auto&            state = DirectDebug();
		std::scoped_lock lock {state.lock};
		state.reasons[&program] = std::string(reason) + " reads=" + std::to_string(program.srt_reads.size()) +
		                          " cf=" + std::to_string(program.control_flow.size()) + " " +
		                          DirectShape(value, 0).substr(0, 200);
	}
	return false;
}

// kind: 0 direct without control flow, 1 direct with control flow, 2 interpreter.
void CountDirectRefresh(const ResourcePlan& program, int kind) {
	auto&            state = DirectDebug();
	std::scoped_lock lock {state.lock};
	++state.counts[kind];
	if (kind == 2) ++state.refreshes[&program];
	const auto total = state.counts[0] + state.counts[1] + state.counts[2];
	if (total % 50000u != 0u) return;
	std::printf("NHL27SRT: refreshes direct=%llu direct_cf=%llu interpreted=%llu\n",
	            static_cast<unsigned long long>(state.counts[0]),
	            static_cast<unsigned long long>(state.counts[1]),
	            static_cast<unsigned long long>(state.counts[2]));
	std::vector<std::pair<uint64_t, const ResourcePlan*>> hot;
	for (const auto& [plan, count]: state.refreshes) hot.emplace_back(count, plan);
	std::sort(hot.rbegin(), hot.rend());
	for (size_t i = 0; i < hot.size() && i < 10; ++i) {
		const auto reason = state.reasons.find(hot[i].second);
		std::printf("NHL27SRT:   %llu %s\n", static_cast<unsigned long long>(hot[i].first),
		            reason != state.reasons.end() ? reason->second.c_str() : "(no reason)");
	}
	std::fflush(stdout);
}

bool BuildDirectSrtReads(const ResourcePlan& program, std::span<const uint8_t> clean_flat_slots) {
	auto& direct = program.direct_srt_reads;
	direct.clear();
	direct.reserve(program.srt_reads.size());
	program.direct_cf_batch = -1;
	const auto user_data_index = [&](Value value, uint32_t& index) {
		value = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetUserData) return false;
		const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
		if (reg < program.user_data_base) return false;
		index = reg - program.user_data_base;
		return true;
	};
	for (const auto& read: program.srt_reads) {
		const auto value = read.value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32) {
			return RejectDirect(program, "not-load", value);
		}
		if (!IsRawRead(program, *inst)) return RejectDirect(program, "not-raw", value);
		if (read.flat_offset >= program.srt_reads.size()) return RejectDirect(program, "flat", value);
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() < 2) {
			return RejectDirect(program, "handle", value);
		}
		ResourcePlan::DirectSrtRead entry;
		if (!user_data_index(handle->Arg(0), entry.low_register) ||
		    !user_data_index(handle->Arg(1), entry.high_register)) {
			return RejectDirect(program, "base", value);
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			return RejectDirect(program, "offset", value);
		}
		const auto& mem       = program.memory_info[inst->Flags<MemoryFlags>().index];
		const auto  immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		entry.relative = (immediate & ~int64_t {3}) + static_cast<int64_t>(offset.U32() & ~3u);
		entry.evaluation_index = inst->EvaluationIndex(program.evaluation_value_count);
		entry.clean = read.flat_offset < clean_flat_slots.size() && clean_flat_slots[read.flat_offset] != 0u;
		direct.push_back(entry);
	}
	auto& offsets = program.direct_descriptor_offsets;
	offsets.assign(program.descriptor_sources.size() * 8u, -1);
	for (size_t source = 0; source < program.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source];
		for (uint32_t i = 0; i < descriptor.dword_count && i < 8u; ++i) {
			const auto* inst = descriptor.dwords[i].Resolve().TryInstruction();
			if (inst == nullptr || inst->GetOpcode() != ValueOpcode::ReadConst) continue;
			const auto slot = inst->Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= program.srt_reads.size())
				continue;
			offsets[source * 8u + i] = static_cast<int32_t>(program.srt_reads[slot.U32()].flat_offset);
		}
	}
	// Group contiguous dwords read through the same pointer into runs (one reader call each).
	for (size_t head = 0; head < direct.size();) {
		size_t next = head + 1;
		while (next < direct.size() && next - head < 256 &&
		       direct[next].clean == direct[head].clean &&
		       direct[next].low_register == direct[head].low_register &&
		       direct[next].high_register == direct[head].high_register &&
		       direct[next].relative == direct[next - 1].relative + 4 &&
		       program.srt_reads[next].flat_offset == program.srt_reads[next - 1].flat_offset + 1) {
			direct[next].run = 0;
			++next;
		}
		direct[head].run = static_cast<uint32_t>(next - head);
		head = next;
	}
	return true;
}

} // namespace

void SetDirectSrtReadsOverride(int value) {
	g_direct_srt_override.store(value, std::memory_order_relaxed);
}

bool SrtWalker::DirectReadRun(uint32_t slot, uint32_t count, std::vector<uint32_t>& flat) {
	const auto& direct = m_program.direct_srt_reads;
	if (count == 0 || slot >= direct.size() || count > direct.size() - slot) return false;
	const auto& head = direct[slot];
	if (head.clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
		return false;
	auto&       evaluator = head.clean ? *m_clean_evaluator : *this;
	const auto& user_data = evaluator.m_runtime.user_data;
	if (head.low_register >= user_data.size() || head.high_register >= user_data.size()) return false;
	// The pointer registers are inputs like any GetUserData (a resource memo collecting its register
	// dependencies otherwise reuses resources read through another pointer).
	if (const auto& observer = m_runtime.observe_user_data != nullptr ? m_runtime : evaluator.m_runtime;
	    observer.observe_user_data != nullptr) {
		observer.observe_user_data(observer.user_data_observer_userdata, head.low_register);
		observer.observe_user_data(observer.user_data_observer_userdata, head.high_register);
	}
	const auto base = ((static_cast<uint64_t>(user_data[head.high_register]) << 32u) |
	                   user_data[head.low_register]) &
	                  AddressMask;
	uint64_t   address = 0;
	const auto bytes   = static_cast<uint64_t>(count) * 4u;
	if (!AddSignedAddress(base & ~uint64_t {3}, head.relative, address) || address > AddressMask ||
	    bytes - 1 > AddressMask - address) {
		return false;
	}
	const auto first = m_program.srt_reads[slot].flat_offset;
	if (first > flat.size() || count > flat.size() - first) return false;
	const auto words = std::span(flat).subspan(first, count);
	// Descriptor evaluation may already have read these words through the memo
	// (a descriptor evaluated before the flat refresh shares its reads with it).
	auto& values = evaluator.m_context.values;
	if (values.size() < m_program.evaluation_value_count) values.resize(m_program.evaluation_value_count);
	bool memo_hit = true;
	for (uint32_t i = 0; i < count; ++i) {
		const auto index = direct[slot + i].evaluation_index;
		if (index >= values.size() || values[index].generation != evaluator.m_context.generation) {
			memo_hit = false;
			break;
		}
	}
	if (memo_hit) {
		for (uint32_t i = 0; i < count; ++i) {
			words[i] = static_cast<uint32_t>(values[direct[slot + i].evaluation_index].value);
		}
	} else if (evaluator.m_runtime.read_memory != nullptr) {
		if (!evaluator.m_runtime.read_memory(evaluator.m_runtime.userdata, address, words)) return false;
	} else if (!Common::Experiments::Enabled(Common::Experiments::FaultPageDirect) ||
	           !TryFaultPageRead(address, words.data(), words.size_bytes())) {
		// Experiment FaultPageDirect: clean bytes of a page that faulted before (the GPU wrote
		// others) come from the backing; the plain read would fault and wait for the GPU.
		std::memcpy(words.data(), reinterpret_cast<const void*>(address), words.size_bytes());
	}
	// Descriptor and condition expressions refer back to these reads through the memo.
	for (uint32_t i = 0; i < count; ++i) {
		values[direct[slot + i].evaluation_index] = {words[i], evaluator.m_context.generation};
	}
	return true;
}

void SrtWalker::ShareDirectFlat(const std::vector<uint32_t>& flat, const std::vector<uint8_t>* filled) {
	// Experiment CleanFromDirect: branch conditions, uniform fills and written descriptors are
	// evaluated strictly; their resource-table words need not be read a second time.
	if (m_clean_evaluator == nullptr || !Common::Experiments::Enabled(Common::Experiments::CleanFromDirect)) {
		return;
	}
	m_clean_evaluator->m_direct_flat   = &flat;
	m_clean_evaluator->m_direct_filled = filled;
}

bool SrtWalker::RefreshDirectFlat(std::vector<uint32_t>& flat) {
	const auto& direct = m_program.direct_srt_reads;
	flat.resize(m_program.srt_reads.size());
	for (size_t slot = 0; slot < direct.size();) {
		// Without batching every slot is its own read, as in the scalar walk.
		const uint32_t count = m_runtime.batch_reads ? direct[slot].run : 1u;
		if (!DirectReadRun(static_cast<uint32_t>(slot), count, flat)) return false;
		slot += count;
	}
	m_direct_flat   = &flat;
	m_direct_filled = nullptr;
	ShareDirectFlat(flat, nullptr);
	return true;
}

namespace {

// The inputs of a branch condition, when they are only user data, flat (resource-table) words
// and the shader base: everything else on the way is a pure operation of its arguments.
bool CollectConditionLeaves(const ResourcePlan& program, Value root,
                            std::vector<ResourcePlan::DirectCfCondition::Leaf>& leaves) {
	using Leaf = ResourcePlan::DirectCfCondition::Leaf;
	using Kind = ResourcePlan::DirectCfCondition::Kind;
	std::vector<const Inst*> stack;
	std::vector<const Inst*> visited;
	const auto push = [&](Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) return true;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (std::find(visited.begin(), visited.end(), inst) != visited.end()) return true;
		visited.push_back(inst);
		stack.push_back(inst);
		return visited.size() <= 256;
	};
	const auto add = [&](Leaf leaf) {
		for (const auto& known: leaves) {
			if (known.kind == leaf.kind && known.index == leaf.index) return true;
		}
		leaves.push_back(leaf);
		return leaves.size() <= ResourcePlan::DirectCfCondition::MaxLeaves;
	};
	if (!push(root)) return false;
	while (!stack.empty()) {
		const auto* inst = stack.back();
		stack.pop_back();
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
				if (reg < program.user_data_base || !add({Kind::UserData, reg - program.user_data_base})) return false;
				break;
			}
			case ValueOpcode::GetShaderBase:
				if (!add({Kind::ShaderBase, 0})) return false;
				break;
			case ValueOpcode::ReadConst: {
				const auto slot = inst->Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 || slot.U32() >= program.srt_reads.size() ||
				    !add({Kind::Flat, program.srt_reads[slot.U32()].flat_offset})) {
					return false;
				}
				break;
			}
			case ValueOpcode::LoadAddressU32: {
				// Only the plan's own direct reads, whose words the refresh has already fetched.
				bool found = false;
				for (size_t slot = 0; slot < program.srt_reads.size(); ++slot) {
					if (program.srt_reads[slot].value.Resolve().TryInstruction() == inst) {
						if (!add({Kind::Flat, program.srt_reads[slot].flat_offset})) return false;
						found = true;
						break;
					}
				}
				if (!found) return false;
				break;
			}
			case ValueOpcode::Phi:
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadConstBuffer:
			case ValueOpcode::LoadBufferU32:
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU32: return false;
			default:
				for (size_t i = 0; i < inst->NumArgs(); ++i) {
					if (!push(inst->Arg(i))) return false;
				}
				break;
		}
	}
	return true;
}

} // namespace

void SrtWalker::BuildDirectCfRuns(bool batch) {
	const auto& reads  = m_program.direct_srt_reads;
	const auto& blocks = m_program.control_flow;
	auto&       runs   = m_program.direct_cf_runs;
	auto&       begin  = m_program.direct_cf_run_begin;
	runs.clear();
	begin.clear();
	begin.reserve(blocks.size() + 1);
	m_program.direct_cf_batch = batch ? 1 : 0;
	m_program.direct_cf_valid = false;
	for (const auto& block: blocks) {
		begin.push_back(static_cast<uint32_t>(runs.size()));
		for (size_t position = 0; position < block.srt_reads.size();) {
			// Extend the run while this block's next slot continues it.
			const auto slot = block.srt_reads[position];
			if (slot >= reads.size()) return;
			uint32_t count = 1;
			while (batch && position + count < block.srt_reads.size() &&
			       block.srt_reads[position + count] == slot + count && slot + count < reads.size() &&
			       reads[slot + count].run == 0u && reads[slot + count].clean == reads[slot].clean) {
				++count;
			}
			runs.push_back({slot, count});
			position += count;
		}
		for (const auto successor: block.successors) {
			if (successor >= blocks.size()) return;
		}
		if (!block.condition.IsEmpty() && block.successors.size() < 2) return;
	}
	begin.push_back(static_cast<uint32_t>(runs.size()));
	auto& conditions = m_program.direct_cf_conditions;
	conditions.assign(blocks.size(), {});
	for (size_t index = 0; index < blocks.size(); ++index) {
		if (blocks[index].condition.IsEmpty()) continue;
		auto& condition     = conditions[index];
		condition.cacheable = CollectConditionLeaves(m_program, blocks[index].condition, condition.leaves);
		if (!condition.cacheable) condition.leaves.clear();
		condition.last_values.resize(condition.leaves.size());
	}
	auto& active = m_program.direct_cf_active;
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: blocks) {
		for (const auto source: block.sources) {
			if (source >= active.size()) return;
			active[source] = 0u;
		}
	}
	m_program.direct_cf_valid = !blocks.empty();
}

// Direct plans with control flow: the blocks reached from the entry read their slots directly;
// branch conditions are evaluated as before (strictly, through the clean evaluator).
bool SrtWalker::RefreshDirectControlFlow(std::vector<uint32_t>& flat) {
	const bool batch = m_runtime.batch_reads;
	if (m_program.direct_cf_batch != (batch ? 1 : 0)) {
		BuildDirectCfRuns(batch);
	}
	if (!m_program.direct_cf_valid) return false;
	const auto& blocks    = m_program.control_flow;
	const auto& runs      = m_program.direct_cf_runs;
	const auto& run_begin = m_program.direct_cf_run_begin;
	const auto& srt_reads = m_program.srt_reads;
	auto&       filled    = m_program.direct_filled;
	flat.assign(srt_reads.size(), 0u);
	filled.assign(srt_reads.size(), 0u);
	// Conditions below see the words of the blocks visited so far.
	ShareDirectFlat(flat, &filled);
	m_program.active_sources = m_program.direct_cf_active;
	auto& active  = m_program.active_sources;
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(blocks.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited[index] != 0u) continue;
		visited[index] = 1u;
		const auto& block = blocks[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (uint32_t r = run_begin[index]; r < run_begin[index + 1]; ++r) {
			const auto run = runs[r];
			if (!DirectReadRun(run.slot, run.count, flat)) return false;
			for (uint32_t i = 0; i < run.count; ++i) {
				const auto offset = srt_reads[run.slot + i].flat_offset;
				if (offset < filled.size()) filled[offset] = 1u;
			}
		}
		uint32_t condition = 0;
		bool     decided   = false;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr) {
			// A condition whose inputs all equal the last ones has the last result.
			auto&    cache = m_program.direct_cf_conditions[index];
			uint64_t now[ResourcePlan::DirectCfCondition::MaxLeaves];
			bool     ready = cache.cacheable;
			for (size_t i = 0; ready && i < cache.leaves.size(); ++i) {
				const auto& leaf = cache.leaves[i];
				using Kind       = ResourcePlan::DirectCfCondition::Kind;
				if (leaf.kind == Kind::UserData) {
					ready  = leaf.index < m_runtime.user_data.size();
					now[i] = ready ? m_runtime.user_data[leaf.index] : 0;
					if (ready && m_runtime.observe_user_data != nullptr) {
						m_runtime.observe_user_data(m_runtime.user_data_observer_userdata, leaf.index);
					}
				} else if (leaf.kind == Kind::Flat) {
					ready  = leaf.index < filled.size() && filled[leaf.index] != 0u;
					now[i] = ready ? flat[leaf.index] : 0;
				} else {
					now[i] = m_runtime.shader_base;
				}
			}
			if (ready && cache.has_last &&
			    std::equal(cache.last_values.begin(), cache.last_values.end(), now)) {
				condition = cache.last_result;
				decided   = true;
			} else if ([&] {
				           const bool nonblocking =
				               Common::Experiments::Enabled(Common::Experiments::GpuConditionsBothPaths);
				           if (nonblocking) SetConditionReadsNonBlocking(true);
				           const bool ok = predicate.Evaluate(block.condition, condition);
				           if (nonblocking) SetConditionReadsNonBlocking(false);
				           return ok;
			           }()) {
				decided = true;
				if (ready) {
					std::copy(now, now + cache.leaves.size(), cache.last_values.begin());
					cache.last_result = condition;
					cache.has_last    = true;
				}
			}
		}
		if (decided) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			for (const auto successor: block.successors) pending.push_back(successor);
		}
	}
	m_direct_flat   = &flat;
	m_direct_filled = &filled;
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	SRT_PROF_SCOPE(SrtProf::Refresh);
	// KYTY_PERF_STATS: one clock pair per call, split by how the refresh was served.
	struct FlatCallTimer {
		bool                         active = FrameBreakdown::Enabled();
		FrameBreakdown::Clock::time_point start {};
		int                          kind = 0;  // 1 = trace hit, 2 = non-CFG tape/walker
		FlatCallTimer() { if (active) start = FrameBreakdown::Clock::now(); }
		~FlatCallTimer() {
			if (!active || kind == 0) return;
			auto& fc = FrameBreakdown::g_counters;
			const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			    FrameBreakdown::Clock::now() - start).count());
			(kind == 1 ? fc.mat_flat_hit_ns : fc.mat_flat_nocfg_ns).fetch_add(ns, std::memory_order_relaxed);
		}
	} call_timer;
	// Upstream's x64 walker (CompileSrtWalker): plans without control flow or captured reads,
	// read straight from guest memory. It fills the flat words and seeds the evaluation memo.
	// KYTY_SRT_JIT=1 (env only; was experiment bit 4).
	static const bool srt_jit = [] {
		const char* value = std::getenv("KYTY_SRT_JIT");
		return value != nullptr && value[0] == '1';
	}();
	if (m_program.srt_code && m_runtime.read_memory == nullptr && srt_jit) {
		flat.resize(m_program.srt_reads.size());
		m_program.active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		const bool ok = m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		    m_context.values.data(), m_context.generation, m_runtime.user_data.size());
		m_context.refreshed_flat = ok ? &flat : nullptr;
		return ok;
	}
	// Direct plans never mark gathered sources; every reader bounds-checks this vector, so an
	// empty one means "none ready" without allocating it for each fresh walker.
	const bool direct_plan = DirectSrtReadsEnabled() && m_active_mask.IsEmpty() &&
	                         m_program.direct_srt_state == ResourcePlan::DirectSrtState::Direct;
	if (m_cfg_gather_enabled && !direct_plan) {
		m_cfg_gather_ready_sources.assign(m_program.descriptor_sources.size(), 0u);
		if (m_clean_evaluator != nullptr)
			m_clean_evaluator->m_cfg_gather_ready_sources.assign(m_program.descriptor_sources.size(), 0u);
	}
	if (!m_program.control_flow.empty() && !m_skip_tape && !m_cfg_verifying &&
	    m_cfg_tape_enabled && SrtTapeVerifyEnabled() &&
	    m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0].state !=
	        ResourcePlan::SrtTape::State::Unsupported) {
		return VerifyCfgRefresh(flat);
	}
	m_context.refreshed_flat = nullptr;
	m_served_mode = -1;
	m_served_tape = nullptr;
	m_served_cfg  = nullptr;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->m_served_mode = -1;
		m_clean_evaluator->m_served_tape = nullptr;
		m_clean_evaluator->m_served_cfg  = nullptr;
	}
	if (!m_program.srt_plan_complete) return false;
	// Experiment DirectSrtReads: plans whose every SRT read is a dword load through a user-data
	// pointer pair read their words directly (claude/nhl27-perf).
	if (DirectSrtReadsEnabled() && m_active_mask.IsEmpty()) {
		if (m_program.direct_srt_state == ResourcePlan::DirectSrtState::Unknown) {
			m_program.direct_srt_state = BuildDirectSrtReads(m_program, m_clean_flat_slots)
			                                 ? ResourcePlan::DirectSrtState::Direct
			                                 : ResourcePlan::DirectSrtState::Unsupported;
		}
		const bool direct = m_program.direct_srt_state == ResourcePlan::DirectSrtState::Direct;
		if (DirectSrtDebug()) {
			CountDirectRefresh(m_program, !direct ? 2 : m_program.control_flow.empty() ? 0 : 1);
		}
		if (direct) {
			return m_program.control_flow.empty() ? RefreshDirectFlat(flat) : RefreshDirectControlFlow(flat);
		}
	}
	using TraceKind = ResourcePlan::CfgTrace::Kind;
	struct TraceReset {
		bool& recording;
		~TraceReset() { recording = false; }
	} trace_reset {m_cfg_trace_recording};
	// Trace replay reproduces DFS tape refreshes without scalar batching or ordered schedules.
	const bool trace_eligible = !m_program.control_flow.empty() && m_cfg_trace_enabled && m_cfg_tape_enabled &&
	                            !m_skip_tape && !m_cfg_schedule_enabled &&
	                            !m_scalar_batch_enabled && m_active_mask.IsEmpty() &&
	                            (m_clean_evaluator == nullptr || m_clean_evaluator->m_active_mask.IsEmpty());
	if (FrameBreakdown::Enabled() && !trace_eligible) {
		const int why = m_program.control_flow.empty() ? 0 : (!m_cfg_trace_enabled || !m_cfg_tape_enabled) ? 1 : 2;
		FrameBreakdown::g_counters.mat_tr_why[why].fetch_add(1, std::memory_order_relaxed);
	}
	bool compiled_missed = false;
	if (trace_eligible && m_compiled_enabled) {
		const int compiled = TryCompiled(flat, true);
		if (compiled > 0) {
			if (FrameBreakdown::Enabled()) FrameBreakdown::g_counters.mat_flat_trace_hits.fetch_add(1, std::memory_order_relaxed);
			call_timer.kind = 1;
			return true;
		}
		compiled_missed = compiled == -2;
	}
	if (trace_eligible && !compiled_missed && ReplayCfgTrace(flat) > 0) {
		if (FrameBreakdown::Enabled()) FrameBreakdown::g_counters.mat_flat_trace_hits.fetch_add(1, std::memory_order_relaxed);
		call_timer.kind = 1;
		return true;
	}
	if (FrameBreakdown::Enabled()) FrameBreakdown::g_counters.mat_flat_trace_other.fetch_add(1, std::memory_order_relaxed);
	if (!m_program.control_flow.empty()) PrepareCfgTape();
	else call_timer.kind = 2;
	auto& trace_steps = m_program.cfg_trace_scratch;
	const auto record = [&](ResourcePlan::CfgTrace::Step step) {
		if (m_cfg_trace_recording) trace_steps.push_back(step);
	};
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (!m_program.control_flow.empty() && read.flat_offset < flat.size()) {
			const auto& tape = m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0];
			const uint32_t root = tape.state == ResourcePlan::SrtTape::State::Ready
			                          ? tape.slot_ops[slot] : UINT32_MAX;
			const int done = RunCfgRoot(root, flat[read.flat_offset]);
			if (done > 0) record({.kind = TraceKind::Flat, .a = root, .b = read.flat_offset});
			if (done >= 0) return done != 0;
			if (!evaluator.Evaluate(read.value, flat[read.flat_offset])) return false;
			record({.kind = TraceKind::Eval, .a = slot, .b = read.flat_offset});
			return true;
		}
		return read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		int run_tape = -1;
		if (m_compiled_enabled) {
			const int compiled = TryCompiled(flat, false);
			if (compiled >= 0) return compiled != 0;
		}
		if (!m_skip_tape && SrtTapeVerifyEnabled()) {
			TapeReadCapture capture {.source = m_runtime};
			auto observed = m_runtime;
			observed.userdata = &capture;
			observed.read_memory = m_runtime.read_memory != nullptr ? CaptureTapeRead<false> : nullptr;
			observed.read_specialization_memory = m_runtime.read_specialization_memory != nullptr
			                                                 ? CaptureTapeRead<true>
			                                                 : nullptr;
			std::vector<ResourcePlan::EvaluationContext::Entry> saved_clean_values;
			const auto* saved_clean_flat = static_cast<const std::vector<uint32_t>*>(nullptr);
			auto saved_values = m_context.values;
			const auto* saved_flat = m_context.refreshed_flat;
			if (m_clean_evaluator != nullptr) {
				saved_clean_values = m_clean_evaluator->m_context.values;
				saved_clean_flat = m_clean_evaluator->m_context.refreshed_flat;
			}
			const auto original_runtime = m_runtime;
			m_runtime = observed;
			run_tape = RunTape(flat);
			m_runtime = original_runtime;
			const size_t tape_idx = m_clean_evaluator != nullptr ? 1u : 0u;
			const auto& tape = m_program.srt_tapes[tape_idx];
			if (run_tape == 1 && CanReplayTape(tape, original_runtime)) {
				if (!HasStableTapeReads(capture.reads)) {
					if (ShouldLogTapeVerify(m_program, TapeVerifyLogKind::UnstableReads)) {
						std::fprintf(stderr,
						             "SRT tape verify inconclusive shader=0x%016llx stage=%u base=0x%016llx: captured reads changed during tape\n",
						             static_cast<unsigned long long>(m_program.shader_hash),
						             static_cast<unsigned>(m_program.stage),
					             static_cast<unsigned long long>(m_runtime.shader_base));
					}
				} else {
					// Compare each compiled op against the recursive evaluator using only
					// the immutable reads captured by this Tape execution. Keep these fresh
					// contexts separate from the production contexts and from whole-plan replay.
					TapeReadReplay op_replay {.reads = capture.reads};
					auto op_runtime = m_runtime;
					op_runtime.userdata = &op_replay;
					op_runtime.read_memory = ReplayTapeRead<false>;
					op_runtime.read_specialization_memory = ReplayTapeRead<true>;
					SrtWalker op_clean(m_program, CleanRuntime(op_runtime), {}, nullptr, {}, true);
					SrtWalker op_walker(m_program, op_runtime, m_clean_flat_slots,
					                    m_clean_evaluator != nullptr ? &op_clean : nullptr, {}, true);
					auto saved_op_active_sources = m_program.active_sources;
					const auto& tape_values = m_program.srt_tape_values;
					for (size_t op_index = 0; op_index < tape.ops.size() && op_index < tape_values.size();
					     ++op_index) {
						const auto& op = tape.ops[op_index];
						if (op.source == nullptr) continue;
						uint64_t walker_value = 0;
						auto& evaluator = op.clean != 0u ? op_clean : op_walker;
						const bool walker_ok = evaluator.EvaluateWide(
						    Value(const_cast<Inst*>(op.source)), walker_value);
						if (op_replay.inconclusive) {
							if (ShouldLogTapeVerify(m_program, TapeVerifyLogKind::OpUnmatchedReplay)) {
								std::fprintf(stderr,
								             "SRT tape op verify inconclusive shader=0x%016llx stage=%u base=0x%016llx op=%zu opcode=%u memo=%u request=%s:0x%016llx words=%zu captured=%zu\n",
								             static_cast<unsigned long long>(m_program.shader_hash),
								             static_cast<unsigned>(m_program.stage),
								             static_cast<unsigned long long>(m_runtime.shader_base), op_index,
								             static_cast<unsigned>(op.opcode), op.memo_index,
								             op_replay.unmatched_strict ? "strict" : "ordinary",
								             static_cast<unsigned long long>(op_replay.unmatched_address),
								             op_replay.unmatched_words, capture.reads.size());
								std::array<size_t, 3> nearest {SIZE_MAX, SIZE_MAX, SIZE_MAX};
								std::array<uint64_t, 3> distances {UINT64_MAX, UINT64_MAX, UINT64_MAX};
								for (size_t read_index = 0; read_index < capture.reads.size(); ++read_index) {
									const auto& read = capture.reads[read_index];
									const auto distance = read.address > op_replay.unmatched_address
									                         ? read.address - op_replay.unmatched_address
									                         : op_replay.unmatched_address - read.address;
									for (size_t near_index = 0; near_index < nearest.size(); ++near_index) {
										if (distance >= distances[near_index]) continue;
										for (size_t move = nearest.size() - 1; move > near_index; --move) {
											distances[move] = distances[move - 1];
											nearest[move] = nearest[move - 1];
										}
										distances[near_index] = distance;
										nearest[near_index] = read_index;
										break;
									}
								}
								for (size_t near_index = 0; near_index < nearest.size(); ++near_index) {
									if (nearest[near_index] == SIZE_MAX) continue;
									const auto& read = capture.reads[nearest[near_index]];
									std::fprintf(stderr,
									             "SRT tape op verify captured domain=%s addr=0x%016llx words=%zu valid=%u first=0x%08x distance=0x%llx\n",
									             read.strict ? "strict" : "ordinary",
									             static_cast<unsigned long long>(read.address), read.values.size(),
									             read.valid ? 1u : 0u,
									             read.values.empty() ? 0u : read.values[0],
									             static_cast<unsigned long long>(distances[near_index]));
								}
							}
							break;
						}
						if (!walker_ok || walker_value != tape_values[op_index]) {
							if (ShouldLogTapeVerify(m_program, TapeVerifyLogKind::OpMismatch)) {
								std::fprintf(stderr,
								             "SRT tape op mismatch shader=0x%016llx stage=%u base=0x%016llx op=%zu opcode=%u memo=%u tape=0x%016llx walker=%s0x%016llx\n",
								             static_cast<unsigned long long>(m_program.shader_hash),
								             static_cast<unsigned>(m_program.stage),
								             static_cast<unsigned long long>(m_runtime.shader_base), op_index,
								             static_cast<unsigned>(op.opcode), op.memo_index,
								             static_cast<unsigned long long>(tape_values[op_index]),
								             walker_ok ? "" : "failed/",
								             static_cast<unsigned long long>(walker_value));
							}
							break;
						}
					}
					m_program.active_sources = std::move(saved_op_active_sources);

					TapeReadReplay replay {.reads = capture.reads};
					auto saved_active_sources = m_program.active_sources;
					auto replay_runtime = m_runtime;
					replay_runtime.userdata = &replay;
					replay_runtime.read_memory = ReplayTapeRead<false>;
					replay_runtime.read_specialization_memory = ReplayTapeRead<true>;
					SrtWalker replay_clean(m_program, CleanRuntime(replay_runtime), {}, nullptr, {}, true);
					SrtWalker replay_walker(m_program, replay_runtime, m_clean_flat_slots,
					                        m_clean_evaluator != nullptr ? &replay_clean : nullptr, {}, true);
					std::vector<uint32_t> walker_flat;
					const bool walker_ok = replay_walker.RefreshFlatBuffer(walker_flat);
					if (replay.inconclusive) {
						m_program.active_sources = std::move(saved_active_sources);
						if (ShouldLogTapeVerify(m_program, TapeVerifyLogKind::UnmatchedReplay)) {
							size_t ordinary_count = 0;
							size_t strict_count = 0;
							for (const auto& read: capture.reads) {
								if (read.strict) {
									++strict_count;
								} else {
									++ordinary_count;
								}
							}
							std::fprintf(stderr,
							             "SRT tape verify inconclusive shader=0x%016llx stage=%u base=0x%016llx request=%s:0x%016llx words=%zu captured={ordinary:%zu,strict:%zu}\n",
							             static_cast<unsigned long long>(m_program.shader_hash),
							             static_cast<unsigned>(m_program.stage),
							             static_cast<unsigned long long>(m_runtime.shader_base),
							             replay.unmatched_strict ? "strict" : "ordinary",
							             static_cast<unsigned long long>(replay.unmatched_address),
							             replay.unmatched_words, ordinary_count, strict_count);
							std::array<size_t, 3> nearest {SIZE_MAX, SIZE_MAX, SIZE_MAX};
							std::array<uint64_t, 3> distances {UINT64_MAX, UINT64_MAX, UINT64_MAX};
							for (size_t i = 0; i < capture.reads.size(); ++i) {
								const auto& read = capture.reads[i];
								const auto distance = read.address > replay.unmatched_address
								                         ? read.address - replay.unmatched_address
								                         : replay.unmatched_address - read.address;
								for (size_t j = 0; j < nearest.size(); ++j) {
									if (distance >= distances[j]) continue;
									for (size_t k = nearest.size() - 1; k > j; --k) {
										distances[k] = distances[k - 1];
										nearest[k] = nearest[k - 1];
									}
									distances[j] = distance;
									nearest[j] = i;
									break;
								}
							}
							for (size_t i = 0; i < nearest.size(); ++i) {
								const auto index = nearest[i];
								if (index == SIZE_MAX) continue;
								const auto& read = capture.reads[index];
								std::fprintf(stderr,
								             "SRT tape verify captured domain=%s addr=0x%016llx words=%zu valid=%u first=0x%08x distance=0x%llx\n",
								             read.strict ? "strict" : "ordinary",
								             static_cast<unsigned long long>(read.address), read.values.size(),
								             read.valid ? 1u : 0u, read.values.empty() ? 0u : read.values[0],
								             static_cast<unsigned long long>(distances[i]));
							}
						}
					} else {
						size_t mismatch = m_program.srt_reads.size();
						for (size_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
							const auto offset = m_program.srt_reads[slot].flat_offset;
							if (offset >= flat.size() || offset >= walker_flat.size() || flat[offset] != walker_flat[offset]) {
								mismatch = slot;
								break;
							}
						}
						if (!walker_ok || mismatch != m_program.srt_reads.size()) {
							const auto offset = mismatch < m_program.srt_reads.size()
							                        ? m_program.srt_reads[mismatch].flat_offset
							                        : UINT32_MAX;
							const auto op_index = mismatch < tape.slot_ops.size() ? tape.slot_ops[mismatch] : UINT32_MAX;
							const auto op_code = op_index < tape.ops.size()
							                         ? static_cast<unsigned>(tape.ops[op_index].opcode)
							                         : UINT32_MAX;
							if (ShouldLogTapeVerify(m_program, TapeVerifyLogKind::Mismatch)) {
								std::fprintf(stderr,
								             "SRT tape mismatch shader=0x%016llx stage=%u base=0x%016llx slot=%s%zu flat=%u tape=0x%08x walker=%s0x%08x op=%u opcode=%u\n",
								             static_cast<unsigned long long>(m_program.shader_hash),
								             static_cast<unsigned>(m_program.stage),
								             static_cast<unsigned long long>(m_runtime.shader_base),
								             mismatch < m_program.srt_reads.size() ? "" : "none/", mismatch, offset,
								             offset < flat.size() ? flat[offset] : 0u,
								             walker_ok ? "" : "failed/",
								             walker_ok && offset < walker_flat.size() ? walker_flat[offset] : 0u,
								             op_index, op_code);
							}
							m_context.values = std::move(saved_values);
							m_context.refreshed_flat = saved_flat;
							if (m_clean_evaluator != nullptr) {
								m_clean_evaluator->m_context.values = std::move(saved_clean_values);
								m_clean_evaluator->m_context.refreshed_flat = saved_clean_flat;
							}
							if (!walker_ok) return false;
							TransferCurrentEvaluationValues(replay_walker.m_context, m_context);
							if (m_clean_evaluator != nullptr) {
								TransferCurrentEvaluationValues(replay_clean.m_context,
								                               m_clean_evaluator->m_context);
							}
							flat = std::move(walker_flat);
							m_context.refreshed_flat = &flat;
							if (m_clean_evaluator != nullptr) m_clean_evaluator->m_context.refreshed_flat = &flat;
							return true;
						}
						m_program.active_sources = std::move(saved_active_sources);
					}
				}
			}
		} else if (!m_skip_tape) {
			run_tape = RunTape(flat);
		}
		if (run_tape >= 0) return run_tape != 0;

		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size();) {
			const auto batched = TryFlatScalarBatch({}, slot, flat);
			if (batched != 0) { slot += static_cast<uint32_t>(batched); continue; }
			if (!refresh(slot)) return false;
			++slot;
		}
		m_context.refreshed_flat = &flat;
		return true;
	}
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) active.at(source) = 0u;
	}
	{
		auto& trace = m_program.cfg_traces[m_clean_evaluator != nullptr ? 1 : 0];
		m_cfg_trace_recording = trace_eligible && !trace.disabled &&
		                        m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0].state ==
		                            ResourcePlan::SrtTape::State::Ready;
		if (m_cfg_trace_recording) {
			trace_steps.clear();
			if (trace.base_active.empty()) trace.base_active = active;
		}
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) continue;
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		uint64_t block_epoch = 0u;
		if (m_cfg_gather_enabled) {
			block_epoch = ++m_context.cfg_gather_epoch;
			if (block_epoch == 0u) {
				std::fill(m_context.cfg_gather_slot_epochs.begin(), m_context.cfg_gather_slot_epochs.end(), 0u);
				if (m_clean_evaluator != nullptr)
					std::fill(m_clean_evaluator->m_context.cfg_gather_slot_epochs.begin(),
					          m_clean_evaluator->m_context.cfg_gather_slot_epochs.end(), 0u);
				block_epoch = ++m_context.cfg_gather_epoch;
			}
			if (m_context.cfg_gather_slot_epochs.size() < m_program.srt_reads.size())
				m_context.cfg_gather_slot_epochs.resize(m_program.srt_reads.size(), 0u);
		}
		const auto mark_gather_slot = [&](uint32_t slot) {
			if (slot >= m_program.srt_reads.size()) return;
			const auto& read = m_program.srt_reads[slot];
			const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
			                   m_clean_flat_slots[read.flat_offset] != 0u;
			auto* evaluator = clean ? m_clean_evaluator : this;
			if (evaluator == nullptr) return;
			auto& epochs = evaluator->m_context.cfg_gather_slot_epochs;
			if (epochs.size() < m_program.srt_reads.size()) epochs.resize(m_program.srt_reads.size(), 0u);
			epochs[slot] = block_epoch;
		};
		for (const auto source: block.sources) active[source] = 1u;
		if (!block.sources.empty()) record({.kind = TraceKind::Activate, .a = index});
		for (size_t read_index = 0; read_index < block.srt_reads.size();) {
			const auto batched = TryFlatScalarBatch(std::span(block.srt_reads).subspan(read_index), 0u, flat);
			if (batched != 0) {
				if (m_cfg_gather_enabled) {
					for (size_t i = 0; i < batched; ++i) mark_gather_slot(block.srt_reads[read_index + i]);
				}
				read_index += batched; continue;
			}
			if (!refresh(block.srt_reads[read_index])) return false;
			if (m_cfg_gather_enabled) mark_gather_slot(block.srt_reads[read_index]);
			++read_index;
		}
		if (m_cfg_gather_enabled) {
			for (const auto source: block.sources) {
				if (source >= m_program.descriptor_gathers.size()) continue;
				const auto& gather = m_program.descriptor_gathers[source];
				if (!gather.valid || !gather.needs_flat) continue;
				bool eligible = true;
				for (uint32_t word_index = 0;
				     word_index < m_program.descriptor_sources[source].dword_count && eligible; ++word_index) {
					const auto& word = gather.words[word_index];
					if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::UserData) {
						eligible = false; // do not alter user-data observer calls
						break;
					}
					if (word.kind != ResourcePlan::DescriptorGather::Word::Kind::Flat) continue;
					const auto slot = word.read_slot;
					if (slot >= m_program.srt_reads.size() ||
					    m_program.srt_reads[slot].flat_offset != word.value) {
						eligible = false;
						break;
					}
					const bool clean = m_program.srt_reads[slot].flat_offset < m_clean_flat_slots.size() &&
					                   m_clean_flat_slots[m_program.srt_reads[slot].flat_offset] != 0u;
					auto* evaluator = clean ? m_clean_evaluator : this;
					auto* inst = m_program.srt_reads[slot].value.ResolveInstruction();
					if (evaluator == nullptr || !evaluator->m_active_mask.IsEmpty() || inst == nullptr) {
						eligible = false; break;
					}
					if (slot >= evaluator->m_context.cfg_gather_slot_epochs.size() ||
					    evaluator->m_context.cfg_gather_slot_epochs[slot] != block_epoch) {
						eligible = false; break;
					}
					const auto memo_index = inst->EvaluationIndex(m_program.evaluation_value_count);
					if (memo_index >= evaluator->m_context.values.size() ||
					    evaluator->m_context.values[memo_index].generation != evaluator->m_context.generation ||
					    word.value >= flat.size() ||
					    static_cast<uint32_t>(evaluator->m_context.values[memo_index].value) != flat[word.value])
						eligible = false;
				}
				if (eligible) {
					record({.kind = TraceKind::Gather, .a = source});
					m_cfg_gather_ready_sources[source] = 1u;
					if (m_clean_evaluator != nullptr && source < m_clean_evaluator->m_cfg_gather_ready_sources.size())
						m_clean_evaluator->m_cfg_gather_ready_sources[source] = 1u;
				}
			}
		}
		uint32_t condition = 0;
		auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		int predicate_done = -1;
		// Immutable predicates (capture_specialization_reads unset) need no strict reader.
		const bool can_decide = m_runtime.read_specialization_memory != nullptr ||
		                        !m_program.capture_specialization_reads;
		if (!block.condition.IsEmpty() && can_decide) {
			const auto& tape = m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0];
			const uint32_t root = tape.state == ResourcePlan::SrtTape::State::Ready
			                          ? tape.condition_ops[index] : UINT32_MAX;
			const bool tri = tape.state == ResourcePlan::SrtTape::State::Ready && tape.condition_tri[index] != 0u;
			predicate_done = RunCfgRoot(root, condition);
			if (predicate_done > 0 && tri) {
				const uint32_t outcome = condition == 2u ? 2u : condition != 0u ? 1u : 0u;
				record({.kind = TraceKind::CondTri, .a = root, .b = outcome});
				if (outcome == 2u) predicate_done = 0; // the walker's predicate failed
			} else if (predicate_done > 0) record({.kind = TraceKind::Cond, .a = root, .b = condition != 0u ? 1u : 0u});
			else if (predicate_done == 0) m_cfg_trace_recording = false;
			else {
				predicate_done = predicate.Evaluate(block.condition, condition) ? 1 : 0;
				record({.kind = TraceKind::CondEval, .a = index,
				        .b = predicate_done == 0 ? 2u : condition != 0u ? 1u : 0u});
			}
		}
		if (!block.condition.IsEmpty() && can_decide && predicate_done > 0) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	if (m_cfg_trace_recording) {
		record({.kind = TraceKind::End});
		const uint32_t paths_before = m_program.cfg_traces[m_clean_evaluator != nullptr ? 1 : 0].paths;
		MergeCfgTrace();
		if (FrameBreakdown::Enabled())
			FrameBreakdown::g_counters.mat_tr_why[m_program.cfg_traces[m_clean_evaluator != nullptr ? 1 : 0].paths != paths_before ? 15 : 14]
			    .fetch_add(1, std::memory_order_relaxed);
	} else if (trace_eligible && FrameBreakdown::Enabled()) {
		FrameBreakdown::g_counters.mat_tr_why[13].fetch_add(1, std::memory_order_relaxed);
	}
	m_context.refreshed_flat = &flat;
	if (m_clean_evaluator != nullptr) m_clean_evaluator->m_context.refreshed_flat = &flat;
	return true;
}

namespace {

const char* const kRejectNames[] = {
	"ok",
	"control_flow",
	"incomplete_plan",
	"active_mask",
	"unsupported_opcode",
	"unsupported_type",
	"circular_or_complex",
	"missing_clean_eval",
};
} // namespace

const char* SrtTapeRejectName(size_t reason) {
	if (reason < std::size(kRejectNames)) return kRejectNames[reason];
	return "unknown";
}

SrtTapeStats GetSrtTapeStats() {
	SrtTapeStats s;
	s.runs = g_tape_stats.runs.load(std::memory_order_relaxed);
	s.failed = g_tape_stats.failed.load(std::memory_order_relaxed);
	s.walker = g_tape_stats.walker.load(std::memory_order_relaxed);
	s.built = g_tape_stats.built.load(std::memory_order_relaxed);
	s.unsupported = g_tape_stats.unsupported.load(std::memory_order_relaxed);
	s.ops = g_tape_stats.ops.load(std::memory_order_relaxed);
	s.cfg_runs = g_tape_stats.cfg_runs.load(std::memory_order_relaxed);
	s.cfg_fallback = g_tape_stats.cfg_fallback.load(std::memory_order_relaxed);
	s.cfg_scheduled = g_tape_stats.cfg_scheduled.load(std::memory_order_relaxed);
	s.cfg_checked = g_tape_stats.cfg_checked.load(std::memory_order_relaxed);
	s.cfg_matched = g_tape_stats.cfg_matched.load(std::memory_order_relaxed);
	s.cfg_mismatch = g_tape_stats.cfg_mismatch.load(std::memory_order_relaxed);
	s.cfg_inconclusive = g_tape_stats.cfg_inconclusive.load(std::memory_order_relaxed);
	s.cfg_trace_hits = g_tape_stats.cfg_trace_hits.load(std::memory_order_relaxed);
	s.cfg_trace_misses = g_tape_stats.cfg_trace_misses.load(std::memory_order_relaxed);
	s.cfg_trace_recorded = g_tape_stats.cfg_trace_recorded.load(std::memory_order_relaxed);
	for (size_t i = 0; i < s.reject.size(); ++i) {
		s.reject[i] = g_tape_stats.reject[i].load(std::memory_order_relaxed);
	}
	return s;
}

namespace {
bool BuildTapeForPlan(const ResourcePlan& plan, bool with_clean,
                      std::span<const uint8_t> clean_flat_slots,
                      ResourcePlan::SrtTape& tape, size_t& reject_reason, bool cfg = false) {
	if (!cfg && !plan.control_flow.empty()) {
		reject_reason = 1;
		return false;
	}
	if (!plan.srt_plan_complete) {
		reject_reason = 2;
		return false;
	}

	tape.ops.clear();
	tape.slot_ops.clear();
	tape.condition_ops.clear();
	tape.condition_tri.clear();
	tape.slot_ops.reserve(plan.srt_reads.size());
	tape.clean_root = with_clean;

	std::unordered_map<TapeInstKey, uint32_t, TapeInstKeyHash> inst_to_op;
	std::unordered_set<TapeInstKey, TapeInstKeyHash> compiling;
	// Immediate values have no evaluation or memory-reader domain. Share their raw
	// 64-bit representation across roots, including values with different IR types.
	std::unordered_map<uint64_t, uint32_t> immediate_to_op;

	auto get_value_u64 = [](Value val, uint64_t& out) -> bool {
		val = val.Resolve();
		if (!val.IsImmediate()) return false;
		switch (val.GetType()) {
			case Type::U1: out = val.U1(); return true;
			case Type::U8: out = val.U8(); return true;
			case Type::U16: out = val.U16(); return true;
			case Type::U32: out = val.U32(); return true;
			case Type::U64: out = val.U64(); return true;
			case Type::F32: out = std::bit_cast<uint32_t>(val.F32Value()); return true;
			default: return false;
		}
	};

	std::function<int(Value, bool)> emit_val;
	emit_val = [&](Value val, bool is_clean) -> int {
		// Bound hostile/cyclic graphs before recursion or op-vector growth.
		if (compiling.size() >= 256u || tape.ops.size() >= 65536u) return -1;
		val = val.Resolve();
		if (val.IsImmediate()) {
			uint64_t imm = 0;
			if (!get_value_u64(val, imm)) return -1;
			if (const auto found = immediate_to_op.find(imm); found != immediate_to_op.end()) {
				return static_cast<int>(found->second);
			}
			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			ResourcePlan::SrtTape::Op op;
			op.kind = ResourcePlan::SrtTape::Kind::Immediate;
			op.imm = imm;
			tape.ops.push_back(op);
			immediate_to_op.emplace(imm, op_idx);
			return static_cast<int>(op_idx);
		}

		const auto* inst = val.TryInstruction();
		if (inst == nullptr) return -1;

		const TapeInstKey key {inst, is_clean};
		if (auto it = inst_to_op.find(key); it != inst_to_op.end()) {
			return static_cast<int>(it->second);
		}
		if (!compiling.insert(key).second) return -1;
		struct CompilationGuard {
			decltype(compiling)& pending;
			TapeInstKey key;
			~CompilationGuard() { pending.erase(key); }
		} guard {compiling, key};

		const auto opcode = inst->GetOpcode();
		ResourcePlan::SrtTape::Op op;
		op.opcode = opcode;
		op.source = inst;
		op.clean = is_clean ? 1 : 0;
		op.memo_index = inst->EvaluationIndex(const_cast<ResourcePlan&>(plan).evaluation_value_count);

		if (opcode == ValueOpcode::GetUserData) {
			op.kind = ResourcePlan::SrtTape::Kind::UserData;
			op.args[0] = RegIndex(inst->Arg(0).ScalarRegister());
			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[key] = op_idx;
			return static_cast<int>(op_idx);
		}
		if (opcode == ValueOpcode::GetShaderBase) {
			op.kind = ResourcePlan::SrtTape::Kind::ShaderBase;
			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[key] = op_idx;
			return static_cast<int>(op_idx);
		}

		if (opcode == ValueOpcode::ReadConst) {
			const auto slot_val = inst->Arg(1).Resolve();
			if (!slot_val.IsImmediate() || slot_val.GetType() != Type::U32) return -1;
			uint32_t slot = slot_val.U32();
			if (slot >= plan.srt_reads.size()) return -1;

			bool target_clean = is_clean;
			if (slot < clean_flat_slots.size() && clean_flat_slots[slot] != 0u && with_clean) {
				target_clean = true;
			}
			const int alias = emit_val(plan.srt_reads[slot].value, target_clean);
			if (!cfg || alias < 0) return alias;
			op.kind = ResourcePlan::SrtTape::Kind::Inst;
			op.args[0] = static_cast<uint32_t>(alias);
			op.dependency_count = 1;
			const auto index = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[key] = index;
			return static_cast<int>(index);
		}

		if (opcode == ValueOpcode::LoadAddressU32 || opcode == ValueOpcode::ReadConstBuffer ||
		    opcode == ValueOpcode::LoadBufferU32) {
			if (cfg && opcode == ValueOpcode::LoadBufferU32 && with_clean && !is_clean) {
				const int alias = emit_val(val, true);
				if (alias < 0) return -1;
				op.opcode = ValueOpcode::ReadConst;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(alias);
				op.dependency_count = 1;
				const auto index = static_cast<uint32_t>(tape.ops.size());
				tape.ops.push_back(op);
				inst_to_op[key] = index;
				return static_cast<int>(index);
			}
			if (!IsRawRead(plan, *inst)) return -1;
			// The recursive evaluator checks the vector-load guard before evaluating
			// its address or touching memory. The tape currently cannot express that
			// conditional dependency, so keep only statically enabled vector reads.
			if (opcode == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				if (!guard.IsImmediate() || guard.GetType() != Type::U1 || !guard.U1()) {
					reject_reason = 4;
					return -1;
				}
			}
			op.kind = ResourcePlan::SrtTape::Kind::RawRead;
			op.aux = inst->Flags<MemoryFlags>().index;
			// Recursive LoadBufferU32 evaluation delegates the whole node to the
			// clean evaluator when available. Keep this op's memo domain as requested,
			// but evaluate all of its dependencies in that same clean domain.
			const bool dependencies_clean =
			    is_clean || (opcode == ValueOpcode::LoadBufferU32 && with_clean);

			const auto* handle = inst->Arg(0).ResolveInstruction();
			if (handle == nullptr) return -1;
			int low_idx = emit_val(handle->Arg(0), dependencies_clean);
			int high_idx = emit_val(handle->Arg(1), dependencies_clean);
			int off_idx = emit_val(inst->Arg(1), dependencies_clean);
			if (low_idx < 0 || high_idx < 0 || off_idx < 0) return -1;
			op.args[0] = static_cast<uint32_t>(low_idx);
			op.args[1] = static_cast<uint32_t>(high_idx);
			op.args[2] = static_cast<uint32_t>(off_idx);
			op.dependency_count = 3;

			if (opcode == ValueOpcode::ReadConstBuffer || opcode == ValueOpcode::LoadBufferU32) {
				if (handle->NumArgs() != 4u) return -1;
				int rec_idx = emit_val(handle->Arg(2), dependencies_clean);
				int w3_idx = emit_val(handle->Arg(3), dependencies_clean);
				if (rec_idx < 0 || w3_idx < 0) return -1;
				op.args[3] = static_cast<uint32_t>(rec_idx);
				op.args[4] = static_cast<uint32_t>(w3_idx);
				op.dependency_count = 5;
			}

			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[key] = op_idx;
			return static_cast<int>(op_idx);
		}

		if (opcode == ValueOpcode::ConditionRef) {
			const int alias = emit_val(inst->Arg(0), is_clean);
			if (!cfg || alias < 0) return alias;
			op.opcode = ValueOpcode::ReadConst; // scalar alias, with its own memo index
			op.kind = ResourcePlan::SrtTape::Kind::Inst;
			op.args[0] = static_cast<uint32_t>(alias);
			op.dependency_count = 1;
			const auto index = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[key] = index;
			return static_cast<int>(index);
		}
		if (opcode == ValueOpcode::LogicalAnd || opcode == ValueOpcode::LogicalOr) {
			// The walker short-circuits and tolerates a failed operand when the other one
			// decides the result. With one immediate operand that reduces to a constant or
			// to bool(other operand), which fails exactly when the walker's result fails.
			// The decisive value is 0 for And and nonzero for Or.
			const bool is_and = opcode == ValueOpcode::LogicalAnd;
			uint64_t imm = 0;
			int other_arg = -1;
			if (get_value_u64(inst->Arg(1), imm)) other_arg = 0;
			else if (get_value_u64(inst->Arg(0), imm)) other_arg = 1;
			if (other_arg < 0) {
				reject_reason = 4;
				return -1;
			}
			const auto emit_imm = [&](uint64_t value) {
				if (const auto found = immediate_to_op.find(value); found != immediate_to_op.end()) {
					return static_cast<int>(found->second);
				}
				ResourcePlan::SrtTape::Op imm_op;
				imm_op.kind = ResourcePlan::SrtTape::Kind::Immediate;
				imm_op.imm = value;
				const auto index = static_cast<uint32_t>(tape.ops.size());
				tape.ops.push_back(imm_op);
				immediate_to_op.emplace(value, index);
				return static_cast<int>(index);
			};
			if ((imm != 0u) != is_and) return emit_imm(is_and ? 0u : 1u);
			const int a = emit_val(inst->Arg(other_arg), is_clean);
			const int zero = emit_imm(0u);
			if (a < 0 || zero < 0) return -1;
			op.opcode = ValueOpcode::LogicalXor; // (a != 0) != false
			op.kind = ResourcePlan::SrtTape::Kind::Inst;
			op.args[0] = static_cast<uint32_t>(a);
			op.args[1] = static_cast<uint32_t>(zero);
			op.dependency_count = 2;
			const auto index = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[key] = index;
			return static_cast<int>(index);
		}

		switch (opcode) {
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::LogicalNot: {
				int a = emit_val(inst->Arg(0), is_clean);
				if (a < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.dependency_count = 1;
				break;
			}
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalXor: {
				int a = emit_val(inst->Arg(0), is_clean);
				int b = emit_val(inst->Arg(1), is_clean);
				if (a < 0 || b < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.args[1] = static_cast<uint32_t>(b);
				op.dependency_count = 2;
				break;
			}
			case ValueOpcode::CompositeExtractU64: {
				int a = emit_val(inst->Arg(0), is_clean);
				const auto idx_val = inst->Arg(1).Resolve();
				if (a < 0 || !idx_val.IsImmediate() || idx_val.GetType() != Type::U32) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.imm = idx_val.U32();
				op.dependency_count = 1;
				break;
			}
			case ValueOpcode::CompositeExtractU32x2: {
				const auto* src = inst->Arg(0).ResolveInstruction();
				const auto idx_val = inst->Arg(1).Resolve();
				if (src == nullptr || !idx_val.IsImmediate() || idx_val.GetType() != Type::U32) return -1;
				uint32_t comp = idx_val.U32();
				if (comp >= 2u) return -1;
				if (src->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					const int alias = emit_val(src->Arg(comp), is_clean);
					if (!cfg || alias < 0) return alias;
					op.opcode = ValueOpcode::ReadConst; // scalar alias, with its own memo index
					op.kind = ResourcePlan::SrtTape::Kind::Inst;
					op.args[0] = static_cast<uint32_t>(alias);
					op.dependency_count = 1;
					break;
				}
				return -1;
			}
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract: {
				int a = emit_val(inst->Arg(0), is_clean);
				int b = emit_val(inst->Arg(1), is_clean);
				int c = emit_val(inst->Arg(2), is_clean);
				if (a < 0 || b < 0 || c < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.args[1] = static_cast<uint32_t>(b);
				op.args[2] = static_cast<uint32_t>(c);
				op.dependency_count = 3;
				break;
			}
			case ValueOpcode::BitFieldInsert: {
				int a = emit_val(inst->Arg(0), is_clean);
				int b = emit_val(inst->Arg(1), is_clean);
				int c = emit_val(inst->Arg(2), is_clean);
				int d = emit_val(inst->Arg(3), is_clean);
				if (a < 0 || b < 0 || c < 0 || d < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.args[1] = static_cast<uint32_t>(b);
				op.args[2] = static_cast<uint32_t>(c);
				op.args[3] = static_cast<uint32_t>(d);
				op.dependency_count = 4;
				break;
			}
		default:
				// Any lazy/short-circuiting op (Select, Phi, ReadFirstLane, LogicalAnd, LogicalOr)
				// falls back to the dynamic walker.
				if (static const bool diag = std::getenv("KYTY_TAPE_REJECT_DIAG") != nullptr; diag) {
					const auto name = ValueOpcodeName(opcode);
					std::fprintf(stderr, "TAPEREJECT cfg=%d op=%.*s\n", cfg ? 1 : 0,
					             static_cast<int>(name.size()), name.data());
				}
				reject_reason = 4;
				return -1;
		}

		uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
		tape.ops.push_back(op);
		inst_to_op[key] = op_idx;
		return static_cast<int>(op_idx);
	};

	// A value whose walker evaluation always fails: an undefined value, or an And whose
	// other operand is a nonzero immediate (the walker cannot short-circuit it).
	std::function<bool(Value)> always_fails = [&](Value val) {
		const auto* inst = val.Resolve().TryInstruction();
		if (inst == nullptr) return false;
		switch (inst->GetOpcode()) {
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU32: return true;
			case ValueOpcode::LogicalAnd: {
				uint64_t imm = 0;
				return (get_value_u64(inst->Arg(1), imm) && imm != 0u && always_fails(inst->Arg(0))) ||
				       (get_value_u64(inst->Arg(0), imm) && imm != 0u && always_fails(inst->Arg(1)));
			}
			default: return false;
		}
	};
	// [Not](Or(A, F)) with F always failing: the walker yields Or = A ? 1 : failure, so the
	// predicate is a tri-state of A. Emit Select(A, 1 or 0, 2), 2 meaning "failed". The op is
	// synthetic: no source instruction and no walker memo slot.
	const auto emit_tri_condition = [&](Value condition, bool is_clean) -> int {
		auto val = condition.Resolve();
		bool negate = false;
		if (const auto* inst = val.TryInstruction(); inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalNot) {
			negate = true;
			val = inst->Arg(0).Resolve();
		}
		const auto* inst = val.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LogicalOr) return -1;
		Value a_val;
		if (always_fails(inst->Arg(1)) && !always_fails(inst->Arg(0))) a_val = inst->Arg(0);
		else if (always_fails(inst->Arg(0)) && !always_fails(inst->Arg(1))) a_val = inst->Arg(1);
		else return -1;
		const int a = emit_val(a_val, is_clean);
		if (a < 0) return -1;
		const auto emit_imm = [&](uint64_t value) {
			if (const auto found = immediate_to_op.find(value); found != immediate_to_op.end()) {
				return found->second;
			}
			ResourcePlan::SrtTape::Op imm_op;
			imm_op.kind = ResourcePlan::SrtTape::Kind::Immediate;
			imm_op.imm = value;
			const auto index = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(imm_op);
			immediate_to_op.emplace(value, index);
			return index;
		};
		ResourcePlan::SrtTape::Op op;
		op.kind = ResourcePlan::SrtTape::Kind::Inst;
		op.opcode = ValueOpcode::SelectU32;
		op.clean = is_clean ? 1 : 0;
		op.args[0] = static_cast<uint32_t>(a);
		op.args[1] = emit_imm(negate ? 0u : 1u);
		op.args[2] = emit_imm(2u);
		op.dependency_count = 3;
		const auto index = static_cast<uint32_t>(tape.ops.size());
		tape.ops.push_back(op);
		return static_cast<int>(index);
	};

	for (size_t slot = 0; slot < plan.srt_reads.size(); ++slot) {
		const auto& read = plan.srt_reads[slot];
		bool clean = read.flat_offset < clean_flat_slots.size() && clean_flat_slots[read.flat_offset] != 0u;
		if (clean && !with_clean) {
			if (cfg) { tape.slot_ops.push_back(UINT32_MAX); continue; }
			reject_reason = 7;
			return false;
		}
		int emitted = emit_val(read.value, clean && with_clean);
		if (emitted < 0) {
			if (cfg) { tape.slot_ops.push_back(UINT32_MAX); continue; }
			if (reject_reason == 0) reject_reason = 6;
			return false;
		}
		tape.slot_ops.push_back(static_cast<uint32_t>(emitted));
	}
	if (cfg) {
		for (const auto& block: plan.control_flow) {
			auto root = block.condition.IsEmpty() ? -1 : emit_val(block.condition, with_clean);
			const bool tri = root < 0 && !block.condition.IsEmpty() &&
			                 (root = emit_tri_condition(block.condition, with_clean)) >= 0;
			tape.condition_ops.push_back(root < 0 ? UINT32_MAX : static_cast<uint32_t>(root));
			tape.condition_tri.push_back(tri ? 1u : 0u);
			if (static const bool diag = std::getenv("KYTY_TAPE_REJECT_DIAG") != nullptr;
			    diag && root < 0 && !block.condition.IsEmpty()) {
				std::string text;
				std::function<void(Value, int)> dump = [&](Value v, int depth) {
					v = v.Resolve();
					if (v.IsImmediate()) { uint64_t iv = 0; get_value_u64(v, iv); text += std::to_string(iv); return; }
					const auto* i = v.TryInstruction();
					if (i == nullptr) { text += "?"; return; }
					text += ValueOpcodeName(i->GetOpcode());
					if (depth == 0) return;
					text += "(";
					for (size_t a = 0; a < i->NumArgs(); ++a) {
						if (a) text += ",";
						if (i->Arg(a).IsEmpty()) text += "-"; else dump(i->Arg(a), depth - 1);
					}
					text += ")";
				};
				dump(block.condition, 4);
				std::fprintf(stderr, "TAPECOND %s\n", text.c_str());
			}
		}
	}

	return true;
}
} // namespace

int SrtWalker::RunCfgSchedule(uint32_t root, uint32_t& result) {
	auto& tape = m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0];
	if (tape.demand_schedules.size() != tape.ops.size()) {
		tape.demand_schedules.resize(tape.ops.size());
	}
	auto& schedule = tape.demand_schedules[root];
	if (schedule.state == ResourcePlan::SrtTape::State::Unbuilt) {
		// Schedules are built once, never per refresh. Bound DAG expansion and
		// aggregate storage: an unusually shared/large expression uses the existing
		// demand stack instead of consuming unbounded memory at shader warmup.
		constexpr size_t MaxRootSteps = 32768;
		constexpr size_t MaxTapeSteps = 262144;
		size_t used = 0;
		for (const auto& cached: tape.demand_schedules) used += cached.steps.size();
		const size_t budget = std::min(MaxRootSteps, MaxTapeSteps - std::min(used, MaxTapeSteps));
		struct Pending { uint32_t op; uint8_t dependency; uint32_t enter; };
		std::vector<Pending> pending;
		std::vector<uint8_t> seen(tape.ops.size(), 0u);
		pending.push_back({root, 0, UINT32_MAX});
		bool supported = true;
		while (!pending.empty()) {
			auto& node = pending.back();
			const auto& op = tape.ops[node.op];
			if (node.enter == UINT32_MAX) {
				if (schedule.steps.size() >= budget) { supported = false; break; }
				node.enter = static_cast<uint32_t>(schedule.steps.size());
				schedule.steps.push_back({node.op, node.enter + 1u});
				// The earlier ordered occurrence either completed (mark/memo hit)
				// or ended the refresh on failure. A repeated DAG edge only needs
				// its enter check, avoiding exponential schedule expansion.
				if (seen[node.op] || op.dependency_count == 0) {
					seen[node.op] = 1u;
					pending.pop_back();
					continue;
				}
				seen[node.op] = 1u;
			}
			if (node.dependency < op.dependency_count) {
				const auto dependency = op.args[node.dependency++];
				// Tape compilation is topological. Reject corrupt/cyclic schedules
				// rather than looping or changing the original fallback behavior.
				if (dependency >= node.op) { supported = false; break; }
				pending.push_back({dependency, 0, UINT32_MAX});
				continue;
			}
			if (schedule.steps.size() >= budget) { supported = false; break; }
			schedule.steps.push_back({node.op, UINT32_MAX});
			schedule.steps[node.enter].next = static_cast<uint32_t>(schedule.steps.size());
			pending.pop_back();
		}
		schedule.state = supported ? ResourcePlan::SrtTape::State::Ready
		                           : ResourcePlan::SrtTape::State::Unsupported;
		if (!supported) std::vector<ResourcePlan::SrtTape::DemandStep>().swap(schedule.steps);
	}
	if (schedule.state != ResourcePlan::SrtTape::State::Ready) return -1;
	g_tape_stats.cfg_scheduled.fetch_add(1, std::memory_order_relaxed);
	auto& values = m_context.tape_values;
	auto& marks = m_context.tape_marks;
	for (uint32_t pc = 0; pc < schedule.steps.size();) {
		if (m_runtime.evaluation_counts) ++m_runtime.evaluation_counts->cfg_schedule_steps;
		const auto step = schedule.steps[pc];
		const auto& op = tape.ops[step.op];
		auto& context = op.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
		if (step.next != UINT32_MAX) {
			// Enter checks precisely the same domain memo before descending.
			// A successful parent skips its entire subtree, including observers.
			if (op.memo_index != UINT32_MAX && op.memo_index < context.values.size()) {
				const auto& memo = context.values[op.memo_index];
				if (memo.generation == context.generation) {
					values[step.op] = memo.value;
					marks[step.op] = 1u;
					pc = step.next;
					continue;
				}
				if (memo.generation == (context.generation | 1u)) {
					if (op.clean && m_clean_evaluator != nullptr) ++m_clean_evaluator->m_cycle_count;
					else ++m_cycle_count;
					return 0;
				}
				if (memo.generation == (context.generation | 2u)) return 0;
			}
			if (marks[step.op] == 1u) { pc = step.next; continue; }
			if (marks[step.op] == 2u) return 0;
			// Leaves have one step; interior nodes execute at their exit.
			if (step.next != pc + 1u) { ++pc; continue; }
		}
		if (!ExecuteTapeOp(op, values, step.op)) {
			marks[step.op] = 2u;
			if (op.memo_index != UINT32_MAX) {
				context.values.resize(m_program.evaluation_value_count);
				context.values[op.memo_index].generation = context.generation | 2u;
			}
			return 0;
		}
		marks[step.op] = 1u;
		++pc;
	}
	result = static_cast<uint32_t>(values[root]);
	return 1;
}

void SrtWalker::PrepareCfgTape() {
	if (m_skip_tape || !m_cfg_tape_enabled || !m_active_mask.IsEmpty() ||
	    (m_clean_evaluator != nullptr && !m_clean_evaluator->m_active_mask.IsEmpty())) {
		g_tape_stats.walker.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	auto& tape = m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0];
	if (tape.state == ResourcePlan::SrtTape::State::Unbuilt) {
		size_t reason = 0;
		const bool built = BuildTapeForPlan(m_program, m_clean_evaluator != nullptr,
		                                    m_clean_flat_slots, tape, reason, true);
		tape.state = built ? ResourcePlan::SrtTape::State::Ready
		                   : ResourcePlan::SrtTape::State::Unsupported;
		// Trace steps index this tape's ops; a rebuilt tape starts a new trace.
		m_program.cfg_traces[m_clean_evaluator != nullptr ? 1 : 0] = {};
	}
	if (tape.state != ResourcePlan::SrtTape::State::Ready) return;
	m_context.tape_values.resize(tape.ops.size());
	m_context.tape_marks.assign(tape.ops.size(), 0u);
	m_context.tape_pending.clear();
}

int SrtWalker::RunCfgRoot(uint32_t root, uint32_t& result) {
	if (m_skip_tape || !m_cfg_tape_enabled || root == UINT32_MAX || !m_active_mask.IsEmpty() ||
	    (m_clean_evaluator != nullptr && !m_clean_evaluator->m_active_mask.IsEmpty())) {
		g_tape_stats.cfg_fallback.fetch_add(1, std::memory_order_relaxed);
		return -1;
	}
	const auto& tape = m_program.cfg_srt_tapes[m_clean_evaluator != nullptr ? 1 : 0];
	if (root >= tape.ops.size()) return -1;
	g_tape_stats.cfg_runs.fetch_add(1, std::memory_order_relaxed);
	if (m_cfg_schedule_enabled) {
		m_cfg_trace_recording = false;
		const int scheduled = RunCfgSchedule(root, result);
		if (scheduled >= 0) return scheduled;
	}
	auto& values = m_context.tape_values;
	auto& marks = m_context.tape_marks;
	auto& pending = m_context.tape_pending;
	auto* trace = m_cfg_trace_recording ? &m_program.cfg_trace_scratch : nullptr;
	using TraceKind = ResourcePlan::CfgTrace::Kind;
	pending.clear();
	pending.emplace_back(root, 0u);
	while (!pending.empty()) {
		const auto index = pending.back().first;
		const auto& op = tape.ops[index];
		auto& context = op.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
		// Check the recursive evaluator's domain-specific memo before descending.
		// Failed reads remain failed; an in-progress memo retains cycle semantics.
		if (op.memo_index != UINT32_MAX && op.memo_index < context.values.size()) {
			const auto& memo = context.values[op.memo_index];
			if (memo.generation == context.generation) {
				values[index] = memo.value;
				if (trace != nullptr && marks[index] != 1u) trace->push_back({.kind = TraceKind::Memo, .a = index});
				marks[index] = 1u;
				pending.pop_back();
				continue;
			}
			if (memo.generation == (context.generation | 1u)) {
				if (op.clean && m_clean_evaluator != nullptr) ++m_clean_evaluator->m_cycle_count;
				else ++m_cycle_count;
				m_cfg_trace_recording = false;
				return 0;
			}
			if (memo.generation == (context.generation | 2u)) { m_cfg_trace_recording = false; return 0; }
		}
		if (marks[index] == 1u) { pending.pop_back(); continue; }
		if (marks[index] == 2u) { m_cfg_trace_recording = false; return 0; }
		if (pending.back().second < op.dependency_count) {
			const auto dependency = op.args[pending.back().second++];
			pending.emplace_back(dependency, 0u);
			continue;
		}
		if (!ExecuteTapeOp(op, values, index)) {
			marks[index] = 2u;
			if (op.memo_index != UINT32_MAX) {
				context.values.resize(m_program.evaluation_value_count);
				context.values[op.memo_index].generation = context.generation | 2u;
			}
			m_cfg_trace_recording = false;
			return 0;
		}
		if (trace != nullptr) trace->push_back({.kind = TraceKind::Exec, .a = index});
		marks[index] = 1u;
		pending.pop_back();
	}
	result = static_cast<uint32_t>(values[root]);
	return 1;
}

int SrtWalker::ReplayCfgTrace(std::vector<uint32_t>& flat) {
	using TraceKind = ResourcePlan::CfgTrace::Kind;
	const size_t variant = m_clean_evaluator != nullptr ? 1 : 0;
	auto& trace = m_program.cfg_traces[variant];
	const auto& tape = m_program.cfg_srt_tapes[variant];
	if (trace.disabled || trace.steps.empty() || tape.state != ResourcePlan::SrtTape::State::Ready ||
	    trace.strict != (m_runtime.read_specialization_memory != nullptr)) {
		if (FrameBreakdown::Enabled()) {
			const int why = trace.disabled ? 4 : trace.steps.empty() ? 3 : tape.state != ResourcePlan::SrtTape::State::Ready ? 5 : 6;
			FrameBreakdown::g_counters.mat_tr_why[why].fetch_add(1, std::memory_order_relaxed);
		}
		return -1;
	}
	int miss_why = 0;
	SRT_PROF_STEPPER(prof);
	SRT_PROF_TICK(prof, SrtProf::Preamble);
	SRT_PROF_COUNT(replays, 1);
	auto& values = m_context.tape_values;
	if (values.size() < tape.ops.size()) values.resize(tape.ops.size());
	{
		FrameBreakdown::Timer alloc_timer(FrameBreakdown::g_counters.mat_flat_alloc_ns, FrameBreakdown::DetailEnabled());
		flat.assign(m_program.srt_reads.size(), 0u);
	}
	auto& active = m_program.active_sources;
	active = trace.base_active;
	const auto context_for = [&](const ResourcePlan::SrtTape::Op& op) -> ResourcePlan::EvaluationContext& {
		return op.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
	};
	// Memo slots this replay published. Every memo written so far holds a correct value
	// for this session, but a miss restores the tape's slots so the DFS fallback records
	// the same op sequence as the trie and the new branch can be merged. Memoized
	// failures from before the refresh are never touched (Exec checks them first).
	auto& published = m_context.tape_pending;
	published.clear();
	const auto miss = [&] {
		SRT_PROF_FINISH(prof);
		if (FrameBreakdown::Enabled()) FrameBreakdown::g_counters.mat_tr_why[miss_why].fetch_add(1, std::memory_order_relaxed);
		for (const auto& [index, unused]: published) {
			const auto& op = tape.ops[index];
			context_for(op).values[op.memo_index].generation = 0;
		}
		published.clear();
		if (m_cfg_gather_enabled) {
			std::fill(m_cfg_gather_ready_sources.begin(), m_cfg_gather_ready_sources.end(), 0u);
			if (m_clean_evaluator != nullptr)
				std::fill(m_clean_evaluator->m_cfg_gather_ready_sources.begin(),
				          m_clean_evaluator->m_cfg_gather_ready_sources.end(), 0u);
		}
		if (++trace.misses > (m_source_reuse_enabled ? 8192u : 256u) && trace.misses > trace.hits) trace.disabled = true;
		g_tape_stats.cfg_trace_misses.fetch_add(1, std::memory_order_relaxed);
		return -1;
	};
	// KYTY_SRT_SOURCE_REUSE: hoisted loop invariants and one batched stats flush per replay.
	const bool     reuse      = m_source_reuse_enabled;
	const bool     stats      = FrameBreakdown::Enabled();
	const auto*    steps_ptr  = trace.steps.data();
	const auto*    ops_ptr    = tape.ops.data();
	uint64_t       n_steps = 0, n_reads = 0, n_memo = 0;
	if (reuse) published.reserve(tape.ops.size());
	for (uint32_t pc = 0;;) {
		const auto& step = reuse ? steps_ptr[pc] : trace.steps[pc];
#if KYTY_SRT_PROFILE
		{
			int slot = SrtProf::StepEnd;
			switch (step.kind) {
				case TraceKind::Exec: {
					const auto k = tape.ops[step.a].kind;
					slot = k == ResourcePlan::SrtTape::Kind::RawRead ? SrtProf::StepRead
					       : k == ResourcePlan::SrtTape::Kind::Inst ? SrtProf::StepAlu
					       : k == ResourcePlan::SrtTape::Kind::Immediate ? SrtProf::StepImm : SrtProf::StepUserData;
					break;
				}
				case TraceKind::Memo: slot = SrtProf::StepMemo; break;
				case TraceKind::Flat: slot = SrtProf::StepFlat; break;
				case TraceKind::Activate: slot = SrtProf::StepActivate; break;
				case TraceKind::Cond:
				case TraceKind::CondTri: slot = SrtProf::StepCond; break;
				case TraceKind::Gather: slot = SrtProf::StepGather; break;
				default: break;
			}
			SRT_PROF_TICK(prof, slot);
		}
#endif
		if (stats) ++n_steps;
		switch (step.kind) {
			case TraceKind::Exec: {
				// Recorded as executed: the DFS saw no memo (value, in-progress or failure) here.
				const auto& op = reuse ? ops_ptr[step.a] : tape.ops[step.a];
				if (stats && op.kind == ResourcePlan::SrtTape::Kind::RawRead) ++n_reads;
				if (op.memo_index != UINT32_MAX) {
					const auto& context = context_for(op);
					if (op.memo_index < context.values.size() &&
					    (context.values[op.memo_index].generation | 3u) == (context.generation | 3u)) {
						miss_why = 7;
						return miss();
					}
				}
				if (!ExecuteTapeOp(op, values, step.a)) { miss_why = 8; return miss(); }
				if (op.memo_index != UINT32_MAX) { published.emplace_back(step.a, 0u); ++n_memo; }
				++pc;
				break;
			}
			case TraceKind::Memo: {
				const auto& op = tape.ops[step.a];
				const auto& context = op.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
				if (op.memo_index >= context.values.size() ||
				    context.values[op.memo_index].generation != context.generation) {
					miss_why = 9;
					return miss();
				}
				values[step.a] = context.values[op.memo_index].value;
				++n_memo;
				++pc;
				break;
			}
			case TraceKind::Flat:
				flat[step.b] = static_cast<uint32_t>(values[step.a]);
				++pc;
				break;
			case TraceKind::Activate:
				for (const auto source: m_program.control_flow[step.a].sources) active[source] = 1u;
				++pc;
				break;
			case TraceKind::Cond: {
				const auto next = step.next[static_cast<uint32_t>(values[step.a]) != 0u ? 1 : 0];
				if (next == UINT32_MAX) { miss_why = 10; return miss(); }
				pc = next;
				break;
			}
			case TraceKind::CondTri: {
				const auto value = static_cast<uint32_t>(values[step.a]);
				const auto next = step.next[value == 2u ? 2 : value != 0u ? 1 : 0];
				if (next == UINT32_MAX) { miss_why = 10; return miss(); }
				pc = next;
				break;
			}
			case TraceKind::Eval: {
				const auto& read = m_program.srt_reads[step.a];
				const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
				                   m_clean_flat_slots[read.flat_offset] != 0u;
				if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
				{ miss_why = 11; return miss(); }
				auto& evaluator = clean ? *m_clean_evaluator : *this;
				if (!evaluator.Evaluate(read.value, flat[step.b])) { miss_why = 11; return miss(); }
				++pc;
				break;
			}
			case TraceKind::Gather:
				// The DFS proved the source's flat slots were refreshed on this path.
				if (m_cfg_gather_enabled) {
					m_cfg_gather_ready_sources[step.a] = 1u;
					if (m_clean_evaluator != nullptr) m_clean_evaluator->m_cfg_gather_ready_sources[step.a] = 1u;
				}
				++pc;
				break;
			case TraceKind::CondEval: {
				auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
				uint32_t condition = 0;
				const bool ok = predicate.Evaluate(m_program.control_flow[step.a].condition, condition);
				const auto next = step.next[!ok ? 2 : condition != 0u ? 1 : 0];
				if (next == UINT32_MAX) { miss_why = 12; return miss(); }
				pc = next;
				break;
			}
			case TraceKind::End:
				SRT_PROF_FINISH(prof);
				++trace.hits;
				if (stats) {
					auto& fc = FrameBreakdown::g_counters;
					fc.mat_hit_steps.fetch_add(n_steps, std::memory_order_relaxed);
					fc.mat_hit_reads.fetch_add(n_reads, std::memory_order_relaxed);
					fc.mat_hit_memo.fetch_add(n_memo, std::memory_order_relaxed);
				}
				g_tape_stats.cfg_trace_hits.fetch_add(1, std::memory_order_relaxed);
				m_context.refreshed_flat = &flat;
				if (m_clean_evaluator != nullptr) m_clean_evaluator->m_context.refreshed_flat = &flat;
				return 1;
		}
	}
}

void SrtWalker::MergeCfgTrace() {
	using TraceKind = ResourcePlan::CfgTrace::Kind;
	// KYTY_SRT_SOURCE_REUSE: allow many more observed control-flow paths per trie (the 32-path cap
	// left programs with varying branch outcomes permanently missing).
	const size_t   kMaxSteps = m_source_reuse_enabled ? size_t {1} << 19 : size_t {1} << 16;
	const uint32_t kMaxPaths = m_source_reuse_enabled ? 512u : 32u;
	auto& trace = m_program.cfg_traces[m_clean_evaluator != nullptr ? 1 : 0];
	const auto& recorded = m_program.cfg_trace_scratch;
	if (trace.disabled || recorded.empty() || recorded.back().kind != TraceKind::End) return;
	const auto append = [&](size_t first) {
		const auto base = static_cast<uint32_t>(trace.steps.size());
		for (size_t j = first; j < recorded.size(); ++j) {
			auto step = recorded[j];
			if (step.kind == TraceKind::Cond || step.kind == TraceKind::CondEval ||
			    step.kind == TraceKind::CondTri) {
				step.next = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
				step.next[step.b] = base + static_cast<uint32_t>(j + 1 - first);
			}
			trace.steps.push_back(step);
		}
		++trace.paths;
		g_tape_stats.cfg_trace_recorded.fetch_add(1, std::memory_order_relaxed);
	};
	const bool strict = m_runtime.read_specialization_memory != nullptr;
	if (trace.steps.empty()) {
		trace.strict = strict;
		if (recorded.size() <= kMaxSteps) append(0);
		return;
	}
	if (trace.strict != strict) return;
	for (uint32_t pc = 0, i = 0; i < recorded.size(); ++i) {
		auto& step = trace.steps[pc];
		const auto& seen = recorded[i];
		// Paths share a prefix up to their first differing condition; anything else
		// means the refresh did not start from equivalent memo state. Keep the trie.
		const bool branch = seen.kind == TraceKind::Cond || seen.kind == TraceKind::CondEval ||
		                    seen.kind == TraceKind::CondTri;
		if (step.kind != seen.kind || step.a != seen.a || (!branch && step.b != seen.b)) return;
		if (seen.kind == TraceKind::End) return;
		if (!branch) { ++pc; continue; }
		if (step.next[seen.b] != UINT32_MAX) { pc = step.next[seen.b]; continue; }
		if (trace.paths >= kMaxPaths || trace.steps.size() + recorded.size() - i > kMaxSteps) return;
		step.next[seen.b] = static_cast<uint32_t>(trace.steps.size());
		append(i + 1);
		return;
	}
}

int SrtWalker::RunTape(std::vector<uint32_t>& flat) {
	static const bool tape_enabled = [] {
		const char* val = std::getenv("KYTY_SRT_TAPE");
		return val == nullptr || val[0] != '0';
	}();
	if (!tape_enabled || !m_active_mask.IsEmpty() ||
	    (m_clean_evaluator != nullptr && !m_clean_evaluator->m_active_mask.IsEmpty())) {
		g_tape_stats.walker.fetch_add(1, std::memory_order_relaxed);
		return -1;
	}

	const size_t tape_idx = (m_clean_evaluator != nullptr) ? 1 : 0;
	auto& tape = m_program.srt_tapes[tape_idx];

	if (tape.state == ResourcePlan::SrtTape::State::Unbuilt) {
		size_t reject_reason = 0;
		if (BuildTapeForPlan(m_program, m_clean_evaluator != nullptr, m_clean_flat_slots, tape, reject_reason)) {
			tape.state = ResourcePlan::SrtTape::State::Ready;
			g_tape_stats.built.fetch_add(1, std::memory_order_relaxed);
		} else {
			tape.state = ResourcePlan::SrtTape::State::Unsupported;
			g_tape_stats.unsupported.fetch_add(1, std::memory_order_relaxed);
			if (reject_reason < 8) {
				g_tape_stats.reject[reject_reason].fetch_add(1, std::memory_order_relaxed);
			}
		}
	}

	if (tape.state != ResourcePlan::SrtTape::State::Ready) {
		g_tape_stats.walker.fetch_add(1, std::memory_order_relaxed);
		return -1;
	}

	flat.resize(m_program.srt_reads.size());
	auto& vals = m_program.srt_tape_values;
	vals.resize(tape.ops.size());

	// Batched per-op counters: same totals as the old per-op increments, one atomic add per run.
	struct TapeOpTally {
		SrtWalker& walker;
		uint64_t   executed = 0;
		~TapeOpTally() {
			if (executed == 0u) return;
			if (walker.m_runtime.evaluation_counts) walker.m_runtime.evaluation_counts->tape_ops += executed;
			if (FrameBreakdown::Enabled()) {
				auto& fc = FrameBreakdown::g_counters;
				fc.mat_flat_ops.fetch_add(executed, std::memory_order_relaxed);
				if (walker.m_program.control_flow.empty()) {
					fc.mat_flat_nocfg_ops.fetch_add(executed, std::memory_order_relaxed);
				}
			}
		}
	} tally {*this};
	for (size_t i = 0; i < tape.ops.size();) {
		if (m_scalar_batch_enabled &&
		    tape.ops[i].kind == ResourcePlan::SrtTape::Kind::RawRead &&
		    tape.ops[i].opcode == ValueOpcode::LoadAddressU32) {
			std::array<Value, 16> candidates {};
			std::array<uint32_t, 16> words {};
			size_t count = 0;
			const auto& first = tape.ops[i];
			for (; count < candidates.size() && i + count < tape.ops.size(); ++count) {
				const auto& op = tape.ops[i + count];
				if (op.kind != ResourcePlan::SrtTape::Kind::RawRead || op.opcode != first.opcode ||
				    op.clean != first.clean || op.args != first.args || op.source == nullptr) break;
				candidates[count] = Value(const_cast<Inst*>(op.source));
			}
			const auto batched = TryScalarReadBatch(std::span(candidates).first(count), first.clean != 0, words);
			if (batched != 0) {
				for (size_t word = 0; word < batched; ++word) vals[i + word] = words[word];
				i += batched;
				continue;
			}
		}
		++tally.executed;
		if (!ExecuteTapeOp(tape.ops[i], vals, static_cast<uint32_t>(i))) return 0;
		++i;
	}
	for (size_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
		const auto& read = m_program.srt_reads[slot];
		if (read.flat_offset >= flat.size()) {
			g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
			return 0;
		}
		flat[read.flat_offset] = static_cast<uint32_t>(vals[tape.slot_ops[slot]]);
	}

	m_context.refreshed_flat = &flat;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->m_context.refreshed_flat = &flat;
	}
	g_tape_stats.runs.fetch_add(1, std::memory_order_relaxed);
	g_tape_stats.ops.fetch_add(tape.ops.size(), std::memory_order_relaxed);
	return 1;
}


namespace {
inline float TapeFloat32(uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); }
// Shared by the tape executor and the compiled executor (KYTY_SRT_COMPILED): one definition of
// every ALU operation so the two paths cannot diverge.
inline bool TapeAlu(ValueOpcode opcode, const std::array<uint32_t, 5>& args, uint64_t imm,
                    const uint64_t* values, uint64_t& result) {
	const uint64_t a = values[args[0]];
	const uint64_t b = values[args[1]];
	const uint64_t c = values[args[2]];
	switch (opcode) {
		case ValueOpcode::ReadConst: result = a; break; // CFG scalar aliases
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: result = a; break;
		case ValueOpcode::ConvertF32U32:
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			break;
		case ValueOpcode::ConvertU32F32: {
			result = SaturateF32ToU32(TapeFloat32(a));
			break;
		}
		case ValueOpcode::FPTrunc32:
			result = std::bit_cast<uint32_t>(std::trunc(TapeFloat32(a)));
			break;
		case ValueOpcode::FPRecipIFlag32: {
			const auto exp = (a >> 23u) & 0xffu;
			if ((a & 0x807fffffu) != 0u || exp == 0u || exp >= 254u) {
				g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			result = (254u - exp) << 23u;
			break;
		}
		case ValueOpcode::FPIsNan32: result = std::isnan(TapeFloat32(a)); break;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); break;
		case ValueOpcode::LogicalNot: result = a == 0u; break;
		case ValueOpcode::CompositeConstructU64:
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			break;
		case ValueOpcode::CompositeExtractU64:
			result = static_cast<uint32_t>(a >> (imm * 32u));
			break;
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); break;
		case ValueOpcode::IAdd64: result = a + b; break;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); break;
		case ValueOpcode::ISub64: result = a - b; break;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); break;
		case ValueOpcode::UMulHi:
			result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			break;
		case ValueOpcode::UMin32:
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			break;
		case ValueOpcode::FPMul32:
			result = std::bit_cast<uint32_t>(TapeFloat32(a) * TapeFloat32(b));
			break;
		case ValueOpcode::FPOrdLessThanEqual32:
			result = TapeFloat32(a) <= TapeFloat32(b);
			break;
		case ValueOpcode::FPOrdGreaterThanEqual32:
			result = TapeFloat32(a) >= TapeFloat32(b);
			break;
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); break;
		case ValueOpcode::BitwiseAnd64: result = a & b; break;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); break;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); break;
		case ValueOpcode::ShiftLeftLogical32:
			result = static_cast<uint32_t>(a) << (b & 31u);
			break;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); break;
		case ValueOpcode::ShiftRightLogical32:
			result = static_cast<uint32_t>(a) >> (b & 31u);
			break;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); break;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(
			    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			break;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			break;
		case ValueOpcode::IEqual32:
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			break;
		case ValueOpcode::INotEqual32:
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			break;
		case ValueOpcode::ULessThan32:
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			break;
		case ValueOpcode::ULessThanEqual32:
			result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
			break;
		case ValueOpcode::UGreaterThan32:
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			break;
		case ValueOpcode::SGreaterThanEqual32:
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			break;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); break;
		case ValueOpcode::SelectU32: result = a != 0u ? b : c; break; // synthetic tri-state conditions
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			const auto mask = width == 32u ? UINT32_MAX : (width == 0u ? 0u : (1u << width) - 1u);
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			break;
		}
		case ValueOpcode::BitFieldSExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (width == 0u) {
				result = 0;
			} else {
				const auto mask = width == 32u ? UINT32_MAX : (1u << width) - 1u;
				auto bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (1u << (width - 1u))) != 0u) bits |= ~mask;
				result = bits;
			}
			break;
		}
		case ValueOpcode::BitFieldInsert: {
			const uint64_t d = values[args[3]];
			const auto offset = static_cast<uint32_t>(c);
			const auto width = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
			} else {
				const auto mask = width == 32u ? UINT32_MAX : ((1u << width) - 1u) << offset;
				result = (static_cast<uint32_t>(a) & ~mask) |
				         ((static_cast<uint32_t>(b) << offset) & mask);
			}
			break;
		}
		default:
			g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
			return false;
	}

	return true;
}
} // namespace

bool SrtWalker::ExecuteTapeOp(const ResourcePlan::SrtTape::Op& op,
                              std::vector<uint64_t>& values, uint32_t index, bool publish) {
	// Per-op counters are tallied once per RunTape (RunTape::TapeOpTally).
	uint64_t result = 0;
	switch (op.kind) {
		case ResourcePlan::SrtTape::Kind::Immediate:
			result = op.imm;
			break;
		case ResourcePlan::SrtTape::Kind::UserData: {
			const uint32_t reg = op.args[0];
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			const auto user_data_index = reg - m_program.user_data_base;
			if (m_runtime.observe_user_data != nullptr) {
				m_runtime.observe_user_data(m_runtime.user_data_observer_userdata, user_data_index);
			}
			result = m_runtime.user_data[user_data_index];
			break;
		}
		case ResourcePlan::SrtTape::Kind::ShaderBase:
			result = m_runtime.shader_base;
			break;
		case ResourcePlan::SrtTape::Kind::RawRead: {
			const auto& mem = m_program.memory_info[op.aux];
			const uint64_t low = values[op.args[0]];
			const uint64_t high = values[op.args[1]];
			const uint64_t offset = values[op.args[2]];
			const auto base = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
			const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
			uint64_t address = 0;

			if (op.opcode == ValueOpcode::ReadConstBuffer || op.opcode == ValueOpcode::LoadBufferU32) {
				const uint64_t records = values[op.args[3]];
				const uint64_t word3 = values[op.args[4]];
				if (immediate < 0) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return false;
				}
				const auto byte_offset =
				    (static_cast<uint64_t>(immediate) & ~uint64_t{3}) + (static_cast<uint32_t>(offset) & ~3u);
				const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
				if (op.opcode == ValueOpcode::LoadBufferU32) {
					if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return false;
					}
					if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
						result = 0;
						break;
					}
				}
				const auto size = stride == 0u
				                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
				                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
				if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
					// PS5 scalar-buffer loads return zero per out-of-range DWORD.
					result = 0;
					break;
				}
				address = (base & ~uint64_t{3}) + byte_offset;
			} else {
				const auto relative = (immediate & ~int64_t{3}) +
				                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
				if (!AddSignedAddress(base & ~uint64_t{3}, relative, address)) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return false;
				}
			}

			if (op.clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
				g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			uint32_t word = 0;
			const bool is_vector = op.opcode == ValueOpcode::LoadBufferU32;
			const auto reader = (is_vector || op.clean) ? m_runtime.read_specialization_memory
			                                            : m_runtime.read_memory;
			SRT_PROF_COUNT(reader_calls, 1);
			SRT_PROF_COUNT(reader_words, 1);
			SRT_PROF_LOG(address, 1, is_vector || op.clean);
			if (reader != nullptr) {
				SRT_PROF_SCOPE(SrtProf::Reader);
				if (FrameBreakdown::Enabled()) {
					auto& fc = FrameBreakdown::g_counters;
					fc.mat_flat_reads.fetch_add(1, std::memory_order_relaxed);
					FrameBreakdown::Timer read_timer(fc.mat_flat_read_ns, FrameBreakdown::DetailEnabled());
					if (!reader(m_runtime.userdata, address, {&word, 1})) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return false;
					}
				} else if (!reader(m_runtime.userdata, address, {&word, 1})) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return false;
				}
			} else {
				if (is_vector) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return false;
				}
				std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
			}
			result = word;
			break;
		}
		case ResourcePlan::SrtTape::Kind::Inst:
			if (!TapeAlu(op.opcode, op.args, op.imm, values.data(), result)) return false;
			break;
	}

	values[index] = result;
	if (!publish) {
		++m_memo_skipped;
		return true;
	}
	SRT_PROF_SCOPE(SrtProf::MemoWrite);
	// A clean op belongs to the clean walker context only. The same IR instruction
	// may have a separate ordinary op with a different memory-reader domain.
	if (op.memo_index != UINT32_MAX && op.clean && m_clean_evaluator != nullptr) {
		auto& clean_context = m_clean_evaluator->m_context;
		if (op.memo_index >= clean_context.values.size()) {
			clean_context.values.resize(m_program.evaluation_value_count);
		}
		clean_context.values[op.memo_index].value = result;
		clean_context.values[op.memo_index].generation = clean_context.generation;
	} else if (op.memo_index != UINT32_MAX) {
		if (op.memo_index >= m_context.values.size()) {
			m_context.values.resize(m_program.evaluation_value_count);
		}
		m_context.values[op.memo_index].value = result;
		m_context.values[op.memo_index].generation = m_context.generation;
	}
	return true;
}
// --- KYTY_SRT_COMPILED -----------------------------------------------------------------------
// A compiled plan is the straight-line tape (or the recorded CFG trace trie) lowered to a flat step
// array. It is not a cache of results: every call re-reads every user-data register and every guest
// dword, recomputes every pointer and re-tests every branch condition. Adjacent scalar loads from
// one pointer window are fetched with a single reader call when their addresses are contiguous, and
// are retried as the original single reads whenever the reader declines the merged request.

namespace {

struct GlobalCompiledStats {
	std::atomic<uint64_t> calls {0}, compiled {0}, failed {0}, nocfg {0}, cfg {0}, builds {0}, steps {0};
	std::atomic<uint64_t> reads {0}, read_words {0}, groups_hit {0}, groups_miss {0};
	std::atomic<uint64_t> spec_recomputed {0}, spec_reused {0}, verify_checks {0}, verify_mismatch {0};
	std::atomic<uint64_t> desc_reuse {0}, desc_eval {0}, memo_skipped {0};
	std::atomic<uint64_t> mask_calls {0}, mask_first {0}, mask_images {0}, mask_samplers {0}, mask_buffer_fmt {0};
	std::atomic<uint64_t> mask_buffer_addr {0}, mask_buffer_size {0}, mask_flat {0}, mask_user_data {0}, mask_all_same {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(SrtCompiledFallback::Count)> fallback {};
} g_compiled_stats;

constexpr uint32_t kMaxCompiledGroup = 16;

bool SrtTapeEnabled() {
	static const bool enabled = [] {
		const char* val = std::getenv("KYTY_SRT_TAPE");
		return val == nullptr || val[0] != '0';
	}();
	return enabled;
}

void CountCompiledFallback(SrtCompiledFallback reason) {
	g_compiled_stats.fallback[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

using CompiledSrt = ResourcePlan::CompiledSrt;

// Immediates are folded into the constant table; every other tape op becomes one step.
bool LowerTapeOp(const ResourcePlan::SrtTape::Op& op, uint32_t index, CompiledSrt::Step& step) {
	using Kind  = ResourcePlan::SrtTape::Kind;
	step        = {};
	step.out    = index;
	step.memo   = op.memo_index;
	step.clean  = op.clean;
	step.opcode = op.opcode;
	step.args   = op.args;
	step.imm    = op.imm;
	switch (op.kind) {
		case Kind::Immediate: return false;
		case Kind::UserData:
			step.code = CompiledSrt::Code::UserData;
			step.aux  = op.args[0];
			return true;
		case Kind::ShaderBase: step.code = CompiledSrt::Code::ShaderBase; return true;
		case Kind::RawRead:
			step.code = CompiledSrt::Code::Read;
			step.aux  = op.aux;
			return true;
		case Kind::Inst: step.code = CompiledSrt::Code::Alu; return true;
	}
	return false;
}

void FoldConstants(const ResourcePlan::SrtTape& tape, CompiledSrt& program) {
	program.consts.clear();
	for (uint32_t i = 0; i < tape.ops.size(); ++i) {
		if (tape.ops[i].kind == ResourcePlan::SrtTape::Kind::Immediate) {
			program.consts.emplace_back(i, tape.ops[i].imm);
		}
	}
	program.value_count = static_cast<uint32_t>(tape.ops.size());
}

// Marks runs of adjacent scalar loads that share one pointer (same low/high operands, domain and
// opcode). Only the first step of a run carries the count; the following steps stay valid single
// reads, which is what executes when the merged request is declined.
void FormReadGroups(CompiledSrt& program) {
	auto& steps         = program.steps;
	program.read_steps  = 0;
	program.group_steps = 0;
	for (size_t i = 0; i < steps.size(); ++i) {
		if (steps[i].code != CompiledSrt::Code::Read) continue;
		++program.read_steps;
		if (steps[i].opcode != ValueOpcode::LoadAddressU32) continue;
		size_t count = 1;
		while (count < kMaxCompiledGroup && i + count < steps.size()) {
			const auto& head = steps[i];
			const auto& next = steps[i + count];
			if (next.code != CompiledSrt::Code::Read || next.opcode != head.opcode ||
			    next.clean != head.clean || next.args[0] != head.args[0] || next.args[1] != head.args[1])
				break;
			++count;
		}
		if (count > 1) {
			steps[i].group = static_cast<uint8_t>(count);
			++program.group_steps;
			program.read_steps += static_cast<uint32_t>(count - 1);
			i += count - 1;
		}
	}
}

// --- dead-memo elimination ----------------------------------------------------------------------
// A compiled step publishes its result into the evaluator memo (value + generation) so that later
// recursive evaluation and the CFG trace-miss rollback can see it. Most published slots are never read:
// the only code that can consult the memo after (or during) a compiled run is
//   * recursive Evaluate / EvaluateDescriptor, whose roots are fixed by the plan: the descriptor source
//     dwords, the indirect-descriptor expressions and the uniform-fill values;
//   * the CFG descriptor gather check, which reads the memo of the srt_read value behind every Flat
//     gather word (those values are reached from the source dwords, so they are roots too);
//   * the compiled Memo steps themselves (kept unconditionally).
// A recursive evaluation stops at the first memoized node, so it can reach a slot only through nodes
// that are not published on every path. The analysis therefore walks the value graph from the roots
// and stops at nodes that every path through the program publishes (in the same evaluator context);
// everything visited that the program publishes is kept, the rest is dead. The trace-miss rollback only
// resets slots that were published, and a slot that was never published carries an older generation,
// which the recording DFS treats exactly like the rolled-back state, so skipping publication is safe.
// Any construct the walk cannot model (Phi, ReadFirstLane, an oversized graph, a branch that does not
// move forward) keeps every publication.
using MemoKey = uintptr_t;
inline MemoKey MakeMemoKey(const Inst* inst, uint32_t ctx) {
	return (reinterpret_cast<uintptr_t>(inst) << 1u) | ctx;
}

void ClassifyMemoPublication(const ResourcePlan& plan, const ResourcePlan::SrtTape& tape, CompiledSrt& program) {
	using Code = CompiledSrt::Code;
	auto& steps = program.steps;
	program.dead_memo_steps = 0;
	if (std::getenv("KYTY_SRT_KEEP_MEMO") != nullptr) return; // diagnostic: publish everything
	const uint32_t contexts = tape.clean_root ? 2u : 1u;
	const size_t   count    = steps.size();
	std::unordered_map<MemoKey, uint32_t> key_index;
	std::vector<int32_t> step_key(count, -1);
	std::vector<MemoKey> keys;
	const auto key_of = [&](const CompiledSrt::Step& step) -> MemoKey {
		const auto& op = tape.ops[step.out];
		return MakeMemoKey(op.source, tape.clean_root && step.clean ? 1u : 0u);
	};
	for (size_t pc = 0; pc < count; ++pc) {
		const auto& step = steps[pc];
		if (step.code != Code::UserData && step.code != Code::ShaderBase && step.code != Code::Alu &&
		    step.code != Code::Read)
			continue;
		if (step.memo == UINT32_MAX || step.out >= tape.ops.size() || tape.ops[step.out].source == nullptr) continue;
		const auto key = key_of(step);
		const auto [it, added] = key_index.emplace(key, static_cast<uint32_t>(keys.size()));
		if (added) keys.push_back(key);
		step_key[pc] = static_cast<int32_t>(it->second);
	}
	if (keys.empty()) return;
	// Keys published on every path from each step to End.
	const size_t words = (keys.size() + 63u) / 64u;
	std::vector<uint64_t> must(count * words, 0u);
	for (size_t pc = count; pc-- > 0;) {
		const auto& step = steps[pc];
		uint64_t*   cur  = &must[pc * words];
		if (step.code == Code::End) continue;
		if (step.code == Code::Cond || step.code == Code::CondTri) {
			bool first = true;
			for (const auto target: step.next) {
				if (target == UINT32_MAX) continue;
				if (target <= pc || target >= count) return; // not a forward trie edge: keep everything
				const uint64_t* succ = &must[static_cast<size_t>(target) * words];
				for (size_t w = 0; w < words; ++w) cur[w] = first ? succ[w] : (cur[w] & succ[w]);
				first = false;
			}
			continue;
		}
		if (pc + 1 >= count) return;
		const uint64_t* succ = &must[(pc + 1) * words];
		for (size_t w = 0; w < words; ++w) cur[w] = succ[w];
		if (step_key[pc] >= 0) cur[static_cast<size_t>(step_key[pc]) / 64u] |= uint64_t {1} << (step_key[pc] % 64u);
	}
	const uint64_t* all = must.data();
	const auto      published_on_every_path = [&](uint32_t index) { return ((all[index / 64u] >> (index % 64u)) & 1u) != 0u; };

	std::unordered_set<MemoKey> visited, needed;
	std::vector<std::pair<const Inst*, uint32_t>> stack;
	const auto push = [&](Value value) {
		value = value.Resolve();
		if (value.IsEmpty() || value.IsImmediate()) return;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return;
		for (uint32_t ctx = 0; ctx < contexts; ++ctx) stack.emplace_back(inst, ctx);
	};
	for (const auto& source: plan.descriptor_sources) {
		for (uint32_t i = 0; i < source.dword_count && i < source.dwords.size(); ++i) push(source.dwords[i]);
		if (source.indirect_descriptor.has_value()) {
			push(source.indirect_descriptor->key_count);
			push(source.indirect_descriptor->selector_first);
			push(source.indirect_descriptor->selector_mask);
		}
	}
	for (const auto& value: plan.uniform_fill.values) push(value);
	for (const auto& gather: plan.descriptor_gathers) {
		if (!gather.valid || !gather.needs_flat) continue;
		for (const auto& word: gather.words) {
			if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::Flat && word.read_slot < plan.srt_reads.size())
				push(plan.srt_reads[word.read_slot].value);
		}
	}
	for (size_t pc = 0; pc < count; ++pc) {
		const auto& step = steps[pc];
		if (step.code == Code::Memo && step.out < tape.ops.size() && tape.ops[step.out].source != nullptr)
			needed.insert(key_of(step));
	}
	constexpr size_t kMaxVisited = 1u << 16;
	while (!stack.empty()) {
		const auto [inst, ctx] = stack.back();
		stack.pop_back();
		const auto key = MakeMemoKey(inst, ctx);
		if (!visited.insert(key).second) continue;
		if (visited.size() > kMaxVisited) return;
		if (const auto found = key_index.find(key); found != key_index.end()) {
			needed.insert(key);
			if (published_on_every_path(found->second)) continue;
		}
		const auto opcode = inst->GetOpcode();
		if (opcode == ValueOpcode::Phi || opcode == ValueOpcode::ReadFirstLane) return;
		if (opcode == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < plan.srt_reads.size())
				push(plan.srt_reads[slot.U32()].value);
		}
		for (size_t i = 0; i < inst->NumArgs(); ++i) push(inst->Arg(i));
	}
	for (size_t pc = 0; pc < count; ++pc) {
		if (step_key[pc] < 0) continue;
		const bool keep = needed.count(keys[static_cast<size_t>(step_key[pc])]) != 0u;
		steps[pc].publish = keep ? 1u : 0u;
		program.dead_memo_steps += keep ? 0u : 1u;
	}
}

bool CompileNocfg(const ResourcePlan& plan, const ResourcePlan::SrtTape& tape, CompiledSrt& program) {
	program.steps.clear();
	program.reason = 0;
	FoldConstants(tape, program);
	program.steps.reserve(tape.ops.size() + plan.srt_reads.size() + 1);
	CompiledSrt::Step step;
	for (uint32_t i = 0; i < tape.ops.size(); ++i) {
		if (LowerTapeOp(tape.ops[i], i, step)) program.steps.push_back(step);
	}
	if (tape.slot_ops.size() != plan.srt_reads.size()) return false;
	for (size_t slot = 0; slot < plan.srt_reads.size(); ++slot) {
		if (plan.srt_reads[slot].flat_offset >= plan.srt_reads.size() || tape.slot_ops[slot] >= tape.ops.size())
			return false;
		step      = {};
		step.code = CompiledSrt::Code::Flat;
		step.out  = tape.slot_ops[slot];
		step.aux  = plan.srt_reads[slot].flat_offset;
		program.steps.push_back(step);
	}
	step      = {};
	step.code = CompiledSrt::Code::End;
	program.steps.push_back(step);
	FormReadGroups(program);
	ClassifyMemoPublication(plan, tape, program);
	return true;
}

// Lowers the recorded trie. Eval/CondEval steps re-run the recursive walker and cannot be expressed
// (reason 1). Branch targets are remapped through `landing` because folded immediates disappear.
bool CompileTrace(const ResourcePlan& plan, const ResourcePlan::SrtTape& tape, const ResourcePlan::CfgTrace& trace,
                  CompiledSrt& program) {
	using TraceKind = ResourcePlan::CfgTrace::Kind;
	program.steps.clear();
	program.reason = 0;
	FoldConstants(tape, program);
	const auto&                    src = trace.steps;
	std::vector<CompiledSrt::Step> lowered(src.size());
	std::vector<uint8_t>           kept(src.size(), 0u);
	for (size_t pc = 0; pc < src.size(); ++pc) {
		const auto& in   = src[pc];
		auto&       step = lowered[pc];
		bool        keep = true;
		switch (in.kind) {
			case TraceKind::Exec:
				if (in.a >= tape.ops.size()) return false;
				keep = LowerTapeOp(tape.ops[in.a], in.a, step);
				break;
			case TraceKind::Memo:
				if (in.a >= tape.ops.size()) return false;
				step.code  = CompiledSrt::Code::Memo;
				step.out   = in.a;
				step.memo  = tape.ops[in.a].memo_index;
				step.clean = tape.ops[in.a].clean;
				break;
			case TraceKind::Flat:
				if (in.a >= tape.ops.size()) return false;
				step.code = CompiledSrt::Code::Flat;
				step.out  = in.a;
				step.aux  = in.b;
				break;
			case TraceKind::Activate:
				step.code = CompiledSrt::Code::Activate;
				step.aux  = in.a;
				break;
			case TraceKind::Cond:
			case TraceKind::CondTri:
				if (in.a >= tape.ops.size()) return false;
				step.code = in.kind == TraceKind::Cond ? CompiledSrt::Code::Cond : CompiledSrt::Code::CondTri;
				step.out  = in.a;
				break;
			case TraceKind::Gather:
				step.code = CompiledSrt::Code::Gather;
				step.aux  = in.a;
				break;
			case TraceKind::End: step.code = CompiledSrt::Code::End; break;
			case TraceKind::Eval:
			case TraceKind::CondEval: program.reason = 1; return false;
		}
		kept[pc] = keep ? 1u : 0u;
	}
	// Flat copies and activity marks only consume finished values, so within a straight-line segment
	// they can be deferred to its end. That makes the loads of one descriptor table adjacent steps,
	// which FormReadGroups then fetches with one reader call per contiguous run. Segments end at
	// branches, Gather (its flat slots must be written before it runs) and End, which never move.
	{
		std::vector<uint32_t> order(src.size());
		for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<uint32_t>(i);
		size_t seg = 0;
		for (size_t pc = 0; pc < src.size(); ++pc) {
			const auto kind = src[pc].kind;
			if (kind != TraceKind::Cond && kind != TraceKind::CondTri && kind != TraceKind::Gather &&
			    kind != TraceKind::End)
				continue;
			std::stable_partition(order.begin() + static_cast<ptrdiff_t>(seg), order.begin() + static_cast<ptrdiff_t>(pc),
			                      [&](uint32_t i) { return src[i].kind != TraceKind::Flat && src[i].kind != TraceKind::Activate; });
			seg = pc + 1;
		}
		std::vector<CompiledSrt::Step> lowered_ordered(src.size());
		std::vector<uint8_t>           kept_ordered(src.size());
		for (size_t pc = 0; pc < src.size(); ++pc) {
			lowered_ordered[pc] = lowered[order[pc]];
			kept_ordered[pc]    = kept[order[pc]];
		}
		lowered = std::move(lowered_ordered);
		kept    = std::move(kept_ordered);
	}
	std::vector<uint32_t> landing(src.size(), UINT32_MAX);
	uint32_t              kept_count = 0;
	for (size_t pc = 0; pc < src.size(); ++pc) kept_count += kept[pc];
	uint32_t next_landing = kept_count;
	uint32_t index        = kept_count;
	for (size_t pc = src.size(); pc-- > 0;) {
		if (kept[pc]) next_landing = --index;
		landing[pc] = next_landing;
	}
	std::vector<CompiledSrt::Step> out;
	out.reserve(kept_count);
	for (size_t pc = 0; pc < src.size(); ++pc) {
		if (!kept[pc]) continue;
		auto step = lowered[pc];
		if (src[pc].kind == TraceKind::Cond || src[pc].kind == TraceKind::CondTri) {
			for (size_t k = 0; k < 3; ++k) {
				const auto target = src[pc].next[k];
				if (target == UINT32_MAX) continue;
				if (target >= src.size() || landing[target] >= kept_count) return false;
				step.next[k] = landing[target];
			}
		}
		out.push_back(step);
	}
	if (out.empty() || out.back().code != CompiledSrt::Code::End) return false;
	program.steps = std::move(out);
	FormReadGroups(program);
	ClassifyMemoPublication(plan, tape, program);
	program.verified_gather = {};
	return true;
}

} // namespace

SrtCompiledStats GetSrtCompiledStats() {
	SrtCompiledStats s;
	const auto       load = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
	s.calls               = load(g_compiled_stats.calls);
	s.compiled            = load(g_compiled_stats.compiled);
	s.failed              = load(g_compiled_stats.failed);
	s.nocfg               = load(g_compiled_stats.nocfg);
	s.cfg                 = load(g_compiled_stats.cfg);
	s.builds              = load(g_compiled_stats.builds);
	s.steps               = load(g_compiled_stats.steps);
	s.reads               = load(g_compiled_stats.reads);
	s.read_words          = load(g_compiled_stats.read_words);
	s.groups_hit          = load(g_compiled_stats.groups_hit);
	s.groups_miss         = load(g_compiled_stats.groups_miss);
	s.spec_recomputed     = load(g_compiled_stats.spec_recomputed);
	s.spec_reused         = load(g_compiled_stats.spec_reused);
	s.verify_checks       = load(g_compiled_stats.verify_checks);
	s.verify_mismatch     = load(g_compiled_stats.verify_mismatch);
	s.desc_reuse          = load(g_compiled_stats.desc_reuse);
	s.desc_eval           = load(g_compiled_stats.desc_eval);
	s.memo_skipped        = load(g_compiled_stats.memo_skipped);
	s.mask_calls          = load(g_compiled_stats.mask_calls);
	s.mask_first          = load(g_compiled_stats.mask_first);
	s.mask_images         = load(g_compiled_stats.mask_images);
	s.mask_samplers       = load(g_compiled_stats.mask_samplers);
	s.mask_buffer_fmt     = load(g_compiled_stats.mask_buffer_fmt);
	s.mask_buffer_addr    = load(g_compiled_stats.mask_buffer_addr);
	s.mask_buffer_size    = load(g_compiled_stats.mask_buffer_size);
	s.mask_flat           = load(g_compiled_stats.mask_flat);
	s.mask_user_data      = load(g_compiled_stats.mask_user_data);
	s.mask_all_same       = load(g_compiled_stats.mask_all_same);
	for (size_t i = 0; i < s.fallback.size(); ++i) s.fallback[i] = load(g_compiled_stats.fallback[i]);
	return s;
}

const char* SrtCompiledFallbackName(size_t reason) {
	static const char* const names[] = {"ineligible", "nocfg_unsupported", "cfg_no_trace", "cfg_eval", "cfg_miss"};
	return reason < std::size(names) ? names[reason] : "unknown";
}

void SrtCompiledCountDesc(uint64_t reuse, uint64_t eval, uint64_t skipped) {
	if (reuse != 0u) g_compiled_stats.desc_reuse.fetch_add(reuse, std::memory_order_relaxed);
	if (eval != 0u) g_compiled_stats.desc_eval.fetch_add(eval, std::memory_order_relaxed);
	if (skipped != 0u) g_compiled_stats.memo_skipped.fetch_add(skipped, std::memory_order_relaxed);
}

void SrtCompiledCountSpec(bool recomputed) {
	(recomputed ? g_compiled_stats.spec_recomputed : g_compiled_stats.spec_reused)
	    .fetch_add(1, std::memory_order_relaxed);
}

void SrtCompiledCountMask(bool first, bool images, bool samplers, bool buffer_fmt, bool buffer_addr,
                          bool buffer_size, bool flat, bool user_data) {
	auto& st = g_compiled_stats;
	st.mask_calls.fetch_add(1, std::memory_order_relaxed);
	if (first) {
		st.mask_first.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	st.mask_images.fetch_add(images, std::memory_order_relaxed);
	st.mask_samplers.fetch_add(samplers, std::memory_order_relaxed);
	st.mask_buffer_fmt.fetch_add(buffer_fmt, std::memory_order_relaxed);
	st.mask_buffer_addr.fetch_add(buffer_addr, std::memory_order_relaxed);
	st.mask_buffer_size.fetch_add(buffer_size, std::memory_order_relaxed);
	st.mask_flat.fetch_add(flat, std::memory_order_relaxed);
	st.mask_user_data.fetch_add(user_data, std::memory_order_relaxed);
	if (!images && !samplers && !buffer_fmt && !buffer_addr && !buffer_size && !flat && !user_data)
		st.mask_all_same.fetch_add(1, std::memory_order_relaxed);
}

void SrtCompiledCountVerify(bool mismatch) {
	g_compiled_stats.verify_checks.fetch_add(1, std::memory_order_relaxed);
	if (mismatch) g_compiled_stats.verify_mismatch.fetch_add(1, std::memory_order_relaxed);
}

bool SrtWalker::CompiledEligible() const {
	return m_compiled_enabled && !m_skip_tape && !m_cfg_verifying && !m_scalar_batch_enabled &&
	       !m_cfg_schedule_enabled && m_active_mask.IsEmpty() &&
	       (m_clean_evaluator == nullptr || m_clean_evaluator->m_active_mask.IsEmpty()) &&
	       m_program.srt_plan_complete && SrtTapeEnabled() && !SrtTapeVerifyEnabled();
}

// Fetches `count` adjacent scalar loads (same pointer) with one reader call per contiguous run.
// Commits nothing unless every run succeeded, so a declined or failed merge leaves the original
// single-read steps to execute exactly as before.
bool SrtWalker::ReadCompiledGroup(const CompiledSrt::Step* steps, uint32_t count,
                                  std::vector<uint64_t>& values, bool cfg) {
	const auto& head = steps[0];
	if (head.clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
		return false;
	const auto  reader = head.clean ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	const uint64_t low  = values[head.args[0]];
	const uint64_t high = values[head.args[1]];
	const auto     base = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	std::array<uint64_t, kMaxCompiledGroup> addresses;
	std::array<uint32_t, kMaxCompiledGroup> words;
	for (uint32_t i = 0; i < count; ++i) {
		const auto& step = steps[i];
		if (step.memo != UINT32_MAX && cfg) {
			const auto& context = step.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
			if (step.memo < context.values.size() &&
			    (context.values[step.memo].generation | 3u) == (context.generation | 3u))
				return false;
		}
		if (step.aux >= m_program.memory_info.size()) return false;
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(m_program.memory_info[step.aux].offset));
		const auto offset    = values[step.args[2]];
		const auto relative  = (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, addresses[i])) return false;
	}
	for (uint32_t first = 0; first < count;) {
		uint32_t run = 1;
		while (first + run < count && addresses[first + run] == addresses[first] + run * 4u) ++run;
		const std::span<uint32_t> out(words.data() + first, run);
		SRT_PROF_COUNT(reader_calls, 1);
		SRT_PROF_COUNT(reader_words, run);
		SRT_PROF_LOG(addresses[first], run, head.clean != 0);
		if (reader != nullptr) {
			SRT_PROF_SCOPE(SrtProf::Reader);
			if (!reader(m_runtime.userdata, addresses[first], out)) return false;
		} else {
			std::memcpy(out.data(), reinterpret_cast<const void*>(addresses[first]), out.size_bytes());
		}
		g_compiled_stats.reads.fetch_add(1, std::memory_order_relaxed);
		first += run;
	}
	g_compiled_stats.read_words.fetch_add(count, std::memory_order_relaxed);
	SRT_PROF_SCOPE(SrtProf::MemoWrite);
	for (uint32_t i = 0; i < count; ++i) {
		const auto& step = steps[i];
		values[step.out] = words[i];
		if (step.memo != UINT32_MAX) {
			if (step.publish == 0u) {
				++m_memo_skipped;
				continue;
			}
			auto& context = step.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
			if (step.memo >= context.values.size()) context.values.resize(m_program.evaluation_value_count);
			context.values[step.memo].value      = words[i];
			context.values[step.memo].generation = context.generation;
			if (cfg) m_context.tape_pending.emplace_back(step.out, 0u);
		}
	}
	return true;
}

int SrtWalker::TryCompiled(std::vector<uint32_t>& flat, bool cfg) {
	using Code   = CompiledSrt::Code;
	const size_t variant = m_clean_evaluator != nullptr ? 1 : 0;
	g_compiled_stats.calls.fetch_add(1, std::memory_order_relaxed);
	if (!CompiledEligible()) {
		CountCompiledFallback(SrtCompiledFallback::Ineligible);
		return -1;
	}
	const ResourcePlan::SrtTape* tape_ptr = nullptr;
	CompiledSrt*                 program  = nullptr;
	ResourcePlan::CfgTrace*      trace    = nullptr;
	if (!cfg) {
		auto& tape = m_program.srt_tapes[variant];
		if (tape.state == ResourcePlan::SrtTape::State::Unbuilt) {
			size_t reject_reason = 0;
			if (BuildTapeForPlan(m_program, m_clean_evaluator != nullptr, m_clean_flat_slots, tape, reject_reason)) {
				tape.state = ResourcePlan::SrtTape::State::Ready;
				g_tape_stats.built.fetch_add(1, std::memory_order_relaxed);
			} else {
				tape.state = ResourcePlan::SrtTape::State::Unsupported;
				g_tape_stats.unsupported.fetch_add(1, std::memory_order_relaxed);
				if (reject_reason < 8) g_tape_stats.reject[reject_reason].fetch_add(1, std::memory_order_relaxed);
			}
		}
		program = &m_program.compiled_nocfg[variant];
		if (tape.state != ResourcePlan::SrtTape::State::Ready) {
			CountCompiledFallback(SrtCompiledFallback::NocfgUnsupported);
			return -1;
		}
		if (program->state == CompiledSrt::State::Unbuilt) {
			program->state = CompileNocfg(m_program, tape, *program) ? CompiledSrt::State::Ready
			                                                          : CompiledSrt::State::Unsupported;
			g_compiled_stats.builds.fetch_add(1, std::memory_order_relaxed);
		}
		if (program->state != CompiledSrt::State::Ready) {
			CountCompiledFallback(SrtCompiledFallback::NocfgUnsupported);
			return -1;
		}
		tape_ptr = &tape;
	} else {
		trace          = &m_program.cfg_traces[variant];
		const auto& tape = m_program.cfg_srt_tapes[variant];
		if (trace->disabled || trace->steps.empty() || tape.state != ResourcePlan::SrtTape::State::Ready ||
		    trace->strict != (m_runtime.read_specialization_memory != nullptr)) {
			CountCompiledFallback(SrtCompiledFallback::CfgNoTrace);
			return -1;
		}
		program = &m_program.compiled_cfg[variant];
		if (program->state == CompiledSrt::State::Unbuilt || program->source_steps != trace->steps.size() ||
		    program->source_paths != trace->paths) {
			program->source_steps = static_cast<uint32_t>(trace->steps.size());
			program->source_paths = trace->paths;
			program->state = CompileTrace(m_program, tape, *trace, *program) ? CompiledSrt::State::Ready
			                                                      : CompiledSrt::State::Unsupported;
			g_compiled_stats.builds.fetch_add(1, std::memory_order_relaxed);
		}
		if (program->state != CompiledSrt::State::Ready) {
			CountCompiledFallback(program->reason == 1 ? SrtCompiledFallback::CfgEval
			                                           : SrtCompiledFallback::CfgNoTrace);
			return -1;
		}
		tape_ptr = &tape;
	}
	const auto& tape = *tape_ptr;
	SRT_PROF_STEPPER(prof);
	SRT_PROF_TICK(prof, SrtProf::Preamble);
	SRT_PROF_COUNT(replays, 1);
	auto&       values = m_context.tape_values;
	if (values.size() < program->value_count) values.resize(program->value_count);
	for (const auto& [index, imm]: program->consts) values[index] = imm;
	auto& active    = m_program.active_sources;
	auto& published = m_context.tape_pending;
	if (cfg) {
		flat.assign(m_program.srt_reads.size(), 0u);
		active = trace->base_active;
		published.clear();
	} else {
		flat.resize(m_program.srt_reads.size());
	}
	const auto context_for = [&](const CompiledSrt::Step& step) -> ResourcePlan::EvaluationContext& {
		return step.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context;
	};
	const auto store_memo = [&](const CompiledSrt::Step& step, uint64_t result) {
		if (step.memo == UINT32_MAX) return;
		if (step.publish == 0u) {
			++m_memo_skipped;
			return;
		}
		SRT_PROF_SCOPE(SrtProf::MemoWrite);
		auto& context = context_for(step);
		if (step.memo >= context.values.size()) context.values.resize(m_program.evaluation_value_count);
		context.values[step.memo].value      = result;
		context.values[step.memo].generation = context.generation;
		if (cfg) published.emplace_back(step.out, 0u);
	};
	uint64_t executed = 0;
	uint64_t* v = values.data();
	const auto* steps = program->steps.data();
	// A divergence restores the memo slots this replay published, so the recording DFS sees the
	// same state as after a failed ReplayCfgTrace.
	const auto miss = [&](bool count_cfg_miss) {
		SRT_PROF_FINISH(prof);
		for (const auto& [index, unused]: published) {
			const auto& op = tape.ops[index];
			(op.clean && m_clean_evaluator != nullptr ? m_clean_evaluator->m_context : m_context)
			    .values[op.memo_index].generation = 0;
		}
		published.clear();
		if (m_cfg_gather_enabled) {
			std::fill(m_cfg_gather_ready_sources.begin(), m_cfg_gather_ready_sources.end(), 0u);
			if (m_clean_evaluator != nullptr)
				std::fill(m_clean_evaluator->m_cfg_gather_ready_sources.begin(),
				          m_clean_evaluator->m_cfg_gather_ready_sources.end(), 0u);
		}
		if (count_cfg_miss) {
			if (++trace->misses > (m_source_reuse_enabled ? 8192u : 256u) && trace->misses > trace->hits)
				trace->disabled = true;
			g_tape_stats.cfg_trace_misses.fetch_add(1, std::memory_order_relaxed);
		}
		g_compiled_stats.steps.fetch_add(executed, std::memory_order_relaxed);
		CountCompiledFallback(SrtCompiledFallback::CfgMiss);
		return -2;
	};
	const auto fail = [&]() {
		SRT_PROF_FINISH(prof);
		g_compiled_stats.steps.fetch_add(executed, std::memory_order_relaxed);
		g_compiled_stats.failed.fetch_add(1, std::memory_order_relaxed);
		g_compiled_stats.compiled.fetch_add(1, std::memory_order_relaxed);
		g_compiled_stats.nocfg.fetch_add(1, std::memory_order_relaxed);
		return 0;
	};
	for (uint32_t pc = 0;;) {
		const auto& step = steps[pc];
#if KYTY_SRT_PROFILE
		{
			int slot = SrtProf::StepEnd;
			switch (step.code) {
				case Code::UserData:
				case Code::ShaderBase: slot = SrtProf::StepUserData; break;
				case Code::Alu: slot = SrtProf::StepAlu; break;
				case Code::Read: slot = SrtProf::StepRead; break;
				case Code::Memo: slot = SrtProf::StepMemo; break;
				case Code::Flat: slot = SrtProf::StepFlat; break;
				case Code::Activate: slot = SrtProf::StepActivate; break;
				case Code::Cond:
				case Code::CondTri: slot = SrtProf::StepCond; break;
				case Code::Gather: slot = SrtProf::StepGather; break;
				default: break;
			}
			SRT_PROF_TICK(prof, slot);
		}
#endif
		switch (step.code) {
			case Code::UserData: {
				if (cfg && step.memo != UINT32_MAX) {
					const auto& context = context_for(step);
					if (step.memo < context.values.size() &&
					    (context.values[step.memo].generation | 3u) == (context.generation | 3u))
						return miss(true);
				}
				const uint32_t reg = step.aux;
				if (reg < m_program.user_data_base || reg - m_program.user_data_base >= m_runtime.user_data.size()) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return cfg ? miss(true) : fail();
				}
				const auto user_data_index = reg - m_program.user_data_base;
				if (m_runtime.observe_user_data != nullptr)
					m_runtime.observe_user_data(m_runtime.user_data_observer_userdata, user_data_index);
				const uint64_t result = m_runtime.user_data[user_data_index];
				v[step.out]           = result;
				store_memo(step, result);
				++executed;
				++pc;
				break;
			}
			case Code::ShaderBase: {
				if (cfg && step.memo != UINT32_MAX) {
					const auto& context = context_for(step);
					if (step.memo < context.values.size() &&
					    (context.values[step.memo].generation | 3u) == (context.generation | 3u))
						return miss(true);
				}
				v[step.out] = m_runtime.shader_base;
				store_memo(step, m_runtime.shader_base);
				++executed;
				++pc;
				break;
			}
			case Code::Alu: {
				if (cfg && step.memo != UINT32_MAX) {
					const auto& context = context_for(step);
					if (step.memo < context.values.size() &&
					    (context.values[step.memo].generation | 3u) == (context.generation | 3u))
						return miss(true);
				}
				uint64_t result = 0;
				if (!TapeAlu(step.opcode, step.args, step.imm, v, result)) return cfg ? miss(true) : fail();
				v[step.out] = result;
				store_memo(step, result);
				++executed;
				++pc;
				break;
			}
			case Code::Read: {
				if (step.group > 1u) {
					if (ReadCompiledGroup(&step, step.group, values, cfg)) {
						g_compiled_stats.groups_hit.fetch_add(1, std::memory_order_relaxed);
						executed += step.group;
						pc += step.group;
						break;
					}
					g_compiled_stats.groups_miss.fetch_add(1, std::memory_order_relaxed);
				}
				if (cfg && step.memo != UINT32_MAX) {
					const auto& context = context_for(step);
					if (step.memo < context.values.size() &&
					    (context.values[step.memo].generation | 3u) == (context.generation | 3u))
						return miss(true);
				}
				// Single reads keep the tape's exact address, bounds and domain handling.
				if (!ExecuteTapeOp(tape.ops[step.out], values, step.out, step.publish != 0u)) return cfg ? miss(true) : fail();
				if (cfg && step.memo != UINT32_MAX && step.publish != 0u) published.emplace_back(step.out, 0u);
				g_compiled_stats.reads.fetch_add(1, std::memory_order_relaxed);
				g_compiled_stats.read_words.fetch_add(1, std::memory_order_relaxed);
				++executed;
				++pc;
				break;
			}
			case Code::Memo: {
				const auto& context = context_for(step);
				if (step.memo >= context.values.size() || context.values[step.memo].generation != context.generation)
					return miss(true);
				v[step.out] = context.values[step.memo].value;
				++pc;
				break;
			}
			case Code::Flat:
				flat[step.aux] = static_cast<uint32_t>(v[step.out]);
				++pc;
				break;
			case Code::Activate:
				for (const auto source: m_program.control_flow[step.aux].sources) active[source] = 1u;
				++pc;
				break;
			case Code::Cond: {
				const auto next = step.next[static_cast<uint32_t>(v[step.out]) != 0u ? 1 : 0];
				if (next == UINT32_MAX) return miss(true);
				pc = next;
				break;
			}
			case Code::CondTri: {
				const auto value = static_cast<uint32_t>(v[step.out]);
				const auto next  = step.next[value == 2u ? 2 : value != 0u ? 1 : 0];
				if (next == UINT32_MAX) return miss(true);
				pc = next;
				break;
			}
			case Code::Gather:
				if (m_cfg_gather_enabled) {
					// The step index names the Gather step: EvaluateDescriptor remembers which steps it verified.
					const uint32_t id = pc + 2u;
					m_cfg_gather_ready_sources[step.aux] = id;
					if (m_clean_evaluator != nullptr) m_clean_evaluator->m_cfg_gather_ready_sources[step.aux] = id;
				}
				++pc;
				break;
			case Code::End: {
				SRT_PROF_FINISH(prof);
				if (m_runtime.evaluation_counts) m_runtime.evaluation_counts->tape_ops += executed;
				if (FrameBreakdown::Enabled()) {
					auto& fc = FrameBreakdown::g_counters;
					fc.mat_flat_ops.fetch_add(executed, std::memory_order_relaxed);
					if (!cfg) fc.mat_flat_nocfg_ops.fetch_add(executed, std::memory_order_relaxed);
				}
				g_compiled_stats.steps.fetch_add(executed, std::memory_order_relaxed);
				g_compiled_stats.compiled.fetch_add(1, std::memory_order_relaxed);
				(cfg ? g_compiled_stats.cfg : g_compiled_stats.nocfg).fetch_add(1, std::memory_order_relaxed);
				m_context.refreshed_flat = &flat;
				if (m_clean_evaluator != nullptr) m_clean_evaluator->m_context.refreshed_flat = &flat;
				m_served_mode = m_clean_evaluator != nullptr ? 1 : 0;
				m_served_tape = tape_ptr;
				m_served_cfg  = cfg ? program : nullptr;
				if (m_clean_evaluator != nullptr) {
					m_clean_evaluator->m_served_mode = 2;
					m_clean_evaluator->m_served_tape = tape_ptr;
					m_clean_evaluator->m_served_cfg  = m_served_cfg;
				}
				if (cfg) {
					++trace->hits;
					g_tape_stats.cfg_trace_hits.fetch_add(1, std::memory_order_relaxed);
				} else {
					g_tape_stats.runs.fetch_add(1, std::memory_order_relaxed);
					g_tape_stats.ops.fetch_add(tape.ops.size(), std::memory_order_relaxed);
				}
				return 1;
			}
		}
	}
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
