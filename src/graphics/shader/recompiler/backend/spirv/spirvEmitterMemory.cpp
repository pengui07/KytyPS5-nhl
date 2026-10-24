#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/shader.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	if (state.lane_count == 2) {
		target = Binary(state, spv::OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto shuffled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                          ConstantU32(state, spv::ScopeSubgroup), source, target);
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                          ConstantU32(state, spv::ScopeSubgroup), exec, target);
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

struct BufferAddress {
	uint32_t offset;
	uint32_t byte;
};

BufferAddress CalculateBufferAddress(EmitterState& state, uint32_t index, uint32_t offset,
                                     uint32_t soffset, uint32_t immediate, uint32_t stride,
                                     uint32_t swizzle, uint32_t index_stride) {
	const auto zero = ConstantU32(state, 0);
	const auto one  = ConstantU32(state, 1);
	const auto add = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero ? rhs : rhs == zero ? lhs
		                                      : Binary(state, spv::OpIAdd, TypeU32(state), lhs, rhs);
	};
	const auto mul = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero || rhs == zero ? zero
		       : lhs == one              ? rhs
		       : rhs == one              ? lhs
		                                 : Binary(state, spv::OpIMul, TypeU32(state), lhs, rhs);
	};
	if (immediate != 0u) {
		offset = add(offset, ConstantU32(state, immediate));
	}
	auto address = add(mul(index, stride), offset);
	if (swizzle != 0u) {
		const auto index_shift =
		    Binary(state, spv::OpIAdd, TypeU32(state), index_stride, ConstantU32(state, 3));
		const auto indices = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            ConstantU32(state, 1), index_shift);
		const auto index_msb =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, index_shift);
		const auto index_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), index,
		           Binary(state, spv::OpISub, TypeU32(state), indices, ConstantU32(state, 1)));
		const auto offset_msb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto msb = mul(add(mul(index_msb, stride), offset_msb), indices);
		const auto lsb = add(Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index_lsb,
		                            ConstantU32(state, 2u)), offset_lsb);
		address = Select(state, TypeU32(state), swizzle, add(msb, lsb), address);
	}
	return {offset, add(address, soffset)};
}

uint32_t BufferLane(EmitterState& state) {
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), EmitSubgroupLocalInvocationId(state),
	              ConstantU32(state, 63));
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	const auto stride = packed & 0x3fffu;
	auto       index  = ctx.Arg(inst, 1);
	if ((packed & (1u << 20u)) != 0u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
	}
	const bool swizzle = stride != 0u && (packed & (1u << 14u)) != 0u;
	return CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3), mem.offset,
	                              ConstantU32(state, stride),
	                              swizzle ? ConstantBool(state, true) : 0u,
	                              ConstantU32(state, (packed >> 16u) & 3u))
	    .byte;
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, spv::OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, spv::OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t AddAddressOffset(EmitterState& state, const IR::MemoryInfo& mem, uint32_t low,
                          uint32_t& high) {
	const auto immediate = static_cast<int32_t>(mem.offset);
	if (immediate == 0) return low;
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	return AddU64Low(state, low, high, immediate_low, immediate_high, high);
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state = ctx.state;
	low = AddAddressOffset(state, mem, low, high);
	const auto valid = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = PackU64(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantU64(state, 0);
		}
		auto base_low = ctx.Arg(*handle, 0);
		if (mem.kind == IR::ResourceKind::ScalarAddress) {
			base_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), base_low,
			                  ConstantU32(state, ~3u));
		}
		const auto base = PackU64(state, base_low, ctx.Arg(*handle, 1));
		address         = Binary(state, spv::OpIAdd, TypeU64(state), base,
		                         Unary(state, spv::OpUConvert, TypeU64(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, spv::OpIAdd, TypeU64(state), address,
	                    ConstantU64(state, static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0), index);
	return pointer;
}

// The BufferCache caching page of a guest address: the index of its BDA page table entry and of
// its bits in the fault buffer.
uint32_t BdaPageIndex(EmitterState& state, uint32_t address) {
	const auto type     = TypeU64(state);
	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed   = Select(state, type, extended,
	                             Binary(state, spv::OpISub, type, address,
	                                    ConstantU64(state,
	                                        LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	                             address);
	const auto page64   = Binary(state, spv::OpShiftRightLogical, type, packed,
	                             ConstantU64(state, BufferCache::CACHING_PAGEBITS));
	return Unary(state, spv::OpUConvert, TypeU32(state), page64);
}

// Sets the page's bit in the write bitmap, so the host learns which pages V#-table stores wrote.
// Most stores find the bit set already; only the first store to a page pays for the atomic.
void RecordBdaWrite(EmitterState& state, uint32_t page) {
	static_assert(BufferCache::WRITE_BITMAP_OFFSET % sizeof(uint32_t) == 0 &&
	              BufferCache::WRITE_BITMAP_OFFSET / sizeof(uint32_t) <= UINT32_MAX);
	const auto word = Binary(
	    state, spv::OpIAdd, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5)),
	    ConstantU32(state,
	                static_cast<uint32_t>(BufferCache::WRITE_BITMAP_OFFSET / sizeof(uint32_t))));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto pointer = FaultElementPointer(state, word);
	const auto scope   = ConstantU32(state, spv::ScopeDevice);
	const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), value, pointer, scope, relaxed);
	const auto missing =
	    Binary(state, spv::OpIEqual, TypeBool(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, bit), ConstantU32(state, 0));
	EmitIfCondition(state, missing, [&]() {
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
		                          pointer, scope, relaxed, bit);
	});
}

void RecordBdaFault(EmitterState& state, uint32_t page) {
	const auto word =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto pointer = FaultElementPointer(state, word);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	state.builder.AddFunction(spv::OpStore, pointer,
	                          Binary(state, spv::OpBitwiseOr, TypeU32(state), value, bit));
	// Dirty flag for FaultManager's gate pass: one word past both bitmaps. Every bitmap store is
	// followed by a plain store of 1 here, so "dirty == 0" implies no bit was set since the last
	// parser clear.
	static_assert(BufferCache::FAULT_DIRTY_WORD <= UINT32_MAX);
	const auto dirty_pointer = FaultElementPointer(
	    state, ConstantU32(state, static_cast<uint32_t>(BufferCache::FAULT_DIRTY_WORD)));
	state.builder.AddFunction(spv::OpStore, dirty_pointer, ConstantU32(state, 1u));
}

uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address, bool coherent = false) {
	auto&      state   = ctx.state;
	const auto bda     = GetBdaPointer(state, address);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
	return EmitValueOrZeroIfCondition(state, present, [&]() {
		auto pointer = state.builder.AllocateId();
		if (coherent) {
			const auto block = state.builder.DecoratedType(
			    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
			                        {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
			                        {spv::OpDecorate, {spv::DecorationBlock}}}, TypeU32(state));
			state.builder.AddFunction(
			    spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block),
			    pointer, bda);
			const auto base = pointer;
			pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer,
			                          base, ConstantU32(state, 0));
		} else {
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          bda);
		}
		const auto         value     = state.builder.AllocateId();
		constexpr uint32_t alignment = sizeof(uint32_t);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
		                          spv::MemoryAccessAlignedMask |
		                              (coherent ? spv::MemoryAccessVolatileMask : spv::MemoryAccessMaskNone),
		                          alignment);
		return value;
	});
}

uint32_t LoadBda(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t bits,
                 bool coherent = false) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto first   = LoadBdaDword(ctx, aligned, coherent);
		const auto byte =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		           Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
		const auto crosses =
		    bits == 8u ? ConstantBool(state, false)
		               : Binary(state, bits == 16u ? spv::OpUGreaterThan : spv::OpINotEqual,
		                        TypeBool(state), byte, ConstantU32(state, bits == 16u ? 2u : 0u));
		const auto second = EmitValueOrZeroIfCondition(state, crosses, [&]() {
			return LoadBdaDword(ctx, Binary(state, spv::OpIAdd, TypeU64(state), aligned,
			                                ConstantU64(state, sizeof(uint32_t))), coherent);
		});
		const auto shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3));
		const auto upper_shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                  Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 4), byte),
		                  ConstantU32(state, 3)),
		           ConstantU32(state, 3));
		const auto merged =
		    Binary(state, spv::OpBitwiseOr, TypeU32(state),
		           Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, shift),
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), second, upper_shift));
		return bits == 32u ? merged
		                   : Binary(state, spv::OpBitwiseAnd, TypeU32(state), merged,
		                            ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
	});
}

// Experiment BdaLoadGroups (key bit 8192). A guest load of several DWORDs through an address
// (FLAT, GLOBAL, S_LOAD) is one LoadAddressU32 per DWORD, and each looks up its page and loads in
// branches of its own, one memory latency after another: the vertex shaders of the most drawn
// shadow pipelines had 32 of them in a row. Consecutive loads of consecutive DWORDs with the same
// address operands form a group, loaded at its first load: when the group (with the DWORD after
// it, for an unaligned address) is in one page, one lookup serves it and its loads issue
// together; otherwise each DWORD loads as before. Faults and values are the same either way.

// Loads whose own emission is LoadBdaDword (scalar) or LoadBda (vector), non-coherent.
bool BdaGroupable(const ValueEmitContext& ctx, const IR::Inst& inst) {
	if (inst.GetOpcode() != IR::ValueOpcode::LoadAddressU32) return false;
	const auto& mem = ctx.Memory(inst);
	return !mem.planning_only && !mem.coherent &&
	       (mem.kind == IR::ResourceKind::ScalarAddress || mem.kind == IR::ResourceKind::Flat ||
	        mem.kind == IR::ResourceKind::Global);
}

// The address offset GuestAddress() adds for the load.
int64_t BdaImmediate(const IR::MemoryInfo& mem) {
	auto immediate = mem.offset;
	if (mem.kind == IR::ResourceKind::ScalarAddress) immediate &= ~3u;
	return static_cast<int64_t>(static_cast<int32_t>(immediate));
}

// Whether `next` loads the DWORD after `previous`'s with the same address operands.
bool BdaGroupContinues(const ValueEmitContext& ctx, const IR::Inst& previous, const IR::Inst& next) {
	const auto& a = ctx.Memory(previous);
	const auto& b = ctx.Memory(next);
	if (a.kind != b.kind || a.address_is_full != b.address_is_full ||
	    previous.NumArgs() != next.NumArgs() || BdaImmediate(b) != BdaImmediate(a) + 4) {
		return false;
	}
	for (size_t arg = 0; arg < previous.NumArgs(); arg++) {
		if (!(previous.Arg(arg).Resolve() == next.Arg(arg).Resolve())) return false;
	}
	return true;
}

const std::unordered_map<const IR::Inst*, std::vector<const IR::Inst*>>&
BdaLoadGroups(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	if (state.bda_load_groups_built) return state.bda_load_groups;
	state.bda_load_groups_built = true;
	constexpr size_t MaxGroup = 16;
	static const bool debug = std::getenv("KYTY_DBG_BDA_GROUPS") != nullptr;
	size_t            loads = 0;
	for (const auto* block: state.program.blocks) {
		std::vector<const IR::Inst*> group;
		const auto close = [&]() {
			if (group.size() >= 2) {
				loads += group.size();
				const auto* first = group.front();
				state.bda_load_groups.emplace(first, std::move(group));
			}
			group.clear();
		};
		for (const auto& inst: *block) {
			if (!BdaGroupable(ctx, inst)) {
				close();
				continue;
			}
			if (group.empty() || group.size() == MaxGroup || !BdaGroupContinues(ctx, *group.back(), inst)) {
				close();
			}
			group.push_back(&inst);
		}
		close();
	}
	if (debug && !state.bda_load_groups.empty()) {
		std::printf("NHL27BDAGROUP: hash=0x%016llx stage=%u groups=%zu loads=%zu\n",
		            static_cast<unsigned long long>(state.program.shader_hash),
		            static_cast<unsigned>(state.program.stage), state.bda_load_groups.size(), loads);
		std::fflush(stdout);
	}
	return state.bda_load_groups;
}

uint32_t LoadPhysicalDword(EmitterState& state, uint32_t address) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
	const auto         value     = state.builder.AllocateId();
	constexpr uint32_t alignment = sizeof(uint32_t);
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask, alignment);
	return value;
}

// "if (condition) values = then(); else values = otherwise();" with a phi per value; without
// `otherwise` the values are 0 where the condition is false.
template <typename Then, typename Otherwise>
std::vector<uint32_t> EmitValuesIfElse(EmitterState& state, uint32_t condition, size_t count,
                                       Then&& then_values, Otherwise&& otherwise_values, uint32_t type = 0) {
	const auto then_label  = state.builder.AllocateId();
	const auto else_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, then_label, else_label);
	EmitLabel(state, then_label);
	const auto then_result = then_values();
	const auto then_exit   = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, else_label);
	const auto else_result = otherwise_values();
	const auto else_exit   = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	std::vector<uint32_t> result(count);
	for (size_t index = 0; index < count; index++) {
		result[index] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, type != 0 ? type : TypeU32(state), result[index], then_result[index],
		                          then_exit, else_result[index], else_exit);
	}
	return result;
}

template <typename Then>
std::vector<uint32_t> EmitValuesOrZerosIfCondition(EmitterState& state, uint32_t condition,
                                                   size_t count, Then&& then_values) {
	return EmitValuesIfElse(state, condition, count, std::forward<Then>(then_values), [&]() {
		return std::vector<uint32_t>(count, ConstantU32(state, 0));
	});
}

// Experiment BdaTwoLookups: a scalar load group (S_LOAD_DWORDX2..X16 through a pointer) that crosses
// into the next caching page looks up its first and its last page, and each DWORD reads at the host
// address of its own page (a missing page's DWORDs read 0; its fault is recorded). Looking up each
// DWORD by itself made the crossing branch of every group the larger one: most of the page lookups of
// NHL 27's heavy pixel shaders (268 lookups for 35 groups in 9eaaf80c).
std::vector<uint32_t> LoadCrossingScalarGroup(EmitterState& state, uint32_t address, uint32_t count) {
	const auto u64     = TypeU64(state);
	const auto u32     = TypeU32(state);
	const auto boolean = TypeBool(state);
	const auto bits    = ConstantU64(state, BufferCache::CACHING_PAGEBITS);
	// DWORDs before the next page.
	const auto next_page = Binary(state, spv::OpShiftLeftLogical, u64,
	                              Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpShiftRightLogical, u64, address, bits),
	                                     ConstantU64(state, 1)),
	                              bits);
	const auto split = Unary(state, spv::OpUConvert, u32,
	                         Binary(state, spv::OpShiftRightLogical, u64, Binary(state, spv::OpISub, u64, next_page, address),
	                                ConstantU64(state, 2)));
	const auto last        = Binary(state, spv::OpIAdd, u64, address, ConstantU64(state, 4u * (count - 1u)));
	const auto first_bda   = GetBdaPointer(state, address);
	const auto last_bda    = GetBdaPointer(state, last);
	const auto first_found = Binary(state, spv::OpINotEqual, boolean, first_bda, ConstantU64(state, 0));
	const auto last_found  = Binary(state, spv::OpINotEqual, boolean, last_bda, ConstantU64(state, 0));
	std::vector<uint32_t> values;
	for (uint32_t index = 0; index < count; index++) {
		const auto on_first = Binary(state, spv::OpULessThan, boolean, ConstantU32(state, index), split);
		const auto host     = Select(state, u64, on_first,
		                             Binary(state, spv::OpIAdd, u64, first_bda, ConstantU64(state, 4u * index)),
		                             Binary(state, spv::OpISub, u64, last_bda, ConstantU64(state, 4u * (count - 1u - index))));
		// Not an OpSelect on booleans.
		const auto found = Binary(state, spv::OpLogicalOr, boolean, Binary(state, spv::OpLogicalAnd, boolean, on_first, first_found),
		                          Binary(state, spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, on_first),
		                                 last_found));
		values.push_back(EmitValueOrZeroIfCondition(state, found, [&]() { return LoadPhysicalDword(state, host); }));
	}
	return values;
}

std::vector<uint32_t> LoadBdaGroup(ValueEmitContext& ctx, const std::vector<const IR::Inst*>& group) {
	auto&       state   = ctx.state;
	const auto& leader  = *group.front();
	const auto& mem     = ctx.Memory(leader);
	const bool  scalar  = mem.kind == IR::ResourceKind::ScalarAddress;
	const auto  count   = static_cast<uint32_t>(group.size());
	const auto  u64     = TypeU64(state);
	const auto  address = GuestAddress(ctx, leader, mem);
	const auto  offset  = [&](uint32_t base, uint32_t bytes) {
		return bytes == 0 ? base : Binary(state, spv::OpIAdd, u64, base, ConstantU64(state, bytes));
	};
	const auto load = [&]() {
		// Scalar addresses are DWORD aligned already; a vector one may need the DWORD after the group.
		const auto aligned =
		    scalar ? address
		           : Binary(state, spv::OpBitwiseAnd, u64, address, ConstantU64(state, ~uint64_t {3}));
		const auto last      = offset(aligned, 4u * (scalar ? count - 1u : count));
		const auto same_page = Binary(state, spv::OpIEqual, TypeBool(state), BdaPageIndex(state, aligned),
		                              BdaPageIndex(state, last));
		return EmitValuesIfElse(
		    state, same_page, count,
		    [&]() {
			    const auto bda     = GetBdaPointer(state, aligned);
			    const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda,
			                                ConstantU64(state, 0));
			    return EmitValuesOrZerosIfCondition(state, present, count, [&]() {
				    std::vector<uint32_t> dwords;
				    for (uint32_t index = 0; index < count + (scalar ? 0u : 1u); index++) {
					    dwords.push_back(LoadPhysicalDword(state, offset(bda, 4u * index)));
				    }
				    if (scalar) return dwords;
				    // As LoadBda merges the two DWORDs an unaligned address spans.
				    const auto u32   = TypeU32(state);
				    const auto byte  = Binary(state, spv::OpBitwiseAnd, u32,
				                              Unary(state, spv::OpUConvert, u32, address),
				                              ConstantU32(state, 3));
				    const auto crosses =
				        Binary(state, spv::OpINotEqual, TypeBool(state), byte, ConstantU32(state, 0));
				    const auto shift = Binary(state, spv::OpShiftLeftLogical, u32, byte, ConstantU32(state, 3));
				    const auto upper_shift = Binary(
				        state, spv::OpShiftLeftLogical, u32,
				        Binary(state, spv::OpBitwiseAnd, u32,
				               Binary(state, spv::OpISub, u32, ConstantU32(state, 4), byte),
				               ConstantU32(state, 3)),
				        ConstantU32(state, 3));
				    std::vector<uint32_t> values;
				    for (uint32_t index = 0; index < count; index++) {
					    const auto second = Select(state, u32, crosses, dwords[index + 1], ConstantU32(state, 0));
					    values.push_back(Binary(
					        state, spv::OpBitwiseOr, u32,
					        Binary(state, spv::OpShiftRightLogical, u32, dwords[index], shift),
					        Binary(state, spv::OpShiftLeftLogical, u32, second, upper_shift)));
				    }
				    return values;
			    });
		    },
		    [&]() {
			    if (scalar && ShaderBdaTwoLookups()) {
				    return LoadCrossingScalarGroup(state, address, count);
			    }
			    std::vector<uint32_t> values;
			    for (uint32_t index = 0; index < count; index++) {
				    const auto component = offset(address, 4u * index);
				    values.push_back(scalar ? LoadBdaDword(ctx, component)
				                            : LoadBda(ctx, component, ConstantBool(state, true), 32u));
			    }
			    return values;
		    });
	};
	if (scalar) return load();
	return EmitValuesOrZerosIfCondition(state, ctx.Arg(leader, leader.NumArgs() - 1), count, load);
}

// The value of a grouped load, or 0 when it loads by itself.
uint32_t LoadGroupedBda(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (!ShaderBdaLoadGroups() || ctx.state.lane_count != 1) return 0;
	if (const auto found = ctx.grouped_loads.find(&inst); found != ctx.grouped_loads.end()) {
		const auto [label, value] = found->second;
		ctx.grouped_loads.erase(found);
		if (label == ctx.state.current_label) return value;
		static const bool debug = std::getenv("KYTY_DBG_BDA_GROUPS") != nullptr;
		if (debug) {
			std::printf("NHL27BDAGROUP: hash=0x%016llx load emitted apart from its group\n",
			            static_cast<unsigned long long>(ctx.state.program.shader_hash));
		}
		return 0;
	}
	const auto& groups = BdaLoadGroups(ctx);
	const auto  group  = groups.find(&inst);
	if (group == groups.end()) return 0;
	const auto values = LoadBdaGroup(ctx, group->second);
	for (size_t index = 1; index < values.size(); index++) {
		ctx.grouped_loads.insert_or_assign(group->second[index],
		                                   std::pair {ctx.state.current_label, values[index]});
	}
	return values.front();
}

// Experiment MovRelArrays: an element of a register array's function-local variable.
uint32_t RegArrayElement(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state     = ctx.state;
	const auto array     = inst.Arg(0).Resolve();
	EXIT_IF(!array.IsImmediate());
	const auto found     = state.reg_array_variables.find(array.U32());
	EXIT_IF(found == state.reg_array_variables.end());
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeU32ElementPointer(state, spv::StorageClassFunction),
	                          pointer, found->second[state.lane_half], ctx.Def(inst.Arg(1)));
	return pointer;
}

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto resource = ctx.state.program.info.buffers.at(mem.resource).descriptor_index;
		const auto prefix = ctx.state.memory_byte_offsets[resource];
		const auto address = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state),
		                            BufferByteAddress(ctx, inst, mem), prefix);
		// The host binding prefix must not wrap an out-of-range guest address into the buffer.
		return Select(ctx.state, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), address, prefix),
		              address, ConstantU32(ctx.state, UINT32_MAX));
	}
	if (mem.kind == IR::ResourceKind::Lds && ctx.state.program.info.function_lds_lane_slots) {
		// Per-lane slots: drop the LaneId * 4 every access shares (see SpirvRequirements).
		uint32_t constant = 0;
		if (!MatchLaneSlotLdsAddress(inst.Arg(0), constant)) {
			EXIT("LDS lane slot address no longer matches\n");
		}
		return ConstantU32(ctx.state, (constant + mem.offset) & 0xffffu);
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto address = ByteAddress(ctx, inst, mem);
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		// RDNA2 DS region addresses retain bits [15:2] after adding the byte offset.
		address = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                 ConstantU32(ctx.state, 0xffffu));
	}
	return Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
	              address, ConstantU32(ctx.state, 2));
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend);

// Experiment RobustLoads: storage-buffer reads out of range return zero on the device itself.
// Shaders that spill guest VGPRs (to LDS or scratch) keep their branches: unconditional loads made
// the one such NHL 27 pixel shader 16-27% slower (more live values), the others 8-23% faster.
bool RobustLoad(const EmitterState& state, IR::ResourceKind kind) {
	const bool spills = state.program.info.function_lds || state.program.scratch_dwords != 0;
	return (kind == IR::ResourceKind::Buffer || kind == IR::ResourceKind::ScalarBuffer) && ShaderRobustLoads() &&
	       (!spills || (ShaderRobustSpillLoads() && state.program.stage != ShaderType::Pixel));
}

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	const auto index = DwordIndex(ctx, inst, mem);
	if (RobustLoad(ctx.state, resource.kind)) {
		return LoadWordInBounds(ctx, resource, index);
	}
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadWordInBounds(ctx, resource, index); });
}

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	if (RobustLoad(ctx.state, mem.kind) && !ShaderRobustLoadsKeepExec()) {
		// Lanes outside EXEC load too; their result is discarded where the register is written.
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	});
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits,
                             bool sign_extend) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	if (RobustLoad(ctx.state, resource.kind)) {
		return LoadSubwordInBounds(ctx, resource, address, index, bits, sign_extend);
	}
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    return LoadSubwordInBounds(ctx, resource, address, index, bits, sign_extend);
	    });
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	const auto value   = ctx.state.builder.AllocateId();
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value, pointer,
	                              resource.memory_access);
	return value;
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend) {
	uint32_t value;
	if (resource.element_bits < 32u) {
		const auto loaded = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad,
		                              ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type, loaded,
		                              EmitMemoryElementPointer(ctx.state, resource, index),
		                              resource.memory_access);
		value = Unary(ctx.state, spv::OpUConvert, TypeU32(ctx.state), loaded);
	} else {
		const auto word = LoadWordInBounds(ctx, resource, index);
		const auto byte = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                         ConstantU32(ctx.state, 3));
		const auto shift = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), byte,
		                          ConstantU32(ctx.state, 3));
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		               Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), word, shift),
		               ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	}
	if (!sign_extend) return value;
	const auto left = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), value,
	                         ConstantU32(ctx.state, 32u - bits));
	return Binary(ctx.state, spv::OpShiftRightArithmetic, TypeU32(ctx.state), left,
	              ConstantU32(ctx.state, 32u - bits));
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits,
                     bool sign_extend) {
	if (RobustLoad(ctx.state, mem.kind) && !ShaderRobustLoadsKeepExec()) {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits, sign_extend);
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits, sign_extend);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto selector = GetDstSel(ctx.state.program.info.buffers[mem.resource].descriptor_swizzle,
	                                output_component);
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                                FormattedSource source, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	auto raw = info.packed_bitfield || bits == 32u ? load_word(component)
	                                                : load_subword(component, bits);
	if (info.packed_bitfield || (bits < 32u && IsSignedFormatComponent(info.type))) {
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, spv::OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(IsSignedFormatComponent(info.type) ? spv::OpBitFieldSExtract
		                                                                 : spv::OpBitFieldUExtract,
		                              type, extracted, source_value,
		                              ConstantU32(ctx.state, info.packed_bitfield
		                                                         ? info.component_bit_offset[component]
		                                                         : 0u),
		                              ConstantU32(ctx.state, bits));
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.element_bits < 32u) {
		ctx.state.builder.AddFunction(
		    spv::OpStore, pointer,
		    Unary(ctx.state, spv::OpUConvert,
		          ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type, data),
		    resource.memory_access);
		return;
	}
	const auto shift   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
	                                   ConstantU32(ctx.state, 3)),
	                            ConstantU32(ctx.state, 3));
	const auto mask    = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), data,
	                                   ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                            shift);
	const auto merge   = [&](uint32_t old) {
		return Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, merge(old));
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	// Only plain 8/16-bit stores: a read-modify-write of the containing dword out of range must not
	// spin on a compare-exchange.
	if (RobustLoad(ctx.state, resource.kind) && ShaderRobustStores() && resource.element_bits < 32u) {
		StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
		return;
	}
	EmitIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	    });
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreSubwordPrepared(ctx, inst, mem, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	ctx.state.builder.AddFunction(spv::OpStore,
	                              EmitMemoryElementPointer(ctx.state, resource, index), data,
	                              resource.memory_access);
}

template <typename Fn>
void ForEachLocalFlatAccess(ValueEmitContext& ctx, const IR::Inst& inst, Fn&& emit) {
	auto& state = ctx.state;
	auto mem = ctx.Memory(inst);
	auto high = ctx.Arg(inst, 2);
	const auto low = AddAddressOffset(state, mem, ctx.Arg(inst, 1), high);
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low, ConstantU32(state, 2));
	const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), ConstantU32(state, 0),
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 3)));
	for (const auto kind: {IR::ResourceKind::Scratch, IR::ResourceKind::Lds}) {
		mem.kind = kind;
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto aperture = kind == IR::ResourceKind::Scratch ? Decoder::PrivateApertureHigh
		                                                       : Decoder::SharedApertureHigh;
		const auto selected = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, aperture));
		const auto valid = AndCondition(state, aligned,
		    AndCondition(state, selected, EmitMemoryElementInBounds(state, resource, index)));
		emit(resource, index, valid);
	}
}

uint32_t LoadLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		uint32_t result = 0;
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			const auto value = EmitValueOrZeroIfCondition(ctx.state, valid, [&]() {
				return LoadWordInBounds(ctx, resource, index);
			});
			result = result == 0 ? value : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), result, value);
		});
		return result;
	});
}

void StoreLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			EmitIfCondition(ctx.state, valid, [&]() {
				StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, inst.NumArgs() - 2));
			});
		});
	});
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data) {
	const auto index = DwordIndex(ctx, inst, mem);
	if (RobustLoad(ctx.state, resource.kind) && ShaderRobustStores()) {
		StoreWordInBounds(ctx, resource, index, data);
		return;
	}
	EmitIfCondition(ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		StoreWordInBounds(ctx, resource, index, data);
	});
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto store = [&](uint32_t resource) {
			if (mem.kind == IR::ResourceKind::Buffer) {
				const auto& buffer = state.program.info.buffers[resource];
				if (buffer.packed_stride == 0u &&
				    buffer.source < state.program.descriptor_sources.size() &&
				    state.program.descriptor_sources[buffer.source].indirect_descriptor) return;
			}
			mem.resource = resource;
			const auto access = PrepareMemoryResourceAccess(state, mem);
			StoreWordPrepared(ctx, inst, mem, access, data);
		};
		if (mem.kind != IR::ResourceKind::Buffer ||
		    state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
			store(mem.resource);
			return;
		}
		const auto& buffer = state.program.info.buffers[mem.resource];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr || buffer.indirect_resources.empty() ||
		    state.flattened_srt_variable == 0 || buffer.indirect_search_iterations == 0u) {
			ctx.Fail(inst, "has no indirect buffer runtime mapping");
			return;
		}
		const auto selected = EmitIndirectResourceIndex(
		    state, ctx.Arg(*handle, 0), buffer.indirect_mapping_offset,
		    buffer.indirect_search_iterations, UINT32_MAX);
		const auto merge = state.builder.AllocateId();
		std::vector<uint32_t> labels(buffer.indirect_resources.size());
		std::vector<uint32_t> branches {spv::OpSwitch, selected, merge};
		for (uint32_t i = 0; i < labels.size(); ++i) {
			labels[i] = state.builder.AllocateId();
			branches.push_back(i);
			branches.push_back(labels[i]);
		}
		state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
		state.builder.AddFunction(branches);
		for (uint32_t i = 0; i < labels.size(); ++i) {
			EmitLabel(state, labels[i]);
			store(buffer.indirect_resources[i]);
			state.builder.AddFunction(spv::OpBranch, merge);
		}
		EmitLabel(state, merge);
	});
}

template <typename Fn>
uint32_t EmitAtomicAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& operation) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto index = DwordIndex(ctx, inst, mem);
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return EmitValueOrZeroIfCondition(
		    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
			    return operation(EmitMemoryElementPointer(ctx.state, resource, index));
		    });
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
}

uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old >= limit ? 0 : old + 1 (unsigned).
	const auto wrap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, spv::OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old == 0 || old > limit ? limit : old - 1 (unsigned).
	const auto zero  = Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, spv::OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, spv::OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, spv::OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	uint32_t                 base = 0;
	uint32_t                 in_bounds = 0;
};

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo& mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info) {
	PreparedFormattedMemory plan;
	plan.info      = info;
	plan.resource  = resource;
	const auto start = ByteAddress(ctx, inst, mem);
	// Check the whole, unaligned format span before aligning the actual memory address.
	const auto end = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), start,
	                        ConstantU32(ctx.state, info.byte_size - 1u));
	const auto last_index = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), end,
	                               ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	plan.in_bounds = AndCondition(
	    ctx.state, Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), end, start),
	    EmitMemoryElementInBounds(ctx.state, plan.resource, last_index));
	plan.base = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), start,
	                   ConstantU32(ctx.state, ~(std::min(4u, info.byte_size) - 1u)));
	return plan;
}

std::pair<uint32_t, uint32_t> FormattedComponentAddress(EmitterState& state,
                                                       const PreparedFormattedMemory& plan,
                                                       uint32_t component) {
	// Components advance sequentially from one address; do not reapply buffer swizzling.
	const auto offset = Format::GetFormatComponentByteOffset(plan.info, component);
	const auto address = offset == 0u ? plan.base : Binary(
	    state, spv::OpIAdd, TypeU32(state), plan.base, ConstantU32(state, offset));
	return {address, Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                        ConstantU32(state, std::countr_zero(plan.resource.element_bits / 8u)))};
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, plan.info, ResolveFormattedSource(ctx, mem, plan.info, output_component),
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource,
		                            FormattedComponentAddress(ctx.state, plan, component).second);
	    },
	    [&](uint32_t component, uint32_t bits) {
		    const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
		    return LoadSubwordInBounds(ctx, plan.resource, address, index, bits, false);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	if (components == 1u) return values[0];
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpCompositeConstruct, TypeU32Composite(state, components),
	                             result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
	}
	return ConstructU32Composite(ctx.state, components, values);
}

// Robust loads: a formatted load reads its components unconditionally (an out-of-range element
// reads zeros on the device) and selects the out-of-bounds value instead of branching around the
// loads (the crowd vertex shaders had ~400 such branches around 16-bit vertex loads). Lanes outside
// EXEC load too; the register write discards their result.
bool BranchlessFormattedLoad(const EmitterState& state, const IR::MemoryInfo& mem) {
	return RobustLoad(state, mem.kind) && !ShaderRobustLoadsKeepExec() && ShaderBranchlessFormattedLoads();
}

uint32_t FormattedLoadSelect(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const PreparedFormattedMemory& plan,
                             uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto loaded = LoadFormattedInBounds(ctx, mem, plan, component);
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = Select(ctx.state, TypeU32(ctx.state), plan.in_bounds, loaded,
		                           FormattedConstant(ctx, plan.info, source.kind));
	}
	return ConstructU32Composite(ctx.state, components, values);
}

uint32_t FormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (BranchlessFormattedLoad(ctx.state, mem)) {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto info     = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type != Format::ComponentType::Unknown) {
			const auto plan  = PrepareFormattedMemory(ctx, inst, mem, resource, info);
			const auto value = FormattedLoadSelect(ctx, mem, plan, 1u);
			if (mem.data_bits != 16u) {
				return value;
			}
			if (info.type == Format::ComponentType::Uint) {
				return EmitUMin32(ctx.state, value, ConstantU32(ctx.state, 0xffffu));
			}
			if (info.type == Format::ComponentType::Sint) {
				const auto lower = EmitSMax32(ctx.state, value, ConstantU32(ctx.state, 0xffff8000u));
				return EmitSMin32(ctx.state, lower, ConstantU32(ctx.state, 0x7fffu));
			}
			return EmitConvertF16F32(ctx.state, EmitBitCastF32U32(ctx.state, value));
		}
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto info = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			return LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource);
		}
		const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
		const auto value = EmitValueOrDefaultIfCondition(
		    ctx.state, plan.in_bounds, TypeU32(ctx.state),
		    FormattedOutOfBoundsValue(ctx, mem, plan, 1u),
		    [&]() { return LoadFormattedInBounds(ctx, mem, plan, 0u); });
		if (mem.data_bits != 16u) {
			return value;
		}
		if (info.type == Format::ComponentType::Uint) {
			return EmitUMin32(ctx.state, value, ConstantU32(ctx.state, 0xffffu));
		}
		if (info.type == Format::ComponentType::Sint) {
			const auto lower = EmitSMax32(ctx.state, value, ConstantU32(ctx.state, 0xffff8000u));
			return EmitSMin32(ctx.state, lower, ConstantU32(ctx.state, 0x7fffu));
		}
		return EmitConvertF16F32(ctx.state, EmitBitCastF32U32(ctx.state, value));
	});
}

uint32_t EncodeFormattedStoreComponent(ValueEmitContext& ctx,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t component, uint32_t data, uint32_t data_bits) {
	auto& state = ctx.state;
	const auto bits = info.component_bits[component];
	if (data_bits == 16u) {
		if (info.type == Format::ComponentType::Uint) {
			data = EmitBitFieldUExtract(state, data, ConstantU32(state, 0), ConstantU32(state, 16));
		} else if (info.type == Format::ComponentType::Sint) {
			data = EmitBitFieldSExtract(state, data, ConstantU32(state, 0), ConstantU32(state, 16));
		} else {
			data = EmitBitCastU32F32(state, EmitF16BitsToF32(state, data));
		}
	}
	const bool normalized = info.type == Format::ComponentType::Unorm ||
	                        info.type == Format::ComponentType::Snorm;
	const bool scaled = info.type == Format::ComponentType::Uscaled ||
	                    info.type == Format::ComponentType::Sscaled;
	if (normalized || scaled) {
		// Two-bit packed channels are unsigned even in signed formats.
		const bool is_signed = bits != 2u && (info.type == Format::ComponentType::Snorm ||
		                                      info.type == Format::ComponentType::Sscaled);
		const auto max_value = static_cast<float>((1u << (bits - (is_signed ? 1u : 0u))) - 1u);
		const auto lower = normalized ? (is_signed ? -1.0f : 0.0f)
		                              : (is_signed ? -max_value - 1.0f : 0.0f);
		const auto upper = normalized ? 1.0f : max_value;
		auto value = Select(state, TypeF32(state), EmitClassifyF32Bits(state, data).nan,
		                    ConstantF32Value(state, 0.0f), EmitBitCastF32U32(state, data));
		value = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
		    state, value, ConstantF32Value(state, lower), ConstantF32Value(state, upper));
		if (normalized) {
			value = EmitFPRoundEven32(
			    state, EmitFPMul32(state, value, ConstantF32Value(state, max_value)));
		}
		// Float-to-integer conversion truncates scaled values toward zero.
		value = Unary(state, is_signed ? spv::OpConvertFToS : spv::OpConvertFToU,
		              is_signed ? TypeI32(state) : TypeU32(state), value);
		return is_signed ? Unary(state, spv::OpBitcast, TypeU32(state), value) : value;
	}
	if (bits == 16u && info.type == Format::ComponentType::Float) {
		const auto pair = EmitCompositeConstructF32x2(
		    state, EmitBitCastF32U32(state, data), ConstantF32Value(state, 0.0f));
		return EmitPackHalf2x16(state, pair);
	}
	if ((bits == 11u || bits == 10u) && info.type == Format::ComponentType::Float) {
		// Unsigned 11/10-bit floats share the half-float exponent: drop the sign and the low
		// mantissa bits of the f16 encoding, clamping negatives to zero and keeping NaN a NaN.
		const auto value = EmitBitCastF32U32(ctx.state, data);
		const auto pair  = EmitCompositeConstructF32x2(ctx.state, value,
		                                               ConstantF32Value(ctx.state, 0.0f));
		const auto half  = EmitPackHalf2x16(ctx.state, pair);
		const auto shift = ConstantU32(ctx.state, bits == 11u ? 4u : 5u);
		const auto packed_small = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
		                          EmitAndConstant(ctx.state, half, 0x7fffu), shift);
		const auto nan   = Binary(ctx.state, spv::OpUGreaterThan, TypeBool(ctx.state),
		                          EmitAndConstant(ctx.state, half, 0x7fffu),
		                          ConstantU32(ctx.state, 0x7c00u));
		const auto negative = Binary(ctx.state, spv::OpINotEqual, TypeBool(ctx.state),
		                             EmitAndConstant(ctx.state, half, 0x8000u),
		                             ConstantU32(ctx.state, 0u));
		const auto finite =
		    Select(ctx.state, TypeU32(ctx.state), negative, ConstantU32(ctx.state, 0u), packed_small);
		return Select(ctx.state, TypeU32(ctx.state), nan,
		              ConstantU32(ctx.state, bits == 11u ? 0x7ffu : 0x3ffu), finite);
	}
	return data;
}

void StoreFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             const Format::BufferFormatInfo& info, uint32_t data,
                             uint32_t components) {
	// RDNA2 formatted stores transfer the entire format, zero-filling missing VGPRs.
	const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	EmitIfCondition(ctx.state, plan.in_bounds, [&]() {
		auto packed = ConstantU32(ctx.state, 0);
		for (uint32_t component = 0; component < info.component_count; component++) {
			auto value = ConstantU32(ctx.state, 0);
			if (component < components) {
				value = data;
				if (components != 1u) {
					value = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
					                              value, data, component);
				}
			}
			value = EncodeFormattedStoreComponent(ctx, info, component, value, mem.data_bits);
			const auto bits = info.component_bits[component];
			if (info.packed_bitfield) {
				packed = EmitBitFieldInsert(
				    ctx.state, packed, value,
				    ConstantU32(ctx.state, info.component_bit_offset[component]),
				    ConstantU32(ctx.state, bits));
			} else {
				const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
				if (bits == 8u || bits == 16u) {
					StoreSubwordInBounds(ctx, mem, plan.resource, address, index, bits, value);
				} else {
					StoreWordInBounds(ctx, plan.resource, index, value);
				}
			}
		}
		if (info.packed_bitfield) {
			StoreWordInBounds(ctx, plan.resource,
			                  FormattedComponentAddress(ctx.state, plan, 0u).second, packed);
		}
	});
}

void FormattedStore(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto data     = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info     = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource, data);
			return;
		}
		StoreFormattedPrepared(ctx, inst, mem, resource, info, data, 1u);
	});
}

uint32_t LoadIndirectScalarBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word1  = ctx.Arg(handle, 1);
	const auto  stride = EmitBitFieldUExtract(state, word1, ConstantU32(state, 16),
	                                         ConstantU32(state, 14));
	const auto  size   = Binary(state, spv::OpIMul, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  EmitUMax32(state, stride, ConstantU32(state, 1))),
	                            Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(handle, 2)));
	const auto  offset = Binary(state, spv::OpIAdd, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                                         ctx.Arg(inst, 1), ConstantU32(state, ~3u))),
	                            ConstantU64(state, ctx.Memory(inst).offset & ~3u));
	const auto  end =
	    Binary(state, spv::OpIAdd, TypeU64(state), offset, ConstantU64(state, sizeof(uint32_t)));
	return EmitValueOrZeroIfCondition(
	    state, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size), [&]() {
		    const auto base = PackU64(state,
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                                     ctx.Arg(handle, 0), ConstantU32(state, ~3u)),
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1,
		                                     ConstantU32(state, 0xffffu)));
		    return LoadBdaDword(ctx, Binary(state, spv::OpIAdd, TypeU64(state), base, offset));
	    });
}

struct IndirectBufferAccess {
	uint32_t address;
	uint32_t offset;
	uint32_t stride;
	uint32_t swizzle;
	uint32_t format;
	uint32_t selector;
	uint32_t mode;
	uint32_t records;
	uint32_t index_in_bounds;
	uint32_t scalar_in_bounds;
	uint32_t raw_records;
	uint32_t raw_index_in_bounds;
};

IndirectBufferAccess PrepareIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto word1   = ctx.Arg(handle, 1);
	const auto records = ctx.Arg(handle, 2);
	const auto word3   = ctx.Arg(handle, 3);
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto index = Binary(
	    state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	    Select(state, TypeU32(state), nonzero(field(word3, 23, 1)), BufferLane(state),
	           ConstantU32(state, 0)));
	const auto soffset = ctx.Arg(inst, 3);
	const auto address = CalculateBufferAddress(state, index, ctx.Arg(inst, 2), soffset,
	                                            ctx.Memory(inst).offset, stride, swizzle,
	                                            field(word3, 21, 2));
	const auto base        = PackU64(state, ctx.Arg(handle, 0), field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	return {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = field(word3, 12, 7),
	    .selector         = ctx.Memory(inst).formatted ? field(word3, 0, 3) : 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	};
}

uint32_t IndirectBufferInBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                uint32_t displacement, uint32_t bytes, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size,
		                  ConstantU32(state, bytes)),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, ConstantU32(state, bytes))));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

uint32_t LoadIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	const auto buffer = PrepareIndirectBuffer(ctx, inst);
	const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
	                                  ConstantU32(state, 0));
	const auto base         = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                                 ConstantU64(state, ~uint64_t {3}));
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; ++component) {
		const auto address = component == 0u ? base
		                                     : Binary(state, spv::OpIAdd, TypeU64(state), base,
		                                              ConstantU64(state, component * 4u));
		values[component] = EmitValueOrZeroIfCondition(
		    state, AndCondition(state, valid_format,
		                        IndirectBufferInBounds(state, buffer, component * 4u, 4u, false)),
		    [&]() { return LoadBdaDword(ctx, address); });
	}
	return ConstructU32Composite(state, components, values);
}

void StoreBdaDword(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t value) {
	auto& state = ctx.state;
	EmitIfCondition(state, active, [&]() {
		// RDNA2 DWORD accesses ignore the two low address bits.
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto bda     = GetBdaPointer(state, aligned);
		const auto present =
		    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
		EmitIfCondition(state, present, [&]() {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          bda);
			constexpr uint32_t alignment = sizeof(uint32_t);
			state.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask,
			                          alignment);
			RecordBdaWrite(state, BdaPageIndex(state, aligned));
		});
	});
}

void StoreIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto buffer       = PrepareIndirectBuffer(ctx, inst);
		const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
		                                 ConstantU32(state, 0));
		const auto base = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
		                         ConstantU64(state, ~uint64_t {3}));
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		for (uint32_t component = 0; component < components; component++) {
			auto value = data;
			if (components != 1u) {
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data,
				                          component);
			}
			const auto address = component == 0u ? base : Binary(
			    state, spv::OpIAdd, TypeU64(state), base,
			    ConstantU64(state, component * 4u));
			StoreBdaDword(ctx, address,
			              AndCondition(state, valid_format,
			                           IndirectBufferInBounds(state, buffer, component * 4u, 4u,
			                                                  false)),
			              value);
		}
	});
}

uint32_t LoadIndirectFormattedX(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		return EmitIndexSwitch(
		    state, buffer.format, static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u,
		    TypeU32(state), [&](uint32_t format) {
			    const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			    if (info.type == Format::ComponentType::Unknown) return ConstantU32(state, 0);
			    const auto constant = Select(
			        state, TypeU32(state),
			        Binary(state, spv::OpIEqual, TypeBool(state), buffer.selector, ConstantU32(state, 1)),
			        FormattedConstant(ctx, info, FormattedSourceKind::One), ConstantU32(state, 0));
			    return EmitValueOrDefaultIfCondition(
			        state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), buffer.selector,
			                      ConstantU32(state, 4)),
			        TypeU32(state), constant, [&]() {
				        const auto in_bounds = IndirectBufferInBounds(state, buffer, 0u, info.byte_size, true);
				        const auto base = Binary(
				            state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
				            ConstantU64(state, ~(uint64_t(std::min(4u, info.byte_size)) - 1u)));
				        const auto load = [&](uint32_t component, uint32_t bits) {
					        const auto offset = Format::GetFormatComponentByteOffset(info, component);
					        const auto address = offset == 0u
					                                 ? base
					                                 : Binary(state, spv::OpIAdd, TypeU64(state),
					                                          base, ConstantU64(state, offset));
					        return LoadBda(ctx, address, in_bounds, bits);
				        };
				        const auto emit_component = [&](uint32_t component) {
					        return LoadFormattedComponent(
					            ctx, info, {FormattedSourceKind::Memory, component},
					            [&](uint32_t source) { return load(source, 32u); }, load);
				        };
				        if (info.component_count == 1u) return emit_component(0u);
				        const auto component = Binary(
				            state, spv::OpUMod, TypeU32(state),
				            Binary(state, spv::OpISub, TypeU32(state), buffer.selector, ConstantU32(state, 4)),
				            ConstantU32(state, info.component_count));
				        return EmitIndexSwitch(state, component, info.component_count, TypeU32(state),
				                               emit_component);
			        });
		    });
	});
}

uint32_t LoadBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	if (const auto mem = ctx.Memory(inst);
	    RobustLoad(state, mem.kind) && !mem.formatted && !ShaderRobustLoadsKeepExec()) {
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			values[component] =
			    LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		}
		return ConstructU32Composite(state, components, values);
	}
	if (const auto mem = ctx.Memory(inst);
	    mem.formatted && mem.kind != IR::ResourceKind::IndirectBuffer && BranchlessFormattedLoad(state, mem)) {
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto info     = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type != Format::ComponentType::Unknown) {
			return FormattedLoadSelect(ctx, mem, PrepareFormattedMemory(ctx, inst, mem, resource, info), components);
		}
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto mem      = ctx.Memory(inst);
		    if (mem.kind == IR::ResourceKind::IndirectBuffer) {
			    return LoadIndirectBuffer(ctx, inst, components);
		    }
		    const auto resource = PrepareMemoryResourceAccess(state, mem);
		    const auto info = Format::GetFormatInfo(
		        mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		    if (info.type != Format::ComponentType::Unknown) {
			    const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
			    return EmitValueOrDefaultIfCondition(
			        state, plan.in_bounds, TypeU32Composite(state, components),
			        FormattedOutOfBoundsValue(ctx, mem, plan, components), [&]() {
				        std::array<uint32_t, 4> values {};
				        for (uint32_t component = 0; component < components; component++) {
					        values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
				        }
				        return ConstructU32Composite(state, components, values);
			        });
		    }
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    values[component] =
			        LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem       = ctx.Memory(inst);
		const auto resource  = PrepareMemoryResourceAccess(state, mem);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info = Format::GetFormatInfo(
		    mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		if (info.type != Format::ComponentType::Unknown) {
			StoreFormattedPrepared(ctx, inst, mem, resource, info, composite, components);
			return;
		}
		for (uint32_t component = 0; component < components; component++) {
			const auto data = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite,
			                          component);
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource, data);
		}
	});
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u
			                               ? base
			                               : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                        ConstantU32(state, component * 4u));
			    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
			                                  address, ConstantU32(state, 2));
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                              ConstantU32(state, 2));
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

} // namespace

uint32_t GetBdaPointer(EmitterState& state, uint32_t address) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result,
	                          state.bda_pointer_function, address);
	return result;
}

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	const auto type            = TypeU64(state);
	const auto function_type   = state.builder.Type(spv::OpTypeFunction, type, type);
	state.bda_pointer_function = state.builder.AllocateId();
	const auto address         = state.builder.AllocateId();
	const auto entry_label     = state.builder.AllocateId();
	state.builder.AddName(state.bda_pointer_function, "get_bda_pointer");
	state.builder.AddFunction(spv::OpFunction, type, state.bda_pointer_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	EmitLabel(state, entry_label);

	const auto page          = BdaPageIndex(state, address);
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64),
	                          entry_pointer, state.bda_pagetable_variable, ConstantU32(state, 0),
	                          page);
	const auto base = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, base, entry_pointer);
	const auto missing = Binary(state, spv::OpIEqual, TypeBool(state), base, ConstantU64(state, 0));
	const auto fault_label     = state.builder.AllocateId();
	const auto available_label = state.builder.AllocateId();
	const auto merge_label     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, missing, fault_label, available_label);

	EmitLabel(state, fault_label);
	RecordBdaFault(state, page);
	state.builder.AddFunction(spv::OpBranch, merge_label);

	EmitLabel(state, available_label);
	const auto offset    = Binary(state, spv::OpBitwiseAnd, type, address,
	                              ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1));
	const auto available = Binary(state, spv::OpIAdd, type, base, offset);
	state.builder.AddFunction(spv::OpBranch, merge_label);

	EmitLabel(state, merge_label);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, type, result, ConstantU64(state, 0), fault_label,
	                          available, available_label);
	state.builder.AddFunction(spv::OpReturnValue, result);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

uint32_t EmitAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem = ctx.Memory(inst);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		const auto scope =
		    mem.kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
		const auto old = EmitAtomicOperation(ctx, inst, pointer, scope);
		EmitAtomicMemoryBarrier(ctx.state, mem.kind);
		return old;
	});
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem   = ctx.Memory(inst);
	auto&       state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    const auto resource =
		        PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3]);
		    const auto byte_address = ByteAddress(ctx, inst, mem);
		    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, EmitMemoryElementInBounds(state, resource, index), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto pointer = EmitStorageBufferElementPointer(
			            state, resource, index, TypeStorageBufferElementPointer(state, 64));
			        const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			        EmitAtomicMemoryBarrier(state, mem.kind);
			        return old;
		        });
	    });
}

void EmitSharedAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	EnsureLdsStorage(state);
	EmitIfCondition(state, ctx.Arg(inst, 2), [&]() {
		const auto address = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                            ByteAddress(ctx, inst, mem), ConstantU32(state, 0xfff8u));
		const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
		                          ConstantU32(state, 3u));
		const auto in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index,
		                              ConstantU32(state, LdsDwordCount(state) / 2u));
		EmitIfCondition(state, in_bounds, [&]() {
			auto native_index = index;
			auto semantics = spv::MemorySemanticsWorkgroupMemoryMask;
			if (state.lds_storage_class == spv::StorageClassStorageBuffer) {
				native_index = EmitAddU32(state, index, EmitBinaryU32(
				    state, spv::OpShiftRightLogical, state.lds_base_dwords, ConstantU32(state, 1)));
				semantics = spv::MemorySemanticsUniformMemoryMask;
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpAccessChain, TypePointer(state, state.lds_storage_class, TypeU64(state)),
			    pointer, state.lds_u64_variable, ConstantU32(state, 0), native_index);
			const auto value = ctx.Arg(inst, 1);
			state.builder.AddFunction(
			    SpirvAtomicOpcode(inst.GetOpcode()), TypeU64(state), state.builder.AllocateId(),
			    pointer, ConstantU32(state, spv::ScopeWorkgroup),
			    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | semantics), value);
		});
	});
}

uint32_t EmitBufferFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::BufferAtomicFMax32;
	return EmitAtomicUpdate(ctx, inst, mem,
	                        [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                        return EmitFloatAtomicReplacement(state, old, value, max_value);
	                        });
}

void EmitSharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
	EmitAtomicUpdate(ctx, inst, mem,
	                 [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                 return EmitDsFloatAtomicReplacement(state, old, value, max_value);
	                 });
}

uint32_t EmitAppendConsume(ValueEmitContext& ctx, const IR::Inst& inst) {
	const bool append = inst.GetOpcode() == IR::ValueOpcode::DataAppend;
	auto&      state  = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto mem           = ctx.Memory(inst);
	const auto offset        = mem.offset & 0xfffcu;
	auto       address       = ConstantU32(state, offset);
	uint32_t   region_bounds = 0;
	if (mem.kind == IR::ResourceKind::Gds) {
		const auto m0 = ctx.Arg(inst, 0);
		const auto base =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
		const auto size =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
		address = Binary(state, spv::OpIAdd, TypeU32(state), base, address);
		// The entire M0 region must fit the PS5's 48 KiB GDS partition.
		region_bounds = AndCondition(
		    state, Binary(state, spv::OpULessThan, TypeBool(state),
		                  ConstantU32(state, offset + 3u), size),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state),
		           Binary(state, spv::OpIAdd, TypeU32(state), base, size),
		           ConstantU32(state, 0xc000u)));
	}
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto access = PrepareMemoryResourceAccess(state, mem);
	const auto exec   = ctx.Arg(inst, 1);
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto low    = state.builder.AllocateId();
	const auto high   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	const auto count = Binary(state, spv::OpIAdd, TypeU32(state),
	                          Unary(state, spv::OpBitCount, TypeU32(state), low),
	                          Unary(state, spv::OpBitCount, TypeU32(state), high));
	const auto first = ctx.FirstLane(ballot);
	const auto source_lane =
	    state.lane_count == 2
	        ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 31))
	        : first;
	const auto is_first       = Binary(state, spv::OpIEqual, TypeBool(state),
	                                   EmitSubgroupLocalInvocationId(state), source_lane);
	auto bounds = EmitMemoryElementInBounds(state, access, index);
	if (mem.kind == IR::ResourceKind::Gds) {
		bounds = AndCondition(state, bounds, region_bounds);
	}
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 state.lane_count == 2 ? Binary(state, spv::OpINotEqual, TypeBool(state), count,
	                                                ConstantU32(state, 0))
	                                       : exec,
	                 bounds));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(append ? spv::OpAtomicIAdd : spv::OpAtomicISub, TypeU32(state),
		                          value, EmitMemoryElementPointer(state, access, index),
		                          ConstantU32(state, mem.kind == IR::ResourceKind::Gds
		                                                 ? spv::ScopeDevice
		                                                 : spv::ScopeWorkgroup),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), count);
		return value;
	});
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), atomic, source_lane);
	return result;
}

uint32_t EmitReadConst(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.flattened_srt_variable == 0) {
		ctx.Fail(inst, "requires the flattened SRT descriptor");
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.flattened_srt_variable, ConstantU32(state, 0),
	                          ctx.Arg(inst, 1));
	return EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
}

void EmitReadConstBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto mem = ctx.Memory(inst);
	if (mem.planning_only) return;
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		ctx.Define(inst, LoadIndirectScalarBuffer(ctx, inst));
		return;
	}
	auto& state        = ctx.state;
	mem.kind           = IR::ResourceKind::ScalarBuffer;
	auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), ctx.Arg(inst, 1),
	                    ConstantU32(state, 2));
	if (mem.offset >= 4u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, mem.offset >> 2u));
	}
	const auto access    = PrepareMemoryResourceAccess(state, mem);
	// Scalar buffer addressing aligns the base and each offset independently.
	const auto element =
	    ShaderAlignedBufferOffsets()
	        ? index
	        : Binary(state, spv::OpIAdd, TypeU32(state), index,
	                 Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byte_offset,
	                        ConstantU32(state, 2)));
	if ((RobustLoad(state, mem.kind) && ShaderRobustConstReads()) || ShaderSkipBoundsChecks()) {
		// As LoadWordPrepared: out of range reads zero on the device. A branch per constant kept
		// each scalar load in a block of its own, so the compiler could neither batch nor hoist them
		// (the heaviest NHL 27 compute shader had 189 such loads).
		ctx.Define(inst, EmitNative<spv::OpLoad, IR::Type::U32>(state, EmitMemoryElementPointer(state, access, element)));
		return;
	}
	const auto condition = EmitMemoryElementInBounds(state, access, element);
	ctx.Define(inst, EmitValueOrZeroIfCondition(state, condition, [&]() {
		           return EmitNative<spv::OpLoad, IR::Type::U32>(
		               state, EmitMemoryElementPointer(state, access, element));
	           }));
}

void EmitLoadMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op  = inst.GetOpcode();
	const auto& mem = ctx.Memory(inst);
	if (op == IR::ValueOpcode::LoadAddressU32 && mem.planning_only) return;
	const auto buffer_components = IR::BufferComponentCount(op);
	const auto shared_components = IR::SharedComponentCount(op);
	const auto address_info      = IR::AddressOpcodeInfoOf(op);
	uint32_t   value;
	if (mem.kind == IR::ResourceKind::FlatLocal)
		value = LoadLocalFlat(ctx, inst);
	else if (buffer_components > 1u ||
	         (buffer_components == 1u && mem.kind == IR::ResourceKind::IndirectBuffer && !mem.formatted))
		value = LoadBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		value = LoadWideShared(ctx, inst, shared_components);
	else if (const auto grouped = BdaGroupable(ctx, inst) ? LoadGroupedBda(ctx, inst) : 0u;
	         grouped != 0u)
		value = grouped;
	else if (mem.kind == IR::ResourceKind::ScalarAddress)
		value = LoadBdaDword(ctx, GuestAddress(ctx, inst, mem));
	else if (address_info.access == IR::AddressAccess::Read &&
	         mem.kind != IR::ResourceKind::Scratch)
		value = LoadBda(ctx, GuestAddress(ctx, inst, mem), ctx.Arg(inst, inst.NumArgs() - 1),
		                address_info.data_bits, mem.coherent);
	else if (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted)
		value = mem.kind == IR::ResourceKind::IndirectBuffer ? LoadIndirectFormattedX(ctx, inst)
		                                                     : FormattedLoad(ctx, inst, mem);
	else if (inst.GetType() == IR::Type::U8)
		value = LoadSubword(ctx, inst, mem, 8, false);
	else if (inst.GetType() == IR::Type::U16)
		value = LoadSubword(ctx, inst, mem, 16, false);
	else
		value = LoadWord(ctx, inst, mem);
	ctx.Define(inst, value);
}

void EmitStoreMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op                = inst.GetOpcode();
	const auto& mem               = ctx.Memory(inst);
	const auto  buffer_components = IR::BufferComponentCount(op);
	const auto  shared_components = IR::SharedComponentCount(op);
	const auto  type              = inst.Arg(inst.NumArgs() - 2).GetType();
	if (mem.kind == IR::ResourceKind::FlatLocal)
		StoreLocalFlat(ctx, inst);
	else if (mem.kind == IR::ResourceKind::IndirectBuffer)
		StoreIndirectBuffer(ctx, inst, std::max(buffer_components, 1u));
	else if (buffer_components > 1u)
		StoreWideBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		StoreWideShared(ctx, inst, shared_components);
	else if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
		FormattedStore(ctx, inst, mem);
	else if (type == IR::Type::U8)
		StoreSubword(ctx, inst, mem, 8);
	else if (type == IR::Type::U16)
		StoreSubword(ctx, inst, mem, 16);
	else
		StoreWord(ctx, inst, mem);
}

uint32_t EmitSharedIncDec(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto replacement =
	    inst.GetOpcode() == IR::ValueOpcode::SharedAtomicInc32 ? AtomicIncrement : AtomicDecrement;
	return EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement);
}

void EmitSharedAtomicMaskedOr32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto keep = Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), ctx.Arg(inst, 1));
	EmitAtomicUpdate(ctx, inst, ctx.Memory(inst),
	                 [keep](EmitterState& state, uint32_t old, uint32_t value) {
		                 return Binary(state, spv::OpBitwiseOr, TypeU32(state),
		                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old, keep), value);
	                 });
}

uint32_t EmitSwizzleU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	state.builder.AddFunction(spv::OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 0));
	const auto source = EmitNative<spv::OpLoad, IR::Type::U32>(state, ctx.scratch_u32_variable);
	const auto target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
	                                            inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

uint32_t EmitPermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state   = ctx.state;
	auto       lane    = EmitSubgroupLocalInvocationId(state);
	if (state.lane_count == 2) {
		lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane, ConstantU32(state, 31));
	}
	const auto address = ctx.Arg(inst, 1);
	const auto word    = Binary(state, spv::OpShiftRightLogical, TypeU32(state), lane,
	                            ConstantU32(state, 5));
	const auto ballot_word = [&](uint32_t predicate) {
		const auto ballot = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), predicate);
		state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), result, ballot, word);
		return result;
	};
	// RDNA2 permutes independently within each 32-lane half. Intersect the source
	// address bit ballots to find this destination's enabled writers without LDS.
	auto writers = ballot_word(ctx.Arg(inst, 2));
	for (uint32_t bit = 0; bit < 5; ++bit) {
		const auto address_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), address,
		                                ConstantU32(state, 1u << (bit + 2)));
		const auto mask = ballot_word(Binary(state, spv::OpINotEqual, TypeBool(state),
		                                      address_bit, ConstantU32(state, 0)));
		const auto lane_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
		                             ConstantU32(state, 1u << bit));
		const auto selected = Select(
		    state, TypeU32(state),
		    Binary(state, spv::OpINotEqual, TypeBool(state), lane_bit, ConstantU32(state, 0)),
		    mask, Unary(state, spv::OpNot, TypeU32(state), mask));
		writers = Binary(state, spv::OpBitwiseAnd, TypeU32(state), writers, selected);
	}
	const auto active = Binary(state, spv::OpINotEqual, TypeBool(state), writers,
	                           ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
	                         ConstantU32(state, ~31u));
	const auto source = Select(state, TypeU32(state), active,
	                           Binary(state, spv::OpBitwiseOr, TypeU32(state), base,
	                                  EmitFindUMsb32(state, writers)), lane);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 0), source);
	return Select(state, TypeU32(state), active, result, ConstantU32(state, 0));
}

uint32_t EmitBpermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto source = ctx.Arg(inst, 0);
	const auto index  = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                  ctx.Arg(inst, 1), ConstantU32(state, 2)),
	                           ConstantU32(state, 31));
	const auto base   = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           EmitSubgroupLocalInvocationId(state), ConstantU32(state, ~31u));
	const auto target = Binary(state, spv::OpBitwiseOr, TypeU32(state), base, index);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

uint32_t EmitRegArrayLoad(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitNative<spv::OpLoad, IR::Type::U32>(ctx.state, RegArrayElement(ctx, inst));
}

void EmitRegArrayStore(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.state.builder.AddFunction(spv::OpStore, RegArrayElement(ctx, inst), ctx.Def(inst.Arg(2)));
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
