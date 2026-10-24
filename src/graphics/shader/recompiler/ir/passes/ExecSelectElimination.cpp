#include "graphics/shader/recompiler/ir/passes/ExecSelectElimination.h"

#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

// Lanes in which a value can be observed: none, only where one exec value is set, or any lane.
struct Demand {
	enum Kind : uint8_t { None, Mask, All };
	Kind        kind = None;
	const Inst* mask = nullptr;
};

// Joins `add` into `into`; true when `into` changed.
bool Join(Demand& into, const Demand& add) {
	if (add.kind == Demand::None || into.kind == Demand::All) return false;
	if (into.kind == Demand::None) {
		into = add;
		return true;
	}
	if (add.kind == Demand::Mask && into.mask == add.mask) return false;
	into = {Demand::All, nullptr};
	return true;
}

// Operations whose result in a lane depends only on their operands in that lane.
bool IsLaneWise(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::Identity:
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU16F16:
		case ValueOpcode::BitCastF16U16:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::BitCastU64F64:
		case ValueOpcode::BitCastF64U64:
		case ValueOpcode::ConvertU16U32:
		case ValueOpcode::ConvertU32U16:
		case ValueOpcode::ConvertU8U32:
		case ValueOpcode::ConvertU32U8:
		case ValueOpcode::ConvertF32F16:
		case ValueOpcode::ConvertF16F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32S32:
		case ValueOpcode::ConvertF64S32:
		case ValueOpcode::ConvertF32F64:
		case ValueOpcode::ConvertF64F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertF64U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeConstructF32x2:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::PackHalf2x16:
		case ValueOpcode::PackSnorm2x16:
		case ValueOpcode::PackUnorm2x16:
		case ValueOpcode::PackFloat2x16Rtz:
		case ValueOpcode::FPAbs32:
		case ValueOpcode::FPNeg32:
		case ValueOpcode::FPSaturate32:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UDiv32:
		case ValueOpcode::SMulHi:
		case ValueOpcode::UMulHi:
		case ValueOpcode::IAbs32:
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
		case ValueOpcode::BitReverse32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::BitCount64:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::FindUMsb64:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::SMin32:
		case ValueOpcode::UMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMinTri32:
		case ValueOpcode::UMinTri32:
		case ValueOpcode::SMaxTri32:
		case ValueOpcode::UMaxTri32:
		case ValueOpcode::SMedTri32:
		case ValueOpcode::UMedTri32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThan64:
		case ValueOpcode::SLessThanEqual64:
		case ValueOpcode::ULessThanEqual64:
		case ValueOpcode::UGreaterThanEqual64:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThan64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThan64:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPOrdEqual64:
		case ValueOpcode::FPOrdLessThanEqual64:
		case ValueOpcode::FPOrdGreaterThanEqual64:
		case ValueOpcode::FPUnordEqual32:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPUnordNotEqual32:
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPUnordLessThan32:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPUnordGreaterThan32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPUnordLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPUnordGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPCmpClass16:
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPFma32:
		case ValueOpcode::FPMad32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPAdd16x2:
		case ValueOpcode::FPMul16x2:
		case ValueOpcode::FPFma16x2:
		case ValueOpcode::FPClamp16x2:
		case ValueOpcode::F16LaneExtract:
		case ValueOpcode::F16LaneInsert:
		case ValueOpcode::FPAdd16Low:
		case ValueOpcode::FPSub16Low:
		case ValueOpcode::FPMul16Low:
		case ValueOpcode::FPFma16Low:
		case ValueOpcode::F16x2Swizzle:
		case ValueOpcode::FPUnary16Low:
		case ValueOpcode::FPMinMax16Low:
		case ValueOpcode::FPMinMax16x2:
		case ValueOpcode::FPAdd64:
		case ValueOpcode::FPMul64:
		case ValueOpcode::FPFma64:
		case ValueOpcode::FPRecip64:
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMin64:
		case ValueOpcode::FPMax64:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32:
		case ValueOpcode::FPMedTri32:
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPRecipSqrt32:
		case ValueOpcode::FPSqrt:
		case ValueOpcode::FPSin:
		case ValueOpcode::FPCos:
		case ValueOpcode::FPExp2:
		case ValueOpcode::FPLog2:
		case ValueOpcode::FPLdexp:
		case ValueOpcode::FPRoundEven32:
		case ValueOpcode::FPFloor32:
		case ValueOpcode::FPCeil32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPFloor64:
		case ValueOpcode::FPCeil64:
		case ValueOpcode::FPTrunc64:
		case ValueOpcode::FPFract32:
		case ValueOpcode::FPFract64: return true;
		default: return false;
	}
}

bool IsSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectF32;
}

// Memory operations take the exec mask as their last operand; the emitter only touches memory (and
// only reads the other operands) in lanes where it is set.
bool IsExecPredicatedMemory(const Inst& inst) {
	const auto op = inst.GetOpcode();
	const bool memory = BufferAccessOf(op) != BufferAccess::None || SharedAccessOf(op) != SharedAccess::None ||
	                    AddressOpcodeInfoOf(op).access != AddressAccess::None ||
	                    ImageOpcodeInfoOf(op).access != ImageAccess::None;
	return memory && inst.NumArgs() != 0 && NumArgsOf(op) == inst.NumArgs() &&
	       ArgTypeOf(op, inst.NumArgs() - 1) == Type::U1;
}

// Debugging: KYTY_DBG_EXECSEL_MODE (bits: 1 select new-operand rule, 2 same-condition folding,
// 4 forward phis, 8 image coordinates as lane-wise, 16 memory operands under exec) selects the
// rules. Default 16: the rules 1, 2 and 4 break NHL 27 when applied to every shader (under study).
uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_DBG_EXECSEL_MODE");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0)) : 16u;
	}();
	return mode;
}

// Sampling with implicit derivatives reads the coordinates of the other lanes of its quad.
bool ReadsNeighborLanes(ValueOpcode op) {
	return op == ValueOpcode::ImageSampleRaw || op == ValueOpcode::ImageGatherRaw ||
	       op == ValueOpcode::ImageQueryLod;
}

} // namespace

bool IsLaneWiseOpcode(ValueOpcode op) {
	return IsLaneWise(op);
}

bool IsExecPredicatedMemoryOp(const Inst& inst) {
	return IsExecPredicatedMemory(inst);
}

bool ReadsNeighborLanesOpcode(ValueOpcode op) {
	return ReadsNeighborLanes(op);
}

uint32_t EliminateExecSelects(const BlockList& blocks) {
	std::vector<Inst*>                       insts;
	std::unordered_map<const Inst*, uint32_t> index;
	for (auto* block: blocks) {
		for (auto& inst: *block) {
			index.emplace(&inst, static_cast<uint32_t>(insts.size()));
			insts.push_back(&inst);
		}
	}
	// A register written twice under the same exec value: select(c, x, select(c, y, z)) reads only z
	// where c is clear, and select(c, select(c, x, y), z) only x where c is set.
	for (auto* inst: insts) {
		if ((Mode() & 2u) == 0 || !IsSelect(inst->GetOpcode()) || inst->NumArgs() != 3) continue;
		const auto condition = inst->Arg(0);
		if (condition.TryInstruction() == nullptr) continue;
		for (size_t arg = 1; arg <= 2; ++arg) {
			for (uint32_t depth = 0; depth < 64; ++depth) {
				const auto* inner = inst->Arg(arg).TryInstruction();
				if (inner == nullptr || !IsSelect(inner->GetOpcode()) || inner->NumArgs() != 3 ||
				    inner->Arg(0) != condition) {
					break;
				}
				inst->SetArg(arg, inner->Arg(arg));
			}
		}
	}
	// Reverse post-order from the entry: every edge into a block with a smaller or equal number
	// is retreating, and every cycle contains one.
	std::unordered_map<const Block*, uint32_t> rpo;
	if (!blocks.empty()) {
		std::vector<const Block*>                    post;
		std::unordered_set<const Block*>             seen {blocks.front()};
		std::vector<std::pair<const Block*, size_t>> stack {{blocks.front(), 0}};
		while (!stack.empty()) {
			const auto* block = stack.back().first;
			const auto  succ  = block->ImmSuccessors();
			if (stack.back().second < succ.size()) {
				const Block* next = succ[stack.back().second++];
				if (seen.insert(next).second) stack.emplace_back(next, 0);
			} else {
				post.push_back(block);
				stack.pop_back();
			}
		}
		for (size_t i = 0; i < post.size(); ++i) rpo[post[i]] = static_cast<uint32_t>(post.size() - 1 - i);
	}
	std::vector<Demand>   demand(insts.size());
	std::vector<uint32_t> pending;
	const auto demand_value = [&](Value value, const Demand& add) {
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return;
		const auto found = index.find(inst);
		if (found == index.end()) return;
		if (Join(demand[found->second], add)) pending.push_back(found->second);
	};
	const Demand all {Demand::All, nullptr};
	// Roots: side effects, predicated memory operations, and values used outside the blocks.
	for (uint32_t i = 0; i < insts.size(); ++i) {
		const auto& inst = *insts[i];
		for (const auto& use: inst.Uses()) {
			if (use.user->Parent() == nullptr && Join(demand[i], all)) pending.push_back(i);
		}
		if (IsExecPredicatedMemory(inst) && (ReadsNeighborLanes(inst.GetOpcode()) && (Mode() & 8u) == 0)) {
			for (size_t arg = 0; arg < inst.NumArgs(); ++arg) demand_value(inst.Arg(arg), all);
		} else if (IsExecPredicatedMemory(inst) && (Mode() & 16u) != 0) {
			const auto  exec = inst.Arg(inst.NumArgs() - 1);
			const auto* mask = exec.TryInstruction();
			const Demand under = mask != nullptr ? Demand {Demand::Mask, mask} : all;
			for (size_t arg = 0; arg + 1 < inst.NumArgs(); ++arg) demand_value(inst.Arg(arg), under);
			demand_value(exec, all);
		} else if (inst.MayHaveSideEffects() || IsExecPredicatedMemory(inst)) {
			for (size_t arg = 0; arg < inst.NumArgs(); ++arg) demand_value(inst.Arg(arg), all);
		}
	}
	while (!pending.empty()) {
		const auto i = pending.back();
		pending.pop_back();
		const auto& inst = *insts[i];
		const auto  d    = demand[i];
		const auto  op   = inst.GetOpcode();
		if (d.kind == Demand::None || IsExecPredicatedMemory(inst)) continue;
		if (op == ValueOpcode::Phi) {
			// Along a forward edge the value is observed in the same pass through the code. A
			// retreating (loop) edge can carry it to a later iteration's different exec: any lane.
			const auto self = rpo.find(inst.Parent());
			for (size_t arg = 0; arg < inst.NumPhiBlocks(); ++arg) {
				const auto pred    = rpo.find(inst.PhiBlock(arg));
				const bool forward = (Mode() & 4u) != 0 && self != rpo.end() && pred != rpo.end() &&
				                     pred->second < self->second;
				demand_value(inst.Arg(arg), forward ? d : all);
			}
		} else if (IsSelect(op) && inst.NumArgs() == 3) {
			// The select reads its new value only where the condition is set, and its old value only
			// where it is clear: a result needed in every lane still needs the new value only under
			// the condition.
			const auto* condition = inst.Arg(0).TryInstruction();
			demand_value(inst.Arg(0), d);
			demand_value(inst.Arg(1), condition != nullptr && (Mode() & 1u) != 0 ? Demand {Demand::Mask, condition} : d);
			if (!(d.kind == Demand::Mask && condition != nullptr && d.mask == condition)) {
				demand_value(inst.Arg(2), d);
			}
		} else if (IsLaneWise(op)) {
			for (size_t arg = 0; arg < inst.NumArgs(); ++arg) demand_value(inst.Arg(arg), d);
		} else {
			for (size_t arg = 0; arg < inst.NumArgs(); ++arg) demand_value(inst.Arg(arg), all);
		}
	}
	// KYTY_DBG_EXECSEL=1: why selects stay (direct users of kept selects, by opcode and demand).
	static const bool debug = std::getenv("KYTY_DBG_EXECSEL") != nullptr;
	if (debug) {
		std::map<std::string, uint32_t> reasons;
		for (uint32_t i = 0; i < insts.size(); ++i) {
			const auto& inst = *insts[i];
			if (!IsSelect(inst.GetOpcode()) || inst.NumArgs() != 3 || demand[i].kind != Demand::All) continue;
			for (const auto& use: inst.Uses()) {
				std::string key(ValueOpcodeName(use.user->GetOpcode()));
				const auto  found = index.find(use.user);
				if (found != index.end()) {
					const auto& ud = demand[found->second];
					key += ud.kind == Demand::All ? ":all" : ud.kind == Demand::Mask ? ":mask" : ":none";
					if (ud.kind == Demand::Mask) key += ud.mask == inst.Arg(0).TryInstruction() ? "=same" : "=other";
				}
				if (IsSelect(use.user->GetOpcode())) key += use.operand == 2 ? "(as old)" : "(as new)";
				++reasons[key];
			}
		}
		std::string line;
		for (const auto& [key, count]: reasons) line += " " + key + "=" + std::to_string(count);
		std::printf("NHL27EXECSELWHY:%s\n", line.c_str());
	}
	uint32_t removed = 0;
	for (uint32_t i = 0; i < insts.size(); ++i) {
		auto& inst = *insts[i];
		if (!IsSelect(inst.GetOpcode()) || inst.NumArgs() != 3 || !inst.HasUses()) continue;
		const auto  d         = demand[i];
		const auto* condition = inst.Arg(0).TryInstruction();
		if (d.kind == Demand::None || (d.kind == Demand::Mask && condition != nullptr && d.mask == condition)) {
			inst.ReplaceUsesWith(inst.Arg(1), true);
			++removed;
		}
	}
	return removed;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
