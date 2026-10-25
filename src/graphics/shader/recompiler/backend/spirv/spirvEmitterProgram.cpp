#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/passes/ExecSelectElimination.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

void EmitKillIfBoolFalse(EmitterState& state, uint32_t active) {
	const auto kill_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, active, merge_label, kill_label);
	EmitLabel(state, kill_label);
	state.builder.AddFunction(spv::OpKill);
	EmitLabel(state, merge_label);
}

void EmitKillIfPixelValidMaskInactive(EmitterState& state) {
	if (state.pixel_valid_mask_variable == 0) {
		return;
	}

	const auto mask_value = state.builder.AllocateId();
	const auto active     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), mask_value,
	                          state.pixel_valid_mask_variable);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, mask_value,
	                          ConstantU32(state, 0));
	EmitKillIfBoolFalse(state, active);
}

uint32_t SpillPointerType(ValueEmitContext& ctx, IR::Type type) {
	const auto value_type = TypeId(ctx.state, type);
	return value_type == 0 ? 0 : TypePointer(ctx.state, spv::StorageClassFunction, value_type);
}

struct DeferredPhiPatch {
	DeferredPhi     phi;
	const IR::Inst* instruction = nullptr;
	uint32_t        half        = 0;
};

// Experiment LaneSkips: a selection skipped by a branch on EXEC or VCC (s_cbranch_execz around an
// "if") tests its condition through a ballot, so every lane runs the body whenever one needs it, and
// the body's VGPR writes select each new value against the old one under EXEC. The host compiler
// keeps both values live through the body, which in NHL 27's material shaders doubled the registers
// of their light loops (33037c60: 191 VGPRs, 145 with these branches per lane). Where the body
// cannot tell the difference, each lane enters it on its own condition instead: the lanes without it
// would only run instructions masked off for them, so the body must
// - stay inside the selection (no return, no branch out but to its merge);
// - mask every side effect, ballot, first-lane read and EXEC/VCC branch so that lanes outside the
//   condition take no part (their EXEC argument implies the condition);
// - not read other lanes (DPP, permutes, lane reads and writes, reductions);
// - leave the merge's phis, where the lanes outside the condition take the old value, with a value
//   equal to it in those lanes (a select under such an EXEC of anything over the old value).
// Merge values the body computed for the whole wave are repaired after the merge's phis instead: a
// lane mask the condition implies (VCC of a V_CMP under EXEC) is false in the lanes outside once any
// lane entered, and a value the same in every lane that entered (an SGPR) is taken from the first of
// them. Any other merge value keeps the branch on the ballot.
bool IsBallotCondition(CFG::BranchCondition kind) {
	return kind == CFG::BranchCondition::ExecZero || kind == CFG::BranchCondition::ExecNonZero ||
	       kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::VccNonZero;
}

bool IsZeroCondition(CFG::BranchCondition kind) {
	return kind == CFG::BranchCondition::ExecZero || kind == CFG::BranchCondition::VccZero;
}

struct LaneSkip {
	IR::Value lane;          // the per-lane condition of entering the body is `lane`, or its negation
	bool      negate = false;
	// Merge phis repaired after the merge's phis: true for a lane mask, false for a wave value.
	std::vector<std::pair<const IR::Inst*, bool>> fixes;
};

// KYTY_DBG_LANE_SKIPS=2: what a merge value, inner branch or side effect that kept a selection on its
// ballot was.
thread_local std::string* t_lane_skip_detail = nullptr;

std::string DescribeLaneSkipValue(IR::Value value, int depth = 2);

class LaneSkipAnalysis {
public:
	// KYTY_DBG_LANE_SKIPS=2: in the last implication asked, the first value it failed on that is not a
	// combination of others.
	std::string failed_leaf;

	// `per_lane_merges`: the merges of the selections inside that branch per lane already, with the
	// phis whose wave value they repair (those stay the same in every lane).
	LaneSkipAnalysis(IR::Value lane, bool negate,
	                 const std::unordered_map<const IR::Block*, std::unordered_set<const IR::Inst*>>& per_lane_merges)
	    : lane_(lane), negate_(negate), per_lane_merges_(per_lane_merges) {}

	// The value (or with `positive` false its negation) is false wherever the lane condition is.
	bool Implies(IR::Value value, bool positive) {
		failed_leaf.clear();
		implies_.Begin();
		const bool result = ImpliesRec(value, positive);
		implies_.Commit();
		return result;
	}

	// `value` equals `old` wherever the lane condition is false.
	bool Masked(IR::Value value, IR::Value old) {
		masked_ = {};
		masked_.Begin();
		return MaskedRec(value, old.Resolve());
	}

	// `value` is the same in every lane that computes it.
	bool Uniform(IR::Value value) {
		uniform_.Begin();
		const bool result = UniformRec(value);
		uniform_.Commit();
		return result;
	}

private:
	// Answers about instructions, where phis (their loops carry values back) are assumed to hold
	// while they are being decided. Answers are monotone in those assumptions, so a "no" is final at
	// once; a "yes" stays provisional on a trail, which a phi that turns out "no" rolls back to where
	// it started, and is final when the outermost question ends.
	struct Memo {
		std::unordered_map<uint64_t, bool> final;
		std::unordered_map<uint64_t, bool> provisional; // false: being decided
		std::vector<uint64_t>              trail;
		void Begin() {
			provisional.clear();
			trail.clear();
		}
		void Commit() {
			for (const auto key: trail) final[key] = true;
			Begin();
		}
		// 1 yes, 0 no, -1 not known yet.
		int Find(uint64_t key) const {
			if (const auto found = final.find(key); found != final.end()) return found->second ? 1 : 0;
			return provisional.contains(key) ? 1 : -1;
		}
		void Yes(uint64_t key) {
			provisional[key] = true;
			trail.push_back(key);
		}
		void No(uint64_t key, size_t mark) {
			for (size_t index = mark; index < trail.size(); index++) provisional.erase(trail[index]);
			trail.resize(mark);
			provisional.erase(key);
			final[key] = false;
		}
	};

	// kind: 0/1 implication of the negation/the value, 2 implication of a lane-mask word's bits, 3 uniform.
	static uint64_t Key(const IR::Inst* inst, uint32_t kind) {
		return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(inst)) * 4u + kind;
	}

	static const IR::Inst* Producer(IR::Value value, IR::ValueOpcode opcode) {
		value             = value.Resolve();
		const auto* inst  = value.IsImmediate() ? nullptr : value.TryInstruction();
		return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
	}

	static bool IsImmediateU32(IR::Value value, uint32_t expected) {
		value = value.Resolve();
		return value.IsImmediate() && value.GetType() == IR::Type::U32 && value.U32() == expected;
	}

	// The lane's bit of a lane mask (Translator::ThreadBit): its mask words, or false.
	static bool MatchThreadBit(const IR::Inst& inst, std::array<IR::Value, 2>& words) {
		if (inst.GetOpcode() != IR::ValueOpcode::INotEqual32 || !IsImmediateU32(inst.Arg(1), 0u)) return false;
		const auto* masked = Producer(inst.Arg(0), IR::ValueOpcode::BitwiseAnd32);
		if (masked == nullptr || !IsImmediateU32(masked->Arg(1), 1u)) return false;
		const auto* shifted = Producer(masked->Arg(0), IR::ValueOpcode::ShiftRightLogical32);
		if (shifted == nullptr) return false;
		const auto* bit = Producer(shifted->Arg(1), IR::ValueOpcode::BitwiseAnd32);
		if (bit == nullptr || Producer(bit->Arg(0), IR::ValueOpcode::LaneId) == nullptr ||
		    !IsImmediateU32(bit->Arg(1), 31u)) {
			return false;
		}
		const auto word = shifted->Arg(0).Resolve();
		if (const auto* half = Producer(word, IR::ValueOpcode::SelectU32)) {
			const auto* low = Producer(half->Arg(0), IR::ValueOpcode::ULessThan32);
			if (low != nullptr && Producer(low->Arg(0), IR::ValueOpcode::LaneId) != nullptr &&
			    IsImmediateU32(low->Arg(1), 32u)) {
				words = {half->Arg(1), half->Arg(2)};
				return true;
			}
		}
		words = {word, word};
		return true;
	}

	bool ImpliesRec(IR::Value value, bool positive) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			const bool result = value.GetType() == IR::Type::U1 && value.U1() != positive;
			if (!result && t_lane_skip_detail != nullptr && failed_leaf.size() < 600) failed_leaf += " imm";
			return result;
		}
		if (value == lane_) return positive != negate_;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		const auto key = Key(inst, positive ? 1u : 0u);
		if (const auto known = implies_.Find(key); known >= 0) return known != 0;
		const auto mark = implies_.trail.size();
		bool       result = false;
		std::array<IR::Value, 2> words;
		switch (inst->GetOpcode()) {
			case IR::ValueOpcode::LogicalNot: result = ImpliesRec(inst->Arg(0), !positive); break;
			case IR::ValueOpcode::LogicalAnd:
			case IR::ValueOpcode::LogicalOr: {
				// A conjunction implies it when either side does, a disjunction when both do.
				const bool any = (inst->GetOpcode() == IR::ValueOpcode::LogicalAnd) == positive;
				result         = ImpliesRec(inst->Arg(0), positive);
				if (result != any) result = ImpliesRec(inst->Arg(1), positive);
				break;
			}
			case IR::ValueOpcode::SelectU1:
				result = ImpliesRec(inst->Arg(1), positive) && ImpliesRec(inst->Arg(2), positive);
				break;
			case IR::ValueOpcode::Phi:
				implies_.provisional[key] = false;
				result                    = true;
				for (size_t index = 0; index < inst->NumArgs() && result; index++) {
					result = ImpliesRec(inst->Arg(index), positive);
				}
				break;
			default:
				// A lane's bit of a saved mask (an EXEC restored from an SGPR pair).
				result = positive && MatchThreadBit(*inst, words) && MaskImpliesRec(words[0]) &&
				         MaskImpliesRec(words[1]);
				if (!result && t_lane_skip_detail != nullptr && failed_leaf.size() < 600) {
					failed_leaf += (positive ? " " : " !") + DescribeLaneSkipValue(value, 3);
				}
				break;
		}
		if (result) {
			implies_.Yes(key);
		} else {
			implies_.No(key, mark);
		}
		return result;
	}

	// Every set bit of the lane-mask word is a lane where the condition holds.
	bool MaskImpliesRec(IR::Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) return value.GetType() == IR::Type::U32 && value.U32() == 0u;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		const auto key = Key(inst, 2u);
		if (const auto known = implies_.Find(key); known >= 0) return known != 0;
		const auto mark = implies_.trail.size();
		bool       result = false;
		switch (inst->GetOpcode()) {
			case IR::ValueOpcode::CompositeExtractU32x4: {
				const auto* ballot = Producer(inst->Arg(0), IR::ValueOpcode::Ballot);
				result             = ballot != nullptr && ImpliesRec(ballot->Arg(0), true);
				break;
			}
			case IR::ValueOpcode::BitwiseAnd32:
				result = MaskImpliesRec(inst->Arg(0)) || MaskImpliesRec(inst->Arg(1));
				break;
			case IR::ValueOpcode::BitwiseOr32:
				result = MaskImpliesRec(inst->Arg(0)) && MaskImpliesRec(inst->Arg(1));
				break;
			case IR::ValueOpcode::SelectU32:
				result = MaskImpliesRec(inst->Arg(1)) && MaskImpliesRec(inst->Arg(2));
				break;
			case IR::ValueOpcode::Phi:
				implies_.provisional[key] = false;
				result                    = true;
				for (size_t index = 0; index < inst->NumArgs() && result; index++) {
					result = MaskImpliesRec(inst->Arg(index));
				}
				break;
			default: break;
		}
		if (result) {
			implies_.Yes(key);
		} else {
			implies_.No(key, mark);
		}
		return result;
	}

	bool MaskedRec(IR::Value value, IR::Value old) {
		value = value.Resolve();
		if (value == old) return true;
		if (value.GetType() == IR::Type::U1 && Implies(value, true) && Implies(old, true)) return true;
		const auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
		if (inst == nullptr) return false;
		const auto key = Key(inst, 1u);
		if (const auto known = masked_.Find(key); known >= 0) return known != 0;
		const auto mark   = masked_.trail.size();
		bool       result = false;
		switch (inst->GetOpcode()) {
			case IR::ValueOpcode::SelectU1:
			case IR::ValueOpcode::SelectU32:
			case IR::ValueOpcode::SelectF32:
				result = Implies(inst->Arg(0), true)
				             ? MaskedRec(inst->Arg(2), old)
				             : MaskedRec(inst->Arg(1), old) && MaskedRec(inst->Arg(2), old);
				break;
			case IR::ValueOpcode::Phi:
				masked_.provisional[key] = false;
				result                   = true;
				for (size_t index = 0; index < inst->NumArgs() && result; index++) {
					result = MaskedRec(inst->Arg(index), old);
				}
				break;
			default: break;
		}
		if (result) {
			masked_.Yes(key);
		} else {
			masked_.No(key, mark);
		}
		return result;
	}

	bool UniformRec(IR::Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) return true;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return true; // a register number
		const auto key = Key(inst, 3u);
		if (const auto known = uniform_.Find(key); known >= 0) return known != 0;
		const auto mark   = uniform_.trail.size();
		bool       result = true;
		switch (inst->GetOpcode()) {
			// Per-lane inputs and reads of other lanes.
			case IR::ValueOpcode::LaneId:
			case IR::ValueOpcode::GetAttribute:
			case IR::ValueOpcode::GetInterpolationParameter:
			case IR::ValueOpcode::GetVectorRegister:
			case IR::ValueOpcode::GetBuiltin:
			case IR::ValueOpcode::GetTessellationAttribute:
			case IR::ValueOpcode::GetThreadBitScalarRegister:
			case IR::ValueOpcode::GetScalarMaskTag:
			case IR::ValueOpcode::DppMoveU32:
			case IR::ValueOpcode::DppUpdateU32:
			case IR::ValueOpcode::Permlane16U32:
			case IR::ValueOpcode::PermuteU32:
			case IR::ValueOpcode::BpermuteU32:
			case IR::ValueOpcode::SwizzleU32:
			case IR::ValueOpcode::WriteLane:
			case IR::ValueOpcode::RegArrayLoad: result = false; break;
			// Wave operations give every lane the same value.
			case IR::ValueOpcode::Ballot:
			case IR::ValueOpcode::ReadFirstLane:
			case IR::ValueOpcode::WaveReduceU32: break;
			case IR::ValueOpcode::ReadLane: result = UniformRec(inst->Arg(1)); break;
			case IR::ValueOpcode::ConditionRef:
				result = IsBallotCondition(inst->Flags<CFG::BranchCondition>()) || UniformRec(inst->Arg(0));
				break;
			case IR::ValueOpcode::Phi: {
				// Lanes join a per-lane selection's merge from both sides: only its repaired wave values
				// are the same in every lane there.
				if (const auto merge = per_lane_merges_.find(inst->Parent()); merge != per_lane_merges_.end()) {
					result = merge->second.contains(inst);
					break;
				}
				uniform_.provisional[key] = false;
				for (size_t index = 0; index < inst->NumArgs() && result; index++) {
					result = UniformRec(inst->Arg(index));
				}
				break;
			}
			default:
				// Atomics and appends give each lane its own result.
				result = !inst->MayHaveSideEffects();
				for (size_t index = 0; index < inst->NumArgs() && result; index++) {
					result = UniformRec(inst->Arg(index));
				}
				break;
		}
		if (result) {
			uniform_.Yes(key);
		} else {
			uniform_.No(key, mark);
		}
		return result;
	}

	IR::Value lane_;
	bool      negate_;
	const std::unordered_map<const IR::Block*, std::unordered_set<const IR::Inst*>>& per_lane_merges_;
	Memo      implies_;
	Memo      masked_;
	Memo      uniform_;
};

std::string DescribeLaneSkipValue(IR::Value value, int depth) {
	value = value.Resolve();
	if (value.IsImmediate()) return "imm";
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return "reg";
	std::string text(IR::ValueOpcodeName(inst->GetOpcode()));
	if (depth > 0 && inst->NumArgs() != 0) {
		text += "(";
		for (size_t index = 0; index < inst->NumArgs() && index < 4; index++) {
			text += (index != 0 ? "," : "") + DescribeLaneSkipValue(inst->Arg(index), depth - 1);
		}
		text += ")";
	}
	return text;
}

// A selection that might branch per lane: its body (the blocks reached from it before the merge) and
// lane condition; nullptr, or why not.
const char* FindLaneSkipRegion(const IR::Block* header, LaneSkip& skip, std::unordered_set<const IR::Block*>& region) {
	const auto& term = header->terminator;
	if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header || term.merge_block == nullptr ||
	    header->condition.IsEmpty()) {
		return "not a selection";
	}
	const auto* merge     = term.merge_block;
	const bool  body_true = term.false_block == merge;
	if (!body_true && term.true_block != merge) return "both targets in the body";
	const auto* body = body_true ? term.true_block : term.false_block;
	if (body == nullptr || body == merge) return "no body";
	auto        condition = header->condition.Resolve();
	bool        negated   = false;
	const auto* reference = condition.IsImmediate() ? nullptr : condition.TryInstruction();
	if (reference != nullptr && reference->GetOpcode() == IR::ValueOpcode::LogicalNot) {
		negated   = true;
		condition = reference->Arg(0).Resolve();
		reference = condition.IsImmediate() ? nullptr : condition.TryInstruction();
	}
	if (reference == nullptr || reference->GetOpcode() != IR::ValueOpcode::ConditionRef) return "no condition";
	const auto kind = reference->Flags<CFG::BranchCondition>();
	if (!IsBallotCondition(kind)) return "not a ballot";
	// The body is entered when some lane has the lane condition: "any(P)" (a non-zero test taken into
	// the body) or "not all(P)" (a zero test taken to the merge); "all(P)" would run it only when
	// every lane agrees.
	const bool flip = negated != !body_true;
	if (IsZeroCondition(kind) != flip) return "entered when every lane agrees";
	skip.lane   = reference->Arg(0).Resolve();
	skip.negate = flip;
	while (const auto* inst = skip.lane.IsImmediate() ? nullptr : skip.lane.TryInstruction()) {
		if (inst->GetOpcode() != IR::ValueOpcode::LogicalNot) break;
		skip.lane   = inst->Arg(0).Resolve();
		skip.negate = !skip.negate;
	}
	if (skip.lane.IsImmediate()) return "constant";
	if (body->ImmPredecessors().size() != 1) return "body has other entries";
	std::vector<const IR::Block*> work {body};
	while (!work.empty()) {
		const auto* block = work.back();
		work.pop_back();
		if (block == merge || !region.insert(block).second) continue;
		if (block == header) return "body loops back";
		const auto& t = block->terminator;
		if (t.kind == CFG::TerminatorKind::Branch && t.true_block != nullptr) {
			work.push_back(t.true_block);
		} else if (t.kind == CFG::TerminatorKind::ConditionalBranch && t.true_block != nullptr &&
		           t.false_block != nullptr && !block->condition.IsEmpty()) {
			work.push_back(t.true_block);
			work.push_back(t.false_block);
		} else {
			return "body returns";
		}
	}
	for (const auto* predecessor: merge->ImmPredecessors()) {
		if (predecessor != header && !region.contains(predecessor)) return "merge has other entries";
	}
	return nullptr;
}

// Whether the body of a FindLaneSkipRegion selection cannot tell per-lane entry (nullptr), with the
// merge phis to repair in skip.fixes, or why not.
const char* CheckLaneSkip(
    const IR::Block* header, LaneSkip& skip, const std::unordered_set<const IR::Block*>& region,
    const std::unordered_map<const IR::Block*, std::unordered_set<const IR::Inst*>>& per_lane_merges) {
	const auto*      merge = header->terminator.merge_block;
	LaneSkipAnalysis analysis(skip.lane, skip.negate, per_lane_merges);
	for (const auto* block: region) {
		for (const auto& inst: *block) {
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::DppUpdateU32:
				case IR::ValueOpcode::ReadLane:
				case IR::ValueOpcode::WriteLane:
				case IR::ValueOpcode::WaveReduceU32:
				case IR::ValueOpcode::Permlane16U32:
				case IR::ValueOpcode::PermuteU32:
				case IR::ValueOpcode::BpermuteU32:
				case IR::ValueOpcode::SwizzleU32: return "reads other lanes";
				case IR::ValueOpcode::Ballot:
					if (!analysis.Implies(inst.Arg(0), true)) return "ballot";
					continue;
				case IR::ValueOpcode::ReadFirstLane:
					if (!analysis.Implies(inst.Arg(1), true)) return "first-lane read";
					continue;
				case IR::ValueOpcode::ConditionRef: {
					// Lanes outside take no part: a zero test's P holds in them, a non-zero test's does not.
					const auto inner = inst.Flags<CFG::BranchCondition>();
					if (IsBallotCondition(inner) && !analysis.Implies(inst.Arg(0), !IsZeroCondition(inner))) {
						if (t_lane_skip_detail != nullptr) {
							*t_lane_skip_detail = std::to_string(static_cast<int>(inner)) + " leaf " +
							                      analysis.failed_leaf + " in " + DescribeLaneSkipValue(inst.Arg(0), 2);
						}
						return "inner branch";
					}
					continue;
				}
				case IR::ValueOpcode::Reference:
				case IR::ValueOpcode::ReferenceU32: continue;
				case IR::ValueOpcode::SetAttribute:
				case IR::ValueOpcode::SetTessellationAttribute:
				case IR::ValueOpcode::MeshAllocate:
				case IR::ValueOpcode::StoreCompletion:
				case IR::ValueOpcode::RegArrayStore:
				case IR::ValueOpcode::Barrier:
				case IR::ValueOpcode::Sendmsg: return "export or wave operation";
				default: break;
			}
			if (inst.MayHaveSideEffects()) {
				const auto args = inst.NumArgs();
				if (args == 0 || inst.Arg(args - 1).GetType() != IR::Type::U1 ||
				    !analysis.Implies(inst.Arg(args - 1), true)) {
					if (t_lane_skip_detail != nullptr) {
						*t_lane_skip_detail = DescribeLaneSkipValue(IR::Value(const_cast<IR::Inst*>(&inst)), 3);
					}
					return "side effect";
				}
			}
		}
	}
	// A merge shared with a selection inside that branches per lane already: its lanes may reach the
	// merge from several places, so only masked values are allowed.
	const bool shared_merge = per_lane_merges.contains(merge);
	for (const auto& phi: *merge) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) break;
		IR::Value old;
		bool      found = false;
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == header) {
				old   = phi.Arg(index);
				found = true;
			}
		}
		if (!found) return "merge phi without the skip";
		bool masked = true;
		for (size_t index = 0; index < phi.NumArgs() && masked; index++) {
			masked = phi.PhiBlock(index) == header || analysis.Masked(phi.Arg(index), old);
		}
		if (masked) continue;
		bool mask = phi.GetType() == IR::Type::U1, uniform = true;
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == header) continue;
			mask    = mask && analysis.Implies(phi.Arg(index), true);
			uniform = uniform && analysis.Uniform(phi.Arg(index));
		}
		if (!shared_merge && mask) {
			skip.fixes.emplace_back(&phi, true);
		} else if (!shared_merge && uniform &&
		           (phi.GetType() == IR::Type::U32 || phi.GetType() == IR::Type::U1)) {
			skip.fixes.emplace_back(&phi, false);
		} else {
			if (t_lane_skip_detail != nullptr) {
				std::string values;
				for (size_t index = 0; index < phi.NumArgs(); index++) {
					values += (index != 0 ? " | " : "") + DescribeLaneSkipValue(phi.Arg(index), 3);
				}
				*t_lane_skip_detail = std::string(IR::TypeName(phi.GetType())) + (shared_merge ? " shared " : " ") + values;
			}
			return "merge value";
		}
	}
	return nullptr;
}

// Experiment LaneSkips: a per-lane selection whose merge has values to repair, and an incoming value
// of its "entered the body" phi.
struct LaneSkipRepair {
	const IR::Block* header   = nullptr;
	const LaneSkip*  skip     = nullptr;
	const IR::Inst*  last_phi = nullptr;
};
struct LaneSkipEnteredPatch {
	DeferredPhi      phi;
	size_t           index       = 0;
	const IR::Block* predecessor = nullptr;
	bool             value       = false;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t> block_exit_labels;
	std::vector<DeferredPhiPatch>                  deferred_phis;
	std::unordered_map<const IR::Block*, LaneSkip> lane_skips;
	std::unordered_map<const IR::Block*, LaneSkipRepair> lane_skip_repairs; // by merge
	std::vector<LaneSkipEnteredPatch>                    lane_skip_entered;
};

// Experiment LaneSkips: after the last phi of a per-lane selection's merge, the repairs of its merge
// values (see LaneSkip): whether the lane entered the body is a phi, a lane mask is cleared in the
// lanes that did not once any lane did, and a wave value comes from the first lane that did.
void EmitLaneSkipRepairs(ValueEmitContext& ctx, StructuredFunctionState& structured, const IR::Block* merge,
                         const LaneSkipRepair& repair) {
	auto&      state        = ctx.state;
	auto&      builder      = state.builder;
	const auto predecessors = merge->ImmPredecessors();
	const auto entered      = builder.AllocateId();
	const auto phi          = builder.AddDeferredPhi(TypeBool(state), entered, predecessors.size());
	for (size_t index = 0; index < predecessors.size(); index++) {
		structured.lane_skip_entered.push_back({phi, index, predecessors[index], predecessors[index] != repair.header});
	}
	builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
	builder.RequireCapability(spv::CapabilityGroupNonUniformShuffle);
	const auto scope  = ConstantU32(state, spv::ScopeSubgroup);
	const auto ballot = builder.AllocateId();
	builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot, scope, entered);
	const auto any = Unary(state, spv::OpAny, TypeBool(state),
	                       Binary(state, spv::OpINotEqual, TypeBoolVector(state, 4), ballot,
	                              ConstantU32CompositeZero(state, 4)));
	const auto first = ctx.FirstLane(ballot);
	for (const auto& [inst, mask]: repair.skip->fixes) {
		const auto value = ctx.Def(IR::Value(const_cast<IR::Inst*>(inst)));
		uint32_t   fixed = 0;
		if (mask) {
			const auto none = Unary(state, spv::OpLogicalNot, TypeBool(state), any);
			fixed = Binary(state, spv::OpLogicalAnd, TypeBool(state), value,
			               Binary(state, spv::OpLogicalOr, TypeBool(state), none, entered));
		} else if (inst->GetType() == IR::Type::U1) {
			// Booleans travel as 0/1.
			const auto bits     = Select(state, TypeU32(state), value, ConstantU32(state, 1), ConstantU32(state, 0));
			const auto shuffled = builder.AllocateId();
			builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled, scope, bits, first);
			fixed = Select(state, TypeBool(state), any,
			               Binary(state, spv::OpINotEqual, TypeBool(state), shuffled, ConstantU32(state, 0)), value);
		} else {
			const auto shuffled = builder.AllocateId();
			builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled, scope, value, first);
			fixed = Select(state, TypeU32(state), any, shuffled, value);
		}
		inst->SetDefinition(0, fixed);
	}
}

struct DispatcherFunctionState {
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2> spills;
	uint32_t                                      header_label       = 0;
	uint32_t                                      select_label       = 0;
	uint32_t                                      after_switch_label = 0;
	uint32_t                                      continue_label     = 0;
	uint32_t                                      merge_label        = 0;
};

void StoreDispatcherPhiEdge(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                            const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) {
			break;
		}
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == from) {
				ctx.state.builder.AddFunction(spv::OpStore, dispatcher.spills[ctx.half].at(&phi),
				                              ctx.Def(phi.Arg(index)));
				break;
			}
		}
	}
}

void EmitReturn(ValueEmitContext& ctx) {
	EmitKillIfPixelValidMaskInactive(ctx.state);
	if (ctx.state.object_primitive_id_variable != 0) {
		auto& state = ctx.state;
		const auto primitive = EmitGetBuiltin(
		    ctx, IR::Value {static_cast<uint32_t>(IR::StageInputKind::PrimitiveId)}, IR::Value {0u});
		const auto object = EmitPushDataLoad(state, ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.object_primitive_id_variable,
		                          EmitAddU32(state, object, primitive));
	}
	ctx.state.builder.AddFunction(spv::OpReturn);
}

// NHL debugging: KYTY_DBG_LOOP_CAP=<n> gives each invocation a budget of n loop-header passes.
// Once exhausted, every branch to a loop merge block (a break) is taken, so runaway
// data-dependent loops end instead of triggering a GPU timeout.
uint32_t LoopCapValue() {
	static const uint32_t cap = [] {
		const char* v = std::getenv("KYTY_DBG_LOOP_CAP");
		return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0u;
	}();
	return cap;
}

bool IsLoopMergeBlock(const IR::Program& program, const IR::Block* target) {
	return target != nullptr && std::ranges::any_of(program.blocks, [&](const IR::Block* other) {
		return other->terminator.loop_header && other->terminator.merge_block == target;
	});
}

uint32_t LoopBudgetExhausted(EmitterState& state) {
	const auto budget = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), budget, state.loop_budget_variable);
	const auto exhausted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), exhausted, budget,
	                          ConstantU32(state, 0));
	return exhausted;
}

void DecrementLoopBudget(EmitterState& state) {
	const auto budget = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), budget, state.loop_budget_variable);
	const auto nonzero = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), nonzero, budget,
	                          ConstantU32(state, 0));
	const auto step = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), step, nonzero, ConstantU32(state, 1),
	                          ConstantU32(state, 0));
	const auto next = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpISub, TypeU32(state), next, budget, step);
	state.builder.AddFunction(spv::OpStore, state.loop_budget_variable, next);
}

// `lane_skip`: experiment LaneSkips, the selection's per-lane condition of entering its body.
void EmitStructuredTerminator(ValueEmitContext& ctx, const IR::Block* block, const LaneSkip* lane_skip = nullptr) {
	const auto& program = ctx.state.program;
	const auto& term       = block->terminator;
	const bool  loop_cap   = ctx.state.loop_budget_variable != 0;
	if (loop_cap && term.loop_header) {
		DecrementLoopBudget(ctx.state);
	}
	const auto  emit_merge = [&]() {
		if (term.loop_header) {
			const auto* merge = term.merge_block;
			const auto* cont  = term.continue_block;
			if (merge != nullptr && cont != nullptr) {
				ctx.state.builder.AddFunction(spv::OpLoopMerge, merge->Definition(),
				                              cont->Definition(), spv::LoopControlMaskNone);
			}
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           term.merge_block != nullptr) {
			ctx.state.builder.AddFunction(spv::OpSelectionMerge, term.merge_block->Definition(),
			                              spv::SelectionControlMaskNone);
		}
	};

	switch (term.kind) {
		case CFG::TerminatorKind::Branch: {
			const auto* target = term.true_block;
			if (target == nullptr) {
				EmitReturn(ctx);
				return;
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranch, target->Definition());
			return;
		}
		case CFG::TerminatorKind::ConditionalBranch: {
			const auto* true_block  = term.true_block;
			const auto* false_block = term.false_block;
			if (true_block == nullptr || false_block == nullptr || block->condition.IsEmpty()) {
				EmitReturn(ctx);
				return;
			}
			auto condition = ctx.Def(block->condition);
			if (lane_skip != nullptr) {
				// The true target is the body unless it is the merge.
				condition = ctx.Def(lane_skip->lane);
				if (lane_skip->negate != (true_block == term.merge_block)) {
					const auto negated = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpLogicalNot, TypeBool(ctx.state), negated, condition);
					condition = negated;
				}
			}
			if (loop_cap) {
				const bool true_breaks  = IsLoopMergeBlock(program, term.true_block);
				const bool false_breaks = IsLoopMergeBlock(program, term.false_block);
				if (true_breaks != false_breaks) {
					const auto exhausted = LoopBudgetExhausted(ctx.state);
					auto       forced    = exhausted;
					if (false_breaks) {
						forced = ctx.state.builder.AllocateId();
						ctx.state.builder.AddFunction(spv::OpLogicalNot, TypeBool(ctx.state), forced,
						                              exhausted);
					}
					const auto combined = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(
					    true_breaks ? spv::OpLogicalOr : spv::OpLogicalAnd, TypeBool(ctx.state),
					    combined, condition, forced);
					condition = combined;
				}
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranchConditional, condition,
			                              true_block->Definition(), false_block->Definition());
			return;
		}
		default: EmitReturn(ctx); return;
	}
}

void EmitDispatcherTarget(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                          const IR::Block* from, const IR::Block* block) {
	if (block != nullptr) {
		StoreDispatcherPhiEdge(ctx, dispatcher, from, block);
		if (ctx.other_half != nullptr) {
			StoreDispatcherPhiEdge(*ctx.other_half, dispatcher, from, block);
		}
	}
}

// NHL debugging: KYTY_DBG_LOOP_CAP in dispatcher form. A target "loops" when it is a back-edge
// (block id not after the current one) or a latch block that branches straight back.
bool DispatcherTargetLoops(const IR::Block* from, const IR::Block* target) {
	if (target == nullptr) {
		return false;
	}
	if (target->id <= from->id) {
		return true;
	}
	return target->terminator.kind == CFG::TerminatorKind::Branch &&
	       target->terminator.true_block != nullptr &&
	       target->terminator.true_block->id <= target->id;
}

uint32_t EmitDispatcherNextPc(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                              const IR::Block* block) {
	const auto& term     = block->terminator;
	const bool  loop_cap = ctx.state.loop_budget_variable != 0;
	switch (term.kind) {
		case CFG::TerminatorKind::Branch:
			if (loop_cap && term.true_block->id <= block->id) {
				DecrementLoopBudget(ctx.state);
			}
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			return ConstantU32(ctx.state, term.true_block->id);
		case CFG::TerminatorKind::ConditionalBranch: {
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			EmitDispatcherTarget(ctx, dispatcher, block, term.false_block);
			auto condition = ctx.Def(block->condition);
			if (loop_cap) {
				const bool true_loops  = DispatcherTargetLoops(block, term.true_block);
				const bool false_loops = DispatcherTargetLoops(block, term.false_block);
				if (true_loops != false_loops) {
					if (term.true_block->id <= block->id || term.false_block->id <= block->id) {
						DecrementLoopBudget(ctx.state);
					}
					const auto exhausted = LoopBudgetExhausted(ctx.state);
					auto       forced    = exhausted;
					if (true_loops) {
						forced = ctx.state.builder.AllocateId();
						ctx.state.builder.AddFunction(spv::OpLogicalNot, TypeBool(ctx.state), forced,
						                              exhausted);
					}
					const auto combined = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(
					    true_loops ? spv::OpLogicalAnd : spv::OpLogicalOr, TypeBool(ctx.state),
					    combined, condition, forced);
					condition = combined;
				}
			}
			const auto selected = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), selected, condition,
			                              ConstantU32(ctx.state, term.true_block->id),
			                              ConstantU32(ctx.state, term.false_block->id));
			return selected;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			for (const auto* target: block->ImmSuccessors()) {
				EmitDispatcherTarget(ctx, dispatcher, block, target);
			}
			uint32_t selected = ConstantU32(ctx.state, UINT32_MAX);
			if (!block->indirect_target.IsEmpty()) {
				const auto selector = ctx.Def(block->indirect_target);
				for (const auto& branch: term.cases) {
					const auto match = ctx.state.builder.AllocateId();
					const auto next  = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), match,
					                              selector, ConstantU32(ctx.state, branch.value));
					ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), next, match,
					                              ConstantU32(ctx.state, branch.target->id),
					                              selected);
					selected = next;
				}
			}
			return selected;
		}
		default: return ConstantU32(ctx.state, UINT32_MAX);
	}
}

template <typename T>
decltype(auto) Arg(ValueEmitContext& ctx, const IR::Inst& inst, size_t index) {
	if constexpr (std::is_same_v<T, const IR::Inst&>) {
		return inst;
	} else if constexpr (std::is_same_v<T, IR::Value>) {
		return inst.Arg(index);
	} else if constexpr (std::is_same_v<T, IR::ScalarReg>) {
		return inst.Arg(index).ScalarRegister();
	} else {
		static_assert(std::is_same_v<T, uint32_t>);
		return ctx.Def(inst.Arg(index));
	}
}

template <typename Context, typename Return, typename... Args>
void Invoke(Return (*emit)(Context&, Args...), ValueEmitContext& ctx, const IR::Inst& inst) {
	// A full instruction keeps metadata and predicated/lane operand loads lazy.
	static_assert(std::is_same_v<Context, ValueEmitContext> ||
	              std::is_same_v<Context, EmitterState>);
	auto& context = [&]() -> Context& {
		if constexpr (std::is_same_v<Context, EmitterState>)
			return ctx.state;
		else
			return ctx;
	}();
	constexpr bool has_inst = (std::is_same_v<Args, const IR::Inst&> || ...);
	[&]<size_t... I>(std::index_sequence<I...>) {
		static_assert(((!std::is_same_v<Args, const IR::Inst&> || I == 0) && ...));
		const auto call = [&] {
			return emit(context, Arg<Args>(ctx, inst, I - (has_inst && I != 0))...);
		};
		if constexpr (std::is_void_v<Return>) {
			call();
		} else {
			static_assert(std::is_same_v<Return, uint32_t>);
			ctx.Define(inst, call());
		}
	}(std::index_sequence_for<Args...> {});
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto op = inst.GetOpcode();
	if (ctx.half != 0 &&
	    (op == Op::Ballot || op == Op::ReadFirstLane || op == Op::SubgroupUMinU32 ||
	     op == Op::ReadConstBuffer ||
	     (op == Op::LoadAddressU32 && ctx.Memory(inst).kind == IR::ResourceKind::ScalarAddress))) {
		// Whole-wave results and SMEM loads (SGPR operands, no EXEC) are identical in both halves.
		ctx.Define(inst, ctx.other_half->Result(inst));
		return;
	}
	switch (op) {
#define VALUE_OPCODE(name, ...)                                                                    \
	case IR::ValueOpcode::name: return Invoke(Emit##name, ctx, inst);
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: ctx.Fail(inst, "has no direct SPIR-V emitter");
	}
}

void EmitStructuredInstruction(ValueEmitContext& ctx, StructuredFunctionState& structured,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0 || inst.NumArgs() == 0) {
			ctx.Fail(inst, "has no native SPIR-V representation");
		}
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			const auto* predecessor = inst.PhiBlock(index);
			if (predecessor == nullptr || predecessor->Definition() == 0) {
				ctx.Fail(inst, "has a predecessor outside the structured function");
			}
		}
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), inst.NumArgs()), &inst,
		     ctx.half});
		if (const auto repair = structured.lane_skip_repairs.find(inst.Parent());
		    repair != structured.lane_skip_repairs.end() && repair->second.last_phi == &inst) {
			EmitLaneSkipRepairs(ctx, structured, inst.Parent(), repair->second);
		}
		return;
	}
	EmitDirectInstruction(ctx, inst);
}

void EmitDispatcherInstruction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0) {
			ctx.Fail(inst, "cannot be loaded by the dispatcher");
		}
		ctx.state.builder.AddFunction(spv::OpLoad, type, ctx.Result(inst),
		                              dispatcher.spills[ctx.half].at(&inst));
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (const auto found = dispatcher.spills[ctx.half].find(&inst);
	    found != dispatcher.spills[ctx.half].end()) {
		ctx.state.builder.AddFunction(spv::OpStore, found->second,
		                              ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))));
	}
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, block->Definition());
	bool emitted_non_phi = false;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			emitted_non_phi = true;
		}
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::StoreCompletion &&
			                  inst.GetOpcode() != IR::ValueOpcode::DataCacheInvalidate &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				emit_instruction(lane, inst);
			}
		}
		ctx.state.lane_half = 0;
	}
}

// Experiment ExecBranches: a run of instructions of a block that are lane-wise or select a guest
// VGPR's new value under one EXEC value (Select(exec, new, old)) is emitted inside
// "if (exec) { ... }": the selects inside it are their new values, and those used after the run
// become phis of the new value (from the branch) and the old one (from before it). Instructions of
// the run used outside it that depend on nothing in it are emitted before the branch.
struct ExecRun {
	const IR::Inst*              begin = nullptr;
	const IR::Inst*              end   = nullptr; // nullptr: the run ends the block
	IR::Value                    exec;
	std::vector<const IR::Inst*> hoisted;
	std::vector<const IR::Inst*> branch;
	std::vector<const IR::Inst*> live_out;
	bool                         memory = false;
};

// Experiment ExecBranchMemory: memory operations under the run's EXEC join it.
bool ExecRunMemoryOps() {
	return ShaderExecBranchMemory();
}

bool IsSelectOpcode(IR::ValueOpcode op) {
	return op == IR::ValueOpcode::SelectU1 || op == IR::ValueOpcode::SelectU32 || op == IR::ValueOpcode::SelectF32;
}

// A value that is `new` where EXEC is set and `old` where it is clear: Select(exec, new, old), and
// LogicalAnd(exec, new) (a V_CMP result under EXEC, false where it is clear). `new_operand` is the
// argument index of the new value.
bool ExecPredicated(const IR::Inst& inst, IR::Value exec, IR::Value& new_value, IR::Value& old_value,
                    size_t& new_operand) {
	if (IsSelectOpcode(inst.GetOpcode()) && inst.NumArgs() == 3 && inst.Arg(0).Resolve() == exec) {
		new_value   = inst.Arg(1);
		old_value   = inst.Arg(2).Resolve();
		new_operand = 1;
		return true;
	}
	if (inst.GetOpcode() == IR::ValueOpcode::LogicalAnd && inst.NumArgs() == 2) {
		for (size_t arg = 0; arg < 2; arg++) {
			if (inst.Arg(arg).Resolve() == exec) {
				new_value   = inst.Arg(1 - arg);
				old_value   = IR::Value(false);
				new_operand = 1 - arg;
				return true;
			}
		}
	}
	return false;
}

bool IsExecSelect(const IR::Inst& inst, IR::Value exec) {
	IR::Value new_value, old_value;
	size_t    new_operand = 0;
	return ExecPredicated(inst, exec, new_value, old_value, new_operand);
}

// The value of an exec-predicated instruction where EXEC is clear, through chains of them inside
// the run (`inside` decides which producers belong to it).
template <typename Inside>
IR::Value ExecOldValue(const IR::Inst& inst, IR::Value exec, Inside&& inside) {
	IR::Value new_value, old_value;
	size_t    new_operand = 0;
	(void)ExecPredicated(inst, exec, new_value, old_value, new_operand);
	while (const auto* producer = old_value.IsImmediate() ? nullptr : old_value.TryInstruction()) {
		IR::Value next_new, next_old;
		if (!inside(producer) || !ExecPredicated(*producer, exec, next_new, next_old, new_operand)) break;
		old_value = next_old;
	}
	return old_value;
}

uint32_t ExecRunMinSelects() {
	static const uint32_t minimum = [] {
		const char* v = std::getenv("KYTY_DBG_EXEC_BRANCH_MIN");
		return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 6u;
	}();
	return minimum;
}

// KYTY_DBG_EXEC_BRANCH=2: per shader, the instructions that end a stretch of run candidates holding
// EXEC selects, and the EXEC selects left outside runs.
thread_local std::unordered_map<std::string_view, uint32_t>* t_exec_breakers = nullptr;

std::vector<ExecRun> FindExecRuns(const IR::Block& block) {
	std::vector<const IR::Inst*> insts;
	for (const auto& inst: block) insts.push_back(&inst);
	// Runs are index ranges of the block: membership by position, not by hash sets rebuilt for
	// every shrink of a run (most of the compile time of large shaders).
	std::unordered_map<const IR::Inst*, size_t> position;
	position.reserve(insts.size());
	for (size_t i = 0; i < insts.size(); i++) position.emplace(insts[i], i);
	std::vector<uint8_t> hoisted_at(insts.size(), 0);
	const auto* condition_inst = block.condition.IsEmpty() || block.condition.IsImmediate()
	                                 ? nullptr
	                                 : block.condition.TryInstruction();
	const auto* target_inst = block.indirect_target.IsEmpty() || block.indirect_target.IsImmediate()
	                              ? nullptr
	                              : block.indirect_target.TryInstruction();
	const bool  memory_ops     = ExecRunMemoryOps();
	const auto  is_memory      = [memory_ops](const IR::Inst& inst) {
		return memory_ops && IR::IsExecPredicatedMemoryOp(inst) && !IR::ReadsNeighborLanesOpcode(inst.GetOpcode());
	};
	const bool transparent_ops = ShaderExecRunTransparent();
	const bool const_reads     = ShaderExecRunConstReads();
	const auto transparent     = [transparent_ops, const_reads](IR::ValueOpcode op) {
		if (!transparent_ops) return false;
		switch (op) {
			case IR::ValueOpcode::ReadConstBuffer:
			case IR::ValueOpcode::ReadConst:
			case IR::ValueOpcode::MakeImageAddress: return const_reads;
			case IR::ValueOpcode::Reference:
			case IR::ValueOpcode::ReferenceU32:
			case IR::ValueOpcode::GetSrtResource:
			case IR::ValueOpcode::GetBufferResource:
			case IR::ValueOpcode::GetAddressResource:
			case IR::ValueOpcode::GetImageResource:
			case IR::ValueOpcode::GetSamplerResource:
			case IR::ValueOpcode::LaneId: return true;
			default: return false;
		}
	};
	const auto candidate = [&](const IR::Inst& inst) {
		const auto op = inst.GetOpcode();
		return is_memory(inst) || transparent(op) ||
		       ((IR::IsLaneWiseOpcode(op) || IsSelectOpcode(op)) && op != IR::ValueOpcode::Phi && !inst.MayHaveSideEffects());
	};
	std::vector<ExecRun> runs;
	size_t               start = 0;
	while (start < insts.size()) {
		if (!candidate(*insts[start])) {
			start++;
			continue;
		}
		size_t end = start;
		while (end < insts.size() && candidate(*insts[end])) end++;
		if (t_exec_breakers != nullptr && end < insts.size()) {
			for (size_t i = start; i < end; i++) {
				if (IsSelectOpcode(insts[i]->GetOpcode())) {
					(*t_exec_breakers)[IR::ValueOpcodeName(insts[end]->GetOpcode())]++;
					break;
				}
			}
		}
		const auto index_of = [&](const IR::Inst* inst) {
			const auto it = position.find(inst);
			return it != position.end() ? it->second : SIZE_MAX;
		};
		struct InRun {
			const decltype(index_of)& index;
			const size_t&             first;
			const size_t&             last;
			[[nodiscard]] bool        contains(const IR::Inst* inst) const {
				const auto i = index(inst);
				return i != SIZE_MAX && i >= first && i < last;
			}
		};
		// The run's EXEC value: the condition of its first select whose condition is an
		// instruction defined before the run.
		const InRun in_run {index_of, start, end};
		IR::Value exec;
		for (size_t i = start; i < end && exec.IsEmpty(); i++) {
			const auto& inst = *insts[i];
			if (!IsSelectOpcode(inst.GetOpcode()) || inst.NumArgs() != 3) continue;
			const auto cond = inst.Arg(0).Resolve();
			if (!cond.IsImmediate() && cond.TryInstruction() != nullptr && !in_run.contains(cond.TryInstruction())) {
				exec = cond;
			}
		}
		if (exec.IsEmpty()) {
			start = end;
			continue;
		}
		// Shrink the run until everything it computes is used inside it, is a select under the
		// EXEC value (a phi after it), or can be computed before the branch.
		for (;;) {
			ExecRun run;
			run.exec = exec;
			std::fill(hoisted_at.begin() + static_cast<ptrdiff_t>(start), hoisted_at.begin() + static_cast<ptrdiff_t>(end),
			          uint8_t {0});
			struct Hoisted {
				const decltype(index_of)& index;
				std::vector<uint8_t>&     at;
				[[nodiscard]] bool        contains(const IR::Inst* inst) const {
					const auto i = index(inst);
					return i != SIZE_MAX && at[i] != 0;
				}
				void insert(const IR::Inst* inst) {
					if (const auto i = index(inst); i != SIZE_MAX) at[i] = 1;
				}
			} hoisted {index_of, hoisted_at};
			size_t                              violation = end;
			uint32_t                            selects   = 0;
			for (size_t i = start; i < end && violation == end; i++) {
				const auto& inst    = *insts[i];
				bool        outside = &inst == condition_inst || &inst == target_inst;
				for (const auto& use: inst.Uses()) {
					IR::Value user_new, user_old;
					size_t    user_new_operand = 0;
					if (!in_run.contains(use.user) || hoisted.contains(use.user)) {
						outside = true;
					} else if (ExecPredicated(*use.user, exec, user_new, user_old, user_new_operand) &&
					           use.operand != user_new_operand) {
						// As an old value or a condition the value is read where EXEC is clear.
						outside = true;
					}
				}
				if (IsExecSelect(inst, exec)) {
					const IR::Value old = ExecOldValue(inst, exec, [&](const IR::Inst* producer) {
						return in_run.contains(producer) && !hoisted.contains(producer);
					});
					const auto* producer = old.IsImmediate() ? nullptr : old.TryInstruction();
					if (producer != nullptr && in_run.contains(producer) && !hoisted.contains(producer)) {
						violation = i;
						break;
					}
					run.branch.push_back(&inst);
					if (outside) run.live_out.push_back(&inst);
					selects++;
					continue;
				}
				if (is_memory(inst)) {
					// Only where EXEC is set (itself or a value under it), in program order.
					const auto predicate = inst.Arg(inst.NumArgs() - 1).Resolve();
					const auto* producer = predicate.IsImmediate() ? nullptr : predicate.TryInstruction();
					const bool  under    = predicate == exec ||
					                   (producer != nullptr && in_run.contains(producer) && !hoisted.contains(producer) &&
					                    IsExecSelect(*producer, exec));
					if (!under || outside) {
						violation = i;
						break;
					}
					run.branch.push_back(&inst);
					run.memory = true;
					continue;
				}
				if (!outside) {
					run.branch.push_back(&inst);
					continue;
				}
				bool independent = true;
				for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
					const auto value    = inst.Arg(arg).Resolve();
					const auto* producer = value.IsImmediate() ? nullptr : value.TryInstruction();
					if (producer != nullptr && in_run.contains(producer) && !hoisted.contains(producer)) {
						independent = false;
					}
				}
				if (!independent) {
					violation = i;
					break;
				}
				hoisted.insert(&inst);
				run.hoisted.push_back(&inst);
			}
			if (violation == end) {
				if (selects >= ExecRunMinSelects() && !run.branch.empty()) {
					run.begin = insts[start];
					run.end   = end < insts.size() ? insts[end] : nullptr;
					runs.push_back(std::move(run));
				}
				break;
			}
			end = violation;
			if (end <= start) break;
		}
		start = std::max(end, start + 1);
	}
	return runs;
}

void EmitExecRun(ValueEmitContext& ctx, StructuredFunctionState& structured, const ExecRun& run) {
	auto& state   = ctx.state;
	auto& builder = state.builder;
	for (const auto* inst: run.hoisted) {
		EmitStructuredInstruction(ctx, structured, *inst);
	}
	const auto condition   = ctx.Def(run.exec);
	const auto then_label  = builder.AllocateId();
	const auto merge_label = builder.AllocateId();
	const auto pre_label   = state.current_label;
	builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	builder.AddFunction(spv::OpBranchConditional, condition, then_label, merge_label);
	EmitLabel(state, then_label);
	const auto* exec_inst = run.exec.TryInstruction();
	const auto  exec_def  = exec_inst != nullptr ? exec_inst->Definition(0) : 0u;
	if (exec_inst != nullptr && run.memory) {
		exec_inst->SetDefinition(0, ConstantBool(state, true));
	}
	std::vector<std::pair<const IR::Inst*, uint32_t>> saved;
	std::unordered_set<const IR::Inst*>                in_branch(run.branch.begin(), run.branch.end());
	for (const auto* inst: run.branch) {
		IR::Value new_value, old_value;
		size_t    new_operand = 0;
		if (ExecPredicated(*inst, run.exec, new_value, old_value, new_operand)) {
			// Inside the branch EXEC is set: the instruction is its new value.
			const auto value = ctx.Def(new_value);
			saved.emplace_back(inst, inst->Definition(0));
			inst->SetDefinition(0, value);
		} else {
			EmitStructuredInstruction(ctx, structured, *inst);
		}
	}
	if (exec_inst != nullptr && run.memory) {
		exec_inst->SetDefinition(0, exec_def);
	}
	const auto then_exit = state.current_label;
	builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	std::vector<std::pair<const IR::Inst*, uint32_t>> new_values;
	for (const auto* inst: run.live_out) new_values.emplace_back(inst, inst->Definition(0));
	for (const auto& [inst, previous]: saved) inst->SetDefinition(0, previous);
	for (const auto& [inst, value]: new_values) {
		const auto old = ExecOldValue(*inst, run.exec, [&](const IR::Inst* producer) {
			return in_branch.contains(producer);
		});
		const auto old_value = ctx.Def(old);
		builder.AddFunction(spv::OpPhi, TypeId(state, inst->GetType()), ctx.Result(*inst), value, then_exit,
		                    old_value, pre_label);
	}
}

void EmitBlockWithExecRuns(ValueEmitContext& ctx, StructuredFunctionState& structured, const IR::Block* block,
                           const std::vector<ExecRun>& runs) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, block->Definition());
	size_t      next = 0;
	const auto& insts = block->Instructions();
	for (auto it = insts.begin(); it != insts.end();) {
		if (next < runs.size() && &*it == runs[next].begin) {
			EmitExecRun(ctx, structured, runs[next]);
			const auto* end = runs[next].end;
			while (it != insts.end() && &*it != end) ++it;
			next++;
			continue;
		}
		EmitStructuredInstruction(ctx, structured, *it);
		++it;
	}
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& patch: structured.lane_skip_entered) {
		const auto found = structured.block_exit_labels.find(patch.predecessor);
		if (found == structured.block_exit_labels.end()) {
			ctx.Fail("a per-lane selection's merge has a predecessor that was not emitted");
		}
		ctx.state.builder.PatchDeferredPhi(patch.phi, patch.index, ConstantBool(ctx.state, patch.value), found->second);
	}
	for (const auto& deferred: structured.deferred_phis) {
		auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
		for (size_t index = 0; index < deferred.instruction->NumArgs(); index++) {
			const auto* predecessor = deferred.instruction->PhiBlock(index);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			ctx.state.builder.PatchDeferredPhi(
			    deferred.phi, index, lane.Def(deferred.instruction->Arg(index)), found->second);
		}
	}
}

void EmitStructuredFunction(ValueEmitContext& ctx) {
	const auto& program = ctx.state.program;
	StructuredFunctionState structured;
	ctx.state.builder.AddFunction(spv::OpBranch, program.blocks.front()->Definition());
	// Experiment ExecBranches: not in loop headers (their OpLoopMerge ends their one block) nor with
	// wave64 emulation (each instruction is emitted per half); in continue targets only with
	// ExecBranchContinue (the selection constructs then lie inside the continue construct, which
	// the back-edge block, the last of them, still post-dominates).
	const bool exec_branches = ShaderExecBranches() && ctx.state.lane_count == 1;
	const bool in_continue   = ShaderExecBranchContinue();
	std::unordered_set<const IR::Block*> loop_blocks;
	if (exec_branches) {
		for (const auto* block: program.blocks) {
			if (block->terminator.loop_header) {
				loop_blocks.insert(block);
				if (block->terminator.continue_block != nullptr && !in_continue) {
					loop_blocks.insert(block->terminator.continue_block);
				}
			}
		}
	}
	// KYTY_DBG_EXEC_BRANCH=1: runs, selects (and LogicalAnds) under EXEC turned into branch code,
	// phis and hoisted instructions per shader (NHL27EXECBR).
	static const bool exec_branch_debug = std::getenv("KYTY_DBG_EXEC_BRANCH") != nullptr;
	static const bool breaker_debug     = [] {
		const char* v = std::getenv("KYTY_DBG_EXEC_BRANCH");
		return v != nullptr && v[0] == '2';
	}();
	std::unordered_map<std::string_view, uint32_t> breakers;
	t_exec_breakers = breaker_debug ? &breakers : nullptr;
	// Experiment LaneSkips (KYTY_DBG_LANE_SKIPS=1: per shader the selections on a ballot, those
	// branching per lane, their repaired merge values and why the others do not, NHL27LANESKIP; =2
	// also each one kept on its ballot, NHL27LANESKIPNO).
	if (ShaderLaneSkips() && ctx.state.lane_count == 1) {
		static const char* lane_skip_env   = std::getenv("KYTY_DBG_LANE_SKIPS");
		static const bool  lane_skip_debug = lane_skip_env != nullptr;
		static const bool  lane_skip_each  = lane_skip_env != nullptr && lane_skip_env[0] == '2';
		std::string        detail;
		t_lane_skip_detail = lane_skip_each ? &detail : nullptr;
		std::map<std::string_view, uint32_t> reasons;
		uint32_t                             selections = 0, repaired = 0;
		const auto reject = [&](const IR::Block* block, const char* reason) {
			reasons[reason]++;
			if (lane_skip_each) {
				std::printf("NHL27LANESKIPNO hash=%016" PRIx64 " label=%u %s: %s\n", program.shader_hash,
				            block->Definition(), reason, detail.c_str());
			}
			detail.clear();
		};
		struct Candidate {
			const IR::Block*                     header = nullptr;
			LaneSkip                             skip;
			std::unordered_set<const IR::Block*> region;
		};
		std::vector<Candidate> candidates;
		for (const auto* block: program.blocks) {
			Candidate              candidate {block};
			const char*            found  = FindLaneSkipRegion(block, candidate.skip, candidate.region);
			const std::string_view reason = found != nullptr ? found : "";
			if (reason == "not a selection" || reason == "no condition" || reason == "not a ballot") continue;
			selections++;
			if (found != nullptr) {
				reject(block, found);
			} else {
				candidates.push_back(std::move(candidate));
			}
		}
		// Innermost first: a selection's repairs make its merge values the same in every lane again,
		// which the selections around it may rely on.
		std::stable_sort(candidates.begin(), candidates.end(),
		                 [](const Candidate& a, const Candidate& b) { return a.region.size() < b.region.size(); });
		std::unordered_map<const IR::Block*, std::unordered_set<const IR::Inst*>> per_lane_merges;
		for (auto& candidate: candidates) {
			if (const char* found = CheckLaneSkip(candidate.header, candidate.skip, candidate.region, per_lane_merges);
			    found != nullptr) {
				reject(candidate.header, found);
				continue;
			}
			const auto* merge  = candidate.header->terminator.merge_block;
			auto&       values = per_lane_merges[merge];
			for (const auto& [phi, mask]: candidate.skip.fixes) {
				if (!mask) values.insert(phi);
			}
			repaired += static_cast<uint32_t>(candidate.skip.fixes.size());
			const auto& skip = structured.lane_skips.emplace(candidate.header, std::move(candidate.skip)).first->second;
			if (!skip.fixes.empty()) {
				const IR::Inst* last_phi = nullptr;
				for (const auto& inst: *merge) {
					if (inst.GetOpcode() != IR::ValueOpcode::Phi) break;
					last_phi = &inst;
				}
				structured.lane_skip_repairs[merge] = {candidate.header, &skip, last_phi};
			}
		}
		t_lane_skip_detail = nullptr;
		if (lane_skip_debug) {
			std::string line;
			for (const auto& [reason, count]: reasons) line += " [" + std::string(reason) + "]=" + std::to_string(count);
			std::printf("NHL27LANESKIP hash=%016" PRIx64 " ballot_selections=%u per_lane=%zu repaired=%u%s\n",
			            program.shader_hash, selections, structured.lane_skips.size(), repaired, line.c_str());
		}
	}
	const auto lane_skip = [&](const IR::Block* block) -> const LaneSkip* {
		const auto found = structured.lane_skips.find(block);
		return found != structured.lane_skips.end() ? &found->second : nullptr;
	};
	size_t            stat_runs = 0, stat_branch = 0, stat_phis = 0, stat_hoisted = 0;
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		if (exec_branches && !loop_blocks.contains(block)) {
			if (const auto runs = FindExecRuns(*block); !runs.empty()) {
				for (const auto& run: runs) {
					stat_runs++;
					stat_branch += run.branch.size();
					stat_phis += run.live_out.size();
					stat_hoisted += run.hoisted.size();
				}
				EmitBlockWithExecRuns(ctx, structured, block, runs);
				structured.block_exit_labels.emplace(block, ctx.state.current_label);
				EmitStructuredTerminator(ctx, block, lane_skip(block));
				continue;
			}
		}
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
		structured.block_exit_labels.emplace(block, ctx.state.current_label);
		EmitStructuredTerminator(ctx, block, lane_skip(block));
	}
	PatchStructuredPhis(ctx, structured);
	t_exec_breakers = nullptr;
	if (breaker_debug && !breakers.empty()) {
		std::vector<std::pair<std::string_view, uint32_t>> sorted(breakers.begin(), breakers.end());
		std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string line;
		for (size_t i = 0; i < sorted.size() && i < 8; i++) {
			line += " " + std::string(sorted[i].first) + "=" + std::to_string(sorted[i].second);
		}
		std::printf("NHL27EXECBREAK: hash=0x%016llx%s\n", static_cast<unsigned long long>(program.shader_hash),
		            line.c_str());
	}
	if (exec_branch_debug && exec_branches) {
		std::printf("NHL27EXECBR: hash=0x%016llx stage=%u runs=%zu branch_insts=%zu phis=%zu hoisted=%zu\n",
		            static_cast<unsigned long long>(program.shader_hash), static_cast<unsigned>(program.stage), stat_runs,
		            stat_branch, stat_phis, stat_hoisted);
		std::fflush(stdout);
	}
}

void EmitDispatcherFunction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher) {
	auto&       state = ctx.state;
	const auto* entry = state.program.blocks.front();
	state.builder.AddFunction(spv::OpBranch, entry->Definition());
	EmitBlock(ctx, entry, [&](ValueEmitContext& lane, const IR::Inst& inst) {
		EmitDispatcherInstruction(lane, dispatcher, inst);
	});
	const auto initial_pc     = EmitDispatcherNextPc(ctx, dispatcher, entry);
	const auto initial_parent = state.current_label;
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);

	state.current_block = nullptr;
	EmitLabel(state, dispatcher.header_label);
	const auto pc      = state.builder.AllocateId();
	const auto next_pc = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), pc, initial_pc, initial_parent, next_pc,
	                          dispatcher.continue_label);
	const auto done = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), done, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
	state.builder.AddFunction(spv::OpLoopMerge, dispatcher.merge_label, dispatcher.continue_label,
	                          spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, done, dispatcher.merge_label,
	                          dispatcher.select_label);

	EmitLabel(state, dispatcher.select_label);
	state.builder.AddFunction(spv::OpSelectionMerge, dispatcher.after_switch_label,
	                          spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, pc, dispatcher.after_switch_label};
	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		words.push_back(state.program.blocks[index]->id);
		words.push_back(state.program.blocks[index]->Definition());
	}
	state.builder.AddFunction(words);
	std::vector<uint32_t> next_pc_words {spv::OpPhi, TypeU32(state), next_pc,
	                                     ConstantU32(state, UINT32_MAX), dispatcher.select_label};

	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		EmitBlock(ctx, state.program.blocks[index],
		          [&](ValueEmitContext& lane, const IR::Inst& inst) {
			          EmitDispatcherInstruction(lane, dispatcher, inst);
		          });
		const auto selected = EmitDispatcherNextPc(ctx, dispatcher, state.program.blocks[index]);
		next_pc_words.push_back(selected);
		next_pc_words.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, dispatcher.after_switch_label);
	}
	state.current_block = nullptr;
	EmitLabel(state, dispatcher.after_switch_label);
	state.builder.AddFunction(next_pc_words);
	state.builder.AddFunction(spv::OpBranch, dispatcher.continue_label);
	EmitLabel(state, dispatcher.continue_label);
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);
	EmitLabel(state, dispatcher.merge_label);
	EmitReturn(ctx);
}

} // namespace

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16: return TypeU32(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
		case IR::Type::F32x2: return TypeF32Vector(state, 2);
		default: return 0;
	}
}

uint32_t ValueEmitContext::Def(IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case IR::Type::U1: return ConstantBool(state, value.U1());
			case IR::Type::U8: return ConstantU32(state, value.U8());
			case IR::Type::U16: return ConstantU32(state, value.U16());
			case IR::Type::U32: return ConstantU32(state, value.U32());
			case IR::Type::U64: return ConstantU64(state, value.U64());
			case IR::Type::F16: return ConstantU32(state, value.F16Bits());
			case IR::Type::F32:
				return ConstantF32(state, std::bit_cast<uint32_t>(value.F32Value()));
			default: break;
		}
	}
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr) {
		Fail("direct SPIR-V emitter received a non-value argument");
	}
	if (dispatcher_spills != nullptr && state.current_block != nullptr &&
	    inst->Parent() != state.current_block) {
		if (const auto found = dispatcher_spills->find(inst); found != dispatcher_spills->end()) {
			if (const auto loaded = dispatcher_block_loads.find(inst);
			    loaded != dispatcher_block_loads.end() &&
			    loaded->second.first == state.current_label) {
				return loaded->second.second;
			}
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), id,
			                          found->second);
			dispatcher_block_loads.insert_or_assign(inst, std::pair {state.current_label, id});
			return id;
		}
	}
	return Result(*inst);
}

uint32_t ValueEmitContext::Arg(const IR::Inst& inst, size_t index) {
	return Def(inst.Arg(index));
}

uint32_t ValueEmitContext::HalfArg(const IR::Inst& inst, size_t index, uint32_t lane_half) {
	return lane_half == half ? Arg(inst, index) : other_half->Arg(inst, index);
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto low         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, low, scope,
	                          other_half == nullptr || half == 0 ? Def(predicate)
	                                                             : other_half->Def(predicate));
	if (other_half == nullptr) {
		return low;
	}
	const auto high      = state.builder.AllocateId();
	const auto low_word  = state.builder.AllocateId();
	const auto high_word = state.builder.AllocateId();
	const auto ballot    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, high, scope,
	                          half == 1 ? Def(predicate) : other_half->Def(predicate));
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low_word, low, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high_word, high, 0);
	state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, low_word, high_word,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	return ballot;
}

uint32_t ValueEmitContext::FirstLane(uint32_t ballot) {
	// An empty ballot (EXEC zero) gives lane 0, which V_READFIRSTLANE reads on hardware. FindILsb
	// returns -1 for zero, unlike BallotFindLSB, whose result is then undefined.
	const auto first = [&](uint32_t word) {
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits, ballot, word);
		return std::pair {bits, EmitGlsl<GLSLstd450FindILsb, IR::Type::U32>(state, bits)};
	};
	const auto [low, low_first] = first(0);
	if (state.program.wave_size == 32u) {
		return EmitGlsl<GLSLstd450SMax, IR::Type::U32>(state, low_first, ConstantU32(state, 0));
	}
	const auto high_first = first(1).second;
	const auto high_lane  = EmitGlsl<GLSLstd450SMax, IR::Type::U32>(
	    state, EmitBinaryU32(state, spv::OpBitwiseOr, high_first, ConstantU32(state, 32)),
	    ConstantU32(state, 0));
	return Select(state, TypeU32(state),
	              Binary(state, spv::OpINotEqual, TypeBool(state), low, ConstantU32(state, 0)),
	              low_first, high_lane);
}

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto low   = state.builder.AllocateId();
	if (other_half == nullptr) {
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          lane);
		return low;
	}
	const auto physical_lane =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto high          = state.builder.AllocateId();
	const auto in_high       = state.builder.AllocateId();
	const auto value         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope,
	                          HalfArg(inst, index, 0), physical_lane);
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, high, scope,
	                          HalfArg(inst, index, 1), physical_lane);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), in_high,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, type, value, in_high, high, low);
	return value;
}

uint32_t ValueEmitContext::Result(const IR::Inst& inst) {
	if (const auto id = inst.Definition(half); id != 0) {
		return id;
	}
	const auto id = state.builder.AllocateId();
	inst.SetDefinition(half, id);
	return id;
}

uint32_t ValueEmitContext::Define(const IR::Inst& inst, uint32_t value) {
	if (const auto id = inst.Definition(half); id != 0) {
		if (id != value) {
			state.builder.AddFunction(spv::OpCopyObject, TypeId(state, inst.GetType()),
			                          id, value);
		}
		return id;
	}
	inst.SetDefinition(half, value);
	return value;
}

uint32_t ValueEmitContext::ResourceIndex(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != opcode) {
		Fail("typed resource handle has the wrong producer");
	}
	return inst->Flags<uint32_t>();
}

const IR::Inst* ValueEmitContext::ImageAddress(IR::Value value) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::MakeImageAddress) {
		Fail("typed image address was not constructed by MakeImageAddress");
	}
	return inst;
}

const IR::MemoryInfo& ValueEmitContext::Memory(const IR::Inst& inst) const {
	return state.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
}

const IR::ExportInfo& ValueEmitContext::Export(const IR::Inst& inst) const {
	return state.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
}

[[noreturn]] void ValueEmitContext::Fail(const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), reason);
	std::abort();
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
	std::abort();
}

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.program.info.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	for (const auto* block: program.blocks) {
		// Re-emission may use a different host subgroup width and ID allocation order.
		for (const auto& inst: *block) {
			inst.ResetDefinitions();
		}
		block->SetDefinition(state.builder.AllocateId());
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
		// Allocate both halves in program order so the SPIR-V is identical across runs.
		const auto add_spill = [&](const IR::Inst* inst) {
			for (uint32_t half = 0; half < state.lane_count; half++) {
				dispatch.spills[half].emplace(inst, state.builder.AllocateId());
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
					continue;
				}
				if (SpillPointerType(ctx, inst.GetType()) == 0) {
					ctx.Fail(inst, "cannot be stored by the dispatcher");
					break;
				}
				add_spill(&inst);
			}
		}
		const auto mark_cross_block = [&](IR::Value value, const IR::Block* consumer) {
			value                  = value.Resolve();
			const auto* definition = value.TryInstruction();
			if (definition == nullptr || definition->Parent() == consumer ||
			    definition->Parent() == program.blocks.front()) {
				return;
			}
			if (SpillPointerType(ctx, definition->GetType()) == 0) {
				ctx.Fail(*definition, "cannot be stored by the dispatcher");
				return;
			}
			if (!dispatch.spills[0].contains(definition)) {
				add_spill(definition);
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* consumer =
					    inst.GetOpcode() == IR::ValueOpcode::Phi ? inst.PhiBlock(index) : block;
					mark_cross_block(inst.Arg(index), consumer);
				}
			}
		}
		for (size_t index = 0; index < program.blocks.size(); index++) {
			mark_cross_block(program.blocks[index]->condition, program.blocks[index]);
			mark_cross_block(program.blocks[index]->indirect_target, program.blocks[index]);
		}
		dispatch.header_label       = state.builder.AllocateId();
		dispatch.select_label       = state.builder.AllocateId();
		dispatch.after_switch_label = state.builder.AllocateId();
		dispatch.continue_label     = state.builder.AllocateId();
		dispatch.merge_label        = state.builder.AllocateId();
		ctx.dispatcher_spills       = &dispatch.spills[0];
		high.dispatcher_spills      = &dispatch.spills[1];
	}
	DefineGetBdaPointer(state);
	DefineIndirectBufferLoads(ctx);
	DefineIndirectBufferFormats(ctx);
	DefineBvhIntersect(state);
	if (program.info.uses_swizzle) {
		ctx.scratch_u32_variable = state.builder.AllocateId();
		if (state.lane_count == 2) {
			high.scratch_u32_variable = state.builder.AllocateId();
		}
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	if (state.program.info.function_lds) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.program.info.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	for (const auto& [array, variables]: state.reg_array_variables) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypeU32ArrayPointer(state, spv::StorageClassFunction, array & 0xffffu),
			                          variables[half], spv::StorageClassFunction);
		}
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
	}
	for (uint32_t half = 0; half < state.lane_count; half++) {
		auto& lane = half == 0 ? ctx : high;
		if (state.program.dispatcher_fallback) {
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					if (const auto found = dispatcher->spills[half].find(&inst);
					    found != dispatcher->spills[half].end()) {
						state.builder.AddFunction(spv::OpVariable,
						                          SpillPointerType(lane, inst.GetType()),
						                          found->second, spv::StorageClassFunction);
					}
				}
			}
		}
		if (lane.scratch_u32_variable != 0) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
	}
	if (LoopCapValue() != 0) {
		state.loop_budget_variable = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.loop_budget_variable, spv::StorageClassFunction);
	}
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	if (state.loop_budget_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.loop_budget_variable,
		                          ConstantU32(state, LoopCapValue()));
	}
	if (state.lds_storage_class == spv::StorageClassStorageBuffer && state.lds_variable != 0) {
		const auto group_x = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		const auto group_y = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto group_z = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 2);
		const auto count_x = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 0);
		const auto count_y = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 1);
		const auto row = EmitAddU32(state, group_y, EmitBinaryU32(state, spv::OpIMul, group_z, count_y));
		const auto index = EmitAddU32(state, group_x, EmitBinaryU32(state, spv::OpIMul, row, count_x));
		state.lds_base_dwords = EmitBinaryU32(state, spv::OpIMul, index,
		                                     ConstantU32(state, LdsDwordCount(state)));
	}
	EmitMemoryOffsets(state);
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	state.current_block = nullptr;
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
