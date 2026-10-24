#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include "graphics/shader/recompiler/backend/spirv/spirvHalfLanes.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

bool HalfVectorEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HALF_VECTOR");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

namespace {

constexpr uint32_t NoId = 0;

class Module {
public:
	explicit Module(const std::vector<uint32_t>& words): m_words(words) {}

	bool Parse() {
		if (m_words.size() < 5 || m_words[0] != spv::MagicNumber) return false;
		m_bound = m_words[3];
		m_def.assign(m_bound, -1);
		for (size_t pos = 5; pos < m_words.size();) {
			const uint32_t count = m_words[pos] >> 16u;
			const auto     op    = static_cast<spv::Op>(m_words[pos] & 0xffffu);
			if (count == 0 || pos + count > m_words.size()) return false;
			bool has_result = false;
			bool has_type   = false;
			spv::HasResultAndType(op, &has_result, &has_type);
			Inst inst {static_cast<uint32_t>(pos), count, op, has_type, NoId};
			if (has_result) {
				const uint32_t at = has_type ? 2u : 1u;
				if (count <= at) return false;
				inst.result = m_words[pos + at];
				if (inst.result >= m_bound) return false;
				m_def[inst.result] = static_cast<int32_t>(m_insts.size());
			}
			m_insts.push_back(inst);
			pos += count;
		}
		BuildUses();
		return true;
	}

	struct Inst {
		uint32_t pos;
		uint32_t count;
		spv::Op  op;
		bool     has_type;
		uint32_t result;
	};

	[[nodiscard]] const Inst* Def(uint32_t id) const {
		return id < m_bound && m_def[id] >= 0 ? &m_insts[static_cast<size_t>(m_def[id])] : nullptr;
	}
	[[nodiscard]] uint32_t Word(const Inst& inst, uint32_t index) const {
		return index < inst.count ? m_words[inst.pos + index] : NoId;
	}
	[[nodiscard]] uint32_t TypeOf(uint32_t id) const {
		const auto* d = Def(id);
		return d != nullptr && d->has_type ? Word(*d, 1) : NoId;
	}
	[[nodiscard]] const std::vector<int32_t>& Users(uint32_t id) const {
		static const std::vector<int32_t> empty;
		const auto it = m_uses.find(id);
		return it != m_uses.end() ? it->second : empty;
	}

	std::vector<Inst>& Insts() { return m_insts; }
	[[nodiscard]] const std::vector<uint32_t>& Words() const { return m_words; }
	[[nodiscard]] uint32_t Bound() const { return m_bound; }

private:
	void BuildUses() {
		for (size_t index = 0; index < m_insts.size(); index++) {
			const auto& inst = m_insts[index];
			if (inst.op == spv::OpName || inst.op == spv::OpMemberName || inst.op == spv::OpDecorate ||
			    inst.op == spv::OpMemberDecorate || inst.op == spv::OpEntryPoint ||
			    inst.op == spv::OpExecutionMode) {
				continue;
			}
			const uint32_t first = inst.has_type ? 3u : (inst.result != NoId ? 2u : 1u);
			for (uint32_t w = first; w < inst.count; w++) {
				const auto value = m_words[inst.pos + w];
				// Literal words that happen to equal an ID only make the users list longer, which
				// keeps the analysis conservative.
				if (value != NoId && value < m_bound) {
					auto& list = m_uses[value];
					if (list.empty() || list.back() != static_cast<int32_t>(index)) {
						list.push_back(static_cast<int32_t>(index));
					}
				}
			}
		}
	}

	const std::vector<uint32_t>&                     m_words;
	uint32_t                                         m_bound = 0;
	std::vector<Inst>                                m_insts;
	std::vector<int32_t>                             m_def;
	std::unordered_map<uint32_t, std::vector<int32_t>> m_uses;
};

class HalfLaneRewriter {
public:
	explicit HalfLaneRewriter(Module& module): m(module) {}

	bool FindTypes() {
		for (const auto& inst: m.Insts()) {
			switch (inst.op) {
				case spv::OpExtInstImport: {
					const char* name = reinterpret_cast<const char*>(&m.Words()[inst.pos + 2]);
					const size_t max = (inst.count - 2u) * 4u;
					if (strnlen(name, max) < max && std::strcmp(name, "GLSL.std.450") == 0) {
						m_glsl = inst.result;
					}
					break;
				}
				case spv::OpTypeInt:
					if (m.Word(inst, 2) == 32 && m.Word(inst, 3) == 0) m_u32 = inst.result;
					break;
				case spv::OpTypeFloat:
					if (m.Word(inst, 2) == 32) m_f32 = inst.result;
					if (m.Word(inst, 2) == 16) m_f16 = inst.result;
					break;
				default: break;
			}
		}
		for (const auto& inst: m.Insts()) {
			if (inst.op == spv::OpTypeVector && m.Word(inst, 3) == 2) {
				if (m.Word(inst, 2) == m_f16 && m_f16 != NoId) m_f16x2 = inst.result;
				if (m.Word(inst, 2) == m_f32 && m_f32 != NoId) m_f32x2 = inst.result;
			}
		}
		return m_glsl != NoId && m_u32 != NoId && m_f32 != NoId && m_f16 != NoId &&
		       m_f16x2 != NoId && m_f32x2 != NoId;
	}

	uint32_t Run(std::vector<uint32_t>& out, HalfLaneStats* stats) {
		auto&    insts = m.Insts();
		uint32_t next  = m.Bound();
		std::unordered_map<size_t, std::vector<uint32_t>> replace;
		HalfLaneStats local;
		const auto emit = [](std::vector<uint32_t>& v, spv::Op op, std::initializer_list<uint32_t> operands) {
			v.push_back((static_cast<uint32_t>(operands.size() + 1u) << 16u) | static_cast<uint32_t>(op));
			v.insert(v.end(), operands.begin(), operands.end());
		};
		for (size_t index = 0; index < insts.size(); index++) {
			const auto& inst = insts[index];
			std::vector<uint32_t> words;
			if (inst.op == spv::OpCompositeExtract && inst.count == 5 && m.Word(inst, 1) == m_f32 &&
			    m.Word(inst, 4) == 0) {
				// R1: UnpackHalf2x16(bfe(src, off, 16)).x -> float(f16vec2(src)[off / 16])
				const auto* unpack = m.Def(m.Word(inst, 3));
				if (unpack == nullptr || unpack->op != spv::OpExtInst || unpack->count != 6 ||
				    m.Word(*unpack, 1) != m_f32x2 || m.Word(*unpack, 3) != m_glsl ||
				    m.Word(*unpack, 4) != GLSLstd450UnpackHalf2x16) {
					continue;
				}
				uint32_t src = NoId, lane = 0;
				if (!LaneOf(m.Word(*unpack, 5), &src, &lane)) continue;
				const uint32_t vec = next++, half = next++;
				emit(words, spv::OpBitcast, {m_f16x2, vec, src});
				emit(words, spv::OpCompositeExtract, {m_f16, half, vec, lane});
				emit(words, spv::OpFConvert, {m_f32, inst.result, half});
				local.reads++;
			} else if (inst.op == spv::OpBitcast && inst.count == 4 && m.Word(inst, 1) == m_f16x2) {
				// R3: f16vec2(bfe(src, off, 16)) whose high lane never reaches a result -> swizzle.
				uint32_t src = NoId, lane = 0;
				if (!LaneOf(m.Word(inst, 3), &src, &lane) || !LowLaneOnly(inst.result, 0)) continue;
				const uint32_t vec = next++;
				emit(words, spv::OpBitcast, {m_f16x2, vec, src});
				emit(words, spv::OpVectorShuffle, {m_f16x2, inst.result, vec, vec, lane, lane});
				local.shuffles++;
			} else if (inst.op == spv::OpBitwiseOr && inst.count == 5 && m.Word(inst, 1) == m_u32) {
				// R2: (old & 0xffff0000) | half  /  (old & 0xffff) | (half << 16)
				const uint32_t a = m.Word(inst, 3), b = m.Word(inst, 4);
				for (int order = 0; order < 2 && words.empty(); order++) {
					const uint32_t masked = order == 0 ? a : b;
					const uint32_t other  = order == 0 ? b : a;
					uint32_t       old = NoId, mask = 0;
					if (!AndConstant(masked, &old, &mask)) continue;
					uint32_t lane = 0, value = other;
					bool     need_clean = false;
					if (mask == 0xffff0000u) {
						lane       = 0;
						need_clean = true;
					} else if (mask == 0x0000ffffu) {
						const auto* shl = m.Def(other);
						if (shl == nullptr || shl->op != spv::OpShiftLeftLogical || shl->count != 5 ||
						    m.Word(*shl, 1) != m_u32 || !IsU32Constant(m.Word(*shl, 4), 16)) {
							continue;
						}
						lane  = 1;
						value = m.Word(*shl, 3);
					} else {
						continue;
					}
					uint32_t source = NoId;
					bool     convert = false;
					if (!HalfSource(value, need_clean, &source, &convert)) continue;
					const uint32_t half = next++, vec = next++, merged = next++;
					if (convert) {
						emit(words, spv::OpFConvert, {m_f16, half, source});
					} else {
						emit(words, spv::OpCompositeExtract, {m_f16, half, source, 0});
					}
					emit(words, spv::OpBitcast, {m_f16x2, vec, old});
					emit(words, spv::OpCompositeInsert, {m_f16x2, merged, half, vec, lane});
					emit(words, spv::OpBitcast, {m_u32, inst.result, merged});
					local.writes++;
				}
			}
			if (!words.empty()) {
				replace.emplace(index, std::move(words));
			}
		}
		const uint32_t total = local.reads + local.shuffles + local.writes;
		if (stats != nullptr) *stats = local;
		if (total == 0) return 0;
		const auto& in = m.Words();
		out.clear();
		out.reserve(in.size() + replace.size() * 12u);
		out.insert(out.end(), in.begin(), in.begin() + 5);
		out[3] = next;
		for (size_t index = 0; index < insts.size(); index++) {
			const auto it = replace.find(index);
			if (it != replace.end()) {
				out.insert(out.end(), it->second.begin(), it->second.end());
			} else {
				const auto& inst = insts[index];
				out.insert(out.end(), in.begin() + inst.pos, in.begin() + inst.pos + inst.count);
			}
		}
		return total;
	}

private:
	[[nodiscard]] bool IsU32Constant(uint32_t id, uint32_t value) const {
		const auto* d = m.Def(id);
		return d != nullptr && d->op == spv::OpConstant && d->count == 4 && m.Word(*d, 1) == m_u32 &&
		       m.Word(*d, 3) == value;
	}
	[[nodiscard]] bool IsF32Zero(uint32_t id) const {
		const auto* d = m.Def(id);
		return d != nullptr && d->op == spv::OpConstant && d->count == 4 && m.Word(*d, 1) == m_f32 &&
		       m.Word(*d, 3) == 0u;
	}
	// value & constant (either operand order)
	bool AndConstant(uint32_t id, uint32_t* value, uint32_t* mask) const {
		const auto* d = m.Def(id);
		if (d == nullptr || d->op != spv::OpBitwiseAnd || d->count != 5 || m.Word(*d, 1) != m_u32) {
			return false;
		}
		for (int order = 0; order < 2; order++) {
			const uint32_t c = m.Word(*d, order == 0 ? 4 : 3);
			const auto*    k = m.Def(c);
			if (k != nullptr && k->op == spv::OpConstant && k->count == 4 && m.Word(*k, 1) == m_u32) {
				*mask  = m.Word(*k, 3);
				*value = m.Word(*d, order == 0 ? 3 : 4);
				return true;
			}
		}
		return false;
	}
	// Skips value & 0xffff and u32->u32 bitcasts; reports whether a mask was crossed.
	uint32_t Strip(uint32_t id, bool* cleaned) const {
		for (int guard = 0; guard < 16; guard++) {
			uint32_t value = NoId, mask = 0;
			if (AndConstant(id, &value, &mask) && mask == 0xffffu) {
				*cleaned = true;
				id       = value;
				continue;
			}
			const auto* d = m.Def(id);
			if (d != nullptr && d->op == spv::OpBitcast && d->count == 4 && m.Word(*d, 1) == m_u32 &&
			    m.TypeOf(m.Word(*d, 3)) == m_u32) {
				id = m.Word(*d, 3);
				continue;
			}
			break;
		}
		return id;
	}
	// id == bfe(src, 0 | 16, 16) up to masks/bitcasts
	bool LaneOf(uint32_t id, uint32_t* src, uint32_t* lane) const {
		bool        cleaned = false;
		const auto* d       = m.Def(Strip(id, &cleaned));
		if (d == nullptr || d->op != spv::OpBitFieldUExtract || d->count != 6 ||
		    m.Word(*d, 1) != m_u32 || !IsU32Constant(m.Word(*d, 5), 16)) {
			return false;
		}
		if (IsU32Constant(m.Word(*d, 4), 0)) {
			*lane = 0;
		} else if (IsU32Constant(m.Word(*d, 4), 16)) {
			*lane = 1;
		} else {
			return false;
		}
		*src = m.Word(*d, 3);
		return m.TypeOf(*src) == m_u32;
	}
	// The low 16 bits of `id` hold a half. need_clean: the upper 16 bits must also be zero.
	bool HalfSource(uint32_t id, bool need_clean, uint32_t* source, bool* convert) const {
		bool        cleaned = false;
		const auto* d       = m.Def(Strip(id, &cleaned));
		if (d == nullptr) return false;
		if (d->op == spv::OpExtInst && d->count == 6 && m.Word(*d, 3) == m_glsl &&
		    m.Word(*d, 4) == GLSLstd450PackHalf2x16) {
			const auto* pair = m.Def(m.Word(*d, 5));
			if (pair != nullptr && pair->op == spv::OpCompositeConstruct && pair->count == 5 &&
			    m.Word(*pair, 1) == m_f32x2 && IsF32Zero(m.Word(*pair, 4))) {
				*source  = m.Word(*pair, 3);
				*convert = true; // upper half is +0.0 = 0
				return true;
			}
			return false;
		}
		if (d->op == spv::OpBitcast && d->count == 4 && m.TypeOf(m.Word(*d, 3)) == m_f16x2 &&
		    (cleaned || !need_clean)) {
			*source  = m.Word(*d, 3);
			*convert = false;
			return true;
		}
		return false;
	}
	// Every use of the f16vec2 `id` only lets its low lane reach anything observable.
	bool LowLaneOnly(uint32_t id, int depth) const {
		if (depth > 6) return false;
		const auto& users = m.Users(id);
		if (users.empty()) return false;
		for (const auto user: users) {
			const auto& inst = m.Insts()[static_cast<size_t>(user)];
			const auto  type = inst.has_type ? m.Word(inst, 1) : NoId;
			switch (inst.op) {
				case spv::OpBitcast: {
					if (type != m_u32) return false;
					const auto& masks = m.Users(inst.result);
					if (masks.empty()) return false;
					for (const auto mu: masks) {
						uint32_t value = NoId, mask = 0;
						if (!AndConstant(m.Insts()[static_cast<size_t>(mu)].result, &value, &mask) ||
						    mask != 0xffffu || value != inst.result) {
							return false;
						}
					}
					break;
				}
				case spv::OpFAdd:
				case spv::OpFSub:
				case spv::OpFMul:
				case spv::OpFNegate:
					if (type != m_f16x2 || !LowLaneOnly(inst.result, depth + 1)) return false;
					break;
				case spv::OpExtInst: {
					const auto which = m.Word(inst, 4);
					if (type != m_f16x2 || m.Word(inst, 3) != m_glsl ||
					    (which != GLSLstd450Fma && which != GLSLstd450FClamp &&
					     which != GLSLstd450NMax && which != GLSLstd450NMin &&
					     which != GLSLstd450FMax && which != GLSLstd450FMin) ||
					    !LowLaneOnly(inst.result, depth + 1)) {
						return false;
					}
					break;
				}
				case spv::OpCompositeExtract:
					if (inst.count != 5 || m.Word(inst, 4) != 0) return false;
					break;
				default: return false;
			}
		}
		return true;
	}

	Module&  m;
	uint32_t m_glsl  = NoId;
	uint32_t m_u32   = NoId;
	uint32_t m_f32   = NoId;
	uint32_t m_f16   = NoId;
	uint32_t m_f16x2 = NoId;
	uint32_t m_f32x2 = NoId;
};

} // namespace

uint32_t RewriteHalfLanes(std::vector<uint32_t>& spirv, HalfLaneStats* stats) {
	Module module(spirv);
	if (!module.Parse()) return 0;
	HalfLaneRewriter rewriter(module);
	if (!rewriter.FindTypes()) return 0;
	std::vector<uint32_t> out;
	const auto count = rewriter.Run(out, stats);
	if (count != 0) {
		spirv = std::move(out);
	}
	return count;
}

bool FastDenormEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FAST_DENORM");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool LaneDemandEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LANE_DEMAND");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

namespace {

bool IsPureValueOp(spv::Op op) {
	switch (op) {
		case spv::OpBitcast:
		case spv::OpBitwiseAnd:
		case spv::OpBitwiseOr:
		case spv::OpBitwiseXor:
		case spv::OpNot:
		case spv::OpBitFieldUExtract:
		case spv::OpBitFieldSExtract:
		case spv::OpBitFieldInsert:
		case spv::OpBitReverse:
		case spv::OpBitCount:
		case spv::OpShiftLeftLogical:
		case spv::OpShiftRightLogical:
		case spv::OpShiftRightArithmetic:
		case spv::OpIAdd:
		case spv::OpISub:
		case spv::OpIMul:
		case spv::OpUDiv:
		case spv::OpSDiv:
		case spv::OpUMod:
		case spv::OpSMod:
		case spv::OpSRem:
		case spv::OpSNegate:
		case spv::OpFAdd:
		case spv::OpFSub:
		case spv::OpFMul:
		case spv::OpFDiv:
		case spv::OpFMod:
		case spv::OpFRem:
		case spv::OpFNegate:
		case spv::OpExtInst:
		case spv::OpConvertFToU:
		case spv::OpConvertFToS:
		case spv::OpConvertUToF:
		case spv::OpConvertSToF:
		case spv::OpFConvert:
		case spv::OpUConvert:
		case spv::OpSConvert:
		case spv::OpQuantizeToF16:
		case spv::OpCompositeConstruct:
		case spv::OpCompositeExtract:
		case spv::OpCompositeInsert:
		case spv::OpVectorShuffle:
		case spv::OpCopyObject:
		case spv::OpIEqual:
		case spv::OpINotEqual:
		case spv::OpULessThan:
		case spv::OpULessThanEqual:
		case spv::OpUGreaterThan:
		case spv::OpUGreaterThanEqual:
		case spv::OpSLessThan:
		case spv::OpSLessThanEqual:
		case spv::OpSGreaterThan:
		case spv::OpSGreaterThanEqual:
		case spv::OpFOrdEqual:
		case spv::OpFOrdNotEqual:
		case spv::OpFOrdLessThan:
		case spv::OpFOrdLessThanEqual:
		case spv::OpFOrdGreaterThan:
		case spv::OpFOrdGreaterThanEqual:
		case spv::OpFUnordEqual:
		case spv::OpFUnordNotEqual:
		case spv::OpFUnordLessThan:
		case spv::OpFUnordLessThanEqual:
		case spv::OpFUnordGreaterThan:
		case spv::OpFUnordGreaterThanEqual:
		case spv::OpIsNan:
		case spv::OpIsInf:
		case spv::OpLogicalAnd:
		case spv::OpLogicalOr:
		case spv::OpLogicalNot:
		case spv::OpLogicalEqual:
		case spv::OpLogicalNotEqual:
		case spv::OpAny:
		case spv::OpAll:
		case spv::OpDot:
		case spv::OpVectorTimesScalar:
		case spv::OpUMulExtended:
		case spv::OpSMulExtended:
		case spv::OpIAddCarry:
		case spv::OpISubBorrow:
		case spv::OpAccessChain:
		case spv::OpInBoundsAccessChain:
		case spv::OpPtrAccessChain:
		case spv::OpArrayLength:
		case spv::OpLoad:
		case spv::OpSampledImage:
		case spv::OpImage:
		case spv::OpImageSampleExplicitLod:
		case spv::OpImageSampleDrefExplicitLod:
		case spv::OpImageSparseSampleExplicitLod:
		case spv::OpImageGather:
		case spv::OpImageDrefGather:
		case spv::OpImageFetch:
		case spv::OpImageRead:
		case spv::OpImageQuerySize:
		case spv::OpImageQuerySizeLod:
		case spv::OpImageQueryLevels:
		case spv::OpImageQuerySamples:
		case spv::OpPhi:
		case spv::OpSelect: return true;
		default: return false;
	}
}

// Instructions whose result can depend on other lanes' values (including lanes the guest had
// switched off): their operands are observed in every lane.
bool IsCrossLaneOp(spv::Op op) {
	const auto v = static_cast<uint32_t>(op);
	if (v >= static_cast<uint32_t>(spv::OpGroupNonUniformElect) &&
	    v <= static_cast<uint32_t>(spv::OpGroupNonUniformQuadSwap)) {
		return true;
	}
	if (v >= static_cast<uint32_t>(spv::OpGroupAll) && v <= static_cast<uint32_t>(spv::OpGroupSMax)) {
		return true;
	}
	if (v >= static_cast<uint32_t>(spv::OpDPdx) && v <= static_cast<uint32_t>(spv::OpFwidthCoarse)) {
		return true;
	}
	if (v >= 4421u && v <= 4432u) { // SPV_KHR_shader_ballot / subgroup_vote / rotate
		return true;
	}
	switch (op) {
		case spv::OpImageSampleImplicitLod:
		case spv::OpImageSampleDrefImplicitLod:
		case spv::OpImageSampleProjImplicitLod:
		case spv::OpImageSampleProjDrefImplicitLod:
		case spv::OpImageSparseSampleImplicitLod:
		case spv::OpImageSparseSampleDrefImplicitLod:
		case spv::OpImageSparseSampleProjImplicitLod:
		case spv::OpImageSparseSampleProjDrefImplicitLod:
		case spv::OpImageQueryLod:
		case spv::OpGroupNonUniformPartitionNV: return true;
		default: return false;
	}
}

// Lane sets are conjunctions of boolean SSA conditions: {} = every lane, {a, b} = a && b.
using LaneSet = std::vector<uint32_t>;

class LaneDemand {
public:
	explicit LaneDemand(Module& module): m(module) {}

	bool Run(std::vector<uint32_t>& out, LaneDemandStats* stats) {
		auto& insts = m.Insts();
		// Single entry function only; calls would need interprocedural demand.
		uint32_t functions = 0;
		for (const auto& inst: insts) {
			if (inst.op == spv::OpFunction) functions++;
			if (inst.op == spv::OpFunctionCall) return false;
			if (inst.op == spv::OpTypeBool) m_bool = inst.result;
		}
		if (functions != 1 || m_bool == NoId) return false;
		m_demand.assign(m.Bound(), {});
		m_has.assign(m.Bound(), 0);
		BuildBlocks();
		// Seeds: operands of everything that is not a pure value computation.
		bool in_function = false;
		for (size_t index = 0; index < insts.size(); index++) {
			const auto& inst = insts[index];
			if (inst.op == spv::OpFunction) in_function = true;
			if (!in_function || inst.op == spv::OpLabel || inst.op == spv::OpFunction ||
			    inst.op == spv::OpFunctionEnd || inst.op == spv::OpVariable ||
			    IsPureValueOp(inst.op)) {
				continue;
			}
			const LaneSet seed = IsCrossLaneOp(inst.op) ? LaneSet {} : Guard(m_block_of[index]);
			ForOperands(inst, [&](uint32_t id) { Add(id, seed); });
		}
		// Backward propagation.
		while (!m_work.empty()) {
			const uint32_t id = m_work.back();
			m_work.pop_back();
			const auto* def = m.Def(id);
			if (def == nullptr || !IsPureValueOp(def->op)) continue;
			const LaneSet d = m_demand[id];
			if (def->op == spv::OpSelect && def->count == 6) {
				const uint32_t cond = m.Word(*def, 3);
				Add(cond, d);
				LaneSet with_cond = d;
				with_cond.push_back(cond);
				Add(m.Word(*def, 4), Simplify(with_cond));
				if (!SetImplies(d, cond)) Add(m.Word(*def, 5), d);
				continue;
			}
			if (def->op == spv::OpLogicalAnd && def->count == 5 && m.Word(*def, 1) == m_bool) {
				// a && b: b only matters in lanes where a holds.
				const uint32_t a = m.Word(*def, 3);
				Add(a, d);
				LaneSet with_a = d;
				with_a.push_back(a);
				Add(m.Word(*def, 4), Simplify(with_a));
				continue;
			}
			ForOperands(*def, [&](uint32_t operand) { Add(operand, d); });
		}
		// Rewrite.
		LaneDemandStats local;
		out.clear();
		const auto& in = m.Words();
		out.insert(out.end(), in.begin(), in.begin() + 5);
		for (const auto& inst: insts) {
			if (inst.op == spv::OpSelect && inst.count == 6) {
				local.selects++;
				const uint32_t cond = m.Word(inst, 3);
				if (m.TypeOf(cond) == m_bool &&
				    (!m_has[inst.result] || SetImplies(m_demand[inst.result], cond))) {
					out.push_back((4u << 16u) | static_cast<uint32_t>(spv::OpCopyObject));
					out.push_back(m.Word(inst, 1));
					out.push_back(inst.result);
					out.push_back(m.Word(inst, 4));
					local.removed++;
					continue;
				}
			}
			out.insert(out.end(), in.begin() + inst.pos, in.begin() + inst.pos + inst.count);
		}
		if (stats != nullptr) *stats = local;
		return local.removed != 0;
	}

private:
	template <typename F>
	void ForOperands(const Module::Inst& inst, F&& f) const {
		uint32_t first = inst.has_type ? 3u : (inst.result != NoId ? 2u : 1u);
		if (inst.op == spv::OpExtInst) first = 5u;
		for (uint32_t w = first; w < inst.count; w++) {
			const auto id = m.Word(inst, w);
			if (id != NoId && id < m.Bound()) f(id);
		}
	}

	void BuildBlocks() {
		const auto& insts = m.Insts();
		m_block_of.assign(insts.size(), NoId);
		uint32_t current = NoId;
		for (size_t index = 0; index < insts.size(); index++) {
			const auto& inst = insts[index];
			if (inst.op == spv::OpLabel) current = inst.result;
			m_block_of[index] = current;
			const auto edge = [&](uint32_t target, uint32_t cond) {
				auto& p = m_preds[target];
				p.count++;
				p.cond = cond;
			};
			if (inst.op == spv::OpBranch) {
				edge(m.Word(inst, 1), NoId);
			} else if (inst.op == spv::OpBranchConditional) {
				const uint32_t t = m.Word(inst, 2), f = m.Word(inst, 3);
				edge(t, t != f ? m.Word(inst, 1) : NoId);
				edge(f, NoId);
			} else if (inst.op == spv::OpSwitch) {
				edge(m.Word(inst, 2), NoId);
				for (uint32_t w = 4; w < inst.count; w += 2) edge(m.Word(inst, w), NoId);
			}
		}
	}

	[[nodiscard]] LaneSet Guard(uint32_t block) const {
		const auto it = m_preds.find(block);
		if (it != m_preds.end() && it->second.count == 1 && it->second.cond != NoId) {
			return {it->second.cond};
		}
		return {};
	}

	bool Implies(uint32_t x, uint32_t c, int depth = 0) {
		if (x == c) return true;
		if (depth > 12 || x == NoId) return false;
		const uint64_t key = (static_cast<uint64_t>(x) << 32u) | c;
		if (const auto it = m_memo.find(key); it != m_memo.end()) return it->second;
		m_memo[key] = false; // cycle guard
		bool        result = false;
		const auto* dx     = m.Def(x);
		if (dx != nullptr && dx->op == spv::OpLogicalAnd && dx->count == 5) {
			result = Implies(m.Word(*dx, 3), c, depth + 1) || Implies(m.Word(*dx, 4), c, depth + 1);
		}
		if (!result && dx != nullptr && dx->op == spv::OpPhi && dx->count >= 5) {
			result = true;
			for (uint32_t w = 3; w < dx->count; w += 2) {
				if (!Implies(m.Word(*dx, w), c, depth + 1)) {
					result = false;
					break;
				}
			}
		}
		if (!result) {
			const auto* dc = m.Def(c);
			if (dc != nullptr && dc->op == spv::OpLogicalAnd && dc->count == 5) {
				result = Implies(x, m.Word(*dc, 3), depth + 1) && Implies(x, m.Word(*dc, 4), depth + 1);
			}
		}
		m_memo[key] = result;
		return result;
	}

	bool SetImplies(const LaneSet& set, uint32_t c) {
		for (const auto x: set) {
			if (Implies(x, c)) return true;
		}
		return false;
	}

	LaneSet Simplify(LaneSet set) {
		std::sort(set.begin(), set.end());
		set.erase(std::unique(set.begin(), set.end()), set.end());
		LaneSet result;
		for (const auto c: set) {
			bool implied = false;
			for (const auto o: set) {
				if (o != c && Implies(o, c)) {
					implied = true;
					break;
				}
			}
			if (!implied) result.push_back(c);
		}
		if (result.size() > 8) result.resize(8); // dropping conditions only widens the lane set
		return result;
	}

	// Union of the two lane sets, widened to something expressible as a conjunction.
	LaneSet Meet(const LaneSet& a, const LaneSet& b) {
		if (a == b) return a;
		LaneSet keep;
		for (const auto c: a) {
			if (SetImplies(b, c)) keep.push_back(c);
		}
		for (const auto c: b) {
			if (SetImplies(a, c)) keep.push_back(c);
		}
		return Simplify(std::move(keep));
	}

	void Add(uint32_t id, const LaneSet& demand) {
		if (id == NoId || id >= m.Bound() || m.Def(id) == nullptr) return;
		if (!m_has[id]) {
			m_has[id]    = 1;
			m_demand[id] = demand;
			m_work.push_back(id);
			return;
		}
		auto merged = Meet(m_demand[id], demand);
		if (merged != m_demand[id]) {
			m_demand[id] = std::move(merged);
			m_work.push_back(id);
		}
	}

	struct Preds {
		uint32_t count = 0;
		uint32_t cond  = NoId;
	};

	Module&                                 m;
	uint32_t                                m_bool = NoId;
	std::vector<LaneSet>                    m_demand;
	std::vector<uint8_t>                    m_has;
	std::vector<uint32_t>                   m_work;
	std::vector<uint32_t>                   m_block_of;
	std::unordered_map<uint32_t, Preds>     m_preds;
	std::unordered_map<uint64_t, bool>      m_memo;
};

} // namespace

uint32_t RewriteLaneDemand(std::vector<uint32_t>& spirv, LaneDemandStats* stats) {
	Module module(spirv);
	if (!module.Parse()) return 0;
	LaneDemand pass(module);
	std::vector<uint32_t> out;
	LaneDemandStats       local;
	if (!pass.Run(out, &local)) {
		if (stats != nullptr) *stats = local;
		return 0;
	}
	if (stats != nullptr) *stats = local;
	spirv = std::move(out);
	return local.removed;
}

// 1 = only inside loops (Oct 17 A/B: loop bodies gained, straight-line code did not), 2 = everywhere.
int LaneRegionsMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_LANE_REGIONS");
		if (value == nullptr) return 0;
		if (value[0] == '1') return 1;
		if (value[0] == '2') return 2;
		return 0;
	}();
	return mode;
}

bool LaneRegionsEnabled() {
	return LaneRegionsMode() != 0;
}

namespace {

// One instruction of the rewritten function, kept as raw words.
struct RInst {
	std::vector<uint32_t> w;

	[[nodiscard]] spv::Op Op() const { return static_cast<spv::Op>(w[0] & 0xffffu); }
	[[nodiscard]] uint32_t Result() const {
		bool has_result = false;
		bool has_type   = false;
		spv::HasResultAndType(Op(), &has_result, &has_type);
		if (!has_result) return NoId;
		const size_t at = has_type ? 2u : 1u;
		return at < w.size() ? w[at] : NoId;
	}
};

RInst MakeInst(spv::Op op, std::initializer_list<uint32_t> operands) {
	RInst inst;
	inst.w.push_back((static_cast<uint32_t>(operands.size() + 1u) << 16u) | static_cast<uint32_t>(op));
	inst.w.insert(inst.w.end(), operands.begin(), operands.end());
	return inst;
}

struct RBlock {
	uint32_t           label = NoId;
	std::vector<RInst> insts; // without OpLabel
};

// KYTY_LANE_REGIONS: work that only feeds select(c, new, old) is moved into "if (c) { ... }" and
// merged back with OpPhi, so the driver masks the lanes instead of emitting a v_cndmask per value.
class LaneRegions {
public:
	static constexpr uint32_t MinExports = 3;
	static constexpr uint32_t MinMoved   = 6;

	bool Run(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, LaneRegionStats* stats) {
		if (in.size() < 5 || in[0] != spv::MagicNumber) return false;
		m_bound = in[3];
		// Split: everything before the function, the function, everything after.
		size_t fn_begin = 0;
		size_t fn_end   = 0;
		uint32_t functions = 0;
		for (size_t pos = 5; pos < in.size();) {
			const uint32_t count = in[pos] >> 16u;
			const auto     op    = static_cast<spv::Op>(in[pos] & 0xffffu);
			if (count == 0 || pos + count > in.size()) return false;
			switch (op) {
				case spv::OpFunction:
					functions++;
					fn_begin = pos;
					break;
				case spv::OpFunctionEnd: fn_end = pos; break;
				case spv::OpFunctionCall: return false;
				case spv::OpTypeBool: m_bool = in[pos + 1]; break;
				case spv::OpConstantFalse:
					if (count == 3) m_false = in[pos + 2];
					break;
				case spv::OpExtInstImport: {
					const char*  name = reinterpret_cast<const char*>(&in[pos + 2]);
					const size_t max  = (count - 2u) * 4u;
					if (strnlen(name, max) < max && std::strcmp(name, "GLSL.std.450") == 0) {
						m_glsl = in[pos + 1];
					}
					break;
				}
				default: break;
			}
			pos += count;
		}
		if (functions != 1 || m_bool == NoId || fn_end <= fn_begin) return false;

		// Function header (OpFunction + parameters) and blocks.
		std::vector<uint32_t> fn_head;
		for (size_t pos = fn_begin; pos < fn_end;) {
			const uint32_t count = in[pos] >> 16u;
			const auto     op    = static_cast<spv::Op>(in[pos] & 0xffffu);
			if (op == spv::OpLabel) {
				m_blocks.push_back(RBlock {in[pos + 1], {}});
			} else if (m_blocks.empty()) {
				fn_head.insert(fn_head.end(), in.begin() + static_cast<ptrdiff_t>(pos),
				               in.begin() + static_cast<ptrdiff_t>(pos + count));
			} else {
				RInst inst;
				inst.w.assign(in.begin() + static_cast<ptrdiff_t>(pos),
				              in.begin() + static_cast<ptrdiff_t>(pos + count));
				m_blocks.back().insts.push_back(std::move(inst));
			}
			pos += count;
		}
		if (m_blocks.empty()) return false;
		m_loops_only = LaneRegionsMode() != 2;
		if (m_loops_only) FindLoopBlocks();

		std::vector<size_t> work;
		for (size_t i = m_blocks.size(); i-- > 0;) work.push_back(i);
		uint32_t guard = 0;
		while (!work.empty() && guard++ < 4096u) {
			const size_t index = work.back();
			work.pop_back();
			if (Process(index)) {
				// Process inserted [then, merge] after `index`: shift queued indices, then revisit
				// the merge block and the (shortened) original block.
				for (auto& w: work) {
					if (w > index) w += 2;
				}
				work.push_back(index + 2);
				work.push_back(index);
			}
		}
		if (m_stats.regions == 0) return false;

		bool add_false = false;
		if (m_false == NoId) {
			m_false   = m_bound++;
			add_false = true;
		}
		out.clear();
		out.insert(out.end(), in.begin(), in.begin() + static_cast<ptrdiff_t>(fn_begin));
		out[3] = m_bound;
		if (add_false) {
			out.push_back((3u << 16u) | static_cast<uint32_t>(spv::OpConstantFalse));
			out.push_back(m_bool);
			out.push_back(m_false);
		}
		for (auto& block: m_blocks) {
			for (auto& inst: block.insts) {
				if (inst.Op() != spv::OpPhi) continue;
				for (size_t w = 3; w < inst.w.size(); w += 2) {
					if (inst.w[w] == FalsePlaceholder) inst.w[w] = m_false;
				}
			}
		}
		out.insert(out.end(), fn_head.begin(), fn_head.end());
		for (const auto& block: m_blocks) {
			out.push_back((2u << 16u) | static_cast<uint32_t>(spv::OpLabel));
			out.push_back(block.label);
			for (const auto& inst: block.insts) out.insert(out.end(), inst.w.begin(), inst.w.end());
		}
		out.insert(out.end(), in.begin() + static_cast<ptrdiff_t>(fn_end), in.end());
		if (stats != nullptr) *stats = m_stats;
		return true;
	}

private:
	static constexpr uint32_t FalsePlaceholder = 0xffffffffu;

	[[nodiscard]] bool Movable(const RInst& inst) const {
		const auto op = inst.Op();
		if (op == spv::OpPhi || op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain ||
		    op == spv::OpPtrAccessChain || op == spv::OpArrayLength || op == spv::OpLoad ||
		    op == spv::OpSampledImage || op == spv::OpImage) {
			return false;
		}
		if (op >= spv::OpImageSampleImplicitLod && op <= spv::OpImageWrite) return false;
		if (op >= spv::OpImageQuerySizeLod && op <= spv::OpImageQuerySamples) return false;
		if (op == spv::OpImageQuerySize || op == spv::OpImageSparseSampleExplicitLod) return false;
		if (op == spv::OpExtInst && (inst.w.size() < 5 || inst.w[3] != m_glsl || m_glsl == NoId)) {
			return false;
		}
		return IsPureValueOp(op) && inst.Result() != NoId;
	}

	// Word indices of ID operands (excluding result type and result).
	template <typename F>
	static void ForIdOperands(const RInst& inst, F&& f) {
		const auto op = inst.Op();
		bool has_result = false;
		bool has_type   = false;
		spv::HasResultAndType(op, &has_result, &has_type);
		size_t first = (has_type ? 1u : 0u) + (has_result ? 1u : 0u) + 1u;
		size_t last  = inst.w.size();
		switch (op) {
			case spv::OpCompositeExtract: last = std::min<size_t>(last, 4); break;
			case spv::OpCompositeInsert:
			case spv::OpVectorShuffle: last = std::min<size_t>(last, 5); break;
			case spv::OpExtInst: first = 5; break;
			default: break;
		}
		for (size_t w = first; w < last; w++) f(w);
	}

	// Every word after the result that might be an ID (literals included: conservative for uses).
	template <typename F>
	static void ForAllOperandWords(const RInst& inst, F&& f) {
		bool has_result = false;
		bool has_type   = false;
		spv::HasResultAndType(inst.Op(), &has_result, &has_type);
		const size_t first = (has_type ? 1u : 0u) + (has_result ? 1u : 0u) + 1u;
		for (size_t w = first; w < inst.w.size(); w++) f(inst.w[w]);
	}

	// Operand words that may be IDs: exact for pure value ops (so literal indices and GLSL
	// instruction numbers are not mistaken for uses), every word otherwise (conservative).
	template <typename F>
	static void ForUseWords(const RInst& inst, F&& f) {
		if (IsPureValueOp(inst.Op()) && inst.Op() != spv::OpPhi) {
			ForIdOperands(inst, [&](size_t w) { f(inst.w[w]); });
		} else {
			ForAllOperandWords(inst, f);
		}
	}

	[[nodiscard]] static std::vector<uint32_t> Successors(const RInst& term) {
		std::vector<uint32_t> successors;
		switch (term.Op()) {
			case spv::OpBranch:
				if (term.w.size() >= 2) successors.push_back(term.w[1]);
				break;
			case spv::OpBranchConditional:
				if (term.w.size() >= 4) {
					successors.push_back(term.w[2]);
					successors.push_back(term.w[3]);
				}
				break;
			case spv::OpSwitch:
				if (term.w.size() >= 3) successors.push_back(term.w[2]);
				for (size_t w = 4; w < term.w.size(); w += 2) successors.push_back(term.w[w]);
				break;
			default: break;
		}
		return successors;
	}

	// Blocks inside a loop construct: reachable from a loop header without passing its merge block.
	void FindLoopBlocks() {
		std::unordered_map<uint32_t, size_t> index_of;
		for (size_t i = 0; i < m_blocks.size(); i++) index_of[m_blocks[i].label] = i;
		for (const auto& header: m_blocks) {
			uint32_t merge = NoId;
			for (const auto& inst: header.insts) {
				if (inst.Op() == spv::OpLoopMerge && inst.w.size() >= 3) merge = inst.w[1];
			}
			if (merge == NoId) continue;
			std::vector<uint32_t> stack {header.label};
			std::unordered_set<uint32_t> seen {header.label};
			while (!stack.empty()) {
				const uint32_t label = stack.back();
				stack.pop_back();
				const auto it = index_of.find(label);
				if (it == index_of.end() || m_blocks[it->second].insts.empty()) continue;
				for (const auto next: Successors(m_blocks[it->second].insts.back())) {
					if (next != merge && seen.insert(next).second) stack.push_back(next);
				}
			}
			m_loop_blocks.insert(seen.begin(), seen.end());
		}
	}

	struct Use {
		size_t block;
		size_t index;
	};

	void BuildUses() {
		m_uses.clear();
		for (size_t b = 0; b < m_blocks.size(); b++) {
			const auto& insts = m_blocks[b].insts;
			for (size_t i = 0; i < insts.size(); i++) {
				ForUseWords(insts[i], [&](uint32_t id) {
					if (id != NoId && id < m_bound) m_uses[id].push_back(Use {b, i});
				});
			}
		}
	}

	[[nodiscard]] const std::vector<Use>& UsesOf(uint32_t id) const {
		static const std::vector<Use> empty;
		const auto it = m_uses.find(id);
		return it != m_uses.end() ? it->second : empty;
	}

	struct Export {
		uint32_t value; // computed inside the region
		uint32_t old;   // value for lanes where c is false
		uint32_t type;
	};

	// select(c, a, b) -> a ; LogicalAnd(c, a) / LogicalAnd(a, c) -> a
	[[nodiscard]] uint32_t ExportedValue(const RInst& inst, uint32_t c) const {
		if (inst.Op() == spv::OpSelect && inst.w.size() == 6 && inst.w[3] == c) return inst.w[4];
		if (inst.Op() == spv::OpLogicalAnd && inst.w.size() == 5) {
			if (inst.w[3] == c && inst.w[4] != c) return inst.w[4];
			if (inst.w[4] == c && inst.w[3] != c) return inst.w[3];
		}
		return NoId;
	}

	bool Process(size_t block_index) {
		if (m_loops_only && m_loop_blocks.count(m_blocks[block_index].label) == 0) return false;
		auto& insts = m_blocks[block_index].insts;
		size_t nphi = 0;
		while (nphi < insts.size() && insts[nphi].Op() == spv::OpPhi) nphi++;
		if (insts.empty()) return false;
		size_t ntail = 1;
		if (insts.size() >= 2 && (insts[insts.size() - 2].Op() == spv::OpSelectionMerge ||
		                          insts[insts.size() - 2].Op() == spv::OpLoopMerge)) {
			ntail = 2;
		}
		for (const auto& inst: insts) {
			if (inst.Op() == spv::OpLoopMerge) return false;
		}
		if (insts.size() < nphi + ntail) return false;
		const size_t mid_begin = nphi;
		const size_t mid_end   = insts.size() - ntail;
		if (mid_end <= mid_begin) return false;

		// Candidate conditions, most selects first.
		std::unordered_map<uint32_t, uint32_t> cond_count;
		for (size_t i = mid_begin; i < mid_end; i++) {
			if (insts[i].Op() == spv::OpSelect && insts[i].w.size() == 6) cond_count[insts[i].w[3]]++;
		}
		std::vector<std::pair<uint32_t, uint32_t>> conds;
		for (const auto& [c, n]: cond_count) {
			if (n >= MinExports) conds.emplace_back(n, c);
		}
		std::sort(conds.begin(), conds.end(), std::greater<>());
		if (conds.size() > 4) conds.resize(4);
		if (conds.empty()) return false;

		BuildUses();
		for (const auto& [n, c]: conds) {
			if (TryCondition(block_index, mid_begin, mid_end, c)) return true;
		}
		return false;
	}

	bool TryCondition(size_t block_index, size_t mid_begin, size_t mid_end, uint32_t c) {
		auto&        insts = m_blocks[block_index].insts;
		const size_t n     = mid_end - mid_begin;
		std::unordered_map<uint32_t, size_t> pos; // result -> mid index
		for (size_t k = 0; k < n; k++) {
			const auto r = insts[mid_begin + k].Result();
			if (r != NoId) pos[r] = k;
		}
		std::unordered_set<uint32_t> region;
		// Backward pass: a value belongs to the region when every use is in this block, later, and
		// is either a region instruction or an exporter that takes it as `new`.
		const auto only_feeds_region = [&](size_t k) {
			const auto& inst = insts[mid_begin + k];
			const auto  r    = inst.Result();
			const auto& uses = UsesOf(r);
			if (uses.empty()) return false;
			for (const auto& use: uses) {
				if (use.block != block_index || use.index < mid_begin || use.index >= mid_end ||
				    use.index <= mid_begin + k) {
					return false;
				}
				const auto& user = insts[use.index];
				const auto  ur   = user.Result();
				if (ur != NoId && region.count(ur) != 0) continue;
				if (ExportedValue(user, c) == r) {
					uint32_t refs = 0;
					ForUseWords(user, [&](uint32_t id) { refs += id == r ? 1u : 0u; });
					if (refs == 1 && (user.Op() != spv::OpSelect || region.count(user.w[5]) == 0)) continue;
				}
				return false;
			}
			return true;
		};
		const auto build_region = [&](const std::unordered_set<uint32_t>* allowed) {
			std::unordered_set<uint32_t> previous = std::move(region);
			region.clear();
			for (size_t k = n; k-- > 0;) {
				const auto& inst = insts[mid_begin + k];
				const auto  r    = inst.Result();
				if (r == NoId || r == c || !Movable(inst) || ExportedValue(inst, c) != NoId) continue;
				if (allowed != nullptr && allowed->count(r) == 0) continue;
				if (only_feeds_region(k)) region.insert(r);
			}
			(void)previous;
		};
		build_region(nullptr);
		if (region.empty()) return false;

		std::unordered_map<uint32_t, Export> exports;
		size_t split = n;
		for (int iteration = 0; iteration < 64; iteration++) {
			exports.clear();
			for (size_t k = 0; k < n; k++) {
				const auto& inst = insts[mid_begin + k];
				const auto  a    = ExportedValue(inst, c);
				if (a == NoId || region.count(a) == 0) continue;
				if (inst.Op() == spv::OpSelect) {
					if (region.count(inst.w[5]) != 0) continue;
					exports[inst.Result()] = Export {a, inst.w[5], inst.w[1]};
				} else {
					exports[inst.Result()] = Export {a, FalsePlaceholder, m_bool};
				}
			}
			// Split point: first instruction outside the region that reads an exported value.
			split = n;
			for (size_t k = 0; k < n && split == n; k++) {
				const auto& inst = insts[mid_begin + k];
				const auto  r    = inst.Result();
				if (r != NoId && (region.count(r) != 0 || exports.count(r) != 0)) continue;
				ForUseWords(inst, [&](uint32_t id) {
					if (exports.count(id) != 0) split = k;
				});
			}
			if (const auto it = pos.find(c); it != pos.end() && it->second >= split) return false;
			if (exports.count(c) != 0) return false;
			std::unordered_set<uint32_t> after; // non-region results placed after the merge
			for (size_t k = split; k < n; k++) {
				const auto r = insts[mid_begin + k].Result();
				if (r != NoId && region.count(r) == 0 && exports.count(r) == 0) after.insert(r);
			}
			std::unordered_set<uint32_t> bad;
			for (size_t k = 0; k < n; k++) {
				const auto& inst = insts[mid_begin + k];
				const auto  r    = inst.Result();
				if (r == NoId || region.count(r) == 0) continue;
				ForIdOperands(inst, [&](size_t w) {
					uint32_t id = inst.w[w];
					if (const auto e = exports.find(id); e != exports.end()) id = e->second.value;
					if (after.count(id) != 0) bad.insert(r);
				});
			}
			for (const auto& [s, e]: exports) {
				uint32_t old = e.old;
				while (exports.count(old) != 0) old = exports[old].old;
				if (after.count(old) != 0) bad.insert(e.value);
			}
			if (bad.empty()) break;
			std::unordered_set<uint32_t> allowed = region;
			for (const auto id: bad) allowed.erase(id);
			build_region(&allowed);
			if (region.empty()) return false;
		}
		if (exports.size() < MinExports || region.size() < MinMoved) return false;

		// Rewrite: [phis, before...] -> if (c) [region] -> [export phis, after..., tail].
		const uint32_t then_label  = m_bound++;
		const uint32_t merge_label = m_bound++;
		const uint32_t self_label  = m_blocks[block_index].label;
		std::vector<RInst> head(insts.begin(), insts.begin() + static_cast<ptrdiff_t>(mid_begin));
		std::vector<RInst> body;
		std::vector<RInst> merge;
		for (size_t k = 0; k < n; k++) {
			const auto& inst = insts[mid_begin + k];
			const auto  r    = inst.Result();
			if (r != NoId && region.count(r) != 0) {
				RInst moved = inst;
				ForIdOperands(moved, [&](size_t w) {
					if (const auto e = exports.find(moved.w[w]); e != exports.end()) moved.w[w] = e->second.value;
				});
				body.push_back(std::move(moved));
			} else if (r != NoId && exports.count(r) != 0) {
				const auto& e   = exports[r];
				uint32_t    old = e.old;
				while (exports.count(old) != 0) old = exports[old].old;
				merge.push_back(MakeInst(spv::OpPhi, {e.type, r, e.value, then_label, old, self_label}));
			} else if (k < split) {
				head.push_back(inst);
			}
		}
		const size_t export_phis = merge.size();
		for (size_t k = split; k < n; k++) {
			const auto& inst = insts[mid_begin + k];
			const auto  r    = inst.Result();
			if (r != NoId && (region.count(r) != 0 || exports.count(r) != 0)) continue;
			merge.push_back(inst);
		}
		merge.insert(merge.end(), insts.begin() + static_cast<ptrdiff_t>(mid_end), insts.end());
		head.push_back(MakeInst(spv::OpSelectionMerge, {merge_label, 0}));
		head.push_back(MakeInst(spv::OpBranchConditional, {c, then_label, merge_label}));
		body.push_back(MakeInst(spv::OpBranch, {merge_label}));

		// Successors now branch from the merge block.
		const std::vector<uint32_t> successors = Successors(merge.back());
		if (m_loops_only) {
			m_loop_blocks.insert(then_label);
			m_loop_blocks.insert(merge_label);
		}
		const uint32_t moved_count = static_cast<uint32_t>(body.size() - 1);
		insts = std::move(head);
		m_blocks.insert(m_blocks.begin() + static_cast<ptrdiff_t>(block_index + 1),
		                RBlock {then_label, std::move(body)});
		m_blocks.insert(m_blocks.begin() + static_cast<ptrdiff_t>(block_index + 2),
		                RBlock {merge_label, std::move(merge)});
		for (auto& block: m_blocks) {
			if (std::find(successors.begin(), successors.end(), block.label) == successors.end()) continue;
			for (auto& inst: block.insts) {
				if (inst.Op() != spv::OpPhi) break;
				for (size_t w = 4; w < inst.w.size(); w += 2) {
					if (inst.w[w] == self_label) inst.w[w] = merge_label;
				}
			}
		}
		m_stats.regions++;
		m_stats.moved += moved_count;
		m_stats.phis += static_cast<uint32_t>(export_phis);
		return true;
	}

	uint32_t                                         m_bound = 0;
	uint32_t                                         m_bool  = NoId;
	uint32_t                                         m_false = NoId;
	uint32_t                                         m_glsl  = NoId;
	std::vector<RBlock>                              m_blocks;
	std::unordered_map<uint32_t, std::vector<Use>>   m_uses;
	LaneRegionStats                                  m_stats;
	bool                                             m_loops_only = true;
	std::unordered_set<uint32_t>                     m_loop_blocks;
};

} // namespace

uint32_t RewriteLaneRegions(std::vector<uint32_t>& spirv, LaneRegionStats* stats) {
	LaneRegions           pass;
	std::vector<uint32_t> out;
	LaneRegionStats       local;
	if (!pass.Run(spirv, out, &local)) {
		if (stats != nullptr) *stats = local;
		return 0;
	}
	if (stats != nullptr) *stats = local;
	spirv = std::move(out);
	return local.regions;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
