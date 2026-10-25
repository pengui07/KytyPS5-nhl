#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <optional>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

// Scalar f16 sources are read with Read16LaneBits, which also selects a word with SDWA (WORD_0,
// WORD_1): those take the native path too (NHL 27's crowd vertex shaders: 480 V_SUB_F16 each).
static bool NativeScalarF16Source(const Decoder::Operand& op) {
	return (op.sdwa_sel == 6u || (ShaderNativeF16Sdwa() && (op.sdwa_sel == 4u || op.sdwa_sel == 5u))) &&
	       !op.dpp && !op.dpp8;
}

bool Translator::NativeF16Eligible(const Decoder::Operand& op) const {
	return op.sdwa_sel == 6u && !op.dpp && !op.dpp8;
}

// Both f16 lanes of a source as one u32 (lo = lane read for the low result, hi = for the high one).
IR::U32 Translator::NativePackedF16Source(const Decoder::Operand& op) {
	const bool plain_reg = (op.kind == Decoder::OperandKind::Vgpr || op.kind == Decoder::OperandKind::Sgpr) &&
	                       !op.absolute;
	if (!plain_reg) {
		return PackU16Lanes(Read16LaneBits(op, false), Read16LaneBits(op, true));
	}
	const auto raw = ReadRawU32(op);
	const auto c16 = IR::U32(IR::Value(16u));
	IR::U32    v   = raw;
	if (ShaderF16LaneViews()) {
		// Experiment F16LaneViews: the lanes op_sel/op_sel_hi pick, as an f16vec2 shuffle the host
		// compiler folds into the packed instruction's own operand selects.
		if (op.op_sel || !op.op_sel_hi) {
			v = IR::U32(ir.Emit(IR::ValueOpcode::F16x2Swizzle,
			                    {raw, IR::Value(op.op_sel ? 1u : 0u), IR::Value(op.op_sel_hi ? 1u : 0u)}));
		}
	} else if (!op.op_sel && op.op_sel_hi) {
		v = raw;
	} else if (op.op_sel && !op.op_sel_hi) {
		v = ir.BitwiseOr(ir.ShiftLeftLogical(raw, c16), ir.ShiftRightLogical(raw, c16));
	} else if (!op.op_sel && !op.op_sel_hi) {
		v = ir.BitwiseOr(ir.BitwiseAnd(raw, IR::U32(IR::Value(0xffffu))), ir.ShiftLeftLogical(raw, c16));
	} else {
		v = ir.BitwiseOr(ir.ShiftRightLogical(raw, c16), ir.BitwiseAnd(raw, IR::U32(IR::Value(0xffff0000u))));
	}
	const uint32_t flip = (op.negate ? 0x8000u : 0u) | (op.negate_hi ? 0x80000000u : 0u);
	if (flip != 0u) {
		v = ir.BitwiseXor(v, IR::U32(IR::Value(flip)));
	}
	return v;
}

// Scalar f16 op through the packed native opcodes: the operand lane goes in the low half (the high
// half is zero and discarded).
bool Translator::NativeScalarF16(const Decoder::Instruction& inst, IR::ValueOpcode opcode, bool reverse,
                                 bool accumulator) {
	if (!ShaderNativeFp16() || inst.dst.omod != 0u) {
		return false;
	}
	IR::ValueOpcode native = IR::ValueOpcode::Void;
	bool            negate_b = false;
	switch (opcode) {
		case IR::ValueOpcode::FPAdd32: native = IR::ValueOpcode::FPAdd16x2; break;
		case IR::ValueOpcode::FPSub32: native = IR::ValueOpcode::FPAdd16x2; negate_b = true; break;
		case IR::ValueOpcode::FPMul32: native = IR::ValueOpcode::FPMul16x2; break;
		case IR::ValueOpcode::FPFma32: native = IR::ValueOpcode::FPFma16x2; break;
		default: return false;
	}
	const bool three = native == IR::ValueOpcode::FPFma16x2;
	if (three ? (!accumulator && inst.src_count != 3u) : inst.src_count != 2u) {
		return false;
	}
	const Decoder::Operand* srcs[3] = {&inst.src0, &inst.src1, nullptr};
	if (three) srcs[2] = accumulator ? &inst.dst : &inst.src2;
	if (reverse) std::swap(srcs[0], srcs[1]);
	for (const auto* s: srcs) {
		if (s != nullptr && !NativeScalarF16Source(*s)) return false;
	}
	if (ShaderF16LaneViews()) {
		NativeScalarF16Lanes(inst, native, negate_b, three, srcs);
		fp16_native = true;
		return true;
	}
	const auto lane = [&](const Decoder::Operand& op) { return Read16LaneBits(op, false); };
	const auto a    = lane(*srcs[0]);
	auto       b    = lane(*srcs[1]);
	if (negate_b) b = ir.BitwiseXor(b, IR::U32(IR::Value(0x8000u)));
	IR::U32 result;
	if (three) {
		result = IR::U32(ir.Emit(native, {a, b, lane(*srcs[2])}));
	} else {
		result = IR::U32(ir.Emit(native, {a, b}));
	}
	if (inst.dst.clamp) {
		result = IR::U32(ir.Emit(IR::ValueOpcode::FPClamp16x2, {result}));
	}
	Write16Bits(DestinationOperand(inst), result);
	fp16_native = true;
	return true;
}

// Experiment F16LaneViews: NativeScalarF16 with the source lanes as F16LaneExtract (plain register
// halves; other sources as Read16LaneBits makes them), the FP*16Low form of the op, and the result
// written into the destination half with F16LaneInsert when the other half is kept.
IR::U32 Translator::NativeF16Lane(const Decoder::Operand& op) {
	if ((op.kind == Decoder::OperandKind::Vgpr || op.kind == Decoder::OperandKind::Sgpr) && !op.absolute &&
	    !op.negate && !op.dpp && !op.dpp8 && (op.sdwa_sel == 4u || op.sdwa_sel == 5u || op.sdwa_sel == 6u)) {
		const uint32_t index = op.sdwa_sel == 5u ? 1u : op.sdwa_sel == 4u ? 0u : op.op_sel ? 1u : 0u;
		return IR::U32(ir.Emit(IR::ValueOpcode::F16LaneExtract, {ReadRawU32(op), IR::Value(index)}));
	}
	return Read16LaneBits(op, false);
}

// The low half of `result` into the instruction's destination half (the other half kept when the
// destination says so), clamped first with the clamp modifier.
void Translator::WriteNativeF16Lane(const Decoder::Instruction& inst, IR::U32 result) {
	if (inst.dst.clamp) {
		result = IR::U32(ir.Emit(IR::ValueOpcode::FPClamp16x2, {result}));
	}
	auto destination = DestinationOperand(inst);
	if (!destination.explicit_sdwa_dst) {
		if (destination.sdwa_sel == 6u) {
			destination.sdwa_sel = 4u;
		}
		destination.sdwa_dst_unused = 2u;
	}
	if (destination.kind == Decoder::OperandKind::Vgpr && destination.sdwa_dst_unused == 2u &&
	    (destination.sdwa_sel == 4u || destination.sdwa_sel == 5u)) {
		const auto plain = PlainOperand(destination);
		WriteRawU32(plain, IR::U32(ir.Emit(IR::ValueOpcode::F16LaneInsert,
		                                   {ReadRawU32(plain), result, IR::Value(destination.sdwa_sel == 5u ? 1u : 0u)})));
		return;
	}
	Write16Bits(DestinationOperand(inst), result);
}

// Experiment NativeF16Ops: a scalar f16 unary op as the host's f16 instruction.
bool Translator::NativeF16Unary(const Decoder::Instruction& inst, IR::F16UnaryOp op) {
	if (!ShaderNativeF16Ops() || !ShaderF16LaneViews() || inst.dst.omod != 0u || !NativeScalarF16Source(inst.src0)) {
		return false;
	}
	WriteNativeF16Lane(inst, IR::U32(ir.Emit(IR::ValueOpcode::FPUnary16Low,
	                                         {NativeF16Lane(inst.src0), IR::Value(static_cast<uint32_t>(op))})));
	fp16_native = true;
	return true;
}

// Experiment NativeF16Ops: V_MIN/MAX_F16 and V_PK_MIN/MAX_F16 as NMin/NMax on f16 (the host's
// v_min_f16 / v_pk_max_f16; as with NativeMinMax for f32, only signalling NaNs and the sign of
// equal zeros may differ).
bool Translator::NativeF16MinMax(const Decoder::Instruction& inst, bool max, bool packed) {
	if (!ShaderNativeF16Ops() || !ShaderF16LaneViews() || inst.dst.omod != 0u || inst.src_count != 2u) {
		return false;
	}
	if (packed) {
		if (!NativeF16Eligible(inst.src0) || !NativeF16Eligible(inst.src1)) {
			return false;
		}
		auto result = IR::U32(ir.Emit(IR::ValueOpcode::FPMinMax16x2,
		                              {NativePackedF16Source(inst.src0), NativePackedF16Source(inst.src1),
		                               IR::Value(max ? 1u : 0u)}));
		if (inst.dst.clamp) {
			result = IR::U32(ir.Emit(IR::ValueOpcode::FPClamp16x2, {result}));
		}
		auto raw  = DestinationOperand(inst);
		raw.omod  = 0u;
		raw.clamp = false;
		WriteOperand(raw, result);
		fp16_native = true;
		return true;
	}
	if (!NativeScalarF16Source(inst.src0) || !NativeScalarF16Source(inst.src1)) {
		return false;
	}
	WriteNativeF16Lane(inst, IR::U32(ir.Emit(IR::ValueOpcode::FPMinMax16Low,
	                                         {NativeF16Lane(inst.src0), NativeF16Lane(inst.src1),
	                                          IR::Value(max ? 1u : 0u)})));
	fp16_native = true;
	return true;
}

void Translator::NativeScalarF16Lanes(const Decoder::Instruction& inst, IR::ValueOpcode native, bool subtract,
                                      bool three, const Decoder::Operand* const* srcs) {
	const auto lane = [&](const Decoder::Operand& op) { return NativeF16Lane(op); };
	IR::ValueOpcode opcode = IR::ValueOpcode::FPFma16Low;
	switch (native) {
		case IR::ValueOpcode::FPAdd16x2: opcode = subtract ? IR::ValueOpcode::FPSub16Low : IR::ValueOpcode::FPAdd16Low; break;
		case IR::ValueOpcode::FPMul16x2: opcode = IR::ValueOpcode::FPMul16Low; break;
		default: break;
	}
	const auto a = lane(*srcs[0]);
	const auto b = lane(*srcs[1]);
	const auto result = three ? IR::U32(ir.Emit(opcode, {a, b, lane(*srcs[2])})) : IR::U32(ir.Emit(opcode, {a, b}));
	WriteNativeF16Lane(inst, result);
}

void Translator::PackedFloat16(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool accumulator, bool quiet_snan) {
	if ((opcode == IR::ValueOpcode::FPMin32 || opcode == IR::ValueOpcode::FPMax32) && !accumulator &&
	    NativeF16MinMax(inst, opcode == IR::ValueOpcode::FPMax32, true)) {
		return;
	}
	if (quiet_snan && ShaderFastMinMax()) {
		// KYTY_FAST_MINMAX: like the f32 min/max path, v_pk_min/max_f16 skip the signaling-NaN
		// quieting (games do not feed sNaN bit patterns to min/max).
		quiet_snan = false;
	}
	if (ShaderNativeFp16() && !quiet_snan) {
		// Native packed f16. Sources are packed into one u32 holding both halves exactly as the
		// fallback reads them (op_sel/op_sel_hi, neg/neg_hi, abs, inline constants); omod falls back.
		IR::ValueOpcode native = IR::ValueOpcode::Void;
		switch (opcode) {
			case IR::ValueOpcode::FPAdd32: native = IR::ValueOpcode::FPAdd16x2; break;
			case IR::ValueOpcode::FPMul32: native = IR::ValueOpcode::FPMul16x2; break;
			case IR::ValueOpcode::FPFma32: native = IR::ValueOpcode::FPFma16x2; break;
			default: break;
		}
		const bool three = native == IR::ValueOpcode::FPFma16x2;
		const bool shape_ok = three ? (accumulator || inst.src_count == 3u) : (inst.src_count == 2u);
		if (native != IR::ValueOpcode::Void && shape_ok && inst.dst.omod == 0u &&
		    NativeF16Eligible(inst.src0) && NativeF16Eligible(inst.src1) &&
		    (!three || NativeF16Eligible(accumulator ? inst.dst : inst.src2))) {
			const auto a = NativePackedF16Source(inst.src0);
			const auto b = NativePackedF16Source(inst.src1);
			IR::U32    result;
			if (three) {
				const auto c = NativePackedF16Source(accumulator ? inst.dst : inst.src2);
				result       = IR::U32(ir.Emit(native, {a, b, c}));
			} else {
				result = IR::U32(ir.Emit(native, {a, b}));
			}
			if (inst.dst.clamp) {
				result = IR::U32(ir.Emit(IR::ValueOpcode::FPClamp16x2, {result}));
			}
			auto raw  = DestinationOperand(inst);
			raw.omod  = 0u;
			raw.clamp = false;
			WriteOperand(raw, result);
			fp16_native = true;
			return;
		}
	}
	const auto translate_lane = [&](bool high) {
		const auto lhs = ReadF16LaneAsF32(inst.src0, high);
		const auto rhs = ReadF16LaneAsF32(inst.src1, high);
		IR::F32    result;
		if (accumulator) {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs, ReadF16LaneAsF32(inst.dst, high)}));
		} else if (inst.src_count == 3u) {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs, ReadF16LaneAsF32(inst.src2, high)}));
		} else {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs}));
		}
		return ApplyF32ResultModifiers(inst.dst, result);
	};
	auto raw    = DestinationOperand(inst);
	raw.omod    = 0u;
	raw.clamp   = false;
	auto result = PackHalf2x16(translate_lane(false), translate_lane(true));
	if (quiet_snan) {
		const auto quiet_snan_lane = [&](const Decoder::Operand& operand, bool high) {
			const auto bits     = ReadU16LaneAsU32(operand, high, false);
			const auto exponent = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x7e00u)));
			const auto payload  = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x01ffu)));
			const auto snan     = ir.LogicalAnd(ir.IEqual(exponent, IR::U32(IR::Value(0x7c00u))),
			                                    ir.INotEqual(payload, IR::U32(IR::Value(0u))));
			return std::pair {snan, ir.BitwiseOr(bits, IR::U32(IR::Value(0x0200u)))};
		};
		const auto override_lane = [&](bool high) {
			const auto [lhs_snan, lhs_quiet] = quiet_snan_lane(inst.src0, high);
			const auto [rhs_snan, rhs_quiet] = quiet_snan_lane(inst.src1, high);
			const auto normal = high ? ir.ShiftRightLogical(result, IR::U32(IR::Value(16u)))
			                         : ir.BitwiseAnd(result, IR::U32(IR::Value(0xffffu)));
			return ir.Select(lhs_snan, lhs_quiet, ir.Select(rhs_snan, rhs_quiet, normal));
		};
		result = PackU16Lanes(override_lane(false), override_lane(true));
	}
	WriteOperand(raw, result);
}

void Translator::Float16Unary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool invalid_negative) {
	{
		std::optional<IR::F16UnaryOp> native;
		switch (opcode) {
			case IR::ValueOpcode::FPRecip32: native = IR::F16UnaryOp::Recip; break;
			case IR::ValueOpcode::FPSqrt: native = IR::F16UnaryOp::Sqrt; break;
			case IR::ValueOpcode::FPRecipSqrt32: native = IR::F16UnaryOp::RecipSqrt; break;
			case IR::ValueOpcode::FPLog2: native = IR::F16UnaryOp::Log2; break;
			case IR::ValueOpcode::FPExp2: native = IR::F16UnaryOp::Exp2; break;
			case IR::ValueOpcode::FPFloor32: native = IR::F16UnaryOp::Floor; break;
			case IR::ValueOpcode::FPCeil32: native = IR::F16UnaryOp::Ceil; break;
			case IR::ValueOpcode::FPTrunc32: native = IR::F16UnaryOp::Trunc; break;
			case IR::ValueOpcode::FPRoundEven32: native = IR::F16UnaryOp::RoundEven; break;
			default: break;
		}
		if (native && NativeF16Unary(inst, *native)) {
			return;
		}
	}
	const auto argument = ReadF16AsF32(inst.src0);
	auto       result   = IR::F32(ir.Emit(opcode, {argument}));
	if (!invalid_negative) {
		WriteF16(DestinationOperand(inst), result);
		return;
	}
	const auto negative =
	    IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {argument, IR::Value::F32(0.0f)}));
	result             = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(negative, invalid, bits));
}

void Translator::Float16Trig(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto argument = ReadF16AsF32(inst.src0);
	const auto magnitude =
	    ir.BitwiseAnd(ir.BitCastU32(argument), IR::U32(IR::Value(0x7fffffffu)));
	auto result = IR::F32(ir.Emit(opcode, {argument}));
	if (opcode == IR::ValueOpcode::FPSin) {
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto whole = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.0f)}));
		const auto half = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.5f)}));
		const auto nonzero = ir.INotEqual(magnitude, IR::U32(IR::Value(0u)));
		const auto cardinal = ir.LogicalOr(ir.LogicalAnd(whole, nonzero), half);
		result = IR::F32(ir.Emit(
		    IR::ValueOpcode::SelectF32, {cardinal, IR::Value::F32(0.0f), result}));
	} else {
		// The architectural quarter-cycle result is exactly zero. Evaluating cos(pi/2) with a
		// rounded F32 PI can otherwise produce a half-precision subnormal instead.
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto quarter = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.25f)}));
		const auto three_quarters = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.75f)}));
		result = IR::F32(
		    ir.Emit(IR::ValueOpcode::SelectF32,
		            {ir.LogicalOr(quarter, three_quarters), IR::Value::F32(0.0f), result}));
	}

	const auto infinite = ir.IEqual(magnitude, IR::U32(IR::Value(0x7f800000u)));
	result              = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(infinite, invalid, bits));
}

void Translator::Float16Binary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool reverse) {
	if ((opcode == IR::ValueOpcode::FPMin32 || opcode == IR::ValueOpcode::FPMax32) && !reverse &&
	    NativeF16MinMax(inst, opcode == IR::ValueOpcode::FPMax32, false)) {
		return;
	}
	if (NativeScalarF16(inst, opcode, reverse, false)) {
		return;
	}
	const auto lhs = ReadF16AsF32(reverse ? inst.src1 : inst.src0);
	const auto rhs = ReadF16AsF32(reverse ? inst.src0 : inst.src1);
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {lhs, rhs})));
}

void Translator::Float16Ternary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                bool accumulator, bool mix) {
	// Experiment NativeF16Ops: V_MIN3/MAX3_F16 as two f16 NMin/NMax (the host's v_min3_f16).
	if ((opcode == IR::ValueOpcode::FPMinTri32 || opcode == IR::ValueOpcode::FPMaxTri32) && !accumulator && !mix &&
	    ShaderNativeF16Ops() && ShaderF16LaneViews() && inst.dst.omod == 0u && inst.src_count == 3u &&
	    NativeScalarF16Source(inst.src0) && NativeScalarF16Source(inst.src1) && NativeScalarF16Source(inst.src2)) {
		const auto max   = IR::Value(opcode == IR::ValueOpcode::FPMaxTri32 ? 1u : 0u);
		const auto first = IR::U32(ir.Emit(IR::ValueOpcode::FPMinMax16Low,
		                                   {NativeF16Lane(inst.src0), NativeF16Lane(inst.src1), max}));
		WriteNativeF16Lane(inst, IR::U32(ir.Emit(IR::ValueOpcode::FPMinMax16Low,
		                                         {first, NativeF16Lane(inst.src2), max})));
		fp16_native = true;
		return;
	}
	if (!mix && inst.opcode != Decoder::Opcode::V_FMAMK_F16 && inst.opcode != Decoder::Opcode::V_FMAAK_F16 &&
	    NativeScalarF16(inst, opcode, false, accumulator)) {
		return;
	}
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		args[index] = mix ? IR::Value(ReadMixF32(operand)) : IR::Value(ReadF16AsF32(operand));
	}
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {args[0], args[1], args[2]})));
}

void Translator::FloatUnary(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto type = IR::ArgTypeOf(opcode, 0);
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {ReadOperand(inst.src0, type)}));
}

void Translator::FloatBinary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                             bool reverse) {
	std::array<IR::Value, 2> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = SourceAt(inst, reverse ? 1u - index : index);
		args[index]        = ReadOperand(operand, IR::ArgTypeOf(opcode, index));
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1]}));
}

void Translator::FloatTernary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool accumulator, bool mix) {
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		const auto type    = IR::ArgTypeOf(opcode, index);
		args[index]        = type == IR::Type::F32 && mix ? IR::Value(ReadMixF32(operand))
		                                                  : ReadOperand(operand, type);
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1], args[2]}));
}

void Translator::V_FREXP_MANT_F32(const Decoder::Instruction& inst) {
	const auto bits = ReadU32(inst.src0);
	const auto exponent =
	    IR::U32(ir.Emit(IR::ValueOpcode::BitFieldUExtract, {bits, IR::Value(23u), IR::Value(8u)}));
	const auto mantissa = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x007fffffu)));
	const auto sign     = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x80000000u)));
	const auto base     = ir.BitwiseOr(sign, IR::U32(IR::Value(0x3f000000u)));
	const auto normal   = ir.BitwiseOr(base, mantissa);
	const auto msb      = IR::U32(ir.Emit(IR::ValueOpcode::FindUMsb32, {mantissa}));
	const auto shift    = ir.ISub(IR::U32(IR::Value(23u)), msb);
	const auto fraction =
	    ir.BitwiseAnd(ir.ShiftLeftLogical(mantissa, shift), IR::U32(IR::Value(0x007fffffu)));
	const auto subnormal = ir.BitwiseOr(base, fraction);
	const auto zero      = ir.IEqual(mantissa, IR::U32(IR::Value(0u)));
	const auto finite    = ir.Select(ir.INotEqual(exponent, IR::U32(IR::Value(0u))), normal,
	                                 ir.Select(zero, bits, subnormal));
	const auto result    = ir.Select(ir.IEqual(exponent, IR::U32(IR::Value(0xffu))), bits, finite);
	WriteOperand(DestinationOperand(inst), ir.BitCastF32(result));
}

void Translator::V_FRACT(const Decoder::Instruction& inst, bool half) {
	if (half && NativeF16Unary(inst, IR::F16UnaryOp::Fract)) {
		return;
	}
	// DX fract returns [0, 1): x - floor(x) rounds to 1.0 for tiny negative x, and the
	// hardware clamps that to the largest value below one (LLVM matches
	// minnum(x - floor(x), nextafter(1.0, 0.0)) to V_FRACT for the same reason).
	const auto source = half ? ReadF16AsF32(inst.src0) : IR::F32(ReadOperand(inst.src0, IR::Type::F32));
	auto       result = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {source}));
	if (!half && ShaderNativeFract()) {
		// The AMD compiler emits Fract as v_fract_f32, which clamps like the guest's.
		WriteOperand(DestinationOperand(inst), result);
		return;
	}
	// Clamp in the destination precision: an F16 result above 0x3bff would round to 1.0.
	const auto below_one =
	    IR::F32(IR::Value::F32(std::bit_cast<float>(half ? 0x3f7fe000u : 0x3f7fffffu)));
	const auto above = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThan32, {result, below_one}));
	result           = SelectF32(above, below_one, result);
	if (half) {
		WriteF16(DestinationOperand(inst), result);
	} else {
		WriteOperand(DestinationOperand(inst), result);
	}
}

void Translator::V_DOT2C_F32_F16(const Decoder::Instruction& inst) {
	auto a          = inst.src0;
	a.op_sel        = false;
	a.op_sel_hi     = true;
	auto b          = inst.src1;
	b.op_sel        = false;
	b.op_sel_hi     = true;
	const auto a_lo = ReadF16LaneAsF32(a, false);
	const auto a_hi = ReadF16LaneAsF32(a, true);
	const auto b_lo = ReadF16LaneAsF32(b, false);
	const auto b_hi = ReadF16LaneAsF32(b, true);
	const auto acc  = ReadMixF32(inst.dst);
	const auto lo   = IR::F32(ir.Emit(IR::ValueOpcode::FPFma32, {a_lo, b_lo, acc}));
	WriteOperand(DestinationOperand(inst), ir.Emit(IR::ValueOpcode::FPFma32, {a_hi, b_hi, lo}));
}

void Translator::V_CUBEID_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 0u);
}

void Translator::V_CUBESC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 1u);
}

void Translator::V_CUBETC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 2u);
}

void Translator::V_CUBEMA_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 3u);
}

void Translator::FloatCube(const Decoder::Instruction& inst, uint32_t result_kind) {
	const auto x  = ReadMixF32(inst.src0);
	const auto y  = ReadMixF32(inst.src1);
	const auto z  = ReadMixF32(inst.src2);
	const auto nx = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {x}));
	const auto ny = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {y}));
	const auto nz = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {z}));
	const auto ax = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {x}));
	const auto ay = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {y}));
	const auto az = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {z}));
	const auto z_face =
	    ir.LogicalAnd(IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ax})),
	                  IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ay})));
	const auto y_face = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {ay, ax}));
	const auto x_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {x, IR::Value::F32(0.0f)}));
	const auto y_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {y, IR::Value::F32(0.0f)}));
	const auto z_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {z, IR::Value::F32(0.0f)}));
	const auto select_face = [&](IR::F32 x_value, IR::F32 y_value, IR::F32 z_value) {
		return SelectF32(z_face, z_value, SelectF32(y_face, y_value, x_value));
	};
	IR::F32 result;
	switch (result_kind) {
		case 0u:
			result = select_face(
			    SelectF32(x_neg, IR::F32(IR::Value::F32(1.0f)), IR::F32(IR::Value::F32(0.0f))),
			    SelectF32(y_neg, IR::F32(IR::Value::F32(3.0f)), IR::F32(IR::Value::F32(2.0f))),
			    SelectF32(z_neg, IR::F32(IR::Value::F32(5.0f)), IR::F32(IR::Value::F32(4.0f))));
			break;
		case 1u: result = select_face(SelectF32(x_neg, z, nz), x, SelectF32(z_neg, nx, x)); break;
		case 2u: result = select_face(ny, SelectF32(y_neg, nz, z), ny); break;
		case 3u:
			result = IR::F32(
			    ir.Emit(IR::ValueOpcode::FPMul32, {select_face(x, y, z), IR::Value::F32(2.0f)}));
			break;
		default: EXIT("invalid cube result kind");
	}
	WriteOperand(DestinationOperand(inst), result);
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
