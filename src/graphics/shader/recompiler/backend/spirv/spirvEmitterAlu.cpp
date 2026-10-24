#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/shader.h"

#include <array>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

struct Pair {
	uint32_t low  = 0;
	uint32_t high = 0;
};

Pair ExtractPair(EmitterState& state, uint32_t value) {
	Pair result {state.builder.AllocateId(), state.builder.AllocateId()};
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result.low, value, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result.high, value, 1);
	return result;
}

uint32_t EmitMinMaxF64(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value) {
	const auto bits_type = TypeU32Vector(state, 2);
	const auto lhs_bits = Unary(state, spv::OpBitcast, bits_type, lhs);
	const auto rhs_bits = Unary(state, spv::OpBitcast, bits_type, rhs);
	const auto ordered = Binary(state, max_value ? spv::OpFOrdGreaterThan : spv::OpFOrdLessThan,
	                            TypeBool(state), lhs, rhs);
	auto result = Select(state, TypeF64(state), ordered, lhs, rhs);

	// Equal values have identical bits except signed zero: min chooses -0, max chooses +0.
	const auto equal = Binary(state, spv::OpFOrdEqual, TypeBool(state), lhs, rhs);
	const auto equal_bits = Binary(state, max_value ? spv::OpBitwiseAnd : spv::OpBitwiseOr,
	                               bits_type, lhs_bits, rhs_bits);
	result = Select(state, TypeF64(state), equal,
	                Unary(state, spv::OpBitcast, TypeF64(state), equal_bits), result);

	// Non-IEEE mode selects the other operand for NaN, including rhs when both are NaN.
	result = Select(state, TypeF64(state), Unary(state, spv::OpIsNan, TypeBool(state), rhs),
	                lhs, result);
	return Select(state, TypeF64(state), Unary(state, spv::OpIsNan, TypeBool(state), lhs),
	              rhs, result);
}

uint32_t EmitMulHigh(EmitterState& state, uint32_t lhs, uint32_t rhs, bool signed_value) {
	const auto operand_type = signed_value ? TypeI32(state) : TypeU32(state);
	const auto pair_type    = signed_value ? TypeI32Pair(state) : TypeU32Pair(state);
	uint32_t   lhs_operand  = lhs;
	uint32_t   rhs_operand  = rhs;
	if (signed_value) {
		lhs_operand = Unary(state, spv::OpBitcast, TypeI32(state), lhs);
		rhs_operand = Unary(state, spv::OpBitcast, TypeI32(state), rhs);
	}
	const auto extended = state.builder.AllocateId();
	state.builder.AddFunction(signed_value ? spv::OpSMulExtended : spv::OpUMulExtended, pair_type,
	                          extended, lhs_operand, rhs_operand);
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, operand_type, high, extended, 1);
	return signed_value ? Unary(state, spv::OpBitcast, TypeU32(state), high) : high;
}

uint32_t EmitFMinMax3(EmitterState& state, uint32_t a, uint32_t b, uint32_t c, bool max_value) {
	return EmitMinMaxF32Value(state, EmitMinMaxF32Value(state, a, b, max_value), c, max_value);
}

uint32_t EmitExt(EmitterState& state, uint32_t type, uint32_t opcode,
                 std::initializer_list<uint32_t> args) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, type, result, GlslStd450(state), opcode, args);
	return result;
}

uint32_t EmitF32ToU32(EmitterState& state, uint32_t src, bool signed_value) {
	const auto trunc         = EmitTruncF32Value(state, src);
	const auto converted_raw = state.builder.AllocateId();
	if (signed_value) {
		const auto converted_i = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertFToS, TypeI32(state), converted_i, trunc);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), converted_raw, converted_i);
	} else {
		state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), converted_raw, trunc);
	}
	const auto nan = EmitClassifyF32(state, src).nan;
	if (signed_value) {
		const auto below = Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), src,
		                          ConstantF32(state, 0xcf000000u));
		const auto above = Binary(state, spv::OpFOrdGreaterThanEqual, TypeBool(state), src,
		                          ConstantF32(state, 0x4f000000u));
		const auto high =
		    Select(state, TypeU32(state), above, ConstantU32(state, 0x7fffffffu), converted_raw);
		const auto low =
		    Select(state, TypeU32(state), below, ConstantU32(state, 0x80000000u), high);
		return Select(state, TypeU32(state), nan, ConstantU32(state, 0), low);
	}
	const auto below =
	    Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), src, ConstantF32(state, 0));
	const auto above = Binary(state, spv::OpFOrdGreaterThanEqual, TypeBool(state), src,
	                          ConstantF32(state, 0x4f800000u));
	const auto zero  = Binary(state, spv::OpLogicalOr, TypeBool(state), nan, below);
	const auto high =
	    Select(state, TypeU32(state), above, ConstantU32(state, 0xffffffffu), converted_raw);
	return Select(state, TypeU32(state), zero, ConstantU32(state, 0), high);
}

} // namespace

namespace {
uint32_t TypeF16Vector2(EmitterState& state) {
	return state.builder.Type(spv::OpTypeVector, state.builder.Type(spv::OpTypeFloat, 16u), 2u);
}

uint32_t BitcastToF16x2(EmitterState& state, uint32_t bits) {
	return Unary(state, spv::OpBitcast, TypeF16Vector2(state), bits);
}

uint32_t BitcastF16x2ToU32(EmitterState& state, uint32_t value) {
	return Unary(state, spv::OpBitcast, TypeU32(state), value);
}

// KYTY_FAST_MINMAX: one GLSL NMin/NMax instead of the exact guest min/max emulation.
uint32_t EmitFastMinMaxF32(EmitterState& state, uint32_t a, uint32_t b, bool max_value) {
	return EmitExt(state, TypeF32(state), max_value ? GLSLstd450NMax : GLSLstd450NMin, {a, b});
}
} // namespace

// Native packed f16 (KYTY_NATIVE_FP16): u32 -> f16vec2, vector op, f16vec2 -> u32. The FMA is a
// single-rounded GLSL Fma, matching v_pk_fma_f16 (the f32 path double-rounds).
uint32_t EmitFPAdd16x2(EmitterState& state, uint32_t a, uint32_t b) {
	return BitcastF16x2ToU32(
	    state, Binary(state, spv::OpFAdd, TypeF16Vector2(state), BitcastToF16x2(state, a),
	                  BitcastToF16x2(state, b)));
}

uint32_t EmitFPMul16x2(EmitterState& state, uint32_t a, uint32_t b) {
	return BitcastF16x2ToU32(
	    state, Binary(state, spv::OpFMul, TypeF16Vector2(state), BitcastToF16x2(state, a),
	                  BitcastToF16x2(state, b)));
}

uint32_t EmitFPFma16x2(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	return BitcastF16x2ToU32(
	    state, EmitExt(state, TypeF16Vector2(state), GLSLstd450Fma,
	                   {BitcastToF16x2(state, a), BitcastToF16x2(state, b),
	                    BitcastToF16x2(state, c)}));
}

uint32_t EmitFPClamp16x2(EmitterState& state, uint32_t a) {
	const auto half = state.builder.Type(spv::OpTypeFloat, 16u);
	const auto zero = state.builder.Constant(spv::OpConstant, half, 0x0000u);
	const auto one  = state.builder.Constant(spv::OpConstant, half, 0x3c00u);
	const auto vec  = TypeF16Vector2(state);
	const auto zero2 = state.builder.Constant(spv::OpConstantComposite, vec, zero, zero);
	const auto one2  = state.builder.Constant(spv::OpConstantComposite, vec, one, one);
	return BitcastF16x2ToU32(
	    state, EmitExt(state, vec, GLSLstd450FClamp, {BitcastToF16x2(state, a), zero2, one2}));
}

namespace {
uint32_t TypeF16(EmitterState& state) {
	return state.builder.Type(spv::OpTypeFloat, 16u);
}

// The f16 a U32 holds in its low half, or the lane of an F16LaneExtract's source: a component of
// the source register viewed as an f16vec2, which the host compiler reads as a register half.
uint32_t HalfOf(ValueEmitContext& ctx, IR::Value value) {
	auto&      state    = ctx.state;
	const auto resolved = value.Resolve();
	uint32_t   bits     = 0;
	uint32_t   lane     = 0;
	const auto* producer = resolved.IsImmediate() ? nullptr : resolved.TryInstruction();
	if (producer != nullptr && producer->GetOpcode() == IR::ValueOpcode::F16LaneExtract &&
	    producer->Arg(1).Resolve().IsImmediate()) {
		bits = ctx.Def(producer->Arg(0));
		lane = producer->Arg(1).Resolve().U32() & 1u;
	} else {
		bits = ctx.Def(value);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF16(state), result, BitcastToF16x2(state, bits), lane);
	return result;
}

uint32_t HalfToLow(EmitterState& state, uint32_t half) {
	const auto zero = state.builder.Constant(spv::OpConstant, TypeF16(state), 0x0000u);
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF16Vector2(state), pair, half, zero);
	return BitcastF16x2ToU32(state, pair);
}

uint32_t EmitHalfBinary(ValueEmitContext& ctx, const IR::Inst& inst, spv::Op opcode) {
	auto& state = ctx.state;
	return HalfToLow(state, Binary(state, opcode, TypeF16(state), HalfOf(ctx, inst.Arg(0)), HalfOf(ctx, inst.Arg(1))));
}
} // namespace

uint32_t EmitF16LaneExtract(EmitterState& state, uint32_t bits, uint32_t lane) {
	const auto offset = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), lane, ConstantU32(state, 4));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), result, bits, offset, ConstantU32(state, 16));
	return result;
}

uint32_t EmitF16LaneInsert(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto lane  = inst.Arg(2).Resolve();
	if (!lane.IsImmediate()) {
		const auto offset = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ctx.Def(inst.Arg(2)),
		                           ConstantU32(state, 4));
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitFieldInsert, TypeU32(state), result, ctx.Def(inst.Arg(0)),
		                          ctx.Def(inst.Arg(1)), offset, ConstantU32(state, 16));
		return result;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeInsert, TypeF16Vector2(state), result, HalfOf(ctx, inst.Arg(1)),
	                          BitcastToF16x2(state, ctx.Def(inst.Arg(0))), lane.U32() & 1u);
	return BitcastF16x2ToU32(state, result);
}

uint32_t EmitFPAdd16Low(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitHalfBinary(ctx, inst, spv::OpFAdd);
}

uint32_t EmitFPSub16Low(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitHalfBinary(ctx, inst, spv::OpFSub);
}

uint32_t EmitFPMul16Low(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitHalfBinary(ctx, inst, spv::OpFMul);
}

uint32_t EmitFPFma16Low(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	return HalfToLow(state, EmitExt(state, TypeF16(state), GLSLstd450Fma,
	                                {HalfOf(ctx, inst.Arg(0)), HalfOf(ctx, inst.Arg(1)), HalfOf(ctx, inst.Arg(2))}));
}

uint32_t EmitFPUnary16Low(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto op    = inst.Arg(1).Resolve();
	EXIT_IF(!op.IsImmediate());
	const auto x = HalfOf(ctx, inst.Arg(0));
	uint32_t   result;
	const auto glsl = [&](GLSLstd450 function) { return EmitExt(state, TypeF16(state), function, {x}); };
	switch (static_cast<IR::F16UnaryOp>(op.U32())) {
		case IR::F16UnaryOp::Recip:
			result = Binary(state, spv::OpFDiv, TypeF16(state),
			                state.builder.Constant(spv::OpConstant, TypeF16(state), 0x3c00u), x);
			break;
		case IR::F16UnaryOp::Sqrt: result = glsl(GLSLstd450Sqrt); break;
		case IR::F16UnaryOp::RecipSqrt: result = glsl(GLSLstd450InverseSqrt); break;
		case IR::F16UnaryOp::Log2: result = glsl(GLSLstd450Log2); break;
		case IR::F16UnaryOp::Exp2: result = glsl(GLSLstd450Exp2); break;
		case IR::F16UnaryOp::Floor: result = glsl(GLSLstd450Floor); break;
		case IR::F16UnaryOp::Ceil: result = glsl(GLSLstd450Ceil); break;
		case IR::F16UnaryOp::Trunc: result = glsl(GLSLstd450Trunc); break;
		case IR::F16UnaryOp::RoundEven: result = glsl(GLSLstd450RoundEven); break;
		case IR::F16UnaryOp::Fract: result = glsl(GLSLstd450Fract); break;
		default: EXIT("unknown f16 unary op %u\n", op.U32());
	}
	return HalfToLow(state, result);
}

uint32_t EmitFPMinMax16Low(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto max   = inst.Arg(2).Resolve();
	EXIT_IF(!max.IsImmediate());
	return HalfToLow(state, EmitExt(state, TypeF16(state), max.U32() != 0u ? GLSLstd450NMax : GLSLstd450NMin,
	                                {HalfOf(ctx, inst.Arg(0)), HalfOf(ctx, inst.Arg(1))}));
}

uint32_t EmitFPMinMax16x2(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto max   = inst.Arg(2).Resolve();
	EXIT_IF(!max.IsImmediate());
	return BitcastF16x2ToU32(
	    state, EmitExt(state, TypeF16Vector2(state), max.U32() != 0u ? GLSLstd450NMax : GLSLstd450NMin,
	                   {BitcastToF16x2(state, ctx.Def(inst.Arg(0))), BitcastToF16x2(state, ctx.Def(inst.Arg(1)))}));
}

uint32_t EmitF16x2Swizzle(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto bits  = ctx.Def(inst.Arg(0));
	const auto lo    = inst.Arg(1).Resolve();
	const auto hi    = inst.Arg(2).Resolve();
	if (!lo.IsImmediate() || !hi.IsImmediate()) {
		const auto low = Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits,
		                        Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ctx.Def(inst.Arg(1)), ConstantU32(state, 4)));
		const auto high = Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits,
		                         Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ctx.Def(inst.Arg(2)), ConstantU32(state, 4)));
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitFieldInsert, TypeU32(state), result,
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 0xffffu)), high,
		                          ConstantU32(state, 16), ConstantU32(state, 16));
		return result;
	}
	const auto vector = BitcastToF16x2(state, bits);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpVectorShuffle, TypeF16Vector2(state), result, vector, vector, lo.U32() & 1u,
	                          hi.U32() & 1u);
	return BitcastF16x2ToU32(state, result);
}

uint32_t EmitFPFma32(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	const auto result = EmitGlsl<GLSLstd450Fma, IR::Type::F32>(state, a, b, c);
	state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	return result;
}

uint32_t EmitFPMad32(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	if (ShaderFusedMad() && (state.program.stage == ShaderType::Pixel || state.program.stage == ShaderType::Compute)) {
		return EmitGlsl<GLSLstd450Fma, IR::Type::F32>(state, a, b, c);
	}
	// Explicit denormal checks typically drop 3DMiniGolf menu FPS from 28-29 to 21-24
	// on NVIDIA RTX 5080 Laptop GPU, so leave them disabled for this experiment.
	// a = EmitFlushF32DenormToSignedZero(state, a);
	// b = EmitFlushF32DenormToSignedZero(state, b);
	// c = EmitFlushF32DenormToSignedZero(state, c);
	const auto product = EmitFPMul32(state, a, b);
	state.builder.AddAnnotation(spv::OpDecorate, product, spv::DecorationNoContraction);
	// const auto sum = EmitFPAdd32(state, EmitFlushF32DenormToSignedZero(state, product), c);
	const auto sum = EmitFPAdd32(state, product, c);
	state.builder.AddAnnotation(spv::OpDecorate, sum, spv::DecorationNoContraction);
	// return EmitFlushF32DenormToSignedZero(state, sum);
	return sum;
}

uint32_t EmitFPMedTri32(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	if (ShaderFastMinMax()) {
		// max(min(a, b), min(max(a, b), c)); the driver can fold this into v_med3_f32.
		const auto lo = EmitFastMinMaxF32(state, a, b, false);
		const auto hi = EmitFastMinMaxF32(state, a, b, true);
		return EmitFastMinMaxF32(state, lo, EmitFastMinMaxF32(state, hi, c, false), true);
	}
	const auto min_ab   = EmitMinMaxF32Value(state, a, b, false);
	const auto min3     = EmitMinMaxF32Value(state, min_ab, c, false);
	const auto max_ab   = EmitMinMaxF32Value(state, a, b, true);
	const auto high_min = EmitMinMaxF32Value(state, max_ab, c, false);
	const auto median   = EmitMinMaxF32Value(state, min_ab, high_min, true);
	const auto nan_ab   = Binary(state, spv::OpLogicalOr, TypeBool(state),
	                             EmitClassifyF32(state, a).nan, EmitClassifyF32(state, b).nan);
	const auto any_nan =
	    Binary(state, spv::OpLogicalOr, TypeBool(state), nan_ab, EmitClassifyF32(state, c).nan);
	return Select(state, TypeF32(state), any_nan, min3, median);
}

uint32_t EmitFindUMsb64(EmitterState& state, uint32_t value) {
	// GLSL FindUMsb is restricted to 32-bit components.
	const auto unpacked = Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), value);
	const auto pair =
	    ExtractPair(state, EmitExt(state, TypeU32Vector(state, 2), GLSLstd450FindUMsb, {unpacked}));
	const auto high_found =
	    Binary(state, spv::OpINotEqual, TypeBool(state), pair.high, ConstantU32(state, UINT32_MAX));
	return Select(state, TypeU32(state), high_found,
	              Binary(state, spv::OpIAdd, TypeU32(state), pair.high, ConstantU32(state, 32)),
	              pair.low);
}

uint32_t EmitConvertU16U32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpBitwiseAnd, IR::Type::U16>(state, arg0, ConstantU32(state, 0xffffu));
}

uint32_t EmitConvertU8U32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpBitwiseAnd, IR::Type::U8>(state, arg0, ConstantU32(state, 0xffu));
}

uint32_t EmitConvertF16F32(EmitterState& state, uint32_t arg0) {
	if (ShaderF16Converts()) {
		state.builder.RequireCapability(spv::CapabilityFloat16);
		return HalfToLow(state, Unary(state, spv::OpFConvert, TypeF16(state), arg0));
	}
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, arg0,
	                          ConstantF32(state, 0));
	return EmitPackHalf2x16(state, pair);
}

uint32_t EmitConvertS32F32(EmitterState& state, uint32_t arg0) {
	return EmitF32ToU32(state, arg0, true);
}

uint32_t EmitConvertU32F32(EmitterState& state, uint32_t arg0) {
	return EmitF32ToU32(state, arg0, false);
}

uint32_t EmitConvertF32F64(EmitterState& state, uint32_t arg0) {
	const auto converted = Unary(state, spv::OpFConvert, TypeF32(state), arg0);
	const auto source    = EmitNative<spv::OpCompositeExtract, IR::Type::U32>(
	    state, Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), arg0), 1u);
	const auto exponent = EmitAndConstant(state, source, 0x7ff00000u);
	const auto overflow =
	    Binary(state, spv::OpLogicalAnd, TypeBool(state),
	           EmitCompareU32Constant(state, spv::OpUGreaterThan, exponent, 0x47e00000u),
	           EmitCompareU32Constant(state, spv::OpULessThan, exponent, 0x7ff00000u));
	const auto clamped = EmitOrU32(state, EmitAndConstant(state, source, 0x80000000u),
	                               ConstantU32(state, 0x7f7fffffu));
	return EmitFlushF32DenormToSignedZero(
	    state, Select(state, TypeF32(state), overflow,
	                  Unary(state, spv::OpBitcast, TypeF32(state), clamped), converted));
}

uint32_t EmitConvertF64F32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpFConvert, IR::Type::F64>(state,
	                                                  EmitFlushF32DenormToSignedZero(state, arg0));
}

uint32_t EmitCompositeConstructU64(ValueEmitContext& ctx, uint32_t arg0, IR::Value arg1) {
	arg1 = arg1.Resolve();
	return arg1.IsImmediate() && arg1.U32() == 0u
	           ? Unary(ctx.state, spv::OpUConvert, TypeU64(ctx.state), arg0)
	           : PackU64(ctx.state, arg0, ctx.Def(arg1));
}

uint32_t EmitCompositeExtractU64(EmitterState& state, uint32_t arg0, IR::Value arg1) {
	const auto value = arg1.U32() == 0u ? arg0
	                                    : Binary(state, spv::OpShiftRightLogical, TypeU64(state),
	                                             arg0, ConstantU32(state, 32));
	return Unary(state, spv::OpUConvert, TypeU32(state), value);
}

uint32_t EmitCompositeExtractU32x2(EmitterState& state, uint32_t arg0, IR::Value arg1) {
	return EmitNative<spv::OpCompositeExtract, IR::Type::U32>(state, arg0, arg1.U32());
}

uint32_t EmitPackFloat2x16Rtz(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	if (ShaderNativePkrtz()) {
		// One lane per PackHalf2x16, merged with integer ops (see the RDNA4 driver note in
		// spirvEmitterImage.cpp). The AMD compiler emits v_cvt_pk_rtz_f16_f32 for PackHalf2x16 also in
		// shaders that declare RoundingModeRTE for f16 (seen in their ISA).
		const auto pack_lane = [&](uint32_t lane) {
			const auto pair = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, lane,
			                          ConstantF32(state, 0));
			return EmitPackHalf2x16(state, pair);
		};
		return Binary(state, spv::OpBitwiseOr, TypeU32(state), pack_lane(arg0),
		              Binary(state, spv::OpShiftLeftLogical, TypeU32(state), pack_lane(arg1),
		                     ConstantU32(state, 16)));
	}
	const auto low  = EmitF32ToF16RtzBits(state, arg0);
	const auto high = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
	                         EmitF32ToF16RtzBits(state, arg1), ConstantU32(state, 16));
	return Binary(state, spv::OpBitwiseOr, TypeU32(state), low, high);
}

uint32_t EmitFPSaturate32(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450FClamp,
	               {arg0, ConstantF32(state, 0), ConstantF32(state, 0x3f800000u)});
}

uint32_t EmitSMulHi(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMulHigh(state, arg0, arg1, true);
}

uint32_t EmitUMulHi(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMulHigh(state, arg0, arg1, false);
}

uint32_t EmitBitCount64(EmitterState& state, uint32_t arg0) {
	// Vulkan bit-count operands remain 32-bit, including when counting a U64.
	const auto unpacked = Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), arg0);
	const auto pair =
	    ExtractPair(state, Unary(state, spv::OpBitCount, TypeU32Vector(state, 2), unpacked));
	return Binary(state, spv::OpIAdd, TypeU32(state), pair.low, pair.high);
}

uint32_t EmitSMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitSMin32(state, arg0, EmitSMin32(state, arg1, arg2));
}

uint32_t EmitSMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitSMax32(state, arg0, EmitSMax32(state, arg1, arg2));
}

uint32_t EmitUMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitUMin32(state, arg0, EmitUMin32(state, arg1, arg2));
}

uint32_t EmitUMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitUMax32(state, arg0, EmitUMax32(state, arg1, arg2));
}

uint32_t EmitSMedTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	const auto high_min = EmitSMin32(state, EmitSMax32(state, arg0, arg1), arg2);
	return EmitSMax32(state, EmitSMin32(state, arg0, arg1), high_min);
}

uint32_t EmitUMedTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	const auto high_min = EmitUMin32(state, EmitUMax32(state, arg0, arg1), arg2);
	return EmitUMax32(state, EmitUMin32(state, arg0, arg1), high_min);
}

uint32_t EmitFPMin32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	if (ShaderFastMinMax()) {
		return EmitFastMinMaxF32(state, arg0, arg1, false);
	}
	return EmitMinMaxF32Value(state, arg0, arg1, false);
}

uint32_t EmitFPMax32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	if (ShaderFastMinMax()) {
		return EmitFastMinMaxF32(state, arg0, arg1, true);
	}
	return EmitMinMaxF32Value(state, arg0, arg1, true);
}

uint32_t EmitFPMin64(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF64(state, arg0, arg1, false);
}

uint32_t EmitFPMax64(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF64(state, arg0, arg1, true);
}

uint32_t EmitFPMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	if (ShaderFastMinMax()) {
		return EmitFastMinMaxF32(state, EmitFastMinMaxF32(state, arg0, arg1, false), arg2, false);
	}
	return EmitFMinMax3(state, arg0, arg1, arg2, false);
}

uint32_t EmitFPMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	if (ShaderFastMinMax()) {
		return EmitFastMinMaxF32(state, EmitFastMinMaxF32(state, arg0, arg1, true), arg2, true);
	}
	return EmitFMinMax3(state, arg0, arg1, arg2, true);
}

uint32_t EmitFPRecip32(EmitterState& state, uint32_t arg0) {
	const auto source = EmitFlushF32DenormToSignedZero(state, arg0);
	return Binary(state, spv::OpFDiv, TypeF32(state), ConstantF32(state, 0x3f800000u), source);
}

uint32_t EmitFPRecip64(EmitterState& state, uint32_t arg0) {
	const auto one = state.builder.Constant(spv::OpConstant, TypeF64(state), 0u, 0x3ff00000u);
	return EmitNative<spv::OpFDiv, IR::Type::F64>(state, one, arg0);
}

uint32_t EmitFPRecipIFlag32(EmitterState& state, uint32_t arg0) {
	// Integer-to-float inputs used by IFLAG cannot be denormal.
	return Binary(state, spv::OpFDiv, TypeF32(state), ConstantF32(state, 0x3f800000u), arg0);
}

uint32_t EmitFPRecipSqrt32(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450InverseSqrt,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPSqrt(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Sqrt,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPExp2(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Exp2,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPLog2(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Log2,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPLdexp(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	const auto exponent = Unary(state, spv::OpBitcast, TypeI32(state), arg1);
	return EmitExt(state, TypeF32(state), GLSLstd450Ldexp, {arg0, exponent});
}

uint32_t EmitFPSin(EmitterState& state, uint32_t arg0) {
	auto source = EmitTrigCycleF32(state, arg0, true);
	source = Binary(state, spv::OpFMul, TypeF32(state), source, ConstantF32(state, 0x40c90fdbu));
	return EmitExt(state, TypeF32(state), GLSLstd450Sin, {source});
}

uint32_t EmitFPCos(EmitterState& state, uint32_t arg0) {
	auto source = EmitTrigCycleF32(state, arg0, false);
	source = Binary(state, spv::OpFMul, TypeF32(state), source, ConstantF32(state, 0x40c90fdbu));
	return EmitExt(state, TypeF32(state), GLSLstd450Cos, {source});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
