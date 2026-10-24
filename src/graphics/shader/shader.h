#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/experiments.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

// Codegen option: native f16 (Float16 capability) for V_PK_{ADD,MUL,FMA,FMAC}_F16 with plain
// operands, when the device supports shaderFloat16 and experiment NativeFp16 (KYTY_NATIVE_FP16=1) is
// on. The option is part of every stage key, so switching it at runtime compiles new variants. The
// mode bits describe which float-controls f16 execution modes the device allows the emitter to declare.
inline constexpr uint32_t kNativeFp16DenormPreserve = 1u << 0u;
inline constexpr uint32_t kNativeFp16SignedZeroInfNan = 1u << 1u;
inline constexpr uint32_t kNativeFp16RoundTiesEven = 1u << 2u;
inline std::atomic<bool>     g_shader_native_fp16 {false};
inline std::atomic<uint32_t> g_shader_native_fp16_modes {0};
[[nodiscard]] inline bool ShaderNativeFp16() {
	return g_shader_native_fp16.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::NativeFp16);
}
// Codegen option: experiment ExecSelects (IR::EliminateExecSelects) and its rule set
// (KYTY_DBG_EXECSEL_MODE, default 16). Part of every stage key: 0 when off.
[[nodiscard]] inline uint32_t ShaderExecSelects() {
	if (!Common::Experiments::Enabled(Common::Experiments::ExecSelects)) return 0;
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_DBG_EXECSEL_MODE");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0)) : 16u;
	}();
	return 0x100u | mode;
}
// Codegen option: experiment LdsLaneSlots (ShaderInfo::function_lds_lane_slots). Part of the
// vertex and pixel stage keys.
[[nodiscard]] inline bool ShaderLdsLaneSlots() {
	return Common::Experiments::Enabled(Common::Experiments::LdsLaneSlots);
}
// Codegen option: experiment NativeMinMax. V_MIN/V_MAX/V_MIN3/V_MAX3/V_MED3_F32 (and the f32
// math of the f16 forms) use GLSL NMin/NMax, which an AMD host compiles to the same v_min/v_max
// as the guest, instead of exact NaN and signed-zero emulation (about ten instructions each). Folded
// into the NativeFp16 key word so keys are unchanged while it is off.
[[nodiscard]] inline bool ShaderNativeMinMax() {
	return Common::Experiments::Enabled(Common::Experiments::NativeMinMax);
}
// Codegen option: experiment FusedMad. V_MAD_F32/V_MAC_F32 (a separately rounded multiply and
// add on the guest) of pixel and compute shaders as one fused multiply-add instead of a
// NoContraction multiply and add. The guest compiler picks V_MAD/V_MAC for contractable multiply-adds
// (the fast form with FP32 denormals flushed); vertex and mesh shaders keep the separate rounding so
// positions computed by different shaders stay bit-identical (depth-equal passes).
[[nodiscard]] inline bool ShaderFusedMad() {
	return Common::Experiments::Enabled(Common::Experiments::FusedMad);
}
// Codegen option: experiment RobustLoads, on devices with robustBufferAccess2 (out-of-range
// storage-buffer reads return zero, as the guest's do). Raw storage-buffer loads then skip their
// EXEC branch and per-dword bounds branch, so the host compiler can batch them.
inline std::atomic<bool> g_shader_robust_buffer_access2 {false};
[[nodiscard]] inline bool ShaderRobustLoads() {
	return g_shader_robust_buffer_access2.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::RobustLoads);
}
// KYTY_ROBUST_LOADS_KEEP_EXEC=1: robust loads still branch on EXEC (skipping waves with no active
// lane), dropping only the bounds branches.
[[nodiscard]] inline bool ShaderRobustLoadsKeepExec() {
	static const bool keep = [] {
		const char* value = std::getenv("KYTY_ROBUST_LOADS_KEEP_EXEC");
		return value != nullptr && value[0] == '1';
	}();
	return keep;
}
// Robust loads also cover scalar constant reads (S_BUFFER_LOAD): no bounds branch per dword
// (code-gen key bit 32; GPU time -6 to -7% per frame in an A/B/A).
[[nodiscard]] inline bool ShaderRobustConstReads() {
	return ShaderRobustLoads();
}
// Robust loads also drop the bounds branch of storage-buffer stores (out of range writes are
// discarded on the device); the EXEC branch stays (key bit 64; GPU time -3 to -4% per frame).
[[nodiscard]] inline bool ShaderRobustStores() {
	return ShaderRobustLoads();
}
// Codegen option: V_CVT_PKRTZ_F16_F32 through GLSL PackHalf2x16, which AMD drivers compile to the
// same v_cvt_pk_rtz_f16_f32 (round toward zero), instead of about forty integer instructions per
// half. Set for AMD devices (key bit 128). Paused A/B/A: GPU time per frame -5%; the crowd vertex
// shaders behind PS f5a3c1b7 halve (46k -> 22k instructions).
inline std::atomic<bool> g_shader_native_pkrtz {false};
[[nodiscard]] inline bool ShaderNativePkrtz() {
	return g_shader_native_pkrtz.load(std::memory_order_relaxed);
}
// Codegen option: native scalar f16 ops (ShaderNativeFp16) also for SDWA word-select sources
// (key bit 256). GPU time unchanged in a paused A/B/A; rounds like the guest (the f32 fallback
// packs its result with PackHalf2x16, which AMD compiles to round-toward-zero).
[[nodiscard]] inline bool ShaderNativeF16Sdwa() {
	return ShaderNativeFp16();
}
// Codegen option: robust (branchless) storage-buffer loads, stores and constant reads also in
// non-pixel shaders that spill guest VGPRs to LDS or scratch (key bit 512; d7cfa519 -3%).
[[nodiscard]] inline bool ShaderRobustSpillLoads() {
	return ShaderRobustLoads();
}
// Codegen option: formatted buffer loads under robust loads read unconditionally and select the
// out-of-bounds value instead of branching (key bit 1024). The crowd vertex shaders had ~400 such
// branches around 16-bit vertex loads: their draws -9.6%, GPU time per frame -4% with these two
// options in a paused crowd scene (A/B/A).
[[nodiscard]] inline bool ShaderBranchlessFormattedLoads() {
	return ShaderRobustLoads();
}
// Codegen option: experiment ExecBranches (key bit 2048). A guest VGPR write under a partial EXEC
// selects the new value per lane; runs of lane-wise instructions under one EXEC value run in a
// divergent branch instead, their selects becoming phis where the run's results are used after it
// (the host compiler masks the writes, as the guest's hardware does).
[[nodiscard]] inline bool ShaderExecBranches() {
	return Common::Experiments::Enabled(Common::Experiments::ExecBranches);
}
// Codegen option: experiment ExecBranchMemory (key bit 4096): ExecBranches runs also take the
// memory operations predicated by their EXEC (in order; inside the branch they need no exec check).
[[nodiscard]] inline bool ShaderExecBranchMemory() {
	return ShaderExecBranches() && Common::Experiments::Enabled(Common::Experiments::ExecBranchMemory);
}
// Codegen option: experiment ExecBranchContinue (key bit 16384): ExecBranches runs also in loop
// continue targets (the bodies of NHL 27's sampling loops, which selected every write per lane).
[[nodiscard]] inline bool ShaderExecBranchContinue() {
	return ShaderExecBranches() && Common::Experiments::Enabled(Common::Experiments::ExecBranchContinue);
}
// Codegen option: experiment F16LaneViews (key bit 32768): scalar f16 ops on SDWA/op_sel lanes as
// f16vec2 components (F16LaneExtract / FP*16Low / F16LaneInsert) instead of integer masking.
[[nodiscard]] inline bool ShaderF16LaneViews() {
	return ShaderNativeFp16() && Common::Experiments::Enabled(Common::Experiments::F16LaneViews);
}
// Codegen option: experiment BdaLoadGroups (key bit 8192): consecutive DWORD loads of one guest
// load through an address share one page lookup when they are in one page.
[[nodiscard]] inline bool ShaderBdaLoadGroups() {
	return Common::Experiments::Enabled(Common::Experiments::BdaLoadGroups);
}
// The device has VK_AMD_shader_early_and_late_fragment_tests (vulkanWindow sets it).
inline std::atomic<bool> g_shader_early_late_tests {false};
// Codegen option: experiment EarlyLateFragmentTests (key bit 65536). A pixel shader that reads memory
// by address records missing pages in the fault buffer, and with those stores (and a kill) the host
// compiler tests depth only after the shader: occluded pixels ran it too. When the guest shader itself
// writes no memory and exports no depth or sample mask, it declares EarlyAndLateFragmentTestsAMD:
// depth is tested before it as well (a fragment that fails the early test records no fault).
[[nodiscard]] inline bool ShaderEarlyLateFragmentTests() {
	return g_shader_early_late_tests.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::EarlyLateFragmentTests);
}
// Codegen option: experiment UndefVgprs (key bit 131072). A VGPR the hardware leaves uninitialized
// read before any write on some path was 0 there: a value written in one branch and read after it
// merged with that 0 (a phi, often a move per path and a register longer live). It is undefined
// instead, as on the hardware.
[[nodiscard]] inline bool ShaderUndefVgprs() {
	return Common::Experiments::Enabled(Common::Experiments::UndefVgprs);
}
// The device can flush 32-bit denormals while 16-bit ones are preserved (vulkanWindow sets it).
inline std::atomic<bool> g_shader_ftz32 {false};
// Codegen option: experiment Ftz32 (key bit 262144). The guest's MODE (0xc0) flushes FP32
// denormals; the shaders declare DenormFlushToZero for 32-bit floats, so the inputs of RCP, RSQ,
// SQRT, EXP and LOG need no flush of their own (an and, a compare and a select each).
[[nodiscard]] inline bool ShaderFtz32() {
	return g_shader_ftz32.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::Ftz32);
}
// Codegen option: experiment MovRelBounds (key bit 524288), see MovRelExtents in Translate.cpp.
[[nodiscard]] inline bool ShaderMovRelBounds() {
	return Common::Experiments::Enabled(Common::Experiments::MovRelBounds);
}
// Codegen option: experiment F16Converts (key bit 1048576). An f16 the f32 emulation of a scalar f16
// op produces was packed with PackHalf2x16, which the AMD compiler emits as v_cvt_pk_rtz_f16_f32
// (round toward zero, not the guest's to nearest even); as OpFConvert it is v_cvt_f16_f32, and the
// host compiler can fold an op between two conversions into its f16 form.
[[nodiscard]] inline bool ShaderF16Converts() {
	return ShaderNativeFp16() && Common::Experiments::Enabled(Common::Experiments::F16Converts);
}
// Codegen option: experiment F16ImagePack (key bit 2097152), see ShaderF16Converts.
[[nodiscard]] inline bool ShaderF16ImagePack() {
	return ShaderF16Converts() && Common::Experiments::Enabled(Common::Experiments::F16ImagePack);
}
// Codegen option: experiment ExecRunTransparent (key bit 4194304). ExecBranches runs ended at any
// instruction that is not lane-wise: in NHL 27's heaviest shaders mostly resource handles (Get*Resource),
// LaneId and Reference (no code), which do not depend on EXEC and may sit in or before the branch.
[[nodiscard]] inline bool ShaderExecRunTransparent() {
	return ShaderExecBranches() && Common::Experiments::Enabled(Common::Experiments::ExecRunTransparent);
}
// Codegen option: experiment ExecRunConstReads (key bit 268435456). With ExecRunTransparent, runs
// still ended at scalar constant reads and image address operands (in NHL 27's heaviest compute
// shaders most of the ends: 52 and 61 constant reads, 19 and 33 image addresses), which do not
// depend on EXEC either.
[[nodiscard]] inline bool ShaderExecRunConstReads() {
	return ShaderExecRunTransparent() && Common::Experiments::Enabled(Common::Experiments::ExecRunConstReads);
}
// Codegen option: experiment AlignedBufferOffsets (key bit 8388608). A buffer binding starts at the
// guest base aligned down to the device's storage offset alignment, and the shader adds the rest
// (the binding prefix, from the shader data). Scalar buffer reads align the base and the offset
// independently, so with an alignment of at most 4 bytes (AMD) their prefix is always zero dwords;
// adding it per read kept the compiler from folding and merging the constant loads.
// vulkanWindow sets g_shader_storage_align4 at device creation.
inline std::atomic<bool> g_shader_storage_align4 {false};
[[nodiscard]] inline bool ShaderAlignedBufferOffsets() {
	return g_shader_storage_align4.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::AlignedBufferOffsets);
}
// Codegen option: experiment NativeFract (key bit 16777216), AMD devices only (see
// g_shader_native_pkrtz): the clamp after V_FRACT_F32's Fract is left out.
[[nodiscard]] inline bool ShaderNativeFract() {
	return g_shader_native_pkrtz.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::NativeFract);
}
// Codegen option: experiment NativeF16Ops (key bit 33554432), AMD devices with native f16 and
// native min/max: the f16 unary ops and min/max run as the host's f16 instructions (the guest's
// own: v_rcp_f16, v_pk_max_f16, ...) instead of through f32 with NaN fix-ups.
[[nodiscard]] inline bool ShaderNativeF16Ops() {
	return ShaderNativeFp16() && ShaderNativeMinMax() && g_shader_native_pkrtz.load(std::memory_order_relaxed) &&
	       Common::Experiments::Enabled(Common::Experiments::NativeF16Ops);
}
// Codegen option: experiment MovRelArrays (key bit 67108864), with MovRelBounds.
[[nodiscard]] inline bool ShaderMovRelArrays() {
	return ShaderMovRelBounds() && Common::Experiments::Enabled(Common::Experiments::MovRelArrays);
}
// Codegen option: experiment BdaTwoLookups (key bit 134217728), with BdaLoadGroups.
[[nodiscard]] inline bool ShaderBdaTwoLookups() {
	return ShaderBdaLoadGroups() && Common::Experiments::Enabled(Common::Experiments::BdaTwoLookups);
}
[[nodiscard]] inline uint32_t ShaderFloatCodegenKey() {
	return static_cast<uint32_t>(ShaderNativeFp16()) | (ShaderNativeMinMax() ? 2u : 0u) |
	       (ShaderFusedMad() ? 4u : 0u) |
	       (ShaderRobustLoads() ? (ShaderRobustLoadsKeepExec() ? 24u : 8u) : 0u) |
	       (ShaderRobustConstReads() ? 32u : 0u) | (ShaderRobustStores() ? 64u : 0u) |
	       (ShaderNativePkrtz() ? 128u : 0u) | (ShaderNativeF16Sdwa() ? 256u : 0u) |
	       (ShaderRobustSpillLoads() ? 512u : 0u) | (ShaderBranchlessFormattedLoads() ? 1024u : 0u) |
	       (ShaderExecBranches() ? 2048u : 0u) | (ShaderExecBranchMemory() ? 4096u : 0u) |
	       (ShaderBdaLoadGroups() ? 8192u : 0u) | (ShaderExecBranchContinue() ? 16384u : 0u) |
	       (ShaderF16LaneViews() ? 32768u : 0u) | (ShaderEarlyLateFragmentTests() ? 65536u : 0u) |
	       (ShaderUndefVgprs() ? 131072u : 0u) | (ShaderFtz32() ? 262144u : 0u) |
	       (ShaderMovRelBounds() ? 524288u : 0u) | (ShaderF16Converts() ? 1048576u : 0u) |
	       (ShaderF16ImagePack() ? 2097152u : 0u) | (ShaderExecRunTransparent() ? 4194304u : 0u) |
	       (ShaderAlignedBufferOffsets() ? 8388608u : 0u) | (ShaderNativeFract() ? 16777216u : 0u) |
	       (ShaderNativeF16Ops() ? 33554432u : 0u) | (ShaderMovRelArrays() ? 67108864u : 0u) |
	       (ShaderBdaTwoLookups() ? 134217728u : 0u) | (ShaderExecRunConstReads() ? 268435456u : 0u);
}
[[nodiscard]] inline uint32_t ShaderNativeFp16Modes() {
	return g_shader_native_fp16_modes.load(std::memory_order_relaxed);
}

// Codegen experiment (KYTY_FAST_MINMAX=1): float min/max/min3/max3/med3 use GLSL NMin/NMax
// directly instead of emulating the guest's signed-zero ordering and NaN operand selection.
// NMin/NMax still return the non-NaN operand; only -0/+0 ordering and NaN payloads may differ.
// Read once per process; off by default.
[[nodiscard]] inline bool ShaderFastMinMax() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FAST_MINMAX");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Codegen experiment (KYTY_SKIP_BOUNDS_CHECKS=1): scalar buffer reads (s_buffer_load) skip the
// explicit per-dword bounds check and rely on robustBufferAccess2, which returns zero for dwords
// outside the bound range just like the check. Only active when the device runs with
// robustBufferAccess2 and a storage-buffer robustness alignment of at most 4 bytes; vulkanWindow
// sets g_shader_robust_buffer2 at device creation. Off by default.
inline std::atomic<bool> g_shader_robust_buffer2 {false};
[[nodiscard]] inline bool ShaderSkipBoundsChecks() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_SKIP_BOUNDS_CHECKS");
		return value != nullptr && value[0] == '1';
	}();
	return requested && g_shader_robust_buffer2.load(std::memory_order_relaxed);
}

namespace ShaderError {

[[nodiscard]] inline bool Fail(std::string* error, std::string_view message) {
	if (error != nullptr) {
		error->assign(message.data(), message.size());
	}
	return false;
}

} // namespace ShaderError

namespace HW {
struct VertexShaderInfo;
struct PixelShaderInfo;
struct ComputeShaderInfo;
struct ShaderRegisters;
} // namespace HW

enum class ShaderType {
	Unknown,
	Vertex,
	Pixel,
	Fetch,
	Compute,
	Mesh,
	Local,
	TessellationControl,
	TessellationEvaluation
};

namespace ShaderRecompiler::IR {
struct CompiledShaderInfo;
} // namespace ShaderRecompiler::IR

struct ShaderStageRuntime {
	const ShaderRecompiler::IR::CompiledShaderInfo* program = nullptr;
	const ShaderRecompiler::IR::ResourceSnapshot*   resources = nullptr;

	[[nodiscard]] explicit operator bool() const {
		return program != nullptr && resources != nullptr;
	}
};

constexpr uint32_t DstSel(uint32_t x, uint32_t y = 0, uint32_t z = 0, uint32_t w = 0) {
	return x | (y << 3u) | (z << 6u) | (w << 9u);
}

inline uint8_t GetDstSel(uint32_t swizzle, uint32_t channel) {
	return (swizzle >> (channel * 3u)) & 0x7u;
}

struct ShaderClipSpaceTransform {
	float scale[2]       = {};
	float offset[2]      = {};
	float half_extent[2] = {};
	bool  enabled        = false;
};

struct ShaderWorkgroupInputInfo {
	uint32_t threads_num[3]      = {0, 0, 0};
	uint32_t lds_size_dwords     = 0;
	uint32_t scratch_size_dwords = 0;
	uint32_t host_subgroup_size  = 64;
	uint32_t wave_size           = 64;
};

struct ShaderMeshInputInfo: ShaderWorkgroupInputInfo {
	uint32_t input_primitive      = 0;
	uint32_t primitives_per_group = 0;
	uint32_t vertices_per_group   = 0;
	uint32_t max_vertices         = 0;
	uint32_t max_primitives       = 0;
	uint32_t provoking_vertex     = 0;
	bool     fast_launch          = false;

	[[nodiscard]] constexpr uint32_t InputPrimitiveSize() const {
		switch (static_cast<Prospero::PrimitiveType>(input_primitive)) {
			case Prospero::PrimitiveType::kPointList: return 1u;
			case Prospero::PrimitiveType::kLineList: return 2u;
			default: return 3u;
		}
	}
	[[nodiscard]] constexpr uint32_t InputPrimitiveStep() const {
		switch (static_cast<Prospero::PrimitiveType>(input_primitive)) {
			case Prospero::PrimitiveType::kTriFan:
			case Prospero::PrimitiveType::kTriStrip: return 1u;
			default: return InputPrimitiveSize();
		}
	}
	[[nodiscard]] constexpr uint32_t InputPrimitiveCount(uint32_t vertices) const {
		const auto size = InputPrimitiveSize();
		return vertices < size ? 0u : (vertices - size) / InputPrimitiveStep() + 1u;
	}
	[[nodiscard]] constexpr uint32_t InputVertexCount(uint32_t primitives) const {
		return primitives == 0u ? 0u : (primitives - 1u) * InputPrimitiveStep() + InputPrimitiveSize();
	}
};

struct ShaderTessellationInputInfo {
	uint32_t input_control_points  = 0;
	uint32_t output_control_points = 0;
	uint32_t ls_stride             = 0;
	uint32_t hs_stride             = 0;
	uint32_t domain                = 0;
	uint32_t partitioning          = 0;
	uint32_t output_topology       = 0;
};

struct NativeDrawShaderInfo {
	std::array<uint32_t, 3> registers {UINT32_MAX, UINT32_MAX, UINT32_MAX};
	bool indexed = false;
	[[nodiscard]] bool Enabled() const {
		return registers[0] != UINT32_MAX || registers[1] != UINT32_MAX || registers[2] != UINT32_MAX;
	}
};

struct ShaderVertexInputInfo {
	NativeDrawShaderInfo native_draw;
	static constexpr int RES_MAX = 32;

	ShaderBufferResource    resources[RES_MAX];
	ShaderVertexDestination resources_dst[RES_MAX];
	ShaderVertexInputBuffer buffers[RES_MAX];
	ShaderStageRuntime      stage;
	ShaderType                  logical_stage        = ShaderType::Vertex;
	int                     resources_num       = 0;
	int                     fetch_attrib_reg    = 0;
	int                     fetch_buffer_reg    = 0;
	int                     buffers_num         = 0;
	uint32_t                wave_size           = 64;
	uint32_t                scratch_size_dwords = 0;
	uint32_t                pa_cl_vs_out_cntl    = 0;
	ShaderClipSpaceTransform clip_space;
	ShaderMeshInputInfo      mesh;
	ShaderTessellationInputInfo tess;
	bool                    fetch_external      = false;
	bool                    fetch_embedded      = false;
};

struct ShaderComputeInputInfo: ShaderWorkgroupInputInfo {
	bool               async_compute              = false;
	uint8_t            float_mode                 = 0xc0;
	uint32_t           dispatch_threads_num[3]    = {0, 0, 0};
	uint32_t           workgroup_counts[3]        = {0, 0, 0};
	bool               group_id[3]                = {false, false, false};
	bool               dispatch_thread_dimensions = false;
	bool               lds_storage                = false;
	int                thread_ids_num             = 0;
	int                workgroup_register         = 0;
	bool               tg_size_en                 = false;
	ShaderStageRuntime stage;
};

enum class ShaderAlphaBlendSource : uint8_t { None, SourceAlpha, SourceAlphaOne, SourceAlphaZero };
enum class ShaderPixelParameterMode : uint8_t { FirstVertex, LastVertex, Rectangle };

struct ShaderPixelInputInfo {
	bool pixel_quad_correct = false;
	uint32_t                                       interpolator_settings[32]    = {0};
	uint32_t                                       input_num                    = 0;
	uint32_t                                       wave_size                    = 64;
	uint32_t                                       ps_system_input_base         = 0;
	uint32_t                                       custom_interpolation_mask    = 0;
	uint32_t                                       ps_perspective_center_vgpr   = UINT32_MAX;
	uint32_t                                       ps_perspective_sample_vgpr   = UINT32_MAX;
	uint32_t                                       ps_perspective_centroid_vgpr = UINT32_MAX;
	uint8_t                                        target_output_mode[8]        = {};
	uint32_t                                       target_shader_mask           = UINT32_MAX;
	std::array<Prospero::ColorComponentMapping, 8> target_export_mapping        = {};
	uint32_t                                       scratch_size_dwords          = 0;
	bool                                           ps_pos_x                     = false;
	bool                                           ps_pos_y                     = false;
	bool                                           ps_pos_z                     = false;
	bool                                           ps_pos_w                     = false;
	bool                                           ps_front_face                = false;
	bool                                           ps_front_face_all_bits       = false;
	bool                                           ps_ancillary                 = false;
	bool                                           ps_no_perspective            = false;
	ShaderPixelParameterMode                       parameter_mode = ShaderPixelParameterMode::FirstVertex;
	bool                                           ps_pixel_kill_enable         = false;
	bool                                           ps_depth_export_enable       = false;
	bool                                           ps_sample_mask_export_enable = false;
	bool                                           ps_sample_shading            = false;
	bool                                           dual_source_blending         = false;
	// Export logical alpha or per-channel source factors through MRT1 after channel swizzling.
	ShaderAlphaBlendSource                         alpha_blend_source = ShaderAlphaBlendSource::None;
	bool                                           ps_early_z                   = false;
	bool                                           ps_execute_on_noop           = false;
	ShaderStageRuntime                             stage;

	bool HasPositionInput() const { return ps_pos_x || ps_pos_y || ps_pos_z || ps_pos_w; }
};

union ShaderStageInputInfo {
	const ShaderVertexInputInfo*  vertex;
	const ShaderPixelInputInfo*   pixel;
	const ShaderComputeInputInfo* compute = nullptr;
};

inline const ShaderWorkgroupInputInfo* ShaderWorkgroupInput(ShaderType           stage,
                                                            ShaderStageInputInfo input) {
	switch (stage) {
		case ShaderType::Compute: return input.compute;
		case ShaderType::Mesh: return &input.vertex->mesh;
		default: return nullptr;
	}
}

uint32_t ShaderPixelParameterMappedLocation(const ShaderPixelInputInfo& info, uint32_t input);
uint32_t ShaderPixelParameterLocation(const ShaderPixelInputInfo& info,
                                      std::span<const uint32_t> active_inputs, uint32_t input);
bool     ShaderPixelParameterIsFlat(const ShaderPixelInputInfo& info, uint32_t input);
bool     ShaderPixelParameterIsCustom(const ShaderPixelInputInfo& info, uint32_t input);

struct ShaderSharp {
	uint16_t offset_dw : 15;
	uint16_t size      : 1;
};

struct ShaderUserData {
	uint16_t*    direct_resource_offset;
	ShaderSharp* sharp_resource_offset[4];
	uint16_t     eud_size_dw;
	uint16_t     srt_size_dw;
	uint16_t     direct_resource_count;
	uint16_t     sharp_resource_count[4];
};

struct ShaderRegisterRange {
	uint16_t start;
	uint16_t end;
};

struct ShaderDrawModifier {
	uint32_t enbl_start_vertex_offset   : 1;
	uint32_t enbl_start_index_offset    : 1;
	uint32_t enbl_start_instance_offset : 1;
	uint32_t enbl_draw_index            : 1;
	uint32_t enbl_user_vgprs            : 1;
	uint32_t render_target_slice_offset : 3;
	uint32_t fuse_draws                 : 1;
	uint32_t compiler_flags             : 23;
	uint32_t is_default                 : 1;
	uint32_t reserved                   : 31;
};

struct ShaderRegister {
	uint32_t offset;
	uint32_t value;
};

struct ShaderSpecialRegs {
	ShaderRegister      ge_cntl;
	ShaderRegister      vgt_shader_stages_en;
	uint32_t            dispatch_modifier;
	ShaderRegisterRange user_data_range;
	ShaderDrawModifier  draw_modifier;
	ShaderRegister      vgt_gs_out_prim_type;
	ShaderRegister      ge_user_vgpr_en;
};

struct ShaderSemantic {
	uint32_t semantic         : 8;
	uint32_t hardware_mapping : 8;
	uint32_t size_in_elements : 4;
	uint32_t is_f16           : 2;
	uint32_t is_flat_shaded   : 1;
	uint32_t is_linear        : 1;
	uint32_t is_custom        : 1;
	uint32_t static_vb_index  : 1;
	uint32_t static_attribute : 1;
	uint32_t reserved         : 1;
	uint32_t default_value    : 2;
	uint32_t default_value_hi : 2;
};

struct Shader {
	uint32_t             file_header;
	uint32_t             version;
	ShaderUserData*      user_data;
	const volatile void* code;
	ShaderRegister*      cx_registers;
	ShaderRegister*      sh_registers;
	ShaderSpecialRegs*   specials;
	ShaderSemantic*      input_semantics;
	ShaderSemantic*      output_semantics;
	uint32_t             header_size;
	uint32_t             shader_size;
	uint32_t             embedded_constant_buffer_size_dqw;
	uint32_t             target;
	uint32_t             num_input_semantics;
	uint16_t             scratch_size_dw_per_thread;
	uint16_t             num_output_semantics;
	uint16_t             special_sizes_bytes;
	uint8_t              type;
	uint8_t              num_cx_registers;
	uint8_t              num_sh_registers;
};

struct ShaderMappedData {
	Prospero::ShaderBinaryType type {};
	ShaderUserData* user_data           = nullptr;
	ShaderSemantic* input_semantics     = nullptr;
	uint32_t        num_input_semantics = 0;
	uint32_t        code_size_bytes     = 0;
	uint32_t        scratch_size_dwords = 0;
};

void ShaderInit();
void ShaderMapUserData(uint64_t addr, const ShaderMappedData& data);
uint32_t ShaderPixelExportTarget(uint32_t shader_mask, uint32_t export_index);

void     ShaderDbgDumpInputInfo(const ShaderVertexInputInfo& info);
void     ShaderDbgDumpInputInfo(const ShaderPixelInputInfo& info);
void     ShaderDbgDumpInputInfo(const ShaderComputeInputInfo& info);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_H_ */
