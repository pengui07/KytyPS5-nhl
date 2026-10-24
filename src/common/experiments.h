#ifndef EMULATOR_SRC_COMMON_EXPERIMENTS_H_
#define EMULATOR_SRC_COMMON_EXPERIMENTS_H_

#include <atomic>
#include <cstdint>

namespace Common::Experiments {

// Startup switches that can also be changed while the emulator runs, so A/B comparisons can
// alternate inside one match instead of comparing noisy separate runs.
// Each bit starts from its KYTY_* environment variable: "1" = on, "0" = off, unset = the default
// noted below (the NHL 27 recommended set is on by default). When KYTY_EXPERIMENT_FILE names a file,
// a background thread reads a decimal mask from it every 500 ms; while the file exists its mask
// replaces the environment defaults.
//
// Features that also have a live runtime flag (common/runtimeFlags.h, flags.ctl) are NOT here:
// KYTY_GUEST_SYNC, KYTY_LONG_BUFFER_GC_AGE, KYTY_RENDER_TARGET_MEMO (flag KYTY_RT_DECODE_MEMO),
// KYTY_FAULT_COALESCE, KYTY_SPARSE_BUFFER_GC, KYTY_DIRECT_EOP_INTERRUPTS, KYTY_READBACK_QUEUE.
// Their bit numbers are reused: 13 ClockWaitAhead (now env-only KYTY_CLOCK_WAIT_AHEAD, bit 13 is GpuConditionsBothPaths), 14 ComputeUnlocked, 16 AheadUnlocked, 17 BackingReadMru, 19
// DispatchAhead, 20 BackingLookupCache. Bit 4 (once SrtJit, now env-only KYTY_SRT_JIT) is CpBackingReads,
// bit 6 (once FalseShareReads, now env-only KYTY_FALSE_SHARE_READS) is WideWriteDownloads. Bit 0 (once GuestSync) is
// EarlySubmitWriters, bit 8 (once GpuThreadPriority, then ReadbackTransfer) is RobustLoads and bit 9 (once
// ImageRecycle, unconditional since the GTA port, then ReadbackSpin) is FusedMad.
// KYTY_RESOURCE_MULTI_CACHE (16-state resource memo) is read only by the resource memo itself.
enum Bit : uint64_t {
	EarlySubmitWriters = 1u << 0u, // KYTY_EARLY_SUBMIT_WRITERS (default off; a draw/dispatch writing bytes resource reads waited for is submitted at once)
	SkipRedundantBinds = 1u << 1u, // KYTY_SKIP_REDUNDANT_BINDS (default off)
	ReadbackCompute    = 1u << 2u, // KYTY_READBACK_COMPUTE (default on; second-queue readbacks on a compute-only queue family)
	CpuCopyForward     = 1u << 3u, // KYTY_CPU_COPY_FORWARD (default off)
	RobustLoads        = 1u << 8u, // KYTY_ROBUST_LOADS (default on; storage-buffer loads without per-load exec and bounds branches, relying on robustBufferAccess2; part of shader keys; was ReadbackTransfer)
	FusedMad           = 1u << 9u, // KYTY_FUSED_MAD (default off; V_MAD/V_MAC_F32 as one fused multiply-add, part of shader keys; was ReadbackSpin)
	CpBackingReads     = 1u << 4u, // KYTY_CP_BACKING_READS (default on; command-processor reads of read-protected pages go through the execution thread and the backing store instead of faulting; was SrtJit, now env-only KYTY_SRT_JIT)
	FaultPageDirect    = 1u << 5u, // KYTY_FAULT_PAGE_DIRECT (default on; direct SRT reads and read-ahead checks of pages that faulted before use the backing)
	WideWriteDownloads = 1u << 6u, // KYTY_WIDE_WRITE_DOWNLOADS (default on; a guest store into a GPU-written page downloads the GPU-written bytes of its 64 KiB window at once; was FalseShareReads, now env-only KYTY_FALSE_SHARE_READS)
	SrtBatchReads      = 1u << 7u, // KYTY_SRT_BATCH_READS (default on; also read by the program cache)
	DirectSrtReads     = 1u << 10u, // KYTY_DIRECT_SRT_READS (default on; SRT reads without the IR walk)
	GpuConditionsBothPaths = 1u << 13u, // KYTY_GPU_CONDITIONS_BOTH_PATHS (default on; a resource walk takes both paths of a branch on GPU-written bytes instead of waiting for them; was ClockWaitAhead, now env-only)
	ComputeUnlocked    = 1u << 14u, // KYTY_COMPUTE_UNLOCKED (default on; the execution thread materializes compute resources without the program lock)
	AheadUnlocked      = 1u << 16u, // KYTY_AHEAD_UNLOCKED (default on; the resolve thread materializes resources ahead without the program lock)
	BackingLookupCache = 1u << 20u, // KYTY_BACKING_LOOKUP_CACHE (default on; backing reads of another mapping use a per-64 KiB lookup cache)
	DispatchAhead      = 1u << 19u, // KYTY_DISPATCH_AHEAD (default on; the resolve thread resolves direct dispatches' programs ahead, as for draws)
	BackingReadMru     = 1u << 17u, // KYTY_BACKING_READ_MRU (default on; backing reads remember the thread's last two mappings, not one)
	NativeMinMax       = 1u << 11u, // KYTY_NATIVE_MINMAX (default on; f32 min/max as GLSL NMin/NMax, part of shader keys; was TextureDescCache)
	ImageLookupHint    = 1u << 12u, // KYTY_IMAGE_LOOKUP_HINT (default on; skip the region scan for known images)
	WideFaultWindows     = 1u << 15u, // KYTY_WIDE_FAULT_WINDOWS (default on; a CPU write fault releases KYTY_FAULT_COALESCE_PAGES pages, else 4; was AdaptiveResourceMemo)
	CleanFromDirect      = 1u << 18u, // KYTY_CLEAN_FROM_DIRECT (default on; strict evaluator reuses direct SRT words)
	AsyncSubmit          = 1u << 21u, // KYTY_ASYNC_SUBMIT (no consumer: the GPU pipeline's recording thread submits)
	LazySharedWrites     = 1u << 22u, // KYTY_LAZY_SHARED_WRITES (default off; CPU stores next to few GPU-written bytes do not wait)
	NativeFp16           = 1u << 23u, // KYTY_NATIVE_FP16 (default off; f16vec2 math for packed f16 ops, needs shaderFloat16)
	FaultPageReads       = 1u << 24u, // KYTY_FAULT_PAGE_READS (default off; SRT reads of pages that faulted before use the backing)
	GpuColorClear        = 1u << 26u, // KYTY_GPU_COLOR_CLEAR (default off; fast-clear checks of GPU-written metadata on the GPU)
	ExecSelects          = 1u << 25u, // KYTY_EXEC_SELECTS (default off; drop EXEC-mask selects nobody observes, part of shader keys)
	PooledBdaBuffers     = 1u << 27u, // KYTY_POOLED_BDA_BUFFERS (default off; small device-address buffers share VMA blocks)
	AsyncGuestReads      = 1u << 28u, // KYTY_ASYNC_GUEST_READS (default off; a guest thread waits for its own readback)
	BdaDirtyRegions      = 1u << 29u, // KYTY_BDA_DIRTY_REGIONS (default off; BDA preparation syncs only CPU-written regions)
	LdsLaneSlots         = 1u << 30u, // KYTY_LDS_LANE_SLOTS (default on; per-lane LDS slots of non-compute stages as a small array, part of shader keys)
	ProtectSharedLock    = uint64_t {1} << 32u, // KYTY_PROTECT_SHARED_LOCK (default off; page-tracking protections of different pages run in parallel)
	ClockWaitPredict     = uint64_t {1} << 33u, // KYTY_CLOCK_WAIT_PREDICT (default off; WAIT_REG_MEM on a pending clock write need not wait for it)
	FalseShareAnyThread  = uint64_t {1} << 34u, // KYTY_FALSE_SHARE_ANY_THREAD (default on; CP and guest reads of clean bytes on GPU-dirty pages use the backing)
	CpCleanRegReads      = uint64_t {1} << 35u, // KYTY_CP_CLEAN_REG_READS (default off; the command processor reads clean indirect register tables from the backing)
	CpSkipIdleSpin       = uint64_t {1} << 36u, // KYTY_CP_SKIP_IDLE_SPIN (default off; work queued while the command processor waited for the execution thread starts at once, not after a 2 ms spin)
	ExecBranches         = uint64_t {1} << 39u, // KYTY_EXEC_BRANCHES (default on; runs of lane-wise instructions under one partial EXEC run in a divergent branch instead of selecting each written VGPR per lane; part of shader keys)
	ExecBranchMemory     = uint64_t {1} << 40u, // KYTY_EXEC_BRANCH_MEMORY (default on; ExecBranches runs also take memory operations under their EXEC; part of shader keys)
	DispatchResolver     = uint64_t {1} << 41u, // KYTY_DISPATCH_RESOLVER (default off; a second resolve thread resolves dispatches ahead while the first resolves draws)
	BdaLoadGroups        = uint64_t {1} << 42u, // KYTY_BDA_LOAD_GROUPS (default on since 2026-10-23; consecutive DWORD loads through an address share one page lookup; part of shader keys)
	DrawResolver         = uint64_t {1} << 43u, // KYTY_DRAW_RESOLVER (default on since 2026-10-22; earlier off for 2 guest job-thread crashes in 8 boots, since fixed causes; a second resolve thread resolves about half of the draws ahead, by shader addresses)
	SpeculateDirtyReads  = uint64_t {1} << 37u, // KYTY_SPECULATE_DIRTY_READS (default off; shader-table reads of GPU-written bytes the GPU keeps rewriting unchanged use the backing, checked later)
	// A/B only (no environment variable): selects the code a pending change replaces.
	AbLegacy = 1u << 31u,
	// A/B only, like AbLegacy, for the changes made after it was taken (AbLegacy also turns off
	// the earlier changes it still selects).
	AbPending = uint64_t {1} << 38u,
	// More A/B-only bits, so that one boot can compare several pending changes (KytyPS5-toolsbmulti.sh).
	AbPending2 = uint64_t {1} << 44u,
	AbPending3 = uint64_t {1} << 45u,
	AbPending4 = uint64_t {1} << 49u,
	// A/B only and read by nothing else: a pending change's A/B changes nothing more.
	AbPending5 = uint64_t {1} << 56u,
	// KYTY_NATIVE_FRACT (default on; AMD devices): V_FRACT_F32 is GLSL Fract alone, which the AMD
	// compiler emits as the same v_fract_f32 (clamped below one), without the clamp that
	// x - floor(x) needs elsewhere; part of shader keys. Was AbPending6.
	NativeFract = uint64_t {1} << 57u,
	AbPending7 = uint64_t {1} << 58u,
	// KYTY_F16_CONVERTS (default on; with NativeFp16): f32 <-> f16 conversions as OpFConvert on f16
	// instead of PackHalf2x16 / UnpackHalf2x16; part of shader keys.
	F16Converts = uint64_t {1} << 59u,
	// KYTY_F16_IMAGE_PACK (default on; with F16Converts): 16-bit float image results packed as f16
	// pairs from OpFConvert instead of PackHalf2x16 per lane; part of shader keys.
	F16ImagePack = uint64_t {1} << 60u,
	// KYTY_EXEC_RUN_TRANSPARENT (default on; with ExecBranches): resource handles, lane ids and
	// references no longer end an ExecBranches run; part of shader keys.
	ExecRunTransparent = uint64_t {1} << 61u,
	// KYTY_ALIGNED_BUFFER_OFFSETS (default on; on devices binding storage buffers at any dword):
	// scalar buffer reads leave out the binding prefix, which is below one dword there; part of
	// shader keys.
	AlignedBufferOffsets = uint64_t {1} << 62u,
	// KYTY_GUEST_OFF_STAGE_CORES (default on): threads without a CPU of their own (guest, driver
	// and helper threads) keep off both logical CPUs of the execution and resolve threads' cores.
	GuestOffStageCores = uint64_t {1} << 46u,
	// KYTY_GUEST_OFF_COPY_CORE (default off; with GuestOffStageCores): the host copy thread's
	// core too. Bit 47 was GuestOffStageSiblings (only the two siblings: no gain).
	GuestOffCopyCore = uint64_t {1} << 47u,
	// KYTY_GUEST_OFF_CP_CORES (default off; with GuestOffStageCores): the command processor's and
	// the recording thread's cores too.
	GuestOffCpCores = uint64_t {1} << 48u,
	// KYTY_EXEC_BRANCH_CONTINUE (default on; with ExecBranches): runs in loop continue targets
	// (loop bodies that branch back) too; part of shader keys.
	ExecBranchContinue = uint64_t {1} << 50u,
	// KYTY_F16_LANE_VIEWS (default on; with NativeFp16): scalar f16 ops read and write their
	// 16-bit lanes as halves of an f16vec2 instead of masking them out of U32s; part of shader keys.
	F16LaneViews = uint64_t {1} << 51u,
	// KYTY_UNDEF_VGPRS (default on): VGPRs the hardware does not initialize read as undefined instead
	// of 0 before their first write; part of shader keys. Bit 52 was LazyGuestBarriers (dropped).
	UndefVgprs = uint64_t {1} << 52u,
	// KYTY_FTZ32 (default on): shaders declare DenormFlushToZero for 32-bit floats (the guest's MODE
	// flushes them) and transcendental inputs are no longer flushed by hand; part of shader keys.
	Ftz32 = uint64_t {1} << 54u,
	// KYTY_MOVREL_BOUNDS (default on): a V_MOVRELS/D_B32 selects over the registers of its array only
	// (see MovRelExtents) instead of all after its base; part of shader keys.
	MovRelBounds = uint64_t {1} << 55u,
	// KYTY_EARLY_LATE_FRAGMENT_TESTS (default on): pixel shaders whose only side effects are the emulator's
	// address fault records keep early depth tests (EarlyAndLateFragmentTestsAMD); shader key bit.
	EarlyLateFragmentTests = uint64_t {1} << 53u,
};

namespace Detail {
// The live mask; bit 63 stays set until the first use reads the environment. Enabled() runs
// millions of times per second on the GPU thread, so its fast path is inline.
inline constexpr uint64_t Uninitialized = uint64_t {1} << 63u;
extern std::atomic<uint64_t> g_mask;
[[nodiscard]] uint64_t InitMask() noexcept;
} // namespace Detail

[[nodiscard]] inline uint64_t Mask() noexcept {
	uint64_t mask = Detail::g_mask.load(std::memory_order_relaxed);
	if ((mask & Detail::Uninitialized) != 0) [[unlikely]] {
		mask = Detail::InitMask();
	}
	return mask;
}

[[nodiscard]] inline bool Enabled(Bit bit) noexcept {
	return (Mask() & bit) != 0;
}

// KYTY_FALSE_SHARE_READS=1 (env only; was experiment bit 6): the old single-GPU-thread design's
// reads of clean bytes on GPU-written pages through the backing.
[[nodiscard]] bool FalseShareReads() noexcept;

} // namespace Common::Experiments

namespace Common::Timeline {

// Debugging: KYTY_DBG_TIMELINE=<seconds> records marks from every thread for 1.5 s
// (KYTY_DBG_TIMELINE_MS) after that delay, then prints them once in time order (NHL27TL lines):
// who waits on whom in a frame.
[[nodiscard]] bool Enabled() noexcept;
void Mark(const char* what, uint64_t value = 0) noexcept;
// While recording: one "ra" mark per return address of the calling stack (up to 10 frames, the
// caller first), so a wait can be traced to its call site (symbolize with the image base).
void MarkStack() noexcept;

} // namespace Common::Timeline

#endif // EMULATOR_SRC_COMMON_EXPERIMENTS_H_
