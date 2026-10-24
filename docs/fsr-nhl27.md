# FSR test branch — 2026-10-17

## Update — 2026-10-18: hidden HDR game-input shadow provider

This supersedes the earlier "no provider" status. KYTY_FSR3_NHL27_SHADOW=1 is
an off-default startup diagnostic (no runtime bit). With the optional FSR3 build,
KYTY_FSR3_HDR=1 and KYTY_FSR3_LUMA_FORMAT_FIX=1, it reads the actual 208-byte
GPU constants and snapshots prepare-pass color/depth/raw motion into owned
images before alias reuse. All nine graph edges and the matching accumulation
constants are checked. Guest passes/history/display stay active. The host output
is hidden and its history is reset every frame; this does not certify temporal
motion direction, motion-jitter inclusion, camera units or reset handling.

Supported shaderInt16 is enabled only for this mode; unsupported hardware stops
the diagnostic. The first three dispatches and every 600th dispatch log SDK
status and a synchronized center 2x2 FP16 finite check. Failure stops subsequent
shadow work. These sparse checks are not a complete image-quality test. GPU
constant readback uses synchronous submission waits; this mode is not a
performance candidate and no gain is established.

The normal-size isolated test passed at Medium integrity with Khronos validation
and exit 0. KYTY_FSR3_TEST_CAPTURE_SIZE=1 optionally runs the GPU fixtures at
2348x1320 -> 3520x1980; all named checks passed including packed HDR and FP16
history, with zero validation stderr. The large-test parent timed out at 20s,
while the test subsequently completed, so its exact exit code was not retained.
Nonfinite reciprocal metadata is rejected in the captured-constant fixture.

Saved implementation commit 3983f3db. Live build 77ee25e6-dirty ran HDR shadow
dispatches at 2348x1320 -> 3520x1980 in user-confirmed, screenshot-verified Free
Skate (Pettersson, Vancouver). Readbacks at guest frames145/745/1345/1945 were
finite and nonblack, with RGB near (1.85,1.90,1.98), preserving values above1.
These checks establish successful SDK execution with actual scene inputs, not
output quality or temporal reconstruction: each frame still resets history.

The initial run exposed missing shaderInt16; the rebuilt run logs enabled=1
with no pCode-08740 errors, SDK failure or device-lost observed. Thirty validation
errors remain: ten each mipLevels-00958 and dstOffset-00150/00151, concerning
128x128 images requesting ten mip levels and out-of-bounds image copies. The
standalone SDK test is clean at the same render/output size. This suggests the
guest texture path, but does not establish a shadow-off baseline. Do not claim
full-game validation clean or displayed FSR3. Motion direction/jitter inclusion,
camera units, history resets and visible output quality remain to be verified.

Workspace evidence prefixes: kyty-play-nhl27/fsr-shadow-resume-20261018-153602,
fsr-shadow-check-20261018-154725 and fsr-shadow-feature-20261018-155534.
An uncompiled experimental mip workaround was preserved outside this checkout in
kyty-play-nhl27/fsr-mip-guard-uncompiled-20261018-155710, then removed from active
source after recovery checks. It is not part of this change.
## Update — 2026-10-18: verified Free Skate bindings and opt-in HDR backend

This section supersedes the older pending-capture and SDR-only backend status below.
The user-triggered F2 dump completed at
F:/emuf/captures/fsr-targeted-20261018-120423/frame_0. Its FP16 temporal output
visually confirms Free Skate. Guest frame index 62 uses render 2348x1320 and
temporal display 3520x1980, followed by 4K LUT/composite processing. Four captured
208-byte GPU constant buffers agree in their first 49 DWORDs, and all nine
inter-pass resource connections match the historical graph.

PrepareInputs-like hash 4618098710b83f8f samples depth/motion/color at image slots
0/1/3. Accumulation hash a36f0b4faa20493a writes UNORM history at slot 6 and the
FP16 image consumed downstream at slot 7. Outputs are promptly aliased/reused.
Keep role names provisional until linked to guest program names.

Across 48,345 interior sample points, prepared depth matches the 3x3 maximum
within 2e-7; prepared motion matches the maximum-depth neighbor's raw motion
times (-0.5,+0.5), maximum error 1.1921e-7. Identity/opposite scaling fail all
sampled nonzero-motion checks at 2e-6 tolerance. This establishes the local
prepare transform, not temporal direction or motion-jitter cancellation.
At this extent, the SDK dispatch motion scale for that transform is (-1174,+660).
All 317 saved jitter records fit the width-derived 17-phase Halton cycle.

Packed-float color has channels above 1 in 84.44% of pixels; FP16 temporal output
reaches 967. Do not clamp these inputs through SDR. For this frame, the SDK-style
depth equation gives candidate near 20 / far 700 in unknown game units; vertical
FOV is 0.29998063 radians and delta is 38.7957767 ms. Validate physical units,
temporal conventions and resets before certifying a provider.

The host backend now supports HDR only when both Fsr3Frame::hdr is true and
the off-by-default environment flag KYTY_FSR3_HDR=1 is set. It accepts packed
B10G11R11 UFLOAT or RGBA16 SFLOAT input and requires RGBA16 SFLOAT output; the SDK
HIGH_DYNAMIC_RANGE context flag preserves the linear HDR path. SDR behavior
remains the default. Existing verified-frame, non-aliasing and luma-format-fix
requirements still apply. There is still no game provider or automatic activation.

The separate GPU test executable passed with FSR3 enabled and Khronos validation:
HDR opt-in rejection, packed-float and FP16 inputs containing (2,4,8), three history
frames per format with FP16 readback within 3%, plus existing FSR1/SDR FSR3 checks.
No validation warnings/errors were reported. A preceding FSR3-disabled build
also passed its FSR1/probe checks; it does not count as HDR validation.
Only build-fsr3-tests was configured/relinked. The running emulator was neither
rebuilt nor interrupted. Test executable and actual test token were Medium (8192);
no security settings changed.

Workspace evidence: kyty-play-nhl27/fsr-targeted-20261018-120423-analysis/analysis.json,
jitter-checks.json in that same folder, and
kyty-play-nhl27/fsr-hdr-enabled-gpu-tests-20261018.log / .err.log.
The full capture hit the readback cap for 35 later image requests, including the
final 4K composite; all recorded temporal raw files are complete. These are
synthetic backend checks and one-frame capture analyses, not gameplay quality or
performance results. No gain established.


## Implemented

`KYTY_FSR1=1` enables AMD FSR 1 EASU + RCAS at presentation. It is **off by
default**, live runtime flag **bit 36** in `common/runtimeFlags.h` (revision now
starts at bit 37). It can be switched with a `KYTY_FSR1=0|1` line in the selected
`KYTY_FLAG_FILE`, using the existing runtime refresh. `KYTY_FSR_SHARPNESS` is a
startup float in [0,1], default 0.2. Invalid values use the default.

This spatial mode needs an already-antialiased SDR image. It preserves guest
TAA and HUD. Equal input/output sizes run RCAS only; a larger source, HDR, blank
frames or unsupported formats retain the normal blit. It does not lower the
game's resolution. `NHL27FSR` logs distinguish EASU+RCAS, RCAS and fallback.
The processing images belong to individual swapchain slots; the existing queue
wait protects destruction/recreation. No new work runs while disabled.

`KYTY_ENABLE_FSR3=ON` builds a separate **FSR 3.1.4 Vulkan SR backend**. It uses
the standalone upscaler, with caller-owned shared resources, avoiding the
combined wrapper's upscaling-only lifetime problem. There is **no NHL temporal
capture provider yet and no automatic FSR3 activation in the game**. No frame
generation is included. The backend requires verified, same-frame, non-aliasing
color/depth/motion snapshots plus jitter, motion scale and clipping/FOV metadata.
It rejects invalid/unverified inputs, duplicates and configuration changes;
missing frames or discontinuities reset history. The caller owns barriers and
must wait for GPU completion before context destruction/configuration changes.

The SDK 1.1.4 Vulkan shaders declare `rw_luma_history` as `rgba8` although the
SDK allocates RGBA16F. Validation caught that mismatch. FSR3 context creation
therefore also requires the off-by-default startup flag
`KYTY_FSR3_LUMA_FORMAT_FIX=1`. Its shader-module wrapper changes only the named
variable's isolated SPIR-V image type to Rgba16f, refusing shared types. Runtime
DLLs remain unchanged. `NHL27FSR3 fix=luma-history-format` records the correction.

## Build and test

FSR 1 shaders are vendored unmodified from AMD FSR v1.0.2, with their MIT license.
Normal builds do not require an external SDK. The optional FSR3 path requires
official FidelityFX SDK **1.1.4** headers and the matching Windows x64 Vulkan
backend/upscaler runtime and import libraries. The pinned runtime DLLs used here
come from the user's referenced DLSSKytyPS5 v1.5.0 release, not from an emulator
executable we run. Preparation script: `tools/prepare-fsr3-sdk.ps1`; it verifies
both archive SHA256 hashes, extracts only the two required runtime DLLs, and
generates import libraries from their exports using LLVM.

Sources:
- https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v1.1.4
- https://github.com/iska0702/DLSSKytyPS5/releases/tag/v1.5.0
- https://github.com/GPUOpen-Effects/FidelityFX-FSR/tree/v1.0.2

Configure with `-DKYTY_ENABLE_FSR3=ON -DFIDELITYFX_SDK_DIR=<prepared SDK root>`.
Build `kyty_emulator` and `fsr_upscaler_tests`; run the latter for **GPU** testing.
The test uses the same production implementations. Synthetic motion/depth are
test fixtures only, never a game fallback.

Final RX 6800 XT run passed with the local Khronos validation layer enabled and
no validation warnings/errors: FSR1 EASU+RCAS over three frames, resource reuse,
native RCAS, resize/downscale rejection, quadrant orientation and zero sharpening;
FSR3 three history frames plus recovery after a rejected frame, output readback
against a constant-color reference and invalid-input rejection. The runtime flag
bit/revision mapping and FSR2 candidate decoder also passed. FSR3 used the luma
format fix above. Evidence: `../kyty-play-nhl27/fsr-final-vvl-tests.log` in the
workspace. The earlier validation failure is retained in `fsr-gpu-tests-final.log`.

An independent `KYTY_ENABLE_FSR3=OFF` build passed the FSR1/probe checks with
validation (`fsr1-only-vvl-tests.log`); its test executable imports no FidelityFX
DLLs. Both final FSR1 SPIR-V modules pass
`spirv-val --target-env vulkan1.3`. These are synthetic GPU checks, not NHL image
quality or performance results.

## Runtime evidence and display sizing

The guarded `5d9715e-dirty` build booted NHL into the main menu with normal 4K
guest video-out, default branch runtime flags plus `KYTY_FSR1=1`,
`KYTY_FSR2_PROBE=1`, and bulk shader warmup disabled. The menu run reached over
12,000 presented frames and closed normally. The last probe summary recorded
4,407,296 compute dispatches, 4,226,867 clean backing reads, 11,075,789 refusals
and **zero candidates**. This covers intro/menu activity only and does not prove
FSR2 is inactive in gameplay. Evidence: `fsr-run-menu-20261017.log` and screenshots
in `fsr-nav/`, under the workspace run directory. Free skate was not reached.

That binary's FSR1 status used the launcher's suppressed LOGF channel. The next
rebuild prints and flushes `NHL27FSR` to stdout. A subsequent 1920x1080-window run
reached screenshot-confirmed free skate with Hronek, and the status explicitly
reported fallback. FSR1 activation remains unverified in NHL; 4K guest video-out
is larger than the display. That run ended at 3,284,992 dispatches / 3,520,217
clean reads / 8,823,212 refusals / zero candidates. Evidence:
`fsr-run-free-skate-20261017.log` and `fsr-nav/001_after-load.png` in the workspace
run directory. It closed normally before the compute-capture rebuild. Title-bar
FPS was not a paired measurement and establishes no performance gain.

The initial run with `KYTY_OUTPUT_1080P=1`, copied experiment flags and bulk warmup
crashed on a guest null write after warmup, before presentation. It changed
several conditions at once, so the cause is unproven. Retained evidence:
`fsr-run-startup-crash.log`. No device-lost occurred in either run.

The monitor is 2560x1440. An oversized 3840x2160 test window was switched to
desktop fullscreen; future `fsr-launch.cmd` runs request a **1920x1080 window**.
Window size and guest/internal render resolution are separate. The workspace's
`tools/kyty-step.ps1` now supports F11 and refuses to send keys unless the emulator
has foreground focus. Navigation lost focus intermittently; do not assume a
sent key changed the menu. No paired performance measurement or gain established.

`KYTY_DBG_FRAME_DUMP_CS_CONSTANTS=1`, off by default, extends the existing F2
frame dump to log all compute bindings and copy bound GPU buffers whose guest
size and host range are at most 4096 bytes. This survey avoids the protected
guest backing read used for specialization bytes in the full trace. Existing
hash-selected compute tracing retains its original behavior. Dumps occur after
each dispatch; they provide evidence, not a temporal provider or a guarantee
that the captured block is cbFSR2. Larger buffers/other offsets can still be missed.
The Release capture build passed (`fsr-capture-build.log`).

The workspace `fsr-launch.cmd` now enables that survey, raw frame dump and
`KYTY_DBG_FRAME_DUMP_DIR=F:\emuf\captures\fsr-free-skate`, keeping the guarded
`nhl27.bat play` launch. It booted successfully as `a34c03f-dirty` and is waiting
at sign-in; automatic foreground focus failed, though the user confirmed manual
keyboard/controller input works. A requested manual free-skate/F2 capture is
pending. The capture feature has not yet been exercised in game. Re-enumerate
host processes before acting; do not reuse recorded PIDs.

## NHL temporal mapping still required

The user's follow-up corrects the initial interpretation: `fJitter`,
`fMotionVectorScale` and related names belong to **18 embedded AMD FSR2 compute
shader permutations**, not Frostbite's own TAA fields. Reflection tables are at
approximately `0xE185D76..0xE1AC079`, with header `6.0.0.0 build 671`; names include
`cbFSR2`, `r_input_color_jittered`, `r_input_motion_vectors`, `r_input_exposure`,
`r_reactive_mask`, `r_dilated_motion_vectors`, `r_dilatedDepth`. Presence in the
executable does not establish that these shaders execute in free skate.

The supplied cbFSR2 byte offsets (reflection/public-layout match, **not yet
verified at runtime**) are:

| Offset | Field |
|---:|---|
| 0 / 8 / 16 / 24 | iRenderSize / iMaxRenderSize / iDisplaySize / iInputColorResourceDimensions (int2 each) |
| 32 / 40 / 44 | iLumaMipDimensions (int2) / iLumaMipLevelToUse / iFrameIndex |
| 48 | fDeviceToViewDepth (float4) |
| 64 / 72 / 80 / 88 | fJitter / fMotionVectorScale / fDownscaleFactor / fMotionVectorJitterCancellation (float2 each) |
| 96 / 100 / 104 / 108 | fPreExposure / fPreviousFramePreExposure / fTanHalfFOV / fJitterSequenceLength |
| 112 / 116 / 120 | fDeltaTime / fDynamicResChangeFactor / fViewSpaceToMetersFactor |

`KYTY_FSR2_PROBE=1` is an off-by-default diagnostic. On compute dispatches it
tries the first 128 bytes of bound buffers at least 128 bytes long, using clean backing
only, with no GPU wait or protected-memory read. It prints at most 64 plausible
matches as `NHL27FSR2 candidate=...`, including raw constants and bound image
slots, and periodic scan/refusal counts. The decoder checks dimensions, finite
camera values and downscale consistency. A match remains a hypothesis and never
enables FSR3. Other layouts/offsets or GPU-dirty constants can be missed.

AMD SDK source confirms `fTanHalfFOV` is **horizontal**, and FSR2's `fDeltaTime`
uses **seconds**. FSR3 dispatch expects vertical FOV and milliseconds; conversions
must be explicit when a verified provider is connected.

Other Ghidra findings remain leads, not GPU bindings:

Data-table references supplied by the user: `ShaderProgram_TemporalAAResolve`
string `0xD882714`, pointers `0xE9E86E8`, `0x118EBC90`;
`TemporalAADeJitterFilterKernel` string `0xDE6B443`, pointers `0xF1DCB38`,
`0xF1DCB70`, `0x1252B5A8`, `0x1252B5F0`;
`VelocityVectorsDeriveFromDepthEnable` string `0xD888277`, pointers `0xEBFEA40`,
`0x11D166B8`. They are likely option/pass registries; consumers have not been
followed. These are executable-analysis addresses, not verified runtime bindings.

- `ShaderProgram_TemporalAAResolve`, `TemporalAADeJitterFilterKernel`,
  `TemporalAADisocclusionRejectionBinaryThreshold`,
  `TemporalAAPostSharpeningAmount`, `TemporalAADofCocFilterEnable`.
- `MainOpaque_MotionVectorsPass…`, `ShaderExpressionMotionVectorsEnable`,
  `fMotionVectorScale`.
- `VelocityVectorsDeriveFromDepthEnable`, `VelocityVectorFromDepth`.
- `UpsamplerDebugMode_PrepareInputs_DilatedMotionVector` was not found as a
  plain string in the follow-up scan; no address is established.
- FXAA/SMAA shader program names exist, but their active selector is unverified.

Next: map the actual TAA/upscaler producer's shader hash, texture slots, viewport,
constant offsets and values. Capture depth and motion **before reuse/aliasing**
and carry those snapshots with the matching prepared frame. Validate motion
direction, scale, jitter inclusion, depth convention and camera planes against
camera movement. Then connect Fsr3::Record and validate HUD/history transitions.
Merely finding a similarly sized texture is insufficient.

Existing captures include 3D match-setup and franchise-menu frames (frame_0,
frame_5, frame_15 inspected); they are not evidence of free-skate temporal capture.
The v1.5.0 ASTRO-specific capture must not be used as an NHL binding map.
An offline scan of 812 existing 128..4096-byte captured buffer files, including
16-byte-aligned offsets, found no plausible cbFSR2 block. Those captures did not
systematically collect all compute constants, so this does not establish that
FSR2 is inactive. Evidence: `../kyty-play-nhl27/fsr2-existing-capture-scan.json`
in the workspace (not part of this checkout).

For any performance claims, use paired interleaved windows in one process.
Lower source resolution must be confirmed independently: `KYTY_OUTPUT_1080P=1`
changes what is reported to the guest, whose actual buffer choice must still be
checked. A 4K guest frame displayed at 1440p takes the downscaling fallback.
