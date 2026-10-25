# NHL 27 performance switches (claude/nhl27-perf)

The 2026-10-17 merge reconciles experiment-bit collisions and compute resolve-ahead
controls; see [PR 8 merge resolution](pr8-merge-resolution.md) before using old decimal
experiment masks from the base branch.

There are two kinds of switches, and a key name belongs to exactly one of them.

* **Runtime flags** (`src/common/runtimeFlags.h`): read from the environment at startup and from
  `flags.ctl` (`KEY=0|1` per line, `KYTY_FLAG_FILE` names another file) every 250 ms, so they can be
  flipped while the game runs. A key missing from both means the default in `FlagSet`; an explicit
  `0`/`1` overrides it.
* **Env-only startup switches** (`src/common/experiments.cpp`): `KYTY_X=1` enables, `KYTY_X=0`
  disables, unset means the default below. They are read once at startup. For in-match A/B tests,
  `KYTY_EXPERIMENT_FILE` names a file holding a decimal mask (bit numbers below); while the file
  exists its mask replaces the environment values.

## Threaded GPU pipeline (merge of pengui07/nhl27-gta-eadb, 2026-10-14)

pengui07's `nhl27-gta-eadb` (upstream main, TheCruZ's GTA V work and its NHL 27 port) runs the
emulated GPU as a pipeline of threads: command processor, resolve (program selection and SRT
materialization ahead of execution), execution (owns the renderer and caches) and recording
(replays the recorded commands into Vulkan command buffers and submits them).
`KYTY_GPU_PIPELINE=0` (or `--gpu-pipeline 0`) runs everything inline; stage threads are pinned
(`KYTY_GPU_THREAD_CPUS`, default 4,6,8) and run at high priority (`KYTY_GPU_THREAD_PRIORITY=0`
keeps normal priority).

Measured on this merge (cross-boot, 120 s of a match then 60 s paused, play-nhl defaults):

| Build | Match fps | Paused fps |
|---|---|---|
| Merge, pipeline on (PGO profile v4) | 18.0 / 22.9 / 18.4 | 20.7 / 22.7 / 19.5 |
| Merge, pipeline on (stale PGO), 6 boots | 17-22, mean 20.0 | 20.6-26.3, mean 23.6 |
| Merge, `KYTY_GPU_PIPELINE=0` | 14.7 / 14.2 | 16.0 / 18.6 |
| pengui07/nhl27-gta-eadb alone (no PGO) | 15.0 | 15.6 |
| claude/nhl27-perf before the merge, 8 boots the same day | 19-24, mean 20.9 | 19-28, mean 23.2 |

So the pipeline is worth about +40% on the merged code, but the merged code inline is slower than
the pre-merge branch was, and the two cancel out in a match: the merge is about level with the
pre-merge build (the game also switches between 60 and 30 Hz flips on its own, which quantizes the
frame rate). Per-boot spread is several fps; these are not in-match A/Bs.

Changes made in the merge:
- `KYTY_READBACK_QUEUE` with the pipeline lost the device in 3 of 5 boots: a second-queue readback
  waited for a timeline tick the recording thread had not submitted yet. It now waits until the
  tick is submitted (`CommandScheduler::WaitSubmitted`); 3 of 3 boots clean, and a failed
  second-queue submission falls back to the ordinary readback.
- Retired experiment bits: 8 (`KYTY_GPU_THREAD_PRIORITY` as an experiment: the stage threads set
  their own priority, and the env key now belongs to the pipeline), 9 (`KYTY_IMAGE_RECYCLE`:
  image recycling is TheCruZ's, unconditional, 768 MiB kept for NHL 27's aliased targets) and 21
  (`KYTY_ASYNC_SUBMIT` has no consumer: the recording thread submits).
- Bit 0 is now `KYTY_EARLY_SUBMIT_WRITERS` (default off): a draw or dispatch that writes bytes a
  strict resource read had to download ends the host command buffer, so the next such read waits
  only for that work on the second queue. In-match A/B, 4 rounds: 27.0 fps off, 20.8 fps on
  (every round slower), so it stays off: the extra submissions cost more than the shorter waits.
- A dispatch whose arguments the CPU wrote gets them from the stream buffer, which has no device
  address; the dispatch-args guard (and the GPU color-clear check) skip it instead of failing
  (`streamBuffer.cpp:121`, once during a match load).
- The first boot after shader-emitter changes recompiles every shader (minutes at 0 fps while
  loading); later boots use the pipeline cache.

### Readbacks on a compute queue (bit 2, `KYTY_READBACK_COMPUTE`, default on)

`KYTY_DBG_CP_READS=1` prints what the synchronous downloads wait for (`NHL27RBWAIT`, every 2 s:
who read, whether the bytes read were GPU-written themselves, how many submissions back the
buffer's last write is, and whether it had finished) and which command-processor reads of guest
memory are slow (`NHL27CPREAD`: indirect register loads and `WAIT_REG_MEM` values, by time).

In a match, about 300 to 4000 downloads per 2 s copy data whose writer had already finished, yet
each took 0.2 to 0.8 ms (up to a third of the execution thread): the second queue of the graphics
family runs on the same engine as the frame's work and waits behind it. The device also has a
compute-only family (async compute, 8 queues on the RX 9070 XT) and a transfer-only family
(SDMA). Every buffer is now shared with them (concurrent sharing; `KYTY_READBACK_FAMILIES=0`
leaves the device as before), and bit 2 copies on the compute family, bit 8
(`KYTY_READBACK_TRANSFER`) on the transfer family. In-match A/B, 6 rounds of 20 s:

| Readback queue | fps | vs graphics |
|---|---|---|
| Graphics family (before) | 25.3 | |
| Compute family | 28.7 | +13.8% (paired +3.5 +- 1.4, every round faster) |
| Transfer family | 24.7 | -2% (+-7.5: copies of finished data take 0.1 ms, but some phases are much slower) |

With the compute queue, the remaining download waits of the execution thread in play (per 2 s):
- 20 to 40 faults on bytes that only share a page with GPU-written ones, each waiting about 6 ms
  for the previous submission (130 to 230 ms): the plain loads of direct resource-table reads
  (`SrtWalker::DirectReadRun`) and of read-ahead checks (`GuestReadLog::StillValid`). Bit 5
  (`KYTY_FAULT_PAGE_DIRECT`, default on) reads such bytes from the backing once their page has
  faulted on this thread (as `KYTY_FAULT_PAGE_READS` already did for ordinary resource reads);
  with it these waits are gone from `NHL27RBWAIT`.
- About 60 reads of bytes the GPU wrote in the command buffer being recorded (strict resource
  reads): 50 to 300 ms, a real dependency.
- Bit 9 (`KYTY_READBACK_SPIN`, fence polling) stays off: within noise.

The game asks for its flip rate every frame (`VideoOutSetFlipRate`; `KYTY_DBG_QUEUE_TIME=1`
prints changes as `NHL27FLIPRATE`): 60 Hz during play (stretches of 60 to 70 s in a match), 30 Hz
in menus, the pause menu and some stoppages. A paused match is therefore capped at 30 fps and is
no longer a useful A/B scene; in play the execution thread is the limit (`NHL27QUEUES`: q0 busy
990 ms/s, about 110,000 operations/s at 9 us each).

The command processor waits about 23% of its time in `WAIT_REG_MEM`: NHL 27 writes the GPU clock
at end of pipe (`RELEASE_MEM`, 64-bit clock source) and waits for it to be nonzero (`NHL27CPREAD:
wait-eop-clock`, 5 to 35 ms each). Bit 13 (`KYTY_CLOCK_WAIT_AHEAD`, default off) lets such a wait
pass on the command processor when the pending clock write is the same queue's and the test holds
for any later clock value. In-match A/B (8 rounds): -8.4% +- 4.8, within noise, so it stays off; in its phases the game kept running normally.

Render resolution, cross-boot benches of 120 s in a match (`nhl27_scale_0p5.json` sets the
game's ResolutionScale to 0.5; the game registers 3840x2160 display buffers):

| Run | Native | Scale 0.5 |
|---|---|---|
| Pair 1 | 29.1 fps, GPU 3D 72-74% | 29.7 fps |
| Pair 2 | 29.6 fps, GPU 3D 72-74% | 33.8 fps, GPU 3D 52-53% |

So pixels are not what holds the frame rate: the GPU stays busy at half resolution and the gain is
small and uneven. `KYTY_OUTPUT_1080P=1` reports a 1080p output to the game (not measured yet).

The resolve and execution threads share one program lock, held for whole resource
materializations (resolve waited on it 14-17% of its time, execution 4-6%). Bit 14
(`KYTY_COMPUTE_UNLOCKED`) releases it while the execution thread materializes a compute program,
bit 16 (`KYTY_AHEAD_UNLOCKED`) while the resolve thread materializes ahead (a per-entry busy flag
keeps the other thread off that program), and bit 17 (`KYTY_BACKING_READ_MRU`) makes backing
reads remember the thread's last two mappings instead of one. Bit 19 (`KYTY_DISPATCH_AHEAD`)
resolves direct dispatches on the resolve thread as draws are. All default off: in-match A/Bs,
-5% for the dispatch read-ahead (6 rounds, +-7.0) and +0.3% for 14+16+17 together (10 rounds,
+-5.4), within the scene-to-scene noise.

Bit 20 (`KYTY_BACKING_LOOKUP_CACHE`, default off): backing reads of another mapping than the
thread's last one look the mapping up in a per-thread cache of 64 slots (one per 64 KiB block)
before the locked map search; it hits 99.7% of the time. In-match A/B (8 rounds): +6.7% +- 4.3, within noise.

Unconditional execution-thread trims (profile-checked, each 1-2% of the thread before):
- The dispatch-args guard reads its trip counter (in video memory the CPU reads uncached) every
  64 indirect dispatches instead of every one.
- Buffer range clamps remember the last 16 committed-run answers per thread while no page
  changed (a 256 MiB range is a thousand bitmap words).
- The CPU fast-clear check compares color metadata eight bytes at a time.

Bit 15 (`KYTY_WIDE_FAULT_WINDOWS`, default on, was the unused `KYTY_ADAPTIVE_RESOURCE_MEMO`): a
CPU write fault releases `KYTY_FAULT_COALESCE_PAGES` pages (16); off, 4 pages. In play the game
takes about 200 write faults per frame and the 16-page windows upload about 11 MB per frame,
mostly unwritten neighbors. In-match A/B (10 rounds), 4-page windows: +2.1% +- 3.4, and in the
steadiest stretch (rounds 5 to 10) 0.6 to 1 fps slower every round, with write faults 14,500 to
33,000 per 2 s. So the wide windows stay.

In-match fps varies by 5 to 7 fps between adjacent 20 s phases (scene changes), so A/Bs of 6 to 10
rounds only resolve effects of about 10% or more.

### Native min/max (bit 11, `KYTY_NATIVE_MINMAX`, default on)

`V_MIN/V_MAX/V_MIN3/V_MAX3/V_MED3_F32` (and the f32 math of the f16 forms) were emulated exactly:
NaN and signed-zero classification of both operands and four selects, about ten host instructions
each, and the bulk of the `v_cndmask` in NHL 27's heavy shaders. With bit 11 they are GLSL
`NMin`/`NMax`, which an AMD host compiles to the same `v_min`/`v_max` the guest ran (differences
only for signalling NaNs and the sign of equal zeros). The option is folded into the NativeFp16
stage-key word, so keys stay as before while it is off; the first boot with it on translates every
shader again. Bit 11 was the unused `KYTY_TEXTURE_DESC_CACHE`, so the play-nhl default mask is
still 1977916646 (with bit 11 off: 1977914598).

Measured in one boot, paused match, GPU timeline captures A (off), B (on), A again (drift +0.2%):
GPU busy per frame 20.9 -> 18.8 ms (-10%), the top 30 draws/dispatches -13% (CS a36f0b4f -30%,
db7e537e -56%, 4c9b2981 -25%, 9380d724 -13%, PS 9eaaf80c -12%, 136eed67 -7%), 34-38 -> 40 fps.
In-match screenshots look the same.

### Robust buffer loads (bit 8, `KYTY_ROBUST_LOADS`, default on)

Every raw storage-buffer load was wrapped in a branch on EXEC and, per dword, a branch on the
bounds (a `buffer_load_dwordx4` became five branches), so the host compiler could neither batch
loads nor hoist them. The device has robustBufferAccess2 (out-of-range reads return zero, as the
guest's do, against the same bound range the explicit check used), so with bit 8 those loads are
plain loads in every lane. Formatted loads keep their checks (their out-of-range value need not be
zero). Shaders that spill guest VGPRs (LDS lane slots or scratch) keep the branches too: the one
such heavy pixel shader (33037c60) got 16-27% slower without them, as more values stayed live.
`KYTY_ROBUST_LOADS_KEEP_EXEC=1` keeps the EXEC branch and drops only the bounds branches.

A/B/A in a paused match (drift -0.2%): top 30 commands -5.0%, GPU busy per frame 16.9 -> 16.1
ms; PS 136eed67 -13%, 9eaaf80c -10%, 90706b3b -18%, 2b39f281 -18%, f4e33d7e -20%, 33037c60 level;
in-match screenshots unchanged. Bit 8 was `KYTY_READBACK_TRANSFER`, which is now env-only (=1);
likewise `KYTY_READBACK_SPIN` (was bit 9). Bit 9 is now `KYTY_FUSED_MAD` (default off): V_MAD/V_MAC_F32
as one fused multiply-add measured -1% (A/B/A), within noise, so the separately rounded multiply
and add stay. Default mask with play-nhl: 1977916902.

Tools for shader-codegen A/Bs: `gpu-ab.sh <prefix> <maskA> <maskB>` (`PAUSE=1` pauses first)
captures A, switches the mask (new variants compile), captures B, switches back, captures A2, and
compares per-shader GPU time with `gpu-compare.py`; GPU time per draw is far steadier than fps.

### Where the pipeline threads wait (`KYTY_DBG_STAGE_STATS=1`)

`NHL27STALLS` (every 5 s): execution-thread busy, idle and program-lock ms/s, and the resolve
thread's lock and full-queue waits; `ab-stalls.py <console>` groups them by mask. In a match the
execution thread is busy about 900 ms/s, idle 90 to 130 (the command processor waits for its
results), and 33 to 60 ms/s of the busy time is spinning for the program lock while the resolve
thread materializes; the resolve thread waits 90 to 300 ms/s for the lock and about 220 ms/s for
room in the execution queue. With bits 14 and 16 together both lock waits drop to about 0, but
in-match fps (8 rounds) and the paused fps did not move beyond noise, so they stay off.

### How a frame is paced

The game's submission thread waits on an event queue for the end-of-pipe interrupt that the
execution thread raises right after it runs the frame's flip, then submits all of the next frame's
command buffers (about 55) at once. A frame therefore takes as long as the execution thread needs
for it, plus its waits for the GPU: the strict resource reads of bytes the GPU wrote in the command
buffer being recorded submit it and wait for all queued GPU work (`NHL27RBWAIT: inline via=other
from=gpu exact=yes lag=0`, now also by shader with `KYTY_DBG_CP_READS=1`). Nearly all of those came
from one small compute shader (45066550b711ac4b, about 17 dispatches per frame): its first block
reads a GPU-written counter and an index table and exits early when the entry is -1; the resource
walk evaluated that branch, so it downloaded the counter (up to 0.7 s per 2 s while a match loads,
2-10% of the execution thread in play).

Bit 13 (`KYTY_GPU_CONDITIONS_BOTH_PATHS`, default on; was `KYTY_CLOCK_WAIT_AHEAD`, now env-only):
while the direct resource walk evaluates a branch condition, strict reads decline GPU-written bytes
instead of waiting; the condition stays undecided and the walk takes both successors (the union of
their resources; the GPU still takes the branch itself). Those waits are gone; a six-boot A/B was
within the cross-boot noise (33.8 vs 32.2 fps, single boots range 28-39).

Crowd shots: with the stands in view a frame has about 4,050 draws (2,360 in normal play: 556
extra tiny draws of PS 1fa12ecb into an R16 1024x256 target, 250 more depth draws, 140 more of PS
f5a3c1b7), and drops to about 18 fps with the GPU half idle: the execution thread is the limit.
`KYTY_DBG_DRAW_COST=1` (`NHL27DRAWCOST`, every 5 s) gives the execution-thread cost per draw by
shader pair: the common draws take 2 to 2.5 us; the draw that carries the CPU-write
synchronization (BDA sync of the pages the CPU wrote since the last one) takes about 1.6 ms.

Device losses still happen at match load or late in a match in about 1 of 6 boots (2 of 6 in one
run, 1 of 4 in another). With `KYTY_DBG_BREADCRUMBS=1` none occurred in 6 boots, so the markers
change the timing; it is not understood yet. `KYTY_DBG_DUMP_SHADERS=<hash>` now also writes the
guest disassembly and the IR (`_Shaders/NNNN_ir_<stage>_<hash>.txt`); a WAIT_REG_MEM that keeps
failing prints `NHL27STUCKWAIT` (the menu hangs: the game then calls Coredump).

Menu hangs (about 1 boot in 6 since 2026-10-14, the game then calls Coredump): `NHL27STUCKWAIT`
showed the graphics queue waiting on a 64-bit label for "not zero" with no pending write left: the
command processor had waited for the end-of-pipe clock write before it, the execution thread had
written the clock and raised the interrupt, the game woke and reset the label to 0, and only then
did the command processor read memory. The execution thread now records the clock values it
writes, and a wait that had to wait for such a write takes the value written.

Bits 14, 16 and 19 are on by default since 2026-10-15 (lock-free materialization on both threads
and dispatch read-ahead): the execution thread's program-lock waits (40 to 125 ms/s, the most in
crowd shots) are gone, 98.6% of dispatches are resolved on the resolve thread, and its checks no
longer fault (`StillValid(true)`). Four 120 s boots, no hang or device loss; the execution thread
idles 130 to 170 ms/s now (it waits for the command processor more often).

Device losses are GPU timeouts (Windows logs LiveKernelEvent 141, VIDEO_ENGINE_TIMEOUT_DETECTED,
at the same second; such events go back to 2026-10-12). `KYTY_DBG_LOOP_CAP=4000000` did not
prevent them. The draws and dispatches recorded are kept in a CPU-side ring with their command
buffer's tick; on a device loss `NHL27HANG` prints the ones in ticks the GPU had not finished.

`KYTY_FORCE_FLIP_RATE=0` keeps 60 Hz flips where the game asks for 30 (menus reach 40+ fps) and
`KYTY_VBLANK_HZ` overrides the vblank rate; the paused match still ran at about 30 fps with both,
because that is what it costs (GPU busy 70%, the execution thread 93%).

Earlier A/Bs of bits 13, 14, 16 and 17 that reported stalls or -25 to -39% were wrong: their base
mask (1979710566) had the retired bits 13 to 20 set, so adding a bit carried into bits up to 25
(NativeFp16, ExecSelects, FaultPageReads off: shader recompiles at 0 fps). Masks are now built by
OR-ing bits into the default mask the boot prints (`Experiments: mask <default> -> ...`,
1977916646 with play-nhl defaults).

Measured and dropped on the way: per-range write ticks (a download waits for the latest write of
its bytes instead of its buffer's): fewer than 1% of the downloads would wait for an older
submission, and the A/B was -14% +- 7.5.

Slow scenes (about 20 fps, some replays and crowd shots) were readback-bound: the execution
thread spent up to half of each second waiting for GPU-to-CPU downloads (`KYTY_DBG_CP_READS=1`,
`NHL27RBWAIT`). Two kinds dominated:

* Command-processor reads of pages that hold GPU-written bytes (NHL 27 embeds labels the GPU
  writes in its command buffers, and keeps register tables and WAIT_REG_MEM values next to
  compute-written bytes). The page is read-protected, so the read faulted and its download waited
  for the writer, about 6-10 ms each while it still ran, though the bytes read were nearly always
  CPU-written ones. Bit 4 (`KYTY_CP_BACKING_READS`, default on; was `KYTY_SRT_JIT`, now env-only):
  on a read-protected page the command processor reads the backing instead (indirect SH/CX/UC
  register tables, WAIT_REG_MEM/decision values, and PM4 packets), and the execution thread checks
  each such read in order: if an earlier operation wrote those bytes it downloads them and compares
  (`NHL27CPSPEC`; a difference turns the reads off). Reading through the execution thread
  synchronously instead cost 0.24 ms per read and stopped the CP running ahead (exec idle 28%).
  In 1.58 million checked reads over six boots: 0.5-1.6% touched GPU-written bytes, all with the same value,
  no mismatch.
* Guest stores into GPU-written pages: once the CP no longer downloaded those pages early, game
  memcpy fills (vmovntdq/vmovdqa) over whole GPU-written pages faulted once per page, each a
  readback (~90-120 us; up to 1,100/s). Bit 6 (`KYTY_WIDE_WRITE_DOWNLOADS`, default on; was
  `KYTY_FALSE_SHARE_READS`, now env-only): such a store downloads the GPU-written bytes of its
  64 KiB window at once (the window a CPU write fault releases), down to about 50-70/s.

Three in-boot A/B boots (`abwin.sh`, 12 s phases alternating bits 4+6 on/off): readback waits
86/74/71 ms/s on vs 98/133/277 off, execution thread busy 4-5% lower; with the PM4 fix as well no
CP fault remained (normal play ~15 ms/s of readback waits in total, slow windows ~30). The
`NHL27OWNW` diagnostic lists the GPU-written ranges of a page a guest store hits;
`NHL27UNMAP` counts unmaps that drained the GPU and why; `NHL27COREDUMP` prints the guest thread
and return addresses when the game calls Coredump (its `Heartbeat` watchdog does when a boot
stalls, for example while every shader recompiles).

Upload copies (CPU-written buffer pages into the staging buffer before their GPU copy, 3.3% of the
execution thread) now go to the host copy thread like the stream buffer's; the submission waits for
them and guest-visible operations drain them first. In-boot A/B against `AbLegacy` (bit 31, the old
copies): execution-thread time per draw 5.72 -> 5.46 us, per dispatch 4.45 -> 4.26 us.

A GPU A/B of the stronger EXEC-select rules (`KYTY_DBG_EXECSEL_MODE=31`) on the heaviest compute
shader alone (c556aaf3, 3.1 ms per frame in a paused scene of 26.8 ms GPU per frame) removed 901 of
1,506 selects for -8% of its time: not worth the rules' known breakage. It uses 154 VGPRs (wave64).

Device losses at match load (TDR, Windows LiveKernelEvent 141): three of five boots on 2026-10-16
lost the device at the same point of the load (around tick 500,000, the failing submit showing
args=2,1,1,97,0x1140100a00); none did in five boots with `KYTY_READBACK_QUEUE=0`. The second-queue
readback (on the compute family since 2026-10-14, when the losses started) made that queue wait
for the writer's submission on the GPU. Now the CPU waits for the writer first, so the queue's
wait (kept for its memory dependency) is already satisfied when it is submitted: 6 of 6 boots
without a loss.

Two more execution-thread fixes: draw checks of resolved-ahead draws (`GuestReadLog::StillValid`)
compare read-protected pages in the backing instead of taking an exception per plain load (the
16-page fault cache thrashed; 2.7% of the thread), and a pending asynchronous guest read marks
only the GPU-written bytes it carries dirty, not its whole pages (a draw check that hit such a page
waited for that download: another 2.7%).

Indirect dispatches are resolved ahead on the resolve thread too (their program does not depend
on the group counts; with bit 19 `KYTY_DISPATCH_AHEAD`): execution-thread time per indirect dispatch
24.9 -> 11.5 us in an in-boot A/B (about 2.5% of the thread).

Breadcrumbs (`KYTY_DBG_BREADCRUMBS=1`) caught the remaining device loss in a soak: the GPU was
running an indirect dispatch of CS 9ba58a112a994ad8 with 50,589,920 x 1 x 1 groups (the record at
0x11c020a000 grows over a match; CS 8714aa79 had just run with the same record). The counts are
within the device limits, so the guard let them through and the GPU ran for seconds (TDR). The
guard now also zeroes records whose product exceeds 4M groups (NHL 27's normal indirect dispatches
stay below 1,000; a full-screen 4K pass of 8x8 groups is about 130,000). The record itself is
still wrong (not reset where the game expects it); not found yet.

Scalar constant reads (S_BUFFER_LOAD) under robust loads no longer branch on their bounds per
dword (a branch per constant kept each load in a block of its own; the heaviest compute shader had
189): GPU time -6 to -7% per frame in a paused-scene A/B/A (29.0 -> 27.0 ms; d7cfa519 -17%,
others -5 to -15%). Part of the shader keys (code-gen key bit 32).

With the readback waits gone, the pipeline is limited by two threads at once: the resolve thread
(resource walks of the draws and dispatches ahead) is about 88% busy and the execution thread about
80%; the command processor mostly waits for free register snapshots. `NHL27STALLS` now also prints
the resolve thread's idle and busy time. Bits 17 (`KYTY_BACKING_READ_MRU`) and 20
(`KYTY_BACKING_LOOKUP_CACHE`), whose earlier A/Bs were invalid (the mask carry bug), are on by
default: resolve-thread time per operation -7.7% in an in-boot A/B (soak: 6 of 6 boots, 2 min in
the match). Deeper queues between the stages (resolved queue 256 -> 768, rings 320 -> 960, register
snapshots 512 -> 1536) cut execution-thread idle time 181 -> 156 ms/s but not the operations per
second (in-boot A/B): dropped.

Storage-buffer stores (32-bit, and plain 8/16-bit ones) under robust loads drop their bounds branch
too (the EXEC branch stays; a read-modify-write of a containing dword keeps it, so an out-of-range
compare-exchange cannot spin): GPU time -3.1 / -3.7% per frame in a paused A/B/A (pixel shaders
with buffer stores -5 to -9%). Code-gen key bit 64.

The resolve thread read resource-table words one dword per call (its SRT runtime never set
`batch_reads`, which the execution thread's has on by default): every dword went through five
calls down to a memcpy, and each switch between the two threads on one program rebuilt the
program's read runs. It now reads contiguous words in one call as well: resolve-thread busy time
-32% per operation in an in-boot A/B (2.8 -> 1.9 us; 740 -> 475 ms/s), so the execution thread
is the limit again (about 95% busy in heavy play). Small backing reads (4/8/16/32 bytes) are
copied inline. A resolve-side memo of whole materializations (four per program, validated by
re-reading) hit only 10% of the time and cost more than it saved: dropped.

The per-draw proof that no scalar resource read overlaps a write of the same draw (it exits the
emulator when one does; it never fired in NHL 27) is opt-in now, `KYTY_DBG_CHECK_SCALAR_READS=1`:
about 1% of the execution thread.

Batch boundaries (`KYTY_BDA_BATCH`, after every indirect buffer, wait and event packet) are a
third of the execution thread's operations, but cheap: carrying them on the next operation instead
took ~110k callbacks/s off and changed nothing per draw (paused A/B: 4.70 vs 4.72 us per draw;
fps 40 vs 41): dropped. In a paused scene the callbacks cost about 0.1 us each.

The execution thread's checks of resolved reads look up the GPU-dirty state (buffer ranges and
GPU-modified images) once per page instead of per read: execution time per dispatch -14%, draws
per second +1.5% in a paused, execution-bound A/B (52 fps).

Page protection churn (in play: ~45k protect calls/s, ~11k write faults/s; 87% of the execution
thread's protect calls came from the BDA synchronization uploading every CPU-dirty page):
- Write epochs (`RegionManager`): the command processor advances an epoch at each batch boundary
  and stamps it on the operations it emits; a page that becomes CPU dirty takes the epoch of that
  moment (0 when the execution or command processor thread writes it). The BDA synchronization of
  an operation uploads (and protects again) only the pages of earlier epochs: a game writes a
  submission's data before submitting it, and data the GPU waits for before the label. Pages the
  game is still filling stay writable until an operation of a later epoch needs them. Protect
  calls and write faults -20% (in-play A/B).
- A write fault releases its 16-page window before the faulting page instead of after, so the
  window, faulting page included, is one protection change instead of three: protect calls -41%,
  those of the faulting (game) threads -56%.
Together about 89k -> 30k protect calls per 2 s. Execution time per draw did not change
measurably. `NHL27FAULTPAGES` (with `KYTY_DBG_CP_READS`) counts write faults per page every 2 s:
about 1,800 pages fault 11k times, two thirds of them on ~470 pages rewritten every 2-8 frames.

`V_CVT_PKRTZ_F16_F32` (pack two floats as halves, rounding toward zero) was emulated with about
forty integer instructions per half; on AMD devices it is now GLSL `PackHalf2x16`, which the AMD
compiler turns into the same `v_cvt_pk_rtz_f16_f32` (seen in the pipeline ISA, also in shaders that
declare RTE for f16). The two vertex shaders drawn with PS f5a3c1b7 (crowd and players) shrink from
46k to 22k and 24k to 12k ISA instructions; GPU time per frame -5% in a paused A/B/A (code-gen key
bit 128). Native scalar f16 ops also take SDWA word-select sources (480 `V_SUB_F16` in each of those
shaders; key bit 256): GPU time unchanged, but they round like the guest.

The execution thread checks every guest read the resolve thread made for a draw or dispatch
(`GuestReadLog::StillValid`): ~37 reads, ~590 bytes on ~17 pages per draw. Skipping every check
(measurement only) cut execution time per draw 12% and per dispatch 26%; the cost is touching that
memory, not the bookkeeping (remembering 8 pages instead of 1 changed nothing). Now a read needs no
check when the resolve thread made it after the execution thread had finished operation `base`,
no later operation writes its page from the emulator's side, and its bytes were not GPU-owned
then (`PageWriteSeq`: per 4 KiB page, the sequence of the last operation that writes it; raised by
the command processor's noted writes, by GPU-written buffer ranges, and by every backing write;
writes from other threads mark the page as always checked; mapping changes force full checks).
The game does not rewrite memory that submitted GPU work still reads before that work signals
completion. Specialization (strict) reads are always checked. `KYTY_DBG_VERIFY_SKIPS=1` still
checks the skipped reads and reports any that changed (`NHL27SKIPBAD`); none in 3 soak boots and
two 2-minute matches. `NHL27CHECKS` (stage stats) prints reads, bytes, page switches, skips and why
the others were checked. The resolve thread looks up the GPU-dirty state of each read's page
(`HasGpuDirtyBytesShared`, 8 pages remembered per log). That query used to call every range dirty
while any asynchronous guest download was pending (most of the time): the pending downloads' pages
are now a shared range set (`m_pending_read_ranges`, filled before their bytes leave the GPU-dirty
ranges, emptied once they are in guest memory). Draw reads skipped: 19 of 35 -> 34 of 36;
dispatch reads 7 of 15 -> 8.5 of 16. Execution time per draw -6.4%, per dispatch -4% in total
(paused A/B; skipping every check would be -12% / -26%). Writes of more than 16 pages (every bind
of a GPU-written buffer) are recorded per 1 MiB chunk: per page they were 2% of the thread.

Crowd shots are GPU-bound (~30-37 ms GPU per frame paused): ~500 draws per frame use the crowd
vertex shaders (0d41096a, 21d8131e), in depth, shadow and main passes. After native PKRTZ they still
wrapped ~400 formatted 16-bit vertex loads in bounds branches (formatted loads were excluded from
robust loads). Now formatted loads under robust loads read unconditionally (an out-of-range element
reads zeros on the device) and select the format's out-of-bounds value (key bit 1024), and robust
loads/stores/constant reads also apply to non-pixel shaders that spill guest VGPRs (key bit 512;
pixel shaders keep their branches, one got slower without them). Crowd VS 21k -> 18.7k
instructions, 408 -> 5 branches; paused crowd scene A/B/A: crowd draws -9.6%, d7cfa519 -3%, GPU
time per frame -4% (36.9 -> 34.8-36.3 ms).

EXEC as branches (experiment bit 39 `KYTY_EXEC_BRANCHES`, default on; code-gen key bit 2048): every
guest VGPR write under a partial EXEC selects the new value per lane, and the heavy compute
shaders are mostly such code (d7cfa519: 2,900 `v_cndmask` among 11,600 host instructions, against
112 `V_CNDMASK` in the guest's 4,600). The emitter now puts runs of a block's lane-wise
instructions under one EXEC value inside `if (exec) { ... }`: their selects (and V_CMP results
`exec && cmp`) are their new values inside, and those used after the run become phis of the new
and the old value, so the host compiler masks the writes as the guest's hardware does. Pure
instructions of the run that are needed outside it and depend on nothing in it are emitted
before the branch; a run ends at memory operations, cross-lane operations and anything with
side effects; loop headers and continue targets, and wave64 emulation, are left alone.
`KYTY_DBG_EXEC_BRANCH=1` prints runs, branch instructions, phis and hoisted instructions per
shader (`NHL27EXECBR`), `KYTY_DBG_EXEC_BRANCH_MIN` the selects a run needs (default 6). Paused
A/B/A in two scenes: top shaders -8% and -6.4% (c556aaf3 -17%, a36f0b4f -16%, 9380d724 -19%,
33037c60 -8%), GPU time per frame -4%; the final image differs from the old build's as much as two
consecutive frames of the old build differ (TAA), and looks the same. Gameplay 49.8 -> 51-52 fps
(two fixed-profile matches). Bit 40 `KYTY_EXEC_BRANCH_MEMORY` (default on; key bit 4096): runs also
take the memory operations predicated by their EXEC (or by `exec && cmp` inside them), in order;
inside the branch their EXEC operand is the constant true. Paused A/B/A: top shaders -4.9% more
(drift -1.6%; PS 9eaaf80c, 136eed67, 90706b3b -4.5 to -5.4%, c556aaf3 -5.9%), GPU time per
frame 23.0 -> 21.9 ms; image unchanged.
`KYTY_DBG_FRAME_DUMP_FROM=<seq>` keeps a frame dump to the last passes (the readback cap skipped
the final images), and a file named `framedump.trigger` in the working directory requests a dump
(KytyPS5-tools `gpu-abd.sh` takes one per phase, `dump-diff.ps1`/`img-diff.ps1` compare them).

The low-energy preset patch (`patches/nhl27_lowenergy.json`: resolution scale 0.5, no FSR3)
changes nothing on the CPU side (same draws per second and thread load).

### Comparable runs and fps by flip rate (2026-10-19)

Match-to-match fps varied 33-43 with the same build. Two causes: the game saves its profile
every boot (Play Now remembers its setup), and it switches its flip rate itself: 60 Hz in play,
30 Hz in menus, faceoff intros and replays (a quarter to a third of a 3-minute window). Now:

* `KytyPS5-tools\play-nhl.ps1` restores a profile snapshot (`save-fixed`) before each boot
  (`-KeepSave` skips it). Two boots then read the same assets (`gameio-diff.py` on consoles with
  `KYTY_DBG_TRACE_GAMEFILES=1`: 291 of 295 reads common).
* The emulator always prints `NHL27FPS` (every 5 s: flips so far and the rate the game asks for)
  and `NHL27FLIPRATE` changes with flip counts; `fps-by-rate.py <console> [--last s]` splits a
  run's fps by rate. Baseline (43eba5e4): 45-47 fps at 60 Hz, 27-28 fps in the 30 Hz scenes
  (those are GPU-bound on the crowd), 41 overall.
* `NHL27CALLBACKS` (stage stats) splits the execution thread's callback time by what the
  callbacks do (BDA boundaries, flushes, flips, flip waits, EOP, events, barriers...). In play
  they are about 3% of the thread; the draws (about 2,900 per frame) are three quarters.

In heavy play both pipeline threads are near their limit: the execution thread about 95% busy,
the resolve thread about 84%. The execution thread's time is spread thin: buffer binds and their
synchronization about 20% (re-protecting pages after BDA uploads alone 4-5%), texture bindings
12%, read checks 6-8%, descriptor commits 6%, compute dispatches 14%. Cuts so far (paused A/B
against `AbPending`, experiment bit 38, A/B only): draw cost -6%, dispatch -3% (fps 43.5 -> 46):
texture slots whose T# is unchanged keep their description; descriptions are copied once per
recording (`Found::serial`); no page walk for texel buffer binds when no image starts at the
address; the GPU-fill filter is tested a word at a time; read checks look up `PageWriteSeq` once
per page run and skip clean page runs whole; the resolve thread's GPU-dirty page answers are
cached per thread while the GPU-modified and pending-download sets keep their version.
Measured and dropped: a memo of written buffer binds (skip the synchronization while no page of
the range's tracking regions became CPU dirty or GPU-clean): no gain, the 4 MiB region epochs
change too often. Protecting all of a BDA synchronization's ranges with one update per tracking
region (merging runs across already protected pages): the execution thread's protect calls -23%
but protected bytes +57%, draw cost unchanged: dropped. On the resolve thread, the permutation
the last resolution of a program took is tried first, and read-watch answers are cached per
thread while no page gains or loses its read watcher: resolve time per operation -7% (paused
A/B, with the GPU-dirty page cache).

Buffer bindings (`NHL27BINDS`, stage stats): about 1.45 million per second (29,000 per frame, 7.5
per draw or dispatch), 53% of them the range the slot had the draw before, 4.6% written. Nearly
all ran a full synchronization (`NHL27BUFFERS2` with `KYTY_PERF_STATS_VERBOSE=1`: 21,500 of
23,000 per frame) that found nothing to upload (under 5 uploads per frame): the memo of read
bindings compared the CPU-dirty epoch of the whole 4 MiB tracking region, which moves whenever the
game writes anywhere in it. Regions now keep an epoch per 64 KiB block as well, advanced only when
one of its pages becomes CPU dirty, and ranges covering a region partly sum their blocks'
epochs: draw cost -7.6%, paused fps 50 -> 53. The memo also takes formatted (texel) reads while
no image starts at the address (only such an image is copied into the buffer): draw cost -10.5%
and dispatch -6% in total, paused fps 55-57 -> 59-62; a clean match 54.8 fps at 60 Hz (steady
54-56). A second resolve thread for dispatches (bit 41 `KYTY_DISPATCH_RESOLVER`, own core) took
20% off the resolve thread but changed no fps while the execution thread was the limit.

A gameplay profile of the execution thread after that (`samples\pr18-execution.txt`): buffer
binds 17% (the texture cache's "image at this address?" check on every texel bind 3.6% of it),
re-protecting pages after BDA uploads 5.8%, description lookups of T#s 3% (a stage alternates
between programs, and the slot memo kept only the stage's last program), building the pipeline
key 2% (member-by-member initialization of a 720-byte key). Now the read memo keeps the image
registration version its "no image here" answer was found at, image memos are kept per program,
and the key is copied from a default one: draw cost -2.6% (paused A/B, 4.29 -> 4.18 us). The
earlier A/B switches of the binding memo work are gone; `AbPending` selects only new changes.
Measured and dropped (paused A/B, -0.5% together): push-descriptor words compared with the last
push in place (instead of building a key vector), and texture bindings kept per program across
stage switches (fewer description copies).

With PGO v6 (`samples\pr19-execution.txt`): re-protecting pages after BDA uploads is 8.4% of
the execution thread (NtProtectVirtualMemory about 7.5 us per call, about 190 calls per frame, 10
BDA syncs per frame), and the execution thread idles 80 ms/s waiting for the resolve thread in
a paused scene: the two threads are now about balanced. In the play captures of the low-overhead
trace, the execution thread waits about 6% of the time in heavy frames (48-50 fps), all of it in
some 10,000 waits per second of a few microseconds each (it caught up with the resolve thread);
in a light stretch (58 fps) a third of the frames also wait 2-10 ms after their flip for the game's
next frame (vsync pacing, not a bottleneck). A second resolve thread for dispatches (bit 41)
took the resolve thread from 834 to 679 ms/s busy but left the execution thread's waits for it
nearly unchanged (94 -> 84 ms/s, paused A/B) and the throughput the same: still off. Deeper
queues again (resolved queue 256 -> 768, resolution rings 320 -> 800, register snapshots 512 ->
1280; cross-build A/B, two paused boots each): draw cost unchanged (4.08 / 4.06 us), dispatch
cost +8%, resolve thread +12% busy for the same work (its rings no longer stay in cache): dropped.

Bit 43 `KYTY_DRAW_RESOLVER` (second draw resolver on CPU 14, its own resolution ring): the
resolve thread hands about half of the draws to it, chosen by the vertex and pixel shader
addresses so the draws of a program pair stay on one thread; a program the other resolver is
materializing is waited for instead of leaving the draw to the execution thread. The execution
thread waits for slots in order, as for the dispatch resolver. Paused A/B (a 60 fps scene): the
execution thread's waits for the resolve thread 84 -> 15 ms/s (the scene reached the 60 Hz cap
and waited for flips instead). In play (unpaused in-boot A/B, 60 Hz windows): waits 89 -> 30 ms/s,
fps 48.8 -> 49.6, but per-draw execution cost +5% (4.58 -> 4.82 us). Two changes since: each
of the two resolution rings holds half as many resolutions while the second resolver is on (as
many in total as one ring before), and a draw state's vertex input info (9.6 KB, almost all of it
arrays filled from the front and read up to their counts) is reset only up to the entries its
last use filled instead of value-initialized whole (1.5 GB/s of writes; resolve thread -9% busy
in play). Now default on: in-boot gameplay A/B, 15 s phases, the window-title fps of paired
phases 55/52, 54/52, 55/52 (+4 to +6%), waits for the resolve thread 110 -> 38 ms/s. Back to
default off: with it on, 2 of 8 boots crashed in guest job threads (a UI table holding -1 right
after the match start; a pointer read from memory the emulator had written 1 ms before, during
loading), and none of the boots without it today. Fixed-save matches, 60 Hz / 30 Hz fps: off
54.2/25.2 and 49.6/25.3, on 59.3/28.0 (the second boot with it on crashed while loading).
Three more boots with it on and `KYTY_DBG_VERIFY_SKIPS=1` (skipped read checks compared anyway,
`NHL27SKIPBAD`): no crash, no mismatch. Both crashes read guest memory the emulator had written
back from the GPU just before (`--- Host writes near guest pointers ---`), so the suspicion is a
download / CPU-write race that the faster pipeline exposes more often, not the resolutions.

Deferred protection (`PageManager::DeferredProtectionScope`): while the execution thread
synchronizes CPU-dirty buffers for BDA, pages that gain write watchers are marked stale in their
region and protected by the host copy thread (an apply call queued before the uploads' copies,
which go through that thread too while it is on); losing watchers (fault handling) stays
immediate, and the apply protects each stale page as its watchers want at that time. In-boot
gameplay A/B (`AbPending` = immediate): execution-thread protect calls 18,330 -> 4,770 per 2 s
(all calls about the same, write faults unchanged), per-draw cost 4.55 -> 4.35 us. Soak: 2 of 3
boots fine; the third hit the guest crash at pc 0x9093c3f41 (job thread, a UI table slot holding
-1 when the game switches to a 30 Hz screen after play), which also crashed boots on 2026-10-14
(74b7740) and 2026-10-15 (6138c3b) and the first draw-resolver boot today: a rare existing crash,
not new with these changes. The draw resolver's other crash (pc 0x909568416, job thread, during
loading) is the only one of its kind so far. Four more boots with the draw resolver on: no crash,
but 60 Hz fps 46.6 / 50.8 / 45.7 / 51.4 against 52.2 / 50.7 for two boots of the same build
without it (fixed save, 240 s each): no gain in whole matches despite the in-boot A/Bs; still off.
A later in-boot A/B (20 s phases) had it behind (paired rounds 42/47, 42/48, 44/40): in much of
play the execution thread waits 70-220 ms/s for the game's next submission (`no-submission`),
and the extra resolver costs the execution thread +4% per draw.

Measured and dropped (in-boot gameplay A/Bs, 60 Hz windows): pipeline threads sleeping after
200 us instead of 2 ms (54.1 / 54.0 fps); presenting a flip that missed its vblank as soon as it
is ready (50.9 / 50.6 fps: flip waits are small, frames are not quantized by vsync); the host
copy thread on the command processor's SMT sibling (two matches 52.5 / 55.6, within noise).

Image filter (`TextureCache::HasImageAt`, `FindImageFromRange`): texel buffer binds ask whether
an image starts at the bound address on nearly every draw and almost never find one; a bit per
hash of the registered images' start addresses (counts kept under the cache lock) answers no
without the lock and the map lookup. In-boot gameplay A/B (with deferred protection): 48.7 ->
51.2 fps at 60 Hz, draw 4.34 -> 3.93 us.

No host-copy drain before dispatches: the execution thread waited for the host copy thread
before every operation that is not a draw (7.6% of the thread), for guest-visible effects
(labels, interrupts, flips, host commands) that let the guest overwrite what earlier draws read.
A dispatch has none: its submission waits for the copies before it, and only with
`CpuCopyForward` (default off) does a dispatch write guest memory. In-boot gameplay A/B: 48.0 ->
49.0 fps at 60 Hz (paired rounds 48/50, 48/49, 45/52, 45/45, 48/51).

Measured and dropped: guest threads kept off the cores of the GPU stage threads (host affinity
of all 41 guest threads switched in boot; Petr's PC has a Ryzen 7 5800X3D, 8 cores, so the five
stage threads on logical CPUs 2-10 leave three cores and the SMT siblings to the game). Off the
stage cores and their siblings (mask 0xf22b): draw 4.13 -> 3.91 us, dispatch 3.23 -> 2.79 us,
but 51.3 -> 47.9 fps; only off the siblings of the execution and resolve threads (0xf77f): draw
4.01 -> 3.73 us, dispatch 3.27 -> 2.68 us, 52.4 -> 50.2 fps, the execution thread waiting for
the game's next submission 34 -> 104 ms/s. The game's job threads are each ~28% busy (5% of it
in the emulator: process time counter, exceptions) and want many cores at once; the SMT
siblings cost the stage threads 7-18% per operation.
Measured and dropped: the command processor on the execution thread's SMT sibling and the
host copy thread on the resolve thread's (re-pinned in boot, `KytyPS5-tools\pin.py`, `PIN_A` /
`PIN_B` in `abwin.sh`), freeing two cores for the game: draw 4.00 -> 4.55 us, 50.7 -> 46.8 fps.
The second draw resolver on the resolve thread's SMT sibling (`KYTY_GPU_THREAD_CPUS=4,6,8,10,2,12,11`,
10 rounds): waits for the resolve thread 72 -> 59 ms/s in heavy play, but the game's submissions
later (34 -> 64 ms/s), 49.3 -> 48.4 fps: the CPU it takes is the game's. The command processor
sleeping on a full register snapshot ring (61% of its time is spent waiting for a free slot,
spinning) until an eighth of the ring is free: the execution thread then waited longer for work
(no-submission 56 -> 93 ms/s), 51.2 -> 50.8 fps.
Command buffer flushes (`NHL27EXECIDLE ... flushes/s`, by origin): ~5,300 a second in play,
nearly all flush callbacks (the submission split every 48 draws with `AsyncSubmit`, the end of
each graphics submission, end-of-pipe interrupts and compute submissions); label-only releases
never reach their 64-item batch. Measured and dropped: interrupt and compute-submission flushes
batched like labels (flushes -10%; 47.4 -> 49.3 fps in one A/B, inside the noise), and with it
no host-copy drain before flush, barrier and BDA callbacks (51.1 -> 45.7 fps).

BDA boundaries (`CommandProcessor::BdaBoundary`, after waits, events, acquires, DMA, indirect
buffers and each submission slice) only make the next draw or dispatch synchronize again; a
boundary with no draw or dispatch emitted since the previous one is no longer an operation
(statistics builds still emit them all). In play: 599k -> 531k operations a second, the
execution thread's waits for the resolve thread 63 -> 49 ms/s, 60.6 -> 61.3 fps (in-boot A/B,
240 Hz display, 60 Hz windows; paired rounds even).

BDA synchronization after a mapping change walked every mapped range (`RangeSet::ForEach`, 3% of
the execution thread in play: the game maps and unmaps many times a frame). Drained dirty
regions outside the mapped ranges are dropped, so only memory mapped since the last
synchronization can hold CPU-dirty pages that no drain brings back; unmaps leave nothing to
upload. The scan now covers just that memory (`RenderContext::m_bda_new_ranges`; AbPending: the
old walk). Sampled execution thread: `PrepareBda` 4.3% -> 2.2%, all of it now the queued dirty
regions; two in-boot A/Bs within their noise (draw cost -3% and +0.4%).

Buffer bindings were 17% of the execution thread in play (`NHL27BUFFERS2`, `NHL27BINDS` with
`KYTY_PERF_STATS_VERBOSE=1` and `KYTY_DBG_STAGE_STATS=1`): about 9,000 buffer lookups a frame, of
which 43% of the read-only ones missed the obtain cache and 4,300 reached `SynchronizeBuffer`,
which found something to upload 4 times a frame. The obtain cache indexed its 512 slots by the low
bits of a multiplicative hash, which only depend on the low bits of the address: 256-byte aligned
ranges used one slot in sixteen. It now takes the high bits, over 2,048 slots. Written storage
buffers also no longer ask the texture cache to invalidate images starting at their address
when the start-address filter has none. In-boot A/B: draw cost 4.68 -> 4.44 us (-5%), dispatch
3.54 -> 3.40 us, execution thread 915 -> 874 ms/s busy.
The read memo of `ObtainBuffer` (lookups that miss the obtain cache, and texel reads) also hits
when its CPU-dirty epoch moved but no page of the range is CPU dirty: the epochs count writes
anywhere in a 64 KiB block, so the memo missed and `SynchronizeBuffer` found nothing to upload.
In-boot A/Bs: draw cost 4.71 -> 4.47 us and 5.04 -> 4.96 us, dispatch 3.42 -> 3.34 us. One of
these boots crashed in a job thread while loading the match, at the same guest pc as one
earlier draw-resolver boot (0x909568416, a null pointer read from a page the emulator had
written back from the GPU 10 ms before): the rare existing crash, not the memo (whose hits are
exactly the ranges where `SynchronizeBuffer` uploads nothing).

Unmaps (about 190 a second in play) drained the whole GPU whenever a deferred completion that
may write guest memory was pending, 3-6 times a second for 4-6 ms each (`NHL27UNMAP` with
`KYTY_DBG_CP_READS=1`; `KytyPS5-tools\unmap-by-mask.py` splits it by A/B phase): 20-36 ms a
second of the execution thread. Most of them waited for the write scan of the indirect write
tables, which was tracked as writing anywhere so that a remap would not inherit its GPU-written
marks. Unmaps are now numbered and the scan skips pages unmapped since it was queued
(`RenderContext::UnmappedSince`), so it no longer counts as a write. In-boot A/B (together with
a wait only up to the command buffer of the last completion writing the range, since reverted):
unmap waits 35.7 -> 13.7 ms/s, draw cost 4.60 -> 4.42 us, execution thread 904 -> 874 ms/s busy.
The waits left are downloads into the unmapped range. The partial wait went back to the whole
drain after a boot ended with a download published into memory unmapped before it
(`Memory: required direct-backing write failed`, the first such exit; second draw resolver on).

Host-copy drains before callbacks: every callback waited for the host copy thread (4.5% of the
execution thread, spinning), although only guest-visible effects need the copies done first. BDA
boundaries, global barriers and cache events (two thirds of the 155,000 callbacks a second in
play) no longer drain; the occlusion counter dump, the only event that writes guest memory, is
tagged `Other` and still does. In-boot A/B: draw cost 4.80 -> 4.62 us, 50.2 -> 51.8 fps at
60 Hz.

The second draw resolver (bit 43) again, on this build: the execution thread's waits for the
resolve thread 79 -> 25 ms/s, its waits for the game's next submission 16 -> 82 ms/s, draw cost
+2%; paired 20 s rounds 476 / 452 fps in sum (+5%), but whole-phase 60 Hz fps 50.7 / 52.0. The CPU
it takes is the game's (8 cores here, fewer on lower-end PCs): still off.

Image bindings per program: a stage's draws alternate between programs, and its one set of
`TextureBinding`s took the description of whichever program drew last, so `ReuseTextureLookup`
copied a description (a few hundred bytes) for most slots. Each program's image memo now keeps
its own bindings, swapped in with the program (`ProgramImageMemo::bindings`): in-boot A/B, draw
cost 4.73 -> 4.35 us (-8%), execution thread 891 -> 876 ms/s busy.

Bit 4 `KYTY_CP_BACKING_READS` is now off by default: the command processor read WAIT_REG_MEM
values and packets on read-protected pages from the backing, checked later in order by the
execution thread. A boot hung at the match start right after that check reported a mismatch at
the very counter the CP then waited on forever (`NHL27CPSPEC: MISMATCH`, `NHL27STUCKWAIT` with the
value one short); the three earlier hangs of that kind left no such line. Its cost, in one boot
with both variants in alternating 15 s phases (`KytyPS5-toolsbmulti.sh`): none (paired rounds
287 / 301 fps in sum with it on / off). `KYTY_LAZY_SHARED_WRITES` off in the same boot: 248, so it
stays on despite its documented gaps. New default mask 1651247152614; A/B-only bits 44 and 45
(`AbPending2`, `AbPending3`) let one boot compare several pending changes.

Test boots reach the match about 75 s sooner: `play-nhl.ps1` polls the boot every 2 s, waits
half as long on the menus and detects the match start in the console instead of sleeping.

Two smaller cuts, compared in one boot (base, and each one's old code, 7 rounds of 15 s): deleting
an image no longer scans every image for stencil planes associated with it unless it is a depth
image that has one (1% of the execution thread: images are replaced all the time); the texel read
memo is versioned by the registrations hashing to its address's filter bit
(`TextureCache::AddressBucketVersion`) instead of every registration. Rounds 2-7 in sum: 295 fps,
290 with the old memo, 281 with the old scan; draw cost 4.17 / 4.30 / 4.21 us.

Asynchronous guest reads (`KYTY_ASYNC_GUEST_READS`, on in the recommended set) published their
download with a plain write of the GPU's bytes. Three of today's guest crashes read objects in pages
that such a publish (the priority thread) had written 1-10 ms before, and the match-start hangs
waited on counters one short in the pages NHL 27 builds command buffers in (0x1131xxxxxx). The
publish now merges like the lazy shared writes: a byte that changed while the download was pending
keeps its new value. `NHL27ASYNCKEEP` reports such cases; in four boots one came up, 8 bytes in
0x1131274000 (the page of an earlier hang's counter) that the old publish would have overwritten.

A later boot still hung in the menus, on two counters one short in a page where two such merges
had just kept changed bytes. The cause: label writes (`TryWriteLabel`, EOP and WRITE_DATA labels)
update guest memory and record a copy into the label's buffer, but did not count as a write of the
buffer, so a synchronous readback on the second queue (`KYTY_READBACK_QUEUE`), which waits only
for the buffer's last writer, copied the buffer before the label's copy ran and published the old
value over the label. `WriteDataBuffer` now sets the buffer's `last_write_tick`. The following
boots: no hang. Also measured: no host-copy drain before flush callbacks (the recording thread
waits for the copies before it submits): 359 / 377 fps in paired rounds, dropped. The second draw
resolver once more in one boot: 345 / 360 (with it, the execution thread is 97% busy): it stays off.

Plain-read buffer slots keep what they bound (`PreparedBindings::buffer_memos`): a slot whose range
is the one it bound the draw before reuses the descriptor while no buffer was registered and the
range stayed CPU-clean (`BufferCache::PlainReadHolds`, the obtain cache's own test, without its
hash lookup); stream-buffer copies are never kept. One boot, 8 paired rounds: 374 / 365 fps in sum
(7 of 8 rounds ahead), draw cost 4.64 -> 4.40 us.
The texel memo's image versions now come from a 4,096-entry table (groups of 16 filter bits)
instead of one version per filter bit (256 KiB, a cache miss on most texel binds: 1.2% of the
execution thread): one boot, draw cost 4.75 -> 4.37 us, rounds 1-6 ahead by 1-4 fps each. A slot
memo for texel reads as well (on top of the per-address memo in `ObtainBuffer`) gained nothing:
388 / 406 fps in sum, dropped.
Written slots keep their binding as well while the GPU ranges version (`m_gpu_ranges_version`:
every download, lazy write and new range moves it) and the buffer set are unchanged: the range is
then still GPU-owned with nothing to upload, and `BufferCache::WrittenBindHolds` does only the
per-operation bookkeeping of a GPU write (fills dropped, last write tick, write generations,
`PageWriteSeq`). One boot: draw cost 4.65 -> 4.51 us, paired rounds 404 / 408 (within noise).
The version changes too often for that (new written ranges, downloads): a written slot now also
holds while its own range is still wholly in the GPU-written ranges and not pending a guest read
(its pages are then GPU-owned with nothing to upload). One boot: 417 / 371 fps in paired rounds
(6 of 8 ahead), draw cost 4.72 -> 4.54 us. `MaterializeColorClear` asks whether its metadata
slices hold GPU-written bytes once instead of up to three times.
With the small version table, the texel slot memo pays off after all (texel reads whose address
has no image keep their binding while the image versions of that address group, the buffer set and
the range's CPU state are unchanged): one boot with three variants, draw cost 4.76 -> 4.46 us,
paired rounds 378 / 370. Reusing identical flattened-SRT and shader-data uploads within a lap of
the stream buffer: draw cost -2% but 378 / 407 fps in the same boot, dropped.

Readback waits of the execution thread (`NHL27RBWAIT`; `KytyPS5-tools
bwait-by-mask.py`) were
65-92 ms/s in play with bit 4 off, nearly all synchronous downloads for command-processor read
faults on bytes the GPU never wrote (pages NHL 27 shares between command buffers and the 16
bytes a compute shader writes). Bit 34 `KYTY_FALSE_SHARE_ANY_THREAD` is now on by default: such a
read on the command processor or a guest thread is emulated from the backing when its bytes are
not GPU-written (exact, unlike the speculative bit 4). One boot with off / bit 4 / bit 34 in
alternating 20 s phases: readback waits 65 / 12 / 24 ms/s, paired rounds 361 / 390 / 382 fps.
Bit 4 in two more boots: 374 / 394 and 464 / 446 fps against off. New default mask 1668427021798.
Two more boots, each with three variants: bit 1 `KYTY_SKIP_REDUNDANT_BINDS` off made draws
cheaper (4.58 -> 4.36 and 4.37 -> 4.29 us: building its descriptor key cost more than the pushes
it saved, which the recording thread has room for), and bit 4 on top of bit 34 cut the readback
waits to 21-22 ms/s (bit 34 alone 54-96 ms/s in these boots). Paired rounds base / bit 1 off /
bit 1 off and bit 4 on: 370 / 375 / 386 (bit 4 on with bit 1 still on) and 343 / 363 / 381. Bit 1 is
now off and bit 4 on again by default (the hang it was suspected of was the label readback
race): default mask 1668427021812.

Display: flips go out at every vblank (`KYTY_FORCE_FLIP_RATE` defaults to 0) and the emulated
display runs at 240 Hz (`KYTY_PLAY_VBLANK_HZ` / `KYTY_MENU_VBLANK_HZ`, 0 = the configured
frequency): menus and loading screens reach 75-86 fps, light play 61-69; the game is timed by
the clock (Petr saw nothing run fast). Measurements since then are uncapped.

Measured and dropped: a resource memo for resolutions ahead (the execution thread's
`ResourceMemo` per program, 4 states, the recorded reads re-read through the resolution's log).
Only 10% of resolutions repeat a program's register inputs (per-object pointers in user data),
so capturing the reads and copying snapshots cost more than the hits saved: resolve thread
828 -> 936 ms/s, 51.2 -> 43.1 fps. Its first version crashed the GPU within seconds: with
`KYTY_RESOURCE_DYNAMIC_DEPS` (default on) a memo depends on the user-data registers the walk
reports, and the direct table reader (`SrtWalker::DirectReadRun`, `DirectSrtReads` default on)
read its pointer registers without reporting them, so a draw whose tables moved reused the
resources read through the old pointer while the old bytes were unchanged. The direct reader and
the cached branch conditions now report their registers. The execution thread's memo
(`KYTY_PROGRAM_RESOURCE_CACHE=1`) had the same exposure, but it rarely hits: `KYTY_DBG_MEMO_CHECK=1`
(every hit materialized again with the live readers and compared, `NHL27MEMOCHECK`) saw fewer than
4,096 hits and no mismatch in a match, with and without the fix.

GPU time per frame without the per-draw timestamps (`KYTY_IDLE_TRACE_NO_CMDS=1` at boot: the
idle trace then stamps only command buffers, so the traced match keeps 48-58 fps at 60 Hz;
`KytyPS5-tools\ts-nocmd.sh`): 14.0-15.4 ms per flip in play (GPU 72-82% busy), 24-27 ms in the
30 Hz scenes (67-81% busy). The per-draw timestamps had inflated both (17-23 and about 50 ms).
Neither is GPU-bound yet; the idle time is spread over gaps of 20 us to 1 ms between the 120
command buffers of a frame, waiting for the CPU side. At 60 fps the GPU has 1.5-2.5 ms to spare.

A sampled match (`samples\pr34-execution.txt`, 10 s around a whistle) had the execution thread
20% in unmaps draining the GPU: at stoppages NHL 27 unmaps memory that a pending download (a
lazy shared write or a texture download) still had to publish into, 24 times in 2 s at 10-13 ms
each. Unmaps are now logged in `RenderContext` (8,192 entries, its own lock) and deferred
downloads publish through `RenderContext::WriteUnlessUnmapped`: under the log's lock they write
only the parts of their range not unmapped since they were queued (a range unmapped and mapped
again keeps the new contents), so an unmap no longer waits for them; GPU-written pages and images
in the range still drain. Bit 38 (`AbPending`) restores the drain. One boot, 7 paired rounds:
unmap waits 39.6 -> 3.7 ms/s, draw cost 4.41 -> 4.20 us. The hang report after a device loss
now prints the first and the last 32 unfinished commands and their count.

The GPU stage threads are pinned to one logical CPU each (`KYTY_GPU_THREAD_CPUS`, default 4, 6,
8, 10, 2 for command processor, execution, recording, resolve, host copy), but the game's threads
were free to run on the SMT siblings of the execution and resolve threads, sharing their cores.
Bit 46 `KYTY_GUEST_OFF_STAGE_CORES` (default on) sets the process default CPU sets
(`SetProcessDefaultCpuSets`) to every logical CPU except both of those two cores' CPUs, so guest,
driver and helper threads keep off them while the pinned stage threads keep their affinity
(`Experiments: threads without a CPU of their own avoid CPUs 0xcc0` at boot on the 8-core
5800X3D; the mask is reapplied when the experiments mask changes). Bit 47 (then
`KYTY_GUEST_OFF_STAGE_SIBLINGS`, excluding only the two siblings) measured no better. Two boots with
base / bit 46 / bit 47: execution-thread cost per draw 4.15 / 3.97 / 3.90 and 4.34 / 4.03 / 4.33 us,
per dispatch 3.18 / 2.57 / 2.47 and 3.29 / 2.71 / 2.95 us, resolve busy 880 / 803 / 828 and
830 / 788 / 818 ms/s; paired rounds 304 / 304 / 283 and 323 / 389 / 363 fps in sum. Default mask
now 72037171199476.

In play the execution thread runs about 350,000 operations a second, half of them callbacks.
A label-only `RELEASE_MEM` (about 37,000 a second) queued its label write and then one more
operation that submits when the command buffer holds a batch of work; the label write's own
operation now checks the batch (`CommandProcessor::FlushIfBatchReady`; `AbPending2` restores the
extra operation). One boot: callbacks per draw 0.99 -> 0.85, callback time 91 -> 83 ms/s. Index
buffers through the obtain cache (as plain-read shader buffers) made draws no cheaper (3.83 against
3.73 us without): dropped.

Shader compiles that reach the execution thread (a compute permutation first seen there) spent
most of their time finding EXEC runs (`FindExecRuns`, 8.8% of the execution thread in a sample at
the start of a match): every shrink of a run rebuilt a hash set of its instructions. Membership is
now by position in the block. A boot comparing the new and the old search on every block compiled
(`KYTY_DBG_EXEC_RUNS_CHECK`, removed again): 69,632 blocks, 38,019 runs, no difference. The guest
CPU sets also keep threads off the cores of the extra resolvers (bits 41 and 43) while those run.

Bit 47 is now `KYTY_GUEST_OFF_COPY_CORE` (the host copy thread's core too) and bit 48
`KYTY_GUEST_OFF_CP_CORES` (also the command processor's and the recording thread's cores), both
off: one boot with base / bit 47 / bits 47 and 48, paired rounds 418 / 419 / 326 fps in sum,
draw cost 3.95 / 3.96 / 5.68 us (with four cores left the game's own threads fall behind).

Readback waits of the execution thread in play were mostly downloads for guest threads' write
faults (stores into pages a shader wrote, `NHL27RBWAIT ... from=guest`) whose writer the GPU had
not finished: 4,445 such waits of 7.4 ms each in one boot, 4% of the execution thread (the 36,000
with a finished writer took 0.12 ms each). Such a download now comes back with the writer's tick
(after submitting the open command buffer when the writer is in it); the guest thread waits for
the tick itself while the execution thread goes on, then asks again (up to four times;
`AbPending3`: the execution thread waits). One boot: readback waits 58.2 -> 16.1 ms/s, execution
thread 869 -> 834 ms/s busy, draw cost 3.82 -> 3.75 us; fps unchanged (53.1 / 53.2 at 60 Hz):
the execution thread waits for the resolve thread a little more (110 -> 118 ms/s) instead.

Each guest command buffer (about 1,400 a second) ended its host command buffer and submitted it.
The end of a guest submission now submits only once the host command buffer holds a batch of work
(`BufferFlushIfBatchReady`, 64 draws and dispatches, as label-only releases do); frame ends,
interrupts and the long-buffer splits still submit, and readbacks of the open command buffer
submit it. Submissions per draw -40%. Three boots, paired rounds against one submission per guest
command buffer: 433 / 399, 430 / 414 and 431 / 393 fps in sum, 60 Hz fps 54.5 / 52.1, 55.8 / 53.9
and 58.2 / 54.8. (Batching the submissions after interrupts as well changed nothing: NHL 27's
releases do not interrupt.) `NHL27EXECIDLE` now also counts who asked for the submissions
(`asked by other / submission / interrupt / suspend / split`): in play about 3,000 a second are
splits of long command buffers. A host copy drain does the copies not yet handed to the copy
thread itself (callbacks -3 ms/s).

With the GPU submissions batched, the heavier play scenes (about 2,850 draws a frame) ran at
55 fps with the resolve thread 90% busy and the execution thread waiting for it 14% of the time;
the GPU was then 75-87% busy, 12.3-16.2 ms per flip (`ts-nocmd.sh`, eight 3 s captures). Extra
resolve threads on the stage threads' SMT siblings (CPU 7 or 11) slowed the thread they shared a
core with (draw cost +20% on CPU 7, resolve waits up on CPU 11), and with their own core kept from
the game it lost a core. Bit 43 `KYTY_DRAW_RESOLVER` on CPU 14, sharing that core with the game's
threads (the CPU sets no longer keep the game off the extra resolvers' cores): one boot, 60 Hz
play 57.4 -> 59.4 fps, paired rounds 413 / 447 (DR rounds at 58-59), resolve busy 840 -> 550
ms/s, the execution thread waiting for it 124 -> 50 ms/s. It is on by default now (the crashes it
was once suspected of had other causes, fixed since); default mask 80833264221684. Two fixed-save
matches with it: 59.1 fps at 60 Hz over 150 s and 58.8 over 96 s, no errors.
GPU time by shader in play (`KYTY_PERF_GPU_TIMING=2`, 30 s): compute dispatches lead, a36f0b4f
(a 4K full-screen pass, 1.4 ms), c556aaf3 (dispatch-indirect, 1.4-1.6 ms each, about one a frame),
9380d724 (0.7 ms), 62ece4db (30 dispatches of 16 us), 40d1ee8b (0.35 ms).

The host copy thread turned out mostly idle in play (91% waiting, a third of it asleep; its
work is a few page-protection calls and small copies), so the execution thread's drains before
guest-visible operations (4% of it) partly wait for it to wake. Pinning it to the resolve thread's
SMT sibling and polling for 20 ms before sleeping: one boot, 60 Hz fps 59.5 / 59.1, paired rounds
449 / 456, callback time 143 / 116 ms/s: no gain, dropped.

The second draw resolver is idle about half the time in play; letting it spin 20 ms instead of
2 ms before it sleeps cut the execution thread's waits for it 48 -> 40 ms/s but not the fps (one
boot, 60 Hz 59.7 / 59.1, paired rounds 460 / 461, most rounds at the 60 fps the game paces to):
dropped. Three 6-minute soak matches on the defaults: 59.2, 58.8 and 59.5 fps at 60 Hz, no errors.
One boot later stopped in the menus on a PM4 marker NOP the emulator did not know (id 0x11 behind
the 0x6875 marker signature); unknown markers are now skipped and reported once
(`NHL27MARKER`).

Both CPU placements were measured on 8 cores / 16 logical CPUs: the second draw resolver now
defaults on only with 16 or more logical CPUs (its default CPU is 14) and the game's threads keep
off the stage cores only with 12 or more; their variables still force either way.

Frame times in play (`ts-nocmd.sh`, eight 3 s captures): in light stretches every frame within
16.6-18.4 ms; elsewhere single frames of 29-37 ms. `KYTY_DBG_CP_READS` now also logs the GPU
thread's own writes into GPU-written pages whose downloads wait (`NHL27GPUWRITEWAIT`) and
`KYTY_DBG_STRICT_READS=1` the strict resource reads of GPU-written bytes (`NHL27STRICT`). In play
the stalls were PM4 `WRITE_DATA` packets of the pipelined command processor: its fast label path
(both copies updated, nothing downloaded) ran only on the GPU thread, so on the execution thread
they went through `PrepareGpuWrite`, which downloaded the page's GPU-written bytes and waited for
their writer (the command buffer just submitted): 101 waits of 6-23 ms in two minutes. The
execution thread's write now takes the label path first (`AbPending4`: the old path). One boot:
write-data waits 14 (334 ms) -> 0 in the same phases, draw cost 3.86 -> 3.79 us. (The 6 ms strict
waits attributed to VS 126d611593f8bf08 happen in the loading screens, not in play.) Hang reports
after a device loss now count in the render scheduler's ticks: the presenter's own scheduler,
made later, had made every recorded command look unfinished.

The heaviest compute shaders spend many instructions on 16-bit lane handling (a36f0b4f: 456
`v_and_b32 0xffff` of 3,285 instructions). Native scalar f16 ops reading their source lanes
without the zero-extending extract (the high result half is discarded anyway): paused A/B/A,
top shaders +2.5% / +2.2% GPU time (drift 0.2%): slower, dropped. Twelve soak matches after the
`WRITE_DATA` fix (six with `KYTY_DBG_BREADCRUMBS=1`): 57.9-59.5 fps at 60 Hz, no errors, no device
loss (the losses before were Windows GPU timeouts, LiveKernelEvent 141/117, about one in 40 boots).

Loading screens: a sample of the execution thread there (`KytyPS5-tools\prof-load.sh`) had 23% of
it in the in-order checks of the command processor's early backing reads (`CheckCpBackingRead`
downloading GPU-written bytes to compare). Checking them asynchronously (a copy after the
recorded work, compared when it lands) removed those waits, but the time went to the execution
thread's own faults on the same GPU-written pages (`StillValid`'s plain loads, 37%), and the boot
reached the match no sooner (243 s against 244-247): dropped.

The program store (`KYTY_SHADER_PREGEN`, on by default; the compile inputs of every program a
session compiled, queued to the compile workers at the next start) was named after the build's
commit, so every new build started without it. It is now named after a hash of the committed
shader translator and program cache directories (`KYTY_GIT_SHADER_TREE`; the per-commit name while
either has local changes). Two boots of one build with it on: 1,424 programs queued at the start
of the second, shaders compiled during the match 104 -> 0, and the match reached 15 s sooner
(212.8 -> 197.5 s). Eighteen soak matches in a row on the code before: 54.8-59.6 fps at 60 Hz, no
errors. `KytyPS5-tools\play-nhl.ps1` now runs with the store on.

Three more soak matches with profile v12: 59.4, 59.0 and 59.2 fps at 60 Hz, no errors.

The 60 fps that play held to was the host display, not the game. The swapchain asks for Mailbox,
which the AMD driver does not offer under Windows (it offers Immediate, FIFO, FIFO relaxed and
FIFO latest ready), so it fell back to FIFO, and on Petr's 60 Hz monitor every `vkQueuePresentKHR`
waited for the next refresh (while holding the queue lock the render submissions use); the
presenter's flips, and so the game's frames, followed at exactly 60 per second (the idle trace's
presents 16.66-16.72 ms apart, whatever the emulated 240 Hz vblank did). FIFO latest ready also
held 60. Without Mailbox the swapchain now takes Immediate (a window the desktop composes does not
tear); `--present-mode Fifo` keeps vsync. The swapchain prints its mode (`NHL27VIDEO: swapchain
present mode`) and the presenter the frames it showed every 5 s (`NHL27PRESENT`). One boot: 65.1 fps
at 60 Hz over 111 s (stretches of 61-78), 30 Hz scenes 39.9. The execution thread is then 91-94%
busy at about 400k operations/s (2.2-2.4 us each) and the GPU near its 14.6 ms per frame: 120 fps
needs both about halved. The game asks whether 119.88 Hz output is supported (`KYTY_VBLANK_HZ=120`
answers yes; `NHL27VIDEO` lines) but did not switch to it, and play ran the same.

GPU side at uncapped rates (2026-10-23):
* The guest's global barriers (about 190 a frame after the dedup, 14k/s) cost 2.1 ms of GPU time
  per frame: a paused scene took 13.0 ms per flip with them and 10.9 ms with them left out (a
  measurement only; leaving them out is what broke the ice image on 2026-10-20). Waiting only for
  the stages recorded since the barrier before (64% compute only, 14% graphics only) changed
  nothing (13.78 vs 13.80 ms): the cost is the drain and the cache maintenance themselves.
* Per-dispatch timing in play (`KYTY_PERF_GPU_TIMING=2`): compute `d7cfa519` (a wave32 guest
  shader with 152 VGPRs; on the host 256 VGPRs and 64 bytes of scratch) takes 2.2-2.8 ms per frame,
  27% of the GPU time; `a36f0b4f` 1.1 ms. In some stretches (a minute of one match) compute
  `d46af19f`, a horizon-search ambient occlusion pass at the internal resolution (two nested loops
  over slices and steps, two depth samples per step, mostly f16), takes 6.4 ms per frame, 40% of
  the GPU time; `KYTY_DBG_LOOP_CAP=256` left it unchanged, so its loops end normally.
* The guest barriers now name the device-side accesses (shader, attachment and transfer writes;
  the reads of every device stage) instead of MEMORY_WRITE/MEMORY_READ, which may also make the
  driver write back and invalidate for the host: paused GPU A/B/A 13.75 / 13.65 / 13.82 ms per flip.
* The game renders at 2348x1320 and upscales to 3840x2160.
* Bit 50 `KYTY_EXEC_BRANCH_CONTINUE` (default on; key bit 16384): ExecBranches runs also in loop
  continue targets, the bodies of the sampling loops (`d46af19f`'s inner loop selected every write
  per lane: 113 -> 64 `v_cndmask`, the loop body 330 -> 270 ISA lines; the pass 6.4-6.6 -> 5.5 ms
  per frame in one boot). All 1,607 SPIR-V modules of a match pass `spirv-val`
  (`KYTY_DBG_DUMP_ALL_SPIRV=1` dumps them). A paused scene without per-draw timestamps did not
  change (14.53-14.56 vs 14.52 ms per flip); with them the heavy shaders read 7-21% faster, which
  was mostly how they overlap. Two soak matches with it on: 69.5 and 64.0 fps at 60 Hz, no errors.
* `KYTY_PERF_GPU_TIMING` reports also print `NHL27GPUKINDS`: every key's time summed by kind.
* Bit 51 `KYTY_F16_LANE_VIEWS` (default on; key bit 32768): scalar f16 ops on SDWA/op_sel lanes read
  their sources as components of the register viewed as an f16vec2 (`F16LaneExtract`), compute in
  half precision (`FPAdd16Low`, `FPSub16Low`, `FPMul16Low`, `FPFma16Low`) and write the destination
  half with `F16LaneInsert`; packed f16 sources pick their lanes with an f16vec2 shuffle
  (`F16x2Swizzle`). The AMD compiler then reads and writes register halves itself
  (`v_sub_f16 v43.l, v11.h, v63.l`) instead of masking: crowd VS 8,691 -> 7,533 ISA lines
  (`v_and_b32` 862 -> 158, `v_xor_b32` 289 -> 1), `a36f0b4f` 75-78 -> 59 VGPRs, `d46af19f` 67 -> 54.
  The new IR opcodes are appended (stored records number opcodes by position). All 1,584 SPIR-V
  modules of a match validate. Paused scene, no per-draw timestamps: 13.70-13.72 -> 13.19 ms GPU
  per flip (-3.8%), 69 -> 76 fps. Two soak matches: 64.2 and 66.4 fps at 60 Hz, no errors, crowd close-ups normal.
* `ExecBranches` off (bits 39/40) made the heaviest compute shaders 19-23% slower again.

Execution thread at uncapped rates: about 400k operations/s, 92-94% busy; draws 3.4-3.7 us,
dispatches 2.2-2.5 us, callbacks 0.5 us. 4% of it spun on the region tracking spin lock, which the
guest threads' write faults hold across page-protection calls; CPU-dirty queries
(`MemoryTracker::IsRegionCpuModified`) now read the region's bit words atomically without it
(in-boot A/B: draw 3.80 -> 3.74 us). Another 4.7% waits in `HostCopyQueue::Drain` before
guest-visible callbacks, for the copy thread's copies and the page protections it applies.
Build variants (paused matches, two boots each; cross-boot scenes vary the draw cost by +-7%):
ThinLTO (`-DKYTY_THIN_LTO=ON`) draw 3.43 -> 3.34 us, dispatch 2.19 -> 2.09 us; AVX2 on top
(`-DKYTY_ARCH_AVX2=ON`, new option, off) draw 3.09 -> 3.31 us, fps 62.7 -> 66.1: nothing clear. Three soak
matches with these changes: 63.5, 65.7 and 59.6 fps at 60 Hz (uncapped), no errors, images normal.
`KYTY_DBG_STAGE_STATS=1` also prints `NHL27DRAINS`: drains before operations by callback tag, those
that waited for the copy thread, and the time.

`KYTY_DBG_LONG_OPS=<us>`: execution-thread operations taking that long print where the time went
(`NHL27LONGOP`: waiting for their resolution, then the parts of executing them; the parts need
`KYTY_PERF_STATS=1`; `KytyPS5-tools\longops.py` sums them up).

Bit 42 `KYTY_BDA_LOAD_GROUPS` (default off; key bit 8192; `KYTY_DBG_BDA_GROUPS=1` prints
`NHL27BDAGROUP` per shader): consecutive DWORD loads of one guest load through an address share
one page lookup and issue together when the group lies in one page. Grouped loads turned up in
about 60 pixel shaders (2 to 36 groups each), not in the shadow vertex shaders it was written for.
A paused GPU A/B (A/B/A, per-draw timestamps) had nearly every shader 8-10% slower with it,
grouped or not, while A and A2 agreed within 1%: inconclusive, left off.

Toward 120 fps (2026-10-23):
* The emulated display now runs at 1000 Hz in play and menus by default (`KYTY_PLAY_VBLANK_HZ`,
  `KYTY_MENU_VBLANK_HZ`), and the presenter's frame pool takes a frame whose last present finished,
  or grows up to 8 frames, instead of waiting for a present still in flight (`FramePool::Acquire`
  was 3.1% of the execution thread). In-boot A/B, 6 rounds: 240 Hz 64.3 fps (flip waits 7 ms/s),
  1000 Hz 64.5 fps (3 ms/s). Two soak matches: 62.9 and 59.9 fps at 60 Hz, no errors.
* Guest threads print `NHL27THREAD: <name> tid=<host id>`; `KytyPS5-tools\tcpu.sh` names the busiest
  threads in 60 Hz play (`sampler\threadcpu.exe`). In play at ~65 fps the six GPU pipeline threads
  each use 91-100% of a CPU (spinning included), the game's own threads 30-47% each.
* Stage stats at ~70 fps: draws 195k/s at 3.57 us (2,716 a frame), dispatches 53k/s at 2.22 us,
  callbacks 151k/s at 0.31 us; execution busy 940 ms/s; the resolve thread busy 593 ms/s and
  waiting for room in the queue 389 ms/s. 120 fps needs the execution thread's work per frame
  about 42% smaller; a paused scene takes 14.5 ms of GPU per flip, so the GPU needs about 43% too.
* `KYTY_DBG_DRAW_REPEATS=1` (`NHL27REPEATS`): per stage, against the last use of its program, the
  T#s were the same 84% of the time, the S#s 99.5%, all V#s 49% (40% of the buffer slots changed),
  the user data 12%. 66% of the draws had every stage's T#s and S#s unchanged; 0.1% repeated all.
* `KYTY_DBG_BARRIER_HAZARDS=1` (`NHL27HAZARDS`, since removed): a guest barrier stays pending until a later draw or
  dispatch reads a range written before it, writes a range read or written before it, or accesses
  memory by address (bound buffers, images, attachments, index, vertex and indirect ranges; the
  emulator's own passes and copies are not counted). In play 59% of the guest barriers (11.7k/s)
  would be kept: 23k read-after-write, 16k write-after-write, 7.5k write-after-read, 6k by-address
  in 5 s; 19% of the draws and dispatches use addresses.
* Bit 52 `KYTY_LAZY_GUEST_BARRIERS` (tried, removed again with the `NHL27HAZARDS` probe: its checks cost
  0.3% of the execution thread even while off): the guest barrier is owed to the recording target
  and recorded before the first draw or dispatch whose ranges overlap the work before it (or that
  uses addresses), and before any other command there that accesses memory (`CommandRecorder::GuestWork`:
  copies, fills, clears, dispatches, render pass begins; draws only after a check). With it 18% of the
  guest barriers went unrecorded (the other commands, mostly the emulator's own uploads, took 28k of
  the 50k recorded in 5 s); images were normal. Paused GPU A/B/A without per-draw timestamps:
  14.46 / 14.24 / 14.40 ms per flip (-1.4%), while draws cost 3.74 -> 4.09 us on the execution thread
  (collecting the ranges): in play 61.5 vs 65.5 fps. The paused scene was 95% GPU-busy at
  14.4 ms per flip: the GPU limits it to about 69 fps.
* GPU time by shader in 60 Hz play (`KYTY_PERF_GPU_TIMING=2`, 744 frames): compute `a36f0b4f` 0.98 ms
  a frame, pixel `33037c605e42f4ce` (a full-screen pass) 0.88, compute `d7cfa519` about 1.1 over three
  indirect dispatch sites, compute `c556aaf3` 0.72, pixel `9eaaf80c6d8ef848` 0.64 (15 draws), compute
  `9380d724` 0.61, pixel `136eed67e0cb25f8` 0.51, compute `62ece4db` 0.48, crowd 0.38.
  `KYTY_DBG_GUEST_VGPRS=1` prints the guest's VGPR allocation (`NHL27GUESTREGS`); against the host's
  (`NHL27PSTAT`): pixel `33037c60` 128 -> 252, `136eed67` 152 -> 256 + scratch, `9eaaf80c` 148 -> 256 +
  scratch, compute `d7cfa519` 152 -> 256 + scratch, `c556aaf3` 96 -> 133, `62ece4db` 60 -> 96 + 256 bytes
  of scratch, `a36f0b4f` 48 -> 59. The three pixel shaders load 136-232 DWORDs through addresses, each
  with a page lookup and a fault record (two stores) in a branch of its own; with those stores and a
  kill, LLPC tests depth only after them (`z_order` LATE_Z in the pipeline's metadata).
* Bit 53 `KYTY_EARLY_LATE_FRAGMENT_TESTS` (default on; key bit 65536; needs
  `VK_AMD_shader_early_and_late_fragment_tests`): a pixel shader that reads through addresses, writes
  no guest memory and exports no depth or sample mask declares `EarlyAndLateFragmentTestsAMD`; the two
  shaders then get EARLY_Z_THEN_LATE_Z. Paused A/B/A without per-draw timestamps: 10.63 / 10.53 /
  10.68 ms per flip (-1.2%).
* Bit 42 `KYTY_BDA_LOAD_GROUPS` is now on by default: the three pixel shaders have 20-36 groups (all
  their address loads); paused A/B/A 13.17 / 12.93 / 13.04 ms per flip (-1.3%). With both, a 5-round
  in-match A/B: 65-66 fps either way (play is CPU-bound); frames normal. Soak with both on: 66.3 and
  64.4 fps at 60 Hz, no errors (an earlier soak lost the device once at a match start, without
  breadcrumbs; five boots since were clean).
* Tried and dropped: loads through addresses without branches (a missing page read a zero page that
  BDA page table entry 0 points to; the shader recorded its last missing page when it returned).
  `33037c60` lost all 312 stores and half its branches and its 170 vector loads became scalar, but
  SGPR spills to VGPR lanes went 183 -> 575 `v_readlane` and VGPRs 252 -> 243 only; a heavy paused
  scene took 25.27 / 25.73 / 25.43 ms per flip (A/B/A, +1.5%).
* Buffer binds per frame (`KYTY_PERF_STATS_VERBOSE=1`, `NHL27BUFFERS2`): `RebindBuffers` sent 11.5k
  bindings to `ObtainBuffer` (926 written, 1 stream copy, 2.6k plain reads through the obtain cache,
  the rest texel reads), and 5.2k of them walked the tracker's pages under the region lock in
  `SynchronizeBuffer` to upload 3 ranges. Now a read of a range without CPU-dirty pages (lock-free bit
  check; a texel read only when no image starts at the address) skips `SynchronizeBuffer` (1.1k walks
  left), and the buffer slots (sources, V# keys, bind memos) are kept per program like the image
  memo, so a slot compares with the program's last draw rather than the stage's (10.3k bindings sent).
  In-boot A/B, 6 rounds: 66.3 fps and 906 ms/s of execution with both, 64.3 / 944 with the old syncs,
  65.6 / 938 with the old slot memos.
* The game's own threads (`KytyPS5-tools\prof-guest.sh` samples them by name): NHL 27's job threads
  (Job0-Job12) spent 20-34% of their time blocked in `AudioOut2PortGetState` on the AudioOut2 state
  mutex, which the audio thread held across `AudioOutOutputs`, a blocking push that waits up to a
  grain (~10 ms) for SDL's queue. The context's PCM and gains are now copied under the state mutex
  and pushed without it; a separate output mutex keeps the audio handles open meanwhile (closing one
  takes it). The job threads no longer block there (Job0 53% busy, the rest 25-47%).
* PGO profile v13, recorded after the binding changes: soak 66.7 and 70.3 fps at 60 Hz.
* Guest write faults (`NHL27FAULTCOST`, ~330 a frame) took 140-190 us each, 45-60 ms of guest thread
  time a frame: 12 ms in protection calls, most of the rest spinning on the tracking region locks,
  which faulting threads held across those calls (the fault's window release in `MarkCpuDirtyAhead`,
  the faulting page in `InvalidateRegion`). Every watcher change now only marks its pages stale
  under the `PageManager` region spin lock; the protection is applied after that lock (and, for a
  guest thread's write fault, after the whole handler: `PageManager::ReleaseScope`) by
  `ApplyStaleNow`, which holds a per-region mutex across its protection calls so the last apply of a
  page sees its latest watchers. A fault on a page whose release is still pending applies it
  (`ApplyPending`, first thing in `HandleFault`), so no thread waits for another's handler. Two
  in-boot A/Bs: execution 3.76 vs 3.95 us per draw (the guest faults contended with its
  synchronizations); fps 69.7 and 62.9 vs 65.2 and 67.2 (scene noise).
* Tried: a per-1 MiB-chunk last-write sequence so `StillValid` skips page lookups in unwritten
  chunks (3.71 vs 3.76 us per draw without it, noise), and `KYTY_PROTECT_SHARED_LOCK` again
  (3.71 us, noise); both dropped/left off.
* The GPU limits play too: `KYTY_IDLE_TRACE_NO_CMDS` captures in play read 12-22 ms of GPU work
  per flip at 80-97% busy (heavier scenes, e.g. 44 fps at 97%). Top shaders in play (per-draw
  timestamps): compute `c556aaf3` 1.4-1.6 ms a frame (one indirect dispatch), pixel `33037c60`
  0.8-1.2, compute `a36f0b4f` ~1.0, compute `6cb32e43` ~0.9 (4-5 indirect dispatches, 256 VGPRs
  and 192 bytes of scratch for the guest's 128), compute `9380d724` 0.6. `KYTY_DBG_DUMP_SHADERS`
  prints their guest code (`_Shaders/*_ir_*.txt`) and `KYTY_DBG_PIPELINE_STATS` the host ISA.
* Bit 52 `KYTY_UNDEF_VGPRS` (default on; key bit 131072): a VGPR the hardware does not initialize,
  read before any write on some path, was 0 there, so a value written in one branch and read after
  it merged with that 0 (a phi: zeroing moves and a longer live register). It is now `UndefU32`
  (`OpUndef`), as on the hardware (`c556aaf3`: 264 -> 202 moves). Paused A/B/A without per-draw
  timestamps: 13.56 / 13.37 / 13.55 ms per flip (-1.4%).
* Bit 54 `KYTY_FTZ32` (default on; key bit 262144; needs `shaderDenormFlushToZeroFloat32` and
  32-bit denormal independence): the guest's MODE (0xc0) flushes FP32 denormals; shaders now declare
  `DenormFlushToZero` for 32 bits (LLPC already ran them so, `denormal_fpenv(float: preservesign)`),
  and the inputs of RCP, RSQ, SQRT, EXP and LOG are no longer flushed by hand (an and, a compare and
  a select each). Paused A/B/A: 26.85 / 26.05 / 26.78 ms per flip (-2.8%). Frames captured in play
  with both: normal.
* Tried: declaring the host-written tables (BDA page table, shader data, flattened SRT) NonWritable
  so the page lookups could be shared: 11.22 / 11.32 / 11.28 ms per flip; dropped.
* Execution-thread waits in play (`NHL27EXECIDLE`, now split): ~30 ms/s for the second draw
  resolver's slots, ~27 ms/s for the resolve thread, 11-30 ms/s of them in the first 256 operations
  after a flip. Fault windows of 4 pages instead of 16 (bit 15 off): 3.38 us per draw either way.
* Guest barriers at the command processor: a partial-flush or cache write-back event becomes a
  global barrier operation there, and a barrier right after another one, with no operation between
  them, is not emitted (the execution thread skipped those anyway, `KYTY_GUEST_BARRIER_DEDUP`, after
  a trip through the pipeline: 118k such operations a second for 13k recorded barriers); ignored
  event types emit nothing. In-boot A/B: operations -12% (322k vs 368k a second), execution 3.52 vs
  3.65 us per draw. The CPU-side crumb ring for hang reports holds 4096 commands (16384, 640 KiB, kept
  missing the L2 cache: 0.9% of the execution thread). Soak: 67.6 fps at 60 Hz, no errors.
* Then a barrier is not emitted whenever nothing since the previous one may record GPU work (only
  barriers, BDA boundaries, end-of-pipe labels and interrupts, WRITE_DATA: label writes order their
  own GPU copies), and a BDA boundary is no operation of its own: the next draw or dispatch, the
  only ones that synchronize, carries it and executes it first. In-boot A/B: operations -13.5%
  (269k vs 311k a second; barriers 18k vs 39k, BDA boundaries 1.3k vs 17k), execution 13.2 vs 13.7
  ms a frame (per draw within noise). Soak: 69.0 fps at 60 Hz over 132 s, no errors.
* End-of-pipe and WRITE_DATA labels no longer make the execution thread wait for the host copy
  thread (`HostCopyQueue::Drain`, 7% of it in play): a label of up to 16 bytes is written into guest
  memory by the copy thread, after the copies queued before it (`BufferCache::TryWriteLabel` with a
  queue; the label's GPU copy is still updated by the execution thread), and a submission does not
  drain at all (the recording thread submits after the copies anyway). A direct interrupt
  (`KYTY_DIRECT_EOP_INTERRUPTS`) or a label written on the execution thread drains first. In-boot
  A/B (AbPending6, bit 57: the old waits): drain waits ~55 -> ~12 ms/s, execution 13.4 vs 14.3 ms a
  frame, 71.0 vs 66.0 fps in the same match. Soak: 66.4 fps at 60 Hz, no errors.
* Bit 55 `KYTY_MOVREL_BOUNDS` (default on; key bit 524288): `V_MOVRELS/D_B32` (register arrays
  indexed by M0) selected over every VGPR after their base; compute `6cb32e43` (in play ~0.9 ms a
  frame) has 31 of them over up to 128 registers: 2.7k guest instructions became 19.5k host ones
  (2.9k `v_cndmask`) with 256 VGPRs and 192 bytes of scratch. `MovRelExtents` (Translate.cpp)
  traces M0 within a block (and into single-predecessor blocks) to an SGPR times a stride, groups
  bases closer than 4 (or the stride) as one array's components, and bounds an index SGPR by the
  fewest elements one of its arrays has before the next array it indexes; untraced ones keep the
  full range (`KYTY_DBG_MOVREL=1` prints them, `NHL27MOVREL`). `6cb32e43`: 23 of 31 bounded (4 and
  10 registers), 247 VGPRs, no scratch. Paused A/B/A: 12.94 / 12.67 / 12.98 and 11.30 / 11.13 /
  11.24 ms per flip (-1.2 to -2.2%); frames captured in play unchanged.
* Per draw: a program's shader data and flattened SRT are uploaded again only when their words
  changed or a new command buffer began (`PreparedBindings::UploadMemo`; the copies were 2.4% of the
  execution thread), a texel read drops the known GPU fills of its memory only when an image was
  copied into the buffer (a lock and a scan per texel bind, 1.7%), and the program memos are found
  through a direct-mapped cache (0.8%). In-boot A/B (AbPending7, bit 58: the old way): 3.68 vs 3.74
  us per draw. NHL 27 issues no occlusion queries in play (a run with every query counting no
  samples saw none). The upload memo was dropped again: counted in play it hit 5% of the uploads
  (1.2M hits, 24.7M misses of ~84 words), so the compare and the copy into the memo cost more than
  the hits saved.
* Bit 62 `KYTY_ALIGNED_BUFFER_OFFSETS` (default on; key bit 8388608; only where
  `minStorageBufferOffsetAlignment` is at most 4, as on AMD): a buffer binding starts at the guest
  base aligned down to that alignment and the shader adds the rest from the shader data (the
  binding prefix). Scalar buffer reads align the base and the offset independently, so their prefix
  in dwords is always zero there; adding it per read kept the compiler from using immediate offsets
  and merging constant loads. Compute `c556aaf3`: `s_buffer_load_b32` 189 -> 143 (now b64/b256 for
  the constant ones), `s_add_co_i32` 207 -> 115, SGPRs 94 -> 86. In play 0.35% of the bindings have
  a prefix (2-byte aligned texel buffers), so vector accesses keep it. Paused A/B/A: 17.32 / 17.50 /
  17.42 ms per flip (-0.7%).
* Execution thread: descriptor sets of pipelines without push descriptors are written by the
  recording thread at their place in the command stream (`CommandRecorder::updateDescriptorSets`;
  `vkUpdateDescriptorSets` in the driver was ~3% of the execution thread, the recording thread is
  ~78% idle; in-boot A/B with AbPending5 = the old way: draw 3.74 vs 3.92 us, fps within noise),
  profiler zones test an inline flag instead of calling `tracy::ProfilerAvailable()` (0.6%), and
  `BufferCache::HasGpuDirtyBytes` remembers its answers per range while the GPU-written set is
  unchanged (`RangeSet::Intersects` 1.9 -> 0.8%). The old alternates of AbPending5/6/7 (bits 56-58)
  are gone, so those bits are free for new A/Bs.
* `V_CVT_U32_F32`, `V_CVT_I32_F32`, `V_CVT_RPI/FLR_I32_F32` emit `ConvertU32F32`/`ConvertS32F32`
  alone: the backend already saturates them like the guest (`EmitF32ToU32`: NaN 0, the 32-bit
  bounds), and the translator's clamp before them doubled each conversion (`c556aaf3` 3070 -> 2980
  instructions, `ea94f09b` 2674 -> 2598). The resource-table walker saturates the same way instead
  of failing on out-of-range values. Bit 57 `KYTY_NATIVE_FRACT` (default on, AMD only; key bit
  16777216): `V_FRACT_F32` is GLSL `Fract` without the clamp below one, as the AMD compiler emits
  `v_fract_f32`, which clamps itself (`c556aaf3` -19 instructions).
* Bit 58 `KYTY_NATIVE_F16_OPS` (default on; AMD with native f16 and native min/max; key bit
  33554432): `V_RCP/SQRT/RSQ/LOG/EXP/FLOOR/CEIL/TRUNC/RNDNE/FRACT_F16`, `V_MIN/MAX_F16`,
  `V_MIN3/MAX3_F16` and `V_PK_MIN/MAX_F16` run as host f16 instructions on register halves
  (`FPUnary16Low`, `FPMinMax16Low`, `FPMinMax16x2`) instead of through f32 with NaN fix-ups (the
  packed min/max quieted signalling NaNs by hand: ~15 instructions each). `9380d724`: 2301 -> 1460
  instructions (24 `v_rcp_f16` for 24 `V_RCP_F16`), `a36f0b4f` 1763 -> 1446, `40d1ee8b` 1173 ->
  1065. Paused per-draw A/B/A: `9380d724` 587 -> 385 us, `a36f0b4f` 1088 -> 896, PS `1ca26bbe`
  474 -> 388, `40d1ee8b` 405 -> 374 (unchanged shaders drifted 4-5% between captures); in-match
  frames look the same. PGO v13 (profile of the current code): draws 3.31 us, dispatches 2.30 us.
* Execution thread, PGO v13 build: read-log compares of other mappings use the backing's MRU and
  lookup cache before the map search (`TryCompareBacking`; `GuestReadLog::StillValid` 8.8 -> 6.7%,
  `CompareCleanBacking` 1.7 -> 1.1%), a stage's flattened SRT and shader data share one stream
  reservation (`NativeUploadPair`; `StreamBuffer::Copy` 2.1 -> 1.4%), and stage T#/S# arrays compare
  as one memcmp. Game threads in play: 13 job threads at 25-58% of a CPU each (about 5.4 CPUs);
  the GPU stage threads spin at ~99% on their CPUs (2 ms spin budget), four of which share cores
  with guest threads; guest write faults cost ~8 ms of CPU per frame across threads
  (`KYTY_PERF_STATS=1`, NHL27FRAME `faults=`).
* MOVREL bounds, second step: `MovRelExtents` meets the traced M0 over all predecessors of a block
  (repeated until the block ends are stable, so loop back edges count) and an index masked by a
  constant (`S_AND_B32 vcc_hi, vcc_lo, 3`, in VCC too) bounds its MOVRELs exactly
  (`max * stride + 1`): `6cb32e43` 23 -> 31 of 31 bounded. Bit 56 `KYTY_MOVREL_ARRAYS` (default on;
  key bit 67108864): a bounded `V_MOVRELS/D_B32` of up to 32 registers stores them into a
  function-local array and loads or stores the element (`RegArrayLoad/Store`); the AMD compiler keeps
  the array in registers and uses `v_movrels_b32`/`v_movreld_b32`. `6cb32e43`: 6444 -> 5284
  instructions, 245 -> 203 VGPRs; `7324a60f` 6382 -> 5188, 252 -> 208. Paused A/B/A against no
  bounds at all: `6cb32e43` 259 -> 64-67 us per dispatch; in-match frames look the same.
* Bit 44 `KYTY_BDA_TWO_LOOKUPS` (default on; key bit 134217728; was AbPending2, whose old alternate
  and those of AbPending3/4 are gone): a scalar BDA load group (`S_LOAD_DWORDX2..X16` through a
  pointer, e.g. NHL 27's per-light records at `s[2:3] + index * 64`) that crosses a caching page looks
  up its first and last pages only (`LoadCrossingScalarGroup`) instead of each DWORD's. 9eaaf80c:
  23834 -> 21201 instructions, SGPR spills (`v_readlane`) 644 -> 313; 33037c60 18728 -> 17310.
  Paused A/B/A: those pixel shaders 1-10% faster (drift ~3%). A first version kept a 64-bit value in
  a 32-bit OpPhi (`EmitValuesIfElse` now takes the phi type): the AMD driver crashed compiling it,
  and warm-up does not validate SPIR-V unless `--shader-validation true` is given; codegen changes are
  now checked with a validated full warm-up (`isa-bench.ps1 -Hashes all -Extra
  @('--shader-validation','true')`). `KYTY_DBG_WARMUP_TRACE=1` prints each warm-up pipeline's shaders.
* Pixel shaders of mesh pipelines are not in the pipeline record, so the warm-up bench cannot build
  them; `KytyPS5-tools/spvisa` compiles a dumped pixel shader with a stub vertex shader (`stubvs.py`,
  `psbench.sh`; dump with `KYTY_DBG_DUMP_SHADERS` in a play run). The heavy material shaders are
  3.3-3.7x their guest size (9eaaf80c 6406 -> 23834 instructions, 212 VGPRs): mostly the BDA loads of
  `S_LOAD ... s[2:3], vcc_lo` (dynamic offsets into resource tables, 35 groups) with 64-bit scalar
  address math, fault records and SGPR spills.
* Host ISA without playing: `KYTY_WARMUP_ONLY=<hex hash>[,...]` warms up only the recorded
  pipelines that use those shaders (the driver cache is left as it was) and `KYTY_EXIT_AFTER_WARMUP=1`
  exits after the warm-up; with `KYTY_DBG_PIPELINE_STATS` that is the shaders' VGPRs, SGPRs and ISA
  in a few seconds (`KytyPS5-tools/isa-bench.ps1 -Hashes ... [-Mask ...]`, `isa-masks.sh`,
  `isa-stat.py`). Only compute pipelines and the graphics pipelines in the record (97) are there.
  Host vs guest VGPRs of the heavy compute shaders: `ea94f09b` 60 -> 160, `6cb32e43` 128 -> 245,
  `7324a60f` 128 -> 252, `0b71107d` 36 -> 72, `c556aaf3` 96 -> 135 (wave32), `62ece4db` 60 -> 96
  plus 272 bytes of scratch (workgroups of 1024), `a36f0b4f` 48 -> 52.
* GPU time in play by pass (per-draw timestamps): compute 55% (dispatches 28%, indirect 26%, async
  2%), the 2348x1320 HDR draws 26%, the 3520x1980 ones 6%, depth-only 5%.
* Bit 59 `KYTY_F16_CONVERTS` (default on with native f16; key bit 1048576): the f32 emulation of a
  scalar f16 op packed its result with `PackHalf2x16`, which the AMD compiler emits as
  `v_cvt_pk_rtz_f16_f32` (rounding toward zero, the guest rounds to nearest even); conversions are
  now `OpFConvert` on f16 (`v_cvt_f16_f32`), and the host compiler folds ops between them into f16
  forms. Compute `a36f0b4f`: 2622 -> 2493 instructions, 222 -> 123 `v_cvt_pk_rtz` (the rest pack
  image results). Paused A/B/A: 10.47 / 10.31 / 10.49 ms per flip (-1.6%); frames normal.
* Bit 60 `KYTY_F16_IMAGE_PACK` (default on with it; key bit 2097152): 16-bit float image results are
  packed as f16 pairs from `OpFConvert` (the `PackHalf2x16` per lane avoided an RDNA4 driver crash
  on two components of one sample; this form compiles). `a36f0b4f`: 2493 -> 2362 instructions,
  `v_cvt_pk_rtz` 123 -> 31. Paused A/B/A: 21.40 / 20.94 / 21.03 ms per flip (-1.3%); frames normal.
* Bit 61 `KYTY_EXEC_RUN_TRANSPARENT` (default on; key bit 4194304): `KYTY_DBG_EXEC_BRANCH=2`
  (`NHL27EXECBREAK`) counts what ends a stretch of ExecBranches candidates that holds EXEC selects; in
  the heaviest shaders mostly `Reference` (no code), `GetImageResource`, `GetBufferResource`,
  `GetSrtResource` and `LaneId`, which do not depend on EXEC. They no longer end a run (used after
  it, they are emitted before the branch). Paused A/B/A: 12.56 / 12.36 / 12.51 ms per flip
  (-1.4%); frames normal.
* Tried: `KYTY_FUSED_MAD` again with GPU time per flip: 12.32 / 12.18 / 12.32 ms (-1.1%); left off
  (it changes rounding).
* Match-start device losses (a GPU hang: flips stop, then `ErrorDeviceLost`, no fault addresses):
  2 in 13 boots with bits 42 and 53 on, 0 in 8 with `KYTY_DBG_BREADCRUMBS=1`, 0 in 3 with bit 53 off.
  Too rare to bisect by boots; bit 53 stays on.
* Tried: skipping the region lock in the BDA dirty-span drain when the span had no CPU-dirty page
  (`ForEachCpuDirtySpanBefore`, where the profile showed 2.8% spinning on that lock): 63.2 vs 64.3 fps,
  draw 3.80 vs 3.82 us; dropped.
* Tried: pointer windows for the material shaders' scalar BDA groups through a user-data pointer
  with a dynamic offset: the 64 KiB after the pointer (clipped to the buffer already holding it)
  bound as a read-only storage buffer, read when the group lies inside it (`OpArrayLength`), through
  BDA otherwise. Both paths stay in the shader, so 9eaaf80c grew 21201 -> 22392 instructions, and two
  paused GPU A/Bs were neutral (-0.4% and +0.4% of the top shaders; the material shaders +3.0, +9.2,
  -5.5, -4.1, +3.7%). The BDA address math is not what limits them; dropped. Two lessons: binding a
  range with `FindBuffer`/`ObtainBuffer` that overruns the buffer holding its start creates and
  merges buffers on every draw (execution 3 -> 31 us per operation), and an in-boot mask switch of a
  key bit recompiles every shader (~4 a second), so in-boot A/Bs of key bits need long phases.
* Paused GPU A/Bs are now measured with `KytyPS5-tools/gpu-abn.sh <prefix> <rounds> <mask>...`: a
  paused match, each mask for a few seconds per round, a 3 s capture of command-buffer timestamps
  only (`KYTY_IDLE_TRACE_NO_CMDS`) per mask and round, GPU busy per flip averaged per mask
  (`WARM=<seconds>` first compiles each shader-key mask's variants). Per-draw timestamps inflate GPU
  time per flip by 30-50% and the GPU, not saturated while paused, changes clocks between captures:
  whole captures moved 5-10% (every shader alike) in A/B/A runs, while gpu-abn's rounds agree within
  about 1%. With it, scoped guest barriers (a CS/VS/PS partial flush waiting for its own stage only,
  a cache acquire for none) measured -0.1% (5 rounds) and were dropped.
* Bit 9 `KYTY_FUSED_MAD` is now on by default for pixel and compute shaders: `V_MAD_F32`/`V_MAC_F32`
  become one fused multiply-add. The guest compiler emits them for multiply-adds it may contract (the
  fast form while FP32 denormals are flushed), so fusing them stays within what the shaders allow;
  vertex and mesh shaders keep the separately rounded form so that positions computed by different
  shaders stay bit-identical. Compute `c556aaf3` has 379 of them (each was a multiply and an add, 2980
  host instructions). Paused GPU A/Bs (gpu-abn): -1.3% (all stages, 3 rounds), then for pixel and
  compute only -0.6% (4 rounds, a heavy 20 ms scene) and -1.2% (6 rounds); frames in play normal.
* `KYTY_IDLE_TRACE` per-command captures (`gpu-abn.sh` with `CMDS=1`) now also time the emulator's
  own commands: barriers, buffer and image copies, clears, blits, buffer updates are reported like
  draws with identity `0x4057'0000'0000'00xx` (`host-*` rows in the shaders file). A paused capture:
  per flip 2841 draws (8.0 ms), 802 dispatches (7.5 ms), 1711 barriers (1.7 ms), 349 buffer copies
  (0.9 ms), all with timestamps around each. Upper bounds measured with gpu-abn: label copies
  (`WriteDataBuffer`, ~210 a flip, each between two barriers) without their barriers -1.3%; images
  written by shaders without the barrier before their next same-layout use (the guest orders its
  own shader writes) +0.8%. The GPU time is the guest's shaders.
* Execution thread: `RuntimeFlags::Current()` is always inlined (it unpacked every flag out of line,
  0.7%), and `TextureDescKey` / `DescriptorValue` compare as one memcmp instead of dword by dword
  (1% and 0.6%). In play the command processor waits 91% of the time and the two resolvers 50-58%
  (`prof-stage.sh`): the execution thread alone sets the pace; per shader pair its draws cost 1.7 us
  (the most frequent, all memos hitting) to 5-16 us (`KYTY_DBG_DRAW_COST=1`).
* Sampled null T#s now go through the description memo like the others (`ResolveTexture`): each draw
  with one built the null description, looked up the null image and its view again; reuse accepts
  unregistered images without a guest range. Color fast-clear checks compare the metadata slice in
  the backing store instead of copying it first. Exec profile in play: `ResolveTexture` 3.4 -> 2.1%,
  `RebindImages` 2.2 -> 1.4%, `RuntimeFlags::Current` 1.0 -> 0.3%; frames normal.
* Reads ahead were re-checked far more than needed: writes into guest memory from threads other
  than the execution thread (the labels the host copy thread writes, download publishes) raised their
  pages to "always check" in `PageWriteSeq`, so no read on such a page was ever skipped again: in
  play every one of a dispatch's ~15 reads (`GuestReadLog::StillValid` was 3.5% of the execution
  thread for dispatches alone) and 5 of a draw's 35. Such a write now belongs after the operation the
  execution thread runs at the time (`g_operation` + 1): a resolve that read the bytes before it had
  seen at most the completion of an earlier operation. Now dispatches skip 10.4 of 14.8 reads, draws
  34.7 of 35.4; `KYTY_DBG_VERIFY_SKIPS=1` (compares every skipped read) reported no changed read. In-boot
  A/B (8 rounds, the old raise as an alternate table): draws 4.07 -> 3.60 us, dispatches 2.68 -> 2.35 us.
* PGO v15 (profile after these changes). Soak: 75.2 / 71.4 / 69.1 fps in 60 Hz play, 49-53 in the
  game's 30 Hz scenes (were 39-48), no errors.
* The execution thread now waits on the resolvers more often (in play ~35 ms/s for the resolve thread
  and ~40 for the draw resolver's slots). A third draw resolver (on the dispatch resolver's CPU,
  draws split in three) cut the slot waits 40 -> 7 ms/s but raised per-draw cost (3.58 -> 3.64 us) and
  left the fps the same; the dispatch resolver (bit 41) moved the waits to its own slots: both left
  off. Memo statistics: 95% of texture slots reuse their lookup; most buffer-slot misses are texel
  (formatted) reads whose range changed since the slot's last draw (per-object vertex buffers); a
  shared memo by range for them: draws -2%, dispatches +3.6%, dropped.
* Upcoming resolutions are prefetched: before each operation the execution thread issues prefetches
  for the next operation's snapshot arrays and read log (`ProgramResolution::Prefetch` step 1) and
  for the one after it the resolution's headers and the draw state it reads first (step 0), so the
  first reads of data the resolvers wrote on other cores no longer miss. In-boot A/Bs, 8 rounds each:
  draws -2.7%, -3.5%, -3.6%; dispatches -1 to -2.6%.
* Per-shader GPU time over a whole match (`KYTY_PERF_GPU_TIMING=2`): in the heaviest quarter of the
  5 s windows (the game's 30 Hz cinematic scenes) the indirect compute `d7cfa519` (256 VGPRs + 64 bytes
  of scratch for the guest's 152, wave32) takes 18% of the GPU time and `d46af19f` (a wave64 horizon
  ambient occlusion: two nested loops whose exit test costs ~14 scalar instructions an iteration and
  ~20 VGPR copies, two depth samples per step) 15%, about 2 and 3 ms per dispatch.
* ThinLTO is now on by default where clang-cl links with lld-link (`KYTY_THIN_LTO`, the main build
  is reconfigured with it): with the PGO profile, a paused cross-build A/B (`exe-ab.sh`, two boots of
  each) took draws from 3.14 to 2.75 us (boots 3.11 / 3.17 against 2.63 / 2.86) and dispatches 2.04 to
  2.00 us. A clean build takes about 14 minutes; later builds re-run the link's code generation for the
  changed modules (`lto.cache` in the build directory).
* Bit 45 `KYTY_EXEC_RUN_CONST_READS` (default on; key bit 268435456): `KYTY_DBG_EXEC_BRANCH=2` showed
  that ExecBranches runs still ended mostly at scalar constant reads (`ReadConstBuffer`: 52 in
  `c556aaf3`, 61 in `d7cfa519`) and image address operands (`MakeImageAddress`, no code: 19 and 33),
  which do not depend on EXEC either. They no longer end a run. `c556aaf3` 2765 -> 2705 instructions,
  `d7cfa519` 7039 -> 6955. Paused GPU A/B (4 rounds, the first one recompiling): +0.3, -1.9, -1.9%;
  the whole warm-up (22,736 programs, 752 pipelines) validates with `--shader-validation true`.
* Tried: the constant compares of the execution thread's lookups (36- to 64-byte T#, V#, image view and
  render target keys) inline with SSE2 instead of the C runtime's `memcmp`, which clang calls for more
  than two loads without AVX (the profile put ~6% of the thread in it): paired in-boot windows were the
  same, the cost is the first touch of the data. Dropped. Prefetching the programs' memos ahead (as the
  resolutions) was neutral too.
* The recording thread spins about 77% of its time (in play the driver's command building takes about
  16% of it). The heavy material pixel shaders and `d7cfa519` keep loop-carried values in two VGPR sets
  (each inner light-loop iteration copies ~50 values one way at the header and back at the latch):
  in `33037c60` 113 of the 203 VGPRs live at the peak are such copies. Broadcasting the scalar loads
  through addresses from the first lane (`OpGroupNonUniformBroadcastFirst`, moving them to SGPRs) took
  it from 217 to 211 VGPRs only; `NonWritable` on their pointee changed nothing, `spirv-opt -O` neither.
  `KYTY_DBG_LOOP_CAP` changes about 1% of the instructions and no VGPR counts.
* With the cheaper draws, the execution thread waited ~110 ms/s in 60 Hz play on the resolvers (the
  resolve thread and the draw resolver's slots) and was ~85% busy. A third draw resolver
  (`KYTY_DRAW_RESOLVERS=2`, the default on 16 logical CPUs; on CPU 12, the dispatch resolver's) takes a
  third of the draws: two in-boot A/Bs (AbPending4 = the old halves) 82.3 vs 78.6 and 77.3 vs 73.1 fps
  in the 60 Hz windows, waits on the resolvers 34 vs 46 and 27 vs 39 ms/s, while a draw costs the
  execution thread 3-5% more (3.30 vs 3.15 us). A 3-minute match afterwards: 77.9 fps at 60 Hz.
* PGO profile v16 (recorded with these changes; the build uses ThinLTO). Soak: 77.5 / 81.0 / 77.0 fps
  in 60 Hz play, 45-48 in the game's 30 Hz scenes, no errors.
* Why the heavy shaders kept loop-carried values twice: a small GLSL compute shader
  (`KytyPS5-tools/spvisa/looptest.py`, 48 accumulators in a loop over the set bits of a uniform mask, three
  light types) compiled through the AMD driver uses 58 VGPRs with `while (mask != 0)`, and 105 when
  the loop leaves through any subgroup operation (`subgroupBallot(z) == subgroupBallot(true)`, the
  emulator's form, or even `subgroupBroadcastFirst`, `subgroupAll`, `subgroupAny`): a convergent
  operation in the exit test keeps LLVM from rotating the loop, and the carried values then live in two
  sets copied at the header and the latch every iteration (as in `d7cfa519`, `33037c60`). If/else chains
  nested in other branches doubled them too in that test (sequential single-armed ifs did not), but the
  emulator's shaders have few of those (8 in `d7cfa519`, 48 in `33037c60`; splitting them changed
  nothing there).
* Bit 32 `KYTY_SCC_BRANCHES` (default on; key bit 536870912): branches on SCC (`S_CBRANCH_SCC0/1`)
  test the condition itself instead of a ballot of it; SCC is uniform (scalar results, and ballots of
  masks for SALU mask operations). Almost every loop of the heavy shaders is tested on SCC at its top.
  `c556aaf3` 134 -> 119 VGPRs and 2705 -> 2623 instructions, `d7cfa519` 80 -> 16 bytes of scratch
  (still 256 VGPRs: 183 of the values live at its peak come from its prologue). Paused GPU A/B, 5 rounds:
  -4.1, -4.5, -5.2, -6.9, -5.6% per flip (mean -5.3%); the whole warm-up validates. Bit 32 was
  `KYTY_PROTECT_SHARED_LOCK` (a shared lock in `ProtectTransient`, measured as noise twice; removed).
  Loops tested on VCC or EXEC (`d46af19f`'s two step loops) still exit through a ballot. Soak (the first
  boot recompiling every shader): 71.3 / 73.4 / 78.6 fps in 60 Hz play (CPU-bound), 49-51 in the game's
  30 Hz scenes (were 45-48); no errors, frames normal.
* Merge of `pengui07/KytyPS5-nhl-p` `nhl27-eadb` (Petr's synced nhl27, upstream main, Stage 2):
  its switches `KYTY_PROGRAM_LOCK_SPLIT`, `KYTY_FILL_PATTERNS`, `KYTY_UNMAP_BUFFER_WAIT`,
  `KYTY_DIRECT_GPU_READS`, `KYTY_ASYNC_READBACK` and `KYTY_SPECULATE_STRICT` (all default off) had
  taken experiment bits 38-43, which this line uses, and their defaults would have switched off
  ExecBranches, BdaLoadGroups and DrawResolver: they are environment-only now
  (`Experiments::EnvEnabled`). PGO v17 afterwards. Test builds leave out the Slint launcher (it needs
  Rust and downloads Slint): `-DKYTY_BUILD_LAUNCHER=OFF`.
* Why the material pixel shaders kept two register sets after SCC branches: in `33037c60` (191
  VGPRs), replacing the ballots of the branches that skip an "if" on EXEC or VCC (`s_cbranch_execz`)
  with per-lane tests gave 145 VGPRs; replacing those of the loop exits gave nothing. Such a branch
  runs the body for every lane whenever one lane needs it, and the body's VGPR writes select the new
  value against the old one under EXEC, so both stay live through the body.
* Bit 49 `KYTY_LANE_SKIPS` (default on; key bit 2147483648; was AbPending4, AbPending (38) is the
  free A/B bit now, its two old alternates removed): such a selection branches per lane where the body
  cannot tell (`FindLaneSkip` in `spirvEmitterProgram.cpp`): it stays inside the selection, every side
  effect, ballot, first-lane read and EXEC/VCC branch in it is under an EXEC that implies the lane
  condition (also through a lane's bit of a saved SGPR mask), it reads no other lane, and every merge
  value is the old one in the other lanes or is repaired after the merge's phis (a lane mask implied
  by the condition is cleared there, a value the same in every lane that entered, an SGPR, comes from
  the first of them). Selections are decided innermost first, as repaired values are uniform again.
  16463 of 21547 such selections branch per lane. `33037c60` 191 -> 150 VGPRs, 15175 -> 13738
  instructions; `136eed67` 209 -> 222 VGPRs, 18905 -> 18209; `9eaaf80c` 216 -> 220, 18780 -> 18239
  (their loops re-enable lanes with `S_ORN2_SAVEEXEC`, so their outer selections keep the ballot, and
  per-lane bodies under a uniform selection cost registers). Keeping every selection inside a kept one
  on its ballot made it much worse (`136eed67` 252 VGPRs), so mixed nests stay. Paused GPU A/B, 4
  settled rounds: -1.8, -5.8, -0.4, -2.7% per flip. In play (CPU-bound) an in-boot A/B, 8 rounds,
  stayed within scene noise (60 Hz windows 71.9 vs 69.3 fps); soak 73.6 / 72.6 / 73.4 fps at 60 Hz,
  no errors, frames normal. `KYTY_DBG_LANE_SKIPS=1` prints per shader the
  selections, those per lane and the repairs (`NHL27LANESKIP`), `=2` also why each other one kept its
  ballot (`NHL27LANESKIPNO`).
* Where a draw's time goes on the execution thread (CMake `-DKYTY_EXEC_REGIONS=ON`, then
  `KYTY_DBG_EXEC_REGIONS=1`: `NHL27REGIONS` every 5 s, ms per second by region of the draw and
  dispatch paths, from the time stamp counter; without the option the marks compile to nothing). In
  play about 205,000 draws and 65,000 dispatches a second at 70 fps: 3.4 us a draw, 2.4 us a dispatch.
  A draw with the programs, T#s, S#s and render targets of the draw before it (55% of the draws, 79%
  have the programs) takes 2.2 us, the others about 6 us, and every region costs two to three times as
  much in them (their programs, memos, images and buffers are not in the cache). Two attempts at that
  changed nothing measurable: grouping the fields lookups touch at the start of `Image`,
  `TextureDescDerivation` and `TextureBinding` (79-82 vs 80-86 ns per image slot), and pulling the next
  draws' programs, program memos and their arrays into the cache over three operations ahead (paused
  draw cost -0.6%), and taking a changed V#'s buffer from the by-range memo of `RebindBuffers` instead
  of the page table in `FindBuffers` (3.41 vs 3.47 us, noise). In a dispatch (2.4 us), `RebindBuffers` takes 1 us: 2.6 slot memo hits (33 ns plain,
  110 ns written), 1.75 by-range memo hits (135 ns) and 1.5 lookups (280 ns) per dispatch.
* A boot froze in the menus (once in about 20 boots today): the execution thread waited for a GPU
  tick (a command processor read of GPU-written memory), the recording thread for the host copies of
  that tick's submission, and the host copy thread was idle with every queued copy done; the trigger
  is not confirmed. Looking for it found a real race: `ProgressCounter` (`spscQueue.h`) kept one wait
  target for all waiters, and the host copy counter has two (the recording thread before each
  submission, the execution thread in `HostCopyQueue::Drain()`). A waiter that finished reset the
  target another had stored after it, and `Notify()` then never woke that one (the execution thread
  could sleep in `Drain()` for good). Each sleeper now claims one of three target slots (more sleepers
  are woken by every `Notify()`); `gpu_pipeline_tests` has a five-waiter case.
* `RebindImages` no longer looks the null images up again on every draw: they are never registered,
  and the rediscovery test now skips unregistered images without a guest range, as the table elements'
  test did.
* Lean draws (default on, `KYTY_LEAN_DRAWS=0` turns them off): a stage whose program binds the T#s
  of its last draw keeps that draw's texture bindings, and skips `RebindImages()` and the layout
  transitions, while no image changed since its lookups last held (to the end of that draw) and the
  draw has the same attachments (views and layouts) and no storage image. `g_texture_epoch`
  (`image.h`) advances on every change bindings depend on: dirty and modified flags, tracking, stencil
  association, usage, registration, creation, deletion, replacement (`BumpGeneration`) and every
  barrier that changes a layout or access; `ProgramImageMemo::lean_epoch` holds the epoch a program's
  bindings held at. 65% of the stages in play keep them. Paused in-boot A/B, 6 rounds: draw 3.496 ->
  3.433 us (-1.8%), fps 65.3 -> 66.8. `KYTY_LEAN_VERIFY=1` runs the full lookup for every kept slot,
  the skipped rebind and transitions, and exits on any difference: none in 30 s of play and a
  3-minute match.
* Tried: the same for each unchanged slot of a draw that changes some of its T#s (the slot keeps its
  binding while the epoch holds, the rebind and transitions run). Verified clean in play, but the
  paused in-boot A/B (6 rounds) was 3.290 vs 3.276 us a draw: dropped.
* `Image::GetBarriers` keeps up to two barriers in the object instead of a vector (a heap allocation
  per transition that needed one was about 0.5% of the execution thread).
* PGO v18 (2026-10-25, recorded on f1541bf1, the merge of the 14 newer upstream commits).
* Buffer slots that miss both memos of `RebindBuffers`, by reason (`NHL27REGIONS` draws,
  `NHL27REGIONSDISP` dispatches; region-timer build only): in play about 70,000 a second in draws
  and 76,000 in dispatches had no by-range memo of their range. Almost none were CPU-modified ranges
  (an upload due); in draws 85% were ranges bound before that another range had pushed out of the
  16384-slot memo, in dispatches a third (the rest are ranges not bound in the last 1M lookups).
  With 65536 slots, paused in-boot A/B (6 rounds): dispatch 1.969 -> 1.948 us, draw 3.161 -> 3.158
  us, fps 79.3 -> 80.3; kept.
* The NHL 27 FSR3 hook skips dispatches other than its four passes before timing them (it read the
  clock twice on each of 60,000 dispatches a second).
* DMA bases (`RenderExecutor::CacheDmaBases`): a stage looks its DMA base windows' buffers up again
  only when its program, the base registers' dwords, the buffer set or the mappings changed
  (`FindBuffer` only creates what is missing). The lookups (mapped-range walk under the mapping lock,
  page-table searches) were 25 ms/s in play, 46,000 stages a second (region `dma`). Paused in-boot
  A/B: draw 3.126 -> 3.111 us.
* Why stages did not keep their bindings (play, region-timer build): of 340,000 stages a second
  234,000 kept them; 62,000 had other T#s, 21,000 had a last draw whose bindings did not hold to its
  end and 23,000 found the texture epoch moved since. Each image now also has its own stamp
  (`Image::binding_epoch`, set from `g_texture_epoch` by every change of its own lookup state, usage
  or layouts), and changes of the image set (creation, registration, deletion, replacement, stencil
  association) advance `g_image_set_epoch`: kept bindings hold while the set is unchanged and none
  of their own images changed (`KeptImagesHold()`), whatever other images did. `KYTY_LEAN_VERIFY=1`:
  no difference in the menus and 60 s of play. Paused in-boot A/B: draw 3.297 -> 3.278 us.
* Execution-thread waits in play (sampled): 1.5% in `CommandScheduler::WaitPriorityOperations`
  from `PopPendingOperations` (a completed pending operation waited for the priority thread's
  downloads of its tick). Draws and dispatches now call `PopPendingOperations(false)`: such an
  operation stays queued for a later call. `Image::Transit` returns before `GetBarriers` when the
  whole image is already in the wanted layout and access. Cold/warm timing (`KYTY_DBG_WARM`):
  a transition 22 ns, a plain buffer memo check 14 / 10 ns, a texture lookup reuse 51 / 23 ns.
* Deferred protection for every upload: `SynchronizeBuffer` on the execution thread (host copy
  thread on, coherent staging) defers the protections of pages that gain write watchers to the
  host copy thread, as the BDA synchronization already did. Samples in play: protect calls 1.9 ->
  0.6% of the thread, `SynchronizeBuffer` 4.8 -> 3.0%. (An in-play in-boot A/B was too noisy:
  scenes ran 47-82 fps.)
* Per-slot keep, now with the per-image stamps: in a draw that changes some of its T#s, a slot with
  the same T# keeps its binding (no `ReuseTextureLookup`) while the program's last bindings held to
  their end, the image set is unchanged and the slot's image did not change (`KeptImageHolds()`);
  the rebind and transitions run for the stage. `KYTY_LEAN_VERIFY=1` checks each kept slot: clean in
  the menus and 50 s of play. Paused in-boot A/B: draw 3.117 -> 3.103 us.
* The shipped `assets/cfgGlobal.json` (which the launcher applies over the user's settings) had
  `KYTY_PERF_STATS=1`: the statistics timers and counters on every hot path, and BDA batch
  boundaries emitted as operations of their own. Two boots in play: draw 4.38 vs 3.2-3.5 us,
  dispatch 2.66 vs 2.1 us; a 3-minute match 55.7 fps at 60 Hz with it, 76-78 without. It is now
  `0` there (set it to `1` again for the NHL27BDA, NHL27BUFFERS and other per-frame statistics).
  A match with every variable of `cfgGlobal.json` on top of `play-nhl.ps1`'s (render scale 0.6667,
  FSR3, loop cap 65536, skipped bounds checks...): 77.4 fps at 60 Hz once its programs were in the
  store; the first boot with those code-generation settings compiles them (34 fps, then 10 fps on a
  30 Hz screen, and the rare guest crash at pc 0x9093c3f41).
* The first access to a range's by-range memo entry in `RebindBuffers` (a random entry of the 6.8
  MB table) was the hottest single line of the execution thread (4.4% of its samples in play).
  `FindBuffers` now prefetches the entry when a slot gets a new range, several hundred ns before
  `RebindBuffers` reads it. Paused in-boot A/B: draw 3.404 -> 3.336 us, dispatch 2.493 -> 2.440 us.
  The same before a non-lean stage's image slots (each description's last lookup and the image's
  lookup-state lines, for all slots first): 3.117 / 3.114 us, no effect, dropped. Prefetching the
  upcoming operations one further ahead (the third operation's resolution, the second's snapshot
  arrays): draw 3.004 vs 3.014 us, fps 73.3 vs 71.5, dropped.
* `CommitBindings` keeps each stage's descriptor write and image descriptor counts and its shader
  stage for the stage's program (`PreparedBindings::commit_counts`) instead of walking the
  descriptor mask every draw (paused A/B: draw 3.018 -> 3.006 us, about noise); statistics timers
  that are off write only their flag. The region-timer build is now configured without PGO
  (`-DKYTY_PGO=none`): every profile refresh made its ThinLTO link fail on mixed profile summaries.
* A 3 x 5-minute soak on ee3f53e2: 77.2 / 77.2 / 77.4 fps at 60 Hz; the third match ended in the
  rare guest crash at pc 0x9093c3f41 (`rax=-1`, a UI table of -1 slots; no emulator write within
  0x400 of the pointers among the last 474,845 logged), now seen twice today in about 13 boots.

## Merged defaults (nhl27 + claude/nhl27-perf)

Where both branches implemented the same feature, only one implementation is left and it is a
runtime flag. The experiment bits for these features are retired (their bit numbers are unused).

### Runtime flags (flags.ctl, live)

| Key | Default | Notes |
|---|---|---|
| `KYTY_GUEST_SYNC` | on | Orders compute work with the guest's own barriers instead of a barrier around every dispatch (+13% with `KYTY_SKIP_REDUNDANT_BINDS`) |
| `KYTY_GUEST_BARRIER_DEDUP` | on | Skips redundant guest global barriers |
| `KYTY_DISPATCH_ARGS_GUARD` | on | See "Always on" below; `0` disables |
| `KYTY_RT_DECODE_MEMO` | on | Memoizes color render-target decoding (this was `KYTY_RENDER_TARGET_MEMO`, +3.4%; that name no longer exists) |
| `KYTY_FAULT_COALESCE` | on | A CPU write fault releases its clean aligned window (`KYTY_FAULT_COALESCE_PAGES`, env only, default 16); +2.0%, write faults -85% |
| `KYTY_BDA_BATCH`, `KYTY_RESOURCE_SPEC_CACHE`, `KYTY_SRT_COMPILED` and the other pengui07 flags | on | See `runtimeFlags.h` |
| `KYTY_LONG_BUFFER_GC_AGE` | off | Keeps idle cache buffers 10x longer; +9.2% in Petr's measurement |
| `KYTY_READBACK_QUEUE` | off | Downloads of a buffer last written by an already submitted command buffer run on a second queue; +1.4% |
| `KYTY_DIRECT_EOP_INTERRUPTS` | off | End-of-pipe interrupts are delivered when the CP reaches them; +0.7% |
| `KYTY_SPARSE_BUFFER_GC` | off | Scans for idle cache buffers every 16th submission; +1.0% |

The four that are off were measured as wins on Petr's tree but are still off in the merged
defaults; turn them on with `KYTY_X=1` or a `flags.ctl` line, or flip the default in `FlagSet`.

### Env-only startup switches (default on, `=0` disables)

| Variable | Mask bit | What it does | Measured |
|---|---|---|---|
| `KYTY_SRT_BATCH_READS` | 128 | Reads contiguous resource-table dwords with one call | +54% in a match |
| `KYTY_SKIP_REDUNDANT_BINDS` | 2 | Skips re-pushing identical descriptors and push constants | (with guest sync) |
| `KYTY_DIRECT_SRT_READS` | 1024 | Evaluates resource-table reads and descriptor copies without the IR interpreter | +22.4% paused match (11.1 -> 13.6 fps) |
| `KYTY_IMAGE_LOOKUP_HINT` | 4096 | Reuses the image a texture last resolved to instead of scanning every page | +2.3% paused match |
| `KYTY_CLEAN_FROM_DIRECT` | 262144 | The strict resource-table evaluator takes words the direct reads already fetched | +3.1% paused match |
| `KYTY_IMAGE_RECYCLE` | 512 | A deleted device image (idle on the GPU by then) is kept and handed out again by the next image with the same creation parameters instead of allocating; at most 32 images / 768 MiB, oldest out first. NHL 27 aliases about seven full-screen targets in one memory range, so the texture cache frees and recreates all of them every frame; each new VMA block cost a `vkAllocateMemory` that waits inside the driver (about 8% of the GPU thread in gameplay, `KYTY_DBG_IMAGE_RECYCLE=1` prints the hit rate) | ~97% of image creations recycled; in-match gameplay A/B +12% fps (+-5.7, 8 rounds) and -13% GPU-thread time per draw (+-9%); paused match unchanged (-1% +-2.4: no churn there) |
| `KYTY_LDS_LANE_SLOTS` | 1073741824 | A vertex or pixel shader whose every LDS access is LaneId * 4 plus constants (the guest compiler spilling VGPRs into per-lane LDS slots) gets a small per-invocation array indexed by the constants instead of a 32 KiB one. Part of the vertex and pixel shader keys, so it switches at runtime (shaders recompile) | NHL 27's heaviest pixel shader (33037c605e42f4ce): scratch 32784 -> 0 bytes; +2.6% paused match (in-match A/B, 6 rounds) |
| `KYTY_LEAN_DRAWS` | (env only) | A draw binding the T#s of its program's last draw keeps that draw's texture bindings and layouts while no image changed (`g_texture_epoch`); `KYTY_LEAN_VERIFY=1` checks each kept binding | -1.8% draw cost, paused in-boot A/B |
| `KYTY_TEXTURE_DESC_CACHE` | 2048 | No consumer in the merged tree (pengui07's draw-prep delta replaces it); accepted for compatibility | was +15.5% |
| `KYTY_ADAPTIVE_RESOURCE_MEMO` | 32768 | No consumer in the merged tree (`KYTY_RESOURCE_MEMO_BYPASS` replaces it); accepted for compatibility | was +2.3% |

Default off (set to `1` to enable): `KYTY_CPU_COPY_FORWARD` (8), `KYTY_FALSE_SHARE_READS` (env only since
2026-10-16; bit 64 is now `KYTY_WIDE_WRITE_DOWNLOADS`),
`KYTY_ASYNC_SUBMIT` (2097152), `KYTY_LAZY_SHARED_WRITES` (4194304), `KYTY_NATIVE_FP16` (8388608),
`KYTY_FAULT_PAGE_READS` (16777216), `KYTY_GPU_COLOR_CLEAR` (67108864), `KYTY_POOLED_BDA_BUFFERS`
(134217728), `KYTY_ASYNC_GUEST_READS` (268435456), `KYTY_BDA_DIRTY_REGIONS` (536870912). All but `KYTY_CPU_COPY_FORWARD`, `KYTY_NATIVE_FP16`
and `KYTY_POOLED_BDA_BUFFERS` are wins for NHL 27 (see the measurement log).
`KYTY_RESOURCE_MULTI_CACHE=1` (16-state resource memo) is read only by the resource memo at startup and has no mask bit.

The bits of the removed switches (1, 4, 256, 512, 8192, 16384, 65536, 131072, 524288, 1048576) are ignored in
an experiment mask; use `flags.ctl` for those features.

Dispatch args guard, `KYTY_PROGRAM_RESOURCE_CACHE=1`, `KYTY_PERF_NO_SYNC_READBACK=1`,
`KYTY_EOP_DIRECT_LABELS=1` and `KYTY_SHADER_PREGEN=0` remain separate environment settings.

## Measurement log (claude/nhl27-perf before the merge)

The table below is the original per-switch log, from when every switch was env-only and default off
(`=1` and the mask bits below describe that build). The merged defaults and key types are the tables
above; read this only for the measurements.

### Recommended for NHL 27 on that build

On top of the fork's usual flags (`KYTY_PROGRAM_RESOURCE_CACHE=1`, `KYTY_PERF_NO_SYNC_READBACK=1`,
`KYTY_EOP_DIRECT_LABELS=1`, `KYTY_SHADER_PREGEN=0`, ...):

| Variable | Mask bit | What it does | Measured |
|---|---|---|---|
| `KYTY_SRT_BATCH_READS=1` | 128 | Reads contiguous resource-table dwords with one call (Codex's switch) | +54% in a match |
| `KYTY_GUEST_SYNC=1` | 1 | Orders compute work with the guest's own barriers instead of a barrier around every dispatch | +13% with the next one |
| `KYTY_SKIP_REDUNDANT_BINDS=1` | 2 | Skips re-pushing identical descriptors and push constants | (with the previous) |
| `KYTY_DIRECT_SRT_READS=1` | 1024 | Evaluates resource-table reads and descriptor copies without the IR interpreter | +22.4% paused match (11.1 -> 13.6 fps) |
| `KYTY_TEXTURE_DESC_CACHE=1` | 2048 | Memoizes texture descriptor decoding | +15.5% in a match |
| `KYTY_IMAGE_LOOKUP_HINT=1` | 4096 | Reuses the image a texture last resolved to instead of scanning every page | +2.3% paused match |
| `KYTY_LONG_BUFFER_GC_AGE=1` | 8192 | Keeps idle cache buffers 10x longer (the collector ages per submission) | +9.2% paused match |
| `KYTY_RENDER_TARGET_MEMO=1` | 16384 | Memoizes color render-target decoding (registers to image description) | +3.4% paused match |
| `KYTY_ADAPTIVE_RESOURCE_MEMO=1` | 32768 | Stops capturing resource-table reads for shaders whose inputs change on every draw (the memo can never hit there) | +2.3% paused match |
| `KYTY_FAULT_COALESCE=1` | 65536 | A CPU write fault on a tracked buffer page releases its whole clean 64 KiB window (`KYTY_FAULT_COALESCE_PAGES`, default 16) | +2.0% paused match, write faults -85% |
| `KYTY_SPARSE_BUFFER_GC=1` | 131072 | Scans for idle cache buffers every 16th guest submission instead of after each one | +1.0% paused match |
| `KYTY_CLEAN_FROM_DIRECT=1` | 262144 | The strict resource-table evaluator (branch conditions, uniform fills, written descriptors) takes words the direct reads already fetched | +3.1% paused match |
| `KYTY_READBACK_QUEUE=1` | 1048576 | GPU-to-CPU downloads of a buffer last written by an already submitted command buffer run on a second queue and wait only for that submission (about 45 us instead of draining everything queued) | +1.4% paused match |
| `KYTY_DIRECT_EOP_INTERRUPTS=1` | 524288 | End-of-pipe interrupts are delivered when the CP reaches them, like the labels (`KYTY_EOP_DIRECT_LABELS`) | +0.7% paused match (+0.8% with the readback queue) |
| `KYTY_FALSE_SHARE_READS=1` | 64 (env only since 2026-10-16) | The GPU thread parses PM4 packets (and serves its own faulting loads) from the backing when the bytes are clean but share a page with GPU-written bytes | +0.5% paused match |
| `KYTY_ASYNC_SUBMIT=1` | 2097152 | `vkQueueSubmit` runs on a worker thread in tick order (about 65 us of GPU-thread time per submit); submit splits default to 48 draws / 0.75 ms with it | +1.5% paused match, +4.5% more with the finer splits |
| `KYTY_LAZY_SHARED_WRITES=1` | 4194304 | A guest thread's store into a page where the GPU wrote at most 256 other bytes no longer waits for that GPU work: the download is queued and merged into guest memory later, keeping bytes the CPU changed meanwhile | +7.3% paused match |
| `KYTY_FAULT_PAGE_READS=1` | 16777216 | Resource-table reads of a page on which the GPU thread took a read fault (the GPU wrote other bytes of it) read clean bytes through the backing instead of faulting and downloading again | +4.8% paused match (busy scene) |
| `KYTY_NATIVE_FP16=1` | 8388608 | f16vec2 math for packed half-float ops instead of 32-bit unpack/pack (now switchable at runtime) | 2-19% less GPU time on several heavy shaders; about +3% gameplay now that the GPU limits part of the time |
| `KYTY_GPU_COLOR_CLEAR=1` | 67108864 | A color target or texture whose fast-clear metadata (DCC/CMASK) the GPU wrote is checked on the GPU; matching slices are cleared through conditional rendering (VK_EXT_conditional_rendering) instead of reading the metadata back and waiting for the GPU | removes a per-frame readback wait (about 5% in scenes that have it); no change where they do not |
| `KYTY_ASYNC_GUEST_READS=1` | 268435456 | A guest thread's read of GPU-written memory no longer stalls the GPU thread: it queues and submits the copy and goes on recording, and the guest thread waits for the copy itself | +5.0% paused match, +4.6% gameplay (PGO build) |
| `KYTY_BDA_DIRTY_REGIONS=1` | 536870912 | With `KYTY_BDA_DIRTY_CACHE=1`, BDA preparation synchronizes only the tracker regions that gained CPU writes instead of every mapped range (a mapping change still scans everything); now an experiment bit, switchable while running | +2.2% paused match, +11% gameplay (8 rounds, noisy) |
| `KYTY_POOLED_BDA_BUFFERS=1` | 134217728 | Cache buffers up to 32 MiB come from VMA's shared blocks instead of one `vkAllocateMemory` each (about 2% of the GPU thread in gameplay) | within noise so far |
| `KYTY_GPU_THREAD_PRIORITY=1` | 256 | The GPU thread runs at highest priority | -8% gameplay (noisy); left off |
| `KYTY_EXEC_SELECTS=1` | 33554432 | Shader pass: VGPR writes under a partial EXEC whose old lanes nobody reads lose their select (part of the shader keys) | experimental: the default rule set removes few selects; `KYTY_DBG_EXECSEL_MODE` enables stronger rules that currently break NHL 27 |

The last five rows were measured after the merge (merged defaults, plus the four runtime flags
that are off by default and `KYTY_PM4_WRITE_DATA_FAST=1`). They are env-only switches, default
off; their mask bits are 64, 2097152, 4194304, 16777216 and 8388608.

### In-match A/B of GPU-thread time

With `KYTY_DBG_QUEUE_TIME=1` each NHL27QUEUES line also prints the experiment mask, so an abmulti run (`-Env 'KYTY_DBG_QUEUE_TIME=1'`) can be scored by GPU-thread time per draw/dispatch as well as by fps: `python KytyPS5-tools/ab-items.py <console.txt>` averages us/item per mask, dropping the 5 s window that straddles each switch. Gameplay fps per 20 s phase swings by scene (replays, celebrations); paused matches are steadier but do not exercise everything (the image churn above does not happen there).
`KYTY_DBG_IMAGE_CHURN=<seconds>` prints image inserts and frees starting that long after the first image (`=1`: from the start), to catch the match instead of the menus.

### Why a frame waits

NHL 27 submits a whole frame only after the previous frame's end-of-pipe interrupt, then rewrites
command-buffer memory that shares pages with bytes a compute shader wrote into the previous
frame's command buffers. That write faults, and the fault used to download the page after
everything queued on the GPU (11-15 ms per frame). With both switches above the download waits
only for the writer's submission. Recorded with `KYTY_DBG_TIMELINE` (below).

With direct EOP interrupts the CP no longer waits there; instead the game's own render thread
then stores its next command buffer into a page that holds 16 bytes a compute shader
(abce16f02df45052) wrote at the end of the frame, and that store waited for the host GPU to
finish the frame (about 6 ms per frame). `KYTY_LAZY_SHARED_WRITES` lets the store go ahead and
merges the 16 bytes afterwards. The merge writes a GPU dword only where guest memory still holds
the value it had at the fault, so a later CPU store wins, except one that rewrites the very same
value; a CPU read of those bytes before the merge sees their old value.

### Submit split

The merged `submit_split` runtime flag ends a host command buffer early during a long guest
command buffer so the GPU can start on it. Each split is one more `vkQueueSubmit` (about 65 us on
Windows), so without `KYTY_ASYNC_SUBMIT` the default thresholds are now 768 draws or 12 ms (`KYTY_SUBMIT_SPLIT_DRAWS`,
`KYTY_SUBMIT_SPLIT_US`; `split.ctl` overrides them at runtime): +3.9% paused match against
48 draws / 0.75 ms. Turning the split off costs 8%. Deferring the submit across queued guest
submissions (one host command buffer for many of them) lost 16%, because the GPU then idles while
the GPU thread records. With the recording thread submitting (the retired `KYTY_ASYNC_SUBMIT` still selects the
short thresholds, and play-nhl sets it) and guest command buffers batched (above), the async
defaults are now 192 draws / 3 ms: an in-boot A/B through `split.ctl` (`KytyPS5-toolsbsplit.sh`)
of 48 / 192 / 768 draws, 433 / 455 / 451 fps in paired rounds, most rounds at 59-61 fps. A
fixed-save match afterwards: 57.5 fps at 60 Hz over 96 s (49.5-51.4 in the morning's builds).

## Profile-guided build

`-DKYTY_PGO=use` (clang-cl) builds the emulator and `common` with the NHL 27 gameplay profile in
`tools/pgo/nhl27.profdata`: 18.6 vs 16.1 fps in gameplay (+15%, three boots of each build).
The profile matches clang 22.1.8; other versions ignore it with a warning. `-DKYTY_PGO=generate`
builds the instrumented emulator that records a new profile, and `-DKYTY_THIN_LTO=ON` adds
ThinLTO; `tools/pgo/README.md` has the steps.

Refresh the profile after larger changes: functions whose code changed get no profile data and
are optimized as cold. Profile v5 (2026-10-19, recorded on 0cc673db; v4 predated the resolve and
execution-thread rewrites): gameplay 45.6 -> 49.8 fps at 60 Hz in a fixed-profile match. Profile
v6 (2026-10-19, recorded on ad057ab9: the binding memo work had changed the hottest functions,
ObtainBuffer, FindBuffers, PrepareBindings, StillValid and the pipeline threads' loops): paused
cross-build A/B (`exe-ab.sh`, two boots each) draw cost 4.27 -> 4.01 us, resolve busy 887 -> 846
ms/s. A new profile file takes effect only after a reconfigure (its hash is a configure-time
define): otherwise ninja rebuilds nothing. Profile v7 (2026-10-20, recorded on 053750b3 after the
execution thread's loop and the BDA synchronization changed): cross-build A/B against v6, the one
comparable pair of paused boots: draw 3.69 / 3.71 us, dispatch 3.01 / 2.89 us (the other pair
landed in 30 fps scenes); kept as it matches the current code. Profile v11 (2026-10-22, recorded on bdc062b3:
the buffer slot memos, texel version table, unmap log and guest CPU sets had changed
`RebindBuffers`, `ObtainBuffer` and the execution loop since v10). Profile v12 (2026-10-23, recorded on 610b5dbe: submission
batching, the second draw resolver and the label path of `WRITE_DATA` since v11). Profile v17
(2026-10-25, recorded on bc8b1125, the merge of nhl27-eadb: the merged build had kept profile data
for only part of its functions, 1107 mismatches): three 3-minute matches 78.4 / 76.3 / 69.9 fps at
60 Hz (the merge on v16: 71.0 with a cold shader cache / 67.0 / 70.7). Profile v18 (2026-10-25,
recorded on f1541bf1: lean draws, the region timers and the merge of 14 more upstream commits).

Single fixed-save matches of the same build differ by up to 5 fps at 60 Hz (52.2 and 57.1 within
an hour): compare builds with `exe-ab.sh` (per-draw cost in paused boots), not one match each.

Comparing builds needs separate boots, and scenes differ between boots, so compare the
window-title fps over a few minutes of gameplay in several boots of each. `KYTY_PERF_STATS=1`
must be off for such comparisons: its timers take tens of millions of timestamps per second.

## Always on

`KYTY_DISPATCH_ARGS_GUARD` (a runtime flag, default on; set to `0` in the environment or `flags.ctl` to disable): every indirect dispatch reads its group counts
through a small GPU pass that zeroes counts above the device limit. NHL 27 occasionally issues an
indirect dispatch with 0xffffffff x3 groups during match loads; without the guard the GPU never
finishes it and the device is lost. Zeroed dispatches are reported as `NHL27GUARD` lines.

## Diagnostics

* `KYTY_PERF_STATS=1`: every 5 s, sync waits by cause, per-frame GPU-thread phase times
  (`NHL27PHASE`, `NHL27DRAW`), buffer/image churn and device memory.
* `KYTY_DBG_BREADCRUMBS=1`: GPU breadcrumbs, now with indirect dispatch arguments, printed on
  device loss.
* `KYTY_DBG_IMAGE_CHURN=1`: logs image inserts and frees.
* `KYTY_DBG_SHARED_WRITES=1`: the instruction behind each guest write fault (`NHL27SHW`) and the
  GPU-written bytes a write fault had to download (`NHL27SHD`).
* `KYTY_DBG_PIPELINE_STATS=<hex hash>[,...]|all`: driver statistics of pipelines using those shaders
  (VGPRs, SGPRs, LDS, scratch; `NHL27PSTAT`) and their ISA in `_Shaders` (VK_KHR_pipeline_executable_properties).
* `-DKYTY_LINE_TABLES=ON` (CMake): source lines and inline frames in the PDB, for the sampler.
* `KYTY_DBG_DIRECT_SRT=1`: counts direct and interpreted resource-table refreshes and why plans fall back (`NHL27SRT`).
* `KYTY_PERF_FAULT_RIPS=gpu`: host instruction addresses of the GPU thread's page faults.
* `KYTY_DBG_TIMELINE=<seconds or trigger file>`: records submissions, GPU-thread idle/wake, host
  GPU waits, event-queue waits and triggers, faults and synchronous GPU commands from all threads
  for 1.5 s, then prints them in time order (`NHL27TL`). A non-numeric value names a file whose
  appearance starts the recording.
* `KYTY_PERF_GPU_TIMING=1|2`: per draw/dispatch GPU time by shader (`NHL27GPU`, every 5 s).
  `2` measures each operation from the completion of the previous one, so the times add up to
  GPU busy time; `1` measures from top of pipe (overlapping).
* `tools/profiling/sampler.exe <pid> <seconds> <out> [interval_ms] [tid|rN]`: out-of-process
  stack sampler (about 400 samples/s, dbghelp); `tools/profiling/foldprof.py` summarizes its output.
* Replay benchmarks on a copy of a `_PipelineCache/*.shaders` record (no Vulkan):
  `shader_recompiler_compute_tests --resource-replay-phases <file> [iterations] [direct]`
  (validates the direct path against the interpreter), `--resource-replay-shapes <file>`,
  `--resource-replay-dump <file> <hex hash>`.
