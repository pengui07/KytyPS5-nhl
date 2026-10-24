#pragma once

// Flags are published as one atomic value; readers never retain mutable buffers.
// Refresh only at frame/submit safe points. Hot probes perform one relaxed load.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>

namespace Common::RuntimeFlags {
inline constexpr int64_t kIntervalMs = 250;
inline constexpr unsigned kRevisionShift = 37;
inline constexpr uint64_t kFlagMask = (uint64_t{1} << kRevisionShift) - 1;
inline constexpr uint64_t kUnpinnedBits = uint64_t{1} << kRevisionShift;
static_assert(kRevisionShift <= 48, "flag bits and the unpinned sentinel must fit uint64_t and leave revision bits");
// A flag's bit is its index in kSwitchableFlags. Defaults live in the FlagSet initializers: a missing
// environment variable / flags.ctl line means the default, an explicit =0 / =1 overrides it.
struct FlagSet {
 bool resource_dynamic_deps = true;
 bool resource_memo_verify = false;
 bool srt_cfg_tape = true;
 bool srt_cfg_schedule = false;
 bool pm4_event_clean = true;
 bool pm4_write_data_fast = false;
 bool shader_clean_reads = false;
 bool srt_scalar_batch = false;
 bool resource_hash_filter = false;
 bool resource_memo_bypass = true;
 bool srt_cfg_gather = true;
 bool srt_cfg_trace = true;
 bool fault_ahead = false;
 bool resource_spec_cache = true;
 bool fault_parse_gate = false;
 bool idle_trace = false;
 bool submit_split = true;
 bool srt_source_reuse = true;
 bool dispatch_fast = true;
 bool buffer_fast = true;
 bool drawprep_delta = true;
 bool srt_compiled = true;
 bool pipeline_memo = true;
 bool guest_sync = true;
 bool guest_barrier_dedup = true;
 bool dispatch_args_guard = true;
 bool long_buffer_gc_age = false;
 bool rt_decode_memo = true;
 bool bda_batch = true;
 bool readback_queue = false;
 bool direct_eop_interrupts = false;
 bool fault_coalesce = true;
 bool sparse_buffer_gc = false;
 bool write_cover_skip = false;
 bool aged_writeback = false;
 bool compute_resolve_ahead = false;
 bool fsr1 = false;
};
inline constexpr const char* kSwitchableFlags[] = {
 "KYTY_RESOURCE_DYNAMIC_DEPS",
 "KYTY_RESOURCE_MEMO_VERIFY",
 "KYTY_SRT_CFG_TAPE",
 "KYTY_SRT_CFG_SCHEDULE",
 "KYTY_PM4_EVENT_CLEAN",
 "KYTY_PM4_WRITE_DATA_FAST",
 "KYTY_SHADER_CLEAN_READS",
 "KYTY_SRT_SCALAR_BATCH",
 "KYTY_RESOURCE_HASH_FILTER",
 "KYTY_RESOURCE_MEMO_BYPASS",
 "KYTY_SRT_CFG_GATHER",
 "KYTY_SRT_CFG_TRACE",
 "KYTY_FAULT_AHEAD",
 "KYTY_RESOURCE_SPEC_CACHE",
 "KYTY_FAULT_PARSE_GATE",
 "KYTY_IDLE_TRACE",
 "KYTY_SUBMIT_SPLIT",
 "KYTY_SRT_SOURCE_REUSE",
 "KYTY_DISPATCH_FAST",
 "KYTY_BUFFER_FAST",
 "KYTY_DRAWPREP_DELTA",
 "KYTY_SRT_COMPILED",
 "KYTY_PIPELINE_MEMO",
 "KYTY_GUEST_SYNC",
 "KYTY_GUEST_BARRIER_DEDUP",
 "KYTY_DISPATCH_ARGS_GUARD",
 "KYTY_LONG_BUFFER_GC_AGE",
 "KYTY_RT_DECODE_MEMO",
 "KYTY_BDA_BATCH",
 "KYTY_READBACK_QUEUE",
 "KYTY_DIRECT_EOP_INTERRUPTS",
 "KYTY_FAULT_COALESCE",
 "KYTY_SPARSE_BUFFER_GC",
 "KYTY_WRITE_COVER_SKIP",
 "KYTY_AGED_WRITEBACK",
 "KYTY_COMPUTE_RESOLVE_AHEAD",
 "KYTY_FSR1"
};
static_assert(sizeof(kSwitchableFlags) / sizeof(kSwitchableFlags[0]) == kRevisionShift, "one bit per switchable flag");
inline constexpr uint64_t Bits(const FlagSet& f) {
 return (uint64_t(f.resource_dynamic_deps)<<0) |
  (uint64_t(f.resource_memo_verify)<<1) |
  (uint64_t(f.srt_cfg_tape)<<2) |
  (uint64_t(f.srt_cfg_schedule)<<3) |
  (uint64_t(f.pm4_event_clean)<<4) |
  (uint64_t(f.pm4_write_data_fast)<<5) |
  (uint64_t(f.shader_clean_reads)<<6) |
  (uint64_t(f.srt_scalar_batch)<<7) |
  (uint64_t(f.resource_hash_filter)<<8) |
  (uint64_t(f.resource_memo_bypass)<<9) |
  (uint64_t(f.srt_cfg_gather)<<10) |
  (uint64_t(f.srt_cfg_trace)<<11) |
  (uint64_t(f.fault_ahead)<<12) |
  (uint64_t(f.resource_spec_cache)<<13) |
  (uint64_t(f.fault_parse_gate)<<14) |
  (uint64_t(f.idle_trace)<<15) |
  (uint64_t(f.submit_split)<<16) |
  (uint64_t(f.srt_source_reuse)<<17) |
  (uint64_t(f.dispatch_fast)<<18) |
  (uint64_t(f.buffer_fast)<<19) |
  (uint64_t(f.drawprep_delta)<<20) |
  (uint64_t(f.srt_compiled)<<21) |
  (uint64_t(f.pipeline_memo)<<22) |
  (uint64_t(f.guest_sync)<<23) |
  (uint64_t(f.guest_barrier_dedup)<<24) |
  (uint64_t(f.dispatch_args_guard)<<25) |
  (uint64_t(f.long_buffer_gc_age)<<26) |
  (uint64_t(f.rt_decode_memo)<<27) |
  (uint64_t(f.bda_batch)<<28) |
  (uint64_t(f.readback_queue)<<29) |
  (uint64_t(f.direct_eop_interrupts)<<30) |
  (uint64_t(f.fault_coalesce)<<31) |
  (uint64_t(f.sparse_buffer_gc)<<32) |
  (uint64_t(f.write_cover_skip)<<33) |
  (uint64_t(f.aged_writeback)<<34) |
  (uint64_t(f.compute_resolve_ahead)<<35) |
  (uint64_t(f.fsr1)<<36);
}
// Bit positions for the few call sites that test g_published directly without decoding a FlagSet.
inline constexpr unsigned kBitIdleTrace = 15;
inline constexpr unsigned kBitSrtSourceReuse = 17;
inline constexpr unsigned kBitSrtCompiled = 21;
inline constexpr unsigned kBitLongBufferGcAge = 26;
inline constexpr unsigned kBitBdaBatch = 28;
inline constexpr unsigned kBitReadbackQueue = 29;
inline constexpr unsigned kBitDirectEopInterrupts = 30;
inline constexpr unsigned kBitFaultCoalesce = 31;
inline constexpr unsigned kBitSparseBufferGc = 32;
inline constexpr unsigned kBitWriteCoverSkip = 33;
static_assert(Bits(FlagSet{.idle_trace=true}) >> kBitIdleTrace & 1, "kBitIdleTrace");
static_assert(Bits(FlagSet{.srt_source_reuse=true}) >> kBitSrtSourceReuse & 1, "kBitSrtSourceReuse");
static_assert(Bits(FlagSet{.srt_compiled=true}) >> kBitSrtCompiled & 1, "kBitSrtCompiled");
static_assert(Bits(FlagSet{.long_buffer_gc_age=true}) >> kBitLongBufferGcAge & 1, "kBitLongBufferGcAge");
static_assert(Bits(FlagSet{.bda_batch=true}) >> kBitBdaBatch & 1, "kBitBdaBatch");
static_assert(Bits(FlagSet{.readback_queue=true}) >> kBitReadbackQueue & 1, "kBitReadbackQueue");
static_assert(Bits(FlagSet{.direct_eop_interrupts=true}) >> kBitDirectEopInterrupts & 1, "kBitDirectEopInterrupts");
static_assert(Bits(FlagSet{.fault_coalesce=true}) >> kBitFaultCoalesce & 1, "kBitFaultCoalesce");
static_assert(Bits(FlagSet{.sparse_buffer_gc=true}) >> kBitSparseBufferGc & 1, "kBitSparseBufferGc");
static_assert(Bits(FlagSet{.write_cover_skip=true}) >> kBitWriteCoverSkip & 1, "kBitWriteCoverSkip");
// Low kRevisionShift bits flags; remaining bits revision. One atomic publication.
// Starts at the defaults (revision 0) so a reader that runs before the first Refresh sees them.
inline std::atomic<uint64_t> g_published{Bits(FlagSet{})};
inline std::atomic<int64_t> g_last_poll{0};
inline std::mutex g_refresh_mutex;
inline std::string g_control_path;
inline bool g_path_set = false;
// Shader memory synchronization can submit while materialization is in progress.
// Pin the complete flag set for that evaluation, even if a safe-point poll publishes.
inline thread_local uint64_t g_evaluation_bits = kUnpinnedBits;
class EvaluationScope {
 uint64_t previous;
public:
 EvaluationScope():previous(g_evaluation_bits) {
  if (previous==kUnpinnedBits) g_evaluation_bits=g_published.load(std::memory_order_relaxed)&kFlagMask;
 }
 ~EvaluationScope() { g_evaluation_bits=previous; }
 EvaluationScope(const EvaluationScope&)=delete;
 EvaluationScope& operator=(const EvaluationScope&)=delete;
};
// Always inlined: callers read one or two flags, and the unused ones then cost nothing (out of
// line, unpacking every flag was 0.7% of the execution thread).
#if defined(__clang__) || defined(__GNUC__)
[[gnu::always_inline]]
#endif
inline FlagSet Current() {
 const auto bits = g_evaluation_bits!=kUnpinnedBits ? g_evaluation_bits : g_published.load(std::memory_order_relaxed);
 FlagSet f;
 f.resource_dynamic_deps=(bits>>0&1u)!=0;
 f.resource_memo_verify=(bits>>1&1u)!=0;
 f.srt_cfg_tape=(bits>>2&1u)!=0;
 f.srt_cfg_schedule=(bits>>3&1u)!=0;
 f.pm4_event_clean=(bits>>4&1u)!=0;
 f.pm4_write_data_fast=(bits>>5&1u)!=0;
 f.shader_clean_reads=(bits>>6&1u)!=0;
 f.srt_scalar_batch=(bits>>7&1u)!=0;
 f.resource_hash_filter=(bits>>8&1u)!=0;
 f.resource_memo_bypass=(bits>>9&1u)!=0;
 f.srt_cfg_gather=(bits>>10&1u)!=0;
 f.srt_cfg_trace=(bits>>11&1u)!=0;
 f.fault_ahead=(bits>>12&1u)!=0;
 f.resource_spec_cache=(bits>>13&1u)!=0;
 f.fault_parse_gate=(bits>>14&1u)!=0;
 f.idle_trace=(bits>>15&1u)!=0;
 f.submit_split=(bits>>16&1u)!=0;
 f.srt_source_reuse=(bits>>17&1u)!=0;
 f.dispatch_fast=(bits>>18&1u)!=0;
 f.buffer_fast=(bits>>19&1u)!=0;
 f.drawprep_delta=(bits>>20&1u)!=0;
 f.srt_compiled=(bits>>21&1u)!=0;
 f.pipeline_memo=(bits>>22&1u)!=0;
 f.guest_sync=(bits>>23&1u)!=0;
 f.guest_barrier_dedup=(bits>>24&1u)!=0;
 f.dispatch_args_guard=(bits>>25&1u)!=0;
 f.long_buffer_gc_age=(bits>>26&1u)!=0;
 f.rt_decode_memo=(bits>>27&1u)!=0;
 f.bda_batch=(bits>>28&1u)!=0;
 f.readback_queue=(bits>>29&1u)!=0;
 f.direct_eop_interrupts=(bits>>30&1u)!=0;
 f.fault_coalesce=(bits>>31&1u)!=0;
 f.sparse_buffer_gc=(bits>>32&1u)!=0;
 f.write_cover_skip=(bits>>33&1u)!=0;
 f.aged_writeback=(bits>>34&1u)!=0;
 f.compute_resolve_ahead=(bits>>35&1u)!=0;
 f.fsr1=(bits>>36&1u)!=0;
 return f;
}
inline uint64_t Revision() { return g_published.load(std::memory_order_relaxed)>>kRevisionShift; }
inline int64_t NowMs() {
 return std::chrono::duration_cast<std::chrono::milliseconds>(
  std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool PollNotDue() {
 const auto last = g_last_poll.load(std::memory_order_relaxed);
 return last && NowMs()-last<kIntervalMs;
}
inline void SetControlFile(const std::string& path) {
 std::lock_guard lock(g_refresh_mutex); g_control_path=path; g_path_set=true;
}
inline bool* FieldFor(FlagSet& f,const char* name) {
 if (!std::strcmp(name,kSwitchableFlags[0])) return &f.resource_dynamic_deps;
 if (!std::strcmp(name,kSwitchableFlags[1])) return &f.resource_memo_verify;
 if (!std::strcmp(name,kSwitchableFlags[2])) return &f.srt_cfg_tape;
 if (!std::strcmp(name,kSwitchableFlags[3])) return &f.srt_cfg_schedule;
 if (!std::strcmp(name,kSwitchableFlags[4])) return &f.pm4_event_clean;
 if (!std::strcmp(name,kSwitchableFlags[5])) return &f.pm4_write_data_fast;
 if (!std::strcmp(name,kSwitchableFlags[6])) return &f.shader_clean_reads;
 if (!std::strcmp(name,kSwitchableFlags[7])) return &f.srt_scalar_batch;
 if (!std::strcmp(name,kSwitchableFlags[8])) return &f.resource_hash_filter;
 if (!std::strcmp(name,kSwitchableFlags[9])) return &f.resource_memo_bypass;
 if (!std::strcmp(name,kSwitchableFlags[10])) return &f.srt_cfg_gather;
 if (!std::strcmp(name,kSwitchableFlags[11])) return &f.srt_cfg_trace;
 if (!std::strcmp(name,kSwitchableFlags[12])) return &f.fault_ahead;
 if (!std::strcmp(name,kSwitchableFlags[13])) return &f.resource_spec_cache;
 if (!std::strcmp(name,kSwitchableFlags[14])) return &f.fault_parse_gate;
 if (!std::strcmp(name,kSwitchableFlags[15])) return &f.idle_trace;
 if (!std::strcmp(name,kSwitchableFlags[16])) return &f.submit_split;
 if (!std::strcmp(name,kSwitchableFlags[17])) return &f.srt_source_reuse;
 if (!std::strcmp(name,kSwitchableFlags[18])) return &f.dispatch_fast;
 if (!std::strcmp(name,kSwitchableFlags[19])) return &f.buffer_fast;
 if (!std::strcmp(name,kSwitchableFlags[20])) return &f.drawprep_delta;
 if (!std::strcmp(name,kSwitchableFlags[21])) return &f.srt_compiled;
 if (!std::strcmp(name,kSwitchableFlags[22])) return &f.pipeline_memo;
 if (!std::strcmp(name,kSwitchableFlags[23])) return &f.guest_sync;
 if (!std::strcmp(name,kSwitchableFlags[24])) return &f.guest_barrier_dedup;
 if (!std::strcmp(name,kSwitchableFlags[25])) return &f.dispatch_args_guard;
 if (!std::strcmp(name,kSwitchableFlags[26])) return &f.long_buffer_gc_age;
 if (!std::strcmp(name,kSwitchableFlags[27])) return &f.rt_decode_memo;
 if (!std::strcmp(name,kSwitchableFlags[28])) return &f.bda_batch;
 if (!std::strcmp(name,kSwitchableFlags[29])) return &f.readback_queue;
 if (!std::strcmp(name,kSwitchableFlags[30])) return &f.direct_eop_interrupts;
 if (!std::strcmp(name,kSwitchableFlags[31])) return &f.fault_coalesce;
 if (!std::strcmp(name,kSwitchableFlags[32])) return &f.sparse_buffer_gc;
 if (!std::strcmp(name,kSwitchableFlags[33])) return &f.write_cover_skip;
 if (!std::strcmp(name,kSwitchableFlags[34])) return &f.aged_writeback;
 if (!std::strcmp(name,kSwitchableFlags[35])) return &f.compute_resolve_ahead;
 if (!std::strcmp(name,kSwitchableFlags[36])) return &f.fsr1;
 return nullptr;
}
inline void Trim(std::string& s) {
 const auto first=s.find_first_not_of(" \t\r\n");
 if (first==std::string::npos) { s.clear(); return; }
 const auto last=s.find_last_not_of(" \t\r\n"); s=s.substr(first,last-first+1);
}
inline FlagSet BuildLocked() {
 FlagSet f; // defaults come from the FlagSet initializers
 for (auto* name:kSwitchableFlags) {
  // Environment: only an explicit 0 or 1 overrides the default (unset or anything else keeps it).
  const char* v=std::getenv(name);
  if (v && (v[0]=='0' || v[0]=='1')) *FieldFor(f,name) = v[0]=='1';
 }
 if (!g_path_set) {
  const char* v=std::getenv("KYTY_FLAG_FILE");
  g_control_path=v && *v ? v : "flags.ctl"; g_path_set=true;
 }
 // flags.ctl (KEY=VALUE per line): only listed flags are overridden; a missing line means the default.
 std::ifstream file(g_control_path,std::ios::binary); std::string line;
 while (std::getline(file,line)) {
  const auto eq=line.find('='); if (eq==std::string::npos) continue;
  auto name=line.substr(0,eq), value=line.substr(eq+1); Trim(name); Trim(value);
  if (value!="0" && value!="1") continue;
  if (auto* field=FieldFor(f,name.c_str())) *field=value=="1";
 }
 return f;
}
inline void Refresh(bool force=false) {
 if (!force && PollNotDue()) return;
 std::lock_guard lock(g_refresh_mutex);
 const auto now=NowMs(), last=g_last_poll.load(std::memory_order_relaxed);
 if (!force && last && now-last<kIntervalMs) return;
 const auto f=BuildLocked(); const auto old=g_published.load(std::memory_order_relaxed);
 g_last_poll.store(now,std::memory_order_relaxed);
 const auto bits=Bits(f);
 if ((old&kFlagMask)==bits && (old>>kRevisionShift)!=0) return;
 const auto revision=(old>>kRevisionShift)+1;
 g_published.store((revision<<kRevisionShift)|bits,std::memory_order_relaxed);
 std::printf("NHL27CFG rev=%llu",static_cast<unsigned long long>(revision));
 auto printable=f;
 for (auto* name:kSwitchableFlags) std::printf(" %s=%d",name,*FieldFor(printable,name));
 std::printf("\n"); std::fflush(stdout);
}
}
