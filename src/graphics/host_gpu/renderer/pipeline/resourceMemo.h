#ifndef KYTY_GRAPHICS_RESOURCE_MEMO_H_
#define KYTY_GRAPHICS_RESOURCE_MEMO_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/runtimeFlags.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct ResourceReuseTraceStats {
    uint64_t evictions = 0, checked = 0, matched = 0;
    uint64_t distance16 = 0, distance32 = 0, distance64 = 0, distance128 = 0;
};

struct ResourceHashFilterStats {
    uint64_t eligible = 0, rejected = 0, matched = 0;
};

namespace ResourceHashFilterDetail {
struct Counters { uint64_t eligible = 0, rejected = 0, matched = 0; };
inline thread_local Counters counters;
} // namespace ResourceHashFilterDetail

inline ResourceHashFilterStats GetResourceHashFilterStats() {
    const auto& c = ResourceHashFilterDetail::counters;
    return {c.eligible, c.rejected, c.matched};
}

namespace ResourceReuseTraceDetail {
struct Counters {
    std::atomic<uint64_t> evictions{0}, checked{0}, matched{0};
    std::atomic<uint64_t> distance16{0}, distance32{0}, distance64{0}, distance128{0};
};
inline Counters& GetCounters() { static Counters counters; return counters; }
} // namespace ResourceReuseTraceDetail

inline ResourceReuseTraceStats GetResourceReuseTraceStats() {
    auto& c = ResourceReuseTraceDetail::GetCounters();
    return {c.evictions.load(std::memory_order_relaxed), c.checked.load(std::memory_order_relaxed),
            c.matched.load(std::memory_order_relaxed), c.distance16.load(std::memory_order_relaxed),
            c.distance32.load(std::memory_order_relaxed), c.distance64.load(std::memory_order_relaxed),
            c.distance128.load(std::memory_order_relaxed)};
}

// KYTY_RESOURCE_MISS_DIAG=1 (single-entry memo only): on each register-input miss,
// record which user-data register changed first and how many changed, and whether the
// re-materialized output differs from the previous one. Answers "do changed registers
// produce different resources, or only different buffer addresses?" Diagnostic only.
namespace ResourceMissDiag {
struct Counters {
    std::atomic<uint64_t> misses{0};
    std::array<std::atomic<uint64_t>, 64> first_index{};
    std::array<std::atomic<uint64_t>, 4> changed{}; // 1, 2, 3-4, 5+
    std::atomic<uint64_t> same_images{0}, same_samplers{0}, same_buffers{0}, same_buffers_masked{0};
    std::atomic<uint64_t> same_flat{0}, same_spec{0}, same_fill{0}, all_but_buffer_base{0}, identical{0};
};
inline Counters& Get() { static Counters counters; return counters; }
inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("KYTY_RESOURCE_MISS_DIAG");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
// V# base address is dwords 0 and the low 16 bits of dword 1. Compare everything else.
inline bool BuffersEqualIgnoringBase(const std::vector<ShaderRecompiler::IR::DescriptorValue>& a,
                                     const std::vector<ShaderRecompiler::IR::DescriptorValue>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].dword_count != b[i].dword_count) return false;
        for (uint32_t w = 0; w < a[i].dword_count && w < 8; ++w) {
            const uint32_t mask = w == 0 ? 0u : w == 1 ? 0xffff0000u : 0xffffffffu;
            if ((a[i].dwords[w] & mask) != (b[i].dwords[w] & mask)) return false;
        }
    }
    return true;
}
inline void Record(uint32_t first, uint32_t changed_count,
                   const ShaderRecompiler::IR::ResourceSnapshot& before,
                   const ShaderRecompiler::IR::ResourceSpecialization& before_spec,
                   const ShaderRecompiler::IR::ResourceSnapshot& after,
                   const ShaderRecompiler::IR::ResourceSpecialization& after_spec) {
    auto& c = Get();
    constexpr auto relaxed = std::memory_order_relaxed;
    c.misses.fetch_add(1, relaxed);
    c.first_index[std::min<uint32_t>(first, 63)].fetch_add(1, relaxed);
    c.changed[changed_count <= 1 ? 0 : changed_count == 2 ? 1 : changed_count <= 4 ? 2 : 3].fetch_add(1, relaxed);
    const bool images = before.images == after.images, samplers = before.samplers == after.samplers;
    const bool buffers = before.buffers == after.buffers;
    const bool masked = BuffersEqualIgnoringBase(before.buffers, after.buffers);
    const bool flat = before.flattened_srt == after.flattened_srt, spec = before_spec == after_spec;
    const bool fill = before.uniform_fill == after.uniform_fill;
    c.same_images.fetch_add(images, relaxed);
    c.same_samplers.fetch_add(samplers, relaxed);
    c.same_buffers.fetch_add(buffers, relaxed);
    c.same_buffers_masked.fetch_add(masked, relaxed);
    c.same_flat.fetch_add(flat, relaxed);
    c.same_spec.fetch_add(spec, relaxed);
    c.same_fill.fetch_add(fill, relaxed);
    c.all_but_buffer_base.fetch_add(images && samplers && masked && spec && fill, relaxed);
    c.identical.fetch_add(images && samplers && buffers && flat && spec && fill, relaxed);
}
} // namespace ResourceMissDiag

// A resource snapshot depends on register inputs AND guest memory. Validate the
// actual reads with their original readers; unchanged pointers alone are insufficient.
class ResourceMemoEntry {
    using Runtime = ShaderRecompiler::IR::SrtRuntime;
    struct Read {
        uint64_t address;
        bool strict;
        std::vector<uint32_t> values;
        bool valid = true;
    };
    struct Capture {
        ResourceMemoEntry& memo;
        const Runtime& source;
    };
    struct UserDataCapture {
        ResourceMemoEntry& memo;
        ShaderRecompiler::IR::SrtUserDataObserver forward;
        void* forward_userdata;
    };
    struct ReadReplay {
        std::span<const Read> reads;
        bool inconclusive = false;
    };

    // Safe points publish one atomic flag value; no clock, lock or file poll here.
    static bool DynamicDependenciesEnabled() {
        return ::Common::RuntimeFlags::Current().resource_dynamic_deps;
    }
    static bool VerifyHitsEnabled() {
        return ::Common::RuntimeFlags::Current().resource_memo_verify;
    }
    static void ObserveUserData(void* userdata, uint32_t index) {
        auto& capture = *static_cast<UserDataCapture*>(userdata);
        auto& memo = capture.memo;
        if (index < memo.m_user_data_seen.size() && memo.m_user_data_seen[index] == 0) {
            memo.m_user_data_seen[index] = 1;
            memo.m_observed_register_inputs.push_back(index);
        }
        if (capture.forward != nullptr) capture.forward(capture.forward_userdata, index);
    }

    static bool ReadValues(const Runtime& runtime, bool strict, uint64_t address,
                           std::span<uint32_t> values) {
        const auto span_reader = strict ? runtime.try_read_specialization_span
                                        : runtime.try_read_memory_span;
        if (values.size() > 1 && span_reader != nullptr &&
            span_reader(runtime.userdata, address, values)) return true;
        const auto reader = strict ? runtime.read_specialization_memory : runtime.read_memory;
        if (reader != nullptr) return reader(runtime.userdata, address, values);
        if (strict) return false;
        std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
        return true;
    }
    static void CaptureValues(Capture& capture, bool strict, uint64_t address,
                              std::span<uint32_t> values, bool valid) {
        auto& reads = capture.memo.m_reads;
        // Preserve each successful reader call's width. Only an explicitly
        // successful span capability may introduce a wider capture; adjacent
        // scalar calls can have width-sensitive or lazy failure behavior.
        if (capture.memo.m_read_count == reads.size()) reads.emplace_back();
        auto& read = reads[capture.memo.m_read_count++];
        read.address = address;
        read.strict = strict;
        read.valid = valid;
        if (valid) read.values.assign(values.begin(), values.end());
        else read.values.assign(values.size(), 0u);
    }
    template <bool strict>
    static bool CaptureRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
        auto& capture = *static_cast<Capture*>(userdata);
        const bool valid = ReadValues(capture.source, strict, address, values);
        CaptureValues(capture, strict, address, values, valid);
        return valid;
    }
    template <bool strict>
    static bool CaptureSpan(void* userdata, uint64_t address, std::span<uint32_t> values) {
        auto& capture = *static_cast<Capture*>(userdata);
        const auto reader = strict ? capture.source.try_read_specialization_span
                                    : capture.source.try_read_memory_span;
        if (reader == nullptr || !reader(capture.source.userdata, address, values)) return false;
        CaptureValues(capture, strict, address, values, true);
        return true;
    }
    template <bool strict>
    static bool ReplayRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
        auto& replay = *static_cast<ReadReplay*>(userdata);
        if (values.size() > (UINT64_MAX - address) / sizeof(uint32_t)) {
            replay.inconclusive = true;
            return false;
        }
        const auto end = address + values.size_bytes();
        bool found = false;
        bool valid = false;
        for (const auto& read: replay.reads) {
            if (read.strict != strict || read.values.size() >
                    (UINT64_MAX - read.address) / sizeof(uint32_t)) continue;
            const auto read_end = read.address + read.values.size() * sizeof(uint32_t);
            if (address < read.address || end > read_end || ((address - read.address) & 3u) != 0u) continue;
            const auto offset = static_cast<size_t>((address - read.address) / sizeof(uint32_t));
            const auto captured = std::span(read.values).subspan(offset, values.size());
            if (!found) {
                std::ranges::copy(captured, values.begin());
                valid = read.valid;
                found = true;
            } else if (valid != read.valid || !std::ranges::equal(captured, values)) {
                replay.inconclusive = true;
                return false;
            }
        }
        if (!found) {
            replay.inconclusive = true;
            return false;
        }
        return valid;
    }
    template <bool strict>
    static bool ReplaySpan(void* userdata, uint64_t address, std::span<uint32_t> values) {
        // A declined optional peek must not make verification inconclusive.
        // Successful wide captures also satisfy recursive DWORD subreads.
        const auto& replay = *static_cast<const ReadReplay*>(userdata);
        ReadReplay probe{replay.reads};
        std::vector<uint32_t> words(values.size());
        if (!ReplayRead<strict>(&probe, address, words)) return false;
        std::ranges::copy(words, values.begin());
        return true;
    }
    static bool SameReadCoverage(const std::vector<std::pair<uint64_t, uint64_t>>& left,
                                 const std::vector<std::pair<uint64_t, uint64_t>>& right) {
        // Batch and scalar reads can cover the same bytes with different range
        // segmentation. Reader values are verified separately in their domains.
        const auto normalize = [](const auto& ranges, auto& output) {
            for (const auto& [address, size] : ranges) {
                if (size == 0 || size > UINT64_MAX - address) return false;
                output.emplace_back(address, address + size);
            }
            std::ranges::sort(output);
            size_t count = 0;
            for (const auto range : output) {
                if (count != 0 && range.first <= output[count - 1].second) {
                    output[count - 1].second = std::max(output[count - 1].second, range.second);
                } else output[count++] = range;
            }
            output.resize(count);
            return true;
        };
        std::vector<std::pair<uint64_t, uint64_t>> a, b;
        return normalize(left, a) && normalize(right, b) && a == b;
    }
    static bool SameSnapshot(const ShaderRecompiler::IR::ResourceSnapshot& left,
                             const ShaderRecompiler::IR::ResourceSnapshot& right) {
        return left.buffers == right.buffers && left.images == right.images &&
               left.samplers == right.samplers && left.flattened_srt == right.flattened_srt &&
               left.user_data == right.user_data &&
               SameReadCoverage(left.specialization_reads, right.specialization_reads) &&
               left.uniform_fill == right.uniform_fill &&
               left.uniform_pattern == right.uniform_pattern &&
               left.uniform_pattern_values == right.uniform_pattern_values;
    }
    // KYTY_DBG_MEMO_CHECK=1: every hit is materialized again with the live readers and compared
    // (NHL27MEMOCHECK; a mismatch is a hit that reused resources the inputs no longer give).
    static bool CheckHitsEnabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("KYTY_DBG_MEMO_CHECK");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
    void CheckHit(const ShaderRecompiler::IR::ResourcePlan& plan, const Runtime& runtime,
                  const ShaderRecompiler::IR::ResourceSnapshot& cached_snapshot,
                  const ShaderRecompiler::IR::ResourceSpecialization& cached_specialization) {
        static std::atomic<uint64_t> hits {0}, mismatches {0}, failed {0};
        auto check = runtime;
        check.observe_user_data = nullptr;
        check.user_data_observer_userdata = nullptr;
        check.specialization_cache = nullptr;
        check.delta = nullptr;
        const auto active_sources = plan.active_sources;
        ShaderRecompiler::IR::ResourceSnapshot fresh_snapshot;
        ShaderRecompiler::IR::ResourceSpecialization fresh_specialization;
        const bool ok = ShaderRecompiler::IR::MaterializeResources(plan, check, fresh_snapshot, fresh_specialization);
        plan.active_sources = active_sources;
        auto expected = cached_snapshot;
        expected.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
        const auto count = hits.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!ok) {
            failed.fetch_add(1, std::memory_order_relaxed);
        } else if (!SameSnapshot(expected, fresh_snapshot) || cached_specialization != fresh_specialization) {
            const auto bad = mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
            if (bad <= 16) {
                std::printf("NHL27MEMOCHECK: mismatch plan=%p buffers %zu/%zu images %zu/%zu same buffers=%d images=%d "
                            "flat=%d spec=%d\n",
                            static_cast<const void*>(&plan), expected.buffers.size(), fresh_snapshot.buffers.size(),
                            expected.images.size(), fresh_snapshot.images.size(),
                            expected.buffers == fresh_snapshot.buffers ? 1 : 0,
                            expected.images == fresh_snapshot.images ? 1 : 0,
                            expected.flattened_srt == fresh_snapshot.flattened_srt ? 1 : 0,
                            cached_specialization == fresh_specialization ? 1 : 0);
                std::fflush(stdout);
            }
        }
        if ((count & 4095u) == 0) {
            std::printf("NHL27MEMOCHECK: hits=%llu mismatches=%llu failed=%llu\n", static_cast<unsigned long long>(count),
                        static_cast<unsigned long long>(mismatches.load(std::memory_order_relaxed)),
                        static_cast<unsigned long long>(failed.load(std::memory_order_relaxed)));
            std::fflush(stdout);
        }
    }
    bool VerifyCachedHit(const ShaderRecompiler::IR::ResourcePlan& plan, const Runtime& runtime,
                         const ShaderRecompiler::IR::ResourceSnapshot& cached_snapshot,
                         const ShaderRecompiler::IR::ResourceSpecialization& cached_specialization,
                         bool& inconclusive) {
        ++m_verify_checks;
        ReadReplay replay{std::span(m_reads).first(m_read_count)};
        auto replay_runtime = runtime;
        replay_runtime.userdata = &replay;
        replay_runtime.read_memory = ReplayRead<false>;
        replay_runtime.read_specialization_memory = ReplayRead<true>;
        replay_runtime.try_read_memory_span = runtime.try_read_memory_span != nullptr
            ? ReplaySpan<false> : nullptr;
        replay_runtime.try_read_specialization_span = runtime.try_read_specialization_span != nullptr
            ? ReplaySpan<true> : nullptr;
        // Verification must not replay dependency notifications to an outer collector.
        replay_runtime.observe_user_data = nullptr;
        replay_runtime.user_data_observer_userdata = nullptr;
        // The reference must derive metadata independently of the component cache.
        replay_runtime.specialization_cache = nullptr;

        const auto active_sources = plan.active_sources;
        auto expected_snapshot = cached_snapshot;
        expected_snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
        ShaderRecompiler::IR::ResourceSnapshot fresh_snapshot;
        ShaderRecompiler::IR::ResourceSpecialization fresh_specialization;
        const bool materialized = ShaderRecompiler::IR::MaterializeResources(
            plan, replay_runtime, fresh_snapshot, fresh_specialization);
        const auto fresh_active_sources = plan.active_sources;
        plan.active_sources = active_sources;
        if (replay.inconclusive) {
            ++m_verify_inconclusive;
            inconclusive = true;
            return false;
        }
        if (!materialized || !SameSnapshot(expected_snapshot, fresh_snapshot) ||
            cached_specialization != fresh_specialization || fresh_active_sources != m_active_sources) {
            ++m_verify_mismatches;
            return false;
        }
        ++m_verify_matches;
        return true;
    }

public:
    struct RegisterTraceState {
        const ShaderRecompiler::IR::ResourcePlan* plan = nullptr;
        uint64_t shader_base = 0;
        size_t user_data_size = 0;
        bool dynamic_dependencies = false;
        std::vector<uint32_t> workgroups;
        std::vector<std::pair<uint32_t, uint32_t>> registers;
        bool operator==(const RegisterTraceState&) const = default;
    };
    RegisterTraceState CaptureRegisterTraceState() const {
        RegisterTraceState state;
        state.plan = m_plan;
        state.shader_base = m_shader_base;
        state.user_data_size = m_user_data.size();
        state.dynamic_dependencies = m_dynamic_dependencies;
        state.workgroups = m_workgroups;
        if (m_plan == nullptr) return state;
        state.registers.reserve(m_register_inputs.size());
        for (const auto index: m_register_inputs) {
            if (index >= m_user_data.size()) return {};
            state.registers.emplace_back(index, m_user_data[index]);
        }
        std::ranges::sort(state.registers, {}, &std::pair<uint32_t, uint32_t>::first);
        return state;
    }
    // The hash is a rejection-only prefilter. A match always falls through to
    // RegisterInputsEqual and the original reader validation path.
    bool CanHashFilter(const ShaderRecompiler::IR::ResourcePlan& plan, const Runtime& runtime,
                       bool dynamic_dependencies) const {
        return m_valid && m_hash_metadata_ready && m_register_inputs_valid &&
               m_plan == &plan && m_shader_base == runtime.shader_base &&
               m_user_data.size() == runtime.user_data.size() &&
               m_dynamic_dependencies == dynamic_dependencies &&
               std::ranges::equal(m_workgroups, runtime.workgroup_counts);
    }
    uint64_t CurrentRegisterStateHash(std::span<const uint32_t> current) const {
        return HashRegisterState(m_register_inputs, current);
    }
    uint64_t StoredRegisterStateHash() const { return m_register_state_hash; }
    bool SameRegisterDependencySet(const ResourceMemoEntry& other) const {
        return m_register_dependency_hash == other.m_register_dependency_hash &&
               m_register_inputs == other.m_register_inputs;
    }
    bool HashFilterPolicyValid(bool dynamic_dependencies) const {
        return m_valid && m_hash_metadata_ready && m_dynamic_dependencies == dynamic_dependencies;
    }
    bool ValidForDependencyPolicy(bool dynamic_dependencies) const {
        return m_valid && m_dynamic_dependencies == dynamic_dependencies;
    }
    bool UsesDynamicDependencies() const { return m_dynamic_dependencies; }
    void PrepareHashFilterMetadata() {
        m_register_inputs_valid = std::ranges::all_of(
            m_register_inputs, [&](uint32_t index) { return index < m_user_data.size(); });
        m_register_dependency_hash = HashDependencySet(m_register_inputs);
        m_register_state_hash = HashRegisterState(m_register_inputs, m_user_data);
        m_hash_metadata_ready = true;
    }
    enum class Miss {
        None, Cold, ShaderBase, Registers, Workgroups, ReadFailed, MemoryChanged,
        VerifyMismatch, VerifyInconclusive, Count
    };
    Miss LastMiss() const { return m_last_miss; }
    size_t RegisterDependencyCount() const { return m_register_inputs.size(); }
    size_t StaticRegisterDependencyCount() const { return m_static_register_count; }
    uint64_t VerifyChecks() const { return m_verify_checks; }
    uint64_t VerifyMatches() const { return m_verify_matches; }
    uint64_t VerifyMismatches() const { return m_verify_mismatches; }
    uint64_t VerifyInconclusive() const { return m_verify_inconclusive; }
    void Invalidate() { m_valid = false; }
    size_t RecordedWords() const {
        size_t total = 0;
        for (const auto& read: std::span(m_reads).first(m_read_count)) total += read.values.size();
        return total;
    }

    bool Refresh(const ShaderRecompiler::IR::ResourcePlan& plan, const Runtime& runtime,
                 ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                 ShaderRecompiler::IR::ResourceSpecialization& specialization,
                 bool enabled, bool& reused, bool probe_only = false,
                 bool update_push_data = true) {
        reused = false;
        m_last_miss = Miss::None;
        const bool dynamic_dependencies = DynamicDependenciesEnabled();
        // A control-file switch must not reuse a dependency list collected under
        // the previous policy. Rebuild each entry once; subsequent probes use the
        // same selected policy for both register validation and notifications.
        if (m_valid && m_dynamic_dependencies != dynamic_dependencies) {
            m_valid = false;
            if (!dynamic_dependencies) m_register_inputs = m_static_register_inputs;
        }
        bool inputs_equal = enabled;
        if (enabled) {
            if (!m_valid || m_plan != &plan) m_last_miss = Miss::Cold;
            else if (m_shader_base != runtime.shader_base) m_last_miss = Miss::ShaderBase;
            else if (!RegisterInputsEqual(runtime.user_data)) m_last_miss = Miss::Registers;
            else if (!std::ranges::equal(m_workgroups, runtime.workgroup_counts)) m_last_miss = Miss::Workgroups;
            inputs_equal = m_last_miss == Miss::None;
        }
        if (inputs_equal) {
            bool equal = true;
            for (const auto& read: std::span(m_reads).first(m_read_count)) {
                m_check.resize(read.values.size());
                const bool valid = ReadValues(runtime, read.strict, read.address, m_check);
                if (valid != read.valid) {
                    m_last_miss = Miss::ReadFailed;
                    equal = false;
                    break;
                }
                if (valid && m_check != read.values) {
                    m_last_miss = Miss::MemoryChanged;
                    equal = false;
                    break;
                }
            }
            if (equal) {
                if (VerifyHitsEnabled()) {
                    bool inconclusive = false;
                    if (!VerifyCachedHit(plan, runtime, snapshot, specialization, inconclusive)) {
                        m_valid = false;
                        m_last_miss = inconclusive ? Miss::VerifyInconclusive : Miss::VerifyMismatch;
                        inputs_equal = false;
                    }
                }
            }
            if (equal && inputs_equal && CheckHitsEnabled()) CheckHit(plan, runtime, snapshot, specialization);
            if (equal && inputs_equal) {
                // Non-resource SGPRs can vary per draw. They still feed push data,
                // but do not require walking the descriptor graph again.
                if (update_push_data) snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
                plan.active_sources = m_active_sources;
                // A caller may be collecting dependencies around this memo. Replay
                // the validated dependency set on hits so it sees the same roots.
                if (runtime.observe_user_data != nullptr) {
                    for (const auto index: m_register_inputs) {
                        runtime.observe_user_data(runtime.user_data_observer_userdata, index);
                    }
                }
                reused = true;
                return true;
            }
        }
        if (probe_only) return false;
        m_valid = false;
        m_read_count = 0;
        if (!enabled) {
            return ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
        }
        Capture capture{*this, runtime};
        auto observed = runtime;
        observed.userdata = &capture;
        observed.read_memory = CaptureRead<false>;
        observed.read_specialization_memory = CaptureRead<true>;
        observed.try_read_memory_span = runtime.try_read_memory_span != nullptr
            ? CaptureSpan<false> : nullptr;
        observed.try_read_specialization_span = runtime.try_read_specialization_span != nullptr
            ? CaptureSpan<true> : nullptr;
        UserDataCapture user_data_capture{*this, runtime.observe_user_data,
                                          runtime.user_data_observer_userdata};
        if (dynamic_dependencies) {
            for (const auto index: m_observed_register_inputs) {
                if (index < m_user_data_seen.size()) m_user_data_seen[index] = 0;
            }
            m_observed_register_inputs.clear();
            if (m_user_data_seen.size() != runtime.user_data.size()) {
                m_user_data_seen.assign(runtime.user_data.size(), 0);
            }
            m_observed_register_inputs.reserve(runtime.user_data.size());
            observed.observe_user_data = ObserveUserData;
            observed.user_data_observer_userdata = &user_data_capture;
        }
        if (!ShaderRecompiler::IR::MaterializeResources(plan, observed, snapshot, specialization)) return false;
        if (dynamic_dependencies) {
            m_register_inputs = m_observed_register_inputs;
        }
        if (m_plan != &plan) {
            m_static_register_inputs.clear();
            for (const auto& value: plan.value_storage) {
                if (value.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) continue;
                const auto reg = ShaderRecompiler::IR::RegIndex(value.Arg(0).ScalarRegister());
                if (reg >= plan.user_data_base) m_static_register_inputs.push_back(reg - plan.user_data_base);
            }
            std::ranges::sort(m_static_register_inputs);
            m_static_register_inputs.erase(
                std::unique(m_static_register_inputs.begin(), m_static_register_inputs.end()),
                m_static_register_inputs.end());
            m_static_register_count = m_static_register_inputs.size();
        }
        if (!dynamic_dependencies) {
            m_register_inputs = m_static_register_inputs;
        }
        m_plan = &plan;
        m_shader_base = runtime.shader_base;
        m_user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
        m_workgroups.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
        // Verification can be enabled after this entry was captured. Activity
        // also belongs to this cached state, not the last alternating material.
        m_active_sources = plan.active_sources;
        m_dynamic_dependencies = dynamic_dependencies;
        m_register_state_hash = HashRegisterState(m_register_inputs, m_user_data);
        m_valid = true;
        return true;
    }

private:
    static uint64_t HashRegisterState(std::span<const uint32_t> indices,
                                      std::span<const uint32_t> values) {
        uint64_t hash = 14695981039346656037ull;
        const auto mix = [&](uint32_t value, uint64_t& target) {
            for (unsigned shift = 0; shift != 32; shift += 8) {
                target ^= static_cast<uint8_t>(value >> shift);
                target *= 1099511628211ull;
            }
        };
        mix(static_cast<uint32_t>(indices.size()), hash);
        for (const auto index: indices) {
            mix(index, hash);
            if (index < values.size()) mix(values[index], hash);
        }
        return hash;
    }
    static uint64_t HashDependencySet(std::span<const uint32_t> indices) {
        uint64_t hash = 14695981039346656037ull;
        for (unsigned shift = 0; shift != 32; shift += 8) {
            hash ^= static_cast<uint8_t>(static_cast<uint32_t>(indices.size()) >> shift);
            hash *= 1099511628211ull;
        }
        for (const auto index: indices) {
            for (unsigned shift = 0; shift != 32; shift += 8) {
                hash ^= static_cast<uint8_t>(index >> shift);
                hash *= 1099511628211ull;
            }
        }
        return hash;
    }
public:
    // ResourceMissDiag: first changed register input and how many changed (0 if none).
    uint32_t DiffRegisterInputs(std::span<const uint32_t> current, uint32_t& changed) const {
        uint32_t first = UINT32_MAX;
        changed = 0;
        if (current.size() != m_user_data.size()) return first;
        for (const auto index: m_register_inputs) {
            if (index < current.size() && current[index] != m_user_data[index]) {
                first = std::min(first, index);
                ++changed;
            }
        }
        return first;
    }

private:
    Miss m_last_miss = Miss::None;
    bool RegisterInputsEqual(std::span<const uint32_t> current) const {
        if (current.size() != m_user_data.size()) return false;
        for (const auto index: m_register_inputs) {
            if (index >= current.size() || current[index] != m_user_data[index]) return false;
        }
        return true;
    }
    const ShaderRecompiler::IR::ResourcePlan* m_plan = nullptr;
    uint64_t m_shader_base = 0;
    bool m_valid = false;
    bool m_dynamic_dependencies = false;
    std::vector<uint32_t> m_user_data;
    std::vector<uint32_t> m_workgroups;
    std::vector<uint8_t> m_active_sources;
    std::vector<uint32_t> m_register_inputs;
    std::vector<uint32_t> m_observed_register_inputs;
    std::vector<uint8_t> m_user_data_seen;
    std::vector<uint32_t> m_static_register_inputs;
    size_t m_static_register_count = 0;
    std::vector<Read> m_reads;
    size_t m_read_count = 0;
    std::vector<uint32_t> m_check;
    uint64_t m_verify_checks = 0;
    uint64_t m_verify_matches = 0;
    uint64_t m_verify_mismatches = 0;
    uint64_t m_verify_inconclusive = 0;
    uint64_t m_register_state_hash = 0;
    uint64_t m_register_dependency_hash = 0;
    bool m_hash_metadata_ready = false, m_register_inputs_valid = false;
};

// Keep multiple validated states so alternating materials do not evict each other.
// Never cache solely by pointer: each candidate still replays its dependency reads.
class ResourceMemo {
public:
    explicit ResourceMemo(size_t capacity = 0) : ResourceMemo(capacity, TraceFromEnvironment()) {}
    ResourceMemo(size_t capacity, bool trace_enabled) : m_capacity(capacity), m_trace_enabled(trace_enabled) {
        if (m_capacity == 0) {
            const char* flag = std::getenv("KYTY_RESOURCE_MULTI_CACHE");
            m_capacity = flag != nullptr && flag[0] == '1' ? 16 : 1;
        }
        m_capacity = std::clamp(m_capacity, size_t{1}, size_t{16});
        m_entries.reserve(m_capacity);
    }
    using Miss = ResourceMemoEntry::Miss;
    Miss LastMiss() const { return m_last_miss; }
    bool Refresh(const ShaderRecompiler::IR::ResourcePlan& plan,
                 const ShaderRecompiler::IR::SrtRuntime& runtime,
                 ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                 ShaderRecompiler::IR::ResourceSpecialization& specialization,
                 bool enabled, bool& reused) {
        reused = false;
        const auto flags = Common::RuntimeFlags::Current();
        const bool hash_filter = flags.resource_hash_filter;
        if (m_trace_enabled && !m_trace_policy_valid) {
            m_trace_policy = Common::RuntimeFlags::Current().resource_dynamic_deps;
            m_trace_policy_valid = true;
        } else if (m_trace_enabled && m_trace_policy != Common::RuntimeFlags::Current().resource_dynamic_deps) {
            m_trace_policy = Common::RuntimeFlags::Current().resource_dynamic_deps;
            m_trace_records.clear(); m_trace_next = 0;
        }
        if (!enabled) {
            m_entries.clear();
            m_trace_records.clear(); m_trace_next = 0;
            m_dependency_groups_ready = false;
            m_last_hit = SIZE_MAX;
            reused = false;
            return ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
        }
        if (m_capacity == 1) {
            // With one state the caller already owns the cached snapshot. Refresh it
            // directly rather than maintaining and copying a second full snapshot.
            if (m_entries.empty()) m_entries.emplace_back();
            auto& memo = m_entries.front().memo;
            if (m_output != &snapshot || m_specialization != &specialization) memo.Invalidate();
            const bool diag = ResourceMissDiag::Enabled();
            uint32_t diag_first = UINT32_MAX, diag_changed = 0;
            if (diag) {
                diag_first = memo.DiffRegisterInputs(runtime.user_data, diag_changed);
                if (diag_changed != 0) {
                    m_diag_snapshot = snapshot;
                    m_diag_specialization = specialization;
                }
            }
            const bool ok = memo.Refresh(plan, runtime, snapshot, specialization, true, reused);
            m_last_miss = memo.LastMiss();
            if (diag && ok && diag_changed != 0 && m_last_miss == Miss::Registers) {
                ResourceMissDiag::Record(diag_first, diag_changed, m_diag_snapshot, m_diag_specialization,
                                         snapshot, specialization);
            }
            m_output = &snapshot;
            m_specialization = &specialization;
            if (!ok || memo.RecordedWords() > 4096) {
                m_entries.clear();
                m_output = m_specialization = nullptr;
            }
            return ok;
        }
        const auto count = m_entries.size();
        if (hash_filter && !m_dependency_groups_ready) {
            for (auto& entry: m_entries) entry.dependency_group = UINT8_MAX;
            for (auto& entry: m_entries) {
                if (entry.memo.ValidForDependencyPolicy(flags.resource_dynamic_deps)) {
                    entry.memo.PrepareHashFilterMetadata();
                }
            }
            for (size_t i = 0; i < count; ++i) {
                if (m_entries[i].memo.ValidForDependencyPolicy(flags.resource_dynamic_deps)) {
                    AssignDependencyGroup(i, flags.resource_dynamic_deps);
                }
            }
            m_dependency_groups_ready = true;
        } else if (!hash_filter) {
            m_dependency_groups_ready = false;
        }
        if (m_trace_enabled) {
            m_trace_resident.clear();
            for (const auto& resident: m_entries) m_trace_resident.push_back(resident.memo.CaptureRegisterTraceState());
        }
        size_t index = m_last_hit < count ? m_last_hit : 0;
        std::array<uint64_t, 16> group_hashes;
        uint16_t ready_groups = 0;
        for (size_t step = 0; step < count; ++step) {
            auto& entry = m_entries[index];
            if (hash_filter && entry.dependency_group < group_hashes.size() &&
                entry.memo.CanHashFilter(plan, runtime, flags.resource_dynamic_deps)) {
                auto& stats = ResourceHashFilterDetail::counters;
                ++stats.eligible;
                const auto group = entry.dependency_group;
                const auto mask = static_cast<uint16_t>(1u << group);
                if ((ready_groups & mask) == 0) {
                    group_hashes[group] = entry.memo.CurrentRegisterStateHash(runtime.user_data);
                    ready_groups |= mask;
                }
                if (group_hashes[group] != entry.memo.StoredRegisterStateHash()) {
                    ++stats.rejected;
                    if (step == 0) m_last_miss = Miss::Registers;
                    if (++index == count) index = 0;
                    continue;
                }
                ++stats.matched; // Hash match only; exact checks/read validation still follow.
            }
            if (!entry.memo.Refresh(plan, runtime, entry.snapshot, entry.specialization,
                                    true, reused, true, false)) {
                if (step == 0) m_last_miss = entry.memo.LastMiss();
                if (++index == count) index = 0;
                continue;
            }
            if (!reused && hash_filter) AssignDependencyGroup(index, flags.resource_dynamic_deps);
            if (m_last_hit != index || m_output != &snapshot || m_specialization != &specialization) {
                snapshot = entry.snapshot;
                specialization = entry.specialization;
            }
            // Probe leaves cached push data alone: update the caller once rather
            // than copying all SGPRs into both the entry and output on every hit.
            snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
            m_last_hit = index;
            m_output = &snapshot;
            m_specialization = &specialization;
            m_last_miss = Miss::None;
            return true;
        }
        if (count == 0) m_last_miss = Miss::Cold;
        // Recycle the victim in place. Its dependency list and captured-read storage
        // survive misses instead of rebuilding/freeing them for every object.
        const size_t selected = count < m_capacity ? count : m_next++ % m_capacity;
        if (count < m_capacity) m_entries.emplace_back();
        auto& entry = m_entries[selected];
        if (m_trace_enabled && count == m_capacity) {
            ++m_trace_sequence; // Distance bins count intervening eviction events only.
            RecordEvicted(entry.memo.CaptureRegisterTraceState());
        }
        if (!entry.memo.Refresh(plan, runtime, entry.snapshot, entry.specialization, true, reused)) {
            m_entries.clear();
            m_last_hit = SIZE_MAX;
            return false;
        }
        if (m_trace_enabled && count == m_capacity) TraceMaterialized(entry.memo.CaptureRegisterTraceState());
        if (entry.memo.RecordedWords() > 4096) {
            snapshot = std::move(entry.snapshot);
            specialization = std::move(entry.specialization);
            m_entries.erase(m_entries.begin() + selected);
            m_last_hit = SIZE_MAX;
            return true;
        }
        if (hash_filter) AssignDependencyGroup(selected, flags.resource_dynamic_deps);
        snapshot = entry.snapshot;
        specialization = entry.specialization;
        m_last_hit = selected;
        m_output = &snapshot;
        m_specialization = &specialization;
        return true;
    }
private:
    ShaderRecompiler::IR::ResourceSnapshot m_diag_snapshot;
    ShaderRecompiler::IR::ResourceSpecialization m_diag_specialization;
    void AssignDependencyGroup(size_t selected, bool dynamic_dependencies) {
        auto& entry = m_entries[selected];
        entry.memo.PrepareHashFilterMetadata();
        dynamic_dependencies = entry.memo.UsesDynamicDependencies();
        std::array<bool, 16> used{};
        for (size_t i = 0; i < m_entries.size(); ++i) {
            if (i == selected) continue;
            const auto& other = m_entries[i];
            if (!other.memo.HashFilterPolicyValid(dynamic_dependencies) ||
                other.dependency_group >= used.size()) continue;
            if (entry.memo.SameRegisterDependencySet(other.memo)) {
                entry.dependency_group = other.dependency_group;
                return;
            }
            used[other.dependency_group] = true;
        }
        for (size_t i = 0; i < used.size(); ++i) {
            if (!used[i]) { entry.dependency_group = static_cast<uint8_t>(i); return; }
        }
        entry.dependency_group = 0; // Unreachable while the bank is capped at 16.
    }
    struct TraceRecord { ResourceMemoEntry::RegisterTraceState state; uint64_t sequence = 0; };
    static bool TraceFromEnvironment() {
        static const bool enabled = [] { const char* value = std::getenv("KYTY_RESOURCE_REUSE_TRACE");
            return value != nullptr && value[0] == '1' && value[1] == '\0'; }();
        return enabled;
    }
    void RecordEvicted(ResourceMemoEntry::RegisterTraceState state) {
        if (state.plan == nullptr || state.dynamic_dependencies != m_trace_policy) return;
        auto& counters = ResourceReuseTraceDetail::GetCounters();
        counters.evictions.fetch_add(1, std::memory_order_relaxed);
        for (auto& record: m_trace_records) {
            if (record.state == state) { record.sequence = m_trace_sequence; return; }
        }
        if (m_trace_records.size() < 128) m_trace_records.push_back({std::move(state), m_trace_sequence});
        else { m_trace_records[m_trace_next] = {std::move(state), m_trace_sequence}; m_trace_next = (m_trace_next + 1) % 128; }
    }
    void TraceMaterialized(ResourceMemoEntry::RegisterTraceState state) {
        if (state.plan == nullptr || state.dynamic_dependencies != m_trace_policy) return;
        auto& counters = ResourceReuseTraceDetail::GetCounters();
        counters.checked.fetch_add(1, std::memory_order_relaxed);
        if (std::ranges::find(m_trace_resident, state) != m_trace_resident.end()) return;
        uint64_t newest = 0;
        for (const auto& record: m_trace_records) if (record.state == state) newest = std::max(newest, record.sequence);
        if (newest == 0) return;
        counters.matched.fetch_add(1, std::memory_order_relaxed);
        const uint64_t distance = m_trace_sequence - newest;
        // Disjoint bins count intervening eviction events, not LRU reuse distance.
        if (distance <= 16) counters.distance16.fetch_add(1, std::memory_order_relaxed);
        else if (distance <= 32) counters.distance32.fetch_add(1, std::memory_order_relaxed);
        else if (distance <= 64) counters.distance64.fetch_add(1, std::memory_order_relaxed);
        else if (distance <= 128) counters.distance128.fetch_add(1, std::memory_order_relaxed);
    }
    struct Entry {
        ResourceMemoEntry memo;
        ShaderRecompiler::IR::ResourceSnapshot snapshot;
        ShaderRecompiler::IR::ResourceSpecialization specialization;
        uint8_t dependency_group = UINT8_MAX;
    };
    std::vector<Entry> m_entries;
    size_t m_capacity;
    size_t m_last_hit = SIZE_MAX, m_next = 0;
    const void* m_output = nullptr;
    const void* m_specialization = nullptr;
    bool m_dependency_groups_ready = false;
    bool m_trace_enabled = false, m_trace_policy_valid = false, m_trace_policy = false;
    uint64_t m_trace_sequence = 0;
    size_t m_trace_next = 0;
    std::vector<TraceRecord> m_trace_records;
    std::vector<ResourceMemoEntry::RegisterTraceState> m_trace_resident;
    Miss m_last_miss = Miss::None;
};

} // namespace Libs::Graphics
#endif
