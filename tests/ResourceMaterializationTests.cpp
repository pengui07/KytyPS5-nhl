#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "common/runtimeFlags.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/host_gpu/renderer/pipeline/resourceMemo.h"
#include "graphics/shader/recompiler/ir/passes/ResourceFixture.h"
#include "graphics/shader/recompiler/ir/passes/ResourcePlanFixture.h"
#include "graphics/shader/recompiler/ir/passes/SrtProf.h"

#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::Program SrtProgram(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return program;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan SrtPlan(uint64_t address) {
  auto program = SrtProgram(address);
  return Libs::Graphics::ShaderRecompiler::IR::ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestCompiledDescriptorGatherMatchesWalker() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.user_data_base = 8;
  auto &block = AddValueBlock(program);
  auto &reg8 = block.AppendNewInst(ValueOpcode::GetUserData,
                                   {Value(static_cast<ScalarReg>(8))});
  auto &reg10 = block.AppendNewInst(ValueOpcode::GetUserData,
                                    {Value(static_cast<ScalarReg>(10))});
  DescriptorSource descriptor;
  descriptor.dword_count = 8;
  descriptor.dwords = {Value(&reg8), Value(0x12345678u), Value(&reg10), Value(&reg8),
                      Value::F32(1.25f), Value(uint64_t{0xfedcba9876543210u}),
                      Value(true), Value(0u)};
  program.descriptor_sources.push_back(descriptor);
  auto plan = ExtractResourcePlan(program);
  Check(plan.descriptor_gathers.size() == 1 && plan.descriptor_gathers[0].valid,
        "literal/user-data descriptor did not compile a gather");
  const auto compare = [&](std::span<const uint32_t> words, bool expected_success) {
    const SrtRuntime runtime{.user_data = words};
    DescriptorValue fast, slow;
    bool fast_ok;
    { SrtWalker walker(plan, runtime); fast_ok = walker.EvaluateDescriptor(0, fast); }
    plan.descriptor_gathers[0].valid = false;
    bool slow_ok;
    { SrtWalker walker(plan, runtime); slow_ok = walker.EvaluateDescriptor(0, slow); }
    plan.descriptor_gathers[0].valid = true;
    Check(fast_ok == expected_success && fast_ok == slow_ok && fast == slow,
          "compiled gather changed descriptor values or failure behavior");
  };
  std::array<uint32_t, 3> words{0x100u, 0x200u, 0x300u};
  compare(words, true);
  words = {0x400u, 0x500u, 0x600u};
  compare(words, true);
  compare(std::span(words).first(2), false);
  // Store absolute register indices; changing the plan's base must not leave
  // stale relative offsets compiled into the gather.
  plan.user_data_base = 10;
  compare(words, false);
  plan.user_data_base = 0;
  std::array<uint32_t, 11> rebased{};
  rebased[8] = 0x700u;
  rebased[10] = 0x800u;
  compare(rebased, true);
  DescriptorValue masked;
  {
    SrtWalker walker(plan, {.user_data = rebased}, {}, nullptr,
                     plan.descriptor_sources[0].dwords[0]);
    Check(walker.EvaluateDescriptor(0, masked) && masked.dwords[0] == 1u &&
              masked.dwords[3] == 1u && masked.dwords[2] == 0x800u,
          "compiled gather bypassed the active-mask evaluation context");
  }
  auto memory_plan = SrtPlan(0x1000u);
  Check(memory_plan.descriptor_gathers[0].valid &&
            memory_plan.descriptor_gathers[0].needs_flat,
        "SRT descriptor lost its refresh requirement");
}

void TestFlatDescriptorGatherRequiresOwnCompleteRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x1000u);
  struct Reads { uint32_t value = 0x100u; uint32_t count = 0; bool fail = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        auto& reads = *static_cast<Reads*>(data);
        ++reads.count;
        if (reads.fail || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }, .userdata = &reads};
  DescriptorValue descriptor;
  std::vector<uint32_t> flat;
  {
    SrtWalker walker(plan, runtime);
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == reads.value &&
              reads.count == 1,
          "unrefreshed flat gather bypassed ordinary memory evaluation");
  }
  for (const auto value: {0x200u, 0x300u}) {
    reads.value = value;
    reads.count = 0;
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(flat) && reads.count == 1 && flat[0] == value,
          "SRT refresh retained values from a prior evaluation session");
    // Indirect descriptor materialization can append and reallocate this vector.
    flat.reserve(flat.capacity() + 128);
    flat.push_back(0xdeadbeefu);
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == value &&
              reads.count == 1,
          "flat descriptor gather lost refreshed values after vector reallocation");
    flat.clear();
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == value &&
              reads.count == 1,
          "short flat storage did not safely fall back to the same-session memo");
  }
  reads.fail = true;
  {
    SrtWalker walker(plan, runtime);
    Check(!walker.RefreshFlatBuffer(flat) && !walker.EvaluateDescriptor(0, descriptor),
          "failed refresh reused a prior session's flat descriptor");
  }
  reads.fail = false;
  {
    const auto mask = plan.descriptor_sources[0].dwords[0];
    SrtWalker walker(plan, runtime, {}, nullptr, mask);
    Check(walker.RefreshFlatBuffer(flat) && walker.EvaluateDescriptor(0, descriptor) &&
              descriptor.dwords[0] == 1u,
          "flat gather bypassed active-mask semantics");
  }
  // A clean writable-descriptor session must not consume another walker's
  // ordinary flattened values. Its own strict reader supplies the descriptor.
  const SrtRuntime mixed{
      .read_memory = runtime.read_memory, .userdata = &reads,
      .read_specialization_memory = +[](void*, uint64_t address, std::span<uint32_t> words) {
        if (address != 0x1000u || words.size() != 1) return false;
        words[0] = 0x555u;
        return true;
      }};
  {
    SrtWalker ordinary(plan, mixed);
    Check(ordinary.RefreshFlatBuffer(flat), "ordinary SRT refresh failed");
    SrtWalker clean(plan, CleanRuntime(mixed));
    Check(clean.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == 0x555u,
          "unrefreshed clean walker consumed ordinary flat values");
  }
  // Control-flow refresh may leave unreachable slots zero. Such sessions must
  // retain the interpreter even after a successful refresh.
  plan.control_flow.emplace_back();
  plan.control_flow[0].srt_reads.push_back(0);
  plan.control_flow[0].sources.push_back(0);
  {
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(flat), "control-flow SRT refresh failed");
    flat[0] = 0xbadu;
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == reads.value,
          "control-flow session used the unconditional flat gather");
  }
}

void TestResourceMemoTracksMemoryAndRuntime() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  uint32_t word = 0x12345678u;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&word));
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemo memo;
  SrtRuntime runtime{};
  bool reused = true;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "memo first refresh reused an absent snapshot");
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "unchanged memory-backed SRT was not reused");
  word = 0xabcdef01u;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.flattened_srt[0] == word,
        "same-pointer SRT update reused stale values");
  runtime.shader_base = 0x1000;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "shader-base change reused an old snapshot");
  std::array<uint32_t, 3> groups{1, 2, 3};
  runtime.workgroup_counts = groups;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "new workgroup counts reused an old snapshot");
  groups[0] = 4;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "same-pointer workgroup update reused stale values");
  Check(memo.Refresh(plan, runtime, snapshot, specialization, false, reused) && !reused,
        "disabled memo reused cached state");
  auto user_plan = UserDataBufferPlan();
  std::array<uint32_t, 1> user{0x8000};
  runtime.user_data = user;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && !reused,
        "different resource plan reused an old snapshot");
  user[0] = 0x9000;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.buffers[0].dwords[0] == user[0] && snapshot.user_data[0] == user[0],
        "same-pointer register update reused stale descriptor or push payload");
  std::array<uint32_t, 2> payload{user[0], 7};
  runtime.user_data = payload;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && !reused,
        "changed user-data length reused an old snapshot");
  payload[1] = 9;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && reused &&
            snapshot.user_data[1] == 9 && snapshot.buffers[0].dwords[0] == payload[0],
        "non-resource register update failed to reuse resources or update push data");
}

void TestResourceMemoHitVerifier() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *flag = std::getenv("KYTY_RESOURCE_MEMO_VERIFY");
  if (flag == nullptr || flag[0] != '1') return;

  uint32_t value = 0x12345678u;
  auto plan = SrtPlan(0x1000u);
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.ordinary;
        if (address != 0x1000u || words.size() != 1u) return false;
        words[0] = 0x12345678u;
        return true;
      },
      .userdata = &reads};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemoEntry memo;
  bool reused = false;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "memo verifier failed to establish a cached resource");
  auto calls = reads.ordinary;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused &&
            memo.VerifyChecks() == 1u && memo.VerifyMatches() == 1u &&
            reads.ordinary == calls + 1u,
        "matched memo verification performed extra live reads");

  snapshot.flattened_srt[0] ^= 1u;
  calls = reads.ordinary;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
            memo.LastMiss() == Libs::Graphics::ResourceMemoEntry::Miss::VerifyMismatch &&
            memo.VerifyMismatches() == 1u && reads.ordinary == calls + 2u &&
            snapshot.flattened_srt[0] == value,
        "memo verifier did not invalidate and rematerialize a corrupted snapshot");

  auto unknown_plan = SrtPlan(0x1000u);
  Libs::Graphics::ResourceMemoEntry unknown_memo;
  ResourceSnapshot unknown_snapshot;
  ResourceSpecialization unknown_specialization;
  Check(unknown_memo.Refresh(unknown_plan, runtime, unknown_snapshot,
                            unknown_specialization, true, reused) && !reused,
        "unknown-read fixture failed to establish a cached resource");
  auto address_changed = false;
  for (auto &inst: unknown_plan.value_storage) {
    if (inst.GetOpcode() != ValueOpcode::GetAddressResource) continue;
    inst.SetArg(0, Value(0x2000u));
    address_changed = true;
    break;
  }
  Check(address_changed, "unknown-read fixture could not find its address node");
  unknown_plan.srt_tapes = {};
  unknown_plan.cfg_srt_tapes = {};
  reads.ordinary = 0;
  const SrtRuntime unknown_runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        if (words.size() != 1u || (address != 0x1000u && address != 0x2000u)) return false;
        words[0] = address == 0x1000u ? 0x12345678u : 0x87654321u;
        return true;
      },
      .userdata = &reads};
  Check(unknown_memo.Refresh(unknown_plan, unknown_runtime, unknown_snapshot,
                            unknown_specialization, true, reused) && !reused &&
            unknown_memo.LastMiss() == Libs::Graphics::ResourceMemoEntry::Miss::VerifyInconclusive &&
            unknown_memo.VerifyInconclusive() == 1u && reads.ordinary == 2u &&
            unknown_snapshot.flattened_srt[0] == 0x87654321u,
        "uncaptured dependency was not treated as inconclusive and rematerialized");

  ResourcePlan domains;
  domains.stage = Libs::Graphics::ShaderType::Compute;
  domains.srt_plan_complete = domains.resource_tracking_complete = true;
  domains.requires_specialization_memory = true;
  domains.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  auto &handle = domains.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x3000u));
  handle.SetArg(1, Value(0u));
  for (uint32_t i = 0; i < 2u; ++i) {
    auto &read = domains.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = 0u});
    read.SetArg(0, Value(&handle));
    read.SetArg(1, Value(0u));
    read.SetArg(2, Value(0u));
    read.SetArg(3, Value(true));
    domains.srt_reads.push_back({Value(&read), i});
    domains.clean_flat_slots.push_back(i == 0u ? 0u : 1u);
  }
  struct DomainsReads { uint32_t ordinary = 0; uint32_t strict = 0; } domain_reads;
  const SrtRuntime domain_runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<DomainsReads *>(data)->ordinary;
        if (address != 0x3000u || words.size() != 1u) return false;
        words[0] = 0x11111111u;
        return true;
      },
      .userdata = &domain_reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<DomainsReads *>(data)->strict;
        if (address != 0x3000u || words.size() != 1u) return false;
        words[0] = 0x22222222u;
        return true;
      }};
  Libs::Graphics::ResourceMemoEntry domain_memo;
  ResourceSnapshot domain_snapshot;
  ResourceSpecialization domain_specialization;
  Check(domain_memo.Refresh(domains, domain_runtime, domain_snapshot,
                            domain_specialization, true, reused) && !reused &&
            domain_snapshot.flattened_srt == std::vector<uint32_t>{0x11111111u, 0x22222222u},
        "strict/ordinary setup failed to produce distinct same-address values");
  const auto ordinary_calls = domain_reads.ordinary;
  const auto strict_calls = domain_reads.strict;
  Check(domain_memo.Refresh(domains, domain_runtime, domain_snapshot,
                            domain_specialization, true, reused) && reused &&
            domain_memo.VerifyMatches() == 1u &&
            domain_reads.ordinary == ordinary_calls + 1u &&
            domain_reads.strict == strict_calls + 1u,
        "strict and ordinary same-address reads were merged during replay");
}

void TestResourceMemoTracksConsumedRegisters() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const bool dynamic = Common::RuntimeFlags::Current().resource_dynamic_deps;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.capture_specialization_reads = true; // Exercise the nested memory-capture runtime wrapper.
  const auto user = [&](uint32_t index) {
    auto &inst = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
    inst.SetArg(0, Value(static_cast<ScalarReg>(index)));
    return Value(&inst);
  };
  const auto guard = user(2);
  const auto selected = user(0);
  const auto inactive = user(1);
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x1000u));
  handle.SetArg(1, Value(0u));
  auto &memory_read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  memory_read.SetFlags(MemoryFlags{.index = 0u});
  memory_read.SetArg(0, Value(&handle));
  memory_read.SetArg(1, Value(0u));
  memory_read.SetArg(2, Value(0u));
  memory_read.SetArg(3, Value(true));
  plan.srt_reads.push_back({Value(&memory_read), 0u});
  plan.descriptor_sources.resize(2);
  for (auto &source: plan.descriptor_sources) {
    source.dword_count = 4;
    source.dwords.fill(Value(0u));
  }
  plan.descriptor_sources[0].dwords[0] = selected;
  plan.descriptor_sources[1].dwords[0] = inactive;
  plan.descriptor_gathers.resize(2);
  for (uint32_t i = 0; i < plan.descriptor_gathers.size(); ++i) {
    auto &gather = plan.descriptor_gathers[i];
    gather.valid = true;
    gather.words[0] = {.value = i == 0 ? 0u : 1u,
                       .kind = ResourcePlan::DescriptorGather::Word::Kind::UserData};
  }
  plan.info.buffers = {{.source = 0}, {.source = 1}};
  plan.control_flow.resize(3);
  plan.control_flow[0].condition = guard;
  plan.control_flow[0].successors = {1u, 2u};
  plan.control_flow[1].srt_reads = {0u};
  plan.control_flow[1].sources = {0u};
  plan.control_flow[2].sources = {1u};

  std::array<uint32_t, 4> regs{0x8000u, 0x9000u, 1u, 17u};
  std::vector<uint32_t> forwarded;
  struct Memory { uint32_t value = 0x12345678u; } memory;
  SrtRuntime runtime{.user_data = regs,
                     .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
                       if (address != 0x1000u || words.size() != 1u) return false;
                       words[0] = static_cast<Memory *>(data)->value;
                       return true;
                     },
                     .userdata = &memory,
                     .observe_user_data = +[](void *data, uint32_t index) {
                       static_cast<std::vector<uint32_t> *>(data)->push_back(index);
                     },
                     .user_data_observer_userdata = &forwarded};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemoEntry memo;
  bool reused = false;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.buffers[0].dwords[0] == regs[0] && snapshot.buffers[1].dwords[0] == 0u,
        "CFG dependency fixture did not materialize only the active descriptor");
  Check(memo.StaticRegisterDependencyCount() == 3u &&
            memo.RegisterDependencyCount() == (dynamic ? 2u : 3u),
        "resource memo reported an incorrect static/consumed register set");

  regs[1] = 0xdead0000u; // Descriptor in the inactive successor must not invalidate in dynamic mode.
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) &&
            reused == dynamic,
        "inactive CFG descriptor register dependency was not handled by the selected mode");
  regs[3] = 99u; // Push data outside every resource expression is always refreshed on a hit.
  forwarded.clear();
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused &&
            snapshot.user_data[3] == 99u,
        "non-resource push data invalidated resources or was not refreshed");
  Check(forwarded.size() == memo.RegisterDependencyCount(),
        "resource memo cache hit did not forward its validated dependencies");

  regs[0] = 0x8100u;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.buffers[0].dwords[0] == regs[0],
        "selected descriptor register change reused a stale resource snapshot");
  regs[2] = 0u; // The branch predicate is a dependency even when active roots change.
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.buffers[1].dwords[0] == regs[1],
        "CFG guard change did not invalidate and select the other resource");

  // The shared ResourcePlan's active_sources belong to its most recent materialization,
  // which may be a different cached entry. Verify against each entry's own path.
  Libs::Graphics::ResourceMemo alternating_memo(2);
  regs[2] = 1u;
  Check(alternating_memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "alternating CFG memo failed to capture its first path");
  regs[2] = 0u;
  Check(alternating_memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "alternating CFG memo failed to capture its second path");
  regs[2] = 1u;
  Check(alternating_memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused &&
            snapshot.buffers[0].dwords[0] == regs[0],
        "alternating CFG cache entry compared against another entry's active path");
}

void TestResourceMemoLazySelectDependencies() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const bool dynamic = Common::RuntimeFlags::Current().resource_dynamic_deps;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &selected = block.AppendNewInst(ValueOpcode::GetUserData,
                                       {Value(static_cast<ScalarReg>(0))});
  auto &unused = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(1))});
  auto &selection = block.AppendNewInst(ValueOpcode::SelectU32,
                                         {Value(true), Value(&selected), Value(&unused)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(&selection);
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> regs{0x8000u, 0xdead0000u};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemoEntry memo;
  bool reused = false;
  Check(memo.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization,
                     true, reused) && !reused && snapshot.buffers[0].dwords[0] == regs[0],
        "lazy-select resource fixture failed to materialize its selected arm");
  regs[1] = 0xbad00000u;
  Check(memo.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization,
                     true, reused) && reused == dynamic,
        "lazy-select unused arm was included in dynamic resource dependencies");
}

void TestResourceMemoAlternatingObjects() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Alternating objects must reuse both entries, and still reject changed memory.
  auto alternating_plan = UserDataBufferPlan();
  ResourceSnapshot alternating_snapshot;
  ResourceSpecialization alternating_specialization;
  Libs::Graphics::ResourceMemo alternating_memo(16);
  std::array<uint32_t, 2> registers{0x8000, 7};
  SrtRuntime alternating_runtime{.user_data = registers};
  bool alternating_reused = false;
  for (uint32_t address: {0x8000u, 0x9000u, 0x8000u, 0x9000u}) {
    registers[0] = address;
    Check(alternating_memo.Refresh(alternating_plan, alternating_runtime,
          alternating_snapshot, alternating_specialization, true, alternating_reused),
          "alternating memo refresh failed");
    Check(alternating_snapshot.buffers[0].dwords[0] == address,
          "multi-entry memo restored the wrong object's descriptor");
  }
  Check(alternating_reused, "alternating objects did not reuse multiple entries");
  registers[1] = 13;
  Check(alternating_memo.Refresh(alternating_plan, alternating_runtime,
        alternating_snapshot, alternating_specialization, true, alternating_reused) &&
        alternating_reused && alternating_snapshot.user_data[1] == 13,
        "multi-entry hit failed to refresh non-resource push registers");

}

void TestSingleResourceMemoChangesOutput() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  std::array<uint32_t, 2> registers{0x8000, 7};
  SrtRuntime runtime{.user_data = registers};
  ResourceSnapshot first, second;
  ResourceSpecialization first_specialization, second_specialization;
  Libs::Graphics::ResourceMemo memo(1);
  bool reused = true;
  Check(memo.Refresh(plan, runtime, first, first_specialization, true, reused) && !reused,
        "single memo reused its absent output");
  Check(memo.Refresh(plan, runtime, second, second_specialization, true, reused) && !reused &&
            second.buffers == first.buffers && second.user_data == first.user_data &&
            second_specialization == first_specialization,
        "single memo treated an empty replacement output as a cached snapshot");
  Check(memo.Refresh(plan, runtime, second, second_specialization, true, reused) && reused,
        "single memo did not retain the current output");
  registers[1] = 19;
  Check(memo.Refresh(plan, runtime, second, second_specialization, true, reused) && reused &&
            second.user_data[1] == 19 && first.user_data[1] == 7,
        "single memo failed to update push data independently of the old output");
  registers[0] = 0x9000;
  Check(memo.Refresh(plan, runtime, first, first_specialization, true, reused) && !reused &&
            first.buffers[0].dwords[0] == 0x9000 && second.buffers[0].dwords[0] == 0x8000,
        "single memo refreshed the wrong output or reused a changed resource register");
  // The runtime capacity switch (SetCapacity) of the claude/nhl27-perf memo is gone: the merged tree
  // uses the pengui07 memo, whose capacity is fixed at construction (KYTY_RESOURCE_MULTI_CACHE).
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(&dword)));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestNativeNestedSrtRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::array<uint32_t, 3> first{0u, 0x12345678u, 0xabcdef01u};
  std::array<uint32_t, 3> second{0u, 0x87654321u, 0x10fedcbau};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address) | 3u,
                                  static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
  };
  auto table = pointer_words(first.data());
  const auto user_data = pointer_words(table.data());
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 2;
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto low = emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))});
  const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto read = [&](Value address, uint32_t offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {address, Value(offset), Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto pointer_low = read(root, 0u);
  const auto pointer_high = read(root, 4u);
  auto *leaf = program.srt_reads[0].value.Instruction();
  leaf->Arg(0).Instruction()->SetArg(0, pointer_low);
  leaf->Arg(0).Instruction()->SetArg(1, pointer_high);
  leaf->SetArg(1, Value(7u));
  read(leaf->Arg(0), 11u);
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_code != nullptr, "nested user-data tree did not compile an x86 walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const SrtRuntime runtime{.user_data = user_data};
  {
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(snapshot.flattened_srt),
          "native SRT refresh failed");
    const auto loaded = first[1];
    first[1] ^= 0xffffffffu;
    uint32_t value = 0;
    Check(walker.Evaluate(plan.srt_reads[0].value, value) && value == loaded,
          "descriptor evaluation reloaded a value already copied by the native walker");
    first[1] = loaded;
  }
  for (const auto *data : {&first, &second}) {
    table = pointer_words(data->data());
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt ==
                  std::vector<uint32_t>{(*data)[1], table[0], table[1], (*data)[2]},
          "native SRT refresh lost pointer masking, alignment, or current values");
  }
  Check(!MaterializeResources(plan, {.user_data = std::span(user_data).first(1)},
                              snapshot, specialization),
        "native SRT walk accepted truncated user data");
  const std::array<uint32_t, 2> fake_user_data{0x1003u, 0xabcd0000u};
  uint32_t reads = 0;
  const SrtRuntime callback_runtime{
      .user_data = fake_user_data,
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++*static_cast<uint32_t *>(data);
        switch (address) {
          case 0x1000u: words[0] = 0x2003u; break;
          case 0x1004u: words[0] = 0xabcd0000u; break;
          case 0x2004u: words[0] = 0x1234u; break;
          case 0x2008u: words[0] = 0x5678u; break;
          default: return false;
        }
        return true;
      },
      .userdata = &reads};
  Check(MaterializeResources(plan, callback_runtime, snapshot, specialization) && reads == 4 &&
            snapshot.flattened_srt ==
                std::vector<uint32_t>{0x1234u, 0x2003u, 0xabcd0000u, 0x5678u},
        "native-eligible SRT bypassed its callback or repeated nested pointer reads");
  leaf->SetArg(0, root);
  leaf->SetArg(1, Value(0u));
  program.srt_reads.resize(1);
  for (const int32_t offset : {-1, 7}) {
    program.memory_info[0].offset = static_cast<uint32_t>(offset);
    auto bounded = ExtractResourcePlan(program);
    auto base = pointer_words(second.data() + 1);
    Check(bounded.srt_code != nullptr &&
              MaterializeResources(bounded, {.user_data = base}, snapshot, specialization) &&
              snapshot.flattened_srt[0] == second[offset < 0 ? 0 : 2],
          "native SRT applied a signed immediate offset incorrectly");
    base = offset < 0 ? std::array<uint32_t, 2>{0u, 0u}
                      : std::array<uint32_t, 2>{0xfffffffcu, 0xffffu};
    Check(!MaterializeResources(bounded, {.user_data = base}, snapshot, specialization),
          "native SRT accepted address underflow or 48-bit overflow");
  }
  high.Instruction()->SetArg(0, Value(ScalarReg(6)));
  Check(ExtractResourcePlan(program).srt_code == nullptr,
        "native SRT accepted nonadjacent user-data pointer words");
  high.Instruction()->SetArg(0, Value(ScalarReg(5)));
  program.memory_info[0].offset = 0;
  program.srt_reads.push_back({pointer_low, 1});
  program.srt_reads.push_back({pointer_high, 2});
  leaf->SetArg(0, emit(ValueOpcode::GetAddressResource, {pointer_low, pointer_high}));
  pointer_high.Instruction()->SetArg(1, Value(8u));
  Check(ExtractResourcePlan(program).srt_code == nullptr,
        "native SRT accepted nonadjacent nested pointer words");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtProgram(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestFailedRuntimeReadIsMemoizedPerRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
      {Value(0x1000u), Value(0u)});
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  read.SetFlags(MemoryFlags{.index = 0});
  struct Reads { uint32_t count = 0; bool available = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.count;
        if (!state.available) return false;
        words[0] = 42u;
        return true;
      },
      .userdata = &reads};
  uint32_t result = 0;
  {
    SrtWalker walker(program, runtime);
    for (uint32_t i = 0; i < 32; ++i)
      Check(!walker.Evaluate(Value(&read), result), "unavailable runtime read was accepted");
    Check(reads.count == 1, "failed runtime read was retried within one refresh");
    // Another evaluation context must not inherit the failed result.
    reads.available = true;
    SrtWalker nested(program, runtime);
    Check(nested.Evaluate(Value(&read), result) && result == 42u && reads.count == 2,
          "nested runtime evaluation inherited a failed result");
  }
  Check(SrtWalker(program, runtime).Evaluate(Value(&read), result) &&
            result == 42u && reads.count == 3,
        "next refresh retained a failed runtime result");
}
void TestCyclicRuntimeFailureCanRecover() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  auto &root = block.AppendNewInst(ValueOpcode::ConditionRef, {Value(false)});
  auto &inverse = block.AppendNewInst(ValueOpcode::LogicalNot, {Value(&root)});
  auto &either = block.AppendNewInst(ValueOpcode::LogicalOr, {Value(&inverse), Value(true)});
  root.SetArg(0, Value(&either));
  SrtWalker walker(program, {});
  uint32_t result = 0;
  Check(walker.Evaluate(Value(&root), result) && result == 1u,
        "short-circuit evaluation failed to resolve a cyclic predicate");
  Check(walker.Evaluate(Value(&inverse), result) && result == 0u,
        "temporary cyclic failure prevented later predicate evaluation");
}
void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t strict = 0; uint32_t ordinary = 0; bool clean = true; } reads;
  const SrtRuntime runtime{
      .read_memory = [](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 999;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = ExtractResourcePlan(program);
    Check(plan.control_flow.empty() && plan.requires_specialization_memory &&
              plan.capture_specialization_reads,
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.strict == 1 && reads.ordinary == 0 &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "vector descriptor input was not read and captured exactly once");
    reads.clean = false;
    reads.strict = 0;
    Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.strict == 1 && reads.ordinary == 0,
          "dirty vector descriptor input fell back to an ordinary memory read");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  program.descriptor_sources.push_back({.dwords = {Value(&inactive)}, .dword_count = 1});
  reads.strict = 0;
  uint32_t result = 99;
  {
    const auto plan = ExtractResourcePlan(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "literal false EXEC read vector memory");
  }
  count.SetArg(4, Value(true));
  handle.SetArg(3, Value(0x204u));
  program.descriptor_sources.back().dwords[0] = Value(&count);
  {
    const auto plan = ExtractResourcePlan(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "invalid vector buffer format read memory");
  }
  handle.SetArg(3, Value(0x16204u));
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestGuardedVectorReadTapeFallback() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.user_data_base = 8;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.requires_specialization_memory = true;
  plan.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &condition = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
  condition.SetArg(0, Value(static_cast<ScalarReg>(8)));
  auto &guard = plan.value_storage.emplace_back(ValueOpcode::INotEqual32);
  guard.SetArg(0, Value(&condition));
  guard.SetArg(1, Value(0u));
  auto &handle = plan.value_storage.emplace_back(
      ValueOpcode::GetBufferResource);
  handle.SetArg(0, Value(0x1000u));
  handle.SetArg(1, Value(4u << 16u));
  handle.SetArg(2, Value(1u));
  handle.SetArg(3, Value(0x16204u));
  auto &load = plan.value_storage.emplace_back(ValueOpcode::LoadBufferU32);
  load.SetFlags(MemoryFlags{.index = 0});
  load.SetArg(0, Value(&handle));
  load.SetArg(1, Value(0u));
  load.SetArg(2, Value(0u));
  load.SetArg(3, Value(0u));
  load.SetArg(4, Value(&guard));
  plan.srt_reads.push_back({Value(&load), 0});
  plan.clean_flat_slots.resize(1u);

  struct ReadState {
    uint32_t calls = 0;
    uint32_t value = 0x76543210u;
  } reads;
  const auto reader = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &state = *static_cast<ReadState *>(data);
    ++state.calls;
    if (address != 0x1000u || words.size() != 1u) return false;
    words[0] = state.value;
    return true;
  };
  const SrtRuntime runtime{
      .user_data = {},
      .read_memory = reader,
      .userdata = &reads,
      .read_specialization_memory = reader,
  };
  std::array<uint32_t, 1> user_data{};
  auto guarded_runtime = runtime;
  guarded_runtime.user_data = user_data;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;

  user_data[0] = 0;
  const bool false_ok = MaterializeResources(plan, guarded_runtime, snapshot, specialization);
  Check(false_ok &&
            snapshot.flattened_srt.size() == 1u && snapshot.flattened_srt[0] == 0u &&
            reads.calls == 0u,
        "false vector-load guard touched memory or returned a nonzero value");

  user_data[0] = 1;
  const bool true_ok = MaterializeResources(plan, guarded_runtime, snapshot, specialization);
  Check(true_ok &&
            snapshot.flattened_srt.size() == 1u &&
            snapshot.flattened_srt[0] == reads.value && reads.calls == 1u,
        "true vector-load guard did not read the descriptor exactly once");
}

void TestTapeVerifyReplaysReaderDomains() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *verify = std::getenv("KYTY_SRT_TAPE_VERIFY");
  const char *tape = std::getenv("KYTY_SRT_TAPE");
  if (verify == nullptr || verify[0] != '1' || (tape != nullptr && tape[0] == '0')) return;

  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = true;
  plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  plan.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &scalar_handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  scalar_handle.SetArg(0, Value(0x2000u));
  scalar_handle.SetArg(1, Value(0u));
  auto &scalar_read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  scalar_read.SetFlags(MemoryFlags{.index = 0});
  scalar_read.SetArg(0, Value(&scalar_handle));
  scalar_read.SetArg(1, Value(0u));
  scalar_read.SetArg(2, Value(0u));
  scalar_read.SetArg(3, Value(true));
  auto &buffer_handle = plan.value_storage.emplace_back(ValueOpcode::GetBufferResource);
  buffer_handle.SetArg(0, Value(0x3000u));
  buffer_handle.SetArg(1, Value(4u << 16u));
  buffer_handle.SetArg(2, Value(1u));
  buffer_handle.SetArg(3, Value(0x16204u));
  auto &vector_read = plan.value_storage.emplace_back(ValueOpcode::LoadBufferU32);
  vector_read.SetFlags(MemoryFlags{.index = 1});
  vector_read.SetArg(0, Value(&buffer_handle));
  vector_read.SetArg(1, Value(0u));
  vector_read.SetArg(2, Value(0u));
  vector_read.SetArg(3, Value(0u));
  vector_read.SetArg(4, Value(true));
  plan.srt_reads.push_back({Value(&scalar_read), 0});
  plan.srt_reads.push_back({Value(&vector_read), 1});

  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.ordinary;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x12345678u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.strict;
        if (address != 0x3000u || words.size() != 1u) return false;
        words[0] = 0x87654321u;
        return true;
      }};
  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, runtime).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x12345678u, 0x87654321u} &&
            reads.ordinary == 1u && reads.strict == 1u,
        "tape verifier reread memory or crossed ordinary/strict reader domains");
  bool has_scalar_source = false;
  bool has_vector_source = false;
  for (const auto &op : plan.srt_tapes[0].ops) {
    has_scalar_source |= op.source == &scalar_read;
    has_vector_source |= op.source == &vector_read;
  }
  Check(has_scalar_source && has_vector_source,
        "Tape verifier lost source instruction provenance");

  reads = {};
  auto failed_runtime = runtime;
  failed_runtime.read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
    ++static_cast<Reads *>(data)->ordinary;
    return false;
  };
  Check(!SrtWalker(plan, failed_runtime).RefreshFlatBuffer(flat) &&
            reads.ordinary == 1u && reads.strict == 0u,
        "tape verifier changed failed-reader behavior");
}

void TestTapeVerifyMismatchFallbackAndUnmatchedReplay() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *verify = std::getenv("KYTY_SRT_TAPE_VERIFY");
  const char *tape = std::getenv("KYTY_SRT_TAPE");
  if (verify == nullptr || verify[0] != '1' || (tape != nullptr && tape[0] == '0')) return;
  const auto add_read = [](ResourcePlan &plan, uint64_t address, uint32_t index) {
    plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
    auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    handle.SetArg(0, Value(static_cast<uint32_t>(address)));
    handle.SetArg(1, Value(static_cast<uint32_t>(address >> 32u)));
    auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = index});
    read.SetArg(0, Value(&handle));
    read.SetArg(1, Value(0u));
    read.SetArg(2, Value(0u));
    read.SetArg(3, Value(true));
    return Value(&read);
  };
  struct MutatingRead {
    ResourcePlan *plan = nullptr;
    Value replacement;
    uint32_t calls = 0;
  };
  const auto reader = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &state = *static_cast<MutatingRead *>(data);
    ++state.calls;
    if (address != 0x4000u || words.size() != 1u) return false;
    words[0] = 0xabcdef01u;
    state.plan->srt_reads[0].value = state.replacement;
    return true;
  };

  ResourcePlan mismatch_plan;
  mismatch_plan.stage = Libs::Graphics::ShaderType::Compute;
  mismatch_plan.srt_plan_complete = true;
  mismatch_plan.resource_tracking_complete = true;
  const auto original = add_read(mismatch_plan, 0x4000u, 0u);
  mismatch_plan.srt_reads.push_back({original, 0u});
  mismatch_plan.srt_reads.push_back({original, 1u});
  auto &descriptor_expression = mismatch_plan.value_storage.emplace_back(ValueOpcode::IAdd32);
  descriptor_expression.SetArg(0, original);
  descriptor_expression.SetArg(1, Value(1u));
  DescriptorSource descriptor_source;
  descriptor_source.dword_count = 1;
  descriptor_source.dwords[0] = Value(&descriptor_expression);
  mismatch_plan.descriptor_sources.push_back(descriptor_source);
  MutatingRead mismatch_state{&mismatch_plan, Value(0x55aa55aau)};
  const SrtRuntime mismatch_runtime{.read_memory = reader, .userdata = &mismatch_state};
  std::vector<uint32_t> flat;
  SrtWalker mismatch_walker(mismatch_plan, mismatch_runtime);
  Check(mismatch_walker.RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x55aa55aau, 0xabcdef01u} && mismatch_state.calls == 1u,
        "tape verifier did not fall back to the recursive value after a synthetic mismatch");
  DescriptorValue descriptor;
  Check(mismatch_walker.EvaluateDescriptor(0u, descriptor) &&
            descriptor.dwords[0] == 0xabcdef02u && mismatch_state.calls == 1u,
        "mismatch fallback did not transfer the validated recursive memo into descriptor evaluation");

  ResourcePlan unmatched_plan;
  unmatched_plan.stage = Libs::Graphics::ShaderType::Compute;
  unmatched_plan.srt_plan_complete = true;
  unmatched_plan.resource_tracking_complete = true;
  const auto first = add_read(unmatched_plan, 0x4000u, 0u);
  const auto second = add_read(unmatched_plan, 0x5000u, 1u);
  unmatched_plan.srt_reads.push_back({first, 0u});
  MutatingRead unmatched_state{&unmatched_plan, second};
  const SrtRuntime unmatched_runtime{.read_memory = reader, .userdata = &unmatched_state};
  Check(SrtWalker(unmatched_plan, unmatched_runtime).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0xabcdef01u} && unmatched_state.calls == 1u,
        "unmatched replay was treated as a mismatch or reread live memory");
}

void TestTapeSeparatesCleanAndOrdinaryReadDomains(bool cfg = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x2000u));
  handle.SetArg(1, Value(0u));
  auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  read.SetFlags(MemoryFlags{.index = 0});
  read.SetArg(0, Value(&handle));
  read.SetArg(1, Value(0u));
  read.SetArg(2, Value(0u));
  read.SetArg(3, Value(true));
  plan.srt_reads.push_back({Value(&read), 0u});
  plan.srt_reads.push_back({Value(&read), 1u});
  plan.clean_flat_slots = {0u, 1u};
  if (cfg) {
    plan.control_flow.emplace_back();
    plan.control_flow[0].srt_reads = {0u, 1u};
  }

  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.ordinary;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x11111111u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.strict;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x22222222u;
        return true;
      }};
  SrtWalker clean(plan, CleanRuntime(runtime));
  SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x11111111u, 0x22222222u},
        "Tape merged ordinary and clean versions of one shared IR read");
  uint32_t ordinary_value = 0;
  uint32_t strict_value = 0;
  Check(walker.Evaluate(Value(&read), ordinary_value) && ordinary_value == 0x11111111u &&
            clean.Evaluate(Value(&read), strict_value) && strict_value == 0x22222222u &&
            reads.ordinary == 1u && reads.strict == 1u,
        "Tape published a read result into the wrong evaluator memo domain");
}

void TestVectorLoadUsesCleanDependencies(bool through_read_const, bool cfg = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  plan.memory_info.push_back({.kind = ResourceKind::Buffer, .data_dwords = 1,
                              .data_bits = 32});

  auto &scalar_handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  scalar_handle.SetArg(0, Value(0x2000u));
  scalar_handle.SetArg(1, Value(0u));
  auto &address_word = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  address_word.SetFlags(MemoryFlags{.index = 0});
  address_word.SetArg(0, Value(&scalar_handle));
  address_word.SetArg(1, Value(0u));
  address_word.SetArg(2, Value(0u));
  address_word.SetArg(3, Value(true));
  Value low(&address_word);
  if (through_read_const) {
    auto &srt = plan.value_storage.emplace_back(ValueOpcode::GetSrtResource);
    auto &read_const = plan.value_storage.emplace_back(ValueOpcode::ReadConst);
    read_const.SetArg(0, Value(&srt));
    read_const.SetArg(1, Value(0u));
    low = Value(&read_const);
  }

  auto &buffer_handle = plan.value_storage.emplace_back(ValueOpcode::GetBufferResource);
  buffer_handle.SetArg(0, low);
  buffer_handle.SetArg(1, Value(4u << 16u));
  buffer_handle.SetArg(2, Value(1u));
  buffer_handle.SetArg(3, Value(0x16204u));
  auto &vector_read = plan.value_storage.emplace_back(ValueOpcode::LoadBufferU32);
  vector_read.SetFlags(MemoryFlags{.index = 1});
  vector_read.SetArg(0, Value(&buffer_handle));
  vector_read.SetArg(1, Value(0u));
  vector_read.SetArg(2, Value(0u));
  vector_read.SetArg(3, Value(0u));
  vector_read.SetArg(4, Value(true));
  plan.srt_reads.push_back({Value(&address_word), 0u});
  plan.srt_reads.push_back({Value(&vector_read), 1u});
  if (cfg) {
    plan.control_flow.emplace_back();
    plan.control_flow[0].srt_reads = {0u, 1u};
  }

  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x3000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->strict;
        if (words.size() != 1u) return false;
        if (address == 0x2000u) words[0] = 0x4000u;
        else if (address == 0x3000u) words[0] = 0xaaaaaaaau;
        else if (address == 0x4000u) words[0] = 0xbbbbbbbbu;
        else return false;
        return true;
      }};
  SrtWalker clean(plan, CleanRuntime(runtime));
  SrtWalker walker(plan, runtime, {}, &clean);
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x3000u, 0xbbbbbbbbu},
        through_read_const
            ? "clean vector-load ReadConst dependency used the ordinary reader"
            : "clean vector-load address dependency used the ordinary reader");
}

void TestCfgTapeVerifierRejectsCorruption() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *verify = std::getenv("KYTY_SRT_TAPE_VERIFY");
  if (!Common::RuntimeFlags::Current().srt_cfg_tape || verify == nullptr || verify[0] != '1') return;
  const auto verifier_before = GetSrtTapeStats();
  auto plan = SrtPlan(0x2000u);
  const Value raw = plan.srt_reads[0].value;
  auto &sum = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
  sum.SetArg(0, raw); sum.SetArg(1, Value(1u));
  plan.srt_reads = {{Value(&sum), 0u}, {raw, 1u}};
  plan.clean_flat_slots = {0u, 1u};
  plan.control_flow.resize(1);
  plan.control_flow[0].srt_reads = {0u, 1u};
  plan.control_flow[0].sources = {0u};
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime {
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary; words[0] = 10u; return true;
      }, .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->strict; words[0] = 20u; return true;
      }};
  std::vector<uint32_t> flat;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    Check(walker.RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{11u, 20u},
          "CFG verifier warmup failed");
  }
  auto &tape = plan.cfg_srt_tapes[1];
  bool corrupted = false;
  for (const auto &op: tape.ops) {
    if (op.source == &sum) {
      tape.ops[op.args[1]].imm = 8u; // Invalid compiled arithmetic, original IR unchanged.
      corrupted = true;
      break;
    }
  }
  Check(corrupted, "CFG verifier fixture did not corrupt a supported compiled root");
  reads = {};
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    uint32_t value = 0;
    // This live read predates the capture; replay must receive its initial memo.
    Check(walker.Evaluate(raw, value) && value == 10u, "CFG verifier preloaded input failed");
    Check(walker.RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{11u, 20u} &&
              reads.ordinary == 1u && reads.strict == 1u &&
              tape.state == ResourcePlan::SrtTape::State::Unsupported,
          "CFG verifier accepted corrupt arithmetic, reread inputs, or failed to disable plan");
    Check(walker.Evaluate(Value(&sum), value) && value == 11u &&
              clean.Evaluate(raw, value) && value == 20u &&
              reads.ordinary == 1u && reads.strict == 1u,
          "CFG mismatch fallback did not restore both validated memo domains");
  }

  ResourcePlan activation;
  activation.srt_plan_complete = activation.resource_tracking_complete = true;
  activation.descriptor_sources.resize(2);
  activation.control_flow.resize(3);
  activation.control_flow[0].condition = Value(true);
  activation.control_flow[0].successors = {1u, 2u};
  activation.control_flow[1].sources = {0u};
  activation.control_flow[2].sources = {1u};
  Check(SrtWalker(activation, runtime).RefreshFlatBuffer(flat), "CFG activation warmup failed");
  auto &activation_tape = activation.cfg_srt_tapes[0];
  activation_tape.ops[activation_tape.condition_ops[0]].imm = 0u;
  reads = {};
  Check(SrtWalker(activation, runtime).RefreshFlatBuffer(flat) &&
            activation.active_sources == std::vector<uint8_t>{1u, 0u} &&
            activation_tape.state == ResourcePlan::SrtTape::State::Unsupported &&
            reads.ordinary == 0u && reads.strict == 0u,
        "CFG verifier ignored an activation-only mismatch");

  auto unmatched = SrtPlan(0x2000u);
  unmatched.control_flow.resize(1);
  unmatched.control_flow[0].srt_reads = {0u};
  Check(SrtWalker(unmatched, runtime).RefreshFlatBuffer(flat), "CFG unmatched warmup failed");
  auto &unmatched_tape = unmatched.cfg_srt_tapes[0];
  for (const auto &op: unmatched_tape.ops) {
    if (op.kind == ResourcePlan::SrtTape::Kind::RawRead) {
      unmatched_tape.ops[op.args[0]].imm = 0x3000u;
      break;
    }
  }
  reads = {};
  flat = {0x55u};
  Check(!SrtWalker(unmatched, runtime).RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{0x55u} &&
            reads.ordinary == 1u && unmatched_tape.state == ResourcePlan::SrtTape::State::Unsupported,
        "unmatched CFG replay reread live memory or published an unvalidated result");

  auto unusual = SrtPlan(0x2000u);
  unusual.clean_flat_slots = {1u};
  unusual.control_flow.resize(1);
  unusual.control_flow[0].srt_reads = {0u};
  reads = {};
  {
    // Deliberately not CleanRuntime(runtime): its ordinary callback is distinct.
    SrtWalker different_clean(unusual, runtime);
    SrtWalker walker(unusual, runtime, unusual.clean_flat_slots, &different_clean);
    Check(!walker.RefreshFlatBuffer(flat) && reads.ordinary == 0u && reads.strict == 0u &&
              unusual.cfg_srt_tapes[1].state == ResourcePlan::SrtTape::State::Unsupported,
          "CFG verifier accepted a distinct clean runtime as the same capture domain");
  }
  const auto verifier_after = GetSrtTapeStats();
  Check(verifier_after.cfg_checked - verifier_before.cfg_checked == 7u &&
            verifier_after.cfg_matched - verifier_before.cfg_matched == 3u &&
            verifier_after.cfg_mismatch - verifier_before.cfg_mismatch == 2u &&
            verifier_after.cfg_inconclusive - verifier_before.cfg_inconclusive == 2u,
        "CFG verifier counters did not report matching, mismatched and rejected refreshes");
}

void TestCfgTapeDemandRoots() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto user = [&](uint32_t reg) {
    auto &inst = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
    inst.SetArg(0, Value(static_cast<ScalarReg>(reg)));
    return Value(&inst);
  };
  const auto raw = [&](Value address) {
    auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    handle.SetArg(0, address); handle.SetArg(1, Value(0u));
    auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = 0});
    read.SetArg(0, Value(&handle)); read.SetArg(1, Value(0u));
    read.SetArg(2, Value(0u)); read.SetArg(3, Value(true));
    return Value(&read);
  };
  const auto shared = raw(user(0));
  const auto poisoned = raw(Value(0xdead0000u));
  auto &selection = plan.value_storage.emplace_back(ValueOpcode::SelectU32);
  selection.SetArg(0, Value(true)); selection.SetArg(1, shared); selection.SetArg(2, poisoned);
  // Unsupported lazy root follows supported shared roots; never execute its poison arm.
  plan.srt_reads = {{shared, 0u}, {poisoned, 1u}, {shared, 2u}, {Value(&selection), 3u}};
  plan.clean_flat_slots = {0u, 0u, 1u, 0u};
  plan.control_flow.resize(3);
  plan.control_flow[0].condition = user(1);
  plan.control_flow[0].successors = {1u, 2u};
  plan.control_flow[1].srt_reads = {0u, 2u, 3u};
  plan.control_flow[1].sources = {0u};
  plan.control_flow[2].srt_reads = {1u};
  plan.control_flow[2].sources = {1u};
  plan.descriptor_sources.resize(2);
  struct Reads {
    std::vector<uint64_t> order;
    uint32_t value = 7;
    bool allow_poison = false;
  } reads;
  const auto ordinary = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &s = *static_cast<Reads *>(data);
    s.order.push_back(address);
    if (words.size() != 1u || (address == 0xdead0000u && !s.allow_poison)) return false;
    words[0] = s.value;
    return true;
  };
  const auto strict = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &s = *static_cast<Reads *>(data);
    s.order.push_back(address | (1ull << 63u));
    if (words.size() != 1u) return false;
    words[0] = s.value + 100u;
    return true;
  };
  std::array<uint32_t, 2> sgpr {0x1000u, 1u};
  const SrtRuntime runtime {.user_data = sgpr, .read_memory = ordinary, .userdata = &reads,
                            .read_specialization_memory = strict};
  std::vector<uint32_t> flat;
  const auto stats_before = GetSrtTapeStats();
  for (uint32_t generation = 0; generation < 2u; ++generation) {
    reads.order.clear();
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    uint32_t preloaded = 0;
    Check(walker.Evaluate(shared, preloaded), "CFG memo warmup failed");
    Check(walker.RefreshFlatBuffer(flat) &&
              flat == std::vector<uint32_t>{reads.value, 0u, reads.value + 100u, reads.value} &&
              reads.order == std::vector<uint64_t>{sgpr[0], sgpr[0] | (1ull << 63u)} &&
              plan.active_sources == std::vector<uint8_t>{1u, 0u},
          "CFG demand tape eagerly read poison, merged domains, or ignored recursive memo");
    sgpr[0] += 0x1000u;
    ++reads.value;
  }
  if (Common::RuntimeFlags::Current().srt_cfg_tape) {
    const auto &tape = plan.cfg_srt_tapes[1];
    Check(tape.state == ResourcePlan::SrtTape::State::Ready &&
              tape.slot_ops[0] != UINT32_MAX && tape.slot_ops[3] == UINT32_MAX &&
              GetSrtTapeStats().cfg_runs > stats_before.cfg_runs &&
              GetSrtTapeStats().cfg_fallback > stats_before.cfg_fallback,
          "CFG test silently bypassed supported or independently unsupported roots");
  }
  // A runtime-unknown condition must preserve traversal of both successors.
  auto &unknown = plan.value_storage.emplace_back(ValueOpcode::GetBuiltin);
  unknown.SetArg(0, Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)));
  unknown.SetArg(1, Value(0u));
  plan.control_flow[0].condition = Value(&unknown);
  // A new plan graph requires a fresh compiled aggregate, as in extracted shader variants.
  for (auto &tape: plan.cfg_srt_tapes) tape = {};
  reads.allow_poison = true;
  reads.order.clear();
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    Check(walker.RefreshFlatBuffer(flat) && flat[1] == reads.value &&
              plan.active_sources == std::vector<uint8_t>{1u, 1u} &&
              reads.order == std::vector<uint64_t>{0xdead0000u, sgpr[0], sgpr[0] | (1ull << 63u)},
          "unknown CFG predicate did not traverse both successors in original read order");
  }
  reads.allow_poison = false;
  reads.order.clear();
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    uint32_t ignored = 0;
    Check(!walker.Evaluate(poisoned, ignored) && !walker.RefreshFlatBuffer(flat) &&
              reads.order == std::vector<uint64_t>{0xdead0000u},
          "CFG tape retried a memoized failed read");
  }
}

void TestCfgTapeCompilationCycleFallsBack() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  auto &cycle = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
  cycle.SetArg(0, Value(&cycle)); cycle.SetArg(1, Value(1u));
  plan.srt_reads = {{Value(4u), 0u}, {Value(&cycle), 1u}};
  plan.control_flow.resize(1);
  plan.control_flow[0].srt_reads = {0u, 1u};
  std::vector<uint32_t> flat;
  Check(!SrtWalker(plan, {}).RefreshFlatBuffer(flat) && flat[0] == 4u,
        "cyclic CFG compilation did not leave supported sibling root usable");
}

void TestTapeUnsupportedRootFallsBack() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  auto &selection = plan.value_storage.emplace_back(ValueOpcode::SelectU32);
  selection.SetArg(0, Value(true));
  selection.SetArg(1, Value(0x22222222u));
  selection.SetArg(2, Value(0x33333333u));
  // Keep a valid Tape op at index zero. Before the fix, the unsupported Select
  // returned `false` (integer zero), silently aliasing this first slot's op.
  plan.srt_reads.push_back({Value(0x11111111u), 0u});
  plan.srt_reads.push_back({Value(&selection), 1u});

  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, {}).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x11111111u, 0x22222222u},
        "unsupported Tape root did not fall back to the recursive evaluator");
}

void TestTapeDeduplicatesRawImmediateBits() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info = {
      {.kind = ResourceKind::ScalarAddress, .offset = 0u, .planning_only = true},
      {.kind = ResourceKind::ScalarAddress, .offset = 4u, .planning_only = true}};

  auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x2000u));
  handle.SetArg(1, Value(0u));
  const auto read = [&](uint32_t memory_index) {
    auto &inst = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    inst.SetFlags(MemoryFlags{.index = memory_index});
    inst.SetArg(0, Value(&handle));
    inst.SetArg(1, Value(0u));
    inst.SetArg(2, Value(0u));
    inst.SetArg(3, Value(true));
    return Value(&inst);
  };
  plan.srt_reads = {
      {read(0u), 0u},
      {Value::F32(1.0f), 1u},
      {Value(0x3f800000u), 2u},
      {read(1u), 3u},
      {Value(uint64_t{0x3f800000u}), 4u}};

  struct Reads {
    std::vector<uint64_t> addresses;
  } reads_state;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        if (words.size() != 1u || (address != 0x2000u && address != 0x2004u)) return false;
        state.addresses.push_back(address);
        words[0] = address == 0x2000u ? 0xaaaaaaaau : 0xbbbbbbbbu;
        return true;
      },
      .userdata = &reads_state};

  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, runtime).RefreshFlatBuffer(flat),
        "tape refresh with duplicate immediate bits failed");
  Check(flat == std::vector<uint32_t>{0xaaaaaaaau, 0x3f800000u, 0x3f800000u,
                                      0xbbbbbbbbu, 0x3f800000u},
        "tape immediate deduplication changed mixed-type raw-bit results");
  Check(reads_state.addresses == std::vector<uint64_t>{0x2000u, 0x2004u},
        "tape immediate deduplication changed raw-read order");
  const char *tape_enabled = std::getenv("KYTY_SRT_TAPE");
  if (tape_enabled != nullptr && tape_enabled[0] == '0') return;
  const auto &tape = plan.srt_tapes[0];
  const auto matching_constants = std::ranges::count_if(tape.ops, [](const auto &op) {
    return op.kind == ResourcePlan::SrtTape::Kind::Immediate && op.imm == 0x3f800000u;
  });
  Check(matching_constants == 1u,
        "tape emitted repeated immediate ops for the same raw 64-bit value");
  Check(tape.ops.size() == 5u,
        "tape did not reduce this mixed-immediate fixture to five shared ops");
}

void TestTapePreservesWideMemoryIndex() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.resize(257);
  plan.memory_info[0] = {.kind = ResourceKind::ScalarAddress, .offset = 0,
                        .planning_only = true};
  plan.memory_info[256] = {.kind = ResourceKind::ScalarAddress, .offset = 4,
                           .planning_only = true};
  auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x2000u));
  handle.SetArg(1, Value(0u));
  auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  read.SetFlags(MemoryFlags{.index = 256});
  read.SetArg(0, Value(&handle));
  read.SetArg(1, Value(0u));
  read.SetArg(2, Value(0u));
  read.SetArg(3, Value(true));
  plan.srt_reads.push_back({Value(&read), 0u});

  const SrtRuntime runtime{
      .read_memory = +[](void *, uint64_t address, std::span<uint32_t> words) {
        if (words.size() != 1u) return false;
        if (address == 0x2000u) words[0] = 0x11111111u;
        else if (address == 0x2004u) words[0] = 0x22222222u;
        else return false;
        return true;
      }};
  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, runtime).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x22222222u},
        "Tape truncated a 32-bit memory-info index");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(SrtWalker(program, {}).Evaluate(Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!SrtWalker(program, {}).Evaluate(Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorPredicateReads(bool memory_condition) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // Only memory-derived host decisions need clean, disjoint descriptor reads.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
      {memory_condition ? Value(&read) : Value(&offset),
       Value(memory_condition ? 0x8000u : 4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.blocks[0]->condition = Value(&condition);
  program.blocks[0]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.blocks[0]->terminator.true_block = program.blocks[1];
  program.blocks[0]->terminator.false_block = program.blocks[2];
  program.blocks[1]->id = 1;
  program.blocks[1]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.blocks[2]->id = 2;
  program.blocks[2]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  Check(plan.capture_specialization_reads == memory_condition && !plan.control_flow.empty(),
        "conditional writable descriptor used the wrong memory dependency policy");
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; bool clean = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 0x8000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  if (memory_condition) {
    Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.ordinary == 0 && reads.strict == 1,
          "GPU-dirty dynamic writable descriptor bypassed strict provenance");
    reads.clean = true;
    reads.strict = 0;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.ordinary == 0 && reads.strict == 1 && snapshot.buffers[0].dwords[0] == 0x8000u &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}},
          "writable descriptor was evaluated twice or scalar EXEC suppressed its read");
    Libs::Graphics::ResourceMemo memo;
    bool reused = true;
    Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
          "strict memo did not establish an initial snapshot");
    Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
          "unchanged strict descriptor did not reuse its snapshot");
    reads.clean = false;
    Check(!memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
          "failed strict read reused a stale writable descriptor");
    reads.clean = true;
    Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
          "failed strict refresh left a reusable snapshot");
  } else {
    // A user-data predicate needs no clean read; the descriptor uses the ordinary reader.
    for (const bool clean : {false, true}) {
      reads = {.clean = clean};
      Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
                reads.ordinary == 1 && reads.strict == 0 &&
                snapshot.buffers[0].dwords[0] == 0x8000u && snapshot.specialization_reads.empty(),
            "immutable writable-descriptor predicate changed reader policy");
    }
  }
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestSharedControlPredicates() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 8;
  for (const auto operation : {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr}) {
    Program program;
    program.user_data_count = 1;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    auto &block = AddValueBlock(program);
    const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
      return Value(&block.AppendNewInst(opcode, args));
    };
    const auto user = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
    const auto uniform = emit(ValueOpcode::INotEqual32, {user, Value(0u)});
    const auto lane = emit(ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    auto varying = emit(ValueOpcode::INotEqual32, {lane, Value(0u)});
    // Both failed and partially uniform subgraphs share their operands exponentially.
    for (uint32_t i = 0; i < depth; ++i)
      varying = emit(ValueOpcode::LogicalOr, {varying, varying});
    auto condition = emit(operation, {varying, uniform});
    for (uint32_t i = 0; i < depth; ++i)
      condition = emit(ValueOpcode::LogicalOr, {condition, condition});
    condition = emit(ValueOpcode::LogicalNot, {condition});
    condition = emit(ValueOpcode::ConditionRef, {condition});
    for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
    for (uint32_t i = 0; i < 2; ++i) {
      program.blocks[i]->condition = condition;
      program.blocks[i]->terminator.kind = TerminatorKind::ConditionalBranch;
      program.blocks[i]->terminator.true_block = program.blocks[i == 0 ? 1 : 3];
      program.blocks[i]->terminator.false_block = program.blocks[2];
    }
    auto plan = ExtractResourcePlan(program);
    Check(plan.value_storage.size() == depth + 5 &&
              plan.control_flow.size() == 4 &&
              plan.control_flow[0].condition == plan.control_flow[1].condition &&
              !plan.capture_specialization_reads,
          "shared predicates expanded into trees or retained varying dependencies");
    for (const uint32_t flag : {0u, 1u}) {
      const std::array<uint32_t, 1> user_data{flag};
      SrtWalker walker(plan, {.user_data = user_data});
      uint32_t result = 99;
      const bool conjunction = operation == ValueOpcode::LogicalAnd;
      const bool known = conjunction ? flag == 0 : flag != 0;
      Check(walker.Evaluate(plan.control_flow[0].condition, result) == known &&
                (!known || result == uint32_t{conjunction}),
            "shared predicate changed its uniform factor or cached runtime values");
      std::vector<uint32_t> flat;
      // Our walker explores both edges of a branch whose predicate depends on per-lane values
      // (upstream also prunes the rejected edge), so only the safety property is checked here:
      // every block the guest can reach is visited.
      const auto expected = known ? (conjunction ? std::vector<uint8_t>{1, 1, 0, 1}
                                                 : std::vector<uint8_t>{1, 0, 1, 0})
                                  : std::vector<uint8_t>{1, 1, 1, 1};
      bool covers = walker.RefreshFlatBuffer(flat) && plan.visited_blocks.size() == expected.size();
      for (size_t i = 0; covers && i < expected.size(); ++i) covers = !expected[i] || plan.visited_blocks[i];
      Check(covers, "shared predicate skipped a reachable branch");
    }
  }
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = ExtractResourcePlan(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
    uint32_t ordinary = 0;
  } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    const auto offset = specialization.images[0].indirect_mapping_offset;
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated scalar reads or bypassed clean provenance");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
}

// The merged SrtWalker batches only plain user-data-pointer loads (the direct path, KYTY_SRT_BATCH_READS)
// and pengui07's KYTY_SRT_SCALAR_BATCH spans; these constant-address plans take neither, so the case checks
// values, grouping metadata and reachability, not reader-call counts.
void TestContiguousScalarReadBatches() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto make_plan = [](bool buffer, uint32_t size = 16, bool gap = false,
                            uint32_t count = 4, uint64_t address = 0x1000) {
    Program program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    auto &block = AddValueBlock(program);
    auto &handle = buffer
        ? block.AppendNewInst(ValueOpcode::GetBufferResource,
            {Value(static_cast<uint32_t>(address)), Value(static_cast<uint32_t>(address >> 32)), Value(size), Value(0u)})
        : block.AppendNewInst(ValueOpcode::GetAddressResource,
            {Value(static_cast<uint32_t>(address)), Value(static_cast<uint32_t>(address >> 32))});
    DescriptorSource source;
    source.dword_count = 4;
    source.dwords.fill(Value(0u));
    for (uint32_t index = 0; index < count; ++index) {
      program.memory_info.push_back({.kind = buffer ? ResourceKind::ScalarBuffer : ResourceKind::ScalarAddress,
                                     .offset = index * 4 + (gap && index > 0 ? 4u : 0u),
                                     .planning_only = true});
      auto &read = buffer
          ? block.AppendNewInst(ValueOpcode::ReadConstBuffer, {Value(&handle), Value(0u)})
          : block.AppendNewInst(ValueOpcode::LoadAddressU32, {Value(&handle), Value(0u), Value(0u), Value(true)});
      read.SetFlags(MemoryFlags{.index = index});
      program.srt_reads.push_back({Value(&read), index});
      // Descriptor dependencies must see the published scalar memo, without rereading.
      if (index == 0 || index == count - 1) source.dwords[index == 0 ? 0 : 1] = Value(&read);
    }
    program.descriptor_sources.push_back(source);
    program.info.samplers.push_back({.source = 0});
    return ExtractResourcePlan(program);
  };
  struct Reads {
    std::vector<uint32_t> words{11, 22, 33, 44, 55};
    uint32_t calls = 0;
    bool available = true;
  } reads;
  const auto reader = +[](void *data, uint64_t address, std::span<uint32_t> values) {
    auto &state = *static_cast<Reads *>(data);
    ++state.calls;
    if (!state.available || address < 0x1000 || address - 0x1000 > state.words.size() * 4 ||
        values.size_bytes() > state.words.size() * 4 - (address - 0x1000)) return false;
    std::copy_n(state.words.begin() + (address - 0x1000) / 4, values.size(), values.begin());
    return true;
  };
  SrtRuntime runtime{.read_memory = reader, .userdata = &reads, .batch_reads = true};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool buffer : {false, true}) {
    auto plan = make_plan(buffer);
    Check(plan.srt_batch_counts[0] == 4, "contiguous scalar reads were not grouped");
    for (const bool batched : {false, true}) {
      runtime.batch_reads = batched;
      reads.calls = 0;
      Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
                snapshot.flattened_srt == std::vector<uint32_t>({11, 22, 33, 44}) &&
                snapshot.samplers[0].dwords[0] == 11 && snapshot.samplers[0].dwords[1] == 44 &&
                reads.calls == 4,
            "batched scalar values or descriptor memo differ from scalar evaluation");
    }
    runtime.batch_reads = true;
    // (Removed: a seeded walker session memo that a later batched RefreshFlatBuffer must not reread.
    // The merged walker refreshes the flat buffer from fresh reads; its own session memo is covered by
    // the resource tracking tests.)
    reads.available = false;
    Check(!MaterializeResources(plan, runtime, snapshot, specialization), "failed batch read was accepted");
    reads.available = true;
    for (const size_t capacity : {size_t{1}, size_t{16}}) {
      Libs::Graphics::ResourceMemo memo(capacity);
      bool reused = false;
      Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
            "initial batched resource memo failed");
      Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
            "unchanged batched resource memo missed");
      ++reads.words[3];
      Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
                snapshot.flattened_srt[3] == reads.words[3], "batch memo reused changed backing words");
      reads.words[3] = 44;
    }
    // A clean batch must use the strict reader exclusively.
    plan.clean_flat_slots.assign(4, 1);
    plan.requires_specialization_memory = true;
    auto &srt = plan.value_storage.emplace_back(ValueOpcode::GetSrtResource);
    for (const uint32_t slot : {0u, 3u}) {
      auto &flat = plan.value_storage.emplace_back(ValueOpcode::ReadConst);
      flat.SetArg(0, Value(&srt));
      flat.SetArg(1, Value(slot));
      plan.descriptor_sources[0].dwords[slot == 0 ? 0 : 1] = Value(&flat);
    }
    uint32_t ordinary_calls = 0;
    struct Pair { Reads *strict; uint32_t *ordinary; } pair{&reads, &ordinary_calls};
    runtime.userdata = &pair;
    runtime.read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
      ++*static_cast<Pair *>(data)->ordinary; return false;
    };
    runtime.read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> values) {
      auto &state = *static_cast<Pair *>(data)->strict;
      ++state.calls;
      std::copy_n(state.words.begin() + (address - 0x1000) / 4, values.size(), values.begin());
      return true;
    };
    reads.calls = 0;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) && ordinary_calls == 0 && reads.calls >= 1,
          "clean batch bypassed or repeated strict reads");
    runtime.read_memory = reader;
    runtime.read_specialization_memory = nullptr;
    runtime.userdata = &reads;
  }
  auto bounded = make_plan(true, 15);
  Check(MaterializeResources(bounded, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>({11, 22, 33, 0}),
        "batch crossing scalar buffer bounds did not zero-fill its out-of-range word");
  auto gap = make_plan(false, 32, true);
  reads.calls = 0;
  Check(gap.srt_batch_counts[0] == 1 && gap.srt_batch_counts[1] == 3 &&
            MaterializeResources(gap, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>({11, 33, 44, 55}),
        "batch crossed a scalar-read gap or lost later grouping");
  auto conditional = make_plan(false);
  conditional.control_flow.emplace_back();
  conditional.control_flow.back().sources = {0};
  conditional.control_flow.back().srt_reads = {0};
  reads.calls = 0;
  Check(MaterializeResources(conditional, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>({11, 0, 0, 0}),
        "conditional refresh read unreachable scalar slots through a batch");
  // The merged materializer caches its control-flow schedule per plan: build a fresh plan per case.
  conditional = make_plan(false);
  conditional.control_flow.emplace_back();
  conditional.control_flow.back().sources = {0};
  conditional.control_flow.back().srt_reads = {1, 2, 3};
  reads.calls = 0;
  Check(MaterializeResources(conditional, runtime, snapshot, specialization) &&
            snapshot.flattened_srt.size() == 4 && snapshot.flattened_srt[1] == 22 &&
            snapshot.flattened_srt[2] == 33 && snapshot.flattened_srt[3] == 44,
        "reached-block batching lost a suffix or fetched an unreachable prefix");
  // The merged materializer caches its control-flow schedule per plan: build a fresh plan per case.
  conditional = make_plan(false);
  conditional.control_flow.emplace_back();
  conditional.control_flow.back().sources = {0};
  conditional.control_flow.back().srt_reads = {0, 2};
  reads.calls = 0;
  Check(MaterializeResources(conditional, runtime, snapshot, specialization) &&
            snapshot.flattened_srt.size() == 4 && snapshot.flattened_srt[0] == 11 &&
            snapshot.flattened_srt[2] == 33,
        "reached-block batch crossed nonconsecutive scalar slots");
  auto mapped = make_plan(false, 16, false, 4, reinterpret_cast<uint64_t>(reads.words.data()));
  Check(MaterializeResources(mapped, {.batch_reads = true}, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>({11, 22, 33, 44}),
        "direct mapped backing batch differs from scalar words");
  reads.words.resize(70);
  for (uint32_t index = 0; index < reads.words.size(); ++index) reads.words[index] = index + 1;
  auto large = make_plan(true, 280, false, 70);
  reads.calls = 0;
  Check(large.srt_batch_counts[0] == 64 && large.srt_batch_counts[64] == 6 &&
            MaterializeResources(large, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == reads.words,
        "scalar batches exceeded their limit or lost the trailing words");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

// The replay fixture is what lets SrtWalker/resourceMemo iteration happen with no
// emulator run, so its two load-bearing properties get checked directly: adjacent
// reads coalesce instead of exploding into one record per word, and an uncaptured
// read is reported rather than silently answered.
void TestResourceFixtureRoundTrip() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;

  const std::string path = "resource_fixture_test.arb";
  std::remove(path.c_str());

  ResourceFixture writer;
  const std::array<uint32_t, 3> user_data {0xAAAA0001u, 0xAAAA0002u, 0xAAAA0003u};
  writer.SetUserData(user_data);
  writer.SetShaderBase(0x1234'5678'9ABCuLL);

  // Three adjacent ordinary words: must collapse into one record.
  writer.AddRead(FixtureRead {.address = 0x5000, .strict = false, .valid = true, .values = {0x11u}});
  writer.AddRead(FixtureRead {.address = 0x5004, .strict = false, .valid = true, .values = {0x22u}});
  writer.AddRead(FixtureRead {.address = 0x5008, .strict = false, .valid = true, .values = {0x33u}});

  // A gap breaks coalescing.
  writer.AddRead(FixtureRead {.address = 0x5100, .strict = false, .valid = true, .values = {0x44u}});

  // A read that legitimately failed must be recorded as invalid, not omitted.
  writer.AddRead(FixtureRead {.address = 0x5200, .strict = false, .valid = false, .values = {0u}});

  Check(writer.TotalReads() == 5u, "fixture counted the raw reads");
  Check(writer.DistinctReads() == 3u, "adjacent reads coalesced into one record");
  Check(writer.Save(path), "fixture saved");

  ResourceFixture reader;
  Check(reader.Load(path), "fixture loaded");
  Check(reader.ShaderBase() == 0x1234'5678'9ABCuLL, "shader base survived the round trip");
  Check(reader.UserData().size() == user_data.size() &&
            reader.UserData()[2] == user_data[2],
        "user data survived the round trip");

  // Replay serves exactly what was captured.
  std::array<uint32_t, 3> out {0u, 0u, 0u};
  bool missing = false;
  Check(reader.Read(nullptr, 0x5000, out, false, &missing) && !missing &&
            out[0] == 0x11u && out[1] == 0x22u && out[2] == 0x33u,
        "a coalesced record serves the full span it covered");

  // Lookup is reader-agnostic: the fixture records the memory access trace, so the
  // same bytes are served whichever reader the plan uses. Recording a strict label
  // would be wrong for a runtime whose ordinary reader is the specialization reader.
  bool again_missing = false;
  std::array<uint32_t, 1> again_out {0u};
  Check(reader.Read(nullptr, 0x5000, again_out, true, &again_missing) && !again_missing &&
            again_out[0] == 0x11u,
        "the same bytes are served whichever reader asks");

  // A width that was never captured at a captured address is missing, not a
  // truncated or over-read answer.
  std::array<uint32_t, 4> wrong_width {0u, 0u, 0u, 0u};
  bool width_missing = false;
  Check(!reader.Read(nullptr, 0x5000, wrong_width, false, &width_missing) && width_missing,
        "a differently sized request at a captured address is reported as missing");

  // The 1-word record at 0x5100 must not be extended to 2 words by the coalesced
  // neighbour at 0x5000.
  std::array<uint32_t, 2> gap_overread {0u, 0u};
  bool gap_missing = false;
  Check(!reader.Read(nullptr, 0x5100, gap_overread, false, &gap_missing) && gap_missing,
        "a record is not extended past the span it captured");

  // An uncaptured request must be reported, never invented or read from live memory.
  std::array<uint32_t, 1> absent {0u};
  bool absent_missing = false;
  Check(!reader.Read(nullptr, 0xDEAD'0000, absent, false, &absent_missing) && absent_missing,
        "an uncaptured read is reported as missing");

  // A recorded-invalid read must report "the fixture knows this read failed", which
  // is NOT the same as "the fixture lacks the record". A partial fixture that
  // reported the latter as the former would look complete when it is not.
  std::array<uint32_t, 1> invalid_out {0u};
  bool invalid_missing = false;
  bool invalid_failed  = false;
  Check(!reader.Read(nullptr, 0x5200, invalid_out, false, &invalid_missing, &invalid_failed) &&
            invalid_failed && !invalid_missing,
        "a recorded-invalid read reports failed, not missing");

  // Version mismatch must be refused rather than misinterpreted.
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(8);
    const uint32_t bad_version = 0xFFFF'FFFFu;
    f.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    f.close();
  }
  ResourceFixture bad;
  Check(!bad.Load(path), "a fixture with an unknown version is refused");
  std::remove(path.c_str());
}

// A fixture is only useful if it can actually drive a runtime, so this exercises the
// replay callbacks the way a plan would: every captured read must be served back
// identically, and anything uncaptured must be reported rather than invented.
void TestResourceFixtureDrivesRuntime() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;

  ResourceFixture fixture;
  fixture.SetUserData(std::array<uint32_t, 2> {0x1111u, 0x2222u});
  fixture.SetShaderBase(0x9000'0000u);
  fixture.AddRead(FixtureRead {.address = 0x2000, .strict = false, .valid = true, .values = {0xA1u}});
  fixture.AddRead(FixtureRead {.address = 0x2004, .strict = false, .valid = true, .values = {0xB2u}});
  fixture.AddRead(FixtureRead {.address = 0x2100, .strict = true, .valid = true, .values = {0xC3u}});

  // Replay callback bound to the fixture, as a plan's reader would be.
  struct Replay {
    ResourceFixture *fixture;
    uint32_t served = 0;
    uint32_t missing = 0;
    uint32_t failed = 0;
  };
  static Replay replay;
  replay = Replay {.fixture = &fixture};

  auto reader = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &state = *static_cast<Replay *>(data);
    bool missing = false;
    bool failed = false;
    const bool ok = state.fixture->Read(nullptr, address, words, false, &missing, &failed);
    if (ok) ++state.served;
    if (missing) ++state.missing;
    if (failed) ++state.failed;
    return ok;
  };

  SrtRuntime runtime {};
  runtime.user_data = fixture.UserData();
  runtime.shader_base = fixture.ShaderBase();
  runtime.read_memory = reader;
  runtime.userdata = &replay;

  // A read spanning two coalesced words must come back byte-identical.
  std::array<uint32_t, 2> span {0u, 0u};
  Check(runtime.read_memory(runtime.userdata, 0x2000, span) && span[0] == 0xA1u && span[1] == 0xB2u,
        "the replay runtime serves a captured span unchanged");
  Check(replay.served == 1u, "the replay runtime counted the served read");

  // A wider request that runs past the captured span must be refused, not truncated.
  std::array<uint32_t, 4> over {0u, 0u, 0u, 0u};
  Check(!runtime.read_memory(runtime.userdata, 0x2000, over) && replay.missing == 1u,
        "a read running past the captured span is refused and reported missing");

  // An uncaptured address must never be answered.
  std::array<uint32_t, 1> absent {0u};
  Check(!runtime.read_memory(runtime.userdata, 0x7777'0000u, absent),
        "an uncaptured address is refused by the replay runtime");
  Check(fixture.ShaderBase() == runtime.shader_base && runtime.user_data.size() == 2u,
        "the replay runtime keeps the recorded base and user data");
}

// The live recorder writes append-only batches rather than rewriting the file, so the
// append path needs its own coverage: Save() alone would not have caught a header
// count written at the wrong offset, which is a bug this test now pins down.
void TestResourceFixtureAppendPath() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;

  const std::string path = "resource_fixture_append_test.arb";
  std::remove(path.c_str());

  ResourceFixture writer;
  writer.SetUserData(std::array<uint32_t, 1> {0x5A5A5A5Au});
  writer.SetShaderBase(0xDEAD'BEEFu);
  Check(writer.BeginFile(path), "append path: header written");

  // Two batches, derived the same way the live recorder derives them: add, rebuild so
  // coalescing is applied, then append only the records beyond the previous count.
  auto append_batch = [&](std::vector<FixtureRead> reads, const char *what) {
    const size_t before = writer.DistinctReads();
    for (auto &read : reads) writer.AddRead(read);
    writer.Rebuild();
    const auto &all = writer.Reads();
    std::vector<FixtureRead> tail;
    for (size_t i = before; i < all.size(); ++i) tail.push_back(all[i]);
    Check(writer.AppendReads(path, tail), what);
    return tail.size();
  };

  // Adjacent words coalesce into one record, so this batch writes a single record.
  const size_t written_one = append_batch(
      {FixtureRead {.address = 0x1000, .strict = false, .valid = true, .values = {0x01u}},
       FixtureRead {.address = 0x1004, .strict = false, .valid = true, .values = {0x02u}}},
      "append path: first batch appended");
  Check(written_one == 1u, "append path: adjacent reads coalesced into one appended record");

  const size_t written_two = append_batch(
      {FixtureRead {.address = 0x2000, .strict = false, .valid = true, .values = {0x03u}}},
      "append path: second batch appended");
  Check(written_two == 1u, "append path: second batch appended one record");

  // The file must load after appends, with both batches present.
  ResourceFixture reader;
  Check(reader.Load(path), "append path: file loads after appends");
  Check(reader.ShaderBase() == 0xDEAD'BEEFu, "append path: shader base intact");
  Check(reader.UserData().size() == 1u && reader.UserData()[0] == 0x5A5A5A5Au,
        "append path: user data intact");
  Check(reader.Reads().size() == 2u, "append path: both appended records survived");

  std::vector<uint32_t> out(2, 0u);
  bool missing = false;
  bool failed = false;
  Check(reader.Read(nullptr, 0x1000, out, false, &missing, &failed) && out[0] == 0x01u && out[1] == 0x02u,
        "append path: the coalesced first batch replays whole");
  out.assign(1, 0u);
  Check(reader.Read(nullptr, 0x2000, out, false, &missing, &failed) && out[0] == 0x03u,
        "append path: second batch replays");

  std::remove(path.c_str());
}

// The accumulator merges overlapping records word by word, so this checks the property
// that matters: after a rebuild, every word that was ever added is still there. An
// earlier record-level dedup silently dropped words beyond a wider kept span.
void TestResourceFixturePreservesEveryWord() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;

  const std::string path = "resource_fixture_words_test.arb";
  std::remove(path.c_str());

  ResourceFixture writer;
  Check(writer.BeginFile(path), "word preservation: header written");

  // Batch 1: a three-word record.
  writer.AddRead(FixtureRead {.address = 0x8000, .strict = false, .valid = true, .values = {0x11u, 0x22u, 0x33u}});
  writer.Rebuild();
  uint64_t appended = 0;
  std::vector<FixtureRead> batch_one;
  for (const auto &read : writer.Reads()) {
    batch_one.push_back(read);
  }
  Check(writer.AppendReads(path, batch_one), "word preservation: batch 1 appended");
  appended = batch_one.size();
  Check(appended == 1u, "word preservation: batch 1 wrote one record");

  // Batch 2: a record that overlaps an already-captured word, plus a word that extends
  // past the end of the wide record. Neither may be lost.
  writer.AddRead(FixtureRead {.address = 0x8004, .strict = false, .valid = true, .values = {0x77u}});
  writer.AddRead(FixtureRead {.address = 0x800C, .strict = false, .valid = true, .values = {0x99u}});
  writer.Rebuild();

  std::vector<uint32_t> out(1, 0u);
  bool missing = false;
  bool failed = false;
  Check(writer.Read(nullptr, 0x8000, out, false, &missing, &failed) && out[0] == 0x11u,
        "word preservation: first word of the wide record survived");
  // Overlap keeps the FIRST captured value: a later record must not rewrite what the
  // guest actually returned at that address, or replay would no longer match the run.
  out.assign(1, 0u);
  Check(writer.Read(nullptr, 0x8004, out, false, &missing, &failed) && out[0] == 0x22u,
        "word preservation: overlapping word keeps the first captured value");
  out.assign(1, 0u);
  Check(writer.Read(nullptr, 0x8008, out, false, &missing, &failed) && out[0] == 0x33u,
        "word preservation: word beyond the overlap survived");
  out.assign(1, 0u);
  Check(writer.Read(nullptr, 0x800C, out, false, &missing, &failed) && out[0] == 0x99u,
        "word preservation: extending word survived");

  // Conservation: distinct addresses are 0x8000, 0x8004, 0x8008, 0x800C = 4 words.
  // The overlapping 0x8004 is a genuine duplicate and must not be double counted, so
  // the retained count staying at exactly 4 is the dedup proof.
  Check(writer.WordsRetained() == 4u,
        "word preservation: retained word count equals the distinct words added");

  std::remove(path.c_str());
}

// The live recorder flushes repeatedly, appending only words the file does not have
// yet. This replays that shape offline: de-coalesced records so each append is
// predictable, several flushes, then assert the file accounts for every word.
void TestResourceFixtureRepeatedFlush() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;

  const std::string path = "resource_fixture_flush_test.arb";
  std::remove(path.c_str());

  ResourceFixture writer;
  writer.EnableCoalescing(false); // one record per read, so appends are countable
  Check(writer.BeginFile(path), "repeated flush: header written");

  std::vector<FixtureRead> batch;
  // Expected pieces, derived with the same word-splitting rule the recorder uses: a
  // record is split so only its genuinely new words are written. Mirrors MergeAndFlush.
  std::vector<FixtureRead> expected_pieces;
  auto word_written = [&](uint64_t address) {
    for (const auto &r : expected_pieces) {
      if (address >= r.address && address < r.address + r.values.size() * sizeof(uint32_t)) return true;
    }
    return false;
  };

  auto flush = [&]() {
    writer.Rebuild();
    batch.clear();
    for (const auto &read : writer.Reads()) {
      auto new_piece = [&]() {
        FixtureRead piece;
        piece.strict = read.strict;
        piece.valid  = read.valid;
        return piece;
      };
      FixtureRead piece;
      bool started = false;
      uint64_t cursor = read.address;
      for (const uint32_t value : read.values) {
        if (!word_written(cursor)) {
          if (!started) {
            piece = new_piece();
            piece.address = cursor;
            started = true;
          }
          piece.values.push_back(value);
        } else if (started) {
          batch.push_back(piece);
          piece = FixtureRead {};
          started = false;
        }
        cursor += sizeof(uint32_t);
      }
      if (started) batch.push_back(piece);
    }
    Check(writer.AppendReads(path, batch), "repeated flush: append succeeded");
    for (const auto &piece : batch) expected_pieces.push_back(piece);
  };

  for (int round = 0; round < 5; ++round) {
    const uint32_t base = 0x10000u + static_cast<uint32_t>(round) * 0x1000u;
    for (uint32_t i = 0; i < 8; ++i) {
      writer.AddRead(FixtureRead {.address = base + i * 4u,
                                  .strict = false,
                                  .valid = true,
                                  .values = {0xAA00u + i}});
    }
    if (round >= 1) {
      // Introduce overlapping values, the case that tripped the earlier cursor logic.
      writer.AddRead(FixtureRead {.address = 0x10004u, .strict = false, .valid = true, .values = {0xDEADu}});
    }
    flush();
  }

  ResourceFixture reader;
  Check(reader.Load(path), "repeated flush: file loads");
  std::printf("  [diag] writer.WordsRetained=%llu expected_pieces=%zu file_records=%zu\n",
              static_cast<unsigned long long>(writer.WordsRetained()), expected_pieces.size(),
              reader.Reads().size());
  // The file may hold fewer pieces than were appended, because appending two adjacent
  // pieces leaves them as two records on disk. What must hold is that the file accounts
  // for every word the accumulator holds: no word lost, none invented.
  uint64_t file_words = 0;
  for (const auto &read : reader.Reads()) file_words += read.values.size();
  Check(file_words == writer.WordsRetained(),
        "repeated flush: file accounts for every retained word");

  // A sample of the earliest and latest rounds must be readable.
  std::vector<uint32_t> out(1, 0u);
  bool missing = false;
  bool failed = false;
  Check(reader.Read(nullptr, 0x10000, out, false, &missing, &failed) && out[0] == 0xAA00u,
        "repeated flush: first round is readable");
  out.assign(1, 0u);
  Check(reader.Read(nullptr, 0x14000, out, false, &missing, &failed) && out[0] == 0xAA00u,
        "repeated flush: last round is readable");

  std::remove(path.c_str());
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

// Reached through Common::Thread on Windows (threads.cpp is linked in for the
// config/experiment globals); without it the target does not link there.
int DbgNotImplementedHandler(const char *, const char *, int) { std::abort(); }

void DbgExit(int) { std::abort(); }

} // namespace Common

// Inspect a recorded fixture without a game run. This is the reader half of the
// record/replay loop: point it at a .arb produced by KYTY_SRT_FIXTURE_RECORD to see
// what was captured, and use it to confirm a recording actually contains a usable
// workload before spending an emulator run on replaying it.
//
// Usage: resource_materialization_tests.exe --inspect-fixture <path.arb>
static int InspectFixture(const char *path) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourceFixture fixture;
  if (!fixture.Load(path)) {
    std::fprintf(stderr, "fixture: FAILED to load '%s' (missing, wrong magic or wrong version)\n", path);
    return 1;
  }
  uint64_t min_address = UINT64_MAX;
  uint64_t max_address = 0;
  uint64_t failed_reads = 0;
  uint64_t words = 0;
  for (const auto &read : fixture.Reads()) {
    if (read.address < min_address) min_address = read.address;
    const uint64_t end = read.address + read.values.size() * sizeof(uint32_t);
    if (end > max_address) max_address = end;
    if (!read.valid) ++failed_reads;
    words += read.values.size();
  }
  std::printf("fixture: %s\n", path);
  std::printf("  distinct reads : %zu\n", fixture.Reads().size());
  std::printf("  words captured : %llu\n", static_cast<unsigned long long>(words));
  std::printf("  failed reads   : %llu\n", static_cast<unsigned long long>(failed_reads));
  std::printf("  shader base    : 0x%llx\n", static_cast<unsigned long long>(fixture.ShaderBase()));
  std::printf("  user data words: %zu\n", fixture.UserData().size());
  if (!fixture.Reads().empty()) {
    std::printf("  address range  : 0x%llx .. 0x%llx\n",
                static_cast<unsigned long long>(min_address),
                static_cast<unsigned long long>(max_address));
  }
  // Exercise the replayer against the fixture's own first record so a structurally
  // valid but unreadable fixture is caught here rather than mid-experiment.
  if (!fixture.Reads().empty()) {
    const auto &first = fixture.Reads().front();
    std::vector<uint32_t> out(first.values.size(), 0u);
    bool missing = false;
    bool failed = false;
    const bool ok = fixture.Read(nullptr, first.address, out, false, &missing, &failed);
    std::printf("  self-read      : %s (missing=%d failed=%d)\n", ok ? "served" : "NOT SERVED",
                missing ? 1 : 0, failed ? 1 : 0);
  }
  return 0;
}

static int ReplayPlanFixture(const char* path) {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 ResourcePlanFixture fixture;
 if (!fixture.Load(path)) {
  std::fprintf(stderr,"plan fixture: load failed: %s (requires a .rpf; legacy .arb has no plan)\n",path);
  return 2;
 }
 const auto r = fixture.Replay();
 std::printf("plan fixture: hash=%llx instructions=%zu result=%s success=%d missing=%llu conflicts=%llu "
             "reads=%llu words=%llu ops=%llu captured_ops=%llu cfg_roots=%llu walker=%llu\n",
  static_cast<unsigned long long>(fixture.plan.shader_hash),fixture.plan.value_storage.size(),
  r.matched ? "MATCH" : "FAIL",r.success,
  static_cast<unsigned long long>(r.missing),static_cast<unsigned long long>(r.conflicts),
  static_cast<unsigned long long>(r.requests),static_cast<unsigned long long>(r.words),
  static_cast<unsigned long long>(r.ops),static_cast<unsigned long long>(fixture.captured_ops),
  static_cast<unsigned long long>(r.cfg_roots),static_cast<unsigned long long>(r.walker));
 std::printf("  recursive_instructions=%llu tape_operations=%llu\n", static_cast<unsigned long long>(r.recursive_ops), static_cast<unsigned long long>(r.tape_ops));
 return r.matched ? 0 : 1;
}

// Load and validate an entire captured workload, then time production materialization
// while keeping each frozen reader map alive across warm-up and measured iterations.
static int BenchmarkResourcePlans(const char* directory, uint64_t iterations) {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 if (iterations == 0 || iterations > 1000000) {
  std::fprintf(stderr, "plan benchmark: iterations must be in 1..1000000\n"); return 2;
 }
 std::error_code ec;
 std::filesystem::path root(directory);
 if (!std::filesystem::is_directory(root, ec) || ec) {
  std::fprintf(stderr, "plan benchmark: not a readable directory: %s\n", directory); return 2;
 }
 std::vector<std::filesystem::path> paths;
 for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
  if (it->is_regular_file(ec) && it->path().extension() == ".rpf") paths.push_back(it->path());
 if (ec) { std::fprintf(stderr, "plan benchmark: directory scan failed: %s\n", directory); return 2; }
 std::sort(paths.begin(), paths.end());
 if (paths.empty()) { std::fprintf(stderr, "plan benchmark: no .rpf files in %s\n", directory); return 2; }

 struct Item { std::filesystem::path path; ResourcePlanFixture fixture; ResourcePlanFixture::ReplayBacking backing; };
 std::vector<std::unique_ptr<Item>> items;
 items.reserve(paths.size());
 for (const auto& path : paths) {
  auto item = std::make_unique<Item>();
  item->path = path;
  if (!item->fixture.Load(path.string())) {
   std::fprintf(stderr, "plan benchmark: failed to load %s\n", path.string().c_str()); return 2;
  }
  item->backing = ResourcePlanFixture::ReplayBacking(item->fixture);
  auto checked = item->fixture.ReplayPrepared(item->backing);
  if (!checked.matched || checked.missing || checked.conflicts) {
   std::fprintf(stderr, "plan benchmark: correctness failed for %s (matched=%d missing=%llu conflicts=%llu)\n",
    path.string().c_str(), checked.matched, static_cast<unsigned long long>(checked.missing),
    static_cast<unsigned long long>(checked.conflicts)); return 1;
  }
  items.push_back(std::move(item));
 }
 // Warm each plan once before starting the timer.
 for (auto& item : items) {
  const auto warm = item->fixture.ReplayPrepared(item->backing);
  if (!warm.matched || warm.missing || warm.conflicts) {
   std::fprintf(stderr, "plan benchmark: warm-up failed for %s\n", item->path.string().c_str()); return 1;
  }
 }

 uint64_t total_us = 0, total_ordinary = 0, total_strict = 0, total_recursive = 0, total_tape = 0;
 uint64_t total_scalar_batch_words = 0, total_gather_eligible = 0, total_gather_hits = 0;
 uint64_t total_schedule_steps = 0, total_cfg_scheduled = 0;
 std::printf("plan benchmark: plans=%zu iterations=%llu (correctness passed)\n", items.size(),
  static_cast<unsigned long long>(iterations));
 for (auto& item : items) {
  uint64_t ordinary = 0, strict = 0, recursive = 0, tape = 0, scalar_batch_words = 0;
  uint64_t gather_eligible = 0, gather_hits = 0;
  uint64_t schedule_steps = 0, cfg_scheduled = 0;
  const auto start = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < iterations; ++i) {
   const auto r = item->fixture.ReplayPrepared(item->backing);
   if (!r.matched || r.missing || r.conflicts) {
    std::fprintf(stderr, "plan benchmark: iteration failed for %s (missing=%llu conflicts=%llu)\n",
     item->path.string().c_str(), static_cast<unsigned long long>(r.missing),
     static_cast<unsigned long long>(r.conflicts)); return 1;
   }
   ordinary += r.ordinary_calls; strict += r.strict_calls;
   recursive += r.recursive_ops; tape += r.tape_ops;
   scalar_batch_words += r.scalar_batch_words;
   gather_eligible += r.cfg_gather_eligible; gather_hits += r.cfg_gather_hits;
   schedule_steps += r.cfg_schedule_steps; cfg_scheduled += r.cfg_scheduled;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
   std::chrono::steady_clock::now() - start).count();
  const uint64_t us = static_cast<uint64_t>(elapsed);
  total_us += us; total_ordinary += ordinary; total_strict += strict;
  total_recursive += recursive; total_tape += tape;
  total_scalar_batch_words += scalar_batch_words;
  total_gather_eligible += gather_eligible; total_gather_hits += gather_hits;
  total_schedule_steps += schedule_steps; total_cfg_scheduled += cfg_scheduled;
  std::printf("  %s avg_us=%.2f ordinary_calls/eval=%llu strict_calls/eval=%llu recursive/eval=%llu tape/eval=%llu scalar_batch_words/eval=%llu cfg_schedule_steps/eval=%llu cfg_scheduled/eval=%llu cfg_gather_eligible/eval=%llu cfg_gather_hits/eval=%llu\n",
   item->path.filename().string().c_str(), static_cast<double>(us) / iterations,
   static_cast<unsigned long long>(ordinary / iterations), static_cast<unsigned long long>(strict / iterations),
   static_cast<unsigned long long>(recursive / iterations), static_cast<unsigned long long>(tape / iterations),
   static_cast<unsigned long long>(scalar_batch_words / iterations),
   static_cast<unsigned long long>(schedule_steps / iterations), static_cast<unsigned long long>(cfg_scheduled / iterations),
   static_cast<unsigned long long>(gather_eligible / iterations), static_cast<unsigned long long>(gather_hits / iterations));
 }
 std::printf("aggregate: avg_us_per_plan=%.2f total_us=%llu ordinary_calls/corpus_sweep=%llu strict_calls/corpus_sweep=%llu recursive/corpus_sweep=%llu tape/corpus_sweep=%llu scalar_batch_words/corpus_sweep=%llu cfg_schedule_steps/corpus_sweep=%llu cfg_scheduled/corpus_sweep=%llu cfg_gather_eligible/corpus_sweep=%llu cfg_gather_hits/corpus_sweep=%llu\n",
  static_cast<double>(total_us) / (items.size() * iterations), static_cast<unsigned long long>(total_us),
  static_cast<unsigned long long>(total_ordinary / iterations), static_cast<unsigned long long>(total_strict / iterations),
  static_cast<unsigned long long>(total_recursive / iterations), static_cast<unsigned long long>(total_tape / iterations),
  static_cast<unsigned long long>(total_scalar_batch_words / iterations),
  static_cast<unsigned long long>(total_schedule_steps / iterations), static_cast<unsigned long long>(total_cfg_scheduled / iterations),
  static_cast<unsigned long long>(total_gather_eligible / iterations), static_cast<unsigned long long>(total_gather_hits / iterations));
 const auto tape_stats = GetSrtTapeStats();
 std::printf("cfg_trace: hits=%llu misses=%llu recorded=%llu\n", static_cast<unsigned long long>(tape_stats.cfg_trace_hits),
  static_cast<unsigned long long>(tape_stats.cfg_trace_misses), static_cast<unsigned long long>(tape_stats.cfg_trace_recorded));
 const auto spec = GetResourceSpecializationCacheStats();
 std::printf("specialization_cache: enabled=%d calls=%llu buffer_hits=%llu buffer_misses=%llu image_hits=%llu image_misses=%llu indirect_uncached=%llu\n",
  Common::RuntimeFlags::Current().resource_spec_cache, static_cast<unsigned long long>(spec.calls),
  static_cast<unsigned long long>(spec.buffer_hits), static_cast<unsigned long long>(spec.buffer_misses),
  static_cast<unsigned long long>(spec.image_hits), static_cast<unsigned long long>(spec.image_misses),
  static_cast<unsigned long long>(spec.indirect_uncached));
 return 0;
}

// Frozen providers isolate ResourceMemo capture/probe overhead from GPU work.
// Unique shader bases force misses only when the graph cannot observe that input.
static int BenchmarkResourceMemo(const char* directory, uint64_t iterations) {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 if (iterations == 0 || iterations > 1000000) {
  std::fprintf(stderr,"resource memo benchmark: iterations must be in 1..1000000\n"); return 2;
 }
 std::error_code ec;
 const std::filesystem::path root(directory);
 if (!std::filesystem::is_directory(root,ec) || ec) return 2;
 std::vector<std::filesystem::path> paths;
 for (std::filesystem::directory_iterator it(root,ec),end; !ec && it!=end; it.increment(ec)) {
  if (it->is_regular_file(ec) && it->path().extension()==".rpf") paths.push_back(it->path());
 }
 if (ec || paths.empty()) return 2;
 std::sort(paths.begin(),paths.end());
 struct Arm {
  Libs::Graphics::ResourceMemo memo{16};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  uint64_t sequence=0, ns=0, evaluations=0, hits=0, reads=0, words=0;
  uint64_t ordinary=0, strict=0;
 };
 struct Item {
  std::filesystem::path path;
  ResourcePlanFixture fixture;
  ResourcePlanFixture::ReplayBacking backing;
  std::array<Arm,3> arms;
 };
 std::vector<std::unique_ptr<Item>> items;
 uint64_t skipped_base=0, skipped_failure=0, skipped_uncacheable=0;
 const auto runtime_for=[](Item& item) {
  auto& f=item.fixture;
  SrtRuntime runtime{.user_data=f.user_data,.shader_base=f.shader_base,
      .userdata=&item.backing,.workgroup_counts=f.workgroups};
  if (f.ordinary_reader) runtime.read_memory=+[](void* p,uint64_t address,std::span<uint32_t> words) {
   return static_cast<ResourcePlanFixture::ReplayBacking*>(p)->Read(false,address,words);
  };
  else runtime.read_memory=+[](void* p,uint64_t,std::span<uint32_t>) {
   auto& result=static_cast<ResourcePlanFixture::ReplayBacking*>(p)->result;
   ++result.ordinary_calls; ++result.missing; return false;
  };
  if (f.strict_reader) runtime.read_specialization_memory=+[](void* p,uint64_t address,std::span<uint32_t> words) {
   return static_cast<ResourcePlanFixture::ReplayBacking*>(p)->Read(true,address,words);
  };
  if (f.ordinary_reader) runtime.try_read_memory_span=+[](void* p,uint64_t address,std::span<uint32_t> words) {
   return static_cast<ResourcePlanFixture::ReplayBacking*>(p)->TryRead(false,address,words);
  };
  if (f.strict_reader) runtime.try_read_specialization_span=+[](void* p,uint64_t address,std::span<uint32_t> words) {
   return static_cast<ResourcePlanFixture::ReplayBacking*>(p)->TryRead(true,address,words);
  };
  return runtime;
 };
 const auto matches=[](const Item& item,const Arm& arm) {
  const auto& expected=item.fixture.expected;
  const auto& actual=arm.snapshot;
  return !item.backing.result.missing && !item.backing.result.conflicts &&
   actual.buffers==expected.buffers && actual.images==expected.images && actual.samplers==expected.samplers &&
   actual.flattened_srt==expected.flattened_srt && actual.user_data==expected.user_data &&
   ResourcePlanFixture::SameRangeCoverage(actual.specialization_reads,expected.specialization_reads) &&
   actual.uniform_fill==expected.uniform_fill && arm.specialization==item.fixture.specialization &&
   item.fixture.plan.active_sources==item.fixture.active;
 };
 const auto run=[&](Item& item,size_t index,bool& reused) {
  auto& arm=item.arms[index];
  auto runtime=runtime_for(item);
  // Both miss arms receive identical changing inputs. Stable-hit calibration
  // holds the captured base fixed; excluded graphs cannot observe either base.
  if (index!=2) runtime.shader_base=++arm.sequence;
  return arm.memo.Refresh(item.fixture.plan,runtime,arm.snapshot,arm.specialization,index!=0,reused);
 };
 for (const auto& path:paths) {
  auto item=std::make_unique<Item>(); item->path=path;
  if (!item->fixture.Load(path.string())) {
   std::fprintf(stderr,"resource memo benchmark: load failed %s\n",path.string().c_str()); return 2;
  }
  item->backing=ResourcePlanFixture::ReplayBacking(item->fixture);
  const auto checked=item->fixture.ReplayPrepared(item->backing);
  if (!checked.matched || checked.missing || checked.conflicts) {
   std::fprintf(stderr,"resource memo benchmark: fixture mismatch %s\n",path.string().c_str()); return 1;
  }
  if (std::ranges::any_of(item->fixture.plan.value_storage,[](const auto& inst) {
       return inst.GetOpcode()==ValueOpcode::GetShaderBase;
      })) {
   ++skipped_base; std::printf("skip %s: graph observes shader_base\n",path.filename().string().c_str()); continue;
  }
  if (!item->fixture.success) {
   ++skipped_failure; std::printf("skip %s: recorded failure cannot calibrate a stable hit\n",path.filename().string().c_str()); continue;
  }
  bool cacheable=true;
  // Fill every miss-bank slot and warm retained vectors before measuring.
  for (size_t arm=0;arm<3;++arm) for (uint32_t warm=0;warm<17;++warm) {
   item->backing.Reset(); bool reused=false;
   if (!run(*item,arm,reused) || !matches(*item,item->arms[arm]) || (arm!=2 && reused)) {
    std::fprintf(stderr,"resource memo benchmark: arm %zu warm-up mismatch %s\n",arm,path.string().c_str()); return 1;
   }
   if (arm==2 && warm!=0 && !reused) cacheable=false;
  }
  if (!cacheable) {
   ++skipped_uncacheable; std::printf("skip %s: stable state is not retained by ResourceMemo\n",path.filename().string().c_str()); continue;
  }
  items.push_back(std::move(item));
 }
 if (items.empty()) {
  std::fprintf(stderr,"resource memo benchmark: no eligible cacheable plans (base=%llu failed=%llu uncacheable=%llu)\n",
   static_cast<unsigned long long>(skipped_base),static_cast<unsigned long long>(skipped_failure),
   static_cast<unsigned long long>(skipped_uncacheable)); return 2;
 }
 std::printf("resource memo benchmark: eligible=%zu loaded=%zu skipped_shader_base=%llu skipped_failure=%llu skipped_uncacheable=%llu iterations=%llu rounds=3 flag_bits=%llu (all arms verified before timing)\n",
  items.size(),paths.size(),static_cast<unsigned long long>(skipped_base),static_cast<unsigned long long>(skipped_failure),
  static_cast<unsigned long long>(skipped_uncacheable),static_cast<unsigned long long>(iterations),
  static_cast<unsigned long long>(Common::RuntimeFlags::Bits(Common::RuntimeFlags::Current())));
 std::array<uint64_t,3> total_ns{},total_evaluations{};
 for (auto& item:items) {
  // Rotate arm order to avoid consistently assigning the earliest window to bypass.
  for (size_t round=0;round<3;++round) for (size_t order=0;order<3;++order) {
   const size_t index=(round+order)%3; auto& arm=item->arms[index];
   item->backing.Reset(); uint64_t hits=0;
   const auto start=std::chrono::steady_clock::now();
   for (uint64_t iteration=0;iteration<iterations;++iteration) {
    bool reused=false;
    if (!run(*item,index,reused) || (index!=2 && reused) || (index==2 && !reused)) {
     std::fprintf(stderr,"resource memo benchmark: measured outcome failed %s arm=%zu\n",item->path.string().c_str(),index); return 1;
    }
    hits+=reused;
   }
   const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
   if (!matches(*item,arm)) {
    std::fprintf(stderr,"resource memo benchmark: measured output mismatch %s arm=%zu\n",item->path.string().c_str(),index); return 1;
   }
   arm.ns+=static_cast<uint64_t>(ns); arm.evaluations+=iterations; arm.hits+=hits;
   arm.reads+=item->backing.result.requests; arm.words+=item->backing.result.words;
   arm.ordinary+=item->backing.result.ordinary_calls; arm.strict+=item->backing.result.strict_calls;
  }
  std::printf("  %s",item->path.filename().string().c_str());
  const char* names[]={"bypass","forced_miss","stable_hit"};
  for (size_t index=0;index<3;++index) {
   const auto& arm=item->arms[index]; total_ns[index]+=arm.ns; total_evaluations[index]+=arm.evaluations;
   std::printf(" %s_us=%.3f reads/eval=%.2f words/eval=%.2f ordinary/eval=%.2f strict/eval=%.2f hits=%llu/%llu",
    names[index],static_cast<double>(arm.ns)/arm.evaluations/1000.0,
    static_cast<double>(arm.reads)/arm.evaluations,static_cast<double>(arm.words)/arm.evaluations,
    static_cast<double>(arm.ordinary)/arm.evaluations,static_cast<double>(arm.strict)/arm.evaluations,
    static_cast<unsigned long long>(arm.hits),static_cast<unsigned long long>(arm.evaluations));
  }
  std::printf("\n");
 }
 std::printf("workload-weighted aggregate (equal iterations per eligible frozen plan):");
 const char* names[]={"bypass","forced_miss","stable_hit"};
 for (size_t index=0;index<3;++index) {
  std::printf(" %s_us=%.3f total_us=%.3f evaluations=%llu",names[index],
   static_cast<double>(total_ns[index])/total_evaluations[index]/1000.0,
   static_cast<double>(total_ns[index])/1000.0,static_cast<unsigned long long>(total_evaluations[index]));
 }
 std::printf("\nFrozen provider timings include callback/capture/probe/output-copy work; base misses underestimate register-probe costs. Equal plan weights are not live draw frequencies or FPS.\n");
 return 0;
}

void TestResourcePlanFixture() {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 // Real extracted graph with a memory load, flattened SRT, descriptor and buffer
 // specialization. The round trip runs the production materializer, not a mock.
 auto plan = SrtPlan(0x2000);
 plan.shader_hash = 0x12345678;
 plan.descriptor_sources[0].dword_count = 4;
 plan.descriptor_sources[0].dwords[2] = Value(16u);
 plan.descriptor_sources[0].dwords[3] = Value(0u);
 plan.descriptor_gathers.clear();
 plan.info.buffers.push_back({.source=0});
 std::array<uint32_t,4> user = {0x2000,0,17,19};
 std::array<uint32_t,3> groups = {2,3,4};
 SrtRuntime runtime {.user_data=user,.shader_base=0x100000,.read_memory=
  +[](void*,uint64_t a,std::span<uint32_t> v) { if (a != 0x2000 || v.size()!=1) return false; v[0]=0x4000; return true; },
  .read_specialization_memory=+[](void*,uint64_t a,std::span<uint32_t> v) { if (a != 0x2000 || v.size()!=1) return false; v[0]=0x4000; return true; },
  .workgroup_counts=groups};
 ResourceSnapshot snapshot; ResourceSpecialization spec;
 const char* path = "resource-plan-roundtrip.rpf";
 Check(CaptureResourcePlan(plan,runtime,snapshot,spec,path),"record real materialization");
 ResourcePlanFixture fixture;
 Check(fixture.Load(path),"load serialized resource graph");
 Check(fixture.plan.shader_hash == plan.shader_hash && fixture.user_data == std::vector<uint32_t>(user.begin(),user.end()) &&
  fixture.workgroups == std::vector<uint32_t>(groups.begin(),groups.end()),"plan identity and runtime preserved");
 const auto first = fixture.Replay();
 Check(first.matched && first.success && !first.missing && first.requests > 0,"real plan replay matches snapshot and specialization");
 Check(first.ops > 0 && first.ops == fixture.captured_ops,"same evaluator operation count is reproducible");
 Check(fixture.Replay().matched,"replay again with fresh materialization outputs");
 fixture.expected.buffers[0].dwords[0] ^= 1;
 Check(!fixture.Replay().matched,"wrong recorded descriptor result rejected");
 fixture.expected.buffers[0].dwords[0] ^= 1;
 const auto captured = fixture.reads;
 fixture.reads.clear();
 const auto missing = fixture.Replay();
 Check(!missing.matched && missing.missing,"missing capture never uses live memory");
 fixture.reads = captured;
 auto conflict = fixture.reads.front(); conflict.words[0] ^= 1; fixture.reads.push_back(conflict);
 Check(fixture.Replay().conflicts && !fixture.Replay().matched,"contradictory same-domain memory rejects replay");
 fixture.reads = captured;
 auto other = fixture.reads.front(); other.strict = !other.strict; other.words[0] ^= 1; fixture.reads.push_back(other);
 Check(!fixture.Replay().conflicts,"strict and ordinary memory remain separate");
 // Conditional graph: an inactive SRT read is intentionally uncaptured, and
 // the strict reader returns a different word for the same guest address.
 plan.capture_specialization_reads = true;
 plan.clean_flat_slots = {0u,1u};
 plan.srt_reads.push_back({plan.srt_reads[0].value,1u});
 auto &condition = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
 condition.SetArg(0,Value(static_cast<ScalarReg>(2)));
 auto &poison = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
 poison.SetFlags(MemoryFlags{.index=0});
 auto &address = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
 address.SetArg(0,Value(0xdead0000u)); address.SetArg(1,Value(0u));
 poison.SetArg(0,Value(&address)); poison.SetArg(1,Value(0u));
 poison.SetArg(2,Value(0u)); poison.SetArg(3,Value(true));
 plan.srt_reads.push_back({Value(&poison),2u});
 plan.clean_flat_slots.push_back(0);
 plan.control_flow.resize(3);
 plan.control_flow[0].condition = Value(&condition);
 plan.control_flow[0].successors = {1,2};
 plan.control_flow[1].srt_reads = {0,1}; plan.control_flow[1].sources = {0};
 plan.control_flow[2].srt_reads = {2};
 runtime.read_specialization_memory = +[](void*,uint64_t a,std::span<uint32_t> v) {
  if (a!=0x2000 || v.size()!=1) return false; v[0]=0x8000; return true;
 };
 Check(CaptureResourcePlan(plan,runtime,snapshot,spec,"resource-plan-cfg.rpf"),"capture conditional dual-reader materialization");
 ResourcePlanFixture cfg;
 Check(cfg.Load("resource-plan-cfg.rpf") && cfg.Replay().matched,"CFG activity and strict reader replay");
 Check(cfg.reads.size()==2 && cfg.expected.flattened_srt==std::vector<uint32_t>{0x4000,0x8000,0},"inactive poison is not read and domains differ");
 Check(cfg.Replay().ops>0 && cfg.Replay().ops==cfg.captured_ops,"CFG evaluator operations reproducible");
 cfg.user_data[2]=0;
 Check(cfg.Replay().missing && !cfg.Replay().matched,"changed CFG branch reports uncaptured memory");
 cfg.user_data[2]=17;
 // A captured failure is distinct from a missing read and faithfully replays.
 runtime.read_memory = +[](void*,uint64_t,std::span<uint32_t>) { return false; };
 Check(!CaptureResourcePlan(plan,runtime,snapshot,spec,"resource-plan-failed.rpf"),"record failed materialization");
 ResourcePlanFixture failed;
 Check(failed.Load("resource-plan-failed.rpf") && failed.Replay().matched && !failed.Replay().success && !failed.Replay().missing,"known failure replays without treating it as missing");
 // A truncated archive must not pass as an empty/failed evaluation.
 std::ifstream original(path,std::ios::binary);
 std::string bytes((std::istreambuf_iterator<char>(original)),{}); original.close();
 std::ofstream truncated("resource-plan-truncated.rpf",std::ios::binary); truncated.write(bytes.data(),bytes.size()-1); truncated.close();
 ResourcePlanFixture bad;
 Check(!bad.Load("resource-plan-truncated.rpf"),"short plan archive rejected");
 std::remove("resource-plan-truncated.rpf");
}
#include "SrtBatchTests.inc"
#include "ResourceMemoRuntimeTests.inc"
#include "ScalarBatchTests.inc"
#include "ScalarBatchFixtureTests.inc"
#include "ScalarBatchMemoTests.inc"
#include "ResourceSpecializationCacheTests.inc"
#include "CompiledSrtTests.inc"
void TestCfgDescriptorGatherMemoShortcut() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct Reads { std::vector<uint64_t> addresses; } reads;
  ResourcePlan plan;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  auto make_read = [&](uint32_t address) {
    auto &h = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    h.SetArg(0, Value(address)); h.SetArg(1, Value(0u));
    auto &r = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    r.SetFlags(MemoryFlags{.index = 0});
    r.SetArg(0, Value(&h)); r.SetArg(1, Value(0u)); r.SetArg(2, Value(0u)); r.SetArg(3, Value(true));
    return Value(&r);
  };
  const auto selected = make_read(0x1000u);
  const auto replacement = make_read(0x2000u);
  const auto poison = make_read(0xdead0000u);
  auto &select = plan.value_storage.emplace_back(ValueOpcode::SelectU32);
  select.SetArg(0, Value(true)); select.SetArg(1, selected); select.SetArg(2, poison);
  plan.srt_reads.push_back({Value(&select), 0u});
  auto &srt = plan.value_storage.emplace_back(ValueOpcode::GetSrtResource);
  auto &read_const = plan.value_storage.emplace_back(ValueOpcode::ReadConst);
  read_const.SetArg(0, Value(&srt)); read_const.SetArg(1, Value(0u));
  DescriptorSource descriptor;
  descriptor.dwords[0] = Value(&read_const); descriptor.dword_count = 1u;
  plan.descriptor_sources.push_back(descriptor);
  ResourcePlan::DescriptorGather gather;
  gather.valid = true; gather.needs_flat = true;
  gather.words[0] = {.value = 0u, .kind = ResourcePlan::DescriptorGather::Word::Kind::Flat,
                     .read_slot = 0u};
  plan.descriptor_gathers.push_back(gather);
  plan.control_flow.resize(1);
  plan.control_flow[0].srt_reads = {0u}; plan.control_flow[0].sources = {0u};

  const auto reader = +[](void *data, uint64_t address, std::span<uint32_t> values) {
    auto &state = *static_cast<Reads *>(data);
    state.addresses.push_back(address);
    if ((address != 0x1000u && address != 0x2000u) || values.size() != 1u) return false;
    values[0] = address == 0x1000u ? 0x12345678u : 0xabcdef01u; return true;
  };
  struct Outcome { DescriptorValue descriptor; SrtEvaluationCounts counts; std::vector<uint64_t> addresses; bool ok; };
  const auto run = [&](bool enabled, bool invalidate_after_refresh = false) {
    // Only the gather bit: FlagSet defaults now turn other materialization paths on.
    const uint64_t gather_bit = Common::RuntimeFlags::Bits(Common::RuntimeFlags::FlagSet {.srt_cfg_gather = true}) ^
                                Common::RuntimeFlags::Bits(Common::RuntimeFlags::FlagSet {.srt_cfg_gather = false});
    const auto old = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    Common::RuntimeFlags::g_published.store((old & ~Common::RuntimeFlags::kFlagMask) |
      (enabled ? gather_bit : 0u), std::memory_order_relaxed);
    reads.addresses.clear();
    Outcome result;
    SrtRuntime runtime {.read_memory = reader, .userdata = &reads, .evaluation_counts = &result.counts};
    std::vector<uint32_t> flat;
    SrtWalker walker(plan, runtime);
    result.ok = walker.RefreshFlatBuffer(flat);
    if (result.ok && invalidate_after_refresh) plan.srt_reads[0].value = replacement;
    if (result.ok) result.ok = walker.EvaluateDescriptor(0u, result.descriptor);
    if (invalidate_after_refresh) plan.srt_reads[0].value = Value(&select);
    result.addresses = reads.addresses;
    return result;
  };
  const auto published = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
  const auto pinned = Common::RuntimeFlags::g_evaluation_bits;
  Common::RuntimeFlags::g_evaluation_bits = Common::RuntimeFlags::kUnpinnedBits;
  const auto off = run(false);
  const auto on = run(true);
  const auto invalidated = run(true, true);
  Common::RuntimeFlags::g_published.store(published, std::memory_order_relaxed);
  Common::RuntimeFlags::g_evaluation_bits = pinned;
  Check(off.ok && on.ok && off.descriptor == on.descriptor &&
            off.descriptor.dwords[0] == 0x12345678u &&
            off.addresses == on.addresses && on.addresses == std::vector<uint64_t>{0x1000u} &&
            on.counts.cfg_gather_eligible != 0u && on.counts.cfg_gather_hits != 0u &&
            on.counts.recursive_ops < off.counts.recursive_ops && invalidated.ok &&
            invalidated.descriptor.dwords[0] == 0xabcdef01u &&
            invalidated.addresses == std::vector<uint64_t>{0x1000u, 0x2000u} &&
            invalidated.counts.cfg_gather_hits == 0u,
        "CFG descriptor gather changed reads/results or did not skip successful memo lookup");
}


// Offline structural dump of one plan's recorded CFG trace trie (--dump-trace): step mix and an
// abbreviated listing. Diagnostic only.
static int DumpCfgTrace(const char* file, uint32_t limit) {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 using TK = ResourcePlan::CfgTrace::Kind;
 using OK = ResourcePlan::SrtTape::Kind;
 ResourcePlanFixture fx;
 if (!fx.Load(file)) { std::fprintf(stderr,"dump-trace: load failed\n"); return 2; }
 auto backing = ResourcePlanFixture::ReplayBacking(fx);
 for (int i=0;i<4;++i) fx.ReplayPrepared(backing);
 const size_t variant = fx.plan.cfg_traces[1].steps.size() > fx.plan.cfg_traces[0].steps.size() ? 1 : 0;
 std::printf("variant=%zu (steps v0=%zu v1=%zu)\n",variant,fx.plan.cfg_traces[0].steps.size(),fx.plan.cfg_traces[1].steps.size());
 const auto& trace = fx.plan.cfg_traces[variant];
 const auto& tape = fx.plan.cfg_srt_tapes[variant];
 std::printf("plan %s: steps=%zu paths=%u hits=%u misses=%u tape_ops=%zu srt_reads=%zu blocks=%zu sources=%zu\n",file,
  trace.steps.size(),trace.paths,trace.hits,trace.misses,tape.ops.size(),fx.plan.srt_reads.size(),
  fx.plan.control_flow.size(),fx.plan.descriptor_sources.size());
 std::map<std::string,uint32_t> mix;
 uint32_t shown=0;
 // One path only: follow the trie from step 0 taking the first recorded outcome.
 std::vector<uint32_t> path;
 for (uint32_t pc=0; pc<trace.steps.size();) {
  const auto& st=trace.steps[pc]; path.push_back(pc);
  if (st.kind==TK::End) break;
  if (st.kind==TK::Cond||st.kind==TK::CondTri||st.kind==TK::CondEval) {
   uint32_t nx=UINT32_MAX; for (auto n: st.next) if (n!=UINT32_MAX) { nx=n; break; }
   if (nx==UINT32_MAX) break; pc=nx;
  } else ++pc;
 }
 std::printf("first path length=%zu\n",path.size());
 for (auto pc: path) {
  const auto& st=trace.steps[pc];
  std::string key;
  switch (st.kind) {
   case TK::Exec: { const auto& op=tape.ops[st.a];
     key = op.kind==OK::Immediate?"Exec:Imm":op.kind==OK::UserData?"Exec:UserData":op.kind==OK::ShaderBase?"Exec:ShaderBase":
       op.kind==OK::RawRead?std::string("Exec:Read:")+std::string(ValueOpcodeName(op.opcode)):std::string("Exec:Alu:")+std::string(ValueOpcodeName(op.opcode));
     key += op.memo_index!=UINT32_MAX?"(memo)":"(nomemo)"; break; }
   case TK::Memo: key="Memo"; break; case TK::Flat: key="Flat"; break; case TK::Activate: key="Activate"; break;
   case TK::Cond: key="Cond"; break; case TK::CondTri: key="CondTri"; break; case TK::Eval: key="Eval"; break;
   case TK::CondEval: key="CondEval"; break; case TK::Gather: key="Gather"; break; case TK::End: key="End"; break;
  }
  ++mix[key];
  if (shown<limit) {
   ++shown;
   if (st.kind==TK::Exec) { const auto& op=tape.ops[st.a];
    if (op.kind==OK::RawRead) { const auto& mi=fx.plan.memory_info[op.aux];
     std::printf("  [%u] READ op%u %s lo=op%u hi=op%u off=op%u imm=%d\n",pc,st.a,std::string(ValueOpcodeName(op.opcode)).c_str(),op.args[0],op.args[1],op.args[2],static_cast<int32_t>(mi.offset)); }
    else std::printf("  [%u] op%u kind=%d %s a=%u,%u,%u imm=%llu\n",pc,st.a,(int)op.kind,std::string(ValueOpcodeName(op.opcode)).c_str(),op.args[0],op.args[1],op.args[2],(unsigned long long)op.imm);
   } else std::printf("  [%u] %s a=%u b=%u\n",pc,st.kind==TK::Cond?"Cond":st.kind==TK::CondTri?"CondTri":st.kind==TK::Flat?"Flat":st.kind==TK::Memo?"Memo":st.kind==TK::Activate?"Activate":st.kind==TK::Gather?"Gather":"Other",st.a,st.b);
  }
 }
 std::printf("first-path step mix:\n");
 for (auto& [k,v]: mix) std::printf("  %5u  %s\n",v,k.c_str());
 return 0;
}

#if KYTY_SRT_PROFILE
// --profile-plans: TSC breakdown of CFG plan materialization (offline, hashed fixture reader). Counts
// every heap allocation made during a call via the replaced global operator new below.
static uint64_t g_alloc_count = 0, g_alloc_bytes = 0;
static std::vector<size_t>* g_alloc_log = nullptr;
void* operator new(size_t n) { ++g_alloc_count; g_alloc_bytes += n; if (g_alloc_log) g_alloc_log->push_back(n); void* p = std::malloc(n ? n : 1); return p; }
void* operator new[](size_t n) { ++g_alloc_count; g_alloc_bytes += n; void* p = std::malloc(n ? n : 1); return p; }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

static int ProfilePlans(const char* directory, uint64_t iterations, bool compiled, bool all_plans) {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 std::error_code ec;
 std::vector<std::filesystem::path> paths;
 for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec))
  if (it->is_regular_file(ec) && it->path().extension() == ".rpf") paths.push_back(it->path());
 std::sort(paths.begin(), paths.end());
 // TSC rate and per-probe overhead.
 const auto c0 = std::chrono::steady_clock::now(); const uint64_t t0 = __rdtsc();
 while (std::chrono::steady_clock::now() - c0 < std::chrono::milliseconds(200)) {}
 const double ghz = double(__rdtsc() - t0) / double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - c0).count());
 SrtProf::g = {}; SrtProf::g.enabled = true;
 for (int i = 0; i < 1000000; ++i) { SRT_PROF_SCOPE(SrtProf::Preamble); }
 const double scope_cycles = double(SrtProf::g.cycles[SrtProf::Preamble]) / 1e6;
 SrtProf::g = {};
 std::printf("profile: executor=%s tsc=%.3f GHz scope_pair_overhead=%.1f cycles (%.1f ns)\n", compiled ? "compiled" : "trace-replay", ghz, scope_cycles, scope_cycles / ghz);
 struct Row { std::string name; double total_ns = 0, ns[SrtProf::Count] = {}; double steps = 0, rcalls = 0, rwords = 0, allocs = 0, alloc_bytes = 0, eval_cnt = 0; };
 std::vector<Row> rows;
 CompiledSrtTest::SetCompiled(compiled);
 for (const auto& path : paths) {
  ResourcePlanFixture fx;
  if (!fx.Load(path.string())) return 2;
  if (!all_plans && fx.plan.control_flow.empty()) continue;
  auto backing = ResourcePlanFixture::ReplayBacking(fx);
  backing.BuildFast();
  for (int w = 0; w < 50; ++w) { const auto r = fx.ReplayPrepared(backing); if (!r.matched) { std::fprintf(stderr, "profile: mismatch %s\n", path.string().c_str()); return 1; } }
  { static std::vector<size_t> log; log.reserve(64); g_alloc_log = &log; fx.ReplayPrepared(backing); g_alloc_log = nullptr;
    std::printf("allocs %s:", path.filename().string().c_str()); for (auto v : log) std::printf(" %zu", v); std::printf("\n"); log.clear(); }
  SrtProf::g.enabled = false;
  const auto a0 = g_alloc_count, b0 = g_alloc_bytes;
  const auto start = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < iterations; ++i) fx.ReplayPrepared(backing);
  const double ns_plain = double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()) / iterations;
  const double allocs = double(g_alloc_count - a0) / iterations, abytes = double(g_alloc_bytes - b0) / iterations;
  // Reader isolation: log one call's reader sequence, then replay it in bulk (no per-call probes).
  std::vector<SrtProf::Data::Call> calls;
  { SrtProf::g = {}; SrtProf::g.log = &calls; fx.ReplayPrepared(backing); SrtProf::g.log = nullptr; }
  double reader_bulk_ns = 0, capture_bulk_ns = 0;
  { std::array<uint32_t, 16> out{}; std::vector<std::pair<uint64_t, uint64_t>> ranges; ranges.reserve(calls.size());
    const uint64_t reps = 2000; uint64_t sink = 0;
    auto t = std::chrono::steady_clock::now();
    for (uint64_t r = 0; r < reps; ++r) for (const auto& c : calls) { backing.Read(c.strict, c.address, std::span<uint32_t>(out.data(), c.words)); sink += out[0]; }
    reader_bulk_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count()) / reps;
    t = std::chrono::steady_clock::now();
    for (uint64_t r = 0; r < reps; ++r) { ranges.clear(); for (const auto& c : calls) { backing.Read(c.strict, c.address, std::span<uint32_t>(out.data(), c.words)); ranges.emplace_back(c.address, c.words * 4ull); sink += out[0]; } }
    capture_bulk_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count()) / reps;
    if (sink == 42) std::printf(" ");
    std::printf("reader-isolated %s: calls=%zu fixture_read_us=%.2f (%.1f ns/call) +capture_push_us=%.2f\n", path.filename().string().c_str(), calls.size(), reader_bulk_ns / 1000, reader_bulk_ns / calls.size(), capture_bulk_ns / 1000); }
  SrtProf::g = {}; SrtProf::g.enabled = true;
  for (uint64_t i = 0; i < iterations; ++i) fx.ReplayPrepared(backing);
  SrtProf::g.enabled = false;
  Row row; row.name = path.filename().string(); row.total_ns = ns_plain;
  for (int k = 0; k < SrtProf::Count; ++k) row.ns[k] = double(SrtProf::g.cycles[k]) / ghz / iterations;
  row.steps = double(SrtProf::g.steps) / iterations; row.rcalls = double(SrtProf::g.reader_calls) / iterations;
  row.rwords = double(SrtProf::g.reader_words) / iterations; row.allocs = allocs; row.alloc_bytes = abytes;
  row.eval_cnt = double(SrtProf::g.count[SrtProf::EvalDesc]) / iterations;
  std::printf("evalmix %s: gather=%.1f recursive=%.1f reused=%.1f\n", path.filename().string().c_str(), double(SrtProf::g.eval_gather) / iterations, double(SrtProf::g.eval_recursive) / iterations, double(SrtProf::g.eval_reused) / iterations);
  rows.push_back(row);
 }
 CompiledSrtTest::SetCompiled(false);
 std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.total_ns > b.total_ns; });
 const auto print = [&](const Row& r) {
  std::printf("%s plain_us=%.2f | profiled(us): Mat=%.2f Refresh=%.2f EvalDesc=%.2f (n=%.0f) | Pre=%.2f UD=%.2f Alu=%.2f Read=%.2f[Reader=%.2f MemoW=%.2f] Imm=%.2f Memo=%.2f Flat=%.2f Act=%.2f Cond=%.2f Gath=%.2f End=%.2f | steps=%.0f rcalls=%.0f rwords=%.0f allocs=%.2f (%.0f B)\n",
   r.name.c_str(), r.total_ns / 1000, r.ns[SrtProf::MatTotal] / 1000, r.ns[SrtProf::Refresh] / 1000, r.ns[SrtProf::EvalDesc] / 1000, r.eval_cnt,
   r.ns[SrtProf::Preamble] / 1000, r.ns[SrtProf::StepUserData] / 1000, r.ns[SrtProf::StepAlu] / 1000, r.ns[SrtProf::StepRead] / 1000,
   r.ns[SrtProf::Reader] / 1000, r.ns[SrtProf::MemoWrite] / 1000, r.ns[SrtProf::StepImm] / 1000, r.ns[SrtProf::StepMemo] / 1000,
   r.ns[SrtProf::StepFlat] / 1000, r.ns[SrtProf::StepActivate] / 1000, r.ns[SrtProf::StepCond] / 1000, r.ns[SrtProf::StepGather] / 1000,
   r.ns[SrtProf::StepEnd] / 1000, r.steps, r.rcalls, r.rwords, r.allocs, r.alloc_bytes);
 };
 for (const auto& r : rows) print(r);
 Row avg; avg.name = "AVERAGE";
 for (const auto& r : rows) { avg.total_ns += r.total_ns; for (int k = 0; k < SrtProf::Count; ++k) avg.ns[k] += r.ns[k];
  avg.steps += r.steps; avg.rcalls += r.rcalls; avg.rwords += r.rwords; avg.allocs += r.allocs; avg.alloc_bytes += r.alloc_bytes; avg.eval_cnt += r.eval_cnt; }
 const double n = double(rows.size());
 avg.total_ns /= n; for (auto& v : avg.ns) v /= n; avg.steps /= n; avg.rcalls /= n; avg.rwords /= n; avg.allocs /= n; avg.alloc_bytes /= n; avg.eval_cnt /= n;
 print(avg);
 return 0;
}
#endif

int main(int argc, char **argv) {
  Common::RuntimeFlags::Refresh(true);
  // The legacy cases assert exact reader-call counts; the compiled executor merges adjacent loads.
  // Run them on the existing path and re-apply the requested flag for the modes that measure it.
  const bool want_compiled = CompiledSrtTest::CompiledRequested();
  CompiledSrtTest::SetCompiled(false);
  TestResourceSpecializationCacheTransitions();
  TestCfgDescriptorGatherMemoShortcut();
  if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--benchmark-resource-memo") == 0) {
    CompiledSrtTest::SetCompiled(want_compiled);
    uint64_t iterations = 1000;
    if (argc == 4) {
      char* end = nullptr;
      iterations = std::strtoull(argv[3], &end, 10);
      if (end == argv[3] || *end != '\0' || iterations == 0 || iterations > 1000000) {
        std::fprintf(stderr,"resource memo benchmark: invalid iteration count: %s\n",argv[3]); return 2;
      }
    }
    return BenchmarkResourceMemo(argv[2], iterations);
  }
  TestCfgScheduleReadFailureAndSharedDag();
  TestCfgFlagSnapshotSurvivesReaderPublication();
  TestResourceMemoRuntimeDependencySwitch();
  TestResourceMemoReuseTrace();
  TestResourceMemoHashFilterDependencies();
  TestScalarBatchDomainsFailureAndActivity();
  TestScalarBatchFixtureCaptureAndReplay();
  TestScalarBatchMemoCaptureHitAndFlagFlip();
  CompiledSrtTest::TestChangedRegisterMemoryAndBranch();
  CompiledSrtTest::TestDescriptorReuseAndDeadMemo();
  if (argc == 3 && std::strcmp(argv[1], "--replay-plan") == 0) {
    CompiledSrtTest::SetCompiled(want_compiled);
    const int status = ReplayPlanFixture(argv[2]);
    CompiledSrtTest::SetCompiled(false);
    if (status != 0) return status;
    // Same capture through the compiled executor: identical outcome, outputs and read sequence on the
    // captured inputs and on perturbed registers, memory and branch conditions.
    return CompiledSrtTest::DifferentialFile(argv[2], 64, true);
  }
  if (argc >= 3 && std::strcmp(argv[1], "--compiled-differential") == 0) {
    const uint32_t perturbations = argc == 4 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 200;
    return CompiledSrtTest::Differential(argv[2], perturbations);
  }
#if KYTY_SRT_PROFILE
  if (argc >= 3 && (std::strcmp(argv[1], "--profile-plans") == 0 || std::strcmp(argv[1], "--profile-plans-compiled") == 0)) {
    const uint64_t iterations = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 3000;
    return ProfilePlans(argv[2], iterations, std::strcmp(argv[1], "--profile-plans-compiled") == 0, argc >= 5);
  }
#endif
  if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--dump-trace") == 0) {
    CompiledSrtTest::SetCompiled(want_compiled);
    return DumpCfgTrace(argv[2], argc == 4 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 120);
  }
  if ((argc == 3 || argc == 4) && std::strcmp(argv[1], "--benchmark-plans") == 0) {
    char* end = nullptr;
    const uint64_t iterations = argc == 4 ? std::strtoull(argv[3], &end, 10) : 100;
    if (argc == 4 && (!end || *end != '\0')) {
      std::fprintf(stderr, "plan benchmark: invalid iteration count: %s\n", argv[3]); return 2;
    }
    CompiledSrtTest::SetCompiled(want_compiled);
    const int status = BenchmarkResourcePlans(argv[2], iterations);
    CompiledSrtTest::SetCompiled(false);
    if (status != 0) return status;
    return CompiledSrtTest::Benchmark(argv[2], iterations);
  }
  TestResourcePlanFixture();
  if (argc >= 3 && std::strcmp(argv[1], "--inspect-fixture") == 0) {
    return InspectFixture(argv[2]);
  }
  TestCfgTapeVerifierRejectsCorruption();
  TestCfgTapeDemandRoots();
  TestCfgTapeCompilationCycleFallsBack();
  TestVectorLoadUsesCleanDependencies(false, true);
  TestVectorLoadUsesCleanDependencies(true, true);
  TestTapeSeparatesCleanAndOrdinaryReadDomains(true);
  TestTapePreservesWideMemoryIndex();
  TestTapeDeduplicatesRawImmediateBits();
  TestTapeUnsupportedRootFallsBack();
  TestVectorLoadUsesCleanDependencies(false);
  TestVectorLoadUsesCleanDependencies(true);
  TestTapeSeparatesCleanAndOrdinaryReadDomains();
  TestTapeVerifyReplaysReaderDomains();
  TestTapeVerifyMismatchFallbackAndUnmatchedReplay();
  TestCompiledDescriptorGatherMatchesWalker();
  TestFlatDescriptorGatherRequiresOwnCompleteRefresh();
  TestResourceMemoTracksMemoryAndRuntime();
  TestResourceMemoHitVerifier();
  TestResourceMemoTracksConsumedRegisters();
  TestResourceMemoLazySelectDependencies();
  TestResourceMemoAlternatingObjects();
  TestSingleResourceMemoChangesOutput();
  TestMappedSrtUsesDirectReaderByDefault();
  TestNativeNestedSrtRefresh();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestFailedRuntimeReadIsMemoizedPerRefresh();
  TestCyclicRuntimeFailureCanRecover();
  TestUniformVectorDescriptorRead();
  TestGuardedVectorReadTapeFallback();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  for (const bool memory_condition : {false, true})
    TestWrittenDescriptorPredicateReads(memory_condition);
  TestFailedMaterializationRejectsStage();
  TestSharedControlPredicates();
  TestFiniteImageRefreshReusesScalarReads();
  TestContiguousScalarReadBatches();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestResourceFixtureRoundTrip();
  TestResourceFixtureDrivesRuntime();
  TestResourceFixtureAppendPath();
  TestResourceFixturePreservesEveryWord();
  TestResourceFixtureRepeatedFlush();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
