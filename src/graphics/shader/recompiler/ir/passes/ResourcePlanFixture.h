#pragma once

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include <bit>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {

// One evaluation, not a process-wide memory union. Separate reader domains and
// contradictory responses are retained so incomplete captures cannot claim success.
struct ResourcePlanFixture {
 struct Read {
  uint64_t address = 0;
  bool strict = false;
  bool valid = false;
  std::vector<uint32_t> words;
 };
 static bool SameRangeCoverage(const std::vector<std::pair<uint64_t,uint64_t>>& a,
                               const std::vector<std::pair<uint64_t,uint64_t>>& b) {
  using Range = std::pair<uint64_t,uint64_t>;
  const auto normalize = [](const auto& ranges, std::vector<Range>& result) {
   result.reserve(ranges.size());
   for (const auto& [address, size] : ranges) {
    if (!size) continue;
    if (size > std::numeric_limits<uint64_t>::max() - address) return false;
    result.emplace_back(address,address+size);
   }
   std::sort(result.begin(),result.end());
   size_t write = 0;
   for (size_t i = 0; i < result.size(); ++i) {
    if (write && result[i].first <= result[write-1].second)
     result[write-1].second = std::max(result[write-1].second,result[i].second);
    else result[write++] = result[i];
   }
   result.resize(write);
   return true;
  };
  std::vector<Range> left, right;
  return normalize(a,left) && normalize(b,right) && left == right;
 }
 ResourcePlan plan;
 std::vector<uint32_t> user_data, workgroups;
 uint64_t shader_base = 0;
 bool ordinary_reader = false, strict_reader = false, success = false;
 std::vector<Read> reads;
 ResourceSnapshot expected;
 ResourceSpecialization specialization;
 std::vector<uint8_t> active;
 uint64_t captured_ops = 0;
 uint64_t captured_recursive_ops = 0, captured_tape_ops = 0;
 // Opcodes added after a capture was recorded (native packed-f16, 2026-10-11) were inserted
 // mid-enum. Captures from before that point are loaded by skipping them in the schema
 // and shifting later numeric ids; a skip count of zero is the current schema.
 static bool LegacyOpcodeSkipped(ValueOpcode op) {
  return op == ValueOpcode::FPAdd16x2 || op == ValueOpcode::FPMul16x2 ||
         op == ValueOpcode::FPFma16x2 || op == ValueOpcode::FPClamp16x2;
 }
 static uint64_t OpcodeSchema(bool legacy = false) {
  uint64_t hash = 14695981039346656037ull;
  for (uint32_t i = 0; i < static_cast<uint32_t>(ValueOpcode::Count); ++i) {
   const auto op = static_cast<ValueOpcode>(i);
   if (legacy && LegacyOpcodeSkipped(op)) continue;
   for (unsigned char c : ValueOpcodeName(op)) { hash ^= c; hash *= 1099511628211ull; }
   hash ^= NumArgsOf(op); hash *= 1099511628211ull;
  }
  return hash;
 }

 // Portable field archive. No object bytes, pointers, padding or evaluation scratch
 // go into the file. Counts are bounded before allocation and short reads fail.
 struct Archive {
  std::istream* input = nullptr;
  std::ostream* output = nullptr;
  bool ok = true;
  std::unordered_map<const Inst*, uint32_t> ids;
  std::vector<Inst*> nodes;
  // Legacy capture: numeric opcodes at or above this id move up by legacy_shift.
  uint32_t legacy_first = UINT32_MAX, legacy_shift = 0;
  void operator()(bool& v) { uint32_t n = v; (*this)(n); if (n > 1) ok = false; v = n != 0; }
  template<class T> requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  void operator()(T& v) {
   using U = std::make_unsigned_t<T>;
   U n = static_cast<U>(v);
   for (size_t i = 0; i < sizeof(T); ++i) {
    if (input) { const int c = input->get(); if (c == EOF) { ok = false; return; }
     if (i == 0) n = 0; n |= static_cast<U>(static_cast<unsigned char>(c)) << (i * 8); }
    else output->put(static_cast<char>(n >> (i * 8)));
   }
   v = static_cast<T>(n);
  }
  template<class T> requires std::is_enum_v<T>
  void operator()(T& v) { auto n = static_cast<std::underlying_type_t<T>>(v); (*this)(n); v = static_cast<T>(n); }
  template<class T, size_t N> void operator()(std::array<T,N>& v) { for (auto& x : v) (*this)(x); }
  template<class T> void operator()(std::vector<T>& v) {
   uint32_t n = static_cast<uint32_t>(v.size()); (*this)(n);
   if (!ok || n > 1048576) { ok = false; return; }
   if (input) v.resize(n);
   for (auto& x : v) { if (!ok) break; (*this)(x); }
  }
  template<class T> void operator()(std::optional<T>& v) {
   bool present = v.has_value(); (*this)(present);
   if (input) { v.reset(); if (present) v.emplace(); }
   if (present && ok) (*this)(*v);
  }
  template<class A, class B> void operator()(std::pair<A,B>& v) { (*this)(v.first); (*this)(v.second); }
  template<class... T> void Fields(T&... v) { ((*this)(v), ...); }
  void operator()(Value& v) {
   Type type = v.TryInstruction() ? Type::Opaque : v.GetType();
   uint64_t bits = 0;
   if (!input) {
    if (auto* p = v.TryInstruction()) {
     const auto found = ids.find(p); if (found == ids.end()) { ok = false; return; } bits = found->second;
    } else switch (type) {
     case Type::Void: break;
     case Type::ScalarReg: bits = static_cast<uint32_t>(v.ScalarRegister()); break;
     case Type::VectorReg: bits = static_cast<uint32_t>(v.VectorRegister()); break;
     case Type::U1: bits = v.U1(); break;
     case Type::U8: bits = v.U8(); break;
     case Type::U16: bits = v.U16(); break;
     case Type::U32: bits = v.U32(); break;
     case Type::U64: bits = v.U64(); break;
     case Type::F16: bits = v.F16Bits(); break;
     case Type::F32: bits = std::bit_cast<uint32_t>(v.F32Value()); break;
     default: ok = false; return;
    }
   }
   Fields(type,bits);
   if (!input || !ok) return;
   switch (type) {
    case Type::Void: v = {}; break;
    case Type::Opaque: if (bits >= nodes.size()) { ok = false; return; } v = Value(nodes[bits]); break;
    case Type::ScalarReg: v = Value(static_cast<ScalarReg>(bits)); break;
    case Type::VectorReg: v = Value(static_cast<VectorReg>(bits)); break;
    case Type::U1: v = Value(bits != 0); break;
    case Type::U8: v = Value(static_cast<uint8_t>(bits)); break;
    case Type::U16: v = Value(static_cast<uint16_t>(bits)); break;
    case Type::U32: v = Value(static_cast<uint32_t>(bits)); break;
    case Type::U64: v = Value(bits); break;
    case Type::F16: v = Value::F16(static_cast<uint16_t>(bits)); break;
    case Type::F32: v = Value::F32(std::bit_cast<float>(static_cast<uint32_t>(bits))); break;
    default: ok = false;
   }
  }
  void operator()(MemoryInfo& v) { Fields(v.kind,v.resource,v.sampler,v.offset,v.secondary_offset,v.dmask,
   v.data_dwords,v.data_bits,v.component_index,v.component_count,v.data_format,v.number_format,
   v.image_sample_flags,v.image_dimension,v.image_address_components,v.address_is_full,v.data_signed,
   v.typed,v.formatted,v.image_has_mip,v.image_r128,v.idxen,v.offen,v.coherent,v.planning_only); }
  void operator()(BufferResource& v) { Fields(v.source,v.first_use_pc,v.max_byte_extent,v.packed_stride,
   v.descriptor_format,v.descriptor_swizzle,v.image_alias,v.read,v.written,v.atomic,v.formatted,v.scalar,
   v.indirect_root,v.indirect_mapping_offset,v.indirect_search_iterations,v.indirect_resources); }
  void operator()(ImageResource& v) { Fields(v.source,v.first_use_pc,v.resource_class,v.numeric_class,
   v.dimension,v.mip_mode,v.mip_count,v.conversion_format,v.shader_swizzle,v.read,v.written,v.atomic,
   v.atomic64,v.depth_compare,v.cube,v.r128,v.indirect_root,v.indirect_mapping_offset,
   v.indirect_search_iterations,v.indirect_resources); }
  void operator()(SamplerResource& v) { Fields(v.source,v.first_use_pc,v.snapshot_index,
   v.force_point_filtering,v.depth_compare,v.integer_border,v.gather_lod); }
  void operator()(SampledResourcePair& v) { Fields(v.image,v.sampler,v.first_use_pc); }
  void operator()(ShaderInfo& v) { Fields(v.buffers,v.images,v.samplers,v.sampled_pairs,
   v.vertex_fetch_components,v.vertex_offset_sgpr,v.instance_offset_sgpr,v.has_bitwise_xor,v.uses_dma,v.native_draw); }
  void operator()(DescriptorSource::IndirectDescriptor::SelectorRead::CountedRows& v) {
    Fields(v.source, v.stride);
  }
  void operator()(std::optional<DescriptorSource::IndirectDescriptor::SelectorRead::CountedRows>& v) {
    bool present = v.has_value();
    (*this)(present);
    if (!input) {
      if (present && ok) (*this)(*v);
      return;
    }
    v.reset();
    if (!present || !ok) return;
    uint32_t source = 0, stride = 0;
    (*this)(source);
    (*this)(stride);
    if (!ok) return;
    v.emplace(source, stride);
  }
  void operator()(DescriptorSource::IndirectDescriptor::SelectorRead& v) {
    Fields(v.source, v.stride, v.offset, v.maximum, v.bit_offset, v.bit_mask,
           v.global, v.bitset, v.rows, v.byte_ranges);
  }
  void operator()(std::optional<DescriptorSource::IndirectDescriptor::SelectorRead>& v) {
    using S = DescriptorSource::IndirectDescriptor::SelectorRead;
    bool present = v.has_value();
    (*this)(present);
    if (!input) {
      if (present && ok) (*this)(*v);
      return;
    }
    v.reset();
    if (!present || !ok) return;
    uint32_t source = 0, stride = 0, offset = 0, maximum = 0, bit_offset = 0, bit_mask = 0;
    bool global = false, bitset = false;
    std::optional<S::CountedRows> rows;
    std::vector<std::pair<uint32_t, uint32_t>> byte_ranges;
    (*this)(source);
    (*this)(stride);
    (*this)(offset);
    (*this)(maximum);
    (*this)(bit_offset);
    (*this)(bit_mask);
    (*this)(global);
    (*this)(bitset);
    (*this)(rows);
    (*this)(byte_ranges);
    if (!ok) return;
    v.emplace(source, stride, offset, maximum, bit_offset, bit_mask, global, bitset,
              std::move(rows), std::move(byte_ranges));
  }
  void operator()(DescriptorSource::IndirectDescriptor::PointerTable& v) {
    Fields(v.offset, v.stride, v.count, v.indexed);
  }
  void operator()(std::optional<DescriptorSource::IndirectDescriptor::PointerTable>& v) {
    bool present = v.has_value();
    (*this)(present);
    if (!input) {
      if (present && ok) (*this)(*v);
      return;
    }
    v.reset();
    if (!present || !ok) return;
    uint32_t offset = 0, stride = 0, count = 0;
    bool indexed = false;
    (*this)(offset);
    (*this)(stride);
    (*this)(count);
    (*this)(indexed);
    if (!ok) return;
    v.emplace(offset, stride, count, indexed);
  }
  void operator()(DescriptorSource::IndirectDescriptor& v) {
    Fields(v.selector, v.table_source, v.table_offset, v.table_immediate, v.table_stride,
           v.table_record_bytes, v.table_scalar, v.indexed, v.signed_count, v.workgroup_axis,
           v.table_mask, v.table_array, v.key_count, v.selector_first, v.selector_mask, v.sources,
           v.table_maximum, v.pointer_table);
  }
  void operator()(std::optional<DescriptorSource::IndirectDescriptor>& v) {
    using D = DescriptorSource::IndirectDescriptor;
    bool present = v.has_value();
    (*this)(present);
    if (!input) {
      if (present && ok) (*this)(*v);
      return;
    }
    v.reset();
    if (!present || !ok) return;
    std::optional<D::SelectorRead> selector;
    uint32_t table_source = 0, table_offset = 0, table_immediate = 0, table_stride = 0;
    uint32_t table_record_bytes = 0, workgroup_axis = 0, table_mask = 0;
    bool table_scalar = false, indexed = false, signed_count = false, table_array = false;
    Value key_count, selector_first, selector_mask;
    std::vector<uint32_t> sources;
    std::optional<uint32_t> table_maximum;
    std::optional<D::PointerTable> pointer_table;
    (*this)(selector);
    (*this)(table_source);
    (*this)(table_offset);
    (*this)(table_immediate);
    (*this)(table_stride);
    (*this)(table_record_bytes);
    (*this)(table_scalar);
    (*this)(indexed);
    (*this)(signed_count);
    (*this)(workgroup_axis);
    (*this)(table_mask);
    (*this)(table_array);
    (*this)(key_count);
    (*this)(selector_first);
    (*this)(selector_mask);
    (*this)(sources);
    (*this)(table_maximum);
    (*this)(pointer_table);
    if (!ok) return;
    v.emplace(std::move(selector), std::move(table_maximum), std::move(pointer_table), table_source,
              table_offset, table_immediate, table_stride, table_record_bytes, table_scalar, indexed,
              signed_count, workgroup_axis, table_mask, table_array, key_count, selector_first,
              selector_mask, std::move(sources));
  }
  void operator()(DescriptorSource& v) { Fields(v.dwords,v.dword_count,v.indirect_descriptor); if (v.dword_count > 8) ok = false; }
  void operator()(ResourcePlan::DescriptorGather::Word& v) { Fields(v.value,v.kind); }
  void operator()(ResourcePlan::DescriptorGather& v) { Fields(v.words,v.valid,v.needs_flat); }
  void operator()(ResourceBlock& v) { Fields(v.condition,v.successors,v.sources,v.srt_reads); }
  void operator()(SrtRead& v) { Fields(v.value,v.flat_offset); }
  void operator()(UniformFill& v) { Fields(v.kind,v.resource,v.group_stride,v.words,v.value); if (v.words > 4) ok = false; }
  void operator()(DescriptorValue& v) { Fields(v.dwords,v.dword_count); if (v.dword_count > 8) ok = false; }
  void operator()(ResourceSnapshot& v) { Fields(v.buffers,v.images,v.samplers,v.flattened_srt,
   v.user_data,v.specialization_reads,v.uniform_fill); }
  void operator()(ResourceSpecialization::Buffer& v) { Fields(v.packed_stride,v.descriptor_format,
   v.descriptor_swizzle,v.zero_stride_oob,v.indirect_root,v.indirect_mapping_offset,v.indirect_search_iterations); }
  void operator()(ResourceSpecialization::Image& v) { Fields(v.numeric_class,v.dimension,v.mip_count,
   v.conversion_format,v.shader_swizzle,v.indirect_root,v.indirect_mapping_offset,
   v.indirect_search_iterations,v.cube,v.fmask); }
  void operator()(ResourceSpecialization& v) { Fields(v.buffers,v.images); }
  void operator()(Read& v) { Fields(v.address,v.strict,v.valid,v.words); }
  void Plan(ResourcePlan& p) {
   uint32_t count = static_cast<uint32_t>(p.value_storage.size()); (*this)(count);
   if (!ok || count > 1048576) { ok = false; return; }
   std::vector<uint32_t> arities;
   if (input) p = ResourcePlan{};
   // Allocate every instruction first; forward references and cycles are legal.
   uint32_t index = 0;
   if (!input) for (auto& inst : p.value_storage) { ids.emplace(&inst,index++); nodes.push_back(&inst); }
   for (uint32_t i = 0; i < count && ok; ++i) {
    ValueOpcode op = input ? ValueOpcode{} : nodes[i]->GetOpcode();
    uint64_t flags = input ? 0 : nodes[i]->Flags<uint64_t>();
    uint32_t n = input ? 0 : static_cast<uint32_t>(nodes[i]->NumArgs());
    Fields(op,flags,n);
    if (input && static_cast<uint32_t>(op) >= legacy_first) op = static_cast<ValueOpcode>(static_cast<uint32_t>(op) + legacy_shift);
    if (op >= ValueOpcode::Count || n > 1048576 || (op != ValueOpcode::Phi && n != NumArgsOf(op))) { ok = false; return; }
    arities.push_back(n);
    if (input) { p.value_storage.emplace_back(op,flags); nodes.push_back(&p.value_storage.back()); }
   }
   for (uint32_t i = 0; i < count && ok; ++i) for (uint32_t j = 0; j < arities[i] && ok; ++j) {
    Value arg = input ? Value{} : nodes[i]->Arg(j); (*this)(arg);
    if (input && ok) { if (nodes[i]->GetOpcode() == ValueOpcode::Phi) nodes[i]->AddPhiOperand(nullptr,arg); else nodes[i]->SetArg(j,arg); }
   }
   Fields(p.stage,p.shader_hash,p.user_data_base,p.user_data_count,p.memory_info,p.descriptor_sources,
    p.descriptor_gathers,p.control_flow,p.srt_reads,p.clean_flat_slots,p.requires_specialization_memory,
    p.capture_specialization_reads,p.srt_plan_complete,p.resource_tracking_complete,p.info,
    p.uniform_fill.fill,p.uniform_fill.values);
   if (input && ok) {
    for (auto& gather : p.descriptor_gathers) {
     for (auto& word : gather.words) {
      word.read_slot = UINT32_MAX;
      if (word.kind != ResourcePlan::DescriptorGather::Word::Kind::Flat) continue;
      for (uint32_t slot = 0; slot < p.srt_reads.size(); ++slot) {
       if (p.srt_reads[slot].flat_offset != word.value) continue;
       if (word.read_slot != UINT32_MAX) { word.read_slot = UINT32_MAX; gather.valid = false; break; }
       word.read_slot = slot;
      }
      if (word.read_slot == UINT32_MAX) gather.valid = false;
     }
    }
   }
  }
 };

 bool Save(const std::string& path, const ResourcePlan& source) {
  std::ofstream out(path + ".tmp", std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write("KYTYRPF1",8);
  Archive a {.output = &out};
  uint32_t version = 1, opcode_count = static_cast<uint32_t>(ValueOpcode::Count);
  uint64_t schema = OpcodeSchema();
  a.Fields(version,opcode_count,schema);
  a.Plan(const_cast<ResourcePlan&>(source));
  a.Fields(user_data,workgroups,shader_base,ordinary_reader,strict_reader,success,reads,
   expected,specialization,active,captured_ops,captured_recursive_ops,captured_tape_ops);
  out.flush(); const bool good = a.ok && out.good(); out.close();
  if (!good) { std::filesystem::remove(path + ".tmp"); return false; }
  std::error_code ec;
  std::filesystem::rename(path + ".tmp",path,ec);
  return !ec;
 }
 bool Load(const std::string& path) {
  std::error_code ec; const auto size = std::filesystem::file_size(path,ec);
  if (ec || size > 64*1024*1024) return false;
  std::ifstream in(path,std::ios::binary);
  char magic[8]; in.read(magic,8);
  if (!in || std::memcmp(magic,"KYTYRPF1",8)) return false;
  Archive a {.input = &in}; uint32_t version = 0, opcode_count = 0;
  uint64_t schema = 0;
  a.Fields(version,opcode_count,schema);
  if (!a.ok || version != 1) return false;
  if (opcode_count != static_cast<uint32_t>(ValueOpcode::Count) || schema != OpcodeSchema()) {
   uint32_t skipped = 0, first = UINT32_MAX;
   for (uint32_t i = 0; i < static_cast<uint32_t>(ValueOpcode::Count); ++i)
    if (LegacyOpcodeSkipped(static_cast<ValueOpcode>(i))) { ++skipped; first = std::min(first,i); }
    // The four skipped opcodes are contiguous, so one shift reconstructs the current numbering.
   if (opcode_count + skipped != static_cast<uint32_t>(ValueOpcode::Count) || schema != OpcodeSchema(true)) return false;
   a.legacy_first = first; a.legacy_shift = skipped;
  }
  a.Plan(plan);
  a.Fields(user_data,workgroups,shader_base,ordinary_reader,strict_reader,success,reads,
   expected,specialization,active,captured_ops,captured_recursive_ops,captured_tape_ops);
  return a.ok && in.peek() == EOF;
 }

 struct ReplayResult {
  bool matched = false, success = false;
  uint64_t missing = 0, conflicts = 0, requests = 0, words = 0, ops = 0, cfg_roots = 0, walker = 0;
  uint64_t ordinary_calls = 0, strict_calls = 0;
  uint64_t recursive_ops = 0, tape_ops = 0;
  uint64_t scalar_batch_words = 0;
  uint64_t cfg_gather_eligible = 0, cfg_gather_hits = 0;
  uint64_t cfg_schedule_steps = 0, cfg_scheduled = 0;
 };
 struct ReplayBacking {
  struct Word { uint32_t value; bool valid; };
  std::map<std::pair<bool,uint64_t>,Word> words;
  std::vector<uint32_t> scratch;
  ReplayResult result;
  ResourceSnapshot actual;
  ResourceSpecialization specialization;
  ResourceSpecializationCache specialization_cache;
  ResourceDelta delta; // KYTY_SRT_COMPILED part A (only attached while the flag is on)
  uint64_t conflict_count = 0;
  ReplayBacking() = default;
  explicit ReplayBacking(const ResourcePlanFixture& fixture) {
   for (const auto& r : fixture.reads) for (size_t i = 0; i < r.words.size(); ++i) {
    if (i > (std::numeric_limits<uint64_t>::max() - r.address) / sizeof(uint32_t)) {
     ++conflict_count; break;
    }
    const Word w {r.words[i],r.valid};
    auto [it,inserted] = words.emplace(std::pair{r.strict,r.address+i*sizeof(uint32_t)},w);
    if (!inserted && (it->second.valid != w.valid || (w.valid && it->second.value != w.value))) ++conflict_count;
   }
  }
  // Bench-only hashed index (KYTY_SRT_COMPILED comparison): the std::map lookup dominates small plans.
  struct FastSlot { uint64_t key = ~uint64_t{0}; Word word {0,false}; };
  std::vector<FastSlot> fast_table;
  uint64_t fast_mask = 0;
  bool fast = false;
  void BuildFast() {
   size_t size = 16;
   while (size < words.size() * 2 + 2) size <<= 1;
   fast_table.assign(size,{}); fast_mask = size - 1;
   for (const auto& [k,w] : words) {
    const uint64_t key = k.second | (k.first ? (uint64_t{1} << 63) : 0);
    size_t at = static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> 20) & fast_mask;
    while (fast_table[at].key != ~uint64_t{0}) at = (at + 1) & fast_mask;
    fast_table[at] = {key,w};
   }
   fast = true;
  }
  const Word* Find(bool strict, uint64_t address) const {
   if (fast && !(address >> 63)) {
    const uint64_t key = address | (strict ? (uint64_t{1} << 63) : 0);
    size_t at = static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> 20) & fast_mask;
    while (fast_table[at].key != ~uint64_t{0}) {
     if (fast_table[at].key == key) return &fast_table[at].word;
     at = (at + 1) & fast_mask;
    }
    return nullptr;
   }
   auto found = words.find({strict,address});
   return found == words.end() ? nullptr : &found->second;
  }
  bool Read(bool strict, uint64_t address, std::span<uint32_t> out) {
   ++result.requests; result.words += out.size();
   if (strict) ++result.strict_calls; else ++result.ordinary_calls;
   if (out.size() > (std::numeric_limits<uint64_t>::max() - address) / sizeof(uint32_t)) {
    ++result.missing; return false;
   }
   bool failed = false, missing = false;
   scratch.resize(out.size());
   for (size_t i = 0; i < out.size(); ++i) {
    const Word* found = Find(strict,address+i*sizeof(uint32_t));
    if (!found) { missing = true; continue; }
    failed |= !found->valid; scratch[i] = found->value;
   }
   if (missing) ++result.missing;
   if (missing || failed) return false;
   std::copy(scratch.begin(),scratch.end(),out.begin()); return true;
  }
  bool TryRead(bool strict, uint64_t address, std::span<uint32_t> out) {
   if (out.size() > (std::numeric_limits<uint64_t>::max() - address) / sizeof(uint32_t)) return false;
   scratch.resize(out.size());
   for (size_t i = 0; i < out.size(); ++i) {
    const Word* found = Find(strict,address+i*sizeof(uint32_t));
    if (!found || !found->valid) return false;
    scratch[i] = found->value;
   }
   ++result.requests; result.words += out.size();
   if (strict) ++result.strict_calls; else ++result.ordinary_calls;
   std::copy(scratch.begin(),scratch.end(),out.begin());
   return true;
  }
  void Reset() { result = {}; result.conflicts = conflict_count; }
 };
 ReplayResult ReplayPrepared(ReplayBacking& backing) const {
  backing.Reset();
  if (backing.result.conflicts) return backing.result;
  SrtRuntime runtime {.user_data=user_data,.shader_base=shader_base,.userdata=&backing,.workgroup_counts=workgroups};
  if (Common::RuntimeFlags::Current().resource_spec_cache)
   runtime.specialization_cache = &backing.specialization_cache;
  if (Common::RuntimeFlags::Current().srt_compiled) runtime.delta = &backing.delta;
  if (ordinary_reader) runtime.read_memory = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<ReplayBacking*>(p)->Read(false,a,v); };
  else runtime.read_memory = +[](void* p,uint64_t,std::span<uint32_t>) {
   auto& r = static_cast<ReplayBacking*>(p)->result; ++r.ordinary_calls; ++r.missing; return false;
  };
  if (ordinary_reader) runtime.try_read_memory_span = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<ReplayBacking*>(p)->TryRead(false,a,v); };
  if (strict_reader) runtime.read_specialization_memory = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<ReplayBacking*>(p)->Read(true,a,v); };
  if (strict_reader) runtime.try_read_specialization_span = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<ReplayBacking*>(p)->TryRead(true,a,v); };
  SrtEvaluationCounts counts;
  runtime.evaluation_counts = &counts;
  const auto before = GetSrtTapeStats();
  backing.result.success = MaterializeResources(plan,runtime,backing.actual,backing.specialization);
  const auto after = GetSrtTapeStats();
  backing.result.recursive_ops = counts.recursive_ops;
  backing.result.tape_ops = counts.tape_ops;
  backing.result.scalar_batch_words = counts.scalar_batch_words;
  backing.result.cfg_gather_eligible = counts.cfg_gather_eligible;
  backing.result.cfg_gather_hits = counts.cfg_gather_hits;
  backing.result.cfg_schedule_steps = counts.cfg_schedule_steps;
  backing.result.cfg_scheduled = after.cfg_scheduled-before.cfg_scheduled;
  backing.result.ops = counts.recursive_ops + counts.tape_ops;
  backing.result.cfg_roots = after.cfg_runs-before.cfg_runs;
  backing.result.walker = after.walker-before.walker;
  backing.result.matched = !backing.result.missing && backing.result.success == success &&
   (!success || (backing.actual.buffers == expected.buffers && backing.actual.images == expected.images &&
    backing.actual.samplers == expected.samplers && backing.actual.flattened_srt == expected.flattened_srt &&
    backing.actual.user_data == expected.user_data && SameRangeCoverage(backing.actual.specialization_reads,expected.specialization_reads) &&
    backing.actual.uniform_fill == expected.uniform_fill && backing.specialization == specialization && plan.active_sources == active));
  return backing.result;
 }
 ReplayResult Replay() { ReplayBacking backing(*this); return ReplayPrepared(backing); }
};

// Shared by live recording and the focused test. The callback allows the live
// entry point to call its implementation without recursively recording itself.
inline bool CaptureResourcePlan(const ResourcePlan& plan, const SrtRuntime& runtime,
 ResourceSnapshot& snapshot, ResourceSpecialization& specialization, const std::string& path,
 bool (*materialize)(const ResourcePlan&,const SrtRuntime&,ResourceSnapshot&,ResourceSpecialization&) = MaterializeResources) {
 struct Capture {
  SrtRuntime source;
  ResourcePlanFixture fixture;
  bool Read(bool strict,uint64_t address,std::span<uint32_t> words) {
   auto reader = strict ? source.read_specialization_memory : source.read_memory;
   bool valid;
   if (!strict && !reader) { std::memcpy(words.data(),reinterpret_cast<const void*>(address),words.size_bytes()); valid = true; }
   else valid = reader && reader(source.userdata,address,words);
   ResourcePlanFixture::Read record {.address=address,.strict=strict,.valid=valid};
   record.words.assign(words.size(),0);
   if (valid) std::copy(words.begin(),words.end(),record.words.begin());
   fixture.reads.push_back(std::move(record));
   return valid;
  }
  bool TryRead(bool strict,uint64_t address,std::span<uint32_t> words) {
   auto reader = strict ? source.try_read_specialization_span : source.try_read_memory_span;
   if (reader == nullptr || !reader(source.userdata,address,words)) return false;
   ResourcePlanFixture::Read record {.address=address,.strict=strict,.valid=true};
   record.words.assign(words.begin(),words.end());
   fixture.reads.push_back(std::move(record));
   return true;
  }
 } capture;
 capture.source = runtime;
 auto& fixture = capture.fixture;
 fixture.user_data.assign(runtime.user_data.begin(),runtime.user_data.end());
 fixture.workgroups.assign(runtime.workgroup_counts.begin(),runtime.workgroup_counts.end());
 fixture.shader_base = runtime.shader_base;
 fixture.ordinary_reader = true;
 fixture.strict_reader = runtime.read_specialization_memory != nullptr;
 SrtRuntime observed = runtime;
 observed.userdata = &capture;
 observed.read_memory = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<Capture*>(p)->Read(false,a,v); };
 if (fixture.strict_reader) observed.read_specialization_memory = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<Capture*>(p)->Read(true,a,v); };
 if (runtime.try_read_memory_span) observed.try_read_memory_span = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<Capture*>(p)->TryRead(false,a,v); };
 if (runtime.try_read_specialization_span) observed.try_read_specialization_span = +[](void* p,uint64_t a,std::span<uint32_t> v) { return static_cast<Capture*>(p)->TryRead(true,a,v); };
 SrtEvaluationCounts counts;
 observed.evaluation_counts = &counts;
 fixture.success = materialize(plan,observed,snapshot,specialization);
 fixture.captured_recursive_ops = counts.recursive_ops;
 fixture.captured_tape_ops = counts.tape_ops;
 fixture.captured_ops = counts.recursive_ops + counts.tape_ops;
 fixture.expected = snapshot; fixture.specialization = specialization; fixture.active = plan.active_sources;
 if (!fixture.Save(path,plan)) std::fprintf(stderr,"NHL27PLAN: save failed: %s\n",path.c_str());
 else std::printf("NHL27PLAN: hash=%llx success=%d reads=%zu ops=%llu -> %s\n",
  static_cast<unsigned long long>(plan.shader_hash),fixture.success,fixture.reads.size(),
  static_cast<unsigned long long>(fixture.captured_ops),path.c_str());
 return fixture.success;
}
} // namespace Libs::Graphics::ShaderRecompiler::IR
