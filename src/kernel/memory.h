#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

namespace Libs::Graphics {
class RenderContext;
enum class PageFaultAccess;
} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

void Initialize();
void Shutdown();

// Guest memory usage for the performance panel. Takes only bookkeeping locks, so it never
// waits on the GPU; call it only while the memory subsystem is initialized.
struct DebugStats {
	uint64_t direct_total        = 0;
	uint64_t direct_allocated    = 0; // sceKernelAllocateDirectMemory.
	uint64_t pooled_allocated    = 0; // Memory pool expansions.
	uint64_t automatic_allocated = 0; // Extended memory.
	uint64_t direct_mapped       = 0;
	uint64_t flexible_total      = 0;
	uint64_t flexible_used       = 0;
	uint64_t pool_committed      = 0;
	uint64_t cpu_page_entries    = 0; // 2 MiB page-table entries in use.
	uint64_t gpu_page_entries    = 0;
	uint64_t page_entries_total  = 0; // Per side.
};
[[nodiscard]] DebugStats GetDebugStats();

struct Lifecycle {
	static constexpr const char* name       = "Memory";
	static constexpr auto        initialize = Libs::LibKernel::Memory::Initialize;
	static constexpr auto        shutdown   = Libs::LibKernel::Memory::Shutdown;
};

using callback_func_t = void (*)(uintptr_t addr, size_t size);

constexpr uint32_t KERNEL_MAXIMUM_NAME_LENGTH = 32;
constexpr uint64_t kExtendedMemoryBase       = 0x080000000000ull;
constexpr uint64_t kExtendedMemorySize       = 512ull * 1024 * 1024 * 1024;

struct VirtualQueryInfo {
	uintptr_t start;
	uintptr_t end;
	uint64_t  offset;
	int32_t   protection;
	int32_t   memory_type;
	uint32_t  is_flexible  : 1;
	uint32_t  is_direct    : 1;
	uint32_t  is_stack     : 1;
	uint32_t  is_pooled    : 1;
	uint32_t  is_committed : 1;
	uint32_t  is_gpu_prt   : 1;
	uint32_t  amm_usage    : 1;
	uint32_t  reserved     : 1;
	char      name[KERNEL_MAXIMUM_NAME_LENGTH];
	uint8_t   gpu_mask_id;
	uint8_t   reserved2;
};

static_assert(sizeof(VirtualQueryInfo) == 72, "VirtualQueryInfo struct size is incorrect");

struct KernelBatchMapEntry {
	void*         start;
	uint64_t      offset;
	uint64_t      length;
	unsigned char protection;
	unsigned char type;
	int16_t       reserved;
	int32_t       operation;
};

static_assert(sizeof(KernelBatchMapEntry) == 32, "KernelBatchMapEntry struct size is incorrect");

struct KernelMemoryPoolBatchEntry {
	uint32_t op;
	uint32_t flags;
	union {
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} commit;
		struct {
			void*    addr;
			uint64_t len;
		} decommit;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
		} protect;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} type_protect;
		struct {
			void*    dst;
			void*    src;
			uint64_t len;
		} move;
		uintptr_t padding[3];
	};
};

static_assert(sizeof(KernelMemoryPoolBatchEntry) == 32,
              "KernelMemoryPoolBatchEntry struct size is incorrect");

struct KernelMemoryPoolBlockStats {
	int32_t available_flushed_blocks;
	int32_t available_cached_blocks;
	int32_t allocated_flushed_blocks;
	int32_t allocated_cached_blocks;
};

static_assert(sizeof(KernelMemoryPoolBlockStats) == 16,
              "KernelMemoryPoolBlockStats struct size is incorrect");

void                   RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func);
void                   SetFlexibleMemorySize(uint64_t size);
int AllocateDirectMemory(int64_t search_start, int64_t search_end, size_t size, size_t alignment,
                         int memory_type, int64_t* phys_addr_out, bool automatic = false);
int MapAutomaticMemory(uint64_t vaddr, size_t size, int type, int prot);
int UnmapAmmMemory(uint64_t vaddr, size_t size);
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
// Crash diagnostics: a ring of recent host (emulated GPU / command processor) writes into guest
// memory. ReportHostWrites prints the logged writes that overlap [vaddr - window, vaddr + window).
enum class HostWriteSource : uint8_t { Backing, CommandProcessor, Label, DmaCopy };
void                   NoteHostWrite(uint64_t vaddr, uint64_t size, HostWriteSource source,
                                     const void* data);
void                   ReportHostWrites(const char* label, uint64_t vaddr, uint64_t window);
// Writes the bytes of `data` whose backing still equals `expected` (see LazySharedWrites).
bool                   TryMergeBacking(uint64_t vaddr, const void* data, const void* expected, uint64_t size);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);
// GPU thread only: reads the backing when none of the bytes is GPU-dirty, so the CPU view of
// them is already current. Never synchronizes with the GPU.
bool                   TryReadCleanBacking(uint64_t vaddr, void* data, uint64_t size);
// KYTY_PERF_GPU_FAULT_STACKS (diagnostics): how this thread's last TryReadCleanBacking call ended, reset
// by the call (0 = no call since the last take).
enum CleanReadResult : int {
	CleanReadServed      = 1,
	CleanReadDirty       = 2,
	CleanReadCapped      = 3,
	CleanReadFailed      = 4,
	CleanReadUnavailable = 5,
};
int                    TakeCleanReadResult();
// Any thread: how many bytes of the range are GPU-dirty (diagnostics).
uint64_t               GpuDirtyByteCount(uint64_t vaddr, uint64_t size);
// Diagnostics: what this thread is reading guest memory for (the shader being materialized).
void                   SetReadContext(uint64_t context);
uint64_t               ReadContext();
// Diagnostics: this thread's host return addresses, innermost first (Windows only; 0 elsewhere).
int                    CaptureHostStack(uint64_t* frames, int max_frames);
// Experiment FalseShareAnyThread: the same for any other thread (command processor, guest): reads the
// backing when the page holds GPU-dirty bytes but the read does not, a limited number of times per page.
bool                   TryReadCleanBackingAnyThread(uint64_t vaddr, void* data, uint64_t size);
// Any thread but the GPU thread: reads the backing when none of the bytes is GPU-dirty (no page
// condition, no limit). False leaves the caller to its ordinary load.
bool                   TryReadCleanBackingShared(uint64_t vaddr, void* data, uint64_t size);
// Optional GPU-thread clean peek: refusal never downloads, faults or changes output.
bool                   TryReadCleanBufferBacking(uint64_t vaddr, void* data, uint64_t size);
// The last 16 pages on which this thread took a GPU-tracking read fault (they are protected because
// the GPU wrote some of their bytes). Readers of guest memory can read such pages through the
// backing when their bytes are clean instead of faulting again.
void                   NoteReadFaultPage(uint64_t vaddr) noexcept;
[[nodiscard]] bool     IsReadFaultPage(uint64_t vaddr, uint64_t size) noexcept;
// Whether a plain read of the range would fault on a page the GPU tracking read-protects (one that
// holds GPU-written bytes not downloaded yet). Any thread.
[[nodiscard]] bool     IsGpuReadProtected(uint64_t vaddr, uint64_t size) noexcept;
// GPU-thread clean validation (DRAWPREP prefetch): Equal = the backing bytes are not GPU-dirty and equal `expected`;
// Different = clean but unequal; Dirty = GPU-dirty bytes (never downloaded); Unavailable = preconditions or mapping failed.
// Never copies, downloads, faults or changes any state.
enum class CleanBackingCompare { Equal, Different, Dirty, Unavailable };
CleanBackingCompare    TryValidateCleanBacking(uint64_t vaddr, const void* expected, uint64_t size);
// Where the backing store keeps [vaddr, vaddr + size) when one mapping holds it, else null. The
// pointer stays readable for the emulator's lifetime and reads what TryReadBacking() reads.
[[nodiscard]] const void* FindBackingPointer(uint64_t vaddr, uint64_t size);
bool                   TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size);
// GPU thread: whether no byte of the range is GPU-dirty in the buffer cache (gpu_images: nor in a
// GPU-modified image). False off the GPU thread. A range that passes passes for each sub-range, so
// callers check a page once for the reads within it.
bool                   IsGpuCleanRange(uint64_t vaddr, uint64_t size, bool gpu_images);
// Any thread: whether bytes of the range may be GPU-dirty (conservatively true while any download
// is pending; see BufferCache::HasGpuDirtyBytesShared).
bool                   HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
// TryValidateCleanBacking for a range IsGpuCleanRange already passed (Dirty is never returned).
CleanBackingCompare    CompareCleanBacking(uint64_t vaddr, const void* expected, uint64_t size);
bool                   TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
// Whether TryReadBacking or TryReadSparseBacking can read the whole range.
bool                   IsBackingReadable(uint64_t vaddr, uint64_t size);
// Changes whenever backing is mapped or unmapped, or a PRT aperture changes: an unchanged value
// means IsBackingReadable still answers as before.
[[nodiscard]] uint64_t BackingEpoch() noexcept;
bool                   CopySparseMemory(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
// As ClampRangeSize, but 0 (instead of a fatal error) when no byte of the range is mapped.
[[nodiscard]] uint64_t ClampRangeSizeOrZero(uint64_t vaddr, uint64_t size);
// Not noexcept: under a Common::SoftExitScope its failure throws to the scope's owner.
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size);
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
void                   InstallGpuResources(Graphics::RenderContext* renderer) noexcept;
[[nodiscard]] bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;

int KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                                               const char* name);
int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags);
int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name);
int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len);
int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len);
size_t KYTY_SYSV_ABI KernelGetDirectMemorySize();
int KYTY_SYSV_ABI    KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                     size_t alignment, int64_t* phys_addr_out,
                                                     size_t* size_out);
int KYTY_SYSV_ABI    KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                             int* gpu_available);
int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name);
int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len);
int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len);
int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled();
int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot);
int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size);
int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size);
int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end);
int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment);
int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot);
int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot);
int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out);
int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size);

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name);
void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name,
                               bool fixed = false);
uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size,
                                  Common::VirtualMemory::Mode mode, const char* name);
bool     ProtectGuestMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
                            Common::VirtualMemory::Mode* old_mode = nullptr);
// Transient PageManager watch state; does not change the guest mapping's semantic protection.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void     TestSetGpuRangeCallback(void (*callback)(bool mapped, uint64_t addr, size_t size));
void     TestBeforeNextBackingMap(callback_func_t callback);
void     TestSetBackingReadCallback(callback_func_t callback);
void     TestFailNextPhysicalMemoryUnmap();
void     TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps);
void     TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps);
void     TestFailNextFixedReserveRangeRegistration();
void     TestFailNextVirtualRangeReplacement();
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
bool     TestGuestFreeRangeBounds();
#endif

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_ */
