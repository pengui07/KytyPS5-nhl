#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <atomic>
#include <array>
#include <bitset>

#include <map>
#include <memory>
#include <mutex>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

class ColorClearCheck;

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	// FindImage that first tries `hint`, an image an earlier lookup of the same description
	// resolved to as an exact backing match: when it is still registered and still the exact
	// match, the region scan over every page of the image is skipped. `exact` reports whether
	// the result is such a match (a valid hint for the next lookup).
	[[nodiscard]] ImageId FindImageHinted(ImageDesc& desc, bool exact_format, ImageId hint,
	                                      bool& exact);
	// KYTY_DRAWPREP_DELTA: bumped whenever a previously resolved T# -> image binding could be stale.
	[[nodiscard]] uint64_t      Generation() const noexcept { return m_generation.load(std::memory_order_relaxed); }
	// Appends {new generation, [begin,end)} to the ring log and bumps the generation. The default range is a
	// wildcard that overlaps every memo.
	void BumpGeneration(uint64_t begin = 0, uint64_t end = UINT64_MAX) noexcept;
	enum class GenCheck : uint8_t { Clean, Overlap, Overrun };
	// Checks every log entry in (memo_gen, current] against the guest range [begin,end). `current` receives the
	// generation the check was made against (read under the log lock). Clean = no entry overlaps.
	[[nodiscard]] GenCheck CheckGenerationLog(uint64_t memo_gen, uint64_t begin, uint64_t end, uint64_t& current) const;
	// The per-call bookkeeping FindImage does on a hit (last-access tick, LRU touch).
	void                        NoteImageUse(ImageId id);
	// Lookup for bindings the shader may never read (bindless table candidates). It resolves
	// like FindImage but returns an empty id instead of converting a depth image to color or
	// back: those conversions replace the image other passes use, which then convert it again.
	[[nodiscard]] ImageId FindImageSpeculative(ImageDesc& desc, bool exact_format = false);
	void                        UpdateImage(ImageId id);
	// Whether a registered image starts at `address` (the only images FindImageFromRange finds).
	[[nodiscard]] bool HasImageAt(uint64_t address) {
		// Nearly every texel bind asks and finds none: a filter bit of the start address is
		// clear then (the lock and the map lookup were 1.5% of the execution thread).
		if (!AddressFilterHit(address)) {
			return false;
		}
		std::scoped_lock lock {m_lock};
		return m_images_by_address.contains(address);
	}
	// Advances after each image registration (the only change that can make HasImageAt() true
	// for an address): a false answer holds while it is unchanged.
	[[nodiscard]] uint64_t AddressIndexVersion() const {
		return m_address_index_version.load(std::memory_order_acquire);
	}
	// The same for the addresses sharing one of 4,096 groups of HasImageAt() filter bits with
	// `address` only (bit 63 set: never equal to an AddressIndexVersion()). A small table: a
	// version per filter bit (256 KiB) missed the cache on most texel binds.
	[[nodiscard]] uint64_t AddressBucketVersion(uint64_t address) const {
		return (uint64_t {1} << 63u) |
		       m_address_filter_versions[AddressFilterIndex(address) >> 4u].load(std::memory_order_acquire);
	}
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	// stencil (optional) receives the stencil association the target refreshed, if any.
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc,
	                                            ImageId* stencil = nullptr);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);
	// Copies storage writes made through block-image shadows back into the block images.
	// Call after recording each draw or dispatch.
	void FlushStorageShadows();

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);
	[[nodiscard]] bool HasImagesInRegion(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// Advances whenever an image is registered or unregistered.
	[[nodiscard]] uint64_t ImageSetEpoch() const { return m_image_set_epoch; }
	// The ImageSetEpoch of the last registration or unregistration of an image overlapping the
	// range's pages: while it is unchanged, a lookup in the range finds the same images.
	[[nodiscard]] uint64_t ImageEpochInRegion(uint64_t address, uint64_t size);
	void RunGarbageCollector();

private:
	struct MetaDataInfo {
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
		// Speculative lookups only: resolving the overlap would replace an owned image.
		bool rejected = false;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged,
	                                           bool speculative = false);
	[[nodiscard]] ImageId       FindImageImpl(ImageDesc& desc, bool exact_format, bool speculative);
	[[nodiscard]] static bool   DepthOverlapNeedsRecreate(const ImageInfo& requested,
	                                                      BindingType binding, const Image& cached);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	// Experiment GpuColorClear: checks GPU-written metadata slices on the GPU and clears the
	// matching image slices through conditional rendering. False when it cannot (CPU path).
	[[nodiscard]] bool          MaterializeColorClearOnGpu(ImageId id, const ImageDesc& desc,
	                                                       uint64_t address, uint64_t slice_size,
	                                                       uint32_t count, uint32_t image_first);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	// Caller holds m_lock and ends rendering. Begins rendering `layers` layers of one color mip
	// through a `format` view, clearing them or, when `clear` is null, loading them; returns the
	// render area.
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               CommandRecorder command);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	// Validates the view and binding of a description; ImageOps::Validate() checks its image
	// information.
	void ValidateImageViewDesc(const ImageDesc& desc) const;
	void ValidateImageDesc(const ImageDesc& desc) const;
	[[nodiscard]] ImageId GetNullImage(const ImageDesc& desc);

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	// tolerant_write: the deferred guest write skips (instead of exiting) if the range was unmapped.
	[[nodiscard]] bool DownloadImageMemory(ImageId id, bool tolerant_write = false);
	// KYTY_ALIAS_WRITEBACK experiment: synchronous download to guest memory before a free.
	[[nodiscard]] bool WriteBackAliasedImage(ImageId id);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	std::vector<ImageId>                              m_shadowed_images;
	std::atomic_bool                                 m_storage_shadows_pending {false};
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	std::shared_ptr<ColorClearCheck>                  m_color_clear_check;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	// Registered images by exact start address (FindImage fast path).
	std::unordered_map<uint64_t, std::vector<ImageId>> m_images_by_address;
	std::atomic<uint64_t>                              m_address_index_version {0};
	// HasImageAt() filter: a bit per hash of the start addresses of registered images, set while
	// the count of registered images hashing to it is not 0 (counts change under m_lock).
	static constexpr size_t AddressFilterBits = size_t {1} << 16u;
	[[nodiscard]] static size_t AddressFilterIndex(uint64_t address) {
		return static_cast<size_t>(((address >> 8u) * 0x9e3779b97f4a7c15ull) >> 48u);
	}
	[[nodiscard]] bool AddressFilterHit(uint64_t address) const {
		const auto index = AddressFilterIndex(address);
		return ((m_address_filter[index / 64].load(std::memory_order_acquire) >> (index % 64)) & 1u) != 0;
	}
	std::array<std::atomic<uint64_t>, AddressFilterBits / 64> m_address_filter {};
	std::array<uint16_t, AddressFilterBits>                   m_address_filter_counts {};
	std::array<std::atomic<uint32_t>, AddressFilterBits / 16> m_address_filter_versions {};
	MultiLevelPageTable<uint64_t, ImagePageTable::kPageBits, ImagePageTable::kAddressSpaceBits,
	                    ImagePageTable::kFirstLevelBits>
	    m_image_page_epochs;
	std::unordered_map<vk::Format, ImageId>           m_null_images;

	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	mutable uint32_t m_image_query_epoch      = 0;
	uint64_t         m_image_set_epoch        = 0;
	bool             m_readback_linear_images = false;
	// Set while an overlap/alias resolution frees an image; only those frees write back.
	bool             m_alias_free_in_progress = false;
	std::atomic<uint64_t> m_generation {0};
	struct GenLogEntry {
		uint64_t generation = 0;
		uint64_t begin      = 0;
		uint64_t end        = 0;
	};
	static constexpr uint64_t GEN_LOG_SIZE = 64;
	mutable std::mutex        m_gen_log_mutex;
	GenLogEntry               m_gen_log[GEN_LOG_SIZE] {};
	// Lookups that found an existing image with the same backing, by description: while no image
	// on the pages of the range was registered or unregistered since, the same lookup finds the
	// same image.
	struct ExactLookup {
		ImageInfo info;
		ImageId   id;
		uint64_t  epoch        = 0;
		bool      exact_format = false;
		bool      valid        = false;
	};
	static constexpr size_t                                   ExactLookupWays = 1024;
	std::unique_ptr<std::array<ExactLookup, ExactLookupWays>> m_exact_lookups =
	    std::make_unique<std::array<ExactLookup, ExactLookupWays>>();
	[[nodiscard]] static size_t ExactLookupSlot(const ImageInfo& info) noexcept;
	[[nodiscard]] uint64_t      ImageEpochInRegionUnlocked(uint64_t address, uint64_t size) const;
	bool             m_readback_tiled_images  = false;


	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
