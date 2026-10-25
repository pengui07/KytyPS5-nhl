#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/experiments.h"
#include "graphics/host_gpu/renderer/nativeIndirect.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/depthAliasLog.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Largest GPU-written tiled image (in texels) that is written back to guest memory on submit.
constexpr uint64_t SmallTiledReadbackTexels = 64 * 64;

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.info.IsDepth()
	                           ? TextureGetSurfaceFormatInfo(desc.info.guest_format).vk_format
	                           : desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodeRegisterColorClear(format, static_cast<uint32_t>(metadata.clear_word),
		                                static_cast<uint32_t>(metadata.clear_word >> 32u),
		                                clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()),
      m_readback_tiled_images(Config::ReadbackTiledImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

namespace {
inline void GenCount(std::atomic<uint64_t>& counter) noexcept {
	if (FrameBreakdown::Verbose()) counter.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

void TextureCache::BumpGeneration(uint64_t begin, uint64_t end) noexcept {
	TextureStateChanged();
	std::lock_guard lock {m_gen_log_mutex};
	const uint64_t generation = m_generation.load(std::memory_order_relaxed) + 1;
	m_gen_log[generation % GEN_LOG_SIZE] = {generation, begin, end};
	m_generation.store(generation, std::memory_order_release);
}

TextureCache::GenCheck TextureCache::CheckGenerationLog(uint64_t memo_gen, uint64_t begin, uint64_t end,
                                                        uint64_t& current) const {
	std::lock_guard lock {m_gen_log_mutex};
	current = m_generation.load(std::memory_order_relaxed);
	if (current < memo_gen || current - memo_gen > GEN_LOG_SIZE) {
		return GenCheck::Overrun;
	}
	for (uint64_t generation = memo_gen + 1; generation <= current; generation++) {
		const auto& entry = m_gen_log[generation % GEN_LOG_SIZE];
		if (entry.generation != generation) {
			return GenCheck::Overrun;
		}
		if (entry.begin < end && begin < entry.end) {
			return GenCheck::Overlap;
		}
	}
	return GenCheck::Clean;
}

// Diagnostic: KYTY_DBG_IMAGE_CHURN=1 prints image inserts and frees (first 4000);
// KYTY_DBG_IMAGE_CHURN=<seconds> starts that long after the first image (e.g. in a match).
static bool ImageChurnLogLine() {
	static const auto start = std::chrono::steady_clock::now();
	static const int64_t from = [] {
		const char* value = std::getenv("KYTY_DBG_IMAGE_CHURN");
		if (value == nullptr || value[0] < '1' || value[0] > '9') return int64_t {-1};
		const auto seconds = std::strtoll(value, nullptr, 10);
		return seconds == 1 ? int64_t {0} : static_cast<int64_t>(seconds);
	}();
	if (from < 0 || std::chrono::steady_clock::now() - start < std::chrono::seconds(from)) return false;
	static std::atomic<uint32_t> lines {0};
	return lines.fetch_add(1, std::memory_order_relaxed) < 4000;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	BufferCache::PerfCount(BufferCache::PerfEvent::ImageInsert);
	if (ImageChurnLogLine()) {
		std::printf("NHL27IMAGE: insert addr=0x%llx size=0x%llx fmt=%s %ux%ux%u levels=%u layers=%u "
		            "tile=%u samples=%u frame=%llu\n",
		            static_cast<unsigned long long>(info.data.address),
		            static_cast<unsigned long long>(info.data.size),
		            vk::to_string(info.pixel_format).c_str(), info.extent.width, info.extent.height,
		            info.extent.depth, info.resources.levels, info.resources.layers,
		            static_cast<uint32_t>(info.tile_mode), info.samples,
		            static_cast<unsigned long long>(BufferCache::PerfFlipCount()));
	}
	// A new image can become the answer for any T# overlapping its guest range (FindImagesInRegion /
	// ResolveOverlap candidates are exactly the registered images overlapping the T# range). Images without
	// guest data are never registered, so no T# lookup can find them.
	GenCount(FrameBreakdown::g_delta.gen_insert);
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	if (!info.data.Empty()) {
		RegisterImage(id);
		BumpGeneration(info.data.address, info.data.address + info.data.size);
	}

	return id;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
	});
	m_images_by_address[image.info.data.address].push_back(id);
	const auto filter_index = AddressFilterIndex(image.info.data.address);
	if (m_address_filter_counts[filter_index]++ == 0) {
		m_address_filter[filter_index / 64].fetch_or(uint64_t {1} << (filter_index % 64), std::memory_order_release);
	}
	m_address_filter_versions[filter_index >> 4u].fetch_add(1, std::memory_order_release);
	m_address_index_version.fetch_add(1, std::memory_order_release);
	TextureStateChanged();
	image.registered = true;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	image.lru_tick   = m_gc_tick;
	const auto epoch = ++m_image_set_epoch;
	ForEachPage(image.info.data.address, image.info.data.size,
	            [this, epoch](uint64_t page) { m_image_page_epochs[page] = epoch; });
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	GenCount(FrameBreakdown::g_delta.gen_unregister);
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	// The unregistered image itself is caught by the memo hit-path checks (slot id generation / registered).
	// It is also logged by range: removing a candidate can change which overlap candidate wins for another
	// T# overlapping it (FindImage picks the last matching candidate / last ResolveOverlap result).
	BumpGeneration(image.info.data.address, image.info.data.address + image.info.data.size);
	image.HeapChanged();

	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	});
	{
		const auto by_address = m_images_by_address.find(image.info.data.address);
		if (by_address == m_images_by_address.end()) {
			EXIT("TextureCache: image missing from address index\n");
		}
		auto& ids = by_address->second;
		const auto found = std::find(ids.begin(), ids.end(), id);
		if (found == ids.end()) {
			EXIT("TextureCache: image missing from address index\n");
		}
		ids.erase(found);
		if (ids.empty()) m_images_by_address.erase(by_address);
		if (const auto index = AddressFilterIndex(image.info.data.address); --m_address_filter_counts[index] == 0) {
			m_address_filter[index / 64].fetch_and(~(uint64_t {1} << (index % 64)), std::memory_order_release);
		}
	}
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	TextureStateChanged();
	image.registered = false;
	const auto epoch = ++m_image_set_epoch;
	ForEachPage(image.info.data.address, image.info.data.size,
	            [this, epoch](uint64_t page) { m_image_page_epochs[page] = epoch; });
}

bool TextureCache::HasImagesInRegion(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return FindImagesInRegion(address, size, false).size() != 0;
}

uint64_t TextureCache::ImageEpochInRegion(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return ImageEpochInRegionUnlocked(address, size);
}

uint64_t TextureCache::ImageEpochInRegionUnlocked(uint64_t address, uint64_t size) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return UINT64_MAX;
	}
	uint64_t epoch = 0;
	ForEachPage(address, size, [&](uint64_t page) {
		if (const auto* page_epoch = m_image_page_epochs.Find(page)) {
			epoch = std::max(epoch, *page_epoch);
		}
	});
	return epoch;
}

size_t TextureCache::ExactLookupSlot(const ImageInfo& info) noexcept {
	uint64_t hash = info.data.address * 0x9e3779b97f4a7c15ull;
	hash ^= (info.data.size + static_cast<uint64_t>(info.pixel_format)) * 0xc2b2ae3d27d4eb4full;
	hash ^= (static_cast<uint64_t>(info.extent.width) << 32u | info.extent.height) *
	        0x165667b19e3779f9ull;
	return static_cast<size_t>(hash >> 32u) % ExactLookupWays;
}

void TextureCache::DeleteImage(ImageId id) {
	GenCount(FrameBreakdown::g_delta.gen_delete);  // UnregisterImage below logs the range
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	if (image->depth_id) {
		std::erase(m_slot_images[image->depth_id].stencil_ids, id);
	} else if (image->stencil_associated) {
		// Detach first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image->stencil_ids, {})) {
			FreeImage(association);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.metadata.kind == ImageMetadataKind::Htile) {
		m_surface_metas.erase(image->info.metadata.range.address);
	}
	UnregisterImage(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferHostOperation([this, id] { m_slot_images.erase(id); });
	} else {
		m_slot_images.erase(id);
	}
}

namespace {

// KYTY_ALIAS_WRITEBACK=1 enables the aliased-image writeback/upload experiment. Off by default.
bool AliasWritebackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ALIAS_WRITEBACK");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

enum class AliasAction : int {
	Writeback,
	ForcedUpload,
	DepthInitFromGuest,
	WritebackSkipUnmapped
};

// Marks an overlap/alias resolution that frees images; restores the previous state on exit.
struct AliasFreeScope {
	bool& flag;
	bool  previous;
	explicit AliasFreeScope(bool& f) : flag(f), previous(f) { flag = true; }
	~AliasFreeScope() { flag = previous; }
	AliasFreeScope(const AliasFreeScope&)            = delete;
	AliasFreeScope& operator=(const AliasFreeScope&) = delete;
};

// One NHL27ALIAS line per (action, address); 5000 lines in total.
void AliasLogOnce(AliasAction action, uint64_t address, uint64_t size, const char* detail) {
	static constexpr uint32_t MaxLines = 5000;
	static std::mutex mutex;
	static std::set<std::pair<int, uint64_t>> seen;
	static uint32_t lines = 0;
	std::lock_guard lock(mutex);
	if (lines >= MaxLines || !seen.insert({static_cast<int>(action), address}).second) {
		return;
	}
	lines++;
	static const char* const names[] = {"writeback", "forced_upload", "depth_init_from_guest",
	                                    "writeback_skip_unmapped"};
	std::printf("NHL27ALIAS: %s addr=0x%llx size=0x%llx %s\n", names[static_cast<int>(action)],
	            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
	            detail);
}

} // namespace

bool TextureCache::WriteBackAliasedImage(ImageId id) {
	// Enqueue only: the download is recorded into the current command buffer and the guest
	// write-back runs as a priority operation. Do not call Finish()/DrainPriorityOperations() here.
	// This runs from FreeImage, reached from FindImage/ResolveOverlap, which still hold image ids.
	// Finish() runs pending deferred erases, which would leave those ids stale (SlotVector
	// operator[] assert). The image memory is released by a deferred erase that runs only after
	// the GPU tick, so the queued download still completes first.
	if (!m_scheduler.Active() || CommandScheduler::InDeferredOperation()) {
		return false;
	}
	const auto& image = m_slot_images[id];
	const auto  address = image.info.data.address;
	const auto  size    = image.info.data.size;
	const auto  format  = vk::to_string(image.backing.format);
	// Refused when the guest range is not backed (TryReadBacking fails) or the image cannot download.
	// The deferred write is tolerant, so an unmap between enqueue and execution is skipped too.
	if (!DownloadImageMemory(id, true)) {
		AliasLogOnce(AliasAction::WritebackSkipUnmapped, address, size, format.c_str());
		return false;
	}
	AliasLogOnce(AliasAction::Writeback, address, size, format.c_str());
	return true;
}

void TextureCache::FreeImage(ImageId id) {
	GenCount(FrameBreakdown::g_delta.gen_free);  // DeleteImage -> UnregisterImage logs the range
	auto& image = m_slot_images[id];
	if (ImageChurnLogLine()) {
		std::printf("NHL27IMAGE: free addr=0x%llx size=0x%llx fmt=%s %ux%u gpu_modified=%d frame=%llu\n",
		            static_cast<unsigned long long>(image.info.data.address),
		            static_cast<unsigned long long>(image.info.data.size),
		            vk::to_string(image.info.pixel_format).c_str(), image.info.extent.width,
		            image.info.extent.height, image.IsGpuModified() ? 1 : 0,
		            static_cast<unsigned long long>(BufferCache::PerfFlipCount()));
	}
	if (image.IsGpuModified()) {
		// Only overlap/alias frees write back. Unmap and GC frees must not enqueue guest writes.
		if (AliasWritebackEnabled() && m_alias_free_in_progress) {
			(void)WriteBackAliasedImage(id);
		}
		if (DepthAliasLog::Enabled() &&
		    DepthAliasLog::Want(image.info.data.address, image.info.data.size) &&
		    DepthAliasLog::TakeLine()) {
			// DIAGNOSTIC (KYTY_DBG_DEPTH_ALIAS): GPU-written content dropped without a download.
			std::printf("NHL27DEPTH: free_gpu_modified_no_download addr=0x%llx size=0x%llx "
			            "fmt=%s extent=%ux%u depth=%d\n",
			            static_cast<unsigned long long>(image.info.data.address),
			            static_cast<unsigned long long>(image.info.data.size),
			            vk::to_string(image.backing.format).c_str(), image.info.extent.width,
			            image.info.extent.height, image.info.IsDepth() ? 1 : 0);
		}
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (image.registered && image.lru_tick != m_gc_tick) {
		image.lru_tick = m_gc_tick;
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	// Generation is intentionally NOT bumped here (KYTY_DRAWPREP_DELTA): this only changes CPU-dirty and page-watch
	// state (flags, maybe-cpu hash, watcher ranges). FindImage selects images from info/registration only
	// (SameBacking, ResolveOverlap) and never reads those; dirtiness is acted on later by RefreshImage.
	// Registration/free changes still bump via Unregister/Delete/Free.
	GenCount(FrameBreakdown::g_delta.gen_dirty_skipped);
	image.MarkMaybeCpuDirty();
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		TextureStateChanged();
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	TextureStateChanged();
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	TextureStateChanged();
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	// No generation bump: watcher-range change only (see MarkAsMaybeDirty). Unregister bumps on removal.
	GenCount(FrameBreakdown::g_delta.gen_untrack_skipped);
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	TextureStateChanged();
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	// No generation bump: watcher-range change only (see MarkAsMaybeDirty).
	GenCount(FrameBreakdown::g_delta.gen_untrack_skipped);
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	TextureStateChanged();
	image.track_addr   = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	// No generation bump: watcher-range change only (see MarkAsMaybeDirty).
	GenCount(FrameBreakdown::g_delta.gen_untrack_skipped);
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	TextureStateChanged();
	image.track_addr_end = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	// Small tiled targets are written back too: games read GPU results such as reductions and
	// simulation state from them on the CPU (PPSA04263 keeps the origin of its ocean wave grid
	// in a 1x1 R32G32 target, and its water physics read it to sample the wave heights).
	const auto& info   = image.info;
	const auto  texels = static_cast<uint64_t>(info.extent.width) * info.extent.height *
	                     info.extent.depth * info.resources.layers;
	if (info.IsTiled() ? m_readback_tiled_images && texels <= SmallTiledReadbackTexels
	                   : m_readback_linear_images) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	return result;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto format = desc.info.pixel_format;
	if (const auto found = m_null_images.find(format); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(format, id);
	return id;
}

void TextureCache::ValidateImageViewDesc(const ImageDesc& desc) const {
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);

	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	const std::array                 depth_copies {
	    vk::BufferImageCopy {0,
	                         depth.info.pitch,
	                         depth.info.extent.height,
	                         {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                         {},
	                         depth.info.extent}};
	const std::array color_copies {
	    vk::BufferImageCopy {0,
	                         color.info.pitch,
	                         color.info.extent.height,
	                         {vk::ImageAspectFlagBits::eColor, 0, 0, layers},
	                         {},
	                         color.info.extent}};

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	if (m_slot_images[destination_id].IsBufferModified()) {
		RefreshImage(destination_id);
	}
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
}

bool TextureCache::DepthOverlapNeedsRecreate(const ImageInfo& requested, BindingType binding,
                                             const Image& cached) {
	// Mirrors ResolveDepthOverlap, which leaves other tile modes to the generic overlap path.
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return false;
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	// PPSA26785: the HTile of the depth source does not matter here, since CopyD16 reads the
	// already-resolved Vulkan depth aspect.
	const bool raw_d16_candidate = binding == BindingType::Texture && cached.info.IsDepth() &&
	                               requested.pixel_format == vk::Format::eR16Uint;
	const bool raw_d16_texture =
	    raw_d16_candidate && cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data.address == cached.info.data.address &&
	    requested.data.size <= cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources == cached.info.resources && requested.type == cached.info.type &&
	    requested.pitch == cached.info.pitch && !requested.HasStencil() &&
	    !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	if (raw_d16_candidate && !raw_d16_texture) {
		static std::atomic<int> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::printf("NHL26DBG: raw D16 texture rejected: guest_fmt=%u/%u backing=%d samples=%u/%u "
			            "addr=0x%llx/0x%llx size=0x%llx/0x%llx extent=%ux%ux%u/%ux%ux%u "
			            "res=%u,%u/%u,%u type=%d/%d pitch=%u/%u stencil=%d/%d meta=%d/%d\n",
			            static_cast<uint32_t>(cached.info.guest_format),
			            static_cast<uint32_t>(requested.guest_format),
			            static_cast<int>(cached.backing.format),
			            static_cast<uint32_t>(cached.backing.samples), requested.samples,
			            static_cast<unsigned long long>(cached.info.data.address),
			            static_cast<unsigned long long>(requested.data.address),
			            static_cast<unsigned long long>(cached.info.data.size),
			            static_cast<unsigned long long>(requested.data.size),
			            cached.info.extent.width, cached.info.extent.height, cached.info.extent.depth,
			            requested.extent.width, requested.extent.height, requested.extent.depth,
			            cached.info.resources.levels, cached.info.resources.layers,
			            requested.resources.levels, requested.resources.layers,
			            static_cast<int>(cached.info.type), static_cast<int>(requested.type),
			            cached.info.pitch, requested.pitch, cached.info.HasStencil(),
			            requested.HasStencil(), static_cast<int>(cached.info.metadata.kind),
			            static_cast<int>(requested.metadata.kind));
			std::fflush(stdout);
		}
	}
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	return recreate;
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	// Frees here (via ResolveOverlap) are overlap frees; the flag is set by the caller.
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	if (!DepthOverlapNeedsRecreate(requested, binding, cached)) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout          = cached.info.mip_layout;
		info.tiling              = cached.info.tiling;
		info.tiled_slice_stride  = cached.info.tiled_slice_stride;
		info.linear_slice_stride = cached.info.linear_slice_stride;
		info.first_tail_level    = cached.info.first_tail_level;
	}
	info.htile_initial_clear  = false;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
		BumpGeneration(cached.info.data.address, cached.info.data.address + cached.info.data.size);
		GenCount(FrameBreakdown::g_delta.gen_rebind);
	}
	const char* copy_path = "none";
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported && AliasWritebackEnabled() && !cached.IsGpuModified()) {
			// Experiment: the old image was never GPU-written, so its contents are not
			// authoritative. Populate the replacement from guest memory instead of copying.
			copy_path = "guest_upload";
			replacement.MarkBufferModified();
			AliasLogOnce(AliasAction::DepthInitFromGuest, cached.info.data.address,
			             cached.info.data.size, vk::to_string(requested.pixel_format).c_str());
		} else if (copy_supported) {
			copy_path = "CopyImage";
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		copy_path = "ReinterpretMsDepth";
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	if (DepthAliasLog::Enabled() &&
	    DepthAliasLog::WantEither(cached.info.data.address, cached.info.data.size,
	                              requested.data.address, requested.data.size) &&
	    DepthAliasLog::TakeLine()) {
		// DIAGNOSTIC (KYTY_DBG_DEPTH_ALIAS): depth image recreated from an incompatible image.
		// "ever uploaded" is not tracked per image; buffer_modified/cpu_dirty are the closest state.
		std::printf("NHL27DEPTH: depth_recreate old addr=0x%llx size=0x%llx fmt=%s backing=%s "
		            "extent=%ux%u samples=%u gpu_modified=%d buffer_modified=%d cpu_dirty=%d "
		            "-> new fmt=%s backing=%s depth=%d copy=%s\n",
		            static_cast<unsigned long long>(cached.info.data.address),
		            static_cast<unsigned long long>(cached.info.data.size),
		            vk::to_string(cached.info.pixel_format).c_str(),
		            vk::to_string(cached.backing.format).c_str(), cached.info.extent.width,
		            cached.info.extent.height, cached.backing.samples,
		            cached.IsGpuModified() ? 1 : 0, cached.IsBufferModified() ? 1 : 0,
		            cached.IsCpuDirty() ? 1 : 0, vk::to_string(requested.pixel_format).c_str(),
		            vk::to_string(replacement.backing.format).c_str(),
		            replacement.info.IsDepth() ? 1 : 0, copy_path);
	}
	FreeImage(cached_id);
	return replacement_id;
}

namespace BcAlias {

bool Matches(const ImageInfo& requested, const ImageInfo& cached) {
	// KYTY_BC_ALIAS=1 (off by default): resolve block/non-block aliases by publishing and freeing
	// one image. Off, compute writes to BC images go through their storage shadows; with both,
	// NHL27's GPU-composited kit textures were freed before their shadows flushed and rendered
	// as the uncoloured template.
	static const bool enabled = [] {
		const char* setting = std::getenv("KYTY_BC_ALIAS");
		return setting != nullptr && setting[0] == '1';
	}();
	return enabled && requested.guest_format != cached.guest_format &&
	       requested.IsBlock() != cached.IsBlock() &&
	       requested.tile_mode == cached.tile_mode &&
	       requested.bytes_per_block == cached.bytes_per_block &&
	       requested.samples == 1 && cached.samples == 1 &&
	       (requested.data != cached.data || requested.resources != cached.resources ||
	        requested.BlockExtent() != cached.BlockExtent()) &&
	       !TextureCache::SameBacking(cached, requested, false);
}

bool Publish(TextureCache& cache, Image& image) {
	if (!cache.CanDownload(image)) {
		return false;
	}
	// ObtainBuffer preserves padding and records ownership before the GPU-only transfer.
	auto [buffer, offset] =
	    cache.m_buffer_cache.ObtainBuffer(image.info.data.address, image.info.data.size, true);
	cache.DownloadImage(image, *buffer, offset, image.info.data.size, image.info.resources.levels);
	image.ClearGpuModified();
	image.MarkBufferModified();
	return true;
}

void Resolve(TextureCache& cache, ImageId cached_id, ImageId merged_id) {
	auto& cached = cache.m_slot_images[cached_id];
	auto* merged = merged_id ? &cache.m_slot_images[merged_id] : nullptr;
	// PS5 raw block images can span several packed BC mips. Transfer their complete
	// guest bytes, including tail padding, before retiring the incompatible backing.
	if (cached.SafeToDownload()) {
		if (merged && merged->SafeToDownload() && !Publish(cache, *merged)) {
			return;
		}
		if (!Publish(cache, cached)) {
			return;
		}
		if (merged) {
			merged->MarkBufferModified();
		}
	}
	cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
	cache.FreeImage(cached_id);
}

} // namespace BcAlias

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id, bool speculative) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr || !owner->registered) {
		return {merged_id};
	}
	if (BcAlias::Matches(requested, owner->info) && !owner->IsCpuDirty()) {
		BcAlias::Resolve(*this, cached_id, merged_id);
		return {merged_id};
	}
	// Every FreeImage below (and in ResolveDepthOverlap/ExpandImage) is an overlap free.
	AliasFreeScope alias_free_scope {m_alias_free_in_progress};
	auto&      cached       = *owner;
	const auto current_tick = m_scheduler.CurrentTick();
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    cached.heap_refs.load(std::memory_order_relaxed) == 0u;

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (speculative && (cached.info.IsDepth() || requested.IsDepth()) &&
		    DepthOverlapNeedsRecreate(requested, binding, cached)) {
			return {.rejected = true};
		}
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     !std::equal(requested.mip_layout.begin(), requested.mip_layout.end(),
		                 cached.info.mip_layout.begin(),
		                 [](const ImageMipInfo& left, const ImageMipInfo& right) {
			                 return left.offset == right.offset && left.size == right.size &&
			                        left.pitch == right.pitch && left.height == right.height &&
			                        left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
			                        left.surface_z == right.surface_z;
		                 }))) {
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		BumpGeneration(cached.info.data.address, cached.info.data.address + cached.info.data.size);
		GenCount(FrameBreakdown::g_delta.gen_rebind);
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (DepthAliasLog::Enabled() &&
	    DepthAliasLog::WantEither(requested.data.address, requested.data.size,
	                              cached.info.data.address, cached.info.data.size) &&
	    DepthAliasLog::TakeLine()) {
		// DIAGNOSTIC (KYTY_DBG_DEPTH_ALIAS): partial overlap that falls through without a copy.
		std::printf("NHL27DEPTH: partial_overlap_skipped requested addr=0x%llx size=0x%llx "
		            "fmt=%s | cached addr=0x%llx size=0x%llx fmt=%s freed=%d\n",
		            static_cast<unsigned long long>(requested.data.address),
		            static_cast<unsigned long long>(requested.data.size),
		            vk::to_string(requested.pixel_format).c_str(),
		            static_cast<unsigned long long>(cached.info.data.address),
		            static_cast<unsigned long long>(cached.info.data.size),
		            vk::to_string(cached.info.pixel_format).c_str(),
		            (requested.data.address >= cached.info.data.address && safe_to_delete) ? 1 : 0);
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
		BumpGeneration(source.info.data.address, source.info.data.address + source.info.data.size);
		GenCount(FrameBreakdown::g_delta.gen_rebind);
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	return expanded_id;
}

bool TextureCache::CanDownload(const Image& image) const {
	const auto& info = image.info;
	if (info.samples != 1 || image.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		return IsSupportedDepthPlaneReadback(info) &&
		       Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
	}
	return info.metadata.compression == VideoOutCompression::Uncompressed &&
	       (!info.IsTiled() || !Prospero::IsFmaskTextureFormat(info.guest_format)) &&
	       info.LinearSize() != 0;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	BufferCache::PerfCount(BufferCache::PerfEvent::ImageUpload);
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto& info        = image.info;
	const auto  binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	if (binding != BindingType::DepthTarget) {
		if ((binding == BindingType::RenderTarget || binding == BindingType::VideoOut) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		if (binding == BindingType::VideoOut &&
		    info.metadata.compression != VideoOutCompression::Uncompressed) {
			EXIT("TextureCache: invalid video-out upload\n");
		}
		const auto linear = m_tiler.Detile(source.Handle(), source_offset, info);
		const auto copies = info.BufferCopies(linear.offset);
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
		return;
	}

	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto layers          = info.resources.layers;
	const auto full_slice_size = info.linear_slice_stride;
	auto       linear          = m_tiler.Detile(source.Handle(), source_offset, info);
	auto       copies          = info.BufferCopies(linear.offset);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		copies.resize(1);
		copies[0].bufferOffset      = linear.offset;
		copies[0].bufferRowLength   = info.pitch;
		copies[0].bufferImageHeight = info.extent.height;
		copies[0].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, 0, layers};
		copies[0].imageExtent       = info.extent;
	}
	destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range = desc.info.metadata.range;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	// Only the selected slices are inspected below. Synchronize actual dirty bytes in that
	// interval, rather than other layers or unrelated GPU-owned bytes on tracked pages.
	const auto slice_size = range.size / layers;
	const auto selected_address = range.address + slice_size * first;
	const auto selected_size = slice_size * count;
	// KYTY_PERF_NO_SYNC_READBACK: when the GPU wrote these slices with a uniform dword fill that
	// the buffer cache recorded, the clear code is known without draining the GPU queue. The
	// fill is dropped by any later write, download or invalidation of the range.
	uint32_t shadow_value = 0;
	// Asked once: the three checks below wanted it each (0.8% of the execution thread in play).
	const bool gpu_dirty = selected_size != 0 && m_buffer_cache.HasGpuDirtyBytes(selected_address, selected_size);
	const bool shadowed  = gpu_dirty && desc.type != BindingType::VideoOut && BufferCache::PerfNoSyncReadback() &&
	                      m_buffer_cache.TryGetGpuFill(selected_address, selected_size, shadow_value);
	if (shadowed) {
		BufferCache::PerfCount(BufferCache::PerfEvent::ColorClearShadowHit);
	}
	// Experiment FillPatterns: the GPU filled every selected slice with a repeating element whose
	// bytes differ, so no slice holds one repeated clear code: nothing to materialize, and no
	// readback is needed to find that out.
	if (!shadowed && selected_size != 0 && desc.type != BindingType::VideoOut &&
	    Common::Experiments::EnvEnabled(Common::Experiments::EnvBit::FillPatterns) &&
	    BufferCache::PerfNoSyncReadback() &&
	    m_buffer_cache.HasGpuDirtyBytes(selected_address, selected_size) &&
	    m_buffer_cache.GpuFillCoversNonUniform(selected_address, selected_size)) {
		if (FrameBreakdown::Enabled()) {
			static std::atomic<uint64_t> hits {0};
			static std::atomic<int64_t>  last_print {0};
			const auto n = hits.fetch_add(1, std::memory_order_relaxed) + 1;
			const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
			                        std::chrono::steady_clock::now().time_since_epoch()).count();
			int64_t last = last_print.load(std::memory_order_relaxed);
			if (now - last >= 5000 && last_print.compare_exchange_strong(last, now)) {
				std::printf("NHL27PATTERN: color-clear readbacks skipped=%llu\n",
				            static_cast<unsigned long long>(n));
				std::fflush(stdout);
			}
		}
		return;
	}
	// Experiment GpuColorClear: the GPU wrote these metadata bytes. Check them on the GPU and clear
	// through conditional rendering instead of reading them back, which waits for that GPU work.
	if (!shadowed && gpu_dirty && m_graphics.conditional_rendering_enabled &&
	    Common::Experiments::Enabled(Common::Experiments::GpuColorClear) &&
	    MaterializeColorClearOnGpu(id, desc, selected_address, slice_size, count, image_first)) {
		return;
	}
	// This may submit; never hold the texture lock across the readback.
	if (!shadowed && gpu_dirty) {
		static const bool debug_cpu = std::getenv("KYTY_DBG_GPU_CLEAR") != nullptr;
		static uint32_t   cpu_logged = 0;
		if (debug_cpu && cpu_logged++ < 32) {
			std::printf("NHL27GPUCLEAR: cpu-readback address=0x%llx size=%llu cond=%d exp=%d\n",
			            static_cast<unsigned long long>(selected_address),
			            static_cast<unsigned long long>(selected_size), m_graphics.conditional_rendering_enabled ? 1 : 0,
			            Common::Experiments::Enabled(Common::Experiments::GpuColorClear) ? 1 : 0);
		}
		m_buffer_cache.ReadMemory(selected_address, selected_size, false,
		                          BufferCache::ReadCaller::ColorClear);
	}
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (shadowed) {
			// Slices are 4 KiB aligned, so the first byte is the fill's low byte.
			code = static_cast<uint8_t>(shadow_value >> (((address - selected_address) & 3u) * 8u));
		} else if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		if (shadowed) {
			// A repeated dword is one repeated byte only when all four of its bytes agree.
			if (shadow_value != static_cast<uint32_t>(code) * 0x01010101u) {
				continue;
			}
		} else {
			// Compared in the backing store, eight bytes at a time, up to the first other byte:
			// copying the slice (tens of KiB) first, for render targets whose first metadata byte is
			// a clear code but whose slice is not uniform, ran on every lookup of them (with a heap
			// allocation, 1.7% of the execution thread in play).
			std::vector<uint8_t> copy;
			const auto*          bytes = static_cast<const uint8_t*>(LibKernel::Memory::FindBackingPointer(address, slice_size));
			if (bytes == nullptr) {
				copy.resize(slice_size);
				if (!LibKernel::Memory::TryReadBacking(address, copy.data(), copy.size())) {
					EXIT("TextureCache: failed to read color metadata slice\n");
				}
				bytes = copy.data();
			}
			const uint64_t pattern = uint64_t {code} * 0x0101010101010101ull;
			size_t         at      = 0;
			bool           uniform = true;
			for (; uniform && at + sizeof(uint64_t) <= slice_size; at += sizeof(uint64_t)) {
				uint64_t word = 0;
				std::memcpy(&word, bytes + at, sizeof(word));
				uniform = word == pattern;
			}
			for (; uniform && at < slice_size; ++at) {
				uniform = bytes[at] == code;
			}
			if (!uniform) {
				continue;
			}
		}
		{
			std::scoped_lock lock {m_lock};
			const auto& image = m_slot_images[id];
			const bool depth = image.info.IsDepth();
			if (depth) {
				const auto value = clear.color.float32[0];
				clear.depthStencil = {value, 0u};
			}
			ClearImage(m_scheduler.Current(), id, depth ? image.backing.format : view.format,
			           {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
			            view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data, and without reading GPU-owned metadata back first.
		if (desc.type != BindingType::VideoOut) {
			const std::vector<uint8_t> bytes(slice_size, uint8_t {0xff});
			if (shadowed) {
				// The expanded keys below replace the GPU's fill, so there is nothing to download.
				m_buffer_cache.DiscardGpuModified(address, slice_size);
			}
			BufferCache::WriteTagScope tag(BufferCache::WriteTag::TextureKeys);
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
	}
}

bool TextureCache::MaterializeColorClearOnGpu(ImageId id, const ImageDesc& desc, uint64_t address,
                                              uint64_t slice_size, uint32_t count, uint32_t image_first) {
	const auto& view = desc.view_info;
	// KYTY_DBG_GPU_CLEAR=1: which path the first checks take (NHL27GPUCLEAR).
	static const bool debug = std::getenv("KYTY_DBG_GPU_CLEAR") != nullptr;
	static uint32_t   logged = 0;
	const auto        note   = [&](const char* what) {
		if (debug && logged++ < 64) {
			std::printf("NHL27GPUCLEAR: %s address=0x%llx slice=%llu count=%u levels=%u volume=%d type=%d\n", what,
			            static_cast<unsigned long long>(address), static_cast<unsigned long long>(slice_size), count,
			            view.level_count, desc.info.IsVolume() ? 1 : 0, static_cast<int>(desc.type));
		}
	};
	if (count == 0 || count > 64 || view.level_count != 1 || desc.info.IsVolume() || slice_size % 4 != 0 ||
	    slice_size / 4 > UINT32_MAX) {
		note("shape");
		return false;
	}
	// The codes a uniform slice could hold that decode to a clear color for this view.
	std::array<uint8_t, 8>             codes {};
	std::array<vk::ClearColorValue, 8> colors {};
	size_t                             candidates = 0;
	for (const uint8_t code: {uint8_t {0x00}, uint8_t {0x20}, uint8_t {0x40}, uint8_t {0x80}, uint8_t {0xc0}}) {
		vk::ClearColorValue color {};
		if (DecodeColorClear(desc, code, color)) {
			codes[candidates]    = code;
			colors[candidates++] = color;
		}
	}
	if (candidates == 0) {
		note("no-candidates");
		return true; // No slice could clear: the CPU path would neither clear nor expand.
	}
	{
		std::scoped_lock lock {m_lock};
		auto&            image = m_slot_images[id];
		if (!(image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment)) {
			note("no-attachment-usage");
			return false;
		}
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const bool expand        = desc.type != BindingType::VideoOut;
	const auto [buffer, offset] = m_buffer_cache.ObtainBuffer(address, slice_size * count, expand, false);
	if (!buffer->HasDeviceAddress()) {
		note("stream-buffer");
		return false;
	}
	if (m_color_clear_check == nullptr) {
		m_color_clear_check = std::make_shared<ColorClearCheck>(m_graphics, m_scheduler);
	}
	const auto [predicates, first] = m_color_clear_check->Record(
	    command, buffer->BufferDeviceAddress() + offset, static_cast<uint32_t>(slice_size / 4), count,
	    std::span(codes.data(), candidates), expand);
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TrackImage(id);
	if (image.IsBufferModified() || image.IsCpuDirty()) {
		InitializeImage(id);
	}
	const auto       native = command.Handle();
	const vk::Extent2D extent {std::max(image.info.extent.width >> view.base_level, 1u),
	                           std::max(image.info.extent.height >> view.base_level, 1u)};
	for (uint32_t slice = 0; slice < count; slice++) {
		ImageViewInfo target {};
		target.format      = view.format;
		target.type        = vk::ImageViewType::e2D;
		target.base_level  = view.base_level;
		target.base_layer  = image_first + slice;
		target.layer_count = 1;
		target.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal, vk::AccessFlagBits2::eColorAttachmentWrite, {}, native);
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(target);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = extent;
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		native.beginRendering(&rendering);
		for (size_t k = 0; k < candidates; k++) {
			vk::ConditionalRenderingBeginInfoEXT condition {};
			condition.buffer = predicates;
			condition.offset = first + (static_cast<uint64_t>(slice) * 8u + k) * sizeof(uint32_t);
			native.beginConditionalRenderingEXT(&condition);
			vk::ClearAttachment clear {};
			clear.aspectMask        = vk::ImageAspectFlagBits::eColor;
			clear.colorAttachment   = 0;
			clear.clearValue.color  = colors[k];
			const vk::ClearRect rect {{{0, 0}, extent}, 0, 1};
			native.clearAttachments(1, &clear, 1, &rect);
			native.endConditionalRenderingEXT();
		}
		native.endRendering();
	}
	CommitGpuWrite(image);
	note("gpu");
	return true;
}

void TextureCache::RefreshImage(ImageId id) {
	FrameBreakdown::Timer tex_timer(FrameBreakdown::g_tex.refresh_ns, FrameBreakdown::TexOn());
	FrameBreakdown::TexAdd(FrameBreakdown::g_tex.refresh_calls);
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		(void)image.ResolveMaybeCpuHash(image.HashGuestEdges());
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	FrameBreakdown::TexAdd(FrameBreakdown::g_tex.uploads);
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		ImageInfo plane {};
		plane.data         = stencil;
		plane.pixel_format = vk::Format::eS8Uint;
		plane.guest_format = Prospero::BufferFormat::k8UInt;
		plane.type         = depth.info.type;
		plane.extent       = depth.info.extent;
		plane.resources    = {depth.stencil_subresources.level_count,
		                      depth.stencil_subresources.layer_count};
		plane.tile_mode    = depth.info.tile_mode;
		plane.pitch =
		    plane.IsTiled() ? TileGetDepthPitch(plane.extent.width, 1, 0) : depth.info.pitch;
		plane.bytes_per_block = 1;
		plane.UpdateSize();
		if (record.depth_id != depth_id) {
			if (record.depth_id) {
				std::erase(m_slot_images[record.depth_id].stencil_ids, association);
			}
			depth.stencil_ids.push_back(association);
		}
		TextureStateChanged();
		record.info                 = plane;
		record.depth_id             = depth_id;
		record.stencil_subresources = depth.stencil_subresources;
	}
	depth.stencil_associated = true;

	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	return FindImageImpl(desc, exact_format, false);
}

ImageId TextureCache::FindImageSpeculative(ImageDesc& desc, bool exact_format) {
	return FindImageImpl(desc, exact_format, true);
}

ImageId TextureCache::FindImageImpl(ImageDesc& desc, bool exact_format, bool speculative) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	// The image description of a remembered lookup was validated when it was remembered.
	ValidateImageViewDesc(desc);
	if (desc.info.data.Empty()) {
		ImageOps::Validate(desc.info);
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	ValidateImageDesc(desc);
	EXIT_IF(desc.info.data.Empty());

	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		// Draws look up the same targets and textures over and over: remember lookups that found
		// an image with the same backing (no overlap resolution, no insertion) until the image set
		// changes.
		auto& exact = (*m_exact_lookups)[ExactLookupSlot(desc.info)];
		if (exact.valid && exact.exact_format == exact_format &&
		    desc.type != BindingType::VideoOut &&
		    std::memcmp(&exact.info, &desc.info, sizeof(ImageInfo)) == 0 &&
		    (exact.epoch == m_image_set_epoch ||
		     ImageEpochInRegionUnlocked(desc.info.data.address, desc.info.data.size) <=
		         exact.epoch)) {
			// No image of the range changed until now: the next lookup compares with now instead of
			// scanning the pages of the range again.
			exact.epoch              = m_image_set_epoch;
			auto& image              = m_slot_images[exact.id];
			image.tick_accessed_last = m_scheduler.CurrentTick();
			TouchImage(image);
			result = exact.id;
		}
	}
	if (result) {
		MaterializeColorClear(result, desc, metadata_base_layer);
		return result;
	}
	ImageOps::Validate(desc.info);
	{
		std::scoped_lock lock {m_lock};
		bool block_alias = false;

		// A color render target can alias a depth buffer's stencil plane (games write stencil via an
		// R8 color target). Stencil association records have no native image, so never hand one
		// out as a color target; a separate color image is created for that memory instead.
		const auto skip_candidate = [&](ImageId id) {
			return desc.type == BindingType::RenderTarget && m_slot_images[id].depth_id;
		};
		// Stencil descriptors name the associated plane; its depth owner supplies the native image
		// and the canonical guest tiling.
		const auto stencil_association = [&](const Image& image) {
			return (desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
			       image.depth_id && image.info.data == desc.info.data &&
			       m_slot_images[image.depth_id].info.samples == desc.info.samples &&
			       image.info.extent == desc.info.extent && image.info.type == desc.info.type &&
			       image.info.resources.levels >= desc.info.resources.levels &&
			       image.info.resources.layers >= desc.info.resources.layers &&
			       desc.info.bytes_per_block == 1 &&
			       desc.info.guest_format == Prospero::BufferFormat::k8UInt;
		};
		const bool stencil_shaped = (desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
		                            desc.info.bytes_per_block == 1 &&
		                            desc.info.guest_format == Prospero::BufferFormat::k8UInt;
		// Fast path: SameBacking requires an identical start address, so every image the region
		// scan below could select is in the address index. With exactly one match the scan
		// would pick that same image; skip walking every owner on the covered 1 MiB pages.
		// Stencil-shaped requests and block-compressed aliases at this address take the scan.
		if (const auto by_address = m_images_by_address.find(desc.info.data.address);
		    !stencil_shaped && by_address != m_images_by_address.end()) {
			uint32_t matches = 0;
			ImageId  match {};
			for (const auto id: by_address->second) {
				if (skip_candidate(id)) continue;
				const auto& image = m_slot_images[id];
				if (SameBacking(image.info, desc.info, exact_format)) {
					match = id;
					++matches;
				} else if (image.SafeToDownload() && BcAlias::Matches(desc.info, image.info)) {
					block_alias = true;
				}
			}
			if (matches == 1 && !block_alias) result = match;
		}
		FrameBreakdown::Add(result ? FrameBreakdown::g_counters.port_hint_hits
		                           : FrameBreakdown::g_counters.port_hint_misses);
		ImageIds candidates;
		if (!result) {
			block_alias = false;
			candidates  = FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
			for (const auto id: candidates) {
				if (skip_candidate(id)) continue;
				const auto& image = m_slot_images[id];
				if (stencil_association(image)) {
					result = id;
					break;
				}
				if (SameBacking(image.info, desc.info, exact_format)) {
					result = id;
				} else if (image.SafeToDownload() && BcAlias::Matches(desc.info, image.info)) {
					block_alias = true;
				}
			}
		}
		const bool same_backing = static_cast<bool>(result);

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				if (skip_candidate(candidate)) {
					continue;
				}
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap =
				    ResolveOverlap(merged_info, desc.type, candidate, result, speculative);
				if (overlap.rejected) {
					return {};
				}
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
		} else if (block_alias) {
			for (const auto candidate: candidates) {
				const auto* owner = m_slot_images.try_get(candidate);
				if (candidate != result && owner != nullptr && owner->registered &&
				    !owner->IsCpuDirty() &&
				    BcAlias::Matches(m_slot_images[result].info, owner->info)) {
					BcAlias::Resolve(*this, candidate, result);
				}
			}
		}

		if (result && !m_slot_images[result].depth_id) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				AliasFreeScope alias_free_scope {m_alias_free_in_progress};
				FreeImage(result);
				result = {};
			}
		}
		if (result && same_backing && desc.type != BindingType::VideoOut) {
			auto& exact = (*m_exact_lookups)[ExactLookupSlot(desc.info)];
			std::memcpy(&exact.info, &desc.info, sizeof(ImageInfo));
			exact.id           = result;
			exact.epoch        = m_image_set_epoch;
			exact.exact_format = exact_format;
			exact.valid        = true;
		}
		if (!result) {
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			} else if (AliasWritebackEnabled()) {
				// Experiment: a fresh image must not stay uninitialized; upload guest memory.
				inserted.MarkBufferModified();
				AliasLogOnce(AliasAction::ForcedUpload, inserted.info.data.address,
				             inserted.info.data.size, vk::to_string(desc.info.pixel_format).c_str());
			} else if (desc.type == BindingType::Texture && DepthAliasLog::Enabled() &&
			           DepthAliasLog::Want(inserted.info.data.address, inserted.info.data.size) &&
			           DepthAliasLog::TakeLine()) {
				// DIAGNOSTIC (KYTY_DBG_DEPTH_ALIAS): sampled image created fresh, nothing to upload.
				std::printf("NHL27DEPTH: fresh_no_upload addr=0x%llx size=0x%llx fmt=%s extent=%ux%u "
				            "depth=%d\n",
				            static_cast<unsigned long long>(inserted.info.data.address),
				            static_cast<unsigned long long>(inserted.info.data.size),
				            vk::to_string(desc.info.pixel_format).c_str(),
				            desc.info.extent.width, desc.info.extent.height,
				            desc.info.IsDepth() ? 1 : 0);
			}
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

ImageId TextureCache::FindImageHinted(ImageDesc& desc, bool exact_format, ImageId hint,
                                      bool& exact) {
	exact = false;
	const auto metadata_base_layer = desc.view_info.base_layer;
	const auto matches             = [&](const Image& image) {
		return image.registered && !(desc.type == BindingType::RenderTarget && image.depth_id) &&
		       SameBacking(image.info, desc.info, exact_format) &&
		       !(image.info.resources < desc.info.resources) &&
		       (!exact_format || image.info.pixel_format == desc.info.pixel_format);
	};
	if (hint && desc.type != BindingType::VideoOut && !desc.info.data.Empty() &&
	    !m_scheduler.Current().IsInvalid()) {
		bool hit = false;
		{
			std::scoped_lock lock {m_lock};
			if (auto* image = m_slot_images.try_get(hint); image != nullptr && matches(*image)) {
				image->tick_accessed_last = m_scheduler.CurrentTick();
				TouchImage(*image);
				hit = true;
			}
		}
		if (hit) {
			MaterializeColorClear(hint, desc, metadata_base_layer);
			exact = true;
			return hint;
		}
	}
	const auto base_level = desc.view_info.base_level;
	const auto id         = FindImage(desc, exact_format);
	{
		std::scoped_lock lock {m_lock};
		const auto* image = m_slot_images.try_get(id);
		exact = image != nullptr && desc.type != BindingType::VideoOut && !desc.info.data.Empty() &&
		        desc.view_info.base_level == base_level &&
		        desc.view_info.base_layer == metadata_base_layer && matches(*image);
	}
	return id;
}

void TextureCache::NoteImageUse(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	image.tick_accessed_last = m_scheduler.CurrentTick();
	TouchImage(image);
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	if (!AddressFilterHit(address)) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	// Only images that start at `address` match. Texel buffer binds ask on every draw and nearly
	// always find none: one hash lookup instead of walking the pages of the range.
	if (!m_images_by_address.contains(address)) {
		return {};
	}
	ImageIds         matches;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr || (ensure_valid && !owner->SafeToDownload())) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) {
			selected = owner->depth_id;
		}
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	if (desc.type == BindingType::Storage && image.NeedsStorageShadow() &&
	    desc.view_info.level_count == 1 && vk::blockExtent(desc.view_info.format)[0] == 1 &&
	    vk::blockSize(desc.view_info.format) == image.info.bytes_per_block) {
		if (std::find(m_shadowed_images.begin(), m_shadowed_images.end(), id) ==
		    m_shadowed_images.end()) {
			m_shadowed_images.push_back(id);
			m_storage_shadows_pending.store(true, std::memory_order_release);
		}
		return image.PrepareShadowStorage(desc.view_info);
	}
	return image.FindView(desc.view_info);
}

void TextureCache::FlushStorageShadows() {
	// Most draws have no block-storage writes; avoid taking the cache lock for them.
	if (!m_storage_shadows_pending.load(std::memory_order_acquire)) return;
	std::scoped_lock lock {m_lock};
	for (const auto id: m_shadowed_images) {
		if (auto* image = m_slot_images.try_get(id); image != nullptr) {
			image->FlushShadowStorage();
		}
	}
	m_shadowed_images.clear();
	m_storage_shadows_pending.store(false, std::memory_order_release);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	if (!image.usage.render_target) TextureStateChanged();
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc, ImageId* stencil) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const auto [metadata, inserted] = m_surface_metas.try_emplace(desc.info.metadata.range.address);
		if (inserted && image.info.htile_initial_clear) {
			metadata->second.clear_mask.set();
		}
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		const auto association = AssociateStencil(id, desc.info.stencil);
		RefreshImage(association);
		if (stencil != nullptr) {
			*stencil = association;
		}
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	image.MarkGpuModified();
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		BeginColorRendering(image, format, range.baseMipLevel, range.baseArrayLayer,
		                    range.layerCount, &clear, command.Handle());
		command.Handle().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	// Guest levels past the host chain alias its last (1x1) level.
	const auto host_levels    = image.backing.mip_levels;
	native_range.baseMipLevel = std::min(native_range.baseMipLevel, host_levels - 1);
	native_range.levelCount =
	    std::min(native_range.levelCount, host_levels - native_range.baseMipLevel);
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

vk::Rect2D TextureCache::BeginColorRendering(Image& image, vk::Format format, uint32_t level,
                                             uint32_t layer, uint32_t layers,
                                             const vk::ClearValue* clear,
                                             CommandRecorder       command) {
	ImageViewInfo view {};
	view.format      = format;
	view.type        = layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	view.base_level  = level;
	view.base_layer  = layer;
	view.layer_count = layers;
	view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	              vk::AccessFlagBits2::eColorAttachmentRead |
	                  vk::AccessFlagBits2::eColorAttachmentWrite,
	              {}, command);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = image.FindView(view);
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	if (clear != nullptr) {
		attachment.loadOp     = vk::AttachmentLoadOp::eClear;
		attachment.clearValue = *clear;
	}
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {std::max(image.info.extent.width >> level, 1u),
	                                  std::max(image.info.extent.height >> level, 1u)};
	rendering.layerCount           = layers;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	return rendering.renderArea;
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	std::unique_lock lock {m_lock, std::defer_lock};
	{
		FrameBreakdown::FaultTimer lock_timer(FrameBreakdown::g_fault_cost.lock_ns);
		lock.lock();
	}
	InvalidateCpuAliases(address, size);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.linear_slice_stride;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			const auto copies = info.BufferCopies(destination_offset);
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
		} else {
			const auto copies = info.BufferCopies();
			m_tiler.TileImage(image, copies, destination.Handle(), destination_offset,
			                  info.data.size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	const std::array copies {vk::BufferImageCopy {0,
	                                              info.pitch,
	                                              info.extent.height,
	                                              {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                                              {},
	                                              info.extent}};
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto guest_linear = tiled ? m_tiler.GetScratchBuffer(info.LinearSize(), host_linear.buffer)
	                          : TileManager::Result {destination.Handle(), destination_offset,
	                                                 destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (tiled) {
		m_tiler.Tile(guest_linear.buffer, guest_linear.offset, guest_linear.size,
		             destination.Handle(), destination_offset, info.data.size, info);
	}
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                 uint64_t destination_size, uint32_t levels) {
	if (!CanDownload(image)) {
		EXIT("TextureCache: invalid image download\n");
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		EXIT_IF(destination_size != image.info.data.size || levels != image.info.resources.levels);
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	const auto& info      = image.info;
	const auto  transform = info.GetColorTransform();
	if (info.IsTiled()) {
		const auto copies = info.BufferCopies(0, levels);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, destination_size,
		                  levels);
	} else if (transform != ColorTransform::None) {
		auto       linear = m_tiler.GetScratchBuffer(destination_size);
		const auto copies = info.BufferCopies(0, levels);
		image.Download(copies, linear.buffer, 0, linear.size);
		m_tiler.TransformColor(linear, {destination.Handle(), destination_offset, destination_size},
		                       transform, false);
	} else {
		const auto copies = info.BufferCopies(destination_offset, levels);
		image.Download(copies, destination.Handle(), destination_offset, destination_size);
	}
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}
	buffer.last_write_tick = m_scheduler.CurrentTick();

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	if (!m_texture_cache.CanDownload(image) ||
	    (TextureCache::UploadBinding(image) == TextureCache::BindingType::DepthTarget &&
	     copy_size != image.info.data.size)) {
		return false;
	}
	ForgetColorMetadataChecks(image.info.data.address, copy_size);
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, levels);
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id, bool tolerant_write) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	if (!image.SafeToDownload() || !CanDownload(image)) {
		return false;
	}
	const auto range    = image.info.data;
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] =
	    download.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	if (mapped == nullptr) {
		EXIT("TextureCache: failed to map reusable download buffer\n");
	}
	download.Commit();
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	download.Flush(offset, range.size);

	DownloadImage(image, download, offset, range.size, image.info.resources.levels);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureDownloadBytes, range.size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	// Only what is still mapped by then (unmaps no longer wait for the publish).
	const auto unmap_sequence = m_scheduler.Context().UnmapSequence();
	m_scheduler.DeferPriorityOperation([this, &download, range, mapped, offset, tolerant_write, unmap_sequence] {
		download.Invalidate(offset, range.size);
		m_scheduler.Context().WriteUnlessUnmapped(
		    unmap_sequence, range.address, range.size, [&](uint64_t address, uint64_t bytes) {
			    const auto* data = static_cast<const uint8_t*>(mapped) + (address - range.address);
			    if (tolerant_write) {
				    (void)LibKernel::Memory::TryWriteBacking(address, data, bytes);
			    } else {
				    LibKernel::Memory::WriteBacking(address, data, bytes);
			    }
		    });
	}, range.address, range.size);
	return true;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	// Only images starting at the address change, and every written storage-buffer bind asks:
	// the start-address filter answers most without the lock and the page walk.
	if (!GuestRange {address, size}.Valid() || !AddressFilterHit(address)) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

bool TextureCache::IsRegionGpuModified(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		// PPSA17168: S_LOAD_DWORD reads shader data at an address overlapping an old
		// render target whose memory the CPU has reused. The cached image still retains
		// its earlier GPU-modified flag.
		if (!image.depth_id && image.IsGpuModified() && !image.IsDefinitelyCpuDirty()) {
			return true;
		}
	}
	return false;
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->Overlaps(address, size)) {
			owner->MarkCpuDirty();
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner->info.data.address;
		const auto image_end   = owner->info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	return found->second.clear_mask.test(slice);
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask.set();
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	found->second.clear_mask.set(slice, is_clear);
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	BufferCache::PerfSetMemoryUsage(m_total_used_memory, m_trigger_gc_memory, 0);
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool           pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool           aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		const uint64_t age       = std::min<uint64_t>(aggressive ? 160 : pressured ? 80 : 16, tick);
		size_t         deletions = aggressive ? 40 : pressured ? 20 : 10;
		std::vector<ImageId> candidates;
		candidates.reserve(deletions);
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() == deletions;
		});
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			--deletions;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				continue;
			}
			if (owner->heap_refs.load(std::memory_order_relaxed) != 0u) {
				// Resident heaps sample it without touching it.
				TouchImage(*owner);
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				if (safe && owner->info.IsTiled()) {
					continue;
				}
				if (safe && !pressured) {
					continue;
				}
				if (safe && !DownloadImageMemory(id)) {
					continue;
				}
			}
			BufferCache::PerfCount(BufferCache::PerfEvent::ImageGcFree);
			FreeImage(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureEvictions);
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
}

void TextureCache::ProcessDownloadImages() {
	if (!m_readback_linear_images && !m_readback_tiled_images) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
