#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <string>

namespace Libs::Graphics {

struct GraphicContext;

// One frame of FSR 4 inputs that are not images (same meaning as Fsr3Frame).
struct Fsr4Params {
	float jitter_x = 0, jitter_y = 0;
	float motion_scale_x = 0, motion_scale_y = 0;
	float near_plane = 0, far_plane = 0, vertical_fov = 0;
	float delta_ms = 0, pre_exposure = 1;
	bool  reset = false;
	bool  reactive = false; // the Reactive image holds a mask for this frame
};

// FSR 4 for the Vulkan renderer (KYTY_FSR4=1, Windows, builds with KYTY_ENABLE_FSR4).
//
// AMD ships FSR 4 for DirectX 12 only, so this runs it on a small D3D12 device on the same
// graphics card. Five images are shared between the two APIs (D3D12 resources imported into
// Vulkan): colour, depth, motion and an optional reactive mask at render size, output at display size. A D3D12 fence imported
// as a Vulkan timeline semaphore orders the work on the GPU:
//   Vulkan copies the inputs in and signals `wait_value` -> D3D12 waits for it, runs FSR 4 and
//   signals `signal_value` -> the next Vulkan submission waits for that before copying the output.
// Shared images stay in VK_IMAGE_LAYOUT_GENERAL on the Vulkan side and are released to / acquired
// from VK_QUEUE_FAMILY_EXTERNAL around the D3D12 work.
class Fsr4Bridge final {
public:
	enum ImageIndex : int { Color = 0, Depth = 1, Motion = 2, Output = 3, Reactive = 4, Count = 5 };

	explicit Fsr4Bridge(GraphicContext& graphics);
	~Fsr4Bridge(); // The Vulkan owner must have finished all work that uses the shared images.
	Fsr4Bridge(const Fsr4Bridge&)            = delete;
	Fsr4Bridge& operator=(const Fsr4Bridge&) = delete;

	// False once anything failed; Status() says what. The caller then falls back to FSR 3.1.
	[[nodiscard]] bool Usable() const;
	[[nodiscard]] const std::string& Status() const;
	// True when Ensure() would recreate the shared images (the caller waits for Vulkan first).
	[[nodiscard]] bool NeedsRecreate(vk::Extent2D render, vk::Extent2D display) const;
	// Creates (or recreates) the shared images and the FSR 4 context for these sizes.
	[[nodiscard]] bool Ensure(vk::Extent2D render, vk::Extent2D display);
	// True for the first frame after Ensure() created the images (their Vulkan layout is undefined).
	[[nodiscard]] bool Fresh() const;
	void ClearFresh();

	[[nodiscard]] vk::Image Image(int index) const;
	[[nodiscard]] vk::Semaphore Semaphore() const;
	// The two timeline values for the next frame: first = Vulkan signals, second = D3D12 signals.
	[[nodiscard]] std::pair<uint64_t, uint64_t> NextValues();

	// Queues the D3D12 work: wait for wait_value, FSR 4, signal signal_value.
	enum class Result {
		NotQueued, // nothing will signal signal_value: the caller must not wait for it
		Failed,    // signal_value will be signalled, but the output holds no new frame
		Upscaled,  // signal_value will be signalled after FSR 4 wrote the output
	};
	[[nodiscard]] Result Dispatch(const Fsr4Params& params, uint64_t wait_value, uint64_t signal_value);

private:
	struct Impl;
	std::unique_ptr<Impl> m;
};

} // namespace Libs::Graphics
