#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <memory>

namespace Libs::Graphics {

struct Fsr3Image {
	vk::Image image;
	vk::Format format = vk::Format::eUndefined;
	vk::Extent2D extent {};
};

// Inputs must be snapshots from one guest frame, in shader-read layout. Color is linear;
// SDR uses UNORM snapshots, HDR uses floating-point color/output and KYTY_FSR3_HDR=1.
// The provider owns color-space conversion before reconstruction and after the output.
// Outputs must be
// non-aliasing storage images in GENERAL. The SDK returns them in those same states.
// No synthetic camera or motion fallback is permitted in the game integration.
struct Fsr3Frame {
	Fsr3Image color, depth, motion;
	Fsr3Image output, dilated_depth, dilated_motion, previous_depth;
	vk::Extent2D render_size {}, output_size {};
	uint64_t frame_id = 0;
	float jitter_x = 0, jitter_y = 0;
	float motion_scale_x = 0, motion_scale_y = 0;
	float near_plane = 0, far_plane = 0, vertical_fov = 0;
	float delta_ms = 0, pre_exposure = 1, sharpness = 0.2f;
	bool verified = false, reset = false;
	bool inverted_depth = false, infinite_depth = false, jittered_motion = false;
	bool hdr = false;
};

class Fsr3 final {
public:
	Fsr3(vk::PhysicalDevice physical, vk::Device device);
	~Fsr3(); // Owner must wait for submitted GPU work first.
	Fsr3(const Fsr3&) = delete;
	Fsr3& operator=(const Fsr3&) = delete;
	// A size/configuration change requires a new instance after the old work completes.
	[[nodiscard]] bool Record(vk::CommandBuffer command, const Fsr3Frame& frame);
	void ResetHistory();
	[[nodiscard]] const std::string& Status() const;

private:
	struct Impl;
	std::unique_ptr<Impl> m;
};
} // namespace Libs::Graphics
