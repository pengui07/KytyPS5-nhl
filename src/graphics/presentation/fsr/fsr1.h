#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <memory>

namespace Libs::Graphics {

// One instance per in-flight presentation slot. The owner must wait for its GPU work before
// destruction or a size change. Color is SDR, already antialiased, in perceptual space.
class Fsr1 final {
public:
	Fsr1(vk::PhysicalDevice physical, vk::Device device);
	~Fsr1();
	Fsr1(const Fsr1&) = delete;
	Fsr1& operator=(const Fsr1&) = delete;
	[[nodiscard]] vk::Image Record(vk::CommandBuffer command, vk::ImageView source,
								   vk::Extent2D input, vk::Extent2D output, float sharpness);

private:
	struct Impl;
	std::unique_ptr<Impl> m;
};

} // namespace Libs::Graphics
