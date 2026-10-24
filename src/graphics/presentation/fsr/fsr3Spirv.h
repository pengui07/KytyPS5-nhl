#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <vector>

namespace Libs::Graphics {

// SDK 1.1.4 creates RGBA16F luma history, but its GLSL callback declares rgba8. Correct only
// that named UAV's type. Other rgba8 UAVs must keep their original types. Empty = no change.
inline std::vector<uint32_t> FixFsr3LumaHistorySpirv(std::span<const uint32_t> code) {
	if (code.size() < 5 || code[0] != spv::MagicNumber)
		return {};
	uint32_t variable = 0, pointer = 0, image = 0;
	size_t image_at = 0;
	for (size_t at = 5; at < code.size();) {
		const auto size = code[at] >> 16, op = code[at] & 0xffff;
		if (!size || at + size > code.size())
			return {};
		if ((op == spv::OpVariable && size < 4) || (op == spv::OpTypePointer && size != 4) ||
			(op == spv::OpTypeImage && size < 9) || (op == spv::OpLoad && size < 4))
			return {};
		if (op == spv::OpName && size >= 3) {
			constexpr char name[] = "rw_luma_history";
			const auto* text = reinterpret_cast<const char*>(code.data() + at + 2);
			if ((size - 2) * 4 >= sizeof(name) && std::memcmp(text, name, sizeof(name)) == 0)
				variable = code[at + 1];
		}
		at += size;
	}
	if (!variable)
		return {};
	for (size_t at = 5; at < code.size(); at += code[at] >> 16)
		if ((code[at] & 0xffff) == spv::OpVariable && code[at + 2] == variable)
			pointer = code[at + 1];
	if (!pointer)
		return {};
	for (size_t at = 5; at < code.size(); at += code[at] >> 16)
		if ((code[at] & 0xffff) == spv::OpTypePointer && code[at + 1] == pointer) {
			if ((code[at] >> 16) != 4)
				return {};
			image = code[at + 3];
		}
	if (!image)
		return {};
	for (size_t at = 5; at < code.size(); at += code[at] >> 16)
		if ((code[at] & 0xffff) == spv::OpTypeImage && code[at + 1] == image)
			image_at = at;
	if (!image_at || (code[image_at] >> 16) < 9 || code[image_at + 8] != spv::ImageFormatRgba8)
		return {};
	bool shared = false;
	for (size_t at = 5; at < code.size(); at += code[at] >> 16) {
		if ((code[at] & 0xffff) != spv::OpTypePointer || code[at + 3] != image)
			continue;
		for (size_t use = 5; use < code.size(); use += code[use] >> 16)
			if ((code[use] & 0xffff) == spv::OpVariable && code[use + 1] == code[at + 1] &&
				code[use + 2] != variable)
				shared = true;
	}
	if (!shared) {
		std::vector<uint32_t> result(code.begin(), code.end());
		result[image_at + 8] = spv::ImageFormatRgba16f;
		return result;
	}
	// Refuse a shared image type: changing it could alter another UAV's format.
	return {};
}
} // namespace Libs::Graphics
