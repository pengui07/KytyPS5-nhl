#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>

namespace Libs::Graphics {

// Candidate decoder only. Layout supplied from NHL's FSR2 reflection tables; a plausible match
// does not certify a shader, resource slot, depth convention or a same-frame snapshot.
struct Fsr2ConstantsCandidate {
	std::array<uint32_t, 32> words {};
	[[nodiscard]] float Float(unsigned byte_offset) const {
		return std::bit_cast<float>(words[byte_offset / 4]);
	}
	[[nodiscard]] bool Plausible() const {
		auto dimension = [](uint32_t x) { return x >= 32 && x <= 16384; };
		for (unsigned i = 0; i < 8; ++i)
			if (!dimension(words[i]))
				return false;
		if (words[0] > words[2] || words[1] > words[3] || words[0] > words[4] ||
			words[1] > words[5] || words[0] > words[6] || words[1] > words[7] || words[10] > 16)
			return false;
		for (unsigned offset = 48; offset <= 120; offset += 4)
			if (!std::isfinite(Float(offset)))
				return false;
		if (std::abs(Float(64)) > 2 || std::abs(Float(68)) > 2 || Float(72) == 0 ||
			Float(76) == 0 || std::abs(Float(72)) > 32768 || std::abs(Float(76)) > 32768 ||
			Float(96) <= 0 || Float(100) <= 0 || Float(104) <= 0.01f || Float(104) > 100 ||
			Float(108) < 1 || Float(108) > 1024 || Float(112) <= 0 || Float(112) > 10 ||
			Float(120) <= 0)
			return false;
		return std::abs(Float(80) - float(words[0]) / float(words[4])) < 0.02f &&
			   std::abs(Float(84) - float(words[1]) / float(words[5])) < 0.02f;
	}
};
static_assert(sizeof(Fsr2ConstantsCandidate) == 128);

// KYTY_FSR2_CAPTURE=1 (NHL 27 PPSA34063 v01.003.001): the guest's AMD FSR2 library keeps the
// constants it uploads for every pass in globalFsr2ConstantBuffers (eboot+0xF01AA80, IDA
// analysis of fsr2Dispatch); its first entry is { uint32Size = 31, Fsr2Constants data }. The
// "cbFSR2" reflection string at eboot+0xDC16F76 identifies the build before anything is read.
namespace Fsr2Capture {
inline std::atomic<uint64_t> g_eboot_base {0};
inline void                  SetEbootBase(uint64_t base) { g_eboot_base.store(base, std::memory_order_release); }
constexpr uint64_t           GlobalConstantsOffset = 0xF01AA80;
constexpr uint64_t           CbNameOffset          = 0xDC16F76;
// The FSR2 shader blobs embedded in the eboot (fsr2_*_blob / blobRelOffsets targets).
constexpr uint64_t           BlobRangeBegin        = 0xE184530;
constexpr uint64_t           BlobRangeEnd          = 0xE1B0000;
} // namespace Fsr2Capture

} // namespace Libs::Graphics
