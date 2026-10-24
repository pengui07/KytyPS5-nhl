#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace Libs::Graphics {

// Frostbite's captured 208-byte block, distinct from the embedded SDK FSR2 layout.
// Only the verified layout/format combination is accepted. Unknown tuning words
// and pass-specific tail words are retained without assigning semantics.
struct Nhl27Fsr3Constants {
	std::array<uint32_t, 52> words {};
	float Float(size_t i) const { return std::bit_cast<float>(words[i]); }
	uint32_t Width() const { return words[4]; }
	uint32_t Height() const { return words[5]; }
	uint32_t OutputWidth() const { return words[8]; }
	uint32_t OutputHeight() const { return words[9]; }
	uint32_t Frame() const { return words[39]; }
	float Near() const { return Float(1) / (1 - Float(0)); }
	float Far() const { return Float(1) / -Float(0); }
	float FovY() const { return 2 * std::atan(Float(3)); }
	bool SameFrame(const Nhl27Fsr3Constants& other) const {
		for (size_t i = 0; i < 49; ++i)
			if (words[i] != other.words[i]) return false;
		return true;
	}
	// Names the first check Decode() fails (diagnostics), or nullptr when it would accept the block.
	static const char* RejectReason(std::span<const uint32_t> source) {
		if (source.size() != 52) return "size";
		Nhl27Fsr3Constants c;
		std::copy(source.begin(), source.end(), c.words.begin());
		if (c.Width() < 320 || c.Height() < 180 || c.OutputWidth() < c.Width() ||
			c.OutputHeight() < c.Height() || c.OutputWidth() > 8192 || c.OutputHeight() > 8192)
			return "dimensions";
		for (size_t i : {size_t(4), size_t(5), size_t(8), size_t(9)})
			if (c.words[i] != c.words[i + 2]) return "dynamic-size";
		for (size_t i = 0; i < 4; ++i) {
			const size_t offset = 12 + i * 2;
			const float w = float(c.words[4 + i * 2]), h = float(c.words[5 + i * 2]);
			if (!std::isfinite(c.Float(offset + 8)) || !std::isfinite(c.Float(offset + 9)) ||
				c.Float(offset) != w || c.Float(offset + 1) != h ||
				std::abs(c.Float(offset + 8) * w - 1) > 1e-5f ||
				std::abs(c.Float(offset + 9) * h - 1) > 1e-5f)
				return "size-floats";
		}
		for (size_t i : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(28), size_t(29),
						size_t(30), size_t(31), size_t(32), size_t(33), size_t(34), size_t(35),
						size_t(36), size_t(37), size_t(38)})
			if (!std::isfinite(c.Float(i))) return "nonfinite";
		if (c.Float(0) >= 0 || c.Float(1) <= 0 || c.Float(2) <= 0 || c.Float(3) <= 0) return "depth-or-fov-sign";
		if (!std::isfinite(c.Near()) || !std::isfinite(c.Far()) || c.Near() <= 0 || c.Far() <= c.Near())
			return "near-far";
		if (c.FovY() >= 3.141593f) return "fov";
		if (c.Float(32) != -0.5f || c.Float(33) != 0.5f) return "motion-scale";
		if (c.Float(36) < 0 || c.Float(36) > 1) return "delta-time";
		if (c.Float(37) <= 0) return "pre-exposure";
		if (c.Float(38) <= 0) return "word38";
		for (size_t i = 28; i < 32; ++i)
			if (std::abs(c.Float(i)) > 0.50001f) return "jitter";
		if (std::abs(c.Float(34) - float(c.Width()) / c.OutputWidth()) > 1e-5f ||
			std::abs(c.Float(35) - float(c.Height()) / c.OutputHeight()) > 1e-5f)
			return "downscale";
		return nullptr;
	}
	// Word 36 (frame time, seconds) is 0 on some frames (player selection does it about once a
	// second); those frames are accepted and the caller substitutes the last real frame time.
	static std::optional<Nhl27Fsr3Constants> Decode(std::span<const uint32_t> source) {
		if (source.size() != 52) return std::nullopt;
		Nhl27Fsr3Constants c;
		std::copy(source.begin(), source.end(), c.words.begin());
		if (c.Width() < 320 || c.Height() < 180 || c.OutputWidth() < c.Width() ||
			c.OutputHeight() < c.Height() || c.OutputWidth() > 8192 || c.OutputHeight() > 8192)
			return std::nullopt;
		// Fail closed for dynamic-resource layouts not yet observed.
		for (size_t i : {size_t(4), size_t(5), size_t(8), size_t(9)})
			if (c.words[i] != c.words[i + 2]) return std::nullopt;
		for (size_t i = 0; i < 4; ++i) {
			const size_t offset = 12 + i * 2;
			const float w = float(c.words[4 + i * 2]), h = float(c.words[5 + i * 2]);
			if (!std::isfinite(c.Float(offset + 8)) || !std::isfinite(c.Float(offset + 9)) ||
				c.Float(offset) != w || c.Float(offset + 1) != h ||
				std::abs(c.Float(offset + 8) * w - 1) > 1e-5f ||
				std::abs(c.Float(offset + 9) * h - 1) > 1e-5f)
				return std::nullopt;
		}
		for (size_t i : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(28), size_t(29),
						size_t(30), size_t(31), size_t(32), size_t(33), size_t(34), size_t(35),
						size_t(36), size_t(37), size_t(38)})
			if (!std::isfinite(c.Float(i))) return std::nullopt;
		if (c.Float(0) >= 0 || c.Float(1) <= 0 || c.Float(2) <= 0 || c.Float(3) <= 0 ||
			!std::isfinite(c.Near()) || !std::isfinite(c.Far()) || c.Near() <= 0 ||
			c.Far() <= c.Near() || c.FovY() >= 3.141593f ||
			c.Float(32) != -0.5f || c.Float(33) != 0.5f ||
			c.Float(36) < 0 || c.Float(36) > 1 || c.Float(37) <= 0 || c.Float(38) <= 0)
			return std::nullopt;
		for (size_t i = 28; i < 32; ++i)
			if (std::abs(c.Float(i)) > 0.50001f) return std::nullopt;
		if (std::abs(c.Float(34) - float(c.Width()) / c.OutputWidth()) > 1e-5f ||
			std::abs(c.Float(35) - float(c.Height()) / c.OutputHeight()) > 1e-5f)
			return std::nullopt;
		return c;
	}
};

// Per-dispatch temporal summary used to decide when FSR history is still valid.
struct Nhl27HistoryState {
	uint32_t frame = 0, width = 0, height = 0, output_width = 0, output_height = 0;
	float near_plane = 0, far_plane = 0, fov = 0;
	float jitter_x = 0, jitter_y = 0, prev_jitter_x = 0, prev_jitter_y = 0;
	static Nhl27HistoryState From(const Nhl27Fsr3Constants& c) {
		return {c.Frame(), c.Width(), c.Height(), c.OutputWidth(), c.OutputHeight(),
				c.Near(), c.Far(), c.FovY(), c.Float(28), c.Float(29), c.Float(30), c.Float(31)};
	}
};
enum class Nhl27ResetReason {
	None, Disabled, First, MissedPass, FrameDiscontinuity, SizeChange, CameraCut, JitterMismatch
};
inline const char* Nhl27ResetReasonName(Nhl27ResetReason r) {
	switch (r) {
		case Nhl27ResetReason::None: return "none";
		case Nhl27ResetReason::Disabled: return "history-disabled";
		case Nhl27ResetReason::First: return "first-frame";
		case Nhl27ResetReason::MissedPass: return "missed-pass";
		case Nhl27ResetReason::FrameDiscontinuity: return "frame-counter-discontinuity";
		case Nhl27ResetReason::SizeChange: return "size-change";
		case Nhl27ResetReason::CameraCut: return "camera-cut";
		case Nhl27ResetReason::JitterMismatch: return "previous-jitter-mismatch";
	}
	return "unknown";
}
// The captured cb carries no reset/cut flag (words 40..48 are constant across captures), so
// continuity is inferred. `prev` is null when no frame was dispatched yet; `missed` is set when
// any pass of the 4-pass chain was rejected/skipped since the last dispatch.
inline Nhl27ResetReason ShouldResetHistory(const Nhl27HistoryState* prev, const Nhl27HistoryState& cur,
										   bool missed, bool history_enabled = true) {
	if (!history_enabled) return Nhl27ResetReason::Disabled;
	if (!prev) return Nhl27ResetReason::First;
	if (missed) return Nhl27ResetReason::MissedPass;
	if (cur.frame - prev->frame != 1u) return Nhl27ResetReason::FrameDiscontinuity;
	if (cur.width != prev->width || cur.height != prev->height ||
		cur.output_width != prev->output_width || cur.output_height != prev->output_height)
		return Nhl27ResetReason::SizeChange;
	auto far_apart = [](float a, float b, float tolerance) {
		return !(std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b)));
	};
	if (far_apart(cur.near_plane, prev->near_plane, 0.01f) ||
		far_apart(cur.far_plane, prev->far_plane, 0.01f) || far_apart(cur.fov, prev->fov, 0.02f))
		return Nhl27ResetReason::CameraCut;
	if (!(std::abs(cur.prev_jitter_x - prev->jitter_x) <= 1e-4f) ||
		!(std::abs(cur.prev_jitter_y - prev->jitter_y) <= 1e-4f))
		return Nhl27ResetReason::JitterMismatch;
	return Nhl27ResetReason::None;
}
} // namespace Libs::Graphics
