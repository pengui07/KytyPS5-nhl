#pragma once

#include "library.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace kyty_launcher {

// Trophy counts of one game. Grade index: 1 platinum, 2 gold, 3 silver, 4 bronze.
struct TrophyProgress
{
	std::array<int, 5> total_grade {};
	std::array<int, 5> earned_grade {};
	int                total  = 0;
	int                earned = 0;

	// Same as the emulator's: points (gold 6, silver 2, bronze 1; platinum not counted), in percent.
	int Percent() const;
};

// Progress of the game in `game_dir`: the trophies come from its sce_sys/trophy2/trophyNN.ucp
// packages, the unlocked ones from <run_dir>\_Trophies\<title_id>\trophies_<user_id>_<NN>.json
// (written by the emulator). False when the game has no trophy package.
bool load_trophy_progress(const std::string& game_dir, const std::string& title_id, const std::string& run_dir, int user_id,
                          TrophyProgress& out);

// One trophy of a game, for its trophy page.
struct TrophyInfo
{
	int         grade = 0; // 1 platinum .. 4 bronze
	std::string name;
	std::string detail;
	std::string reward;    // free text from the metadata, only when the package's hasReward flag is set
	std::string icon;      // PNG copied out of the package into icon_cache_dir; empty when it has none
	bool        hidden = false;
	bool        earned = false;
	std::string earned_at; // "2026-10-19 14:05" (local time), empty when unknown
};

// Every trophy of the game, texts in `console_language` (the launcher's 0..29 index), icons
// written under `icon_cache_dir`. Sorted by package, then id.
bool load_trophy_list(const std::string& game_dir, const std::string& title_id, const std::string& run_dir, int user_id,
                      int console_language, const std::string& icon_cache_dir, std::vector<TrophyInfo>& out);

// One game of an overview scan; `game_index` is the position in the scanned games list.
struct TrophyScanEntry
{
	std::size_t    game_index = 0;
	TrophyProgress progress;
};

// The overview's per-game progress, reused while the game's files are unchanged. `stamp` is the latest
// write time and the total size of the .ucp packages and of the matching unlock files; the scan
// recomputes it on its worker thread and reparses only when it differs.
struct TrophyCache
{
	struct Entry
	{
		std::string    stamp;
		TrophyProgress progress;
		bool           has_trophies = false;
	};

	std::mutex                   mutex; // the scan runs on a worker thread; the UI thread reads cache hits
	std::map<std::string, Entry> entries;
};

// Scans all games (file I/O; call it off the UI thread) and returns one entry per game that has a trophy
// package, in `games` order. Unchanged games come from `cache`; cancelled scans return nothing.
std::vector<TrophyScanEntry> scan_trophy_progress(const std::vector<GameEntry>& games, const std::string& run_dir, int user_id,
                                                  TrophyCache& cache, const std::atomic<bool>& cancel);

// The cached entries of these games, without touching any file: what the page shows while a scan runs.
std::vector<TrophyScanEntry> cached_trophy_progress(const std::vector<GameEntry>& games, const std::string& run_dir, int user_id,
                                                    TrophyCache& cache);

// Drops one game's cached progress (the emulator may have written unlocks before it exited).
void invalidate_trophy_cache(TrophyCache& cache, const std::string& game_dir, const std::string& title_id, const std::string& run_dir,
                             int user_id);

} // namespace kyty_launcher
