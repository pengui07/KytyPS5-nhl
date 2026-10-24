#pragma once

#include "paths.h"
#include "settings.h"

#include <string>
#include <vector>

namespace kyty_launcher {

// What an import of the Qt launcher's Kyty.ini did; text() is the summary shown to the user.
struct ImportReport
{
	bool                     ok = false; // the file was read and imported
	std::string              source;     // the Kyty.ini
	std::string              error;      // why not, when !ok
	int                      settings = 0; // launcher + emulator settings applied
	std::vector<std::string> games;        // one line per game that got a configs\games file
	std::vector<std::string> skipped;      // "what: reason" for values and games that could not be imported
	std::vector<std::string> not_imported; // things the Qt launcher stored that have no equivalent here

	std::string text() const;
};

// The Qt launcher's settings file, probed in order: Kyty.ini in the current folder, next to the launcher exe, in its
// parent folder; then the system/user location QSettings used (%ProgramData%\Kyty\Kyty.ini on Windows,
// ~/.config/Kyty on Linux and macOS). "" when none exists.
std::string find_qt_ini(const Paths& paths);

// Reads `ini_path` and applies it to `s` (only the keys the file holds change) and writes configs\games\<TITLEID>.json for
// each game with custom settings (only the keys that differ from the imported global settings). Never touches the
// Kyty.ini. The caller saves `s`. `first_run`: the emulator and run folder next to the Kyty.ini are always taken; otherwise
// only when no emulator is set (or the one set is missing).
ImportReport import_qt_ini(const std::string& ini_path, const Paths& paths, Settings& s, bool first_run);

} // namespace kyty_launcher
