#pragma once

#include "app.h"
#include "settings.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace kyty_launcher {

// The emulator builds the launcher can run: KytyPS5 GitHub releases unpacked into a
// versions folder, custom exes and the configured emulator. Drives the VM global of
// ui/version_manager.slint and the build dropdown in the main window. The selected build
// is Settings::emulator.
class VersionManager
{
public:
	using SaveFn = std::function<bool()>;
	// A path chosen by the user, "" when cancelled. `folder` picks a folder, otherwise an emulator exe.
	using PickFn = std::function<std::string(const std::string& current, bool folder)>;

	VersionManager(const slint::ComponentHandle<AppWindow>& ui, Settings& settings, std::string default_versions_dir, std::string cache_dir, SaveFn save,
	               PickFn pick);
	// Stops the downloads still running and waits for their threads.
	~VersionManager();
	VersionManager(const VersionManager&)            = delete;
	VersionManager& operator=(const VersionManager&) = delete;

	// For the per-game emulator pins (Settings > Games).
	// The name shown for the build with this exe: its Rename, else its folder name (also for a build that is
	// not in the installed list any more).
	std::string BuildName(const std::string& exe) const;
	// How many library games are pinned to any of these exes; the Remove dialog says so. Pins are never cleared.
	using PinCountFn = std::function<int(const std::vector<std::string>& exes)>;
	void SetPinCounter(PinCountFn fn);
	// Called whenever the installed list is rebuilt (scan, add, remove, rename).
	void SetOnBuildsChanged(std::function<void()> fn);

private:
	struct Impl;
	std::shared_ptr<Impl> m_impl;
};

} // namespace kyty_launcher
