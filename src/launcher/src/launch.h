#pragma once

#include "paths.h"
#include "settings.h"

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kyty_launcher {

struct EnvChange
{
	enum class Mode { Set, SetIfUnset, Unset };

	std::string name;
	std::string value;
	Mode        mode = Mode::Set;
};

// Everything needed to start the emulator; built from Settings, shown by the dry run.
struct LaunchPlan
{
	std::string              exe;
	std::vector<std::string> args;
	std::string              extra_args; // appended to the command line as typed
	std::string              cwd;
	// Where the emulator's output is teed: <root>/logs/last-run.log. Empty means cwd/last-run.log (make_launch_plan
	// always fills it; make_plan alone does not know the paths).
	std::string              log_file;
	std::vector<EnvChange>   env;
	std::vector<std::string> warnings;
	std::vector<std::string> layers; // config layer files looked at, one line each (shown by the dry run)
};

LaunchPlan make_plan(const Settings& s);

// The plan for starting `user.game`: the settings resolved through the config layers (build config, user,
// per-game override of the game's title id) and then make_plan. The game's emulator pin (see
// emulator_for_game) picks the exe, and the build config is looked up next to that exe.
LaunchPlan make_launch_plan(const Settings& user, const Paths& paths);

// The emulator exe a game starts with: the pin in configs/games/<TITLEID>.json when it is set and the file exists,
// else user.emulator. A pin that is set but missing gives `warning` (optional) and falls back to user.emulator.
//
// A pin is a path. Pinning a "-latest" build (KytyPS5-main-latest) follows its updates: the folder keeps its path
// and its contents are replaced. Pinning a dated folder is exact: that release and nothing else.
std::string emulator_for_game(const Settings& user, const Paths& paths, const std::string& title_id, std::string* warning = nullptr);

using EnvList = std::vector<std::pair<std::string, std::string>>;

EnvList     current_environment();
EnvList     apply_env(const EnvList& parent, const std::vector<EnvChange>& changes);
std::string command_line(const LaunchPlan& plan);

// The command, the env changes and the resulting child env.
std::string describe(const LaunchPlan& plan);

// Runs one emulator; stdout+stderr are teed to plan.log_file (the launcher's logs folder, created as needed) or to
// <cwd>/last-run.log when the plan has none, and handed to on_output
// from a reader thread.
class ChildProcess
{
public:
	using OutputFn = std::function<void(std::string chunk)>;
	using ExitFn   = std::function<void(int code)>;

	ChildProcess() = default;
	~ChildProcess();
	ChildProcess(const ChildProcess&)            = delete;
	ChildProcess& operator=(const ChildProcess&) = delete;

	bool Start(const LaunchPlan& plan, OutputFn on_output, ExitFn on_exit, std::string* error);
	void Stop();
	void Wait(); // until the running emulator exits
	bool Running() const;

private:
	struct Impl;
	std::shared_ptr<Impl> m_impl;
	std::thread           m_reader;
};

} // namespace kyty_launcher
