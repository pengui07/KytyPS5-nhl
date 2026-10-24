// Thin Slint launcher for KytyPS5: shows the main settings, starts the emulator and tails its output.
//
//   kyty_slint_launcher.exe                  open the window
//   kyty_slint_launcher.exe --dry-run [file] write the command and child env (no window)
//   kyty_slint_launcher.exe --play           start the emulator with the saved settings (no window)

#include "app.h"
#include "input_mapping.h" // STEP6:6f: the Controls page's bindings and capture
#include "launch.h"
#include "library.h"
#include "paths.h"
#include "settings.h"
#include "platform.h"
#include "qt_import.h"
#include "trophies.h"
#include "version_manager.h"

#include <atomic>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <condition_variable> // STEP6:6c: the image decoder pool
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque> // STEP6:6c
#include <filesystem>
#include <fstream>
#include <functional>
#include <list> // STEP6:6c: the icon LRU
#include <map>
#include <memory>
#include <mutex> // STEP6:6c
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <shobjidl.h>
#else
#include <unistd.h>
#endif

using namespace kyty_launcher;
namespace fs = std::filesystem;

namespace {

constexpr size_t kMaxLogLines = 4000;

std::string ToStd(const slint::SharedString& s)
{
	return std::string(std::string_view(s));
}

#ifdef _WIN32
// Modern Explorer dialog; returns "" when cancelled.
std::string PickPath(const std::string& current, bool folder, const wchar_t* filter_name, const wchar_t* filter_spec)
{
	std::string      result;
	HRESULT          co   = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	IFileOpenDialog* dlg  = nullptr;
	if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg))))
	{
		FILEOPENDIALOGOPTIONS opts = 0;
		dlg->GetOptions(&opts);
		dlg->SetOptions(opts | FOS_FORCEFILESYSTEM | (folder ? FOS_PICKFOLDERS : 0));
		if (!folder && filter_name != nullptr)
		{
			COMDLG_FILTERSPEC spec[] = {{filter_name, filter_spec}, {L"All files", L"*.*"}};
			dlg->SetFileTypes(2, spec);
		}
		fs::path        cur = utf8_path(current);
		std::error_code ec;
		fs::path        start = folder ? cur : cur.parent_path();
		if (!start.empty() && fs::is_directory(start, ec))
		{
			IShellItem* item = nullptr;
			if (SUCCEEDED(SHCreateItemFromParsingName(start.wstring().c_str(), nullptr, IID_PPV_ARGS(&item))))
			{
				dlg->SetFolder(item);
				item->Release();
			}
		}
		if (SUCCEEDED(dlg->Show(GetActiveWindow())))
		{
			IShellItem* item = nullptr;
			if (SUCCEEDED(dlg->GetResult(&item)))
			{
				PWSTR name = nullptr;
				if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &name)))
				{
					result = path_utf8(fs::path(name));
					CoTaskMemFree(name);
				}
				item->Release();
			}
		}
		dlg->Release();
	}
	if (SUCCEEDED(co))
	{
		CoUninitialize();
	}
	return result;
}
#else
std::string PickPath(const std::string& current, bool folder, const wchar_t*, const wchar_t*)
{
	return pick_path_dialog(current, folder);
}
#endif

// "#rrggbb" or "rrggbb" (any case) as "#rrggbb"; "" when it is not a colour.
std::string NormalizeHexColor(std::string text)
{
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
	{
		text.erase(text.begin());
	}
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
	{
		text.pop_back();
	}
	if (!text.empty() && text[0] == '#')
	{
		text.erase(text.begin());
	}
	if (text.size() != 6 || text.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
	{
		return {};
	}
	std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return "#" + text;
}

// The swatch next to the lightbar field: the colour typed, or transparent while it is not a colour.
slint::Color ParseHexColor(const std::string& text)
{
	const std::string hex = NormalizeHexColor(text);
	if (hex.empty())
	{
		return slint::Color::from_argb_uint8(0, 0, 0, 0);
	}
	const unsigned long v = std::strtoul(hex.c_str() + 1, nullptr, 16);
	return slint::Color::from_rgb_uint8(static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v));
}

// STEP6:6c: a small fixed-size worker pool. Images are decoded here, never on the UI thread, and the pool
// keeps its threads for the whole session: `bg_threads` used to add one thread per image and join them only
// at exit. Jobs that are still queued when the pool stops are dropped.
class WorkPool
{
public:
	explicit WorkPool(size_t threads)
	{
		for (size_t i = 0; i < threads; i++)
		{
			workers_.emplace_back([this] { Run(); });
		}
	}
	~WorkPool() { Stop(); }
	WorkPool(const WorkPool&)            = delete;
	WorkPool& operator=(const WorkPool&) = delete;

	void Post(std::function<void()> job)
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (stopping_)
			{
				return;
			}
			jobs_.push_back(std::move(job));
		}
		wake_.notify_one();
	}

	// Stops taking work and joins the workers; safe to call twice.
	void Stop()
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stopping_ = true;
		}
		wake_.notify_all();
		for (auto& t: workers_)
		{
			if (t.joinable())
			{
				t.join();
			}
		}
		workers_.clear();
	}

private:
	void Run()
	{
		for (;;)
		{
			std::function<void()> job;
			{
				std::unique_lock<std::mutex> lock(mutex_);
				wake_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
				if (stopping_)
				{
					return;
				}
				job = std::move(jobs_.front());
				jobs_.pop_front();
			}
			job();
		}
	}

	std::mutex                        mutex_;
	std::condition_variable           wake_;
	std::deque<std::function<void()>> jobs_;
	std::vector<std::thread>          workers_;
	bool                              stopping_ = false;
};

// STEP6:6c: the decoded icons kept in memory, least recently used one evicted past `capacity`. Only touched
// on the UI thread (requests and completions are posted there), so it needs no lock.
class IconCache
{
public:
	explicit IconCache(size_t capacity): capacity_(capacity) {}

	// The cached image, or nullptr. A hit moves the icon to the front of the eviction order.
	const slint::Image* Get(const std::string& path)
	{
		auto it = slots_.find(path);
		if (it == slots_.end())
		{
			return nullptr;
		}
		order_.splice(order_.begin(), order_, it->second.second);
		return &it->second.first;
	}

	bool Contains(const std::string& path) const { return slots_.find(path) != slots_.end(); }

	void Put(const std::string& path, slint::Image image)
	{
		if (auto it = slots_.find(path); it != slots_.end())
		{
			it->second.first = std::move(image);
			order_.splice(order_.begin(), order_, it->second.second);
			return;
		}
		order_.push_front(path);
		slots_.emplace(path, std::make_pair(std::move(image), order_.begin()));
		while (slots_.size() > capacity_)
		{
			slots_.erase(order_.back());
			order_.pop_back();
		}
	}

private:
	size_t capacity_;
	std::list<std::string> order_; // front = most recently used
	std::map<std::string, std::pair<slint::Image, std::list<std::string>::iterator>> slots_;
};

// Icons are decoded at this edge: sharp in the 46 px list row and in the 192 px
// grid card, never full size.
constexpr int kIconPixels = 256;
// Play time only counts a session at least this long: a game that crashes while booting is not play time.
constexpr std::chrono::seconds kMinPlaySession{10};

// STEP6:6c: builds one library row. The icon is never decoded here: the row starts with an empty placeholder
// and `request_icon` asks the decoder pool for it (the completion patches that one row in the model).
GameRow MakeRow(const GameEntry& g, const std::set<std::string>& pinned_keys,
                const std::function<void(const std::string&)>& request_icon = {})
{
	GameRow r;
	r.name         = slint::SharedString(g.name);
	r.serial       = slint::SharedString(g.title_id);
	r.game_version = slint::SharedString(g.game_version);
	r.firmware     = slint::SharedString(g.firmware_version);
	r.size         = slint::SharedString(format_size(g.size));
	r.path         = slint::SharedString(g.dir);
	r.status       = slint::SharedString(status_text(g.status));
	r.status_code  = static_cast<int>(g.status);
	r.comment      = slint::SharedString(g.comment);
	r.play_time    = slint::SharedString(format_play_time(g.play_seconds));
	r.pinned       = pinned_keys.count(favourite_key(g)) != 0;
	if (!g.icon.empty() && request_icon)
	{
		request_icon(g.icon);
	}
	return r;
}

// STEP6:6e: helper for the environment table and the .zar entries.
std::string TrimText(const std::string& s)
{
	size_t b = 0;
	size_t e = s.size();
	while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0)
	{
		b++;
	}
	while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0)
	{
		e--;
	}
	return s.substr(b, e - b);
}

// A .zar archive (the emulator's ZArchive format), whatever the case of the extension.
bool IsArchivePath(const std::string& path)
{
	std::string ext = path_utf8(utf8_path(path).extension());
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return ext == ".zar";
}

// Whether `child` is `parent` or sits inside it; both must be canonical paths.
bool PathIsUnder(const fs::path& child, const fs::path& parent)
{
	auto c = child.begin();
	for (auto p = parent.begin(); p != parent.end(); ++p, ++c)
	{
		if (c == child.end() || *c != *p)
		{
			return false;
		}
	}
	return true;
}

// STEP7:1: the build the selected game is pinned to, for the bottom bar (the pin saved in
// configs/games/<TITLEID>.json). Empty when the game has no pin: the bar says nothing, since the library's own
// build is already shown by the toolbar's build list. `missing` says the pinned exe is gone (shown in red).
void UpdateSelectedEmu(const slint::ComponentHandle<AppWindow>& ui, const Settings& s, const Paths& paths, const std::vector<GameEntry>& games)
{
	std::string name;
	bool        missing = false;
	const int   index   = ui->get_selected();
	if (index >= 0 && static_cast<size_t>(index) < games.size())
	{
		const std::string& title_id = games[static_cast<size_t>(index)].title_id;
		if (!title_id.empty())
		{
			if (const std::string pin = read_game_emulator(paths, title_id); !pin.empty())
			{
				name = build_display_name(s, pin);
				std::error_code ec;
				missing = !fs::exists(utf8_path(pin), ec);
			}
		}
	}
	ui->set_selected_emu(slint::SharedString(name));
	ui->set_selected_emu_missing(missing);
}

// STEP7:2: one row of Settings > Games' "Removed games" list. The saved key is a title id, or the start file
// (eboot.bin, a .bin/.elf or a .zar) of a game without one; the label is what the user recognises, the detail
// the saved key itself.
RemovedGameRow RemovedRow(const std::string& key)
{
	RemovedGameRow row;
	row.label  = slint::SharedString(key);
	row.detail = slint::SharedString(key);
	// A path (drive letter or separator) rather than a title id.
	if (key.find('/') == std::string::npos && key.find('\\') == std::string::npos && key.find(':') == std::string::npos)
	{
		return row;
	}
	const fs::path path = utf8_path(key);
	// A game without a title id is saved by its start file, so its name is the folder's; an archive is the game.
	const fs::path label = IsArchivePath(key) ? path.filename() : path.parent_path().filename();
	if (!label.empty())
	{
		row.label = slint::SharedString(path_utf8(label));
	}
	return row;
}

// True when a file can be created in `dir` right now; the probe file is removed again. The first-launch setup
// uses it to turn the portable choice off (with the reason) instead of failing when Continue is pressed.
bool folder_writable(const fs::path& dir)
{
	std::error_code ec;
	if (!fs::is_directory(dir, ec))
	{
		return false;
	}
	const fs::path probe = dir / ".kytyps5-write-probe";
	{
		std::ofstream f(probe, std::ios::binary | std::ios::trunc);
		if (!f)
		{
			return false;
		}
		f << "probe";
		if (!f)
		{
			f.close();
			fs::remove(probe, ec);
			return false;
		}
	}
	fs::remove(probe, ec);
	return true;
}

// Two folder strings naming the same folder: an empty run dir means the default one, so it has to be told apart.
bool same_path(const std::string& a, const std::string& b)
{
	if (a.empty() || b.empty())
	{
		return false;
	}
	return utf8_path(a).lexically_normal() == utf8_path(b).lexically_normal();
}

#ifdef _WIN32
constexpr const char* kEmulatorExe = "kyty_emulator.exe";
#else
constexpr const char* kEmulatorExe = "kyty_emulator";
#endif

// What the first-launch setup wrote down; `note` says what went wrong, if anything.
struct FirstRunOutcome
{
	bool         portable = false; // portable.txt was written: the data root is the launcher's folder now
	bool         imported = false;
	ImportReport import;
	std::string  note;
};

// The first launch: ask where the data root and the saves go, and whether to bring over the Qt launcher's settings,
// then write the answers down so the setup never shows again. `paths` arrives as the per-user root and leaves as the
// root the answers imply. Nothing is written before Continue or the window being closed.
FirstRunOutcome run_first_run_setup(const char* argv0, Paths& paths)
{
	FirstRunOutcome   out;
	const fs::path    portable_dir = portable_root(argv0);
	const std::string ini          = find_qt_ini(paths);

	// The Qt launcher's emulator folder: where its _SaveData and _Trophies already are.
	fs::path        qt_run_dir;
	std::error_code ec;
	if (!ini.empty())
	{
		const fs::path dir = utf8_path(ini).parent_path();
		if (fs::exists(dir / kEmulatorExe, ec))
		{
			qt_run_dir = dir;
		}
	}

	const bool portable_ok = folder_writable(portable_dir);

	auto setup = FirstRunWindow::create();
	setup->set_app_data_root(slint::SharedString(path_utf8(paths.root)));
	setup->set_portable_root(slint::SharedString(path_utf8(portable_dir)));
	setup->set_portable_enabled(portable_ok);
	setup->set_portable_disabled_reason(slint::SharedString(portable_ok ? std::string() : "This folder can't be written to."));
	setup->set_portable_selected(false);
	setup->set_app_data_run_dir(slint::SharedString(path_utf8(paths.run_dir)));
	setup->set_portable_run_dir(slint::SharedString(path_utf8(portable_dir / "run")));
	// No Kyty.ini: no import checkbox, and no Qt folder to offer for the saves.
	setup->set_qt_found(!ini.empty());
	setup->set_qt_ini_path(slint::SharedString(ini));
	setup->set_qt_run_dir(slint::SharedString(qt_run_dir.empty() ? std::string() : path_utf8(qt_run_dir)));
	setup->set_qt_run_dir_available(!qt_run_dir.empty());
	// Those users' saves are already in the Qt launcher's folder, so that is the preselected answer for them.
	setup->set_run_choice(qt_run_dir.empty() ? 0 : 1);
	setup->set_picked_run_dir(slint::SharedString(std::string()));
	setup->set_import_qt(!ini.empty());
	setup->on_pick_run_dir(
	    [&] {
		    const std::string picked = PickPath(ToStd(setup->get_chosen_run_dir()), true, nullptr, nullptr);
		    if (!picked.empty())
		    {
			    setup->set_picked_run_dir(slint::SharedString(picked));
			    setup->set_run_choice(2);
		    }
	    });
	setup->on_continue_clicked([&] { setup->hide(); });
	// Closing the window ends this too: what it shows is the answer either way.
	setup->run();

	const bool        portable_chosen = setup->get_portable_selected() && portable_ok;
	const std::string chosen_run      = ToStd(setup->get_chosen_run_dir());
	const bool        do_import       = setup->get_import_qt() && !ini.empty();

	// Portable: the marker is what compute_paths looks for, so it is written before the paths are recomputed.
	if (portable_chosen)
	{
		std::ofstream marker(portable_dir / "portable.txt", std::ios::binary | std::ios::trunc);
		if (marker)
		{
			marker << "KytyPS5 keeps its files in this folder.\n";
			out.portable = marker.good();
		}
		if (!out.portable)
		{
			out.note = "Could not write " + path_utf8(portable_dir / "portable.txt") + ": the app data folder is used";
		}
	}
	paths = compute_paths(argv0);

	Settings s = default_settings();
	if (do_import)
	{
		out.import   = import_qt_ini(ini, paths, s, true);
		out.imported = out.import.ok;
		if (!out.import.ok && out.note.empty())
		{
			out.note = out.import.text() + " (defaults kept)";
		}
	}
	// An empty run dir means the default (root/run), as everywhere else; anything else is a folder the user chose.
	s.run_dir = chosen_run.empty() || same_path(chosen_run, path_utf8(paths.run_dir)) ? std::string() : chosen_run;
	if (!save_settings(paths.settings_file(), s) && out.note.empty())
	{
		out.note = "Could not save " + path_utf8(paths.settings_file());
	}
	return out;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
	// Slint menus render styled (like the launcher) instead of native: the menu bar then
	// follows slint-style/fluent-menu.slint (uniform #303030 with the toolbar) and the
	// right-click game menu uses our popup styling. Must precede any Slint backend use.
	SetEnvironmentVariableW(L"SLINT_NO_MUDA", L"1");
#endif
	const char* const argv0 = argc > 0 && argv[0] != nullptr ? argv[0] : "";
	// Every launcher file lives under these folders (see paths.h); they are created when first written.
	Paths       paths = compute_paths(argv0);
	Settings    s     = default_settings();
	std::string startup_note; // shown in the status line and the console, and by --dry-run

	// First launch: no portable.txt beside the exe and no kyty.json in the per-user root. The windowed launcher asks
	// where the data and the saves go before it reads anything; --dry-run and --play stay unattended, as before.
	const bool      headless = argc >= 2 && (std::string(argv[1]) == "--dry-run" || std::string(argv[1]) == "--play");
	std::error_code first_ec;
	const bool      first_start = !paths.portable && !fs::exists(paths.settings_file(), first_ec);
	ImportReport    import_report;
	bool            imported = false;
	std::string     setup_note;
	if (!headless && first_start)
	{
		const FirstRunOutcome first = run_first_run_setup(argv0, paths);
		import_report                = first.import;
		imported                     = first.imported;
		setup_note                   = first.note;
	}
	const fs::path settings_path = paths.settings_file();
	const bool     had_settings  = load_settings(settings_path, s, &startup_note);
	if (!setup_note.empty())
	{
		// The setup's own problem (a folder it could not write) outranks what loading the settings says.
		startup_note = setup_note;
	}
	// The monitor: its size is the recommended window size, and the first run starts with it.
	int monitor_w = 0;
	int monitor_h = 0;
	const bool have_monitor = detect_primary_monitor(&monitor_w, &monitor_h);
	if (have_monitor && (!had_settings || first_start))
	{
		s.width  = monitor_w;
		s.height = monitor_h;
	}
	{
		// No machine-specific defaults here: the run folder is under the data root, and the emulator sits next
		// to the launcher (or, on Windows, in the downloaded main build).
		const fs::path here = paths.exe_dir;
		if (s.run_dir.empty())
		{
			s.run_dir = path_utf8(paths.run_dir);
		}
		std::error_code ec;
#ifdef _WIN32
		const char* const exe_name = "kyty_emulator.exe";
#else
		const char* const exe_name = "kyty_emulator";
#endif
		if (s.emulator.empty() && fs::exists(here / exe_name, ec))
		{
			s.emulator = path_utf8(here / exe_name);
		}
#ifdef _WIN32
		// Same folder the Version Manager downloads the newest main build to.
		const fs::path latest = (s.versions_dir.empty() ? paths.versions_dir : utf8_path(s.versions_dir)) / "KytyPS5-main-latest" / exe_name;
		if (s.emulator.empty() && fs::exists(latest, ec))
		{
			s.emulator = path_utf8(latest);
		}
#endif
	}

	// First run with no kyty.json yet, and nothing was asked (--dry-run, --play): bring over the Qt launcher's
	// Kyty.ini, saved right away; the summary is shown once. The windowed first launch did this in its own setup.
	{
		std::error_code ec;
		if (!fs::exists(settings_path, ec))
		{
			const std::string ini = find_qt_ini(paths);
			if (!ini.empty())
			{
				import_report = import_qt_ini(ini, paths, s, true);
				if (import_report.ok)
				{
					imported     = true;
					Settings out = s;
					if (out.run_dir == path_utf8(paths.run_dir))
					{
						out.run_dir.clear();
					}
					if (!save_settings(settings_path, out))
					{
						startup_note = "Could not save the settings imported from " + ini;
					}
				} else
				{
					startup_note = import_report.text() + " (defaults kept)";
				}
			}
		}
	}

	if (argc >= 2 && std::string(argv[1]) == "--dry-run")
	{
		LaunchPlan  plan = make_launch_plan(s, paths);
		if (!startup_note.empty())
		{
			plan.warnings.insert(plan.warnings.begin(), startup_note);
		}
		std::string text = (imported ? import_report.text() + "\n\n" : std::string()) + describe(plan);
		if (argc >= 3)
		{
			std::ofstream(utf8_path(argv[2]), std::ios::binary) << text;
		} else
		{
#ifdef _WIN32
			// GUI subsystem: borrow the console of the shell that started us.
			if (AttachConsole(ATTACH_PARENT_PROCESS) != FALSE)
			{
				FILE* out = nullptr;
				freopen_s(&out, "CONOUT$", "w", stdout);
			}
#endif
			std::fputs(text.c_str(), stdout);
		}
		return 0;
	}

	if (argc >= 2 && std::string(argv[1]) == "--play")
	{
		// Same launch as the Play button, without a window; output goes to the log file only.
		ChildProcess child;
		std::string  error;
		int          exit_code = -1;
		const auto   started   = std::chrono::steady_clock::now(); // from the launch: Start forks the emulator
		if (!child.Start(make_launch_plan(s, paths), [](const std::string&) {}, [&](int code) { exit_code = code; }, &error))
		{
			return 1;
		}
		child.Wait();
		// A headless launch is counted too: same key, same minimum and same file as the window.
		const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started).count();
		if (elapsed >= kMinPlaySession.count())
		{
			const int64_t now = static_cast<int64_t>(
				std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
			record_play_session(path_utf8(paths.play_time_file()), play_time_key(s.game), static_cast<uint64_t>(elapsed), now);
		}
		return exit_code;
	}

	auto ui = AppWindow::create();

	auto log_model  = std::make_shared<slint::VectorModel<slint::SharedString>>();

	// STEP6:6e: the Advanced tab's environment table (C++ owns the rows) and the device pickers of the page.
	auto env_model   = std::make_shared<slint::VectorModel<EnvVar>>();
	auto audio_model = std::make_shared<slint::VectorModel<slint::SharedString>>();
	auto gpu_model   = std::make_shared<slint::VectorModel<slint::SharedString>>();
	ui->global<EnvEditor>().set_vars(env_model);
	ui->global<Devices>().set_audio_inputs(audio_model);
	ui->global<Devices>().set_gpu_choices(gpu_model);
	{
		// A row edit (the page sends the whole row on every keystroke, so both fields stay in step).
		auto& env_editor = ui->global<EnvEditor>();
		env_editor.on_edited(
		    [env_model](int row, const slint::SharedString& name, const slint::SharedString& value)
		    {
			    if (row < 0 || static_cast<size_t>(row) >= env_model->row_count())
			    {
				    return;
			    }
			    EnvVar v;
			    v.name  = name;
			    v.value = value;
			    env_model->set_row_data(static_cast<size_t>(row), v);
		    });
		env_editor.on_removed(
		    [env_model](int row)
		    {
			    if (row >= 0 && static_cast<size_t>(row) < env_model->row_count())
			    {
				    env_model->erase(static_cast<size_t>(row));
			    }
		    });
		env_editor.on_added([env_model] { env_model->push_back(EnvVar {}); });
	}
	{
		// STEP6:6e: the microphone and GPU lists are enumerated at most once per session (a few ms), again when
		// the page asks for a rescan, and never where the platform has no enumeration (macOS, Linux).
		auto device_cache = std::make_shared<std::vector<std::string>>(); // audio input devices
		auto gpu_cache    = std::make_shared<std::vector<std::string>>();
		auto enumerated   = std::make_shared<bool>(false);
		auto push_devices = [audio_model, gpu_model, device_cache, gpu_cache, enumerated](bool rescan)
		{
			if (rescan || !*enumerated)
			{
				*device_cache = list_audio_input_devices();
				*gpu_cache    = list_gpu_devices();
				*enumerated   = true;
			}
			std::vector<slint::SharedString> audio;
			audio.reserve(device_cache->size());
			for (const auto& name: *device_cache)
			{
				audio.emplace_back(name);
			}
			audio_model->set_vector(audio);
			// Index 0 is "Auto" (the emulator's gpu index -1), so the device at index N is gpu index N-1.
			std::vector<slint::SharedString> gpu;
			gpu.reserve(gpu_cache->size() + 1);
			gpu.emplace_back("Auto");
			for (const auto& name: *gpu_cache)
			{
				gpu.emplace_back(name);
			}
			gpu_model->set_vector(gpu);
		};
		ui->global<Devices>().on_refresh([push_devices] { push_devices(false); });
		ui->global<Devices>().on_rescan([push_devices] { push_devices(true); });
	}
	{
		// Index in the Window size list (1280x720, 1600x900, 1920x1080, 2560x1440, 3840x2160); -1: not a preset.
		constexpr int kPresets[][2] = {{1280, 720}, {1600, 900}, {1920, 1080}, {2560, 1440}, {3840, 2160}};
		int           recommended = -1;
		for (int i = 0; have_monitor && i < 5; i++)
		{
			if (kPresets[i][0] == monitor_w && kPresets[i][1] == monitor_h)
			{
				recommended = i;
			}
		}
		ui->set_recommended_size(recommended);
	}
	// Puts the emulator settings of `v` into the settings page. The global page shows the user's settings; per-game mode
	// shows what the game resolves to (see resolve_for_launch).
	auto push_emu = [&](const Settings& v)
	{
		ui->set_width_text(slint::SharedString(std::to_string(v.width)));
		ui->set_height_text(slint::SharedString(std::to_string(v.height)));
		ui->set_extra_args(slint::SharedString(v.extra_args));
		ui->set_present_mode(v.present_mode);
		ui->set_shader_opt(v.shader_opt);
		ui->set_shader_log(v.shader_log);
		ui->set_printf_dir(v.printf_dir);
		ui->set_vblank_text(slint::SharedString(std::to_string(v.vblank)));
		ui->set_gpu_text(slint::SharedString(v.gpu_index >= 0 ? std::to_string(v.gpu_index) : std::string()));
		ui->set_readback(v.readback);
		ui->set_readback_tiled(v.readback_tiled);
		ui->set_tessellation(v.tessellation);
		ui->set_sync_raw(v.sync_raw);
		ui->set_hdr(v.hdr);
		ui->set_vulkan_validation(v.vulkan_validation);
		ui->set_shader_validation(v.shader_validation);
		ui->set_renderdoc(v.renderdoc);
		ui->set_cmd_dump(v.cmd_dump);
		ui->set_profiler(v.profiler);
		ui->set_printf_file(slint::SharedString(v.printf_file));
		ui->set_shader_log_folder(slint::SharedString(v.shader_log_folder));
		ui->set_cmd_dump_folder(slint::SharedString(v.cmd_dump_folder));
		ui->set_user_name(slint::SharedString(v.user_name));
		ui->set_user_id_text(slint::SharedString(std::to_string(v.user_id)));
		ui->set_console_language(v.console_language);
		ui->set_microphone(slint::SharedString(v.microphone));
		ui->set_amd_cpu(v.amd_cpu);
		ui->set_red_zone(v.red_zone);
		ui->set_trophy(v.trophy_notifications);
		ui->set_trophy_duration(v.trophy_duration);
		ui->set_fullscreen(v.fullscreen);
		ui->set_hide_cursor(v.hide_cursor);
		ui->set_perf_panel(v.perf_overlay);
		ui->set_skip_boot_logos(v.skip_notice_screen);
		ui->set_network_connected(v.network_connected);
		ui->set_controller_color(slint::SharedString(v.controller_color));
		ui->set_controller_swatch(ParseHexColor(v.controller_color));
		ui->set_controller_vibration(static_cast<float>(v.controller_vibration));
		ui->set_controller_volume(static_cast<float>(v.controller_volume));
		ui->set_keyboard_player(v.keyboard_player);
		// STEP6:6e: the environment table, and the GPU picker's index (0 = Auto, so the device at N is gpu index N-1).
		{
			std::vector<EnvVar> rows;
			rows.reserve(v.env.size());
			for (const auto& [name, value]: v.env)
			{
				EnvVar row;
				row.name  = slint::SharedString(name);
				row.value = slint::SharedString(value);
				rows.push_back(std::move(row));
			}
			env_model->set_vector(rows);
		}
		ui->global<Devices>().set_gpu_choice(v.gpu_index + 1);
	};
	// Library column widths, in header order ("240,100,...", in logical pixels).
	auto push_columns = [&](const std::string& columns)
	{
		auto&              cols = ui->global<Cols>();
		std::vector<float> w;
		for (size_t pos = 0; pos < columns.size();)
		{
			w.push_back(static_cast<float>(std::atof(columns.c_str() + pos)));
			size_t comma = columns.find(',', pos);
			pos          = comma == std::string::npos ? columns.size() : comma + 1;
		}
		if (w.size() == 9 && std::all_of(w.begin(), w.end(), [](float v) { return v >= 40.0f && v <= 4000.0f; }))
		{
			cols.set_name(w[0]);
			cols.set_serial(w[1]);
			cols.set_version(w[2]);
			cols.set_firmware(w[3]);
			cols.set_size(w[4]);
			cols.set_path(w[5]);
			cols.set_status(w[6]);
			cols.set_comment(w[7]);
			cols.set_playtime(w[8]);
		}
	};
	// STEP6:6e: every field of the settings page (and of the launcher's own bar) from one Settings: startup, and
	// Settings > Discard, which is why this is one function.
	auto push_all = [&](const Settings& v)
	{
		ui->set_game_dirs(slint::SharedString(v.game_dirs));
		ui->set_emulator(slint::SharedString(v.emulator));
		ui->set_game(slint::SharedString(v.game));
		ui->set_run_dir(slint::SharedString(v.run_dir));
		ui->set_addons_dir(slint::SharedString(v.addons_dir));
		// STEP7:3: the update check is a launcher setting, not an emulator one.
		ui->set_check_updates(v.check_updates);
		push_columns(v.columns);
		push_emu(v);
	};
	push_all(s);
#ifdef _WIN32
	ui->set_is_windows(true);
#endif
	ui->set_status(slint::SharedString(imported ? "Imported settings from " + import_report.source : startup_note.empty() ? "Ready" : startup_note));
	if (imported)
	{
		ui->set_import_summary_text(slint::SharedString(import_report.text()));
		ui->set_import_summary_open(true);
	}

	// Title id of the game whose settings the page is editing (per-game mode); empty on the global page.
	std::string game_mode_tid;
	// STEP6:6e: what the per-game page showed when it was opened: Discard puts it back.
	Settings game_mode_opened;
	// STEP6:6e: the last settings written to disk; Settings > Discard reloads them.
	Settings saved_settings = s;
	// STEP6:6e: set by collect_emu when the environment table has a row that can't be saved.
	std::string env_error;
	// The emulator settings shown on the settings page, written into `t`.
	auto collect_emu = [&](Settings& t)
	{
		t.width      = std::atoi(ToStd(ui->get_width_text()).c_str());
		t.height     = std::atoi(ToStd(ui->get_height_text()).c_str());
		t.extra_args = ToStd(ui->get_extra_args());
		t.present_mode = ui->get_present_mode();
		t.shader_opt = ui->get_shader_opt();
		t.shader_log = ui->get_shader_log();
		t.printf_dir = ui->get_printf_dir();
		t.vblank = std::atoi(ToStd(ui->get_vblank_text()).c_str());
		{
			const std::string g = ToStd(ui->get_gpu_text());
			t.gpu_index = g.empty() ? -1 : std::atoi(g.c_str());
		}
		t.readback = ui->get_readback();
		t.readback_tiled = ui->get_readback_tiled();
		t.tessellation = ui->get_tessellation();
		t.sync_raw = ui->get_sync_raw();
		t.hdr = ui->get_hdr();
		t.vulkan_validation = ui->get_vulkan_validation();
		t.shader_validation = ui->get_shader_validation();
		t.renderdoc = ui->get_renderdoc();
		t.cmd_dump = ui->get_cmd_dump();
		t.profiler = ui->get_profiler();
		t.printf_file = ToStd(ui->get_printf_file());
		t.shader_log_folder = ToStd(ui->get_shader_log_folder());
		t.cmd_dump_folder = ToStd(ui->get_cmd_dump_folder());
		t.user_name        = ToStd(ui->get_user_name());
		t.user_id          = std::atoi(ToStd(ui->get_user_id_text()).c_str());
		t.console_language = ui->get_console_language();
		t.microphone       = ToStd(ui->get_microphone());
		t.amd_cpu          = ui->get_amd_cpu();
		t.red_zone         = ui->get_red_zone();
		t.trophy_notifications = ui->get_trophy();
		t.trophy_duration      = ui->get_trophy_duration();
		t.fullscreen           = ui->get_fullscreen();
		t.hide_cursor          = ui->get_hide_cursor();
		t.perf_overlay         = ui->get_perf_panel();
		t.skip_notice_screen   = ui->get_skip_boot_logos();
		t.network_connected    = ui->get_network_connected();
		t.controller_color     = NormalizeHexColor(ToStd(ui->get_controller_color()));
		t.controller_vibration = std::clamp(static_cast<int>(std::lround(ui->get_controller_vibration())), 0, 100);
		t.controller_volume    = std::clamp(static_cast<int>(std::lround(ui->get_controller_volume())), 0, 100);
		t.keyboard_player      = std::clamp(ui->get_keyboard_player(), 0, 4);
		// STEP6:6e: the environment table. Names are trimmed; an empty name is an unused row, a blank value
		// leaves the variable out (the launcher unsets what is not in the table for the child), and a bad row is
		// reported (env_error) instead of being saved silently.
		env_error.clear();
		t.env.clear();
		for (size_t i = 0; i < env_model->row_count(); i++)
		{
			const auto row = env_model->row_data(i);
			if (!row.has_value())
			{
				continue;
			}
			const std::string name  = TrimText(ToStd(row->name));
			const std::string value = ToStd(row->value);
			if (name.empty())
			{
				continue;
			}
			const bool bad_first = std::isalpha(static_cast<unsigned char>(name[0])) == 0 && name[0] != '_';
			const bool bad_char  = std::any_of(name.begin(), name.end(),
			                                   [](unsigned char c) { return std::isalnum(c) == 0 && c != '_'; });
			if (bad_first || bad_char)
			{
				env_error = "Not an environment variable name: " + name + " (letters, digits, underscore)";
				continue;
			}
			if (t.env.count(name) != 0)
			{
				env_error = "Duplicate environment variable: " + name;
				continue;
			}
			// A line break would break the child's environment block.
			bool has_line_break = value.find(static_cast<char>(10)) != std::string::npos
			                      || value.find(static_cast<char>(13)) != std::string::npos;
			if (has_line_break)
			{
				env_error = "The value of " + name + " can't hold a line break";
				continue;
			}
			if (value.empty())
			{
				continue; // an empty value removes the variable
			}
			t.env[name] = value;
		}
	};
	auto collect = [&]
	{
		const auto& cols = ui->global<Cols>();
		s.columns.clear();
		for (float v: {cols.get_name(), cols.get_serial(), cols.get_version(), cols.get_firmware(), cols.get_size(), cols.get_path(),
		               cols.get_status(), cols.get_comment(), cols.get_playtime()})
		{
			s.columns += (s.columns.empty() ? "" : ",") + std::to_string(static_cast<int>(v + 0.5f));
		}

		s.emulator   = ToStd(ui->get_emulator());
		s.game       = ToStd(ui->get_game());
		s.run_dir    = ToStd(ui->get_run_dir());
		s.game_dirs  = ToStd(ui->get_game_dirs());
		s.addons_dir = ToStd(ui->get_addons_dir());
		// STEP7:3
		s.check_updates = ui->get_check_updates();
		// STEP6:6c: the library's view state (the sort key and direction, list or grid).
		s.library_sort      = ToStd(ui->get_sort_key());
		s.library_sort_desc = ui->get_sort_desc();
		s.library_grid      = ui->get_library_grid();
		// In per-game mode the page holds one game's values, not the user's settings.
		if (game_mode_tid.empty())
		{
			collect_emu(s);
		}
	};
	auto save = [&]
	{
		collect();
		// The default run folder is stored as "" so a portable/installed move does not leave a stale path.
		Settings out = s;
		if (out.run_dir == path_utf8(paths.run_dir))
		{
			out.run_dir.clear();
		}
		const bool ok = save_settings(settings_path, out);
		if (ok)
		{
			// STEP6:6e: what Settings > Discard puts back (the effective values, not the sparse file).
			saved_settings = s;
		}
		return ok;
	};

	// View > Console window: created on first use, shows the emulator output.
	auto console = std::make_shared<std::optional<slint::ComponentHandle<ConsoleWindow>>>();
	// STEP6:6a: output is coalesced into 30 ms batches, trimmed once per batch and pinned only while following.
	auto pending_lines = std::make_shared<std::vector<std::string>>();
	auto flush_timer   = std::make_shared<slint::Timer>();
	auto flush         = std::make_shared<std::function<void()>>();
	*flush             = [log_model, console, pending_lines]
	{
		if (pending_lines->empty())
		{
			return;
		}
		std::vector<std::string> lines;
		lines.swap(*pending_lines);
		const size_t total = log_model->row_count() + lines.size();
		if (total > kMaxLogLines)
		{
			// One model reset for the whole batch: keep the tail of the old model plus the new lines.
			const size_t                     drop     = total - kMaxLogLines;
			const size_t                     skip_new = drop > log_model->row_count() ? drop - log_model->row_count() : 0;
			std::vector<slint::SharedString> kept;
			kept.reserve(kMaxLogLines);
			for (size_t i = drop; i < log_model->row_count(); i++)
			{
				if (auto row = log_model->row_data(i))
				{
					kept.push_back(*row);
				}
			}
			for (size_t i = skip_new; i < lines.size(); i++)
			{
				kept.push_back(slint::SharedString(lines[i]));
			}
			log_model->set_vector(std::move(kept));
		} else
		{
			for (const auto& l: lines)
			{
				log_model->push_back(slint::SharedString(l));
			}
		}
		if (console->has_value())
		{
			(**console)->invoke_pin();
		}
	};
	auto append_lines = [pending_lines, flush_timer, flush](const std::vector<std::string>& lines)
	{
		pending_lines->insert(pending_lines->end(), lines.begin(), lines.end());
		if (!flush_timer->running())
		{
			flush_timer->start(slint::TimerMode::SingleShot, std::chrono::milliseconds(30), [flush] { (*flush)(); });
		}
	};
	auto append_text = [append_lines](const std::string& text)
	{
		std::vector<std::string> lines;
		size_t                   pos = 0;
		while (pos <= text.size())
		{
			size_t nl = text.find('\n', pos);
			if (nl == std::string::npos)
			{
				nl = text.size();
			}
			if (nl > pos || nl < text.size())
			{
				lines.push_back(text.substr(pos, nl - pos));
			}
			pos = nl + 1;
		}
		append_lines(lines);
	};
	if (!startup_note.empty())
	{
		append_text("warning: " + startup_note);
	}
	if (imported)
	{
		append_text(import_report.text());
	}

	// Game library: scanned off the UI thread; results are posted back to the event loop.
	auto        games        = std::make_shared<std::vector<GameEntry>>();
	auto        games_model  = std::make_shared<slint::VectorModel<GameRow>>();
	auto        grid_model   = std::make_shared<slint::VectorModel<GridRow>>(); // grid view: rows of indices into `games`
	auto        grid_columns = std::make_shared<int>(4);
	auto        size_cache   = std::make_shared<SizeCache>();
	// Play time totals: read once here; from now on only the UI thread writes this copy (see finish_session).
	const std::string play_time_file = path_utf8(paths.play_time_file());
	auto        play_times   = std::make_shared<PlayTimeDb>(load_play_times(play_time_file));
	auto        scan_active  = std::make_shared<std::atomic<bool>>(false);
	auto        games_changed = std::make_shared<std::function<void()>>(); // refreshes the Games tab (set once the Version Manager exists)
	std::thread scan_thread;
	ui->set_games(games_model);
	ui->set_grid_rows(grid_model);

	// STEP6:6c: the library order. Pinned games come first, then the saved sort key; both views use this one
	// order, because the grid rows only hold indices into `games`.
	auto favourite_keys = std::make_shared<std::set<std::string>>();
	auto favourite_list = std::make_shared<std::vector<std::string>>(s.favourite_games);
	for (const auto& key: *favourite_list)
	{
		favourite_keys->insert(path_key(key));
	}
	auto sort_key  = std::make_shared<GameSortKey>(game_sort_key_from_string(s.library_sort));
	auto sort_desc = std::make_shared<bool>(s.library_sort_desc);
	s.library_sort = game_sort_key_name(*sort_key); // keep only a key the code knows
	auto resort    = [games, sort_key, sort_desc, favourite_keys] { sort_games(*games, *sort_key, *sort_desc, *favourite_keys); };
	// The window shows the saved view state; the callbacks below write changes back to the app keys.
	ui->set_library_grid(s.library_grid);
	ui->set_sort_key(slint::SharedString(s.library_sort));
	ui->set_sort_desc(*sort_desc);

	// STEP6:6c: icons and backdrop art are decoded by this one bounded pool: never on the UI thread, and never
	// with a thread per image (the backdrop threads used to grow for the whole session).
	auto pool       = std::make_shared<WorkPool>(2);
	auto icon_cache = std::make_shared<IconCache>(200);
	auto icon_jobs  = std::make_shared<std::set<std::string>>(); // icons the pool is decoding right now

	// STEP6:6c: asks for one game's icon. The result comes back on the event loop and is matched to the game by
	// eboot path, because sorting may have moved its row since the request; nothing is changed for a game that
	// is no longer in the list.
	auto request_icon = std::make_shared<std::function<void(const std::string&)>>();
	*request_icon     = [pool, icon_cache, icon_jobs, games, games_model, favourite_keys,
	                     weak_ui = slint::ComponentWeakHandle<AppWindow>(ui)](const std::string& path)
	{
		if (path.empty() || icon_cache->Contains(path) || !icon_jobs->insert(path).second)
		{
			return;
		}
		pool->Post(
		    [icon_cache, icon_jobs, games, games_model, favourite_keys, weak_ui, path]
		    {
			    std::vector<uint8_t> rgba;
			    int                  w   = 0;
			    int                  hgt = 0;
			    const bool           ok  = load_image_rgba(path, kIconPixels, kIconPixels, rgba, w, hgt);
			    slint::invoke_from_event_loop(
			        [=, rgba = std::move(rgba)]
			        {
				        icon_jobs->erase(path);
				        if (!weak_ui.lock())
				        {
					        return;
				        }
				        const size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(hgt) * 4;
				        slint::Image img;
				        if (ok && w > 0 && hgt > 0 && rgba.size() >= bytes)
				        {
					        slint::SharedPixelBuffer<slint::Rgba8Pixel> buf(static_cast<uint32_t>(w), static_cast<uint32_t>(hgt));
					        std::memcpy(buf.begin(), rgba.data(), bytes);
					        img = slint::Image(std::move(buf));
				        }
				        // A failed decode stays cached as an empty image, so it is not retried on every rebuild.
				        icon_cache->Put(path, img);
				        for (size_t i = 0; i < games->size(); i++)
				        {
					        if ((*games)[i].icon != path)
					        {
						        continue;
					        }
					        GameRow row = MakeRow((*games)[i], *favourite_keys);
					        row.icon    = img;
					        games_model->set_row_data(static_cast<int>(i), row);
				        }
			        });
		    });
	};

	// Backdrop art: decoded and downscaled off the UI thread, kept in memory, crossfaded in Slint.
	using BgBuffer = slint::SharedPixelBuffer<slint::Rgba8Pixel>;
	auto bg_cache   = std::make_shared<std::map<std::string, BgBuffer>>();
	auto bg_loading = std::make_shared<std::set<std::string>>();
	auto bg_wanted  = std::make_shared<std::string>();
	auto bg_shown   = std::make_shared<std::string>();
	auto want_bg    = std::make_shared<std::function<void(const std::string&)>>();
	*want_bg = [pool, weak_ui = slint::ComponentWeakHandle<AppWindow>(ui), bg_cache, bg_loading, bg_wanted,
	            bg_shown](const std::string& path)
	{
		auto h = weak_ui.lock();
		if (!h)
		{
			return;
		}
		*bg_wanted = path;
		if (path == *bg_shown)
		{
			return;
		}
		auto show = [&](const std::string& p, const BgBuffer* buf)
		{
			(*h)->invoke_set_background(buf != nullptr ? slint::Image(*buf) : slint::Image());
			*bg_shown = p;
		};
		if (path.empty())
		{
			show(path, nullptr);
			return;
		}
		if (auto it = bg_cache->find(path); it != bg_cache->end())
		{
			show(path, &it->second);
			return;
		}
		if (!bg_loading->insert(path).second)
		{
			return;
		}
		// STEP6:6c: the same bounded pool decodes the backdrop; the "still wanted" check below survives.
		pool->Post(
		    [path, weak_ui, bg_cache, bg_loading, bg_wanted, bg_shown]
		    {
			    std::vector<uint8_t> rgba;
			    int                  w   = 0;
			    int                  hgt = 0;
			    const bool           ok  = load_image_rgba(path, 1600, 900, rgba, w, hgt);
			    slint::invoke_from_event_loop(
			        [=, rgba = std::move(rgba)]
			        {
				        bg_loading->erase(path);
				        auto ui_h = weak_ui.lock();
				        if (!ok || !ui_h)
				        {
					        return;
				        }
				        if (bg_cache->size() > 24)
				        {
					        bg_cache->clear();
				        }
				        BgBuffer buf(static_cast<uint32_t>(w), static_cast<uint32_t>(hgt));
				        std::memcpy(buf.begin(), rgba.data(), rgba.size());
				        auto it = bg_cache->emplace(path, std::move(buf)).first;
				        if (*bg_wanted == path)
				        {
					        (*ui_h)->invoke_set_background(slint::Image(it->second));
					        *bg_shown = path;
				        }
			        });
		    });
	};

	const std::string compat_cache = path_utf8(paths.compat_cache_file());

	// STEP6:6c: builds both models from `games`, which is already in display order. The grid rows hold indices,
	// so the list and the grid can never disagree; an icon that is already decoded is put in without a request.
	auto rebuild = [games, games_model, grid_model, grid_columns, icon_cache, favourite_keys, request_icon]()
	{
		std::vector<GameRow> rows;
		rows.reserve(games->size());
		for (const auto& g: *games)
		{
			GameRow row = MakeRow(g, *favourite_keys, *request_icon);
			if (!g.icon.empty())
			{
				if (const slint::Image* cached = icon_cache->Get(g.icon))
				{
					row.icon = *cached;
				}
			}
			rows.push_back(std::move(row));
		}
		games_model->set_vector(rows);

		std::vector<GridRow> grid;
		const size_t         columns = static_cast<size_t>(std::max(1, *grid_columns));
		for (size_t i = 0; i < games->size(); i += columns)
		{
			auto    cells = std::make_shared<slint::VectorModel<int>>();
			GridRow row;
			for (size_t k = i; k < games->size() && k < i + columns; k++)
			{
				cells->push_back(static_cast<int>(k));
			}
			row.cells = cells;
			grid.push_back(std::move(row));
		}
		grid_model->set_vector(grid);
	};

	// A rescan, a re-sort or a pin keeps the user's pick; at startup nothing is selected.
	auto restore_selection = [games, want_bg, &s, &paths, weak = slint::ComponentWeakHandle<AppWindow>(ui)]()
	{
		auto h = weak.lock();
		if (!h)
		{
			return;
		}
		int               selected = -1;
		const bool        keep     = (*h)->get_selected() >= 0;
		const std::string current  = path_key(ToStd((*h)->get_game()));
		if (keep)
		{
			for (size_t i = 0; i < games->size(); i++)
			{
				if (path_key((*games)[i].eboot) == current)
				{
					selected = static_cast<int>(i);
					break;
				}
			}
		}
		(*h)->set_selected(selected);
		(*want_bg)(selected >= 0 ? (*games)[static_cast<size_t>(selected)].background : std::string());
		// STEP7:1: the same pick decides the build named in the bottom bar.
		UpdateSelectedEmu(*h, s, paths, *games);
	};

	auto publish = [games, size_cache, scan_active, games_changed, resort, rebuild, restore_selection, play_times,
	                weak = slint::ComponentWeakHandle<AppWindow>(ui)](std::vector<GameEntry> list, SizeCache sizes, bool done)
	{
		auto h = weak.lock();
		if (!h)
		{
			return;
		}
		*size_cache = std::move(sizes);
		*games      = std::move(list);
		// The totals belong to the UI thread, so they go on this fresh list here and not in the scan thread.
		apply_play_times(*games, *play_times);
		// A rescan and a header click sort with the same comparator, before the rows are built.
		resort();
		rebuild();
		if (*games_changed)
		{
			(*games_changed)();
		}
		restore_selection();
		if (done)
		{
			(*h)->set_scanning(false);
			*scan_active = false;
		}
	};

	auto refresh = [&, publish]
	{
		if (scan_active->exchange(true))
		{
			return;
		}
		if (scan_thread.joinable())
		{
			scan_thread.join();
		}
		collect();
		ui->set_scanning(true);
		scan_thread = std::thread(
		    [roots = s.game_dirs, hidden = s.hidden_games, cache = *size_cache, compat_cache, publish]() mutable
		    {
			    std::vector<GameEntry> list = scan_games(roots, cache);
			    std::set<std::string>  hidden_keys;
			    for (const auto& h: hidden)
			    {
				    hidden_keys.insert(path_key(h));
			    }
			    // Hidden by title id (by eboot path for a game that has none).
			    std::erase_if(list,
			                  [&](const GameEntry& g)
			                  { return g.title_id.empty() ? hidden_keys.count(path_key(g.eboot)) != 0 : hidden_keys.count(path_key(g.title_id)) != 0; });
			    const std::string      saved = read_text_file(compat_cache);
			    apply_compat(list, parse_compat(saved));
			    slint::invoke_from_event_loop([publish, list, cache]() mutable { publish(list, cache, false); });

			    std::string fresh = fetch_compat_json();
			    if (!fresh.empty() && fresh != saved)
			    {
				    CompatDb db = parse_compat(fresh);
				    if (!db.empty())
				    {
					    write_text_file(compat_cache, fresh);
					    apply_compat(list, db);
				    }
			    }
			    slint::invoke_from_event_loop([publish, list, cache]() mutable { publish(list, cache, true); });
		    });
	};

	ui->on_refresh_library(refresh);
	ui->on_select_game(
	    [&, want_bg](int i)
	    {
		    const int previous = ui->get_selected();
		    if (i < 0)
		    {
			    ui->set_selected(-1);
			    (*want_bg)(std::string());
		    } else if (static_cast<size_t>(i) < games->size())
		    {
			    ui->set_selected(i);
			    ui->set_game(slint::SharedString((*games)[static_cast<size_t>(i)].eboot));
			    (*want_bg)((*games)[static_cast<size_t>(i)].background);
		    } else
		    {
			    return;
		    }
		    // STEP6:6e: the previous game's status line does not apply to this one. Only the game-specific
		    // texts go (the version manager's and the library's own messages stay), and never while a game runs.
		    if (i != previous && !ui->get_running())
		    {
			    const std::string status = ToStd(ui->get_status());
			    for (const char* prefix: {"No save data yet", "Game settings saved for", "Could not save the settings of",
			                              "Removed ", "Restored "})
			    {
				    if (status.rfind(prefix, 0) == 0)
				    {
					    ui->set_status(slint::SharedString());
					    break;
				    }
			    }
			    // STEP6:6e: a .zar archive: the launcher can only see the file name (see library.cpp).
			    if (i >= 0 && (*games)[static_cast<size_t>(i)].archive)
			    {
				    ui->set_status(slint::SharedString((*games)[static_cast<size_t>(i)].name
				                                       + " is a .zar archive: the name and serial come from the file name; reading its "
				                                         "metadata needs zarchive, which the launcher doesn't have"));
			    }
		    }
		    // STEP7:1: the build this game starts with, named after it in the bottom bar.
		    UpdateSelectedEmu(ui, s, paths, *games);
	    });

	// STEP6:6c: a header click (or a flip of the sorted column) re-sorts the library and saves the order.
	ui->on_sort_by(
	    [&, resort, rebuild, restore_selection, sort_key, sort_desc](slint::SharedString key, bool desc)
	    {
		    *sort_key           = game_sort_key_from_string(ToStd(key));
		    *sort_desc          = desc;
		    s.library_sort      = game_sort_key_name(*sort_key);
		    s.library_sort_desc = desc;
		    resort();
		    rebuild();
		    restore_selection();
		    save();
	    });
	// STEP6:6c: "Pin to top" / "Unpin". Keyed by title id (by eboot path for a game without one); the pinned
	// games keep their place in the chosen sort.
	ui->on_pin_game(
	    [&, resort, rebuild, restore_selection, favourite_keys, favourite_list](int i)
	    {
		    if (i < 0 || static_cast<size_t>(i) >= games->size())
		    {
			    return;
		    }
		    const GameEntry&  g   = (*games)[static_cast<size_t>(i)];
		    const std::string key = favourite_key(g);
		    if (favourite_keys->erase(key) == 0)
		    {
			    favourite_keys->insert(key);
			    favourite_list->push_back(g.title_id.empty() ? g.eboot : g.title_id);
		    } else
		    {
			    std::erase_if(*favourite_list, [&](const std::string& item) { return path_key(item) == key; });
		    }
		    s.favourite_games = *favourite_list;
		    resort();
		    rebuild();
		    restore_selection();
		    save();
	    });
	// STEP6:6c: how many cards the grid fits per row; Slint debounces the library box width and calls this.
	ui->on_grid_columns_changed(
	    [grid_columns, rebuild](int columns)
	    {
		    const int n = std::max(1, columns);
		    if (n == *grid_columns)
		    {
			    return;
		    }
		    *grid_columns = n;
		    rebuild();
	    });
	// STEP6:6c: list/grid toggle. The mode is one of the app keys.
	ui->on_library_view_changed(
	    [&](bool grid)
	    {
		    s.library_grid = grid;
		    save();
	    });
	// Game folders list: the ';'-separated game-dirs shown one per row.
	auto game_dir_model = std::make_shared<slint::VectorModel<slint::SharedString>>();
	ui->set_game_dir_list(game_dir_model);
	auto sync_game_dirs = [&, game_dir_model]
	{
		std::vector<slint::SharedString> rows;
		const std::string                dirs = ToStd(ui->get_game_dirs());
		for (size_t pos = 0; pos <= dirs.size();)
		{
			size_t end = dirs.find(';', pos);
			if (end == std::string::npos)
			{
				end = dirs.size();
			}
			if (end > pos)
			{
				rows.emplace_back(dirs.substr(pos, end - pos));
			}
			pos = end + 1;
		}
		game_dir_model->set_vector(rows);
		ui->set_game_dir_selected(-1);
	};
	sync_game_dirs();

	// STEP6:6b Trophies page: progress of every library game that has a trophy package. The overview is
	// scanned off the UI thread (bg_threads joins the scans when the launcher exits) and cached by file stamp.
	auto trophy_rows     = std::make_shared<std::vector<size_t>>(); // overview row -> library index
	auto trophy_cache    = std::make_shared<TrophyCache>();
	auto trophy_scan_gen = std::make_shared<std::atomic<uint64_t>>(0);
	auto trophy_cancel   = std::make_shared<std::atomic<bool>>(false); // cancel token of the running scan
	const slint::ComponentWeakHandle<AppWindow> trophy_weak(ui);

	auto publish_trophy_scan = [games, trophy_rows, weak = trophy_weak](
	                               const std::vector<TrophyScanEntry>&                  entries,
	                               const std::shared_ptr<const std::vector<GameEntry>>& scanned, bool done)
	{
		auto h = weak.lock();
		if (!h)
		{
			return;
		}
		const auto& t = (*h)->global<Trophies>();
		// Discard a scan whose library no longer matches: the row indices would be wrong.
		bool stale = games->size() != scanned->size();
		for (size_t i = 0; !stale && i < scanned->size(); i++)
		{
			stale = path_key((*games)[i].eboot) != path_key((*scanned)[i].eboot);
		}
		if (stale)
		{
			t.set_loading(false);
			return;
		}
		std::vector<TrophyGame> rows;
		TrophyProgress          all;
		trophy_rows->clear();
		for (const auto& entry: entries)
		{
			if (entry.game_index >= games->size())
			{
				continue;
			}
			const auto& g = (*games)[entry.game_index];
			const auto& p = entry.progress;
			TrophyGame  r;
			r.name     = slint::SharedString(g.name);
			r.serial   = slint::SharedString(g.title_id);
			r.percent  = p.Percent();
			r.earned   = p.earned;
			r.total    = p.total;
			r.platinum = p.earned_grade[1];
			r.gold     = p.earned_grade[2];
			r.silver   = p.earned_grade[3];
			r.bronze   = p.earned_grade[4];
			if (!g.icon.empty())
			{
				r.icon = slint::Image::load_from_path(slint::SharedString(g.icon));
			}
			rows.push_back(r);
			trophy_rows->push_back(entry.game_index);
			all.earned += p.earned;
			for (size_t k = 1; k < all.earned_grade.size(); k++)
			{
				all.earned_grade[k] += p.earned_grade[k];
			}
		}
		t.set_games(std::make_shared<slint::VectorModel<TrophyGame>>(rows));
		t.set_earned(all.earned);
		t.set_platinum(all.earned_grade[1]);
		t.set_gold(all.earned_grade[2]);
		t.set_silver(all.earned_grade[3]);
		t.set_bronze(all.earned_grade[4]);
		if (done)
		{
			t.set_loading(false);
		}
	};
	auto open_trophy_overview = [&, trophy_rows]
	{
		const std::string run_dir = ToStd(ui->get_run_dir());
		const int         user_id = std::atoi(ToStd(ui->get_user_id_text()).c_str());

		// One scan per open; the previous one (if still running) stops at its next check.
		trophy_cancel->store(true);
		auto           cancel = std::make_shared<std::atomic<bool>>(false);
		trophy_cancel         = cancel;
		const uint64_t gen    = trophy_scan_gen->fetch_add(1) + 1;

		auto scanned = std::shared_ptr<const std::vector<GameEntry>>(std::make_shared<std::vector<GameEntry>>(*games));
		{
			const auto& t = ui->global<Trophies>();
			t.set_loading(true);
			t.set_run_dir(slint::SharedString(run_dir));
			// Cached rows paint at once; the fresh scan replaces them when it finishes.
			publish_trophy_scan(cached_trophy_progress(*scanned, run_dir, user_id, *trophy_cache), scanned, false);
		}

		pool->Post(
		    [scanned, trophy_cache, publish_trophy_scan, run_dir, user_id, cancel, gen, trophy_scan_gen, trophy_weak]
		    {
			    std::vector<TrophyScanEntry> entries;
			    try
			    {
				    entries = scan_trophy_progress(*scanned, run_dir, user_id, *trophy_cache, *cancel);
			    }
			    catch (...)
			    {
				    entries.clear(); // an unreadable game must still clear "Reading trophies…"
			    }
			    if (cancel->load() || trophy_scan_gen->load() != gen)
			    {
				    return;
			    }
			    slint::invoke_from_event_loop(
			        [publish_trophy_scan, entries = std::move(entries), scanned, cancel, gen, trophy_scan_gen, trophy_weak]
			        {
				        if (cancel->load() || trophy_scan_gen->load() != gen)
				        {
					        return; // the page was reopened (or the scan was replaced)
				        }
				        auto h = trophy_weak.lock();
				        if (!h)
				        {
					        return;
				        }
				        if (!(*h)->get_show_trophies())
				        {
					        // Closed while scanning: drop the result; the next open scans again.
					        (*h)->global<Trophies>().set_loading(false);
					        return;
				        }
				        publish_trophy_scan(entries, scanned, true);
			        });
		    });
	};
	ui->global<Trophies>().on_opened(open_trophy_overview);
	// STEP6:6b: the emulator may have written unlocks: drop the running game's cache and refresh an open page.
	auto trophy_after_exit = std::make_shared<std::function<void()>>();
	*trophy_after_exit    = [&, games, trophy_cache]
	{
		const std::string eboot   = path_key(ToStd(ui->get_game()));
		const std::string run_dir = ToStd(ui->get_run_dir());
		const int         user_id = std::atoi(ToStd(ui->get_user_id_text()).c_str());
		for (const auto& g: *games)
		{
			if (path_key(g.eboot) == eboot)
			{
				invalidate_trophy_cache(*trophy_cache, g.dir, g.title_id, run_dir, user_id);
				break;
			}
		}
		if (ui->get_show_trophies())
		{
			open_trophy_overview();
		}
	};
	// One game's trophies, by library index.
	auto open_game_trophies = [&](size_t gi)
	{
		if (gi >= games->size())
		{
			return;
		}
		const auto&             g = (*games)[gi];
		std::vector<TrophyInfo> list;
		const std::string       cache = path_utf8(paths.trophy_cache_dir(g.title_id));
		const auto&             t     = ui->global<Trophies>();
		t.set_loading(true);
		if (!load_trophy_list(g.dir, g.title_id, ToStd(ui->get_run_dir()), std::atoi(ToStd(ui->get_user_id_text()).c_str()),
		                      ui->get_console_language(), cache, list))
		{
			t.set_detail_open(false);
			t.set_loading(false);
			ui->set_status(slint::SharedString(g.name + " has no trophies"));
			return;
		}
		std::vector<TrophyItem> items;
		TrophyProgress          p;
		for (const auto& info: list)
		{
			TrophyItem item;
			item.name      = slint::SharedString(info.name);
			item.detail    = slint::SharedString(info.detail);
			item.reward    = slint::SharedString(info.reward);
			item.grade     = info.grade;
			item.earned    = info.earned;
			item.earned_at = slint::SharedString(info.earned_at);
			item.hidden    = info.hidden;
			if (!info.icon.empty())
			{
				item.icon = slint::Image::load_from_path(slint::SharedString(info.icon));
			}
			items.push_back(item);
			p.total++;
			p.total_grade[info.grade]++;
			if (info.earned)
			{
				p.earned++;
				p.earned_grade[info.grade]++;
			}
		}
		TrophyGame header;
		header.name     = slint::SharedString(g.name);
		header.serial   = slint::SharedString(g.title_id);
		header.percent  = p.Percent();
		header.earned   = p.earned;
		header.total    = p.total;
		header.platinum = p.earned_grade[1];
		header.gold     = p.earned_grade[2];
		header.silver   = p.earned_grade[3];
		header.bronze   = p.earned_grade[4];
		t.set_detail_game(header);
		t.set_items(std::make_shared<slint::VectorModel<TrophyItem>>(items));
		t.set_detail_open(true);
		t.set_loading(false);
	};
	ui->global<Trophies>().on_open_game(
	    [trophy_rows, open_game_trophies](int row)
	    {
		    if (row >= 0 && static_cast<size_t>(row) < trophy_rows->size())
		    {
			    open_game_trophies((*trophy_rows)[static_cast<size_t>(row)]);
		    }
	    });
	ui->on_open_game_trophies(
	    [&, open_trophy_overview, open_game_trophies](int i)
	    {
		    open_trophy_overview();
		    if (i >= 0)
		    {
			    open_game_trophies(static_cast<size_t>(i));
		    }
	    });
	ui->on_open_save_folder(
	    [&](int i)
	    {
		    if (i < 0 || static_cast<size_t>(i) >= games->size())
		    {
			    return;
		    }
		    const std::string serial = (*games)[static_cast<size_t>(i)].title_id;
		    const fs::path    dir    = utf8_path(ToStd(ui->get_run_dir())) / "_SaveData" / utf8_path(serial);
		    std::error_code   ec;
		    if (serial.empty() || !fs::is_directory(dir, ec))
		    {
			    ui->set_status(slint::SharedString("No save data yet: " + path_utf8(dir)));
			    return;
		    }
		    open_in_system(path_utf8(dir));
	    });
	// STEP6:6e: "Delete save data…" - only <run folder>/_SaveData/<serial>, never the game folder, _Trophies or
	// anything else. The dialog in the launcher window asks first; Settings > Games routes through GameActions.
	auto delete_save_data = [&](int i)
	{
		if (i < 0 || static_cast<size_t>(i) >= games->size())
		{
			return;
		}
		const GameEntry& g = (*games)[static_cast<size_t>(i)];
		if (g.title_id.empty())
		{
			ui->set_status(slint::SharedString(g.name + " has no title id, so it has no save data folder"));
			return;
		}
		std::error_code ec;
		const fs::path  save_root = utf8_path(ToStd(ui->get_run_dir())) / "_SaveData";
		const fs::path  target    = save_root / utf8_path(g.title_id);
		if (!fs::is_directory(target, ec))
		{
			ui->set_status(slint::SharedString("No save data for " + g.name + " (" + g.title_id + ")"));
			return;
		}
		// Canonical paths, so a symlink or a "<serial>/.." trick can't lead out of _SaveData.
		std::error_code ec_root;
		std::error_code ec_target;
		const fs::path  canon_root   = fs::weakly_canonical(save_root, ec_root);
		const fs::path  canon_target = fs::weakly_canonical(target, ec_target);
		if (ec_root || ec_target || canon_root.empty() || canon_target == canon_root || !PathIsUnder(canon_target, canon_root))
		{
			ui->set_status(slint::SharedString("Not deleting " + path_utf8(target) + ": it is not inside " + path_utf8(save_root)));
			return;
		}
		// Counted first for the status line (a save folder holds a handful of files).
		uint64_t        files = 0;
		uint64_t        bytes = 0;
		std::error_code wec;
		for (fs::recursive_directory_iterator it(canon_target, fs::directory_options::skip_permission_denied, wec), end; !wec && it != end;
		     it.increment(wec))
		{
			std::error_code fec;
			if (it->is_regular_file(fec))
			{
				files++;
				bytes += it->file_size(fec);
			}
		}
		std::error_code del;
		fs::remove_all(canon_target, del);
		if (del)
		{
			ui->set_status(slint::SharedString("Could not delete the save data of " + g.name + ": " + del.message()));
			return;
		}
		ui->set_status(slint::SharedString("Deleted save data for " + g.name + " (" + std::to_string(files) + " files, " + format_size(bytes) + ")"));
	};
	ui->on_delete_save_data(delete_save_data);
	ui->global<GameActions>().on_delete_save_data(
	    [&](int i)
	    {
		    if (i < 0 || static_cast<size_t>(i) >= games->size())
		    {
			    return;
		    }
		    ui->set_delete_save_row(i);
		    ui->set_delete_save_prompt_open(true);
	    });
	ui->on_remove_game(
	    [&, want_bg, rebuild]
	    {
		    const int i = ui->get_selected();
		    if (i < 0 || static_cast<size_t>(i) >= games->size())
		    {
			    return;
		    }
		    // Only takes the game off the list; its files are never touched.
		    const GameEntry& hide = (*games)[static_cast<size_t>(i)];
		    s.hidden_games.push_back(hide.title_id.empty() ? hide.eboot : hide.title_id);
		    games->erase(games->begin() + i);
		    // STEP6:6c: rebuilds both models, so the grid does not keep the removed game's card.
		    rebuild();
		    ui->set_selected(-1);
		    (*want_bg)(std::string());
		    save();
	    });
	ui->on_add_game(
	    [&, sync_game_dirs]
	    {
		    auto p = PickPath(ToStd(ui->get_game()), false, L"Game (eboot.bin, .bin, .elf, .zar)", L"eboot.bin;*.bin;*.elf;*.zar");
		    if (p.empty())
		    {
			    return;
		    }
		    // A folder holding eboot.bin is scanned as a game of its own. STEP6:6e: a .zar archive is listed as
		    // the game itself, so its full path goes into the game folders (scan_games takes file roots).
		    const std::string dir = IsArchivePath(p) ? p : path_utf8(utf8_path(p).parent_path());
		    std::string       cur = ToStd(ui->get_game_dirs());
		    if (!dir.empty() && (";" + path_key(cur) + ";").find(";" + path_key(dir) + ";") == std::string::npos)
		    {
			    ui->set_game_dirs(slint::SharedString(cur.empty() ? dir : cur + ";" + dir));
		    }
		    sync_game_dirs();
		    ui->set_game(slint::SharedString(p));
		    {
			    // Adding a game puts it back on the list if it was removed earlier.
			    const std::string pk    = path_key(p);
			    const std::string title = path_key(read_title_id(p));
			    std::erase_if(s.hidden_games, [&](const std::string& item) { return path_key(item) == pk || (!title.empty() && path_key(item) == title); });
		    }
		    save();
		    refresh();
	    });
	ui->on_browse_game_dir(
	    [&, sync_game_dirs]
	    {
		    auto p = PickPath(ToStd(ui->get_game_dirs()), true, nullptr, nullptr);
		    if (p.empty())
		    {
			    return;
		    }
		    std::string cur = ToStd(ui->get_game_dirs());
		    if ((";" + path_key(cur) + ";").find(";" + path_key(p) + ";") == std::string::npos)
		    {
			    ui->set_game_dirs(slint::SharedString(cur.empty() ? p : cur + ";" + p));
		    }
		    sync_game_dirs();
		    save();
		    refresh();
	    });
	ui->on_remove_game_dir(
	    [&, sync_game_dirs, game_dir_model](int index)
	    {
		    if (index < 0 || static_cast<size_t>(index) >= game_dir_model->row_count())
		    {
			    return;
		    }
		    std::string out;
		    for (size_t i = 0; i < game_dir_model->row_count(); i++)
		    {
			    if (static_cast<int>(i) != index)
			    {
				    out += (out.empty() ? "" : ";") + ToStd(*game_dir_model->row_data(i));
			    }
		    }
		    ui->set_game_dirs(slint::SharedString(out));
		    sync_game_dirs();
		    save();
		    refresh();
	    });

	// The one add-ons (DLC) folder: Browse picks it, the field can be typed in, Clear empties it.
	ui->on_browse_addons_dir(
	    [&]
	    {
		    auto p = PickPath(ToStd(ui->get_addons_dir()), true, nullptr, nullptr);
		    if (!p.empty())
		    {
			    ui->set_addons_dir(slint::SharedString(p));
			    save();
		    }
	    });
	// Settings > Paths > Import from Kyty.ini...: the Qt launcher's file, into the current settings (only the keys it holds).
	ui->on_import_kyty_ini(
	    [&, append_text, sync_game_dirs]
	    {
		    auto p = PickPath(path_utf8(paths.exe_dir / "Kyty.ini"), false, L"Kyty.ini (*.ini)", L"*.ini");
		    if (p.empty())
		    {
			    return;
		    }
		    collect();
		    const ImportReport rep = import_qt_ini(p, paths, s, false);
		    if (rep.ok)
		    {
			    push_emu(s);
			    ui->set_emulator(slint::SharedString(s.emulator));
			    ui->set_run_dir(slint::SharedString(s.run_dir));
			    ui->set_game_dirs(slint::SharedString(s.game_dirs));
			    ui->set_addons_dir(slint::SharedString(s.addons_dir));
			    sync_game_dirs();
			    ui->set_status(slint::SharedString(save() ? "Imported settings from " + p : "Imported settings from " + p + ", but could not save them"));
			    refresh();
		    } else
		    {
			    ui->set_status(slint::SharedString(rep.text()));
		    }
		    append_text(rep.text());
		    ui->set_import_summary_text(slint::SharedString(rep.text()));
		    ui->set_import_summary_open(true);
	    });
	ui->on_controller_color_edited([&](const slint::SharedString& text) { ui->set_controller_swatch(ParseHexColor(ToStd(text))); });

	ui->on_browse_emulator(
	    [&]
	    {
		    auto p = PickPath(ToStd(ui->get_emulator()), false, L"Emulator (*.exe)", L"*.exe");
		    if (!p.empty())
		    {
			    ui->set_emulator(slint::SharedString(p));
		    }
	    });
	ui->on_browse_game(
	    [&]
	    {
		    auto p = PickPath(ToStd(ui->get_game()), false, L"Game (eboot.bin, .bin, .elf, .zar)", L"eboot.bin;*.bin;*.elf;*.zar");
		    if (!p.empty())
		    {
			    ui->set_game(slint::SharedString(p));
		    }
	    });
	ui->on_browse_run_dir(
	    [&]
	    {
		    auto p = PickPath(ToStd(ui->get_run_dir()), true, nullptr, nullptr);
		    if (!p.empty())
		    {
			    ui->set_run_dir(slint::SharedString(p));
		    }
	    });

	ui->on_save(
	    [&]
	    {
		    if (!game_mode_tid.empty())
		    {
			    // Per-game mode: only the keys that differ from what the game gets without its file are written.
			    std::string      tid      = game_mode_tid;
			    const Settings   baseline = [&]
			    {
				    Settings user = s;
				    user.emulator = emulator_for_game(s, paths, tid);
				    return resolve_for_launch(user, paths, tid, nullptr, false);
			    }();
			    Settings edited = baseline;
			    collect_emu(edited);
			    const bool ok = save_game_overrides(paths, tid, baseline, edited);
			    // STEP6:6e: a bad environment row is reported instead of "saved" (the other values did save).
			    ui->set_status(slint::SharedString(!ok ? "Could not save the settings of " + tid
			                                           : env_error.empty() ? "Game settings saved for " + tid : env_error));
			    return;
		    }
		    const bool ok = save();
		    // STEP7:1: Save can change the emulator the whole library runs, so the bottom bar's build name follows.
		    UpdateSelectedEmu(ui, s, paths, *games);
		    // STEP6:6e: same for the env table (save() ran collect(), which fills env_error).
		    ui->set_status(slint::SharedString(!env_error.empty() ? env_error
		                                         : ok                 ? std::string("Settings saved")
		                                                              : std::string("Could not save settings")));
	    });
	// Settings > Discard: put the saved values back; nothing is written. The page goes back to the library on
	// its own after calling this (ui/settings.slint), so there is no status line to set.
	ui->on_discard_settings(
	    [&]
	    {
		    if (!game_mode_tid.empty())
		    {
			    // Per-game mode: the values the page showed when it was opened (the emulator pin is untouched).
			    push_emu(game_mode_opened);
			    return;
		    }
		    s = saved_settings;
		    push_all(s);
		    sync_game_dirs();
	    });
	// Emulator builds: the Version Manager page and the build dropdown.
	VersionManager version_manager(
	    ui, s, path_utf8(paths.versions_dir), path_utf8(paths.cache_dir), save,
	    [](const std::string& current, bool folder)
	    {
		    return folder ? PickPath(current, true, nullptr, nullptr)
		                  : PickPath(current, false, L"Emulator (kyty_emulator.exe)", L"*.exe");
	    });

	// STEP7:2: Settings > Games' "Removed games" list: the keys saved in hidden_games, in their saved order, with
	// a Restore that drops the key and rescans (the game's files were never touched).
	auto removed_model = std::make_shared<slint::VectorModel<RemovedGameRow>>();
	ui->set_removed_games(removed_model);
	auto refresh_removed_games = [&, removed_model]
	{
		std::vector<RemovedGameRow> rows;
		rows.reserve(s.hidden_games.size());
		for (const auto& key: s.hidden_games)
		{
			rows.push_back(RemovedRow(key));
		}
		removed_model->set_vector(rows);
		// The selection is an index into the list before it was rebuilt.
		if (ui->get_removed_selected() >= static_cast<int>(rows.size()))
		{
			ui->set_removed_selected(rows.empty() ? -1 : static_cast<int>(rows.size()) - 1);
		}
	};
	ui->on_restore_game(
	    [&, refresh_removed_games](int row)
	    {
		    if (row < 0 || static_cast<size_t>(row) >= s.hidden_games.size())
		    {
			    return;
		    }
		    // The list is built from hidden_games in order, so the row is the entry (checked again above).
		    const std::string label = ToStd(removed_model->row_data(static_cast<size_t>(row))->label);
		    s.hidden_games.erase(s.hidden_games.begin() + row);
		    ui->set_removed_selected(-1);
		    refresh_removed_games();
		    const bool ok = save();
		    ui->set_status(slint::SharedString("Restored " + label + (ok ? "" : ", but the list could not be saved")));
		    refresh(); // the rescan puts the game back on the library
	    });

	// Games tab (emulator pins) and per-game settings, both keyed by title id.
	auto game_config_model = std::make_shared<slint::VectorModel<GameConfigRow>>();
	ui->set_game_configs(game_config_model);
	auto refresh_game_configs = [&, game_config_model]
	{
		std::vector<GameConfigRow> rows;
		for (const auto& g: *games)
		{
			GameConfigRow r;
			r.name         = slint::SharedString(g.name);
			r.serial       = slint::SharedString(g.title_id);
			r.game_version = slint::SharedString(g.game_version);
			r.can_config   = !g.title_id.empty();
			const std::string pin = read_game_emulator(paths, g.title_id);
			std::error_code   ec;
			r.pinned      = !pin.empty();
			r.emu_missing = r.pinned && !fs::exists(utf8_path(pin), ec);
			r.emu         = slint::SharedString(!r.can_config ? std::string("-")
			                                    : !r.pinned   ? std::string("None")
			                                                  : version_manager.BuildName(pin) + (r.emu_missing ? " (missing)" : ""));
			if (!g.icon.empty())
			{
				r.icon = slint::Image::load_from_path(slint::SharedString(g.icon));
			}
			rows.push_back(r);
		}
		game_config_model->set_vector(rows);
		// STEP7:1 and STEP7:2: a rescan, a pin or a build change moves the Removed list and the bottom bar's build
		// name as well (this runs from games_changed and from the Games tab being opened).
		refresh_removed_games();
		UpdateSelectedEmu(ui, s, paths, *games);
	};
	*games_changed = refresh_game_configs;
	version_manager.SetOnBuildsChanged(refresh_game_configs);
	// "N games are pinned to this build" in the Remove dialog; the pins themselves are kept.
	version_manager.SetPinCounter(
	    [&](const std::vector<std::string>& exes)
	    {
		    std::set<std::string> keys;
		    for (const auto& e: exes)
		    {
			    keys.insert(path_key(e));
		    }
		    std::set<std::string> titles;
		    for (const auto& g: *games)
		    {
			    const std::string pin = read_game_emulator(paths, g.title_id);
			    if (!pin.empty() && keys.count(path_key(pin)) != 0)
			    {
				    titles.insert(g.title_id);
			    }
		    }
		    return static_cast<int>(titles.size());
	    });
	ui->on_games_opened(refresh_game_configs);
	ui->on_set_game_emu(
	    [&, refresh_game_configs](int row, int build)
	    {
		    if (row < 0 || static_cast<size_t>(row) >= games->size() || (*games)[static_cast<size_t>(row)].title_id.empty())
		    {
			    return;
		    }
		    const GameEntry& g = (*games)[static_cast<size_t>(row)];
		    std::string      exe;
		    if (build >= 0)
		    {
			    const auto builds = ui->get_pin_builds();
			    if (!builds || static_cast<size_t>(build) >= builds->row_count())
			    {
				    return;
			    }
			    exe = ToStd(builds->row_data(static_cast<size_t>(build))->path);
		    }
		    if (!set_game_emulator(paths, g.title_id, exe))
		    {
			    ui->set_status(slint::SharedString("Could not save the emulator of " + g.name + " (" + g.title_id + ")"));
			    return;
		    }
		    refresh_game_configs();
		    ui->set_status(slint::SharedString(exe.empty() ? g.name + " uses the default emulator"
		                                                   : g.name + " is pinned to " + version_manager.BuildName(exe)));
	    });
	// The page shows what a game resolves to: defaults < build config (of the exe the game runs) < user < game file.
	auto show_game_values = [&](const std::string& tid)
	{
		Settings user = s;
		user.emulator = emulator_for_game(s, paths, tid);
		const Settings resolved = resolve_for_launch(user, paths, tid);
		// STEP6:6e: what Discard puts back in per-game mode.
		game_mode_opened = resolved;
		push_emu(resolved);
	};
	ui->on_open_game_settings(
	    [&](int row, bool from_library)
	    {
		    if (row < 0 || static_cast<size_t>(row) >= games->size())
		    {
			    return;
		    }
		    const GameEntry& g = (*games)[static_cast<size_t>(row)];
		    if (g.title_id.empty())
		    {
			    ui->set_status(slint::SharedString(g.name + " has no title id, it can't have its own settings"));
			    return;
		    }
		    collect(); // keeps the edits made on the global page
		    game_mode_tid = g.title_id;
		    show_game_values(game_mode_tid);
		    ui->set_game_mode_name(slint::SharedString(g.name));
		    ui->set_game_mode_serial(slint::SharedString(g.title_id));
		    ui->set_game_mode_from_library(from_library);
		    ui->set_settings_tab(0);
		    ui->set_game_mode(true);
	    });
	ui->on_leave_game_settings(
	    [&]
	    {
		    if (game_mode_tid.empty())
		    {
			    return;
		    }
		    game_mode_tid.clear();
		    push_emu(s);
		    ui->set_game_mode(false);
	    });
	ui->on_reset_game_settings(
	    [&]
	    {
		    if (game_mode_tid.empty())
		    {
			    return;
		    }
		    if (reset_game_overrides(paths, game_mode_tid))
		    {
			    show_game_values(game_mode_tid);
			    ui->set_status(slint::SharedString("Settings of " + game_mode_tid + " reset to global"));
		    } else
		    {
			    ui->set_status(slint::SharedString("Could not reset the settings of " + game_mode_tid));
		    }
	    });

	// ---- STEP6:6f: Controls page (keyboard/mouse and gamepad bindings) ----
	// The page edits a working copy of Settings::host_input_mapping; Save writes it into `s` and saves the file.
	auto input_mapping = std::make_shared<InputMapping>();
	auto capture_timer = std::make_shared<slint::Timer>();
	{
		std::vector<slint::SharedString> labels;
		for (const auto& source: InputMapping::Sources())
		{
			labels.push_back(slint::SharedString(source.label));
		}
		ui->global<Controls>().set_pad_sources(std::make_shared<slint::VectorModel<slint::SharedString>>(labels));
	}
	auto control_rows = [](const std::vector<ControlBinding>& rows)
	{
		std::vector<ControlRow> out;
		out.reserve(rows.size());
		for (const auto& r: rows)
		{
			ControlRow row;
			row.id            = slint::SharedString(r.id);
			row.label         = slint::SharedString(r.label);
			row.binding       = slint::SharedString(r.binding);
			row.section       = slint::SharedString(r.section);
			row.section_start = r.section_start;
			out.push_back(row);
		}
		return out;
	};
	// "1.5": one decimal without going through the locale's decimal point (snprintf %f would).
	auto one_decimal = [](double v)
	{
		const int tenths = static_cast<int>(std::lround(v * 10.0));
		char      buf[24];
		std::snprintf(buf, sizeof(buf), "%d.%d", tenths / 10, tenths % 10);
		return std::string(buf);
	};
	auto refresh_controls = [ui, input_mapping, control_rows]
	{
		auto& c = ui->global<Controls>();
		c.set_gamepad_rows(std::make_shared<slint::VectorModel<ControlRow>>(control_rows(input_mapping->Rows(true))));
		c.set_keyboard_rows(std::make_shared<slint::VectorModel<ControlRow>>(control_rows(input_mapping->Rows(false))));
	};
	auto stop_controls_capture = [ui, input_mapping, capture_timer]
	{
		capture_timer->stop();
		input_mapping->CancelCapture();
		auto& c = ui->global<Controls>();
		c.set_listening(false);
		c.set_capture_id("");
		c.set_capture_label("");
		c.set_capture_message("");
	};
	ui->global<Controls>().on_opened(
	    [&]
	    {
		    auto& c = ui->global<Controls>();
		    input_mapping->Load(s);
		    c.set_tab(0);
		    c.set_listening(false);
		    c.set_capture_id("");
		    c.set_capture_label("");
		    c.set_capture_message("");
		    c.set_notice("");
		    c.set_sensitivity(static_cast<float>(input_mapping->Sensitivity()));
		    c.set_sensitivity_text(slint::SharedString(one_decimal(input_mapping->Sensitivity()) + "\xc3\x97"));
		    c.set_skin_index(input_mapping->SkinIndex());
		    c.set_show_cursor(input_mapping->ShowCursor());
		    c.set_status(slint::SharedString(input_mapping->PadStatus()));
		    refresh_controls();
	    });
	// A click on a binding starts a capture: the timer polls the keyboard/mouse or the gamepad until it has one.
	ui->global<Controls>().on_begin_capture(
	    [&](const slint::SharedString& id, bool pad)
	    {
		    const std::string key = ToStd(id);
		    std::string       error;
		    stop_controls_capture();
		    auto& c = ui->global<Controls>();
		    c.set_capture_id(slint::SharedString(key));
		    c.set_capture_label(slint::SharedString(input_mapping->RowLabel(pad, key)));
		    c.set_listening(true);
		    if (!input_mapping->BeginCapture(pad, key, &error))
		    {
			    c.set_capture_message(slint::SharedString(error));
			    return;
		    }
		    c.set_capture_message(slint::SharedString(pad ? "Press a button or trigger \xe2\x80\x94 Esc cancels"
		                                                   : "Press a key / mouse button \xe2\x80\x94 Esc cancels"));
		    capture_timer->start(
		        slint::TimerMode::Repeated, std::chrono::milliseconds(14),
		        [&]
		        {
			        const CaptureHit hit = input_mapping->PollCapture();
			        auto&             c  = ui->global<Controls>();
			        switch (hit.kind)
			        {
				        case CaptureHit::Kind::None: return;
				        case CaptureHit::Kind::Cancelled: stop_controls_capture(); return;
				        case CaptureHit::Kind::Reserved:
				        case CaptureHit::Kind::Unsupported: c.set_capture_message(slint::SharedString(hit.name)); return;
				        case CaptureHit::Kind::Key:
				        case CaptureHit::Kind::Pad:
				        {
					        const bool pad_hit = hit.kind == CaptureHit::Kind::Pad;
					        input_mapping->Assign(pad_hit, ToStd(c.get_capture_id()), hit.name);
					        refresh_controls();
					        stop_controls_capture();
					        return;
				        }
			        }
		        });
	    });
	// The dropdown of every pad source: the buttons XInput can't report (Guide, Touchpad, paddles, Misc1).
	ui->global<Controls>().on_pick_source(
	    [&](int index)
	    {
		    auto&       c       = ui->global<Controls>();
		    const auto& sources = InputMapping::Sources();
		    if (!c.get_listening() || index < 0 || static_cast<size_t>(index) >= sources.size())
		    {
			    return;
		    }
		    input_mapping->Assign(true, ToStd(c.get_capture_id()), sources[static_cast<size_t>(index)].id);
		    refresh_controls();
		    stop_controls_capture();
	    });
	ui->global<Controls>().on_clear_binding(
	    [&](const slint::SharedString& id, bool pad)
	    {
		    input_mapping->Clear(pad, ToStd(id));
		    refresh_controls();
	    });
	ui->global<Controls>().on_restore_defaults(
	    [&](bool pad)
	    {
		    input_mapping->RestoreDefaults(pad);
		    auto& c = ui->global<Controls>();
		    c.set_sensitivity(static_cast<float>(input_mapping->Sensitivity()));
		    c.set_sensitivity_text(slint::SharedString(one_decimal(input_mapping->Sensitivity()) + "\xc3\x97"));
		    refresh_controls();
	    });
	ui->global<Controls>().on_sensitivity_edited(
	    [&](float value)
	    {
		    const double rounded = std::round(std::clamp(static_cast<double>(value), 0.1, 5.0) * 10.0) / 10.0;
		    auto&        c       = ui->global<Controls>();
		    c.set_sensitivity(static_cast<float>(rounded));
		    c.set_sensitivity_text(slint::SharedString(one_decimal(rounded) + "\xc3\x97"));
	    });
	ui->global<Controls>().on_cancel_capture(stop_controls_capture);
	ui->global<Controls>().on_save(
	    [&]
	    {
		    auto&    c      = ui->global<Controls>();
		    Settings edited = s;
		    input_mapping->SetTouchOptions(static_cast<double>(c.get_sensitivity()), c.get_skin_index(), c.get_show_cursor());
		    input_mapping->SaveMapping(edited);
		    const std::vector<std::string> before = s.host_input_mapping;
		    s.host_input_mapping                  = edited.host_input_mapping;
		    if (!save())
		    {
			    s.host_input_mapping = before;
			    c.set_notice(slint::SharedString("Could not save the controls"));
			    ui->set_status(slint::SharedString("Could not save the controls"));
			    return;
		    }
		    c.set_notice("");
		    stop_controls_capture();
		    ui->set_show_controls(false);
		    ui->set_status(slint::SharedString("Controls saved"));
	    });
	// Cancel and Back leave the page without writing anything.
	ui->global<Controls>().on_cancel(
	    [&]
	    {
		    stop_controls_capture();
		    ui->set_show_controls(false);
	    });
	ui->global<Controls>().on_close(
	    [&]
	    {
		    stop_controls_capture();
		    ui->set_show_controls(false);
	    });

	ui->on_open_console(
	    [console, log_model, pending_lines, weak_app = slint::ComponentWeakHandle<AppWindow>(ui), &paths]
	    {
		    if (!console->has_value())
		    {
			    auto c = ConsoleWindow::create();
			    c->set_log_lines(log_model);
			    // STEP6:6a: clear drops queued lines too and starts following again.
			    c->on_clear(
			        [log_model, pending_lines, weak_console = slint::ComponentWeakHandle<ConsoleWindow>(c)]
			        {
				        log_model->clear();
				        pending_lines->clear();
				        if (auto h = weak_console.lock())
				        {
					        (*h)->invoke_scroll_to_end();
				        }
			        });
			    // Last log: replace the view with the tail of logs/last-run.log (the full
			    // output, unlike the trimmed live buffer). New output keeps appending after it.
			    c->on_open_last_log(
			        [log_model, weak_app, &paths, weak_console = slint::ComponentWeakHandle<ConsoleWindow>(c)]
			        {
				        const fs::path file = paths.logs_dir / "last-run.log";
				        std::error_code ec;
				        if (!fs::exists(file, ec))
				        {
					        if (auto h = weak_app.lock())
					        {
						        (*h)->set_status(slint::SharedString("Last log not found: " + path_utf8(file)));
					        }
					        return;
				        }
				        const std::string        text = read_text_file(path_utf8(file));
				        std::vector<std::string> all;
				        for (size_t b = 0;;)
				        {
					        const size_t e = text.find('\n', b);
					        std::string  line =
					            e == std::string::npos ? text.substr(b) : text.substr(b, e - b);
					        if (!line.empty() && line.back() == '\r')
					        {
						        line.pop_back();
					        }
					        all.push_back(std::move(line));
					        if (e == std::string::npos)
					        {
						        break;
					        }
					        b = e + 1;
				        }
				        if (!all.empty() && all.back().empty())
				        {
					        all.pop_back(); // trailing newline is not a row
				        }
				        const size_t from = all.size() > kMaxLogLines ? all.size() - kMaxLogLines : 0;
				        std::vector<slint::SharedString> rows;
				        rows.reserve(all.size() - from);
				        for (size_t i = from; i < all.size(); i++)
				        {
					        rows.emplace_back(all[i]);
				        }
				        log_model->set_vector(std::move(rows));
				        if (auto h = weak_console.lock())
				        {
					        (*h)->invoke_scroll_to_end();
				        }
			        });
			    *console = c;
		    }
		    (**console)->show();
		    (**console)->invoke_scroll_to_end();
	    });

	ui->on_open_url(
	    [](const slint::SharedString& url)
	    {
		    open_in_system(ToStd(url));
	    });

	ui->on_dry_run(
	    [&]
	    {
		    save();
		    append_text("---- dry run ----\n" + describe(make_launch_plan(s, paths)));
		    ui->set_status("Dry run: nothing started");
	    });

	ChildProcess child;
	auto         weak = slint::ComponentWeakHandle<AppWindow>(ui);

	// Play time: the session running now, and the one place a finished one is written. Only the UI thread
	// touches it, so `play_times` is always that thread's own copy; the key is taken when the game starts,
	// because the list selection can name another game by the time it exits.
	struct Session
	{
		std::string                           key;
		std::chrono::steady_clock::time_point start;
	};
	auto session        = std::make_shared<std::optional<Session>>();
	auto finish_session = std::make_shared<std::function<void()>>();
	*finish_session     = [session, play_times, play_time_file, games, resort, rebuild, restore_selection]
	{
		if (!session->has_value())
		{
			return;
		}
		// Taken and reset at once: the exit callback and the close path both come here, and only one records.
		const Session run = **session;
		session->reset();
		const auto elapsed =
			std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - run.start).count();
		// Sleeping or hibernating with the game open counts as play: steady_clock keeps running. Only a session
		// shorter than kMinPlaySession (a crash on boot) is dropped.
		if (elapsed < kMinPlaySession.count())
		{
			return;
		}
		const int64_t now = static_cast<int64_t>(
			std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
		*play_times = record_play_session(play_time_file, run.key, static_cast<uint64_t>(elapsed), now);
		apply_play_times(*games, *play_times);
		resort();
		rebuild();
		restore_selection();
	};

	auto play = [&]
	{
		    save();
		    LaunchPlan plan = make_launch_plan(s, paths);
		    std::string pre = "---- play ----\n";
		    for (const auto& w: plan.warnings)
		    {
			    pre += "warning: " + w + "\n";
		    }
		    append_text(pre + command_line(plan) + "\n");

		    // The reader thread keeps the unfinished last line until more output arrives.
		    auto partial   = std::make_shared<std::string>();
		    auto on_output = [partial, append_text](std::string chunk)
		    {
			    std::string text = *partial + chunk;
			    size_t      cut  = text.rfind('\n');
			    if (cut == std::string::npos)
			    {
				    *partial = std::move(text);
				    return;
			    }
			    *partial = text.substr(cut + 1);
			    text.resize(cut);
			    std::string clean;
			    clean.reserve(text.size());
			    for (char c: text)
			    {
				    if (c != '\r')
				    {
					    clean += c;
				    }
			    }
			    slint::invoke_from_event_loop([append_text, clean] { append_text(clean); });
		    };
		    auto on_exit = [partial, append_text, weak, trophy_after_exit, finish_session, flush](int code)
		    {
			    std::string rest = *partial;
			    slint::invoke_from_event_loop(
			        [append_text, weak, rest, code, trophy_after_exit, finish_session, flush]
			        {
				        if (!rest.empty())
				        {
					        append_text(rest);
				        }
				        append_text("---- emulator exited, code " + std::to_string(code) + " ----");
				        (*flush)(); // STEP6:6a: show the last lines now, not in 30 ms
				        // The session is over: add its time to the totals. This runs even when the window handle is gone,
				        // and is a no-op when the close path recorded it first.
				        (*finish_session)();
				        if (auto h = weak.lock())
				        {
					        (*h)->set_running(false);
					        // STEP6:6b: the game may have written unlock files; refresh the trophy page if it is open.
					        if (*trophy_after_exit)
					        {
						        (*trophy_after_exit)();
					        }
					        (*h)->set_status(slint::SharedString("Exited with code " + std::to_string(code)));
					        // The game may have left the window half-painted, so repaint everything once it has exited.
					        (*h)->window().request_redraw();
				        }
			        });
		    };

		    std::string error;
		    if (!child.Start(plan, on_output, on_exit, &error))
		    {
			    append_text("error: " + error);
			    ui->set_status(slint::SharedString(error));
			    return;
		    }
		    ui->set_running(true);
		    // The launched game's key, taken now: the selection may name another game by the time it exits.
		    session->emplace(Session{play_time_key(ToStd(ui->get_game())), std::chrono::steady_clock::now()});
		    {
			    // Name shown in "<game> is running": the library entry, else the game's folder.
			    const std::string eboot = ToStd(ui->get_game());
			    std::string       name  = path_utf8(utf8_path(eboot).parent_path().filename());
			    for (const auto& g: *games)
			    {
				    if (path_key(g.eboot) == path_key(eboot) && !g.name.empty())
				    {
					    name = g.name;
					    break;
				    }
			    }
			    ui->set_running_name(slint::SharedString(name));
		    }
		    const std::string log_file = plan.log_file.empty() ? path_utf8(utf8_path(plan.cwd) / "last-run.log") : plan.log_file;
		    ui->set_status(slint::SharedString("Running - output is also in " + log_file));
	};
	ui->on_play(play);
	ui->on_play_game(
	    [&](int i)
	    {
		    if (i < 0 || static_cast<size_t>(i) >= games->size())
		    {
			    return;
		    }
		    ui->set_selected(i);
		    ui->set_game(slint::SharedString((*games)[static_cast<size_t>(i)].eboot));
		    if (!child.Running())
		    {
			    play();
		    }
	    });
	ui->on_stop(
	    [&]
	    {
		    child.Stop();
		    ui->set_status("Stopping…");
	    });

	// Closing the window while a game runs asks first (the game ends with the launcher).
	ui->window().on_close_requested(
	    [&]
	    {
		    if (child.Running())
		    {
			    ui->set_close_prompt_open(true);
			    return slint::CloseRequestResponse::KeepWindowShown;
		    }
		    slint::quit_event_loop();
		    return slint::CloseRequestResponse::HideWindow;
	    });
	ui->on_confirm_close(
	    [&]
	    {
		    child.Stop();
		    // Stop() has ended the process, so the session is final now. The exit callback posted by on_exit may never
		    // run once the event loop quits; finish_session is a no-op when it already did.
		    (*finish_session)();
		    slint::quit_event_loop();
	    });

#ifdef _WIN32
	// Right-click menus are drawn by Slint, not native Win32 menus: a native context menu runs a
	// modal loop, so the window does not repaint the row the right-click selected, and a
	// right-click on another row only closes the menu. SLINT_NO_MUDA is read every time a
	// context menu opens, but the menu bar is set up once when the window is shown, so setting
	// it after that keeps the native Windows menu bar.
	slint::Timer::single_shot(std::chrono::milliseconds(0), [] { SetEnvironmentVariableW(L"SLINT_NO_MUDA", L"1"); });
#elif defined(__APPLE__)
	// Same on macOS: native context menus, but keep the native menu bar.
	slint::Timer::single_shot(std::chrono::milliseconds(0), [] { setenv("SLINT_NO_MUDA", "1", 1); });
#endif
	// STEP6:6e: the window's last size and position (window_x/y/window_w/window_h). Sizes that don't look like a
	// real window, or a position off the primary monitor (whose rect is taken as 0,0..w,h), fall back to the
	// Slint defaults: the page's preferred-width/height, placed by the window manager.
	const bool restore_size = s.window_w >= 800 && s.window_h >= 500;
	if (restore_size)
	{
		ui->window().set_size(slint::LogicalSize(slint::Size<float>{static_cast<float>(s.window_w), static_cast<float>(s.window_h)}));
	}
	if (restore_size && s.window_x >= 0 && s.window_y >= 0 && have_monitor && s.window_x < monitor_w && s.window_y < monitor_h)
	{
		ui->window().set_position(slint::LogicalPosition(slint::Point<float>{static_cast<float>(s.window_x), static_cast<float>(s.window_y)}));
	}
	refresh();
	ui->run();
	// STEP6:6e: keep the geometry for the next start. Read once, after run() returns: the native window is gone
	// but the component handle still reports its last size and position (never saved on every resize).
	{
		const float                   scale  = ui->window().scale_factor();
		const float                   factor = scale > 0.01f ? scale : 1.0f;
		const slint::PhysicalSize     size   = ui->window().size();
		const slint::PhysicalPosition pos    = ui->window().position();
		const int w = static_cast<int>(std::lround(static_cast<float>(size.width) / factor));
		const int h = static_cast<int>(std::lround(static_cast<float>(size.height) / factor));
		// A platform that answers nothing after the window closed leaves the saved values alone.
		if (w >= 800 && h >= 500)
		{
			s.window_w = w;
			s.window_h = h;
			s.window_x = static_cast<int>(std::lround(static_cast<float>(pos.x) / factor));
			s.window_y = static_cast<int>(std::lround(static_cast<float>(pos.y) / factor));
		}
	}
	save();
	if (scan_thread.joinable())
	{
		scan_thread.join();
	}
	// STEP6:6c: the image pool had its threads for the whole session; join them before the window goes.
	pool->Stop();
	// Closing stopped the game; wait for its output to finish (into the log file).
	child.Wait();
	return 0;
}
