#include "version_manager.h"

#include "library.h"
#include "net.h"
#include "platform.h"
#include "secret_store.h"
#include "versions.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace kyty_launcher {

namespace {

std::string ToStd(const slint::SharedString& s)
{
	return std::string(std::string_view(s));
}

} // namespace

// Everything runs on the UI thread; worker threads only hand results back through
// slint::invoke_from_event_loop with a strong reference to this object.
struct VersionManager::Impl: std::enable_shared_from_this<Impl>
{
	Impl(const slint::ComponentHandle<AppWindow>& window, Settings& settings, std::string default_dir, std::string cache, SaveFn save_fn, PickFn pick_fn)
	    : ui(window), s(settings), default_versions_dir(std::move(default_dir)), cache_dir(std::move(cache)), save(std::move(save_fn)), pick(std::move(pick_fn))
	{
	}

	slint::ComponentWeakHandle<AppWindow> ui;
	Settings&                             s;
	std::string                           default_versions_dir;
	std::string                           cache_dir;
	SaveFn                                save;
	PickFn                                pick;

	PinCountFn                  pin_counter;
	std::function<void()>       builds_changed;
	std::vector<InstalledBuild> installed;
	std::vector<size_t>         menu_map; // toolbar menu row -> index in `installed`
	std::vector<size_t>         row_map;  // Installed table row (after the search and the tracked filter) -> index in `installed`
	// Selected builds of the Installed table, by path_key of the exe; the focused one is current-installed,
	// the anchor is where a Shift-click range starts.
	std::set<std::string>       selected;
	std::set<std::string>       marquee_base;
	std::string                 focus_key;
	std::string                 anchor_key;
	std::vector<ReleaseInfo>    releases; // of the current source
	bool                        busy = false;
	// The build whose About dialog is open: its exe ("Go to file") and release page ("Open link").
	std::string                 about_path;
	std::string                 about_url;

	// "<tab>-latest" builds and the source each is downloaded from.
	using LatestTarget = std::pair<std::string, InstalledBuild>; // repo, build
	std::vector<LatestTarget> offered;  // builds in the update popup
	std::set<std::string>     notified; // "folder|tag" already offered this session (not offered again)
	slint::Timer              update_timer;

	// Release sources: the default KytyPS5 repo first, then the ones added with +.
	struct Source
	{
		std::string repo; // owner/repo
		std::string name; // tab title
	};
	std::vector<Source>                              sources;
	int                                              current_source = 0;
	std::shared_ptr<slint::VectorModel<slint::SharedString>> sources_model = std::make_shared<slint::VectorModel<slint::SharedString>>();

	std::shared_ptr<slint::VectorModel<EmuVersion>>          installed_model = std::make_shared<slint::VectorModel<EmuVersion>>();
	std::shared_ptr<slint::VectorModel<ReleaseItem>>         available_model = std::make_shared<slint::VectorModel<ReleaseItem>>();
	std::shared_ptr<slint::VectorModel<ReleaseItem>>         older_model     = std::make_shared<slint::VectorModel<ReleaseItem>>();
	std::shared_ptr<slint::VectorModel<slint::SharedString>> dropdown_model  = std::make_shared<slint::VectorModel<slint::SharedString>>();
	// Every installed build for the pin picker: not filtered by the search box or the kind selector (unlike installed_model).
	std::shared_ptr<slint::VectorModel<BuildChoice>>         pin_model      = std::make_shared<slint::VectorModel<BuildChoice>>();

	// Worker threads, joined in Shutdown; each gets a token that CancelAll sets. Started from the UI
	// thread only. Finished ones are joined and dropped before the next one starts, so an hourly
	// update check does not pile threads up for the whole session.
	struct Worker
	{
		std::thread                        thread;
		std::shared_ptr<std::atomic<bool>> token;
		std::shared_ptr<std::atomic<bool>> done;
	};
	std::vector<Worker> workers;

	// Joins the workers that returned, and forgets them (and their tokens).
	void ReapFinished()
	{
		for (auto& w: workers)
		{
			if (w.done->load() && w.thread.joinable())
			{
				w.thread.join();
			}
		}
		std::erase_if(workers, [](const Worker& w) { return w.done->load(); });
	}

	// Runs fn(token) on a new thread; the work polls *token to stop early.
	template<class F>
	void Spawn(F fn)
	{
		ReapFinished();
		Worker w;
		w.token = std::make_shared<std::atomic<bool>>(false);
		w.done  = std::make_shared<std::atomic<bool>>(false);
		const auto token = w.token;
		const auto done  = w.done;
		w.thread         = std::thread(
		    [fn = std::move(fn), token, done]() mutable
		    {
			    fn(token);
			    done->store(true);
		    });
		workers.push_back(std::move(w));
	}

	void CancelAll()
	{
		for (const auto& w: workers)
		{
			if (w.token)
			{
				w.token->store(true);
			}
		}
	}

	// The launcher is closing: stop the hourly check and the workers, and wait for them (a request in
	// flight ends within its timeout).
	void Shutdown()
	{
		update_timer.stop();
		CancelAll();
		for (auto& w: workers)
		{
			if (w.thread.joinable())
			{
				w.thread.join();
			}
		}
		workers.clear();
	}

	// Display name of a build: the name given with Rename, else its folder name.
	std::string DisplayName(const InstalledBuild& b) const
	{
		for (const auto& item: s.build_names)
		{
			size_t bar = item.find('|');
			if (bar != std::string::npos && path_key(item.substr(0, bar)) == path_key(b.exe))
			{
				return item.substr(bar + 1);
			}
		}
		return b.name;
	}

	// The default KytyPS5-main build is always the first installed row. Rename may change its name, so
	// the folder it was downloaded into identifies it; a path without a folder falls back to the name.
	static bool IsMainLatest(const InstalledBuild& b)
	{
		const std::string folder = path_utf8(utf8_path(b.exe).parent_path().filename());
		return path_key(folder.empty() ? b.name : folder) == path_key("KytyPS5-main-latest");
	}

	static std::string Lower(std::string v)
	{
		std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return v;
	}

	static int CompareText(const std::string& a, const std::string& b)
	{
		const std::string la = Lower(a);
		const std::string lb = Lower(b);
		return la < lb ? -1 : la > lb ? 1 : 0;
	}

	// The persisted Installed table sort key; anything unknown means the display name.
	static std::string SortKeyName(const std::string& key)
	{
		return key == "codename" || key == "date" || key == "tracked" || key == "path" ? key : "name";
	}

	bool IsPinned(const InstalledBuild& b) const
	{
		return std::any_of(s.pinned_builds.begin(), s.pinned_builds.end(),
		                   [&](const std::string& p) { return path_key(p) == path_key(b.exe); });
	}

	size_t PinPosition(const std::string& exe) const
	{
		for (size_t i = 0; i < s.pinned_builds.size(); i++)
		{
			if (path_key(s.pinned_builds[i]) == path_key(exe))
			{
				return i;
			}
		}
		return s.pinned_builds.size();
	}

	// Main-latest first, then the pinned builds in their saved order, then the chosen sort. The
	// tie-break (display name, then path) keeps the order deterministic for equal keys.
	void SortInstalled()
	{
		const std::string key  = SortKeyName(s.vm_sort);
		const bool        desc = s.vm_sort_desc;
		auto              by_key = [&](const InstalledBuild& a, const InstalledBuild& b) -> int
		{
			if (key == "codename")
			{
				return CompareText(a.codename, b.codename);
			}
			if (key == "date")
			{
				return CompareText(a.date, b.date);
			}
			if (key == "tracked")
			{
				return a.managed == b.managed ? 0 : a.managed ? -1 : 1;
			}
			if (key == "path")
			{
				return CompareText(path_key(a.exe), path_key(b.exe));
			}
			return CompareText(DisplayName(a), DisplayName(b));
		};
		std::stable_sort(
		    installed.begin(), installed.end(),
		    [&](const InstalledBuild& a, const InstalledBuild& b)
		    {
			    const bool a_main = IsMainLatest(a);
			    const bool b_main = IsMainLatest(b);
			    if (a_main != b_main)
			    {
				    return a_main;
			    }
			    // Main-latest cannot be pinned (it is always first anyway); its pins are left alone.
			    if (!a_main)
			    {
				    const bool a_pin = IsPinned(a);
				    const bool b_pin = IsPinned(b);
				    if (a_pin != b_pin)
				    {
					    return a_pin;
				    }
				    if (a_pin && b_pin)
				    {
					    const size_t a_pos = PinPosition(a.exe);
					    const size_t b_pos = PinPosition(b.exe);
					    if (a_pos != b_pos)
					    {
						    return a_pos < b_pos;
					    }
				    }
			    }
			    int c = by_key(a, b);
			    if (desc)
			    {
				    c = -c;
			    }
			    if (c != 0)
			    {
				    return c < 0;
			    }
			    c = CompareText(DisplayName(a), DisplayName(b));
			    if (c != 0)
			    {
				    return c < 0;
			    }
			    return path_key(a.exe) < path_key(b.exe);
		    });
	}

	// The KytyPS5-main that comes by default (the default emulator path) keeps its name.
	static bool IsOriginalMain(const InstalledBuild& b)
	{
		const std::string def = default_settings().emulator;
		return !def.empty() && path_key(b.exe) == path_key(def);
	}

	void Rename(int index, std::string name)
	{
		if (index < 0 || static_cast<size_t>(index) >= installed.size() || IsOriginalMain(installed[static_cast<size_t>(index)]))
		{
			return;
		}
		std::erase_if(name, [](char c) { return c == '|' || c == '\r' || c == '\n'; });
		const std::string        exe = installed[static_cast<size_t>(index)].exe;
		std::vector<std::string> kept;
		for (const auto& item: s.build_names)
		{
			size_t bar = item.find('|');
			if (bar == std::string::npos || path_key(item.substr(0, bar)) != path_key(exe))
			{
				kept.push_back(item);
			}
		}
		if (!name.empty())
		{
			kept.push_back(exe + "|" + name);
		}
		s.build_names = std::move(kept);
		save();
		RefreshInstalled();
	}

	std::string VersionsDir() const { return s.versions_dir.empty() ? default_versions_dir : s.versions_dir; }
	// Release lists are kept on disk (next to the settings), not in memory, between fetches.
	std::string CacheDir() const { return cache_dir; }

	// "3 lists, 41 KB" on the Settings tab.
	void ShowCacheInfo()
	{
		int      files = 0;
		uint64_t bytes = 0;
		release_cache_size(CacheDir(), &files, &bytes);
		const std::string info = files == 0 ? std::string("No saved release lists")
		                                    : std::to_string(files) + " saved release list" + (files == 1 ? "" : "s") + ", " +
		                                          std::to_string((bytes + 1023) / 1024) + " KB in " + CacheDir();
		WithUi([&](AppWindow&, const VM& vm) { vm.set_cache_info(slint::SharedString(info)); });
	}

	// Runs f(window, vm global) when the window still exists.
	template <typename F> void WithUi(F&& f)
	{
		if (auto h = ui.lock())
		{
			f(**h, (*h)->template global<VM>());
		}
	}

	// The status line of the page, and of the main window (prefixed, so it reads on its own there).
	// Page-only lines are list counts and hints that mean nothing outside the page.
	void Status(const std::string& text, bool page_only = false)
	{
		WithUi(
		    [&](AppWindow& w, const VM& vm)
		    {
			    vm.set_status(slint::SharedString(text));
			    if (!page_only)
			    {
				    w.set_status(slint::SharedString("Version Manager: " + text));
			    }
		    });
	}

	void SetBusy(bool value)
	{
		busy = value;
		WithUi([&](AppWindow&, const VM& vm) { vm.set_busy(value); });
	}

	bool IsInstalled(const ReleaseInfo& r) const
	{
		return std::any_of(installed.begin(), installed.end(), [&](const InstalledBuild& b) { return b.tag == r.tag || b.name == r.folder; });
	}

	static std::string TabName(const std::string& repo)
	{
		if (repo == kDefaultRepo)
		{
			return "KytyPS5-main";
		}
		size_t slash = repo.find('/');
		if (slash == std::string::npos)
		{
			return repo;
		}
		// A fork that keeps the project's name (owner/KytyPS5) is told apart by its owner.
		std::string name = repo.substr(slash + 1);
		std::string lower = name;
		std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return lower == "kytyps5" ? repo.substr(0, slash) : name;
	}

	// The tab title of an added source: the name given with Rename, else the repo's name.
	std::string SourceTitle(const std::string& repo) const
	{
		for (const auto& item: s.source_names)
		{
			const size_t bar = item.find('|');
			if (bar != std::string::npos && item.substr(0, bar) == repo)
			{
				return item.substr(bar + 1);
			}
		}
		return TabName(repo);
	}

	// Renames the tab of the added source at `index` (the default source keeps its name).
	void RenameSource(int index, std::string name)
	{
		if (index <= 0 || static_cast<size_t>(index) >= sources.size())
		{
			return;
		}
		std::erase_if(name, [](char c) { return c == '|' || c == '\r' || c == '\n'; });
		const std::string        repo = sources[static_cast<size_t>(index)].repo;
		std::vector<std::string> kept;
		for (const auto& item: s.source_names)
		{
			const size_t bar = item.find('|');
			if (bar == std::string::npos || item.substr(0, bar) != repo)
			{
				kept.push_back(item);
			}
		}
		if (!name.empty())
		{
			kept.push_back(repo + "|" + name);
		}
		s.source_names = std::move(kept);
		save();
		BuildSources();
	}

	void BuildSources()
	{
		sources.clear();
		sources.push_back({kDefaultRepo, TabName(kDefaultRepo)});
		for (const auto& repo: s.sources)
		{
			sources.push_back({repo, SourceTitle(repo)});
		}
		std::vector<slint::SharedString> names;
		for (const auto& src: sources)
		{
			names.emplace_back(src.name);
		}
		sources_model->set_vector(names);
		current_source = std::min(current_source, static_cast<int>(sources.size()) - 1);
		WithUi([&](AppWindow&, const VM& vm) { vm.set_current_source(current_source); });
	}

	void ShowReleases()
	{
		std::vector<ReleaseItem> items;
		for (const auto& r: releases)
		{
			const bool installed_already = IsInstalled(r);
			ReleaseItem item;
			item.text      = slint::SharedString(r.date + "  " + r.codename + (installed_already ? "  (installed)" : ""));
			item.installed = installed_already;
			items.push_back(item);
		}
		older_model->set_vector(items.size() > 1 ? std::vector<ReleaseItem>(items.begin() + 1, items.end()) : std::vector<ReleaseItem> {});
		available_model->set_vector(std::move(items));
	}

	void RefreshInstalled()
	{
		std::vector<std::string> extra = s.custom_builds;
		extra.push_back(s.emulator);
		installed = scan_installed(VersionsDir(), extra, s.hidden_builds);
		ApplyInstalled();
		ShowReleases();
	}

	// The index in `installed` of table row `row` (-1 when there is none).
	int RealIndex(int row) const
	{
		return row >= 0 && static_cast<size_t>(row) < row_map.size() ? static_cast<int>(row_map[static_cast<size_t>(row)]) : -1;
	}

	// Fills the Installed table and the toolbar menu from `installed`, with the search box and the
	// Any / Tracked / Untracked selector applied to the table.
	void ApplyInstalled()
	{
		// The order is the table order, the toolbar menu order and what RealIndex maps into.
		SortInstalled();
		std::string query;
		int         kind = 0;
		WithUi(
		    [&](AppWindow&, const VM& vm)
		    {
			    query = ToStd(vm.get_installed_filter());
			    kind  = vm.get_installed_kind();
			    vm.set_sort_key(slint::SharedString(SortKeyName(s.vm_sort)));
			    vm.set_sort_desc(s.vm_sort_desc);
		    });
		std::transform(query.begin(), query.end(), query.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		auto matches = [&](const InstalledBuild& b)
		{
			if ((kind == 1 && !b.managed) || (kind == 2 && b.managed))
			{
				return false;
			}
			if (query.empty())
			{
				return true;
			}
			std::string hay = DisplayName(b) + "\n" + b.codename + "\n" + b.date + "\n" + b.exe + "\n" + b.tag;
			std::transform(hay.begin(), hay.end(), hay.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return hay.find(query) != std::string::npos;
		};

		std::vector<EmuVersion>          rows;
		std::vector<slint::SharedString> names;
		int                              current = -1;
		const auto&                      unticked = s.menu_hidden;
		menu_map.clear();
		row_map.clear();
		for (size_t i = 0; i < installed.size(); i++)
		{
			const auto& b = installed[i];
			EmuVersion  v;
			v.name     = slint::SharedString(DisplayName(b));
			v.renamable = !IsOriginalMain(b);
			v.codename = slint::SharedString(b.codename);
			v.date     = slint::SharedString(b.date);
			v.path     = slint::SharedString(b.exe);
			v.tracked  = b.managed;
			v.active   = path_key(b.exe) == path_key(s.emulator);
			v.shown    = std::none_of(unticked.begin(), unticked.end(), [&](const std::string& u) { return path_key(u) == path_key(b.exe); });
			v.selected = selected.count(path_key(b.exe)) != 0;
			const bool main_latest = IsMainLatest(b);
			v.pinned               = !main_latest && IsPinned(b);
			v.main_latest          = main_latest;
			if (matches(b))
			{
				rows.push_back(v);
				row_map.push_back(i);
			}
			// The ticked builds are in the toolbar's build list; the running one is always in
			// it, so the list is never empty and always shows what the launcher runs.
			if (v.shown || v.active)
			{
				if (v.active)
				{
					current = static_cast<int>(menu_map.size());
				}
				menu_map.push_back(i);
				names.emplace_back(b.codename == "Local" ? DisplayName(b) + " (local)" : DisplayName(b));
			}
		}
		installed_model->set_vector(rows);
		dropdown_model->set_vector(names);
		{
			std::vector<BuildChoice> all;
			for (const auto& b: installed)
			{
				BuildChoice c;
				c.name = slint::SharedString(DisplayName(b) + (b.codename == "Local" ? " (local)" : ""));
				c.path = slint::SharedString(b.exe);
				all.push_back(c);
			}
			pin_model->set_vector(std::move(all));
		}
		WithUi([&](AppWindow& w, const VM&) { w.set_emu_build_index(current); });
		// Builds the filter hides (or that are gone) leave the selection.
		std::set<std::string> visible;
		for (size_t r = 0; r < row_map.size(); r++)
		{
			visible.insert(path_key(installed[row_map[r]].exe));
		}
		std::erase_if(selected, [&](const std::string& k) { return visible.count(k) == 0; });
		ShowSelection();
		if (builds_changed)
		{
			builds_changed();
		}
	}

	std::string BuildName(const std::string& exe) const
	{
		for (const auto& b: installed)
		{
			if (path_key(b.exe) == path_key(exe))
			{
				return DisplayName(b);
			}
		}
		return build_display_name(s, exe);
	}

	// Makes the build at `index` the one the launcher runs. The build does not have to be
	// shown in the toolbar's build list: it can be run while it is unticked.
	void Activate(int index)
	{
		if (index < 0 || static_cast<size_t>(index) >= installed.size())
		{
			return;
		}
		const InstalledBuild b = installed[static_cast<size_t>(index)];
		if (path_key(b.exe) == path_key(s.emulator))
		{
			Status("Already using " + DisplayName(b));
			return;
		}
		s.emulator = b.exe;
		WithUi([&](AppWindow& w, const VM&) { w.set_emulator(slint::SharedString(b.exe)); });
		save();
		RefreshInstalled();
		Status("Using " + DisplayName(b));
	}

	// A click on an Installed header: the same key flips the direction, a new key sorts ascending.
	void SortBy(const std::string& key)
	{
		const std::string name = SortKeyName(key);
		if (name == s.vm_sort)
		{
			s.vm_sort_desc = !s.vm_sort_desc;
		} else
		{
			s.vm_sort      = name;
			s.vm_sort_desc = false;
		}
		save();
		ApplyInstalled();
	}

	// Right-click > Pin to top / Unpin. Pins only change the Installed order (main-latest stays first).
	void TogglePin(int index)
	{
		if (index < 0 || static_cast<size_t>(index) >= installed.size())
		{
			return;
		}
		const InstalledBuild b = installed[static_cast<size_t>(index)];
		if (IsMainLatest(b))
		{
			Status(DisplayName(b) + " always stays first.");
			return;
		}
		auto&      pins  = s.pinned_builds;
		const auto it    = std::find_if(pins.begin(), pins.end(), [&](const std::string& p) { return path_key(p) == path_key(b.exe); });
		const bool unpin = it != pins.end();
		if (unpin)
		{
			pins.erase(it);
		} else
		{
			pins.push_back(b.exe);
		}
		save();
		ApplyInstalled();
		Status((unpin ? "Unpinned " : "Pinned ") + DisplayName(b));
	}

	// ---- About of one build: size and config files next to its exe

	// Total size of the build folder; an unreadable folder counts as 0.
	static uint64_t FolderSize(const std::filesystem::path& dir)
	{
		std::error_code ec;
		if (!std::filesystem::is_directory(dir, ec))
		{
			return 0;
		}
		uint64_t total = 0;
		std::filesystem::recursive_directory_iterator it(dir, std::filesystem::directory_options::skip_permission_denied, ec);
		const std::filesystem::recursive_directory_iterator end;
		while (it != end)
		{
			if (ec)
			{
				ec.clear();
			} else
			{
				std::error_code file_ec;
				if (it->is_regular_file(file_ec))
				{
					total += it->file_size(file_ec);
				}
			}
			it.increment(ec);
		}
		return total;
	}

	// The repo a managed build came from: its "-latest" source when known, else the default repo for a
	// tagged build under the versions folder. "" for a local build.
	std::string AboutRepo(const InstalledBuild& b) const
	{
		if (!b.managed)
		{
			return {};
		}
		const std::string latest = LatestRepo(b);
		if (!latest.empty())
		{
			return latest;
		}
		return b.tag.empty() ? std::string() : std::string(kDefaultRepo);
	}

	void ShowAbout(int row)
	{
		const int index = RealIndex(row);
		if (index < 0 || static_cast<size_t>(index) >= installed.size())
		{
			return;
		}
		const InstalledBuild b    = installed[static_cast<size_t>(index)];
		const auto          dir   = utf8_path(b.exe).parent_path();
		const std::string   repo  = AboutRepo(b);
		BuildAbout          about;
		about.title  = slint::SharedString(DisplayName(b));
		about.author = slint::SharedString(repo.empty() ? std::string() : repo.substr(0, repo.find('/')));
		about.date   = slint::SharedString(b.date);
		about.path   = slint::SharedString(b.exe);
		about.size   = slint::SharedString(format_size(FolderSize(dir)));
		// The build's config files: cfgGlobal.json and the cfg<TITLEID>.json files the loader reads.
		std::error_code ec;
		if (!dir.empty() && std::filesystem::is_regular_file(dir / "cfgGlobal.json", ec))
		{
			about.global_config = "cfgGlobal.json";
		}
		std::vector<std::string> configs;
		if (!dir.empty())
		{
			std::filesystem::directory_iterator it(dir, ec);
			const std::filesystem::directory_iterator end;
			for (; !ec && it != end; it.increment(ec))
			{
				std::error_code file_ec;
				if (!it->is_regular_file(file_ec))
				{
					continue;
				}
				const std::string name = path_utf8(it->path().filename());
				const std::string low  = Lower(name);
				if (low.starts_with("cfg") && low.ends_with(".json") && low != "cfgglobal.json")
				{
					configs.push_back(name);
				}
			}
		}
		std::sort(configs.begin(), configs.end(), [](const std::string& a, const std::string& c) { return Lower(a) < Lower(c); });
		const size_t max_configs = 50;
		const size_t total       = configs.size();
		if (configs.size() > max_configs)
		{
			configs.resize(max_configs);
		}
		std::vector<slint::SharedString> cfg_names;
		for (const auto& cfg: configs)
		{
			cfg_names.emplace_back(cfg);
		}
		auto model = std::make_shared<slint::VectorModel<slint::SharedString>>();
		model->set_vector(std::move(cfg_names));
		about.serial_configs = model;
		if (!repo.empty() && !b.tag.empty())
		{
			about.github = slint::SharedString("https://github.com/" + repo + "/releases/tag/" + b.tag);
		}
		about_path = b.exe;
		about_url  = ToStd(about.github);
		WithUi([&](AppWindow&, const VM& vm) { vm.set_about(about); });
		if (total > max_configs)
		{
			Status("Showing the first " + std::to_string(max_configs) + " serial configs of " + std::to_string(total) + ".", true);
		}
	}

	// Right-click > About > Go to file: the folder holding the exe, never the emulator itself.
	void ShowAboutFile()
	{
		if (about_path.empty())
		{
			return;
		}
		const auto dir = utf8_path(about_path).parent_path();
		open_in_system(path_utf8(dir.empty() ? utf8_path(about_path) : dir));
	}

	void OpenAboutLink()
	{
		if (about_url.empty())
		{
			return;
		}
		const std::string url = about_url;
		WithUi([&](AppWindow& w, const VM&) { w.invoke_open_url(slint::SharedString(url)); });
	}

	// Lists the current source's releases in the background (from the disk cache while it is fresh; `force`
	// asks GitHub regardless), then runs `then` on the UI thread.
	void FetchReleasesAsync(std::function<void()> then, bool force = false)
	{
		SetBusy(true);
		Status("Fetching the release list…", true);
		Spawn(
		    [self = shared_from_this(), then = std::move(then), repo = sources[static_cast<size_t>(current_source)].repo, cache = CacheDir(),
		     force](const auto& cancel)
		    {
			    std::string error;
			    std::string note;
			    auto        list = fetch_releases(repo, cache, force, &error, &note, cancel.get());
			    slint::invoke_from_event_loop(
			        [self, then, list = std::move(list), error, note, repo]
			        {
				        self->SetBusy(false);
				        if (list.empty())
				        {
					        self->Status(TabName(repo) + ": " + (error.empty() ? std::string("no Windows builds found") : error));
					        return;
				        }
				        // The user may have switched tabs while this was loading.
				        if (self->sources[static_cast<size_t>(self->current_source)].repo != repo)
				        {
					        return;
				        }
				        self->releases = list;
				        self->ShowReleases();
				        self->Status(TabName(repo) + ": " + std::to_string(list.size()) + " builds available" + (note.empty() ? "" : " (" + note + ")") +
				                         ". Double-click one to download it.",
				                     true);
				        if (then)
				        {
					        then();
				        }
			        });
		    });
	}

	void SelectSource(int index)
	{
		if (index < 0 || static_cast<size_t>(index) >= sources.size() || index == current_source)
		{
			return;
		}
		current_source = index;
		WithUi([&](AppWindow&, const VM& vm) { vm.set_current_source(index); vm.set_current_available(-1); });
		releases.clear();
		ShowReleases();
		if (!busy)
		{
			FetchReleasesAsync(nullptr);
		}
	}

	// Opening the page refreshes both lists: the installed builds and the current source's releases.
	void Opened()
	{
		RefreshInstalled();
		if (!busy && !sources.empty())
		{
			FetchReleasesAsync(nullptr);
		}
	}

	// Leaving the page drops the release lists; Opened fetches them again.
	void Closed()
	{
		releases.clear();
		releases.shrink_to_fit();
		available_model->set_vector({});
		older_model->set_vector({});
		WithUi([&](AppWindow&, const VM& vm) { vm.set_current_available(-1); });
	}

	// Right-click > Stop tracking on a source tab: drops it. The default KytyPS5-main tab stays.
	void StopTracking(int index)
	{
		if (index <= 0 || static_cast<size_t>(index) >= sources.size())
		{
			return;
		}
		const std::string repo = sources[static_cast<size_t>(index)].repo;
		std::vector<std::string> kept;
		for (const auto& src: s.sources)
		{
			if (path_key(src) != path_key(repo))
			{
				kept.push_back(src);
			}
		}
		s.sources = std::move(kept);
		save();
		if (current_source == index)
		{
			current_source = 0;
		} else if (current_source > index)
		{
			current_source--;
		}
		BuildSources(); // rebuilds the tabs from the saved list
		releases.clear();
		ShowReleases();
		Status("Stopped tracking " + repo);
		if (!busy)
		{
			FetchReleasesAsync(nullptr);
		}
	}

	// A link pasted into the + dialog: check that the repo has releases, then add its tab.
	void AddSource(const std::string& link)
	{
		std::string repo = parse_repo_link(link);
		auto        fail = [&](const std::string& text)
		{
			WithUi(
			    [&](AppWindow&, const VM& vm)
			    {
				    vm.set_add_source_busy(false);
				    vm.set_add_source_error(slint::SharedString(text));
			    });
		};
		if (repo.empty())
		{
			fail("That is not a GitHub link. Use something like https://github.com/owner/repo/releases");
			return;
		}
		auto same = [&](const Source& src)
		{
			std::string a = src.repo;
			std::string b = repo;
			for (auto* t: {&a, &b})
			{
				for (auto& ch: *t)
				{
					ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
				}
			}
			return a == b;
		};
		if (std::any_of(sources.begin(), sources.end(), same))
		{
			fail(repo + " is already added.");
			return;
		}
		WithUi(
		    [&](AppWindow&, const VM& vm)
		    {
			    vm.set_add_source_error("");
			    vm.set_add_source_busy(true);
		    });
		// Checked on a worker: the release list, then that the newest zip holds kyty_emulator.exe.
		Spawn(
		    [self = shared_from_this(), repo, cache = CacheDir()](const auto& cancel)
		    {
			    std::string error;
			    auto        list = fetch_releases(repo, cache, false, &error, nullptr, cancel.get());
			    bool        unverified = false; // the zip could not be checked without downloading it
			    if (!list.empty())
			    {
				    // Read only the zip's central directory (HTTP Range) instead of the whole file.
				    std::string probe_error;
				    switch (probe_release_zip(list.front(), &probe_error))
				    {
					    case ZipProbe::HasEmulator: break;
					    case ZipProbe::NoEmulator:
						    error = std::string("The newest release (") + list.front().tag + ") has no " + "kyty_emulator.exe in its zip.";
						    list.clear();
						    break;
					    case ZipProbe::Unsupported:
						    // No Range support (or an odd zip). Adding a tab must not download a whole build, so take the
						    // source as is; Download reports a zip without the emulator.
						    unverified = true;
						    break;
				    }
			    }
			    slint::invoke_from_event_loop(
			        [self, repo, list = std::move(list), error, unverified]
			        {
				        if (list.empty())
				        {
					        self->WithUi(
					            [&](AppWindow&, const VM& vm)
					            {
						            vm.set_add_source_busy(false);
						            vm.set_add_source_error(slint::SharedString(error.empty() ? "No releases with a Windows zip were found in " + repo : error));
					            });
					        return;
				        }
				        self->s.sources.push_back(repo);
				        self->save();
				        self->BuildSources();
				        self->current_source = static_cast<int>(self->sources.size()) - 1;
				        self->releases       = list;
				        self->ShowReleases();
				        self->WithUi(
				            [&](AppWindow&, const VM& vm)
				            {
					            vm.set_current_source(self->current_source);
					            vm.set_current_available(-1);
					            vm.set_add_source_busy(false);
					            vm.set_add_source_open(false);
				            });
				        self->Status("Added " + repo + " (" + std::to_string(list.size()) + " builds)" +
				                     (unverified ? ". Its zips could not be checked for the emulator before downloading." : ""));
			        });
		    });
	}

	// Downloads and unpacks a release, then selects it.
	void Download(ReleaseInfo r, bool latest)
	{
		if (latest)
		{
			// The newest build goes to "<tab>-latest" (e.g. KytyPS5-main-latest), replaced on each download.
			r.folder = TabName(sources[static_cast<size_t>(current_source)].repo) + "-latest";
		}
		SetBusy(true);
		Status("Downloading " + r.tag + "…");
		Spawn(
		    [self = shared_from_this(), r, dir = VersionsDir()](const auto& cancel)
		    {
			    int         last_pct = -1;
			    std::string error;
			    std::string exe = install_release(
			        r, dir,
			        [&](uint64_t done, uint64_t total)
			        {
				        int pct = total > 0 ? static_cast<int>(done * 100 / total) : -1;
				        if (pct == last_pct)
				        {
					        return;
				        }
				        last_pct         = pct;
				        std::string text = "Downloading " + r.tag + ": " + std::to_string(done >> 20) + " / " + std::to_string(total >> 20) + " MB";
				        slint::invoke_from_event_loop([self, text] { self->Status(text); });
			        },
			        &error, cancel.get());
			    slint::invoke_from_event_loop(
			        [self, r, exe, error]
			        {
				        self->SetBusy(false);
				        if (exe.empty())
				        {
					        self->Status(error == "Cancelled" ? "Download cancelled" : "Download failed: " + error);
					        return;
				        }
				        self->Unhide(exe);
				        self->RefreshInstalled();
				        for (size_t i = 0; i < self->installed.size(); i++)
				        {
					        if (path_key(self->installed[i].exe) == path_key(exe))
					        {
						        self->Activate(static_cast<int>(i));
					        }
				        }
				        self->Status("Installed " + r.tag);
			        });
		    });
	}

	// The source a "<tab>-latest" build is downloaded from (Download names the folder after the tab);
	// "" for any other build, or when its source is no longer tracked.
	std::string LatestRepo(const InstalledBuild& b) const
	{
		if (!b.managed)
		{
			return {};
		}
		for (const auto& src: sources)
		{
			if (b.name == TabName(src.repo) + "-latest")
			{
				return src.repo;
			}
		}
		return {};
	}

	// The "-latest" builds, all of them or only the selected ones.
	std::vector<LatestTarget> LatestBuilds(bool selected_only) const
	{
		std::vector<LatestTarget> out;
		for (const auto& b: installed)
		{
			if (selected_only && selected.count(path_key(b.exe)) == 0)
			{
				continue;
			}
			if (std::string repo = LatestRepo(b); !repo.empty())
			{
				out.emplace_back(repo, b);
			}
		}
		return out;
	}

	// Asks each source whether a newer build is out than its "-latest" folder holds (from the saved
	// release list while it is fresh) and offers the ones that are behind. Quiet when nothing is.
	void CheckForUpdates()
	{
		// STEP7:3: "Check for updates on startup" off (Settings > General): no automatic check at all. The
		// Version Manager's Update buttons still work, and so does turning it back on mid-session.
		if (!s.check_updates)
		{
			return;
		}
		auto targets = LatestBuilds(false);
		if (busy || targets.empty())
		{
			return;
		}
		Spawn(
		    [self = shared_from_this(), targets = std::move(targets), cache = CacheDir()](const auto& cancel)
		    {
			    std::vector<std::pair<LatestTarget, ReleaseInfo>> found;
			    for (const auto& t: targets)
			    {
				    std::string error;
				    auto        list = fetch_releases(t.first, cache, false, &error, nullptr, cancel.get());
				    if (!list.empty() && list.front().tag != t.second.tag && list.front().date >= t.second.date)
				    {
					    found.emplace_back(t, list.front());
				    }
			    }
			    slint::invoke_from_event_loop([self, found = std::move(found)] { self->OfferUpdates(found); });
		    });
	}

	// The "newer build" popup (on any page). A release is offered once per session; Later leaves it.
	void OfferUpdates(const std::vector<std::pair<LatestTarget, ReleaseInfo>>& found)
	{
		bool any_new = false;
		for (const auto& [t, r]: found)
		{
			any_new = notified.insert(t.second.name + "|" + r.tag).second || any_new;
		}
		if (!any_new)
		{
			return;
		}
		offered.clear();
		std::string text;
		for (const auto& [t, r]: found)
		{
			offered.push_back(t);
			text += (text.empty() ? "" : "\n") + DisplayName(t.second) + ": " + r.date + "  " + r.codename + " (installed: " + t.second.date +
			        "  " + t.second.codename + ")";
		}
		const std::string title = offered.size() == 1 ? std::string("A newer build is available")
		                                              : std::to_string(offered.size()) + " newer builds are available";
		WithUi(
		    [&](AppWindow&, const VM& vm)
		    {
			    vm.set_update_prompt_title(slint::SharedString(title));
			    vm.set_update_prompt_text(slint::SharedString(text));
			    vm.set_update_prompt_open(true);
		    });
	}

	// Downloads the newest release of each target's source into its "-latest" folder, one after
	// another; the ones already on the newest release are left as they are.
	void UpdateBuilds(std::vector<LatestTarget> targets)
	{
		if (targets.empty())
		{
			return;
		}
		if (busy)
		{
			Status("Wait for the running download to finish, then use Update.");
			return;
		}
		SetBusy(true);
		Status("Checking for updates…");
		Spawn(
		    [self = shared_from_this(), targets = std::move(targets), dir = VersionsDir(), cache = CacheDir()](const auto& cancel)
		    {
			    std::vector<std::string> updated_exes;
			    std::vector<std::string> updated; // "name to codename"
			    std::vector<std::string> failed;  // "name: error"
			    int                      up_to_date = 0;
			    bool                     cancelled  = false;
			    for (size_t i = 0; i < targets.size() && !cancelled; i++)
			    {
				    const auto& [repo, b] = targets[i];
				    std::string error;
				    auto        list = fetch_releases(repo, cache, true, &error, nullptr, cancel.get());
				    if (list.empty())
				    {
					    cancelled = error == "Cancelled";
					    if (!cancelled)
					    {
						    failed.push_back(b.name + ": " + (error.empty() ? std::string("no Windows builds found") : error));
					    }
					    continue;
				    }
				    ReleaseInfo r = list.front();
				    if (r.tag == b.tag)
				    {
					    up_to_date++;
					    continue;
				    }
				    r.folder = b.name;
				    const std::string prefix =
				        "Updating " + b.name + (targets.size() > 1 ? " (" + std::to_string(i + 1) + "/" + std::to_string(targets.size()) + ")" : "") + ": ";
				    int         last_pct = -1;
				    std::string exe      = install_release(
				        r, dir,
				        [&](uint64_t done, uint64_t total)
				        {
					        int pct = total > 0 ? static_cast<int>(done * 100 / total) : -1;
					        if (pct == last_pct)
					        {
						        return;
					        }
					        last_pct         = pct;
					        std::string text = prefix + std::to_string(done >> 20) + " / " + std::to_string(total >> 20) + " MB";
					        slint::invoke_from_event_loop([self, text] { self->Status(text); });
				        },
				        &error, cancel.get());
				    if (exe.empty())
				    {
					    cancelled = error == "Cancelled";
					    if (!cancelled)
					    {
						    failed.push_back(b.name + ": " + error);
					    }
					    continue;
				    }
				    updated_exes.push_back(exe);
				    updated.push_back(b.name + " to " + r.codename);
			    }
			    slint::invoke_from_event_loop(
			        [self, updated_exes, updated, failed, up_to_date, cancelled, total = targets.size()]
			        {
				        self->SetBusy(false);
				        for (const auto& exe: updated_exes)
				        {
					        self->Unhide(exe);
				        }
				        self->RefreshInstalled();
				        std::string text;
				        if (updated.size() == 1)
				        {
					        text = "Updated " + updated[0];
				        } else if (!updated.empty())
				        {
					        text = "Updated " + std::to_string(updated.size()) + " builds";
				        } else if (up_to_date > 0 && failed.empty() && !cancelled)
				        {
					        text = total == 1 ? std::string("Already on the newest build") : "All " + std::to_string(total) + " builds are on the newest build";
				        }
				        if (cancelled)
				        {
					        text += (text.empty() ? "" : "; ") + std::string("update cancelled");
				        }
				        for (const auto& f: failed)
				        {
					        text += (text.empty() ? "Update failed: " : "; failed: ") + f;
				        }
				        self->Status(text);
			        });
		    });
	}

	void AddCustom()
	{
		std::string p = pick(s.emulator, false);
		if (p.empty())
		{
			return;
		}
		auto& list = s.custom_builds;
		if (std::none_of(list.begin(), list.end(), [&](const std::string& e) { return path_key(e) == path_key(p); }))
		{
			list.push_back(p);
		}
		s.emulator = p;
		WithUi([&](AppWindow& w, const VM&) { w.set_emulator(slint::SharedString(p)); });
		save();
		RefreshInstalled();
		Status("Added " + p);
	}

	// A downloaded build that is downloaded again shows up again.
	void Unhide(const std::string& exe)
	{
		auto&        hidden = s.hidden_builds;
		const size_t before = hidden.size();
		hidden.erase(std::remove_if(hidden.begin(), hidden.end(), [&](const std::string& h) { return path_key(h) == path_key(exe); }), hidden.end());
		if (hidden.size() != before)
		{
			save();
		}
	}

	// Removes the selected builds from the Installed list. Downloaded builds keep their folder unless
	// delete_files is set; custom and local builds only leave the custom list.
	void RemoveSelected(bool delete_files)
	{
		std::vector<InstalledBuild> picked;
		for (const auto& b: installed)
		{
			if (selected.count(path_key(b.exe)) != 0)
			{
				picked.push_back(b);
			}
		}
		if (picked.empty())
		{
			return;
		}
		auto&       custom  = s.custom_builds;
		auto&       hidden  = s.hidden_builds;
		int         removed = 0;
		std::string error;
		for (const auto& b: picked)
		{
			if (!b.managed)
			{
				custom.erase(std::remove_if(custom.begin(), custom.end(), [&](const std::string& e) { return path_key(e) == path_key(b.exe); }),
				             custom.end());
			} else if (delete_files)
			{
				if (!delete_build_folder(b, VersionsDir(), &error))
				{
					break;
				}
			} else
			{
				// Hidden until it is downloaded again.
				hidden.push_back(b.exe);
			}
			removed++;
		}
		save();
		selected.clear();
		RefreshInstalled();
		if (!error.empty())
		{
			Status(error);
		} else if (removed == 1)
		{
			const auto& b = picked[0];
			Status(!b.managed     ? "Removed " + DisplayName(b) + " from the list"
			       : delete_files ? "Deleted " + DisplayName(b) + " and its files"
			                      : "Removed " + DisplayName(b) + " from the list (its folder is still on disk)");
		} else
		{
			Status((delete_files ? "Deleted " : "Removed ") + std::to_string(removed) + " builds" + (delete_files ? "" : " from the list"));
		}
	}

	// Opens the Remove confirmation for the selected builds.
	void RequestRemove()
	{
		int         count = 0;
		bool        any_tracked = false;
		bool        any_local   = false;
		std::string first;
		std::vector<std::string> exes;
		for (const auto& b: installed)
		{
			if (selected.count(path_key(b.exe)) != 0)
			{
				exes.push_back(b.exe);
				if (count++ == 0)
				{
					first = DisplayName(b);
				}
				any_tracked = any_tracked || b.managed;
				any_local   = any_local || !b.managed;
			}
		}
		if (count == 0 || busy)
		{
			return;
		}
		const int pinned = pin_counter ? pin_counter(exes) : 0;
		WithUi(
		    [&](AppWindow&, const VM& vm)
		    {
			    vm.set_remove_pinned_count(pinned);
			    vm.set_remove_count(count);
			    vm.set_remove_name(slint::SharedString(first));
			    vm.set_remove_has_tracked(any_tracked);
			    vm.set_remove_has_local(any_local);
			    vm.set_remove_delete_files(false);
			    vm.set_remove_dialog_open(true);
		    });
	}

	// ---- selection of the Installed table (rows are table rows, after the filter)

	std::string RowKey(int row) const
	{
		const int index = RealIndex(row);
		return index < 0 ? std::string() : path_key(installed[static_cast<size_t>(index)].exe);
	}

	int RowOf(const std::string& key) const
	{
		for (size_t r = 0; r < row_map.size(); r++)
		{
			if (path_key(installed[row_map[r]].exe) == key)
			{
				return static_cast<int>(r);
			}
		}
		return -1;
	}

	// Pushes `selected` into the table rows that changed, and the focused row and count to the page.
	void ShowSelection()
	{
		for (size_t r = 0; r < row_map.size(); r++)
		{
			const bool on = selected.count(path_key(installed[row_map[r]].exe)) != 0;
			if (auto v = installed_model->row_data(r); v && v->selected != on)
			{
				v->selected = on;
				installed_model->set_row_data(r, *v);
			}
		}
		const int  focus     = focus_key.empty() ? -1 : RowOf(focus_key);
		const auto& unticked = s.menu_hidden;
		const bool all_shown = std::none_of(unticked.begin(), unticked.end(), [&](const std::string& u) { return selected.count(path_key(u)) != 0; });
		const int  latest    = static_cast<int>(LatestBuilds(true).size());
		WithUi(
		    [&](AppWindow&, const VM& vm)
		    {
			    vm.set_current_installed(focus);
			    vm.set_selected_count(static_cast<int>(selected.size()));
			    vm.set_selected_all_shown(all_shown);
			    vm.set_selected_latest_count(latest);
		    });
	}

	// Ticks (or unticks) Show on every selected build: listed in the toolbar's build menu or not.
	void ShowSelected(bool on)
	{
		auto& list = s.menu_hidden;
		std::erase_if(list, [&](const std::string& u) { return selected.count(path_key(u)) != 0; });
		if (!on)
		{
			for (const auto& b: installed)
			{
				if (selected.count(path_key(b.exe)) != 0)
				{
					list.push_back(b.exe);
				}
			}
		}
		save();
		// The running build stays in the toolbar's build list, ticked or not.
		if (!on)
		{
			for (const auto& b: installed)
			{
				if (selected.count(path_key(b.exe)) != 0 && path_key(b.exe) == path_key(s.emulator))
				{
					Status(DisplayName(b) + " is still the running build; it stays in the build list.");
					break;
				}
			}
		}
		RefreshInstalled();
	}

	// Left press on a row (-1: the empty space under the rows). Ctrl toggles, Shift extends from the anchor.
	void RowPressed(int row, bool ctrl, bool shift)
	{
		const std::string key = RowKey(row);
		if (key.empty())
		{
			if (!ctrl)
			{
				selected.clear();
				focus_key.clear();
			}
		} else if (shift && RowOf(anchor_key) >= 0)
		{
			if (!ctrl)
			{
				selected.clear();
			}
			const int a = RowOf(anchor_key);
			for (int r = std::min(a, row); r <= std::max(a, row); r++)
			{
				selected.insert(RowKey(r));
			}
			focus_key = key;
		} else if (ctrl)
		{
			if (selected.erase(key) == 0)
			{
				selected.insert(key);
			}
			anchor_key = key;
			focus_key  = key;
		} else
		{
			selected  = {key};
			anchor_key = key;
			focus_key  = key;
		}
		ShowSelection();
	}

	// Right-click on a row: a row outside the selection becomes the only selected one.
	void RowMenu(int row)
	{
		const std::string key = RowKey(row);
		if (key.empty())
		{
			return;
		}
		if (selected.count(key) == 0)
		{
			selected   = {key};
			anchor_key = key;
		}
		focus_key = key;
		ShowSelection();
	}

	// Marquee drag: the rows first..last on top of what was selected when it began (with Ctrl held).
	void MarqueeBegin(bool additive)
	{
		marquee_base = additive ? selected : std::set<std::string> {};
	}

	void Marquee(int first, int last)
	{
		selected = marquee_base;
		first    = std::max(first, 0);
		last     = std::min(last, static_cast<int>(row_map.size()) - 1);
		for (int r = first; r <= last; r++)
		{
			selected.insert(RowKey(r));
		}
		if (first <= last)
		{
			focus_key = RowKey(last);
		}
		ShowSelection();
	}

	void SelectAll()
	{
		for (size_t r = 0; r < row_map.size(); r++)
		{
			selected.insert(RowKey(static_cast<int>(r)));
		}
		ShowSelection();
	}

	void BrowsePath()
	{
		std::string p = pick(VersionsDir(), true);
		if (p.empty())
		{
			return;
		}
		s.versions_dir = p;
		WithUi([&](AppWindow&, const VM& vm) { vm.set_versions_path(slint::SharedString(p)); });
		save();
		RefreshInstalled();
	}

	// "Refresh List" under Installed: also picks up a path typed into the field.
	void RescanInstalled()
	{
		WithUi(
		    [&](AppWindow&, const VM& vm)
		    {
			    std::string typed = ToStd(vm.get_versions_path());
			    s.versions_dir    = typed == default_versions_dir ? std::string() : typed;
		    });
		save();
		RefreshInstalled();
		Status(std::to_string(installed.size()) + " builds installed", true);
	}

	void Wire()
	{
		auto self = shared_from_this();
		BuildSources();
		WithUi(
		    [&](AppWindow& w, const VM& vm)
		    {
			    w.set_emu_builds(dropdown_model);
			    w.set_pin_builds(pin_model);
			    w.on_choose_emu_build(
			        [self](int index)
			        {
				        if (index >= 0 && static_cast<size_t>(index) < self->menu_map.size())
				        {
					        self->Activate(static_cast<int>(self->menu_map[static_cast<size_t>(index)]));
				        }
			        });

			    vm.set_versions_path(slint::SharedString(VersionsDir()));
			    // Settings tab: the GitHub token and the saved release lists.
			    // The token is write-only: the UI only learns whether one is set, never its text.
			    {
				    const std::string saved = load_github_token();
				    set_github_token(saved);
				    vm.set_gh_token_set(!saved.empty());
				    vm.set_gh_token_hint(slint::SharedString(token_hint(saved)));
				    vm.set_gh_token_storable(secret_store_supported());
			    }
			    vm.on_gh_token_changed(
			        [self](const slint::SharedString& text)
			        {
				        // Pasted tokens often carry stray spaces or a newline.
				        std::string  token = ToStd(text);
				        const size_t b     = token.find_first_not_of(" \t\r\n");
				        token = b == std::string::npos ? std::string() : token.substr(b, token.find_last_not_of(" \t\r\n") - b + 1);
				        std::string status;
				        if (token.empty())
				        {
					        clear_github_token();
					        status = "GitHub token cleared";
				        } else if (save_github_token(token))
				        {
					        status = "GitHub token saved";
				        } else
				        {
					        // No credential store: net.cpp keeps it in memory until the launcher closes.
					        status = "GitHub token set for this session only (no credential store)";
				        }
				        set_github_token(token);
				        self->WithUi(
				            [&](AppWindow&, const VM& vm)
				            {
					            vm.set_gh_token_set(!token.empty());
					            vm.set_gh_token_hint(slint::SharedString(token_hint(token)));
				            });
				        self->Status(status);
			        });
			    vm.on_settings_opened([self] { self->ShowCacheInfo(); });
			    vm.on_delete_cache(
			        [self]
			        {
				        const int n = clear_release_cache(self->CacheDir());
				        self->ShowCacheInfo();
				        self->Status(n == 0 ? std::string("No saved release lists to delete")
				                            : "Deleted " + std::to_string(n) + " saved release list" + (n == 1 ? "" : "s"));
			        });
			    vm.set_installed(installed_model);
			    vm.set_available(available_model);
			    vm.set_older(older_model);
			    vm.set_sources(sources_model);

			    // The Show checkbox lists the build in the toolbar menu (the menu picks the one that runs).
			    vm.on_select_version(
			        [self](int row, bool on)
			        {
				        const int index = self->RealIndex(row);
				        if (index >= 0 && static_cast<size_t>(index) < self->installed.size())
				        {
					        const std::string exe  = self->installed[static_cast<size_t>(index)].exe;
					        auto&             list = self->s.menu_hidden;
					        list.erase(std::remove_if(list.begin(), list.end(), [&](const std::string& u) { return path_key(u) == path_key(exe); }),
					                   list.end());
					        if (!on)
					        {
						        list.push_back(exe);
					        }
					        self->save();
					        // The running build stays in the toolbar's build list, ticked or not.
					        if (!on && path_key(exe) == path_key(self->s.emulator))
					        {
						        self->Status(self->DisplayName(self->installed[static_cast<size_t>(index)]) + " is still the running build; it stays in the build list.");
					        }
				        }
				        self->RefreshInstalled();
			        });
			    // Double-click in the Installed table: the launcher runs that build.
			    vm.on_activate_version([self](int row) { self->Activate(self->RealIndex(row)); });
			    vm.on_browse_path([self] { self->BrowsePath(); });
			    vm.on_refresh_installed([self] { self->RescanInstalled(); });
			    vm.on_add_custom([self] { self->AddCustom(); });
			    vm.on_request_remove([self] { self->RequestRemove(); });
			    vm.on_confirm_remove([self](bool delete_files) { self->RemoveSelected(delete_files); });
			    vm.on_row_pressed([self](int row, bool ctrl, bool shift) { self->RowPressed(row, ctrl, shift); });
			    vm.on_row_menu([self](int row) { self->RowMenu(row); });
			    vm.on_marquee_begin([self](bool additive) { self->MarqueeBegin(additive); });
			    vm.on_marquee([self](int first, int last) { self->Marquee(first, last); });
			    vm.on_select_all([self] { self->SelectAll(); });
			    vm.on_show_selected([self](bool on) { self->ShowSelected(on); });
			    vm.on_refresh_available([self] { self->FetchReleasesAsync(nullptr, true); });
			    vm.on_download_version(
			        [self](int index)
			        {
				        if (index >= 0 && static_cast<size_t>(index) < self->releases.size())
				        {
					        self->Download(self->releases[static_cast<size_t>(index)], index == 0);
				        }
			        });
			    vm.on_select_source([self](int index) { self->SelectSource(index); });
			    vm.on_rename_source([self](int index, const slint::SharedString& name) { self->RenameSource(index, ToStd(name)); });
			    // The release's page on GitHub, opened with the launcher's link opener.
			    vm.on_open_release(
			        [self](int index)
			        {
				        if (index >= 0 && static_cast<size_t>(index) < self->releases.size())
				        {
					        const std::string url = "https://github.com/" + self->sources[static_cast<size_t>(self->current_source)].repo +
					                                "/releases/tag/" + self->releases[static_cast<size_t>(index)].tag;
					        self->WithUi([&](AppWindow& w, const VM&) { w.invoke_open_url(slint::SharedString(url)); });
				        }
			        });
			    vm.on_stop_tracking([self](int index) { self->StopTracking(index); });
			    vm.on_rename_version([self](int row, const slint::SharedString& name) { self->Rename(self->RealIndex(row), ToStd(name)); });
			    vm.on_sort_by([self](const slint::SharedString& key) { self->SortBy(ToStd(key)); });
			    vm.on_toggle_pin([self](int row) { self->TogglePin(self->RealIndex(row)); });
			    vm.on_about_requested([self](int row) { self->ShowAbout(row); });
			    vm.on_about_show_file([self] { self->ShowAboutFile(); });
			    vm.on_about_open_link([self] { self->OpenAboutLink(); });
			    vm.on_filter_changed([self] { self->ApplyInstalled(); });
			    vm.on_opened([self] { self->Opened(); });
			    vm.on_cancel_busy([self] { self->CancelAll(); });
			    vm.on_closed([self] { self->Closed(); });
			    vm.on_add_source([self](const slint::SharedString& link) { self->AddSource(ToStd(link)); });
			    // Right-click > Update on the selected "-latest" builds, and Update in the newer-build popup.
			    vm.on_update_selected([self] { self->UpdateBuilds(self->LatestBuilds(true)); });
			    vm.on_update_prompt_accept([self] { self->UpdateBuilds(self->offered); });
		    });
		RefreshInstalled();
		// Newer builds for the "-latest" folders: looked for now and every hour while the launcher is open.
		CheckForUpdates();
		update_timer.start(slint::TimerMode::Repeated, std::chrono::hours(1),
		                   [weak = weak_from_this()]
		                   {
			                   if (auto self = weak.lock())
			                   {
				                   self->CheckForUpdates();
			                   }
		                   });
	}
};

VersionManager::VersionManager(const slint::ComponentHandle<AppWindow>& ui, Settings& settings, std::string default_versions_dir, std::string cache_dir,
                               SaveFn save, PickFn pick)
    : m_impl(std::make_shared<Impl>(ui, settings, std::move(default_versions_dir), std::move(cache_dir), std::move(save), std::move(pick)))
{
	m_impl->Wire();
}

VersionManager::~VersionManager()
{
	m_impl->Shutdown();
}

std::string VersionManager::BuildName(const std::string& exe) const
{
	return m_impl->BuildName(exe);
}

void VersionManager::SetPinCounter(PinCountFn fn)
{
	m_impl->pin_counter = std::move(fn);
}

void VersionManager::SetOnBuildsChanged(std::function<void()> fn)
{
	m_impl->builds_changed = std::move(fn);
}

} // namespace kyty_launcher
