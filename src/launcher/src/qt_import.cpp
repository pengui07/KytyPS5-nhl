#include "qt_import.h"

#include "library.h"
#include "qt_ini.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>

namespace kyty_launcher {

namespace fs = std::filesystem;

namespace {

constexpr char kGlobal[] = "GlobalConfiguration";
constexpr char kLauncher[] = "Launcher";
constexpr char kGames[]  = "GameConfigurations";

#ifdef _WIN32
constexpr char kEmulatorExe[] = "kyty_emulator.exe";
#else
constexpr char kEmulatorExe[] = "kyty_emulator";
#endif

bool IsFile(const fs::path& p)
{
	std::error_code ec;
	return fs::is_regular_file(p, ec);
}

bool IsTrue(const std::string& v)
{
	std::string low = v;
	std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return low == "true" || low == "1";
}

// The Qt launcher stores these once per game entry (and empty in the global section); they are not settings.
bool IsGameInfoKey(const std::string& k)
{
	return k == "name" || k == "basedir" || k == "game_path" || k == "custom_settings";
}

// Keys the Qt launcher wrote that have no equivalent here, with what they were.
const std::pair<const char*, const char*> kKnownUnsupported[] = {
    {"MainDialog/geometry", "main window size and position"},
    {"MainDialog/theme", "theme"},
    {"ConfigurationEditDialog/geometry", "settings window size and position"},
    {"Launcher/grid_view", "grid view"},
    {"Launcher/grid_icon_width", "grid icon size"},
};

// A key with array indexes as "N", so the entries of an array count once.
std::string GenericKey(const std::string& key)
{
	std::string out;
	size_t      pos = 0;
	while (pos <= key.size())
	{
		size_t slash = key.find('/', pos);
		if (slash == std::string::npos)
		{
			slash = key.size();
		}
		std::string part = key.substr(pos, slash - pos);
		if (!part.empty() && std::all_of(part.begin(), part.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
		{
			part = "N";
		}
		out += (out.empty() ? "" : "/") + part;
		pos = slash + 1;
	}
	return out;
}

// The title id of a game entry: game_path (a game folder, or an archive), else basedir. `why` says what was wrong.
std::string IdentifyGame(const std::string& game_path, const std::string& basedir, const fs::path& ini_dir, std::string& why)
{
	std::error_code ec;
	bool            found = false;
	std::string     first_tried;
	for (const std::string& c: {game_path, basedir})
	{
		if (c.empty())
		{
			continue;
		}
		fs::path p = utf8_path(c);
		if (p.is_relative() && fs::exists(ini_dir / p, ec))
		{
			p = ini_dir / p;
		}
		if (first_tried.empty())
		{
			first_tried = c;
		}
		if (!fs::exists(p, ec))
		{
			continue;
		}
		found                = true;
		const std::string id = read_title_id(path_utf8(p));
		if (!id.empty())
		{
			return id;
		}
	}
	if (first_tried.empty())
	{
		why = "the entry has no game path";
	} else if (!found)
	{
		why = "game not found: " + first_tried;
	} else
	{
		why = "title id not found (an archive, or no sce_sys/param.json): " + first_tried;
	}
	return {};
}

std::string Plural(size_t n, const char* one, const char* many)
{
	return std::to_string(n) + " " + (n == 1 ? one : many);
}

} // namespace

std::string ImportReport::text() const
{
	if (!ok)
	{
		return "Could not import " + source + ": " + error;
	}
	std::string out = "Imported from " + source + "\n\n";
	out += "Settings imported: " + std::to_string(settings) + "\n";
	out += "Games with their own settings: " + std::to_string(games.size()) + "\n";
	for (const auto& g: games)
	{
		out += "  - " + g + "\n";
	}
	if (!skipped.empty())
	{
		out += "\nSkipped (" + std::to_string(skipped.size()) + "):\n";
		for (const auto& g: skipped)
		{
			out += "  - " + g + "\n";
		}
	}
	if (!not_imported.empty())
	{
		out += "\nNot imported (" + std::to_string(not_imported.size()) + "):\n";
		for (const auto& g: not_imported)
		{
			out += "  - " + g + "\n";
		}
	}
	out += "\nThe Kyty.ini was not changed.";
	return out;
}

std::string find_qt_ini(const Paths& paths)
{
	std::vector<fs::path> candidates;
	std::error_code       ec;
	candidates.push_back(fs::current_path(ec) / "Kyty.ini");
	candidates.push_back(paths.exe_dir / "Kyty.ini");
	candidates.push_back(paths.exe_dir.parent_path() / "Kyty.ini");
#ifdef _WIN32
	// QSettings(IniFormat, SystemScope, "Kyty", "Kyty").
	if (const char* pd = std::getenv("ProgramData"); pd != nullptr && *pd != '\0')
	{
		candidates.push_back(fs::path(pd) / "Kyty" / "Kyty.ini");
	}
#else
	// QSettings(IniFormat, UserScope, "Kyty", "Kyty"): Qt names the file Kyty.conf on these systems; Kyty.ini is probed too.
	const char* xdg  = std::getenv("XDG_CONFIG_HOME");
	const char* home = std::getenv("HOME");
	fs::path    conf = xdg != nullptr && *xdg != '\0' ? fs::path(xdg) : (home != nullptr && *home != '\0' ? fs::path(home) / ".config" : fs::path());
	if (!conf.empty())
	{
		candidates.push_back(conf / "Kyty" / "Kyty.ini");
		candidates.push_back(conf / "Kyty" / "Kyty.conf");
	}
#endif
	for (const auto& c: candidates)
	{
		if (IsFile(c))
		{
			return path_utf8(c);
		}
	}
	return {};
}

ImportReport import_qt_ini(const std::string& ini_path, const Paths& paths, Settings& s, bool first_run)
{
	ImportReport r;
	r.source = ini_path;
	QtIni ini;
	if (!read_qt_ini(ini_path, ini, &r.error))
	{
		return r;
	}
	r.ok = true;

	const fs::path        ini_dir = utf8_path(ini_path).parent_path();
	std::set<std::string> used; // INI keys that were looked at; the rest is reported as not recognised
	auto                  use = [&](const std::string& k) { used.insert(k); };

	// [Launcher]: game folders (game_dirs, or the older single game_dir) and the add-ons folder.
	{
		std::vector<std::string> dirs;
		ini.get_list(std::string(kLauncher) + "/game_dirs", dirs);
		if (dirs.empty())
		{
			ini.get_list(std::string(kLauncher) + "/game_dir", dirs);
		}
		use(std::string(kLauncher) + "/game_dirs");
		use(std::string(kLauncher) + "/game_dir");
		std::string joined;
		for (const auto& d: dirs)
		{
			if (!d.empty())
			{
				joined += (joined.empty() ? "" : ";") + d;
			}
		}
		if (!joined.empty())
		{
			s.game_dirs = joined;
			r.settings++;
		}
		std::string addons;
		use(std::string(kLauncher) + "/addons_dir");
		if (ini.get(std::string(kLauncher) + "/addons_dir", addons) && !addons.empty())
		{
			s.addons_dir = addons;
			r.settings++;
		}
	}

	// [MainDialog]: the Qt launcher's own "check for updates on startup" (a launcher setting, not the emulator's).
	// STEP7:3.
	{
		const std::string key = "MainDialog/check_updates_on_startup";
		use(key);
		std::string text;
		if (ini.get(key, text))
		{
			s.check_updates = IsTrue(text);
			r.settings++;
		}
	}

	// [GlobalConfiguration]: the emulator settings (same key names as the "emu" object of kyty.json).
	for (const auto& key: ini.keys_in(kGlobal))
	{
		const std::string full = std::string(kGlobal) + "/" + key;
		if (IsGameInfoKey(key))
		{
			use(full);
			continue;
		}
		if (key == "elf")
		{
			use(full);
			std::string elf;
			if (ini.get(full, elf) && !elf.empty() && elf != "eboot.bin")
			{
				r.not_imported.push_back("global start file (elf): " + elf);
			}
			continue;
		}
		if (key == "host_input_mapping")
		{
			use(full);
			std::vector<std::string> list;
			if (ini.get_list(full, list))
			{
				list.erase(std::remove(list.begin(), list.end(), std::string()), list.end());
				s.host_input_mapping = list;
				s.user_keys.insert("host_input_mapping");
				r.settings++;
			}
			continue;
		}
		std::string text;
		std::string why;
		const bool  have = ini.get(full, text);
		// The key is recognised when apply_qt_emu_value knows it; an unknown one stays out of `used`.
		if (!have)
		{
			Settings probe = s;
			if (apply_qt_emu_value(probe, key, "0", &why) != 0)
			{
				use(full);
				r.skipped.push_back(std::string(kGlobal) + "/" + key + ": unsupported value type");
			}
			continue;
		}
		const int res = apply_qt_emu_value(s, key, text, &why);
		if (res == 1)
		{
			use(full);
			r.settings++;
		} else if (res < 0)
		{
			use(full);
			r.skipped.push_back(std::string(kGlobal) + "/" + key + ": " + why);
		}
	}

	// The emulator next to the Kyty.ini, and its run folder (_SaveData, _Trophies, _Patches carry over).
	{
		const fs::path exe = ini_dir / kEmulatorExe;
		std::error_code ec;
		if (IsFile(exe))
		{
			if (first_run || s.emulator.empty() || !fs::exists(utf8_path(s.emulator), ec))
			{
				s.emulator = path_utf8(exe);
				s.run_dir  = path_utf8(ini_dir);
				r.settings += 2;
			} else if (path_key(s.emulator) != path_key(path_utf8(exe)))
			{
				r.not_imported.push_back("emulator next to the Kyty.ini (" + path_utf8(exe) + "): the emulator already set was kept");
			}
		}
	}

	// [GameConfigurations]: one file per game that has custom settings.
	{
		int size = 0;
		use(std::string(kGames) + "/size");
		std::string size_text;
		if (ini.get(std::string(kGames) + "/size", size_text))
		{
			size = std::atoi(size_text.c_str());
		}
		for (const auto& [k, v]: ini.values) // also entries past a wrong size
		{
			if (k.rfind(std::string(kGames) + "/", 0) == 0)
			{
				const int idx = std::atoi(k.c_str() + std::string(kGames).size() + 1);
				size          = std::max(size, idx);
			}
		}
		std::set<std::string> seen_ids;
		size_t                plain = 0;
		for (int i = 1; i <= size; i++)
		{
			const std::string prefix = std::string(kGames) + "/" + std::to_string(i);
			const auto        keys   = ini.keys_in(prefix);
			if (keys.empty())
			{
				continue;
			}
			auto value = [&](const char* key)
			{
				std::string t;
				ini.get(prefix + "/" + key, t);
				return t;
			};
			for (const auto& key: keys)
			{
				if (IsGameInfoKey(key))
				{
					use(prefix + "/" + key);
				}
			}
			const std::string game_path = value("game_path");
			const std::string basedir   = value("basedir");
			std::string       name      = value("name");
			const std::string label     = name.empty() ? (game_path.empty() ? "entry " + std::to_string(i) : game_path) : name;
			if (!IsTrue(value("custom_settings")))
			{
				plain++;
				for (const auto& key: keys)
				{
					use(prefix + "/" + key);
				}
				continue;
			}
			std::string why;
			const std::string tid = IdentifyGame(game_path, basedir, ini_dir, why);
			if (tid.empty())
			{
				r.skipped.push_back(label + ": " + why);
				for (const auto& key: keys)
				{
					use(prefix + "/" + key);
				}
				continue;
			}
			if (!seen_ids.insert(tid).second)
			{
				r.skipped.push_back(label + ": " + tid + " already imported from another entry");
				for (const auto& key: keys)
				{
					use(prefix + "/" + key);
				}
				continue;
			}
			Settings edited = s; // the imported global settings; the file keeps only what differs
			for (const auto& key: keys)
			{
				const std::string full = prefix + "/" + key;
				if (IsGameInfoKey(key) || key.rfind("controller_", 0) == 0) // controller settings are global in the Qt launcher
				{
					use(full);
					continue;
				}
				if (key == "elf")
				{
					use(full);
					const std::string elf = value("elf");
					if (!elf.empty() && elf != "eboot.bin")
					{
						r.not_imported.push_back(tid + " start file (elf): " + elf);
					}
					continue;
				}
				if (key == "host_input_mapping")
				{
					use(full);
					std::vector<std::string> list;
					if (ini.get_list(full, list))
					{
						list.erase(std::remove(list.begin(), list.end(), std::string()), list.end());
						edited.host_input_mapping = list;
					}
					continue;
				}
				std::string text;
				if (!ini.get(full, text))
				{
					continue; // special values: reported with the unrecognised keys
				}
				std::string why_not;
				const int   res = apply_qt_emu_value(edited, key, text, &why_not);
				if (res != 0)
				{
					use(full);
				}
				if (res < 0)
				{
					r.skipped.push_back(tid + " " + key + ": " + why_not);
				}
			}
			if (!save_game_overrides(paths, tid, s, edited))
			{
				r.skipped.push_back(label + ": could not write " + path_utf8(paths.game_config_file(tid)));
				continue;
			}
			size_t differing = 0;
			{
				JsonValue root;
				if (parse_json(read_text_file(path_utf8(paths.game_config_file(tid))), root) && root.type == JsonValue::Type::Object)
				{
					if (auto it = root.o.find("emu"); it != root.o.end())
					{
						differing = it->second.o.size() - it->second.o.count("env");
					}
				}
			}
			r.games.push_back(tid + (name.empty() ? "" : " (" + name + ")") + ": " + Plural(differing, "setting differs", "settings differ") +
			                  " from the global ones");
		}
		if (plain > 0)
		{
			r.not_imported.push_back(Plural(plain, "game entry", "game entries") + " without custom settings (nothing to carry over)");
		}
	}

	// What is left of the file.
	for (const auto& [key, what]: kKnownUnsupported)
	{
		if (ini.find(key) != nullptr)
		{
			use(key);
			r.not_imported.push_back(std::string(what) + " (" + key + ")");
		}
	}
	{
		std::set<std::string> unknown;
		for (const auto& [k, v]: ini.values)
		{
			if (used.count(k) == 0)
			{
				unknown.insert(GenericKey(k));
			}
		}
		std::string list;
		size_t      n = 0;
		for (const auto& k: unknown)
		{
			if (n++ < 12)
			{
				list += (list.empty() ? "" : ", ") + k;
			}
		}
		if (!unknown.empty())
		{
			r.not_imported.push_back("keys not recognised: " + list + (unknown.size() > 12 ? ", and " + std::to_string(unknown.size() - 12) + " more" : ""));
		}
	}
	if (ini.lines_ignored > 0)
	{
		r.not_imported.push_back(Plural(static_cast<size_t>(ini.lines_ignored), "unreadable line", "unreadable lines"));
	}
	return r;
}

} // namespace kyty_launcher
