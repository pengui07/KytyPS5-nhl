#pragma once

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace kyty_launcher {

struct Paths;

// "version" in kyty.json and the per-game files.
constexpr int kConfigVersion = 1;

// Settings strings are UTF-8; these keep non-ASCII paths intact on Windows.
inline std::filesystem::path utf8_path(const std::string& s)
{
	return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

inline std::string path_utf8(const std::filesystem::path& p)
{
	auto u = p.u8string();
	return std::string(u.begin(), u.end());
}

// Minimal JSON tree for the settings file and sce_sys/param.json.
struct JsonValue
{
	enum class Type { Null, Bool, Number, String, Object, Array } type = Type::Null;
	bool                             b = false;
	double                           n = 0.0;
	std::string                      s;
	std::map<std::string, JsonValue> o;
	std::vector<JsonValue>           a;
};

bool parse_json(const std::string& text, JsonValue& out);

struct Settings
{
	std::string                 emulator;
	std::string                 game;
	std::string                 run_dir;
	std::string                 game_dirs; // ';'-separated folders scanned for games
	int                         width  = 2560;
	int                         height = 1440;
	std::string                 extra_args;
	std::string                 addons_dir; // the one add-ons (DLC) folder, --addcont-dir; empty = none (global, never layered)
	std::string                 columns; // library column widths, "240,100,..." (logical px)
	// User profile and compatibility (the emulator's --user-name, --user-id, --console-language, --mic, --amd-cpu, --redzone)
	std::string                 user_name        = "Kyty";
	int                         user_id          = 1000;
	int                         console_language = 1; // English (United States)
	std::string                 microphone;           // empty = None
	bool                        amd_cpu          = false;
	bool                        red_zone         = false;
	// Trophy popups (the emulator's --no-trophy-notifications, --trophy-duration)
	bool                        trophy_notifications = true;
	int                         trophy_duration      = 6; // seconds, 5..10
	// General (the emulator's --fullscreen, --hide-cursor, --perf-overlay, --skip-notice-screen, --network-connected)
	bool                        fullscreen         = false;
	bool                        hide_cursor        = false;
	bool                        perf_overlay       = false;
	bool                        skip_notice_screen = false;
	bool                        network_connected  = false;
	// Controller (--controller-color, --controller-vibration, --controller-volume, --keyboard-player)
	std::string                 controller_color;              // "#rrggbb"; empty = the default lightbar
	int                         controller_vibration = 100;    // 0..100
	int                         controller_volume    = 50;     // 0..100
	int                         keyboard_player      = 0;      // 0 = shared with player 1, 1..4
	// Graphics and debug (the emulator's --present-mode, --vblank-frequency, ... defaults)
	int                         present_mode = 1;
	int                         vblank = 60;
	int                         shader_opt = 0;
	int                         gpu_index = -1;
	int                         shader_log = 0;
	int                         printf_dir = 0;
	bool                        readback = false;
	bool                        readback_tiled = false;
	bool                        tessellation = false;
	bool                        sync_raw = false;
	bool                        hdr = false;
	bool                        vulkan_validation = false;
	bool                        shader_validation = false;
	bool                        renderdoc = false;
	bool                        cmd_dump = false;
	bool                        profiler = false;
	std::string                 printf_file; // empty = emulator default
	std::string                 shader_log_folder; // empty = emulator default
	std::string                 cmd_dump_folder; // empty = emulator default
	// Extra environment for the emulator process only (the launcher's own environment is untouched). Saved
	// as the "env" object under "emu" in kyty.json: {"NAME": "value"}; an empty value removes NAME from the
	// child's environment. Edited on Settings > Advanced (global and per-game mode): a row with an empty value
	// leaves the variable out of the table.
	std::map<std::string, std::string> env;
	// Keyboard/pad bindings, one `--keymap <entry>` each ("Pad:Cross=South", "Keyboard=Disabled"), as the Qt launcher's
	// input mapping dialog wrote them to host_input_mapping. Saved as the "host_input_mapping" array under "emu".
	// Edited on the Controls page (src/input_mapping.*).
	std::vector<std::string> host_input_mapping;
	// Version Manager
	std::string versions_dir;  // downloaded builds; empty = <data root>\versions
	// Lists are stored as JSON arrays, so a path may hold any character.
	std::vector<std::string> custom_builds; // emulator exes added with Add Custom
	std::vector<std::string> sources;       // extra GitHub release sources (owner/repo) added with +
	std::vector<std::string> hidden_builds; // builds taken off the Installed list (folders stay on disk)
	std::vector<std::string> build_names;   // "exe|name" display names given with Rename
	std::vector<std::string> menu_hidden;   // build exes unticked in the Version Manager (not in the toolbar menu)
	std::vector<std::string> source_names;  // "owner/repo|name" tab names given with Rename
	// STEP6:6d app keys: begin
	std::vector<std::string> pinned_builds;  // build exes pinned to the top of the Installed table, in pin order
	std::string              vm_sort      = "name"; // Installed table sort: name, codename, date, tracked, path
	bool                     vm_sort_desc = false;
	// STEP6:6d app keys: end
	std::vector<std::string> hidden_games;  // title ids (eboot path for a game without one) taken off the library (files stay on disk)
	// STEP7:3 app keys: begin
	// Look for newer "-latest" builds when the launcher starts and every hour while it is open. Off: only the
	// Version Manager's own Update buttons fetch anything.
	bool check_updates = true;
	// STEP7:3 app keys: end
	// STEP6:6e app keys: begin
	// The launcher window's last geometry. Not the emulator's window size (width/height above are that).
	// window_x/window_y are the screen position (-1: let the window manager place it); window_w/window_h
	// 0 means the Slint default (the page's preferred-width/height).
	int window_x = -1;
	int window_y = -1;
	int window_w = 0;
	int window_h = 0;
	// STEP6:6e app keys: end
	// STEP6:6c app keys: begin
	std::string              library_sort      = "name"; // library sort key: name, status, serial, firmware, size, version, playtime, path, comment
	bool                     library_sort_desc = false;  // descending instead of ascending
	bool                     library_grid      = false;  // grid view instead of the table
	std::vector<std::string> favourite_games;            // "Pin to top": title ids (eboot path for a game without one)
	// STEP6:6c app keys: end
	// The "emu" keys that were present in the user file. Together with "differs from the default" this decides
	// which keys the user layer sets (written back on save; a build's config files may still override them at
	// launch - see resolve_for_launch).
	std::set<std::string> user_keys;
};

Settings default_settings();

// Reads the user file (configs/kyty.json): missing keys keep their defaults. Returns false when the file is
// absent, unreadable or written by a newer launcher (version > kConfigVersion); then `s` is untouched and
// `note` (if given) says why for the newer case. save_settings never overwrites such a file.
bool load_settings(const std::filesystem::path& path, Settings& s, std::string* note = nullptr);
bool save_settings(const std::filesystem::path& path, const Settings& s);

// Applies the "emu" object of a sparse layer file (a build's cfgGlobal.json or a per-game file) to `s`: only the
// keys it holds change, and "env" merges by name. Returns whether it was applied; `status` says why not
// ("not found", "unreadable, ignored", "newer version N, ignored") or "applied".
bool apply_layer_file(const std::filesystem::path& path, Settings& s, std::string* status = nullptr);

// The settings the emulator is started with: `user` (kyty.json) < build config (cfgGlobal.json, cfg<TITLEID>.json
// next to the emulator) < configs/games/<TITLEID>.json. A build's config files are requirements, not defaults:
// they add and overwrite on top of the user's own values, for the launches that use that build only. Nothing is
// written back, so another build (or one without those files) brings the user's values back.
// `report` gets one line per layer file looked at, naming the keys a layer set and the ones it took from the user.
// `overrides` gets one sentence per build layer that beat a saved value (the launcher shows those as warnings).
// `with_game_file` false leaves the per-game layer out: what the game would get without its file (the baseline
// the per-game settings page saves its differences from).
Settings resolve_for_launch(const Settings& user, const Paths& paths, const std::string& title_id, std::vector<std::string>* report = nullptr,
                            bool with_game_file = true, std::vector<std::string>* overrides = nullptr);

// Per-game file configs/games/<TITLEID>.json:
//   { "version": 1, "emulator": "<exe path>", "emu": { ...sparse keys, as in kyty.json "emu"... } }
// "emulator" is the game's emulator pin (outside "emu": it is not a setting of the emulator). Both parts are optional;
// a file left with neither is deleted. A file from a newer launcher, or one that can't be read, is never overwritten
// (the functions below return false).
std::string read_game_emulator(const Paths& paths, const std::string& title_id); // "" = no pin (use the default emulator)
bool        set_game_emulator(const Paths& paths, const std::string& title_id, const std::string& exe); // "" removes the pin
// Writes only the "emu" keys of `edited` that differ from `baseline`; keeps the pin. The "env" table is
// written as its differences too: a variable the page changed gets its new value, one it removed gets an
// empty value ("unset in the child's environment"), and values inherited from another layer are never
// baked into the file (variables the page did not touch are left exactly as the file held them).
bool save_game_overrides(const Paths& paths, const std::string& title_id, const Settings& baseline, const Settings& edited);
bool reset_game_overrides(const Paths& paths, const std::string& title_id); // drops "emu", keeps the pin

// Kyty.ini import: applies one value of [GlobalConfiguration] or of a game's entry (the text as QSettings holds it:
// "R2560X1440", "Mailbox", "true", "60") to the "emu" key of the same name, and marks the key as set by the user.
// Returns 1 when applied, 0 when `key` is not an emulator key (also "env" and "host_input_mapping", which are lists),
// -1 when the value is invalid (`why` says how); `s` is unchanged then.
int apply_qt_emu_value(Settings& s, const std::string& key, const std::string& text, std::string* why = nullptr);

// Name shown for a build exe that is not (or not known to be) in the Version Manager's list: the name given with
// Rename, else the build's folder name.
std::string build_display_name(const Settings& s, const std::string& exe);

} // namespace kyty_launcher
