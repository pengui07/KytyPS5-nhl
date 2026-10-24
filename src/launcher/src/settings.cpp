#include "settings.h"

#include "paths.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace kyty_launcher {

Settings default_settings()
{
	Settings s;
	// No machine-specific paths: main() fills in the emulator and run folder it finds next to the launcher.
	return s;
}

namespace {

class JsonParser
{
public:
	explicit JsonParser(const std::string& text): m_t(text) {}

	bool Parse(JsonValue& v)
	{
		return Value(v) && (Ws(), m_p == m_t.size());
	}

private:
	// Whitespace and // line comments (hand-written config files). Strings never reach here, so a
	// "//" inside one ("https://...") is kept.
	void Ws()
	{
		for (;;)
		{
			while (m_p < m_t.size() && std::isspace(static_cast<unsigned char>(m_t[m_p])) != 0)
			{
				m_p++;
			}
			if (m_t.compare(m_p, 2, "//") != 0)
			{
				return;
			}
			while (m_p < m_t.size() && m_t[m_p] != '\n')
			{
				m_p++;
			}
		}
	}

	bool Lit(const char* w)
	{
		size_t len = std::char_traits<char>::length(w);
		if (m_t.compare(m_p, len, w) != 0)
		{
			return false;
		}
		m_p += len;
		return true;
	}

	bool Str(std::string& out)
	{
		if (m_t[m_p] != '"')
		{
			return false;
		}
		m_p++;
		while (m_p < m_t.size() && m_t[m_p] != '"')
		{
			char c = m_t[m_p++];
			if (c != '\\')
			{
				out += c;
				continue;
			}
			if (m_p >= m_t.size())
			{
				return false;
			}
			char e = m_t[m_p++];
			switch (e)
			{
				case 'n': out += '\n'; break;
				case 't': out += '\t'; break;
				case 'r': out += '\r'; break;
				case 'b': out += '\b'; break;
				case 'f': out += '\f'; break;
				case 'u':
				{
					if (m_p + 4 > m_t.size())
					{
						return false;
					}
					unsigned cp = std::stoul(m_t.substr(m_p, 4), nullptr, 16);
					m_p += 4;
					// BMP only; the writer never emits \u for anything else.
					if (cp < 0x80)
					{
						out += static_cast<char>(cp);
					} else if (cp < 0x800)
					{
						out += static_cast<char>(0xC0 | (cp >> 6));
						out += static_cast<char>(0x80 | (cp & 0x3F));
					} else
					{
						out += static_cast<char>(0xE0 | (cp >> 12));
						out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
						out += static_cast<char>(0x80 | (cp & 0x3F));
					}
					break;
				}
				default: out += e; break;
			}
		}
		if (m_p >= m_t.size())
		{
			return false;
		}
		m_p++;
		return true;
	}

	bool Value(JsonValue& v)
	{
		Ws();
		if (m_p >= m_t.size())
		{
			return false;
		}
		char c = m_t[m_p];
		if (c == '{')
		{
			v.type = JsonValue::Type::Object;
			m_p++;
			Ws();
			if (m_p < m_t.size() && m_t[m_p] == '}')
			{
				m_p++;
				return true;
			}
			for (;;)
			{
				Ws();
				std::string key;
				if (m_p >= m_t.size() || !Str(key))
				{
					return false;
				}
				Ws();
				if (m_p >= m_t.size() || m_t[m_p] != ':')
				{
					return false;
				}
				m_p++;
				if (!Value(v.o[key]))
				{
					return false;
				}
				Ws();
				if (m_p < m_t.size() && m_t[m_p] == ',')
				{
					m_p++;
					continue;
				}
				if (m_p < m_t.size() && m_t[m_p] == '}')
				{
					m_p++;
					return true;
				}
				return false;
			}
		}
		if (c == '[')
		{
			v.type = JsonValue::Type::Array;
			m_p++;
			Ws();
			if (m_p < m_t.size() && m_t[m_p] == ']')
			{
				m_p++;
				return true;
			}
			for (;;)
			{
				v.a.emplace_back();
				if (!Value(v.a.back()))
				{
					return false;
				}
				Ws();
				if (m_p < m_t.size() && m_t[m_p] == ',')
				{
					m_p++;
					continue;
				}
				if (m_p < m_t.size() && m_t[m_p] == ']')
				{
					m_p++;
					return true;
				}
				return false;
			}
		}
		if (c == '"')
		{
			v.type = JsonValue::Type::String;
			return Str(v.s);
		}
		if (Lit("true"))
		{
			v.type = JsonValue::Type::Bool;
			v.b    = true;
			return true;
		}
		if (Lit("false"))
		{
			v.type = JsonValue::Type::Bool;
			return true;
		}
		if (Lit("null"))
		{
			return true;
		}
		size_t start = m_p;
		while (m_p < m_t.size() && m_t[m_p] != 0 &&
		       (std::isdigit(static_cast<unsigned char>(m_t[m_p])) != 0 || std::strchr("+-.eE", m_t[m_p]) != nullptr))
		{
			m_p++;
		}
		if (start == m_p)
		{
			return false;
		}
		v.type = JsonValue::Type::Number;
		v.n    = std::strtod(m_t.substr(start, m_p - start).c_str(), nullptr);
		return true;
	}

	const std::string& m_t;
	size_t             m_p = 0;
};

std::string Quote(const std::string& s)
{
	std::string out = "\"";
	for (char c: s)
	{
		switch (c)
		{
			case '"': out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n"; break;
			case '\r': out += "\\r"; break;
			case '\t': out += "\\t"; break;
			default:
				if (static_cast<unsigned char>(c) < 0x20)
				{
					char buf[8];
					std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
					out += buf;
				} else
				{
					out += c;
				}
		}
	}
	return out + "\"";
}

std::string QuoteList(const std::vector<std::string>& items)
{
	std::string out = "[";
	for (size_t i = 0; i < items.size(); i++)
	{
		out += (i == 0 ? "" : ", ") + Quote(items[i]);
	}
	return out + "]";
}

// The emulator's enum names, as Kyty.ini and the command line spell them; the index is the UI's.
const std::vector<const char*> kPresentModes        = {"Fifo", "Mailbox", "Immediate"};
const std::vector<const char*> kShaderOptimizations = {"None", "Size", "Performance"};
const std::vector<const char*> kLogDirections       = {"Silent", "Console", "File"};

const char* Name(const std::vector<const char*>& names, int i)
{
	return names[i >= 0 && i < static_cast<int>(names.size()) ? i : 0];
}

std::vector<std::string> SplitList(const std::string& s)
{
	std::vector<std::string> out;
	for (size_t pos = 0; pos <= s.size();)
	{
		size_t semi = s.find(';', pos);
		if (semi == std::string::npos)
		{
			semi = s.size();
		}
		if (semi > pos)
		{
			out.push_back(s.substr(pos, semi - pos));
		}
		pos = semi + 1;
	}
	return out;
}

std::string JoinList(const std::vector<std::string>& items)
{
	std::string out;
	for (const auto& i: items)
	{
		out += (out.empty() ? "" : ";") + i;
	}
	return out;
}

// Typed readers over one JSON object; a key of the wrong type is ignored. `seen` collects the keys found.
struct KeyReader
{
	const JsonValue&       root;
	std::set<std::string>* seen = nullptr;

	const JsonValue* find(const char* key, JsonValue::Type type) const
	{
		auto it = root.o.find(key);
		if (it == root.o.end() || it->second.type != type)
		{
			return nullptr;
		}
		if (seen != nullptr)
		{
			seen->insert(key);
		}
		return &it->second;
	}
	void str(const char* key, std::string& out) const
	{
		if (const auto* v = find(key, JsonValue::Type::String))
		{
			out = v->s;
		}
	}
	// A JSON array of strings.
	void list(const char* key, std::vector<std::string>& out) const
	{
		if (const auto* v = find(key, JsonValue::Type::Array))
		{
			out.clear();
			for (const auto& e: v->a)
			{
				if (e.type == JsonValue::Type::String && !e.s.empty())
				{
					out.push_back(e.s);
				}
			}
		}
	}
	void num(const char* key, int& out) const
	{
		if (const auto* v = find(key, JsonValue::Type::Number))
		{
			out = static_cast<int>(v->n);
		}
	}
	void flag(const char* key, bool& out) const
	{
		if (const auto* v = find(key, JsonValue::Type::Bool))
		{
			out = v->b;
		}
	}
	// An emulator enum, by its name ("Mailbox").
	void choice(const char* key, const std::vector<const char*>& names, int& out) const
	{
		if (const auto* v = find(key, JsonValue::Type::String))
		{
			for (size_t i = 0; i < names.size(); i++)
			{
				if (v->s == names[i])
				{
					out = static_cast<int>(i);
				}
			}
		}
	}
};

// The "app" keys: launcher settings that are never layered.
void ReadApp(const JsonValue& root, Settings& s)
{
	const KeyReader r {root};
	r.str("emulator", s.emulator);
	r.str("game", s.game);
	r.str("run_dir", s.run_dir);
	if (root.o.count("game_dirs") != 0)
	{
		std::vector<std::string> dirs;
		r.list("game_dirs", dirs);
		s.game_dirs = JoinList(dirs);
	}
	r.str("addons_dir", s.addons_dir);
	r.str("columns", s.columns);
	// STEP6:6e app keys: begin
	r.num("window_x", s.window_x);
	r.num("window_y", s.window_y);
	r.num("window_w", s.window_w);
	r.num("window_h", s.window_h);
	// STEP6:6e app keys: end
	r.str("versions_dir", s.versions_dir);
	r.list("custom_builds", s.custom_builds);
	r.list("sources", s.sources);
	r.list("hidden_builds", s.hidden_builds);
	r.list("hidden_games", s.hidden_games);
	// STEP7:3 app keys: begin
	r.flag("check_updates", s.check_updates);
	// STEP7:3 app keys: end
	r.list("source_names", s.source_names);
	r.list("menu_hidden", s.menu_hidden);
	r.list("build_names", s.build_names);
	// STEP6:6d app keys: begin
	r.list("pinned_builds", s.pinned_builds);
	r.str("vm_sort", s.vm_sort);
	r.flag("vm_sort_desc", s.vm_sort_desc);
	// STEP6:6d app keys: end
	// STEP6:6c app keys: begin
	r.str("library_sort", s.library_sort);
	r.flag("library_sort_desc", s.library_sort_desc);
	r.flag("library_grid", s.library_grid);
	r.list("favourite_games", s.favourite_games);
	// STEP6:6c app keys: end
}

// The "emu" keys, the ones every config layer may carry (keys follow the official launcher's Kyty.ini
// [GlobalConfiguration]). Only keys present in `root` change `s`; `env` merges by name.
void ReadEmu(const JsonValue& root, Settings& s, std::set<std::string>* seen)
{
	const KeyReader r {root, seen};
	if (const auto* v = r.find("screen_resolution", JsonValue::Type::String))
	{
		int w = 0;
		int h = 0;
		if (std::sscanf(v->s.c_str(), "R%dX%d", &w, &h) == 2 && w > 0 && h > 0)
		{
			s.width  = w;
			s.height = h;
		}
	}
	r.str("extra_args", s.extra_args);
	r.choice("present_mode", kPresentModes, s.present_mode);
	r.num("vblank_frequency", s.vblank);
	r.choice("shader_optimization_type", kShaderOptimizations, s.shader_opt);
	r.num("gpu_index", s.gpu_index);
	r.choice("shader_log_direction", kLogDirections, s.shader_log);
	r.choice("printf_direction", kLogDirections, s.printf_dir);
	r.str("printf_output_file", s.printf_file);
	r.str("shader_log_folder", s.shader_log_folder);
	r.str("command_buffer_dump_folder", s.cmd_dump_folder);
	r.flag("readback_linear_images", s.readback);
	r.flag("readback_tiled_images", s.readback_tiled);
	r.flag("tessellation_enabled", s.tessellation);
	r.flag("sync_raw_image_buffers", s.sync_raw);
	r.flag("hdr_enabled", s.hdr);
	r.flag("vulkan_validation_enabled", s.vulkan_validation);
	r.flag("shader_validation_enabled", s.shader_validation);
	r.flag("renderdoc_enabled", s.renderdoc);
	r.flag("command_buffer_dump_enabled", s.cmd_dump);
	r.flag("profiler_enabled", s.profiler);
	r.str("user_name", s.user_name);
	r.num("user_id", s.user_id);
	r.num("console_language", s.console_language);
	r.str("audio_input_device", s.microphone);
	r.flag("amd_cpu_enabled", s.amd_cpu);
	r.flag("red_zone_protection_enabled", s.red_zone);
	r.flag("trophy_enabled", s.trophy_notifications);
	r.num("trophy_duration_seconds", s.trophy_duration);
	s.trophy_duration = std::clamp(s.trophy_duration, 5, 10);
	r.flag("fullscreen_enabled", s.fullscreen);
	r.flag("hide_cursor_enabled", s.hide_cursor);
	r.flag("perf_overlay_enabled", s.perf_overlay);
	r.flag("skip_notice_screen", s.skip_notice_screen);
	r.flag("network_connected", s.network_connected);
	r.str("controller_color", s.controller_color);
	r.list("host_input_mapping", s.host_input_mapping);
	r.num("controller_vibration_intensity", s.controller_vibration);
	r.num("controller_speaker_volume", s.controller_volume);
	r.num("controller_keyboard_player", s.keyboard_player);
	s.controller_vibration = std::clamp(s.controller_vibration, 0, 100);
	s.controller_volume    = std::clamp(s.controller_volume, 0, 100);
	s.keyboard_player      = std::clamp(s.keyboard_player, 0, 4);
	// Only string values are taken; an empty one means "unset in the child".
	if (const auto* v = r.find("env", JsonValue::Type::Object))
	{
		for (const auto& [k, e]: v->o)
		{
			if (e.type == JsonValue::Type::String)
			{
				s.env[k] = e.s;
			}
		}
	}
}

// Each "emu" key of `s` with its value as JSON text, in file order. Used to write the file and to compare
// a value with the default.
using EmuEntries = std::vector<std::pair<const char*, std::string>>;

EmuEntries EmuKeys(const Settings& s)
{
	auto      b = [](bool v) { return std::string(v ? "true" : "false"); };
	EmuEntries e;
	e.emplace_back("screen_resolution", Quote("R" + std::to_string(s.width) + "X" + std::to_string(s.height)));
	e.emplace_back("extra_args", Quote(s.extra_args));
	e.emplace_back("present_mode", Quote(Name(kPresentModes, s.present_mode)));
	e.emplace_back("vblank_frequency", std::to_string(s.vblank));
	e.emplace_back("shader_optimization_type", Quote(Name(kShaderOptimizations, s.shader_opt)));
	e.emplace_back("gpu_index", std::to_string(s.gpu_index));
	e.emplace_back("shader_log_direction", Quote(Name(kLogDirections, s.shader_log)));
	e.emplace_back("printf_direction", Quote(Name(kLogDirections, s.printf_dir)));
	e.emplace_back("printf_output_file", Quote(s.printf_file));
	e.emplace_back("shader_log_folder", Quote(s.shader_log_folder));
	e.emplace_back("command_buffer_dump_folder", Quote(s.cmd_dump_folder));
	e.emplace_back("readback_linear_images", b(s.readback));
	e.emplace_back("readback_tiled_images", b(s.readback_tiled));
	e.emplace_back("tessellation_enabled", b(s.tessellation));
	e.emplace_back("sync_raw_image_buffers", b(s.sync_raw));
	e.emplace_back("hdr_enabled", b(s.hdr));
	e.emplace_back("vulkan_validation_enabled", b(s.vulkan_validation));
	e.emplace_back("shader_validation_enabled", b(s.shader_validation));
	e.emplace_back("renderdoc_enabled", b(s.renderdoc));
	e.emplace_back("command_buffer_dump_enabled", b(s.cmd_dump));
	e.emplace_back("profiler_enabled", b(s.profiler));
	e.emplace_back("user_name", Quote(s.user_name));
	e.emplace_back("user_id", std::to_string(s.user_id));
	e.emplace_back("console_language", std::to_string(s.console_language));
	e.emplace_back("audio_input_device", Quote(s.microphone));
	e.emplace_back("amd_cpu_enabled", b(s.amd_cpu));
	e.emplace_back("red_zone_protection_enabled", b(s.red_zone));
	e.emplace_back("trophy_enabled", b(s.trophy_notifications));
	e.emplace_back("trophy_duration_seconds", std::to_string(std::clamp(s.trophy_duration, 5, 10)));
	e.emplace_back("fullscreen_enabled", b(s.fullscreen));
	e.emplace_back("hide_cursor_enabled", b(s.hide_cursor));
	e.emplace_back("perf_overlay_enabled", b(s.perf_overlay));
	e.emplace_back("skip_notice_screen", b(s.skip_notice_screen));
	e.emplace_back("network_connected", b(s.network_connected));
	e.emplace_back("controller_color", Quote(s.controller_color));
	e.emplace_back("controller_vibration_intensity", std::to_string(std::clamp(s.controller_vibration, 0, 100)));
	e.emplace_back("controller_speaker_volume", std::to_string(std::clamp(s.controller_volume, 0, 100)));
	e.emplace_back("controller_keyboard_player", std::to_string(std::clamp(s.keyboard_player, 0, 4)));
	e.emplace_back("host_input_mapping", QuoteList(s.host_input_mapping));
	std::string env = "{";
	for (const auto& [k, v]: s.env)
	{
		env += (env.size() > 1 ? ", " : "") + Quote(k) + ": " + Quote(v);
	}
	e.emplace_back("env", env + "}");
	return e;
}

// The user file keeps only the "emu" keys the user set: ones that were in the file, or that now differ from
// the built-in default. Everything else stays absent so the build config (cfgGlobal.json) can supply it.
bool UserSet(const Settings& user, const EmuEntries& values, const EmuEntries& defaults, size_t i)
{
	return user.user_keys.count(values[i].first) != 0 || values[i].second != defaults[i].second;
}

bool ReadJsonFile(const std::filesystem::path& path, JsonValue& root)
{
	std::ifstream f(path, std::ios::binary);
	if (!f)
	{
		return false;
	}
	std::stringstream ss;
	ss << f.rdbuf();
	std::string text = ss.str();
	// Editors (and PowerShell's Set-Content -Encoding utf8) may add a UTF-8 BOM.
	if (text.rfind("\xEF\xBB\xBF", 0) == 0)
	{
		text.erase(0, 3);
	}
	return JsonParser(text).Parse(root) && root.type == JsonValue::Type::Object;
}

// The file's top-level "version"; a missing one is 1.
int SchemaVersion(const JsonValue& root)
{
	auto it = root.o.find("version");
	return it != root.o.end() && it->second.type == JsonValue::Type::Number ? static_cast<int>(it->second.n) : 1;
}

const JsonValue* Section(const JsonValue& root, const char* name)
{
	auto it = root.o.find(name);
	return it != root.o.end() && it->second.type == JsonValue::Type::Object ? &it->second : nullptr;
}

void Dump(const JsonValue& v, std::string& out, int depth)
{
	switch (v.type)
	{
		case JsonValue::Type::Null: out += "null"; break;
		case JsonValue::Type::Bool: out += v.b ? "true" : "false"; break;
		case JsonValue::Type::Number:
		{
			char buf[40];
			if (v.n == static_cast<double>(static_cast<long long>(v.n)))
			{
				std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v.n));
			} else
			{
				std::snprintf(buf, sizeof(buf), "%.17g", v.n);
			}
			out += buf;
			break;
		}
		case JsonValue::Type::String: out += Quote(v.s); break;
		case JsonValue::Type::Array:
			out += "[";
			for (size_t i = 0; i < v.a.size(); i++)
			{
				out += i == 0 ? "" : ", ";
				Dump(v.a[i], out, depth + 1);
			}
			out += "]";
			break;
		case JsonValue::Type::Object:
		{
			if (v.o.empty())
			{
				out += "{}";
				break;
			}
			const std::string pad(static_cast<size_t>(depth + 1) * 2, ' ');
			out += "{\n";
			size_t i = 0;
			for (const auto& [k, e]: v.o)
			{
				out += pad + Quote(k) + ": ";
				Dump(e, out, depth + 1);
				out += ++i < v.o.size() ? ",\n" : "\n";
			}
			out += std::string(static_cast<size_t>(depth) * 2, ' ') + "}";
			break;
		}
	}
}

// The per-game file as a tree; a missing file is an empty object. False when it exists but must not be
// rewritten (unreadable, or from a newer launcher).
bool LoadGameRoot(const std::filesystem::path& file, JsonValue& root)
{
	std::error_code ec;
	root = JsonValue();
	root.type = JsonValue::Type::Object;
	if (!std::filesystem::exists(file, ec))
	{
		return true;
	}
	JsonValue parsed;
	if (!ReadJsonFile(file, parsed) || SchemaVersion(parsed) > kConfigVersion)
	{
		return false;
	}
	root = std::move(parsed);
	return true;
}

bool IsEmptyObject(const JsonValue* v)
{
	return v == nullptr || (v->type == JsonValue::Type::Object && v->o.empty());
}

// Writes the per-game file: "version", "emulator", "emu" first, then any other key. A file with neither a pin
// nor "emu" is removed instead.
bool WriteGameRoot(const std::filesystem::path& file, JsonValue root)
{
	std::error_code ec;
	auto            emulator = root.o.find("emulator");
	if (emulator != root.o.end() && (emulator->second.type != JsonValue::Type::String || emulator->second.s.empty()))
	{
		root.o.erase(emulator);
	}
	auto emu = root.o.find("emu");
	if (emu != root.o.end() && IsEmptyObject(&emu->second))
	{
		root.o.erase(emu);
	}
	root.o.erase("version");
	if (root.o.empty())
	{
		std::filesystem::remove(file, ec);
		return !std::filesystem::exists(file, ec);
	}
	std::string out = "{\n  \"version\": " + std::to_string(kConfigVersion);
	for (const char* first: {"emulator", "emu"})
	{
		if (auto it = root.o.find(first); it != root.o.end())
		{
			out += ",\n  " + Quote(first) + ": ";
			Dump(it->second, out, 1);
		}
	}
	for (const auto& [k, e]: root.o)
	{
		if (k != "emulator" && k != "emu")
		{
			out += ",\n  " + Quote(k) + ": ";
			Dump(e, out, 1);
		}
	}
	out += "\n}\n";
	std::filesystem::create_directories(file.parent_path(), ec);
	std::filesystem::path tmp = file;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		if (!f || !(f << out))
		{
			return false;
		}
	}
	std::filesystem::rename(tmp, file, ec);
	return !ec;
}

} // namespace

bool parse_json(const std::string& text, JsonValue& out)
{
	return JsonParser(text).Parse(out);
}

bool load_settings(const std::filesystem::path& path, Settings& s, std::string* note)
{
	JsonValue root;
	if (!ReadJsonFile(path, root))
	{
		return false;
	}
	if (SchemaVersion(root) > kConfigVersion)
	{
		if (note != nullptr)
		{
			*note = path_utf8(path) + " is from a newer launcher (version " + std::to_string(SchemaVersion(root)) +
			        "): using defaults and not saving over it";
		}
		return false;
	}
	if (const auto* app = Section(root, "app"))
	{
		ReadApp(*app, s);
	}
	if (const auto* emu = Section(root, "emu"))
	{
		ReadEmu(*emu, s, &s.user_keys);
	}
	return true;
}

bool apply_layer_file(const std::filesystem::path& path, Settings& s, std::string* status)
{
	std::string  ignore;
	std::string& st = status != nullptr ? *status : ignore;
	std::error_code ec;
	if (!std::filesystem::exists(path, ec))
	{
		st = "not found";
		return false;
	}
	JsonValue root;
	if (!ReadJsonFile(path, root))
	{
		st = "unreadable, ignored";
		return false;
	}
	if (SchemaVersion(root) > kConfigVersion)
	{
		st = "newer version " + std::to_string(SchemaVersion(root)) + ", ignored";
		return false;
	}
	if (const auto* emu = Section(root, "emu"))
	{
		ReadEmu(*emu, s, nullptr);
	}
	st = "applied";
	return true;
}

int apply_qt_emu_value(Settings& s, const std::string& key, const std::string& text, std::string* why)
{
	std::string  ignore;
	std::string& reason = why != nullptr ? *why : ignore;
	if (key == "env" || key == "host_input_mapping")
	{
		return 0;
	}
	const EmuEntries defaults = EmuKeys(default_settings());
	auto             def      = std::find_if(defaults.begin(), defaults.end(), [&](const auto& e) { return key == e.first; });
	if (def == defaults.end())
	{
		return 0;
	}
	std::string trimmed = text;
	while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back())) != 0)
	{
		trimmed.pop_back();
	}
	while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front())) != 0)
	{
		trimmed.erase(trimmed.begin());
	}
	auto bad = [&](const std::string& msg)
	{
		reason = msg;
		return -1;
	};

	JsonValue obj;
	obj.type      = JsonValue::Type::Object;
	JsonValue& v  = obj.o[key];
	const std::string& dflt = def->second;
	if (dflt == "true" || dflt == "false")
	{
		std::string low = trimmed;
		std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (low != "true" && low != "false" && low != "1" && low != "0")
		{
			return bad("\"" + text + "\" is not true or false");
		}
		v.type = JsonValue::Type::Bool;
		v.b    = low == "true" || low == "1";
	} else if (dflt[0] == '"')
	{
		v.type = JsonValue::Type::String;
		v.s    = text;
		const std::vector<const char*>* names = key == "present_mode"               ? &kPresentModes
		                                        : key == "shader_optimization_type" ? &kShaderOptimizations
		                                        : key == "shader_log_direction" || key == "printf_direction" ? &kLogDirections
		                                                                                                      : nullptr;
		if (names != nullptr)
		{
			// The enum's name; a number is its index (a QVariant of the enum may be stored that way).
			bool found = false;
			for (size_t i = 0; i < names->size(); i++)
			{
				if (trimmed == (*names)[i] || trimmed == std::to_string(i))
				{
					v.s   = (*names)[i];
					found = true;
				}
			}
			if (!found)
			{
				return bad("unknown value \"" + text + "\"");
			}
		} else if (key == "screen_resolution")
		{
			// "R2560X1440" (as stored), or "2560x1440" (as shown), or the index of the Qt launcher's four sizes.
			static const char* const kQtSizes[] = {"R1280X720", "R1920X1080", "R2560X1440", "R3840X2160"};
			int                      w = 0;
			int                      h = 0;
			std::string              up = trimmed;
			std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
			if (up.size() == 1 && up[0] >= '0' && up[0] <= '3')
			{
				up = kQtSizes[up[0] - '0'];
			}
			if (up.rfind("R", 0) == 0)
			{
				up.erase(0, 1);
			}
			if (std::sscanf(up.c_str(), "%dX%d", &w, &h) != 2 || w <= 0 || h <= 0)
			{
				return bad("unknown resolution \"" + text + "\"");
			}
			v.s = "R" + std::to_string(w) + "X" + std::to_string(h);
		} else if (key == "controller_color")
		{
			std::string hex = trimmed;
			if (!hex.empty() && hex[0] == '#')
			{
				hex.erase(0, 1);
			}
			if (hex.empty())
			{
				v.s.clear();
			} else if (hex.size() != 6 || hex.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
			{
				return bad("\"" + text + "\" is not a #rrggbb colour");
			} else
			{
				std::transform(hex.begin(), hex.end(), hex.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
				v.s = "#" + hex;
			}
		}
	} else
	{
		char*      end = nullptr;
		const long n   = std::strtol(trimmed.c_str(), &end, 10);
		if (trimmed.empty() || end == nullptr || *end != '\0')
		{
			return bad("\"" + text + "\" is not a number");
		}
		if ((key == "console_language" && (n < 0 || n > 29)) || (key == "vblank_frequency" && (n < 30 || n > 360)) ||
		    (key == "gpu_index" && n < -1))
		{
			return bad(std::to_string(n) + " is out of range");
		}
		v.type = JsonValue::Type::Number;
		v.n    = static_cast<double>(n);
	}
	ReadEmu(obj, s, &s.user_keys);
	return 1;
}

Settings resolve_for_launch(const Settings& user, const Paths& paths, const std::string& title_id, std::vector<std::string>* report,
                            bool with_game_file, std::vector<std::string>* overrides)
{
	auto note = [&](const char* kind, const std::filesystem::path& file, const std::string& status)
	{
		if (report != nullptr)
		{
			report->push_back(std::string(kind) + " " + path_utf8(file) + ": " + status);
		}
	};
	auto join = [](const std::vector<std::string>& names)
	{
		std::string text;
		for (size_t i = 0; i < names.size(); i++)
		{
			text += (i == 0 ? std::string() : ", ") + names[i];
		}
		return text;
	};

	// The user's kyty.json is the base. A build's config files are requirements, not defaults: they add and
	// overwrite on top of it. Nothing here is written back - this Settings only becomes the command line - so a
	// build without those files, or any other build, brings the user's own values back.
	Settings              r        = user;
	const EmuEntries      defaults = EmuKeys(default_settings());
	const EmuEntries      mine     = EmuKeys(user);
	std::set<std::string> build_touched; // keys a build config file set (whatever the value)

	// Applies one sparse layer, then names the keys it changed.
	auto apply = [&](const char* kind, const std::filesystem::path& file, bool is_build)
	{
		const EmuEntries before = EmuKeys(r);
		std::string      st;
		if (!apply_layer_file(file, r, &st))
		{
			note(kind, file, st);
			return;
		}
		const EmuEntries         after = EmuKeys(r);
		std::vector<std::string> changed;
		for (size_t i = 0; i < before.size() && i < after.size(); i++)
		{
			if (before[i].second == after[i].second)
			{
				continue;
			}
			changed.emplace_back(after[i].first);
			if (is_build)
			{
				build_touched.emplace(after[i].first);
			}
		}
		std::string text = st;
		if (!changed.empty())
		{
			text += ", " + std::to_string(changed.size()) + " key(s) set: " + join(changed);
		}
		note(kind, file, text);
	};

	std::error_code ec;
	note("user ", paths.settings_file(), std::filesystem::exists(paths.settings_file(), ec) ? "loaded" : "not found");

	// Build config: requirements of the build the emulator pin chose, so they go over the user's values.
	if (!user.emulator.empty())
	{
		const auto dir = utf8_path(user.emulator).parent_path();
		apply("build", dir / "cfgGlobal.json", true);
		if (!title_id.empty())
		{
			apply("build", dir / ("cfg" + title_id + ".json"), true);
		}
	}
	else if (report != nullptr)
	{
		report->push_back("build: no emulator set, no build config looked for");
	}

	// Per-game file last: a choice made for this game outranks the build's files too.
	if (!title_id.empty() && with_game_file)
	{
		apply("game ", paths.game_config_file(title_id), false);
	}

	// Only warn where the value the emulator gets really differs from the saved one: a per-game file that sets
	// the key back means the build did not win after all.
	const EmuEntries         final_state = EmuKeys(r);
	std::vector<std::string> took;
	for (size_t i = 0; i < final_state.size() && i < mine.size(); i++)
	{
		if (final_state[i].second != mine[i].second && build_touched.count(final_state[i].first) != 0 && UserSet(user, mine, defaults, i))
		{
			took.emplace_back(final_state[i].first);
		}
	}
	if (!took.empty() && overrides != nullptr)
	{
		overrides->push_back("this build's config files set " + join(took) + " over the saved value");
	}
	return r;
}

std::string read_game_emulator(const Paths& paths, const std::string& title_id)
{
	JsonValue root;
	if (title_id.empty() || !ReadJsonFile(paths.game_config_file(title_id), root) || SchemaVersion(root) > kConfigVersion)
	{
		return std::string();
	}
	auto it = root.o.find("emulator");
	return it != root.o.end() && it->second.type == JsonValue::Type::String ? it->second.s : std::string();
}

bool set_game_emulator(const Paths& paths, const std::string& title_id, const std::string& exe)
{
	JsonValue  root;
	const auto file = paths.game_config_file(title_id);
	if (title_id.empty() || !LoadGameRoot(file, root))
	{
		return false;
	}
	if (exe.empty())
	{
		root.o.erase("emulator");
	} else
	{
		root.o["emulator"].type = JsonValue::Type::String;
		root.o["emulator"].s    = exe;
	}
	return WriteGameRoot(file, std::move(root));
}

bool save_game_overrides(const Paths& paths, const std::string& title_id, const Settings& baseline, const Settings& edited)
{
	JsonValue  root;
	const auto file = paths.game_config_file(title_id);
	if (title_id.empty() || !LoadGameRoot(file, root))
	{
		return false;
	}
	JsonValue emu;
	emu.type = JsonValue::Type::Object;
	const EmuEntries was = EmuKeys(baseline);
	const EmuEntries now = EmuKeys(edited);
	for (size_t i = 0; i < now.size(); i++)
	{
		// "env" is written below, as its differences (see the comment in settings.h).
		if (std::string(now[i].first) != "env" && now[i].second != was[i].second)
		{
			if (!JsonParser(now[i].second).Parse(emu.o[now[i].first]))
			{
				return false;
			}
		}
	}
	// The file's own env object, kept as it is unless the page changed a variable.
	JsonValue file_env;
	file_env.type = JsonValue::Type::Object;
	if (const auto* old = Section(root, "emu"))
	{
		if (auto env = old->o.find("env"); env != old->o.end() && env->second.type == JsonValue::Type::Object)
		{
			file_env = env->second;
		}
	}
	// STEP6:6e: only the differences go into the file: a variable the page added or changed gets its value, a
	// variable it removed gets an empty value (removed from the child's environment). Values inherited from
	// the global table or a build config are never baked in.
	bool env_changed = false;
	for (const auto& [name, value]: edited.env)
	{
		const auto it = baseline.env.find(name);
		if (it != baseline.env.end() && it->second == value)
		{
			continue;
		}
		if (value.empty())
		{
			file_env.o.erase(name);
		} else
		{
			JsonValue v;
			v.type           = JsonValue::Type::String;
			v.s              = value;
			file_env.o[name] = std::move(v);
		}
		env_changed = true;
	}
	for (const auto& entry: baseline.env)
	{
		if (edited.env.count(entry.first) != 0)
		{
			continue;
		}
		JsonValue v;
		v.type                  = JsonValue::Type::String; // empty: unset it in the child's environment
		file_env.o[entry.first] = std::move(v);
		env_changed             = true;
	}
	if (env_changed)
	{
		emu.o["env"] = std::move(file_env);
	} else if (const auto* old = Section(root, "emu"))
	{
		if (auto it = old->o.find("env"); it != old->o.end() && it->second.type == JsonValue::Type::Object)
		{
			emu.o["env"] = it->second;
		}
	}
	root.o["emu"] = std::move(emu);
	return WriteGameRoot(file, std::move(root));
}

bool reset_game_overrides(const Paths& paths, const std::string& title_id)
{
	JsonValue  root;
	const auto file = paths.game_config_file(title_id);
	if (title_id.empty() || !LoadGameRoot(file, root))
	{
		return false;
	}
	root.o.erase("emu");
	return WriteGameRoot(file, std::move(root));
}

std::string build_display_name(const Settings& s, const std::string& exe)
{
	auto same = [](const std::string& a, const std::string& b)
	{
		auto key = [](std::string v)
		{
			std::replace(v.begin(), v.end(), '\\', '/');
			std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return v;
		};
		return key(a) == key(b);
	};
	for (const auto& item: s.build_names)
	{
		const size_t bar = item.find('|');
		if (bar != std::string::npos && same(item.substr(0, bar), exe))
		{
			return item.substr(bar + 1);
		}
	}
	const std::string folder = path_utf8(utf8_path(exe).parent_path().filename());
	return folder.empty() ? exe : folder;
}

bool save_settings(const std::filesystem::path& path, const Settings& s)
{
	{
		// A file written by a newer launcher is left alone.
		JsonValue existing;
		if (ReadJsonFile(path, existing) && SchemaVersion(existing) > kConfigVersion)
		{
			return false;
		}
	}
	std::error_code ec;
	std::filesystem::create_directories(path.parent_path(), ec);

	std::string out = "{\n";
	out += "  \"version\": " + std::to_string(kConfigVersion) + ",\n";
	out += "  \"app\": {\n";
	out += "    \"emulator\": " + Quote(s.emulator) + ",\n";
	out += "    \"game\": " + Quote(s.game) + ",\n";
	out += "    \"run_dir\": " + Quote(s.run_dir) + ",\n";
	out += "    \"game_dirs\": " + QuoteList(SplitList(s.game_dirs)) + ",\n";
	out += "    \"addons_dir\": " + Quote(s.addons_dir) + ",\n";
	out += "    \"columns\": " + Quote(s.columns) + ",\n";
	// STEP6:6e app keys: begin
	out += "    \"window_x\": " + std::to_string(s.window_x) + ",\n";
	out += "    \"window_y\": " + std::to_string(s.window_y) + ",\n";
	out += "    \"window_w\": " + std::to_string(s.window_w) + ",\n";
	out += "    \"window_h\": " + std::to_string(s.window_h) + ",\n";
	// STEP6:6e app keys: end
	out += "    \"versions_dir\": " + Quote(s.versions_dir) + ",\n";
	out += "    \"custom_builds\": " + QuoteList(s.custom_builds) + ",\n";
	out += "    \"sources\": " + QuoteList(s.sources) + ",\n";
	out += "    \"hidden_builds\": " + QuoteList(s.hidden_builds) + ",\n";
	out += "    \"hidden_games\": " + QuoteList(s.hidden_games) + ",\n";
	// STEP7:3 app keys: begin
	out += "    \"check_updates\": " + std::string(s.check_updates ? "true" : "false") + ",\n";
	// STEP7:3 app keys: end
	out += "    \"source_names\": " + QuoteList(s.source_names) + ",\n";
	out += "    \"menu_hidden\": " + QuoteList(s.menu_hidden) + ",\n";
	out += "    \"build_names\": " + QuoteList(s.build_names) + ",\n";
	// STEP6:6d app keys: begin
	out += "    \"pinned_builds\": " + QuoteList(s.pinned_builds) + ",\n";
	out += "    \"vm_sort\": " + Quote(s.vm_sort) + ",\n";
	out += "    \"vm_sort_desc\": " + std::string(s.vm_sort_desc ? "true" : "false") + ",\n";
	// STEP6:6d app keys: end
	// STEP6:6c app keys: begin
	out += "    \"library_sort\": " + Quote(s.library_sort) + ",\n";
	out += "    \"library_sort_desc\": " + std::string(s.library_sort_desc ? "true" : "false") + ",\n";
	out += "    \"library_grid\": " + std::string(s.library_grid ? "true" : "false") + ",\n";
	out += "    \"favourite_games\": " + QuoteList(s.favourite_games) + "\n";
	// STEP6:6c app keys: end
	out += "  },\n";
	out += "  \"emu\": {";
	const EmuEntries defaults = EmuKeys(default_settings());
	const EmuEntries values   = EmuKeys(s);
	bool             first    = true;
	for (size_t i = 0; i < values.size(); i++)
	{
		if (UserSet(s, values, defaults, i))
		{
			out += first ? "\n" : ",\n";
			out += std::string("    \"") + values[i].first + "\": " + values[i].second;
			first = false;
		}
	}
	out += first ? "}\n}\n" : "\n  }\n}\n";

	std::filesystem::path tmp = path;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		if (!f || !(f << out))
		{
			return false;
		}
	}
	std::filesystem::rename(tmp, path, ec);
	return !ec;
}

} // namespace kyty_launcher
