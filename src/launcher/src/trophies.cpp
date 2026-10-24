#include "trophies.h"

#include "library.h"
#include "settings.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string_view>

namespace kyty_launcher {

namespace fs = std::filesystem;

namespace {

// Unlock times are 1 µs ticks since 0001-01-01; this is 1970-01-01.
constexpr uint64_t kUnixEpochTick = 62135596800000000ULL;

uint64_t ReadBigEndian(const std::string& data, size_t at, size_t size)
{
	uint64_t value = 0;
	for (size_t i = 0; i < size; i++)
	{
		value = (value << 8u) | static_cast<uint8_t>(data[at + i]);
	}
	return value;
}

// The files inside a .ucp package (the same container the emulator reads), as name -> bytes.
std::map<std::string, std::string> ReadPackage(const std::string& data)
{
	std::map<std::string, std::string> files;
	if (data.size() < 0x40 || ReadBigEndian(data, 0, 4) != 0xb228c60a || ReadBigEndian(data, 4, 4) != 1)
	{
		return files;
	}
	const uint64_t size  = ReadBigEndian(data, 8, 8);
	const uint64_t count = ReadBigEndian(data, 0x10, 4);
	const uint64_t toc   = ReadBigEndian(data, 0x14, 4);
	if (size < 0x40 || size > data.size() || count > 4096 || toc > size || 0x20 + count * 0x40 > size - toc)
	{
		return files;
	}
	for (uint64_t i = 0; i < count; i++)
	{
		const size_t      entry = static_cast<size_t>(toc + 0x20 + i * 0x40);
		const char*       name  = data.data() + entry;
		const std::string filename(name, std::find(name, name + 0x20, '\0') - name);
		const uint64_t    offset = ReadBigEndian(data, entry + 0x20, 8);
		const uint64_t    length = ReadBigEndian(data, entry + 0x28, 8);
		if (offset > size || length > size - offset)
		{
			return {};
		}
		files.emplace(filename, data.substr(static_cast<size_t>(offset), static_cast<size_t>(length)));
	}
	return files;
}

// A trophy id written as a number or as a string of digits; -1 when neither.
int ReadId(const JsonValue& v)
{
	if (v.type == JsonValue::Type::Number)
	{
		return v.n >= 0 && v.n <= 2147483647.0 ? static_cast<int>(v.n) : -1;
	}
	if (v.type == JsonValue::Type::String && !v.s.empty() && v.s.size() < 10 &&
	    std::all_of(v.s.begin(), v.s.end(), [](char c) { return c >= '0' && c <= '9'; }))
	{
		return std::stoi(v.s);
	}
	return -1;
}

const JsonValue* Field(const JsonValue& obj, const char* key, JsonValue::Type type)
{
	if (obj.type != JsonValue::Type::Object)
	{
		return nullptr;
	}
	auto it = obj.o.find(key);
	return it != obj.o.end() && it->second.type == type ? &it->second : nullptr;
}

std::string ReadBinary(const fs::path& path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	const auto size = static_cast<size_t>(in.tellg());
	if (size > (size_t {1} << 30u))
	{
		return {};
	}
	in.seekg(0);
	std::string data(size, '\0');
	in.read(data.data(), static_cast<std::streamsize>(size));
	return data;
}

// The sce_sys/trophy2/trophyNN.ucp packages of the game, by package label.
std::map<int, fs::path> PackagePaths(const std::string& game_dir)
{
	std::error_code ec;
	const fs::path  dir = utf8_path(game_dir) / "sce_sys" / "trophy2";
	std::map<int, fs::path> packages;
	if (game_dir.empty() || !fs::is_directory(dir, ec))
	{
		return packages;
	}
	for (const auto& entry: fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec))
	{
		const std::string file = path_utf8(entry.path().filename());
		if (file.size() < 11 || file.rfind("trophy", 0) != 0 || file.substr(file.size() - 4) != ".ucp")
		{
			continue;
		}
		const std::string digits = file.substr(6, file.size() - 10);
		if (digits.size() <= 6 && std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
		{
			packages.emplace(std::stoi(digits), entry.path());
		}
	}
	return packages;
}

// Calls fn(label, package files) for each sce_sys/trophy2/trophyNN.ucp of the game, by label.
bool ForEachPackage(const std::string& game_dir, const std::function<void(int, const std::map<std::string, std::string>&)>& fn)
{
	bool found = false;
	for (const auto& [label, path]: PackagePaths(game_dir))
	{
		auto files = ReadPackage(ReadBinary(path));
		if (files.count("tropconf.json") != 0)
		{
			fn(label, files);
			found = true;
		}
	}
	return found;
}

struct Unlocks
{
	std::set<int>           ids;
	std::map<int, uint64_t> ticks;
};

// The emulator's unlock file of one package: <run_dir>\_Trophies\<title>\trophies_<user>_<label>.json.
Unlocks ReadUnlocks(const std::string& run_dir, const std::string& title_id, int user_id, int label)
{
	Unlocks u;
	if (title_id.empty())
	{
		return u;
	}
	const fs::path path = utf8_path(run_dir) / "_Trophies" / utf8_path(title_id) /
	                      ("trophies_" + std::to_string(user_id) + "_" + std::to_string(label) + ".json");
	JsonValue json;
	if (!parse_json(read_text_file(path_utf8(path)), json))
	{
		return u;
	}
	if (const auto* list = Field(json, "unlockedTrophies", JsonValue::Type::Array))
	{
		for (const auto& v: list->a)
		{
			if (int id = ReadId(v); id >= 0)
			{
				u.ids.insert(id);
			}
		}
	}
	if (const auto* at = Field(json, "unlockedAt", JsonValue::Type::Object))
	{
		for (const auto& [key, v]: at->o)
		{
			JsonValue k;
			k.type = JsonValue::Type::String;
			k.s    = key;
			if (int id = ReadId(k); id >= 0 && v.type == JsonValue::Type::Number && v.n >= static_cast<double>(kUnixEpochTick))
			{
				u.ticks[id] = static_cast<uint64_t>(v.n);
			}
		}
	}
	return u;
}

std::string FormatTick(uint64_t tick)
{
	const auto seconds = static_cast<std::time_t>((tick - kUnixEpochTick) / 1000000);
	std::tm    tm {};
#ifdef _WIN32
	localtime_s(&tm, &seconds);
#else
	localtime_r(&seconds, &tm);
#endif
	char buf[32];
	std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
	return buf;
}

// Cache identity: a game directory can be reached through different spellings.
std::string TrophyKey(const std::string& game_dir, const std::string& title_id, const std::string& run_dir, int user_id)
{
	return path_key(game_dir) + "|" + title_id + "|" + path_key(run_dir) + "|" + std::to_string(user_id);
}

// Latest write time and total size (plus a count) of the game's trophy packages and of its unlock files.
// Empty when the game has no trophy package; any change makes the stamp differ.
std::string TrophyStamp(const std::string& game_dir, const std::string& title_id, const std::string& run_dir, int user_id)
{
	const std::map<int, fs::path> packages = PackagePaths(game_dir);
	if (packages.empty())
	{
		return {};
	}
	uint64_t        latest = 0;
	uint64_t        total  = 0;
	int             count  = 0;
	const auto      add    = [&](const fs::path& file)
	{
		std::error_code size_ec;
		const auto      size = fs::file_size(file, size_ec);
		if (size_ec)
		{
			return;
		}
		std::error_code time_ec;
		const auto      time = fs::last_write_time(file, time_ec);
		if (!time_ec)
		{
			latest = std::max(latest, static_cast<uint64_t>(time.time_since_epoch().count()));
		}
		total += static_cast<uint64_t>(size);
		count++;
	};
	for (const auto& [label, path]: packages)
	{
		add(path);
		if (!title_id.empty())
		{
			add(utf8_path(run_dir) / "_Trophies" / utf8_path(title_id) /
			    ("trophies_" + std::to_string(user_id) + "_" + std::to_string(label) + ".json"));
		}
	}
	return std::to_string(latest) + ":" + std::to_string(total) + ":" + std::to_string(count);
}

// "P", "G", "S", "B" -> 1..4; 0 otherwise.
int Grade(const JsonValue& t)
{
	const auto* g = Field(t, "grade", JsonValue::Type::String);
	if (g == nullptr || g->s.size() != 1)
	{
		return 0;
	}
	const size_t i = std::string_view("PGSB").find(g->s[0]);
	return i == std::string_view::npos ? 0 : static_cast<int>(i) + 1;
}

} // namespace

int TrophyProgress::Percent() const
{
	const int points        = total_grade[2] * 6 + total_grade[3] * 2 + total_grade[4];
	const int earned_points = earned_grade[2] * 6 + earned_grade[3] * 2 + earned_grade[4];
	return points == 0 ? 0 : earned_points * 100 / points;
}

bool load_trophy_progress(const std::string& game_dir, const std::string& title_id, const std::string& run_dir, int user_id,
                          TrophyProgress& out)
{
	out = {};
	return ForEachPackage(game_dir,
	                      [&](int label, const std::map<std::string, std::string>& files)
	                      {
		                      JsonValue conf;
		                      if (!parse_json(files.at("tropconf.json"), conf))
		                      {
			                      return;
		                      }
		                      const auto* list = Field(conf, "trophies", JsonValue::Type::Array);
		                      if (list == nullptr)
		                      {
			                      return;
		                      }
		                      const Unlocks unlocks = ReadUnlocks(run_dir, title_id, user_id, label);
		                      for (const auto& t: list->a)
		                      {
			                      const int grade = Grade(t);
			                      const auto* id  = t.type == JsonValue::Type::Object && t.o.count("id") != 0 ? &t.o.at("id") : nullptr;
			                      if (grade == 0 || id == nullptr || ReadId(*id) < 0)
			                      {
				                      continue;
			                      }
			                      out.total++;
			                      out.total_grade[grade]++;
			                      if (unlocks.ids.count(ReadId(*id)) != 0)
			                      {
				                      out.earned++;
				                      out.earned_grade[grade]++;
			                      }
		                      }
	                      });
}

bool load_trophy_list(const std::string& game_dir, const std::string& title_id, const std::string& run_dir, int user_id,
                      int console_language, const std::string& icon_cache_dir, std::vector<TrophyInfo>& out)
{
	static constexpr std::array<const char*, 30> kLocales = {
	    "ja-JP", "en-US", "fr-FR", "es-ES", "de-DE", "it-IT", "nl-NL", "pt-PT", "ru-RU",  "ko-KR",
	    "zh-Hant", "zh-Hans", "fi-FI", "sv-SE", "da-DK", "no-NO", "pl-PL", "pt-BR", "en-GB", "tr-TR",
	    "es-419", "ar-AE", "fr-CA", "cs-CZ", "hu-HU", "el-GR", "ro-RO", "th-TH", "vi-VN",  "id-ID"};
	out.clear();
	return ForEachPackage(
	    game_dir,
	    [&](int label, const std::map<std::string, std::string>& files)
	    {
		    JsonValue conf;
		    if (!parse_json(files.at("tropconf.json"), conf))
		    {
			    return;
		    }
		    const auto* list = Field(conf, "trophies", JsonValue::Type::Array);
		    if (list == nullptr)
		    {
			    return;
		    }

		    // Texts: the console language, else the package's default language.
		    struct Text
		    {
			    std::string name;
			    std::string detail;
			    std::string reward;
		    };
		    std::map<int, Text> texts;
		    {
			    const char* locale = kLocales[console_language >= 0 && console_language < 30 ? console_language : 1];
			    auto        meta   = files.find(std::string("tropmeta_") + locale + ".json");
			    if (meta == files.end())
			    {
				    if (const auto* def = Field(conf, "defaultLanguage", JsonValue::Type::String))
				    {
					    meta = files.find("tropmeta_" + def->s + ".json");
				    }
			    }
			    JsonValue json;
			    if (meta != files.end() && parse_json(meta->second, json))
			    {
				    const auto* md = Field(json, "metadata", JsonValue::Type::Object);
				    const auto* tm = md != nullptr ? Field(*md, "trophyMetadata", JsonValue::Type::Array) : nullptr;
				    for (const auto& t: tm != nullptr ? tm->a : std::vector<JsonValue> {})
				    {
					    const auto* id = t.type == JsonValue::Type::Object && t.o.count("id") != 0 ? &t.o.at("id") : nullptr;
					    if (id == nullptr || ReadId(*id) < 0)
					    {
						    continue;
					    }
					    const auto* name   = Field(t, "name", JsonValue::Type::String);
					    const auto* detail = Field(t, "detail", JsonValue::Type::String);
					    const auto* reward = Field(t, "reward", JsonValue::Type::String);
					    texts[ReadId(*id)] = {name != nullptr ? name->s : "", detail != nullptr ? detail->s : "",
					                          reward != nullptr ? reward->s : ""};
				    }
			    }
		    }

		    const Unlocks   unlocks = ReadUnlocks(run_dir, title_id, user_id, label);
		    const fs::path  icons   = utf8_path(icon_cache_dir) / std::to_string(label);
		    std::error_code ec;
		    fs::create_directories(icons, ec);

		    std::vector<std::pair<int, TrophyInfo>> trophies;
		    for (const auto& t: list->a)
		    {
			    const int  grade = Grade(t);
			    const auto* idv  = t.type == JsonValue::Type::Object && t.o.count("id") != 0 ? &t.o.at("id") : nullptr;
			    const int  id    = idv != nullptr ? ReadId(*idv) : -1;
			    if (grade == 0 || id < 0)
			    {
				    continue;
			    }
			    TrophyInfo info;
			    info.grade = grade;
			    if (auto text = texts.find(id); text != texts.end())
			    {
				    info.name   = text->second.name;
				    info.detail = text->second.detail;
				    info.reward = text->second.reward;
			    }
			    // Stable fallback: the package trophy id, never the row index (the list may be reordered).
			    if (info.name.empty())
			    {
				    info.name = "Trophy " + std::to_string(id);
			    }
			    const auto* hidden = Field(t, "hidden", JsonValue::Type::Bool);
			    info.hidden        = hidden != nullptr && hidden->b;
			    // The .ucp trophy record has a hasReward flag and a free-text reward in the metadata (there is no
			    // points value); show the text only when the flag is set, like the emulator's trophy viewer.
			    const auto* has_reward = Field(t, "hasReward", JsonValue::Type::Bool);
			    if (has_reward == nullptr || !has_reward->b)
			    {
				    info.reward.clear();
			    }
			    info.earned        = unlocks.ids.count(id) != 0;
			    if (auto tick = unlocks.ticks.find(id); info.earned && tick != unlocks.ticks.end())
			    {
				    info.earned_at = FormatTick(tick->second);
			    }
			    char png[16];
			    std::snprintf(png, sizeof(png), "trop%04d.png", id);
			    if (auto icon = files.find(png); icon != files.end())
			    {
				    // Written once per game and package; the slint image loader reads files.
				    const fs::path file = icons / png;
				    if (!fs::exists(file, ec) || fs::file_size(file, ec) != icon->second.size())
				    {
					    std::ofstream(file, std::ios::binary | std::ios::trunc).write(icon->second.data(),
					                                                                  static_cast<std::streamsize>(icon->second.size()));
				    }
				    info.icon = path_utf8(file);
			    }
			    trophies.emplace_back(id, std::move(info));
		    }
		    std::sort(trophies.begin(), trophies.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
		    for (auto& [id, info]: trophies)
		    {
			    out.push_back(std::move(info));
		    }
	    });
}

std::vector<TrophyScanEntry> scan_trophy_progress(const std::vector<GameEntry>& games, const std::string& run_dir, int user_id,
                                                  TrophyCache& cache, const std::atomic<bool>& cancel)
{
	std::vector<TrophyScanEntry> out;
	for (std::size_t i = 0; i < games.size(); i++)
	{
		if (cancel.load())
		{
			return {};
		}
		const GameEntry&  game  = games[i];
		const std::string stamp = TrophyStamp(game.dir, game.title_id, run_dir, user_id);
		if (stamp.empty())
		{
			continue;
		}
		const std::string  key = TrophyKey(game.dir, game.title_id, run_dir, user_id);
		TrophyCache::Entry entry;
		bool               hit = false;
		{
			std::lock_guard<std::mutex> lock(cache.mutex);
			if (const auto it = cache.entries.find(key); it != cache.entries.end() && it->second.stamp == stamp)
			{
				entry = it->second;
				hit   = true;
			}
		}
		if (!hit)
		{
			TrophyProgress progress;
			const bool     has_trophies = load_trophy_progress(game.dir, game.title_id, run_dir, user_id, progress);
			if (cancel.load())
			{
				return {};
			}
			entry.stamp        = stamp;
			entry.progress     = progress;
			entry.has_trophies = has_trophies;
			std::lock_guard<std::mutex> lock(cache.mutex);
			cache.entries[key] = entry;
		}
		if (entry.has_trophies)
		{
			out.push_back(TrophyScanEntry {i, entry.progress});
		}
	}
	return out;
}

std::vector<TrophyScanEntry> cached_trophy_progress(const std::vector<GameEntry>& games, const std::string& run_dir, int user_id,
                                                    TrophyCache& cache)
{
	std::vector<TrophyScanEntry> out;
	std::lock_guard<std::mutex>  lock(cache.mutex);
	for (std::size_t i = 0; i < games.size(); i++)
	{
		const auto it = cache.entries.find(TrophyKey(games[i].dir, games[i].title_id, run_dir, user_id));
		if (it != cache.entries.end() && it->second.has_trophies)
		{
			out.push_back(TrophyScanEntry {i, it->second.progress});
		}
	}
	return out;
}

void invalidate_trophy_cache(TrophyCache& cache, const std::string& game_dir, const std::string& title_id, const std::string& run_dir,
                             int user_id)
{
	std::lock_guard<std::mutex> lock(cache.mutex);
	cache.entries.erase(TrophyKey(game_dir, title_id, run_dir, user_id));
}

} // namespace kyty_launcher
