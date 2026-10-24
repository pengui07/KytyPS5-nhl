#include "versions.h"

#include "library.h"
#include "net.h"
#include "settings.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kyty_launcher {

namespace fs = std::filesystem;

namespace {

constexpr const char* kTagPrefix   = "KytyPS5-";
// Folders next to a build while it is replaced: the new one being unpacked, the previous one set aside.
constexpr const char* kStagingSuffix = ".part";
constexpr const char* kOldSuffix     = ".old";

// What a release of this platform looks like: the emulator inside it and the archive it ships as
// (KytyPS5-<date>-<commit>-Windows-x64.zip, -macOS-x86_64.zip, -Linux-x86_64.tar.gz).
#ifdef _WIN32
constexpr const char* kExeName    = "kyty_emulator.exe";
constexpr const char* kArchiveExt = ".zip";
constexpr const char* kThisOs[]   = {"windows", "win64", "win-x64"};
constexpr const char* kOtherOs[]  = {"linux", "mac"};
#elif defined(__APPLE__)
constexpr const char* kExeName    = "kyty_emulator";
constexpr const char* kArchiveExt = ".zip";
constexpr const char* kThisOs[]   = {"macos", "osx", "mac"};
constexpr const char* kOtherOs[]  = {"windows", "win64", "win-x64", "linux"};
#else
constexpr const char* kExeName    = "kyty_emulator";
constexpr const char* kArchiveExt = ".tar.gz";
constexpr const char* kThisOs[]   = {"linux"};
constexpr const char* kOtherOs[]  = {"windows", "win64", "win-x64", "mac"};
#endif

bool EndsWith(const std::string& s, const std::string& suffix)
{
	return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool ContainsAny(const std::string& s, const char* const* words, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		if (s.find(words[i]) != std::string::npos)
		{
			return true;
		}
	}
	return false;
}

// KytyPS5-2026-10-19-e5f0ee7 -> date 2026-10-19, codename e5f0ee7.
void SplitTag(const std::string& tag, std::string& date, std::string& codename)
{
	std::string rest = tag.rfind(kTagPrefix, 0) == 0 ? tag.substr(std::char_traits<char>::length(kTagPrefix)) : tag;
	if (rest.size() > 11 && rest[4] == '-' && rest[7] == '-' && rest[10] == '-')
	{
		date     = rest.substr(0, 10);
		codename = rest.substr(11);
	} else
	{
		date.clear();
		codename = rest;
	}
}

std::string FileDate(const fs::path& p)
{
	std::error_code ec;
	auto            t = fs::last_write_time(p, ec);
	if (ec)
	{
		return {};
	}
#ifdef _WIN32
	auto       sys  = std::chrono::clock_cast<std::chrono::system_clock>(t);
#else
	// libc++ has no clock_cast yet: shift the file time by the difference of the two clocks.
	auto       sys  = std::chrono::system_clock::now() + std::chrono::duration_cast<std::chrono::system_clock::duration>(t - fs::file_time_type::clock::now());
#endif
	std::time_t tt  = std::chrono::system_clock::to_time_t(sys);
	std::tm     tm {};
#ifdef _WIN32
	localtime_s(&tm, &tt);
#else
	localtime_r(&tt, &tm);
#endif
	char buf[16];
	std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
	return buf;
}

// The emulator at most three folders down (zips may wrap everything in a folder). The shallowest
// one wins: the macOS zip holds one at its top and another inside KytyPS5.app.
std::string FindExe(const fs::path& dir)
{
	std::error_code ec;
	std::string     best;
	int             best_depth = 1000;
	for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; it != end && !ec;
	     it.increment(ec))
	{
		if (it.depth() > 3)
		{
			it.disable_recursion_pending();
			continue;
		}
		if (it.depth() < best_depth && it->is_regular_file(ec) && it->path().filename() == kExeName)
		{
			best       = path_utf8(it->path());
			best_depth = it.depth();
		}
	}
	return best;
}

#ifdef _WIN32
// Windows 10+ ships bsdtar as tar.exe, which also unpacks zips.
bool Unzip(const fs::path& zip, const fs::path& dir, std::string* error)
{
	wchar_t sys[MAX_PATH];
	GetSystemDirectoryW(sys, MAX_PATH);
	std::wstring tar = std::wstring(sys) + L"\\tar.exe";
	std::wstring cmd = L"\"" + tar + L"\" -xf \"" + zip.wstring() + L"\" -C \"" + dir.wstring() + L"\"";

	STARTUPINFOW        si {sizeof(si)};
	PROCESS_INFORMATION pi {};
	if (CreateProcessW(tar.c_str(), cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi) == FALSE)
	{
		*error = "Could not start tar.exe";
		return false;
	}
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	if (code != 0)
	{
		*error = "Unpacking failed (tar exit code " + std::to_string(code) + ")";
		return false;
	}
	return true;
}
#else
// Runs a program without a shell; its exit code (or -1).
int RunProgram(const std::vector<std::string>& args)
{
	std::vector<char*> argv;
	std::vector<std::string> copy = args;
	for (auto& a: copy)
	{
		argv.push_back(a.data());
	}
	argv.push_back(nullptr);
	pid_t pid = fork();
	if (pid < 0)
	{
		return -1;
	}
	if (pid == 0)
	{
		execvp(argv[0], argv.data());
		_exit(127);
	}
	int status = 0;
	if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status))
	{
		return -1;
	}
	return WEXITSTATUS(status);
}

// macOS ships unzip; Linux releases are .tar.gz and tar is always there.
bool Unzip(const fs::path& archive, const fs::path& dir, std::string* error)
{
#ifdef __APPLE__
	const std::vector<std::string> cmd = {"unzip", "-q", "-o", path_utf8(archive), "-d", path_utf8(dir)};
#else
	const std::vector<std::string> cmd = {"tar", "-xzf", path_utf8(archive), "-C", path_utf8(dir)};
#endif
	const int code = RunProgram(cmd);
	if (code != 0)
	{
		*error = "Unpacking failed (" + cmd[0] + " exit code " + std::to_string(code) + ")";
		return false;
	}
	return true;
}
#endif

} // namespace

std::string parse_repo_link(const std::string& link)
{
	std::string s = link;
	// Trim spaces, then drop the scheme and host.
	while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0)
	{
		s.pop_back();
	}
	size_t b = 0;
	while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0)
	{
		b++;
	}
	s = s.substr(b);
	for (const char* prefix: {"https://", "http://"})
	{
		if (s.rfind(prefix, 0) == 0)
		{
			s = s.substr(std::char_traits<char>::length(prefix));
		}
	}
	for (const char* host: {"www.github.com/", "github.com/"})
	{
		if (s.rfind(host, 0) == 0)
		{
			s = s.substr(std::char_traits<char>::length(host));
		}
	}
	// owner/repo, then anything after (releases, tag, query...).
	size_t slash1 = s.find('/');
	if (slash1 == std::string::npos || slash1 == 0)
	{
		return {};
	}
	size_t end = s.find_first_of("/?#", slash1 + 1);
	std::string repo = s.substr(slash1 + 1, end == std::string::npos ? std::string::npos : end - slash1 - 1);
	if (repo.size() > 4 && repo.compare(repo.size() - 4, 4, ".git") == 0)
	{
		repo.resize(repo.size() - 4);
	}
	std::string owner = s.substr(0, slash1);
	auto        ok    = [](const std::string& part)
	{
		return !part.empty() && part.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") == std::string::npos;
	};
	return ok(owner) && ok(repo) ? owner + "/" + repo : std::string();
}

// The release list straight from the API. `validator` applies to page 1 only: when GitHub says it is
// unchanged (validator->not_modified) nothing is returned and the caller keeps its copy.
static std::vector<ReleaseInfo> FetchFromGitHub(const std::string& repo, HttpValidator* validator, std::string* error,
                                                const std::atomic<bool>* cancel)
{
	std::vector<ReleaseInfo> out;
	// GitHub returns at most 100 releases a page; walk the pages until a short one (bounded, since each page
	// costs one of the 60 unauthenticated requests an hour).
	constexpr int kPerPage  = 100;
	constexpr int kMaxPages = 10;
	for (int page = 1; page <= kMaxPages; page++)
	{
		if (cancel != nullptr && cancel->load())
		{
			// A partial list would pass for the whole one.
			out.clear();
			*error = "Cancelled";
			return out;
		}
		std::string body;
		std::string page_error;
		if (!http_get("https://api.github.com/repos/" + repo + "/releases?per_page=" + std::to_string(kPerPage) + "&page=" + std::to_string(page),
		              [&](const char* p, size_t n) { body.append(p, n); return true; }, nullptr, &page_error, std::string(), nullptr,
		              page == 1 ? validator : nullptr))
		{
			// A later page failing still leaves the newer releases usable.
			if (page == 1)
			{
				// GitHub answers 403 (or 429) once the 60 unauthenticated requests an hour are used up.
				const bool limited = page_error.rfind("HTTP 403", 0) == 0 || page_error.rfind("HTTP 429", 0) == 0;
				const bool not_found = page_error.rfind("HTTP 404", 0) == 0;
				*error = limited     ? "GitHub's hourly API limit was reached. Try again later, or set GITHUB_TOKEN for a higher limit."
				         : not_found ? (github_token_set() ? "Repository not found, or the GitHub token has no access to it."
				                                           : "Repository not found. If it is private, set a GitHub token in Settings > Versions.")
				                     : page_error;
			}
			break;
		}
		if (page == 1 && validator != nullptr && validator->not_modified)
		{
			return out;
		}
		JsonValue root;
		if (!parse_json(body, root) || root.type != JsonValue::Type::Array)
		{
			if (page == 1)
			{
				*error = "Unexpected reply from GitHub";
			}
			break;
		}
		for (const auto& rel: root.a)
		{
			auto tag = rel.o.find("tag_name");
			auto assets = rel.o.find("assets");
			if (tag == rel.o.end() || assets == rel.o.end() || assets->second.type != JsonValue::Type::Array)
			{
				continue;
			}
			// This platform's archive; failing that, the only archive of that kind that names no other OS.
			const JsonValue* pick = nullptr;
			for (const auto& a: assets->second.a)
			{
				auto name = a.o.find("name");
				auto url  = a.o.find("browser_download_url");
				if (name == a.o.end() || url == a.o.end())
				{
					continue;
				}
				std::string lower = name->second.s;
				for (auto& ch: lower)
				{
					ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
				}
				if (!EndsWith(lower, kArchiveExt))
				{
					continue;
				}
	#if defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
				// The releases are built for x86-64 only.
				if (lower.find("x86_64") != std::string::npos || lower.find("x64") != std::string::npos)
				{
					continue;
				}
	#endif
				if (ContainsAny(lower, kThisOs, std::size(kThisOs)))
				{
					pick = &a;
					break;
				}
				if (pick == nullptr && !ContainsAny(lower, kOtherOs, std::size(kOtherOs)))
				{
					pick = &a;
				}
			}
			if (pick != nullptr)
			{
				ReleaseInfo r;
				r.tag     = tag->second.s;
				r.zip_url = pick->o.find("browser_download_url")->second.s;
				if (auto api = pick->o.find("url"); api != pick->o.end())
				{
					r.api_url = api->second.s;
				}
				if (auto size = pick->o.find("size"); size != pick->o.end())
				{
					r.zip_size = static_cast<uint64_t>(size->second.n);
				}
				SplitTag(r.tag, r.date, r.codename);
				if (r.date.empty())
				{
					if (auto pub = rel.o.find("published_at"); pub != rel.o.end() && pub->second.s.size() >= 10)
					{
						r.date = pub->second.s.substr(0, 10);
					}
				}
				r.folder = repo == kDefaultRepo ? r.tag : repo.substr(repo.find('/') + 1) + "-" + r.tag;
				out.push_back(r);
			}
		}
		if (root.a.size() < kPerPage)
		{
			break;
		}
	}
	// The API orders by creation; sort by tag date so the newest build is first.
	std::stable_sort(out.begin(), out.end(), [](const ReleaseInfo& a, const ReleaseInfo& b) { return a.date > b.date; });
	return out;
}

namespace {

// A release list saved on disk: one file per repo, "etag\t<etag>", "fetched\t<unix seconds>", then one
// tab-separated release per line (tags and URLs hold no tabs or newlines).
struct CachedReleases
{
	std::string              etag;
	int64_t                  fetched = 0;
	std::vector<ReleaseInfo> list;
};

// Within this many seconds of the last check the saved list is used without asking GitHub at all
// (unauthenticated 304 answers still count against the hourly limit).
constexpr int64_t kCacheFreshSeconds = 10 * 60;

int64_t NowSeconds()
{
	return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

fs::path CacheFile(const std::string& cache_dir, const std::string& repo)
{
	std::string name = repo;
	std::replace(name.begin(), name.end(), '/', '-');
	return utf8_path(cache_dir) / ("releases-" + name + ".tsv");
}

std::vector<std::string> SplitTabs(const std::string& line)
{
	std::vector<std::string> out;
	size_t                   start = 0;
	for (;;)
	{
		size_t tab = line.find('\t', start);
		out.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
		if (tab == std::string::npos)
		{
			return out;
		}
		start = tab + 1;
	}
}

bool LoadCache(const fs::path& file, CachedReleases& c)
{
	std::ifstream f(file, std::ios::binary);
	std::string   line;
	if (!f || !std::getline(f, line) || line.rfind("etag\t", 0) != 0)
	{
		return false;
	}
	c.etag = line.substr(5);
	if (!std::getline(f, line) || line.rfind("fetched\t", 0) != 0)
	{
		return false;
	}
	c.fetched = std::strtoll(line.c_str() + 8, nullptr, 10);
	while (std::getline(f, line))
	{
		auto parts = SplitTabs(line);
		if (parts.size() != 7)
		{
			return false;
		}
		ReleaseInfo r;
		r.tag      = parts[0];
		r.folder   = parts[1];
		r.date     = parts[2];
		r.codename = parts[3];
		r.zip_url  = parts[4];
		r.api_url  = parts[5];
		r.zip_size = std::strtoull(parts[6].c_str(), nullptr, 10);
		c.list.push_back(std::move(r));
	}
	return !c.list.empty();
}

// Written to a temporary file first so a crash never leaves half a list.
void SaveCache(const fs::path& file, const CachedReleases& c)
{
	std::error_code ec;
	fs::create_directories(file.parent_path(), ec);
	fs::path tmp = file;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		f << "etag\t" << c.etag << "\n" << "fetched\t" << c.fetched << "\n";
		for (const auto& r: c.list)
		{
			f << r.tag << '\t' << r.folder << '\t' << r.date << '\t' << r.codename << '\t' << r.zip_url << '\t' << r.api_url << '\t' << r.zip_size << "\n";
		}
		if (!f)
		{
			f.close();
			fs::remove(tmp, ec);
			return;
		}
	}
	fs::rename(tmp, file, ec);
}

std::string Age(int64_t seconds)
{
	if (seconds < 60 * 60)
	{
		return std::to_string(std::max<int64_t>(seconds / 60, 1)) + " min ago";
	}
	if (seconds < 48 * 60 * 60)
	{
		return std::to_string(seconds / (60 * 60)) + " h ago";
	}
	return std::to_string(seconds / (24 * 60 * 60)) + " days ago";
}

} // namespace

std::vector<ReleaseInfo> fetch_releases(const std::string& repo, const std::string& cache_dir, bool force, std::string* error, std::string* note,
                                        const std::atomic<bool>* cancel)
{
	const fs::path file = cache_dir.empty() ? fs::path() : CacheFile(cache_dir, repo);
	CachedReleases cached;
	const bool     have = !file.empty() && LoadCache(file, cached);
	const int64_t  now  = NowSeconds();
	if (have && !force && now >= cached.fetched && now - cached.fetched < kCacheFreshSeconds)
	{
		return std::move(cached.list);
	}

	HttpValidator validator;
	validator.etag = have ? cached.etag : std::string();
	std::string fetch_error;
	auto        list = FetchFromGitHub(repo, &validator, &fetch_error, cancel);
	if (have && validator.not_modified)
	{
		cached.fetched = now;
		SaveCache(file, cached);
		return std::move(cached.list);
	}
	if (!list.empty())
	{
		if (!file.empty())
		{
			SaveCache(file, CachedReleases {validator.etag, now, list});
		}
		return list;
	}
	if (have && fetch_error != "Cancelled")
	{
		// Offline or out of requests: the saved list is still worth showing.
		if (note != nullptr)
		{
			*note = "list saved " + Age(now - cached.fetched) + (fetch_error.empty() ? std::string() : " (" + fetch_error + ")");
		}
		return std::move(cached.list);
	}
	*error = fetch_error;
	return {};
}

// Visits the cached release lists (only files fetch_releases writes, so nothing else in the folder is touched).
template<class F>
static void ForEachCacheFile(const std::string& cache_dir, F fn)
{
	std::error_code ec;
	if (cache_dir.empty() || !fs::is_directory(utf8_path(cache_dir), ec))
	{
		return;
	}
	for (const auto& entry: fs::directory_iterator(utf8_path(cache_dir), fs::directory_options::skip_permission_denied, ec))
	{
		const std::string name = path_utf8(entry.path().filename());
		if (entry.is_regular_file(ec) && name.rfind("releases-", 0) == 0 && (EndsWith(name, ".tsv") || EndsWith(name, ".tsv.tmp")))
		{
			fn(entry);
		}
	}
}

void release_cache_size(const std::string& cache_dir, int* files, uint64_t* bytes)
{
	*files = 0;
	*bytes = 0;
	ForEachCacheFile(cache_dir,
	                 [&](const fs::directory_entry& e)
	                 {
		                 std::error_code ec;
		                 (*files)++;
		                 *bytes += e.file_size(ec);
	                 });
}

int clear_release_cache(const std::string& cache_dir)
{
	std::vector<fs::path> victims;
	ForEachCacheFile(cache_dir, [&](const fs::directory_entry& e) { victims.push_back(e.path()); });
	int deleted = 0;
	for (const auto& p: victims)
	{
		std::error_code ec;
		deleted += fs::remove(p, ec) ? 1 : 0;
	}
	return deleted;
}

std::vector<InstalledBuild> scan_installed(const std::string& versions_dir, const std::vector<std::string>& extra_exes,
                                           const std::vector<std::string>& hidden_exes)
{
	std::vector<InstalledBuild> out;
	std::error_code             ec;
	fs::path                    root = utf8_path(versions_dir);
	if (!versions_dir.empty() && fs::is_directory(root, ec))
	{
		for (const auto& entry: fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec))
		{
			if (!entry.is_directory(ec))
			{
				continue;
			}
			const std::string folder = path_utf8(entry.path().filename());
			if (EndsWith(folder, kStagingSuffix) || EndsWith(folder, kOldSuffix))
			{
				continue;
			}
			std::string exe = FindExe(entry.path());
			if (exe.empty())
			{
				continue;
			}
			if (std::any_of(hidden_exes.begin(), hidden_exes.end(), [&](const std::string& h) { return path_key(h) == path_key(exe); }))
			{
				continue;
			}
			InstalledBuild b;
			b.name    = path_utf8(entry.path().filename());
			b.exe     = exe;
			b.managed = true;
			// release.txt names the tag this folder holds (a "-latest" folder's name does not).
			std::string tag = read_text_file(path_utf8(entry.path() / "release.txt"));
			tag             = tag.substr(0, tag.find_first_of("\r\n"));
			b.tag           = tag.empty() ? b.name : tag;
			SplitTag(b.tag, b.date, b.codename);
			if (b.date.empty())
			{
				b.date = FileDate(utf8_path(exe));
			}
			out.push_back(b);
		}
		std::sort(out.begin(), out.end(), [](const InstalledBuild& a, const InstalledBuild& b) { return a.date > b.date; });
	}
	for (const auto& exe: extra_exes)
	{
		if (std::any_of(hidden_exes.begin(), hidden_exes.end(), [&](const std::string& h) { return path_key(h) == path_key(exe); }))
		{
			continue;
		}
		if (exe.empty() || !fs::exists(utf8_path(exe), ec) ||
		    std::any_of(out.begin(), out.end(), [&](const InstalledBuild& b) { return path_key(b.exe) == path_key(exe); }))
		{
			continue;
		}
		InstalledBuild b;
		fs::path       p = utf8_path(exe);
		// <repo>\build\kyty_emulator.exe -> "<repo>"
		fs::path dir = p.parent_path();
		b.name       = path_utf8(dir.filename() == "build" ? dir.parent_path().filename() : dir.filename());
		b.codename   = "Local";
		b.date       = FileDate(p);
		b.exe        = exe;
		out.push_back(b);
	}
	return out;
}

// A release file: its public download link first; when that is a 404 and a token is set (a private repo),
// the asset API, which takes the token. *via_api remembers that for the next call (the zip probe makes
// several). Only the answer that counts reaches *error: a 404 the API then gets past is never shown.
static bool GetReleaseFile(const ReleaseInfo& r, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length,
                           std::string* error, const std::string& range, bool* range_ignored, bool* via_api)
{
	if (!*via_api)
	{
		std::string public_error;
		if (http_get(r.zip_url, sink, content_length, &public_error, range, range_ignored))
		{
			error->clear();
			return true;
		}
		const bool not_found = public_error.rfind("HTTP 404", 0) == 0;
		if (!not_found || r.api_url.empty() || !github_token_set())
		{
			*error = not_found && !github_token_set()
			             ? "HTTP 404 from GitHub. If the repository is private, set a GitHub token in Settings > Versions."
			             : public_error;
			return false;
		}
		*via_api = true;
	}
	std::string api_error;
	if (http_get_github_asset(r.api_url, sink, content_length, &api_error, range, range_ignored))
	{
		error->clear();
		return true;
	}
	// 404 / 401: the token doesn't reach this repo (403 is more often the rate limit, so it is shown as is).
	const bool denied = api_error.rfind("HTTP 404", 0) == 0 || api_error.rfind("HTTP 401", 0) == 0;
	*error = denied ? "The GitHub token can't download from this repository (" + api_error.substr(0, api_error.find(" from ")) + ")" : api_error;
	return false;
}

std::string install_release(const ReleaseInfo& r, const std::string& versions_dir, const ProgressFn& progress, std::string* error,
                            const std::atomic<bool>* cancel)
{
	std::error_code    ec;
	const fs::path     root = utf8_path(versions_dir);
	const std::string  name = r.folder.empty() ? r.tag : r.folder;
	const fs::path     dir  = root / utf8_path(name);
	const fs::path     part = root / utf8_path(name + kStagingSuffix);
	const fs::path     old  = root / utf8_path(name + kOldSuffix);
	auto               cancelled = [&] { return cancel != nullptr && cancel->load(); };

	// A crash between the two renames below leaves the working build as "<folder>.old": put it back.
	if (fs::exists(old, ec) && !fs::exists(dir, ec))
	{
		fs::rename(old, dir, ec);
	}
	fs::remove_all(old, ec);
	fs::remove_all(part, ec);
	fs::create_directories(part, ec);
	if (ec)
	{
		*error = "Could not create " + path_utf8(part);
		return {};
	}
	auto fail = [&](const std::string& why)
	{
		*error = why;
		fs::remove_all(part, ec);
		return std::string();
	};

	fs::path zip = part / (std::string("download") + kArchiveExt);
	{
		std::ofstream f(zip, std::ios::binary | std::ios::trunc);
		uint64_t      total = r.zip_size;
		uint64_t      done  = 0;
		std::string   http_error;
		bool          via_api = false;
		bool ok = f && GetReleaseFile(
		                   r,
		                   [&](const char* p, size_t n)
		                   {
			                   if (cancelled())
			                   {
				                   return false;
			                   }
			                   f.write(p, static_cast<std::streamsize>(n));
			                   done += n;
			                   progress(done, total);
			                   return static_cast<bool>(f);
		                   },
		                   &total, &http_error, std::string(), nullptr, &via_api);
		f.close();
		if (!ok)
		{
			return fail(cancelled() ? "Cancelled" : http_error.empty() ? "Could not write " + path_utf8(zip) : http_error);
		}
	}
	if (cancelled())
	{
		return fail("Cancelled");
	}
	std::string unzip_error;
	if (!Unzip(zip, part, &unzip_error))
	{
		return fail(unzip_error);
	}
	fs::remove(zip, ec);
	{
		// Which release this folder holds, for the installed check and the date.
		std::ofstream marker(part / "release.txt", std::ios::trunc);
		marker << r.tag << "\n";
	}
	if (FindExe(part).empty())
	{
		return fail(std::string("No ") + kExeName + " in the release archive");
	}

	// Swap in: the previous build (if any) is set aside first and put back if the rename fails
	// (on Windows that is what happens while its emulator is running).
	// A folder can't be renamed while a file in it is open, and right after unpacking the antivirus is often
	// still scanning the new exe, so each rename is retried for a while.
	auto rename_retry = [&](const fs::path& from, const fs::path& to)
	{
		for (int attempt = 0;; attempt++)
		{
			fs::rename(from, to, ec);
			if (!ec || attempt >= 40 || cancelled())
			{
				return !ec;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
	};
	const bool had_old = fs::exists(dir, ec);
	if (had_old && !rename_retry(dir, old))
	{
		return fail("Could not replace " + path_utf8(dir) + " (is that build running?): " + ec.message());
	}
	if (!rename_retry(part, dir))
	{
		const std::string why = ec.message();
		std::error_code   ec2;
		if (had_old)
		{
			fs::rename(old, dir, ec2);
		}
		return fail("Could not move the new build into " + path_utf8(dir) + ": " + why);
	}
	if (had_old)
	{
		fs::remove_all(old, ec);
	}
	std::string exe = FindExe(dir);
#ifndef _WIN32
	if (!exe.empty())
	{
		// Archives made by CI keep the executable bit; make sure of it anyway.
		fs::permissions(utf8_path(exe), fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
	}
#endif
	return exe;
}

namespace {

uint16_t Rd16(const std::string& b, size_t o)
{
	return static_cast<uint16_t>(static_cast<uint8_t>(b[o]) | (static_cast<uint8_t>(b[o + 1]) << 8));
}

uint32_t Rd32(const std::string& b, size_t o)
{
	return static_cast<uint32_t>(Rd16(b, o)) | (static_cast<uint32_t>(Rd16(b, o + 2)) << 16);
}

uint64_t Rd64(const std::string& b, size_t o)
{
	return static_cast<uint64_t>(Rd32(b, o)) | (static_cast<uint64_t>(Rd32(b, o + 4)) << 32);
}

// File names in a zip central directory; false when the bytes are not one.
bool ListCentralDirectory(const std::string& cd, std::vector<std::string>& names)
{
	size_t pos = 0;
	while (pos + 46 <= cd.size() && Rd32(cd, pos) == 0x02014b50)
	{
		size_t n = Rd16(cd, pos + 28);
		size_t m = Rd16(cd, pos + 30);
		size_t k = Rd16(cd, pos + 32);
		if (pos + 46 + n + m + k > cd.size())
		{
			return false;
		}
		names.push_back(cd.substr(pos + 46, n));
		pos += 46 + n + m + k;
	}
	return !names.empty();
}

// kyty_emulator.exe at most three folders down, like FindExe() looks after unpacking.
bool ListsEmulator(const std::vector<std::string>& names)
{
	const std::string exe = kExeName;
	for (std::string name: names)
	{
		std::replace(name.begin(), name.end(), '\\', '/');
		if (name.size() < exe.size() || name.compare(name.size() - exe.size(), exe.size(), exe) != 0)
		{
			continue;
		}
		if (name.size() > exe.size() && name[name.size() - exe.size() - 1] != '/')
		{
			continue;
		}
		if (std::count(name.begin(), name.end(), '/') <= 3)
		{
			return true;
		}
	}
	return false;
}

} // namespace

ZipProbe probe_release_zip(const ReleaseInfo& r, std::string* error)
{
	if (std::string(kArchiveExt) != ".zip")
	{
		// A .tar.gz has no index at its end; the caller installs it to find out.
		*error = "Not a zip archive";
		return ZipProbe::Unsupported;
	}
	constexpr uint64_t kTailBytes = 128 * 1024;     // the end of the zip: its last entries and the EOCD record
	constexpr uint64_t kMaxCdBytes = 16 * 1024 * 1024; // a central directory bigger than this is not a build zip
	const uint64_t     size       = r.zip_size;
	if (size < 22)
	{
		*error = "The release does not say how big its zip is";
		return ZipProbe::Unsupported;
	}

	bool ignored = false;
	bool via_api = false;
	// Bytes [from, to] of the zip.
	auto fetch = [&](uint64_t from, uint64_t to, std::string& out)
	{
		out.clear();
		return GetReleaseFile(
		    r, [&](const char* p, size_t n) { out.append(p, n); return out.size() <= to - from + 1; }, nullptr, error,
		    std::to_string(from) + "-" + std::to_string(to), &ignored, &via_api);
	};

	std::string    tail;
	const uint64_t tail_len   = std::min(size, kTailBytes);
	const uint64_t tail_start = size - tail_len;
	if (!fetch(tail_start, size - 1, tail) || tail.size() != tail_len)
	{
		if (error->empty())
		{
			*error = "Short reply to a Range request";
		}
		return ZipProbe::Unsupported;
	}

	// End of central directory record: the last "PK\5\6" whose comment reaches the end of the file.
	size_t eocd = std::string::npos;
	for (size_t i = tail.size() - 22 + 1; i-- > 0;)
	{
		if (Rd32(tail, i) == 0x06054b50 && i + 22 + Rd16(tail, i + 20) == tail.size())
		{
			eocd = i;
			break;
		}
	}
	if (eocd == std::string::npos)
	{
		*error = "No zip end record found";
		return ZipProbe::Unsupported;
	}
	uint64_t cd_size   = Rd32(tail, eocd + 12);
	uint64_t cd_offset = Rd32(tail, eocd + 16);
	if (cd_size == 0xFFFFFFFFu || cd_offset == 0xFFFFFFFFu || Rd16(tail, eocd + 10) == 0xFFFF)
	{
		// zip64: the locator sits right before the EOCD and points at the zip64 record.
		if (eocd < 20 || Rd32(tail, eocd - 20) != 0x07064b50)
		{
			*error = "Unreadable zip64 end record";
			return ZipProbe::Unsupported;
		}
		uint64_t    rec_off = Rd64(tail, eocd - 20 + 8);
		std::string rec;
		if (rec_off >= tail_start && rec_off - tail_start + 56 <= tail.size())
		{
			rec = tail.substr(rec_off - tail_start, 56);
		} else if (rec_off + 56 > size || !fetch(rec_off, rec_off + 55, rec) || rec.size() != 56)
		{
			*error = "Unreadable zip64 end record";
			return ZipProbe::Unsupported;
		}
		if (Rd32(rec, 0) != 0x06064b50)
		{
			*error = "Unreadable zip64 end record";
			return ZipProbe::Unsupported;
		}
		cd_size   = Rd64(rec, 40);
		cd_offset = Rd64(rec, 48);
	}
	if (cd_size == 0 || cd_size > kMaxCdBytes || cd_offset + cd_size > size)
	{
		*error = "Odd zip directory size";
		return ZipProbe::Unsupported;
	}

	std::string cd;
	if (cd_offset >= tail_start)
	{
		cd = tail.substr(cd_offset - tail_start, cd_size);
	} else if (!fetch(cd_offset, cd_offset + cd_size - 1, cd) || cd.size() != cd_size)
	{
		if (error->empty())
		{
			*error = "Short reply to a Range request";
		}
		return ZipProbe::Unsupported;
	}

	std::vector<std::string> names;
	if (!ListCentralDirectory(cd, names))
	{
		*error = "Could not read the zip's file list";
		return ZipProbe::Unsupported;
	}
	return ListsEmulator(names) ? ZipProbe::HasEmulator : ZipProbe::NoEmulator;
}

// Deletes a downloaded build's folder from the versions folder (only a folder directly inside it).
bool delete_build_folder(const InstalledBuild& b, const std::string& versions_dir, std::string* error)
{
	if (b.name.empty() || b.name.find_first_of("/\\") != std::string::npos || b.name == "." || b.name == "..")
	{
		*error = "Refusing to delete " + b.name;
		return false;
	}
	std::error_code ec;
	fs::remove_all(utf8_path(versions_dir) / utf8_path(b.name), ec);
	if (ec)
	{
		*error = "Could not delete " + b.name + ": " + ec.message();
		return false;
	}
	return true;
}

} // namespace kyty_launcher
