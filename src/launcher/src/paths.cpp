#include "paths.h"

#include <cstdlib>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace kyty_launcher {

namespace fs = std::filesystem;

namespace {

#ifndef _WIN32
// $name when set and non-empty, else `fallback`.
fs::path EnvDir(const char* name, const fs::path& fallback)
{
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' ? fs::path(v) : fallback;
}

fs::path Home()
{
	const char* h = std::getenv("HOME");
	return h != nullptr && *h != '\0' ? fs::path(h) : fs::path("/");
}
#endif

fs::path ExePath(const char* argv0)
{
#ifdef _WIN32
	(void)argv0;
	wchar_t buf[MAX_PATH * 4];
	DWORD   n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
	return fs::path(std::wstring(buf, n));
#elif defined(__APPLE__)
	char     buf[4096];
	uint32_t size = sizeof(buf);
	if (_NSGetExecutablePath(buf, &size) == 0)
	{
		std::error_code ec;
		auto            p = fs::weakly_canonical(buf, ec);
		return ec ? fs::path(buf) : p;
	}
	return fs::absolute(argv0);
#else
	std::error_code ec;
	auto            p = fs::read_symlink("/proc/self/exe", ec);
	return ec ? fs::absolute(argv0) : p;
#endif
}

} // namespace

fs::path exe_dir(const char* argv0)
{
	return ExePath(argv0).parent_path();
}

fs::path portable_root(const char* argv0)
{
	// portable.txt sits next to the exe; on macOS next to the .app bundle (<bundle>/Contents/MacOS/<exe>).
	fs::path beside = exe_dir(argv0);
#ifdef __APPLE__
	for (fs::path q = beside; q.has_parent_path() && q != q.parent_path(); q = q.parent_path())
	{
		if (q.extension() == ".app")
		{
			beside = q.parent_path();
			break;
		}
	}
#endif
	return beside;
}

Paths compute_paths(const char* argv0)
{
	Paths p;
	p.exe_dir = exe_dir(argv0);

	const fs::path beside = portable_root(argv0);
	std::error_code ec;
	p.portable = fs::exists(beside / "portable.txt", ec);

	fs::path config_root; // holds kyty.json (and games/)
	if (p.portable)
	{
		p.root      = beside;
		config_root = p.root / "configs";
	} else
	{
#ifdef _WIN32
		PWSTR local = nullptr;
		if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &local)) && local != nullptr)
		{
			p.root = fs::path(local) / "KytyPS5";
		}
		if (local != nullptr)
		{
			CoTaskMemFree(local);
		}
		if (p.root.empty())
		{
			// Never expected; stay usable by keeping everything next to the exe.
			p.root = beside / "KytyPS5Data";
		}
		config_root = p.root / "configs";
#elif defined(__APPLE__)
		p.root      = Home() / "Library" / "Application Support" / "KytyPS5";
		config_root = p.root / "configs";
#else
		p.root      = EnvDir("XDG_DATA_HOME", Home() / ".local" / "share") / "kytyps5";
		config_root = EnvDir("XDG_CONFIG_HOME", Home() / ".config") / "kytyps5";
#endif
	}
	p.config_dir       = config_root;
	p.games_config_dir = config_root / "games";
	p.versions_dir     = p.root / "versions";
	p.cache_dir        = p.root / "cache";
	p.logs_dir         = p.root / "logs";
	p.run_dir          = p.root / "run";
	return p;
}

} // namespace kyty_launcher
