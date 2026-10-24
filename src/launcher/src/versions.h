#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kyty_launcher {

// A KytyPS5 GitHub release with a Windows build.
struct ReleaseInfo
{
	std::string tag;      // e.g. KytyPS5-2026-10-19-e5f0ee7
	std::string folder;   // where it is unpacked inside the versions folder (the tag for KytyPS5/KytyPS5)
	std::string date;     // 2026-10-19
	std::string codename; // e5f0ee7 (the commit)
	std::string zip_url;  // the ...-Windows-x64.zip asset
	std::string api_url;  // the same asset through the API (takes the token: private repos)
	uint64_t    zip_size = 0;
};

// An emulator build on disk.
struct InstalledBuild
{
	std::string name;
	std::string codename;
	std::string date;
	std::string exe;
	std::string tag;            // the release tag it was downloaded from (empty for local builds)
	bool        managed = false; // lives in the versions folder (Delete removes it)
};

// The default release source: its tab is called KytyPS5-main.
constexpr const char* kDefaultRepo = "KytyPS5/KytyPS5";

// "https://github.com/owner/repo/releases/tag/x" (or just owner/repo) -> "owner/repo"; "" when it is not a GitHub repo link.
std::string parse_repo_link(const std::string& link);

// Releases of a GitHub repo ("owner/repo") that carry a Windows zip, newest first.
// Empty with *error set on failure.
// The list is saved in cache_dir (empty = no cache) and reused without a request for a few minutes;
// after that GitHub is asked whether it changed (ETag), and `force` skips the wait. When GitHub
// can't be reached the saved list comes back anyway, with *note (optional) saying how old it is.
// `cancel` (optional) is checked between pages; a cancelled fetch returns nothing with *error "Cancelled".
std::vector<ReleaseInfo> fetch_releases(const std::string& repo, const std::string& cache_dir, bool force, std::string* error,
                                        std::string* note = nullptr, const std::atomic<bool>* cancel = nullptr);

// The release lists saved by fetch_releases in cache_dir: how many files and bytes, and deleting them.
void release_cache_size(const std::string& cache_dir, int* files, uint64_t* bytes);
int  clear_release_cache(const std::string& cache_dir); // returns the number of files deleted

// Builds unpacked in versions_dir (one folder per release tag), newest first, then the
// extra exes (custom builds and the configured emulator) that are not already listed.
// Exes in hidden_exes are left out of both (their folders stay on disk).
std::vector<InstalledBuild> scan_installed(const std::string& versions_dir, const std::vector<std::string>& extra_exes,
                                           const std::vector<std::string>& hidden_exes);

// Deletes a downloaded build's folder from the versions folder (only a folder directly inside it).
bool delete_build_folder(const InstalledBuild& b, const std::string& versions_dir, std::string* error);

// Does the release's zip hold kyty_emulator.exe (up to three folders deep)? Answered by reading
// only the zip's central directory with HTTP Range requests (a few KB, not the 28 MB zip).
// Unsupported (with *error saying why) when the server ignores Range or the zip can't be read
// that way: the caller then downloads, unpacks and searches instead.
enum class ZipProbe { HasEmulator, NoEmulator, Unsupported };
ZipProbe probe_release_zip(const ReleaseInfo& r, std::string* error);

using ProgressFn = std::function<void(uint64_t done, uint64_t total)>;

// Downloads the release zip and unpacks it into versions_dir/<folder>. The work happens in a
// "<folder>.part" folder that replaces the old one only once everything succeeded, so a failed or
// cancelled download leaves an existing build as it was. `cancel` (optional) stops the download.
// Returns the path of the kyty_emulator.exe inside, or "" with *error set.
std::string install_release(const ReleaseInfo& r, const std::string& versions_dir, const ProgressFn& progress, std::string* error,
                            const std::atomic<bool>* cancel = nullptr);


} // namespace kyty_launcher
