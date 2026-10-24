#include "launch.h"

#include "library.h"

#include <algorithm>
#include <cctype>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
extern char** environ; // NOLINT
#endif

namespace kyty_launcher {

namespace fs = std::filesystem;

namespace {

#ifdef _WIN32
std::wstring Widen(const std::string& s)
{
	if (s.empty())
	{
		return {};
	}
	int          n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring w(n, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
	return w;
}

std::string Narrow(const wchar_t* w, size_t len)
{
	if (len == 0)
	{
		return {};
	}
	int         n = WideCharToMultiByte(CP_UTF8, 0, w, static_cast<int>(len), nullptr, 0, nullptr, nullptr);
	std::string s(n, '\0');
	WideCharToMultiByte(CP_UTF8, 0, w, static_cast<int>(len), s.data(), n, nullptr, nullptr);
	return s;
}
#endif

bool SameName(const std::string& a, const std::string& b)
{
#ifdef _WIN32
	return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y)
	                                          { return std::toupper(static_cast<unsigned char>(x)) == std::toupper(static_cast<unsigned char>(y)); });
#else
	return a == b;
#endif
}

// Quote one argument for CommandLineToArgvW / the MSVC CRT.
std::string QuoteArg(const std::string& a)
{
	if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos)
	{
		return a;
	}
	std::string out = "\"";
	size_t      backslashes = 0;
	for (char c: a)
	{
		if (c == '\\')
		{
			backslashes++;
			continue;
		}
		if (c == '"')
		{
			out.append(backslashes * 2 + 1, '\\');
		} else
		{
			out.append(backslashes, '\\');
		}
		backslashes = 0;
		out += c;
	}
	out.append(backslashes * 2, '\\');
	return out + "\"";
}

// ---- Per-build --help probing: old builds die on unknown options ----

// How long "<exe> --help" gets before it counts as a hang (killed, build unchecked).
constexpr uint64_t kHelpProbeTimeoutMs = 5000;

// Runs "<exe> --help" and returns whatever it printed (stdout and stderr merged: usage
// goes to either depending on the build). Empty on any failure: missing exe or pipe,
// spawn error, timeout kill, crash, or silence.
std::string CaptureHelp(const std::string& exe)
{
#ifdef _WIN32
	SECURITY_ATTRIBUTES sa {sizeof(sa), nullptr, TRUE};
	HANDLE              rd = nullptr;
	HANDLE              wr = nullptr;
	if (CreatePipe(&rd, &wr, &sa, 0) == FALSE)
	{
		return {};
	}
	SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
	HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

	STARTUPINFOW si {};
	si.cb         = sizeof(si);
	si.dwFlags    = STARTF_USESTDHANDLES;
	si.hStdInput  = nul != INVALID_HANDLE_VALUE ? nul : nullptr;
	si.hStdOutput = wr;
	si.hStdError  = wr;

	const fs::path  dir = utf8_path(exe).parent_path();
	std::wstring    cmd = Widen(QuoteArg(exe) + " --help"); // mutable: CreateProcessW may write into it
	PROCESS_INFORMATION pi {};
	const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
	                               dir.empty() ? nullptr : dir.wstring().c_str(), &si, &pi);
	CloseHandle(wr);
	if (nul != INVALID_HANDLE_VALUE)
	{
		CloseHandle(nul);
	}
	if (ok == FALSE)
	{
		CloseHandle(rd);
		return {};
	}
	CloseHandle(pi.hThread);

	// Drain while waiting: a chatty --help must never wedge the pipe and look like a hang.
	std::string    out;
	const uint64_t deadline = GetTickCount64() + kHelpProbeTimeoutMs;
	for (;;)
	{
		const DWORD status = WaitForSingleObject(pi.hProcess, 250);
		for (;;)
		{
			DWORD avail = 0;
			if (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) == FALSE || avail == 0)
			{
				break;
			}
			char        buf[4096];
			DWORD       got = 0;
			const DWORD n   = avail < sizeof(buf) ? avail : static_cast<DWORD>(sizeof(buf));
			if (ReadFile(rd, buf, n, &got, nullptr) == FALSE || got == 0)
			{
				break;
			}
			out.append(buf, got);
		}
		if (status == WAIT_OBJECT_0)
		{
			break;
		}
		if (GetTickCount64() >= deadline)
		{
			TerminateProcess(pi.hProcess, 1);
			WaitForSingleObject(pi.hProcess, 2000);
			CloseHandle(rd);
			CloseHandle(pi.hProcess);
			return {};
		}
	}
	CloseHandle(rd);
	CloseHandle(pi.hProcess);
	return out;
#else
	int fds[2] = {-1, -1};
	if (pipe(fds) != 0)
	{
		return {};
	}
	fcntl(fds[0], F_SETFD, FD_CLOEXEC); // a grandchild must not hold our read end open
	const pid_t pid = fork();
	if (pid < 0)
	{
		close(fds[0]);
		close(fds[1]);
		return {};
	}
	if (pid == 0)
	{
		dup2(fds[1], STDOUT_FILENO);
		dup2(fds[1], STDERR_FILENO);
		close(fds[0]);
		close(fds[1]);
		const int nul = open("/dev/null", O_RDONLY);
		if (nul >= 0)
		{
			dup2(nul, STDIN_FILENO);
			if (nul != STDIN_FILENO)
			{
				close(nul);
			}
		}
#ifdef __linux__
		prctl(PR_SET_PDEATHSIG, SIGKILL); // an orphaned probe dies with the launcher
#endif
		execl(exe.c_str(), exe.c_str(), "--help", static_cast<char*>(nullptr));
		_exit(127);
	}
	close(fds[1]);

	std::string out;
	bool        finished = false; // child reaped
	bool        eof      = false; // write end closed
	struct timespec start {};
	clock_gettime(CLOCK_MONOTONIC, &start);
	const auto elapsed_ms = [&]
	{
		struct timespec now {};
		clock_gettime(CLOCK_MONOTONIC, &now);
		return static_cast<uint64_t>(now.tv_sec - start.tv_sec) * 1000u +
		       static_cast<uint64_t>(now.tv_nsec - start.tv_nsec) / 1000000u;
	};
	while ((!finished || !eof) && elapsed_ms() < kHelpProbeTimeoutMs)
	{
		fd_set         rfds;
		FD_ZERO(&rfds);
		FD_SET(fds[0], &rfds);
		struct timeval slice {0, 100000}; // 100 ms
		const int      ready = select(fds[0] + 1, &rfds, nullptr, nullptr, &slice);
		if (ready < 0)
		{
			if (errno == EINTR)
			{
				continue;
			}
			break;
		}
		if (ready > 0 && !eof)
		{
			char          buf[4096];
			const ssize_t n = read(fds[0], buf, sizeof(buf));
			if (n > 0)
			{
				out.append(buf, static_cast<size_t>(n));
			} else
			{
				eof = true;
			}
		}
		int         status = 0;
		const pid_t r      = waitpid(pid, &status, WNOHANG);
		if (r == pid)
		{
			finished = true;
		}
	}
	if (!finished)
	{
		kill(pid, SIGKILL);
		int status = 0;
		waitpid(pid, &status, 0);
		close(fds[0]);
		return {}; // hung: the build is unchecked
	}
	// Reaped in time: drain whatever is left, then hand over (possibly empty) output.
	for (;;)
	{
		fd_set         rfds;
		FD_ZERO(&rfds);
		FD_SET(fds[0], &rfds);
		struct timeval slice {0, 0};
		if (select(fds[0] + 1, &rfds, nullptr, nullptr, &slice) <= 0)
		{
			break;
		}
		char          buf[4096];
		const ssize_t n = read(fds[0], buf, sizeof(buf));
		if (n <= 0)
		{
			break;
		}
		out.append(buf, static_cast<size_t>(n));
	}
	close(fds[0]);
	return out;
#endif
}

// Every "--option" token heading a "  --" usage line (the --help format).
std::set<std::string> ParseHelpOptions(const std::string& text)
{
	std::set<std::string> out;
	size_t                pos = 0;
	while (pos < text.size())
	{
		size_t eol = text.find('\n', pos);
		if (eol == std::string::npos)
		{
			eol = text.size();
		}
		std::string line = text.substr(pos, eol - pos);
		if (!line.empty() && line.back() == '\r')
		{
			line.pop_back();
		}
		if (line.rfind("  --", 0) == 0)
		{
			const size_t end = line.find_first_of(" \t=", 3);
			const std::string opt = line.substr(2, end == std::string::npos ? end : end - 2);
			if (opt.size() > 2)
			{
				out.insert(opt);
			}
		}
		pos = eol + 1;
	}
	return out;
}

// The usable --help option list, or false when the build could not be checked (missing,
// failed spawn, timeout kill, crash, or silence). Cached per exe path, size and mtime,
// so a rebuild is checked again but normal launches add no delay.
bool SupportedEmuOptions(const std::string& exe, std::set<std::string>& options)
{
	static std::mutex                    mutex;
	static std::map<std::string, std::pair<bool, std::set<std::string>>> cache; // key -> (usable, options)

	std::error_code     ec;
	const uintmax_t     size  = fs::file_size(utf8_path(exe), ec);
	std::error_code     ec2;
	const auto          mtime = fs::last_write_time(utf8_path(exe), ec2);
	if (ec || ec2 || exe.empty())
	{
		return false;
	}
	const std::string key =
	    exe + "|" + std::to_string(size) + "|" + std::to_string(mtime.time_since_epoch().count());
	{
		std::lock_guard lock(mutex);
		const auto      it = cache.find(key);
		if (it != cache.end())
		{
			options = it->second.second;
			return it->second.first;
		}
	}
	const std::set<std::string> found = ParseHelpOptions(CaptureHelp(exe));
	const bool                  usable = !found.empty();
	{
		std::lock_guard lock(mutex);
		cache[key] = {usable, found};
		if (usable)
		{
			options = found;
		}
	}
	return usable;
}

bool IsOptionToken(const std::string& tok)
{
	return tok.size() > 2 && tok[0] == '-' && tok[1] == '-';
}

std::string OptionName(const std::string& tok)
{
	const size_t eq = tok.find('=');
	return tok.substr(0, eq);
}

// Options every build is assumed to accept (without --game there is no launch at all).
bool AlwaysSentOption(const std::string& opt)
{
	return opt == "--game" || opt == "--screen-width" || opt == "--screen-height" || opt == "--printf-direction" ||
	       opt == "--shader-log-folder" || opt == "--redzone";
}

const char* OptionHint(const std::string& opt)
{
	if (opt == "--trophy-duration")
	{
		return "trophy popup time";
	}
	return nullptr;
}

// Drops options rejected by `keep` (with their values: a following token that is not
// itself an option) and names them in `dropped`, in order.
template <typename Keep>
void FilterArgs(std::vector<std::string>& args, Keep keep, std::vector<std::string>& dropped)
{
	std::vector<std::string> out;
	for (size_t i = 0; i < args.size();)
	{
		const std::string& tok = args[i];
		if (IsOptionToken(tok) && !keep(OptionName(tok)))
		{
			dropped.push_back(OptionName(tok));
			i++;
			if (i < args.size() && !IsOptionToken(args[i]))
			{
				i++; // its value
			}
		}
		else
		{
			out.push_back(tok);
			i++;
		}
	}
	args = std::move(out);
}

// Drops what this build's --help does not list (warning per option) or, when the build
// could not be checked, everything but the basic set. Hand-typed extra_args always pass
// through as written.
void FilterUnsupportedOptions(LaunchPlan& p, const std::string& exe)
{
	if (exe.empty())
	{
		return; // nothing to check against; the missing-emulator warning below still fires
	}
	std::set<std::string> supported;
	if (SupportedEmuOptions(exe, supported))
	{
		std::vector<std::string> dropped;
		FilterArgs(p.args, [&](const std::string& opt) { return opt == "--game" || supported.count(opt) != 0; }, dropped);
		std::set<std::string> warned;
		for (const auto& opt: dropped)
		{
			if (!warned.insert(opt).second)
			{
				continue;
			}
			std::string w = "This build doesn't support " + opt;
			if (const char* hint = OptionHint(opt))
			{
				w += " (" + std::string(hint) + ")";
			}
			w += "; skipped";
			p.warnings.push_back(std::move(w));
		}
		std::set<std::string> extra_warned;
		std::istringstream      parts(p.extra_args);
		for (std::string tok; parts >> tok;)
		{
			if (!IsOptionToken(tok))
			{
				continue;
			}
			const std::string name = OptionName(tok);
			if (name != "--game" && supported.count(name) == 0 && extra_warned.insert(name).second)
			{
				p.warnings.push_back("Extra argument " + name + " isn't in this build's --help; passing it through anyway");
			}
		}
	} else
	{
		std::vector<std::string> dropped;
		FilterArgs(p.args, [&](const std::string& opt) { return AlwaysSentOption(opt); }, dropped);
		(void)dropped;
		p.warnings.push_back("Couldn't check " + exe + " for supported options; sending only the basic set");
	}
}

} // namespace

LaunchPlan make_plan(const Settings& s)
{
	LaunchPlan p;
	p.exe = s.emulator;
	p.cwd = s.run_dir;

	int w = s.width > 0 ? s.width : 2560;
	int h = s.height > 0 ? s.height : 1440;
	if (w != s.width || h != s.height)
	{
		p.warnings.push_back("Bad window size, using " + std::to_string(w) + "x" + std::to_string(h));
	}

	static const char* const direction[] = {"Silent", "Console", "File"};
	auto in_range = [](int v, int n) { return v >= 0 && v < n ? v : 0; };

	// The emulator accepts 1-16 bytes; cut at a character boundary.
	std::string user_name = s.user_name.substr(0, 16);
	while (!user_name.empty() && s.user_name.size() > user_name.size() && (static_cast<unsigned char>(s.user_name[user_name.size()]) & 0xC0) == 0x80)
	{
		user_name.pop_back();
	}

	p.args = {"--game",
	          s.game,
	          "--screen-width",
	          std::to_string(w),
	          "--screen-height",
	          std::to_string(h),
	          "--printf-direction",
	          direction[in_range(s.printf_dir, 3)],
	          "--shader-log-folder",
	          s.shader_log_folder.empty() ? path_utf8(utf8_path(s.run_dir) / "shaders") : s.shader_log_folder,
	          "--user-name",
	          s.user_name.empty() ? "Kyty" : user_name,
	          "--user-id",
	          std::to_string(s.user_id >= 0 && s.user_id <= 0x7ffffcff && s.user_id != 0xfe && s.user_id != 0xff ? s.user_id : 1000),
	          "--console-language",
	          std::to_string(s.console_language >= 0 && s.console_language <= 29 ? s.console_language : 1)};
#ifdef _WIN32
	// The red zone protection is a Windows (SysV ABI on the Windows stack) option, off unless the flag is given.
	if (s.red_zone)
	{
		p.args.push_back("--redzone");
	}
#endif
	// One --keymap per binding (the Qt launcher's host_input_mapping).
	for (const auto& binding: s.host_input_mapping)
	{
		p.args.push_back("--keymap");
		p.args.push_back(binding);
	}
	if (!s.microphone.empty())
	{
		p.args.push_back("--mic");
		p.args.push_back(s.microphone);
	}
	if (s.amd_cpu)
	{
		p.args.push_back("--amd-cpu");
	}
	// Fullscreen, cursor, performance panel, boot logos and network are off by default; each is a bare flag.
	for (const auto& [on, flag]: {std::pair {s.fullscreen, "--fullscreen"},
	                              std::pair {s.hide_cursor, "--hide-cursor"},
	                              std::pair {s.perf_overlay, "--perf-overlay"},
	                              std::pair {s.skip_notice_screen, "--skip-notice-screen"},
	                              std::pair {s.network_connected, "--network-connected"}})
	{
		if (on)
		{
			p.args.push_back(flag);
		}
	}
	// Controller: the lightbar colour only when custom (#RRGGBB, as the emulator parses it); the emulator's own
	// defaults (vibration 100, volume 50, keyboard shared with player 1) are not passed.
	if (s.controller_color.size() == 7 && s.controller_color[0] == '#' &&
	    s.controller_color.find_first_not_of("0123456789abcdefABCDEF", 1) == std::string::npos)
	{
		p.args.push_back("--controller-color");
		p.args.push_back(s.controller_color);
	}
	if (s.controller_vibration != 100)
	{
		p.args.push_back("--controller-vibration");
		p.args.push_back(std::to_string(std::clamp(s.controller_vibration, 0, 100)));
	}
	if (s.controller_volume != 50)
	{
		p.args.push_back("--controller-volume");
		p.args.push_back(std::to_string(std::clamp(s.controller_volume, 0, 100)));
	}
	if (s.keyboard_player != 0)
	{
		p.args.push_back("--keyboard-player");
		p.args.push_back(std::to_string(std::clamp(s.keyboard_player, 0, 4)));
	}
	// Trophy popups are on by default; the emulator takes a bare flag to turn them off.
	if (!s.trophy_notifications)
	{
		p.args.push_back("--no-trophy-notifications");
	}
	p.args.push_back("--trophy-duration");
	p.args.push_back(std::to_string(std::clamp(s.trophy_duration, 5, 10)));

	{
		static const char* const present[] = {"Fifo", "Mailbox", "Immediate"};
		static const char* const optimize[] = {"None", "Size", "Performance"};
		p.args.push_back("--present-mode");
		p.args.push_back(present[in_range(s.present_mode, 3)]);
		p.args.push_back("--vblank-frequency");
		p.args.push_back(std::to_string(s.vblank > 0 ? s.vblank : 60));
		p.args.push_back("--shader-optimization-type");
		p.args.push_back(optimize[in_range(s.shader_opt, 3)]);
		p.args.push_back("--shader-log-direction");
		p.args.push_back(direction[in_range(s.shader_log, 3)]);
		if (s.gpu_index >= 0)
		{
			p.args.push_back("--gpu");
			p.args.push_back(std::to_string(s.gpu_index));
		}
		// The emulator's switches are bare flags: present means on, a following "true"/"false" is rejected.
		for (const auto& [on, flag]: {std::pair {s.readback, "--readback-linear-images"},
			                              std::pair {s.readback_tiled, "--readback-tiled-images"},
		                              std::pair {s.sync_raw, "--sync-raw-image-buffers"},
		                              std::pair {s.vulkan_validation, "--vulkan-validation"},
		                              std::pair {s.shader_validation, "--shader-validation"},
		                              std::pair {s.cmd_dump, "--command-buffer-dump"}})
		{
			if (on)
			{
				p.args.push_back(flag);
			}
		}
		if (!s.cmd_dump_folder.empty())
		{
			p.args.push_back("--command-buffer-dump-folder");
			p.args.push_back(s.cmd_dump_folder);
		}
		if (!s.printf_file.empty())
		{
			p.args.push_back("--printf-output-file");
			p.args.push_back(s.printf_file);
		}
		if (s.tessellation)
		{
			p.args.push_back("--tessellation");
		}
		if (s.hdr)
		{
			p.args.push_back("--hdr");
		}
		if (s.renderdoc)
		{
			p.args.push_back("--rd");
		}
		if (s.profiler)
		{
			p.args.push_back("--profile");
		}
	}
	// The one add-ons (DLC) folder; a folder that is gone is left out rather than handed to the emulator.
	if (!s.addons_dir.empty())
	{
		std::error_code addons_ec;
		if (fs::is_directory(utf8_path(s.addons_dir), addons_ec))
		{
			p.args.push_back("--addcont-dir");
			p.args.push_back(s.addons_dir);
		} else
		{
			p.warnings.push_back("Add-ons folder not found, not passed: " + s.addons_dir);
		}
	}
	p.extra_args = s.extra_args;

	// Old builds die on unknown options: keep only what this build's --help lists
	// (warning per dropped option), or the basic set when it can't be checked.
	// Hand-typed extra_args always pass through as written.
	FilterUnsupportedOptions(p, s.emulator);

	if (!fs::exists(utf8_path(s.emulator)))
	{
		p.warnings.push_back("Emulator not found: " + s.emulator);
	}
	if (!fs::exists(utf8_path(s.game)))
	{
		p.warnings.push_back("Game not found: " + s.game);
	}

	// The launcher sets SLINT_NO_MUDA on itself; the emulator must not inherit it.
	p.env.push_back({"SLINT_NO_MUDA", "", EnvChange::Mode::Unset});

	// User env table (Settings::env): an empty value removes the variable from the child's environment.
	for (const auto& [name, value]: s.env)
	{
		if (!name.empty() && name.find('=') == std::string::npos)
		{
			p.env.push_back({name, value, value.empty() ? EnvChange::Mode::Unset : EnvChange::Mode::Set});
		}
	}
	return p;
}

std::string emulator_for_game(const Settings& user, const Paths& paths, const std::string& title_id, std::string* warning)
{
	const std::string pin = read_game_emulator(paths, title_id);
	if (pin.empty())
	{
		return user.emulator;
	}
	std::error_code ec;
	if (fs::exists(utf8_path(pin), ec))
	{
		return pin;
	}
	if (warning != nullptr)
	{
		*warning = "Pinned emulator " + build_display_name(user, pin) + " is missing, using default";
	}
	return user.emulator;
}

LaunchPlan make_launch_plan(const Settings& user, const Paths& paths)
{
	std::vector<std::string> layers = {"defaults: built in"};
	const std::string        title  = read_title_id(user.game);
	layers.push_back("title id: " + (title.empty() ? std::string("(none, param.json not found)") : title));
	// The pin decides which emulator runs, so everything that depends on the emulator (the build config next to
	// it, the exe itself) sees the exe actually launched.
	std::string pin_warning;
	Settings    launch = user;
	launch.emulator    = emulator_for_game(user, paths, title, &pin_warning);
	if (launch.emulator != user.emulator)
	{
		layers.push_back("emulator pin: " + launch.emulator);
	}
	std::vector<std::string> overridden;
	LaunchPlan plan = make_plan(resolve_for_launch(launch, paths, title, &layers, true, &overridden));
	// A build's config files outrank the user's values, so say so: this is the usual reason a build "ignores" a
	// setting. Shown with the plan (--dry-run) and in the console log before the command line.
	for (auto& w: overridden)
	{
		plan.warnings.push_back(std::move(w));
	}
	if (!pin_warning.empty())
	{
		plan.warnings.insert(plan.warnings.begin(), pin_warning);
	}
	plan.layers = std::move(layers);
	// The output of the run lives in the launcher's logs folder, not in the saves folder: a run dir pointed at
	// Documents or at the Qt launcher's folder must not grow a log file.
	plan.log_file = path_utf8(paths.logs_dir / "last-run.log");
	return plan;
}

EnvList current_environment()
{
	EnvList list;
#ifdef _WIN32
	wchar_t* block = GetEnvironmentStringsW();
	for (const wchar_t* e = block; e != nullptr && *e != L'\0'; e += wcslen(e) + 1)
	{
		// Entries like "=C:=C:\dir" are per-drive cwds; keep them, the name is the leading '='.
		const wchar_t* eq = wcschr(e + 1, L'=');
		if (eq == nullptr)
		{
			continue;
		}
		list.emplace_back(Narrow(e, eq - e), Narrow(eq + 1, wcslen(eq + 1)));
	}
	FreeEnvironmentStringsW(block);
#else
	for (char** e = environ; *e != nullptr; e++)
	{
		const char* eq = std::strchr(*e, '=');
		if (eq != nullptr)
		{
			list.emplace_back(std::string(static_cast<const char*>(*e), eq), std::string(eq + 1));
		}
	}
#endif
	return list;
}

EnvList apply_env(const EnvList& parent, const std::vector<EnvChange>& changes)
{
	EnvList env = parent;
	for (const auto& c: changes)
	{
		auto it = std::find_if(env.begin(), env.end(), [&](const auto& kv) { return SameName(kv.first, c.name); });
		switch (c.mode)
		{
			case EnvChange::Mode::Unset:
				if (it != env.end())
				{
					env.erase(it);
				}
				break;
			case EnvChange::Mode::SetIfUnset:
				if (it == env.end())
				{
					env.emplace_back(c.name, c.value);
				}
				break;
			case EnvChange::Mode::Set:
				if (it != env.end())
				{
					it->second = c.value;
				} else
				{
					env.emplace_back(c.name, c.value);
				}
				break;
		}
	}
	// Windows wants the block sorted by name, case-insensitively.
	std::sort(env.begin(), env.end(),
	          [](const auto& a, const auto& b)
	          {
		          return std::lexicographical_compare(a.first.begin(), a.first.end(), b.first.begin(), b.first.end(), [](char x, char y)
		                                              { return std::toupper(static_cast<unsigned char>(x)) < std::toupper(static_cast<unsigned char>(y)); });
	          });
	return env;
}

std::string command_line(const LaunchPlan& plan)
{
	std::string cmd = QuoteArg(plan.exe);
	for (const auto& a: plan.args)
	{
		cmd += " " + QuoteArg(a);
	}
	if (!plan.extra_args.empty())
	{
		cmd += " " + plan.extra_args;
	}
	return cmd;
}

std::string describe(const LaunchPlan& plan)
{
	std::string out;
	for (const auto& w: plan.warnings)
	{
		out += "warning: " + w + "\n";
	}
	out += "config layers (lowest first):\n";
	for (const auto& l: plan.layers)
	{
		out += "  " + l + "\n";
	}
	out += "exe: " + plan.exe + "\n";
	out += "cwd: " + plan.cwd + "\n";
	out += "log: " + (plan.log_file.empty() ? path_utf8(utf8_path(plan.cwd) / "last-run.log") : plan.log_file) + "\n";
	out += "cmd: " + command_line(plan) + "\n";
	out += "env changes:\n";
	for (const auto& c: plan.env)
	{
		switch (c.mode)
		{
			case EnvChange::Mode::Set: out += "  " + c.name + "=" + c.value + "\n"; break;
			case EnvChange::Mode::SetIfUnset: out += "  " + c.name + "=" + c.value + "  (if not already set)\n"; break;
			case EnvChange::Mode::Unset: out += "  unset " + c.name + "\n"; break;
		}
	}
	out += "child env:\n";
	for (const auto& [k, v]: apply_env(current_environment(), plan.env))
	{
		out += "  " + k + "=" + v + "\n";
	}
	return out;
}

#ifdef _WIN32

struct ChildProcess::Impl
{
	std::atomic<bool> running {false};
	std::mutex mutex;
	HANDLE     process = nullptr;
	HANDLE     read    = nullptr;
};

bool ChildProcess::Start(const LaunchPlan& plan, OutputFn on_output, ExitFn on_exit, std::string* error)
{
	if (Running())
	{
		*error = "The emulator is already running";
		return false;
	}
	if (m_reader.joinable())
	{
		m_reader.join();
	}

	std::error_code ec;
	fs::create_directories(utf8_path(plan.cwd), ec);

	std::wstring block;
	for (const auto& [k, v]: apply_env(current_environment(), plan.env))
	{
		block += Widen(k) + L"=" + Widen(v);
		block.push_back(L'\0');
	}
	block.push_back(L'\0');

	SECURITY_ATTRIBUTES sa {sizeof(sa), nullptr, TRUE};
	HANDLE              rd = nullptr;
	HANDLE              wr = nullptr;
	if (CreatePipe(&rd, &wr, &sa, 0) == FALSE)
	{
		*error = "CreatePipe failed";
		return false;
	}
	SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
	HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

	STARTUPINFOW si {};
	si.cb         = sizeof(si);
	si.dwFlags    = STARTF_USESTDHANDLES;
	si.hStdInput  = nul;
	si.hStdOutput = wr;
	si.hStdError  = wr;

	std::wstring        exe  = Widen(plan.exe);
	std::wstring        cmd  = Widen(command_line(plan));
	std::wstring        cwd  = Widen(plan.cwd);
	PROCESS_INFORMATION pi {};
	// Started suspended so it is in the job before it runs any code.
	BOOL ok = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
	                         CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW | CREATE_SUSPENDED, block.data(),
	                         cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
	DWORD err = GetLastError();
	CloseHandle(wr);
	if (nul != INVALID_HANDLE_VALUE)
	{
		CloseHandle(nul);
	}
	if (ok == FALSE)
	{
		CloseHandle(rd);
		*error = "CreateProcess failed, error " + std::to_string(err);
		return false;
	}
	{
		// One job for the launcher's lifetime; its handle closes when the launcher exits
		// (however it exits), and the kill-on-close limit then ends the emulator.
		static HANDLE job = []
		{
			HANDLE                               h = CreateJobObjectW(nullptr, nullptr);
			JOBOBJECT_EXTENDED_LIMIT_INFORMATION info {};
			info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			if (h != nullptr)
			{
				SetInformationJobObject(h, JobObjectExtendedLimitInformation, &info, sizeof(info));
			}
			return h;
		}();
		if (job != nullptr)
		{
			AssignProcessToJobObject(job, pi.hProcess);
		}
	}
	ResumeThread(pi.hThread);
	CloseHandle(pi.hThread);

	m_impl          = std::make_shared<Impl>();
	m_impl->process = pi.hProcess;
	m_impl->read    = rd;
	m_impl->running = true;

	// <root>/logs/last-run.log (set by make_launch_plan); cwd/last-run.log when the plan has no log path.
	fs::path log_path = plan.log_file.empty() ? utf8_path(plan.cwd) / "last-run.log" : utf8_path(plan.log_file);
	std::error_code log_ec;
	if (!log_path.parent_path().empty())
	{
		fs::create_directories(log_path.parent_path(), log_ec);
	}
	m_reader          = std::thread(
        [impl = m_impl, log_path, on_output = std::move(on_output), on_exit = std::move(on_exit)]
        {
            std::ofstream log(log_path, std::ios::binary | std::ios::trunc);
            char          buf[8192];
            DWORD         got = 0;
            while (ReadFile(impl->read, buf, sizeof(buf), &got, nullptr) != FALSE && got > 0)
            {
                log.write(buf, got);
                log.flush();
                on_output(std::string(buf, got));
            }
            CloseHandle(impl->read);
            WaitForSingleObject(impl->process, INFINITE);
            DWORD code = 0;
            GetExitCodeProcess(impl->process, &code);
            {
                std::lock_guard lock(impl->mutex);
                CloseHandle(impl->process);
                impl->process = nullptr;
            }
            impl->running = false;
            on_exit(static_cast<int>(code));
        });
	return true;
}

void ChildProcess::Stop()
{
	if (m_impl)
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->process != nullptr)
		{
			TerminateProcess(m_impl->process, 1);
		}
	}
}

#else

struct ChildProcess::Impl
{
	std::atomic<bool> running {false};
	std::mutex mutex;
	pid_t      pid  = -1;
	int        read = -1;
};

bool ChildProcess::Start(const LaunchPlan& plan, OutputFn on_output, ExitFn on_exit, std::string* error)
{
	if (Running())
	{
		*error = "The emulator is already running";
		return false;
	}
	if (m_reader.joinable())
	{
		m_reader.join();
	}

	std::error_code ec;
	fs::create_directories(utf8_path(plan.cwd), ec);

	std::vector<std::string> env_strings;
	for (const auto& [k, v]: apply_env(current_environment(), plan.env))
	{
		env_strings.push_back(k + "=" + v);
	}
	std::vector<char*> envp;
	for (auto& e: env_strings)
	{
		envp.push_back(e.data());
	}
	envp.push_back(nullptr);

	// Extra args are split on spaces here; there is no shell in between.
	std::vector<std::string> argv_strings = {plan.exe};
	argv_strings.insert(argv_strings.end(), plan.args.begin(), plan.args.end());
	for (size_t pos = 0; pos < plan.extra_args.size();)
	{
		size_t next = plan.extra_args.find(' ', pos);
		if (next == std::string::npos)
		{
			next = plan.extra_args.size();
		}
		if (next > pos)
		{
			argv_strings.push_back(plan.extra_args.substr(pos, next - pos));
		}
		pos = next + 1;
	}
	std::vector<char*> argv;
	for (auto& a: argv_strings)
	{
		argv.push_back(a.data());
	}
	argv.push_back(nullptr);

	int fds[2];
	if (pipe(fds) != 0)
	{
		*error = "pipe failed";
		return false;
	}
	pid_t pid = fork();
	if (pid == 0)
	{
#ifdef __linux__
		// Ends with the launcher, like the kill-on-close job on Windows.
		prctl(PR_SET_PDEATHSIG, SIGKILL);
#endif
		dup2(fds[1], 1);
		dup2(fds[1], 2);
		close(fds[0]);
		close(fds[1]);
		int nul = open("/dev/null", O_RDONLY);
		dup2(nul, 0);
		if (!plan.cwd.empty() && chdir(plan.cwd.c_str()) != 0)
		{
			_exit(126);
		}
		execve(plan.exe.c_str(), argv.data(), envp.data());
		_exit(127);
	}
	close(fds[1]);
	if (pid < 0)
	{
		close(fds[0]);
		*error = "fork failed";
		return false;
	}

	m_impl       = std::make_shared<Impl>();
	m_impl->pid  = pid;
	m_impl->read = fds[0];
	m_impl->running = true;

	// <root>/logs/last-run.log (set by make_launch_plan); cwd/last-run.log when the plan has no log path.
	fs::path log_path = plan.log_file.empty() ? utf8_path(plan.cwd) / "last-run.log" : utf8_path(plan.log_file);
	std::error_code log_ec;
	if (!log_path.parent_path().empty())
	{
		fs::create_directories(log_path.parent_path(), log_ec);
	}
	m_reader          = std::thread(
        [impl = m_impl, log_path, on_output = std::move(on_output), on_exit = std::move(on_exit)]
        {
            std::ofstream log(log_path, std::ios::binary | std::ios::trunc);
            char          buf[8192];
            ssize_t       got = 0;
            while ((got = ::read(impl->read, buf, sizeof(buf))) > 0)
            {
                log.write(buf, got);
                log.flush();
                on_output(std::string(buf, static_cast<size_t>(got)));
            }
            close(impl->read);
            int status = 0;
            waitpid(impl->pid, &status, 0);
            {
                std::lock_guard lock(impl->mutex);
                impl->pid = -1;
            }
            impl->running = false;
            on_exit(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        });
	return true;
}

void ChildProcess::Stop()
{
	if (m_impl)
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->pid > 0)
		{
			kill(m_impl->pid, SIGTERM);
		}
	}
}

#endif

bool ChildProcess::Running() const
{
	return m_impl && m_impl->running.load();
}

void ChildProcess::Wait()
{
	if (m_reader.joinable())
	{
		m_reader.join();
	}
}

ChildProcess::~ChildProcess()
{
	Wait();
}

} // namespace kyty_launcher
