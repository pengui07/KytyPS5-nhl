#include "platform.h"

#include "settings.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#include <shellapi.h>
#else
#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#endif
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kyty_launcher {

namespace fs = std::filesystem;

#ifdef _WIN32

void open_in_system(const std::string& target)
{
	ShellExecuteW(nullptr, L"open", utf8_path(target).wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

std::string run_capture(const std::vector<std::string>&, bool* ok)
{
	if (ok != nullptr)
	{
		*ok = false;
	}
	return {};
}

bool detect_primary_monitor(int* width, int* height)
{
	// The real pixel size; GetSystemMetrics would report a scaled-down one for a process that isn't DPI aware.
	DEVMODEW mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) == FALSE || mode.dmPelsWidth == 0 || mode.dmPelsHeight == 0)
	{
		return false;
	}
	*width  = static_cast<int>(mode.dmPelsWidth);
	*height = static_cast<int>(mode.dmPelsHeight);
	return true;
}

std::string pick_path_dialog(const std::string&, bool)
{
	return {};
}

namespace {

// STEP6:6e: device enumeration for the settings page.
// winmm is loaded on demand: the launcher must not gain a link library for two calls (CMake is not ours).
using WaveInGetNumDevsFn  = UINT(__stdcall*)();
using WaveInGetDevCapsWFn = MMRESULT(__stdcall*)(UINT_PTR, WAVEINCAPSW*, UINT);

// Minimal Vulkan declarations: only the physical device names are needed, so no Vulkan headers and no
// vulkan-1.dll link library are used (the DLL is loaded on demand).
using VkInstance       = struct VkInstance_T*;
using VkPhysicalDevice = struct VkPhysicalDevice_T*;
using VkResult         = int32_t;
using VkFlags          = uint32_t;

struct VkApplicationInfo
{
	int32_t     sType;
	const void* pNext;
	const char* pApplicationName;
	uint32_t    applicationVersion;
	const char* pEngineName;
	uint32_t    engineVersion;
	uint32_t    apiVersion;
};

struct VkInstanceCreateInfo
{
	int32_t                  sType;
	const void*              pNext;
	VkFlags                  flags;
	const VkApplicationInfo* pApplicationInfo;
	uint32_t                 enabledLayerCount;
	const char* const*       ppEnabledLayerNames;
	uint32_t                 enabledExtensionCount;
	const char* const*       ppEnabledExtensionNames;
};

// deviceName sits at its real offset; the block behind it is only here because the driver writes the whole
// struct (limits, sparse properties), which must not run past this buffer.
struct alignas(8) VkPhysicalDeviceProperties
{
	uint32_t apiVersion;
	uint32_t driverVersion;
	uint32_t vendorID;
	uint32_t deviceID;
	int32_t  deviceType;
	char     deviceName[256];
	uint8_t  pipelineCacheUUID[16];
	uint8_t  rest[4096];
};

using PfnVkVoidFunction              = void(__stdcall*)();
using PfnGetInstanceProcAddr         = PfnVkVoidFunction(__stdcall*)(VkInstance, const char*);
using PfnCreateInstance              = VkResult(__stdcall*)(const VkInstanceCreateInfo*, const void*, VkInstance*);
using PfnEnumeratePhysicalDevices    = VkResult(__stdcall*)(VkInstance, uint32_t*, VkPhysicalDevice*);
using PfnGetPhysicalDeviceProperties = void(__stdcall*)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
using PfnDestroyInstance             = void(__stdcall*)(VkInstance, const void*);

constexpr VkResult kVkSuccess                  = 0;
constexpr int32_t  kVkStructureApplicationInfo = 0;         // VK_STRUCTURE_TYPE_APPLICATION_INFO
constexpr int32_t  kVkStructureInstanceCreate  = 1;         // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
constexpr uint32_t kVkApiVersion10             = 1u << 22;  // VK_MAKE_API_VERSION(0, 1, 0, 0)

template<typename Fn>
Fn LoadProc(HMODULE dll, const char* name)
{
	return reinterpret_cast<Fn>(GetProcAddress(dll, name));
}

} // namespace

std::vector<std::string> list_audio_input_devices()
{
	std::vector<std::string> out;
	HMODULE                  dll = LoadLibraryW(L"winmm.dll");
	if (dll == nullptr)
	{
		return out;
	}
	const auto count_devices = LoadProc<WaveInGetNumDevsFn>(dll, "waveInGetNumDevs");
	const auto device_caps   = LoadProc<WaveInGetDevCapsWFn>(dll, "waveInGetDevCapsW");
	if (count_devices != nullptr && device_caps != nullptr)
	{
		const UINT count = count_devices();
		out.reserve(count);
		for (UINT i = 0; i < count; i++)
		{
			WAVEINCAPSW caps {};
			if (device_caps(static_cast<UINT_PTR>(i), &caps, sizeof(caps)) != MMSYSERR_NOERROR)
			{
				continue;
			}
			caps.szPname[MAXPNAMELEN - 1] = L'\0';
			const std::string name        = path_utf8(fs::path(caps.szPname));
			// Two devices can carry the same name; the picker lists each name once.
			if (!name.empty() && std::find(out.begin(), out.end(), name) == out.end())
			{
				out.push_back(name);
			}
		}
	}
	FreeLibrary(dll);
	return out;
}

std::vector<std::string> list_gpu_devices()
{
	std::vector<std::string> out;
	HMODULE                  dll = LoadLibraryW(L"vulkan-1.dll");
	if (dll == nullptr)
	{
		return out; // no Vulkan runtime: the page keeps its text field
	}
	const auto get_proc = LoadProc<PfnGetInstanceProcAddr>(dll, "vkGetInstanceProcAddr");
	if (get_proc != nullptr)
	{
		const auto create_instance = reinterpret_cast<PfnCreateInstance>(get_proc(nullptr, "vkCreateInstance"));
		if (create_instance != nullptr)
		{
			VkApplicationInfo app {};
			app.sType              = kVkStructureApplicationInfo;
			app.pApplicationName   = "Kyty launcher";
			app.applicationVersion = 1;
			app.pEngineName        = "Kyty launcher";
			app.engineVersion      = 1;
			app.apiVersion         = kVkApiVersion10;
			VkInstanceCreateInfo info {};
			info.sType            = kVkStructureInstanceCreate;
			info.pApplicationInfo = &app;
			VkInstance instance   = nullptr;
			if (create_instance(&info, nullptr, &instance) == kVkSuccess && instance != nullptr)
			{
				const auto enumerate  = reinterpret_cast<PfnEnumeratePhysicalDevices>(get_proc(instance, "vkEnumeratePhysicalDevices"));
				const auto properties = reinterpret_cast<PfnGetPhysicalDeviceProperties>(get_proc(instance, "vkGetPhysicalDeviceProperties"));
				const auto destroy    = reinterpret_cast<PfnDestroyInstance>(get_proc(instance, "vkDestroyInstance"));
				if (enumerate != nullptr && properties != nullptr)
				{
					uint32_t count = 0;
					if (enumerate(instance, &count, nullptr) == kVkSuccess && count > 0)
					{
						std::vector<VkPhysicalDevice> devices(count, nullptr);
						if (enumerate(instance, &count, devices.data()) == kVkSuccess)
						{
							devices.resize(count);
							out.reserve(count);
							for (uint32_t i = 0; i < count; i++)
							{
								VkPhysicalDeviceProperties props {};
								properties(devices[i], &props);
								props.deviceName[sizeof(props.deviceName) - 1] = '\0';
								std::string name(props.deviceName);
								// The list position is the emulator's gpu index + 1, so keep empty names in it.
								if (name.empty())
								{
									name = "GPU " + std::to_string(i);
								}
								out.push_back(std::move(name));
							}
						}
					}
				}
				if (destroy != nullptr)
				{
					destroy(instance, nullptr);
				}
			}
		}
	}
	FreeLibrary(dll);
	return out;
}

#else

namespace {

std::string Trim(std::string s)
{
	while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0)
	{
		s.pop_back();
	}
	size_t b = 0;
	while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0)
	{
		b++;
	}
	return s.substr(b);
}

// Whether `name` is an executable in one of the PATH folders.
bool ProgramExists(const std::string& name)
{
	const char* path = std::getenv("PATH");
	if (path == nullptr)
	{
		return false;
	}
	const std::string all = path;
	for (size_t pos = 0; pos <= all.size();)
	{
		size_t end = all.find(':', pos);
		if (end == std::string::npos)
		{
			end = all.size();
		}
		const std::string dir = all.substr(pos, end - pos);
		if (!dir.empty() && access((dir + "/" + name).c_str(), X_OK) == 0)
		{
			return true;
		}
		pos = end + 1;
	}
	return false;
}

} // namespace

std::string run_capture(const std::vector<std::string>& args, bool* ok)
{
	if (ok != nullptr)
	{
		*ok = false;
	}
	if (args.empty())
	{
		return {};
	}
	std::vector<std::string> copy = args;
	std::vector<char*>       argv;
	for (auto& a: copy)
	{
		argv.push_back(a.data());
	}
	argv.push_back(nullptr);

	int fds[2];
	if (pipe(fds) != 0)
	{
		return {};
	}
	pid_t pid = fork();
	if (pid < 0)
	{
		close(fds[0]);
		close(fds[1]);
		return {};
	}
	if (pid == 0)
	{
		dup2(fds[1], 1);
		close(fds[0]);
		close(fds[1]);
		execvp(argv[0], argv.data());
		_exit(127);
	}
	close(fds[1]);
	std::string out;
	char        buf[4096];
	ssize_t     got = 0;
	while ((got = read(fds[0], buf, sizeof(buf))) > 0)
	{
		out.append(buf, static_cast<size_t>(got));
	}
	close(fds[0]);
	int status = 0;
	waitpid(pid, &status, 0);
	if (ok != nullptr)
	{
		*ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
	}
	return Trim(out);
}

void open_in_system(const std::string& target)
{
	// Detached: the opener may run for as long as the user keeps the document open.
	pid_t pid = fork();
	if (pid == 0)
	{
		if (fork() != 0)
		{
			_exit(0);
		}
#ifdef __APPLE__
		execlp("open", "open", target.c_str(), static_cast<char*>(nullptr));
#else
		execlp("xdg-open", "xdg-open", target.c_str(), static_cast<char*>(nullptr));
#endif
		_exit(127);
	}
	if (pid > 0)
	{
		int status = 0;
		waitpid(pid, &status, 0);
	}
}

std::string pick_path_dialog(const std::string& current, bool folder)
{
	// Start in the folder of the current value when it exists.
	std::error_code ec;
	fs::path        start = utf8_path(current);
	if (!fs::is_directory(start, ec))
	{
		start = start.parent_path();
	}
	if (start.empty() || !fs::is_directory(start, ec))
	{
		const char* home = std::getenv("HOME");
		start            = home != nullptr ? home : "/";
	}
	const std::string dir = path_utf8(start);

	bool ok = false;
#ifdef __APPLE__
	const std::string script = std::string("POSIX path of (choose ") + (folder ? "folder" : "file") +
	                           " default location POSIX file \"" + dir + "\")";
	std::string picked = run_capture({"osascript", "-e", script}, &ok);
	if (ok && !picked.empty())
	{
		return picked;
	}
	return {};
#else
	if (ProgramExists("zenity"))
	{
		std::vector<std::string> zenity = {"zenity", "--file-selection", "--filename=" + dir + "/"};
		if (folder)
		{
			zenity.push_back("--directory");
		}
		return run_capture(zenity, &ok);
	}
	if (ProgramExists("kdialog"))
	{
		return run_capture({"kdialog", folder ? "--getexistingdirectory" : "--getopenfilename", dir}, &ok);
	}
	return {};
#endif
}

#ifndef __APPLE__
namespace {

// "2560x1440+0+0" (the geometry xrandr prints for an active output) -> 2560, 1440.
bool ParseGeometry(const std::string& token, int* width, int* height)
{
	int w = 0;
	int h = 0;
	int x = 0;
	int y = 0;
	char tail = 0;
	// The offsets must follow, so a mode list entry like "1920x1080" alone is not taken.
	if (std::sscanf(token.c_str(), "%dx%d+%d+%d%c", &w, &h, &x, &y, &tail) == 4 && w > 0 && h > 0)
	{
		*width  = w;
		*height = h;
		return true;
	}
	return false;
}

} // namespace
#endif

bool detect_primary_monitor(int* width, int* height)
{
#ifdef __APPLE__
	CGDisplayModeRef mode = CGDisplayCopyDisplayMode(CGMainDisplayID());
	if (mode == nullptr)
	{
		return false;
	}
	// Pixels, not the points of a scaled "looks like" mode.
	const size_t w = CGDisplayModeGetPixelWidth(mode);
	const size_t h = CGDisplayModeGetPixelHeight(mode);
	CGDisplayModeRelease(mode);
	if (w == 0 || h == 0)
	{
		return false;
	}
	*width  = static_cast<int>(w);
	*height = static_cast<int>(h);
	return true;
#else
	// X11 and XWayland. Wayland-only sessions without xrandr answer nothing, which is fine: no recommendation.
	if (!ProgramExists("xrandr"))
	{
		return false;
	}
	bool              ok  = false;
	const std::string out = run_capture({"xrandr", "--current"}, &ok);
	if (!ok)
	{
		return false;
	}
	bool found = false;
	for (size_t pos = 0; pos < out.size();)
	{
		size_t            eol  = out.find('\n', pos);
		const std::string line = out.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
		pos                    = eol == std::string::npos ? out.size() : eol + 1;
		if (line.find(" connected") == std::string::npos)
		{
			continue;
		}
		// "DP-1 connected primary 2560x1440+0+0 (...)" or "HDMI-1 connected 1920x1080+2560+0 (...)"
		const bool primary = line.find(" primary ") != std::string::npos;
		int        w       = 0;
		int        h       = 0;
		for (size_t start = 0; start < line.size();)
		{
			size_t end = line.find(' ', start);
			if (end == std::string::npos)
			{
				end = line.size();
			}
			if (ParseGeometry(line.substr(start, end - start), &w, &h))
			{
				break;
			}
			start = end + 1;
		}
		if (w == 0)
		{
			continue; // connected but not active
		}
		if (!found || primary)
		{
			*width  = w;
			*height = h;
			found   = true;
		}
		if (primary)
		{
			break;
		}
	}
	return found;
#endif
}

// STEP6:6e: no device enumeration outside Windows yet; the settings page keeps its text fields.
std::vector<std::string> list_audio_input_devices()
{
	return {};
}

std::vector<std::string> list_gpu_devices()
{
	return {};
}

#endif

} // namespace kyty_launcher
