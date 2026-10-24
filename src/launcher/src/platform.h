#pragma once

#include <string>
#include <vector>

namespace kyty_launcher {

// Opens a folder, file or link with the system's default handler (Explorer / Finder / xdg-open).
void open_in_system(const std::string& target);

// Runs a program without a shell and returns what it printed (stdout), trimmed; `ok` says whether
// it started and exited with 0. Used for the file dialogs of the desktop on macOS and Linux.
std::string run_capture(const std::vector<std::string>& args, bool* ok);

// A folder or file picked in the desktop's own dialog (osascript on macOS, zenity or kdialog on
// Linux); "" when cancelled or when no dialog program is installed.
std::string pick_path_dialog(const std::string& current, bool folder);

// The pixel size of the primary monitor (Win32 on Windows, CoreGraphics on macOS, `xrandr` on Linux).
// False when it can't be told (for example on Wayland without XWayland); the outputs are then unchanged.
bool detect_primary_monitor(int* width, int* height);

// STEP6:6e: the device pickers of the settings page.
// Names of the system's audio input devices, in the system's order (Windows: waveInGetDevCaps on winmm.dll,
// loaded on demand). Empty on macOS and Linux, and when nothing is found: the page then keeps its text field.
std::vector<std::string> list_audio_input_devices();

// Names of the Vulkan physical devices, in Vulkan's enumeration order (the emulator's --gpu index - 1 is the
// position in this list). vulkan-1.dll is loaded on demand; empty when Vulkan is unavailable, and on macOS
// and Linux. A device without a name is listed as "GPU <n>" so the indices stay right.
std::vector<std::string> list_gpu_devices();

} // namespace kyty_launcher
