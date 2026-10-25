// Controls page: the working copy of Settings::host_input_mapping, and the keyboard/mouse + XInput capture.
//
// The names and the tables are the Qt launcher's (src/launcher/src/inputMappingDialog.cpp): its key names and
// PAD_SOURCES ids are what the emulator reads from --keymap, so they are reproduced exactly.

#include "input_mapping.h"

#include "settings.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#endif

namespace kyty_launcher
{
namespace
{

constexpr double kDefaultSensitivity = 1.0;
constexpr double kMinSensitivity     = 0.1;
constexpr double kMaxSensitivity     = 5.0;

constexpr char kMouseSensitivity[]   = "MouseSensitivity=";
constexpr char kTouchParticleSkin[]  = "TouchParticleSkin=";
constexpr char kTouchCursorPrefix[]  = "TouchCursorImage=";
constexpr char kTouchCursorOff[]     = "TouchCursorImage=Off";
constexpr char kKeyboardDisabled[]   = "Keyboard=Disabled";
constexpr char kPadDisabled[]        = "Pad=Disabled";
constexpr char kPadPrefix[]          = "Pad:";

// Order of the sections on the page, as in upstream's InputMappingDialog.
enum Section
{
	Face,
	Dpad,
	Shoulders,
	Sticks,
	System,
	LeftStick,
	RightStick,
	Shortcuts,
};

constexpr const char* kSectionTitles[] = {
    "Face buttons",
    "D-pad",
    "Shoulders & triggers",
    "Stick clicks",
    "System",
    "Left stick",
    "Right stick",
    "In-game shortcuts (cycle)",
};

// pad_default is null for controls that have no gamepad equivalent (no pad row then).
struct ControlDef
{
	const char* id;
	const char* label;
	const char* keyboard_default; // "" = the keyboard row starts unbound
	const char* pad_default;
	Section     section;
};

constexpr ControlDef kControls[] = {
    {"Cross", "Cross", "J", "South", Face},
    {"Circle", "Circle", "L", "East", Face},
    {"Square", "Square", "K", "West", Face},
    {"Triangle", "Triangle", "I", "North", Face},
    {"Up", "Up", "Up", "DpadUp", Dpad},
    {"Down", "Down", "Down", "DpadDown", Dpad},
    {"Left", "Left", "Left", "DpadLeft", Dpad},
    {"Right", "Right", "Right", "DpadRight", Dpad},
    {"L1", "L1", "Q", "LeftShoulder", Shoulders},
    {"R1", "R1", "E", "RightShoulder", Shoulders},
    {"L2", "L2", "Z", "LeftTrigger", Shoulders},
    {"R2", "R2", "C", "RightTrigger", Shoulders},
    {"L3", "L3 (left stick click)", "Left Shift", "LeftStick", Sticks},
    {"R3", "R3 (right stick click)", "Left Ctrl", "RightStick", Sticks},
    {"Options", "Options", "Return", "Start", System},
    {"TouchPad", "Touch pad click", "Backspace", "Touchpad", System},
    {"TouchPadRight", "Touch pad right", "Tab", nullptr, System},
    {"GyroToggle", "Gyro emulation toggle", "", "", System},
    {"TouchPadMode", "Emulated touch pad mode", "F6", "", System},
    {"LeftStickUp", "Up", "W", nullptr, LeftStick},
    {"LeftStickDown", "Down", "S", nullptr, LeftStick},
    {"LeftStickLeft", "Left", "A", nullptr, LeftStick},
    {"LeftStickRight", "Right", "D", nullptr, LeftStick},
    {"RightStickUp", "Up", "T", nullptr, RightStick},
    {"RightStickDown", "Down", "G", nullptr, RightStick},
    {"RightStickLeft", "Left", "F", nullptr, RightStick},
    {"RightStickRight", "Right", "H", nullptr, RightStick},
    {"SpeakerVolume", "Speaker volume", "1", nullptr, Shortcuts},
    {"VibrationIntensity", "Vibration intensity", "2", nullptr, Shortcuts},
    {"TriggerEffectIntensity", "Trigger intensity", "3", nullptr, Shortcuts},
};

bool StartsWith(std::string_view text, std::string_view prefix)
{
	return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool EqualsIgnoreCase(const std::string& a, const std::string& b)
{
	if (a.size() != b.size())
	{
		return false;
	}
	for (size_t i = 0; i < a.size(); i++)
	{
		const auto ca = static_cast<unsigned char>(a[i]);
		const auto cb = static_cast<unsigned char>(b[i]);
		if (std::tolower(ca) != std::tolower(cb))
		{
			return false;
		}
	}
	return true;
}

// One decimal, as the emulator's MouseSensitivity is written ("1.5"): QString::number(v, 'f', 1). Integer
// formatting keeps the decimal point the config expects whatever the C locale says.
std::string OneDecimal(double v)
{
	const int tenths = static_cast<int>(std::lround(v * 10.0));
	char      buf[32];
	std::snprintf(buf, sizeof(buf), "%d.%d", tenths / 10, tenths % 10);
	return buf;
}

#ifdef _WIN32

// The XInput structures, so xinput.h (and its link library) is not needed: xinput1_4.dll is loaded at run time.
struct XInputGamepad
{
	WORD  wButtons;
	BYTE  bLeftTrigger;
	BYTE  bRightTrigger;
	SHORT sThumbLX;
	SHORT sThumbLY;
	SHORT sThumbRX;
	SHORT sThumbRY;
};

struct XInputState
{
	DWORD         dwPacketNumber;
	XInputGamepad Gamepad;
};

using XInputGetStateFn = DWORD(WINAPI*)(DWORD, XInputState*);

// xinput1_4.dll on Windows 8 and later, then the 1.3 / 9.1.0 redistributables older installs have. The library
// stays loaded: the poll holds the function pointer.
XInputGetStateFn LoadXInput()
{
	static const XInputGetStateFn fn = []
	{
		for (const wchar_t* name: {L"xinput1_4.dll", L"xinput1_3.dll", L"xinput9_1_0.dll"})
		{
			if (HMODULE lib = LoadLibraryW(name))
			{
				if (auto proc = reinterpret_cast<XInputGetStateFn>(GetProcAddress(lib, "XInputGetState")))
				{
					return proc;
				}
			}
		}
		return static_cast<XInputGetStateFn>(nullptr);
	}();
	return fn;
}

struct PadButtonBit
{
	WORD        mask;
	const char* id;
};

// The order a button is picked in when several new ones show up in the same poll.
constexpr PadButtonBit kPadButtonBits[] = {
    {0x1000, "South"},         // A
    {0x2000, "East"},          // B
    {0x4000, "West"},          // X
    {0x8000, "North"},         // Y
    {0x0100, "LeftShoulder"},  // LB
    {0x0200, "RightShoulder"}, // RB
    {0x0020, "Back"},
    {0x0010, "Start"},
    {0x0040, "LeftStick"},
    {0x0080, "RightStick"},
    {0x0001, "DpadUp"},
    {0x0002, "DpadDown"},
    {0x0004, "DpadLeft"},
    {0x0008, "DpadRight"},
};

// Trigger that counts as pressed (upstream's preview uses the same threshold).
constexpr BYTE kTriggerThreshold = 30;

// The virtual keys the poll watches and the name each one gets, exactly as the Qt dialog names them. Keypad
// Enter shares VK_RETURN with Return: GetAsyncKeyState has no extended-key flag, so it can't be told apart.
const std::vector<std::pair<int, std::string>>& CaptureKeys()
{
	static const std::vector<std::pair<int, std::string>> keys = []
	{
		std::vector<std::pair<int, std::string>> k;
		for (int c = 'A'; c <= 'Z'; c++)
		{
			k.emplace_back(c, std::string(1, static_cast<char>(c)));
		}
		for (int c = '0'; c <= '9'; c++)
		{
			k.emplace_back(c, std::string(1, static_cast<char>(c)));
		}
		for (int i = 1; i <= 24; i++)
		{
			k.emplace_back(VK_F1 + i - 1, "F" + std::to_string(i));
		}
		for (int i = 0; i <= 9; i++)
		{
			k.emplace_back(VK_NUMPAD0 + i, "Keypad " + std::to_string(i));
		}
		const std::pair<int, const char*> special[] = {
		    {VK_SPACE, "Space"},
		    {VK_RETURN, "Return"},
		    {VK_BACK, "Backspace"},
		    {VK_TAB, "Tab"},
		    {VK_LSHIFT, "Left Shift"},
		    {VK_RSHIFT, "Left Shift"},
		    {VK_SHIFT, "Left Shift"},
		    {VK_LCONTROL, "Left Ctrl"},
		    {VK_RCONTROL, "Left Ctrl"},
		    {VK_CONTROL, "Left Ctrl"},
		    {VK_LMENU, "Left Alt"},
		    {VK_RMENU, "Left Alt"},
		    {VK_MENU, "Left Alt"},
		    {VK_LWIN, "Left GUI"},
		    {VK_RWIN, "Left GUI"},
		    {VK_INSERT, "Insert"},
		    {VK_DELETE, "Delete"},
		    {VK_HOME, "Home"},
		    {VK_END, "End"},
		    {VK_PRIOR, "PageUp"},
		    {VK_NEXT, "PageDown"},
		    {VK_LEFT, "Left"},
		    {VK_RIGHT, "Right"},
		    {VK_UP, "Up"},
		    {VK_DOWN, "Down"},
		    {VK_CAPITAL, "CapsLock"},
		    {VK_NUMLOCK, "Numlock"},
		    {VK_SCROLL, "ScrollLock"},
		    {VK_PAUSE, "Pause"},
		    {VK_SNAPSHOT, "PrintScreen"},
		    {VK_DIVIDE, "Keypad /"},
		    {VK_MULTIPLY, "Keypad *"},
		    {VK_SUBTRACT, "Keypad -"},
		    {VK_ADD, "Keypad +"},
		    {VK_DECIMAL, "Keypad ."},
		    {VK_LBUTTON, "Mouse:Left"},
		    {VK_RBUTTON, "Mouse:Right"},
		    {VK_MBUTTON, "Mouse:Middle"},
		    {VK_XBUTTON1, "Mouse:X1"},
		    {VK_XBUTTON2, "Mouse:X2"},
		};
		for (const auto& [vk, name]: special)
		{
			k.emplace_back(vk, name);
		}
		return k;
	}();
	return keys;
}

bool IsReservedKey(int vk)
{
	return vk == VK_F1 || vk == VK_F2 || vk == VK_F7 || vk == VK_F8 || vk == VK_F11;
}

bool KeyDown(int vk)
{
	return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

// Millisecond clock for the arming delay.
long long NowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#endif // _WIN32

} // namespace

const std::vector<PadSourceOption>& InputMapping::Sources()
{
	static const std::vector<PadSourceOption> sources = {
	    {"South", "A / Cross"},
	    {"East", "B / Circle"},
	    {"West", "X / Square"},
	    {"North", "Y / Triangle"},
	    {"Back", "View / Back / Share"},
	    {"Guide", "Guide / Home"},
	    {"Start", "Menu / Start"},
	    {"LeftStick", "Left stick click"},
	    {"RightStick", "Right stick click"},
	    {"LeftShoulder", "LB / L1"},
	    {"RightShoulder", "RB / R1"},
	    {"DpadUp", "D-pad Up"},
	    {"DpadDown", "D-pad Down"},
	    {"DpadLeft", "D-pad Left"},
	    {"DpadRight", "D-pad Right"},
	    {"Misc1", "Share / Misc"},
	    {"RightPaddle1", "Right paddle 1"},
	    {"LeftPaddle1", "Left paddle 1"},
	    {"RightPaddle2", "Right paddle 2"},
	    {"LeftPaddle2", "Left paddle 2"},
	    {"Touchpad", "Touch pad click"},
	    {"LeftTrigger", "LT / L2"},
	    {"RightTrigger", "RT / R2"},
	};
	return sources;
}

std::string InputMapping::SourceLabel(const std::string& id)
{
	for (const auto& source: Sources())
	{
		if (source.id == id)
		{
			return source.label;
		}
	}
	return id;
}

const std::vector<std::string>& InputMapping::SkinIds()
{
	static const std::vector<std::string> ids = {"Classic", "BlueNeon", "Monochrome", "Crimson", "RedBlue"};
	return ids;
}

const std::vector<std::string>& InputMapping::SkinLabels()
{
	static const std::vector<std::string> labels = {"Classic", "Blue neon", "Monochrome", "Crimson", "Red/Blue"};
	return labels;
}

bool InputMapping::CaptureAvailable()
{
#ifdef _WIN32
	return true;
#else
	return false;
#endif
}

void InputMapping::Load(const Settings& s)
{
	m_keyboard.clear();
	m_pad.clear();
	m_custom_keyboard = false;
	m_custom_pad      = false;
	m_sensitivity     = kDefaultSensitivity;
	m_skin            = 0;
	m_show_cursor     = true;
	bool keyboard_disabled = false;
	bool pad_disabled      = false;
	for (const auto& entry: s.host_input_mapping)
	{
		if (entry == kKeyboardDisabled)
		{
			keyboard_disabled = true;
			continue;
		}
		if (entry == kPadDisabled)
		{
			pad_disabled = true;
			continue;
		}
		if (StartsWith(entry, kMouseSensitivity))
		{
			const std::string value = entry.substr(sizeof(kMouseSensitivity) - 1);
			char*             end   = nullptr;
			const double      v     = std::strtod(value.c_str(), &end);
			m_sensitivity = end != nullptr && end != value.c_str() ? std::clamp(v, kMinSensitivity, kMaxSensitivity) : kDefaultSensitivity;
			continue;
		}
		if (StartsWith(entry, kTouchParticleSkin))
		{
			const std::string id   = entry.substr(sizeof(kTouchParticleSkin) - 1);
			const auto&       ids  = SkinIds();
			const auto        find = std::find(ids.begin(), ids.end(), id);
			m_skin = find == ids.end() ? 0 : static_cast<int>(find - ids.begin());
			continue;
		}
		if (StartsWith(entry, kTouchCursorPrefix))
		{
			m_show_cursor = entry != kTouchCursorOff;
			continue;
		}
		if (StartsWith(entry, kPadPrefix))
		{
			const std::string body = entry.substr(sizeof(kPadPrefix) - 1);
			const auto        sep  = body.find('=');
			if (sep != std::string::npos && sep > 0 && sep + 1 < body.size())
			{
				AssignRaw(true, body.substr(0, sep), body.substr(sep + 1));
				m_custom_pad = true;
			}
			continue;
		}
		const auto sep = entry.find('=');
		if (sep != std::string::npos && sep > 0 && sep + 1 < entry.size())
		{
			AssignRaw(false, entry.substr(0, sep), entry.substr(sep + 1));
		}
	}
	// A tab is custom when its own entries are there; otherwise it shows the table defaults and Save leaves it
	// out of the list, so a user who never opens a tab never writes it.
	m_custom_keyboard = !m_keyboard.empty() || keyboard_disabled;
	m_custom_pad      = m_custom_pad || pad_disabled;
	if (!m_custom_keyboard)
	{
		FillDefaults(false);
	}
	if (!m_custom_pad)
	{
		FillDefaults(true);
	}
}

const std::string* InputMapping::Find(bool pad, const std::string& id) const
{
	const auto& map = pad ? m_pad : m_keyboard;
	const auto  it  = map.find(id);
	return it == map.end() ? nullptr : &it->second;
}

bool InputMapping::HasRow(bool pad, const std::string& id) const
{
	for (const auto& control: kControls)
	{
		if (id == control.id && (!pad || control.pad_default != nullptr))
		{
			return true;
		}
	}
	return false;
}

void InputMapping::FillDefaults(bool pad)
{
	auto& map = pad ? m_pad : m_keyboard;
	map.clear();
	for (const auto& control: kControls)
	{
		const char* value = pad ? control.pad_default : control.keyboard_default;
		if (value != nullptr && value[0] != '\0')
		{
			map[control.id] = value;
		}
	}
}

void InputMapping::AssignRaw(bool pad, const std::string& id, const std::string& binding)
{
	if (!HasRow(pad, id))
	{
		return;
	}
	auto& map = pad ? m_pad : m_keyboard;
	// One key or source triggers one control per tab: the same binding is taken from the control that had it.
	for (auto it = map.begin(); it != map.end();)
	{
		if (it->first != id && EqualsIgnoreCase(it->second, binding))
		{
			it = map.erase(it);
		}
		else
		{
			++it;
		}
	}
	if (binding.empty())
	{
		map.erase(id);
	}
	else
	{
		map[id] = binding;
	}
}

void InputMapping::Assign(bool pad, const std::string& id, const std::string& binding)
{
	if (!HasRow(pad, id))
	{
		return;
	}
	AssignRaw(pad, id, binding);
	if (pad)
	{
		m_custom_pad = true;
	}
	else
	{
		m_custom_keyboard = true;
	}
}

void InputMapping::Clear(bool pad, const std::string& id)
{
	if (!HasRow(pad, id))
	{
		return;
	}
	AssignRaw(pad, id, std::string());
	if (pad)
	{
		m_custom_pad = true;
	}
	else
	{
		m_custom_keyboard = true;
	}
}

void InputMapping::RestoreDefaults(bool pad)
{
	if (pad)
	{
		FillDefaults(true);
		m_custom_pad = false;
		return;
	}
	FillDefaults(false);
	// Upstream restores the mouse sensitivity with the keyboard tab.
	m_sensitivity     = kDefaultSensitivity;
	m_custom_keyboard = false;
}

std::vector<ControlBinding> InputMapping::Rows(bool pad) const
{
	std::vector<ControlBinding> rows;
	for (const auto& control: kControls)
	{
		if (pad && control.pad_default == nullptr)
		{
			continue;
		}
		const std::string id = control.id;
		ControlBinding    row;
		row.id            = id;
		row.label         = !pad && id == "TouchPad" ? "Touch pad left" : control.label;
		row.section       = kSectionTitles[static_cast<int>(control.section)];
		row.section_start = rows.empty() || rows.back().section != row.section;
		const std::string* binding = Find(pad, id);
		if (binding != nullptr && !binding->empty())
		{
			row.binding = pad ? SourceLabel(*binding) : *binding;
		}
		else
		{
			row.binding = "None";
		}
		rows.push_back(row);
	}
	return rows;
}

std::string InputMapping::RowLabel(bool pad, const std::string& id) const
{
	for (const auto& control: kControls)
	{
		if (id == control.id && (!pad || control.pad_default != nullptr))
		{
			return !pad && id == "TouchPad" ? "Touch pad left" : control.label;
		}
	}
	return id;
}

void InputMapping::SetTouchOptions(double sensitivity, int skin_index, bool show_cursor)
{
	// One decimal, like the spin box upstream: the saved value and the default test stay exact.
	m_sensitivity = std::round(std::clamp(sensitivity, kMinSensitivity, kMaxSensitivity) * 10.0) / 10.0;
	m_skin        = std::clamp(skin_index, 0, static_cast<int>(SkinIds().size()) - 1);
	m_show_cursor = show_cursor;
}

std::vector<std::string> InputMapping::Mapping() const
{
	std::vector<std::string> result;
	if (m_custom_keyboard)
	{
		for (const auto& control: kControls)
		{
			const std::string* binding = Find(false, control.id);
			if (binding != nullptr && !binding->empty())
			{
				result.push_back(std::string(control.id) + "=" + *binding);
			}
		}
		if (result.empty())
		{
			result.push_back(kKeyboardDisabled);
		}
	}
	if (m_custom_pad)
	{
		const size_t start = result.size();
		for (const auto& control: kControls)
		{
			const std::string* source = Find(true, control.id);
			if (source != nullptr && !source->empty())
			{
				result.push_back(std::string(kPadPrefix) + control.id + "=" + *source);
			}
		}
		if (result.size() == start)
		{
			result.push_back(kPadDisabled);
		}
	}
	if (m_sensitivity != kDefaultSensitivity)
	{
		result.push_back(std::string(kMouseSensitivity) + OneDecimal(m_sensitivity));
	}
	if (m_skin > 0 && m_skin < static_cast<int>(SkinIds().size()))
	{
		result.push_back(std::string(kTouchParticleSkin) + SkinIds()[static_cast<size_t>(m_skin)]);
	}
	if (!m_show_cursor)
	{
		result.push_back(kTouchCursorOff);
	}
	return result;
}

void InputMapping::SaveMapping(Settings& s) const
{
	s.host_input_mapping = Mapping();
}

bool InputMapping::BeginCapture(bool pad, const std::string& id, std::string* error)
{
	m_capturing = false;
	if (!HasRow(pad, id))
	{
		if (error != nullptr)
		{
			*error = "That control is not on this tab.";
		}
		return false;
	}
#ifndef _WIN32
	if (error != nullptr)
	{
		*error = "Capture is not available on this platform";
	}
	return false;
#else
	m_capture_pad = pad;
	m_capture_id  = id;
	m_arm_ms      = NowMs() + 200;
	m_key_down.clear();
	if (pad)
	{
		// Whatever is held now counts only after it has been released and pressed again.
		m_pad_seen    = false;
		m_pad_index   = -1;
		m_pad_buttons = 0;
		m_pad_lt      = false;
		m_pad_rt      = false;
	}
	else
	{
		for (const auto& key: CaptureKeys())
		{
			m_key_down[key.first] = KeyDown(key.first);
		}
		m_key_down[VK_ESCAPE] = KeyDown(VK_ESCAPE);
	}
	m_capturing = true;
	if (error != nullptr)
	{
		error->clear();
	}
	return true;
#endif
}

void InputMapping::CancelCapture()
{
	m_capturing = false;
	m_capture_id.clear();
	m_key_down.clear();
	m_pad_seen  = false;
	m_pad_index = -1;
}

CaptureHit InputMapping::PollCapture()
{
	CaptureHit hit;
	if (!m_capturing)
	{
		return hit;
	}
#ifdef _WIN32
	const bool armed = NowMs() >= m_arm_ms;
	if (m_capture_pad)
	{
		const auto get = LoadXInput();
		if (get == nullptr)
		{
			return hit;
		}
		XInputState state {};
		int         index = -1;
		for (DWORD i = 0; i < 4; i++)
		{
			if (get(i, &state) == ERROR_SUCCESS)
			{
				index = static_cast<int>(i);
				break;
			}
		}
		if (index < 0)
		{
			m_pad_seen  = false;
			m_pad_index = -1;
			return hit;
		}
		const bool lt = state.Gamepad.bLeftTrigger > kTriggerThreshold;
		const bool rt = state.Gamepad.bRightTrigger > kTriggerThreshold;
		if (!m_pad_seen || index != m_pad_index)
		{
			m_pad_seen    = true;
			m_pad_index   = index;
			m_pad_buttons = state.Gamepad.wButtons;
			m_pad_lt      = lt;
			m_pad_rt      = rt;
			return hit;
		}
		const WORD pressed = static_cast<WORD>(state.Gamepad.wButtons & ~m_pad_buttons);
		m_pad_buttons      = state.Gamepad.wButtons;
		const bool lt_edge = lt && !m_pad_lt;
		const bool rt_edge = rt && !m_pad_rt;
		m_pad_lt           = lt;
		m_pad_rt           = rt;
		if (!armed)
		{
			return hit;
		}
		for (const auto& button: kPadButtonBits)
		{
			if ((pressed & button.mask) != 0)
			{
				hit.kind    = CaptureHit::Kind::Pad;
				hit.name    = button.id;
				m_capturing = false;
				return hit;
			}
		}
		if (lt_edge)
		{
			hit.kind    = CaptureHit::Kind::Pad;
			hit.name    = "LeftTrigger";
			m_capturing = false;
			return hit;
		}
		if (rt_edge)
		{
			hit.kind    = CaptureHit::Kind::Pad;
			hit.name    = "RightTrigger";
			m_capturing = false;
			return hit;
		}
		return hit;
	}
	// Esc cancels the capture, not the page.
	{
		const bool down = KeyDown(VK_ESCAPE);
		const bool was  = m_key_down[VK_ESCAPE];
		m_key_down[VK_ESCAPE] = down;
		if (down && !was)
		{
			m_capturing = false;
			hit.kind    = CaptureHit::Kind::Cancelled;
			return hit;
		}
	}
	for (const auto& [vk, name]: CaptureKeys())
	{
		const bool down = KeyDown(vk);
		const bool was  = m_key_down[vk];
		m_key_down[vk]  = down;
		if (!down || was || !armed)
		{
			continue;
		}
		if (IsReservedKey(vk))
		{
			hit.kind = CaptureHit::Kind::Reserved;
			hit.name = "That key is reserved by the emulator.";
			return hit;
		}
		hit.kind    = CaptureHit::Kind::Key;
		hit.name    = name;
		m_capturing = false;
		return hit;
	}
#endif
	return hit;
}

std::string InputMapping::PadStatus() const
{
#ifdef _WIN32
	if (const auto get = LoadXInput())
	{
		for (DWORD i = 0; i < 4; i++)
		{
			XInputState state {};
			if (get(i, &state) == ERROR_SUCCESS)
			{
				return "Connected: Controller " + std::to_string(static_cast<int>(i) + 1);
			}
		}
	}
	return "No controller connected";
#else
	return "Capture is not available on this platform";
#endif
}

} // namespace kyty_launcher
