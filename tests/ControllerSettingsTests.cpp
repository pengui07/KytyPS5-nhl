// Exercise the production controller with deterministic host output and time.
#include <SDL3/SDL.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "ControllerSettingsTests: %s\n", text);
		std::abort();
	}
}

struct Rumble {
	Uint16 large, small;
	Uint32 duration;
	int    id = 0; // The host gamepad (or haptics controller) that received the request.
};
struct Led {
	int id;
	std::array<Uint8, 3> color;
};
std::vector<Led>                   leds;
Uint64                             now = 1000;
std::vector<Rumble>                rumble;
std::vector<Rumble>                haptics;
std::vector<std::array<Uint8, 32>> effects;
std::vector<int>                   effect_ids;
bool                               haptics_handles_rumble = false;
std::array<Rumble, 6>               pad_rumble {};
std::vector<int>                    closed_pads;
std::array<std::array<Uint8, 3>, 6> pad_colors {};
} // namespace

namespace Fake {
Uint64 GetTicks() {
	return now;
}
int PadId(SDL_Gamepad* pad) {
	return static_cast<int>(reinterpret_cast<uintptr_t>(pad));
}
SDL_Gamepad* GetGamepadFromID(SDL_JoystickID id) {
	return id >= 1 && id <= 5 ? reinterpret_cast<SDL_Gamepad*>(static_cast<uintptr_t>(id))
	                          : nullptr;
}
SDL_GamepadType GetGamepadType(SDL_Gamepad*) {
	return SDL_GAMEPAD_TYPE_PS5;
}
bool RumbleGamepad(SDL_Gamepad* pad, Uint16 large, Uint16 small, Uint32 duration) {
	rumble.push_back({large, small, duration, PadId(pad)});
	pad_rumble[reinterpret_cast<uintptr_t>(pad)] = rumble.back();
	return true;
}
bool SendGamepadEffect(SDL_Gamepad* pad, const void* data, int size) {
	Check(size == 32, "unexpected DualSense effect size");
	std::array<Uint8, 32> effect {};
	std::memcpy(effect.data(), data, effect.size());
	effects.push_back(effect);
	effect_ids.push_back(PadId(pad));
	return true;
}
bool SetGamepadLED(SDL_Gamepad* pad, Uint8 r, Uint8 g, Uint8 b) {
	leds.push_back({PadId(pad), {r, g, b}});
	pad_colors[reinterpret_cast<uintptr_t>(pad)] = {r, g, b};
	return true;
}
bool GamepadHasSensor(SDL_Gamepad*, SDL_SensorType) {
	return false;
}
bool SetGamepadSensorEnabled(SDL_Gamepad*, SDL_SensorType, bool) {
	return true;
}
void CloseGamepad(SDL_Gamepad* pad) {
	closed_pads.push_back(static_cast<int>(reinterpret_cast<uintptr_t>(pad)));
}
void Delay(Uint32) {}
} // namespace Fake

#define SDL_GetTicks                Fake::GetTicks
#define SDL_GetGamepadFromID        Fake::GetGamepadFromID
#define SDL_GetGamepadType          Fake::GetGamepadType
#define SDL_RumbleGamepad           Fake::RumbleGamepad
#define SDL_SendGamepadEffect       Fake::SendGamepadEffect
#define SDL_SetGamepadLED           Fake::SetGamepadLED
#define SDL_GamepadHasSensor        Fake::GamepadHasSensor
#define SDL_SetGamepadSensorEnabled Fake::SetGamepadSensorEnabled
#define SDL_CloseGamepad            Fake::CloseGamepad
#define SDL_Delay                   Fake::Delay
#include "libs/controller.cpp"
#undef SDL_GetTicks
#undef SDL_GetGamepadFromID
#undef SDL_GetGamepadType
#undef SDL_RumbleGamepad
#undef SDL_SendGamepadEffect
#undef SDL_SetGamepadLED
#undef SDL_GamepadHasSensor
#undef SDL_SetGamepadSensorEnabled
#undef SDL_CloseGamepad
#undef SDL_Delay

namespace Libs::Controller {
// No virtual headset in these tests (padVr.cpp is not linked here).
int PadVrPortHand(int) { return -1; }
int PadVrPort(int, int, int, bool) { return PAD_ERROR_INVALID_ARG; }
int PadVrCheckHandle(int) { return PAD_ERROR_INVALID_HANDLE; }
int PadVrClose(int) { return PAD_ERROR_INVALID_HANDLE; }
} // namespace Libs::Controller

namespace Libs::Controller::DualSenseHaptics {
bool SetVibration(int controller, uint8_t large_motor, uint8_t small_motor, uint32_t duration_ms) {
	haptics.push_back({large_motor, small_motor, duration_ms, controller});
	return haptics_handles_rumble;
}
void Shutdown() {}
} // namespace Libs::Controller::DualSenseHaptics

namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	return now * 1000;
}
} // namespace Libs::LibKernel

namespace Loader::Timer {
double GetTimeMs() {
	return static_cast<double>(now);
}
} // namespace Loader::Timer

namespace {
using namespace Libs::Controller;

void ClearOutputs() {
	leds.clear();
	rumble.clear();
	haptics.clear();
	effects.clear();
	effect_ids.clear();
}

struct Controller {
	Controller() {
		now                    = 1000;
		haptics_handles_rumble = false;
		Initialize();
		Check(PadInit() == 0 && PadOpen(Config::GetUserId(), 0, 0, nullptr) == 1,
		      "the standard controller port did not open");
		Connect(1);
		Check(GetSettingScale(Setting::SpeakerVolume) ==
		              Config::GetControllerSpeakerVolume() / 50.0f &&
		          GetSettingScale(Setting::VibrationIntensity) ==
		              Config::GetControllerVibrationIntensity() / 100.0f &&
		          GetSettingScale(Setting::TriggerEffectIntensity) == 1.0f,
		      "controller initialization retained old settings");
		ClearOutputs();
	}
	~Controller() { Shutdown(); }
};

DualSenseEffects LastEffect() {
	Check(!effects.empty(), "missing trigger output");
	DualSenseEffects effect {};
	std::memcpy(&effect, effects.back().data(), sizeof(effect));
	return effect;
}

int ZoneStrength(const uint8_t* effect, int zone) {
	const unsigned active = effect[1] | (effect[2] << 8u);
	const uint32_t packed = effect[3] | (effect[4] << 8u) | (effect[5] << 16u) |
	                        (static_cast<uint32_t>(effect[6]) << 24u);
	return (active & (1u << zone)) != 0 ? 1 + ((packed >> (3 * zone)) & 7u) : 0;
}

void SetRumble(uint8_t large, uint8_t small) {
	const PadVibrationParam param {large, small};
	Check(PadSetVibration(1, &param) == 0, "vibration request failed");
}

void TestPadPortOwnership() {
	Initialize();
	const int user = Config::GetUserId();
	constexpr auto not_initialized = static_cast<int32_t>(0x80920005u);
	constexpr auto already_opened = static_cast<int32_t>(0x80920004u);
	constexpr auto invalid_handle = static_cast<int32_t>(0x80920003u);
	constexpr auto no_handle = static_cast<int32_t>(0x80920008u);
	PadData data {};
	Check(PadOpen(user, 0, 0, nullptr) == not_initialized &&
	          PadGetHandle(user, 0, 0) == not_initialized &&
	          PadReadState(1, &data) == not_initialized,
	      "an uninitialized library exposed a controller port");
	Check(PadInit() == 0 && PadGetHandle(user, 0, 0) == no_handle,
	      "initializing the library opened a controller port");
	const int handle = PadOpen(user, 0, 0, nullptr);
	Check(handle == 1 && PadInit() == 0 && PadGetHandle(user, 0, 0) == handle,
	      "reinitializing the library lost the open port");
	for (int duplicate = 0; duplicate < 4; duplicate++) {
		const int rejected = PadOpen(user, 0, 0, nullptr);
		Check(rejected == already_opened,
		      "opening one pad five times exposed duplicate connected controllers");
		Check(PadClose(rejected) == invalid_handle && PadGetHandle(user, 0, 0) == handle,
		      "closing a rejected duplicate released the original owner");
	}
	Check(PadGetHandle(user + 1, 0, 0) == no_handle &&
	          PadGetHandle(user, 2, 0) == no_handle && PadGetHandle(user, 0, 1) == no_handle,
	      "a different port tuple aliased the open standard pad");
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, true);
	Check(PadReadState(handle, &data) == 0 && data.connected &&
	          data.buttons == PAD_BUTTON_CROSS,
	      "keyboard Cross did not reach the owned port");
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, false);
	Connect(1);
	SetButton(1, PAD_BUTTON_CROSS, true);
	Check(PadReadState(handle, &data) == 0 && data.buttons == PAD_BUTTON_CROSS,
	      "the physical controller did not share the owned port");
	Disconnect(1);
	Check(PadGetHandle(user, 0, 0) == handle && PadOpen(user, 0, 0, nullptr) == already_opened,
	      "host disconnection released guest port ownership");
	Check(PadClose(handle) == 0 && PadGetHandle(user, 0, 0) == no_handle &&
	          PadReadState(handle, &data) == invalid_handle &&
	          PadRead(handle, &data, 1) == invalid_handle && PadClose(handle) == invalid_handle,
	      "a closed port retained a usable handle");
	Check(PadOpen(user, 0, 0, nullptr) == handle && PadReadState(handle, &data) == 0,
	      "a closed port could not reopen");
	Shutdown();
}

int PlayerUser(int player) {
	return GetPlayerUserId(player);
}

PadData ReadPlayer(int handle) {
	PadData data {};
	Check(PadReadState(handle, &data) == OK, "a player pad handle could not read state");
	return data;
}

void TestTwoPlayerInputIsolation() {
	Controller controller;
	Connect(2);
	Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "Player2 did not open its own pad handle");

	SetButton(1, PAD_BUTTON_CROSS, true);
	SetButton(2, PAD_BUTTON_CIRCLE, true);
	SetAxis(2, Axis::LeftX, 10);
	Check(ReadPlayer(1).buttons == PAD_BUTTON_CROSS && ReadPlayer(2).buttons == PAD_BUTTON_CIRCLE,
	      "two host controllers did not retain independent button state");
	Check(ReadPlayer(1).left_stick_x == 128 && ReadPlayer(2).left_stick_x == 10,
	      "one controller's axis reached another player");

	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_OPTIONS, true);
	SetTouchPad(HOST_INPUT_CONTROLLER_ID, 0, true, 0.5f, 0.5f);
	Check(ReadPlayer(1).buttons == (PAD_BUTTON_CROSS | PAD_BUTTON_OPTIONS | PAD_BUTTON_TOUCH_PAD) &&
	          ReadPlayer(2).buttons == PAD_BUTTON_CIRCLE && ReadPlayer(2).touch_data.touch_num == 0,
	      "keyboard input did not remain isolated to Player1");

	// Input from a device that no player owns is ignored.
	SetButton(5, PAD_BUTTON_SQUARE, true);
	Check(ReadPlayer(1).buttons == (PAD_BUTTON_CROSS | PAD_BUTTON_OPTIONS | PAD_BUTTON_TOUCH_PAD) &&
	          ReadPlayer(2).buttons == PAD_BUTTON_CIRCLE,
	      "an unassigned controller changed a player's state");

	Disconnect(1);
	const auto player_one = ReadPlayer(1);
	Check(ReadPlayer(2).buttons == PAD_BUTTON_CIRCLE && ReadPlayer(2).connected,
	      "disconnecting Player1's gamepad changed Player2's state");
	Check(player_one.connected && player_one.buttons == 0,
	      "Player1 did not fall back to the neutral keyboard pseudo-device");
}

void TestPlayerSlotCapacityAndReuse() {
	Controller controller;
	const int  user_two = PlayerUser(1), user_three = PlayerUser(2), user_four = PlayerUser(3);
	Check(GetPlayerControllerIdForUser(PlayerUser(0)) == 1 &&
	          GetPlayerControllerIdForUser(user_two) == -1 && !IsPlayerControllerConnected(1) &&
	          IsPlayerControllerConnected(0),
	      "Player1 lost its gamepad or keyboard connection, or Player2 started connected");

	int32_t user      = 0;
	bool    connected = false;
	Check(!PollPlayerConnectionEvent(&user, &connected), "a primary player queued a login event");

	Connect(2);
	Connect(3);
	Connect(4);
	Check(GetPlayerControllerIdForUser(user_two) == 2 && GetPlayerControllerIdForUser(user_three) == 3 &&
	          GetPlayerControllerIdForUser(user_four) == 4,
	      "controllers were not assigned to slots in connection order");

	Connect(5);
	Check(GetPlayerControllerIdForUser(PlayerUser(0)) == 1 && GetPlayerControllerIdForUser(user_four) == 4,
	      "a fifth controller displaced a player");
	SetButton(5, PAD_BUTTON_CROSS, true);
	Disconnect(5);
	for (int player = 0; player < PLAYER_SLOT_COUNT; ++player) {
		Check(player == 0 || IsPlayerControllerConnected(player), "ignoring a fifth controller cleared a slot");
	}
	Connect(2); // Duplicate connections do not take a second slot.
	Disconnect(99);

	Disconnect(3);
	Check(GetPlayerControllerIdForUser(user_two) == 2 && GetPlayerControllerIdForUser(user_three) == -1 &&
	          GetPlayerControllerIdForUser(user_four) == 4 && IsPlayerControllerConnected(1) &&
	          !IsPlayerControllerConnected(2) && IsPlayerControllerConnected(3),
	      "a disconnect cleared a slot other than its own");

	Connect(5);
	Check(GetPlayerControllerIdForUser(user_three) == 5,
	      "a later controller did not fill the freed slot");

	// One login per secondary connection and one logout, in order.
	const struct {
		int32_t user_id;
		bool    connected;
	} expected[] = {{user_two, true}, {user_three, true}, {user_four, true}, {user_three, false}, {user_three, true}};
	for (const auto& event: expected) {
		Check(PollPlayerConnectionEvent(&user, &connected) && user == event.user_id &&
		          connected == event.connected,
		      "connection events were lost, repeated, or out of order");
	}
	Check(!PollPlayerConnectionEvent(&user, &connected), "extra connection events were queued");

	// Player one's pad can also come and go without events; the keyboard keeps the slot connected.
	Disconnect(1);
	Check(IsPlayerControllerConnected(0) && !PollPlayerConnectionEvent(&user, &connected),
	      "Player1's gamepad changed its login state");
	Connect(1);
	Check(GetPlayerControllerIdForUser(PlayerUser(0)) == 1, "Player1 did not reclaim its slot");
}

void TestPerPlayerPadPorts() {
	Controller controller;
	constexpr auto already_opened = static_cast<int32_t>(0x80920004u);
	constexpr auto invalid_arg    = static_cast<int32_t>(0x80920001u);
	constexpr auto invalid_handle = static_cast<int32_t>(0x80920003u);
	constexpr auto no_handle      = static_cast<int32_t>(0x80920008u);
	for (int player = 1; player < PLAYER_SLOT_COUNT; ++player) {
		Check(PadGetHandle(PlayerUser(player), 0, 0) == no_handle,
		      "an unopened player port exposed a handle");
		Check(PadOpen(PlayerUser(player), 0, 0, nullptr) == player + 1,
		      "a player did not receive the handle matching its slot");
		Check(PadOpen(PlayerUser(player), 0, 0, nullptr) == already_opened,
		      "a player port opened twice");
	}
	Check(PadClose(3) == OK && PadGetHandle(PlayerUser(2), 0, 0) == no_handle,
	      "closing a player's port did not release it");
	Check(PadGetHandle(PlayerUser(0), 0, 0) == 1 && PadGetHandle(PlayerUser(1), 0, 0) == 2 &&
	          PadGetHandle(PlayerUser(3), 0, 0) == 4,
	      "closing one player's port closed another's");
	PadData data {};
	Check(PadReadState(3, &data) == invalid_handle && PadRead(3, &data, 1) == invalid_handle &&
	          PadClose(3) == invalid_handle && ReadPlayer(2).connected == false,
	      "a closed player handle stayed usable");
	for (const int handle: {0, 5, -1, 1000}) {
		Check(PadReadState(handle, &data) == invalid_handle && PadClose(handle) == invalid_handle,
		      "an out-of-range handle aliased a player");
	}
	Check(PadOpen(12345, 0, 0, nullptr) == invalid_arg &&
	          PadOpen(PlayerUser(1), 1, 0, nullptr) == invalid_arg &&
	          PadOpen(PlayerUser(2), 0, 1, nullptr) == invalid_arg &&
	          PadGetHandle(12345, 0, 0) == no_handle,
	      "an invalid port tuple was accepted");
	Check(PadOpen(PlayerUser(2), 0, 0, nullptr) == 3, "a closed player port could not reopen");
	Check(PadGetHandle(PlayerUser(1), 0, 1) == no_handle && PadGetHandle(PlayerUser(1), 2, 0) == no_handle,
	      "a different tuple aliased a player's pad");
}

void TestWaitingControllerTakesFreedSlot() {
	Controller controller;
	Connect(2);
	Connect(3);
	Connect(4);
	Connect(5);
	Connect(5); // A repeated SDL notification must not queue the same device twice.
	Disconnect(2);
	Check(GetPlayerControllerIdForUser(PlayerUser(1)) == 5,
	      "an already-connected fifth controller did not take the freed Player2 slot");
	Check(GetPlayerControllerIdForUser(PlayerUser(0)) == 1 &&
	          GetPlayerControllerIdForUser(PlayerUser(2)) == 3 &&
	          GetPlayerControllerIdForUser(PlayerUser(3)) == 4,
	      "promoting the waiting controller changed another player's assignment");
	Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "promoted controller did not open Player2's pad");
	SetButton(5, PAD_BUTTON_CIRCLE, true);
	Check(ReadPlayer(2).buttons == PAD_BUTTON_CIRCLE,
	      "the promoted controller did not feed its new player's pad");
	Disconnect(5);
	Check(!IsPlayerControllerConnected(1), "a duplicate waiting entry resurrected a disconnected pad");
	Connect(2);
	Connect(5);
	Disconnect(5); // Removing a waiting device must remove it from the queue as well.
	Disconnect(2);
	Check(!IsPlayerControllerConnected(1), "an unplugged waiting controller took a free slot");
}

void TestPerPlayerReadHistory() {
	Controller controller;
	Connect(2);
	Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "Player2 did not open");
	SetButton(1, PAD_BUTTON_CROSS, true);
	SetButton(1, PAD_BUTTON_CROSS, false);
	SetButton(2, PAD_BUTTON_CIRCLE, true);
	SetButton(2, PAD_BUTTON_SQUARE, true);

	PadData history[8] {};
	Check(PadRead(1, history, 8) == 2 && history[0].buttons == PAD_BUTTON_CROSS && history[1].buttons == 0,
	      "Player1's history was not returned in order");
	Check(PadRead(1, history, 8) == 1 && history[0].buttons == 0,
	      "Player1's history was not drained");
	Check(PadRead(2, history, 8) == 2 && history[0].buttons == PAD_BUTTON_CIRCLE &&
	          history[1].buttons == (PAD_BUTTON_CIRCLE | PAD_BUTTON_SQUARE),
	      "draining Player1 consumed Player2's history");

	// A departed device's state never reaches the next device in that slot.
	Disconnect(2);
	PadData gone = ReadPlayer(2);
	Check(!gone.connected && gone.buttons == 0, "a disconnected slot kept the departed device's state");
	Check(PadRead(2, history, 8) == 1 && !history[0].connected && history[0].buttons == 0,
	      "a disconnected slot replayed the departed device's history");
	Connect(3);
	PadData fresh = ReadPlayer(2);
	Check(fresh.connected && fresh.buttons == 0 && fresh.left_stick_x == 128,
	      "a new device inherited the previous device's state");
	Check(PadRead(2, history, 8) == 1 && history[0].buttons == 0,
	      "a new device inherited the previous device's history");
	SetButton(2, PAD_BUTTON_CROSS, true); // The old device is no longer assigned.
	Check(ReadPlayer(2).buttons == 0, "a departed device still drives its old slot");
	SetButton(3, PAD_BUTTON_L1, true);
	Check(ReadPlayer(2).buttons == PAD_BUTTON_L1 && ReadPlayer(1).buttons == 0,
	      "the replacement device did not drive only its slot");
}

void TestPerPlayerOutputIsolation() {
	Controller controller;
	Connect(2);
	Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "Player2 did not open");
	const auto expect_only = [](const std::vector<Rumble>& requests, int id, const char* message) {
		Check(!requests.empty(), message);
		for (const auto& request: requests) {
			Check(request.id == id, message);
		}
	};

	const PadVibrationParam player_two {200, 100};
	ClearOutputs();
	Check(PadSetVibration(2, &player_two) == OK, "Player2 vibration failed");
	expect_only(rumble, 2, "Player2's vibration reached another device");

	ClearOutputs();
	const PadVibrationParam player_one {50, 25};
	Check(PadSetVibration(1, &player_one) == OK, "Player1 vibration failed");
	expect_only(rumble, 1, "Player1's vibration reached another device");

	PadTriggerEffectParam trigger {};
	trigger.trigger_mask       = 1;
	trigger.command[0].mode    = 1;
	trigger.command[0].data[1] = 8;
	ClearOutputs();
	Check(PadSetTriggerEffect(2, &trigger) == OK && !effect_ids.empty(), "Player2 trigger request failed");
	for (const int id: effect_ids) {
		Check(id == 2, "Player2's trigger effect reached another device");
	}

	// Cycling the global level re-sends each player's own cached request to its own device.
	ClearOutputs();
	CycleSetting(Setting::VibrationIntensity);
	bool saw_one = false, saw_two = false;
	for (const auto& request: rumble) {
		saw_one |= request.id == 1;
		saw_two |= request.id == 2;
	}
	Check(saw_one && saw_two, "a level change did not reach every player's controller");

	// Output through the haptics path keeps its controller too.
	haptics_handles_rumble = true;
	ClearOutputs();
	Check(PadSetVibration(2, &player_two) == OK, "Player2 haptics vibration failed");
	expect_only(haptics, 2, "Player2's haptics vibration was routed to another device");
	Check(rumble.empty(), "haptics-handled vibration also used plain rumble");

	// Without a physical controller (keyboard only) a player's output has nowhere to go.
	Disconnect(1);
	ClearOutputs();
	haptics_handles_rumble = false;
	Check(PadSetVibration(1, &player_one) == OK && PadSetTriggerEffect(1, &trigger) == OK,
	      "keyboard-only player output failed");
	Check(rumble.empty() && effects.empty(), "keyboard input received controller output");
	Disconnect(2);
	ClearOutputs();
	Check(PadSetVibration(2, &player_two) == OK && rumble.empty(),
	      "a disconnected player's output reached a device");
}

void TestVirtualUserIdAllocation() {
	struct Case {
		int32_t configured;
		int32_t expected[3];
	};
	const Case cases[] = {{1000, {1001, 1002, 1003}},
	                      {1001, {1002, 1003, 1004}},
	                      {2147483647, {1001, 1002, 1003}},
	                      {0, {1001, 1002, 1003}}};
	for (const auto& test: cases) {
		Config::ConfigOptions options;
		options.user_id = test.configured;
		Config::Load(options);
		for (int round = 0; round < 2; ++round) { // Deterministic across initializations.
			Controller controller;
			Check(PlayerUser(0) == test.configured, "Player1 lost the configured user ID");
			for (int player = 1; player < PLAYER_SLOT_COUNT; ++player) {
				Check(PlayerUser(player) == test.expected[player - 1],
				      "a virtual user received an unexpected ID");
			}
			for (int a = 0; a < PLAYER_SLOT_COUNT; ++a) {
				Check(PlayerUser(a) >= 0 && PlayerUser(a) != 0xfe && PlayerUser(a) != 0xff,
				      "a player received an invalid or reserved user ID");
				for (int b = a + 1; b < PLAYER_SLOT_COUNT; ++b) {
					Check(PlayerUser(a) != PlayerUser(b), "two players share a user ID");
				}
			}
		}
	}
	Check(PlayerUser(-1) == -1 && PlayerUser(PLAYER_SLOT_COUNT) == -1, "an invalid slot returned a user");
	Config::Load(Config::ConfigOptions {});
}

void TestPlayerLightBarColors() {
	// Startup colors identify slots; each guest light bar write can then select its own color.
	Config::ConfigOptions options;
	options.controller_color = Config::ControllerColor {12, 34, 56};
	Config::Load(options);
	Initialize();
	Check(PadInit() == OK, "pad initialization failed");
	ClearOutputs();
	const std::array<Config::ControllerColor, 4> expected {{
	    {12, 34, 56}, {255, 0, 0}, {0, 255, 0}, {255, 32, 160}
	}};
	for (int player = 0; player < 4; ++player) {
		Connect(player + 1);
		Check(!leds.empty() && leds.back().id == player + 1 && leds.back().color == expected[player],
		      "connecting a controller did not apply its own player's color");
		Check(PadOpen(PlayerUser(player), 0, 0, nullptr) == player + 1, "player light bar port did not open");
		const PadLightBarParam game_color {static_cast<uint8_t>(player + 1), 2, 3, 0};
		const Config::ControllerColor wanted = player == 0 ? expected[0]
		    : Config::ControllerColor {static_cast<uint8_t>(player + 1), 2, 3};
		const auto before = leds.size();
		Check(PadSetLightBar(player + 1, &game_color) == OK &&
		          leds.size() == before + 1 && leds.back().id == player + 1 && leds.back().color == wanted,
		      "the game's light bar request was blocked or reached another player's device");
	}
	Disconnect(2);
	ClearOutputs();
	Connect(5);
	Check(leds.size() == 1 && leds.back().id == 5 && leds.back().color == Config::ControllerColor {2, 2, 3},
	      "a replacement controller did not restore Player2's last game-set color");
	Shutdown();
	Config::Load(Config::ConfigOptions {});
	{
		Controller controller;
		// NHL uses its own order: Player1 red and Player2 blue must override the startup palette.
		Connect(2);
		Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "Player2 did not open its light bar port");
		ClearOutputs();
		const PadLightBarParam game_color {255, 0, 0, 0};
		Check(PadSetLightBar(1, &game_color) == OK && leds.back().id == 1 &&
		          leds.back().color == Config::ControllerColor {255, 0, 0},
		      "Player1's startup palette blocked the game's red light bar");
		const PadLightBarParam second_color {0, 0, 255, 0};
		Check(PadSetLightBar(2, &second_color) == OK && leds.size() == 2 &&
		          leds.back().id == 2 && leds.back().color == Config::ControllerColor {0, 0, 255},
		      "Player2's startup palette blocked the game's blue light bar");
	}
}

void TestGameLightBarRestoredOnReconnect() {
	Controller controller;
	Connect(2);
	Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "Player2's pad did not open");
	const PadLightBarParam red {255, 0, 0, 0}, blue {0, 0, 255, 0}, off {0, 0, 0, 0};
	Check(PadSetLightBar(1, &red) == OK && PadSetLightBar(2, &blue) == OK,
	      "game light bar requests failed");
	Disconnect(2);
	Check(PadClose(2) == OK, "disconnected player's pad did not close");
	ClearOutputs();
	Connect(3);
	Check(leds.size() == 1 && leds.back().id == 3 && leds.back().color == Config::ControllerColor {0, 0, 255},
	      "Player2 reconnect restored the startup palette instead of the game's blue");
	Check(PadOpen(PlayerUser(1), 0, 0, nullptr) == 2, "reconnected pad did not reopen");
	Disconnect(1);
	ClearOutputs();
	Connect(4);
	Check(leds.size() == 1 && leds.back().id == 4 && leds.back().color == Config::ControllerColor {255, 0, 0},
	      "Player1 reconnect lost the game color or changed another player's LED");
	Check(PadSetLightBar(2, &off) == OK, "game could not turn off Player2's light bar");
	Disconnect(3);
	ClearOutputs();
	Connect(3);
	Check(leds.size() == 1 && leds.back().color == Config::ControllerColor {0, 0, 0},
	      "reconnect treated a game-requested off light bar as an unset color");
	Disconnect(3);
	ClearOutputs();
	const PadLightBarParam absent_color {17, 28, 39, 0};
	Check(PadSetLightBar(2, &absent_color) == OK && leds.empty(),
	      "setting a disconnected player's color touched another device");
	Connect(5);
	Check(leds.size() == 1 && leds.back().id == 5 && leds.back().color == Config::ControllerColor {17, 28, 39},
	      "reconnect lost the latest color requested while the controller was absent");
}

void TestSettingCycles() {
	Controller controller;
	for (auto setting:
	     {Setting::SpeakerVolume, Setting::VibrationIntensity, Setting::TriggerEffectIntensity}) {
		Check(GetSettingScale(setting) == 1.0f, "initial setting is not strong");
		const std::vector<float> levels = setting == Setting::SpeakerVolume
		                                      ? std::vector<float> {0.0f, 0.02f, 0.12f, 0.42f, 1.0f}
		                                      : std::vector<float> {0.0f, 0.33f, 0.66f, 1.0f};
		for (int cycle = 0; cycle < 2; ++cycle) {
			for (float level: levels) {
				CycleSetting(setting);
				Check(GetSettingScale(setting) == level, "setting did not cycle or wrap");
			}
		}
	}
	CycleSetting(Setting::SpeakerVolume);
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
}

void TestGlobalControllerLevels() {
	Config::ConfigOptions options;
	options.controller_speaker_volume      = 25;
	options.controller_vibration_intensity = 25;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.5f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.25f &&
		          GetSettingScale(Setting::TriggerEffectIntensity) == 1.0f,
		      "global controller levels did not scale their outputs independently");
		SetRumble(200, 100);
		Check(rumble.back().large == 50 * 257 && rumble.back().small == 25 * 257,
		      "global vibration level did not scale rumble");
		CycleSetting(Setting::SpeakerVolume);
		CycleSetting(Setting::VibrationIntensity);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0f,
		      "cycle hotkeys did not mute globally scaled outputs");
		CycleSetting(Setting::SpeakerVolume);
		CycleSetting(Setting::VibrationIntensity);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.01f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0825f,
		      "cycle hotkeys did not multiply the global controller levels");
		Check(rumble.back().large == 17 * 257 && rumble.back().small == 8 * 257,
		      "cycling vibration did not apply the combined intensity to cached rumble");
	}
	options.controller_speaker_volume = 50;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 1.0f,
		      "speaker midpoint did not preserve the previous default level");
	}
	options.controller_speaker_volume = 100;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 2.0f,
		      "speaker maximum did not add 6 dB of gain");
		CycleSetting(Setting::SpeakerVolume);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f,
		      "speaker maximum prevented the hotkey from muting");
	}
	options.controller_speaker_volume      = 0;
	options.controller_vibration_intensity = 0;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0f,
		      "zero global controller levels did not mute outputs");
		SetRumble(200, 100);
		Check(rumble.back().large == 0 && rumble.back().small == 0,
		      "zero global vibration level did not mute motors");
	}
	Config::Load(Config::ConfigOptions {});
}

void TestVibrationLifetime() {
	Controller controller;
	SetRumble(255, 1);
	Check(rumble.size() == 1 && rumble.back().large == 65535 && rumble.back().small == 257 &&
	          rumble.back().duration == 65535,
	      "full vibration changed");
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.back().large == 0 && rumble.back().small == 0, "muting did not stop vibration");
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.back().large == 84 * 257 && rumble.back().small == 257 &&
	          rumble.back().duration == 65335,
	      "changing intensity lost a small motor or extended the vibration deadline");
	Check(effects.empty(), "vibration intensity resent triggers");
	now              = 1000 + 65535;
	const auto calls = rumble.size();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.size() == calls || (rumble.back().large == 0 && rumble.back().small == 0),
	      "expired vibration was resurrected");
	SetRumble(0, 0);
	const auto stopped = rumble.size();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.size() == stopped || (rumble.back().large == 0 && rumble.back().small == 0),
	      "explicitly stopped vibration was resurrected");
}

void TestMaskedTriggersAndValidation() {
	Controller            controller;
	PadTriggerEffectParam param {};
	param.trigger_mask       = 3;
	param.command[0].mode    = 1;
	param.command[0].data[0] = 2;
	param.command[0].data[1] = 8;
	param.command[1].mode    = 2;
	param.command[1].data[0] = 2;
	param.command[1].data[1] = 7;
	param.command[1].data[2] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0, "initial trigger request failed");
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(LastEffect().left_trigger[0] == 5 && LastEffect().right_trigger[0] == 5,
	      "muting did not disable both triggers");
	for (uint32_t mode = 1; mode <= 7; ++mode) {
		auto invalid            = param;
		invalid.command[1]      = {};
		invalid.command[1].mode = mode;
		std::memset(invalid.command[1].data, 255, sizeof(invalid.command[1].data));
		const auto calls = effects.size();
		Check(PadSetTriggerEffect(1, &invalid) == PAD_ERROR_INVALID_ARG && effects.size() == calls,
		      "muting bypassed trigger validation or sent a partial invalid request");
	}
	param.trigger_mask = 4;
	Check(PadSetTriggerEffect(1, &param) == PAD_ERROR_INVALID_ARG, "invalid trigger mask accepted");
	param.trigger_mask       = 1;
	param.command[0].data[1] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0 && LastEffect().enable_bits == 8,
	      "left-only update touched the right trigger");
	CycleSetting(Setting::TriggerEffectIntensity);
	auto effect = LastEffect();
	Check(effect.enable_bits == 12 && effect.left_trigger[0] == 0x21 &&
	          ZoneStrength(effect.left_trigger, 1) == 0 &&
	          ZoneStrength(effect.left_trigger, 2) == 2 && effect.right_trigger[0] == 0x25 &&
	          effect.right_trigger[1] == 0x84 && effect.right_trigger[3] == 1,
	      "masked updates lost a cached trigger or failed to scale its strength");
	CycleSetting(Setting::TriggerEffectIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	effect = LastEffect();
	Check(ZoneStrength(effect.left_trigger, 2) == 6 && effect.right_trigger[3] == 5,
	      "restoring strong intensity did not recover original trigger strengths");
	Check(rumble.empty() && haptics.empty(), "trigger intensity resent vibration");
}

void TestIndependentOutputsAndPadSwitch() {
	Controller controller;
	haptics_handles_rumble = true;
	SetRumble(200, 100);
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.empty() && haptics.back().large == 66 && haptics.back().small == 33 &&
	          haptics.back().duration == 65435,
	      "haptics rumble did not receive scaled motors and remaining duration");
	PadTriggerEffectParam param {};
	param.trigger_mask       = 1;
	param.command[0].mode    = 1;
	param.command[0].data[1] = 8;
	Check(PadSetTriggerEffect(1, &param) == 0, "trigger request failed");
	const auto vibration_calls = haptics.size();
	const auto trigger_calls   = effects.size();
	CycleSetting(Setting::SpeakerVolume);
	Check(haptics.size() == vibration_calls && effects.size() == trigger_calls,
	      "speaker volume resent controller effects");
	Connect(2);
	Disconnect(1);
	Check(GetPlayerControllerIdForUser(PlayerUser(0)) == -1 &&
	          GetPlayerControllerIdForUser(PlayerUser(1)) == 2,
	      "the second controller did not become Player2");
	Check(GetActiveControllerId() == HOST_INPUT_CONTROLLER_ID,
	      "disconnecting player 1 stole player 2's pad");
	Disconnect(2);
	Connect(2);
	Check(GetActiveControllerId() == 2, "active controller did not change");
	ClearOutputs();
	haptics.clear();
	effects.clear();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(effects.empty(), "previous controller's trigger effects leaked to new pad");
	for (const auto& request: haptics) {
		Check(request.large == 0 && request.small == 0,
		      "previous controller's vibration leaked to new pad");
	}
	SetRumble(200, 100);
	Check(PadSetTriggerEffect(1, &param) == 0, "replacement pad trigger request failed");
	EmergencyShutdown();
	Check(GetPlayerControllerIdForUser(PlayerUser(1)) == -1 && !IsPlayerControllerConnected(1),
	      "released controller remained assigned");
	ClearOutputs();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(effects.empty(), "released controller received a trigger request");
	for (const auto& request: haptics) {
		Check(request.large == 0 && request.small == 0,
		      "released controller retained a cached vibration");
	}
}

void TestPlayerUserIds() {
	Check(Config::IsConfiguredUserIdValid(Config::MAX_USER_ID) &&
	          !Config::IsConfiguredUserIdValid(Config::MAX_USER_ID + 1) &&
	          !Config::IsConfiguredUserIdValid(254) && !Config::IsConfiguredUserIdValid(255),
	      "configured user IDs did not reserve a valid range for guests");
	Config::ConfigOptions options;
	options.user_id = Config::MAX_USER_ID;
	Config::Load(options);
	for (int player = 0; player < Config::MAX_PLAYERS; player++) {
		Check(Config::GetPlayerFromUserId(Config::GetPlayerUserId(player)) == player,
		      "player user ID did not round-trip at the upper limit");
	}
	Check(Config::GetPlayerUserId(3) == std::numeric_limits<int32_t>::max() &&
	          Config::GetPlayerFromUserId(-1) == -1,
	      "guest user ID overflowed or accepted an unknown user");
	Config::Load(Config::ConfigOptions {});
}

void TestMultiplayerRouting() {
	Controller controller;
	for (int player = 1; player < Config::MAX_PLAYERS; player++) {
		Check(PadOpen(Config::GetPlayerUserId(player), 0, 0, nullptr) == player + 1,
		      "a guest could not open its own pad port");
	}
	for (int id = 2; id <= 5; id++) {
		Connect(id);
	}
	Connect(2); // Duplicate device reports must not consume another port.
	const auto read = [](int handle) {
		PadData data {};
		Check(PadReadState(handle, &data) == OK, "player pad read failed");
		return data;
	};
	SetButton(1, PAD_BUTTON_CROSS, true);
	SetButton(2, PAD_BUTTON_CIRCLE, true);
	SetButton(5, PAD_BUTTON_TRIANGLE, true);
	Check(read(1).buttons == PAD_BUTTON_CROSS && read(2).buttons == PAD_BUTTON_CIRCLE &&
	          read(3).buttons == 0 && read(4).buttons == 0,
	      "physical pads did not route independently or overflow reached a player");

	SetRumble(100, 50);
	CycleKeyboardPlayer(); // Dedicated player 1; physical pad 1 waits.
	Check(GetActiveControllerId() == HOST_INPUT_CONTROLLER_ID &&
	          read(2).buttons == PAD_BUTTON_CIRCLE && pad_rumble[1].large == 0 &&
	          pad_rumble[1].small == 0,
	      "keyboard reassignment changed an unaffected player or left pad output "
	      "running");
	for (int mode = 2; mode <= Config::MAX_PLAYERS; mode++) {
		CycleKeyboardPlayer();
	}
	CycleKeyboardPlayer(); // Shared player 1; all original pads must return.
	ResetInputState();
	for (int id = 1; id <= 5; id++) {
		SetButton(id, 1u << id, true);
	}
	for (int handle = 1; handle <= Config::MAX_PLAYERS; handle++) {
		Check(read(handle).buttons == (1u << handle),
		      "cycling keyboard players lost or duplicated a connected pad");
	}

	// Moving the keyboard clears its held input from the old player.
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_TRIANGLE, true);
	CycleKeyboardPlayer();
	CycleKeyboardPlayer();
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_TRIANGLE, true);
	CycleKeyboardPlayer(); // Keyboard now moves from player 2 to player 3.
	Check((read(2).buttons & PAD_BUTTON_TRIANGLE) == 0,
	      "moving the keyboard retained a held button on its old player");
	CycleKeyboardPlayer();
	CycleKeyboardPlayer();
	Disconnect(2);
	SetButton(5, PAD_BUTTON_SQUARE, true);
	Check(read(2).buttons == PAD_BUTTON_SQUARE && GetActiveControllerId() == 1,
	      "waiting pad did not fill a free slot or an unaffected pad moved");
	Disconnect(99);
	Check(PadCheckHandle(std::numeric_limits<int>::min()) == PAD_ERROR_INVALID_HANDLE &&
	          PadClose(std::numeric_limits<int>::max()) == PAD_ERROR_INVALID_HANDLE,
	      "an extreme handle aliased a player port");
	closed_pads.clear();
	EmergencyShutdown();
	std::sort(closed_pads.begin(), closed_pads.end());
	Check(closed_pads == std::vector<int> {1, 3, 4, 5},
	      "shutdown did not close each registered host pad exactly once");
}

void TestDedicatedKeyboardOutputs() {
	Config::ConfigOptions options;
	options.keyboard_player  = 1;
	options.controller_color = Config::ControllerColor {30, 40, 50};
	Config::Load(options);
	Initialize();
	PadInit();
	for (int player = 0; player < Config::MAX_PLAYERS; player++) {
		Check(PadOpen(Config::GetPlayerUserId(player), 0, 0, nullptr) == player + 1,
		      "dedicated keyboard port did not open");
	}
	Connect(1);
	Connect(2);
	const PadVibrationParam request {100, 50};
	Check(PadSetVibration(3, &request) == OK, "player three vibration failed");
	SetButton(2, PAD_BUTTON_CROSS, true);
	CycleKeyboardPlayer(); // Pad 1 moves from player 2 to player 1.
	PadData data {};
	Check(GetActiveControllerId() == 1 && pad_colors[1] == *options.controller_color,
	      "moving a pad to an earlier slot cleared its configured lightbar");
	Check(PadReadState(3, &data) == OK && data.buttons == PAD_BUTTON_CROSS &&
	          pad_rumble[2].large == 100 * 257 && pad_rumble[2].small == 50 * 257,
	      "keyboard reassignment reset an unaffected player's input or output");
	haptics_handles_rumble = true;
	SetRumble(200, 100);
	for (int mode = 0; mode < 4; mode++) {
		CycleKeyboardPlayer();
	}
	Check(GetActiveControllerId() == HOST_INPUT_CONTROLLER_ID && haptics.back().large == 0 &&
	          haptics.back().small == 0 && haptics.back().duration == 0,
	      "moving player one's pad retained its DualSense vibration request");
	Shutdown();
	Config::Load(Config::ConfigOptions {});
}

void TestTriggerEffectState() {
	Controller                       controller;
	PadTriggerEffectStateInformation info {{-1, -1}};
	const auto check_state = [&](int32_t left, int32_t right, const char* message) {
		Check(PadGetTriggerEffectState(1, &info) == OK && info.state[0] == left &&
		          info.state[1] == right,
		      message);
	};
	const auto axis = [&](int left, int right) {
		SetAxis(1, Axis::TriggerLeft, left);
		SetAxis(1, Axis::TriggerRight, right);
	};
	PadTriggerEffectParam param {};
	const auto            set_effect = [&] {
		Check(PadSetTriggerEffect(1, &param) == OK, "trigger request failed");
	};
	Check(PadGetTriggerEffectState(2, &info) == PAD_ERROR_INVALID_HANDLE && info.state[0] == -1 &&
	          info.state[1] == -1,
	      "invalid trigger handle changed the output");
	Check(PadGetTriggerEffectState(2, nullptr) == PAD_ERROR_INVALID_HANDLE,
	      "trigger state checked the output before the handle");
	Check(PadGetTriggerEffectState(1, nullptr) == PAD_ERROR_INVALID_ARG,
	      "trigger state accepted a null output");
	check_state(0, 0, "unset trigger effects did not report off");

	param.trigger_mask       = 3;
	param.command[0].mode    = 1;
	param.command[0].data[0] = 7;
	param.command[0].data[1] = 8;
	param.command[1].mode    = 2;
	param.command[1].data[0] = 2;
	param.command[1].data[1] = 7;
	param.command[1].data[2] = 6;
	set_effect();
	axis(191, 63);
	check_state(1, 3, "triggers activated before the documented start positions");
	axis(192, 64);
	check_state(2, 4, "triggers did not activate at the documented start positions");
	axis(255, 223);
	check_state(2, 4, "weapon fired before its end position");
	SetAxis(1, Axis::TriggerRight, 224);
	SetAxis(1, Axis::TriggerRight, 100);
	check_state(2, 5, "unpolled weapon firing was lost on partial release");
	check_state(2, 5, "reading trigger state changed the weapon latch");
	set_effect();
	check_state(2, 5, "repeating a weapon request rearmed a held trigger");
	CycleSetting(Setting::TriggerEffectIntensity);
	check_state(2, 5, "muting trigger effects changed their reported states");

	// A rejected two-trigger request must leave both cached commands and states intact.
	auto invalid               = param;
	invalid.command[0].mode    = 0;
	invalid.command[1].data[1] = 2;
	Check(PadSetTriggerEffect(1, &invalid) == PAD_ERROR_INVALID_ARG,
	      "invalid weapon end position was accepted");
	SetAxis(1, Axis::TriggerLeft, 255);
	check_state(2, 5, "an invalid request changed cached trigger commands or states");
	SetAxis(1, Axis::TriggerRight, 63);
	SetAxis(1, Axis::TriggerRight, 100);
	check_state(2, 4, "unpolled release did not rearm the weapon");
	param.trigger_mask       = 2;
	param.command[1].data[1] = 8;
	set_effect();
	axis(255, 253);
	check_state(2, 4, "weapon end position 8 fired before output 254");
	SetAxis(1, Axis::TriggerRight, 254);
	check_state(2, 5, "weapon end position 8 did not fire at output 254");

	// Masked updates preserve the other trigger, including its fired state.
	param.trigger_mask       = 1;
	param.command[0].data[0] = 9;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 254);
	check_state(1, 5, "feedback position 9 activated before full travel");
	SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::TriggerLeft, 255);
	check_state(2, 5, "keyboard trigger travel did not activate feedback");
	param.command[0].data[0] = 0;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 0);
	check_state(2, 5, "feedback at position 0 was suppressed by a dead zone");
	param.command[0].data[1] = 0;
	set_effect();
	check_state(1, 5, "zero-strength feedback reported active");

	param.command[0]         = {};
	param.command[0].mode    = 4;
	param.command[0].data[0] = 4;
	param.command[0].data[2] = 4;
	param.command[0].data[8] = 4;
	set_effect();
	const int feedback_cases[][2] = {{0, 2},  {1, 1},   {31, 1},  {32, 2},  {63, 2},
	                                 {64, 1}, {223, 1}, {224, 2}, {254, 2}, {255, 1}};
	for (const auto& test: feedback_cases) {
		SetAxis(1, Axis::TriggerLeft, test[0]);
		check_state(test[1], 5, "multi-position feedback used the wrong position");
	}

	param.command[0]         = {};
	param.command[0].mode    = 5;
	param.command[0].data[0] = 3;
	param.command[0].data[1] = 7;
	param.command[0].data[2] = 2;
	param.command[0].data[3] = 6;
	set_effect();
	const int slope_cases[][2] = {{63, 1}, {64, 2}, {191, 2}, {192, 1}, {255, 1}};
	for (const auto& test: slope_cases) {
		SetAxis(1, Axis::TriggerLeft, test[0]);
		check_state(test[1], 5, "slope feedback did not respect its start and end");
	}
	param.command[0].data[0] = 0;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 0);
	check_state(2, 5, "slope feedback at position 0 was suppressed");

	param.command[0]         = {};
	param.command[0].mode    = 3;
	param.command[0].data[0] = 6;
	param.command[0].data[1] = 4;
	param.command[0].data[2] = 20;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 159);
	check_state(6, 5, "vibration activated before position 6");
	SetAxis(1, Axis::TriggerLeft, 160);
	check_state(7, 5, "vibration did not activate at position 6");
	param.command[0].data[0] = 0;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 0);
	check_state(7, 5, "vibration at position 0 was suppressed");
	param.command[0].data[2] = 0;
	set_effect();
	check_state(6, 5, "zero-frequency vibration reported active");

	param.command[0]          = {};
	param.command[0].mode     = 6;
	param.command[0].data[0]  = 20;
	param.command[0].data[10] = 4;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 254);
	check_state(6, 5, "multi-position vibration activated before position 9");
	SetAxis(1, Axis::TriggerLeft, 255);
	check_state(7, 5, "multi-position vibration did not activate at position 9");
	param.command[0].data[0] = 0;
	set_effect();
	check_state(6, 5, "zero-frequency multi-position vibration reported active");

	ResetInputState();
	check_state(6, 3, "input reset retained a fired weapon");
	param.command[0].mode = 0;
	set_effect();
	check_state(0, 3, "off mode retained an effect state");
	Connect(2);
	Disconnect(1);
	check_state(0, 0, "switching controllers retained effect states");
}

} // namespace

int main() {
	Config::Initialize();
	TestPlayerUserIds();
	TestPadPortOwnership();
	TestTwoPlayerInputIsolation();
	TestPlayerSlotCapacityAndReuse();
	TestWaitingControllerTakesFreedSlot();
	TestPerPlayerPadPorts();
	TestPerPlayerReadHistory();
	TestPerPlayerOutputIsolation();
	TestVirtualUserIdAllocation();
	TestPlayerLightBarColors();
	TestGameLightBarRestoredOnReconnect();
	TestSettingCycles();
	TestGlobalControllerLevels();
	// Each following test initializes another controller and requires strong defaults.
	TestVibrationLifetime();
	TestMaskedTriggersAndValidation();
	TestIndependentOutputsAndPadSwitch();
	TestTriggerEffectState();
	TestMultiplayerRouting();
	TestDedicatedKeyboardOutputs();
	Config::Shutdown();
	return 0;
}
