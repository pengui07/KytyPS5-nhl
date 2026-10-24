// Real SDL virtual devices exercise the controller and UserService together.
#include "libs/controller.cpp"
#include "libs/libUserService.cpp"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace Libs::Controller::DualSenseHaptics {
bool SetVibration(int, uint8_t, uint8_t, uint32_t) { return false; }
void Shutdown() {}
} // namespace Libs::Controller::DualSenseHaptics
namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
  return SDL_GetTicksNS() / 1000;
}
} // namespace Libs::LibKernel
namespace Loader {
void SymbolDatabase::Add(const SymbolResolve &, uint64_t, const std::string &) {
}
namespace Timer {
double GetTimeMs() { return static_cast<double>(SDL_GetTicks()); }
} // namespace Timer
} // namespace Loader

namespace {
using namespace Libs::Controller;
using namespace Libs::UserService;

void Check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "ControllerSdlIntegrationTests: %s (SDL: %s)\n",
                 message, SDL_GetError());
    std::abort();
  }
}

// Use the same SDL event-to-pad boundary as the desktop window, with real SDL
// IDs.
void Pump() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_EVENT_GAMEPAD_ADDED:
      if (!SDL_IsJoystickVirtual(event.gdevice.which))
        break;
      Check(SDL_OpenGamepad(event.gdevice.which) != nullptr,
            "opening a gamepad failed");
      Connect(static_cast<int>(event.gdevice.which));
      break;
    case SDL_EVENT_GAMEPAD_REMOVED:
      Disconnect(static_cast<int>(event.gdevice.which));
      if (auto *pad = SDL_GetGamepadFromID(event.gdevice.which))
        SDL_CloseGamepad(pad);
      break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
      if (event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH)
        SetButton(static_cast<int>(event.gbutton.which), PAD_BUTTON_CROSS,
                  event.gbutton.down);
      break;
    default:
      break;
    }
  }
}

PadData Read(int handle) {
  PadData data{};
  Check(PadReadState(handle, &data) == 0, "pad read failed");
  return data;
}
} // namespace

int main() {
  // Keep hardware out of this automated test; virtual devices still use SDL's
  // real events.
  SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI, "0");
  SDL_SetHint(SDL_HINT_JOYSTICK_RAWINPUT, "0");
  SDL_SetHint(SDL_HINT_XINPUT_ENABLED, "0");
  SDL_SetHint(SDL_HINT_JOYSTICK_DIRECTINPUT, "0");
  SDL_SetHint(SDL_HINT_JOYSTICK_GAMEINPUT, "0");
  Check(SDL_Init(SDL_INIT_GAMEPAD), "SDL initialization failed");
  Config::Initialize();
  Initialize();
  Check(PadInit() == 0, "pad initialization failed");
  std::array<SDL_JoystickID, 4> ids{};
  std::array<SDL_Joystick *, 4> joysticks{};
  for (int player = 0; player < 4; ++player) {
    SDL_VirtualJoystickDesc desc{};
    SDL_INIT_INTERFACE(&desc);
    desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    desc.button_mask = 1u << SDL_GAMEPAD_BUTTON_SOUTH;
    desc.name = "Kyty local co-op test";
    ids[player] = SDL_AttachVirtualJoystick(&desc);
    Check(ids[player] != 0, "attaching a virtual gamepad failed");
    Pump();
    joysticks[player] =
        SDL_GetGamepadJoystick(SDL_GetGamepadFromID(ids[player]));
    Check(joysticks[player] != nullptr,
          "SDL did not enumerate an attached gamepad");
    Check(GetPlayerControllerIdForUser(GetPlayerUserId(player)) ==
              static_cast<int>(ids[player]),
          "SDL gamepad was assigned to the wrong player");
    Check(PadOpen(GetPlayerUserId(player), 0, 0, nullptr) == player + 1,
          "distinct user did not open its own pad handle");
    char name[32]{};
    Check(UserServiceGetUserName(GetPlayerUserId(player), name, sizeof(name)) ==
              0,
          "connected player has no user name");
    Check(std::string(name) == (player == 0
                                    ? Config::GetUserName()
                                    : "Player" + std::to_string(player + 1)),
          "connected player has the wrong user name");
  }
  for (int player = 0; player < 4; ++player) {
    Check(SDL_SetJoystickVirtualButton(joysticks[player],
                                       SDL_GAMEPAD_BUTTON_SOUTH, true),
          "virtual button press failed");
    SDL_UpdateJoysticks();
    Pump();
    for (int other = 0; other < 4; ++other)
      Check(Read(other + 1).buttons == (other == player ? PAD_BUTTON_CROSS : 0),
            "real SDL input leaked to another player's handle");
    Check(SDL_SetJoystickVirtualButton(joysticks[player],
                                       SDL_GAMEPAD_BUTTON_SOUTH, false),
          "virtual button release failed");
    SDL_UpdateJoysticks();
    Pump();
  }
  Check(SDL_DetachVirtualJoystick(ids[1]), "detaching Player2 failed");
  Pump();
  Check(!Read(2).connected && Read(1).connected && Read(3).connected &&
            Read(4).connected,
        "disconnecting Player2 changed another player's connection");
  UserServiceLoginUserIdList list{};
  Check(UserServiceGetLoginUserIdList(&list) == 0 && list.user_id[0] == 1000 &&
            list.user_id[1] == 1002 && list.user_id[2] == 1003 &&
            list.user_id[3] == -1,
        "UserService did not remove the disconnected player");

  // Reproduce the reported trigger at the SDL/guest API boundary. A held button,
  // old SDL instance, or stale open port must not survive a user logout/relogin.
  Check(PadClose(2) == 0, "closing the disconnected player's pad failed");
  SceUserServiceEvent user_event{};
  while (UserServiceGetEvent(&user_event) == 0) {}
  for (int cycle = 0; cycle < 64; ++cycle) {
    SDL_VirtualJoystickDesc desc{};
    SDL_INIT_INTERFACE(&desc);
    desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    desc.button_mask = 1u << SDL_GAMEPAD_BUTTON_SOUTH;
    desc.name = "Kyty reconnect test";
    ids[1] = SDL_AttachVirtualJoystick(&desc);
    Check(ids[1] != 0, "reattaching Player2 failed");
    Pump();
    Check(PadOpen(1001, 0, 0, nullptr) == 2, "reconnected Player2 could not reopen its handle");
    Check(UserServiceGetEvent(&user_event) == 0 && user_event.user_id == 1001 &&
              user_event.event_type == UserServiceEventTypeLogin,
          "reconnect did not deliver Player2's login");
    Check(Read(2).connected && Read(2).buttons == 0 && Read(2).connected_count == cycle + 2,
          "reconnect inherited state or lost its connection count");
    joysticks[1] = SDL_GetGamepadJoystick(SDL_GetGamepadFromID(ids[1]));
    Check(SDL_SetJoystickVirtualButton(joysticks[1], SDL_GAMEPAD_BUTTON_SOUTH, true),
          "reconnected gamepad button failed");
    SDL_UpdateJoysticks();
    Pump();
    Check(Read(2).buttons == PAD_BUTTON_CROSS && Read(1).buttons == 0 &&
              Read(3).buttons == 0 && Read(4).buttons == 0,
          "reconnected input did not remain isolated");
    Check(SDL_DetachVirtualJoystick(ids[1]), "repeated disconnect failed");
    Pump();
    Check(!Read(2).connected && Read(2).buttons == 0 && Read(1).connected &&
              Read(3).connected && Read(4).connected,
          "repeated disconnect retained input or affected another player");
    Check(PadClose(2) == 0, "repeated pad close failed");
    Check(UserServiceGetEvent(&user_event) == 0 && user_event.user_id == 1001 &&
              user_event.event_type == UserServiceEventTypeLogout &&
              UserServiceGetEvent(&user_event) == USER_SERVICE_ERROR_NO_EVENT,
          "repeated disconnect lost or duplicated a user event");
  }
  Shutdown();
  Config::Shutdown();
  SDL_Quit();
  std::puts("Four real SDL virtual gamepads: user identity, handles, input "
            "isolation and 64 reconnect cycles passed.");
}
