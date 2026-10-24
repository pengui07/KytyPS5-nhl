// Exercise the guest-visible UserService roster against a fake controller roster.
#include "libs/libUserService.cpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "UserServiceMultiUserTests: %s\n", text);
		std::abort();
	}
}

struct Event {
	int32_t user_id;
	bool    connected;
};

// The roster the production controller module would report.
std::array<int32_t, 4> g_users {1000, 1001, 1002, 1003};
std::array<bool, 4>    g_connected {true, false, false, false};
std::deque<Event>      g_events;
int                   g_disconnect_after_query = -1;

void SetConnected(int player, bool connected) {
	g_connected[player] = connected;
	g_events.push_back({g_users[player], connected});
}
} // namespace

namespace Loader {
void SymbolDatabase::Add(const SymbolResolve&, uint64_t, const std::string&) {}
namespace Timer {
double GetTimeMs() {
	return 0.0;
}
} // namespace Timer
} // namespace Loader

namespace Libs::Controller {
int32_t GetPlayerUserId(int player_index) {
	return player_index >= 0 && player_index < PLAYER_SLOT_COUNT ? g_users[player_index] : -1;
}
bool IsPlayerControllerConnected(int player_index) {
	const bool connected = player_index >= 0 && player_index < PLAYER_SLOT_COUNT && g_connected[player_index];
	if (player_index == g_disconnect_after_query) {
		g_connected[player_index] = false;
		g_disconnect_after_query = -1;
	}
	return connected;
}
// The configured-guest-user path asks this; the test's guests are its controller slots.
bool IsPlayerConnected(int player) {
	return player >= 0 && player < PLAYER_SLOT_COUNT && g_connected[player];
}
bool PollPlayerConnectionEvent(int32_t* user_id, bool* connected) {
	if (g_events.empty()) {
		return false;
	}
	*user_id   = g_events.front().user_id;
	*connected = g_events.front().connected;
	g_events.pop_front();
	return true;
}
} // namespace Libs::Controller

namespace {
using namespace Libs::UserService;

constexpr int NOT_LOGGED_IN = USER_SERVICE_ERROR_NOT_LOGGED_IN;

struct Roster {
	std::array<int, 4> users;
};

Roster LoginList() {
	UserServiceLoginUserIdList list {};
	Check(UserServiceGetLoginUserIdList(&list) == OK, "login list failed");
	return {{list.user_id[0], list.user_id[1], list.user_id[2], list.user_id[3]}};
}

std::string NameOf(int user_id) {
	char name[32] {};
	Check(UserServiceGetUserName(user_id, name, sizeof(name)) == OK, "a user had no name");
	return name;
}

int NumberOf(int user_id) {
	int32_t number = 0;
	Check(UserServiceGetUserNumber(user_id, &number) == OK, "a user had no number");
	return number;
}

// Every user-scoped call rejects a user that is not logged in.
void CheckRejected(int user_id, const char* text) {
	char                   name[32] {};
	int32_t                value = 0;
	uint32_t               age   = 0;
	UserServiceGamePresets presets {};
	Check(UserServiceGetUserName(user_id, name, sizeof(name)) == NOT_LOGGED_IN &&
	          UserServiceGetUserNumber(user_id, &value) == NOT_LOGGED_IN &&
	          UserServiceGetGamePresets(user_id, &presets) == NOT_LOGGED_IN &&
	          UserServiceGetAgeLevel(user_id, &age) == NOT_LOGGED_IN &&
	          UserServiceGetAccessibilityVibration(user_id, &value) == NOT_LOGGED_IN &&
	          UserServiceGetAccessibilityTriggerEffect(user_id, &value) == NOT_LOGGED_IN &&
	          UserServiceGetAccessibilityChatTranscription(user_id, &value) == NOT_LOGGED_IN &&
	          UserServiceGetAccessibilityPressAndHoldDelay(user_id, &value) == NOT_LOGGED_IN &&
	          UserServiceGetAccessibilityZoomEnabled(user_id, &value) == NOT_LOGGED_IN &&
	          UserServiceGetAccessibilityZoomFollowFocus(user_id, &value) == NOT_LOGGED_IN,
	      text);
}

void CheckAccepted(int user_id, const char* text) {
	int32_t                value = 0;
	uint32_t               age   = 0;
	UserServiceGamePresets presets {};
	Check(UserServiceGetGamePresets(user_id, &presets) == OK &&
	          UserServiceGetAgeLevel(user_id, &age) == OK &&
	          UserServiceGetAccessibilityVibration(user_id, &value) == OK &&
	          UserServiceGetAccessibilityTriggerEffect(user_id, &value) == OK &&
	          UserServiceGetAccessibilityChatTranscription(user_id, &value) == OK &&
	          UserServiceGetAccessibilityPressAndHoldDelay(user_id, &value) == OK &&
	          UserServiceGetAccessibilityZoomEnabled(user_id, &value) == OK &&
	          UserServiceGetAccessibilityZoomFollowFocus(user_id, &value) == OK,
	      text);
}

// Runs first: the initial login is delivered once and the queued transitions follow in order.
void TestEventOrdering() {
	SceUserServiceEvent event {};
	Check(UserServiceGetEvent(&event) == OK && event.event_type == UserServiceEventTypeLogin &&
	          event.user_id == g_users[0],
	      "Player1's initial login was not delivered first");
	Check(UserServiceGetEvent(&event) == USER_SERVICE_ERROR_NO_EVENT,
	      "Player1's login was delivered twice");

	// Two connections and a disconnection happen before the guest polls again.
	SetConnected(1, true);
	SetConnected(3, true);
	SetConnected(1, false);
	const Event expected[] = {{g_users[1], true}, {g_users[3], true}, {g_users[1], false}};
	for (const auto& want: expected) {
		Check(UserServiceGetEvent(&event) == OK && event.user_id == want.user_id &&
		          event.event_type ==
		              (want.connected ? UserServiceEventTypeLogin : UserServiceEventTypeLogout),
		      "a queued transition was lost, repeated, or reordered");
	}
	Check(UserServiceGetEvent(&event) == USER_SERVICE_ERROR_NO_EVENT,
	      "a transition was delivered more than once");
	SetConnected(3, false);
	g_events.clear();
}

void TestPlayerOneIdentity() {
	int initial = 0;
	Check(UserServiceGetInitialUser(&initial) == OK && initial == Config::GetUserId() &&
	          g_users[0] == Config::GetUserId(),
	      "the initial user is not the configured player");
	const auto list = LoginList();
	Check(list.users == std::array<int, 4> {g_users[0], -1, -1, -1},
	      "a lone player's login list was wrong");
	Check(NameOf(g_users[0]) == Config::GetUserName() && NumberOf(g_users[0]) == 1,
	      "Player1 lost its configured name or number");
	CheckAccepted(g_users[0], "Player1 was rejected by a user-scoped call");
	CheckRejected(g_users[1], "a disconnected secondary user was accepted");
	CheckRejected(12345, "an unknown user was accepted");
	CheckRejected(-1, "an invalid user was accepted");
}

void TestSecondaryPlayers() {
	SetConnected(1, true);
	SetConnected(3, true);
	g_events.clear();

	// Connected players in slot order, with the gap between them closed.
	Check(LoginList().users == std::array<int, 4> {g_users[0], g_users[1], g_users[3], -1},
	      "the login list was not the connected players in slot order");
	Check(NameOf(g_users[1]) == "Player2" && NameOf(g_users[3]) == "Player4",
	      "secondary players had the wrong names");
	Check(NumberOf(g_users[1]) == 2 && NumberOf(g_users[3]) == 4,
	      "secondary players had the wrong numbers");
	CheckAccepted(g_users[1], "Player2 was rejected by a user-scoped call");
	CheckAccepted(g_users[3], "Player4 was rejected by a user-scoped call");
	CheckRejected(g_users[2], "the empty Player3 slot was accepted");

	char small_name[8] {};
	Check(UserServiceGetUserName(g_users[1], small_name, sizeof(small_name)) == OK &&
	          std::string(small_name) == "Player2",
	      "a seven-character name did not fit its buffer");
	Check(UserServiceGetUserNumber(g_users[1], nullptr) ==
	          USER_SERVICE_ERROR_INVALID_ARGUMENT,
	      "a null number output was accepted");

	SetConnected(1, false);
	g_events.clear();
	CheckRejected(g_users[1], "a user that logged out was still accepted");
	Check(LoginList().users == std::array<int, 4> {g_users[0], g_users[3], -1, -1},
	      "the login list kept a logged-out player");
	Check(NameOf(g_users[3]) == "Player4" && NumberOf(g_users[3]) == 4,
	      "a remaining player changed identity when another left");

	// A player that returns is the same virtual user.
	SetConnected(1, true);
	g_events.clear();
	Check(NameOf(g_users[1]) == "Player2" && NumberOf(g_users[1]) == 2,
	      "a returning player changed identity");
}
} // namespace

int main() {
	Config::Initialize();
	g_users[0] = Config::GetUserId();
	TestEventOrdering();
	TestPlayerOneIdentity();
	TestSecondaryPlayers();
	// Simulate a main-thread hot-unplug immediately after the guest thread's roster query.
	g_disconnect_after_query = 1;
	int32_t number = -1;
	Check(UserServiceGetUserNumber(g_users[1], &number) == OK && number == 2,
	      "a concurrent disconnect produced a successful invalid player number");
	Check(UserServiceGetUserNumber(g_users[1], &number) == USER_SERVICE_ERROR_NOT_LOGGED_IN,
	      "the next query accepted the disconnected player");
	Config::Shutdown();
	return 0;
}
