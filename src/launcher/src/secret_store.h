#pragma once

#include <string>

namespace kyty_launcher {

// The GitHub token lives in the OS credential store, never in kyty.json, and the UI can only replace or clear it.
// Windows: Credential Manager (encrypted for the signed-in user).
// macOS / Linux: not implemented yet (Keychain / libsecret); nothing is saved and GITHUB_TOKEN / GH_TOKEN still work.

// False when this platform has no credential store wired up: a token then lasts only until the launcher closes.
bool secret_store_supported();

// The saved token, or empty.
std::string load_github_token();

// Replaces the saved token; false when it could not be stored.
bool save_github_token(const std::string& token);

void clear_github_token();

// What the UI shows for a saved token, enough to tell which one it is: "ghp_1a2b3c4d5e6f..." (prefix + 12 chars).
std::string token_hint(const std::string& token);

} // namespace kyty_launcher
