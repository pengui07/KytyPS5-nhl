#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace kyty_launcher {

// GET with redirects (GitHub sends release assets to another host). The body goes to `sink`,
// which returns false to stop. `range` ("100-199", "-65536") asks for part of the file; a
// server that ignores it answers 200 with the whole file, which is refused here
// (*range_ignored) before any of it is read. `content_length` gets the size when known.
// WinHTTP on Windows, libcurl elsewhere.
// `validator` (optional) makes the request conditional: its etag goes out as If-None-Match, a 304
// comes back as success with no body and not_modified set, and a 200 leaves the new ETag in etag.
struct HttpValidator
{
	std::string etag;
	bool        not_modified = false;
};
// The GitHub token sent to api.github.com (from the Version Manager settings); empty falls back to the
// GITHUB_TOKEN / GH_TOKEN environment variables.
void set_github_token(const std::string& token);

bool http_get(const std::string& url, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length, std::string* error,
              const std::string& range = std::string(), bool* range_ignored = nullptr, HttpValidator* validator = nullptr);

// Is a GitHub token set (in the settings or the environment)?
bool github_token_set();

// A release file through GitHub's asset API (api.github.com/repos/o/r/releases/assets/<id>), which takes the
// token, so files of private repos can be downloaded. The redirect to the signed download URL is followed
// here without the token. Each call costs one API request; same arguments as http_get otherwise.
bool http_get_github_asset(const std::string& api_url, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length,
                           std::string* error, const std::string& range = std::string(), bool* range_ignored = nullptr);

} // namespace kyty_launcher
