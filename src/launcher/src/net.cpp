#include "net.h"

#include "settings.h"

#include <cstdlib>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#else
#include <curl/curl.h>
#endif

namespace kyty_launcher {

namespace {

std::mutex  g_token_mutex;
std::string g_token;

// A token raises GitHub's API limit from 60 to 5000 requests an hour. It goes only to the API host, never
// to the hosts that release downloads redirect to.
std::string GitHubAuthHeader(const std::string& url)
{
	if (url.rfind("https://api.github.com/", 0) != 0)
	{
		return {};
	}
	{
		std::lock_guard lock(g_token_mutex);
		if (!g_token.empty())
		{
			return "Authorization: Bearer " + g_token;
		}
	}
	for (const char* name: {"GITHUB_TOKEN", "GH_TOKEN"})
	{
		const char* v = std::getenv(name);
		if (v != nullptr && *v != '\0')
		{
			return std::string("Authorization: Bearer ") + v;
		}
	}
	return {};
}

} // namespace

void set_github_token(const std::string& token)
{
	// Pasted tokens often carry stray spaces or a newline.
	const size_t b = token.find_first_not_of(" \t\r\n");
	const size_t e = token.find_last_not_of(" \t\r\n");
	std::lock_guard lock(g_token_mutex);
	g_token = b == std::string::npos ? std::string() : token.substr(b, e - b + 1);
}

#ifdef _WIN32
namespace {

std::wstring Widen(const std::string& s)
{
	return utf8_path(s).wstring();
}

} // namespace

// GET with redirects (GitHub sends release assets to another host). The body goes to `sink`.
// `range` ("100-199", "-65536") asks for part of the file; a server that ignores it answers
// 200 with the whole file, which is refused here (*range_ignored) before any of it is read.
// http_get, plus a mode for GitHub's asset API (`redirect` set): the file is asked for as raw bytes and a
// redirect is not followed but handed back in *redirect, so the token never goes to the host it points at.
static bool HttpRequest(const std::string& url, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length,
                        std::string* error, const std::string& range, bool* range_ignored, HttpValidator* validator, std::string* redirect)
{
	std::wstring     wurl = Widen(url);
	URL_COMPONENTS   uc {};
	wchar_t          host[256];
	wchar_t          path[2048];
	uc.dwStructSize     = sizeof(uc);
	uc.lpszHostName     = host;
	uc.dwHostNameLength = static_cast<DWORD>(std::size(host));
	uc.lpszUrlPath      = path;
	uc.dwUrlPathLength  = static_cast<DWORD>(std::size(path));
	// The query string comes back separately; it is appended to the path below.
	wchar_t extra[1024];
	uc.lpszExtraInfo     = extra;
	uc.dwExtraInfoLength = static_cast<DWORD>(std::size(extra));
	if (WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc) == FALSE)
	{
		*error = "Bad URL: " + url;
		return false;
	}
	std::wstring object = std::wstring(uc.lpszUrlPath, uc.dwUrlPathLength) + std::wstring(uc.lpszExtraInfo, uc.dwExtraInfoLength);

	HINTERNET session = WinHttpOpen(L"Kyty-Launcher", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
	HINTERNET conn    = nullptr;
	HINTERNET req     = nullptr;
	if (session != nullptr)
	{
		WinHttpSetTimeouts(session, 10000, 10000, 15000, 60000);
		conn = WinHttpConnect(session, std::wstring(uc.lpszHostName, uc.dwHostNameLength).c_str(), uc.nPort, 0);
	}
	if (conn != nullptr)
	{
		req = WinHttpOpenRequest(conn, L"GET", object.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
		                         uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0);
	}
	if (req != nullptr && redirect != nullptr)
	{
		DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
		WinHttpSetOption(req, WINHTTP_OPTION_REDIRECT_POLICY, &policy, sizeof(policy));
	}
	// The asset API answers with the file's details (JSON) unless only raw bytes are accepted.
	std::wstring headers = redirect != nullptr ? L"Accept: application/octet-stream\r\n"
	                                           : L"Accept: application/vnd.github+json, application/octet-stream\r\n";
	if (!range.empty())
	{
		headers += L"Range: bytes=" + Widen(range) + L"\r\n";
	}
	if (const std::string auth = GitHubAuthHeader(url); !auth.empty())
	{
		headers += Widen(auth) + L"\r\n";
	}
	if (validator != nullptr && !validator->etag.empty())
	{
		headers += L"If-None-Match: " + Widen(validator->etag) + L"\r\n";
	}
	bool ok = req != nullptr && WinHttpSendRequest(req, headers.c_str(), static_cast<DWORD>(-1L), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) != FALSE &&
	          WinHttpReceiveResponse(req, nullptr) != FALSE;
	DWORD status    = 0;
	bool  skip_body = false;
	if (ok)
	{
		DWORD len = sizeof(status);
		WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &len,
		                    WINHTTP_NO_HEADER_INDEX);
		ok = range.empty() ? status == 200 : status == 206;
		if (redirect != nullptr && status >= 301 && status <= 308 && status != 304)
		{
			// The signed URL can be long: ask for the size first.
			DWORD loc_len = 0;
			WinHttpQueryHeaders(req, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, WINHTTP_NO_OUTPUT_BUFFER, &loc_len, WINHTTP_NO_HEADER_INDEX);
			std::wstring loc(loc_len / sizeof(wchar_t) + 1, L'\0');
			if (loc_len > 0 &&
			    WinHttpQueryHeaders(req, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, loc.data(), &loc_len, WINHTTP_NO_HEADER_INDEX) != FALSE)
			{
				loc.resize(loc_len / sizeof(wchar_t));
				*redirect = path_utf8(std::filesystem::path(loc));
			}
			ok        = !redirect->empty();
			skip_body = true;
			if (!ok)
			{
				*error = "HTTP " + std::to_string(status) + " without a location from " + url;
			}
		} else if (validator != nullptr && status == 304)
		{
			// Nothing to read: the caller's copy is current.
			validator->not_modified = true;
			ok                      = true;
		} else if (validator != nullptr && ok)
		{
			wchar_t buf[256];
			DWORD   etag_len = sizeof(buf);
			validator->etag.clear();
			if (WinHttpQueryHeaders(req, WINHTTP_QUERY_ETAG, WINHTTP_HEADER_NAME_BY_INDEX, buf, &etag_len, WINHTTP_NO_HEADER_INDEX) != FALSE)
			{
				validator->etag = path_utf8(std::filesystem::path(std::wstring(buf, etag_len / sizeof(wchar_t))));
			}
		}
		if (!ok)
		{
			if (!range.empty() && status == 200)
			{
				if (range_ignored != nullptr)
				{
					*range_ignored = true;
				}
				*error = "The server ignores Range requests";
			} else
			{
				*error = "HTTP " + std::to_string(status) + " from " + url;
			}
		}
	} else
	{
		*error = "Could not connect (error " + std::to_string(GetLastError()) + ")";
	}
	if (ok && !skip_body && content_length != nullptr)
	{
		wchar_t buf[32];
		DWORD   len = sizeof(buf);
		if (WinHttpQueryHeaders(req, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, buf, &len, WINHTTP_NO_HEADER_INDEX) != FALSE)
		{
			*content_length = std::wcstoull(buf, nullptr, 10);
		}
	}
	std::string chunk(64 * 1024, '\0');
	while (ok && !skip_body)
	{
		DWORD got = 0;
		if (WinHttpReadData(req, chunk.data(), static_cast<DWORD>(chunk.size()), &got) == FALSE)
		{
			*error = "Download interrupted";
			ok     = false;
			break;
		}
		if (got == 0)
		{
			break;
		}
		if (!sink(chunk.data(), got))
		{
			*error = "Could not write the download";
			ok     = false;
		}
	}
	for (HINTERNET h: {req, conn, session})
	{
		if (h != nullptr)
		{
			WinHttpCloseHandle(h);
		}
	}
	return ok;
}


#else

namespace {

struct CurlContext
{
	const std::function<bool(const char*, size_t)>* sink = nullptr;
	CURL*                                          handle = nullptr;
	bool                                           ranged = false;
	bool                                           first  = true;
	bool                                           range_ignored = false;
	bool                                           bad_status    = false;
	bool                                           sink_failed   = false;
	bool                                           not_modified  = false;
	HttpValidator*                                 validator     = nullptr;
	bool                                           manual_redirect = false; // a 3xx answer is the result, its body is dropped
	bool                                           redirected      = false;
	long                                           status        = 0;
	uint64_t*                                      content_length = nullptr;
};

size_t WriteCallback(char* data, size_t size, size_t count, void* user)
{
	auto* c = static_cast<CurlContext*>(user);
	if (!c->first && c->redirected)
	{
		return size * count;
	}
	if (c->first)
	{
		// Headers are in: check the status before any of the body is used.
		c->first = false;
		curl_easy_getinfo(c->handle, CURLINFO_RESPONSE_CODE, &c->status);
		if (c->validator != nullptr && c->status == 304)
		{
			c->not_modified = true;
			return 0;
		}
		if (c->manual_redirect && c->status >= 301 && c->status <= 308)
		{
			c->redirected = true;
			return size * count;
		}
		if (c->ranged ? c->status != 206 : c->status != 200)
		{
			c->range_ignored = c->ranged && c->status == 200;
			c->bad_status    = true;
			return 0;
		}
		if (c->content_length != nullptr)
		{
			curl_off_t len = -1;
			if (curl_easy_getinfo(c->handle, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len) == CURLE_OK && len >= 0)
			{
				*c->content_length = static_cast<uint64_t>(len);
			}
		}
	}
	if (!(*c->sink)(data, size * count))
	{
		c->sink_failed = true;
		return 0;
	}
	return size * count;
}

// Keeps the ETag response header for a conditional request.
size_t HeaderCallback(char* data, size_t size, size_t count, void* user)
{
	auto*             c = static_cast<CurlContext*>(user);
	const std::string line(data, size * count);
	if (c->validator != nullptr && line.size() > 5 && (line.compare(0, 5, "ETag:") == 0 || line.compare(0, 5, "etag:") == 0))
	{
		size_t b = line.find_first_not_of(" \t", 5);
		size_t e = line.find_last_not_of(" \t\r\n");
		c->validator->etag = b == std::string::npos || e < b ? std::string() : line.substr(b, e - b + 1);
	}
	return size * count;
}

} // namespace

// http_get, plus a mode for GitHub's asset API (`redirect` set): the file is asked for as raw bytes and a
// redirect is not followed but handed back in *redirect, so the token never goes to the host it points at.
static bool HttpRequest(const std::string& url, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length,
                        std::string* error, const std::string& range, bool* range_ignored, HttpValidator* validator, std::string* redirect)
{
	static std::once_flag once;
	std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });

	CURL* handle = curl_easy_init();
	if (handle == nullptr)
	{
		*error = "Could not start libcurl";
		return false;
	}
	CurlContext ctx;
	ctx.sink           = &sink;
	ctx.handle         = handle;
	ctx.ranged         = !range.empty();
	ctx.content_length = content_length;
	ctx.validator      = validator;
	ctx.manual_redirect = redirect != nullptr;
	std::string sent_etag;
	if (validator != nullptr)
	{
		sent_etag = validator->etag;
		validator->etag.clear(); // refilled from the response
	}

	// The asset API answers with the file's details (JSON) unless only raw bytes are accepted.
	curl_slist* headers = curl_slist_append(nullptr, redirect != nullptr ? "Accept: application/octet-stream"
	                                                                     : "Accept: application/vnd.github+json, application/octet-stream");
	if (const std::string auth = GitHubAuthHeader(url); !auth.empty())
	{
		headers = curl_slist_append(headers, auth.c_str());
	}
	if (!sent_etag.empty())
	{
		headers = curl_slist_append(headers, ("If-None-Match: " + sent_etag).c_str());
	}
	curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
	curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(handle, CURLOPT_USERAGENT, "Kyty-Launcher");
	curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, redirect != nullptr ? 0L : 1L);
	curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 10L);
	// A transfer that moves less than 1 byte/s for 30 s is stuck.
	curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, 1L);
	curl_easy_setopt(handle, CURLOPT_LOW_SPEED_TIME, 30L);
	curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
	if (!range.empty())
	{
		curl_easy_setopt(handle, CURLOPT_RANGE, range.c_str());
	}
	curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, WriteCallback);
	curl_easy_setopt(handle, CURLOPT_WRITEDATA, &ctx);
	curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, HeaderCallback);
	curl_easy_setopt(handle, CURLOPT_HEADERDATA, &ctx);

	const CURLcode rc = curl_easy_perform(handle);
	if (ctx.first && rc == CURLE_OK)
	{
		// An empty body never reached the callback.
		curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &ctx.status);
		ctx.not_modified = validator != nullptr && ctx.status == 304;
		ctx.redirected   = ctx.manual_redirect && ctx.status >= 301 && ctx.status <= 308 && ctx.status != 304;
		ctx.bad_status = !ctx.not_modified && !ctx.redirected && ctx.ranged ? ctx.status != 206 : ctx.status != 200;
		ctx.range_ignored = ctx.ranged && ctx.status == 200;
	}
	if (ctx.redirected)
	{
		const char* loc = nullptr;
		if (curl_easy_getinfo(handle, CURLINFO_REDIRECT_URL, &loc) == CURLE_OK && loc != nullptr)
		{
			*redirect = loc;
		}
	}
	curl_slist_free_all(headers);
	curl_easy_cleanup(handle);

	if (ctx.redirected)
	{
		if (redirect->empty())
		{
			*error = "HTTP " + std::to_string(ctx.status) + " without a location from " + url;
			return false;
		}
		return true;
	}

	if (ctx.not_modified)
	{
		validator->not_modified = true;
		validator->etag         = sent_etag;
		return true;
	}
	if (ctx.bad_status)
	{
		if (ctx.range_ignored)
		{
			if (range_ignored != nullptr)
			{
				*range_ignored = true;
			}
			*error = "The server ignores Range requests";
		} else
		{
			*error = "HTTP " + std::to_string(ctx.status) + " from " + url;
		}
		return false;
	}
	if (ctx.sink_failed)
	{
		*error = "Could not write the download";
		return false;
	}
	if (rc != CURLE_OK)
	{
		*error = std::string("Could not connect: ") + curl_easy_strerror(rc);
		return false;
	}
	return true;
}

#endif

bool http_get(const std::string& url, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length, std::string* error,
              const std::string& range, bool* range_ignored, HttpValidator* validator)
{
	return HttpRequest(url, sink, content_length, error, range, range_ignored, validator, nullptr);
}

bool github_token_set()
{
	return !GitHubAuthHeader("https://api.github.com/").empty();
}

bool http_get_github_asset(const std::string& api_url, const std::function<bool(const char*, size_t)>& sink, uint64_t* content_length,
                           std::string* error, const std::string& range, bool* range_ignored)
{
	std::string location;
	if (!HttpRequest(api_url, sink, content_length, error, range, range_ignored, nullptr, &location))
	{
		return false;
	}
	if (location.empty())
	{
		return true; // the API sent the bytes itself
	}
	// The signed download URL is on another host, so no token goes with it (it would be refused anyway).
	return http_get(location, sink, content_length, error, range, range_ignored);
}

} // namespace kyty_launcher
