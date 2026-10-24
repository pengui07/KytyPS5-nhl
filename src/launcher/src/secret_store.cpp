#include "secret_store.h"

#ifdef _WIN32
#include <windows.h>
#include <wincred.h>
#endif

namespace kyty_launcher {

#ifdef _WIN32

namespace {

// Shows up under Windows Credentials in the Credential Manager control panel.
constexpr wchar_t kTarget[] = L"kyty-launcher/github-token";

} // namespace

bool secret_store_supported()
{
	return true;
}

std::string load_github_token()
{
	PCREDENTIALW cred = nullptr;
	if (CredReadW(kTarget, CRED_TYPE_GENERIC, 0, &cred) == FALSE)
	{
		return {};
	}
	std::string token(reinterpret_cast<const char*>(cred->CredentialBlob), cred->CredentialBlobSize);
	SecureZeroMemory(cred->CredentialBlob, cred->CredentialBlobSize);
	CredFree(cred);
	return token;
}

bool save_github_token(const std::string& token)
{
	if (token.empty() || token.size() > CRED_MAX_CREDENTIAL_BLOB_SIZE)
	{
		return false;
	}
	CREDENTIALW cred {};
	cred.Type               = CRED_TYPE_GENERIC;
	cred.TargetName         = const_cast<wchar_t*>(kTarget);
	cred.CredentialBlobSize = static_cast<DWORD>(token.size());
	cred.CredentialBlob     = reinterpret_cast<BYTE*>(const_cast<char*>(token.data()));
	cred.Persist            = CRED_PERSIST_LOCAL_MACHINE; // this user on this PC, not roamed
	cred.UserName           = const_cast<wchar_t*>(L"github");
	return CredWriteW(&cred, 0) != FALSE;
}

void clear_github_token()
{
	CredDeleteW(kTarget, CRED_TYPE_GENERIC, 0);
}

#else

bool secret_store_supported()
{
	return false;
}

std::string load_github_token()
{
	return {};
}

bool save_github_token(const std::string& /*token*/)
{
	return false;
}

void clear_github_token() {}

#endif

std::string token_hint(const std::string& token)
{
	// Keep the type prefix ("ghp_", "github_pat_") whole, then a few characters of the secret part.
	const size_t prefix = token.rfind('_', 11);
	const size_t keep   = (prefix == std::string::npos ? 0 : prefix + 1) + 12;
	return token.size() <= keep ? std::string(token.size(), '*') : token.substr(0, keep) + "...";
}

} // namespace kyty_launcher
