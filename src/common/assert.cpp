#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
// windows.h must come first
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace Common {

static thread_local int g_soft_exit_depth = 0;

SoftExitScope::SoftExitScope(bool enable): m_enable(enable) {
	if (m_enable) {
		g_soft_exit_depth++;
	}
}

SoftExitScope::~SoftExitScope() {
	if (m_enable) {
		g_soft_exit_depth--;
	}
}

static void ThrowIfSoft(std::string_view text, const char* file, int line) {
	if (g_soft_exit_depth > 0) {
		throw SoftExitError(fmt::format("{} ({}:{})", text, file, line));
	}
}

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

// Fatal reports carry the host call stack: an EXIT_IF in a shared helper (CommandScheduler::Current,
// say) names only the helper, not the caller that broke its contract.
static std::string BuildHostStack() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void*      frames[32] {};
	const auto count = RtlCaptureStackBackTrace(2, 32, frames, nullptr);
	HANDLE     process = GetCurrentProcess();
	static const bool sym_ok = [process] {
		SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
		return SymInitialize(process, nullptr, TRUE) != FALSE;
	}();
	std::string out = "--- Host Stack ---\n";
	for (USHORT i = 0; i < count; i++) {
		const auto address = reinterpret_cast<DWORD64>(frames[i]);
		HMODULE    module  = nullptr;
		GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                   static_cast<LPCSTR>(frames[i]), &module);
		char module_name[MAX_PATH] {};
		if (module != nullptr) {
			GetModuleFileNameA(module, module_name, MAX_PATH);
		}
		const char* base_name = module_name;
		for (const char* c = module_name; *c != 0; c++) {
			if (*c == '\\' || *c == '/') base_name = c + 1;
		}
		out += fmt::format("  #{:02} {}+0x{:x}", i, base_name,
		                   address - reinterpret_cast<DWORD64>(module));
		if (sym_ok) {
			alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512] {};
			auto* symbol         = reinterpret_cast<SYMBOL_INFO*>(buffer);
			symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
			symbol->MaxNameLen   = 511;
			DWORD64 displacement = 0;
			if (SymFromAddr(process, address, &displacement, symbol) != FALSE) {
				out += fmt::format(" {}+0x{:x}", symbol->Name, displacement);
			}
			IMAGEHLP_LINE64 line {};
			line.SizeOfStruct    = sizeof(line);
			DWORD line_displacement = 0;
			if (SymGetLineFromAddr64(process, address, &line_displacement, &line) != FALSE) {
				out += fmt::format(" ({}:{})", line.FileName, line.LineNumber);
			}
		}
		out += "\n";
	}
	return out;
#else
	return {};
#endif
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	Log::WriteFatal(BuildFatalReport(title, text, file, line) + BuildHostStack());
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	ThrowIfSoft(fmt::format("Error: condition ({}) is true", expr), file, line);
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	ThrowIfSoft(fmt::format("Not implemented ({})", expr), file, line);
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	ThrowIfSoft(text, file, line);
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line) + BuildHostStack());
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	ThrowIfSoft(text, file, line);
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line) + BuildHostStack());
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
