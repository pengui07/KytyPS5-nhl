// CPU time of a process's busiest threads over a window: threadcpu <pid> <seconds> [top]
// Prints "tid percent" lines, busiest first (100 = one core all the time).
#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static unsigned long long ThreadCpu(HANDLE thread) {
	FILETIME c, e, k, u;
	if (!GetThreadTimes(thread, &c, &e, &k, &u)) return 0;
	return ((static_cast<unsigned long long>(k.dwHighDateTime) << 32u) | k.dwLowDateTime) +
	       ((static_cast<unsigned long long>(u.dwHighDateTime) << 32u) | u.dwLowDateTime);
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "threadcpu <pid> <seconds> [top]\n");
		return 1;
	}
	const DWORD  pid     = std::strtoul(argv[1], nullptr, 10);
	const double seconds = std::atof(argv[2]);
	const size_t top     = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 20;
	std::vector<std::pair<DWORD, HANDLE>> threads;
	HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	THREADENTRY32 entry {sizeof(entry)};
	for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
		if (entry.th32OwnerProcessID != pid) continue;
		if (HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID)) {
			threads.emplace_back(entry.th32ThreadID, h);
		}
	}
	CloseHandle(snap);
	std::vector<unsigned long long> before;
	for (auto& t: threads) before.push_back(ThreadCpu(t.second));
	Sleep(static_cast<DWORD>(seconds * 1000));
	std::vector<std::pair<unsigned long long, DWORD>> busy;
	for (size_t i = 0; i < threads.size(); i++) {
		busy.emplace_back(ThreadCpu(threads[i].second) - before[i], threads[i].first);
		CloseHandle(threads[i].second);
	}
	std::sort(busy.rbegin(), busy.rend());
	for (size_t i = 0; i < busy.size() && i < top; i++) {
		std::printf("%lu %.1f\n", busy[i].second, busy[i].first / (seconds * 1e5));
	}
	return 0;
}
