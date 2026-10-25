@echo off
rem Builds sampler.exe and threadcpu.exe with clang-cl (on PATH) in a Visual Studio x64 developer environment.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
clang-cl /nologo /O2 /std:c++20 /EHsc sampler.cpp /Fe:sampler.exe /link dbghelp.lib winmm.lib || exit /b 1
clang-cl /nologo /O2 /std:c++20 /EHsc threadcpu.cpp /Fe:threadcpu.exe || exit /b 1
echo SAMPLER_OK
