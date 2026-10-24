# Prepares the optional Windows Vulkan SDK from already-downloaded, pinned release archives.
# It never downloads or launches the external emulator. Only two FSR runtime DLLs are extracted.
param(
    [Parameter(Mandatory)][string]$SdkArchive,
    [Parameter(Mandatory)][string]$RuntimeArchive,
    [Parameter(Mandatory)][string]$Destination,
    [Parameter(Mandatory)][string]$LlvmBin
)
$ErrorActionPreference = 'Stop'
if ((Get-FileHash -LiteralPath $SdkArchive -Algorithm SHA256).Hash -ne '0216556BFB0E243CEC30004A2A98D38F4E3F7406CB7938E3C1B85C758E95D952') {
    throw 'Expected the official FidelityFX-SDK-v1.1.4.zip archive; SHA256 differs.'
}
if ((Get-FileHash -LiteralPath $RuntimeArchive -Algorithm SHA256).Hash -ne 'B8DA12ACEDB534A2B5EFDBBDE47435489FB29823734C5608CFACFE1CF2E1403A') {
    throw 'Expected DLSSKytyPS5-v1.5.0-Windows-x64.zip; SHA256 differs.'
}
if (Test-Path -LiteralPath $Destination) { throw 'Destination must be a new directory.' }
$readobj = Join-Path $LlvmBin 'llvm-readobj.exe'
$lib = Join-Path $LlvmBin 'llvm-lib.exe'
if (!(Test-Path -LiteralPath $readobj) -or !(Test-Path -LiteralPath $lib)) { throw 'LLVM tools not found.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory((Resolve-Path -LiteralPath $SdkArchive).Path, $Destination)
$bin = Join-Path $Destination 'sdk/bin/ffx_sdk'
New-Item -ItemType Directory -Path $bin -Force | Out-Null
$zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $RuntimeArchive).Path)
try {
    foreach ($component in @('backend_vk', 'fsr3upscaler')) {
        $base = "ffx_${component}_x64"
        $entry = $zip.Entries | Where-Object { $_.FullName -eq "DLSSKytyPS5-v1.5.0-Windows-x64/$base.dll" }
        if (!$entry) { throw "Missing runtime: $base" }
        $dll = Join-Path $bin "$base.dll"
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dll, $false)
        $raw = & $readobj --coff-exports $dll
        if ($LASTEXITCODE -ne 0) { throw "Cannot read exports: $base" }
        $exports = $raw | ForEach-Object { if ($_ -match '^  Name: (.+)$') { $Matches[1] } }
        $def = Join-Path $bin "$base.def"
        @("LIBRARY $base.dll", 'EXPORTS') + $exports | Set-Content -LiteralPath $def -Encoding ascii
        & $lib /machine:x64 "/def:$def" "/out:$(Join-Path $bin "$base.lib")"
        if ($LASTEXITCODE -ne 0) { throw "Cannot create import library: $base" }
    }
} finally { $zip.Dispose() }
Write-Output "FSR3 Vulkan SDK prepared at $Destination"
