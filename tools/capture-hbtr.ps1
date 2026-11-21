param(
    [string]$Exe = '.\target\release\hbd-sandbox.exe',
    [string[]]$AppArgs = @('--static-camera'),
    [int]$Frames = 6,
    [int]$IntervalMs = 250,
    [string]$OutDir = '.\hbtr-capture'
)

# Explicit no-ghosting capture harness for HBTR/HBTRU.
# Starts the sandbox, captures its window with PrintWindow (works even when
# occluded), and writes numbered PNGs plus a manifest of the launch arguments.
# Compare consecutive frames by eye for smearing behind moving objects; record
# the result in the PR. This is a capture tool, never part of a render frame.

Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class HbtrCapture {
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
'@

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$manifest = @("exe=$Exe", "args=$($AppArgs -join ' ')", "frames=$Frames", "interval_ms=$IntervalMs")
$proc = Start-Process -FilePath $Exe -ArgumentList $AppArgs -PassThru
try {
    Start-Sleep -Milliseconds 2500
    $handle = [IntPtr]::Zero
    for ($i = 0; $i -lt 30; $i++) {
        $proc.Refresh()
        $handle = $proc.MainWindowHandle
        if ($handle -ne [IntPtr]::Zero) { break }
        Start-Sleep -Milliseconds 100
    }
    if ($handle -eq [IntPtr]::Zero) { throw 'launched process has no window' }

    for ($i = 0; $i -lt $Frames; $i++) {
        $rect = New-Object HbtrCapture+RECT
        [HbtrCapture]::GetWindowRect($handle, [ref]$rect) | Out-Null
        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        $bitmap = New-Object System.Drawing.Bitmap $width, $height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $hdc = $graphics.GetHdc()
        [HbtrCapture]::PrintWindow($handle, $hdc, 2) | Out-Null
        $graphics.ReleaseHdc($hdc)
        $path = Join-Path $OutDir ("frame_{0:D3}.png" -f $i)
        $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bitmap.Dispose()
        Start-Sleep -Milliseconds $IntervalMs
    }
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

$manifest | Set-Content (Join-Path $OutDir 'manifest.txt')
Write-Host "captured $Frames frames to $OutDir"
