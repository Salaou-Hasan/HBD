param(
    [Parameter(Mandatory = $true)][string]$ReferenceDir,
    [Parameter(Mandatory = $false)][string]$TestDir = $ReferenceDir,
    [string]$OutCsv = ''
)

# Compares HBTRU capture sequences against a native/AA-off reference.
# Inputs are PNG sequences produced by tools/capture-hbtr.ps1. The script
# reports per-frame mean absolute error, p95 pixel error, and an edge-weighted
# "ghost score" that emphasises differences around high-contrast edges, where
# temporal smearing is visible. Lower is better; a ghost-free pass should score
# close to the native reference noise floor.

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Runtime.InteropServices

$refFiles = @(Get-ChildItem -Path $ReferenceDir -Filter 'frame_*.png' | Sort-Object Name)
$testFiles = @(Get-ChildItem -Path $TestDir -Filter 'frame_*.png' | Sort-Object Name)
if ($refFiles.Count -eq 0) { throw "no frame_*.png in $ReferenceDir" }
if ($refFiles.Count -ne $testFiles.Count) {
    throw "frame count mismatch: reference=$($refFiles.Count) test=$($testFiles.Count)"
}

$rows = New-Object System.Collections.Generic.List[string]
$rows.Add('frame,mae,p95,ghost_score')
$sumMae = 0.0
$sumGhost = 0.0
$sumP95 = 0.0

function Read-Pixels([string]$Path, [ref]$Width, [ref]$Height) {
    $bitmap = [System.Drawing.Bitmap]::FromFile($Path)
    $w = $bitmap.Width
    $h = $bitmap.Height
    $rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
    $data = $bitmap.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, $bitmap.PixelFormat)
    $stride = [Math]::Abs($data.Stride)
    $bytes = New-Object byte[] ($stride * $h)
    [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
    $bitmap.UnlockBits($data)
    $bitmap.Dispose()
    $Width.Value = $w
    $Height.Value = $h
    return ,@($bytes, $stride, $w, $h)
}

for ($i = 0; $i -lt $refFiles.Count; $i++) {
    $rw = 0; $rh = 0; $tw = 0; $th = 0
    $ref = Read-Pixels $refFiles[$i].FullName ([ref]$rw) ([ref]$rh)
    $test = Read-Pixels $testFiles[$i].FullName ([ref]$tw) ([ref]$th)
    if ($rw -ne $tw -or $rh -ne $th) {
        throw "size mismatch on frame ${i}: reference=${rw}x${rh} test=${tw}x${th}"
    }

    $refBytes = $ref[0]; $refStride = $ref[1]
    $testBytes = $test[0]; $testStride = $test[1]
    $histogram = New-Object int[] 256
    $diffSum = 0.0
    $weightedSum = 0.0
    $weightSum = 0.0
    $count = 0

    for ($y = 0; $y -lt ($rh - 1); $y++) {
        for ($x = 0; $x -lt ($rw - 1); $x++) {
            $ri = $y * $refStride + $x * 4
            $ti = $y * $testStride + $x * 4
            $dr = [Math]::Abs([int]$refBytes[$ri + 2] - [int]$testBytes[$ti + 2])
            $dg = [Math]::Abs([int]$refBytes[$ri + 1] - [int]$testBytes[$ti + 1])
            $db = [Math]::Abs([int]$refBytes[$ri] - [int]$testBytes[$ti])
            $diff = ($dr + $dg + $db) / (3.0 * 255.0)
            $diffSum += $diff
            $bin = [int][Math]::Min(255, [int]($diff * 255.0))
            $histogram[$bin] = $histogram[$bin] + 1
            $count++

            # Reference edge magnitude from two neighbours (luma proxy).
            $rx = $ri + 4
            $ry = $ri + $refStride
            $lum = 0.2126 * $refBytes[$ri + 2] + 0.7152 * $refBytes[$ri + 1] + 0.0722 * $refBytes[$ri]
            $lumX = 0.2126 * $refBytes[$rx + 2] + 0.7152 * $refBytes[$rx + 1] + 0.0722 * $refBytes[$rx]
            $lumY = 0.2126 * $refBytes[$ry + 2] + 0.7152 * $refBytes[$ry + 1] + 0.0722 * $refBytes[$ry]
            $edge = [Math]::Min(1.0, ([Math]::Abs($lumX - $lum) + [Math]::Abs($lumY - $lum)) / 64.0)
            $weightedSum += $diff * $edge
            $weightSum += $edge
        }
    }

    $mae = $diffSum / [Math]::Max(1, $count)
    $target = [int][Math]::Floor($count * 0.95)
    $running = 0
    $p95 = 0.0
    for ($bin = 0; $bin -lt 256; $bin++) {
        $running += $histogram[$bin]
        if ($running -ge $target) { $p95 = $bin / 255.0; break }
    }
    $ghost = $weightedSum / [Math]::Max(0.0001, $weightSum)
    $rows.Add(("{0},{1:N6},{2:N6},{3:N6}" -f $i, $mae, $p95, $ghost))
    $sumMae += $mae
    $sumP95 += $p95
    $sumGhost += $ghost
}

$n = $refFiles.Count
Write-Host ("frames=$n  mean MAE={0:N6}  mean p95={1:N6}  mean ghost_score={2:N6}" -f ($sumMae / $n), ($sumP95 / $n), ($sumGhost / $n))
if ($OutCsv -ne '') {
    $rows | Set-Content $OutCsv
    Write-Host "wrote $OutCsv"
}
