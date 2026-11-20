param(
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,
    [double]$MaxGpuP99Ms = 8.0,
    [double]$MaxSsaoP99Ms = 2.5,
    [double]$MaxGeometryP99Ms = 2.5,
    [double]$MaxShadowP99Ms = 0.8,
    [double]$MaxBloomP99Ms = 1.0,
    [double]$MaxTonemapP99Ms = 1.0
)

if (-not (Test-Path $CsvPath)) {
    throw "profile CSV not found: $CsvPath"
}

$lines = Get-Content $CsvPath
if ($lines.Count -lt 2) {
    throw "profile CSV has no data rows: $CsvPath"
}

$header = @($lines[0] -split ',')
$index = @{}
for ($i = 0; $i -lt $header.Count; $i++) {
    $index[$header[$i].Trim()] = $i
}

$required = @('pass_frame', 'pass_ssao', 'pass_geometry', 'pass_shadow', 'pass_bloom', 'pass_tonemap')
foreach ($name in $required) {
    if (-not $index.ContainsKey($name)) {
        throw "profile CSV is missing required column: $name"
    }
}

$values = @{}
foreach ($name in $required) { $values[$name] = New-Object System.Collections.Generic.List[double] }
foreach ($line in $lines[1..($lines.Count - 1)]) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = @($line -split ',')
    foreach ($name in $required) {
        $values[$name].Add([double]$parts[$index[$name]]) | Out-Null
    }
}

function Get-Percentile {
    param([System.Collections.Generic.List[double]]$samples, [double]$fraction)
    if ($samples.Count -eq 0) { return 0.0 }
    $arr = @($samples.ToArray())
    $sorted = $arr | Sort-Object
    $position = [int][Math]::Round(($sorted.Count - 1) * $fraction)
    return [double]$sorted[$position]
}

$results = @(
    [pscustomobject]@{ Pass = 'frame'; P99 = (Get-Percentile $values['pass_frame'] 0.99); Limit = $null }
    [pscustomobject]@{ Pass = 'ssao'; P99 = (Get-Percentile $values['pass_ssao'] 0.99); Limit = $MaxSsaoP99Ms }
    [pscustomobject]@{ Pass = 'geometry'; P99 = (Get-Percentile $values['pass_geometry'] 0.99); Limit = $MaxGeometryP99Ms }
    [pscustomobject]@{ Pass = 'shadow'; P99 = (Get-Percentile $values['pass_shadow'] 0.99); Limit = $MaxShadowP99Ms }
    [pscustomobject]@{ Pass = 'bloom'; P99 = (Get-Percentile $values['pass_bloom'] 0.99); Limit = $MaxBloomP99Ms }
    [pscustomobject]@{ Pass = 'tonemap'; P99 = (Get-Percentile $values['pass_tonemap'] 0.99); Limit = $MaxTonemapP99Ms }
)

$results | Format-Table -AutoSize

$failed = @()
foreach ($result in $results) {
    if ($null -ne $result.Limit -and $result.P99 -gt $result.Limit) {
        $failed += "$($result.Pass): $($result.P99) ms > $($result.Limit) ms"
    }
}
if ($results[0].P99 -gt $MaxGpuP99Ms) {
    $failed += "frame: $($results[0].P99) ms > $MaxGpuP99Ms ms"
}

if ($failed.Count -gt 0) {
    Write-Host "PROFILE CHECK FAILED" -ForegroundColor Red
    foreach ($failure in $failed) { Write-Host "  $failure" -ForegroundColor Red }
    exit 1
}

Write-Host "PROFILE CHECK PASSED" -ForegroundColor Green
exit 0
