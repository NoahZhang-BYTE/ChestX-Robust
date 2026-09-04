param(
    [Parameter(Mandatory = $true)][int]$TrainingPid,
    [Parameter(Mandatory = $true)][string]$OutputCsv
)

$ErrorActionPreference = 'SilentlyContinue'
$records = [System.Collections.Generic.List[object]]::new()

while (Get-Process -Id $TrainingPid) {
    $samples = Get-Counter -Counter '\Memory\Committed Bytes', '\Memory\Commit Limit', '\Memory\Available MBytes'
    if ($samples) {
        $counterSamples = $samples.CounterSamples
        $commit = [double](($counterSamples | Where-Object Path -like '*committed bytes').CookedValue)
        $limit = [double](($counterSamples | Where-Object Path -like '*commit limit').CookedValue)
        $available = [double](($counterSamples | Where-Object Path -like '*available mbytes').CookedValue)
        $process = Get-Process -Id $TrainingPid
        $record = [pscustomobject]@{
            timestamp = (Get-Date).ToString('o')
            commit_used_gb = [math]::Round($commit / 1GB, 3)
            commit_limit_gb = [math]::Round($limit / 1GB, 3)
            commit_pct = [math]::Round(100 * $commit / $limit, 3)
            ram_available_gb = [math]::Round($available / 1024, 3)
            process_private_gb = [math]::Round($process.PrivateMemorySize64 / 1GB, 3)
        }
        $records.Add($record) | Out-Null
        $records | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
        if ($record.commit_pct -gt 80) {
            Stop-Process -Id $TrainingPid -Force
            break
        }
    }
    Start-Sleep -Seconds 15
}

if ($records.Count -gt 0) {
    $records | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
}
