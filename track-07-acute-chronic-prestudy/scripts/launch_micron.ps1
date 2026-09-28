# MICRON: 4 seeds on MIMIC-III chrono first; when their manifests exist, 3 seeds on MIMIC-IV.
$env:PYTHONIOENCODING = 'utf-8'
$py = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
$root = 'C:\Users\Administrator\Desktop\acute-chronic-prestudy'
$script = "$root\scripts\micron_train_dump.py"
$log = "$root\logs\launch_micron.log"
"$(Get-Date -Format s) launching MIMIC-III chrono MICRON seeds" | Out-File $log -Encoding utf8
$m3 = @(@(1203, 2048, 'seed0'), @(1, 1, 'seed1'), @(2, 2, 'seed2'), @(3, 3, 'seed3'))
foreach ($j in $m3) {
    $ts = $j[0]; $ns = $j[1]; $rid = $j[2]
    $p = Start-Process -FilePath $py -ArgumentList @($script, '--dataset', 'mimic3chrono', '--torch-seed', "$ts", '--numpy-seed', "$ns", '--run-id', $rid, '--threads', '1') -WorkingDirectory $root -RedirectStandardOutput "$root\logs\micron_mimic3chrono_$rid.log" -RedirectStandardError "$root\logs\micron_mimic3chrono_$rid.err.log" -WindowStyle Hidden -PassThru
    "$(Get-Date -Format s) started mimic3chrono $rid pid $($p.Id)" | Out-File $log -Append -Encoding utf8
}
$need = 0..3 | ForEach-Object { "$root\out\mimic3_chrono_micron\seed$_\run_manifest.json" }
while ($true) {
    $done = ($need | Where-Object { Test-Path $_ }).Count
    $alive = (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*micron_train_dump.py*' -and $_.CommandLine -like '*--dataset mimic3chrono *' -and $_.Name -like 'python*' } | Measure-Object).Count
    if ($done -eq 4 -or $alive -eq 0) { break }
    Start-Sleep -Seconds 60
}
"$(Get-Date -Format s) mimic3 done=$done alive=$alive; launching MIMIC-IV seeds" | Out-File $log -Append -Encoding utf8
foreach ($ts in @(1203, 2207, 3319)) {
    $rid = "seed-$ts"
    $p = Start-Process -FilePath $py -ArgumentList @($script, '--dataset', 'mimic4', '--torch-seed', "$ts", '--numpy-seed', '2048', '--run-id', $rid, '--threads', '2') -WorkingDirectory $root -RedirectStandardOutput "$root\logs\micron_mimic4_$rid.log" -RedirectStandardError "$root\logs\micron_mimic4_$rid.err.log" -WindowStyle Hidden -PassThru
    "$(Get-Date -Format s) started mimic4 $rid pid $($p.Id)" | Out-File $log -Append -Encoding utf8
}
