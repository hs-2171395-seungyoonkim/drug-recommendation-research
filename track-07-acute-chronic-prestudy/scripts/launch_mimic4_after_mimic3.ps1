# Waits until the four MIMIC-III chrono GAMENet runs have written run_manifest.json,
# then launches three MIMIC-IV GAMENet seeds (1203, 2207, 3319) detached.
$env:PYTHONIOENCODING = 'utf-8'
$py = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
$root = 'C:\Users\Administrator\Desktop\acute-chronic-prestudy'
$script = "$root\scripts\gamenet_train_dump.py"
$need = 0..3 | ForEach-Object { "$root\out\mimic3_chrono_gamenet\seed$_\run_manifest.json" }
$log = "$root\logs\launch_mimic4_after_mimic3.log"
"$(Get-Date -Format s) waiting for MIMIC-III chrono GAMENet manifests" | Out-File $log -Encoding utf8
while ($true) {
    $done = ($need | Where-Object { Test-Path $_ }).Count
    $alive = (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*gamenet_train_dump.py*' -and $_.CommandLine -like '*--dataset mimic3chrono *' -and $_.Name -like 'python*' } | Measure-Object).Count
    if ($done -eq 4 -or $alive -eq 0) { break }
    Start-Sleep -Seconds 120
}
"$(Get-Date -Format s) mimic3 done=$done alive=$alive; launching MIMIC-IV seeds" | Out-File $log -Append -Encoding utf8
foreach ($ts in @(1203, 2207, 3319)) {
    $rid = "seed-$ts"
    $out = "$root\logs\gamenet_mimic4_$rid.log"; $err = "$root\logs\gamenet_mimic4_$rid.err.log"
    $p = Start-Process -FilePath $py -ArgumentList @($script, '--dataset', 'mimic4', '--torch-seed', "$ts", '--numpy-seed', '2048', '--run-id', $rid, '--threads', '2') -WorkingDirectory $root -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    "$(Get-Date -Format s) started mimic4 $rid pid $($p.Id)" | Out-File $log -Append -Encoding utf8
}
