$ErrorActionPreference = "Stop"
$root = Resolve-Path .
$py = "D:\Anaconda\envs\pytorch\python.exe"
$project = "outputs/experiments/multiseed_cp_catf_paper_mode"
$data = "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/data.yaml"
$probe = "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/probe.yaml"
$logs = Join-Path $root "outputs\experiments\multiseed_cp_catf_paper_mode\logs"
$statusPath = Join-Path $logs "full_run_status.json"
$statuses = @()
function Write-Status($runId, $status, $exitCode) {
  $script:statuses += [ordered]@{ run_id = $runId; status = $status; exit_code = $exitCode; timestamp = (Get-Date).ToString("s") }
  $script:statuses | ConvertTo-Json -Depth 4 | Set-Content -Path $statusPath -Encoding UTF8
}
function Invoke-Run($runId, [string[]]$extraArgs) {
  $log = Join-Path $logs "$runId.log"
  Write-Status $runId "running" $null
  $common = @(
    "scripts/train_yolo_default_with_inloop_feedback.py",
    "--model", "yolo11n.pt",
    "--data", $data,
    "--epochs", "50",
    "--imgsz", "1024",
    "--batch", "2",
    "--workers", "0",
    "--device", "0",
    "--project", $project,
    "--run-id", $runId,
    "--skip-doc-update"
  )
  & $py @common @extraArgs 2>&1 | Tee-Object -FilePath $log
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    Write-Status $runId "failed" $code
    exit $code
  }
  Write-Status $runId "completed" $code
}
foreach ($seed in 0,1,2) {
  Invoke-Run "clean_seed_$seed" @("--seed", "$seed", "--feedback-enabled", "false", "--industrial-aug-enabled", "false", "--save-preview", "false")
}
foreach ($seed in 0,1,2) {
  Invoke-Run "cp_catf_seed_$seed" @(
    "--seed", "$seed",
    "--probe-data", $probe,
    "--train-core-data", $data,
    "--catf-version", "v2",
    "--paper-probe-mode", "true",
    "--causal-probe-mode", "true",
    "--class-aware-feedback", "true",
    "--roi-aware-aug", "true",
    "--sample-aware-routing", "true",
    "--threshold-calibration-report", "true",
    "--top-k-active-classes", "2",
    "--top-m-ops-per-class", "2",
    "--feedback-enabled", "true",
    "--feedback-interval", "5",
    "--feedback-start-epoch", "5",
    "--industrial-aug-enabled", "true",
    "--forbid-final-val-policy-selection", "true"
  )
}
Write-Status "all" "completed" 0
