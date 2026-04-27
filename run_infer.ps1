param(
    [string]$CondaExe = "conda",
    [string]$CondaEnvName = "vega-cu122",
    [string]$Checkpoint = "checkpoint/IEMOCAP.pth",
    [string]$Dataset = "IEMOCAP",
    [int]$BatchSize = 9,
    [int]$NumWorkers = 0,
    [switch]$Cpu,
    [string]$ExprImgFolder = "35",
    [double]$Rand = 0.4
)

$ErrorActionPreference = "Stop"

function Test-CondaReady {
    param(
        [string]$CondaCommand,
        [string]$EnvName
    )

    if (-not (Get-Command $CondaCommand -ErrorAction SilentlyContinue)) {
        throw "未找到 conda：$CondaCommand"
    }

    $envListText = & $CondaCommand env list
    if ($envListText -notmatch "(^|\s)$EnvName(\s|$)") {
        throw "未找到 conda 环境：$EnvName。请先运行 setup_cuda122_env.ps1 创建环境。"
    }
}

function Test-RequiredFiles {
    param(
        [string]$DatasetName,
        [string]$CheckpointPath,
        [string]$ExprFolder
    )

    $required = @($CheckpointPath)

    if ($DatasetName -eq "IEMOCAP") {
        $required += "data/IEMOCAP.pkl"
    } elseif ($DatasetName -eq "MELD") {
        $required += "data/MELD.pkl"
    } else {
        throw "不支持的数据集：$DatasetName（仅支持 IEMOCAP / MELD）"
    }

    $anchorPt = "anchor/${ExprFolder}_anchor.pt"
    $anchorDirA = "anchor/$ExprFolder"
    $anchorDirB = "anchor/${ExprFolder}_anchor"

    if ((Test-Path $anchorPt) -or (Test-Path $anchorDirA) -or (Test-Path $anchorDirB)) {
        # anchor resource exists, pass
    } else {
        $required += $anchorPt
    }

    $missing = @()
    foreach ($file in $required) {
        if (-not (Test-Path $file)) {
            $missing += $file
        }
    }

    if ($missing.Count -gt 0) {
        throw ("缺少必需文件：`n - " + ($missing -join "`n - "))
    }
}

Test-CondaReady -CondaCommand $CondaExe -EnvName $CondaEnvName
Test-RequiredFiles -DatasetName $Dataset -CheckpointPath $Checkpoint -ExprFolder $ExprImgFolder

$args = @(
    "inference.py",
    "--checkpoint", $Checkpoint,
    "--dataset", $Dataset,
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--expr_img_folder", $ExprImgFolder,
    "--rand", "$Rand",
    "--clip_loss"
)

if ($Cpu) {
    $args += "--cpu"
}

Write-Host "使用 Conda: $CondaExe" -ForegroundColor Cyan
Write-Host "Conda 环境: $CondaEnvName" -ForegroundColor Cyan
Write-Host "启动推理命令: $CondaExe run -n $CondaEnvName python $($args -join ' ')" -ForegroundColor Cyan

& $CondaExe run -n $CondaEnvName python @args
