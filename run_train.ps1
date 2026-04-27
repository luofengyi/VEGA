param(
    [string]$CondaExe = "conda",
    [string]$CondaEnvName = "vega-cu122",
    [string]$Dataset = "IEMOCAP",
    [string]$CLIPModel = "openai/clip-vit-base-patch32",
    [switch]$NoClipLoss,
    [switch]$NoClsLoss,
    [switch]$NoClipAllClipKlLoss,
    [switch]$NoClsAllClsKlLoss
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
    param([string]$DatasetName)

    if ($DatasetName -eq "IEMOCAP") {
        $required = @(
            "data/IEMOCAP.pkl",
            "anchor/35_anchor.pt"
        )
    } else {
        throw "当前脚本仅内置 IEMOCAP 检查。传入 Dataset=$DatasetName 暂未支持。"
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
Test-RequiredFiles -DatasetName $Dataset

$args = @(
    "run.py",
    "--Dataset", $Dataset,
    "--CLIP_Model", $CLIPModel
)

if ($NoClsLoss) {
    $args += "--no_cls_loss"
} else {
    $args += "--cls_loss"
}

if ($NoClipLoss) {
    $args += "--no_clip_loss"
} else {
    $args += "--clip_loss"
}

if ($NoClipAllClipKlLoss) {
    $args += "--no_clip_all_clip_kl_loss"
} else {
    $args += "--clip_all_clip_kl_loss"
}

if ($NoClsAllClsKlLoss) {
    $args += "--no_cls_all_cls_kl_loss"
} else {
    $args += "--cls_all_cls_kl_loss"
}

Write-Host "使用 Conda: $CondaExe" -ForegroundColor Cyan
Write-Host "Conda 环境: $CondaEnvName" -ForegroundColor Cyan
Write-Host "启动训练命令: $CondaExe run -n $CondaEnvName python $($args -join ' ')" -ForegroundColor Cyan

& $CondaExe run -n $CondaEnvName python @args
