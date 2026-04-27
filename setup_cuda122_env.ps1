param(
    [string]$CondaExe = "conda",
    [string]$CondaEnvName = "vega-cu122",
    [string]$PythonVersion = "3.10",
    [ValidateSet("cu121", "cu124")]
    [string]$TorchCuda = "cu121"
)

$ErrorActionPreference = "Stop"

Write-Host "=== VEGA Conda 环境配置脚本（面向 CUDA 12.2）===" -ForegroundColor Cyan
Write-Host "Conda : $CondaExe"
Write-Host "Env   : $CondaEnvName"
Write-Host "PyVer : $PythonVersion"
Write-Host "Torch : $TorchCuda"

if (-not (Get-Command $CondaExe -ErrorAction SilentlyContinue)) {
    throw "未找到 conda 可执行文件：$CondaExe"
}

Write-Host "`n[1/5] 检查并创建 Conda 环境..." -ForegroundColor Yellow
$envListText = & $CondaExe env list
if ($envListText -notmatch "(^|\s)$CondaEnvName(\s|$)") {
    & $CondaExe create -n $CondaEnvName "python=$PythonVersion" -y
} else {
    Write-Host "Conda 环境已存在，跳过创建。" -ForegroundColor DarkYellow
}

Write-Host "`n[2/5] 升级 pip/setuptools/wheel..." -ForegroundColor Yellow
& $CondaExe run -n $CondaEnvName python -m pip install --upgrade pip setuptools wheel

Write-Host "`n[3/5] 安装 PyTorch CUDA 版本..." -ForegroundColor Yellow
& $CondaExe run -n $CondaEnvName python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TorchCuda"

Write-Host "`n[4/5] 安装 VEGA 依赖..." -ForegroundColor Yellow
& $CondaExe run -n $CondaEnvName python -m pip install numpy pandas scikit-learn tqdm transformers pillow pytz

Write-Host "`n[5/5] 环境自检..." -ForegroundColor Yellow
& $CondaExe run -n $CondaEnvName python -c "import torch, transformers, numpy, pandas, sklearn, PIL, pytz, tqdm; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

Write-Host "`n完成。激活环境命令：" -ForegroundColor Green
Write-Host "conda activate $CondaEnvName"
Write-Host "`n备注："
Write-Host "1) 该脚本默认安装 PyTorch cu121 轮子，在 CUDA 12.2 驱动环境通常可正常运行。"
Write-Host "2) 如果你希望尝试更新轮子，可执行：.\setup_cuda122_env.ps1 -CondaEnvName $CondaEnvName -TorchCuda cu124"
Write-Host "3) 训练前请确认 data/、anchor/、checkpoint/ 资源文件已就位。"
