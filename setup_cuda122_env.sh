#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vega-cu122}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu121}" # cu121 or cu124

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  CONDA_ENV_NAME=vega-cu122 PYTHON_VERSION=3.10 TORCH_CUDA=cu121 ./setup_cuda122_env.sh

Environment variables:
  CONDA_EXE       Conda executable (default: conda)
  CONDA_ENV_NAME  Conda env name (default: vega-cu122)
  PYTHON_VERSION  Python version (default: 3.10)
  TORCH_CUDA      PyTorch CUDA index tag: cu121 or cu124 (default: cu121)
EOF
  exit 0
fi

echo "=== VEGA Conda 环境配置脚本（Linux/CUDA 12.2）==="
echo "Conda : ${CONDA_EXE}"
echo "Env   : ${CONDA_ENV_NAME}"
echo "PyVer : ${PYTHON_VERSION}"
echo "Torch : ${TORCH_CUDA}"

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  echo "错误：未找到 conda 命令：${CONDA_EXE}"
  exit 1
fi

echo
echo "[1/5] 检查并创建 Conda 环境..."
if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  "${CONDA_EXE}" create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "Conda 环境已存在，跳过创建。"
fi

echo
echo "[2/5] 升级 pip/setuptools/wheel..."
"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo
echo "[3/5] 安装 PyTorch CUDA 版本..."
"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo
echo "[4/5] 安装 VEGA 依赖..."
"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python -m pip install numpy pandas scikit-learn tqdm transformers pillow pytz

echo
echo "[5/5] 环境自检..."
"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python -c "import torch, transformers, numpy, pandas, sklearn, PIL, pytz, tqdm; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"

echo
echo "完成。常用命令："
echo "  conda activate ${CONDA_ENV_NAME}"
echo "备注："
echo "1) 默认安装 cu121 轮子，在 CUDA 12.2 驱动环境通常可运行。"
echo "2) 若要尝试更新轮子，可用 TORCH_CUDA=cu124 ./setup_cuda122_env.sh"
echo "3) 训练前请确认 data/、anchor/、checkpoint/ 资源文件已就位。"
