#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vega-cu122}"
DATASET="${DATASET:-IEMOCAP}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"

NO_CLIP_LOSS="${NO_CLIP_LOSS:-0}"
NO_CLS_LOSS="${NO_CLS_LOSS:-0}"
NO_CLIP_ALL_CLIP_KL_LOSS="${NO_CLIP_ALL_CLIP_KL_LOSS:-0}"
NO_CLS_ALL_CLS_KL_LOSS="${NO_CLS_ALL_CLS_KL_LOSS:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  CONDA_ENV_NAME=vega-cu122 ./run_train.sh

Optional env vars:
  CONDA_EXE=conda
  CONDA_ENV_NAME=vega-cu122
  DATASET=IEMOCAP
  CLIP_MODEL=openai/clip-vit-base-patch32
  NO_CLIP_LOSS=1
  NO_CLS_LOSS=1
  NO_CLIP_ALL_CLIP_KL_LOSS=1
  NO_CLS_ALL_CLS_KL_LOSS=1
EOF
  exit 0
fi

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  echo "错误：未找到 conda 命令：${CONDA_EXE}"
  exit 1
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "错误：未找到 conda 环境：${CONDA_ENV_NAME}，请先运行 ./setup_cuda122_env.sh"
  exit 1
fi

if [[ "${DATASET}" == "IEMOCAP" ]]; then
  required_files=("data/IEMOCAP.pkl" "anchor/35_anchor.pt")
else
  echo "错误：当前脚本仅内置 IEMOCAP 检查，收到 DATASET=${DATASET}"
  exit 1
fi

missing=()
for f in "${required_files[@]}"; do
  if [[ ! -e "${f}" ]]; then
    missing+=("${f}")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "错误：缺少必需文件："
  printf ' - %s\n' "${missing[@]}"
  exit 1
fi

args=(
  "run.py"
  "--Dataset" "${DATASET}"
  "--CLIP_Model" "${CLIP_MODEL}"
)

if [[ "${NO_CLS_LOSS}" == "1" ]]; then args+=("--no_cls_loss"); else args+=("--cls_loss"); fi
if [[ "${NO_CLIP_LOSS}" == "1" ]]; then args+=("--no_clip_loss"); else args+=("--clip_loss"); fi
if [[ "${NO_CLIP_ALL_CLIP_KL_LOSS}" == "1" ]]; then args+=("--no_clip_all_clip_kl_loss"); else args+=("--clip_all_clip_kl_loss"); fi
if [[ "${NO_CLS_ALL_CLS_KL_LOSS}" == "1" ]]; then args+=("--no_cls_all_cls_kl_loss"); else args+=("--cls_all_cls_kl_loss"); fi

echo "使用 Conda: ${CONDA_EXE}"
echo "Conda 环境: ${CONDA_ENV_NAME}"
echo "启动训练命令: ${CONDA_EXE} run -n ${CONDA_ENV_NAME} python ${args[*]}"

"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python "${args[@]}"
