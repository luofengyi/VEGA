#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vega-cu122}"
CHECKPOINT="${CHECKPOINT:-checkpoint/IEMOCAP.pth}"
DATASET="${DATASET:-IEMOCAP}"
BATCH_SIZE="${BATCH_SIZE:-9}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CPU="${CPU:-0}"
EXPR_IMG_FOLDER="${EXPR_IMG_FOLDER:-35}"
RAND="${RAND:-0.4}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  CONDA_ENV_NAME=vega-cu122 ./run_infer.sh

Optional env vars:
  CONDA_EXE=conda
  CONDA_ENV_NAME=vega-cu122
  CHECKPOINT=checkpoint/IEMOCAP.pth
  DATASET=IEMOCAP
  BATCH_SIZE=9
  NUM_WORKERS=0
  CPU=1
  EXPR_IMG_FOLDER=35
  RAND=0.4
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

required_files=("${CHECKPOINT}")
if [[ "${DATASET}" == "IEMOCAP" ]]; then
  required_files+=("data/IEMOCAP.pkl")
elif [[ "${DATASET}" == "MELD" ]]; then
  required_files+=("data/MELD.pkl")
else
  echo "错误：不支持的数据集 ${DATASET}（仅支持 IEMOCAP / MELD）"
  exit 1
fi

anchor_pt="anchor/${EXPR_IMG_FOLDER}_anchor.pt"
anchor_dir_a="anchor/${EXPR_IMG_FOLDER}"
anchor_dir_b="anchor/${EXPR_IMG_FOLDER}_anchor"
if [[ ! -e "${anchor_pt}" && ! -d "${anchor_dir_a}" && ! -d "${anchor_dir_b}" ]]; then
  required_files+=("${anchor_pt}")
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
  "inference.py"
  "--checkpoint" "${CHECKPOINT}"
  "--dataset" "${DATASET}"
  "--batch_size" "${BATCH_SIZE}"
  "--num_workers" "${NUM_WORKERS}"
  "--expr_img_folder" "${EXPR_IMG_FOLDER}"
  "--rand" "${RAND}"
  "--clip_loss"
)

if [[ "${CPU}" == "1" ]]; then
  args+=("--cpu")
fi

echo "使用 Conda: ${CONDA_EXE}"
echo "Conda 环境: ${CONDA_ENV_NAME}"
echo "启动推理命令: ${CONDA_EXE} run -n ${CONDA_ENV_NAME} python ${args[*]}"

"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python "${args[@]}"
