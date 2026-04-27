#!/usr/bin/env bash
set -euo pipefail

# One-click runner for VEGA main + ablation experiments.
# Outputs:
#   output/paper_suite/logs/*.log
#   output/paper_suite/summary/metrics.csv
#   output/paper_suite/summary/metrics.json
#   output/paper_suite/figures/anchor_similarity_tsne.png
#   output/paper_suite/tables/*.md
#   output/paper_suite/tables/*.tex

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vega-cu122}"
DATASET="${DATASET:-IEMOCAP}"              # IEMOCAP or MELD
SEED="${SEED:-64}"
BATCH_SIZE="${BATCH_SIZE:-9}"
EPOCHS="${EPOCHS:-400}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
EXPR_IMG_FOLDER="${EXPR_IMG_FOLDER:-35}"
OUT_ROOT="${OUT_ROOT:-output/paper_suite}"
RUN_VIS="${RUN_VIS:-1}"                    # 1: generate figure, 0: skip
VIS_CHECKPOINT="${VIS_CHECKPOINT:-checkpoint/IEMOCAP.pth}"
RUN_TABLE="${RUN_TABLE:-1}"                # 1: generate markdown/latex tables, 0: skip
BASELINE_JSON="${BASELINE_JSON:-}"         # optional baseline json path for comparison table

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  CONDA_ENV_NAME=vega-cu122 DATASET=IEMOCAP ./run_paper_experiments.sh

Optional env vars:
  CONDA_EXE=conda
  CONDA_ENV_NAME=vega-cu122
  DATASET=IEMOCAP|MELD
  SEED=64
  BATCH_SIZE=9
  EPOCHS=400
  CLIP_MODEL=openai/clip-vit-base-patch32
  EXPR_IMG_FOLDER=35
  OUT_ROOT=output/paper_suite
  RUN_VIS=1
  VIS_CHECKPOINT=checkpoint/IEMOCAP.pth
  RUN_TABLE=1
  BASELINE_JSON=baseline_iemocap.json
EOF
  exit 0
fi

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  echo "错误：未找到 conda 命令：${CONDA_EXE}"
  exit 1
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "错误：未找到 conda 环境：${CONDA_ENV_NAME}"
  exit 1
fi

if [[ ! -f "run.py" ]]; then
  echo "错误：请在仓库根目录执行该脚本（run.py 同级）"
  exit 1
fi

LOG_DIR="${OUT_ROOT}/logs"
SUMMARY_DIR="${OUT_ROOT}/summary"
FIG_DIR="${OUT_ROOT}/figures"
TABLE_DIR="${OUT_ROOT}/tables"
mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${FIG_DIR}" "${TABLE_DIR}"

run_exp() {
  local exp_name="$1"
  shift
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "==============================================================="
  echo "[run] ${exp_name}"
  echo "[run] log: ${log_file}"
  echo "==============================================================="

  "${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python run.py \
    --name "${exp_name}" \
    --Dataset "${DATASET}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --expr_img_folder "${EXPR_IMG_FOLDER}" \
    --CLIP_Model "${CLIP_MODEL}" \
    "$@" 2>&1 | tee "${log_file}"
}

# 1) Full VEGA
run_exp "full_vega" \
  --cls_loss --clip_loss --clip_all_clip_kl_loss --cls_all_cls_kl_loss

# 2) w/o VEGA Branch (degrades to backbone/SDT-like setup)
run_exp "wo_vega_branch" \
  --cls_loss --cls_all_cls_kl_loss --no_clip_loss --no_clip_all_clip_kl_loss

# 3) w/o Stochastic Anchor Sampling
run_exp "wo_stochastic_anchor_sampling" \
  --cls_loss --clip_loss --clip_all_clip_kl_loss --cls_all_cls_kl_loss \
  --rand 0.0

# 4) w/o Unimodal Anchoring
run_exp "wo_unimodal_anchoring" \
  --cls_loss --clip_loss --clip_all_clip_kl_loss --cls_all_cls_kl_loss \
  --a_clip_lambda 0 --t_clip_lambda 0 --v_clip_lambda 0 \
  --a_clip_all_clip_kl_lambda 0 --t_clip_all_clip_kl_lambda 0 --v_clip_all_clip_kl_lambda 0

# 5) w/o Multimodal Anchoring
run_exp "wo_multimodal_anchoring" \
  --cls_loss --clip_loss --clip_all_clip_kl_loss --cls_all_cls_kl_loss \
  --all_clip_lambda 0

# 6) w/o Self-Distillation in VEGA
run_exp "wo_self_distillation_in_vega" \
  --cls_loss --clip_loss --no_clip_all_clip_kl_loss --cls_all_cls_kl_loss

echo "[collect] Parsing logs..."
"${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python tools/collect_vega_metrics.py \
  --log_dir "${LOG_DIR}" \
  --out_csv "${SUMMARY_DIR}/metrics.csv" \
  --out_json "${SUMMARY_DIR}/metrics.json"

if [[ "${RUN_VIS}" == "1" ]]; then
  echo "[plot] Generating anchor similarity figure..."
  "${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python tools/plot_anchor_similarity.py \
    --checkpoint "${VIS_CHECKPOINT}" \
    --dataset "${DATASET}" \
    --expr_img_folder "${EXPR_IMG_FOLDER}" \
    --output_png "${FIG_DIR}/anchor_similarity_tsne.png"
fi

if [[ "${RUN_TABLE}" == "1" ]]; then
  echo "[table] Generating markdown/latex tables..."
  table_args=(
    "tools/make_table.py"
    "--metrics_csv" "${SUMMARY_DIR}/metrics.csv"
    "--dataset" "${DATASET}"
    "--out_dir" "${TABLE_DIR}"
  )
  if [[ -n "${BASELINE_JSON}" ]]; then
    table_args+=("--baseline_json" "${BASELINE_JSON}")
  fi
  "${CONDA_EXE}" run -n "${CONDA_ENV_NAME}" python "${table_args[@]}"
fi

echo
echo "完成："
echo "  日志目录   : ${LOG_DIR}"
echo "  指标 CSV   : ${SUMMARY_DIR}/metrics.csv"
echo "  指标 JSON  : ${SUMMARY_DIR}/metrics.json"
if [[ "${RUN_VIS}" == "1" ]]; then
  echo "  可视化图片 : ${FIG_DIR}/anchor_similarity_tsne.png"
fi
if [[ "${RUN_TABLE}" == "1" ]]; then
  echo "  表格目录   : ${TABLE_DIR}"
fi
