#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="python"
DATASET="IEMOCAP"
SEED=24
EPOCHS=0
BATCH_SIZE=0
DRY_RUN=0
STOP_ON_ERROR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-exe)
      PYTHON_EXE="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --stop-on-error)
      STOP_ON_ERROR=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/run_ablation_graph.sh [options]

Options:
  --python-exe <path>     Python executable (default: python)
  --dataset <name>        Dataset name (default: IEMOCAP)
  --seed <int>            Random seed (default: 24)
  --epochs <int>          Override epochs when >0 (default: 0, keep config)
  --batch-size <int>      Override batch size when >0 (default: 0, keep config)
  --dry-run               Print commands only, no training
  --stop-on-error         Stop immediately when one experiment fails
  -h, --help              Show this help
EOF
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1 ;;
  esac
done

timestamp="$(date +"%Y%m%d-%H%M%S")"
log_dir="output/ablation_logs/${timestamp}"
mkdir -p "${log_dir}"
summary_csv="${log_dir}/summary.csv"

echo "Ablation suite: Baseline / +SimpleGraph / +PyG(noCL) / +PyG(CL)"
echo "Dataset: ${DATASET} | Seed: ${SEED} | DryRun: ${DRY_RUN} | StopOnError: ${STOP_ON_ERROR}"

echo "experiment,status,exit_code,best_cls_f1,duration_sec,log_path,command,started_at,finished_at" > "${summary_csv}"

run_experiment() {
  local title="$1"
  local log_path="$2"
  shift 2
  local extra_args=("$@")

  local base_args=("run.py" "--Dataset" "${DATASET}" "--seed" "${SEED}")
  if [[ "${EPOCHS}" -gt 0 ]]; then
    base_args+=("--epochs" "${EPOCHS}")
  fi
  if [[ "${BATCH_SIZE}" -gt 0 ]]; then
    base_args+=("--batch-size" "${BATCH_SIZE}")
  fi

  local cmd=("${PYTHON_EXE}" "${base_args[@]}" "${extra_args[@]}")
  local cmd_text="${cmd[*]}"

  echo ""
  echo "==================== ${title} ===================="
  echo "${cmd_text}"
  echo "Log: ${log_path}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '"%s","dry_run","","","","%s","%s","",""\n' \
      "${title}" "${log_path}" "${cmd_text//\"/\"\"}" >> "${summary_csv}"
    return 0
  fi

  local started_at finished_at
  started_at="$(date +"%Y-%m-%dT%H:%M:%S")"
  local start_ts end_ts duration exit_code status best_cls_f1
  start_ts=$(date +%s)

  set +e
  "${cmd[@]}" > "${log_path}" 2>&1
  exit_code=$?
  set -e

  end_ts=$(date +%s)
  duration=$((end_ts - start_ts))
  finished_at="$(date +"%Y-%m-%dT%H:%M:%S")"
  status="ok"
  if [[ ${exit_code} -ne 0 ]]; then
    status="failed"
    echo "Experiment '${title}' failed (exit code ${exit_code})."
  fi

  best_cls_f1="$(grep -Eo 'Best CLS F1:[[:space:]]*[0-9]+(\.[0-9]+)?' "${log_path}" | awk '{print $4}' | tail -n 1 || true)"

  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "${title}" "${status}" "${exit_code}" "${best_cls_f1}" "${duration}" "${log_path}" \
    "${cmd_text//\"/\"\"}" "${started_at}" "${finished_at}" >> "${summary_csv}"

  if [[ "${status}" == "failed" && "${STOP_ON_ERROR}" -eq 1 ]]; then
    exit "${exit_code}"
  fi
}

run_experiment "Baseline" "${log_dir}/baseline.log" \
  --name ablation_baseline --cls_loss --no_clip_loss --no_use_graph_agg

run_experiment "+SimpleGraph" "${log_dir}/simple_graph.log" \
  --name ablation_simple_graph --cls_loss --no_clip_loss --use_graph_agg --no_use_pyg_graph_agg \
  --graph_wp 8 --graph_wf 8 --graph_drop 0.1

run_experiment "+PyG(noGraphCL)" "${log_dir}/pyg_no_cl.log" \
  --name ablation_pyg_no_cl --cls_loss --no_clip_loss --use_graph_agg --use_pyg_graph_agg \
  --no_graph_cl_loss --disable_graph_cl --graph_wp 8 --graph_wf 8 --graph_drop 0.1

run_experiment "+PyG(GraphCL)" "${log_dir}/pyg_graph_cl.log" \
  --name ablation_pyg_graph_cl --cls_loss --no_clip_loss --use_graph_agg --use_pyg_graph_agg \
  --graph_cl_loss --no_disable_graph_cl --graph_cl_lambda 0.05 --graph_fm_drop_rate 0.25 \
  --graph_ep_perturb_rate 0.1 --graph_gp_topk 3 --graph_cl_tau 0.2 --graph_wp 8 --graph_wf 8 --graph_drop 0.1

echo ""
echo "Ablation suite finished."
echo "Log directory: ${log_dir}"
echo "Summary CSV: ${summary_csv}"
