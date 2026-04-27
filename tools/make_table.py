#!/usr/bin/env python
"""Generate Markdown/LaTeX tables from VEGA metrics CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPERIMENT_DISPLAY = {
    "full_vega": "Full VEGA Model",
    "wo_vega_branch": "w/o VEGA Branch",
    "wo_stochastic_anchor_sampling": "w/o Stochastic Anchor Sampling",
    "wo_unimodal_anchoring": "w/o Unimodal Anchoring",
    "wo_multimodal_anchoring": "w/o Multimodal Anchoring",
    "wo_self_distillation_in_vega": "w/o Self-Distillation in VEGA",
}

EXPERIMENT_ORDER = [
    "full_vega",
    "wo_vega_branch",
    "wo_stochastic_anchor_sampling",
    "wo_unimodal_anchoring",
    "wo_multimodal_anchoring",
    "wo_self_distillation_in_vega",
]


def _to_float(v: str) -> Optional[float]:
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() == "none":
        return None
    return float(v)


def load_metrics_csv(csv_path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_key = row["experiment"]
            rows[exp_key] = {
                "experiment": exp_key,
                "display": EXPERIMENT_DISPLAY.get(exp_key, exp_key),
                "acc": _to_float(row.get("best_all_acc")),
                "wf1": _to_float(row.get("best_all_f1")),
                "acc_epoch": _to_float(row.get("best_all_acc_epoch")),
                "wf1_epoch": _to_float(row.get("best_all_f1_epoch")),
            }
    return rows


def _fmt(v: Optional[float], ndigits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{ndigits}f}"


def build_ablation_rows(data: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for key in EXPERIMENT_ORDER:
        if key in data:
            rows.append(data[key])
    for key in data:
        if key not in EXPERIMENT_ORDER:
            rows.append(data[key])
    return rows


def ablation_markdown(rows: List[dict], dataset: str) -> str:
    header = [
        f"## Ablation Results ({dataset})",
        "",
        "| Setting | Accuracy | Weighted F1 | ACC Epoch | WF1 Epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    body = [
        f"| {r['display']} | {_fmt(r['acc'])} | {_fmt(r['wf1'])} | {_fmt(r['acc_epoch'], 0)} | {_fmt(r['wf1_epoch'], 0)} |"
        for r in rows
    ]
    return "\n".join(header + body) + "\n"


def ablation_latex(rows: List[dict], dataset: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Ablation Results on {dataset}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Setting & Accuracy & Weighted F1 & ACC Epoch & WF1 Epoch \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['display']} & {_fmt(r['acc'])} & {_fmt(r['wf1'])} & {_fmt(r['acc_epoch'], 0)} & {_fmt(r['wf1_epoch'], 0)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def load_baseline_json(path: Optional[Path]) -> List[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("baseline_json must be a list of objects")
    out = []
    for item in payload:
        out.append(
            {
                "model": item["model"],
                "acc": float(item["acc"]),
                "wf1": float(item["wf1"]),
            }
        )
    return out


def build_comparison_rows(full_vega: dict, baselines: List[dict]) -> List[dict]:
    rows = [{"model": "VEGA (Ours)", "acc": full_vega["acc"], "wf1": full_vega["wf1"]}]
    rows.extend(baselines)
    rows = sorted(rows, key=lambda x: (x["wf1"], x["acc"]), reverse=True)
    return rows


def comparison_markdown(rows: List[dict], dataset: str) -> str:
    header = [
        f"## SOTA Comparison ({dataset})",
        "",
        "| Model | Accuracy | Weighted F1 |",
        "|---|---:|---:|",
    ]
    body = [f"| {r['model']} | {_fmt(r['acc'])} | {_fmt(r['wf1'])} |" for r in rows]
    return "\n".join(header + body) + "\n"


def comparison_latex(rows: List[dict], dataset: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{SOTA Comparison on {dataset}}}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Model & Accuracy & Weighted F1 \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(f"{r['model']} & {_fmt(r['acc'])} & {_fmt(r['wf1'])} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_pair(stem: Path, md_text: str, tex_text: str) -> Tuple[Path, Path]:
    md_path = stem.with_suffix(".md")
    tex_path = stem.with_suffix(".tex")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    tex_path.write_text(tex_text, encoding="utf-8")
    return md_path, tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown/LaTeX tables from VEGA metrics CSV.")
    parser.add_argument("--metrics_csv", required=True, help="Path to metrics.csv (from collect_vega_metrics.py).")
    parser.add_argument("--dataset", default="IEMOCAP", help="Dataset name used in table captions.")
    parser.add_argument("--out_dir", default="output/paper_suite/tables", help="Directory for generated tables.")
    parser.add_argument(
        "--baseline_json",
        default="",
        help="Optional JSON list for SOTA models: [{\"model\":\"SDT\",\"acc\":70.12,\"wf1\":70.05}, ...]",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)
    data = load_metrics_csv(metrics_path)
    ablation_rows = build_ablation_rows(data)

    abl_md = ablation_markdown(ablation_rows, args.dataset)
    abl_tex = ablation_latex(ablation_rows, args.dataset)
    abl_md_path, abl_tex_path = write_pair(out_dir / "ablation_table", abl_md, abl_tex)

    print(f"[table] Ablation Markdown: {abl_md_path}")
    print(f"[table] Ablation LaTeX   : {abl_tex_path}")

    if "full_vega" in data:
        baseline_path = Path(args.baseline_json) if args.baseline_json else None
        baselines = load_baseline_json(baseline_path)
        cmp_rows = build_comparison_rows(data["full_vega"], baselines)
        cmp_md = comparison_markdown(cmp_rows, args.dataset)
        cmp_tex = comparison_latex(cmp_rows, args.dataset)
        cmp_md_path, cmp_tex_path = write_pair(out_dir / "comparison_table", cmp_md, cmp_tex)
        print(f"[table] Comparison Markdown: {cmp_md_path}")
        print(f"[table] Comparison LaTeX   : {cmp_tex_path}")
    else:
        print("[table] full_vega not found in metrics.csv; comparison table skipped.")


if __name__ == "__main__":
    main()
