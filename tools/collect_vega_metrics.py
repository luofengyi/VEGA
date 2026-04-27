#!/usr/bin/env python
"""Collect VEGA experiment metrics from training logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


BEST_METRIC_RE = re.compile(r"^(all_acc|all_f1|a_f1|v_f1|t_f1):\s*([0-9.]+),\s*idx:\s*([0-9]+)\s*$")
FINAL_F1_RE = re.compile(r"Best CLS F1:\s*([0-9.]+)")
METHOD_RE = re.compile(r"^\s*0\s+Model\s+")
FLOAT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


@dataclass
class ExperimentResult:
    experiment: str
    log_file: str
    best_all_acc: Optional[float] = None
    best_all_acc_epoch: Optional[int] = None
    best_all_f1: Optional[float] = None
    best_all_f1_epoch: Optional[int] = None
    best_a_f1: Optional[float] = None
    best_a_f1_epoch: Optional[int] = None
    best_v_f1: Optional[float] = None
    best_v_f1_epoch: Optional[int] = None
    best_t_f1: Optional[float] = None
    best_t_f1_epoch: Optional[int] = None
    final_report_wf1: Optional[float] = None
    final_best_cls_f1: Optional[float] = None


def parse_log(log_path: Path) -> ExperimentResult:
    result = ExperimentResult(experiment=log_path.stem, log_file=str(log_path))
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        m = BEST_METRIC_RE.match(line.strip())
        if m:
            metric_name, val, epoch = m.group(1), float(m.group(2)), int(m.group(3))
            setattr(result, f"best_{metric_name}", val)
            setattr(result, f"best_{metric_name}_epoch", epoch)
            continue

        m = FINAL_F1_RE.search(line)
        if m:
            result.final_best_cls_f1 = float(m.group(1))

    # Parse the last "0   Model ... w-F1" table row if present.
    for line in reversed(lines):
        if METHOD_RE.match(line):
            nums = [float(x) for x in FLOAT_RE.findall(line)]
            if nums:
                result.final_report_wf1 = nums[-1]
            break

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect VEGA experiment metrics from log files.")
    parser.add_argument("--log_dir", required=True, help="Directory containing *.log files.")
    parser.add_argument("--out_csv", required=True, help="Path to output CSV file.")
    parser.add_argument("--out_json", required=True, help="Path to output JSON file.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No log files found in: {log_dir}")

    results = [parse_log(log_file) for log_file in logs]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[collect] Parsed {len(results)} logs")
    print(f"[collect] CSV : {out_csv}")
    print(f"[collect] JSON: {out_json}")


if __name__ == "__main__":
    main()
