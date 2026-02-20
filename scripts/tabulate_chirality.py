#!/usr/bin/env python3
import argparse
import glob
import math
import os
from typing import List, Optional


def _parse_token(tok: str):
    tok = tok.strip()
    if tok.startswith("np.int64(") and tok.endswith(")"):
        tok = tok[len("np.int64("):-1]
    if tok.lower() == "nan":
        return float("nan")
    try:
        if "." in tok or "e" in tok.lower():
            return float(tok)
        return int(tok)
    except ValueError:
        return tok


def _parse_metric_line(line: str) -> Optional[List[object]]:
    if "|" not in line:
        return None
    metric_part = line.split("|", 1)[0].strip()
    if not (metric_part.startswith("[") and metric_part.endswith("]")):
        return None
    inner = metric_part[1:-1].strip()
    if not inner:
        return None
    toks = [t for t in inner.split(",")]
    return [_parse_token(t) for t in toks]


def _pick_latest(path: str) -> str:
    if os.path.isfile(path):
        return path
    matches = glob.glob(path)
    if not matches:
        raise FileNotFoundError(f"No files match: {path}")
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description="Tabulate chirality metrics from FlowER eval results")
    ap.add_argument(
        "result_file",
        nargs="?",
        default="/home/ptim/orcd/scratch/FlowERrs_results/flower_new_dataset/mit_normal_gpu_chi_test/test-*.txt",
        help="Result file or glob pattern (default: latest test-*.txt for current experiment)",
    )
    args = ap.parse_args()

    result_file = _pick_latest(args.result_file)

    n = 0
    sum_correct = 0
    sum_wrong_smi_conserved = 0
    sum_wrong_smi_non_conserved = 0
    sum_no_smi_conserved = 0
    sum_no_smi_non_conserved = 0
    sum_correct_cv = 0
    sum_correct_centers = 0
    sum_fp = 0
    sum_fn = 0

    # New metrics (added recently)
    sum_correct_chiral = 0
    sum_total_chiral = 0
    sum_wrong_sign = 0
    rows_with_chiral_targets = 0

    with open(result_file, "r") as f:
        for line in f:
            row = _parse_metric_line(line)
            if row is None:
                continue
            if len(row) < 9:
                continue
            n += 1

            sum_correct += int(row[0])
            sum_wrong_smi_conserved += int(row[1])
            sum_wrong_smi_non_conserved += int(row[2])
            sum_no_smi_conserved += int(row[3])
            sum_no_smi_non_conserved += int(row[4])
            sum_correct_cv += int(row[5])
            sum_correct_centers += int(row[6])
            sum_fp += int(row[7])
            sum_fn += int(row[8])

            if len(row) >= 13:
                c = int(row[9])
                t = int(row[10])
                w = int(row[11])
                sum_correct_chiral += c
                sum_total_chiral += t
                sum_wrong_sign += w
                if t > 0:
                    rows_with_chiral_targets += 1

    if n == 0:
        print(f"No parseable metric rows found in: {result_file}")
        return

    total_samples = (
        sum_correct
        + sum_wrong_smi_conserved
        + sum_wrong_smi_non_conserved
        + sum_no_smi_conserved
        + sum_no_smi_non_conserved
    )
    sample_size_est = total_samples / n if n else float("nan")

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    print(f"file: {result_file}")
    print(f"rows (reactions): {n}")
    print(f"estimated sample_size: {sample_size_est:.3f}")
    print()
    print("smiles-level metrics")
    print(f"  correct:               {sum_correct} ({pct(sum_correct, total_samples):.2f}%)")
    print(f"  wrong_smi_conserved:   {sum_wrong_smi_conserved} ({pct(sum_wrong_smi_conserved, total_samples):.2f}%)")
    print(f"  wrong_smi_non_conserv: {sum_wrong_smi_non_conserved} ({pct(sum_wrong_smi_non_conserved, total_samples):.2f}%)")
    print(f"  no_smi_conserved:      {sum_no_smi_conserved} ({pct(sum_no_smi_conserved, total_samples):.2f}%)")
    print(f"  no_smi_non_conserved:  {sum_no_smi_non_conserved} ({pct(sum_no_smi_non_conserved, total_samples):.2f}%)")
    print(f"  topk/sample accuracy:  {sum_correct}/{total_samples} = {pct(sum_correct, total_samples):.2f}%")
    print()
    print("chirality (existing)")
    print(f"  exact chiral-vector matches (correct_cv): {sum_correct_cv}/{total_samples} = {pct(sum_correct_cv, total_samples):.2f}%")
    print(f"  correct_centers (includes zeros): {sum_correct_centers}")
    print(f"  false_positives: {sum_fp}")
    print(f"  false_negatives: {sum_fn}")

    if sum_total_chiral > 0:
        print()
        print("chirality (chiral-only centers)")
        print(f"  reactions with chiral targets: {rows_with_chiral_targets}/{n}")
        print(f"  correct_chiral_centers: {sum_correct_chiral}")
        print(f"  total_chiral_centers:   {sum_total_chiral}")
        print(f"  chiral_center_acc:      {sum_correct_chiral}/{sum_total_chiral} = {pct(sum_correct_chiral, sum_total_chiral):.2f}%")
        print(f"  wrong_sign_chiral:      {sum_wrong_sign}/{sum_total_chiral} = {pct(sum_wrong_sign, sum_total_chiral):.2f}%")
    else:
        print()
        print("chirality (chiral-only centers)")
        print("  no target chiral centers found in this file")


if __name__ == "__main__":
    main()
