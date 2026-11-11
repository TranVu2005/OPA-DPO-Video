"""Threshold calibration of yes/no runs: how much of a gain is only a yes/no bias shift.

Every run is scored with "yes iff margin > 0". Calibration keeps the model and moves the
threshold: "yes iff margin > t", with t picked on VH_VAL (the t with the highest VH_VAL q_acc;
ties go to the t closest to 0) and then applied unchanged to VH_TEST and EventHallusion. That is
a one-parameter model (a constant added to the "No" logit), so

  calibrated base - base          = gain from fixing the yes/no bias alone
  method - calibrated base        = gain beyond a bias fix
  calibrated method - calibrated base  = the same comparison with both sides calibrated

Reads <eval_dir>/<run>/margins.jsonl (written by evaluate.py); no GPU, a few seconds.

  python -m opadpo_video.calibrate --vh $DATA/eval_vh.jsonl --eh_root $EH_ROOT --eval_dir $OUT/eval \
      --runs base binary_sft_full binary_dpo_lr1e-4_full

Writes <eval_dir>/<run>/calibrated.csv and <eval_dir>/calibration_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPORT = ("VH_VAL", "VH_TEST", "EH_ALL")


def pick_threshold(margins: list[float], is_yes: list[bool]) -> float:
    """Threshold with the highest accuracy of "yes iff margin > t"; ties go to the t closest to 0."""
    m = np.asarray(margins, dtype=float)
    y = np.asarray(is_yes, dtype=bool)
    u = np.unique(m)
    cands = np.concatenate([[0.0, u[0] - 1.0, u[-1] + 1.0], (u[:-1] + u[1:]) / 2])
    acc = np.array([np.mean((m > t) == y) for t in cands])
    best = cands[acc == acc.max()]
    return float(best[np.argmin(np.abs(best))])


def shifted(marg: dict, t: float) -> dict:
    return {k: v - t for k, v in marg.items()}


def calibrate_run(items: list[dict], marg: dict) -> tuple[float, list[dict]]:
    from .evaluate import metrics, split_groups

    groups = split_groups([i for i in items if i["id"] in marg])
    val = groups["VH_VAL"]
    t = pick_threshold([marg[i["id"]] for i in val], [i["answer"] == "yes" for i in val])
    rows = []
    for split in REPORT:
        if split not in groups:
            continue
        raw, cal = metrics(groups[split], marg, None), metrics(groups[split], shifted(marg, t), None)
        rows.append({"split": split, "n": raw["n"], "threshold": t,
                     "q_acc": raw["q_acc"], "q_acc_cal": cal["q_acc"],
                     "pair_acc": raw["pair_acc"], "pair_acc_cal": cal["pair_acc"],
                     "yes_rate": raw["yes_rate"], "yes_rate_cal": cal["yes_rate"]})
    return t, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    from .data import read_jsonl
    from .evaluate import load_eventhallusion

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vh", required=True, help="eval_vh.jsonl (VH_VAL + VH_TEST)")
    p.add_argument("--eh_root", default=None, help="EventHallusion dir; omit to skip EH")
    p.add_argument("--eval_dir", required=True, help="directory holding <run>/margins.jsonl")
    p.add_argument("--runs", nargs="*", default=None, help="default: every run in eval_dir")
    a = p.parse_args()

    items = read_jsonl(a.vh) + (load_eventhallusion(a.eh_root) if a.eh_root else [])
    eval_dir = Path(a.eval_dir)
    runs = a.runs or sorted(d.name for d in eval_dir.iterdir() if (d / "margins.jsonl").exists())
    summary = []
    for run in runs:
        path = eval_dir / run / "margins.jsonl"
        if not path.exists():
            print(f"skip {run}: no {path}")
            continue
        marg = {r["id"]: r["margin"] for r in read_jsonl(path)}
        t, rows = calibrate_run(items, marg)
        rows = [{"run": run, **r} for r in rows]
        _write_csv(eval_dir / run / "calibrated.csv", rows)
        summary += rows
        test = {r["split"]: r for r in rows}
        vh, eh = test.get("VH_TEST"), test.get("EH_ALL")
        line = f"{run:34s} t={t:+.3f}"
        if vh:
            line += (f" | VH Q {vh['q_acc']:.4f}->{vh['q_acc_cal']:.4f} pair {vh['pair_acc']:.4f}->{vh['pair_acc_cal']:.4f}"
                     f" yes {vh['yes_rate']:.3f}->{vh['yes_rate_cal']:.3f}")
        if eh:
            line += f" | EH {eh['q_acc']:.4f}->{eh['q_acc_cal']:.4f} yes {eh['yes_rate']:.3f}->{eh['yes_rate_cal']:.3f}"
        print(line)
    if summary:
        _write_csv(eval_dir / "calibration_summary.csv", summary)
        print(f"summary: {eval_dir / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
