"""Step 2b: join rollouts and expert reports into the OPA (SFT) and OPA-DPO datasets.

  python -m opadpo_video.build_dataset [--min_align 0.5]

Writes work/sft.jsonl (GT and revised responses, Algorithm 1 phase 1) and
work/dpo.jsonl (y_GT, y_Gen, y_Rev with sentence spans, phase 2).
"""
from __future__ import annotations

from .config import cli
from .data import read_jsonl, write_jsonl
from .weights import build_weighted_responses


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--min_align", type=float, default=0.5,
                                                      help="drop reports whose sentences cannot be located in y_Gen"))
    rollouts = {r["id"]: r for p in sorted((cfg.work / "rollouts").glob("rank*.jsonl")) for r in read_jsonl(p)}
    reports = {r["id"]: r["report"] for r in read_jsonl(cfg.work / "reports.jsonl")}

    sft, dpo, skipped = [], [], {"no_rollout": 0, "align": 0, "empty_rev": 0}
    for sid, report in reports.items():
        r = rollouts.get(sid)
        if r is None:
            skipped["no_rollout"] += 1
            continue
        w = build_weighted_responses(r["generated"], report, cfg.weights.hal, cfg.weights.img)
        if w["align_ratio"] < args.min_align:
            skipped["align"] += 1
            continue
        if not w["revised"].strip():
            skipped["empty_rev"] += 1
            continue
        gt = (r.get("gt_response") or "").strip() or None
        base = {"video": r["video"], "prompt": r["prompt"]}
        if gt:
            sft.append({"id": f"{sid}#gt", **base, "response": gt})
        sft.append({"id": f"{sid}#rev", **base, "response": w["revised"]})
        dpo.append({
            "id": sid, **base, "gt": gt, "gen": r["generated"], "gen_finished": r.get("gen_finished", True),
            "rev": w["revised"], "gen_spans": w["gen_spans"], "rev_spans": w["rev_spans"],
            "n_hallucinated": w["n_hallucinated"],
        })

    write_jsonl(cfg.work / "sft.jsonl", sft)
    write_jsonl(cfg.work / "dpo.jsonl", dpo)
    n_hal = sum(1 for d in dpo if d["n_hallucinated"])
    print(f"sft rows: {len(sft)}  dpo examples: {len(dpo)} ({n_hal} with >=1 hallucinated sentence)  skipped: {skipped}")


if __name__ == "__main__":
    main()
