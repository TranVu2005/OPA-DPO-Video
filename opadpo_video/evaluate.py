"""Yes/no hallucination evaluation, matching Adaptive-SEED (evaluation/*.py, models/qwen.py):

  prompt  = question + "\\nAnswer with yes or no."  (video first, chat template, generation prompt)
  margin  = logit("Yes") - logit("No") at the first answer position; predicted "yes" iff margin > 0
  metrics = q_acc, pair_acc (VideoHallucer: both questions of a pair right), yes_rate and, against
            a baseline run, delta_pp / wrong_to_correct / correct_to_wrong / mean_abs_margin_change

  # base model first (its margins are the reference for w2c / c2w):
  python -m opadpo_video.evaluate --name base --vh data/eval_vh.jsonl --eh_root $EH_ROOT
  # any adapter (OPA, OPA-DPO, plain DPO, ...):
  python -m opadpo_video.evaluate --name opadpo --adapter runs/opadpo/final --vh data/eval_vh.jsonl \
      --eh_root $EH_ROOT --baseline runs/eval/base/margins.jsonl

Blind evaluation (--blind black | swap) scores the same questions without their video: all-black
frames, or the video of another question. What accuracy is left is read from the question text alone;
what is lost needed the video. Use a separate --name (eval.pbs appends _blind_<mode>).

Writes runs/eval/<name>/margins.jsonl (appended, so a killed job resumes) and results.csv.
Before trusting adapter numbers, check that the base rows match the `base` rows of SEED's results.csv.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import cli
from .data import read_jsonl
from .qwen import Row, answer_ids, collate_rows, encode_prompt, last_logits, load_model, load_processor, prepare_video
from .prepare_videohallucer import YN_SUFFIX

EH_SUBDIRS = {"entire": ["entire"], "mix": ["interleave", "mix"], "misleading": ["misleading"]}


# ----------------------------------------------------------------------------- EventHallusion
def _eh_candidates(vid: str, sp: str) -> list[str]:
    pre, _, num = vid.partition("_")
    n = str(int(num)) if num.isdigit() else num
    out = [vid, f"{pre}_{n}", f"{pre}{n}", num, n]
    if sp == "mix":
        out += [f"interleave_{num}", f"interleave_{n}", f"interleave{n}"]
    return list(dict.fromkeys(out))


def load_eventhallusion(root: str, splits=("entire", "mix", "misleading")) -> list[dict]:
    """Same id -> file resolution as Adaptive-SEED's data/eventhallusion.py (questions/<split>_questions.json)."""
    by_dir = {}
    for p in glob.glob(f"{root}/**/*.*", recursive=True):
        if "__MACOSX" in p or not p.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
            continue
        by_dir.setdefault(os.path.basename(os.path.dirname(p)).lower(), {})[os.path.splitext(os.path.basename(p))[0]] = p
    items, missing = [], []
    for sp in splits:
        with open(f"{root}/questions/{sp}_questions.json", encoding="utf-8") as f:
            videos = json.load(f)
        for v in videos:
            path = next((by_dir[d][c] for d in EH_SUBDIRS.get(sp, [sp]) if d in by_dir
                         for c in _eh_candidates(v["id"], sp) if c in by_dir[d]), None)
            if path is None:
                missing.append(v["id"])
                continue
            for qi, q in enumerate(v["questions"]):
                items.append(dict(id=f"eh/{v['id']}/{qi}|eh", split=f"eh_{sp}", subset=f"eh_{sp}",
                                  pair_id=f"eh/{v['id']}/{qi}", role="eh", video=path, question=q["question"],
                                  answer=q["answer"].strip().strip(".").lower()))
    if missing:
        raise SystemExit(f"EventHallusion: {len(missing)} videos not found under {root} (e.g. {missing[:5]})")
    return items


# ----------------------------------------------------------------------------- blind evaluation
BLIND_MODES = ("none", "black", "swap")


def swap_videos(items: list[dict], seed: int = 0) -> dict[str, str]:
    """item id -> the video of another item, for --blind swap. Videos are permuted within VideoHallucer
    and within EventHallusion separately (a cycle over the shuffled unique videos), so no question
    keeps its own video and questions that share a video share the same wrong one."""
    groups = {}
    for it in items:
        groups.setdefault(it["split"].startswith("VH"), set()).add(it["video"])
    to = {}
    for vids in groups.values():
        vids = sorted(vids)
        random.Random(seed).shuffle(vids)
        to.update({v: vids[(k + 1) % len(vids)] for k, v in enumerate(vids)})
    return {it["id"]: to[it["video"]] for it in items}


def blind(video: torch.Tensor, mode: str) -> torch.Tensor:
    """--blind black: same shape and fps, every pixel 0 (uint8 frames before the processor)."""
    return torch.zeros_like(video) if mode == "black" else video


# ----------------------------------------------------------------------------- metrics
def split_groups(items: list[dict]) -> dict[str, list[dict]]:
    """VH_VAL, VH_TEST, VH_TEST per subset, EH_ALL and each EH split."""
    groups = {}
    for it in items:
        groups.setdefault(it["split"], []).append(it)
        if it["split"] == "VH_TEST":
            groups.setdefault(f"VH_TEST_{it['subset']}", []).append(it)
        if it["split"].startswith("eh_"):
            groups.setdefault("EH_ALL", []).append(it)
    return groups


def metrics(items: list[dict], marg: dict, base: dict | None) -> dict:
    correct = np.array([(marg[i["id"]] > 0) == (i["answer"] == "yes") for i in items])
    row = {"n": len(items), "q_acc": float(correct.mean()),
           "yes_rate": float(np.mean([marg[i["id"]] > 0 for i in items]))}
    if items and items[0]["split"].startswith("VH"):
        ok = {}
        for i, c in zip(items, correct):
            ok.setdefault(i["pair_id"], []).append(c)
        row["pair_acc"] = float(np.mean([all(v) for v in ok.values()]))
    else:
        row["pair_acc"] = float("nan")
    if base is not None:
        old = np.array([(base[i["id"]] > 0) == (i["answer"] == "yes") for i in items])
        row.update(delta_pp=100 * float(correct.mean() - old.mean()),
                   wrong_to_correct=int((~old & correct).sum()), correct_to_wrong=int((old & ~correct).sum()),
                   mean_abs_margin_change=float(np.mean([abs(marg[i["id"]] - base[i["id"]]) for i in items])))
    return row


# ----------------------------------------------------------------------------- main
def main():
    cfg, args = cli(__doc__, lambda p: (
        p.add_argument("--name", required=True),
        p.add_argument("--adapter", default=None),
        p.add_argument("--vh", default="data/eval_vh.jsonl"),
        p.add_argument("--eh_root", default=None, help="EventHallusion dir (questions/ + videos); omit to skip"),
        p.add_argument("--baseline", default=None, help="margins.jsonl of the base run, for w2c / c2w"),
        p.add_argument("--out", default="runs/eval"),
        p.add_argument("--blind", choices=BLIND_MODES, default="none",
                       help="black: all-black frames; swap: another question's video. Measures how much of "
                            "the accuracy comes from the question text alone")))
    items = read_jsonl(args.vh) if args.vh else []
    if args.eh_root:
        items += load_eventhallusion(args.eh_root)
    src = swap_videos(items) if args.blind == "swap" else {it["id"]: it["video"] for it in items}
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    marg_path = out_dir / "margins.jsonl"
    marg = {r["id"]: r["margin"] for r in read_jsonl(marg_path)} if marg_path.exists() else {}
    todo = [i for i in items if i["id"] not in marg]
    print(f"{len(items)} items, {len(todo)} to score, blind={args.blind}", flush=True)

    if todo:
        proc = load_processor(cfg)
        tok = proc.tokenizer
        yes, no = answer_ids(proc)  # "Yes"/"No" for every backbone (docs/unified_config.md section 4)
        model = load_model(cfg, adapter=args.adapter, device_map={"": 0} if torch.cuda.is_available() else None)
        with open(marg_path, "a", encoding="utf-8") as f, torch.no_grad():
            for n, it in enumerate(todo, 1):
                video, fps = prepare_video(src[it["id"]], cfg)
                video = blind(video, args.blind)
                batch, _ = collate_rows([Row(encode_prompt(proc, it["question"].strip() + YN_SUFFIX, video, fps), [])],
                                        tok.pad_token_id)
                logits = last_logits(model, batch)[0]
                marg[it["id"]] = float(logits[yes] - logits[no])
                f.write(json.dumps({"id": it["id"], "margin": marg[it["id"]]}) + "\n")
                if n % 25 == 0:
                    f.flush()
                    print(f"{n}/{len(todo)}", flush=True)

    base = {r["id"]: r["margin"] for r in read_jsonl(args.baseline)} if args.baseline else None
    rows = [{"run": args.name, "split": s, **metrics(g, marg, base)} for s, g in sorted(split_groups(items).items())]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"results: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
