"""Checks that OPA-DPO-Video uses exactly Adaptive-SEED's VideoHallucer split. Login node, no GPU, ~1 min.

Runs Adaptive-SEED's own split code (src/seed_video/data/videohallucer.py, `run`) offline on the local
VideoHallucer copy, with the CFG values of exp1 (subsets temporal + object_relation, split_seed 0,
150 / 50 / 300 pairs), and compares its TRAIN / VAL / TEST with what OPA-DPO-Video trains and scores on:

  train  <- sft.jsonl of work_binary_all (the yes/no SFT / DPO data, from build_binary.py)
  val    <- eval_vh.jsonl, split VH_VAL
  test   <- eval_vh.jsonl, split VH_TEST
  and the pair ids of split.json (from prepare_videohallucer.py)

For every question (key pair_id|role, as SEED's `key`) it compares subset, video file name, question
text and answer. Exit code 0 when everything matches, 1 otherwise.

  python scripts/check_seed_split.py --seed_repo /scratch/hn98/dd9648/Adaptive-SEED \\
      --vh_root /scratch/jp09/dd9648/datasets_video_hallu/videohallucer \\
      --data $OUT/data --train $OUT/work_binary_all/sft.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from types import SimpleNamespace

YN_SUFFIX = "\nAnswer with yes or no."


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sig(subset: str, video: str, question: str, answer: str) -> tuple:
    return subset, os.path.basename(video), question.strip(), answer.strip().strip(".").lower()


def seed_split(seed_repo: str, vh_root: str, cfg: dict) -> dict[str, dict[str, tuple]]:
    """Adaptive-SEED's TRAIN / VAL / TEST, computed by its own code: {split: {key: sig}}."""
    import numpy as np

    # Accept the repo root or its src/ directory.
    src = seed_repo if os.path.isdir(os.path.join(seed_repo, "seed_video")) else os.path.join(seed_repo, "src")
    if not os.path.isfile(os.path.join(src, "seed_video", "data", "videohallucer.py")):
        raise SystemExit(f"no seed_video/data/videohallucer.py under {seed_repo} (give the Adaptive-SEED checkout)")
    sys.path.insert(0, src)
    from seed_video.data import videohallucer

    os.environ["HF_HUB_OFFLINE"] = "1"  # SEED's offline branch: index the local files, download nothing
    runtime = SimpleNamespace(os=os, json=json, glob=glob, random=random, np=np,
                              args=SimpleNamespace(train_dataset="videohallucer"),
                              CFG=dict(cfg, data_root=vh_root))
    videohallucer.run(runtime)
    out = {}
    for name, items in (("train", runtime.TRAIN), ("val", runtime.VAL), ("test", runtime.TEST)):
        out[name] = {runtime.key(i): sig(i["subset"], i["video"], i["question"], i["answer"]) for i in items}
    return out


def opadpo_split(data_dir: str, train_path: str) -> tuple[dict[str, dict[str, tuple]], dict[str, list]]:
    """What OPA-DPO-Video trains and scores on: {split: {key: sig}}, plus split.json's pair ids."""
    out = {"train": {}, "val": {}, "test": {}}
    for r in read_jsonl(train_path):
        key = r["id"].removesuffix("|bin")
        prompt = r["prompt"].removesuffix(YN_SUFFIX)
        subset = key.split("/")[0]
        out["train"][key] = sig(subset, r["video"], prompt, r["response"])
    for r in read_jsonl(os.path.join(data_dir, "eval_vh.jsonl")):
        name = {"VH_VAL": "val", "VH_TEST": "test"}[r["split"]]
        out[name][r["id"]] = sig(r["subset"], r["video"], r["question"], r["answer"])
    with open(os.path.join(data_dir, "split.json"), encoding="utf-8") as f:
        s = json.load(f)
    return out, {k: s[k] for k in ("train", "val", "test")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed_repo", required=True, help="Adaptive-SEED checkout (holds src/seed_video)")
    p.add_argument("--vh_root", required=True, help="local VideoHallucer copy (one folder per subset)")
    p.add_argument("--data", required=True, help="$OUT/data of OPA-DPO-Video (eval_vh.jsonl, split.json)")
    p.add_argument("--train", required=True, help="sft.jsonl of the yes/no baselines")
    p.add_argument("--subsets", nargs="+", default=["temporal", "object_relation"])
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--n_train_pairs", type=int, default=150)
    p.add_argument("--n_val_pairs", type=int, default=50)
    p.add_argument("--n_test_pairs", type=int, default=300)
    a = p.parse_args()

    cfg = dict(subsets=a.subsets, split_seed=a.split_seed, n_train_pairs=a.n_train_pairs,
               n_val_pairs=a.n_val_pairs, n_test_pairs=a.n_test_pairs, hf_repo="bigai-nlco/VideoHallucer",
               hf_revision="c93475bf1be466c1af90d005749ccb4e6b3ad066", hf_token=None, download_workers=1)
    print("== Adaptive-SEED split (its own code)")
    seed = seed_split(a.seed_repo, a.vh_root, cfg)
    ours, split_json = opadpo_split(a.data, a.train)

    ok = True
    print("\n== comparison (questions; key = pair_id|role)")
    for name in ("train", "val", "test"):
        s, o = seed[name], ours[name]
        only_s, only_o = sorted(set(s) - set(o)), sorted(set(o) - set(s))
        diff = sorted(k for k in set(s) & set(o) if s[k] != o[k])
        pairs_json = set(split_json[name])
        pairs_seed = {k.rsplit("|", 1)[0] for k in s}
        good = not (only_s or only_o or diff) and pairs_json == pairs_seed
        ok &= good
        print(f"{name:<5}: SEED {len(s)} | OPA-DPO-Video {len(o)} | only SEED {len(only_s)} | only ours {len(only_o)} "
              f"| different content {len(diff)} | split.json pairs {'same' if pairs_json == pairs_seed else 'DIFFERENT'}"
              f"  -> {'OK' if good else 'MISMATCH'}")
        for k in (only_s[:3] + only_o[:3]):
            print(f"    missing on one side: {k}")
        for k in diff[:3]:
            print(f"    {k}\n      SEED: {s[k]}\n      ours: {o[k]}")
    print("\nRESULT:", "identical split (train, val, test)" if ok else "split differs, see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
