"""Build OPA-DPO training prompts and evaluation items from VideoHallucer, using EXACTLY the
train/val/test split of Adaptive-SEED (data/videohallucer.py: same de-duplication, pair ids,
seeded shuffle and video-disjoint selection), so results are comparable and no val/test video
is ever trained on.

Runs on a login node (reads json + lists files, no GPU):
  python -m opadpo_video.prepare_videohallucer --root /scratch/jp09/dd9648/datasets_video_hallu/videohallucer \
      --out data --train_subsets temporal --describe 4

Writes:
  data/split.json       pair ids and videos of each split
  data/eval_vh.jsonl    VAL + TEST yes/no items for evaluate.py
  data/prompts.jsonl    training prompts (TRAIN videos only):
      * every TRAIN question of --train_subsets as "yes/no, then explain", label in `reference`
      * --describe N open-ended temporal prompts per TRAIN video, the video's TRAIN Q/A as `reference`
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

from .data import write_jsonl

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")
YN_SUFFIX = "\nAnswer with yes or no."  # evaluation prompt of Adaptive-SEED
TRAIN_SUFFIX = "\nAnswer with yes or no, then explain why."
DESCRIBE_PROMPTS = [
    "Describe the events in this video in the order they happen.",
    "What happens at the beginning, in the middle, and at the end of this video?",
    "Describe the actions of the people or objects in this video step by step, in chronological order.",
    "Describe how the scene changes over time in this video, including the direction of any movement.",
]


def video_index(root: str, subsets: list[str]) -> dict:
    """(subset, basename) -> path relative to root; first occurrence wins (as SEED's setdefault)."""
    index = {}
    for sub in subsets:
        for dirpath, _, names in os.walk(f"{root}/{sub}"):
            for n in names:
                rel = os.path.relpath(os.path.join(dirpath, n), root).replace("\\", "/")
                if rel.lower().endswith(VIDEO_EXT):
                    index.setdefault((rel.split("/")[0], os.path.basename(rel)), rel)
    return index


def load_items(root: str, subsets: list[str]) -> list[dict]:
    index = video_index(root, subsets)
    items = []
    for sub in subsets:
        seen = set()
        for jp in sorted(glob.glob(f"{root}/{sub}/*.json")):
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            for ex in data:
                b, h = ex["basic"], ex["hallucination"]
                sig = (b["video"], b["question"], h["video"], h["question"])
                if sig in seen:
                    continue
                seen.add(sig)  # counted before the video check, as in SEED (keeps pair ids identical)
                if any((sub, os.path.basename(q["video"])) not in index for q in (b, h)):
                    continue
                pid = f"{sub}/{len(seen) - 1}"
                for role, q in (("basic", b), ("hallucination", h)):
                    rp = index[sub, os.path.basename(q["video"])]
                    items.append(dict(subset=sub, pair_id=pid, role=role, video=os.path.join(root, rp),
                                      question=q["question"], answer=q["answer"].strip().lower()))
    return items


def seed_split(items: list[dict], n_train: int, n_val: int, n_test: int, seed: int) -> dict:
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], []).append(it)
    pair_ids = sorted(pairs)
    random.Random(seed).shuffle(pair_ids)

    def take(cands, n, banned):
        ids, vids = [], set()
        for pid in cands:
            if len(ids) >= n:
                break
            v = {i["video"] for i in pairs[pid]}
            if v & banned:
                continue
            ids.append(pid)
            vids |= v
        return ids, vids

    train, train_v = take(pair_ids, n_train, set())
    rest = [p for p in pair_ids if p not in set(train)]
    val, val_v = take(rest, n_val, train_v)
    rest = [p for p in rest if p not in set(val)]
    test, test_v = take(rest, n_test, train_v | val_v)
    assert not (train_v & val_v) and not (train_v & test_v) and not (val_v & test_v), "splits share videos"
    return {"pairs": pairs, "train": train, "val": val, "test": test,
            "videos": {"train": sorted(train_v), "val": sorted(val_v), "test": sorted(test_v)}}


def build(items, split, train_subsets: list[str], describe: int) -> tuple[list[dict], list[dict]]:
    pairs = split["pairs"]
    key = lambda it: f"{it['pair_id']}|{it['role']}"  # noqa: E731  (SEED's item key)
    evals = [{"id": key(it), "split": name, **it}
             for name, ids in (("VH_VAL", split["val"]), ("VH_TEST", split["test"])) for p in ids for it in pairs[p]]

    train_items = [it for p in split["train"] for it in pairs[p] if it["subset"] in train_subsets]
    prompts = [{"id": f"{key(it)}|yn", "video": it["video"], "subset": it["subset"],
                "prompt": it["question"].strip() + TRAIN_SUFFIX,
                "reference": f"Correct answer to this question: {it['answer']}."} for it in train_items]
    by_video = {}
    for it in train_items:
        by_video.setdefault(it["video"], []).append(it)
    for video, its in sorted(by_video.items()):
        facts = "\n".join(f"Q: {it['question'].strip()} A: {it['answer']}" for it in its)
        for k, text in enumerate(DESCRIBE_PROMPTS[:describe]):
            prompts.append({"id": f"desc/{its[0]['subset']}/{os.path.basename(video)}/{k}", "video": video,
                            "subset": its[0]["subset"], "prompt": text,
                            "reference": f"Verified question/answer pairs about this video:\n{facts}"})
    held_out = set(split["videos"]["val"]) | set(split["videos"]["test"])
    assert not any(p["video"] in held_out for p in prompts), "a training prompt uses a val/test video"
    return prompts, evals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="VideoHallucer directory with one folder per subset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--subsets", nargs="+", default=["temporal", "object_relation"],
                    help="subsets that define the split; must equal Adaptive-SEED's CFG['subsets']")
    ap.add_argument("--train_subsets", nargs="+", default=["temporal"], help="subsets whose TRAIN pairs become prompts")
    ap.add_argument("--n_train", type=int, default=150)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--describe", type=int, default=4, help=f"open-ended prompts per TRAIN video (0-{len(DESCRIBE_PROMPTS)})")
    a = ap.parse_args()

    items = load_items(a.root, a.subsets)
    split = seed_split(items, a.n_train, a.n_val, a.n_test, a.split_seed)
    prompts, evals = build(items, split, a.train_subsets, a.describe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "prompts.jsonl", prompts)
    write_jsonl(out / "eval_vh.jsonl", evals)
    (out / "split.json").write_text(json.dumps(
        {"args": vars(a), "train": split["train"], "val": split["val"], "test": split["test"],
         "videos": split["videos"]}, indent=1), encoding="utf-8")
    n_pairs = len(split["pairs"])
    print(f"{n_pairs} pairs -> train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])} pairs")
    print(f"prompts: {len(prompts)} ({sum(p['id'].endswith('|yn') for p in prompts)} yes/no, "
          f"{sum(p['id'].startswith('desc/') for p in prompts)} describe) | eval items: {len(evals)}")
    print("Check against Adaptive-SEED's log line 'train 150 cặp | val 50 cặp | test ... cặp'.")


if __name__ == "__main__":
    main()
