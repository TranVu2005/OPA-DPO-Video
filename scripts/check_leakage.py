"""Checks the yes/no training data for leakage into VH_VAL / VH_TEST / EventHallusion. Standard library only.

  python scripts/check_leakage.py --train $OUT/work_binary_all/sft.jsonl \
      --eval_vh $DATA/eval_vh.jsonl --eh_root /scratch/hn98/dd9648/runs/exp1/EventHallusion

1. Same video file: by path, by basename, and by MD5 of the file content (catches a clip copied under
   another name or into another subset folder).
2. Same question text (lower-cased, whitespace collapsed) in train and in an eval split, and whether the
   answers agree.
3. Text-only shortcut: a bag-of-words naive Bayes fitted on the TRAIN questions alone (no video) and
   scored on the eval questions. Accuracy far above 0.5 means the answer can be read off the wording,
   so a fine-tuned model can gain on VH without looking at the video.
4. With --eval_dir and --runs (no GPU, reuses the margins.jsonl of finished evals): accuracy of each run
   on the questions the text-only model gets right vs wrong, on questions seen / unseen in TRAIN, and how
   often the run's answer equals the text-only answer. A run that learned the wording gains on "text
   right", loses on "text wrong" and follows the text-only answer more than the base does; a run that
   looks at the video gains on both.

  python scripts/check_leakage.py ... --eval_dir $OUT/eval --runs base_qwen3 binary_sft_qwen3_full \
      binary_dpo_lr1e-4_qwen3_full
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
from collections import Counter

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(q: str) -> str:
    q = q.split("\n")[0]  # drops the "Answer with yes or no." suffix of the training prompts
    return re.sub(r"\s+", " ", q.strip().lower())


def md5(path: str, cache: dict) -> str:
    if path not in cache:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[path] = h.hexdigest()
    return cache[path]


def words(q: str) -> list[str]:
    return re.findall(r"[a-z']+", norm(q))


def naive_bayes(train: list[tuple[str, str]]):
    counts = {"yes": Counter(), "no": Counter()}
    prior = Counter(a for _, a in train)
    for q, a in train:
        counts[a].update(set(words(q)))
    vocab = set(counts["yes"]) | set(counts["no"])

    def predict(q: str) -> str:
        score = {}
        for a in ("yes", "no"):
            total = sum(counts[a].values()) + len(vocab)
            score[a] = math.log(prior[a] / len(train)) + sum(
                math.log((counts[a][w] + 1) / total) for w in set(words(q)) if w in vocab)
        return max(score, key=score.get)
    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="sft.jsonl of the yes/no baselines")
    p.add_argument("--eval_vh", required=True, help="eval_vh.jsonl (VH_VAL + VH_TEST)")
    p.add_argument("--eh_root", default=None, help="EventHallusion dir (questions/ + videos)")
    p.add_argument("--no_md5", action="store_true", help="skip hashing the video files")
    p.add_argument("--eval_dir", default=None, help="$OUT/eval, for section 4")
    p.add_argument("--runs", nargs="*", default=[], help="eval run names (base first), for section 4")
    a = p.parse_args()

    train = [{"video": r["video"], "question": r["prompt"],
              "answer": "yes" if r["response"].lower().startswith("yes") else "no"} for r in read_jsonl(a.train)]
    evals = {}
    for r in read_jsonl(a.eval_vh):
        evals.setdefault(r["split"], []).append({"id": r["id"], "video": r["video"], "question": r["question"],
                                                 "answer": r["answer"]})
    if a.eh_root:
        eh_videos = [v for v in glob.glob(f"{a.eh_root}/**/*.*", recursive=True)
                     if "__MACOSX" not in v and v.lower().endswith(VIDEO_EXT)]
        evals["EH"] = [{"id": None, "video": v, "question": None, "answer": None} for v in eh_videos]
        for jp in sorted(glob.glob(f"{a.eh_root}/questions/*_questions.json")):
            with open(jp, encoding="utf-8") as f:
                for v in json.load(f):
                    for qi, q in enumerate(v["questions"]):  # same ids as evaluate.load_eventhallusion
                        evals["EH"].append({"id": f"eh/{v['id']}/{qi}|eh", "video": None, "question": q["question"],
                                            "answer": q["answer"].strip().strip(".").lower()})

    print(f"train: {len(train)} questions, {len({t['video'] for t in train})} videos")
    for name, rows in evals.items():
        print(f"{name}: {sum(r['question'] is not None for r in rows)} questions, "
              f"{len({r['video'] for r in rows if r['video']})} videos")

    cache = {}
    tr_paths = {t["video"] for t in train}
    tr_base = {os.path.basename(v) for v in tr_paths}
    tr_hash = {} if a.no_md5 else {md5(v, cache): v for v in tr_paths}
    tr_q = {}
    for t in train:
        tr_q.setdefault(norm(t["question"]), set()).add(t["answer"])
    nb = naive_bayes([(t["question"], t["answer"]) for t in train])

    print("\n== 1. shared videos with TRAIN ==")
    for name, rows in evals.items():
        vids = {r["video"] for r in rows if r["video"]}
        same_path = vids & tr_paths
        same_base = {v for v in vids if os.path.basename(v) in tr_base}
        same_hash = set() if a.no_md5 else {v for v in vids if md5(v, cache) in tr_hash}
        print(f"{name}: same path {len(same_path)}, same file name {len(same_base)}, "
              f"same content (md5) {'skipped' if a.no_md5 else len(same_hash)}")
        for v in sorted(same_hash)[:5]:
            print(f"    {v}  ==  {tr_hash[md5(v, cache)]}")

    print("\n== 2. same question text as TRAIN ==")
    for name, rows in evals.items():
        qs = [r for r in rows if r["question"]]
        hit = [r for r in qs if norm(r["question"]) in tr_q]
        same = sum(tr_q[norm(r["question"])] == {r["answer"]} for r in hit)
        print(f"{name}: {len(hit)}/{len(qs)} questions also in TRAIN ({same} where TRAIN has only this answer)")
        for r in hit[:3]:
            print(f"    {norm(r['question'])!r}: eval {r['answer']}, train {sorted(tr_q[norm(r['question'])])}")

    print("\n== 3. text-only accuracy (naive Bayes on TRAIN questions, no video) ==")
    for name, rows in evals.items():
        qs = [r for r in rows if r["question"]]
        if qs:
            acc = sum(nb(r["question"]) == r["answer"] for r in qs) / len(qs)
            yes = sum(r["answer"] == "yes" for r in qs) / len(qs)
            print(f"{name}: acc {acc:.4f} (majority class {max(yes, 1 - yes):.4f})")

    if a.eval_dir and a.runs:
        runs = {}
        for run in a.runs:
            path = os.path.join(a.eval_dir, run, "margins.jsonl")
            if os.path.exists(path):
                runs[run] = {r["id"]: r["margin"] for r in read_jsonl(path)}
            else:
                print(f"(no {path}, skipped)")
        print("\n== 4. finished runs split by what the question text alone predicts ==")
        print("acc = accuracy on the bucket; follow = share of answers equal to the text-only answer")
        for name in ("VH_TEST", "EH"):
            qs = [r for r in evals.get(name, []) if r["question"]]
            if not qs:
                continue
            for r in qs:
                r["nb"] = nb(r["question"])
            buckets = {"all": qs, "text right": [r for r in qs if r["nb"] == r["answer"]],
                       "text wrong": [r for r in qs if r["nb"] != r["answer"]]}
            if name.startswith("VH"):
                buckets["question in TRAIN"] = [r for r in qs if norm(r["question"]) in tr_q]
                buckets["question not in TRAIN"] = [r for r in qs if norm(r["question"]) not in tr_q]
            print(f"\n{name}")
            print(f"  {'bucket':<22}{'n':>5}" + "".join(f"  {run[:28]:>28}" for run in runs))
            for b, rows in buckets.items():
                cells = []
                for marg in runs.values():
                    got = [r for r in rows if r["id"] in marg]
                    if not got:
                        cells.append(f"{'-':>28}")
                        continue
                    pred = ["yes" if marg[r["id"]] > 0 else "no" for r in got]
                    acc = sum(p == r["answer"] for p, r in zip(pred, got)) / len(got)
                    follow = sum(p == r["nb"] for p, r in zip(pred, got)) / len(got)
                    cells.append(f"{f'acc {acc:.4f} follow {follow:.4f}':>28}")
                print(f"  {b:<22}{len(rows):>5}" + "".join(f"  {c}" for c in cells))


if __name__ == "__main__":
    main()
