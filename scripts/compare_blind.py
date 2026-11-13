"""Compares blind (black-frame / swapped-video) margins with the normal margins of the same run.
Standard library only, login node.

  python scripts/compare_blind.py --eval_dir $OUT/eval --vh $DATA/eval_vh.jsonl \
      --pairs base_internvl:quick_base_internvl_blind_black \
              binary_dpo_lr1e-4_internvl_full_b1.0:quick_binary_dpo_lr1e-4_internvl_full_b1.0_blind_black

Each pair is <normal run>:<blind run>; only the questions scored in both are used. Columns:
  acc normal / acc blind   accuracy with the real video / without it
  flip                     share of questions whose yes/no answer changes when the video is removed.
                           Near 0 means the answer does not depend on the video at all.
  |m| normal / |m| blind   mean |logit(Yes) - logit(No)|: how sure the model is with and without video
  yes blind                share of "yes" without video (the model's prior for the question text)
"""
from __future__ import annotations

import argparse
import json
import os


def read_margins(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["margin"] for r in map(json.loads, f) if r}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--vh", required=True, help="eval_vh.jsonl, for the answers")
    p.add_argument("--pairs", nargs="+", required=True, help="<normal run>:<blind run>")
    a = p.parse_args()
    with open(a.vh, encoding="utf-8") as f:
        answer = {r["id"]: r["answer"] for r in map(json.loads, f) if r}

    print(f"{'run':<42}{'n':>5}{'acc normal':>12}{'acc blind':>11}{'flip':>7}{'|m| normal':>12}{'|m| blind':>11}"
          f"{'yes blind':>11}")
    for pair in a.pairs:
        normal_name, blind_name = pair.split(":")
        paths = [os.path.join(a.eval_dir, n, "margins.jsonl") for n in (normal_name, blind_name)]
        missing = [x for x in paths if not os.path.exists(x)]
        if missing:
            print(f"{normal_name:<42} missing {missing}")
            continue
        normal, blind = (read_margins(x) for x in paths)
        ids = [i for i in blind if i in normal and i in answer]
        if not ids:
            print(f"{normal_name:<42} no shared questions")
            continue
        n = len(ids)
        right = lambda m, i: (m[i] > 0) == (answer[i] == "yes")  # noqa: E731
        acc_n = sum(right(normal, i) for i in ids) / n
        acc_b = sum(right(blind, i) for i in ids) / n
        flip = sum((normal[i] > 0) != (blind[i] > 0) for i in ids) / n
        mag_n = sum(abs(normal[i]) for i in ids) / n
        mag_b = sum(abs(blind[i]) for i in ids) / n
        yes_b = sum(blind[i] > 0 for i in ids) / n
        print(f"{normal_name[:41]:<42}{n:>5}{acc_n:>12.4f}{acc_b:>11.4f}{flip:>7.3f}{mag_n:>12.2f}{mag_b:>11.2f}"
              f"{yes_b:>11.3f}")


if __name__ == "__main__":
    main()
