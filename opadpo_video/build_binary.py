"""Yes/no baseline WITHOUT an expert: DPO data straight from VideoHallucer TRAIN labels.

Each TRAIN question of --train_subsets (same split as Adaptive-SEED) gives one example:
prompt = question + "\\nAnswer with yes or no." (exactly the evaluation prompt),
y_Rev = correct label ("Yes."/"No."), y_Gen = the opposite label, no y_GT.

This is NOT OPA-DPO / TL-DPO: with one-token answers both labels are on-policy (OPA has
nothing to align), sentence weights and target spans collapse to the whole answer. What
stays meaningful is DPO on the label plus the image-focus term (m vs a distorted m').
Report these rows as "DPO yes/no" and "DPO yes/no + m'".

  python -m opadpo_video.build_binary --root $VH_ROOT work_dir=$OUT/work_binary

Writes <work_dir>/dpo.jsonl and <work_dir>/sft.jsonl (label SFT, for a "SFT yes/no" row).
"""
from __future__ import annotations

from .config import cli
from .data import write_jsonl
from .prepare_videohallucer import YN_SUFFIX, load_items, seed_split


def binary_rows(items: list[dict], split: dict, train_subsets: list[str]) -> tuple[list[dict], list[dict]]:
    pairs = split["pairs"]
    dpo, sft = [], []
    for pid in split["train"]:
        for it in pairs[pid]:
            if it["subset"] not in train_subsets:
                continue
            right = "Yes." if it["answer"] == "yes" else "No."
            wrong = "No." if right == "Yes." else "Yes."
            base = {"id": f"{it['pair_id']}|{it['role']}|bin", "video": it["video"],
                    "prompt": it["question"].strip() + YN_SUFFIX}
            whole = lambda s: [[0, len(s), 1.0, 1.0]]  # noqa: E731  (one sentence, weight 1)
            dpo.append({**base, "gt": None, "gen": wrong, "gen_finished": True, "rev": right,
                        "gen_spans": whole(wrong), "rev_spans": whole(right), "n_hallucinated": 1})
            sft.append({**base, "response": right})
    return dpo, sft


def main():
    cfg, a = cli(__doc__, lambda p: (
        p.add_argument("--root", required=True, help="VideoHallucer directory"),
        p.add_argument("--subsets", nargs="+", default=["temporal", "object_relation"]),
        p.add_argument("--train_subsets", nargs="+", default=["temporal"]),
        p.add_argument("--n_train", type=int, default=150),
        p.add_argument("--n_val", type=int, default=50),
        p.add_argument("--n_test", type=int, default=300),
        p.add_argument("--split_seed", type=int, default=0)))
    items = load_items(a.root, a.subsets)
    split = seed_split(items, a.n_train, a.n_val, a.n_test, a.split_seed)
    dpo, sft = binary_rows(items, split, a.train_subsets)
    write_jsonl(cfg.work / "dpo.jsonl", dpo)
    write_jsonl(cfg.work / "sft.jsonl", sft)
    n_yes = sum(r["rev"] == "Yes." for r in dpo)
    print(f"{len(dpo)} yes/no examples from {len(split['train'])} train pairs ({n_yes} yes / {len(dpo) - n_yes} no) "
          f"-> {cfg.work}")


if __name__ == "__main__":
    main()
