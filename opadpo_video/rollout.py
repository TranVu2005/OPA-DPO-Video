"""Step 1: sample responses y_Gen from the initial policy (top-k 30, top-p 0.95, T=1.0).

Runs with plain `python` (1 GPU) or `torchrun --nproc-per-node N` (data sharded by rank).
Each rank appends to work/rollouts/rank{R}.jsonl and skips ids already written, so a
killed job can simply be resubmitted.

  python -m opadpo_video.rollout --prompts data/prompts.jsonl
"""
from __future__ import annotations

import json
import os

import torch

from .config import cli
from .data import read_jsonl
from .qwen import IM_END, Row, collate_rows, encode_prompt, load_model, load_processor, prepare_video, to_device


def main():
    cfg, args = cli(__doc__, lambda p: (p.add_argument("--prompts", required=True),
                                        p.add_argument("--adapter", default=None)))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(cfg.train.seed + rank)

    out_dir = cfg.work / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank{rank}.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for i, r in enumerate(read_jsonl(args.prompts)) if i % world == rank and r["id"] not in done]
    print(f"[rank {rank}] {len(todo)} prompts to roll out ({len(done)} already done)", flush=True)

    proc = load_processor(cfg)
    model = load_model(cfg, adapter=args.adapter, device_map={"": torch.cuda.current_device()})
    tok = proc.tokenizer
    eos_id = tok.convert_tokens_to_ids(IM_END)
    rc = cfg.rollout

    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), rc.batch_size):
            chunk = todo[start:start + rc.batch_size]
            rows = []
            for r in chunk:
                video, fps = prepare_video(r["video"], cfg)
                rows.append(Row(encode_prompt(proc, r["prompt"], video, fps), []))
            batch, _ = collate_rows(rows, tok.pad_token_id)
            batch = to_device(batch, model.device)
            with torch.no_grad():
                out = model.generate(
                    **batch, do_sample=True, temperature=rc.temperature, top_k=rc.top_k, top_p=rc.top_p,
                    max_new_tokens=rc.max_new_tokens, eos_token_id=eos_id, pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
            new = out[:, batch["input_ids"].shape[1]:]
            for r, ids in zip(chunk, new.tolist()):
                finished = eos_id in ids
                if finished:
                    ids = ids[: ids.index(eos_id)]
                text = tok.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps({**r, "generated": text, "gen_finished": finished}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[rank {rank}] {start + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
