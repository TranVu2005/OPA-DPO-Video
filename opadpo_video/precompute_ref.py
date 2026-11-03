"""Step 4a: reference log-probs of pi_OPA for every DPO row, computed once.

pi_OPA is frozen during OPA-DPO, so its per-token log-probs are stored instead of
keeping a second model in memory. m' is seeded by sample id, so the training pass
masks exactly the same pixels.

  torchrun --nproc-per-node 4 -m opadpo_video.precompute_ref --adapter runs/opa/final
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Subset

from .config import cli
from .data import DPOCollator, DPODataset, unpack_slots
from .qwen import load_model, load_processor, response_logprobs


def load_ref_logps(work_dir) -> dict:
    ref = {}
    for p in sorted((work_dir / "ref_logps").glob("rank*.pt")):
        ref.update(torch.load(p))
    return ref


def main():
    cfg, args = cli(__doc__, lambda p: (p.add_argument("--adapter", required=True, help="OPA adapter (pi_OPA)"),
                                        p.add_argument("--data", default=None),
                                        p.add_argument("--batch_size", type=int, default=2)))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    proc = load_processor(cfg)
    model = load_model(cfg, adapter=args.adapter, device_map={"": torch.cuda.current_device()})

    data = DPODataset(args.data or cfg.work / "dpo.jsonl", proc, cfg)
    shard = Subset(data, list(range(rank, len(data), world)))
    loader = DataLoader(shard, batch_size=args.batch_size, collate_fn=DPOCollator(proc.tokenizer.pad_token_id),
                        num_workers=cfg.train.dataloader_num_workers)
    out, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            logps = response_logprobs(model, batch["batch"], batch["lens"])
            per_role = unpack_slots(logps, batch["slots"], len(batch["ids"]))
            for b, sid in enumerate(batch["ids"]):
                out[sid] = {role: v[b].float().cpu() for role, v in per_role.items() if v[b] is not None}
            n += len(batch["ids"])
            if n % 50 < args.batch_size:
                print(f"[rank {rank}] {n}/{len(shard)}", flush=True)
    path = cfg.work / "ref_logps" / f"rank{rank}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"[rank {rank}] wrote {len(out)} examples to {path}")


if __name__ == "__main__":
    main()
