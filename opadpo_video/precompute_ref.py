"""Step 4a: reference log-probs for every DPO row, computed once.

The reference is frozen during DPO, so its per-token log-probs are stored instead of
keeping a second model in memory. m' is seeded by sample id, so the training pass
distorts exactly the same pixels / frames.

  OPA-DPO (reference = pi_OPA):
    torchrun --nproc-per-node N -m opadpo_video.precompute_ref --adapter runs/opa/final
  Plain DPO without OPA (reference = base model), kept in its own directory:
    python -m opadpo_video.precompute_ref --adapter none dpo.ref_dir=ref_logps_base

work_dir/<dpo.ref_dir>/meta.json records which reference and which m' produced the
files; train_dpo refuses to start if they do not match its own settings.
"""
from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from .config import Config, cli
from .data import DPOCollator, DPODataset, unpack_slots
from .qwen import load_model, load_processor, response_logprobs


def ref_meta(cfg: Config, adapter: str | None) -> dict:
    """Everything that changes the reference log-probs."""
    v = cfg.video
    return {"base_model": cfg.model.base_model, "adapter": adapter, "n_frames": v.n_frames,
            "frame_size": v.frame_size, "max_pixels": v.max_pixels, "min_pixels": v.min_pixels,
            "mask_method": v.mask_method, "mask_ratio": v.mask_ratio,
            "max_response_tokens": cfg.train.max_response_tokens}


def load_ref_logps(cfg: Config, adapter: str | None) -> dict:
    ref_dir = cfg.work / cfg.dpo.ref_dir
    meta_path = ref_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing: run precompute_ref first")
    saved, wanted = json.loads(meta_path.read_text()), ref_meta(cfg, adapter)
    diff = {k: (saved.get(k), v) for k, v in wanted.items() if saved.get(k) != v}
    if diff:
        raise SystemExit(f"reference log-probs in {ref_dir} were made with other settings "
                         f"(saved vs now): {diff}. Recompute them or set dpo.ref_dir.")
    ref = {}
    for p in sorted(ref_dir.glob("rank*.pt")):
        ref.update(torch.load(p))
    return ref


def main():
    cfg, args = cli(__doc__, lambda p: (p.add_argument("--adapter", required=True,
                                                       help="reference LoRA (pi_OPA), or 'none' for the base model"),
                                        p.add_argument("--data", default=None),
                                        p.add_argument("--batch_size", type=int, default=2)))
    adapter = None if args.adapter.lower() == "none" else args.adapter
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    proc = load_processor(cfg)
    model = load_model(cfg, adapter=adapter, device_map={"": torch.cuda.current_device()})

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
    ref_dir = cfg.work / cfg.dpo.ref_dir
    ref_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, ref_dir / f"rank{rank}.pt")
    if rank == 0:
        (ref_dir / "meta.json").write_text(json.dumps(ref_meta(cfg, adapter), indent=2))
    print(f"[rank {rank}] wrote {len(out)} examples to {ref_dir}")


if __name__ == "__main__":
    main()
