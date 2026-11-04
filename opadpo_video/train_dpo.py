"""Step 4b: OPA-DPO training (Eq. 10, Algorithm 1 phase 2), continuing the OPA LoRA.

  torchrun --nproc-per-node 4 -m opadpo_video.train_dpo train.init_adapter=runs/opa/final \
      train.output_dir=runs/opadpo

Paper settings: 4 epochs, lr 1e-6 cosine, global batch 32, beta 0.1, gamma1 0.2,
gamma2 1.0, delta 0, 30% pixel mask. The reference is pi_OPA, whose log-probs come from
precompute_ref.py.

Plain DPO baseline without OPA (reference = base model, fresh LoRA, no token weights,
no image-focus / anchor terms):
  python -m opadpo_video.train_dpo train.init_adapter=null dpo.ref_dir=ref_logps_base \
      dpo.use_weights=false dpo.gamma_if=0 dpo.gamma_anc=0 train.output_dir=runs/dpo_plain
"""
from __future__ import annotations

from collections import defaultdict

from transformers import Trainer

from .config import Config, cli
from .data import ROLES, DPOCollator, DPODataset, unpack_slots
from .losses import opa_dpo_loss
from .precompute_ref import load_ref_logps
from .qwen import load_model, load_processor, response_logprobs
from .train_opa import _last_checkpoint, save_final, training_args


class OPADPOTrainer(Trainer):
    def __init__(self, *a, opa_cfg: Config, **kw):
        super().__init__(*a, **kw)
        self.opa_cfg = opa_cfg
        self._metrics = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        n = len(inputs["ids"])
        logps = response_logprobs(model, inputs["batch"], inputs["lens"])
        pol = unpack_slots(logps, inputs["slots"], n)
        ref = {role: [r.get(role) for r in inputs["ref"]] for role in ROLES}
        w = inputs["w"] if self.opa_cfg.dpo.use_weights else {k: [None] * n for k in inputs["w"]}
        loss, metrics = opa_dpo_loss(pol, ref, w, self.opa_cfg.dpo)
        for k, v in metrics.items():
            self._metrics[k].append(v)
        return loss

    def log(self, logs, *args, **kwargs):
        if self._metrics and "loss" in logs:
            logs.update({f"dpo/{k}": sum(v) / len(v) for k, v in self._metrics.items()})
            self._metrics.clear()
        super().log(logs, *args, **kwargs)


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--data", default=None, help="default: work/dpo.jsonl"))
    adapter = cfg.train.init_adapter or None
    if adapter is None:
        print("train.init_adapter is null: plain DPO from the base model with a fresh LoRA (reference = base)")
    # The reference must be the initial policy: pi_OPA for OPA-DPO, the base model otherwise.
    ref = load_ref_logps(cfg, adapter)
    proc = load_processor(cfg)
    model = load_model(cfg, adapter=adapter, trainable=True)
    data = DPODataset(args.data or cfg.work / "dpo.jsonl", proc, cfg, ref_logps=ref)
    missing = [r["id"] for r in data.rows if r["id"] not in ref]
    if missing:
        raise SystemExit(f"{len(missing)} examples have no reference log-probs (e.g. {missing[:3]}); run precompute_ref")
    trainer = OPADPOTrainer(model=model, args=training_args(cfg, len(data)), train_dataset=data,
                            data_collator=DPOCollator(proc.tokenizer.pad_token_id), opa_cfg=cfg)
    trainer.train(resume_from_checkpoint=_last_checkpoint(cfg.train.output_dir))
    save_final(trainer, proc, cfg.train.output_dir)


if __name__ == "__main__":
    main()
