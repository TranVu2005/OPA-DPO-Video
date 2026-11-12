"""Step 3: On-Policy Alignment. LoRA-SFT on GT + expert-revised responses (Eq. 4, Algorithm 1 phase 1).

  torchrun --nproc-per-node 4 -m opadpo_video.train_opa train.output_dir=runs/opa

Paper settings: 2 epochs, lr 2e-5 cosine, global batch 128, LoRA r=256 / alpha=512 on the
whole backbone (vision encoder and merger included).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .config import Config, cli, total_steps
from .data import SFTCollator, SFTDataset
from .losses import anchored_loss
from .qwen import _lm_head_last, load_model, load_processor


class AnchoredSFTTrainer(Trainer):
    """SFT on the answer token, anchored to the base model (train.kl_to_base). The base logits come from
    the same forward with the LoRA disabled, so nothing has to be precomputed."""

    def __init__(self, *a, kl_weight: float, **kw):
        super().__init__(*a, **kw)
        self.kl_weight = kl_weight
        # The loss below is a mean over the micro-batch, so the Trainer must average the accumulation
        # steps itself (as it does for any model without loss kwargs).
        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        target = inputs.pop("labels")[:, -1]  # answer_token_only: the one label is the last token
        # Position -2 predicts the answer token (rows are left-padded). The base pass runs first, so its
        # activations are freed before the student's are stored for the backward.
        with _lm_head_last(model, 2):
            with torch.no_grad(), _unwrap_peft(model).disable_adapter():
                base = model(**inputs, use_cache=False).logits[:, 0].float()
            student = model(**inputs, use_cache=False).logits[:, 0].float()
        loss, ce, kl = anchored_loss(student, base, target, self.kl_weight)
        with torch.no_grad():
            self._metrics["ce"].append(float(ce.mean()))
            self._metrics["kl_base"].append(float(kl.mean()))
            self._metrics["p_target"].append(float(student.softmax(-1).gather(1, target[:, None]).mean()))
        return loss.mean()

    def log(self, logs, *args, **kwargs):
        if self._metrics and "loss" in logs:
            logs.update({f"sft/{k}": sum(v) / len(v) for k, v in self._metrics.items()})
            self._metrics.clear()
        super().log(logs, *args, **kwargs)


def _unwrap_peft(model):
    """The PeftModel inside any accelerate / DDP wrapper (it owns disable_adapter)."""
    while not hasattr(model, "disable_adapter") and hasattr(model, "module"):
        model = model.module
    return model


def training_args(cfg: Config, n_examples: int | None = None, **overrides) -> TrainingArguments:
    t = cfg.train
    kwargs = dict(
        output_dir=t.output_dir, num_train_epochs=t.num_train_epochs, learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type, warmup_ratio=t.warmup_ratio,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        bf16=True, tf32=torch.cuda.is_available(), logging_steps=t.logging_steps, save_strategy="steps", save_steps=t.save_steps,
        save_total_limit=3, seed=t.seed, report_to=t.report_to, dataloader_num_workers=t.dataloader_num_workers,
        remove_unused_columns=False, ddp_find_unused_parameters=False, deepspeed=t.deepspeed,
        weight_decay=0.0, max_grad_norm=1.0,
    )
    if n_examples:
        kwargs["max_steps"] = total_steps(n_examples, cfg)
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


def save_final(trainer: Trainer, processor, out_dir: str) -> None:
    final = Path(out_dir) / "final"
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(final))
        print(f"adapter saved to {final}")


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--data", default=None, help="default: work/sft.jsonl"))
    proc = load_processor(cfg)
    model = load_model(cfg, adapter=cfg.train.init_adapter, trainable=True)
    data = SFTDataset(args.data or cfg.work / "sft.jsonl", proc, cfg)
    common = dict(model=model, args=training_args(cfg, len(data)), train_dataset=data,
                  data_collator=SFTCollator(proc.tokenizer.pad_token_id))
    if cfg.train.kl_to_base:
        if not cfg.train.answer_token_only:
            raise SystemExit("train.kl_to_base needs train.answer_token_only=true (one answer token per row)")
        print(f"SFT anchored to the base: CE + {cfg.train.kl_to_base} * KL(student || base) at the answer token")
        trainer = AnchoredSFTTrainer(**common, kl_weight=cfg.train.kl_to_base)
    else:
        trainer = Trainer(**common)
    trainer.train(resume_from_checkpoint=_last_checkpoint(cfg.train.output_dir))
    save_final(trainer, proc, cfg.train.output_dir)


def _last_checkpoint(out_dir: str) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(out_dir) if Path(out_dir).is_dir() else None


if __name__ == "__main__":
    main()
