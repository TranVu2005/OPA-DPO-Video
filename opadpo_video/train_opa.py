"""Step 3: On-Policy Alignment. LoRA-SFT on GT + expert-revised responses (Eq. 4, Algorithm 1 phase 1).

  torchrun --nproc-per-node 4 -m opadpo_video.train_opa train.output_dir=runs/opa

Paper settings: 2 epochs, lr 2e-5 cosine, global batch 128, LoRA r=256 / alpha=512 on the
whole backbone (vision encoder and merger included).
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .config import Config, cli, total_steps
from .data import SFTCollator, SFTDataset
from .qwen import load_model, load_processor


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
    trainer = Trainer(model=model, args=training_args(cfg, len(data)), train_dataset=data,
                      data_collator=SFTCollator(proc.tokenizer.pad_token_id))
    trainer.train(resume_from_checkpoint=_last_checkpoint(cfg.train.output_dir))
    save_final(trainer, proc, cfg.train.output_dir)


def _last_checkpoint(out_dir: str) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(out_dir) if Path(out_dir).is_dir() else None


if __name__ == "__main__":
    main()
