"""Configuration for every OPA-DPO-Video stage.

One YAML file (configs/default.yaml) holds all sections; each entry point reads the
sections it needs. Any field can be overridden on the command line with dotted keys,
e.g. `train.learning_rate=1e-6 video.n_frames=16`.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    attn_implementation: str = "sdpa"  # "flash_attention_2" if flash-attn is installed
    gradient_checkpointing: bool = True
    # LoRA covers the whole backbone, including vision encoder and merger (paper App. A.1).
    lora_rank: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.0
    lora_vision: bool = True


@dataclass
class VideoConfig:
    n_frames: int = 8  # must be even: Qwen2.5-VL merges frames in temporal pairs (8 = Adaptive-SEED exp1)
    # Square resize with PIL bicubic, exactly as Adaptive-SEED does (multiple of 28).
    # null keeps the aspect ratio and uses min/max_pixels instead.
    frame_size: int | None = 448
    max_pixels: int = 448 * 448  # per frame, only when frame_size is null
    min_pixels: int = 128 * 28 * 28
    # Distorted video m' for the image-focus loss (Eq. 8).
    # Spatial: "pixel" = random pixels per frame (paper), "tube" = same pixels in every frame,
    #          "frame" = whole frames blanked (mask_ratio applies to these three).
    # Temporal: "shuffle" = frame pairs in a random non-identity order, "reverse" = time reversed,
    #           "freeze" = the middle frame repeated (no motion).
    mask_ratio: float = 0.3
    mask_method: str = "pixel"


@dataclass
class RolloutConfig:
    max_new_tokens: int = 1024
    temperature: float = 1.0  # paper App. A.1: top-k 30, top-p 0.95, temperature 1.0
    top_k: int = 30
    top_p: float = 0.95
    batch_size: int = 4


@dataclass
class ExpertConfig:
    # Any OpenAI-compatible chat endpoint: OpenAI, Azure OpenAI, Gemini's OpenAI endpoint,
    # or a local vLLM server (e.g. Qwen2.5-VL-72B) when the compute node has no internet.
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"
    n_frames: int = 8  # frames sent to the expert, as JPEG
    frame_max_side: int = 768
    temperature: float = 0.0
    max_tokens: int = 4096
    workers: int = 8
    max_retries: int = 5


@dataclass
class WeightConfig:
    # Sentence hallucination score (1 = totally wrong .. 4 = correct) -> W_hal (Eq. 7).
    # Values follow the official implementation (utils/data_utils_dpo.py), where the most
    # severe sentences get the largest weight.
    hal: dict = field(default_factory=lambda: {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.0})
    # Error type -> W_img (Eq. 8, Table 7). "temporal error" is the video addition.
    img: dict = field(default_factory=lambda: {
        "correct": 1.0,
        "language comprehension error": 1.0,
        "visual recognition error": 3.0,
        "temporal error": 3.0,
    })


@dataclass
class TrainConfig:
    output_dir: str = "runs/opa"
    init_adapter: str | None = None  # LoRA to continue from (OPA adapter for the DPO stage)
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_response_tokens: int = 1024
    save_steps: int = 100
    logging_steps: int = 1
    seed: int = 42
    report_to: str = "none"  # "wandb" works with WANDB_MODE=offline on Gadi
    dataloader_num_workers: int = 4
    deepspeed: str | None = None


@dataclass
class DPOConfig:
    beta: float = 0.1
    gamma_if: float = 0.2  # gamma_1, image-focus weight
    gamma_anc: float = 1.0  # gamma_2, anchor weight
    delta: float = 0.0  # anchor value
    # False: every token weight is 1 (no W_hal / W_img), e.g. for a plain-DPO baseline.
    use_weights: bool = True
    # Sub-directory of work_dir holding the reference log-probs. Use a different one per
    # reference policy (pi_OPA vs the base model), so runs never mix them.
    ref_dir: str = "ref_logps"
    # "paper": sequence-level Eq. (7)-(10).
    # "official": token-level loss and anchor signs of the released code (dpo_trainer.py).
    loss_mode: str = "paper"
    # Normaliser of the token-level loss in "official" mode (the released code averages over
    # a fixed padded response length of 896).
    official_response_len: int = 896


@dataclass
class Config:
    work_dir: str = "work"
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    @property
    def work(self) -> Path:
        return Path(self.work_dir)


def _merge(obj: Any, values: dict) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key {key!r} for {type(obj).__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)
    return obj


def _set_dotted(cfg: Config, dotted: str, raw: str) -> None:
    *parents, leaf = dotted.split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config key {dotted!r}")
    value, current = yaml.safe_load(raw), getattr(obj, leaf)
    # YAML 1.1 reads "2e-5" (no dot) as a string; follow the default's type for numbers.
    if isinstance(current, float) and isinstance(value, (int, str)) and not isinstance(value, bool):
        value = float(value)
    elif isinstance(current, int) and not isinstance(current, bool) and isinstance(value, str):
        value = int(value)
    setattr(obj, leaf, value)


def load_config(path: str | None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key, raw)
    cfg.weights.hal = {int(k): float(v) for k, v in cfg.weights.hal.items()}
    if cfg.video.n_frames % 2:
        raise ValueError("video.n_frames must be even for Qwen2.5-VL")
    return cfg


_OVERRIDE = re.compile(r"^[A-Za-z_][\w.]*=")


def split_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pull `key=value` overrides out of argv wherever they are, so an nargs="+" option
    (e.g. --train_subsets temporal work_dir=...) cannot swallow them."""
    over = [a for a in argv if _OVERRIDE.match(a)]
    return [a for a in argv if not _OVERRIDE.match(a)], over


def cli(description: str, extra: callable | None = None,
        argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description,
                                     epilog="Any key=value argument is a dotted config override, "
                                            "e.g. train.learning_rate=1e-6.")
    parser.add_argument("--config", default="configs/default.yaml")
    if extra:
        extra(parser)
    rest, overrides = split_overrides(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(rest)
    args.overrides = overrides
    return load_config(args.config, overrides), args
