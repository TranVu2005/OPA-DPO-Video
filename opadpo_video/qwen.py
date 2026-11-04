"""Qwen2.5-VL glue: model/LoRA loading, input building and response log-probs.

A "row" is one (video, prompt, response) triple. Rows are encoded one by one
(prompt through the processor, response through the tokenizer, so token offsets map
back to the expert's sentence spans) and left-padded into a batch. Left padding puts
every response at the end of its row, so only the last R+1 logits are needed.
"""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .video import load_video, mask_video, sample_seed

IM_END = "<|im_end|>"


# ----------------------------------------------------------------------------- loading
def load_processor(cfg: Config):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(cfg.model.base_model)
    proc.tokenizer.padding_side = "left"
    return proc


def load_model(cfg: Config, adapter: str | None = None, trainable: bool = False, device_map=None):
    from transformers import Qwen2_5_VLForConditionalGeneration

    kwargs = dict(attn_implementation=cfg.model.attn_implementation)
    if "dtype" in inspect.signature(Qwen2_5_VLForConditionalGeneration.from_pretrained).parameters:
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model.base_model, **kwargs)
    model.config.use_cache = False
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = add_lora(model, cfg)
    if trainable and cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if not trainable:
        model.eval().requires_grad_(False)
    return model


SEED_LORA_TARGETS = r"(.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.mlp\.(0|2))"


def lora_target_regex(lora_vision: bool, targets: str = "paper") -> str:
    if targets == "seed":
        return SEED_LORA_TARGETS  # = Adaptive-SEED runtime.LORA_TARGETS
    if targets != "paper":
        raise ValueError(f"model.lora_targets must be paper or seed, got {targets!r}")
    # Language model linears; optionally the ViT blocks and the patch merger too.
    text = r".*model\.(language_model\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    if not lora_vision:
        return text
    vision = r".*visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate|up|down)_proj)|merger\.mlp\.(0|2))"
    return f"({text})|({vision})"


def add_lora(model, cfg: Config):
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(
        r=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=lora_target_regex(cfg.model.lora_vision, cfg.model.lora_targets),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------------------- inputs
def target_size(h: int, w: int, cfg: Config) -> tuple[int, int]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    return smart_resize(h, w, factor=28, min_pixels=cfg.video.min_pixels, max_pixels=cfg.video.max_pixels)


def _resize_square_pil(video: torch.Tensor, size: int) -> torch.Tensor:
    """Same resize as Adaptive-SEED's decoder: PIL RGB, (size, size), bicubic."""
    from PIL import Image

    frames = [
        np.asarray(Image.fromarray(f).convert("RGB").resize((size, size), Image.BICUBIC))
        for f in video.permute(0, 2, 3, 1).numpy()
    ]
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def prepare_video(path: str, cfg: Config, masked: bool = False, sample_id: str = "") -> tuple[torch.Tensor, float]:
    """Decode, resize to the processor grid (so the mask is applied at final resolution), mask if asked."""
    video, fps = load_video(path, cfg.video.n_frames)
    if cfg.video.frame_size:
        video = _resize_square_pil(video, cfg.video.frame_size)
    else:
        h, w = target_size(video.shape[-2], video.shape[-1], cfg)
        video = F.interpolate(video.float(), size=(h, w), mode="bicubic", align_corners=False)
        video = video.round().clamp(0, 255).to(torch.uint8)
    if masked:
        video = mask_video(video, cfg.video.mask_ratio, cfg.video.mask_method, sample_seed(sample_id))
    return video, fps


def chat_prompt(processor, prompt: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt.strip()}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_prompt(processor, prompt: str, video: torch.Tensor, fps: float) -> dict:
    text = chat_prompt(processor, prompt)
    try:
        enc = processor(text=[text], videos=[video], fps=[fps], return_tensors="pt")
    except TypeError:
        enc = processor(text=[text], videos=[video], return_tensors="pt")
    return dict(enc)


def encode_response(tokenizer, text: str, add_eos: bool = True, max_tokens: int | None = None):
    """Token ids and char offsets of the response; <|im_end|> appended as EOS."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if max_tokens is not None and len(ids) >= max_tokens:
        ids, offsets = ids[: max_tokens - 1], offsets[: max_tokens - 1]
        add_eos = False  # truncated: no EOS
    if add_eos:
        ids.append(tokenizer.convert_tokens_to_ids(IM_END))
        offsets.append((len(text), len(text)))  # weight of an EOS falls back to the last span (see data.py)
    return ids, offsets


@dataclass
class Row:
    prompt: dict  # output of encode_prompt
    response_ids: list[int]


def collate_rows(rows: list[Row], pad_id: int) -> tuple[dict, list[int]]:
    """Left-pad rows into one batch. Returns model inputs and per-row response lengths."""
    seqs = [torch.cat([r.prompt["input_ids"][0], torch.tensor(r.response_ids, dtype=torch.long)]) for r in rows]
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, width - len(s):] = s
        attention_mask[i, width - len(s):] = 1
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": torch.cat([r.prompt["pixel_values_videos"] for r in rows]),
        "video_grid_thw": torch.cat([r.prompt["video_grid_thw"] for r in rows]),
    }
    if all("second_per_grid_ts" in r.prompt for r in rows):
        # Python floats, as the HF processor returns them: get_rope_index multiplies them with a
        # CPU arange, so a CUDA tensor here raises "Expected all tensors to be on the same device".
        batch["second_per_grid_ts"] = [float(x) for r in rows
                                       for x in torch.as_tensor(r.prompt["second_per_grid_ts"]).reshape(-1).tolist()]
    return batch, [len(r.response_ids) for r in rows]


def to_device(batch: dict, device) -> dict:
    """Move tensors to `device` (video in bf16); non-tensors such as second_per_grid_ts stay as is."""
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out["pixel_values_videos"] = out["pixel_values_videos"].to(torch.bfloat16)
    return out


# ----------------------------------------------------------------------------- log-probs
def _unwrap(model):
    base = model
    while hasattr(base, "get_base_model") or hasattr(base, "module"):
        base = base.get_base_model() if hasattr(base, "get_base_model") else base.module
    return base


@contextmanager
def _lm_head_last(model, keep: int):
    """Apply lm_head to the last `keep` positions only (transformers < 4.52 has no logits_to_keep
    for Qwen2.5-VL); avoids materialising (B, S, 152k) logits."""
    head = _unwrap(model).get_output_embeddings()
    orig = head.forward
    head.forward = lambda h: orig(h[:, -keep:])
    try:
        yield
    finally:
        del head.forward


def response_logprobs(model, batch: dict, resp_lens: list[int]) -> list[torch.Tensor]:
    """Per-token log pi(y_i | x, m, y_<i) for each row's response (float32 list)."""
    device = next(p for p in model.parameters()).device
    batch = to_device(batch, device)
    keep = max(resp_lens) + 1
    if "logits_to_keep" in inspect.signature(_unwrap(model).forward).parameters:
        logits = model(**batch, use_cache=False, logits_to_keep=keep).logits
    else:
        with _lm_head_last(model, keep):
            logits = model(**batch, use_cache=False).logits
    logits = logits[:, -keep:-1].float()  # (B, R, V), predicts the last R tokens
    labels = batch["input_ids"][:, -(keep - 1):]
    logps = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(-1)).squeeze(-1)
    return [logps[i, logps.shape[1] - n:] for i, n in enumerate(resp_lens)]


def last_logits(model, batch: dict) -> torch.Tensor:
    """Next-token logits after the prompt, (B, V) float32 (rows are left-padded, so position -1)."""
    device = next(p for p in model.parameters()).device
    batch = to_device(batch, device)
    with _lm_head_last(model, 1):
        return model(**batch, use_cache=False).logits[:, -1].float()
