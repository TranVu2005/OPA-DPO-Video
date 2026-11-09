"""Model glue: model/LoRA loading, input building and response log-probs.

Three backends, chosen from model.base_model (see model_family):
  qwen2_5   Qwen/Qwen2.5-VL-7B-Instruct       (default; the OPA-DPO pipeline was written for it)
  qwen3     Qwen/Qwen3-VL-8B-Instruct         (transformers >= 4.57)
  internvl  OpenGVLab/InternVL3_5-8B-Instruct (trust_remote_code; frames as separate 448px images)
Inputs follow Adaptive-SEED's models/qwen.py and models/internvl.py, so the yes/no protocol is the
same as SEED's for every backend. Only the expert-free yes/no path (build_binary -> SFT / DPO ->
evaluate) is supported on qwen3 / internvl; rollout and the expert stay Qwen2.5-only.

A "row" is one (video, prompt, response) triple. Rows are encoded one by one
(prompt through the processor, response through the tokenizer, so token offsets map
back to the expert's sentence spans) and left-padded into a batch. Left padding puts
every response at the end of its row, so only the last R+1 logits are needed.
"""
from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .video import load_video, mask_video, sample_seed

IM_END = "<|im_end|>"
FAMILIES = ("qwen2_5", "qwen3", "internvl")
# Token each model actually emits after "Answer with yes or no." (Adaptive-SEED models/qwen.py:
# Qwen3-VL answers lowercase, ~99% of the mass on 'yes'/'no').
ANSWER_WORDS = {"qwen2_5": ("Yes", "No"), "qwen3": ("yes", "no"), "internvl": ("Yes", "No")}
INTERNVL_FRAME_SIZE = 448  # one 448px tile per frame (no dynamic tiling), as Adaptive-SEED
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def model_family(name_or_path: str) -> str:
    """qwen2_5 | qwen3 | internvl, from the hub id or, for a local path, its config.json."""
    name = name_or_path
    cfg_file = os.path.join(name_or_path, "config.json")
    if os.path.isfile(cfg_file):
        with open(cfg_file, encoding="utf-8") as f:
            name = json.load(f).get("model_type", "") + " " + name_or_path
    n = name.lower().replace("_", "-")
    if "internvl" in n:
        return "internvl"
    if "qwen3-vl" in n:
        return "qwen3"
    if "qwen2.5-vl" in n or "qwen2-5-vl" in n:
        return "qwen2_5"
    raise ValueError(f"unsupported model {name_or_path!r}: expected Qwen2.5-VL, Qwen3-VL or InternVL3.5")


def family_of(processor) -> str:
    """Backend of a processor returned by load_processor."""
    return getattr(processor, "opadpo_family", "qwen2_5")


def answer_ids(processor) -> tuple[int, int]:
    """Token ids whose logit difference is the yes/no margin."""
    tok = processor.tokenizer
    yes, no = ANSWER_WORDS[family_of(processor)]
    return tok.encode(yes, add_special_tokens=False)[0], tok.encode(no, add_special_tokens=False)[0]


def render_answer(processor, text: str) -> str:
    """Yes/no targets ("Yes." / "No.", from build_binary) in the spelling the model itself uses, so
    SFT / DPO move the token the evaluation reads. Any other text is returned unchanged."""
    if text.strip() in ("Yes.", "No.") and ANSWER_WORDS[family_of(processor)][0].islower():
        return text.lower()
    return text


def check_answer_tokens(processor) -> None:
    """Fail early if a rendered yes/no target does not start with the token the evaluation scores."""
    yes, no = answer_ids(processor)
    tok = processor.tokenizer
    for label, want in (("Yes.", yes), ("No.", no)):
        got = tok.encode(render_answer(processor, label), add_special_tokens=False)[0]
        if got != want:
            raise RuntimeError(f"{family_of(processor)}: {label!r} starts with token {got}, evaluation reads {want}")
    eos = tok.convert_tokens_to_ids(IM_END)
    if eos is None or eos == tok.unk_token_id:
        raise RuntimeError(f"{family_of(processor)}: tokenizer has no {IM_END}")


class InternVLProcessor:
    """Tokenizer + chat template + frame layout of InternVL3.5 (Adaptive-SEED models/internvl.py)."""

    opadpo_family = "internvl"

    def __init__(self, name_or_path: str):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True, use_fast=False)
        self.tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        vision = config.vision_config
        self.num_image_token = int((vision.image_size // vision.patch_size) ** 2 * config.downsample_ratio ** 2)
        self.template = config.template
        self._get_conv_template = get_class_from_dynamic_module("conversation.get_conv_template", name_or_path)

    def prompt(self, question: str, n_frames: int) -> str:
        # The template's system_message is what InternVLChatModel uses (model.system_message).
        conv = self._get_conv_template(self.template)
        frames = "".join(f"Frame{i + 1}: <image>\n" for i in range(n_frames))
        conv.append_message(conv.roles[0], frames + question)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        image = "<img>" + "<IMG_CONTEXT>" * self.num_image_token + "</img>"
        for _ in range(n_frames):
            text = text.replace("<image>", image, 1)
        return text

    def save_pretrained(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)


# ----------------------------------------------------------------------------- loading
def load_processor(cfg: Config):
    family = model_family(cfg.model.base_model)
    if family == "internvl":
        return InternVLProcessor(cfg.model.base_model)
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(cfg.model.base_model)
    proc.tokenizer.padding_side = "left"
    proc.opadpo_family = family
    return proc


def _dtype_kwargs(fn) -> dict:
    return {("dtype" if "dtype" in inspect.signature(fn).parameters else "torch_dtype"): torch.bfloat16}


def _load_base(cfg: Config, device_map=None):
    family = model_family(cfg.model.base_model)
    kwargs = {} if device_map is None else {"device_map": device_map}
    if family == "internvl":
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(cfg.model.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                          use_flash_attn=False, trust_remote_code=True, **kwargs)
        tok = AutoTokenizer.from_pretrained(cfg.model.base_model, trust_remote_code=True, use_fast=False)
        model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")  # forward() reads it
        return model
    if family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as cls
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    return cls.from_pretrained(cfg.model.base_model, attn_implementation=cfg.model.attn_implementation,
                               **_dtype_kwargs(cls.from_pretrained), **kwargs)


def load_model(cfg: Config, adapter: str | None = None, trainable: bool = False, device_map=None):
    model = _load_base(cfg, device_map)
    model.config.use_cache = False
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = add_lora(model, cfg)
    if trainable and cfg.model.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("gradient checkpointing enabled")
            model.enable_input_require_grads()
        except (AttributeError, NotImplementedError, ValueError) as e:  # remote-code models may lack it
            print(f"gradient checkpointing not enabled: {e}")
    if not trainable:
        model.eval().requires_grad_(False)
    return model


def disable_base_dropout(model) -> dict[str, int]:
    """Turn off dropout / drop-path in the frozen backbone, keeping LoRA's own dropout.

    DPO compares the policy (train mode) with reference log-probs computed in eval mode, so any
    stochastic layer in the backbone makes the two disagree at step 0 (TRL's DPOTrainer disables
    dropout for the same reason). InternVL's vision tower has drop-path; Qwen has none."""
    import torch.nn as nn

    off = {"dropout": 0, "drop_path": 0}
    for name, m in model.named_modules():
        if "lora_dropout" in name:
            continue
        if isinstance(m, nn.Dropout) and m.p > 0:
            m.p = 0.0
            off["dropout"] += 1
        elif type(m).__name__ == "DropPath" and getattr(m, "drop_prob", 0) > 0:
            m.drop_prob = 0.0
            off["drop_path"] += 1
    print(f"disabled in the backbone: {off['dropout']} dropout, {off['drop_path']} drop-path modules")
    return off


# Adaptive-SEED runtime.LORA_TARGETS (training/losses.py): LM attention q/k/v/o + vision-to-LM projector.
SEED_LORA_TARGETS = {
    "qwen2_5": r"(.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.mlp\.(0|2))",
    "qwen3": r"(.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.(mlp\.(0|2)|linear_fc[12]))",
    "internvl": r"(.*language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*mlp1\.(1|3))",
}


def lora_target_regex(lora_vision: bool, targets: str = "paper", family: str = "qwen2_5") -> str:
    if targets == "seed":
        return SEED_LORA_TARGETS[family]
    if targets != "paper":
        raise ValueError(f"model.lora_targets must be paper or seed, got {targets!r}")
    if family != "qwen2_5":
        raise ValueError(f"lora_targets=paper is defined for Qwen2.5-VL only; use model.lora_targets=seed for {family}")
    # Language model linears; optionally the ViT blocks and the patch merger too.
    text = r".*model\.(language_model\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    if not lora_vision:
        return text
    vision = r".*visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate|up|down)_proj)|merger\.mlp\.(0|2))"
    return f"({text})|({vision})"


def add_lora(model, cfg: Config):
    from peft import LoraConfig, get_peft_model

    family = model_family(cfg.model.base_model)
    lcfg = LoraConfig(
        r=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=lora_target_regex(cfg.model.lora_vision, cfg.model.lora_targets, family),
        # PeftModelForCausalLM passes inputs_embeds=None, which InternVLChatModel.forward rejects;
        # a plain PeftModel forwards kwargs untouched (as Adaptive-SEED wraps it).
        task_type=None if family == "internvl" else "CAUSAL_LM",
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


def _internvl_pixels(video: torch.Tensor) -> torch.Tensor:
    """uint8 (T, C, H, W) -> ImageNet-normalised float (T, 3, 448, 448), as SEED's transform."""
    if tuple(video.shape[-2:]) != (INTERNVL_FRAME_SIZE, INTERNVL_FRAME_SIZE):
        video = _resize_square_pil(video, INTERNVL_FRAME_SIZE)
    x = video.float() / 255.0
    return (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def encode_prompt(processor, prompt: str, video: torch.Tensor, fps: float) -> dict:
    family = family_of(processor)
    if family == "internvl":
        enc = processor.tokenizer(processor.prompt(prompt.strip(), video.shape[0]), return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "pixel_values": _internvl_pixels(video),
                "image_flags": torch.ones((video.shape[0], 1), dtype=torch.long)}
    text = chat_prompt(processor, prompt)
    if family == "qwen3":
        # Frames are already sampled; Qwen3-VL writes per-frame timestamps from this metadata (as SEED).
        n = int(video.shape[0])
        meta = {"total_num_frames": n, "fps": float(fps), "duration": n / float(fps), "frames_indices": list(range(n))}
        return dict(processor(text=[text], videos=[video], video_metadata=[meta], do_sample_frames=False,
                              return_tensors="pt"))
    try:
        enc = processor(text=[text], videos=[video], fps=[fps], return_tensors="pt")
    except TypeError:
        enc = processor(text=[text], videos=[video], return_tensors="pt")
    return dict(enc)


def _decode_offsets(tokenizer, ids: list[int], n_chars: int) -> list[tuple[int, int]]:
    """Char offsets from the lengths of decoded prefixes; only the span weights (sentence level) use them."""
    offsets, start = [], 0
    for i in range(len(ids)):
        end = min(len(tokenizer.decode(ids[: i + 1], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)), n_chars)
        end = max(end, start)
        offsets.append((start, end))
        start = end
    return offsets


def encode_response(tokenizer, text: str, add_eos: bool = True, max_tokens: int | None = None):
    """Token ids and char offsets of the response; <|im_end|> appended as EOS."""
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    else:  # InternVL's slow tokenizer (as its official code) has no offset mapping
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        offsets = _decode_offsets(tokenizer, ids, len(text))
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


PIXEL_KEYS = ("pixel_values_videos", "pixel_values")


def collate_rows(rows: list[Row], pad_id: int) -> tuple[dict, list[int]]:
    """Left-pad rows into one batch. Returns model inputs and per-row response lengths.

    Vision inputs are concatenated in row order (pixel_values_videos / video_grid_thw for Qwen,
    pixel_values / image_flags for InternVL): every backend fills its visual placeholder tokens
    in that order, which left padding keeps."""
    seqs = [torch.cat([r.prompt["input_ids"][0], torch.tensor(r.response_ids, dtype=torch.long)]) for r in rows]
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, width - len(s):] = s
        attention_mask[i, width - len(s):] = 1
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in rows[0].prompt:
        if key in ("input_ids", "attention_mask"):
            continue
        if key == "second_per_grid_ts":
            # Python floats, as the HF processor returns them: get_rope_index multiplies them with a
            # CPU arange, so a CUDA tensor here raises "Expected all tensors to be on the same device".
            batch[key] = [float(x) for r in rows for x in torch.as_tensor(r.prompt[key]).reshape(-1).tolist()]
            continue
        values = [r.prompt[key] for r in rows]
        if not all(torch.is_tensor(v) for v in values):
            raise TypeError(f"cannot batch processor output {key!r} ({type(values[0]).__name__})")
        if any(v.dim() == 2 and v.shape == r.prompt["input_ids"].shape for v, r in zip(values, rows)):
            raise ValueError(f"processor output {key!r} is aligned with input_ids: collate_rows must pad it")
        batch[key] = torch.cat(values)
    return batch, [len(r.response_ids) for r in rows]


def to_device(batch: dict, device) -> dict:
    """Move tensors to `device` (pixels in bf16); non-tensors such as second_per_grid_ts stay as is."""
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    for key in PIXEL_KEYS:
        if key in out:
            out[key] = out[key].to(torch.bfloat16)
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
