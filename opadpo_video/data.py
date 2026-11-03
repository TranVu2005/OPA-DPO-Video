"""Datasets and collators for the OPA (SFT) and OPA-DPO stages.

File formats (JSONL, one object per line):
  prompts.jsonl : {"id", "video", "prompt", "gt_response"?}
  sft.jsonl     : {"id", "video", "prompt", "response"}
  dpo.jsonl     : {"id", "video", "prompt", "gt"|null, "gen", "gen_finished",
                   "rev", "gen_spans", "rev_spans"}
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import Config
from .qwen import Row, collate_rows, encode_prompt, encode_response, prepare_video
from .weights import span_weights_to_tokens

# DPO roles: response x video. "_m" = original video m, "_mm" = distorted video m'.
ROLES = ("gt_m", "gen_m", "rev_m", "gt_mm", "rev_mm")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _eos_fill(weights: list[float], ids: list[int], eos_id: int) -> list[float]:
    # EOS takes the weight of the token before it (as pad_eos in the released code).
    if weights and ids[-1] == eos_id and len(weights) > 1:
        weights[-1] = weights[-2]
    return weights


# ----------------------------------------------------------------------------- OPA / SFT
class SFTDataset(Dataset):
    def __init__(self, path: str | Path, processor, cfg: Config):
        self.rows, self.proc, self.cfg = read_jsonl(path), processor, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        video, fps = prepare_video(r["video"], self.cfg)
        ids, _ = encode_response(self.proc.tokenizer, r["response"], max_tokens=self.cfg.train.max_response_tokens)
        return Row(encode_prompt(self.proc, r["prompt"], video, fps), ids)


class SFTCollator:
    """Loss on response tokens only (Eq. 4)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[Row]) -> dict:
        batch, lens = collate_rows(rows, self.pad_id)
        labels = torch.full_like(batch["input_ids"], -100)
        for i, n in enumerate(lens):
            labels[i, -n:] = batch["input_ids"][i, -n:]
        batch["labels"] = labels
        return batch


# ----------------------------------------------------------------------------- OPA-DPO
class DPODataset(Dataset):
    """Encodes the five rows of one example plus token weights and (optionally) reference log-probs."""

    def __init__(self, path: str | Path, processor, cfg: Config, ref_logps: dict | None = None):
        self.rows, self.proc, self.cfg, self.ref = read_jsonl(path), processor, cfg, ref_logps
        self.eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.rows)

    def _response(self, text: str, spans=None, add_eos=True):
        ids, offsets = encode_response(
            self.proc.tokenizer, text, add_eos=add_eos, max_tokens=self.cfg.train.max_response_tokens
        )
        if spans is None:
            return ids, None, None
        spans = [tuple(s) for s in spans]
        w_hal = _eos_fill(span_weights_to_tokens(offsets, spans, 2), ids, self.eos_id)
        w_img = _eos_fill(span_weights_to_tokens(offsets, spans, 3), ids, self.eos_id)
        return ids, torch.tensor(w_hal), torch.tensor(w_img)

    def __getitem__(self, i):
        r = self.rows[i]
        video, fps = prepare_video(r["video"], self.cfg)
        masked, _ = prepare_video(r["video"], self.cfg, masked=True, sample_id=r["id"])
        p_m = encode_prompt(self.proc, r["prompt"], video, fps)
        p_mm = encode_prompt(self.proc, r["prompt"], masked, fps)

        gen_ids, gen_hal, _ = self._response(r["gen"], r["gen_spans"], add_eos=r.get("gen_finished", True))
        rev_ids, rev_hal, rev_img = self._response(r["rev"], r["rev_spans"])
        item = {
            "id": r["id"],
            "rows": {"gen_m": Row(p_m, gen_ids), "rev_m": Row(p_m, rev_ids), "rev_mm": Row(p_mm, rev_ids)},
            "w": {"gen_hal": gen_hal, "rev_hal": rev_hal, "rev_img": rev_img},
        }
        if r.get("gt"):
            gt_ids, _, _ = self._response(r["gt"])
            item["rows"]["gt_m"] = Row(p_m, gt_ids)
            item["rows"]["gt_mm"] = Row(p_mm, gt_ids)
        if self.ref is not None:
            item["ref"] = self.ref[r["id"]]
        return item


class DPOCollator:
    """Packs every row of every example into ONE batch, so a step is a single forward/backward
    (safe under DDP). `slots[k] = (role, example_index)` tells which row is which.
    Examples without a GT response simply have no gt_* rows."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, items: list[dict]) -> dict:
        slots, rows = [], []
        for b, it in enumerate(items):
            for role in ROLES:
                if role in it["rows"]:
                    slots.append((role, b))
                    rows.append(it["rows"][role])
        batch, lens = collate_rows(rows, self.pad_id)
        out = {"ids": [it["id"] for it in items], "batch": batch, "lens": lens, "slots": slots,
               "w": {k: [it["w"][k] for it in items] for k in ("gen_hal", "rev_hal", "rev_img")}}
        if "ref" in items[0]:
            out["ref"] = [it["ref"] for it in items]
        return out


def unpack_slots(logps: list, slots: list[tuple[str, int]], n_examples: int) -> dict:
    """Row-wise list -> {role: [tensor or None per example]}."""
    out = {role: [None] * n_examples for role in ROLES}
    for lp, (role, b) in zip(logps, slots):
        out[role][b] = lp
    return out
