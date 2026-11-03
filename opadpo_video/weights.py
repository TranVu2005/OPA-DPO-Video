"""Turn the expert's sentence-level report into weighted text spans.

y_Gen is kept exactly as the policy generated it (it must stay on-policy); each expert
sentence is located in it and its character span gets (W_hal, W_img). Text the expert
did not copy keeps weight 1. y_Rev is assembled from the rewritten sentences plus the
"added" part, each piece carrying the weights of its source sentence.

A span list is [(start, end, w_hal, w_img), ...] over characters; token weights are
derived later from the tokenizer's offset mapping (see `span_weights_to_tokens`).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

Span = tuple[int, int, float, float]


def _norm_error_type(value: str | None) -> str:
    v = (value or "correct").strip().lower().replace("_", " ")
    if v == "image recognition error":  # paper wording for the image case
        v = "visual recognition error"
    return v


def _score(s: dict) -> int:
    try:
        return min(4, max(1, int(s.get("score", 4))))
    except (TypeError, ValueError):
        return 4


def _locate(text: str, sentence: str, start: int) -> tuple[int, int] | None:
    """Find `sentence` in `text` at or after `start`; fall back to fuzzy matching."""
    sentence = sentence.strip()
    if not sentence:
        return None
    pos = text.find(sentence, start)
    if pos >= 0:
        return pos, pos + len(sentence)
    # Whitespace-insensitive exact match.
    pattern = r"\s+".join(map(re.escape, sentence.split()))
    m = re.compile(pattern).search(text, start)
    if m:
        return m.span()
    # Fuzzy: best window of the same length after `start`.
    window = len(sentence)
    best, best_pos = 0.0, None
    step = max(1, window // 8)
    for pos in range(start, max(start + 1, len(text) - window // 2), step):
        r = SequenceMatcher(None, text[pos:pos + window], sentence, autojunk=False).ratio()
        if r > best:
            best, best_pos = r, pos
    if best_pos is not None and best >= 0.8:
        return best_pos, min(len(text), best_pos + window)
    return None


def build_weighted_responses(generated: str, report: dict, hal_map: dict, img_map: dict) -> dict:
    """Return y_Rev plus span weights for y_Gen and y_Rev.

    `report` follows the schema produced by expert.py:
      {"sentences": [{"copied_content", "score", "error_type", "rewritten_content", ...}],
       "added": str}
    """
    gen_spans: list[Span] = []
    rev_parts: list[tuple[str, float, float]] = []
    cursor = 0
    n_located = 0
    for s in report.get("sentences", []):
        copied = (s.get("copied_content") or "").strip()
        rewritten = (s.get("rewritten_content") or "").strip()
        w_hal = float(hal_map.get(_score(s), 1.0))
        w_img = float(img_map.get(_norm_error_type(s.get("error_type")), 1.0))
        span = _locate(generated, copied, cursor)
        if span:
            gen_spans.append((span[0], span[1], w_hal, w_img))
            cursor = span[1]
            n_located += 1
        if rewritten:
            rev_parts.append((rewritten, w_hal, w_img))
    added = (report.get("added") or "").strip()
    if added:
        rev_parts.append((added, 1.0, 1.0))

    revised, rev_spans = "", []
    for text, w_hal, w_img in rev_parts:
        if revised:
            revised += " "
        rev_spans.append((len(revised), len(revised) + len(text), w_hal, w_img))
        revised += text
    n_sent = len(report.get("sentences", []))
    return {
        "revised": revised,
        "gen_spans": gen_spans,
        "rev_spans": rev_spans,
        "align_ratio": n_located / n_sent if n_sent else 0.0,
        "n_hallucinated": sum(1 for s in report.get("sentences", []) if _score(s) < 4),
    }


def span_weights_to_tokens(offsets: list[tuple[int, int]], spans: list[Span], which: int) -> list[float]:
    """Token weight = weight of the span containing the token's first non-space char (else 1).

    `which` = 2 for W_hal, 3 for W_img.
    """
    out = []
    j = 0
    spans = sorted(spans)
    for start, end in offsets:
        w = 1.0
        if end > start:
            while j < len(spans) and spans[j][1] <= start:
                j += 1
            if j < len(spans) and spans[j][0] < end and start < spans[j][1]:
                w = spans[j][which]
        out.append(w)
    return out
