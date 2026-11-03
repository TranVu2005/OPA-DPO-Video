"""CPU tests for the pieces that do not need a model: span weights, masking, losses."""
import math

import torch

from opadpo_video.config import DPOConfig, load_config
from opadpo_video.losses import opa_dpo_loss
from opadpo_video.video import mask_video, sample_seed
from opadpo_video.weights import build_weighted_responses, span_weights_to_tokens

HAL = {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.0}
IMG = {"correct": 1.0, "language comprehension error": 1.0, "visual recognition error": 3.0, "temporal error": 3.0}


def test_weighted_responses_keep_generated_text_and_weight_bad_sentence():
    gen = "A man walks a dog. Then he throws a red ball. The sky is clear."
    report = {"sentences": [
        {"copied_content": "A man walks a dog.", "score": 4, "error_type": "correct", "rewritten_content": "A man walks a dog."},
        {"copied_content": "Then he throws a red ball.", "score": 2, "error_type": "image recognition error",
         "rewritten_content": "Then he throws a stick."},
        {"copied_content": "The sky is clear.", "score": 4, "error_type": "correct", "rewritten_content": "The sky is clear."},
    ], "added": ""}
    w = build_weighted_responses(gen, report, HAL, IMG)
    assert w["revised"] == "A man walks a dog. Then he throws a stick. The sky is clear."
    assert w["align_ratio"] == 1.0 and w["n_hallucinated"] == 1
    s, e, wh, wi = w["gen_spans"][1]
    assert gen[s:e] == "Then he throws a red ball." and (wh, wi) == (2.0, 3.0)
    s, e, wh, wi = w["rev_spans"][1]
    assert w["revised"][s:e] == "Then he throws a stick." and (wh, wi) == (2.0, 3.0)


def test_fuzzy_locate_tolerates_whitespace():
    gen = "First  line.\nSecond sentence here."
    report = {"sentences": [{"copied_content": "First line.", "score": 3, "rewritten_content": "First line!"}]}
    w = build_weighted_responses(gen, report, HAL, IMG)
    assert w["gen_spans"] and w["gen_spans"][0][2] == 1.5


def test_span_weights_to_tokens():
    offsets = [(0, 2), (2, 5), (5, 9), (9, 9)]
    spans = [(2, 9, 2.5, 3.0)]
    assert span_weights_to_tokens(offsets, spans, 2) == [1.0, 2.5, 2.5, 1.0]
    assert span_weights_to_tokens(offsets, spans, 3) == [1.0, 3.0, 3.0, 1.0]


def test_mask_is_deterministic_and_ratio():
    v = torch.randint(0, 255, (4, 3, 28, 28), dtype=torch.uint8)
    seed = sample_seed("vid-1")
    a, b = mask_video(v, 0.3, "pixel", seed), mask_video(v, 0.3, "pixel", seed)
    assert torch.equal(a, b) and not torch.equal(a, v)
    changed = (a != v).any(dim=1).float().mean().item()
    assert 0.2 < changed <= 0.31
    assert torch.equal(mask_video(v, 0.5, "frame", seed).float().std(dim=(2, 3)).eq(0).sum(dim=1).gt(0).sum(), torch.tensor(2))


def _fake(B=2, gt=True, same=True):
    torch.manual_seed(0)
    pol, ref = {}, {}
    for role in ("gt_m", "gen_m", "rev_m", "gt_mm", "rev_mm"):
        pol[role], ref[role] = [], []
        for _ in range(B):
            if role.startswith("gt") and not gt:
                pol[role].append(None)
                ref[role].append(None)
                continue
            p = (-torch.rand(7)).requires_grad_()
            pol[role].append(p)
            ref[role].append(p.detach().clone() if same else -torch.rand(7))
    w = {k: [torch.full((7,), 2.0) for _ in range(B)] for k in ("gen_hal", "rev_hal", "rev_img")}
    return pol, ref, w


def test_paper_loss_at_init_equals_log2_terms():
    pol, ref, w = _fake()
    cfg = DPOConfig()
    loss, m = opa_dpo_loss(pol, ref, w, cfg)
    # policy == reference: every log-sigmoid term is log 2; LC 2 terms, IF 2 terms, Anc 2 terms.
    expected = math.log(2) * (2 + cfg.gamma_if * 2 + cfg.gamma_anc * 2)
    assert abs(loss.item() - expected) < 1e-5
    loss.backward()
    assert pol["rev_m"][0].grad is not None and pol["rev_m"][0].grad.sum() < 0  # pushes y_Rev up


def test_loss_without_gt_and_official_mode_run():
    pol, ref, w = _fake(gt=False, same=False)
    for mode in ("paper", "official"):
        loss, m = opa_dpo_loss(pol, ref, w, DPOConfig(loss_mode=mode))
        assert torch.isfinite(loss) and "r_gt" not in m


def test_config_overrides(tmp_path):
    cfg = load_config(None, ["dpo.beta=0.2", "video.n_frames=8", "train.init_adapter=runs/opa/final"])
    assert cfg.dpo.beta == 0.2 and cfg.video.n_frames == 8 and cfg.train.init_adapter == "runs/opa/final"
