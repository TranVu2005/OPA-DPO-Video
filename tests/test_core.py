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


def test_expert_sees_reference_but_not_as_reference_answer(monkeypatch):
    from opadpo_video import expert

    monkeypatch.setattr(expert, "_frames_b64", lambda path, cfg: ["AAAA"])
    cfg = load_config(None, [])
    row = {"video": "v.mp4", "prompt": "Is it raining?\nAnswer with yes or no, then explain why.",
           "generated": "Yes. Drops fall.", "reference": "Correct answer to this question: no."}
    text = expert.build_messages(row, cfg)[1]["content"][-1]["text"]
    assert "VERIFIED FACTS:\nCorrect answer to this question: no." in text
    assert "REFERENCE ANSWER" not in text


def test_yes_no_answer_sentence_gets_its_own_weight():
    gen = "Yes. The man throws the ball before he jumps."
    report = {"sentences": [
        {"copied_content": "Yes.", "score": 1, "error_type": "temporal error", "rewritten_content": "No."},
        {"copied_content": "The man throws the ball before he jumps.", "score": 2, "error_type": "temporal error",
         "rewritten_content": "The man jumps before he throws the ball."}]}
    w = build_weighted_responses(gen, report, HAL, IMG)
    assert w["revised"] == "No. The man jumps before he throws the ball."
    assert w["gen_spans"][0] == (0, 4, 2.5, 3.0) and w["rev_spans"][0] == (0, 3, 2.5, 3.0)


def test_build_dataset_reference_only_has_no_gt(tmp_path, monkeypatch):
    import json
    import sys

    from opadpo_video import build_dataset
    from opadpo_video.data import read_jsonl

    work = tmp_path / "work"
    (work / "rollouts").mkdir(parents=True)
    row = {"id": "t/1|basic", "video": "v.mp4", "prompt": "Q?", "reference": "Correct answer: no.",
           "generated": "Yes. It rains.", "gen_finished": True}
    (work / "rollouts" / "rank0.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rep = {"sentences": [{"copied_content": "Yes.", "score": 1, "error_type": "visual recognition error",
                          "rewritten_content": "No."},
                         {"copied_content": "It rains.", "score": 1, "error_type": "visual recognition error",
                          "rewritten_content": "It is dry."}]}
    (work / "reports.jsonl").write_text(json.dumps({"id": "t/1|basic", "report": rep}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--config", "", f"work_dir={work.as_posix()}"])
    build_dataset.main()
    sft, dpo = read_jsonl(work / "sft.jsonl"), read_jsonl(work / "dpo.jsonl")
    assert [r["id"] for r in sft] == ["t/1|basic#rev"] and sft[0]["response"] == "No. It is dry."
    assert dpo[0]["gt"] is None and "reference" not in sft[0]


def test_cli_overrides_not_swallowed_by_nargs_plus():
    from opadpo_video.config import cli

    cfg, a = cli("t", lambda p: p.add_argument("--train_subsets", nargs="+", default=["temporal"]),
                 argv=["--train_subsets", "temporal", "work_dir=/tmp/x", "dpo.beta=0.2", "--config=configs/default.yaml"])
    assert a.train_subsets == ["temporal"]
    assert cfg.work_dir == "/tmp/x" and cfg.dpo.beta == 0.2


def test_override_numbers_follow_default_type():
    from opadpo_video.config import load_config

    cfg = load_config("configs/default.yaml", ["train.learning_rate=2e-5", "dpo.gamma_if=0",
                                               "train.gradient_accumulation_steps=128"])
    assert cfg.train.learning_rate == 2e-5 and isinstance(cfg.train.learning_rate, float)
    assert cfg.dpo.gamma_if == 0.0 and isinstance(cfg.dpo.gamma_if, float)
    assert cfg.train.gradient_accumulation_steps == 128


def test_total_steps_counts_short_last_step():
    from opadpo_video.config import load_config
    from opadpo_video.config import total_steps

    cfg = load_config("configs/default.yaml", ["train.gradient_accumulation_steps=32", "train.num_train_epochs=4"])
    assert total_steps(130, cfg, world=1) == 20          # 5 steps/epoch (4 full + 1 of 2 examples)
    cfg = load_config("configs/default.yaml", ["train.gradient_accumulation_steps=8", "train.num_train_epochs=1"])
    assert total_steps(130, cfg, world=1) == 17
    assert total_steps(128, cfg, world=1) == 16


def test_seed_lora_targets():
    import re

    from opadpo_video.qwen import lora_target_regex

    rx = re.compile(lora_target_regex(True, "seed"))
    assert rx.fullmatch("base_model.model.model.layers.3.self_attn.q_proj")
    assert rx.fullmatch("model.visual.merger.mlp.0")
    assert not rx.fullmatch("model.layers.3.mlp.down_proj")
    assert not rx.fullmatch("model.visual.blocks.0.attn.qkv")


def test_model_family_and_answer_spelling():
    import pytest

    from opadpo_video.qwen import model_family, render_answer

    assert model_family("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2_5"
    assert model_family("Qwen/Qwen3-VL-8B-Instruct") == "qwen3"
    assert model_family("OpenGVLab/InternVL3_5-8B-Instruct") == "internvl"
    with pytest.raises(ValueError):
        model_family("llava-hf/llava-1.5-7b-hf")

    class P:  # a processor tagged by load_processor
        def __init__(self, fam):
            self.opadpo_family = fam

    # Every backbone is scored on Yes/No (docs/unified_config.md section 4), Qwen3 included.
    for fam in ("qwen2_5", "qwen3", "internvl"):
        assert render_answer(P(fam), "Yes.") == "Yes." and render_answer(P(fam), "No.") == "No."
    assert render_answer(P("qwen3"), "Yes. The man waves.") == "Yes. The man waves."  # long text untouched


def test_encode_response_with_slow_tokenizer():
    from opadpo_video.qwen import IM_END, encode_response

    class Slow:  # InternVL's slow tokenizer: no return_offsets_mapping
        is_fast = False
        pieces = {1: "Yes", 2: ".", 3: " The", 4: " man", 5: " waves", 9: IM_END}

        def __call__(self, text, add_special_tokens=False, **kw):
            assert "return_offsets_mapping" not in kw
            return {"input_ids": [1, 2, 3, 4, 5, 2]}

        def decode(self, ids, **kw):
            return "".join(self.pieces[i] for i in ids)

        def convert_tokens_to_ids(self, tok):
            return 9

    text = "Yes. The man waves."
    ids, offsets = encode_response(Slow(), text)
    assert ids == [1, 2, 3, 4, 5, 2, 9]
    assert offsets == [(0, 3), (3, 4), (4, 8), (8, 12), (12, 18), (18, 19), (19, 19)]
    assert [text[a:b] for a, b in offsets[:3]] == ["Yes", ".", " The"]


def test_answer_token_only_trains_the_scored_token():
    from opadpo_video.data import DPODataset, answer_token

    class Tok:
        ids = {"yes": 9454, "no": 2152, "Yes": 9580, "No": 2753}

        def encode(self, word, add_special_tokens=False):
            return [self.ids[word]]

    class Proc:
        opadpo_family = "qwen3"
        tokenizer = Tok()

    cfg = load_config(None, ["train.answer_token_only=true"])
    assert answer_token(Proc(), cfg, "Yes.") == [9580] and answer_token(Proc(), cfg, "No.") == [2753]
    assert answer_token(Proc(), cfg, "Yes. The man waves.") is None  # free text keeps the full response
    assert answer_token(Proc(), load_config(None, []), "Yes.") is None  # option off: unchanged

    ds = DPODataset.__new__(DPODataset)
    ds.proc, ds.cfg = Proc(), cfg
    ids, w_hal, w_img = ds._response("No.", [[0, 3, 1.0, 1.0]])
    assert ids == [2753] and w_hal.tolist() == [1.0] and w_img.tolist() == [1.0]


def test_old_qwen3_reference_is_stale_after_answer_token_change(tmp_path):
    import json

    import pytest

    from opadpo_video.precompute_ref import load_ref_logps, ref_meta

    def check(model):
        cfg = load_config(None, [f"model.base_model={model}", f"work_dir={tmp_path.as_posix()}",
                                 "dpo.ref_dir=ref"])
        old = ref_meta(cfg, None)
        del old["answer_words"], old["answer_token_only"]  # meta written by the older code
        (tmp_path / "ref").mkdir(exist_ok=True)
        (tmp_path / "ref" / "meta.json").write_text(json.dumps(old))
        return cfg

    with pytest.raises(SystemExit):  # Qwen3 references were made on yes/no: recompute
        load_ref_logps(check("Qwen/Qwen3-VL-8B-Instruct"), None)
    assert load_ref_logps(check("Qwen/Qwen2.5-VL-7B-Instruct"), None) == {}  # unchanged: reused


def test_calibration_threshold_and_metrics():
    from opadpo_video.calibrate import calibrate_run, pick_threshold

    # yes-biased model: every "no" question sits at a small positive margin
    assert pick_threshold([3.0, 2.0, 0.5, 0.4], [True, True, False, False]) == 1.25
    assert pick_threshold([1.0, -1.0], [True, False]) == 0.0  # already right: tie goes to 0

    items, marg = [], {}
    for split, n in (("VH_VAL", 2), ("VH_TEST", 2)):
        for p in range(n):
            for role, ans, m in (("basic", "yes", 2.0), ("hallucination", "no", 0.5)):
                k = f"{split}/{p}|{role}"
                items.append({"id": k, "split": split, "subset": "temporal", "pair_id": f"{split}/{p}",
                              "answer": ans})
                marg[k] = m
    t, rows = calibrate_run(items, marg)
    test = {r["split"]: r for r in rows}["VH_TEST"]
    assert t == 1.25
    assert test["q_acc"] == 0.5 and test["pair_acc"] == 0.0 and test["yes_rate"] == 1.0
    assert test["q_acc_cal"] == 1.0 and test["pair_acc_cal"] == 1.0 and test["yes_rate_cal"] == 0.5


def test_language_attention_targets_skip_projector_and_vision():
    import re

    from opadpo_video.qwen import lora_target_regex

    names = {
        "qwen2_5": ("base_model.model.model.layers.3.self_attn.v_proj",
                    ["base_model.model.visual.merger.mlp.0", "base_model.model.visual.blocks.0.attn.qkv",
                     "base_model.model.model.layers.3.mlp.up_proj"]),
        "qwen3": ("base_model.model.model.language_model.layers.3.self_attn.o_proj",
                  ["model.visual.merger.linear_fc1", "model.visual.blocks.0.attn.qkv",
                   "model.visual.deepstack_merger_list.0.linear_fc1"]),
        "internvl": ("base_model.model.language_model.model.layers.3.self_attn.q_proj",
                     ["base_model.model.mlp1.1", "base_model.model.vision_model.encoder.layers.0.attn.qkv"]),
    }
    for fam, (keep, frozen) in names.items():
        rx = re.compile(lora_target_regex(True, "language_attention", fam))
        assert rx.fullmatch(keep), (fam, keep)
        assert not any(rx.fullmatch(n) for n in frozen), fam
        assert re.compile(lora_target_regex(True, "seed", fam)).fullmatch(keep)  # a subset of the SEED targets


def test_disable_base_dropout_keeps_lora_dropout():
    import torch.nn as nn

    from opadpo_video.qwen import disable_base_dropout

    class DropPath(nn.Module):  # as timm's, used by InternVL's vision tower
        def __init__(self, p):
            super().__init__()
            self.drop_prob = p

    m = nn.Module()
    m.vision = nn.Sequential(DropPath(0.1), nn.Dropout(0.1))
    m.layer = nn.Module()
    m.layer.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.05)})
    assert disable_base_dropout(m) == {"dropout": 1, "drop_path": 1}
    assert m.vision[0].drop_prob == 0 and m.vision[1].p == 0
    assert m.layer.lora_dropout["default"].p == 0.05


def test_seed_lora_targets_per_backbone():
    import re

    from opadpo_video.qwen import lora_target_regex

    q3 = re.compile(lora_target_regex(True, "seed", "qwen3"))
    assert q3.fullmatch("base_model.model.model.language_model.layers.0.self_attn.k_proj")
    assert q3.fullmatch("model.visual.merger.linear_fc1") and q3.fullmatch("model.visual.merger.linear_fc2")
    assert not q3.fullmatch("model.visual.deepstack_merger_list.0.linear_fc1")
    assert not q3.fullmatch("model.visual.blocks.0.attn.qkv")
    iv = re.compile(lora_target_regex(True, "seed", "internvl"))
    assert iv.fullmatch("base_model.model.language_model.model.layers.5.self_attn.o_proj")
    assert iv.fullmatch("base_model.model.mlp1.1") and iv.fullmatch("base_model.model.mlp1.3")
    assert not iv.fullmatch("base_model.model.mlp1.0")  # LayerNorm
    assert not iv.fullmatch("base_model.model.vision_model.encoder.layers.0.attn.qkv")


def test_collate_rows_internvl_and_qwen3_inputs():
    import torch

    from opadpo_video.qwen import Row, collate_rows, to_device

    def iv(n_tok, frames):
        return {"input_ids": torch.arange(n_tok)[None], "attention_mask": torch.ones(1, n_tok, dtype=torch.long),
                "pixel_values": torch.zeros(frames, 3, 448, 448), "image_flags": torch.ones(frames, 1, dtype=torch.long)}

    batch, lens = collate_rows([Row(iv(5, 8), [1, 2]), Row(iv(7, 8), [3])], pad_id=0)
    assert batch["input_ids"].shape == (2, 8) and lens == [2, 1]  # 5+2 and 7+1 tokens, left-padded
    assert batch["pixel_values"].shape == (16, 3, 448, 448) and batch["image_flags"].shape == (16, 1)
    assert to_device(batch, "cpu")["pixel_values"].dtype == torch.bfloat16

    def q3(n_tok):
        return {"input_ids": torch.arange(n_tok)[None], "attention_mask": torch.ones(1, n_tok, dtype=torch.long),
                "pixel_values_videos": torch.zeros(4 * 196, 1536), "video_grid_thw": torch.tensor([[4, 28, 28]])}

    batch, _ = collate_rows([Row(q3(10), []), Row(q3(12), [])], pad_id=0)
    assert batch["video_grid_thw"].shape == (2, 3) and "second_per_grid_ts" not in batch
