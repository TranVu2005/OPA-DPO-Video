"""End-to-end smoke test on CPU with a tiny randomly initialised Qwen2.5-VL.

Downloads only the Qwen2.5-VL processor/tokenizer files. Enable with OPADPO_SMOKE=1:
  OPADPO_SMOKE=1 pytest -q tests/test_smoke_qwen.py
"""
import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("OPADPO_SMOKE") != "1", reason="set OPADPO_SMOKE=1")

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"


def _write_video(path, n=24, h=80, w=112):
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=8)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            arr[:, (i * 4) % w: (i * 4) % w + 16] = 255  # moving bar
            for pkt in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    root = tmp_path_factory.mktemp("smoke")
    conf = AutoConfig.from_pretrained(BASE)
    conf.hidden_size, conf.num_hidden_layers, conf.num_attention_heads = 64, 2, 4
    conf.num_key_value_heads, conf.intermediate_size = 2, 128
    conf.rope_scaling = {**conf.rope_scaling, "mrope_section": [2, 3, 3]}
    v = conf.vision_config
    v.depth, v.hidden_size, v.num_heads, v.intermediate_size, v.out_hidden_size = 2, 32, 2, 64, 64
    v.fullatt_block_indexes = [1]
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(conf).to(torch.bfloat16)
    model.save_pretrained(root / "tiny")
    AutoProcessor.from_pretrained(BASE).save_pretrained(root / "tiny")
    video = root / "v.mp4"
    _write_video(video)

    from opadpo_video.config import load_config

    cfg = load_config(None, [f"model.base_model={(root / 'tiny').as_posix()}", "video.n_frames=4",
                             "video.max_pixels=12544", "model.lora_rank=4", "model.lora_alpha=8",
                             "model.gradient_checkpointing=false", f"work_dir={(root / 'work').as_posix()}",
                             "train.dataloader_num_workers=0", "train.max_response_tokens=64"])
    return cfg, str(video), root


def test_pipeline(setup):
    from opadpo_video.data import DPOCollator, DPODataset, SFTCollator, SFTDataset, unpack_slots, write_jsonl
    from opadpo_video.losses import opa_dpo_loss
    from opadpo_video.qwen import (Row, collate_rows, encode_prompt, load_model, load_processor,
                                   prepare_video, response_logprobs)
    from opadpo_video.weights import build_weighted_responses

    cfg, video, root = setup
    proc = load_processor(cfg)
    tok = proc.tokenizer

    # Rollout-style generation with left padding, two rows of different prompt length.
    model = load_model(cfg)
    v, fps = prepare_video(video, cfg)
    rows = [Row(encode_prompt(proc, "Describe the video.", v, fps), []),
            Row(encode_prompt(proc, "What moves in this video, and in which direction?", v, fps), [])]
    batch, _ = collate_rows(rows, tok.pad_token_id)
    batch["pixel_values_videos"] = batch["pixel_values_videos"].to(torch.bfloat16)
    out = model.generate(**batch, do_sample=True, top_k=30, top_p=0.95, max_new_tokens=5, use_cache=True)
    assert out.shape[1] > batch["input_ids"].shape[1]

    # Left padding must not change a row's log-probs.
    r1 = Row(rows[0].prompt, tok("A white bar moves right.", add_special_tokens=False)["input_ids"])
    alone = response_logprobs(model, *collate_rows([r1], tok.pad_token_id))[0]
    padded = response_logprobs(model, *collate_rows([r1, Row(rows[1].prompt, [1, 2, 3])], tok.pad_token_id))[0]
    assert torch.allclose(alone, padded, atol=2e-2), (alone - padded).abs().max()

    # Build tiny SFT / DPO data from a fake expert report.
    gen = "A white bar moves left. The background is black."
    report = {"sentences": [
        {"copied_content": "A white bar moves left.", "score": 2, "error_type": "temporal error",
         "rewritten_content": "A white bar moves right."},
        {"copied_content": "The background is black.", "score": 4, "error_type": "correct",
         "rewritten_content": "The background is black."}], "added": ""}
    w = build_weighted_responses(gen, report, cfg.weights.hal, cfg.weights.img)
    gt = "A vertical white bar slides from left to right over a black background."
    write_jsonl(cfg.work / "sft.jsonl", [{"id": "a#gt", "video": video, "prompt": "Describe the video.", "response": gt},
                                         {"id": "a#rev", "video": video, "prompt": "Describe the video.", "response": w["revised"]}])
    dpo_rows = [{"id": "a", "video": video, "prompt": "Describe the video.", "gt": gt, "gen": gen, "gen_finished": True,
                 "rev": w["revised"], "gen_spans": w["gen_spans"], "rev_spans": w["rev_spans"]},
                {"id": "b", "video": video, "prompt": "Describe the video.", "gt": None, "gen": gen, "gen_finished": False,
                 "rev": w["revised"], "gen_spans": w["gen_spans"], "rev_spans": w["rev_spans"]}]
    write_jsonl(cfg.work / "dpo.jsonl", dpo_rows)

    # OPA SFT forward/backward with LoRA.
    sft = SFTDataset(cfg.work / "sft.jsonl", proc, cfg)
    sb = SFTCollator(tok.pad_token_id)([sft[0], sft[1]])
    lora = load_model(cfg, trainable=True)
    sb["pixel_values_videos"] = sb["pixel_values_videos"].to(torch.bfloat16)
    loss = lora(**sb).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for n, p in lora.named_parameters() if "visual" in n and "lora" in n)
    lora.save_pretrained(root / "opa")

    # Reference pass with the OPA adapter, then the DPO loss: policy == reference at init.
    ref_model = load_model(cfg, adapter=str(root / "opa"))
    dpo = DPODataset(cfg.work / "dpo.jsonl", proc, cfg)
    db = DPOCollator(tok.pad_token_id)([dpo[0], dpo[1]])
    assert sum(1 for s in db["slots"] if s[0].startswith("gt")) == 2  # only example "a" has GT
    ref_lp = unpack_slots(response_logprobs(ref_model, db["batch"], db["lens"]), db["slots"], 2)
    policy = load_model(cfg, adapter=str(root / "opa"), trainable=True)
    pol_lp = unpack_slots(response_logprobs(policy, db["batch"], db["lens"]), db["slots"], 2)
    # m' differs from m, and is identical across two encodings of the same example.
    assert not torch.allclose(pol_lp["rev_m"][0], pol_lp["rev_mm"][0])
    loss, metrics = opa_dpo_loss(pol_lp, ref_lp, db["w"], cfg.dpo)
    assert abs(metrics["r_rev"]) < 1e-2, metrics
    loss.backward()
    # Weighted tokens: the hallucinated sentence of y_Rev has W_hal 2.0 and W_img 3.0.
    assert db["w"]["rev_hal"][0].max().item() == 2.0 and db["w"]["rev_img"][0].max().item() == 3.0


def test_trainer_step(setup):
    """One optimizer step of OPADPOTrainer through the HF Trainer (collator, device moves, log hook)."""
    from opadpo_video.data import DPOCollator, DPODataset, unpack_slots
    from opadpo_video.qwen import load_model, load_processor, response_logprobs
    from opadpo_video.train_dpo import OPADPOTrainer
    from opadpo_video.train_opa import training_args

    cfg, video, root = setup
    proc = load_processor(cfg)
    ref_model = load_model(cfg, adapter=str(root / "opa"))
    data = DPODataset(cfg.work / "dpo.jsonl", proc, cfg)
    col = DPOCollator(proc.tokenizer.pad_token_id)
    ref = {}
    for i in range(len(data)):
        b = col([data[i]])
        lp = unpack_slots(response_logprobs(ref_model, b["batch"], b["lens"]), b["slots"], 1)
        ref[b["ids"][0]] = {k: v[0] for k, v in lp.items() if v[0] is not None}
    data.ref = ref
    policy = load_model(cfg, adapter=str(root / "opa"), trainable=True)
    args = training_args(cfg, output_dir=str(root / "dpo"), max_steps=2, per_device_train_batch_size=2,
                         gradient_accumulation_steps=1, learning_rate=1e-3, use_cpu=True, save_strategy="no")
    trainer = OPADPOTrainer(model=policy, args=args, train_dataset=data, data_collator=col, opa_cfg=cfg)
    result = trainer.train()
    assert np.isfinite(result.training_loss)
    assert any("dpo/lc" in h for h in trainer.state.log_history)
