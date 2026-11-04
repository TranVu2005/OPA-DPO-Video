"""CPU tests for the comparison protocol: SEED-identical split, temporal m', metrics, reference checks."""
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opadpo_video.config import load_config
from opadpo_video.evaluate import metrics, split_groups
from opadpo_video.prepare_videohallucer import build, load_items, seed_split, video_index
from opadpo_video.video import mask_video

SEED_SRC = Path(__file__).resolve().parents[2] / "Adaptive-SEED-" / "src"


def _fake_vh(root: Path, n_pairs=40, seed=1):
    """VideoHallucer-like tree: <sub>/<sub>.json + <sub>/videos/*.mp4, with shared and missing videos."""
    rng = random.Random(seed)
    for sub in ("temporal", "object_relation"):
        (root / sub / "videos").mkdir(parents=True)
        data = []
        for i in range(n_pairs):
            vb = f"{sub[:3]}_{rng.randint(0, n_pairs // 2)}.mp4"  # some videos reused across pairs
            vh = vb if rng.random() < 0.5 else f"{sub[:3]}_h{i}.mp4"
            for v in (vb, vh):
                if not (i % 17 == 3 and v == vh):  # a few hallucination videos missing on disk
                    (root / sub / "videos" / v).write_bytes(b"")
            pair = {"basic": {"video": f"videos/{vb}", "question": f"Q{i} basic?", "answer": "yes"},
                    "hallucination": {"video": f"videos/{vh}", "question": f"Q{i} hallu?", "answer": "No "}}
            data.append(pair)
            if i % 9 == 0:
                data.append(pair)  # exact duplicate, must be de-duplicated
        (root / sub / f"{sub}.json").write_text(json.dumps(data))


@pytest.mark.skipif(not SEED_SRC.exists(), reason="Adaptive-SEED checkout not found next to this repo")
def test_split_identical_to_adaptive_seed(tmp_path):
    import glob as glob_mod

    root = tmp_path / "VideoHallucer"
    _fake_vh(root)
    subsets = ["temporal", "object_relation"]
    sys.path.insert(0, str(SEED_SRC))
    try:
        from seed_video.data import videohallucer as seed_vh
    finally:
        sys.path.pop(0)
    rt = SimpleNamespace(glob=glob_mod, json=json, os=os, ROOT=str(root),
                         VIDEO_INDEX={k: v for k, v in video_index(str(root), subsets).items()})
    seed_items = seed_vh.load_videohallucer(rt, subsets)
    ours = load_items(str(root), subsets)
    assert seed_items and [{k: d[k] for k in ours[0]} for d in seed_items] == ours

    # SEED's split (data/videohallucer.py run()) vs ours.
    rt.pairs = {}
    for it in seed_items:
        rt.pairs.setdefault(it["pair_id"], []).append(it)
    pids = sorted(rt.pairs)
    random.Random(0).shuffle(pids)
    tr, trv = seed_vh.take(rt, pids, 12, set())
    rest = [p for p in pids if p not in set(tr)]
    va, vav = seed_vh.take(rt, rest, 5, trv)
    rest = [p for p in rest if p not in set(va)]
    te, _ = seed_vh.take(rt, rest, 30, trv | vav)
    split = seed_split(ours, 12, 5, 30, 0)
    assert (split["train"], split["val"], split["test"]) == (tr, va, te)


def test_prompts_only_use_train_videos(tmp_path):
    root = tmp_path / "VH"
    _fake_vh(root)
    items = load_items(str(root), ["temporal", "object_relation"])
    split = seed_split(items, 12, 5, 30, 0)
    prompts, evals = build(items, split, ["temporal"], describe=2)
    held = set(split["videos"]["val"]) | set(split["videos"]["test"])
    assert prompts and not any(p["video"] in held for p in prompts)
    assert all(p["subset"] == "temporal" for p in prompts)
    yn = [p for p in prompts if p["id"].endswith("|yn")]
    assert yn and all(p["prompt"].endswith("then explain why.") and "Correct answer" in p["reference"] for p in yn)
    assert "gt_response" not in prompts[0]
    assert {e["split"] for e in evals} == {"VH_VAL", "VH_TEST"} and all(e["answer"] in ("yes", "no") for e in evals)


@pytest.mark.parametrize("method", ["shuffle", "reverse", "freeze"])
def test_temporal_masks_keep_frame_content(method):
    t = 8
    v = torch.arange(t, dtype=torch.uint8).view(t, 1, 1, 1).expand(t, 3, 4, 4).contiguous()
    out = mask_video(v, 0.3, method, seed=5)
    order = out[:, 0, 0, 0].tolist()
    assert out.shape == v.shape and torch.equal(out, mask_video(v, 0.3, method, seed=5))
    if method == "reverse":
        assert order == list(range(t))[::-1]
    elif method == "freeze":
        assert order == [t // 2] * t
    else:
        assert sorted(order) == list(range(t)) and order != list(range(t))
        assert all(order[i + 1] == order[i] + 1 for i in range(0, t, 2))  # pairs kept together


def test_metrics_match_seed_definitions():
    items = [
        {"id": "p1|basic", "pair_id": "p1", "split": "VH_TEST", "subset": "temporal", "answer": "yes"},
        {"id": "p1|hallucination", "pair_id": "p1", "split": "VH_TEST", "subset": "temporal", "answer": "no"},
        {"id": "p2|basic", "pair_id": "p2", "split": "VH_TEST", "subset": "object_relation", "answer": "yes"},
        {"id": "p2|hallucination", "pair_id": "p2", "split": "VH_TEST", "subset": "object_relation", "answer": "no"},
        {"id": "e1|eh", "pair_id": "e1", "split": "eh_entire", "subset": "eh_entire", "answer": "no"},
    ]
    marg = {"p1|basic": 2.0, "p1|hallucination": -1.0, "p2|basic": 1.0, "p2|hallucination": 0.5, "e1|eh": -3.0}
    base = {"p1|basic": -2.0, "p1|hallucination": -1.0, "p2|basic": 1.0, "p2|hallucination": -0.5, "e1|eh": -3.0}
    groups = split_groups(items)
    assert set(groups) == {"VH_TEST", "VH_TEST_temporal", "VH_TEST_object_relation", "eh_entire", "EH_ALL"}
    m = metrics(groups["VH_TEST"], marg, base)
    assert m["q_acc"] == 0.75 and m["pair_acc"] == 0.5 and m["yes_rate"] == 0.75
    assert (m["wrong_to_correct"], m["correct_to_wrong"]) == (1, 1) and m["delta_pp"] == 0.0
    assert np.isnan(metrics(groups["EH_ALL"], marg, None)["pair_acc"])


def test_reference_settings_must_match(tmp_path):
    from opadpo_video.precompute_ref import load_ref_logps, ref_meta

    cfg = load_config(None, [f"work_dir={tmp_path.as_posix()}", "dpo.ref_dir=ref_x"])
    (tmp_path / "ref_x").mkdir()
    (tmp_path / "ref_x" / "meta.json").write_text(json.dumps(ref_meta(cfg, "runs/opa/final")))
    torch.save({"a": {"rev_m": torch.zeros(3)}}, tmp_path / "ref_x" / "rank0.pt")
    assert "a" in load_ref_logps(cfg, "runs/opa/final")
    with pytest.raises(SystemExit, match="adapter"):
        load_ref_logps(cfg, None)  # plain DPO must not reuse pi_OPA log-probs
    cfg.video.mask_method = "shuffle"
    with pytest.raises(SystemExit, match="mask_method"):
        load_ref_logps(cfg, "runs/opa/final")


def test_binary_baseline_rows(tmp_path):
    from opadpo_video.build_binary import binary_rows
    from opadpo_video.prepare_videohallucer import YN_SUFFIX
    from opadpo_video.weights import span_weights_to_tokens

    root = tmp_path / "VH"
    _fake_vh(root)
    items = load_items(str(root), ["temporal", "object_relation"])
    split = seed_split(items, 12, 5, 30, 0)
    dpo, sft = binary_rows(items, split, ["temporal"])
    held = set(split["videos"]["val"]) | set(split["videos"]["test"])
    assert dpo and len(dpo) == len(sft) and not any(r["video"] in held for r in dpo)
    for r in dpo:
        assert r["prompt"].endswith(YN_SUFFIX) and r["gt"] is None and {r["gen"], r["rev"]} == {"Yes.", "No."}
        assert span_weights_to_tokens([(0, 3), (3, 4)], r["rev_spans"], 2) == [1.0, 1.0]
    truth = {f"{it['pair_id']}|{it['role']}|bin": it["answer"] for it in items}
    assert all(r["rev"].lower().rstrip(".") == truth[r["id"]] for r in dpo)
    assert {r["rev"] for r in dpo} == {"Yes.", "No."}  # both labels present (basic=yes, hallucination=no)
