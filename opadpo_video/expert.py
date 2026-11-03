"""Step 2: expert feedback. Sentence-level hallucination scores, error types and minimal revisions.

Needs network access to the expert endpoint. On Gadi compute nodes have no internet:
run this on a login node / copyq, or point `expert.base_url` at a vLLM server you start
inside the GPU job (e.g. Qwen2.5-VL-72B-Instruct).

  OPENAI_API_KEY=... python -m opadpo_video.expert
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config, cli
from .data import read_jsonl
from .video import load_video

SYSTEM_PROMPT = """You review answers produced by a video-language model and correct its hallucinations.

You receive: frames sampled uniformly over time from a video (in temporal order), the user's question, the MODEL RESPONSE, and possibly a REFERENCE ANSWER. The reference may be incomplete or partly wrong; the video frames are the primary evidence.

Go through the MODEL RESPONSE sentence by sentence:
- A sentence that is accurate and relevant stays exactly as it is.
- A partly correct sentence gets the smallest edit that fixes it.
- A sentence with hallucinated or irrelevant content is rewritten (or emptied if nothing of it can be kept), staying as close to the original wording as possible.
Only change something when you are certain it is wrong. Never introduce new unverifiable details. Keep the revision minimal: the corrected response must stay as close as possible to the original text.

Pay special attention to video-specific errors: objects or people that never appear, wrong actions, wrong order of events, wrong counts of repeated actions, wrong direction of motion, and events attributed to the wrong moment.

Return a JSON object:
{
  "video_description": "<thorough description of what happens in the video, in temporal order>",
  "sentences": [
    {
      "copied_content": "<the sentence copied verbatim from the MODEL RESPONSE>",
      "score": <1-4>,
      "error_type": "<correct | visual recognition error | temporal error | language comprehension error>",
      "objects": ["<wrong -> right>", ...],
      "rewritten_content": "<corrected sentence; identical to copied_content if score is 4>",
      "reason": "<why>"
    }
  ],
  "added": "<only if the corrected response still misses information essential to answer the question; else empty>"
}

Scores: 4 = fully correct; 3 = mostly correct, minor fix (e.g. a count, size or colour); 2 = substantial error (wrong object, wrong action, wrong relation or event order); 1 = completely wrong or irrelevant.
Error types: "correct" only with score 4; "visual recognition error" when something visible is misperceived (object, attribute, action); "temporal error" when the content is right but its order, timing, duration or repetition is wrong; "language comprehension error" when the visual content is understood but the statement is wrong for non-visual reasons (knowledge, reasoning, misreading the question).
Every sentence of the MODEL RESPONSE must appear exactly once in "sentences", in order."""


def _frames_b64(path: str, cfg: Config) -> list[str]:
    from PIL import Image

    video, _ = load_video(path, cfg.expert.n_frames)
    out = []
    for frame in video.permute(0, 2, 3, 1).numpy():
        img = Image.fromarray(frame)
        img.thumbnail((cfg.expert.frame_max_side, cfg.expert.frame_max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def build_messages(row: dict, cfg: Config) -> list[dict]:
    content = [{"type": "text", "text": f"Video frames ({cfg.expert.n_frames}, in temporal order):"}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in _frames_b64(row["video"], cfg)]
    text = f"QUESTION:\n{row['prompt']}\n\nMODEL RESPONSE:\n{row['generated']}\n"
    if row.get("gt_response"):
        text += f"\nREFERENCE ANSWER:\n{row['gt_response']}\n"
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def query(client, row: dict, cfg: Config) -> dict:
    ec = cfg.expert
    last = None
    for attempt in range(ec.max_retries):
        try:
            resp = client.chat.completions.create(
                model=ec.model, messages=build_messages(row, cfg), temperature=ec.temperature,
                max_tokens=ec.max_tokens, response_format={"type": "json_object"},
            )
            report = json.loads(resp.choices[0].message.content)
            if not isinstance(report.get("sentences"), list):
                raise ValueError("report has no 'sentences' list")
            return report
        except Exception as e:  # network errors, rate limits, malformed JSON
            last = e
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"{row['id']}: {last}")


def main():
    cfg, _ = cli(__doc__)
    from openai import OpenAI

    client = OpenAI(base_url=cfg.expert.base_url, api_key=os.environ.get(cfg.expert.api_key_env, "EMPTY"))
    rollouts = [r for p in sorted((cfg.work / "rollouts").glob("rank*.jsonl")) for r in read_jsonl(p)]
    out_path = cfg.work / "reports.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rollouts if r["id"] not in done and r["generated"].strip()]
    print(f"{len(todo)} rollouts to review ({len(done)} done)", flush=True)

    lock = threading.Lock()
    failed = 0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(cfg.expert.workers) as pool:
        futures = {pool.submit(query, client, r, cfg): r for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                report = fut.result()
            except Exception as e:
                failed += 1
                print(f"failed: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps({"id": row["id"], "report": report}, ensure_ascii=False) + "\n")
                f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)} ({failed} failed)", flush=True)
    print(f"done, {failed} failed (rerun to retry them)")


if __name__ == "__main__":
    main()
