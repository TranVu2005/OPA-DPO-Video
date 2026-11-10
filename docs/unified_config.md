# Cấu hình thống nhất: Adaptive-SEED, OPA-DPO-Video, TL-DPO

Mọi phương pháp (SFT, DPO, OPA-DPO, TL-DPO, SEED) được chấm theo **một** quy trình eval và train
với **một** budget, để chênh lệch giữa các dòng trong bảng chỉ đến từ phương pháp.
Quy trình eval dưới đây là của Adaptive-SEED (`src/seed_video/models/*.py`, `evaluation/`) và đã được
OPA-DPO-Video tái hiện (`opadpo_video/evaluate.py`).

**Điều kiện bắt buộc:** trước khi báo cáo một phương pháp, chạy mô hình gốc (không train) bằng pipeline
của phương pháp đó và kiểm tra base ra **đúng** các số ở mục 6. Nếu base lệch thì pipeline eval đang khác,
và kết quả sau train không so sánh được.

## 1. Dữ liệu

### VideoHallucer
- Nguồn: HF dataset `bigai-nlco/VideoHallucer`, revision `c93475bf1be466c1af90d005749ccb4e6b3ad066`.
- Subset: `temporal` và `object_relation`.
- Mỗi cặp gồm câu `basic` (đáp án "yes") và câu `hallucination` (đáp án "no"). Khử trùng theo
  (video, câu hỏi) của cả hai câu.
- Chia: sắp xếp `pair_id`, xáo bằng `random.Random(0)`, rồi lấy lần lượt 150 cặp train, 50 cặp val,
  tối đa 300 cặp test. Bỏ cặp nào có video đã nằm ở split trước, để train, val và test không chung video.
  Kết quả: **train 150 cặp, val 50 cặp (100 câu), test 155 cặp (310 câu)**.
- Dùng thẳng các file đã tạo sẵn trên Gadi, không tự chia lại:
  - `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/data/split.json`: danh sách pair id của từng split.
  - `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/data/eval_vh.jsonl`: các câu val và test.

### EventHallusion
- Chỉ dùng để eval, không train. Gồm cả **409 câu**: `entire` 114, `misleading` 102, `mix` 193.
- Thư mục: `/scratch/hn98/dd9648/runs/exp1/EventHallusion` (`questions/<split>_questions.json` + video).

## 2. Video

- Đọc bằng decord, lấy **8 frame** đều theo thời gian: `idx = np.linspace(0, N - 1, 8).round()`,
  với N là tổng số frame.
- Mỗi frame: PIL RGB, resize về **448 × 448**, bicubic, không giữ tỉ lệ.
- fps truyền cho model = `8 / (N / avg_fps)`.
- Theo backbone:
  - Qwen2.5-VL: `processor(text=..., videos=[video], fps=[fps])`.
  - Qwen3-VL: `processor(..., video_metadata=[{"total_num_frames": 8, "fps": fps,
    "duration": 8 / fps, "frames_indices": [0..7]}], do_sample_frames=False)`.
  - InternVL3.5: mỗi frame là 1 ảnh 448 (không chia tile), chuẩn hoá ImageNet
    (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), 256 token ảnh mỗi frame, `image_flags` = 1.

## 3. Prompt

- Nội dung user: `question.strip() + "\nAnswer with yes or no."`
- Qwen: chat template mặc định của processor, video đặt trước text, `add_generation_prompt=True`,
  không thêm system prompt.
- InternVL: conversation template của model (giữ system message mặc định), nội dung user là
  `"Frame1: <image>\n...Frame8: <image>\n" + câu hỏi`. Mỗi `<image>` được thay bằng
  `<img>` + `<IMG_CONTEXT>` × 256 + `</img>`.

## 4. Cách chấm

- Không sinh câu trả lời. Chỉ chạy **một lượt forward**, lấy logit ở vị trí cuối của prompt.
- `margin = logit(YES) − logit(NO)`. Dự đoán "yes" khi `margin > 0`.
- Token đáp án lấy bằng `tokenizer.encode(word, add_special_tokens=False)[0]`:

  | Backbone | YES | NO |
  |---|---|---|
  | Qwen2.5-VL-7B-Instruct | `Yes` | `No` |
  | Qwen3-VL-8B-Instruct | **`yes`** | **`no`** (chữ thường: Qwen3 đặt khoảng 99% xác suất vào hai token này, dưới 0.5% vào `Yes`/`No`) |
  | InternVL3_5-8B-Instruct | `Yes` | `No` |

- Batch 1, bf16, `model.eval()`.
- Attention: Qwen dùng `sdpa`; InternVL dùng eager (`use_flash_attn=False`, `trust_remote_code=True`).
- Môi trường: Qwen3-VL và InternVL3.5 cần transformers 4.57.6 (venv `/scratch/jp09/dd9648/venv_qwen3`).

## 5. Chỉ số báo cáo

| Cột | Cách tính |
|---|---|
| VH Q acc | tỉ lệ câu đúng trên 310 câu VH_TEST |
| VH pair acc | tỉ lệ cặp đúng **cả hai** câu trên 155 cặp VH_TEST |
| EH acc | tỉ lệ câu đúng trên 409 câu EventHallusion |

Nên ghi thêm `yes_rate` (tỉ lệ trả lời "yes"). Chỉ dùng VH_VAL để chọn cấu hình, không dùng test.

## 6. Base chuẩn (mô hình gốc, không train)

Pipeline nào cũng phải ra đúng các số này trước khi báo cáo phương pháp:

| Backbone | VH Q acc | VH pair acc | EH acc | Số câu đúng (VH / cặp / EH) |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 0.7419 | 0.5032 | 0.6601 | 230/310 · 78/155 · 270/409 |
| Qwen3-VL-8B-Instruct | 0.7806 | 0.6194 | 0.6406 | 242/310 · 96/155 · 262/409 |
| InternVL3_5-8B-Instruct | 0.6839 | 0.4258 | 0.4841 | 212/310 · 66/155 · 198/409 |

Margin của base cho từng câu nằm ở `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/eval/{base,base_qwen3,base_internvl}/margins.jsonl`.
Nếu tổng số không khớp, so margin từng câu với file này để tìm chỗ lệch.

## 7. Budget train (student budget của Adaptive-SEED)

| Mục | Giá trị |
|---|---|
| Dữ liệu train | toàn bộ TRAIN split: 150 cặp, khoảng 300 câu, cả `temporal` lẫn `object_relation` |
| Adapter | LoRA r = 16, α = 32, dropout = 0.05, bias none |
| Module có LoRA | attention q/k/v/o của LM + projector thị giác (regex bên dưới) |
| Optimizer | AdamW, lr 1e-4, weight decay 0, clip grad norm 1.0 |
| Lịch lr | cosine, warmup 5% số bước |
| Epoch / batch | 1 epoch, batch 1 × gradient accumulation 8 (batch hiệu dụng 8) |
| Độ chính xác | bf16 |
| Seed | 0 |
| Checkpoint | lấy checkpoint cuối epoch, không early stopping |

Regex module có LoRA (dùng làm `target_modules` của PEFT):

```text
Qwen2.5-VL : (.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.mlp\.(0|2))
Qwen3-VL   : (.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.(mlp\.(0|2)|linear_fc[12]))
InternVL3.5: (.*language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*mlp1\.(1|3))
```

Số tham số train được: 10,395,648 với Qwen2.5-VL; 15,597,568 với InternVL3.5.

Với phương pháp dạng DPO: reference là mô hình gốc; β và các hệ số riêng của phương pháp ghi rõ
trong báo cáo. Phần DPO trong OPA-DPO-Video hiện dùng β = 0.1.

## 8. Chỗ OPA-DPO-Video hiện chưa khớp mục 7

Các dòng SFT yes/no và DPO yes/no hiện có được train với hai khác biệt so với budget trên:

1. Dữ liệu train chỉ có `temporal` (130 câu, 17 bước) thay vì cả 2 subset (khoảng 300 câu, khoảng 38 bước).
2. Loss tính trên cả câu trả lời (`Yes` + `.` + `<|im_end|>`) thay vì chỉ trên token đáp án như SEED.

Kết quả eval của các dòng này vẫn đúng quy trình ở mục 1–5. Các run có `FULL=1`
(`train_opa.pbs`, `train_dpo.pbs`, dữ liệu từ `scripts/gadi/prepare_binary.sh`) khớp hoàn toàn mục 7:
train trên cả TRAIN split, loss / log-prob chỉ trên token đáp án, seed 0. Output có hậu tố `_full`.

## 9. Cách nhanh nhất để chấm checkpoint của phương pháp khác

Nếu checkpoint là adapter PEFT LoRA trên đúng model gốc ở mục 6, có thể chấm thẳng bằng script của
OPA-DPO-Video, như vậy chắc chắn cùng quy trình:

```bash
cd /scratch/hn98/dd9648/OPA-DPO-Video && git pull
qsub -l walltime=00:30:00,mem=64GB,jobfs=10GB \
  -v MODEL=qwen3,NAME=tldpo_qwen3,ADAPTER=/đường/dẫn/adapter,ENV_NAME=none,VENV=/scratch/jp09/dd9648/venv_qwen3 \
  scripts/gadi/eval.pbs
```

`MODEL` là `qwen25`, `qwen3` hoặc `internvl`. Với `qwen25` thì bỏ `ENV_NAME` và `VENV`.
Kết quả nằm ở `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/eval/<NAME>/results.csv`.
