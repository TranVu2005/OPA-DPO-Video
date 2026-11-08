# OPA-DPO-Video

Bản cài đặt lại **OPA-DPO** ("Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key", CVPR 2025, arXiv 2501.09695) cho **video**, trên **Qwen2.5-VL**. Code viết lại theo paper, không dùng LLaVA. Repo gốc (`../OPA-DPO`) chỉ dùng để đối chiếu chi tiết cài đặt.

## Pipeline (Algorithm 1)

| Bước | Lệnh | Đầu ra |
|---|---|---|
| 0. Chuẩn bị dữ liệu theo đúng split của Adaptive-SEED | `python -m opadpo_video.prepare_videohallucer --root <VideoHallucer>` | `data/prompts.jsonl`, `data/eval_vh.jsonl`, `data/split.json` |
| 1. Rollout y_Gen từ policy gốc (top-k 30, top-p 0.95, T=1) | `python -m opadpo_video.rollout --prompts data/prompts.jsonl` | `work/rollouts/rank*.jsonl` |
| 2. Expert chấm từng câu (S_hal 1-4, loại lỗi) và sửa tối thiểu → y_Rev | `python -m opadpo_video.expert` | `work/reports.jsonl` |
| 2b. Ghép dataset | `python -m opadpo_video.build_dataset` | `work/sft.jsonl`, `work/dpo.jsonl` |
| 3. OPA: LoRA-SFT trên y_GT + y_Rev (Eq. 4) | `torchrun ... -m opadpo_video.train_opa` | `runs/opa/final` |
| 4a. Log-prob tham chiếu của π_OPA | `torchrun ... -m opadpo_video.precompute_ref --adapter runs/opa/final` | `work/ref_logps/rank*.pt` |
| 4b. OPA-DPO (Eq. 10), tiếp tục LoRA của OPA | `torchrun ... -m opadpo_video.train_dpo train.init_adapter=runs/opa/final` | `runs/opadpo/final` (LoRA adapter) |
| 5. Đánh giá yes/no như Adaptive-SEED | `python -m opadpo_video.evaluate --name ... [--adapter ...]` | `runs/eval/<name>/results.csv` |

Mọi tham số nằm trong [configs/default.yaml](configs/default.yaml). Có thể ghi đè bằng `key=value` trên dòng lệnh, ví dụ `dpo.beta=0.2 video.n_frames=32`.

## Dữ liệu đầu vào

`data/prompts.jsonl`, mỗi dòng:

```json
{"id": "vid_0001", "video": "/g/data/ab12/videos/0001.mp4", "prompt": "Describe this video in detail.", "gt_response": "..."}
```

- `gt_response` là câu trả lời tham chiếu, đóng vai trò y_GT. Paper lấy y_GT từ response được ưa thích của RLAIF-V. Với video có thể lấy caption chi tiết của LLaVA-Video-178K hoặc ShareGPT4Video. Trường này **không bắt buộc**: thiếu GT thì các cặp có y_GT bị bỏ, chỉ còn cặp Rev/Gen.
- `reference` là thông tin đúng **chỉ gửi cho expert**, không bao giờ được train. Expert coi đây là nhãn chắc chắn đúng.
- Nên dùng prompt mô tả hoặc câu hỏi mở có câu trả lời dài. OPA-DPO dựa vào việc sửa ảo giác ở từng câu, nên câu trả lời chỉ có "Yes."/"No." không có gì để sửa.

### Câu hỏi yes/no (ví dụ VideoHallucer)

Dùng dạng "trả lời yes/no rồi giải thích", để câu trả lời vẫn dài nhiều câu. Đưa nhãn vào `reference`, **để trống `gt_response`**:

```json
{"id": "temporal/86|basic", "video": ".../v_7p99ez6MEeo.mp4",
 "prompt": "Does 'they continue lobbing the ball over the net' happen later than 'a group of people are on a sandy volleyball field'?\nAnswer with yes or no, then explain why.",
 "reference": "Correct answer to this question: yes."}
```

Lý do: nếu đặt `gt_response = "Yes."`, bước OPA sẽ dạy model trả lời cụt một chữ, và cặp (GT ≻ Gen) thành so 1 token với cả đoạn. Để trống thì dữ liệu chỉ dùng cặp Rev ≻ Gen và (Rev, m ≻ m′).

Lưu ý thêm:
- Giữ nguyên câu hỏi như lúc đánh giá, chỉ thêm đuôi yêu cầu giải thích. Đánh giá vẫn dùng đúng prompt `"\nAnswer with yes or no."`.
- Expert được dặn tách "Yes."/"No." thành câu riêng, để câu đáp án sai nhận đúng trọng số W_hal.
- Giữ cả câu hỏi basic và hallucination của cùng video (một câu đáp án "yes", một câu "no"), để model không học thiên lệch về một phía.
- Paper dùng khoảng 4.8k mẫu.

## Ánh xạ paper → code

| Paper | Code |
|---|---|
| Eq. 7, language correction: (GT ≻ Gen) và (Rev ≻ Gen) có trọng số W_hal | `losses.py`, nhánh `lc` |
| Eq. 8, image focus: (GT, m ≻ m′) và (Rev, m ≻ m′) có trọng số W_img | nhánh `if_`; m′ tạo bằng `video.mask_video` |
| Eq. 9, anchor σ(β·r − δ) cho GT và Rev | nhánh `anc` |
| Eq. 10, L = L_LC + γ1·L_IF + γ2·L_Anc với γ1=0.2, γ2=1, β=0.1, δ=0 | `opa_dpo_loss`, `configs/default.yaml` |
| Table 6, W_hal | `weights.hal`: điểm 1→2.5, 2→2.0, 3→1.5, 4→1.0 |
| Table 7, W_img | `weights.img`: lỗi nhận dạng thị giác 3.0, còn lại 1.0 |
| A.1: LoRA r=256, α=512 trên cả backbone; OPA 2 epoch, lr 2e-5, batch 128; DPO 4 epoch, lr 1e-6, batch 32 | `model.*`, `scripts/gadi/train_*.pbs` |

### Chỗ paper và code gốc khác nhau

1. **Table 6 in ngược.** Paper ghi điểm 4 ("không ảo giác") ứng với trọng số 2.5. Code gốc (`utils/data_utils_dpo.py`) lại cho điểm 1 (sai hoàn toàn) trọng số 2.5. Repo này theo code gốc: câu càng sai thì trọng số càng lớn, đúng với mục đích "tập trung cập nhật vào phần sai".
2. **DPO theo token hay theo chuỗi.** Paper (Eq. 7-9) viết DPO theo chuỗi: σ(β·Σ_i W_i·log-ratio). Code gốc lại áp sigmoid **cho từng vị trí token**, ghép chosen/rejected theo chỉ số, rồi chia trung bình cho độ dài pad cố định 896.
3. **Dấu anchor.** Anchor trong code gốc kéo reward của y_GT lên nhưng lại kéo reward của y_Rev **xuống**, và mỗi chiều được tính hai lần. Paper (Eq. 9) kéo cả hai lên.

Mặc định `dpo.loss_mode=paper` theo paper. Đặt `dpo.loss_mode=official` để tái hiện đúng code gốc (điểm 2 và 3).

### Phần thêm cho video (paper không có)

- **Tiền xử lý giống Adaptive-SEED:** 8 frame lấy đều theo thời gian, resize vuông 448×448 bằng PIL bicubic (`video.frame_size`), `fps = số frame / thời lượng` truyền cho processor để mRoPE có thông tin thời gian. Đặt `video.frame_size=null` để giữ tỉ lệ khung hình.
- **Video méo m′** (`video.mask_method`), seed theo `id` nên lượt reference và lúc train méo đúng cùng chỗ:

  | Kiểu | Phá gì |
  |---|---|
  | `pixel` (paper) | Che 30% pixel mỗi frame bằng màu trung bình |
  | `tube` | Che cùng vị trí pixel ở mọi frame |
  | `frame` | Che nguyên 30% số frame |
  | `shuffle` | Xáo thứ tự các **cặp** frame (Qwen gộp 2 frame thành một token thời gian), không bao giờ giữ nguyên thứ tự |
  | `reverse` | Đảo ngược thời gian |
  | `freeze` | Lặp frame giữa, mất chuyển động |

- Prompt cho expert ([expert.py](opadpo_video/expert.py)) được viết lại cho video. Có thêm loại lỗi `temporal error` (sai thứ tự, thời điểm hoặc số lần lặp của sự kiện), trọng số W_img = 3.0.
- y_Gen giữ **nguyên văn bản policy sinh ra**. Mỗi câu expert chép lại được dò vị trí trong y_Gen (khớp chính xác, rồi bỏ qua khoảng trắng, rồi khớp mờ), và trọng số gán theo khoảng ký tự. Code gốc thì ghép lại y_Gen từ các câu expert chép.

## Thiết kế

- **Reference tính trước.** Reference cố định suốt DPO, nên log-prob từng token của nó được tính một lần (`precompute_ref`). Lúc train chỉ cần một model.
- **Không dùng nhầm reference.** Mỗi thư mục `work/<dpo.ref_dir>/meta.json` ghi lại adapter, kiểu m′, số frame... đã dùng. `train_dpo` từ chối chạy nếu khác cấu hình hiện tại.
- **Một forward mỗi bước.** Năm hàng (GT·m, Gen·m, Rev·m, GT·m′, Rev·m′) của mọi mẫu được gộp và pad trái trong một batch. Cách này an toàn với DDP và chỉ tính logits cho phần response.
- Policy DPO **tiếp tục train chính LoRA của OPA**, như paper A.1.

## Thí nghiệm so sánh với Adaptive-SEED

Mọi hàng dùng cùng Qwen2.5-VL-7B, cùng split VideoHallucer (`temporal` + `object_relation`, seed 0, 150/50/300 cặp), cùng 8 frame 448², cùng cách đánh giá.

| Hàng | Cách chạy trên Gadi |
|---|---|
| Qwen2.5-VL gốc | `qsub -v NAME=base scripts/gadi/eval.pbs` |
| Chỉ OPA (SFT) | `train_opa.pbs`, rồi eval với `ADAPTER=$OUT/opa/final` |
| DPO thường (không OPA, không trọng số, không IF/anchor) | `qsub -v PLAIN=1 scripts/gadi/train_dpo.pbs` |
| OPA-DPO bản paper (m′ = pixel) | `qsub scripts/gadi/train_dpo.pbs` |
| OPA-DPO với m′ theo thời gian | `qsub -v MASK=shuffle` (hoặc `reverse`, `freeze`) `scripts/gadi/train_dpo.pbs` |
| SFT yes/no (không expert) | `qsub -v BINARY=1 scripts/gadi/train_opa.pbs` |
| DPO yes/no (không expert) | `qsub -v BINARY=1,GAMMA_IF=0 scripts/gadi/train_dpo.pbs` |
| DPO yes/no + m′ (không expert) | `qsub -v BINARY=1,MASK=shuffle scripts/gadi/train_dpo.pbs` (hoặc `MASK=pixel`) |

### Baseline yes/no không cần expert

`build_binary.py` (chạy trong `prepare.sh`) tạo dữ liệu thẳng từ nhãn TRAIN của VideoHallucer, cùng split với SEED, vào `$OUT/work_binary`: prompt **giống hệt prompt đánh giá**, y_Rev = nhãn đúng ("Yes."/"No."), y_Gen = nhãn ngược lại, không có y_GT. Không có rollout, không có expert.

Đây **không phải** OPA-DPO hay TL-DPO: với câu trả lời một token thì cả hai nhãn đều on-policy (bước OPA không còn gì để làm), trọng số theo câu và target span đều trùng với toàn bộ câu trả lời. Phần còn ý nghĩa là DPO trên nhãn và số hạng so sánh video gốc với video méo (L_IF). Khi báo cáo, gọi đúng tên "DPO yes/no" và "DPO yes/no + m′"; không dùng chúng để kết luận về OPA-DPO.

`results.csv` có các dòng `VH_VAL`, `VH_TEST`, `VH_TEST_temporal`, `VH_TEST_object_relation`, `EH_ALL`, `eh_entire`, `eh_misleading`, `eh_mix`, với cột `q_acc`, `pair_acc`, `yes_rate`, và so với model gốc: `delta_pp`, `wrong_to_correct`, `correct_to_wrong`, `mean_abs_margin_change`. Định nghĩa giống `results.csv` của Adaptive-SEED.

**Kiểm tra trước khi tin kết quả:** dòng `base` của repo này phải khớp dòng `base` trong `results.csv` của Adaptive-SEED (cùng `VH_TEST`, cùng `EH_ALL`). Nếu lệch, nghĩa là tiền xử lý hoặc prompt chưa giống, phải sửa trước khi so sánh.

### Backbone khác: Qwen3-VL-8B và InternVL3.5-8B

Baseline yes/no (SFT, DPO, DPO + m′) và đánh giá chạy được trên cùng hai checkpoint mà Adaptive-SEED dùng: `Qwen/Qwen3-VL-8B-Instruct` và `OpenGVLab/InternVL3_5-8B-Instruct`. Chọn bằng `MODEL=qwen3|internvl` (mặc định `qwen25`). Rollout, expert và OPA-DPO đầy đủ vẫn chỉ có cho Qwen2.5-VL.

- **Đầu vào giống SEED** (`models/qwen.py`, `models/internvl.py`): Qwen3-VL nhận video kèm `video_metadata` và `do_sample_frames=False`. InternVL nhận mỗi frame là một ảnh 448², prompt `Frame{i}: <image>`, chuẩn hoá ImageNet, không chia tile.
- **Token trả lời:** Qwen3-VL trả lời chữ thường, nên margin là `logit("yes") − logit("no")` và nhãn train "Yes."/"No." được đổi thành "yes."/"no.". Qwen2.5 và InternVL giữ "Yes"/"No". Job dừng ngay nếu token đầu của nhãn train khác token được chấm.
- **LoRA** dùng đúng `LORA_TARGETS` của SEED cho từng model: attention q/k/v/o của LM, cộng `visual.merger` (Qwen3) hoặc `mlp1.1/3` (InternVL).
- **Env riêng:** cần transformers 4.57, build một lần thành `opadpo-video-tf457` (`env_requirements_tf457.txt`). `env.sh` tự chọn tarball này khi `MODEL` khác `qwen25`.
- **Tên output** thêm hậu tố `_qwen3` hoặc `_internvl`: `binary_sft_qwen3`, `binary_dpo_lr1e-4_qwen3`, `work_binary/ref_logps_qwen3_base_pixel`, `eval/base_qwen3`. Dữ liệu `work_binary` và `eval_vh.jsonl` dùng chung cho mọi model.
- **Vị trí model:** `env.sh` tự tìm HF cache có model, trong `$S/hf_cache`, `/g/data/hn98/$USER/models/hf_cache_*` và `/scratch/jp09/$USER/hf_cache_*`. `bash scripts/gadi/find_models.sh` liệt kê mọi bản đang có. Nếu model nằm chỗ khác, truyền `HF_HOME=<thư mục chứa hub/>`.

## Chạy trên Gadi (1 H200, giống `run.md` của Adaptive-SEED)

Đường dẫn (sửa trong [scripts/gadi/env.sh](scripts/gadi/env.sh) nếu khác):

| Mục | Giá trị |
|---|---|
| Project / queue | `hn98`, `gpuhopper`, 1 GPU, 12 CPU, 160 GB |
| Code | `/scratch/hn98/$USER/OPA-DPO-Video` |
| venv, HF cache | Dùng chung với Adaptive-SEED: `/scratch/hn98/$USER/venv`, `/scratch/hn98/$USER/hf_cache` |
| VideoHallucer | `/scratch/jp09/dd9648/datasets_video_hallu/videohallucer` (chỉ đọc) |
| EventHallusion | `/scratch/hn98/$USER/runs/exp1/EventHallusion` (đã dựng bằng `prep_eh` của Adaptive-SEED) |
| Đầu ra | `/scratch/hn98/$USER/runs/opadpo/{data,work,work_binary,opa,opadpo_*,binary_*,eval}` |

```bash
# Login node (có internet), một lần, sau mục 1.2-1.3 của run.md Adaptive-SEED:
cd /scratch/hn98/$USER/OPA-DPO-Video
bash scripts/gadi/setup.sh            # cài thêm openai, av, pyyaml vào venv chung
bash scripts/gadi/prepare.sh          # split giống SEED, prompts.jsonl, eval_vh.jsonl

qsub -v NAME=base scripts/gadi/eval.pbs   # mốc; so với dòng base của SEED
qsub scripts/gadi/rollout.pbs             # 1
qsub scripts/gadi/expert_copyq.pbs        # 2 + 2b (API, cần ~/.opadpo_key)
#   hoặc qsub scripts/gadi/expert_vllm.pbs   (Qwen2.5-VL-72B-AWQ offline)
qsub scripts/gadi/train_opa.pbs           # 3
qsub scripts/gadi/train_dpo.pbs           # 4 (thêm -v MASK=... hoặc -v PLAIN=1 cho các hàng khác)
qsub -v NAME=opadpo_pixel,ADAPTER=/scratch/hn98/$USER/runs/opadpo/opadpo_pixel/final scripts/gadi/eval.pbs
```

Ghi chú:
- **Node GPU không có internet.** Model đọc từ `hf_cache` với `HF_HUB_OFFLINE=1`. Bước gọi API expert chạy ở `copyq`.
- Theo dõi: `qstat -u $USER`, log ở `opadpo_*.o<jobid>` trong thư mục repo.
- Mọi bước chạy lại được khi job bị ngắt: rollout, expert, evaluate bỏ qua các `id` đã xong; train tự resume từ checkpoint mới nhất.
- `prepare.sh` mặc định lấy prompt từ cặp train của `temporal` (yes/no + giải thích, cộng 4 prompt mô tả thời gian cho mỗi video). Đổi bằng `TRAIN_SUBSETS="temporal object_relation" DESCRIBE=2 bash scripts/gadi/prepare.sh`.
- Dữ liệu train nhỏ (vài trăm video), xa so với 4.8k mẫu của paper. Xem thêm phần thảo luận về nguồn video ngoài trước khi báo cáo kết quả chính.

## Test

```bash
pytest -q                                           # CPU: loss, trọng số, mask, split giống SEED, chỉ số đánh giá
OPADPO_SMOKE=1 pytest -q tests/test_smoke_qwen.py   # Qwen2.5-VL thu nhỏ, chạy end-to-end trên CPU
```

`tests/test_protocol.py` gọi thẳng code chia dữ liệu của `../Adaptive-SEED-` trên một bộ VideoHallucer giả và kiểm tra split giống hệt (bỏ qua nếu không có thư mục đó).
