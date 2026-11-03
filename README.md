# OPA-DPO-Video

Bản cài đặt lại **OPA-DPO** ("Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key", CVPR 2025, arXiv 2501.09695) cho **video**, trên **Qwen2.5-VL**. Code viết lại theo paper, không dùng LLaVA. Repo gốc (`../OPA-DPO`) chỉ dùng để đối chiếu chi tiết cài đặt.

## Pipeline (Algorithm 1)

| Bước | Lệnh | Đầu ra |
|---|---|---|
| 1. Rollout y_Gen từ policy gốc (top-k 30, top-p 0.95, T=1) | `python -m opadpo_video.rollout --prompts data/prompts.jsonl` | `work/rollouts/rank*.jsonl` |
| 2. Expert chấm từng câu (S_hal 1-4, loại lỗi) và sửa tối thiểu → y_Rev | `python -m opadpo_video.expert` | `work/reports.jsonl` |
| 2b. Ghép dataset | `python -m opadpo_video.build_dataset` | `work/sft.jsonl`, `work/dpo.jsonl` |
| 3. OPA: LoRA-SFT trên y_GT + y_Rev (Eq. 4) | `torchrun ... -m opadpo_video.train_opa` | `runs/opa/final` |
| 4a. Log-prob tham chiếu của π_OPA | `torchrun ... -m opadpo_video.precompute_ref --adapter runs/opa/final` | `work/ref_logps/rank*.pt` |
| 4b. OPA-DPO (Eq. 10), tiếp tục LoRA của OPA | `torchrun ... -m opadpo_video.train_dpo train.init_adapter=runs/opa/final` | `runs/opadpo/final` (LoRA adapter) |

Mọi tham số nằm trong [configs/default.yaml](configs/default.yaml). Có thể ghi đè bằng `key=value` trên dòng lệnh, ví dụ `dpo.beta=0.2 video.n_frames=32`.

## Dữ liệu đầu vào

`data/prompts.jsonl`, mỗi dòng:

```json
{"id": "vid_0001", "video": "/g/data/ab12/videos/0001.mp4", "prompt": "Describe this video in detail.", "gt_response": "..."}
```

- `gt_response` là câu trả lời tham chiếu, đóng vai trò y_GT. Paper lấy y_GT từ response được ưa thích của RLAIF-V. Với video có thể lấy caption chi tiết của LLaVA-Video-178K hoặc ShareGPT4Video. Trường này **không bắt buộc**: thiếu GT thì các cặp có y_GT bị bỏ, chỉ còn cặp Rev/Gen.
- Nên dùng prompt mô tả hoặc câu hỏi mở có câu trả lời dài. OPA-DPO dựa vào việc sửa ảo giác ở từng câu, nên dữ liệu hỏi yes/no (như VideoHallucer) gần như không có câu để sửa. Các bộ yes/no đó nên để đánh giá.
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

- Mỗi video lấy `video.n_frames` frame đều theo thời gian. Processor nhận `fps` để mRoPE của Qwen2.5-VL có thông tin thời gian.
- m′ là video bị che 30% pixel ở mỗi frame, phần bị che thay bằng màu trung bình của frame (`mask_method=pixel`). Có thêm hai kiểu che: `tube` (cùng vị trí pixel ở mọi frame) và `frame` (che nguyên frame, nhắm vào hiểu biết thời gian). Mask có seed theo `id`, nên lượt tính reference và lúc train che đúng cùng một chỗ.
- Prompt cho expert ([expert.py](opadpo_video/expert.py)) được viết lại cho video. Có thêm loại lỗi `temporal error` (sai thứ tự, thời điểm hoặc số lần lặp của sự kiện), trọng số W_img = 3.0.
- y_Gen giữ **nguyên văn bản policy sinh ra**. Mỗi câu expert chép lại được dò vị trí trong y_Gen (khớp chính xác, rồi bỏ qua khoảng trắng, rồi khớp mờ), và trọng số gán theo khoảng ký tự. Code gốc thì ghép lại y_Gen từ các câu expert chép.

## Thiết kế

- **Reference tính trước.** π_OPA cố định suốt OPA-DPO, nên log-prob từng token của nó được tính một lần (`precompute_ref`). Nhờ vậy lúc train chỉ cần một model; reference forward một lần thay vì một lần mỗi epoch.
- **Một forward mỗi bước.** Năm hàng (GT·m, Gen·m, Rev·m, GT·m′, Rev·m′) của mọi mẫu được gộp và pad trái trong một batch. Cách này an toàn với DDP và chỉ tính logits cho phần response.
- Policy DPO **tiếp tục train chính LoRA của OPA**, như paper A.1.

## Chạy trên Gadi (H200)

```bash
# Trên login node (có internet), một lần:
PROJECT_CODE=ab12 bash scripts/gadi/setup.sh
# Sửa PROJECT_CODE trong scripts/gadi/env.sh và dòng #PBS -P / storage trong các file .pbs

qsub scripts/gadi/rollout.pbs                 # 1
qsub scripts/gadi/expert_copyq.pbs            # 2 + 2b, API qua copyq (có internet)
#   hoặc qsub scripts/gadi/expert_vllm.pbs    #   expert chạy offline bằng vLLM trên H200
qsub scripts/gadi/train_opa.pbs               # 3
qsub scripts/gadi/train_dpo.pbs               # 4a + 4b
```

Ghi chú Gadi:
- **Node tính toán không có internet.** Model phải tải trước về `/g/data`, và `env.sh` đặt `HF_HUB_OFFLINE=1`. Bước gọi API expert chạy ở `copyq`, hoặc dùng `expert_vllm.pbs` để chạy expert là model mở ngay trên node GPU.
- **Tên queue H200** (`gpuhopper`), số CPU và RAM mỗi GPU trong các file `.pbs` là giá trị phỏng đoán. Hãy kiểm tra bằng `nqstat` và tài liệu NCI rồi sửa.
- Môi trường dùng CUDA 12.4, torch 2.6, transformers 4.51.3, peft 0.15.2 (giống Adaptive-SEED). Môi trường gốc của OPA-DPO (CUDA 11.7, torch 2.0) không hỗ trợ Hopper.
- Bộ nhớ: mặc định 16 frame × 448² và batch 1 mỗi GPU, có gradient checkpointing. H200 141GB còn dư, có thể tăng `video.n_frames`.
- Mọi bước đều chạy lại được khi job bị ngắt: rollout và expert bỏ qua các `id` đã xong, còn train tự resume từ checkpoint mới nhất.

## Test

```bash
pytest -q                                    # loss, trọng số, mask (CPU, không cần model)
OPADPO_SMOKE=1 pytest -q tests/test_smoke_qwen.py   # Qwen2.5-VL thu nhỏ, chạy end-to-end trên CPU (~8 phút)
```

## Đánh giá

Repo này chưa có script benchmark. Dùng `runs/opadpo/final` (một LoRA adapter, nạp bằng `PeftModel.from_pretrained`) với các bộ đánh giá video ảo giác, ví dụ VideoHallucer và EventHallusion trong `../Adaptive-SEED-`. Nên có thêm một benchmark caption hoặc mô tả, vì OPA-DPO tối ưu cho câu trả lời dài.
