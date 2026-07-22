# Hướng phát triển tương lai — các nhánh ĐÃ GỠ khỏi pipeline chính

Tài liệu này lưu lại hai hướng thí nghiệm đã được **gỡ khỏi pipeline train chính** (để tập trung
báo cáo thực nghiệm trên nhánh single-stage pose→text + RL fine-tune decoder, ở các mức subset
PHOENIX 5 / 10 / 25% + thí nghiệm phụ How2Sign 10/25%).
Chúng **không bị xoá khỏi ý tưởng** — ghi lại đây làm hướng mở rộng sau này, kèm đủ chi tiết thiết
kế để cài lại khi cần.

Pipeline hiện tại (sau khi gỡ) gồm: **6 pose encoder × (XE + SCST)** + **8 thuật toán/biến thể RL
fine-tune decoder** (SCST, PPO, MRT, RAML, DPO + ablation REINFORCE/A2C/Curriculum) + **4 reward
combo** + **latency** + **baseline sàn**. Xem `run_all.py`.

---

## A. Nhánh GLOSS — Two-stage Pose → Gloss → Text (P7) — ĐÃ GỠ

### Ý tưởng
Thay vì dịch thẳng pose → text (single-stage), tách làm 2 giai đoạn qua **gloss** (chú giải ký hiệu,
cột `orth` trong PHOENIX-2014-T*.corpus.csv):

1. **Stage 1 — Pose → Gloss (CTC).** Encoder pose (dùng chung factory `build_pose_encoder`) + 1
   CTC head (`nn.Linear(d_model, gloss_vocab_size)`, index 0 = `<blank>`). Train bằng CTC loss
   (Graves et al., 2006) — không cần alignment tường minh frame↔gloss, CTC tự học. Đánh giá bằng
   **WER** trên chuỗi gloss.
2. **Stage 2 — Gloss → Text (NMT).** Một Transformer seq2seq nhỏ thuần text (gloss vocab nhỏ, câu
   ngắn hơn pose sequence nhiều), độc lập hoàn toàn với stage 1.

Đánh giá end-to-end: nối 2 stage, đo BLEU-4 text cuối (`test_bleu4_e2e`).

### Vì sao gỡ
- Cần cột `orth` (gloss) — thêm một trục dữ liệu/annotation nữa, trong khi báo cáo hiện tại chỉ tập
  trung so sánh **encoder × thuật toán RL** trên nhánh dịch thẳng.
- Cột `orth` chưa được kiểm chứng trên dữ liệu PHOENIX thật ở pipeline này (chỉ self-verify khi
  chạy) → rủi ro cho một báo cáo dựa số liệu.
- Hai stage nhân đôi số model cần train (CTC 60 epoch + NMT 60 epoch), tốn quota mà không phục vụ
  câu hỏi nghiên cứu chính (RL có khắc phục exposure bias của CE không).

### Muốn cài lại
Khôi phục các file đã xoá (từ lịch sử bản trước): `models/gloss_ctc_head.py`,
`models/gloss2text_nmt.py`, `data/gloss_vocab.py`, `training/train_ctc_gloss.py`,
`training/train_gloss2text.py`, `main_twostage.py`, và `scripts/extract_gloss_segments.py`. Bật lại
cờ `use_gloss` + `gloss_vocab_size` trong `configs/config.py`, và nhánh đọc `orth` trong
`data/dataset.py::PhoenixSLTDataset` (tham số `gloss_vocab`, key `gloss_ids/gloss_raw` trong
`collate_fn`). Thêm lại nhóm `twostage` vào `run_all.py::GROUP_ORDER`.

---

## B. Nhánh RL NGOÀI DECODER — RL × CV — ĐÃ GỠ

Cho tới nay RL chỉ fine-tune **decoder text** (SCST/PPO/…). Ba trong các quyết định của pipeline
lại nằm ở **tầng thị giác** và đều là **quyết định rời rạc** (không có gradient → đúng dấu hiệu bài
toán RL). Ý tưởng: mở rộng RL ra các quyết định đó. Mỗi policy là một mạng nhỏ train RIÊNG bằng
REINFORCE (baseline = trung bình reward trong batch, kiểu SCST rẻ); model SLT chính **đóng băng
hoàn toàn**, reward = BLEU câu do model chính sinh ra trên input đã qua policy.

### B.1 — Frame selection (F.6, topk) & Adaptive temporal sampling (F.9)
Policy GRU 2 chiều cho điểm từng frame → chọn tập con frame (Bernoulli/top-K theo `keep_ratio`), có
phạt theo tỉ lệ frame giữ lại để khuyến khích nén. `adaptive` (F.9) resample ngân sách K theo mật
độ xác suất (dày ở đoạn "thông tin cao").

### B.2 — Landmark selection (F.8)
Policy quyết định MỘT LẦN/câu cho mỗi nhóm landmark (body / tay trái / tay phải) — có nên tắt nhóm
bị occlude nặng không. Bản đơn giản hoá tractable của ý tưởng "chọn theo từng frame".

### B.3 — Decode policy (F.5)
Policy nhỏ đọc `memory` đã encode (mean-pool theo thời gian) → chọn **temperature** sample_decode
phù hợp THEO TỪNG INPUT (từ tập rời rạc, vd `[0.7, 1.0, 1.3]`) thay vì 1 giá trị cố định toàn cục.
Action = chọn 1 temperature; reward = BLEU câu kết quả.

### Giới hạn ĐÃ ĐÍNH CHÍNH (lý do chính để gỡ, không phải xoá)
> **Frame/landmark selection = SOFT-MASK (zero-hoá), KHÔNG xoá khỏi chuỗi.** Độ dài chuỗi giữ
> nguyên, encoder vẫn xử lý đủ T frame. Vì vậy đo được "tín hiệu chọn frame/landmark có giúp/hại
> BLEU không" NHƯNG **KHÔNG giảm compute thực tế** và KHÔNG chứng minh "nén hiệu quả".

Muốn giảm compute thật cần **re-index chuỗi** theo frame đã chọn TRƯỚC khi vào encoder (đường dữ
liệu khác hẳn) — đây chính là hướng mở rộng nghiêm túc của nhánh này.

### Counter-baseline cần có (khi cài lại)
Policy chỉ được coi là "học được" nếu **thắng lựa-chọn-không-học ở cùng `keep_ratio`, cùng cơ chế
soft-mask**: `base_frames_random`, `base_frames_uniform` (đối thủ chính — grid cách đều),
`base_drop_body/lhand/rhand` (cho landmark); và cho decode policy: `base_fixed_temp_*` (policy
per-input phải thắng fixed-temp tốt nhất). Các baseline này đã nằm trong bản trước của
`scripts/eval_baselines.py` (`--kind selection|temp`).

### Muốn cài lại
Khôi phục `training/train_selection_policy.py`, `training/train_decode_policy.py`; thêm lại config
`selection_policy_*` / `decode_policy_*` trong `configs/config.py`; các hàm
`run_baseline_selection` / `run_baseline_temp` + helper `_apply_frame_mask` / `_apply_landmark_mask`
trong `scripts/eval_baselines.py`; và nhóm `selection` / `decode` trong `run_all.py`.

---

## C. Hướng mở rộng khác (ghi nhận, chưa cài)
- **Frame selection giảm compute THẬT**: re-index chuỗi trước encoder (xem B).
- **Reward ngữ nghĩa (BERTScore, Reward 5)**: có sẵn cờ `reward_bert_weight` trong config nhưng tắt
  mặc định (chi phí GPU cao trong reward loop) — bật ở subset nhỏ khi cần.
- **BLEU-1 / ROUGE-L**: pipeline hiện chỉ tính BLEU-4; thêm vào `training/train_xe.py::evaluate`
  nếu paper cần các cột đó (hiện để `--` thay vì bịa số).
