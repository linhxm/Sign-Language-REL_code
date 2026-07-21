# SLT + RL — Dịch ngôn ngữ ký hiệu (pose-based) tối ưu bằng Reinforcement Learning

Pipeline **video → pose 183-d → encoder → decoder → text** (PHOENIX-2014T), trong đó RL
(SCST/PPO/MRT/RAML/DPO) fine-tune trực tiếp theo BLEU để khắc phục exposure bias của cross-entropy.
Câu hỏi nghiên cứu chính: **RL có cải thiện BLEU so với CE-only không, và thuật toán RL / kiến trúc
pose-encoder nào tốt nhất** — đo trên PHOENIX-2014T ở mức **5% dataset** (train 5% split train,
dev/test luôn full).

> **Phạm vi đã thu gọn (07/2026):** hai nhánh mở rộng — **gloss / two-stage P7** (pose→gloss→text)
> và **RL ngoài decoder** (frame/landmark selection, decode-policy) — đã được **gỡ khỏi pipeline**
> để tập trung báo cáo. Chúng được lưu làm hướng phát triển trong
> [`docs/2_Huong_Phat_Trien.md`](docs/2_Huong_Phat_Trien.md).

## Sơ đồ kiến trúc

```
                        ┌───────────────────────────┐
                        │   INPUT: Video / Pose      │  PHOENIX-2014T (~7K câu)
                        └─────────────┬─────────────┘
                                      │ pose 183-d (MediaPipe Holistic)
                                      v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L1 · DATA  (data/)                                                        [CV] │
│  extract_poses (99+42+42) · dataset (norm/augment/curriculum) · tokenizer BPE  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                     │ [B,T,183] + pose_mask
                                     v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L2 · POSE ENCODER — build_pose_encoder() factory, 1 interface chung       [CV] │
│  P1 Transformer 8.26M │ P2 GCN 9.38M ⚠ │ P3 ST-GCN 6.33M ★ │                   │
│  P4 GraphTransf 9.48M │ P5 TCN 7.47M   │ P6 Perceiver 9.59M                    │
│  ⚠ GCN KHÔNG nhẹ hơn Transformer (đã đo)   ★ chỉ ST-GCN thực sự nhẹ nhất        │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                     │ memory [B,T,d_model]
                                     v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L3 · MODEL + PHASE 1                                                 [dùng chung]
│  SLTTransformer (greedy/sample/beam, tie weights) + ValueHead (critic PPO/A2C) │
│  train_xe: Cross-Entropy · teacher forcing  →  best_xe.pt  (+ evaluate BLEU-4) │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                     │ best_xe.pt nạp vào mọi thuật toán RL
                                     v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L4 · PHASE 2 — RL FINE-TUNE DECODER  (training/)                          [RL] │
│  SCST │ REINFORCE(no-baseline) │ Curriculum │ PPO(GAE+clip) │ A2C(no-clip)     │
│  MRT  │ RAML                   │ DPO(preference tự sinh)                       │
│                  <══>  compute_reward: BLEU + rep_penalty + len_penalty        │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                     v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L5 · ĐÁNH GIÁ & BẢNG SO SÁNH  (scripts/)                                       │
│  eval_baselines (BASE cơ bản nhất) · measure_latency (6 encoder)               │
│  aggregate_results: quét mọi *_results/*_history/latency_*.json                │
│                     → comparison_table.csv / .md  (tự cập nhật)                │
└──────────────────────────────────────────────────────────────────────────────┘

 Điều phối: configs/config.py · main.py (--encoder --algo --phase --subset, 1 experiment đơn lẻ)
            run_all.py (chạy TOÀN BỘ ma trận cho 1 subset, resumable) ·
            train_select.py (chọn phạm vi hẹp: single / encoder_allrl / rl_allenc)
 Notebook: Sign-Language-REL_pose-extract.ipynb (trích pose, CPU-only, chạy TRƯỚC) ·
           Sign-Language-REL_smoke-5pct.ipynb (extract 5% + train TOÀN BỘ ma trận ở 5%) ·
           KAGGLE_NOTEBOOK.ipynb (train đa dạng: chọn nhóm/encoder/algo + % data qua 1 cell config)
 [CV] thị giác · [RL] policy · ⚠ giới hạn đã đính chính · 6 encoder / 8 RL
```

## Baseline bắt buộc — đọc bảng so sánh TƯƠNG ĐỐI so với các sàn này

| Base | Đối chứng cho | Ghi chú |
|---|---|---|
| `xe` (CE-only) | mọi thuật toán RL | có sẵn trong mọi run (`--phase all`) |
| `base_empty` / `base_most_frequent` | mọi model | sàn không-cần-model; PHOENIX lặp nhiều nên sàn này có thể > 0 đáng kể |

## Chạy

Quy trình chuẩn gồm **2 bước** (2 notebook Kaggle riêng biệt):

```bash
# 1) BƯỚC 0 — trích pose (MỘT LẦN, mọi subset dùng chung).
#    Trên Kaggle: Sign-Language-REL_pose-extract.ipynb, Accelerator = CPU-only.
#    (Muốn chỉ trích đúng 5% để chạy nhanh: dùng Sign-Language-REL_smoke-5pct.ipynb MODE="extract".)
python data/extract_poses.py \
    --input_dir <PHOENIX-2014-T>/features/fullFrame-210x260px --out_dir ./poses_out

# 2) TRAIN. Mức báo cáo CHÍNH = 5%. MỘT lệnh cho toàn bộ ma trận, resumable (hết giờ session cứ
#    chạy lại ĐÚNG lệnh, bước đã xong tự bỏ qua). Cuối mỗi lần chạy tự sinh bảng + biểu đồ trong
#    <work_dir>/report/:
python run_all.py --subset 0.05              # toàn ma trận ở 5% (~6-9h)
python run_all.py --subset 0.05 --groups core,encoders   # giới hạn phạm vi để debug nhanh
python run_all.py --subset 0.25              # chạy thêm khi có quota

# 3) (Tuỳ chọn) chạy hẹp/tay 1 cấu hình:
python train_select.py --mode single --encoder transformer --algo scst --subset 0.05
python main.py --subset 0.05 --encoder transformer --algo scst --phase all
python scripts/eval_baselines.py --subset 0.05
python scripts/aggregate_results.py --work_dir /kaggle/working --out /kaggle/working/comparison_table
python scripts/make_report.py --work_dir /kaggle/working
```

**Nhóm thí nghiệm hợp lệ cho `--groups`:** `core` (Transformer XE+SCST) · `encoders` (5 encoder còn
lại × XE+SCST) · `algos` (PPO/MRT/RAML/DPO trên Transformer) · `ablations` (REINFORCE/A2C/Curriculum)
· `reward` (4 reward combo) · `latency` (đo 6 encoder). `all` = tất cả.

## Bảng + biểu đồ cho báo cáo/paper

`run_all.py` tự gọi `scripts/make_report.py` ở cuối mỗi lần chạy (chạy tay được, chỉ đọc lại kết
quả đã có, không train gì thêm). Output trong `<work_dir>/report/`:

| Đường dẫn | Nội dung |
|---|---|
| `tables/table_*.csv` `.md` | 6 bảng đã lọc sẵn: main (XE vs mọi algo RL) · encoders · reward ablation · ablation khác (REINFORCE/A2C/Curriculum) · baseline sàn · latency |
| `tables/tab_main.tex` `tab_reward.tex` `tab_encresults.tex` | Dán thẳng vào `paper/sn-article.tex`. Cột BLEU-1/ROUGE-L để `--` vì pipeline chỉ tính BLEU-4 — không bịa số |
| `figures/*.png` `*.pdf` | 4 biểu đồ (cột có SỐ trên biểu đồ): BLEU theo epoch (RL rẽ nhánh từ đúng điểm warm-start XE) · trade-off reward ablation · so sánh 6 encoder · so sánh thuật toán. *(Đã bỏ "ΔBLEU theo subset" vì mỗi subset train đủ epoch độc lập → so sánh giữa subset vô nghĩa.)* |

## Compute (T4×2, ~30 GPU-h/tuần; epoch KHÔNG giảm theo subset)

`xe_epochs=80` / `rl_epochs=20` cố định cho mọi subset → thời gian ~tỉ lệ lượng dữ liệu train.
**dev/test luôn full** (không co theo subset) nên 5% KHÔNG nhanh gọn gấp 20 lần 100%.

| Subset | Toàn bộ ma trận (6 enc + 8 RL + reward + latency) |
|--------|----------------------------------------------------|
| **5%** (báo cáo chính) | **~6–9h** (gọn trong quota, có thể 1 session) |
| 25%    | ~20–25h (vừa quota 1 tuần) |
| 100%   | rất lớn — chạy dần theo `--groups`/`train_select.py`, nhiều session |

## Kết quả thực nghiệm 5% (test BLEU-4, đã chạy)

| Hạng mục | Số |
|---|---|
| Baseline sàn | `base_empty` **0.0** · `base_most_frequent` **0.19** |
| 6 encoder (tốt→tệ) | **TCN 5.40** (7.47M, 108ms) · Transformer 4.16 · GCN 4.10 · Perceiver 4.08 (nhanh nhất 85ms) · ST-GCN 3.62 · GraphTransf 2.66 |
| 8 RL trên Transformer (XE 4.16) | SCST **4.31** · A2C 4.31 · PPO 4.19 · DPO 4.16 · MRT 4.08 · RAML 3.99 · REINFORCE 3.91 · Curriculum 3.77 |
| 4 reward (SCST) | bleu_only 3.98 · len_only 3.90 · both 3.87 · default 3.75 |

**Kết luận:** BLEU tuyệt đối rất thấp (mới 5% train). **RL CHƯA vượt CE một cách tin cậy** — chênh SCST−CE (+0.15) nhỏ hơn nhiễu run-to-run (~0.5; một lần chạy lại cùng cấu hình SCST chỉ 3.75) → finding **H3-neutral**, khớp Kiegeland 2021. Encoder mạnh nhất là **TCN** (không phải ST-GCN → H4 sai). Hệ under-generate (len_ratio ≈ 0.10) nên BLEU bị brevity bóp; rep ≈ 0 (không degeneracy lặp).

## Giới hạn đã đính chính (đọc trước khi diễn giải kết quả)

- **GCN/GraphTransformer KHÔNG nhẹ hơn Transformer** (param đo thật) — chỉ ST-GCN nhẹ hơn (nhưng lại chậm, 294ms).
- Số trên là **1 seed ở 5%** — RL variance cao, cần **≥2 seed** hoặc subset lớn hơn để claim chắc (§H.6).
- Pipeline chỉ tính **BLEU-4** (sacrebleu) + rep_rate + len_ratio; chưa có BLEU-1/ROUGE-L.

## Đọc sâu

| Tài liệu | Nội dung |
|---|---|
| [`docs/0_Architecture.md`](docs/0_Architecture.md) | Sơ đồ kiến trúc hệ thống (ASCII, luồng dữ liệu) |
| [`docs/1_Thuyet_Trinh_Tong_Hop.md`](docs/1_Thuyet_Trinh_Tong_Hop.md) | **Tài liệu chính** — flow thuyết trình + mục tra cứu §A–§L |
| [`docs/2_Huong_Phat_Trien.md`](docs/2_Huong_Phat_Trien.md) | Hướng phát triển: nhánh gloss/P7 + RL-ngoài-decoder đã gỡ |
