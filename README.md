# SLT + RL — Dịch ngôn ngữ ký hiệu (pose-based) tối ưu bằng Reinforcement Learning

Pipeline **video → pose 183-d → encoder → decoder → text** (PHOENIX-2014T), trong đó RL
(SCST/PPO/MRT/RAML/DPO) fine-tune trực tiếp theo BLEU để khắc phục exposure bias của cross-entropy.
Câu hỏi nghiên cứu chính: **RL có cải thiện BLEU so với CE-only không, và thuật toán RL / kiến trúc
pose-encoder nào tốt nhất** — đo trên PHOENIX-2014T qua **sweep 5 / 10 / 25% dataset** (train %split,
dev/test luôn full) + thí nghiệm phụ How2Sign. Phát hiện chính: gain RL **xuất hiện theo quy mô dữ liệu**
(không đáng tin ở 5%, rõ ở 10/25%), và xếp hạng encoder đảo theo data — xem mục Kết quả.

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

# 2) TRAIN. Thiết kế báo cáo = 3 mức subset PHOENIX 5 / 10 / 25% (ĐÃ CHẠY ĐỦ) + 1 THÍ NGHIỆM PHỤ
#    ("train thử") trên How2Sign 10/25/50% (I3D vs pose). MỘT lệnh cho toàn bộ
#    ma trận mỗi mức, resumable (hết giờ session cứ chạy lại ĐÚNG lệnh, bước đã xong tự bỏ qua).
#    Cuối mỗi lần chạy tự sinh bảng + biểu đồ trong <work_dir>/report/:
python run_all.py --subset 0.05              # toàn ma trận ở 5%  (~6-9h) — đã chạy
python run_all.py --subset 0.10              # 10% (~12-18h) — đã chạy
python run_all.py --subset 0.25              # 25% (~20-25h) — đã chạy
python run_all.py --subset 0.05 --groups core,encoders   # giới hạn phạm vi để debug nhanh
#    How2Sign (thí nghiệm phụ): train vào work_dir RIÊNG để tách dataset — pose trích bằng
#    --mode video (xem data/extract_poses.py), rồi trỏ make_overview.py vào từng root (xem dưới).

# 3) (Tuỳ chọn) chạy hẹp/tay 1 cấu hình:
python train_select.py --mode single --encoder transformer --algo scst --subset 0.05
python main.py --subset 0.05 --encoder transformer --algo scst --phase all
python scripts/eval_baselines.py --subset 0.05
python scripts/aggregate_results.py --work_dir /kaggle/working --out /kaggle/working/comparison_table
python scripts/make_report.py --work_dir /kaggle/working
python scripts/make_overview.py --root phoenix=/kaggle/working --out /kaggle/working/overview  # bảng TỔNG
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
| `figures/*.png` `*.pdf` | 4 biểu đồ mỗi work_dir (cột có SỐ): BLEU theo epoch (RL rẽ nhánh từ đúng điểm warm-start XE) · trade-off reward ablation · so sánh 6 encoder · so sánh thuật toán. *(Biểu đồ ΔBLEU-theo-subset là biểu đồ CHÉO nhiều work_dir → dựng bằng notebook/`make_overview.py`, xem `analysis_out/figures/datasize_phoenix.png`.)* |

## Bảng TỔNG (pivot) gộp mọi subset + dataset — `scripts/make_overview.py`

Toàn bộ sweep (5/10/25%) + How2Sign đã gộp sẵn vào `analysis_out/overview.md` bằng `make_overview.py`:
**một bảng pivot** (hàng = method/encoder, cột = subset%, tách theo dataset) — thay vì mở từng
`comparison_table`. Chỉ đọc, không train. Ô `–` = subset/dataset đó chưa có run (không bịa số).

```bash
# 1 dataset (mặc định phoenix):
python scripts/make_overview.py --root phoenix=/kaggle/working --out results/overview
# 2 dataset song song — mỗi dataset train vào work_dir RIÊNG rồi trỏ 2 root:
python scripts/make_overview.py \
    --root phoenix=/kaggle/input/phoenix-runs \
    --root how2sign=/kaggle/input/how2sign-runs \
    --out results/overview
python scripts/make_overview.py --root phoenix=/kaggle/working --manifest   # xem file có/thiếu + dung lượng
```
→ `overview.md` (dán thẳng lên slide/report) + `overview.csv`.

**Sau khi train xong trên Kaggle NÊN TẢI FILE NÀO VỀ?** (output rất nặng — đừng tải hết). Với mỗi
thư mục run chỉ cần các **file JSON nhỏ**, bỏ hẳn checkpoint:

| File | Tải? | Vì sao |
|---|---|---|
| `test_results.json` | ✅ bắt buộc | BLEU-4 test cuối của từng method |
| `*_history.json` (xe/rl/ppo/mrt/raml/dpo) | ✅ | best dev BLEU, rep_rate, len_ratio, #epoch |
| `latency_*.json` | ✅ | #params, latency, throughput, peak memory |
| `.done_*` | ⚪ bỏ được | chỉ là marker resume, không chứa số |
| `best_*.pt` / `last_*.pt` | ❌ KHÔNG | checkpoint ~43MB/cái, chỉ cần nếu chạy lại inference |

Thực tế: zip toàn bộ `*.json` là đủ dựng lại mọi bảng — ma trận 5% chỉ ~**0.14 MB** JSON so với
~**1.5 GB** checkpoint `.pt`. Chạy `--manifest` để in chính xác file nào đang có/thiếu ở mỗi root.

## Compute (T4×2, ~30 GPU-h/tuần; epoch KHÔNG giảm theo subset)

`xe_epochs=80` / `rl_epochs=20` cố định cho mọi subset → thời gian ~tỉ lệ lượng dữ liệu train.
**dev/test luôn full** (không co theo subset) nên 5% KHÔNG nhanh gọn gấp 20 lần 100%.

| Subset | Toàn bộ ma trận (6 enc + 8 RL + reward + latency) |
|--------|----------------------------------------------------|
| **5%** (đã chạy) | **~6–9h** (gọn trong quota, có thể 1 session) |
| **10%** (đã chạy) | ~12–18h |
| **25%** (đã chạy) | ~20–25h (vừa quota 1 tuần) |
| How2Sign 10/25/50% (phụ, đã chạy) | I3D vs pose, work_dir riêng |
| 100%   | rất lớn — chạy dần theo `--groups`/`train_select.py`, nhiều session |

## Kết quả thực nghiệm — toàn bộ sweep 5/10/25% (test BLEU-4, đã chạy đủ)

Nguồn: `analysis_out/overview.md`. Test luôn là full 642 câu PHOENIX bất kể subset → so sánh được trực tiếp.

| Test BLEU-4 (Transformer core) | 5% | 10% | 25% |
|---|---|---|---|
| Sàn (empty / most-freq) | 0.0 / 0.19 | 0.0 / 0.19 | 0.0 / 0.19 |
| XE (CE-only) | 4.16 | 5.04 | 5.86 |
| SCST | 4.31 | 5.23 | 5.94 |
| PPO | 4.19 | 5.28 | 6.31 |
| **MRT (best RL)** | 4.08 | **5.80** | **6.49** |
| **Best Δ (best RL − XE)** | **+0.15** | **+0.76** | **+0.63** |

| 6 encoder (SCST BLEU-4) | 5% | 10% | 25% | Params |
|---|---|---|---|---|
| Transformer | 4.31 | 5.23 | 5.94 | 8.26M |
| GCN | 4.10 | 5.55 | 7.27 | 9.38M |
| ST-GCN | 3.62 | 4.99 | 6.74 | **6.33M** |
| Graph-Transformer | 2.66 | 5.53 | **7.64** | 9.48M |
| TCN | **5.40** | **6.00** | 7.47 | 7.47M |
| Perceiver IO | 4.08 | 4.77 | 6.54 | 9.59M |

Reward ablation (SCST): `bleu_only` **luôn tốt nhất** (3.98/5.19/5.96); mọi penalty đều giảm BLEU, `rep+len` sụp ở 10/25% (3.85/4.44).

**How2Sign (thí nghiệm phụ, I3D vs pose, test BLEU-4):** I3D (video) XE→SCST = 2.11→2.22 / 3.28→3.52 / 4.25→4.35 · pose (99-d) = 1.67→1.59 / 1.76→1.93 / 1.62→1.25. **I3D ≫ pose** ở mọi mức, chỉ I3D scale theo data.

**Kết luận (đọc qua cả sweep):**
1. **Gain RL xuất hiện theo quy mô dữ liệu** — ở 5% Δ +0.15 < nhiễu ~0.5 (không tin cậy), nhưng ở 10/25% best-Δ +0.76/+0.63 vượt hẳn nhiễu, do **MRT/PPO** gánh (không phải SCST thuần) → **đảo ngược H1** (khớp Kiegeland 2021).
2. **Xếp hạng encoder đảo theo data** — 5% TCN nhất & graph tệ nhất; 25% Graph-Transformer/GCN vượt Transformer (giờ tệ nhất), ST-GCN 6.74 > Transformer 5.94 với ít hơn 23% param → **H4 sai ở 5%, đúng ở 25%**.
3. **Penalty reward chỉ tổn hại BLEU** — rep ≈ 0, câu đã quá ngắn (len_ratio 0.11→0.03) → không có reward-hacking để phạt (**H2 không được ủng hộ**).
4. **How2Sign tái lập cùng kết luận** trên ASL + modality khác.

## Giới hạn đã đính chính (đọc trước khi diễn giải kết quả)

- **GCN/GraphTransformer KHÔNG nhẹ hơn Transformer** (param đo thật) — chỉ ST-GCN nhẹ hơn (và ở 25% cũng nhanh nhất, 28ms/36 sent-s; latency biến động theo độ dài chuỗi của subset nên đọc param làm trục ổn định).
- Phần lớn là **1 seed** — RL variance cao (~0.5), nên ưu tiên **xu hướng lặp lại qua nhiều subset + nhiều method** hơn là 1 ô đơn lẻ; muốn claim chắc hơn cần ≥2 seed (§H.6).
- Pipeline chỉ tính **BLEU-4** (sacrebleu) + rep_rate + len_ratio; chưa có BLEU-1/ROUGE-L.

## Đọc sâu

| Tài liệu | Nội dung |
|---|---|
| [`docs/0_Architecture.md`](docs/0_Architecture.md) | Sơ đồ kiến trúc hệ thống (ASCII, luồng dữ liệu) |
| [`docs/1_Thuyet_Trinh_Tong_Hop.md`](docs/1_Thuyet_Trinh_Tong_Hop.md) | **Tài liệu chính** — flow thuyết trình + mục tra cứu §A–§L |
| [`docs/2_Huong_Phat_Trien.md`](docs/2_Huong_Phat_Trien.md) | Hướng phát triển: nhánh gloss/P7 + RL-ngoài-decoder đã gỡ |
