# SLT + RL — Dịch ngôn ngữ ký hiệu (pose-based) tối ưu bằng Reinforcement Learning

Pipeline **video → pose 183-d → encoder → decoder → text** (PHOENIX-2014T), trong đó RL
(SCST/PPO/...) fine-tune trực tiếp theo BLEU để khắc phục exposure bias của cross-entropy,
và **mở rộng RL ra ngoài decoder** — chọn frame/landmark/chiến lược decode — làm cầu nối 2 môn
(RL + Xử lý ảnh & video). Mọi khối trong sơ đồ dưới **đều đã có code + smoke-test**; trích pose
chạy trên Kaggle CPU-only qua `KAGGLE_NOTEBOOK_EXTRACT.ipynb`, train nặng chạy trên Kaggle T4×2
qua `KAGGLE_NOTEBOOK.ipynb` (mỗi subset 25/50/100% chỉ 1 lệnh `run_all.py`, xem mục "Chạy").

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
│  gloss_vocab (BLANK=0/UNK=1, cho nhánh P7)                                     │
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
│                  <══>  compute_reward: BLEU + rep_penalty + len_penalty (+sem) │
└──────────────┬──────────────────────────────────────┬────────────────────────┘
               │ cùng cơ chế RL                        │
               v                                       │
┌───────────────────────────────────────────────────┐ │
│ L5 · RL VƯỢT NGOÀI DECODER — mục F     [RL × CV]   │ │
│  Frame select F.6 ⚠soft-mask · Adaptive F.9        │ │
│  Landmark F.8 · Decode-policy F.5 · CTC-segment F.14│ │
│  ⚠ zero-hoá frame, CHƯA giảm compute thật           │ │
└───────────────────────────────────┬────────────────┘ │
                                     │                   │
┌────────────────────────────────────┼──────────────┐   │
│ P7 · NHÁNH TWO-STAGE (song song)   v      [gloss]  │   │
│  CTC pose→gloss (WER) ───> NMT gloss→text          │   │
│  ⚠ cột orth: tự verify khi chạy, chưa test data thật│  │
└────────────────────────────────────┬──────────────┘   │
                                     v                   v
┌──────────────────────────────────────────────────────────────────────────────┐
│ L6 · ĐÁNH GIÁ & BẢNG SO SÁNH  (scripts/)                                       │
│  eval_baselines (BASE cơ bản nhất) · measure_latency (6 encoder)               │
│  aggregate_results: quét mọi *_results/*_history/latency_*.json                │
│                     → comparison_table.csv / .md  (tự cập nhật)                │
└──────────────────────────────────────────────────────────────────────────────┘

 Điều phối: configs/config.py · main.py (--encoder --algo --phase --subset, 1 experiment đơn lẻ)
            main_twostage.py (P7) · run_all.py (chạy TOÀN BỘ ma trận cho 1 subset, resumable —
            dùng cái này trên Kaggle) · KAGGLE_NOTEBOOK.ipynb (train, T4×2) ·
            KAGGLE_NOTEBOOK_EXTRACT.ipynb (trích pose, CPU-only, chạy TRƯỚC)
 [CV] thị giác · [RL] policy · ⚠ giới hạn đã đính chính · 6 encoder / 8 RL / 5 RL-CV
```

## Baseline bắt buộc — đọc bảng so sánh TƯƠNG ĐỐI so với các sàn này

| Base | Đối chứng cho | Ghi chú |
|---|---|---|
| `xe` (CE-only) | mọi thuật toán RL | có sẵn trong mọi run (`--phase all`) |
| `base_empty` / `base_most_frequent` | mọi model | sàn không-cần-model; PHOENIX lặp nhiều nên sàn này có thể > 0 đáng kể |
| `base_frames_random` / `base_frames_uniform` | frame-selection policy (F.6/F.9) | cùng `keep_ratio`, cùng soft-mask; policy phải thắng **uniform** mới là "học được" |
| `base_fixed_temp_*` | decode policy (F.5) | policy per-input phải thắng fixed-temp tốt nhất |

## Chạy

Quy trình chuẩn giờ đây gồm **2 notebook Kaggle riêng biệt**, không còn chạy cell-by-cell:

```bash
# 1) BƯỚC 0 — trích pose (MỘT LẦN DUY NHẤT, mọi subset dùng chung).
#    Trên Kaggle: KAGGLE_NOTEBOOK_EXTRACT.ipynb, Accelerator = CPU-only (không tốn quota GPU).
#    Local (nếu có máy đủ mạnh): tương tự lệnh dưới rồi upload thư mục out_dir thành Kaggle
#    Dataset `phoenix-poses`.
python data/extract_poses.py \
    --input_dir <PHOENIX-2014-T>/features/fullFrame-210x260px --out_dir ./poses_out

# 2) TOÀN BỘ ma trận thí nghiệm cho 1 subset, MỘT lệnh (resumable — hết giờ session cứ chạy lại
#    ĐÚNG LỆNH NÀY, các bước đã xong tự bỏ qua nhờ marker .done_*). Cuối mỗi lần chạy tự sinh
#    lại bảng + biểu đồ so sánh trong <work_dir>/report/ (xem mục dưới), không cần lệnh nào khác:
python run_all.py --subset 0.25   # rồi khi nào sẵn sàng:
python run_all.py --subset 0.5
python run_all.py --subset 1.0
#    Muốn giới hạn phạm vi (debug nhanh): --groups core,encoders (mặc định --groups all)

# 3) (Tuỳ chọn) Chạy tay 1 cấu hình đơn lẻ để debug -- không cần cho quy trình chuẩn ở trên,
#    run_all.py đã tự gọi các hàm này cho TẤT CẢ encoder/algo/ablation:
python main.py --subset 0.25 --encoder transformer --algo scst --phase all
python main_twostage.py --subset 0.25 --encoder transformer
python scripts/eval_baselines.py --kind trivial --subset 0.25
python scripts/aggregate_results.py --work_dir /kaggle/working --out /kaggle/working/comparison_table
python scripts/make_report.py --work_dir /kaggle/working   # cũng tự chạy cuối mỗi run_all.py
```

## Bảng + biểu đồ cho báo cáo/paper

`run_all.py` tự gọi `scripts/make_report.py` ở cuối mỗi lần chạy (chạy tay được, đọc lại kết quả
đã có, không train gì thêm). Output trong `<work_dir>/report/`:

| Đường dẫn | Nội dung |
|---|---|
| `tables/table_*.csv` `.md` | 6 bảng đã lọc sẵn: main (XE vs mọi algo RL) · encoders · reward ablation · ablation khác (REINFORCE/A2C/Curriculum) · baseline sàn · latency |
| `tables/tab_main.tex` `tab_reward.tex` `tab_encresults.tex` | Dán thẳng vào `paper/sn-article.tex` (khớp `\label{tab:main}`/`tab:reward`/`tab:encresults`). Cột BLEU-1/ROUGE-L để `--` vì pipeline chỉ tính BLEU-4 — không bịa số |
| `figures/*.png` `*.pdf` | 5 biểu đồ: BLEU theo epoch (XE vs mọi RL algo) · ΔBLEU theo subset (Exp.11) · trade-off reward ablation (Exp.9) · so sánh 6 encoder (Exp.4) · so sánh thuật toán (Exp.1/7). PNG xem nhanh, PDF vector để `\includegraphics` |

Chạy tay: `python scripts/make_report.py --work_dir /kaggle/working [--subset 25]` (mặc định dùng
subset lớn nhất đã có dữ liệu).

Trên Kaggle: chạy `KAGGLE_NOTEBOOK_EXTRACT.ipynb` trước (dataset input: `phoenix-2014t`) để có
Kaggle Dataset `phoenix-poses`. Sau đó `KAGGLE_NOTEBOOK.ipynb` (add 3 dataset: `phoenix-2014t`,
`phoenix-poses`, `slt-rl-code`; Accelerator GPU T4×2) — mỗi subset chỉ 1 cell (`run_all.py
--subset ...`), cell cuối ra bảng so sánh + nén tải về.

## Compute (T4×2, ~30 GPU-h/tuần; epoch KHÔNG giảm theo subset)

Ước tính cho riêng CORE (Transformer+SCST, XE→RL) — chạy **toàn bộ ma trận** (6 encoder, 5 algo,
ablation, reward ablation, P7, selection/decode policy) sẽ tốn nhiều hơn đáng kể, xem cảnh báo
trong `run_all.py` và `docs/1_Thuyet_Trinh_Tong_Hop.md §K`:

| Subset | Total XE | Total RL | Total core | Toàn bộ ma trận (ước tính) |
|--------|----------|----------|-------------|------------------------------|
| 25%    | ~2.5h    | ~1h      | ~3.5h       | ~25-30h (vừa quota 1 tuần)   |
| 50%    | ~5h      | ~2h      | ~7h         | lớn hơn đáng kể — trải ra nhiều tuần |
| 100%   | ~10h     | ~4h      | ~14h (>1 session) | rất lớn — chạy dần theo `--groups`, không cần chạy hết trong 1 lần |

## Giới hạn đã đính chính (đọc trước khi diễn giải kết quả)

- **GCN/GraphTransformer KHÔNG nhẹ hơn Transformer** (param đo thật ở sơ đồ trên) — chỉ ST-GCN nhẹ hơn.
- **Frame selection = soft-mask** (zero-hoá, giữ nguyên độ dài chuỗi) — đo được "frame nào quan trọng", **chưa** chứng minh giảm compute.
- **P7**: cột `orth` được code tự verify khi chạy, **chưa** chạy trên dữ liệu PHOENIX thật.
- Mọi claim BLEU/WER chỉ có sau khi chạy Kaggle — smoke-test chỉ đảm bảo đúng shape/gradient, không đảm bảo "học tốt".

## Đọc sâu — chỉ còn 2 tài liệu

Toàn bộ 14 file docs cũ đã được **gộp vào một tài liệu duy nhất**, tổ chức theo flow thuyết trình.

| Tài liệu | Nội dung |
|---|---|
| [`docs/0_Architecture.md`](docs/0_Architecture.md) | Sơ đồ kiến trúc hệ thống (ASCII, luồng dữ liệu L1→L8) |
| [`docs/1_Thuyet_Trinh_Tong_Hop.md`](docs/1_Thuyet_Trinh_Tong_Hop.md) | **Tài liệu chính** — xem mục lục bên dưới |

**Phần I — theo flow thuyết trình** (12 slide, khớp 1-1 với [`slides/index.html`](../slides/index.html)):
Cover · Bài toán · Pipeline base · Vấn đề · Giải pháp · Vì sao RL · Thuật toán · Reward · RL ngoài decoder · Thực nghiệm · Đọc kết quả · Kết luận.
Mỗi slide có 🎙️ kịch bản nói · 🔬 phân tích sâu · ❓ câu hỏi phản biện.

**Phần II — tra cứu** (code comment trỏ thẳng vào các mục này):

| Mục | Nội dung | Mục | Nội dung |
|---|---|---|---|
| **§A** | Pipeline P1–P8 + đã loại | **§G** | Nhật ký thiết kế 41 mục |
| **§B** | Khảo sát kiến trúc encoder | **§H** | Đề xuất luận văn (RQ, H1-H5, hạn chế) |
| **§C** | Khảo sát 6 dataset | **§I** | Roadmap & rủi ro |
| **§D** | Pipeline CV (pose, temporal, augment) | **§J** | Code review & 9 bug đã sửa |
| **§E** | 13 experiment + baseline sàn | **§K** | **Hướng dẫn chạy để lấy số liệu** |
| **§F** | Metric chi tiết | **§L** | References (đã xác minh nguồn) |
