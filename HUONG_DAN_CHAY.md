# Hướng dẫn thực thi — Extract pose → Train → Lấy số liệu cho paper/slide

> Tài liệu này là quy trình **chạy thật** ngắn gọn. Chi tiết thiết kế xem
> [`docs/1_Thuyet_Trinh_Tong_Hop.md`](docs/1_Thuyet_Trinh_Tong_Hop.md) §K. Thiết kế báo cáo = **3
> mức subset PHOENIX 5 / 10 / 25%** (train %split train, **dev/test luôn full**; 5% đã xong làm mốc,
> 10/25% chạy tiếp khi có quota) + **thí nghiệm phụ How2Sign 10/25%** ("train thử", work_dir riêng).
> Bắt buộc đúng thứ tự: **Bước 0 (extract) → Bước 1 (train)**.
>
> Nhánh **gloss/P7** và **RL-ngoài-decoder** đã gỡ khỏi pipeline — xem
> [`docs/2_Huong_Phat_Trien.md`](docs/2_Huong_Phat_Trien.md).

---

## 0. Chuẩn bị Kaggle (làm 1 lần)

| Kaggle Dataset | Nội dung | Khi nào có |
|---|---|---|
| `rwth-phoenix-2014-t` | Bản gốc PHOENIX-2014-T (thư mục ảnh PNG, **không phải video**) | Upload sẵn |
| `phoenix-2014t-annotations` | 3 file `.corpus.csv` (đã đóng gói sẵn trong `code/` cũng được) | Upload sẵn |
| `phoenix-poses` | Pose `.npz` đã trích | **Sinh ra ở Bước 0** |

Đường dẫn trong [`configs/config.py`](configs/config.py) đã khớp sẵn Kaggle — notebook tự trỏ
`pose_cache_dir` vào dataset pose đã mount, **không cần sửa tay**.

---

## Bước 0 — Trích pose (MỘT LẦN DUY NHẤT, CPU-only)

Mở [`Sign-Language-REL_pose-extract.ipynb`](Sign-Language-REL_pose-extract.ipynb) trên Kaggle.

1. **Accelerator = None (CPU)** — MediaPipe chạy CPU, **không tốn quota GPU**.
2. Add dataset input: `rwth-phoenix-2014-t` (ảnh).
3. Chạy tuần tự các cell:
   - Cell 3 (`--limit 5`) = **smoke-test đường dẫn** — BẮT BUỘC làm trước. 10–20h là quá đắt để phát hiện sai đường dẫn.
   - Cell 3b kiểm tra 1 file `.npz`: phải ra shape `[T, 183]`, `float32`, `tỉ lệ phần tử = 0` không được ~1.0.
   - Cell 4 (`--workers 0`) = chạy FULL. Resumable: hết giờ session cứ chạy lại, file `.npz` đã có tự bỏ qua.
4. Cuối script in ra **`missing_counts` (body / lhand / rhand)** — 📊 **GHI LẠI 3 con số này** (bằng chứng định lượng cho vấn đề pose-quality, có thể đưa vào paper).
5. Kỳ vọng **8257 file `.npz`** (PHOENIX-2014T full). Xong thì tạo Kaggle Dataset tên **`phoenix-poses`** từ tab Output.

⏱️ ~10–20h CPU. Tăng tốc: `--workers 8` (≈ 1.7h) hoặc chia nhiều máy bằng `--num_shards N --shard i`.

> **Chỉ cần báo cáo 5%?** Dùng thẳng [`Sign-Language-REL_smoke-5pct.ipynb`](Sign-Language-REL_smoke-5pct.ipynb)
> `MODE="extract"`: nó chỉ trích **5% train + full dev/test** (nhanh hơn nhiều), đóng gói `poses_5pct.zip`
> để bạn tạo dataset pose nhỏ.

---

## Bước 1 — Train (GPU T4×2)

**Cách A — báo cáo 5% trong 1 notebook** ([`Sign-Language-REL_smoke-5pct.ipynb`](Sign-Language-REL_smoke-5pct.ipynb)):
đổi `MODE="train"`, Accelerator GPU, Add dataset pose (từ `poses_5pct.zip`) → chạy. Cell train gọi
`run_all.py --subset 0.05 --groups all` (toàn ma trận ở 5%).

**Cách B — train đa dạng** ([`KAGGLE_NOTEBOOK.ipynb`](KAGGLE_NOTEBOOK.ipynb)):

1. **Accelerator = GPU T4×2**, Internet ON. Add dataset `phoenix-poses`.
2. **Cell 1 (config):** đặt `SUBSET` + chọn phạm vi:
   - `SCOPE="matrix"` → `GROUPS` = 1 hay nhiều nhóm (`core,encoders,algos,ablations,reward,latency`) hoặc `"all"`.
   - `SCOPE="select"` → `SELECT_MODE`/`ENCODER`/`ALGO` (train 1 encoder/algo cụ thể, nhẹ).
3. Cell 2–4 (deps, clone, trỏ pose) → **Cell 5 (chạy)** → Cell 6 (bảng) → Cell 7 (đóng gói tải về).
4. Phiên sau đổi config, lặp lại; **Cell 8 gộp** mọi phiên thành 1 bảng + hình.

Lệnh tương đương chạy tay:
```bash
python run_all.py --subset 0.05                       # toàn ma trận ở 5%
python run_all.py --subset 0.05 --groups core,encoders # giới hạn phạm vi
python train_select.py --mode single --encoder transformer --algo scst --subset 0.05
```

⏱️ Ước tính (toàn ma trận): **5% ≈ 6–9h** (gọn trong quota) · 25% ≈ 20–25h · 100% rất lớn (trải
nhiều session, dùng `--groups`/`train_select.py`). **Resumable** (marker `.done_*`): hết giờ chạy
lại đúng lệnh, bước đã xong tự bỏ qua.

---

## Bước 2 — Số liệu ra ở đâu (tự sinh, không cần lệnh thêm)

Sau mỗi lần `run_all.py`, trong `/kaggle/working/`:

| Đường dẫn | Nội dung | Dùng cho |
|---|---|---|
| `comparison_table.csv` / `.md` | 1 bảng gộp MỌI run/subset | Tra cứu tổng |
| `report/tables/table_*.csv` `.md` | 6 bảng lọc sẵn (main · encoders · reward · ablations · baseline · latency) | Đọc/kiểm |
| `report/tables/tab_main.tex` `tab_reward.tex` `tab_encresults.tex` | 3 bảng LaTeX **dán thẳng vào paper** (khớp `\label{tab:main/reward/encresults}`) | **Paper** |
| `report/figures/*.pdf` `*.png` | 4 biểu đồ (BLEU/epoch · reward trade-off · 6 encoder · thuật toán) | **Paper + Slide** |

> Lưu ý trung thực: pipeline chỉ tính **BLEU-4** (+ rep-rate, len-ratio), nên cột **BLEU-1/ROUGE-L
> trong `tab_main` để `--`**, không bịa số. Muốn có thật phải thêm code vào `evaluate()`.

### Bảng TỔNG gộp mọi subset + dataset (khi có >1 mức / có How2Sign)

`run_all.py` sinh bảng **cho từng work_dir**. Muốn gộp 5/10/25% (và PHOENIX vs How2Sign) vào **một
bảng pivot** để dán slide, chạy thêm (chỉ đọc, không train):

```bash
python scripts/make_overview.py --root phoenix=/kaggle/working --out results/overview
# 2 dataset: train mỗi cái vào work_dir riêng rồi trỏ 2 root
python scripts/make_overview.py --root phoenix=<dir1> --root how2sign=<dir2> --out results/overview
python scripts/make_overview.py --root phoenix=/kaggle/working --manifest   # file có/thiếu + dung lượng
```
→ `overview.md` + `.csv`. Ô `–` = chưa train (không bịa số).

**Tải file nào về từ Kaggle?** Chỉ cần **`*.json`** trong mỗi run: `test_results.json` (bắt buộc),
`*_history.json`, `latency_*.json`. **BỎ** `best_*.pt`/`last_*.pt` (~43MB/cái) và `.done_*` — không
cần để dựng bảng. Ma trận 5% chỉ ~**0.14 MB** JSON so với ~**1.5 GB** checkpoint. Xem chi tiết ở
[`README.md`](README.md) mục "Bảng TỔNG".

---

## (Tuỳ chọn) Kiểm tra pipeline ở máy local trước khi tốn quota Kaggle

Không cần PHOENIX thật / GPU. Cần: `torch numpy pandas sentencepiece sacrebleu matplotlib` (KHÔNG
cần `mediapipe`/`opencv` — chỉ dùng cho extract). Chạy 1 config đơn lẻ để debug:

```bash
python main.py --subset 0.05 --encoder transformer --algo scst --phase all
```
