# Hướng dẫn thực thi — Extract pose → Train → Lấy số liệu cho paper/slide

> Tài liệu này là quy trình **chạy thật** ngắn gọn. Chi tiết thiết kế xem
> [`docs/1_Thuyet_Trinh_Tong_Hop.md`](docs/1_Thuyet_Trinh_Tong_Hop.md) §K. Subset chuẩn = **25 / 50 / 100 %**
> (đã đồng bộ paper + slide + docs + code). Bắt buộc chạy đúng thứ tự: **Bước 0 (extract) → Bước 1 (train)**.

---

## 0. Chuẩn bị Kaggle (làm 1 lần)

Cần 3 thứ trên Kaggle:

| Kaggle Dataset | Nội dung | Khi nào có |
|---|---|---|
| `phoenix-2014t` | Bản gốc PHOENIX-2014-T (thư mục ảnh PNG, **không phải video**) | Upload sẵn |
| `slt-rl-code`   | Chính thư mục `code/` này (nén/upload thành dataset) | Upload sẵn |
| `phoenix-poses` | Pose `.npz` đã trích | **Sinh ra ở Bước 0** |

Đường dẫn trong [`configs/config.py`](configs/config.py) đã khớp sẵn Kaggle (`/kaggle/input/phoenix-2014t`,
`/kaggle/input/phoenix-poses`, `/kaggle/working`) — **không cần sửa gì** nếu đặt tên dataset đúng như trên.

---

## Bước 0 — Trích pose (MỘT LẦN DUY NHẤT, CPU-only)

Mở notebook [`KAGGLE_NOTEBOOK_EXTRACT.ipynb`](KAGGLE_NOTEBOOK_EXTRACT.ipynb) trên Kaggle.

1. **Accelerator = None (CPU)** — MediaPipe chạy CPU, **không tốn quota GPU**.
2. Add dataset input: `phoenix-2014t`, `slt-rl-code`.
3. Chạy tuần tự các cell:
   - Cell 3 (`--limit 5`) = **smoke-test đường dẫn** — BẮT BUỘC làm trước. 10–20h là quá đắt để phát hiện sai đường dẫn.
   - Cell 3b kiểm tra 1 file `.npz`: phải ra shape `[T, 183]`, `float32`, `tỉ lệ phần tử = 0` không được ~1.0.
   - Cell 4 (`--workers 0`) = chạy FULL. Resumable: hết giờ session cứ chạy lại, file `.npz` đã có tự bỏ qua.
4. Cuối script in ra **`missing_counts` (body / lhand / rhand)** — 📊 **GHI LẠI 3 con số này** (bằng chứng định lượng cho vấn đề pose-quality, dùng cho F.8 và có thể đưa vào paper).
5. Kỳ vọng **8257 file `.npz`** (PHOENIX-2014T full). Xong thì tạo Kaggle Dataset tên **`phoenix-poses`** từ tab Output.

⏱️ ~10–20h CPU. Tăng tốc: `--workers 8` (8 process ≈ 1.7h) hoặc chia nhiều máy bằng `--num_shards N --shard i`.

---

## Bước 1 — Train toàn bộ ma trận thí nghiệm (GPU T4×2)

Mở notebook [`KAGGLE_NOTEBOOK.ipynb`](KAGGLE_NOTEBOOK.ipynb).

1. **Accelerator = GPU T4 x2**.
2. Add 3 dataset input: `phoenix-2014t`, `phoenix-poses`, `slt-rl-code`.
3. Cell 3 kiểm tra `phoenix-poses` mount đúng (in ra số file > 0).
4. **Cell 4 = 1 lệnh duy nhất cho TẤT CẢ:**
   ```bash
   python run_all.py --subset 0.25
   ```
   Lệnh này tự chạy: baseline sàn · Exp 1/4/7/8/9/11/15 · P7 two-stage · reward ablation · selection/decode policy · latency 6 encoder → rồi **tự sinh bảng + biểu đồ** ở cuối. **Resumable** (marker `.done_*`): hết giờ session, chạy lại **đúng lệnh này**, bước đã xong tự bỏ qua.
5. Khi có thêm quota, chạy tiếp (không bắt buộc ngay):
   ```bash
   python run_all.py --subset 0.5
   python run_all.py --subset 1.0
   ```

⏱️ Ước tính (toàn ma trận): **25% ≈ 25–30h** (vừa ~1 tuần quota) · **50% lớn hơn đáng kể** · **100% rất lớn** (trải nhiều session, dùng `--groups` để giới hạn nếu cần, ví dụ `--groups core,encoders`).

> **Chỉ có 1 tuần quota?** Bước 0 + `run_all.py --subset 0.25` là **đủ** để có toàn bộ bảng so sánh chính và bảo vệ luận điểm. 50/100% là mở rộng.

---

## Bước 2 — Số liệu ra ở đâu (tự sinh, không cần lệnh thêm)

Sau mỗi lần `run_all.py`, trong `/kaggle/working/`:

| Đường dẫn | Nội dung | Dùng cho |
|---|---|---|
| `comparison_table.csv` / `.md` | 1 bảng gộp MỌI run/subset | Tra cứu tổng |
| `report/tables/table_*.csv` `.md` | 6 bảng lọc sẵn (main · encoders · reward · ablations · baseline · latency) | Đọc/kiểm |
| `report/tables/tab_main.tex` `tab_reward.tex` `tab_encresults.tex` | 3 bảng LaTeX **dán thẳng vào paper** (khớp `\label{tab:main/reward/encresults}`) | **Paper** |
| `report/figures/*.pdf` `*.png` | 5 biểu đồ (BLEU/epoch · ΔBLEU/subset · reward trade-off · 6 encoder · thuật toán) | **Paper + Slide** |

Cell 8 nén tất cả (`all_logs.tar.gz`) để tải về.

> Lưu ý trung thực đã ghi sẵn: pipeline chỉ tính **BLEU-4** (+ rep-rate, len-ratio), nên cột **BLEU-1/ROUGE-L trong `tab_main` để `--`**, không bịa số. Muốn có thật phải thêm code vào `evaluate()`.

---

## Bước 3 — Gửi mình số liệu để hoàn thiện paper + slide

Sau khi chạy xong (dù chỉ subset 25%), gửi mình **1 trong 2**:

- **Cách nhanh:** file `comparison_table.csv` + 3 con số `missing_counts` ở Bước 0.
- **Cách đầy đủ:** cả thư mục `report/` (`all_logs.tar.gz`).

Mình sẽ điền các ô `-` trong [`../paper/sn-article.tex`](../paper/sn-article.tex) (Bảng \ref{tab:main}, \ref{tab:reward}, \ref{tab:encresults}), gắn biểu đồ, và cập nhật phần kết quả của slide cho khớp số thật.

---

## (Tuỳ chọn) Kiểm tra pipeline ở máy local trước khi tốn quota Kaggle

Không cần PHOENIX thật / GPU. Pipeline đã được **smoke-test end-to-end** (21/21 pass: 6 encoder × XE, 5 thuật toán RL, baseline, selection/decode policy, P7, latency, bảng + biểu đồ). Cần: `torch numpy pandas sentencepiece sacrebleu matplotlib` (KHÔNG cần `mediapipe`/`opencv` — chỉ dùng cho extract). Chạy 1 config đơn lẻ để debug:

```bash
python main.py --subset 0.25 --encoder transformer --algo scst --phase all
```
