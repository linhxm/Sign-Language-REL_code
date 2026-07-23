# Demo — dịch ngôn ngữ ký hiệu (XE vs SCST)

Video ký hiệu → MediaPipe Holistic (pose 183-d/frame) → SLTTransformer → câu tiếng Đức.
Cùng một chuỗi pose chạy qua **hai checkpoint** của cùng một run để so sánh trực tiếp:

| Checkpoint | Pha |
|---|---|
| `best_xe.pt` | Cross-Entropy (baseline) |
| `last_rl.pt` | RL / SCST — checkpoint **cuối** pha RL |

Dùng `last_rl.pt` chứ không phải `best_rl.pt` là có chủ đích: ở nhiều run SCST không vượt XE nên
`best_*.pt` không tồn tại, và chính hành vi của model cuối pha RL mới là thứ cần xem.

## File

| File | Vai trò |
|---|---|
| `slt_demo.py` | Lõi: trích pose từ video, nạp checkpoint, decode (greedy/beam) |
| `app.py` | Giao diện Gradio |
| `Demo_Colab.ipynb` | Notebook chạy trên Google Colab (khuyến nghị) |

`slt_demo.py` **import thẳng** `models/`, `configs/`, `data/tokenizer.py` từ repo train thay vì
chép lại kiến trúc — chép lại chỉ cần lệch một tham số là `load_state_dict` vẫn chạy nhưng output
thành rác. Thư mục code được tìm theo thứ tự: `$SLT_CODE_DIR` → `../code` → `/content/Sign-Language-REL_code`.

## Chạy trên Colab

Mở `Demo_Colab.ipynb`, upload lên Drive một thư mục như sau rồi chạy lần lượt các cell:

```
MyDrive/slt_demo/
    run1_transformer_subset25/
        best_xe.pt        # copy từ results/phoenix/run1_transformer_subset25/
        last_rl.pt
    spm.model             # copy từ evidence/phoenix25/spm.model
    slt_demo.py           # 2 file trong thư mục demo/ này
    app.py
```

`spm.model` ở `evidence/phoenix5`, `phoenix10`, `phoenix25` là **cùng một file** (tokenizer train
một lần trên toàn bộ text train split, dùng chung mọi subset) — lấy cái nào cũng được.

Có thể bỏ 2 file `.py` khỏi Drive nếu bạn commit thư mục `demo/` vào repo
`Sign-Language-REL_code` — cell 4 của notebook ưu tiên tìm chúng trong repo đã clone.

## Chạy local

```bash
pip install "mediapipe==0.10.14" sentencepiece gradio opencv-python-headless torch
python demo/app.py --results results/phoenix --spm evidence/phoenix25/spm.model
```

Cờ: `--device cuda|cpu`, `--share` (link public), `--port`.

`--results` là thư mục **cha**; app tự quét đệ quy mọi thư mục con có `best_xe.pt` và cho chọn
trong dropdown, nên chọn được cả các run encoder khác (`gcn`, `tcn`, `perceiver`, …) và các mức
subset 5/10/25%. Encoder được suy ra từ chính tên tham số trong checkpoint (`detect_encoder`).

Không cần giao diện thì gọi trực tiếp:

```python
from slt_demo import PoseExtractor, Translator
pose, stats, _ = PoseExtractor().extract("video.mp4")
tr = Translator.from_checkpoint("results/phoenix/run1_transformer_subset25/last_rl.pt",
                                "evidence/phoenix25/spm.model")
print(tr.translate(pose, decode="beam", beam_size=4))
```

## Đầu vào

- **Video** (`.mp4`, upload hoặc webcam). MediaPipe chạy CPU ~10–20 fps; `stride=0` (mặc định)
  tự bỏ bớt frame sao cho số frame xử lý ≤ 600, sau đó lấy mẫu đều về ≤300 frame đúng như
  `data/dataset.py` làm lúc train.
- **File pose `.npz`** đã trích sẵn (key `pose`, shape `[T,183]`) — vd một câu trong
  `evidence/phoenix-poses.zip`. Bỏ qua hoàn toàn MediaPipe, chạy tức thì, và cho output sát thực
  tế nhất vì đúng phân phối train.

Tiền xử lý trong `slt_demo.py` khớp từng bước với `data/extract_poses.py` (layout 33 body×(x,y,vis)
+ 21 tay trái×(x,y) + 21 tay phải×(x,y) = 183, nội suy tuyến tính từng nhóm cho frame thiếu
landmark) và `data/dataset.py` (uniform-sample về `max_frames=300`). Không có chuẩn hoá nào khác.

## Kỳ vọng chất lượng

Model train trên PHOENIX-2014T — bản tin thời tiết tiếng Đức, một người ký chính diện trên nền
xám, 210×260px — với test BLEU-4 ≈ 6 (`results/phoenix/run1_transformer_subset25/test_results.json`:
XE 5.86 · SCST 5.94). Video webcam ở bối cảnh khác nằm ngoài phân phối train nên câu sinh ra vẫn
là tiếng Đức trôi chảy về thời tiết nhưng **không liên quan nội dung ký hiệu**. Đó là hành vi đúng
của model ở quy mô dữ liệu này, không phải lỗi demo.

Ví dụ chạy thật trên một câu PHOENIX (`01April_2010_Thursday_heute-6696`, beam=4):

```
XE  : dabei wird es sehr windig
SCST: ich wünsche ihnen noch einen schönen abend und machen sie es gut
```
