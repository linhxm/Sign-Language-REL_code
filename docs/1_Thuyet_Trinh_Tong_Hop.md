---
title: "1. SLT × RL — Tài liệu tổng hợp theo flow thuyết trình"
tags: [slt, reinforcement-learning, thuyet-trinh, tong-hop]
slide: "../../slides/index.html"
---

# 1. SLT × RL — Tài liệu tổng hợp

> [!warning] CẬP NHẬT PHẠM VI (07/2026) — đọc trước
> Pipeline đã **thu gọn** để báo cáo thực nghiệm ở **5%** (train 5% split train, dev/test luôn full).
> Hai nhánh sau **đã gỡ khỏi pipeline** và chuyển thành **hướng phát triển** — xem
> [[2_Huong_Phat_Trien]] (`docs/2_Huong_Phat_Trien.md`):
> - **Gloss / two-stage P7** (pose→gloss→text): các mục §A/§E/§F/§K nhắc tới P7, CTC gloss, NMT gloss→text đều thuộc nhánh này.
> - **RL ngoài decoder** (mục §F: frame/landmark selection F.6/F.8/F.9, decode-policy F.5, CTC-segment F.14) + Experiment 8.
>
> Các phần bên dưới GIỮ NGUYÊN để tham chiếu lịch sử; khi đọc, coi mọi nội dung gloss/RL-ngoài-decoder
> là **future work**, không phải thí nghiệm đang chạy. Ma trận hiện tại: 6 encoder + 8 RL + reward
> ablation + latency + baseline sàn.

> [!info] Đây là tài liệu DUY NHẤT của đề tài (ngoài [[0_Architecture]])
> Gộp toàn bộ 13 file docs cũ (tổng quan, RL, dataset, pipeline, CV, kiến trúc, papers, thực nghiệm, metrics, đề xuất luận văn, roadmap, code review, hướng dẫn chạy) vào một chỗ, **tổ chức theo flow thuyết trình**.
>
> **Cấu trúc:**
> - **Phần I** — đi theo 12 slide của [`slides/index.html`](../../slides/index.html). Mỗi slide có: 🎙️ **Kịch bản** (nói gì) · 🔬 **Phân tích sâu** (không có trên slide, dùng khi bị đào sâu) · ❓ **Câu hỏi dự kiến**.
> - **Phần II** — bảng tra cứu đầy đủ (pipeline, kiến trúc, dataset, 13 experiment, metrics, nhật ký thiết kế 41 mục, roadmap, code review, hướng dẫn chạy, references).
>
> **Ký hiệu giữ nguyên từ docs cũ** để code comment vẫn trỏ đúng: `§A/B/C/D/E/F/G` (RL), `P1–P8` (pipeline), `Experiment N`, `A.N/B.N/C.N` (code review).
>
> Slide viết **hoàn toàn tiếng Anh**, chỉ mang keyword; diễn giải nằm trong modal (click từng ô). Tài liệu này viết tiếng Việt, thuật ngữ giữ tiếng Anh.

> [!abstract] Luận điểm xuyên suốt — nếu chỉ nhớ một câu
> Cross-Entropy dạy model **bắt chước từng token**; nhưng ta **chấm điểm cả câu** bằng một metric không khả vi, và lúc chạy thật model phải tự đi trên chính output của nó. Ba khe hở đó — exposure bias, loss-metric mismatch, degeneracy — chính là **định nghĩa của một bài toán RL**. Và một khi đã dựng được khung RL, nó không dừng ở decoder: **mọi quyết định rời rạc trong pipeline thị giác** đều là một MDP.

> [!tip] Cấu trúc slide — điểm cần biết trước khi trình bày
> **Slide 03, 04, 05 dùng CHUNG một canvas.** Node xương sống **không bao giờ di chuyển**; chỉ lớp overlay đổi:
> - **03** pipeline trần
> - **04** cùng pipeline mờ đi + 7 ô **đỏ**, mỗi ô cắm thẳng xuống đúng mắt xích bị hỏng
> - **05** cùng vị trí đó, ô đỏ **thành ô tím/hồng** = phương pháp giải quyết
>
> Khi thuyết trình hãy bấm 03 → 04 → 05 **liên tục, không dừng lâu** để mắt bắt được phần thay đổi, rồi mới quay lại giảng từng ô.

---
---

# PHẦN I — THEO FLOW THUYẾT TRÌNH

---

## Slide 01 — Cover

### 🎙️ Kịch bản

Ba câu, không đi vào chi tiết:

1. Đề tài **Sign Language Translation kết hợp Deep Learning và Reinforcement Learning** — đồ án liên môn giữa *Reinforcement Learning* và *Xử lý ảnh & video*.
2. Hệ thống chạy **pose-only 183 chiều/frame**, dataset **PHOENIX-2014T ~7.000 câu**, huấn luyện **2 phase: Cross-Entropy warm-start → RL fine-tune**.
3. Điểm em muốn bảo vệ nhất **không phải** "RL làm BLEU cao hơn", mà là **RL là khung tối ưu cho mọi quyết định rời rạc trong pipeline**, không chỉ là một mẹo fine-tune decoder.

> [!tip] Câu chuyển
> "Trước khi nói RL giải quyết gì, em xin phát biểu chính xác bài toán và tuyên bố phạm vi."

### 🔬 Phân tích sâu

4 con số trên slide là 4 **ràng buộc** định hình mọi quyết định phía sau — nên trình bày chúng là *ràng buộc*, không phải *thành tích*:

| Con số | Ý nghĩa thật | Hệ quả thiết kế |
|---|---|---|
| `183-d` | Chỉ có pose, không RGB | Loại toàn bộ nhóm kiến trúc RGB-backbone (§B.4) |
| `~7.000 câu` | Dataset **rất nhỏ** cho seq2seq | `d_model=256`, `dropout=0.3` — nhỏ có chủ đích; RL cực nhạy overfit |
| `2-phase` | RL không train from scratch | Reward ≈ 0 ở early training nếu bỏ CE warm-start |
| `13 experiment` | Ngân sách ~30 GPU-h/tuần T4×2 | Phải có bảng ưu tiên, không chạy hết được |

**Nguyên tắc xuyên suốt đề tài** (nói ở đây hoặc để dành cho slide 12): *không tách rời hai hướng* — mọi lựa chọn CV đều đi kèm câu hỏi "RL khai thác được gì?", mọi thiết kế RL đều bắt rễ từ một pipeline thị giác cụ thể.

### ❓ Câu hỏi dự kiến

> [!question] "Em kỳ vọng BLEU bao nhiêu?"
> Mốc tham chiếu: Camgoz 2018 end-to-end **9.58**; Sign Language Transformers 2020 **~21.8**; SOTA gloss-free hiện tại **~22–27** BLEU-4. Nhưng đề tài chạy trên **subset 25/50/100%** với model nhỏ, nên **con số tuyệt đối không phải mục tiêu**. Mục tiêu là **ΔBLEU giữa CE-only và CE+RL từ cùng checkpoint gốc**, cộng khoảng cách tới baseline sàn. Em sẽ nhấn lại ở slide 10.

---

## Slide 02 — The Task

### 🎙️ Kịch bản

**Định nghĩa.** SLT ánh xạ chuỗi video ký hiệu `V = (v₁..v_T)` sang câu văn bản `Y = (y₁..y_N)`. Đây là seq2seq **xuyên phương thức**: input thị giác liên tục, output chuỗi rời rạc có ngữ pháp. Hai đầu không cùng "đơn vị" — đó là nguồn gốc của mọi khó khăn phía sau.

**Ba ô lớn** (click từng ô để mở chi tiết):

- **Input** — pose-only 183-d: `33 body × (x, y, visibility) = 99` + `2 tay × 21 × (x, y) = 84`.
- **Output** — text translation, đánh giá BLEU / ROUGE / METEOR / BERTScore. Gloss chỉ là trạm trung gian tuỳ chọn (nhánh P7).
- **Task choice** — vì sao SLT chứ không phải SLR/SLP.

**Hàng "OUT OF SCOPE"** — bốn thứ bị loại, mỗi cái click ra được lý do:

- **RGB**: dùng trực tiếp frame video. Loại vì không có cache RGB feature; trích on-the-fly phá vỡ ngân sách GPU.
- **Optical flow**: vector dịch chuyển pixel từ frame `t` sang `t+1`. Loại vì phụ thuộc RGB + chi phí RAFT/TV-L1.
- **Face landmark**: điểm biểu cảm gương mặt. Loại để giữ vector gọn.
- **LLM decoder**: quá nhiều ẩn số chưa verify (LLM tiếng Đức scale nhỏ, licence, LoRA trên T4 16GB).

**Câu chốt:** cả bốn bị loại vì **chưa verify được tiền đề trong ngân sách**, không phải vì kém giá trị — tất cả đều nằm ở hướng phát triển (§H.7).

### 🔬 Phân tích sâu

**Vì sao SLT chứ không phải SLR/SLP** — đây là thứ dẫn thẳng tới lý do dùng RL:

| | SLR | **SLT (đề tài này)** | SLP |
|---|---|---|---|
| Ánh xạ | Video → Gloss/nhãn | **Video → Text** | Text → Video/Pose |
| Bản chất | Classification / CTC | **MT xuyên phương thức** | Generative sinh chuyển động |
| RL phù hợp? | Ít — nhãn đã rõ | **Rất phù hợp** | Ngoài phạm vi |

Pipeline chính P1 là end-to-end + gloss-free; P7 (two-stage qua gloss) làm đối chứng nội bộ.

> [!warning] Điểm dễ bị bắt bẻ nhất của slide này
> Loại face landmark là một **hạn chế thật**, không phải một tối ưu. Trong DGS (ngôn ngữ ký hiệu Đức), *non-manual markers* — nhíu mày, hướng nhìn, chuyển động đầu — **mang nghĩa ngữ pháp** (phủ định, câu hỏi, mệnh đề điều kiện). Bỏ chúng là bỏ một phần tín hiệu ngữ pháp.
> **Cách nói đúng:** "Em loại face landmark vì ngân sách và vì muốn cô lập biến, nhưng đây là hạn chế được ghi rõ ở §H.6 và nằm trong hướng phát triển §H.7 — không phải kết luận rằng biểu cảm mặt vô ích."

**Vì sao pose-only hợp lý về khoa học, không chỉ về ngân sách:**

- Bất biến với nền, ánh sáng, trang phục → giảm lượng lớn nhiễu không liên quan.
- Có tiền lệ trực tiếp: **SPOTER** (WACV-W 2022) là Transformer nhẹ trên pose chuẩn hoá.
- Giữ compute đủ rẻ để chạy **nhiều lần** — mà RL bắt buộc phải chạy nhiều lần (variance cao).

**Thay thế rẻ cho optical flow:** đạo hàm hữu hạn trên chính pose, `pose[t] − pose[t−1]`, biến 183-d thành 366-d. Gần như miễn phí và nắm phần lớn thông tin chuyển động mà flow cung cấp, ở đúng mức trừu tượng bài toán cần (chuyển động **khớp**, không phải chuyển động **pixel**). Chi tiết §D.3.

**Về dataset:** PHOENIX-2014T được chọn PRIMARY vì là dataset duy nhất **vừa có translation reference** (bắt buộc để tính reward) **vừa đủ nhỏ** cho nhiều lần chạy RL, **vừa không có rào cản truy cập**. Bảng khảo sát đầy đủ 6 dataset ở §C.

### ❓ Câu hỏi dự kiến

> [!question] "Sao không dùng RGB? RGB chắc chắn nhiều thông tin hơn pose."
> Đúng là nhiều thông tin hơn. Ba lý do: (1) không có cache RGB feature, trích on-the-fly phá ngân sách GPU — bài học từ chính việc pose extraction đã tốn 10–20h CPU và phải tách rời hoàn toàn khỏi vòng train; (2) xây pipeline RGB tự nó là một hạng mục lớn ngoài scope; (3) quan trọng nhất — **câu chuyện chính là RL, không phải modality**. Thêm RGB là thêm một biến nhiễu lớn vào đúng phép so sánh cần cô lập.

> [!question] "Pose 183-d có mất thông tin về signing space không?"
> Có, một phần. MediaPipe chuẩn hoá `[0,1]` **theo khung hình**, chưa theo vai/hông — nên khoảng cách signer↔camera vẫn là biến nhiễu. Chuẩn hoá "signing space" (chia khoảng cách vai, trừ tâm hông — kỹ thuật SPOTER) là cải tiến **chi phí thấp, lợi ích cao nhất** đang nằm đầu danh sách §D.5.

> [!question] "Vì sao bỏ visibility của tay mà giữ của body?"
> Vì visibility của tay từ MediaPipe **không đáng tin** — nó dao động mạnh ngay cả khi tay đang nhìn thấy rõ. Giữ lại tức là thêm nhiễu có cấu trúc vào đúng nhóm landmark quan trọng nhất (§G.1).

> [!question] "Sao không dùng CSL-Daily, nó lớn hơn nhiều?"
> CSL-Daily (20.654 câu) tốt về kỹ thuật nhưng cần **giảng viên ký thoả thuận** — rủi ro deadline. Nguyên tắc: chỉ theo đuổi nếu bắt đầu xin quyền từ tuần 1-2, không để chặn tiến độ. Ngoài ra BLEU tiếng Trung nhạy cách tách từ nên số khó so. Xem §C.

---

## Slide 03 — Base Pipeline

### 🎙️ Kịch bản

"Đây là hệ thống **trước khi có RL**. Em đi từ trái sang phải." (Mỗi node click được để xem chi tiết cấu hình.)

1. **INPUT** — PHOENIX-2014T, ~7K câu DGS, split 7096/519/642.
2. **L1 · EXTRACT** — MediaPipe Holistic → 183-d/frame, cache `.npz`.
3. **L1 · PREPROCESS** — điền landmark thiếu bằng `np.interp`, cắt về tối đa 300 frame bằng `np.linspace`.
4. **L2 · ENCODER** — Transformer pose encoder, 8.26M tham số → sinh `memory`.
5. **L3 · DECODER** — SLTTransformer enc–dec, weight tying, `d_model=256`, 4 head, 4 layer.
6. **L4 · TRAIN** — Cross-Entropy teacher forcing, label smoothing 0.1 → `best_xe.pt`.
7. **INFERENCE** — tự hồi quy từ `<bos>`, greedy/beam, temperature cố định.
8. Nhánh dưới: **TOKENIZER** BPE vocab 3000 cấp target token; **EVAL** BLEU-4 bằng sacreBLEU.

> [!important] Câu phải nói ở slide này
> "**Pose extraction tách rời hoàn toàn khỏi vòng train.** MediaPipe chạy CPU mất 10–20 giờ; nếu chạy lại mỗi lần train thì không còn ngân sách GPU cho bất kỳ experiment RL nào. Toàn bộ thiết kế phía sau đều bị định hình bởi ràng buộc này." (§G.34)

### 🔬 Phân tích sâu — vì sao từng con số là con số đó

Dùng khi hội đồng hỏi "sao chọn tham số này". Trích nhật ký thiết kế 41 mục ở §G.

| Tham số | Giá trị | Lý do |
|---|---|---|
| `pose_dim` | 183 | 33×3 + 21×2 + 21×2; giữ visibility body, bỏ của tay (§G.1) |
| `max_frames` | 300 | Cân bằng thông tin vs `O(T²)` attention trên T4 16GB (§G.2) |
| Truncate | `np.linspace` uniform | Baseline **có chủ đích đơn giản** — để chừa chỗ cho RL adaptive sampling F.9 (§G.3) |
| `vocab_size` | 3000 BPE | Corpus nhỏ; vocab lớn sinh token hiếm không đủ dữ liệu để học (§G.4) |
| `max_text_len` | 60, cắt **giữa** | Giữ `<bos>`/`<eos>` — cắt đuôi thì model không bao giờ thấy `<eos>` ở câu dài (§G.5) |
| `d_model` | 256 / 4 head / 4 layer | **Nhỏ có chủ đích** vì dataset nhỏ; RL cực nhạy overfit (§G.6) |
| `dropout` | 0.3 | Regularization mạnh hơn chuẩn, hệ quả của dòng trên (§G.7) |
| `label_smoothing` | 0.1 | Giảm overconfidence **và giữ entropy của policy** cho exploration khi vào RL (§G.8) |
| `xe_lr` | 5e-4 + warmup 2000 + cosine | Transformer from scratch nhạy lr giai đoạn đầu (§G.9) |
| Weight tying | `out_proj = tok_embed` | Press & Wolf 2017 — quan trọng khi vocab nhỏ (§G.19) |
| Pre-LN | `norm_first=True` **+ final LayerNorm** | Ổn định train from scratch. ⚠️ Thiếu final norm là **bug đã sửa** — xem §J.8 (§G.20) |
| PE | Sinusoidal, không learned | Không tốn tham số, extrapolate được chuỗi dài hơn lúc train (§G.21) |

> [!note] Nội suy landmark — chi tiết nhỏ nhưng đúng chất CV
> Landmark thiếu được nội suy `np.interp` **theo từng segment riêng** (body / tay trái / tay phải), **không** zero-fill. Lý do: zero-fill tạo nhiễu giả — model không phân biệt được "tay đứng yên ở gốc toạ độ" với "thiếu dữ liệu". Segment thiếu **toàn video** vẫn giữ 0 vì không có neo — nhiễu tiềm ẩn còn sót, nay đã được **đếm và in ra cuối `extract_poses.py`** thay vì bỏ qua (§G.40).

### ❓ Câu hỏi dự kiến

> [!question] "Vì sao encoder và decoder tách làm 2 khối trên sơ đồ trong khi code là một `SLTTransformer`?"
> Vì **encoder thay thế được**: factory `build_pose_encoder()` có 6 lựa chọn P1–P6 dùng chung một interface `forward(pose, pose_mask) -> [B,T,d_model]`. Đổi bằng cờ `--encoder`, phần còn lại không đổi. Đó là hạ tầng cho Experiment 4. Bảng đầy đủ §A và §B.

> [!question] "300 frame có đủ không?"
> PHOENIX phần lớn dưới 300 frame nên `linspace` ít khi kích hoạt. Nhưng chính cơ chế cắt uniform này là **vấn đề số 2 ở slide sau**.

> [!question] "Vì sao dùng BPE chứ không phải word-level?"
> Corpus PHOENIX nhỏ (~7K câu) và tiếng Đức có từ ghép dài. Word-level sẽ tạo vocab lớn với nhiều token tần suất 1 — không học được. BPE 3000 là điểm cân bằng; nếu SentencePiece báo lỗi không đủ ký tự để tạo 3000 merge thì giảm còn 2000 và **ghi lại vì nó đổi §G.4**.

---

## Slide 04 — Where It Breaks

> [!abstract] Slide bản lề của cả bài
> Slide 03 vẽ hệ thống. Slide 04 chỉ vào từng mắt xích và nói nó hỏng ở đâu. Slide 05 là câu trả lời, **ở đúng vị trí đó**. Khi nói, hãy nói rõ "cái này em sẽ quay lại ngay slide sau".

### 🎙️ Kịch bản — trái sang phải, mỗi ô 20–30 giây

**① Pose quality (trỏ MediaPipe Holistic).** Tay che mặt hoặc che nhau, chuyển động nhanh → MediaPipe mất track. Điểm mấu chốt: **nhóm landmark nào đáng tin thay đổi theo từng câu** — không có quy tắc cố định nào đúng cho mọi mẫu.

**② Temporal sampling (trỏ Preprocess).** `np.linspace` cắt frame đều nhau, đối xử frame tĩnh (signer giữ tay) và frame chuyển tiếp giữa hai ký hiệu **như nhau**. Trong khi frame chuyển tiếp mới mang thông tin phân biệt.

**③ Sequence length (trỏ Encoder).** T=300 frame nén xuống ~10 token — tỉ lệ 30:1. Cộng attention `O(T²)`. Cộng **co-articulation**: ranh giới giữa hai ký hiệu bị "nuốt" khi ký liên tục.

**④ Degeneracy (trỏ Decoder) — §A.3.** Decoder thiên về token "an toàn", lặp cụm từ (Holtzman 2020). Nhấn: **đây là failure mode quan sát thật trong log, không phải giả định lý thuyết** — nó là lý do `repetition_penalty` tồn tại chứ không phải tuỳ chọn thừa.

**⑤ Exposure bias (trỏ Cross-Entropy) — §A.1.** Train với teacher forcing, decoder **luôn** nhận ground-truth. Inference, nó nhận token do chính nó vừa sinh. Hai phân phối khác nhau → một lỗi ở bước `t` làm lệch toàn bộ phần còn lại. Dataset nhỏ (~7K câu, overfit pattern hẹp) + input thị giác nhiễu làm nặng hơn NMT thường.

**⑥ Decode strategy (trỏ Inference).** Một `temperature` cho mọi câu. Câu dễ và câu khó dùng chung một siêu tham số, và cái "tốt nhất" chỉ là kết quả một lần grid-search.

**⑦ Loss-metric mismatch (ô dưới, trỏ Eval) — §A.2.** Ô quan trọng nhất. Train tối ưu CE **token level**. Đánh giá BLEU **sequence level**, và BLEU **không khả vi** — không tồn tại `∂BLEU/∂θ`. Ta tối ưu một thứ và chấm điểm bằng thứ khác.

> [!tip] Câu chuyển sang slide 05
> "Ba trong bảy ô này — exposure bias, loss-metric mismatch, degeneracy — không phải ba lỗi riêng lẻ. Chúng có **chung một nghiệm**." *(bấm sang slide 05 ngay)*

### 🔬 Phân tích sâu

**Phân loại 7 vấn đề theo bản chất** — hữu ích nếu bị hỏi "sao lại gom chung":

| Nhóm | Vấn đề | Bản chất |
|---|---|---|
| **Thuộc hàm mục tiêu** | ⑤ exposure bias · ⑦ mismatch · ④ degeneracy | Sai ở *cách huấn luyện* → **RL trên decoder** |
| **Thuộc quyết định rời rạc** | ① landmark · ② sampling · ⑥ decode strategy | Đang quyết bằng heuristic cố định → **RL ngoài decoder** |
| **Thuộc kiến trúc** | ③ độ dài chuỗi | Cần đổi kiến trúc, không phải đổi loss → **Perceiver / ST-GCN** |

Đây là lý do đề tài **không** chỉ là "áp RL vào SLT": bảng trên cho thấy RL trả lời 6/7 ô, nhưng bằng **hai cơ chế khác nhau**, và ô còn lại thì thành thật rằng RL không giải quyết được.

**Bảng khó khăn cốt lõi ↔ điểm chạm CV / RL** (mở rộng nếu cần):

| Khó khăn | Hướng CV | Hướng RL |
|---|---|---|
| **Temporal dependency** (300+ frame) | Transformer/GCN encoder, giảm độ dài chuỗi | Frame selection F.6–F.9 |
| **Co-articulation** (ranh giới ký hiệu bị nuốt) | Segmentation | Sign segmentation F.14 (CTC forced-alignment) |
| **Signer variation** | Chuẩn hoá vai/hông + augmentation cố định (§D.5, §D.6) | **Không dùng RL** — RL-cho-augmentation đã loại (tốn vòng lặp ngoài) |
| **Occlusion** (tay che mặt/tay) | Missing-point recovery (§D.5) | Dynamic landmark selection F.8 |
| **Long sequence** (nén T frame → 10-20 token) | TCN/Perceiver (P5/P6) | Adaptive temporal sampling F.9, memory selection F.16 |
| **Data scarcity** (~7K câu) | Augmentation + kiến trúc nhẹ (ST-GCN) | RL tối ưu trực tiếp metric, **nhưng cũng cần nhiều rollout** — vừa là lý do vừa là giới hạn |
| **Domain shift** | Fine-tune transfer (Experiment 13) | Curriculum RL (§C.12 + F.18) |

> [!warning] Đừng hứa quá ở slide này
> Có **một dòng mà RL bị loại có chủ đích**: signer variation. RL-cho-augmentation cần vòng lặp ngoài (train lại nhiều lần) — vượt ngân sách GPU. Nói ra điều này làm bài đáng tin hơn nhiều so với để hội đồng tự phát hiện.

### ❓ Câu hỏi dự kiến

> [!question] "Exposure bias có thật sự nghiêm trọng không? Nhiều paper nói ảnh hưởng bị thổi phồng."
> Phản biện đúng và có nguồn: **Kiegeland & Kreutzer (NAACL 2021)** lập luận phần lớn "gain" của RL cho NMT là ảo, đến từ artifact thiết kế thí nghiệm chứ không phải từ việc sửa exposure bias. Em **trích dẫn bắt buộc** bài này và thiết kế ablation tránh đúng các artifact họ chỉ ra: so sánh từ **cùng checkpoint gốc**, có **baseline sàn không-học**, báo cáo trên **test set**. Kết luận của họ không phải "RL vô dụng" mà là "gain có thật nếu exploration và scaling đúng" — nên đây là ràng buộc phương pháp, không phải lý do bỏ đề tài.

> [!question] "Sao không sửa exposure bias bằng scheduled sampling cho rẻ?"
> Scheduled sampling giải quyết ⑤ nhưng **không** giải quyết ⑦ — nó vẫn tối ưu CE token-level, vẫn không có gradient của BLEU. RL giải quyết cả hai bằng một cơ chế, cộng ④ nhờ reward đa mục tiêu.

---

## Slide 05 — How RL Fixes It

> [!important] Cách trình bày slide này
> **Đừng giảng lại từ đầu.** Nói: "Cùng sơ đồ, cùng vị trí — mỗi ô đỏ vừa nãy giờ là một phương pháp cụ thể." Rồi đi lại đúng thứ tự ①→⑦ để người xem ghép cặp được.
> Chú ý **màu**: hồng = RL × CV (ngoài decoder) · tím = RL trên decoder · xanh dương = **không phải RL**, là kiến trúc.

### 🎙️ Kịch bản

| Ô | Vấn đề (04) | Phương pháp (05) | Loại |
|---|---|---|---|
| ① | Pose quality | **F.8** Landmark selection — chọn nhóm khớp đáng tin | RL × CV |
| ② | Uniform sampling | **F.6 / F.9** Frame policy — học frame nào quan trọng | RL × CV |
| ③ | Sequence length | **Perceiver · ST-GCN** — độ phức tạp tuyến tính | **Kiến trúc, không phải RL** |
| ④ | Degeneracy | **Repetition penalty** trong reward đa mục tiêu | RL |
| ⑤ | Exposure bias | **SCST · PPO · MRT** — rollout = inference | RL |
| ⑥ | Decode strategy | **F.5** Decode policy — temperature per-input | RL × CV |
| ⑦ | Loss-metric mismatch | **max E[R(y)]** — R không cần khả vi | RL |

> [!important] Câu quan trọng nhất slide này
> "Ô ③ màu khác vì nó **không phải RL**. Độ dài chuỗi là vấn đề kiến trúc, và em không giả vờ RL giải quyết được nó. Nói ra điều đó làm sáu ô còn lại đáng tin hơn."

### 🔬 Phân tích sâu

**Vì sao "RL ngoài decoder" là đóng góp chứ không phải trang trí** — ba lý do, xếp theo sức thuyết phục:

1. **Nó làm đề tài liên môn thật sự liên môn.** Nếu RL chỉ chạm decoder thì đây là hai đồ án ghép cạnh nhau: một đồ án CV (pose encoder) và một đồ án RL (SCST). Việc RL ra quyết định *trên chính dữ liệu thị giác* mới là điểm giao.
2. **Nó tổng quát hoá được.** Khung "quyết định rời rạc trong pipeline → MDP" không phụ thuộc SLT. Frame selection, landmark selection, decode strategy là mẫu hình xuất hiện ở mọi pipeline video.
3. **Nó có counter-baseline nghiêm túc** — điều làm nó thành khoa học chứ không phải demo (slide 10).

> [!danger] Cái bẫy lớn nhất của toàn bộ nhóm F
> Một frame-selection policy học được "giữ 50% frame mà BLEU không giảm" **nghe rất hay nhưng có thể hoàn toàn vô nghĩa** — vì `uniform-stride` giữ 50% frame có khi cũng cho kết quả y hệt, mà nó **không cần học gì cả**. Đây là lý do `eval_baselines.py --kind selection` là **bắt buộc**. Policy phải thắng **uniform-stride cùng `keep_ratio`, cùng cơ chế soft-mask**. Thắng mỗi `full-frame` thì chưa chứng minh được gì.

### ❓ Câu hỏi dự kiến

> [!question] "RL có thật sự cần thiết không, hay chỉ là làm phức tạp thêm?"
> Câu trả lời nằm ở ô ⑦: BLEU **không khả vi**. Không có cách nào tối ưu trực tiếp nó bằng gradient descent. Mọi phương án khác — scheduled sampling, beam search tốt hơn, reranking — đều là cách gián tiếp. RL là cách trực tiếp duy nhất, và nó đồng thời sửa luôn ⑤ và ④. Còn ①②⑥ thì đó là các quyết định **rời rạc**, cũng không có gradient — cùng một lý do.

---

## Slide 06 — Why Reinforcement Learning

### 🎙️ Kịch bản

"Ba vấn đề thuộc hàm mục tiêu ánh xạ 1-1 sang ba cơ chế của RL."

| Vấn đề | RL giải quyết bằng gì |
|---|---|
| **§A.1 Exposure bias** | Rollout **giống hệt** inference — phân phối train khớp phân phối test |
| **§A.2 Loss-metric mismatch** | Tối ưu trực tiếp `E[R(y)]`, `R` **không cần khả vi** |
| **§A.3 Degeneracy** | Reward **đa mục tiêu**: BLEU + chống lặp + chống cụt |

**§B — Phát biểu MDP** (5 ô, click từng ô):

- **Agent** = decoder `π_θ` (`decode_step`) — thực thể **duy nhất** ra quyết định. Nhấn: **encoder KHÔNG phải agent** — nó chỉ tạo observation.
- **Environment** = `memory` từ encoder + reference (cố định trong episode) + bộ scorer reward.
- **State** = `(memory, y_<t)`. Transition **tất định**: `s_{t+1} = concat(s_t, a_t)`.
- **Action** = chọn 1 token trong vocab 3000.
- **Reward** = metric sequence-level ở **cuối câu** — **sparse**, 1 lần/episode. **Episode** = 1 câu bos→eos.

"Hai kết luận em muốn nói thẳng:"

**Thứ nhất — CE là điều kiện cần nhưng không đủ.** RL from scratch trên vocab 3000 có reward ≈ 0 ở early training: model sinh chuỗi ngẫu nhiên, BLEU bằng 0, không có tín hiệu học. Nên bắt buộc pipeline 2 phase `CE → RL`, và hai phase **tách biệt** (`main.py --phase all`).

**Thứ hai — giới hạn của RL.** Variance cao, train lâu dễ reward-hack hoặc policy collapse. Quan trọng nhất: **RL không tạo ra thông tin ngôn ngữ mới**, nó chỉ định hướng lại phân phối đã học từ CE. Hệ quả: luôn so CE-only với CE+RL **từ cùng một checkpoint gốc**.

### 🔬 Phân tích sâu

**Vì sao MDP này "dễ" hơn MDP điển hình — và khó hơn ở chỗ khác:**

*Dễ hơn:* transition **tất định** (không có ngẫu nhiên trong môi trường), environment **hoàn toàn quan sát được**, episode ngắn (~10–15 bước), reward tính **chính xác** chứ không phải ước lượng.

*Khó hơn:* reward **cực kỳ thưa** — 1 tín hiệu cho cả episode 15 bước. Đây chính là lý do **credit assignment** khó: khi cả câu được 0.3 BLEU, token nào đáng khen, token nào đáng phạt? Actor-Critic và GAE sinh ra để trả lời, nhưng chúng cần critic học được `V(s_t)` — mà với reward chỉ ở cuối câu và 7K câu thì critic rất khó học tốt. **Đây là lý do trung tâm của H3: PPO không nhất thiết thắng SCST.**

**RL giải quyết đúng 3 vấn đề §A:** (1) tối ưu trực tiếp `E[R(y)]` không cần R khả vi; (2) rollout giống hệt inference → giảm exposure bias; (3) reward đa mục tiêu (BLEU + chống lặp + chống cụt).

**Vì sao đúng 2 phase, và vì sao tách biệt** (§G.32, §G.33):

- Train chung `CE + λRL` từ đầu: RL component có reward ≈ 0 nhiều epoch đầu → chỉ là nhiễu cộng vào gradient CE, tốn compute vô ích.
- `--phase rl` khởi tạo **model instance mới** rồi mới load `best_xe.pt` — tránh rò rỉ optimizer/scheduler state. Nếu sai, lr scheduler của XE (đang ở cuối cosine, lr rất thấp) sẽ giết chết phase RL.
- `rl_lr = 5e-6`, **thấp hơn XE 100 lần** — **siêu tham số nhạy nhất toàn hệ thống** (§G.15), chuẩn theo literature SCST/RLHF.

### ❓ Câu hỏi dự kiến

> [!question] "Vì sao encoder không phải agent? Nó cũng nhận gradient từ RL loss mà."
> Nó **có** nhận gradient, nhưng không **ra quyết định rời rạc** nào. Agent là thực thể chọn action từ một không gian action; encoder chỉ là phép biến đổi tất định từ pose sang `memory`. Việc nó nhận gradient chỉ là chain rule đi ngược qua.
> Phân biệt này quan trọng về kỹ thuật: ở PPO, `memory` **bắt buộc phải tính lại có gradient ở mỗi `ppo_epoch`** — nếu tái dùng `memory` từ rollout `no_grad` thì encoder **không bao giờ** nhận gradient từ PPO loss (§G.36). **Ghi chú 07/2026:** đúng lỗi này đã được phát hiện trong `train_scst.py` ở nhánh `rl_baseline_eval_mode=False` và đã sửa — xem §J.2.

> [!question] "Reward sparse như vậy thì có nên dùng reward shaping không?"
> Có, đó là **Reward 9** (§E): incremental-BLEU từng bước, dạng potential-based (Ng 1999) để đảm bảo không đổi policy tối ưu. Hiện ở dạng ghi chú trong `train_ppo.py`, bật sau khi PPO cơ bản chạy ổn — vì bật cùng lúc thì không tách được nguyên nhân khi kết quả xấu.

> [!question] "Làm sao biết RL đang học thật chứ không phải reward-hack?"
> Ba tín hiệu (nói kỹ ở slide 11): BLEU tăng mà ROUGE giảm (length hacking) · tỉ lệ lặp tri-gram tăng · `avg_advantage` sụp về ≈0 sớm. Về hạ tầng, `eval_every=1` chính là để bắt được điều này — RL có thể trông như đang học trong khi thực ra đang hack, và chỉ dev BLEU thật mới phát hiện được (§G.28).

---

## Slide 07 — RL Algorithms

### 🎙️ Kịch bản

"8 thuật toán đã code, lọc từ 13. Em nhóm theo vai trò." (Click từng ô xem cơ chế + trạng thái.)

**Xương sống:**
- **SCST** (Rennie 2017, §C.2) — baseline là reward của **greedy decode**. Ưu điểm quyết định: "miễn phí", không cần critic, chỉ 2 forward/batch. **Baseline bắt buộc**, mọi so sánh quy về nó.
- **REINFORCE** (Williams 1992, §C.1) — `b = 0`, variance rất cao. Không dùng để thắng, dùng làm **ablation chứng minh vai trò baseline**.

**Trust-region:**
- **PPO** (Schulman 2017, §C.4) — clipped surrogate `min(rA, clip(r,1±ε)A)`, nhiều epoch/rollout, kèm GAE (§C.6). **Ưu tiên cao nhất — Experiment 7.**
- **A2C** (§C.5) — PPO bỏ clip, 1 epoch/rollout. Không phải experiment riêng mà là **ablation đo giá trị của trust-region**.

**Multi-sample:** **MRT** (Shen 2016, §C.9) — risk kỳ vọng `Σ Q(y)·(1−R(y))` trên N candidate, dạng tổng quát của multi-sample SCST.

**"RL rẻ" — không rollout:** **RAML** (Norouzi 2016, §C.10) và **DPO** (Rafailov 2023, §C.7).

**Hỗ trợ:** **Curriculum RL** (§C.12) — reward ramp + câu ngắn trước dài sau.

"Ba thuật toán **đã loại có lý do**: RLHF thật (§C.8 — không ngân sách gán nhãn) · Offline RL (§C.11 — cơ chế chưa đặc tả, và tiền đề 'compute là nút thắt' hoá ra không đúng) · Hierarchical RL (§C.13 — cần 2 policy lồng nhau chưa đặc tả). Cả ba giữ ở hướng phát triển §H.7."

> [!important] Câu quan trọng nhất slide này
> "Giả thuyết H3 của em là **trung lập có chủ đích**: PPO **không nhất thiết** vượt SCST. PPO thiết kế cho reward dày đặc, còn đây reward chỉ có ở cuối episode. Kết quả theo chiều nào cũng hợp lệ **nếu em phân tích được lý do**. Em không đi tìm một chiến thắng."

### 🔬 §C — Bảng 10 thuật toán khả thi (lọc từ 13)

Ký hiệu: `b` baseline, `A = R − b` advantage.

| Mục | Thuật toán | Cơ chế 1 dòng | Đã code | Ưu tiên |
|---|---|---|---|---|
| C.1 | REINFORCE (Williams 1992) | `∇J = E[(R−b)·∇log π]`, b=0 → variance rất cao | ✅ `train_scst.py` (`rl_use_baseline=False`) | Cao (ablation rẻ) |
| C.2 | **SCST** (Rennie 2017) | baseline = reward greedy decode ("miễn phí", không critic); 2 forward/batch | ✅ `train_scst.py` — **xương sống** | Baseline bắt buộc |
| C.3 | Actor-Critic | critic `V(s_t)` cho advantage từng token — nền tảng lý thuyết PPO/A2C | (ValueHead trong PPO) | — |
| C.4 | **PPO** (Schulman 2017) | clipped surrogate + nhiều epoch/rollout — ổn định, sample-efficient | ✅ `train_ppo.py` | **Cao nhất** — Exp 7 |
| C.5 | A2C | PPO bỏ clip + 1 epoch/rollout — đo giá trị trust-region | ✅ `train_ppo.py` (`ppo_use_clip=False`) | TB (gộp ablation PPO) |
| C.6 | GAE (Schulman 2015) | `Σ(γλ)^l δ` cân bằng bias/variance; thử `λ∈{0.9,0.95,1.0}` | (trong PPO, `ppo_gae_lambda`) | Sub-ablation Exp 7 |
| C.7 | DPO (Rafailov 2023) | loss phân loại trên cặp preference tự sinh, ref model đóng băng | ✅ `train_dpo.py` | Thấp (mở rộng rẻ) |
| C.8 | ~~RLHF thật~~ | **LOẠI**: cần preference con người — không ngân sách gán nhãn | — | Hướng mở rộng §H.7 |
| C.9 | MRT (Shen 2016) | risk kỳ vọng `Σ Q(y)·(1−R(y))` trên N candidate | ✅ `train_mrt.py` | Cao — nối Exp 3 |
| C.10 | RAML (Norouzi 2016) | MLE trên target nhiễu quanh ground-truth theo `exp(R/τ)` | ✅ `train_raml.py` | Trung bình |
| C.11 | ~~Offline RL~~ | **LOẠI**: cơ chế chưa đặc tả + tiền đề "compute là nút thắt" không đúng | — | — |
| C.12 | Curriculum RL | reward ramp theo epoch + câu ngắn→dài | ✅ `train_scst.py` + `LengthCurriculumSampler` | Trung bình |
| C.13 | ~~Hierarchical RL~~ | **LOẠI**: cần 2 policy lồng nhau + reward 2 tầng chưa đặc tả | — | Hướng mở §H.7 |

**Vì sao SCST vẫn là xương sống dù PPO "hiện đại hơn":** SCST khai thác đúng một tính chất bài toán này có mà PPO không tận dụng — ta **có sẵn** một baseline tốt và miễn phí, chính greedy decode của model hiện tại. Baseline này tự động thích nghi khi policy cải thiện, không cần học, không có bias của một critic đang học dở. Trong khi PPO phải học `V(s_t)` từ tín hiệu reward chỉ xuất hiện ở cuối câu, trên 7K câu. **Nhiều tham số hơn không đồng nghĩa với ước lượng tốt hơn khi dữ liệu ít.**

**Chi tiết cài đặt nếu bị hỏi sâu:**

- `rl_baseline_eval_mode = True` (§G.22): baseline greedy chạy ở **eval mode**. Bật dropout thì greedy baseline không tất định → thêm variance **vô ích** vào advantage; eval-mode cũng khớp hành vi inference thật.
- `rl_entropy_coef = 0.0` mặc định, chỉ bật ~`1e-3` khi thấy `avg_advantage` sụp sớm (§G.23).
- `grad_clip = 1.0` — quan trọng nhất ở phase RL, vì reward variance cao sinh gradient spike (§G.11).
- `rl_batch_size = 8` < xe 16 — RL cần 2 forward + lưu log-prob, tốn VRAM hơn (§G.14).
- `rl_epochs = 20` — RL chỉ fine-tune, train lâu dễ policy collapse (§G.13).
- AMP bọc forward/backward nhưng **không** bọc vòng reward (string ops CPU) — nhớ khi thêm BERTScore (§G.41).

### ❓ Câu hỏi dự kiến

> [!question] "Vì sao không dùng GRPO — nó đang mới nhất và không cần critic?"
> Đúng, và có tiền lệ trực tiếp: **RVLF (arXiv:2512.07273, 12/2025)** là công trình đầu tiên áp GRPO cho SLT, reward BLEU+ROUGE, cải thiện **+1.11 BLEU-4 trên chính PHOENIX-2014T** (và +5.1 CSL-Daily, +1.4 How2Sign, +1.61 OpenASL). GRPO về bản chất rất gần với **multi-sample SCST** đã có trong code (`rl_n_samples`) — cùng ý tưởng lấy advantage tương đối trong một nhóm sample thay vì dùng critic. Nên phần lớn giá trị của GRPO đã được cover bởi Experiment 3. Thêm GRPO đúng chuẩn là mở rộng hợp lý và rẻ.

> [!question] "DPO có phải RL không? Nó không có rollout."
> Về hình thức thì không — DPO là loss phân loại có giám sát trên cặp preference. Nhưng nó **tương đương** với việc tối ưu cùng một mục tiêu KL-regularized reward của RLHF, chỉ là đã giải nghiệm dạng đóng nên bỏ được vòng rollout. Em ghi rõ nó vào nhóm "RL rẻ, không rollout" cùng RAML thay vì gọi chung chung là RL. Tiền lệ gần nhất: **SignDPO 2026** — DPO trên skeleton.

> [!question] "8 thuật toán có phải quá nhiều, làm hời hợt không?"
> Bảng ưu tiên chính là câu trả lời. **Bắt buộc** chỉ có 2 experiment (1 và 9). 6 thuật toán còn lại phần lớn là **ablation của nhau** chứ không phải 8 hướng độc lập — A2C là PPO bỏ clip, REINFORCE là SCST bỏ baseline, MRT là SCST multi-sample tổng quát.

---

## Slide 08 — Reward Design

### 🎙️ Kịch bản

"Toàn bộ reward nằm trong một công thức, và mỗi trọng số đặt về 0 là tắt thành phần đó."

```
R = w_bleu·BLEU − w_rep·rep_penalty − w_len·len_penalty  (+ w_bert·BERTScore)
```

"Điểm quan trọng về thiết kế: công thức này **chính là hạ tầng của reward ablation** ở Experiment 9 — không cần viết thêm code, chỉ đổi config."

**Ba thành phần, mỗi cái có một quyết định kỹ thuật đằng sau:**

1. **sentence-BLEU** với `effective_order=True` (§G.25). Không có cờ này, câu ngắn hơn 4-gram — **rất phổ biến ở PHOENIX** — bị tính reward = 0 **sai lệch**, model sẽ học rằng câu ngắn luôn tệ.
2. **repetition_penalty** trên **tri-gram** (§G.26). n=2 phạt nhầm cụm chức năng hợp lệ; n≥4 phát hiện quá chậm trên câu ngắn.
3. **length_penalty** = `(r−h)/r` chặn `[0,1]` (§G.27). Chặn để **cùng thang đo** với BLEU và rep — nhờ vậy ba trọng số mới so sánh được với nhau. Lý do tồn tại: brevity penalty của BLEU nằm trong log và bị các `p_n=0` nuốt mất tín hiệu.

**Chọn metric nào làm reward:** BLEU trong vòng lặp train vì rất rẻ. BERTScore và COMET chỉ ở đánh giá cuối.

> [!important] Nguyên tắc bất di bất dịch
> Mọi reward dùng ở dạng **advantage so với baseline**: `R(sample) − R(greedy)`, **không bao giờ** dùng reward tuyệt đối. Lý do: reward tuyệt đối trộn lẫn "câu này vốn dễ" và "policy này tốt". Câu dễ luôn cho reward cao dù policy chẳng học được gì.

### 🔬 §D — Metric làm reward

| Metric | Đặc điểm | Chi phí | Rủi ro hack |
|---|---|---|---|
| **BLEU** | n-gram precision + brevity penalty (sacrebleu) | Rất thấp | Lặp n-gram, câu vừa-đủ-dài → lý do có rep/len penalty |
| ROUGE | recall-oriented | Thấp | Câu dài dàn trải |
| METEOR | alignment + synonym | Trung bình | Ít hơn BLEU |
| CIDEr | TF-IDF n-gram đa reference — kém hợp SLT (chỉ 1 ref/câu) | Trung bình | Trung bình |
| BERTScore | embedding similarity, bắt synonym | Cao (forward BERT) | Thấp |
| COMET | trained scorer, SOTA correlation | Cao nhất | Thấp nhất nhưng **hộp đen** |

**Khuyến nghị:** sentence-BLEU trong vòng lặp train (rẻ); BERTScore/COMET chỉ ở đánh giá cuối — trừ Reward 5 thử riêng trên subset 25%.

### 🔬 §E — 10 biến thể reward

| # | Reward | Cấu hình | Trạng thái |
|---|---|---|---|
| Reward 1 | BLEU thuần | `w_rep=0, w_len=0` | ✅ |
| Reward 2 | BLEU − rep penalty | `w_rep=0.5` | ✅ **mặc định** |
| Reward 3 | BLEU − length penalty | `w_len>0` | ✅ (mặc định tắt) |
| Reward 4 | + semantic similarity | `w_sem·cos_sim(embed)` | ❌ cần sentence-embedding (LaBSE) |
| Reward 5 | + BERTScore | `w_bert>0` | ✅ đã cài `bertscore_reward()` (mặc định tắt — §G.38) |
| Reward 6 | + COMET | `w_comet>0` | ❌ chi phí cao nhất, chỉ proof-of-concept 25% |
| Reward 7 | Human preference | reward model từ người | ❌ ngoài roadmap (§C.8) |
| Reward 8 | Multi-objective | scalarization nhiều thành phần | ✅ chính là công thức tổng quát |
| Reward 9 | Reward shaping | incremental-BLEU potential-based (Ng 1999) | ❌ ghi chú trong `train_ppo.py` |
| Reward 10 | Curriculum reward | `w_rep/w_len` tăng dần theo epoch | ✅ `rl_curriculum_epochs` |

**Vì sao mặc định `w_bleu=1, w_rep=0.5, w_len=0`** (§G.24): `w_rep` bật vì **lặp là failure mode quan sát thật**; `w_len` tắt vì **brevity penalty của BLEU đã xử lý một phần** — bật cả hai là phạt chồng, nên nó được bật riêng ở Experiment 9.

> [!warning] Bẫy khi thêm BERTScore vào reward
> `reward_bert_weight = 0.0` mặc định **dù đã cài xong** `bertscore_reward()` (§G.38). Lý do thuần kỹ thuật: một forward BERT cho mỗi sample, không batch hoá, làm chậm **mọi** run. Chỉ bật thủ công ở Experiment 9 trên subset nhỏ. Nhớ §G.41 — vòng reward không nằm trong AMP scope.

### ❓ Câu hỏi dự kiến

> [!question] "Sao không dùng thẳng BERTScore làm reward? Nó tương quan với đánh giá người tốt hơn."
> Đúng về chất lượng, sai về chi phí. RL cần **hàng chục nghìn** lần tính reward mỗi epoch; BLEU tốn micro-giây, BERTScore tốn một forward pass GPU. Với ~30 GPU-h/tuần, dùng BERTScore trong reward loop đồng nghĩa **không chạy được experiment nào khác**. Nên nó ở tầng đánh giá cuối, và có biến thể riêng (Reward 5) chạy trên subset 25%.

> [!question] "Trọng số 0.5 lấy ở đâu ra?"
> Là điểm khởi đầu có lý do (cùng thang đo với BLEU nên nửa trọng số là điểm trung dung), **chưa phải kết luận**. Experiment 9 quét 8 tổ hợp để trả lời chính xác câu hỏi này — và đó là **bảng trung tâm của chương RL**.

> [!question] "Reward đa mục tiêu bằng cách cộng có vấn đề gì không?"
> Có — scalarization tuyến tính chỉ tìm được các điểm trên **bao lồi** của Pareto front, bỏ sót phần lõm. Phương án đúng hơn về lý thuyết là multi-objective RL thật (Pareto-based). Nhưng với 3 thành phần cùng thang đo và ngân sách này, scalarization + ablation có hệ thống là đánh đổi hợp lý — và Experiment 9 chính là cách kiểm tra bao lồi đó có đủ hay không.

---

## Slide 09 — RL Beyond the Decoder

### 🎙️ Kịch bản

"Đây là phần em coi là đóng góp riêng."

"Cho tới giờ RL mới fine-tune decoder — chuyện đã có tiền lệ, SCST là chuẩn trong image captioning từ 2017. **Nhưng nhìn lại slide 04**: ba trong các ô đỏ không nằm ở decoder mà ở tầng thị giác. Điểm chung: chúng đang được quyết định bằng **heuristic cố định**, và đều là **quyết định rời rạc**. Mà quyết định rời rạc thì không có gradient — đó chính là dấu hiệu của một bài toán RL."

**Sáu MDP** (click từng ô xem action/reward cụ thể): F.6 frame selection · F.9 adaptive sampling · F.8 landmark selection · F.5 decode policy · F.13 beam policy · F.18 curriculum.

"11/13 ý tưởng đã có code. Nhưng em phải nói thẳng hai điều."

> [!important] Hai điều phải nói thẳng
> **① Trung thực về hiện thực.** F.14 sign segmentation làm bằng **CTC forced-alignment**, F.16 memory selection làm bằng **kiến trúc** (Perceiver IO) — không phải vòng RL đúng nghĩa đen. Ghi rõ trong báo cáo thay vì gọi tất cả là "RL". Lý do chọn vậy: với memory selection, gradient trực tiếp qua cross-attention hiệu quả hơn hẳn policy rời rạc.
>
> **② Giới hạn của Experiment 8.** Frame bị loại được **zero-hoá, KHÔNG xoá khỏi chuỗi** — encoder vẫn xử lý đủ T frame. Nghĩa là đo được "frame nào mang tín hiệu quan trọng" (soft-masking), **chưa** chứng minh được giảm compute. Compute-saving thật cần re-index chuỗi — hướng mở rộng §H.7 mục 7, không phải kết quả đợt này. H5 đã được điều chỉnh cho khớp.

### 🔬 §F — 13 ý tưởng RL nâng cao (lọc từ 20)

Chứng minh RL là khung tối ưu cho **mọi quyết định rời rạc trong pipeline CV+NLP**, không chỉ fine-tune decoder.

| # | Ý tưởng | MDP tóm tắt | Đã code | Ưu tiên |
|---|---|---|---|---|
| F.1 | SCST | (§C.2) | ✅ `train_scst.py` | — |
| F.2 | Multi-sample SCST | sample N câu/input, advantage theo nhóm | ✅ `rl_n_samples` | Cao — Exp 3 |
| F.3 | PPO fine-tuning | (§C.4) | ✅ `train_ppo.py` | Cao nhất |
| F.4 | Actor-Critic Transformer | ValueHead trên hidden decoder | ✅ `slt_transformer.py::ValueHead` | Trung bình |
| F.5 | RL for decoding strategy | action = chọn temperature per-input; reward = BLEU | ✅ `train_decode_policy.py` | Trung bình |
| F.6 | RL for frame selection | Bernoulli giữ/bỏ frame; reward = BLEU(model đóng băng) − phạt frame | ✅ `train_selection_policy.py` | Cao — Exp 8 |
| F.7 | RL for keyframe extraction | biến thể F.6 | ❌ trùng cơ chế F.6, không tách riêng | — |
| F.8 | RL for landmark selection | chọn nhóm body/tay-trái/tay-phải (occlusion) | ✅ `train_selection_policy.py` — 1 quyết định/câu | Trung bình |
| F.9 | RL for adaptive temporal sampling | tốc độ lấy mẫu thay đổi thay vì linspace | ✅ `train_selection_policy.py` (frame/adaptive) | Cao |
| F.13 | RL for beam search policy | chọn nhánh beam theo reward cuối | ✅ `train_mrt.py` (`mrt_candidate_source="beam"`) | Thấp-TB |
| F.14 | RL for sign segmentation | ranh giới gloss | ⚠️ hiện thực bằng **CTC forced-alignment**, không phải vòng RL riêng | Trung bình |
| F.16 | RL for memory selection | chọn frame giữ trong attention | ⚠️ hiện thực **kiến trúc** (P6 Perceiver — gradient trực tiếp hiệu quả hơn) | Trung bình |
| F.18 | RL for curriculum learning | thứ tự mẫu ngắn→dài | ✅ `LengthCurriculumSampler` | Trung bình |

**Baseline đối chứng bắt buộc** cho F.5/F.6/F.8/F.9: `scripts/eval_baselines.py` (§E của phần thực nghiệm, slide 10).

**7 ý tưởng đã loại** — vẫn hợp lệ về ý tưởng, chỉ thiếu tiền đề:

| Ý tưởng | Vì sao loại |
|---|---|
| #10 Augmentation policy · #11 Hyperparameter RL | Vòng lặp ngoài = train lại nhiều lần → vượt ngân sách GPU |
| #12 Modality selection · #19 Adaptive fusion | Cần RGB/multimodal — đã loại ở slide 02 |
| #15 Attention routing | Action/reward chưa đặc tả cụ thể |
| #17 Active learning | Cần ngân sách gán nhãn |
| #20 Prompt optimization | Cần LLM decoder |

### ❓ Câu hỏi dự kiến

> [!question] "Frame selection với model đóng băng — sao không train chung cả hai?"
> Vì như vậy sẽ không tách được nguyên nhân: BLEU thay đổi là do policy chọn frame tốt hơn, hay do model SLT thích nghi với việc mất frame? Đóng băng model là cách cô lập biến. Train chung là bước tiếp theo hợp lý, nhưng phải sau khi có số của phiên bản đóng băng để so.

> [!question] "F.8 chỉ 1 quyết định/câu thì có quá thô không?"
> Có, lựa chọn có chủ đích để giữ không gian action nhỏ (3 nhóm → 8 tổ hợp). Phiên bản per-frame tự nhiên hơn về hiện tượng (occlusion xảy ra theo đoạn thời gian), nhưng làm không gian action nổ ra trong khi reward vẫn chỉ ở cuối câu → credit assignment gần như bất khả thi với 7K câu.

> [!question] "Vậy đóng góp chính là kết quả số hay khung tư duy?"
> Là **khung tư duy có kiểm chứng thực nghiệm**: hệ thống hoá 10 thuật toán kèm bảng ưu tiên theo compute thực tế · so sánh 10 biến thể reward · và chứng minh — **hoặc bác bỏ có cơ sở, so với counter-baseline không-học** — rằng RL là khung tối ưu xuyên pipeline. Câu "hoặc bác bỏ có cơ sở" là có chủ đích.

---

## Slide 10 — Experimental Design

### 🎙️ Kịch bản

"13 experiment, nhưng chỉ **2 cái bắt buộc**."

- **Experiment 1 — CE vs SCST**: cùng checkpoint gốc, cả 3 subset. **Câu chuyện chính.**
- **Experiment 9 — Reward ablation**: 8 tổ hợp bật/tắt trọng số → **bảng trung tâm của chương RL**.

"Nhóm ưu tiên cao: Exp 7 (PPO vs SCST) · 11 (data size) · 4 (6 encoder) · 3 (multi-sample)."

"Nhưng phần em nhấn nhất không phải danh sách experiment, mà là **baseline sàn**."

1. **`trivial`** — `base_empty` và `base_most_frequent`. PHOENIX là bản tin thời tiết, **lặp rất nhiều** — câu phổ biến nhất tự nó đạt BLEU đáng kể **mà không cần model nào**. Không vượt sàn này thì BLEU vô nghĩa.
2. **`selection`** — `base_frames_full/random/uniform` cùng `keep_ratio` và cùng soft-mask (import `_apply_frame_mask` để không lệch), cộng `base_drop_body/lhand/rhand`. Đối chứng Experiment 8 và F.8.
3. **`temp`** — `base_fixed_temp_{0.7, 1.0, 1.3}` cùng seed. Đối chứng F.5.

> [!important] Câu chốt của slide 10
> "Nguyên tắc: **mọi con số phải đọc tương đối so với sàn, không bao giờ đọc tuyệt đối**. Một BLEU 15 nghe có vẻ ổn cho tới khi biết `base_most_frequent` đạt 12."
> *(⚠️ con số 15/12 hiện là **ví dụ giả định** — thay bằng số thật sau khi chạy, xem §K.)*

"Nếu hết quota GPU giữa chừng, ưu tiên tuyệt đối là Experiment 1, 9, 11 — đủ chứng minh luận điểm chính."

### 🔬 Phân tích sâu — 5 giả thuyết phát biểu TRƯỚC

Nói ra được thì rất thuyết phục, vì chứng minh không phải kết luận ngược từ kết quả:

- **H1** (Exp 1, 11): CE+SCST > CE-only trên test, cải thiện **lớn hơn ở subset nhỏ** (exposure bias nặng hơn).
- **H2** (Exp 2, 9): Reward BLEU thuần → tỉ lệ lặp tri-gram cao hơn đáng kể so với có `repetition_penalty` (reward hacking).
- **H3** *(trung lập có chủ đích)* (Exp 7): PPO **không nhất thiết** vượt SCST — PPO thiết kế cho reward dày đặc, SLT reward chỉ ở cuối episode.
- **H4** (Exp 4): ST-GCN đạt BLEU tương đương/gần Transformer với ít tham số hơn đáng kể (6.33M vs 8.26M) — graph inductive bias hợp data nhỏ.
- **H5** *(đã thu hẹp phạm vi)* (Exp 8): Frame-selection policy giữ/cải thiện BLEU khi giảm `keep_frac`, **và phải thắng `base_frames_uniform` cùng keep_ratio**. Implementation soft-mask nên H5 chỉ kiểm chứng "tín hiệu frame quan trọng", **chưa** kiểm chứng compute-saving.

**Biến nhiễu phải khai báo ở Exp 4:** ST-GCN dùng input **150-d hiệu dụng** (bỏ kênh visibility để 75 khớp cùng số kênh) so với 183-d của Transformer (§G.39).

**5 câu hỏi nghiên cứu (RQ):** RQ1 SCST có cải thiện nhất quán qua 25/50/100%? · RQ2 thiết kế reward ảnh hưởng chất lượng thực thế nào? · RQ3 PPO có đáng độ phức tạp ở bài toán reward-thưa? · RQ4 kiến trúc encoder ảnh hưởng khả năng "hưởng lợi" từ RL? · RQ5 RL áp được cho quyết định rời rạc tầng thị giác không?

Bảng 13 experiment đầy đủ ở §E.

### ❓ Câu hỏi dự kiến

> [!question] "Chạy 1 seed mà kết luận được à?"
> Không, và em không giấu. RL variance cao, nên với hai cấu hình gần nhau — điển hình PPO vs SCST — kết luận từ 1 seed là không đủ. Nguyên tắc: chạy **≥2 seed** cho các cặp so sánh sát nhau nếu ngân sách cho phép, và **nói rõ là 1 seed** ở chỗ không đủ, kèm khuyến cáo đọc thận trọng (§H.6).

> [!question] "Sao không chạy full dataset mà lại subset 25/50/100%?"
> Ba lý do: (1) ngân sách ~30 GPU-h/tuần mà RL cần chạy nhiều lần; (2) subset **không phải hạn chế bị ép mà là một trục thí nghiệm miễn phí** — chính là Experiment 11; (3) ba mức subset (25/50/100%) dùng chung seed cố định, nên so sánh giữa các mức được kiểm soát (§G.31); 50/100% chạy thêm khi có quota.

> [!question] "Nếu RL không cải thiện gì thì sao?"
> Thì đó là kết quả và em báo cáo đúng như vậy kèm phân tích nguyên nhân — vẫn đúng tinh thần **Kiegeland & Kreutzer 2021**. Đề tài được thiết kế để **đo** chứ không phải để **thắng**. *(Code đã sửa để luôn lưu `last_rl.pt`, nên trường hợp "RL kém hơn CE" vẫn vào được bảng so sánh thay vì biến mất — §J.3.)*

---

## Slide 11 — Metrics and Reading Results

### 🎙️ Kịch bản

"Slide này về kỷ luật đọc số — cách để không tự lừa mình."

**Nhóm 1 — chất lượng dịch:** BLEU-1..4 (báo cáo **cả 4 bậc**, PHOENIX ~10 từ/câu nên BLEU-4 có thể ≈ 0 trong khi BLEU-1/2 vẫn có nghĩa) · ROUGE-L · METEOR · BERTScore · COMET.

**Nhóm 2 — hiệu năng hệ thống:** Latency (batch 1 real-time, batch 16 throughput, warm-up GPU trước khi đo) · FPS · Memory · Parameters.

**Ba tín hiệu phát hiện reward hacking:**

1. **BLEU tăng nhưng ROUGE giảm** → *length hacking*, model rút ngắn câu để giữ precision.
2. **Tỉ lệ lặp tri-gram tăng** → lặp cụm ngắn vẫn đẩy BLEU ở trường hợp biên.
3. **`avg_advantage` sụp về ≈0 sớm** → policy tự "an toàn hoá", ngừng exploration. Cân nhắc `rl_entropy_coef ≈ 1e-3`.

**Bốn nguyên tắc đọc số:** `eval_every=1` · **không** dùng BLEU-4 làm early-stop duy nhất · báo cáo trên **test** (dev chỉ để early-stop/chọn checkpoint) · ≥2 seed khi so hai cấu hình gần nhau.

> [!warning] Một phát hiện ngược kỳ vọng, phải nói ra
> Đo tham số thật cho thấy **GCN và Graph Transformer KHÔNG nhẹ hơn Transformer** (9.38M và 9.48M so với 8.26M) — nguyên nhân là `joint_pool = Linear(75×hidden, d_model)`. **Chỉ ST-GCN mới nhẹ thật** (encoder 1.35M). Đây là kỳ vọng ban đầu bị dữ liệu bác bỏ, và em báo cáo đúng như vậy. Không phải bug — muốn P2/P4 nhẹ thì giảm `hidden` hoặc thay `joint_pool` bằng mean-pool.

### 🔬 Phân tích sâu

**Cặp BLEU/ROUGE là công cụ chẩn đoán, không phải hai metric song song:**

| BLEU | ROUGE | Chẩn đoán |
|---|---|---|
| ↑ | ↑ | Cải thiện thật |
| ↑ | ↓ | **Length hacking** — câu ngắn lại để giữ precision |
| ↓ | ↑ | Câu dài dàn trải, thêm nội dung thừa |
| ↓ | ↓ | Policy collapse hoặc lr quá cao |

**Về Experiment 15 (latency):** so `xe` vs `rl`, **kỳ vọng không đổi** — RL chỉ thay trọng số, không đổi kiến trúc hay số bước decode. Khác biệt đáng kể = **bug**, không phải finding. Đây là *sanity check* trá hình thành experiment.

**Cách trình bày bảng chính trong luận văn:** mỗi hàng = 1 cấu hình (CE-only / CE+SCST / CE+PPO / …), cột = BLEU1-4, ROUGE-L, METEOR, BERTScore, tham số, latency — đo trên **test set**. Và **hàng đầu tiên phải là baseline sàn**, để người đọc thấy ngay mọi con số nên đọc tương đối so với cái gì.

Bảng metric chi tiết (định nghĩa, công cụ, trạng thái code) ở §F.

### ❓ Câu hỏi dự kiến

> [!question] "BLEU có phải metric tốt cho SLT không? Nhiều người chê BLEU."
> Chê đúng — BLEU không nắm nghĩa, phạt oan diễn đạt tương đương, và với câu ngắn thì BLEU-4 rất nhiễu. Nhưng ba lý do vẫn dùng: (1) nó là **metric chuẩn của toàn bộ literature SLT**, không dùng thì không so được với Camgoz 2018/2020; (2) nó **cực rẻ**, mà RL cần tính reward hàng chục nghìn lần; (3) em **không dùng nó một mình**. Nói cách khác: BLEU là *reward rẻ để tối ưu* và *một trong nhiều metric để đánh giá* — hai vai trò khác nhau.

> [!question] "Không có human evaluation thì kết luận có đáng tin không?"
> Đây là hạn chế ghi rõ ở §H.6. Giảm nhẹ bằng cách báo cáo nhiều metric có tính chất khác nhau (precision/recall/embedding-based), cộng các tín hiệu chẩn đoán reward hacking. Nhưng nó không thay thế được human eval, và em không tuyên bố là thay thế được.

---

## Slide 12 — Limitations & Conclusion

### 🎙️ Kịch bản

**Đóng góp:** ① RL xuyên pipeline · ② hệ thống hoá thực dụng theo compute thật · ③ định vị so với 3 công trình gần nhất.

**Hạn chế — nói trước khi bị hỏi:** data nhỏ · không human evaluation · pose-only bỏ face landmark · không RLHF thật · phần lớn 1 seed · gloss-free là chính nên so với literature gloss-based không hoàn toàn công bằng.

**Hướng phát triển:** 7 hướng (§H.7), quan trọng nhất là **frame selection re-index chuỗi thật** để đo compute-saving — nối tiếp trực tiếp H5.

**Ba tiền lệ gần nhất** (số liệu đã xác minh, §L.1):

| Công trình | Gần ở điểm nào | Khác ở điểm nào |
|---|---|---|
| **Panaro 2020** (RIT MS thesis) | SCST + PPO trên **chính PHOENIX-2014T** | ⚠️ Không peer-reviewed; không mở RL ra ngoài decoder |
| **RVLF 2025** (arXiv:2512.07273) | **GRPO đầu tiên cho SLT**, reward BLEU+ROUGE; **+1.11 BLEU-4 trên PHOENIX-2014T** | Quy mô LLM + DINOv2; không mở ra tầng thị giác |
| **SignDPO 2026** (arXiv:2604.18034) | **DPO trên skeleton** — trùng modality | ⚠️ Đánh giá CSL-Daily/How2Sign/OpenASL, **KHÔNG có PHOENIX-2014T** → số không so trực tiếp được |

> [!important] Câu kết
> "Nguyên tắc xuyên suốt là **không tách rời hai hướng**: mọi lựa chọn CV đều đi kèm câu hỏi *'RL khai thác được gì từ đây?'*, và mọi thiết kế RL đều bắt rễ từ một pipeline thị giác cụ thể. Đó là lý do đây là đồ án liên môn chứ không phải hai chương ghép cạnh nhau."

### ❓ Câu hỏi dự kiến

> [!question] "Nếu chỉ được chọn một đóng góp để bảo vệ thì là gì?"
> Là **khung ánh xạ "quyết định rời rạc trong pipeline → MDP"**, cùng bộ counter-baseline nghiêm túc để kiểm chứng nó. Con số BLEU phụ thuộc dataset và ngân sách; khung tư duy thì chuyển được sang bài toán video-to-text khác.

> [!question] "Đề tài này khác gì với việc áp SCST vào một bài toán mới?"
> Nếu chỉ có Experiment 1 thì đúng là vậy — và em thừa nhận phần đó không mới. Cái mới ở ba chỗ: (1) nhóm F — RL cho quyết định tầng thị giác, có counter-baseline không-học; (2) so sánh 10 biến thể reward có hệ thống; (3) đặt tất cả trong ràng buộc **data nhỏ + compute nhỏ**. Cụ thể với 3 tiền lệ: **cả ba đều chỉ áp RL vào mục tiêu sinh câu**, không ai mở ra tầng thị giác.

> [!question] "P7 (two-stage gloss) đâu, sao không thấy trên slide?"
> P7 là **nhánh đối chứng song song**, không nằm trên đường chính của câu chuyện RL — em để ngoài slide để giữ mạch. Nó là Stage 1 CTC (pose→gloss, đánh giá WER) rồi Stage 2 NMT (gloss→text), dùng để so gloss-based với gloss-free **trong cùng codebase**, tránh so với literature ở điều kiện khác nhau. ⚠️ Trạng thái trung thực: **chưa verify trên dữ liệu thật** — format PHOENIX chuẩn *được biết* có cột `orth` và code raise rõ nếu thiếu, nhưng xác nhận thật chỉ xảy ra khi chạy Kaggle.

---
---

# PHẦN II — TRA CỨU

---

## §A — Pipeline khả thi (P1–P8, lọc từ 16)

> Chỉ giữ pipeline **thực thi được** với tài nguyên thật: data chỉ có pose MediaPipe, không LLM infra, Kaggle T4×2. **Repo hiện tại = P1 + P8 đã fuse sẵn.** Không tách "End-to-end"/"Gloss-free" thành pipeline riêng — đó là thuộc tính của P1, không phải kiến trúc khác.

### A.1. Bảng pipeline khả thi

| # | Pipeline | Trạng thái | Ghi chú thực thi |
|---|---|---|---|
| P1 | **Pose → Transformer → Text** | ✅ đã code, đã chạy | Baseline chính (Camgoz 2018/19, SPOTER) |
| P2 | Pose → GCN → Transformer | ✅ `encoders.py::GCNPoseEncoder` | Tái dùng adjacency từ ST-GCN (Yan 2018) |
| P3 | **Pose → ST-GCN → Text** | ✅ `stgcn_encoder.py` | Encoder nhẹ nhất — Experiment 4 |
| P4 | Pose → Graph Transformer | ✅ `encoders.py::GraphTransformerPoseEncoder` | Attention trên 75 khớp |
| P5 | Pose → TCN → Transformer | ✅ `encoders.py::TCNPoseEncoder` | Conv1D dilated nén thời gian |
| P6 | Pose → Perceiver → Decoder | ✅ `encoders.py::PerceiverPoseEncoder` | Nén T frame vào latent — tuyến tính thay `O(T²)` (Jaegle 2021) |
| P7 | Two-stage: Pose → Gloss (CTC) → NMT → Text | ✅ `main_twostage.py` | ⚠️ **CHƯA verify trên dữ liệu thật** (Camgoz 2018 2-stage) |
| P8 | **RL fine-tuning layer** (SCST/PPO/MRT/RAML/DPO trên P1–P6) | ✅ cả 5 (`main.py --algo`) | RL loop không phụ thuộc encoder — chỉ cần memory `[B,T,D]`. P7 stage 2 vẫn CE-only |

### A.2. Bảng so sánh (tham số ĐO THẬT, `d_model=256`, vocab 3000)

> Decoder+embedding dùng chung ≈ 4.98M — chênh lệch giữa các dòng là riêng encoder. Cột compute là **ước lượng định tính chưa đo wall-clock**.

| # | Tổng / riêng encoder | Compute/epoch (ước lượng) | Vai trò | Rủi ro |
|---|---|---|---|---|
| P1 transformer | **8.26M** / 3.27M | 1x baseline | Baseline mọi experiment | Không |
| P2 gcn | **9.38M** / 4.39M | ~1x (encoder KHÔNG rẻ hơn P1) | Biến thể graph cố định | Thấp |
| P3 stgcn | **6.33M** / 1.35M | ~0.7-0.8x (nhẹ nhất) | **Experiment 4** | Thấp |
| P4 graph_transformer | **9.48M** / 4.49M | ~1.1-1.3x | Mở rộng Exp 4 | ⚠️ **OOM risk trên T4** (xem §J) |
| P5 tcn | **7.47M** / 2.48M | ~1x | Phụ trợ F.9 | Thấp |
| P6 perceiver | **9.59M** / 4.61M | ~0.8-1x (lợi khi T lớn) | Long sequence, F.16 | TB (tune latent) |
| P7 | 2 model riêng (CTC-enc + NMT ~0.3M) | ~1.5-2x | Đối chứng gloss-based vs gloss-free | TB (chất lượng gloss chưa có số) |
| P8 | +ValueHead ~0.03M (PPO) | +30-50% (SCST/MRT/RAML) · +50-100% (PPO) · +100% (DPO) | **Trục chính đề tài** | Thấp SCST |

> ⚠️ **Đảo ngược kỳ vọng**: GCN/GraphTransformer **KHÔNG nhẹ hơn** Transformer (do `joint_pool = Linear(75×hidden, d_model)`) — chỉ ST-GCN nhẹ thật. Muốn P2/P4 nhẹ: giảm `hidden` hoặc thay `joint_pool` bằng mean-pool (hướng tinh chỉnh, không phải bug).

### A.3. Ma trận lựa chọn theo mục tiêu

| Mục tiêu | Chọn |
|---|---|
| Baseline nhanh | P1 |
| Encoder rẻ nhất | P3 / P5 |
| Experiment 4 | P1 vs P3 (bắt buộc), +P4 nếu kịp |
| Long sequence | P6 |
| Chứng minh RL (câu chuyện chính) | P8 trên P1/P3 |
| Gloss-based vs gloss-free | P7 |
| Rủi ro cao nhất | P7 (gloss chưa kiểm chứng), P6 (chưa tiền lệ SLT) |

### A.4. Thứ tự CHẠY trên Kaggle

P1+P8(SCST) Cell 4 → P3 Cell 5 → P8(PPO) Cell 6 → P2/P4/P5/P6 Cell 6b → MRT/RAML/DPO Cell 6c → P7 Cell 6f. Chạy `aggregate_results.py` (Cell 9) bất kỳ lúc nào.

### A.5. Pipeline đã loại

| Nhóm chặn | Pipeline | Lý do |
|---|---|---|
| Cần RGB thô | RGB→CNN/I3D→Transformer, RGB+Pose Fusion, VideoMAE→Decoder | Không có cache RGB feature; trích on-the-fly phá ngân sách GPU (§G.34); xây pipeline RGB = 1 hạng mục lớn ngoài scope |
| Cần optical flow | RGB+Pose+Flow Multimodal | Phụ thuộc RGB + chi phí RAFT/TV-L1 |
| Cần LLM/LoRA infra | Pose+LLM Decoder, Q-Former→LLM, BLIP2-style | Chưa verify: LLM tiếng Đức scale nhỏ, licence, LoRA trên T4 16GB |

Các hướng này giữ ở §H.7 — bị loại vì **chưa verify tiền đề**, không phải vì kém giá trị.

---

## §B — Khảo sát kiến trúc model (pose-only)

### B.1–B.3. Bảng kiến trúc khả thi

| Kiến trúc | Cơ chế 1 dòng | Vai trò | Trong code |
|---|---|---|---|
| RNN/LSTM/GRU | hồi quy có cổng | baseline lịch sử (Camgoz 2018 dùng GRU) — không khuyến nghị làm baseline chính | ❌ |
| **TCN** | Conv1D dilated theo thời gian, không cần graph, song song tốt | P5 — temporal aggregator rẻ | ✅ `encoders.py::TCNPoseEncoder` |
| **Transformer** | self-attention `O(T²)` chuẩn | P1 — **baseline chính** | ✅ `slt_transformer.py` |
| **GCN** | graph conv trên đồ thị khớp cố định | P2 | ✅ `encoders.py::GCNPoseEncoder` |
| **ST-GCN** (Yan 2018) | graph conv không gian + Conv1D thời gian, xếp block | P3 — **nhẹ nhất** (encoder 1.35M đo thật) | ✅ `stgcn_encoder.py` |
| MS-G3D | multi-scale disentangled + G3D | nâng cấp nếu ST-GCN chưa đủ | ❌ hướng mở |
| **Graph Transformer** | self-attention trên 75 khớp | P4 | ✅ `encoders.py::GraphTransformerPoseEncoder` |
| **Perceiver IO** (Jaegle 2021) | cross-attention nén T frame vào latent cố định — **tuyến tính** | P6 — long sequence, liên quan F.16 | ✅ `encoders.py::PerceiverPoseEncoder` |

### B.4. Kiến trúc đã loại

| Nhóm chặn | Kiến trúc | Lý do |
|---|---|---|
| Cần RGB thô | CNN 2D, ViT, I3D, SlowFast, TimeSformer, VideoMAE/v2, Video Swin, InternVideo | Chỉ có vai trò RGB encoder — vô nghĩa khi không có RGB; InternVideo thêm lý do 6B+ tham số |
| Cần hạ tầng LLM | Q-Former (BLIP-2 bridge) | Tích hợp LLM tự nó đã ngoài scope |
| Rủi ro chưa verify | Mamba/SSM | `mamba-ssm` cần custom CUDA kernel, chưa kiểm chứng compile trên Kaggle T4 |
| Rủi ro chưa verify | Diffusion decoder (Diffusion-LM/DiffuSeq) | Chưa có paper SLT nào dùng — không có tiền lệ về độ khó thực tế |

Nhóm "rủi ro kỹ thuật" vẫn là hướng nghiên cứu mở — chỉ loại khỏi bảng lựa chọn thực thi (tiêu chí: "verify được trước khi cam kết code").

---

## §C — Khảo sát Dataset

> Neo thực tế: pipeline đã dùng **PHOENIX-2014T** (`data/dataset.py` đọc format `name|video|start|end|speaker|orth|translation`). Số liệu *(approx, verify)* cần đối chiếu nguồn gốc trước khi trích dẫn chính thức.

### C.1. Bảng so sánh

| Dataset | Bài toán | SL/đích | Câu (train/dev/test) | Gloss | Translation | Pose sẵn | Access | Phù hợp RL? |
|---|---|---|---|---|---|---|---|---|
| **PHOENIX-2014T** ⭐ | Continuous SLT | DGS/Đức | 8.257 (7.096/519/642) | ✅ | ✅ | ❌ (tự trích — đã có) | CC BY-NC-SA, tải trực tiếp | **Rất phù hợp** — có reference cho reward, nhỏ vừa ngân sách, SOTA gloss-free ~22-27 BLEU để so |
| CSL-Daily | Continuous SLT | CSL/Trung | 20.654 | ✅ | ✅ | ❌ | Cần giảng viên ký thoả thuận — **rủi ro deadline** | Tốt kỹ thuật, kém tiếp cận |
| How2Sign | Continuous SLT | ASL/Anh | ~35K câu, 80h+ *(approx)* | Một phần | ✅ | ✅ 2D/3D | CC BY-NC, tải tự do | Tốt làm **phụ** (Exp 13); BLEU baseline thấp (~8-12) → reward thưa, dùng subset |
| WLASL / MS-ASL / AUTSL | Isolated word | ASL/TSL | 21-38K clip | nhãn từ đơn | ❌ | WLASL/AUTSL ✅ | non-commercial; **rủi ro link rot** | ❌ Không có translation → không tính BLEU reward; chỉ pretrain encoder |

### C.2. Đề xuất

- **PRIMARY: PHOENIX-2014T** — duy nhất vừa có translation reference vừa đủ nhỏ cho nhiều lần chạy RL; không rào cản truy cập; đã tích hợp sẵn.
- **SECONDARY (Exp 13): How2Sign** — ưu tiên hơn CSL-Daily vì tải được ngay + khác hẳn ngôn ngữ ký hiệu (ASL vs DGS) → phép thử generalization mạnh hơn. Subset ~2-5K câu, seed cố định.
- **CSL-Daily**: chỉ theo đuổi nếu xin quyền từ tuần 1-2. Lưu ý BLEU tiếng Trung nhạy cách tách từ.
- WLASL/MS-ASL/AUTSL: loại khỏi phần RL cốt lõi — nêu trong luận văn là đã khảo sát đủ 6 dataset có cơ sở loại trừ.

### C.3. Số liệu cần xác minh trước khi vào luận văn

Thời lượng PHOENIX (10.5h vs 11h) · vocab gloss (1.066 vs 1.085) · CSL-Daily ~23h + BLEU ~45.6 (nghi do cách tính tiếng Trung) · split How2Sign (31.128/1.741/2.322 — nguồn thứ cấp) · số mẫu AUTSL (38.336 vs 36.302).

---

## §D — Pipeline xử lý ảnh & video (CV)

> Phục vụ môn Xử lý ảnh & video. Preprocessing cấp pixel (resize/denoise/illumination) **ngoài phạm vi** — pipeline là pose-only, MediaPipe tự xử lý nội bộ.

### D.2. Pose extraction — so sánh công cụ

| Công cụ | Đặc điểm | Trạng thái |
|---|---|---|
| **MediaPipe Holistic** | Nhẹ, CPU real-time; 33 body + 21×2 hand (bỏ face → 183-d); đôi khi mất track tay khi chuyển động nhanh | ✅ **Đang dùng** |
| OpenPose | PAF bottom-up, ổn định hơn với occlusion, chậm hơn | ❌ — ứng viên ablation "chất lượng pose ảnh hưởng BLEU" |
| MMPose / HRNet / ViTPose | Top-down SOTA, chậm/tốn hơn nhiều | ❌ — chỉ cân nhắc nếu MediaPipe track kém |

Đề xuất thực nghiệm phụ (ngoài 13 experiment chính): trích lại pose bằng HRNet cho subset 25%, so BLEU — bằng chứng "pose estimator là nút thắt" nếu chênh lệch lớn.

### D.3. Temporal processing

| Kỹ thuật | Trạng thái | Điểm chạm RL |
|---|---|---|
| Frame sampling | ✅ nhưng **uniform** (`np.linspace` khi > `max_frames=300`) | Chính là mục tiêu F.9 — ✅ đã code |
| Keyframe extraction | ❌ | F.6/F.7 — Experiment 8 |
| Motion estimation trên pose | ❌ | Đạo hàm hữu hạn `pose[t]−pose[t−1]` (183d→366d) — **gần miễn phí**, thay thế optical flow |

### D.5. Landmark processing — nhóm cải tiến chi phí thấp, lợi ích cao nhất

| Kỹ thuật | Trạng thái | Ghi chú |
|---|---|---|
| Normalization theo cơ thể | ⚠️ Một phần — MediaPipe chuẩn hoá `[0,1]` theo khung hình, **chưa** theo vai/hông | Chuẩn hoá "signing space" (chia khoảng cách vai, trừ tâm hông) — kỹ thuật SPOTER, giảm signer variation |
| Smoothing (chống jitter) | ❌ | Moving average / Savitzky-Golay trên chuỗi toạ độ — rẻ, hậu xử lý `.npz` |
| Interpolation frame lỗi | ✅ | `np.interp` theo từng segment thay zero-fill (§G.40) |
| Missing point recovery | ✅ | Cùng cơ chế, độc lập body/tay trái/tay phải; segment thiếu toàn video vẫn giữ 0 |

### D.6. Data augmentation (áp dụng CỐ ĐỊNH theo config — RL-cho-augmentation đã loại)

Thứ tự ưu tiên (rẻ → đắt): **Landmark noise** (Gaussian nhỏ) → **Rotation/Scaling** (ma trận 2D) → **Frame dropping** → **Temporal jitter** → **Flip ngang** (⚠️ phải swap index tay trái↔phải, không chỉ lật toạ độ) → Mixup/Cutmix (khó align sequence, ưu tiên thấp nhất). Tất cả hiện ❌ chưa bật mặc định.

### D.7. Thứ tự triển khai phần CV

1. Landmark processing (D.5) → 2. Motion features (D.3) → 3. Augmentation rẻ (D.6) → 4. Adaptive sampling RL (F.9 — làm sau khi baseline CE+SCST ổn định, không đổi input pipeline giữa chừng khi đang so sánh).

---

## §E — Thiết kế thực nghiệm (13 experiment, lọc từ 15)

> Exp 5/6 (cần RGB) **đã loại** — giữ nguyên số thứ tự gốc để tham chiếu chéo không lệch. Ngân sách: ~30 GPU-h/tuần T4×2 (25%→~3.5h, 50%→~7h, 100%→~14h cho XE+RL).

### E.1. Bảng ưu tiên & trạng thái

| # | Experiment | Ưu tiên | Trạng thái code |
|---|---|---|---|
| 1 | CE vs SCST | **Bắt buộc** (câu chuyện chính) | ✅ `--phase xe` vs `--phase all` |
| 9 | Reward ablation | **Bắt buộc** | ✅ đổi config weight |
| 3 | Single vs Multi-sample | Cao | ✅ `rl_n_samples` |
| 7 | PPO vs SCST (+MRT/RAML/DPO) | Cao | ✅ `train_ppo/mrt/raml/dpo.py` |
| 4 | Transformer vs GCN (6 kiến trúc) | Cao | ✅ `models/encoders.py` |
| 11 | Data size ablation | Cao | ✅ `subset_ratios` sẵn |
| 2 | BLEU vs BLEU+Penalty | TB (con của #9) | ✅ |
| 8 | RL frame selection | TB | ✅ `train_selection_policy.py` |
| 10 | Decoder ablation | TB | ✅ đổi `n_dec_layers`/`d_model` |
| 15 | Inference latency | TB | ✅ `scripts/measure_latency.py` |
| 12 | Generalization gap | TB | ✅ `test_results.json` qua `--phase eval` |
| 14 | Few-shot | Thấp | ✅ mở rộng `subset_ratios` |
| 13 | Cross-dataset transfer (How2Sign) | Thấp | ❌ cần loader riêng — ngoài đợt này |
| 5, 6 | ~~RGB vs Pose / Fusion~~ | **ĐÃ LOẠI** | Cần pipeline RGB không có (§A.5) |

### E.2. Baseline bắt buộc — đọc mọi con số TƯƠNG ĐỐI so với sàn

> `scripts/eval_baselines.py` sinh 3 nhóm base, merge vào `test_results.json` đúng định dạng `aggregate_results.py` quét — tự xuất hiện trong `comparison_table`.

| `--kind` | Baseline | Đối chứng cho | Vì sao bắt buộc |
|---|---|---|---|
| `trivial` | `base_empty`, `base_most_frequent` | **Mọi** model | PHOENIX lặp nhiều — câu train phổ biến nhất tự đạt BLEU đáng kể **không cần model**; không vượt sàn = BLEU vô nghĩa |
| `selection` | `base_frames_full/random/uniform` (cùng `keep_ratio`, cùng soft-mask — import `_apply_frame_mask`), `base_drop_body/lhand/rhand` | Exp 8 (F.6/F.9), F.8 | Policy phải thắng **uniform-stride** (heuristic mạnh nhất không cần học) mới là "học được"; thắng mỗi full-frame chưa đủ |
| `temp` | `base_fixed_temp_{0.7,1.0,1.3}` (cùng seed) | Decode policy F.5 | Policy per-input phải thắng fixed-temp tốt nhất mới hơn được 1 lần grid-search |

```bash
python scripts/eval_baselines.py --kind trivial   --subset 0.25
python scripts/eval_baselines.py --kind selection --subset 0.25 --encoder transformer --ckpt <run>/best_xe.pt
python scripts/eval_baselines.py --kind temp      --subset 0.25 --encoder transformer --ckpt <run>/best_xe.pt
```

### E.3. Chi tiết từng experiment

**Exp 1 — CE vs SCST**: `best_xe.pt` vs `best_rl.pt` cùng checkpoint gốc, cả 3 subset. Metric BLEU1-4/ROUGE/METEOR trên **test** + khoảng cách tới sàn. *Kỳ vọng:* RL tăng BLEU-4 nhưng có thể đánh đổi ROUGE/METEOR nếu reward chỉ tối ưu BLEU.

**Exp 2 — BLEU vs BLEU+Penalty**: `w_rep ∈ {0, 0.5}`, `w_len ∈ {0, >0}`, subset 25%. Đo BLEU + tỉ lệ lặp tri-gram + độ dài. *Kỳ vọng:* BLEU thuần → lặp cao hơn (xác nhận §A.3).

**Exp 3 — Single vs Multi-sample SCST**: `rl_n_samples ∈ {1,4,8}`, subset 25%. Đo đường BLEU theo epoch + variance advantage. N=8 tốn ~8x rollout — giới hạn N≤4 nếu thiếu ngân sách; *kỳ vọng* diminishing returns.

**Exp 4 — So sánh 6 encoder**: đổi `--encoder`, decoder giữ nguyên để cô lập biến, subset 25%. Metric: BLEU, param (đo thật §A.2), time/epoch. ⚠️ ST-GCN input 150-d hiệu dụng vs 183-d (§G.39) — biến nhiễu cần nêu.

**Exp 7 — PPO vs SCST**: subset 25% và 100% (PPO cần data cho critic — kém hơn ở subset nhỏ cũng là finding). Đo BLEU theo epoch, độ ổn định, wall-clock. **Kỳ vọng trung thực**: PPO *không chắc thắng* (khớp Kiegeland & Kreutzer 2021). Sub-ablation: `ppo_use_clip=False` (A2C), `ppo_gae_lambda ∈ {0.9,0.95,1.0}`.

**Exp 8 — RL frame selection (F.6)**: policy GRU + REINFORCE, model SLT đóng băng, reward = BLEU − phạt keep_frac; subset 25%. **So với 3 counter-base cùng keep_ratio**. ⚠️ **Giới hạn quan trọng**: frame bị loại được **zero-hoá, KHÔNG xoá khỏi chuỗi** → đo được "tín hiệu frame nào quan trọng", **KHÔNG** chứng minh giảm compute (H5 đã điều chỉnh).

**Exp 9 — Reward ablation**: 8 tổ hợp bật/tắt `w_bleu/w_rep/w_len` (+`w_bert` nếu kịp), subset 25% (~8×1h). Bảng BLEU/ROUGE + tỉ lệ lặp + độ dài mỗi tổ hợp — **bảng trung tâm của chương RL**. Bỏ tổ hợp "tất cả=0" (không có gradient).

**Exp 10 — Decoder ablation**: `n_dec_layers ∈ {2,4,6}`, `d_model ∈ {128,256,384}`, subset 25%. Câu hỏi: decoder lớn có tận dụng RL tốt hơn, hay overfit CE trước khi RL kịp phát huy?

**Exp 11 — Data size ablation**: ΔBLEU (RL−CE) theo subset 25/50/100%. 2 giả thuyết cạnh tranh: RL lợi hơn ở data ít (exposure bias nặng) vs data nhiều (advantage ước lượng ổn định).

**Exp 12 — Generalization gap**: (train BLEU − test BLEU) của xe vs rl + 1 lần `evaluate()` trên train_loader. *Kỳ vọng:* RL giảm gap — bằng chứng gián tiếp cho cơ chế §B.

**Exp 13 — Cross-dataset transfer (How2Sign)**: fine-tune transfer (không zero-shot — vocab khác hẳn), so với train-from-scratch cùng subset. Chi phí cao nhất — chỉ làm nếu tuần 9-10 dư.

**Exp 14 — Few-shot**: subset 0.01/0.02 (~70-140 câu). Giả thuyết: RL **kém ổn định** ở few-shot cực đoan → xác định ranh giới "RL cần tối thiểu bao nhiêu data".

**Exp 15 — Latency**: `scripts/measure_latency.py`, batch 1 và 16, warm-up trước khi đo, trên T4. So: xe vs rl checkpoint (*kỳ vọng* **không đổi** — khác biệt đáng kể = bug) và giữa 6 encoder.

### E.4. Experiment bổ sung (ngoài 13 chính)

AMP vs FP32 (throughput + BLEU không đổi?) · shared-encoder vs 2x-encoder · **beam vs greedy trên cả xe lẫn rl checkpoint** (gain RL và gain beam cộng dồn hay che lấp nhau?) · embedding-scaling ablation (§J, A.3).

---

## §F — Chỉ số đánh giá

### F.1. Nhóm chất lượng dịch

| Metric | Định nghĩa | Công cụ | Trạng thái | Lưu ý cho SLT+RL |
|---|---|---|---|---|
| **BLEU-1..4** | Precision n-gram có brevity penalty, trung bình hình học | `sacrebleu` | ✅ (`corpus_bleu` eval, `BLEU(effective_order=True)` sentence-level reward) | BLEU-4 là metric báo cáo chính nhưng nên **luôn báo cáo cả BLEU-1..3** vì SLT câu ngắn (PHOENIX ~10 từ/câu) — BLEU-4 có thể ≈ 0 dù BLEU-1/2 vẫn có nghĩa |
| **ROUGE** (ROUGE-L) | LCS giữa hyp/ref, đo recall | `rouge-score` | ❌ Chưa có | Bổ sung cho BLEU (precision-oriented) — cặp BLEU/ROUGE phát hiện lệch precision/recall (dấu hiệu length hacking) |
| **METEOR** | Alignment unigram có stem/synonym + penalty chunk fragmentation | `nltk.translate.meteor_score` | ❌ Chưa có | Cần WordNet — tiếng Đức hỗ trợ synonym kém hơn tiếng Anh, nên **chủ yếu hữu ích nếu chạy How2Sign** |
| **BERTScore** | Cosine similarity theo cặp token dùng embedding BERT, F1 greedy-matching | `bert-score` (cần checkpoint BERT Đức) | ❌ Chưa có (reward đã cài) | Chi phí GPU đáng kể — chỉ tính **ở tầng đánh giá cuối**, **không** đưa vào reward loop trừ Reward 5 trên subset nhỏ |
| **COMET** | Regression model học từ human judgment, input (src, hyp, ref) | `unbabel-comet` | ❌ Chưa có | Cần GPU riêng, chi phí cao nhất — **chỉ test-set cuối**, không làm reward trực tiếp |

### F.2. Nhóm hiệu năng hệ thống

| Metric | Định nghĩa | Cách đo | Trạng thái | Lưu ý |
|---|---|---|---|---|
| **Latency** | Thời gian suy luận 1 câu (ms) | `time.perf_counter()` quanh `greedy_decode()`, warm-up GPU trước | ✅ `measure_latency.py` | Đo trên GPU cố định (T4) để so công bằng (Exp 15) |
| **FPS** | Số câu/giây | `batch_size / latency_per_batch` | ✅ | batch=1 phản ánh real-time, batch lớn phản ánh throughput — báo cáo cả 2 |
| **Memory** | VRAM peak | `torch.cuda.max_memory_allocated()` | ✅ | Quan trọng để chứng minh khả thi trên T4 16GB |
| **Parameters** | Tổng tham số trainable | `sum(p.numel() ...)` | ✅ (in ở `main.py`) | Tách bảng riêng so sánh encoder |

### F.3. Cách trình bày trong báo cáo

- **Bảng chính**: mỗi hàng = 1 cấu hình, cột = BLEU1-4, ROUGE-L, METEOR, BERTScore, tham số, latency — đo trên **test set**, không phải dev (dev chỉ để early-stop/chọn checkpoint). **Hàng đầu = baseline sàn.**
- **Không dùng BLEU-4 làm tiêu chí early-stopping/chọn checkpoint duy nhất** — theo dõi thêm tỉ lệ lặp n-gram và độ dài trung bình để phát hiện reward hacking.
- **Báo cáo độ lệch chuẩn/khoảng tin cậy** khi so 2 cấu hình gần nhau (PPO vs SCST) — chạy ≥2 seed nếu ngân sách cho phép, tránh kết luận từ 1 lần chạy (RL variance cao).

---

## §G — Nhật ký quyết định thiết kế (41 mục)

> Code comment trỏ `§N` vào đây — **KHÔNG đánh số lại**.

1. `pose_dim=183` = 33 body×3 + 21×2 + 21×2; giữ visibility cho body, bỏ cho tay (không đáng tin).
2. `max_frames=300` — cân bằng thông tin vs `O(T²)` attention trên T4 16GB.
3. Truncate `np.linspace` uniform — baseline đơn giản, **để chỗ** cho RL adaptive sampling (F.9).
4. `vocab_size=3000` BPE — corpus nhỏ, vocab lớn hơn sinh token hiếm học không đủ.
5. `max_text_len=60`, cắt giữa giữ `<bos>/<eos>` — tránh model không bao giờ thấy `<eos>` ở câu dài.
6. `d_model=256/4head/4layer` — nhỏ có chủ đích vì dataset nhỏ; RL cực nhạy overfit.
7. `dropout=0.3` — regularization mạnh hơn chuẩn do #6.
8. `label_smoothing=0.1` — giảm overconfidence, giữ entropy policy cho exploration khi vào RL.
9. `xe_lr=5e-4` + warmup 2000 + cosine — Transformer from scratch nhạy lr đầu.
10. `xe_weight_decay=1e-4` — nhẹ, không triệt tiêu embedding hiếm.
11. `grad_clip=1.0` — quan trọng nhất ở RL phase (reward variance cao sinh gradient spike).
12. `xe_epochs=80` + early stop patience 10.
13. `rl_epochs=20` — RL chỉ fine-tune, train lâu dễ policy collapse.
14. `rl_batch_size=8` < xe 16 — RL cần 2 forward + lưu log-prob, tốn VRAM hơn.
15. `rl_lr=5e-6` — **thấp hơn XE 100 lần**, hyperparameter nhạy nhất hệ thống (chuẩn SCST/RLHF literature).
16. `rl_sample_temp=1.0` — tăng = thêm exploration nhưng thêm variance. ⚠️ PPO **bắt buộc** = 1.0 (§J.4).
17. `rl_n_samples` — đã wired (Multi-sample SCST, Experiment 3).
18. `subset_ratios=[0.25,0.10,0.05]` — vừa kiểm soát compute vừa là trục Experiment 11 miễn phí.
19. Weight tying out_proj = tok_embed (Press & Wolf 2017) — giảm tham số, quan trọng với vocab nhỏ.
20. Pre-LN (`norm_first=True`) — ổn định train from scratch, tránh loss spike. ⚠️ **phải kèm final LayerNorm** (§J.8).
21. Sinusoidal PE (không learned) — không tốn tham số, extrapolate được chuỗi dài hơn train.
22. **Baseline SCST eval-mode** (`rl_baseline_eval_mode=True`): dropout bật làm greedy baseline không deterministic → thêm variance vô ích vào advantage; eval-mode cũng khớp hành vi inference thật. Nên chạy ablation True vs False. ⚠️ Ablation này **chỉ có nghĩa sau khi sửa §J.2**.
23. `rl_entropy_coef=0.0` mặc định — chỉ bật (~1e-3) khi thấy `avg_advantage` sụp về ≈0 sớm.
24. Default `w_bleu=1, w_rep=0.5, w_len=0` — rep penalty bật vì lặp là failure mode quan sát thật (§A.3); len tắt vì brevity penalty của BLEU đã xử lý một phần, bật ở Experiment 9.
25. `sentence_bleu` dùng `effective_order=True` — không có nó, câu < 4-gram (rất phổ biến ở PHOENIX) bị reward = 0 sai.
26. `repetition_penalty` đo tri-gram — n=2 phạt nhầm cụm chức năng, n=4+ phát hiện chậm trên câu ngắn.
27. `length_penalty = (r−h)/r` chặn `[0,1]` — cùng thang với BLEU/rep để các trọng số so sánh được.
28. `eval_every=1` — RL có thể "trông có vẻ học" trong khi reward-hack; chỉ dev BLEU thật phát hiện được.
29. `save_every=5` nhưng best checkpoint theo dev BLEU luôn được lưu riêng.
30. `early_stop_patience=10` chung cho XE (12.5% tổng epoch) và RL (50%) — RL noisy hơn cần patience tương đối cao hơn.
31. `seed=42` cho cả subset sampling lẫn init — 3 mức subset là tập lồng nhau nhất quán giữa các lần chạy.
32. 2-phase XE→RL tách biệt (không train chung `CE+λRL` từ đầu) — RL from scratch trên vocab 3000 có reward ≈ 0 ở early training.
33. `--phase rl` khởi tạo model instance mới rồi mới load `best_xe.pt` — tránh rò rỉ optimizer/scheduler state từ phase XE.
34. Pose extraction tách rời hoàn toàn, cache `.npz` upload Kaggle Dataset — MediaPipe trên CPU là bottleneck 10-20h.
35. Share `memory` giữa greedy-baseline và sample **chỉ khi** `rl_baseline_eval_mode=False`. ⚠️ **Đã sửa** — cách share cũ làm encoder mất gradient (§J.2).
36. **PPO: encoder/hidden PHẢI tính lại (có gradient) mỗi ppo_epoch** — tái dùng memory từ rollout no_grad thì encoder không bao giờ nhận gradient từ PPO loss (lỗi dễ mắc nhất khi cài PPO cho seq2seq).
37. `ppo_gae_lambda=0.95` mặc định (không phải 1.0) — thận trọng cho lần cài đầu; thử `λ∈{0.9,0.95,1.0}` như sub-ablation Exp 7.
38. `reward_bert_weight=0.0` mặc định dù đã cài `bertscore_reward()` — 1 forward BERT/sample không batch hoá làm chậm mọi run.
39. ST-GCN bỏ kênh visibility (chỉ x,y) để 75 khớp cùng số kênh — input 150 chiều hiệu dụng vs 183 của Transformer, là **biến nhiễu cần nêu** khi diễn giải Experiment 4.
40. Nội suy landmark thiếu bằng `np.interp` theo từng segment (body/tay) thay vì zero-fill; segment thiếu toàn video giữ 0 (không có neo) — nay **được đếm và in ra** cuối `extract_poses.py`.
41. AMP bọc forward/backward model, **KHÔNG** bọc vòng reward (string ops CPU) — nhớ khi thêm phép tính GPU (BERTScore) vào reward.

---

## §H — Đề xuất luận văn

### H.1. Mục tiêu

Xây dựng và đánh giá thực nghiệm pipeline **SLT (video/pose → text)** trong đó **RL** fine-tune trực tiếp theo metric (khắc phục loss-metric mismatch + exposure bias của CE), đồng thời **mở rộng RL ra ngoài decoder** — quyết định rời rạc ở tầng thị giác — thể hiện tích hợp 2 môn (RL + Xử lý ảnh & video) thay vì tách rời.

### H.2. Câu hỏi nghiên cứu

1. **RQ1**: SCST có cải thiện BLEU/BERTScore so với CE thuần trên SLT data nhỏ, và có nhất quán qua 25/50/100% không?
2. **RQ2**: Thiết kế reward (BLEU thuần vs +penalty vs +semantic) ảnh hưởng thế nào tới chất lượng thực (lặp, độ dài) — reward nào cân bằng nhất?
3. **RQ3**: RL nâng cao hơn SCST (PPO) có đáng độ phức tạp thêm ở bài toán reward-thưa-cuối-câu?
4. **RQ4**: Kiến trúc pose-encoder ảnh hưởng thế nào tới khả năng "hưởng lợi" từ RL?
5. **RQ5**: RL áp dụng được cho quyết định rời rạc tầng thị giác (frame selection) không, và có cải thiện đồng thời chất lượng + hiệu quả?

### H.3. Giả thuyết

Xem slide 10 (H1–H5).

### H.4. Đóng góp khoa học

1. Hệ thống hoá + kiểm chứng **10 thuật toán RL khả thi** (lọc từ 13) cho SLT data nhỏ, có bảng ưu tiên theo compute thực tế — góc thực dụng ít paper đề cập.
2. So sánh **10 biến thể reward có hệ thống** (Exp 9) — trade-off điểm số vs chất lượng thực, thường bị paper SOTA lướt qua.
3. **RL ngoài decoder** (frame selection, Exp 8) — chứng minh (hoặc bác bỏ có cơ sở, so với counter-baseline không-học) RL là khung tối ưu xuyên pipeline CV+NLP.
4. Đối chiếu trực tiếp 3 công trình RL-cho-SLT gần nhất: Panaro 2020 · RVLF 2025 · SignDPO 2026 — định vị đóng góp trong bức tranh hiện tại.

### H.5. Đóng góp thực nghiệm

Codebase pose-SLT hoàn chỉnh tái lập được trên hạ tầng miễn phí (T4×2) · 13 experiment + **bảng so sánh tự động kèm baseline sàn** trên 3 mức subset · nhật ký thiết kế minh bạch 41 mục (§G).

### H.6. Hạn chế

- Data nhỏ (PHOENIX subset) — không tổng quát hoá trực tiếp; giảm nhẹ một phần bởi Exp 13.
- Không human evaluation — mọi kết luận theo metric tự động.
- Pose-only, bỏ face landmark (non-manual markers); RGB đã loại hẳn khỏi phạm vi.
- Không RLHF thật — "RL" đều dùng metric tự động làm reward (§C.8).
- Phần lớn experiment 1 seed do giới hạn GPU — RL variance cao, đọc kết luận thận trọng.
- Gloss-free là chính — so sánh với phương pháp gloss-based trong literature không hoàn toàn công bằng (P7 làm đối chứng nội bộ).

### H.7. Hướng phát triển

1. RLHF thật (preference con người quy mô nhỏ).
2. DPO với preference giả từ multi-sample ranking (đã code — mở rộng nhiễu loạn kiểu SignDPO).
3. Video Encoder → Q-Former → LLM nhỏ (đã loại vì thiếu LLM infra — đề tài tiếp nối).
4. Bật face landmarks (non-manual markers).
5. Hierarchical RL (§C.13).
6. Curriculum kép data+reward (§C.12 + Reward 10) đầy đủ.
7. **Frame selection re-index chuỗi thật** để đo compute-saving (nối tiếp H5).

### H.8. Phần có thể publish (thấp → cao rủi ro)

1. **Reward ablation cho SLT ít tài nguyên** (Exp 2+9) — dạng phân tích ít paper làm đầy đủ; workshop CVPR/WACV.
2. **SCST baseline eval-mode vs train-mode** (§G.22) — finding engineering nhỏ có số liệu.
3. **SCST vs PPO cho SLT data nhỏ** (RQ3/H3, đối chiếu Panaro 2020 + RVLF 2025) — giá trị dù kết quả chiều nào.
4. **RL frame selection cho SLT** (nếu Exp 8 thắng `base_frames_uniform`) — góc mới nhất, ứng viên short paper.
5. Pose→Q-Former→LLM — mới nhất nhưng ngoài hạ tầng hiện có.

Không publish riêng "ST-GCN vs Transformer" — kết quả dự đoán được từ literature; giá trị chính là hạ tầng so sánh nội bộ.

---

## §I — Roadmap & rủi ro

> Roadmap gốc giả định từ số 0. Thực tế: **tuần 1-6 + toàn bộ hạng mục CODE của tuần 7-9 đã hoàn thành** — phần còn lại là **CHẠY trên Kaggle và viết báo cáo**.

| Tuần | Nội dung | Trạng thái |
|---|---|---|
| 1-2 | Survey | ✅ bộ `docs/` |
| 3 | Data (extract_poses, tokenizer, dataset) | ✅ code xong — ⚠️ `extract_poses.py` vừa sửa bug chặn (§J.1) |
| 4-5 | Baseline Transformer + XE | ✅ (P1); tuần 5 dùng cho Experiment 4 |
| 6 | SCST | ✅ — dùng để chạy Exp 1 & 11 lấy số liệu đầu |
| 7 | Reward engineering | ✅ code xong — **còn lại: CHẠY Exp 2 & 9** trên subset 25% |
| 8 | Ablation | ✅ code xong — **còn lại: CHẠY Exp 3, 4, 10, 15** |
| 9 | PPO (+MRT/RAML/DPO, selection/decode policy) | ✅ code xong — **còn lại: CHẠY Exp 7, 8 + `eval_baselines.py`** |
| 10 | Viết báo cáo | 🔲 Tổng hợp `comparison_table` → luận văn; chạy Exp 12; rà số liệu *(approx, verify)* ở §C.3 |

**Ghi chú kỹ thuật PPO** (code comment `train_ppo.py` trỏ vào đây): PPO bản đầu **đơn giản hoá có chủ đích để giảm rủi ro debug** — reward chỉ ở cuối episode (chưa dùng incremental-BLEU shaping, Reward 9), GAE `γ=1` cho episode ngắn ≤60 token, `λ` chỉnh qua config. Mở rộng bật sau khi bản cơ bản ổn định trên Kaggle.

### I.5. Rủi ro & dự phòng

| Rủi ro | Dự phòng |
|---|---|
| PPO không hội tụ trong ngân sách | "PPO chưa vượt SCST" **vẫn là kết quả hợp lệ** (khớp H3) — không ép thắng |
| Hết quota GPU giữa chừng | Ưu tiên tuyệt đối Exp 1, 9, 11; bỏ Exp 13 nếu cần |
| BERTScore quá chậm cho full ablation | Chỉ chạy ở 1 tổ hợp best theo BLEU, nêu rõ giới hạn |
| Kết quả policy CV không thắng baseline | Báo cáo trung thực kèm `base_frames_uniform`/`base_fixed_temp` — "không thắng heuristic" cũng là finding về ranh giới của RL |
| P4 Graph Transformer OOM trên T4 | Giảm batch size **chỉ cho encoder này** + ghi rõ là biến nhiễu (§J) |

---

## §J — Code review & bug đã sửa

### J.0. Kiến trúc hiện tại (tóm tắt)

`Pose 183-d (MediaPipe, offline) → PoseEmbed/encoder factory → TransformerDecoder → CE pretrain → RL fine-tune`. Thiết kế đúng đắn nên giữ: tách 2 phase CE→RL, baseline eval-mode có chủ đích (§G.22), reward bật/tắt bằng weight=0, tokenizer cache dùng chung mọi subset, pose extraction tách rời training loop.

### J.1–J.9. Bug đã sửa (07/2026)

| # | File | Vấn đề | Ảnh hưởng |
|---|---|---|---|
| **J.1** | `data/extract_poses.py` | Chỉ quét `**/*.mp4`, nhưng PHOENIX-2014T là **thư mục ảnh PNG** → tìm được 0 file, không báo lỗi | 🔴 **Chặn toàn bộ pipeline** |
| **J.2** | `training/train_scst.py` | Nhánh `rl_baseline_eval_mode=False` share `memory` tính trong `no_grad` → **encoder không bao giờ nhận gradient** | 🔴 Ablation §G.22 đang so nhầm "encoder được train" vs "encoder đóng băng". **Kết quả cũ của ablation này không dùng được** |
| **J.3** | `train_scst.py` + `main.py` | RL không vượt XE → không lưu checkpoint → `test_results.json` không có dòng RL | 🟡 **Không báo cáo được "RL kém hơn CE"** — kết quả hợp lệ theo H3. Nay luôn lưu `last_rl.pt` |
| **J.4** | `training/train_ppo.py` | `rl_sample_temp ≠ 1.0` làm importance ratio sai (old có temperature, new không) | 🟡 PPO ra số vô nghĩa mà không báo. Nay raise ngay |
| **J.5** | `data/dataset.py` | File pose thiếu → im lặng trả vector 0 | 🔴 Nếu `pose_cache_dir` sai thì train 80 epoch trên toàn số 0 mà **không crash**. Nay fail-fast + cảnh báo |
| **J.6** | `data/dataset.py` | Hard-code `annotations/PHOENIX-2014-T.*.csv`, release chuẩn để ở `annotations/manual/` | 🟡 Nay thử 3 layout + quét đệ quy |
| **J.7** | 8 file trainer | `torch.cuda.amp` deprecate từ PyTorch 2.4 → FutureWarning **mỗi batch** | 🟢 Log ngập cảnh báo che mất BLEU/advantage. Nay dùng `utils/amp_compat.py` |
| **J.8** | `slt_transformer.py` + `encoders.py` | **Pre-LN thiếu final LayerNorm** (`norm_first=True` nhưng `norm=None`) | 🔴 **Nghiêm trọng nhất về mặt học** — xem dưới |
| **J.9** | `slt_transformer.py` | **Init embedding sai scale** (`N(0,1)` mặc định nhưng `decode_step` nhân `sqrt(d_model)`; cộng weight tying) | 🔴 Cộng hưởng với J.8 |

> [!danger] Chi tiết J.8 + J.9 — phát hiện bằng smoke test, không phải suy đoán
> **Đo thực tế trước khi sửa** (model khởi tạo, `d_model=256`, vocab 3000):
> ```
> decoder hidden std = 17.82   (Pre-LN đúng phải ≈ 1)
> logits std         = 303.88  (đáng lẽ ≈ 1)
> log_prob của token sampled = 0.0  CHÍNH XÁC
> entropy            = 0.0     (max có thể = 8.0 nats)
> encoder gradient   = 0.0
> ```
> **Cơ chế hỏng:** logits khổng lồ → softmax bão hoà thành one-hot → `log π(a) = 0` → `∇ log π = 0` → **toàn bộ policy gradient bằng 0**. Nghĩa là SCST/PPO/MRT sẽ chạy đủ 20 epoch, in log bình thường, nhưng **model không cập nhật gì cả**. Không crash, không cảnh báo.
>
> **Sau khi sửa** (`norm=nn.LayerNorm(d_model)` + init embedding `std = d_model^-0.5`):
> ```
> logits std = 0.99 … 1.02   (cả 6 encoder)
> entropy    = 5.4 … 7.0 nats
> encoder gradient: 30–85 tensor có grad ≠ 0
> ```
> Entropy ban đầu ~7/8 nats đúng với chủ đích của `label_smoothing=0.1` (§G.8).
>
> ⚠️ **Hệ quả lên số liệu:** thêm LayerNorm làm param count tăng ~0.01M. Số đã cập nhật khắp nơi:
> P1 **8.26M** · P2 **9.38M** · P3 6.33M · P4 **9.48M** · P5 **7.47M** · P6 9.59M.

### J.10. Điểm yếu còn lại (chấp nhận có chủ đích)

| # | Điểm yếu | Trạng thái |
|---|---|---|
| A.2 | Pad value 0.0 trùng nghĩa với missing-landmark trong cùng tensor | ⚠️ chấp nhận (pose_mask che pad đúng; chỉ dễ nhầm khi debug) |
| A.3 | PoseEmbed không scale `√d_model` như decoder embedding | ⚠️ có chủ đích (đã qua LayerNorm) — nên ablation thay vì để ngầm định |
| A.4 | Reward tính tuần tự CPU từng sample | ⚠️ chấp nhận ở batch 8; theo dõi khi multi-sample N=8 |
| B.5 | Không KV-cache khi decode (`O(L³)` mỗi câu) | ⚠️ chấp nhận với `max_text_len=60` — hướng tối ưu |
| B.8 | Causal mask tính lại mỗi bước | ⚠️ chi phí nhỏ, chưa ưu tiên |
| C.15 | Không có sanity-check tự động | ⚠️ smoke-test thủ công đã chạy; chưa có CI |

### J.11. Rủi ro chưa sửa — cần biết trước khi chạy

- **P4 `graph_transformer` có thể OOM trên T4** (spatial attention `[B*T, 75, 75]`, với B=16/T=300 là 4800 attention map/layer). Giảm batch size nếu gặp, **ghi rõ là biến nhiễu**.
- **P7 two-stage gloss chưa verify trên dữ liệu thật.**
- **`vocab_size=3000` có thể quá lớn với corpus PHOENIX.** SentencePiece sẽ raise nếu không đủ ký tự. Nếu gặp, giảm còn 2000 và **ghi lại vì nó đổi §G.4**.
- **AMP + `Categorical` ở fp16**: `sample_decode` tạo phân phối từ logits fp16 trong `autocast`. Chưa quan sát thấy NaN nhưng nếu RL loss thành `nan` thì đây là nghi phạm đầu tiên — thử `use_amp=False` cho phase RL.

---

## §K — Hướng dẫn thực thi

> Trả lời đúng một câu hỏi: **chạy lệnh nào, theo thứ tự nào, để điền đầy các ô `-` trong `paper/sn-article.tex`.**
> Quy trình hiện tại dùng **notebook extract** (`Sign-Language-REL_pose-extract.ipynb`) rồi notebook
> train (`Sign-Language-REL_smoke-5pct.ipynb` cho mức 5%, hoặc `KAGGLE_NOTEBOOK.ipynb` để train đa
> dạng) và **một orchestrator** `run_all.py` chạy TOÀN BỘ ma trận cho 1 subset trong 1 lệnh. Mức
> báo cáo CHÍNH giờ là **5%** (train 5%, dev/test full). Đọc số run thật ở `comparison_table.csv`.

### K.0. Tóm tắt

| Ưu tiên | Chạy gì | Ra số gì | Ước tính |
|---|---|---|---|
| **🔴 BƯỚC 0** | `Sign-Language-REL_pose-extract.ipynb` (kernel CPU-only) | `.npz` cho toàn corpus (hoặc chỉ 5% qua `smoke-5pct`) | **10–20h CPU** (không tốn quota GPU) |
| **🔴 BẮT BUỘC** | `run_all.py --subset 0.05` | Baseline sàn + Exp 1/2/3/4/7/9/10/11/12/15, TẤT CẢ ở 5% | ~6–9h GPU (gọn 1 session) |
| 🟡 Khi có thêm quota | `run_all.py --subset 0.25` | Cùng ma trận, subset 25% — Exp 11 (data-size ablation) | ~20–25h GPU |
| 🟡 Khi có thêm quota | `run_all.py --subset 1.0` | Cùng ma trận, subset 100% | rất lớn — trải nhiều session, dùng `--groups`/`train_select.py` |

**Nếu chỉ có 1 session:** BƯỚC 0 rồi `run_all.py --subset 0.05` (mặc định `--groups all`) là đủ để
có toàn bộ bảng so sánh chính ở 5% — đủ bảo vệ luận điểm chính. 25%/100% là mở rộng khi có quota.

### K.1. BƯỚC 0 — Extract pose (`Sign-Language-REL_pose-extract.ipynb`)

> [!danger] Chỗ dễ hỏng nhất và đắt nhất
> **PHOENIX-2014T KHÔNG có file video.** Mỗi câu là **một thư mục ảnh PNG**:
> `PHOENIX-2014-T/features/fullFrame-210x260px/<split>/<name>/images0001.png`

Chạy trên **notebook Kaggle riêng, Accelerator = CPU-only** (MediaPipe không cần GPU — không tốn
quota GPU hàng tuần dành cho train). Notebook đã có sẵn smoke-test (`--limit 5` + kiểm tra
shape/dtype 1 file `.npz`) trước khi chạy full, và hướng dẫn resume qua nhiều session nếu 10-20h
vượt quá 1 session — xem trực tiếp `Sign-Language-REL_pose-extract.ipynb`, không lặp lại lệnh ở đây.

> [!important] 📊 SỐ LIỆU CẦN GHI #1
> **Ba con số `missing_counts`** in ra cuối script — bằng chứng định lượng cho vấn đề ① (pose quality) ở slide 04 và động cơ thực nghiệm cho F.8. Hiện slide chỉ nói định tính. Nếu `lhand`/`rhand` > 5% corpus thì đó là **finding đáng đưa vào paper**.

Xong thì tạo Kaggle Dataset `phoenix-poses` từ tab Output của notebook đó (đã khớp sẵn
`cfg.data.pose_cache_dir = "/kaggle/input/phoenix-poses"` trong `configs/config.py`, không cần sửa gì).

### K.2. BẮT BUỘC — TOÀN BỘ ma trận cho 1 subset (`Sign-Language-REL_smoke-5pct.ipynb` / `KAGGLE_NOTEBOOK.ipynb`)

```bash
python run_all.py --subset 0.05
```

Lệnh này (đọc kỹ docstring đầu `run_all.py`) tự chạy tuần tự, resumable qua marker
`<log_dir>/.done_<key>` (chạy lại đúng lệnh này nếu bị cắt giữa chừng — không mất phần đã xong):

1. Baseline sàn (`base_empty`, `base_most_frequent`) — Exp bắt buộc trước mọi model.
2. **Core**: Transformer + SCST, XE→RL (Exp 1).
3. 5 encoder còn lại × SCST (Exp 4 + H4).
4. PPO/MRT/RAML/DPO trên Transformer, tái dùng `best_xe.pt` của core (Exp 7).
5. REINFORCE (no-baseline) / A2C (no-clip) / Curriculum RL — 3 ablation.
6. Reward ablation — 4 tổ hợp `rw_bleu_only/rw_default/rw_len_only/rw_both` (Exp 9).
7. Latency (Exp 15) cho 6 encoder.
8. `aggregate_results` → `comparison_table.csv/.md` (tự chạy lại ở CUỐI MỖI lần gọi `run_all.py`,
   kể cả khi 1 vài bước ở trên lỗi/bị skip — luôn phản ánh đúng trạng thái hiện tại).
9. `make_report` → `report/tables/*.csv/.md/.tex` (6 bảng đã lọc theo câu hỏi so sánh + 3 bảng
   LaTeX dán thẳng `tab:main`/`tab:reward`/`tab:encresults`) và `report/figures/*.png/.pdf` (5
   biểu đồ: BLEU theo epoch, ΔBLEU theo subset, trade-off reward ablation, so sánh encoder, so
   sánh thuật toán) — cũng tự chạy lại mỗi lần. Chạy tay: `python scripts/make_report.py --work_dir /kaggle/working`.

> Nhánh **P7 two-stage** và **selection/decode policy (RL ngoài decoder)** đã gỡ — xem [[2_Huong_Phat_Trien]].

Muốn giới hạn phạm vi (debug nhanh, hoặc cố tình bỏ bớt để tiết kiệm quota): `--groups
core,encoders,algos,ablations,reward,latency` (mặc định `all`). Muốn chọn từng encoder/algo riêng:
`train_select.py --mode single|encoder_allrl|rl_allenc`.

Output mỗi encoder/algo vẫn theo đúng convention cũ trong `/kaggle/working/run1_{encoder}_subset{pct}/`:

| File | Chứa gì |
|---|---|
| `best_xe.pt` | Checkpoint XE tốt nhất |
| `best_{algo}.pt` | Checkpoint RL — **chỉ có nếu RL vượt XE** (`best_scst.pt`, `best_ppo.pt`, ...) |
| `last_rl.pt` | Checkpoint RL cuối (SCST) — **luôn có** (J.3) |
| `xe_history.json` / `rl_history.json` / `ppo_history.json` | loss, `dev_bleu4`, `avg_advantage`, `avg_rep_rate`, `avg_len_ratio` theo epoch |
| `test_results.json` | BLEU-4 test — **merge** qua nhiều thuật toán chạy vào cùng thư mục (bug ghi đè cũ đã sửa) |

> [!important] 📊 SỐ LIỆU CẦN GHI — quan trọng nhất
> - `test_bleu4` của `xe` và mọi `--algo` đã chạy, ở **mọi subset đã có** → Table `tab:main` + **ΔBLEU của Exp 11** (so sánh giữa các subset khi đã chạy thêm 0.5/1.0)
> - Đường `dev_bleu4` theo epoch → **HÌNH #1** (§K.4)
> - `avg_advantage` epoch đầu vs cuối → tín hiệu reward hacking
> - Reward ablation (4 tổ hợp): `test_bleu4` · `avg_rep_rate` · `avg_len_ratio` epoch cuối → Table `tab:reward` (kiểm chứng H2)

> [!tip] Nếu RL không vượt XE
> Log in `[!] RL KHÔNG vượt XE ...`. **Không phải lỗi.** Pipeline tự rơi về `last_rl.pt` nên con số âm vẫn vào bảng. Báo cáo nó — đó là H3 hoạt động đúng.

⚠️ `graph_transformer` có thể OOM — `run_all.py` chỉ log lỗi và bỏ qua bước đó, không phá cả ma
trận; ghi rõ vào phần biến nhiễu nếu xảy ra.

📊 **Exp 8 — cách đọc:** policy chỉ "học được" nếu **thắng `base_frames_uniform` ở cùng `keep_frac`**. Thắng `base_frames_random` là đương nhiên; thắng `base_frames_full` cũng không chứng minh gì.

**Exp 12 — generalization gap:** thêm 1 lần `evaluate()` trên `train_loader` cho cả 2 checkpoint, ghi `(train_bleu − test_bleu)` (chưa tự động hoá trong `run_all.py`, chạy tay khi cần).

### K.3. Chạy tay 1 cấu hình đơn lẻ (debug, KHÔNG cần cho quy trình chuẩn)

`run_all.py` đã gọi các hàm này cho toàn bộ ma trận; chỉ dùng trực tiếp khi cần debug 1 cấu hình:

```bash
python main.py --subset 0.05 --encoder transformer --algo scst --phase all --tag run1
python main.py --subset 0.05 --encoder transformer --algo scst --phase rl --tag rw_bleu_only \
    --xe_ckpt /kaggle/working/run1_transformer_subset5/best_xe.pt   # tái dùng XE tag khác
python train_select.py --mode single --encoder stgcn --algo ppo --subset 0.05   # phạm vi hẹp
python scripts/eval_baselines.py --subset 0.05
python scripts/measure_latency.py --ckpt /kaggle/working/run1_transformer_subset5/best_xe.pt --encoder transformer
```

### K.4. Tổng hợp + HÌNH cần vẽ

```bash
python scripts/aggregate_results.py --work_dir /kaggle/working --out /kaggle/working/comparison_table
```

> [!note] Hình hiện có
> `paper/sn-article.tex` đã có `fig:pipeline` vẽ bằng **TikZ, self-contained** — không cần file ảnh ngoài.

Ba hình **nên bổ sung**, xếp theo giá trị:

| # | Hình | Dữ liệu nguồn | Vì sao đáng vẽ |
|---|---|---|---|
| **1** | **BLEU theo epoch: XE vs SCST vs PPO** | `xe_history.json`, `rl_history.json`, `ppo_history.json` | Hình thuyết phục nhất — cho thấy RL cải thiện ở đâu, có ổn định không |
| **2** | **ΔBLEU (RL−CE) theo subset** (cột, 3 mức) | `test_results.json` × 3 | Trả lời trực tiếp Exp 11 + H1 |
| **3** | **Trade-off reward ablation** (scatter: BLEU × `avg_rep_rate`, 4 điểm) | Exp 9 | Hình hoá H2 |

Vẽ bằng matplotlib từ `comparison_table.csv`, export **PDF vector** rồi `\includegraphics`.

> [!tip] Nếu chỉ vẽ được 1 hình
> Vẽ **hình #1** — nó cho thấy cùng lúc: RL có học không, có ổn định không, có collapse không, early-stop kích hoạt ở đâu.

### K.5. Checklist trước khi viết báo cáo

- [ ] Có ít nhất **subset 25%** cho Exp 1 (mở rộng 50%/100% khi có thêm quota)
- [ ] Có **baseline sàn** merged vào `comparison_table`
- [ ] Mỗi bảng có dòng sàn ở **đầu bảng**
- [ ] Exp 9 có đủ **3 cột**: BLEU, rep_rate, len_ratio
- [ ] Exp 8 có **cả 3 counter-baseline** cùng `keep_frac`
- [ ] Latency đo **sau warm-up**, báo cáo cả batch 1 và 16
- [ ] Mọi bảng ghi rõ **số seed** — nếu 1 seed thì nói thẳng
- [ ] Ghi lại `missing_counts` từ bước extract
- [ ] Nếu `graph_transformer` phải giảm batch size → **ghi vào phần biến nhiễu**
- [ ] Thay câu "BLEU 15 vs sàn 12" ở slide 10 bằng **số thật**

---

## §L — References (xác minh 20/07/2026)

> [!warning] Nguyên tắc trích dẫn
> Mọi mục ở §L.1 đã được **kiểm chứng bằng cách truy cập nguồn gốc** (arXiv / RIT repository), không dẫn theo trí nhớ. Cột "Dùng vào đâu" ghi rõ đề tài lấy gì từ bài đó.
> Hai preprint 2025–2026 **cần xác nhận venue trước khi trích dẫn chính thức**; trong bài nói gọi là "preprint gần đây", không phải "công trình đã công bố".

### L.1. Tiền lệ trực tiếp RL cho SLT — 3 bài quan trọng nhất

| Bài | Nguồn | Dùng vào đâu |
|---|---|---|
| **Panaro, J. (2020)** — *Fine-Tuning Sign Language Translation Systems Through Deep Reinforcement Learning*. MS thesis, Computer Engineering, RIT. Advisor: Ifeoma Nwogu. | [repository.rit.edu/theses/10653](https://repository.rit.edu/theses/10653/) | **Tiền lệ gần nhất về setting**: SCST + PPO trên chính RWTH-PHOENIX-Weather-2014T. Định vị đóng góp (slide 12, §H.4). ⚠️ Không peer-reviewed |
| **Rao, Z., Zhou, Y., Zhou, B., Huang, Y., Escalera, S., Wan, J. (2025)** — *RVLF: A Reinforcing Vision-Language Framework for Gloss-Free Sign Language Translation*. arXiv:2512.07273, nộp 08/12/2025. | [arxiv.org/abs/2512.07273](https://arxiv.org/abs/2512.07273) | **GRPO đầu tiên cho SLT**, reward = BLEU + ROUGE. BLEU-4: **+5.1** CSL-Daily, **+1.11 PHOENIX-2014T**, +1.4 How2Sign, +1.61 OpenASL. Trả lời "sao không dùng GRPO" (slide 07) |
| **Pu, M., Wu, X.-M., Lim, M.K., Chong, C.Y., Li, W., Loy, C.C. (2026)** — *SignDPO: Multi-level Direct Preference Optimisation for Skeleton-based Gloss-free Sign Language Translation*. arXiv:2604.18034, nộp 20/04/2026. | [arxiv.org/abs/2604.18034](https://arxiv.org/abs/2604.18034) | **DPO trên skeleton** — trùng modality. Hierarchical perturbation + self-guiding qua decoder cross-attention. Điểm so sánh cho `train_dpo.py`, hướng phát triển §H.7 #2. ⚠️ **CSL-Daily/How2Sign/OpenASL, KHÔNG có PHOENIX-2014T** |

### L.2. Nền tảng SLT

| Bài | Dùng vào đâu |
|---|---|
| **Camgoz et al. (CVPR 2018)** — *Neural Sign Language Translation* | Định nghĩa SLT ≠ SLR; giới thiệu **PHOENIX-2014T**; mốc BLEU-4 e2e **9.58** |
| **Camgoz et al. (CVPR 2020)** — *Sign Language Transformers* | Kiến trúc gần pipeline hiện tại nhất (joint CTC + Transformer decoder); mốc **~21.8** |
| **Bohacek & Hruz (WACV-W 2022)** — *SPOTER* | Tiền lệ pose-only gần nhất về compute regime; nguồn kỹ thuật chuẩn hoá signing space (§D.5) |
| **Zhou et al. (ICCV 2023)** — *GFSLT-VLP* | Tiền lệ gloss-free pretrain kiểu CLIP |
| **Wong et al. (ICLR 2024)** — *Sign2GPT* | Bằng chứng hướng LLM khả thi → cơ sở đưa vào §H.7 #3 |
| Stein/RWTH (~2004-12) — SMT gloss↔text | Chỉ trích dẫn lịch sử — xây chính corpus PHOENIX |
| STMC/STMC-Transformer · MMTLB · TwoStream-SLT · SignLLM | Bối cảnh SOTA; ⚠️ đều cần RGB/multi-cue nên ngoài thực thi |

### L.3. RL cho sequence generation

| Bài | Dùng vào đâu |
|---|---|
| **Williams (1992)** — REINFORCE | Công thức `∇J = E[(R−b)∇log π]`; ablation §C.1 |
| **Ranzato et al. (ICLR 2016)** — MIXER | Nguồn khái niệm **exposure bias** (§A.1); CE warm-start + REINFORCE |
| **Rennie et al. (CVPR 2017)** — **SCST** | **Xương sống** — `train_scst.py` |
| **Shen et al. (ACL 2016)** — MRT | `train_mrt.py` |
| **Norouzi et al. (NeurIPS 2016)** — RAML | `train_raml.py` |
| **Schulman et al. (ICLR 2016)** — **GAE** | `ppo_gae_lambda` trong `train_ppo.py` |
| **Schulman et al. (2017)** — **PPO** | `train_ppo.py`; Experiment 7 |
| **Rafailov et al. (NeurIPS 2023)** — **DPO** | `train_dpo.py` |
| **Ouyang et al. (NeurIPS 2022)** — InstructGPT/RLHF | Chỉ mượn khái niệm; lý do loại RLHF thật (§C.8) |
| **Kiegeland & Kreutzer (NAACL 2021)** | **Trích dẫn bắt buộc** — phản biện "gain RL cho NMT là ảo"; cơ sở của **H3 trung lập** |
| **Ng et al. (1999)** — potential-based reward shaping | Cơ sở lý thuyết Reward 9 (chưa bật) |
| Bahdanau et al. (2016) — Actor-Critic seq | Nền tảng §C.3; nổi tiếng khó train |

### L.4. Kiến trúc & thành phần

| Bài | Dùng vào đâu |
|---|---|
| **Vaswani et al. (NeurIPS 2017)** — Transformer | Kiến trúc lõi `SLTTransformer` (P1) |
| **Press & Wolf (EACL 2017)** — weight tying | `out_proj = tok_embed` (§G.19) |
| **Yan et al. (AAAI 2018)** — **ST-GCN** | P3 — encoder nhẹ nhất (1.35M), cơ sở H4 |
| **Jaegle et al. (ICLR 2022)** — **Perceiver IO** | P6 — độ phức tạp tuyến tính, trả lời vấn đề ③ |
| **Holtzman et al. (ICLR 2020)** — neural text degeneration | Nguồn khái niệm **degeneracy** (§A.3); lý do có `repetition_penalty` |

### L.5. Metric

| Bài | Dùng vào đâu |
|---|---|
| **Papineni et al. (ACL 2002)** — BLEU | Metric chính + reward chính |
| **Post (WMT 2018)** — sacreBLEU | Công cụ; `effective_order=True` cho sentence-level |
| **Zhang et al. (ICLR 2020)** — BERTScore | Reward 5 + đánh giá cuối |
| **Rei et al. (EMNLP 2020)** — COMET | Reward 6 (chưa cài) + đánh giá cuối; ghi rõ là hộp đen |

### L.6. Công cụ

- **MediaPipe Holistic** — trích 183-d pose. ⚠️ Nhược điểm: mất track tay khi occlusion/chuyển động nhanh (vấn đề ① slide 04).
- **SentencePiece** — BPE tokenizer, vocab 3000.
- **PyTorch** — `torch.cuda.amp` deprecate từ 2.4 → dùng `utils/amp_compat.py` (§J.7).

> [!note] Bài CHƯA xác minh được venue
> "Camgoz — Keypoint-based SLT" từng được nhắc trong docs cũ với ghi chú *(verify venue)*. Tài liệu này **không** trích dẫn nó vì chưa xác minh được nguồn gốc. Nếu muốn đưa vào luận văn thì phải tìm bản gốc trước.

---

## Phụ lục — Tra cứu nhanh

### Bản đồ vấn đề ↔ giải pháp ↔ experiment

| Vấn đề (slide 04) | Giải pháp (slide 05) | Experiment | Baseline đối chứng |
|---|---|---|---|
| ⑤ Exposure bias | RL rollout = inference | Exp 1, 12 | `base_empty`, `base_most_frequent` |
| ⑦ Loss-metric mismatch | Tối ưu `E[R(y)]` trực tiếp | Exp 1 | như trên |
| ④ Degeneracy | `repetition_penalty` | Exp 2, 9 | Reward 1 (BLEU thuần) |
| ① Pose quality / occlusion | F.8 landmark selection | — | `base_drop_body/lhand/rhand` |
| ② Uniform sampling | F.6 / F.9 frame policy | Exp 8 | `base_frames_full/random/uniform` |
| ⑥ Decode strategy cố định | F.5 decode policy | — | `base_fixed_temp_{0.7,1.0,1.3}` |
| ③ Độ dài chuỗi / `O(T²)` | Perceiver IO (P6), ST-GCN (P3) | Exp 4, 15 | P1 Transformer |

### Số liệu cần thuộc

| Hạng mục | Con số |
|---|---|
| Pose | 183-d = 33×3 + 21×2 + 21×2 |
| Dataset | PHOENIX-2014T, 8.257 câu, split 7.096/519/642 |
| Model | `d_model=256`, 4 head, 4 layer, vocab BPE 3000, `max_frames=300`, `max_text_len=60` |
| Tham số (đo thật) | P1 8.26M · P2 9.38M · **P3 6.33M (encoder 1.35M)** · P4 9.48M · P5 7.47M · P6 9.59M |
| Learning rate | XE `5e-4` · **RL `5e-6` (thấp hơn 100 lần)** |
| Epoch | XE 80 (patience 10) · RL 20 |
| Reward mặc định | `w_bleu=1, w_rep=0.5, w_len=0` |
| Subset | 25% / 50% / 100%, seed 42 |
| Ngân sách | ~30 GPU-h/tuần, Kaggle T4×2 |
| Mốc literature | Camgoz 2018 e2e **9.58** · Camgoz 2020 **~21.8** · SOTA gloss-free **~22–27** · RVLF **+1.11** trên PHOENIX |

### Ba câu trả lời cứu nguy

> [!tip] Khi không biết câu trả lời
> **"Đây là điều em chưa kiểm chứng."** Rồi nói rõ nó sẽ được kiểm chứng bằng cách nào. Toàn bộ tài liệu này viết theo tinh thần phân biệt rõ *đã đo* / *đã code chưa chạy* / *chưa verify* — trả lời như vậy là **nhất quán** với đề tài chứ không phải né tránh.

> [!tip] Khi bị chê phạm vi hẹp
> **"Đúng, và đó là lựa chọn có chủ đích."** Mỗi thứ bị loại đều có lý do cụ thể là *chưa verify được tiền đề trong ngân sách*, không phải *không có giá trị*, và tất cả nằm ở §H.7.

> [!tip] Khi bị hỏi "kết quả có tốt không"
> **"Câu hỏi đúng phải là: tốt hơn cái gì?"** Rồi kéo về baseline sàn, và so sánh chính là **ΔBLEU giữa CE-only và CE+RL từ cùng checkpoint gốc**, không phải BLEU tuyệt đối so với SOTA chạy full dataset.
