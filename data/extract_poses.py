"""
Bước 0 của TOÀN BỘ pipeline — chạy MỘT LẦN, trên máy local hoặc 1 Kaggle kernel RIÊNG
(không phải kernel train). Output: 1 file .npz cho mỗi sequence → nén lại → upload thành
Kaggle Dataset → mount khi train.

CRITICAL: đừng extract pose mỗi lần train. MediaPipe chạy CPU tốn 10-20h/lần cho cả corpus;
chạy lại mỗi run sẽ phá vỡ toàn bộ ngân sách GPU. Đây là ràng buộc định hình mọi thiết kế
phía sau (docs/1_Thuyet_Trinh_Tong_Hop.md §G.34).

------------------------------------------------------------------------------------------
PHOENIX-2014T KHÔNG PHẢI FILE VIDEO — ĐỌC KỸ PHẦN NÀY
------------------------------------------------------------------------------------------
Bản phát hành chuẩn `PHOENIX-2014-T-release-v3` lưu mỗi câu thành 1 THƯ MỤC ẢNH PNG:

    PHOENIX-2014-T/features/fullFrame-210x260px/<split>/<name>/images0001.png
                                                              /images0002.png
                                                              ...

trong đó `<name>` khớp đúng cột `name` của file annotation csv, ví dụ
`01April_2010_Thursday_heute_default-0`. KHÔNG có file .mp4 nào cả.

Vì vậy script này mặc định chạy ở `--mode frames` (quét thư mục ảnh). `--mode video` chỉ
dành cho dataset khác thật sự có file video (vd How2Sign, Experiment 13).

Usage
-----
# PHOENIX-2014T (mặc định) — trỏ vào thư mục CHỨA các thư mục <split>
python data/extract_poses.py \
    --input_dir  /path/PHOENIX-2014-T/features/fullFrame-210x260px \
    --out_dir    ./poses

# Dataset dạng video rời (How2Sign, ...)
python data/extract_poses.py --mode video --input_dir /path/videos --out_dir ./poses --ext mp4

Kiểm tra nhanh trước khi chạy full (rất nên làm — 10-20h là quá đắt để phát hiện sai đường dẫn):
python data/extract_poses.py --input_dir <...> --out_dir ./poses --limit 5
"""
import os, glob, argparse, time
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

# Lấy module Holistic (legacy solutions API). Trên một số image Kaggle (Python 3.12), bản
# mediapipe cài được có `python/solutions/` ĐẦY ĐỦ trên đĩa nhưng top-level `mediapipe/__init__.py`
# lại KHÔNG re-export `solutions` (do uninstall/reinstall bẩn) -> `mp.solutions` ném AttributeError.
# Import thẳng submodule để né __init__ hỏng. (mediapipe >= 0.10.31 gỡ HẲN solutions -> phải ghim
# <= 0.10.21, xem KAGGLE_NOTEBOOK_EXTRACT.ipynb.)
try:
    mp_holistic = mp.solutions.holistic
except AttributeError:
    from mediapipe.python.solutions import holistic as mp_holistic

# MediaPipe KHÔNG thread-safe và cũng không pickle được -> mỗi PROCESS phải tự tạo instance
# riêng và giữ lại dùng chung cho mọi sequence mà process đó xử lý (tạo lại mỗi lần sẽ tốn
# ~1s khởi tạo model, nhân với 8257 sequence là mất hàng giờ vô ích).
_WORKER = {}


def _init_worker(model_complexity: int):
    """Chạy 1 lần khi Pool sinh ra mỗi process con."""
    cv2.setNumThreads(1)   # tránh OpenCV tự spawn thread, gây tranh CPU với các process khác
    _WORKER["holistic"] = mp_holistic.Holistic(static_image_mode=False,
                                               model_complexity=model_complexity)

# Layout 183-d — PHẢI khớp cfg.data.pose_dim (configs/config.py)
N_BODY_FEAT = 33 * 3   # 33 khớp body × (x, y, visibility) = 99
N_HAND_FEAT = 21 * 2   # 21 khớp tay × (x, y)              = 42  (bỏ visibility: không đáng tin)
POSE_DIM = N_BODY_FEAT + 2 * N_HAND_FEAT  # 99 + 42 + 42 = 183


def _interpolate_segment(seg: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính theo thời gian cho các frame present=False (landmark thiếu do
    occlusion/detection fail), dùng frame present gần nhất trước/sau (np.interp extrapolate
    hằng số ở 2 đầu = forward/backward-fill biên).

    Vì sao KHÔNG zero-fill: vector 0 tuyệt đối làm model học nhầm "tay đứng yên ở gốc toạ độ"
    thay vì "thiếu dữ liệu" — 2 tình huống hoàn toàn khác nhau về ngữ nghĩa.

    Nếu KHÔNG frame nào present -> giữ nguyên 0 (không có neo để nội suy). Đây là nhiễu tiềm
    ẩn còn sót, được đếm và in ra ở cuối main() để biết mức độ nghiêm trọng thay vì bỏ qua.
    """
    T = seg.shape[0]
    valid_idx = np.where(present)[0]
    if len(valid_idx) == 0 or len(valid_idx) == T:
        return seg
    out = seg.copy()
    all_idx = np.arange(T)
    for d in range(seg.shape[1]):
        out[:, d] = np.interp(all_idx, valid_idx, seg[valid_idx, d])
    return out


def _landmarks_to_vec(results):
    """1 frame MediaPipe -> (body[99], lh[42], rh[42], present_flags)."""
    body = np.zeros(N_BODY_FEAT, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            body[i * 3: i * 3 + 3] = [lm.x, lm.y, lm.visibility]

    lh = np.zeros(N_HAND_FEAT, dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i * 2: i * 2 + 2] = [lm.x, lm.y]

    rh = np.zeros(N_HAND_FEAT, dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i * 2: i * 2 + 2] = [lm.x, lm.y]

    return body, lh, rh, (results.pose_landmarks is not None,
                          results.left_hand_landmarks is not None,
                          results.right_hand_landmarks is not None)


def _finalize(frames_body, frames_lh, frames_rh, present_b, present_l, present_r):
    """Stack + nội suy từng nhóm ĐỘC LẬP -> [T, 183]. Trả thêm cờ 'nhóm thiếu toàn bộ'."""
    if not frames_body:
        return np.zeros((1, POSE_DIM), dtype=np.float32), (True, True, True)
    body = _interpolate_segment(np.stack(frames_body), np.array(present_b))
    lh = _interpolate_segment(np.stack(frames_lh), np.array(present_l))
    rh = _interpolate_segment(np.stack(frames_rh), np.array(present_r))
    all_missing = (not any(present_b), not any(present_l), not any(present_r))
    return np.concatenate([body, lh, rh], axis=1), all_missing


def extract_from_frame_dir(frame_dir: str, holistic, img_exts=("png", "jpg", "jpeg")):
    """PHOENIX-2014T: 1 câu = 1 thư mục ảnh PNG đã sort theo tên (images0001, images0002...).
    Thứ tự file CHÍNH LÀ thứ tự thời gian, nên sorted() là đúng và bắt buộc."""
    paths = []
    for ext in img_exts:
        paths.extend(glob.glob(os.path.join(frame_dir, f"*.{ext}")))
    paths = sorted(paths)

    fb, fl, fr, pb, pl, pr = [], [], [], [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        b, l, r, (has_b, has_l, has_r) = _landmarks_to_vec(results)
        fb.append(b); fl.append(l); fr.append(r)
        pb.append(has_b); pl.append(has_l); pr.append(has_r)
    return _finalize(fb, fl, fr, pb, pl, pr)


def extract_from_video(video_path: str, holistic):
    """Dataset dạng video rời (How2Sign...). Giữ lại để Experiment 13 dùng được cùng script."""
    cap = cv2.VideoCapture(video_path)
    fb, fl, fr, pb, pl, pr = [], [], [], [], [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        b, l, r, (has_b, has_l, has_r) = _landmarks_to_vec(results)
        fb.append(b); fl.append(l); fr.append(r)
        pb.append(has_b); pl.append(has_l); pr.append(has_r)
    cap.release()
    return _finalize(fb, fl, fr, pb, pl, pr)


def _discover_frame_dirs(input_dir: str):
    """Tìm mọi thư mục LÁ có chứa ảnh. Chấp nhận cả 2 cách trỏ --input_dir:
       .../fullFrame-210x260px            (chứa train/ dev/ test/)
       .../fullFrame-210x260px/train      (chứa thẳng các thư mục câu)
    Tên output .npz = tên thư mục lá = đúng cột `name` trong annotation csv."""
    seqs = []
    for split in sorted(e.path for e in os.scandir(input_dir) if e.is_dir()):
        children = sorted(e.path for e in os.scandir(split) if e.is_dir())
        if children:            # split chứa các sequence dir → lấy chúng, DỪNG (không vào trong)
            seqs.extend(children)
        else:                   # input_dir vốn đã là 1 split → chính split là sequence
            seqs.append(split)
    return sorted(seqs)


def _process_one(task):
    """Xử lý 1 sequence trong process con. Trả về tuple kết quả để process cha tổng hợp.
    KHÔNG raise ra ngoài — 1 sequence hỏng không được phép giết cả Pool sau nhiều giờ chạy."""
    unit, out_dir, mode = task
    seq_id = os.path.basename(unit.rstrip("/\\")) if mode == "frames" \
        else os.path.splitext(os.path.basename(unit))[0]
    out_path = os.path.join(out_dir, f"{seq_id}.npz")
    if os.path.exists(out_path):
        return ("skip", seq_id, None, None)
    try:
        holistic = _WORKER["holistic"]
        if mode == "frames":
            poses, all_missing = extract_from_frame_dir(unit, holistic)
        else:
            poses, all_missing = extract_from_video(unit, holistic)
        assert poses.shape[1] == POSE_DIM, f"shape sai: {poses.shape}"
        # Ghi ra file tạm rồi đổi tên: nếu bị kill giữa chừng (hết giờ Kaggle, tắt máy) thì
        # KHÔNG để lại file .npz hỏng dở — lần chạy sau sẽ tưởng nó đã xong và bỏ qua.
        # Ghi qua file handle đang mở: nếu truyền THẲNG tên "X.npz.tmp", np.savez_compressed
        # tự chèn ".npz" -> ghi nhầm ra "X.npz.tmp.npz" khiến os.replace không thấy file.
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, pose=poses)
        os.replace(tmp, out_path)
        return ("ok", seq_id, all_missing, poses.shape[0])
    except Exception as e:
        return ("err", seq_id, str(e), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="frames: thư mục cha chứa các thư mục ảnh · video: thư mục chứa file video")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["frames", "video"], default="frames",
                        help="frames = PHOENIX-2014T (mặc định) · video = dataset có file video thật")
    parser.add_argument("--ext", default="mp4", help="chỉ dùng với --mode video")
    parser.add_argument("--limit", type=int, default=0,
                        help=">0: chỉ xử lý N sequence đầu — DÙNG ĐỂ SMOKE-TEST đường dẫn trước khi chạy full")
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                        help="0 nhanh nhất/kém nhất, 2 chậm nhất/tốt nhất. 1 = cân bằng (mặc định)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Số process song song. 0 = tự chọn (cpu_count-1). 1 = tuần tự (dễ debug). "
                             "ĐÂY LÀ CỜ QUAN TRỌNG NHẤT: MediaPipe chạy CPU và song song hoá hoàn "
                             "toàn được -> 8 process rút ~13h xuống ~1.7h.")
    parser.add_argument("--shard", type=int, default=0,
                        help="Chỉ số shard (0-indexed) — chia việc cho nhiều MÁY/KERNEL khác nhau")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Tổng số shard. Vd 4 máy: mỗi máy chạy --num_shards 4 --shard 0|1|2|3")
    parser.add_argument("--names_file", default=None,
                        help="Chỉ extract các sequence có TÊN (basename thư mục = cột `name` annotation) "
                             "nằm trong file này (mỗi tên 1 dòng). Dùng để extract ĐÚNG subset mà loader "
                             "sẽ chọn (vd 5%% train + full dev/test cho smoke-test) thay vì cả corpus.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "frames":
        units = _discover_frame_dirs(args.input_dir)
        kind = "thư mục ảnh"
    else:
        units = sorted(glob.glob(os.path.join(args.input_dir, f"**/*.{args.ext}"), recursive=True))
        kind = f"file .{args.ext}"

    print(f"Tìm thấy {len(units)} {kind} trong {args.input_dir}")
    if not units:
        # Fail nhanh, có hướng dẫn — thay vì chạy 0 vòng lặp rồi báo "xong" gây hiểu nhầm.
        raise SystemExit(
            "Không tìm thấy dữ liệu nào.\n"
            "  · PHOENIX-2014T lưu ảnh PNG theo thư mục, KHÔNG có file .mp4 — hãy dùng --mode frames\n"
            "    và trỏ --input_dir vào .../PHOENIX-2014-T/features/fullFrame-210x260px\n"
            "  · Nếu dataset thật sự là video rời, thêm --mode video --ext mp4"
        )
    if args.names_file:
        with open(args.names_file, encoding="utf-8") as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        units = [u for u in units if os.path.basename(u.rstrip("/\\")) in wanted]
        print(f"--names_file: lọc còn {len(units)} sequence khớp tên (yêu cầu {len(wanted)} tên)")
        if not units:
            raise SystemExit("Không sequence nào khớp names_file — kiểm tra tên có đúng cột `name`?")
    if args.limit > 0:
        units = units[: args.limit]
        print(f"--limit {args.limit}: chỉ xử lý {len(units)} sequence đầu (smoke-test)")
    if args.num_shards > 1:
        units = units[args.shard::args.num_shards]     # xen kẽ -> tải đều giữa các shard
        print(f"Shard {args.shard}/{args.num_shards}: {len(units)} sequence")

    workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    print(f"Dùng {workers} process song song (máy có {cpu_count()} CPU)")

    tasks = [(u, args.out_dir, args.mode) for u in units]
    n_done = n_skip = n_err = 0
    n_frames = 0
    missing_counts = {"body": 0, "lhand": 0, "rhand": 0}
    errors = []
    t0 = time.time()

    def _tally(res):
        nonlocal n_done, n_skip, n_err, n_frames
        status, seq_id, payload, nfr = res
        if status == "skip":
            n_skip += 1
        elif status == "ok":
            n_done += 1; n_frames += nfr or 0
            for key, miss in zip(("body", "lhand", "rhand"), payload):
                missing_counts[key] += int(miss)
        else:
            n_err += 1; errors.append((seq_id, payload))

    if workers == 1:
        _init_worker(args.model_complexity)
        for t in tqdm(tasks):
            _tally(_process_one(t))
    else:
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(args.model_complexity,)) as pool:
            # imap_unordered: nhận kết quả ngay khi có, thanh tiến trình mượt và
            # không giữ toàn bộ kết quả trong RAM.
            for res in tqdm(pool.imap_unordered(_process_one, tasks, chunksize=4),
                            total=len(tasks)):
                _tally(res)

    dt = time.time() - t0
    print(f"\nXong sau {dt/60:.1f} phút: {n_done} mới · {n_skip} đã có · {n_err} lỗi")
    if n_done:
        print(f"Tốc độ: {n_frames/max(dt,1e-9):.0f} frame/s tổng · {n_frames} frame đã xử lý")
    if errors:
        print(f"\n{len(errors)} lỗi (10 đầu):")
        for sid, msg in errors[:10]:
            print(f"  {sid}: {msg}")
    print(f"Nhóm landmark THIẾU TOÀN BỘ video (giữ 0, không nội suy được): {missing_counts}")
    print(f"-> Upload thư mục {args.out_dir} thành Kaggle Dataset, rồi đặt "
          f"cfg.data.pose_cache_dir trỏ vào đó.")


if __name__ == "__main__":
    main()