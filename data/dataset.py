"""PHOENIX-2014T dataset loader. Đọc pose đã extract sẵn + text annotation."""
import os, json, glob, warnings
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

# PHOENIX-2014T phân chia: train/dev/test trong file annotation csv
# Format: name|video|start|end|speaker|orth|translation
# Ta chỉ cần: name (để load pose) và translation (text Đức). Cột `orth` (gloss) KHÔNG dùng —
# nhánh gloss/P7 đã gỡ khỏi pipeline (xem docs/2_Huong_Phat_Trien.md).

class PhoenixSLTDataset(Dataset):
    def __init__(self, annotation_csv: str, pose_dir: str, tokenizer,
                 max_frames: int = 300, max_text_len: int = 60,
                 subset_ratio: float = 1.0, seed: int = 42,
                 pose_dim: int = 183):
        self.df = pd.read_csv(annotation_csv, sep="|")
        self.df = self.df.dropna(subset=["translation"]).reset_index(drop=True)

        # Subset: cố định seed để reproducible
        if subset_ratio < 1.0:
            self.df = self.df.sample(frac=subset_ratio, random_state=seed).reset_index(drop=True)

        self.pose_dir = pose_dir
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_text_len = max_text_len
        self.pose_dim = pose_dim
        self._n_missing = 0        # đếm file pose thiếu — xem cảnh báo trong __getitem__

        # Fail-fast: nếu pose_cache_dir sai hoàn toàn thì phát hiện NGAY, đừng để train xong
        # 80 epoch mới nhận ra model học từ toàn vector 0. Đây là chế độ hỏng nguy hiểm nhất
        # của pipeline này vì nó KHÔNG crash — loss vẫn giảm, BLEU vẫn ~0, trông như "model kém".
        if not os.path.isdir(pose_dir):
            raise FileNotFoundError(
                f"pose_cache_dir không tồn tại: {pose_dir}\n"
                f"-> Chạy data/extract_poses.py trước, rồi trỏ cfg.data.pose_cache_dir vào output."
            )
        n_npz = len(glob.glob(os.path.join(pose_dir, "*.npz")))
        if n_npz == 0:
            raise FileNotFoundError(
                f"Không có file .npz nào trong {pose_dir} — pose chưa được extract hoặc mount sai."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]
        text = str(row["translation"])

        # Load pose
        pose_path = os.path.join(self.pose_dir, f"{name}.npz")
        if os.path.exists(pose_path):
            pose = np.load(pose_path)["pose"].astype(np.float32)
        else:
            # Fallback để 1 file thiếu lẻ tẻ không làm crash cả run — NHƯNG phải kêu to,
            # vì im lặng ở đây nghĩa là model được train trên vector 0 mà không ai biết.
            self._n_missing += 1
            if self._n_missing in (1, 10, 100, 1000):
                warnings.warn(
                    f"[{os.path.basename(self.pose_dir)}] thiếu {self._n_missing} file pose "
                    f"(gần nhất: {name}.npz). Nếu con số này lớn thì tên file .npz KHÔNG khớp "
                    f"cột `name` của annotation csv — kiểm tra lại data/extract_poses.py.",
                    RuntimeWarning, stacklevel=2)
            pose = np.zeros((1, self.pose_dim), dtype=np.float32)

        # Truncate
        if len(pose) > self.max_frames:
            # Uniform sample
            idxs = np.linspace(0, len(pose) - 1, self.max_frames).astype(int)
            pose = pose[idxs]

        # Tokenize text: <bos> ... <eos>. Cắt phần GIỮA, giữ nguyên <bos>/<eos>
        # (cắt thẳng [:max_text_len] sẽ làm mất <eos> ở câu dài -> model không học dừng).
        ids = self.tokenizer.encode(text, add_special=True)
        if len(ids) > self.max_text_len:
            ids = ids[: self.max_text_len - 1] + [self.tokenizer.eos_id]

        return {
            "pose": torch.from_numpy(pose),         # [T, D]
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "text_raw": text,
            "name": name,
        }


def collate_fn(batch, pad_id: int = 0):
    poses = [b["pose"] for b in batch]
    texts = [b["text_ids"] for b in batch]
    pose_lens = torch.tensor([p.size(0) for p in poses], dtype=torch.long)
    text_lens = torch.tensor([t.size(0) for t in texts], dtype=torch.long)

    pose_pad = pad_sequence(poses, batch_first=True, padding_value=0.0)   # [B, T_max, D]
    text_pad = pad_sequence(texts, batch_first=True, padding_value=pad_id)  # [B, L_max]

    # Padding masks: True = padded
    B, T_max, _ = pose_pad.shape
    pose_mask = (torch.arange(T_max)[None, :] >= pose_lens[:, None])
    L_max = text_pad.size(1)
    text_mask = (torch.arange(L_max)[None, :] >= text_lens[:, None])

    return {
        "pose": pose_pad, "pose_mask": pose_mask, "pose_lens": pose_lens,
        "text_ids": text_pad, "text_mask": text_mask, "text_lens": text_lens,
        "text_raw": [b["text_raw"] for b in batch],
        "names": [b["name"] for b in batch],
    }


class LengthCurriculumSampler(Sampler):
    """Curriculum RL (C.12, docs/1_Thuyet_Trinh_Tong_Hop.md) + Ý tưởng F.18: duyệt batch theo
    câu ngắn->dài (dựa độ dài text tham chiếu, không phải độ dài pose) thay vì shuffle ngẫu nhiên --
    câu ngắn dễ đạt reward dương sớm hơn, giảm cold-start RL. Batch nội bộ đã sort (đồng đều độ dài,
    cũng giảm padding lãng phí); THỨ TỰ các batch được xáo trộn theo seed để không cố định 100%."""
    def __init__(self, dataset, batch_size: int, seed: int = 0):
        lengths = dataset.df["translation"].astype(str).str.split().apply(len).values
        self.sorted_idx = list(np.argsort(lengths, kind="stable"))
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch  # đổi seed xáo trộn thứ tự batch mỗi epoch (không đổi nội dung batch)

    def __iter__(self):
        chunks = [self.sorted_idx[i:i + self.batch_size]
                 for i in range(0, len(self.sorted_idx), self.batch_size)]
        rng = np.random.RandomState(self.seed + self._epoch)
        rng.shuffle(chunks)
        for chunk in chunks:
            for idx in chunk:
                yield idx

    def __len__(self):
        return len(self.sorted_idx)


def make_curriculum_loader(base_loader: DataLoader, seed: int = 0) -> DataLoader:
    """Dựng lại 1 DataLoader dùng chung dataset/collate_fn/batch_size với `base_loader` nhưng thay
    `shuffle` ngẫu nhiên bằng `LengthCurriculumSampler` -- dùng cho `rl_curriculum_length_sort=True`
    trong `training/train_scst.py` (chỉ áp dụng ở N epoch đầu, xem `rl_curriculum_epochs`)."""
    ds = base_loader.dataset
    sampler = LengthCurriculumSampler(ds, base_loader.batch_size, seed=seed)
    return DataLoader(ds, batch_size=base_loader.batch_size, sampler=sampler,
                      collate_fn=base_loader.collate_fn, num_workers=base_loader.num_workers,
                      pin_memory=True)


def find_annotation_csv(phoenix_root: str, split: str) -> str:
    """Tìm file annotation của `split` ("train"/"dev"/"test") mà KHÔNG giả định 1 layout duy nhất.

    Bản phát hành chuẩn PHOENIX-2014-T-release-v3 đặt csv ở `annotations/manual/`, nhưng khi
    người dùng tự upload lên Kaggle Dataset thì thư mục thường bị làm phẳng thành `annotations/`
    hoặc thậm chí nằm ngay gốc. Trước đây code hard-code đúng 1 đường dẫn -> FileNotFoundError
    khó hiểu ở dòng pd.read_csv. Giờ thử lần lượt rồi báo lỗi có kèm những chỗ đã tìm."""
    fname = f"PHOENIX-2014-T.{split}.corpus.csv"
    candidates = [
        os.path.join(phoenix_root, "annotations", "manual", fname),   # layout gốc release-v3
        os.path.join(phoenix_root, "annotations", fname),             # layout đã làm phẳng
        os.path.join(phoenix_root, fname),                            # đặt thẳng ở gốc
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Cứu cánh cuối: quét đệ quy (Kaggle Dataset hay thêm 1 lớp thư mục thừa khi giải nén)
    hits = glob.glob(os.path.join(phoenix_root, "**", fname), recursive=True)
    if hits:
        return sorted(hits)[0]
    raise FileNotFoundError(
        f"Không tìm thấy {fname}. Đã thử:\n  " + "\n  ".join(candidates) +
        f"\n  (và quét đệ quy dưới {phoenix_root})\n"
        f"-> Kiểm tra lại cfg.data.phoenix_root trong configs/config.py."
    )


def make_loaders(cfg, tokenizer, subset_ratio: float = 1.0):
    train_csv = find_annotation_csv(cfg.data.phoenix_root, "train")
    dev_csv   = find_annotation_csv(cfg.data.phoenix_root, "dev")
    test_csv  = find_annotation_csv(cfg.data.phoenix_root, "test")

    train_ds = PhoenixSLTDataset(train_csv, cfg.data.pose_cache_dir, tokenizer,
                                 cfg.data.max_frames, cfg.data.max_text_len,
                                 subset_ratio=subset_ratio, seed=cfg.seed,
                                 pose_dim=cfg.data.pose_dim)
    dev_ds   = PhoenixSLTDataset(dev_csv, cfg.data.pose_cache_dir, tokenizer,
                                 cfg.data.max_frames, cfg.data.max_text_len,
                                 pose_dim=cfg.data.pose_dim)
    test_ds  = PhoenixSLTDataset(test_csv, cfg.data.pose_cache_dir, tokenizer,
                                 cfg.data.max_frames, cfg.data.max_text_len,
                                 pose_dim=cfg.data.pose_dim)

    # functools.partial thay vì lambda: picklable -- cần thiết khi num_workers>0 trên start method
    # "spawn" (Windows/macOS mặc định); "fork" (Linux/Kaggle mặc định) không cần nhưng vẫn tương
    # thích. Không đổi hành vi, chỉ đổi cách truyền tham số cố định pad_id vào collate_fn.
    collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.xe_batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.train.xe_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.xe_batch_size, shuffle=False,
                             collate_fn=collate, num_workers=2, pin_memory=True)
    return train_loader, dev_loader, test_loader
