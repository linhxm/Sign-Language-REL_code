"""Lõi inference cho demo: video -> MediaPipe pose [T,183] -> SLTTransformer -> câu tiếng Đức.

Dùng LẠI nguyên si code train (models/, configs/, data/tokenizer.py) thay vì chép lại kiến trúc —
nếu chép, chỉ cần lệch 1 tham số (d_model, norm cuối, weight tying) là load_state_dict vẫn chạy
nhưng output thành rác, rất khó phát hiện. Vì vậy file này BẮT BUỘC tìm được thư mục `code/`.

Tiền xử lý pose ở đây phải khớp TỪNG BƯỚC với lúc train, nếu không model nhận input lệch phân phối:
  · data/extract_poses.py : layout 183-d (33 body*(x,y,vis) + 21 lh*(x,y) + 21 rh*(x,y)),
                            nội suy tuyến tính frame thiếu landmark theo TỪNG NHÓM độc lập.
  · data/dataset.py       : nếu T > max_frames(300) thì lấy mẫu ĐỀU (np.linspace), không cắt đuôi.
Không có chuẩn hoá nào khác (toạ độ MediaPipe đã ở [0,1] theo khung hình).

Dùng:
    from slt_demo import PoseExtractor, Translator
    pose, stats, frames = PoseExtractor().extract("video.mp4")
    tr = Translator.from_checkpoint("best_xe.pt", "spm.model")
    print(tr.translate(pose))
"""
from __future__ import annotations

import os
import sys
import glob
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------------------
# Tìm thư mục code/ (chứa models/, configs/, data/) rồi đưa vào sys.path
# ---------------------------------------------------------------------------------------
_CANDIDATE_CODE_DIRS = [
    os.environ.get("SLT_CODE_DIR", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"),
    "/content/Sign-Language-REL_code",
    "/content/code",
    os.path.join(os.getcwd(), "code"),
    os.getcwd(),
]


def find_code_dir() -> str:
    for c in _CANDIDATE_CODE_DIRS:
        if c and os.path.isfile(os.path.join(c, "models", "slt_transformer.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Không tìm thấy thư mục code/ (cần models/slt_transformer.py).\n"
        "-> git clone https://github.com/linhxm/Sign-Language-REL_code.git\n"
        "   rồi đặt biến môi trường SLT_CODE_DIR trỏ vào đó."
    )


CODE_DIR = find_code_dir()
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import torch  # noqa: E402  (import sau khi set sys.path là cố ý)

from configs.config import Config  # noqa: E402
from data.tokenizer import Tokenizer  # noqa: E402
from models.slt_transformer import SLTTransformer  # noqa: E402

POSE_DIM = 183
N_BODY_FEAT = 33 * 3
N_HAND_FEAT = 21 * 2

ENCODER_CHOICES = ["transformer", "gcn", "stgcn", "graph_transformer", "tcn", "perceiver"]


# =======================================================================================
# 1. Trích pose từ video
# =======================================================================================
@dataclass
class PoseStats:
    n_frames_video: int          # tổng frame trong file
    n_frames_processed: int      # số frame thực sự chạy MediaPipe (sau khi bỏ theo stride)
    n_frames_model: int          # số frame đưa vào model (sau uniform-sample về max_frames)
    fps: float
    duration_s: float
    stride: int
    det_body: float              # tỉ lệ frame detect được nhóm landmark tương ứng
    det_lhand: float
    det_rhand: float
    extract_s: float

    def as_markdown(self) -> str:
        return (
            f"| Chỉ số | Giá trị |\n|---|---|\n"
            f"| Video | {self.n_frames_video} frame · {self.fps:.1f} fps · {self.duration_s:.1f}s |\n"
            f"| Đã xử lý MediaPipe | {self.n_frames_processed} frame (stride={self.stride}) |\n"
            f"| Đưa vào model | {self.n_frames_model} frame × {POSE_DIM}-d |\n"
            f"| Tỉ lệ detect body | {self.det_body*100:.1f}% |\n"
            f"| Tỉ lệ detect tay trái | {self.det_lhand*100:.1f}% |\n"
            f"| Tỉ lệ detect tay phải | {self.det_rhand*100:.1f}% |\n"
            f"| Thời gian trích pose | {self.extract_s:.1f}s |\n"
        )


def _interpolate_segment(seg: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Copy nguyên logic data/extract_poses.py::_interpolate_segment — nội suy tuyến tính theo
    thời gian cho frame thiếu landmark. KHÔNG zero-fill: vector 0 dạy model "tay đứng yên ở gốc
    toạ độ" thay vì "thiếu dữ liệu"."""
    T = seg.shape[0]
    valid = np.where(present)[0]
    if len(valid) == 0 or len(valid) == T:
        return seg
    out = seg.copy()
    all_idx = np.arange(T)
    for d in range(seg.shape[1]):
        out[:, d] = np.interp(all_idx, valid, seg[valid, d])
    return out


def _landmarks_to_vec(results):
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


class PoseExtractor:
    """Bọc MediaPipe Holistic. Giữ 1 instance dùng lại giữa các lần gọi (khởi tạo model tốn ~1s,
    tạo lại mỗi request làm demo giật)."""

    def __init__(self, model_complexity: int = 1):
        import cv2
        import mediapipe as mp
        try:
            holistic_mod = mp.solutions.holistic
            self._drawing = mp.solutions.drawing_utils
            self._styles = mp.solutions.drawing_styles
        except AttributeError:
            # Một số bản mediapipe cài bẩn không re-export `solutions` ở top-level (xem
            # data/extract_poses.py) -> import thẳng submodule.
            from mediapipe.python.solutions import holistic as holistic_mod
            from mediapipe.python.solutions import drawing_utils as _du
            from mediapipe.python.solutions import drawing_styles as _ds
            self._drawing, self._styles = _du, _ds
        self._cv2 = cv2
        self._holistic_mod = holistic_mod
        self.holistic = holistic_mod.Holistic(static_image_mode=False,
                                              model_complexity=model_complexity)

    def _annotate(self, frame_bgr, results):
        img = frame_bgr.copy()
        h = self._holistic_mod
        self._drawing.draw_landmarks(
            img, results.pose_landmarks, h.POSE_CONNECTIONS,
            landmark_drawing_spec=self._styles.get_default_pose_landmarks_style())
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            self._drawing.draw_landmarks(
                img, hand, h.HAND_CONNECTIONS,
                landmark_drawing_spec=self._styles.get_default_hand_landmarks_style())
        return self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)

    def extract(self, video_path: str, max_frames: int = 300, stride: int = 0,
                n_preview: int = 6, progress_cb=None
                ) -> Tuple[np.ndarray, PoseStats, List[np.ndarray]]:
        """video -> (pose [T,183] float32, PoseStats, list ảnh RGB đã vẽ landmark).

        stride=0 -> tự chọn: MediaPipe chạy CPU ~10-20 fps, một clip webcam 30fps dài 30s là 900
        frame ≈ 1 phút chờ. Tự đặt stride sao cho số frame xử lý <= 2*max_frames, vẫn thừa độ phân
        giải thời gian vì dataset dù sao cũng lấy mẫu về <=300 frame.
        """
        cv2 = self._cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if stride <= 0:
            stride = max(1, int(np.ceil(n_total / (2 * max_frames)))) if n_total > 0 else 1

        preview_at = set()
        if n_total > 0 and n_preview > 0:
            est = max(1, n_total // stride)
            preview_at = {int(i) for i in np.linspace(0, est - 1, min(n_preview, est))}

        fb, fl, fr, pb, pl, pr = [], [], [], [], [], []
        previews: List[np.ndarray] = []
        t0 = time.time()
        i_read = i_proc = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i_read % stride != 0:
                i_read += 1
                continue
            res = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            b, l, r, (hb, hl, hr) = _landmarks_to_vec(res)
            fb.append(b); fl.append(l); fr.append(r)
            pb.append(hb); pl.append(hl); pr.append(hr)
            if i_proc in preview_at:
                previews.append(self._annotate(frame, res))
            i_proc += 1
            i_read += 1
            if progress_cb and i_proc % 10 == 0:
                progress_cb(i_proc, max(1, n_total // stride))
        cap.release()
        extract_s = time.time() - t0

        if not fb:
            raise RuntimeError("Không đọc được frame nào từ video (file hỏng hoặc codec không hỗ trợ).")

        body = _interpolate_segment(np.stack(fb), np.array(pb))
        lh = _interpolate_segment(np.stack(fl), np.array(pl))
        rh = _interpolate_segment(np.stack(fr), np.array(pr))
        pose = np.concatenate([body, lh, rh], axis=1).astype(np.float32)
        assert pose.shape[1] == POSE_DIM, f"layout sai: {pose.shape}"

        n_proc = pose.shape[0]
        pose = subsample_to_max(pose, max_frames)

        stats = PoseStats(
            n_frames_video=n_total or i_read,
            n_frames_processed=n_proc,
            n_frames_model=pose.shape[0],
            fps=fps,
            duration_s=(n_total / fps) if (n_total and fps) else 0.0,
            stride=stride,
            det_body=float(np.mean(pb)), det_lhand=float(np.mean(pl)), det_rhand=float(np.mean(pr)),
            extract_s=extract_s,
        )
        return pose, stats, previews


def subsample_to_max(pose: np.ndarray, max_frames: int = 300) -> np.ndarray:
    """Khớp data/dataset.py: video dài hơn max_frames thì lấy mẫu ĐỀU trên toàn chuỗi
    (np.linspace), KHÔNG cắt đuôi — cắt đuôi sẽ mất nửa cuối câu."""
    if len(pose) <= max_frames:
        return pose
    idxs = np.linspace(0, len(pose) - 1, max_frames).astype(int)
    return pose[idxs]


def load_pose_npz(path: str, max_frames: int = 300) -> np.ndarray:
    """Đọc .npz đã trích sẵn (vd phoenix-poses) — cho phép demo trên đúng dữ liệu PHOENIX mà
    không cần chạy lại MediaPipe."""
    pose = np.load(path)["pose"].astype(np.float32)
    return subsample_to_max(pose, max_frames)


# =======================================================================================
# 2. Model
# =======================================================================================
def detect_encoder(state_dict) -> str:
    """Checkpoint không lưu config (chỉ có model/epoch/bleu) -> suy ra encoder_type từ chính tên
    tham số. Sai encoder = load_state_dict báo lỗi thiếu key, nên đoán sai vẫn an toàn (không âm
    thầm cho ra rác)."""
    keys = list(state_dict.keys())

    def has(prefix):
        return any(k.startswith(prefix) for k in keys)

    if has("pose_encoder.latents") or has("pose_encoder.in_cross_attn"):
        return "perceiver"
    if has("pose_encoder.tcn_blocks"):
        return "tcn"
    if has("pose_encoder.spatial_encoder"):
        return "graph_transformer"
    if has("pose_encoder.gc1"):
        return "gcn"
    if has("pose_encoder.st_blocks") or has("pose_encoder.blocks"):
        return "stgcn"
    return "transformer"


@dataclass
class Translator:
    model: SLTTransformer
    tokenizer: Tokenizer
    cfg: Config
    device: str = "cpu"
    name: str = "model"
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, spm_path: str, encoder: Optional[str] = None,
                        device: Optional[str] = None, name: str = "model") -> "Translator":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        enc = encoder or detect_encoder(state)

        tokenizer = Tokenizer(spm_path)
        if tokenizer.vocab_size == 0:
            raise FileNotFoundError(f"Không load được sentencepiece model: {spm_path}")

        ckpt_vocab = state["tok_embed.weight"].shape[0]
        if ckpt_vocab != tokenizer.vocab_size:
            # Vocab lệch = spm.model không phải cái đã train cùng checkpoint này. Nếu bỏ qua,
            # embedding vẫn load được (size khác -> lỗi) hoặc tệ hơn là decode ra token sai hoàn
            # toàn. Fail sớm với chỉ dẫn rõ.
            raise ValueError(
                f"Vocab lệch: checkpoint {ckpt_vocab} vs spm.model {tokenizer.vocab_size}.\n"
                f"-> Dùng đúng spm.model đã train kèm run này (evidence/phoenix{{5,10,25}}/spm.model)."
            )

        cfg = Config()
        cfg.model.encoder_type = enc
        cfg.device = device
        model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size,
                               pose_dim=cfg.data.pose_dim, encoder_type=enc)
        model.load_state_dict(state)
        model.eval().to(device)

        meta = {k: v for k, v in ckpt.items() if k != "model"} if isinstance(ckpt, dict) else {}
        meta["encoder"] = enc
        meta["ckpt"] = os.path.basename(ckpt_path)
        return cls(model=model, tokenizer=tokenizer, cfg=cfg, device=device, name=name, meta=meta)

    @torch.no_grad()
    def translate(self, pose: np.ndarray, decode: str = "beam", beam_size: int = 4,
                  max_len: Optional[int] = None) -> Tuple[str, float]:
        """pose [T,183] -> (câu, latency giây). B=1 nên không cần pose_mask (không có padding)."""
        max_len = max_len or self.cfg.data.max_text_len
        x = torch.from_numpy(np.ascontiguousarray(pose)).unsqueeze(0).to(self.device)
        tok = self.tokenizer
        t0 = time.time()
        if decode == "greedy":
            ys = self.model.greedy_decode(x, None, tok.bos_id, tok.eos_id, max_len=max_len)
        else:
            ys = self.model.beam_search_decode(x, None, tok.bos_id, tok.eos_id,
                                               max_len=max_len, beam_size=beam_size)
        dt = time.time() - t0
        return tok.decode(ys[0].tolist()).strip(), dt


# =======================================================================================
# 3. Tìm checkpoint
# =======================================================================================
def discover_runs(results_dir: str) -> dict:
    """Quét thư mục results/ -> {tên_run: {"xe": path, "scst": path}}.

    XE  = best_xe.pt.
    SCST= last_rl.pt (checkpoint CUỐI của phase RL). Cố ý KHÔNG ưu tiên best_rl.pt: ở nhiều run,
    SCST không vượt XE nên best_*.pt không tồn tại, và bản thân kết luận của báo cáo (RL phụ thuộc
    quy mô dữ liệu) nằm ở chính hành vi của model cuối phase RL.
    """
    runs = {}
    for xe_path in sorted(glob.glob(os.path.join(results_dir, "**", "best_xe.pt"), recursive=True)):
        d = os.path.dirname(xe_path)
        entry = {"xe": xe_path}
        for cand in ("last_rl.pt", "best_scst.pt", "best_rl.pt"):
            p = os.path.join(d, cand)
            if os.path.exists(p):
                entry["scst"] = p
                break
        runs[os.path.relpath(d, results_dir).replace("\\", "/")] = entry
    return runs


def find_spm(search_roots: List[str]) -> Optional[str]:
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", "spm.model"), recursive=True))
        if hits:
            return hits[0]
    return None
