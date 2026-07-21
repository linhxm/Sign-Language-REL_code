"""Ý tưởng F.14 (RL for sign segmentation, docs/1_Thuyet_Trinh_Tong_Hop.md §F) — hiện thực hoá
bằng forced-alignment CTC thay vì 1 policy RL riêng: model CTC đã train (training/train_ctc_gloss.py,
P7) tự học 1 alignment ẩn frame<->gloss trong lúc tối ưu CTC loss; đọc argmax mỗi frame TRƯỚC KHI
collapse lặp/blank (khác `_ctc_greedy_decode` dùng để tính WER) cho trực tiếp ranh giới segment gần
đúng — không cần thêm vòng lặp RL "reward qua downstream" như mô tả gốc của ý tưởng, nhưng cùng mục
tiêu: chia chuỗi pose liên tục thành các đoạn ứng với từng gloss, phục vụ phân tích "co-articulation"
([1_Tong_Quan.md](../docs/1_Thuyet_Trinh_Tong_Hop.md)) hoặc làm input cho hướng nghiên cứu mở rộng sau này
(reward-via-downstream RL segmentation thật sự, nêu ở docs/1_Thuyet_Trinh_Tong_Hop.md §H §10.7).

Dùng: python scripts/extract_gloss_segments.py --ckpt <best_ctc.pt> --encoder transformer \
    --gloss_vocab <gloss_vocab.json> --pose_dir <poses> --csv <PHOENIX-2014-T.dev.corpus.csv> \
    --out segments.json
"""
import argparse, os, json, sys
import numpy as np
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CFG
from data.gloss_vocab import GlossVocab
from models.gloss_ctc_head import GlossCTCModel


def _segments_from_framewise(frame_labels, blank_id: int = 0):
    """frame_labels: list[int] argmax mỗi frame (CHƯA collapse). Trả list (start, end, label)
    theo từng RUN liên tục cùng nhãn non-blank (blank = ranh giới, không sinh segment)."""
    segments = []
    start = None
    prev = None
    for t, lab in enumerate(frame_labels):
        if lab == blank_id:
            if start is not None:
                segments.append((start, t - 1, prev))
                start = None
            prev = None
            continue
        if lab != prev:
            if start is not None:
                segments.append((start, t - 1, prev))
            start = t
            prev = lab
    if start is not None:
        segments.append((start, len(frame_labels) - 1, prev))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="transformer",
                    choices=["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"])
    ap.add_argument("--gloss_vocab", required=True)
    ap.add_argument("--pose_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="segments.json")
    ap.add_argument("--max_videos", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gloss_vocab = GlossVocab.load(args.gloss_vocab)
    CFG.model.encoder_type = args.encoder
    model = GlossCTCModel(CFG, gloss_vocab.vocab_size, CFG.data.pose_dim, args.encoder)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"]); model = model.to(device).eval()

    df = pd.read_csv(args.csv, sep="|").dropna(subset=["orth"]).reset_index(drop=True)
    if len(df) > args.max_videos:
        df = df.iloc[:args.max_videos]

    results = {}
    with torch.no_grad():
        for _, row in df.iterrows():
            name = row["name"]
            pose_path = os.path.join(args.pose_dir, f"{name}.npz")
            if not os.path.exists(pose_path):
                continue
            pose = np.load(pose_path)["pose"].astype(np.float32)
            if len(pose) > CFG.data.max_frames:
                idxs = np.linspace(0, len(pose) - 1, CFG.data.max_frames).astype(int)
                pose = pose[idxs]
            pose_t = torch.from_numpy(pose).unsqueeze(0).to(device)
            log_probs = model(pose_t, None)  # [1,T,V]
            frame_labels = log_probs.argmax(dim=-1)[0].tolist()
            segs = _segments_from_framewise(frame_labels, gloss_vocab.BLANK)
            results[name] = [
                {"start_frame": s, "end_frame": e, "gloss": gloss_vocab.id2token.get(lab, "<unk>")}
                for s, e, lab in segs
            ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {len(results)} video -> {args.out}")


if __name__ == "__main__":
    main()
