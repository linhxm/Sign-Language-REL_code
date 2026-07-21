"""Các baseline CƠ BẢN NHẤT cho bảng so sánh — mọi phương pháp trong comparison_table phải được
đọc TƯƠNG ĐỐI so với các "sàn" này, không phải con số tuyệt đối (docs/1_Thuyet_Trinh_Tong_Hop.md §E, docs/1_Thuyet_Trinh_Tong_Hop.md §F).

3 nhóm baseline (chọn qua --kind):

1. trivial   — sàn không-cần-model:
     - base_empty         : hypothesis rỗng cho mọi câu (sàn tuyệt đối của BLEU).
     - base_most_frequent : lặp lại câu train xuất hiện NHIỀU NHẤT cho mọi câu test. PHOENIX-2014T
       là domain thời tiết lặp nhiều — sàn này có thể > 0 đáng kể; model nào không vượt qua nó
       thì BLEU của model đó vô nghĩa.

2. selection — counter-baseline cho Experiment 8 / F.6-F.9 (train_selection_policy.py). Policy chọn
   frame chỉ được coi là "học được" nếu thắng lựa-chọn-không-học ở CÙNG keep_ratio, CÙNG cơ chế
   soft-mask (dùng lại _apply_frame_mask/_apply_landmark_mask — không tự viết lại để tránh lệch):
     - base_frames_full      : giữ 100% frame (sanity — phải ≈ test BLEU của checkpoint xe).
     - base_frames_random    : chọn ngẫu nhiên top-K theo score rand (cùng thủ tục top-K với
                               _evaluate_with_policy để số frame giữ lại khớp nhau từng sample).
     - base_frames_uniform   : chọn K frame cách đều (stride uniform) — heuristic mạnh nhất
                               không cần học; đây mới là đối thủ chính của policy.
     - base_drop_body/lhand/rhand : tắt hẳn từng nhóm landmark — sàn cho F.8 (landmark selection).

3. temp      — counter-baseline cho F.5 (train_decode_policy.py): sample_decode ở TỪNG nhiệt độ
   CỐ ĐỊNH trong cfg.train.decode_policy_temp_choices. Policy chọn temperature per-input phải
   thắng fixed-temp TỐT NHẤT thì mới có giá trị.

Kết quả merge vào <work_dir>/baseline_{encoder}_subset{pct}/test_results.json — đúng định dạng
mà scripts/aggregate_results.py đã quét, nên các dòng baseline TỰ xuất hiện trong
comparison_table.csv/.md không cần sửa gì thêm.

Usage (Kaggle):
    python scripts/eval_baselines.py --kind trivial   --subset 0.25
    python scripts/eval_baselines.py --kind selection --subset 0.25 --encoder transformer \
        --ckpt /kaggle/working/run1_transformer_subset25/best_xe.pt
    python scripts/eval_baselines.py --kind temp      --subset 0.25 --encoder transformer \
        --ckpt /kaggle/working/run1_transformer_subset25/best_xe.pt
"""
import argparse, os, sys, json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CFG
from data.tokenizer import Tokenizer
from data.dataset import make_loaders
from models.slt_transformer import SLTTransformer
from training.train_selection_policy import _apply_frame_mask, _apply_landmark_mask


def _merge_results(out_dir: str, new_entries: dict):
    """Merge (không ghi đè key khác) vào test_results.json của thư mục baseline."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "test_results.json")
    results = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
    results.update(new_entries)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Đã merge {len(new_entries)} baseline vào {path}")


def _corpus_bleu(hyps, refs):
    from sacrebleu import corpus_bleu
    return round(corpus_bleu(hyps, [refs]).score, 3)


# ---------------------------------------------------------------- kind=trivial
def eval_trivial(train_loader, test_loader):
    refs = [r for batch in test_loader for r in batch["text_raw"]]
    most_freq = train_loader.dataset.df["translation"].astype(str).value_counts().idxmax()
    print(f"Câu train phổ biến nhất ({int(train_loader.dataset.df['translation'].astype(str).value_counts().max())} lần): {most_freq!r}")
    return {
        "base_empty": {"test_bleu4": _corpus_bleu([""] * len(refs), refs)},
        "base_most_frequent": {"test_bleu4": _corpus_bleu([most_freq] * len(refs), refs),
                               "hypothesis": most_freq},
    }


# -------------------------------------------------------------- kind=selection
def _topk_keep_mask(scores, pose_mask, keep_ratio):
    """CÙNG thủ tục top-K với _evaluate_with_policy (train_selection_policy.py): Kmax theo batch,
    lọc lại bằng ~pose_mask — đảm bảo số frame giữ lại của baseline khớp cách policy được eval."""
    scores = scores.masked_fill(pose_mask, -1.0)
    valid_len = (~pose_mask).float().sum(dim=1)
    Kmax = max(1, int((valid_len * keep_ratio).clamp_min(1).long().max().item()))
    topk_idx = scores.topk(Kmax, dim=1).indices
    keep_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, topk_idx, True)
    return keep_mask & (~pose_mask)


def _uniform_scores(pose_mask):
    """Trả về scores sao cho topk(K) ≈ K vị trí CÁCH ĐỀU trên đoạn hợp lệ, đúng với MỌI K:
    xếp hạng theo thứ tự "midpoint" (2 đầu mút trước, rồi điểm giữa từng khoảng, BFS) —
    lấy top-K bất kỳ của thứ tự này luôn cho một lưới xấp xỉ đều (uniform stride)."""
    B, T = pose_mask.shape
    scores = torch.zeros(B, T)
    for i in range(B):
        valid = int((~pose_mask[i]).sum().item())
        if valid <= 0:
            continue
        chosen, seen, queue = [], set(), [(0, valid - 1)]
        for endpoint in (0, valid - 1):
            if endpoint not in seen:
                chosen.append(endpoint); seen.add(endpoint)
        while queue:
            lo, hi = queue.pop(0)
            mid = (lo + hi) // 2
            if mid not in seen:
                chosen.append(mid); seen.add(mid)
            if mid - lo > 1: queue.append((lo, mid))
            if hi - mid > 1: queue.append((mid, hi))
        for r, pos in enumerate(chosen):
            scores[i, pos] = valid - r
    return scores


@torch.no_grad()
def eval_selection(model, test_loader, tokenizer, cfg, keep_ratio, seed):
    device = cfg.device
    variants = ["base_frames_full", "base_frames_random", "base_frames_uniform",
                "base_drop_body", "base_drop_lhand", "base_drop_rhand"]
    hyps = {v: [] for v in variants}
    keep_frac_sum = {v: 0.0 for v in variants}
    refs, n_batches = [], 0
    g = torch.Generator().manual_seed(seed)

    for batch in test_loader:
        pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
        refs.extend(batch["text_raw"]); n_batches += 1
        valid_len = (~pose_mask).float().sum(dim=1).clamp_min(1)

        rand_keep = _topk_keep_mask(torch.rand(pose.shape[:2], generator=g).to(device),
                                    pose_mask, keep_ratio)
        unif_keep = _topk_keep_mask(_uniform_scores(pose_mask.cpu()).to(device),
                                    pose_mask, keep_ratio)
        full_keep = ~pose_mask

        group_masks = {}
        for j, name in enumerate(["base_drop_body", "base_drop_lhand", "base_drop_rhand"]):
            keep_group = torch.ones(pose.size(0), 3, dtype=torch.bool, device=device)
            keep_group[:, j] = False
            group_masks[name] = keep_group

        inputs = {
            "base_frames_full": _apply_frame_mask(pose, full_keep),
            "base_frames_random": _apply_frame_mask(pose, rand_keep),
            "base_frames_uniform": _apply_frame_mask(pose, unif_keep),
        }
        for name, kg in group_masks.items():
            inputs[name] = _apply_landmark_mask(pose, kg)
        for name, km in [("base_frames_full", full_keep), ("base_frames_random", rand_keep),
                         ("base_frames_uniform", unif_keep)]:
            keep_frac_sum[name] += (km.float().sum(dim=1) / valid_len).mean().item()
        for name in group_masks:
            keep_frac_sum[name] += 2.0 / 3.0  # giữ 2/3 nhóm landmark

        for name, filtered in inputs.items():
            gen = model.greedy_decode(filtered, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                                      max_len=cfg.data.max_text_len)
            hyps[name].extend(tokenizer.decode(gen[i].tolist()) for i in range(gen.size(0)))

    return {v: {"test_bleu4": _corpus_bleu(hyps[v], refs),
                "keep_frac": round(keep_frac_sum[v] / max(1, n_batches), 3)}
            for v in variants}


# ------------------------------------------------------------------- kind=temp
@torch.no_grad()
def eval_fixed_temps(model, test_loader, tokenizer, cfg, seed):
    device = cfg.device
    out = {}
    for t in cfg.train.decode_policy_temp_choices:
        torch.manual_seed(seed)  # cùng seed cho mọi temp — khác biệt chỉ đến từ temperature
        hyps, refs = [], []
        for batch in test_loader:
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            ys, _, _ = model.sample_decode(pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                                           max_len=cfg.data.max_text_len, temperature=t)
            hyps.extend(tokenizer.decode(ys[i].tolist()) for i in range(ys.size(0)))
            refs.extend(batch["text_raw"])
        out[f"base_fixed_temp_{t}"] = {"test_bleu4": _corpus_bleu(hyps, refs)}
        print(f"  fixed temp={t}: BLEU4={out[f'base_fixed_temp_{t}']['test_bleu4']}")
    return out


def run_baseline_trivial(subset: float, cfg=None) -> dict:
    """Wrapper mỏng quanh eval_trivial() để run_all.py gọi thẳng trong process (không subprocess).
    Build loader rồi merge kết quả vào baseline_data_subset{pct}/test_results.json."""
    cfg = cfg or CFG
    tokenizer = Tokenizer(os.path.join(cfg.data.work_dir, "spm.model"))
    train_loader, _, test_loader = make_loaders(cfg, tokenizer, subset_ratio=subset)
    out_dir = os.path.join(cfg.data.work_dir, f"baseline_data_subset{int(subset*100)}")
    entries = eval_trivial(train_loader, test_loader)
    _merge_results(out_dir, entries)
    return entries


def _load_ckpt_model(cfg, encoder: str, ckpt_path: str, tokenizer: Tokenizer):
    cfg.model.encoder_type = encoder
    model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                           encoder_type=encoder)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model = model.to(cfg.device).eval()
    print(f"Loaded {ckpt_path} (BLEU dev gốc={ckpt.get('bleu', float('nan')):.2f})")
    return model


def run_baseline_selection(subset: float, encoder: str, ckpt: str, keep_ratio: float = None,
                           cfg=None) -> dict:
    """Wrapper mỏng quanh eval_selection() -- counter-baseline cho Experiment 8 (F.6/F.8/F.9)."""
    cfg = cfg or CFG
    tokenizer = Tokenizer(os.path.join(cfg.data.work_dir, "spm.model"))
    _, _, test_loader = make_loaders(cfg, tokenizer, subset_ratio=subset)
    model = _load_ckpt_model(cfg, encoder, ckpt, tokenizer)
    keep_ratio = keep_ratio if keep_ratio is not None else cfg.train.selection_policy_keep_ratio
    out_dir = os.path.join(cfg.data.work_dir, f"baseline_{encoder}_subset{int(subset*100)}")
    entries = eval_selection(model, test_loader, tokenizer, cfg, keep_ratio, cfg.seed)
    _merge_results(out_dir, entries)
    return entries


def run_baseline_temp(subset: float, encoder: str, ckpt: str, cfg=None) -> dict:
    """Wrapper mỏng quanh eval_fixed_temps() -- counter-baseline cho decode policy (F.5)."""
    cfg = cfg or CFG
    tokenizer = Tokenizer(os.path.join(cfg.data.work_dir, "spm.model"))
    _, _, test_loader = make_loaders(cfg, tokenizer, subset_ratio=subset)
    model = _load_ckpt_model(cfg, encoder, ckpt, tokenizer)
    out_dir = os.path.join(cfg.data.work_dir, f"baseline_{encoder}_subset{int(subset*100)}")
    entries = eval_fixed_temps(model, test_loader, tokenizer, cfg, cfg.seed)
    _merge_results(out_dir, entries)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["trivial", "selection", "temp"], required=True)
    ap.add_argument("--subset", type=float, default=0.25)
    ap.add_argument("--encoder",
                    choices=["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"],
                    default="transformer")
    ap.add_argument("--ckpt", default=None, help="best_xe.pt (bắt buộc với kind=selection/temp)")
    ap.add_argument("--keep_ratio", type=float, default=None,
                    help="mặc định = cfg.train.selection_policy_keep_ratio (khớp policy)")
    args = ap.parse_args()

    tokenizer = Tokenizer(os.path.join(CFG.data.work_dir, "spm.model"))
    train_loader, _, test_loader = make_loaders(CFG, tokenizer, subset_ratio=args.subset)

    if args.kind == "trivial":
        out_dir = os.path.join(CFG.data.work_dir, f"baseline_data_subset{int(args.subset*100)}")
        _merge_results(out_dir, eval_trivial(train_loader, test_loader))
        return

    assert args.ckpt and os.path.exists(args.ckpt), "--ckpt (best_xe.pt) bắt buộc với kind này"
    CFG.model.encoder_type = args.encoder
    model = SLTTransformer(CFG, vocab_size=tokenizer.vocab_size, pose_dim=CFG.data.pose_dim,
                           encoder_type=args.encoder)
    ckpt = torch.load(args.ckpt, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    model = model.to(CFG.device).eval()
    print(f"Loaded {args.ckpt} (BLEU dev gốc={ckpt.get('bleu', float('nan')):.2f})")

    out_dir = os.path.join(CFG.data.work_dir,
                           f"baseline_{args.encoder}_subset{int(args.subset*100)}")
    if args.kind == "selection":
        keep_ratio = args.keep_ratio if args.keep_ratio is not None \
            else CFG.train.selection_policy_keep_ratio
        _merge_results(out_dir, eval_selection(model, test_loader, tokenizer, CFG,
                                               keep_ratio, CFG.seed))
    else:
        _merge_results(out_dir, eval_fixed_temps(model, test_loader, tokenizer, CFG, CFG.seed))


if __name__ == "__main__":
    main()
