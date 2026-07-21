"""
Entry point. Chạy 1 experiment cho 1 subset ratio.
Trên Kaggle: !python main.py --subset 0.25 --phase all
Để chạy TOÀN BỘ ma trận thí nghiệm cho 1 subset trong 1 lệnh, dùng run_all.py thay vì gọi file
này rời từng cái -- file này vẫn giữ nguyên để chạy tay 1 cấu hình đơn lẻ khi cần debug.

Phases:
    xe   : chỉ train cross-entropy
    rl   : chỉ train RL (--algo scst|ppo, cần XE checkpoint)
    all  : XE → RL
    eval : load checkpoint, eval test set

Kiến trúc encoder (--encoder transformer|stgcn) và thuật toán RL (--algo scst|ppo) chọn được qua
CLI để phục vụ Experiment 4 (Transformer vs GCN) và Experiment 7 (PPO vs SCST) mà không cần sửa
code — xem docs/1_Thuyet_Trinh_Tong_Hop.md §E.
"""
import argparse, os, sys, random, json
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import CFG
from data.tokenizer import build_tokenizer_from_train, Tokenizer
from data.dataset import make_loaders, find_annotation_csv
from models.slt_transformer import SLTTransformer
from training.train_xe import train_xe, evaluate
from training.train_scst import train_scst
from training.train_ppo import train_ppo
from training.train_mrt import train_mrt
from training.train_raml import train_raml
from training.train_dpo import train_dpo

ALGO_FNS = {"ppo": train_ppo, "mrt": train_mrt, "raml": train_raml, "dpo": train_dpo}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _ensure_tokenizer(cfg) -> Tokenizer:
    """Build tokenizer 1 lần (dùng chung mọi subset/mọi run) nếu chưa có."""
    spm_model = os.path.join(cfg.data.work_dir, "spm.model")
    if not os.path.exists(spm_model):
        # find_annotation_csv: KHÔNG hard-code "annotations/" -- khớp cả khi CSV để phẳng
        # (vd data/archive/*.corpus.csv). Đồng bộ với make_loaders để tokenizer không lệch loader.
        train_csv = find_annotation_csv(cfg.data.phoenix_root, "train")
        return build_tokenizer_from_train(train_csv, cfg.data.work_dir, cfg.data.vocab_size)
    return Tokenizer(spm_model)


def _merge_json(path: str, new_entries: dict):
    """Đọc JSON đã có (nếu tồn tại) rồi update thay vì ghi đè -- BUG cũ: main() tạo `results = {}`
    rỗng mỗi lần chạy phase eval, nên chạy --algo thứ 2 vào CÙNG log_dir (vd reward ablation tái
    dùng best_xe.pt) sẽ XOÁ MẤT kết quả --algo đầu tiên đã ghi trước đó. Cùng pattern với
    scripts/eval_baselines.py::_merge_results."""
    results = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
    results.update(new_entries)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_experiment(cfg, subset: float, encoder: str, algo: str, phase: str, tag: str = "exp",
                   xe_ckpt_override: str = None, tokenizer: Tokenizer = None) -> dict:
    """Chạy 1 experiment (đúng logic main() cũ, nay tách thành hàm để run_all.py gọi thẳng trong
    process thay vì subprocess !python -- tránh init lại CUDA/torch mỗi lần và cho phép orchestrator
    bọc try/except riêng từng bước).

    xe_ckpt_override: dùng checkpoint XE từ tag/log_dir KHÁC (vd core "run1") thay vì
    `log_dir/best_xe.pt` của chính tag này -- cần cho reward ablation: chạy --phase rl với 1 tag
    MỚI (để không ghi đè history của lần chạy reward mặc định) nhưng KHÔNG train lại XE.
    """
    cfg.model.encoder_type = encoder
    set_seed(cfg.seed)
    log_dir = os.path.join(cfg.data.work_dir, f"{tag}_{encoder}_subset{int(subset*100)}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log dir: {log_dir}")

    tokenizer = tokenizer or _ensure_tokenizer(cfg)
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_loader, dev_loader, test_loader = make_loaders(cfg, tokenizer, subset_ratio=subset)
    print(f"Train: {len(train_loader.dataset)}  Dev: {len(dev_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    def new_model():
        return SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                              encoder_type=encoder)

    model = new_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder: {encoder} | Params: {n_params/1e6:.2f}M")

    xe_ckpt = xe_ckpt_override or os.path.join(log_dir, "best_xe.pt")
    rl_ckpt = os.path.join(log_dir, f"best_{algo}.pt")  # best_scst.pt hoặc best_ppo.pt ...

    if phase in ("xe", "all"):
        print("\n=== Phase 1: XE pretraining ===")
        train_xe(model, train_loader, dev_loader, tokenizer, cfg, log_dir)

    if phase in ("rl", "all"):
        print(f"\n=== Phase 2: RL fine-tuning (algo={algo}) ===")
        if not os.path.exists(xe_ckpt):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint XE: {xe_ckpt}\n"
                f"-> Chạy --phase xe (hoặc all) trước, hoặc truyền xe_ckpt_override đúng."
            )
        # Tải lại model fresh và load checkpoint XE bên trong train_scst/train_ppo/...
        model = new_model()
        algo_fn = ALGO_FNS.get(algo)
        if algo_fn is not None:
            algo_fn(model, train_loader, dev_loader, tokenizer, cfg, log_dir, xe_ckpt)
        else:
            train_scst(model, train_loader, dev_loader, tokenizer, cfg, log_dir, xe_ckpt)
            # train_scst lưu ra "best_rl.pt" cố định (tương thích ngược) -> đồng bộ tên nếu cần
            legacy_path = os.path.join(log_dir, "best_rl.pt")
            if algo == "scst" and os.path.exists(legacy_path) and legacy_path != rl_ckpt:
                import shutil; shutil.copyfile(legacy_path, rl_ckpt)

    results = {}
    if phase in ("eval", "all"):
        print("\n=== Test evaluation ===")
        # Nếu RL không vượt XE thì best_{algo}.pt không tồn tại; rơi về last_rl.pt để vẫn
        # ĐO ĐƯỢC và BÁO CÁO ĐƯỢC trường hợp "RL kém hơn CE" (xem train_scst.py).
        eval_rl_ckpt = rl_ckpt
        if not os.path.exists(eval_rl_ckpt):
            fallback = os.path.join(log_dir, "last_rl.pt")
            if os.path.exists(fallback):
                print(f"[i] Không có {os.path.basename(eval_rl_ckpt)} (RL chưa vượt XE) "
                      f"-> eval bằng last_rl.pt để kết quả âm vẫn vào bảng so sánh.")
                eval_rl_ckpt = fallback
        for name, ckpt_path in [("xe", xe_ckpt), (algo, eval_rl_ckpt)]:
            if not os.path.exists(ckpt_path):
                continue
            ckpt = torch.load(ckpt_path, map_location=cfg.device)
            model = new_model()
            model.load_state_dict(ckpt["model"]); model = model.to(cfg.device)
            bleu, loss, samples = evaluate(model, test_loader, tokenizer, cfg)
            results[name] = {"test_bleu4": bleu, "test_loss": loss}
            print(f"[{name}] Test BLEU4 = {bleu:.2f}")
        _merge_json(os.path.join(log_dir, "test_results.json"), results)

    return {"log_dir": log_dir, "n_params": n_params, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=0.25)
    parser.add_argument("--phase", choices=["xe", "rl", "all", "eval"], default="all")
    parser.add_argument("--tag", type=str, default="exp")
    parser.add_argument("--encoder",
                        choices=["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"],
                        default="transformer",
                        help="Kiến trúc pose encoder P1-P6 (Experiment 4, docs/1_Thuyet_Trinh_Tong_Hop.md §A)")
    parser.add_argument("--algo", choices=["scst", "ppo", "mrt", "raml", "dpo"], default="scst",
                        help="Thuật toán RL phase 2 (docs/1_Thuyet_Trinh_Tong_Hop.md §C)")
    parser.add_argument("--xe_ckpt", type=str, default=None,
                        help="Đường dẫn best_xe.pt của tag KHÁC để tái dùng ở --phase rl/eval "
                             "(vd reward ablation) thay vì <log_dir>/best_xe.pt mặc định")
    args = parser.parse_args()

    run_experiment(CFG, args.subset, args.encoder, args.algo, args.phase, args.tag,
                  xe_ckpt_override=args.xe_ckpt)


if __name__ == "__main__":
    main()
