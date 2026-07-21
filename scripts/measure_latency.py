"""Đo Latency / FPS / Memory / Parameters (Experiment 15, docs/1_Thuyet_Trinh_Tong_Hop.md §E;
nhóm metric hệ thống, docs/1_Thuyet_Trinh_Tong_Hop.md §F §9.2).

Usage:
    python scripts/measure_latency.py --ckpt /kaggle/working/run1_subset25/best_xe.pt \
        --encoder transformer --vocab_size 3000 --batch_sizes 1,16 --n_runs 30

Không cần dataset thật — dùng pose ngẫu nhiên (chỉ đo chi phí compute, không đo chất lượng).
Chạy trên đúng GPU dùng để train (T4) để số liệu so sánh được nhất quán giữa các kiến trúc.
"""
import argparse, os, sys, time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import CFG
from models.slt_transformer import SLTTransformer


def measure(model, pose_dim, max_frames, max_text_len, bos_id, eos_id,
            batch_size, n_runs, device, decode_fn="greedy"):
    pose = torch.randn(batch_size, max_frames, pose_dim, device=device)
    pose_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=device)

    # warm-up (loại bỏ chi phí khởi tạo CUDA context/cuDNN autotune khỏi phép đo)
    for _ in range(3):
        with torch.no_grad():
            model.greedy_decode(pose, pose_mask, bos_id, eos_id, max_len=max_text_len)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model.greedy_decode(pose, pose_mask, bos_id, eos_id, max_len=max_text_len)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    latency_ms_per_batch = (elapsed / n_runs) * 1000
    latency_ms_per_sentence = latency_ms_per_batch / batch_size
    fps = batch_size * n_runs / elapsed  # câu/giây (throughput)
    mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": round(latency_ms_per_batch, 2),
        "latency_ms_per_sentence": round(latency_ms_per_sentence, 2),
        "throughput_sentences_per_s": round(fps, 2),
        "peak_memory_mb": round(mem_mb, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Đường dẫn checkpoint (best_xe.pt/best_rl.pt/best_ppo.pt)")
    p.add_argument("--encoder",
                  choices=["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"],
                  default="transformer")
    p.add_argument("--vocab_size", type=int, default=3000)
    p.add_argument("--pose_dim", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--max_text_len", type=int, default=None)
    p.add_argument("--batch_sizes", type=str, default="1,16")
    p.add_argument("--n_runs", type=int, default=30)
    args = p.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    pose_dim = args.pose_dim or CFG.data.pose_dim
    max_frames = args.max_frames or CFG.data.max_frames
    max_text_len = args.max_text_len or CFG.data.max_text_len

    CFG.model.encoder_type = args.encoder
    model = SLTTransformer(CFG, vocab_size=args.vocab_size, pose_dim=pose_dim,
                           encoder_type=args.encoder)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"Encoder: {args.encoder} | Params: {n_params/1e6:.2f}M | Device: {device}")

    results = []
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        r = measure(model, pose_dim, max_frames, max_text_len,
                   bos_id=1, eos_id=2, batch_size=bs, n_runs=args.n_runs, device=device)
        r["n_params"] = n_params
        r["encoder"] = args.encoder
        results.append(r)
        print(r)

    out_path = os.path.join(os.path.dirname(args.ckpt), f"latency_{args.encoder}.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Đã lưu: {out_path}")


if __name__ == "__main__":
    main()
