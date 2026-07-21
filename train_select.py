"""Chọn PHẠM VI train qua --mode — để không phải chạy cả ma trận (run_all.py) khi chỉ cần 1 phần.
Tất cả tái dùng run_experiment() của main.py (XE checkpoint được tái dùng đúng chỗ), cộng marker
`.done_*` để chạy lại là bỏ qua bước đã xong trong cùng session.

  --mode single        : 1 encoder + 1 algo          (XE -> RL -> eval)
  --mode encoder_allrl : 1 encoder x 5 thuật toán RL  (XE train 1 LẦN, 5 RL dùng chung best_xe.pt)
  --mode rl_allenc     : 1 algo   x 6 encoder         (mỗi encoder tự train XE + algo đó)
  --mode all           : toàn bộ ma trận              (uỷ quyền cho run_all.py)

Ví dụ:
  python train_select.py --mode single       --encoder transformer --algo scst --subset 1.0
  python train_select.py --mode encoder_allrl --encoder stgcn                  --subset 0.25
  python train_select.py --mode rl_allenc                        --algo ppo   --subset 0.25
  python train_select.py --mode all                                           --subset 1.0

  # 2 run "train từng phần" ở 10%: (1) CE-vs-SCST trên transformer (reward mặc định),
  # (2) encoder tốt nhất (TCN) + SCST, reward BLEU thuần (số 5% cho thấy đây là reward tốt nhất
  #     trong 4 combo; penalty đều làm tệ hơn ở scale nhỏ):
  python train_select.py --mode single --encoder transformer --algo scst --subset 0.10 --tag run1
  python train_select.py --mode single --encoder tcn --algo scst --subset 0.10 --tag run1 \
         --w_bleu 1.0 --w_rep 0.0 --w_len 0.0

Tái dùng best_xe.pt qua nhiều session (cho subset 100% train không hết 1 session):
  # session 1: train XE (+scst) rồi tải best_xe.pt về, up thành dataset
  python train_select.py --mode single --encoder transformer --algo scst --subset 1.0
  # session 2: chạy các algo còn lại, KHÔNG train lại XE
  python train_select.py --mode encoder_allrl --encoder transformer --subset 1.0 \
         --xe_ckpt /kaggle/input/<dataset>/best_xe.pt
"""
import argparse, os, sys, time, subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.config import CFG
from main import run_experiment, _ensure_tokenizer

ALL_ENCODERS = ["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"]
ALL_ALGOS = ["scst", "ppo", "mrt", "raml", "dpo"]
OTHER_ALGOS = ["ppo", "mrt", "raml", "dpo"]   # scst = mặc định, đã chạy ở phase "all"


def _step(key, log_dir, fn, *a, **kw):
    """Chạy 1 bước, tự bỏ qua nếu marker đã có (resume trong cùng session), có đo giờ."""
    os.makedirs(log_dir, exist_ok=True)
    marker = os.path.join(log_dir, f".done_{key}")
    if os.path.exists(marker):
        print(f"[SKIP] {key} (đã xong)")
        return
    print(f"\n{'='*72}\n[RUN] {key}\n  log_dir = {log_dir}\n{'='*72}")
    t0 = time.time()
    fn(*a, **kw)
    open(marker, "w").close()
    print(f"[OK] {key} — {(time.time()-t0)/60:.1f} phút")


def _ld(cfg, tag, enc, subset):
    return os.path.join(cfg.data.work_dir, f"{tag}_{enc}_subset{int(subset*100)}")


def do_single(cfg, enc, algo, subset, tag, tok, xe_ckpt):
    ld = _ld(cfg, tag, enc, subset)
    _step(f"{enc}_{algo}_all", ld, run_experiment,
          cfg, subset, enc, algo, "all", tag, xe_ckpt_override=xe_ckpt, tokenizer=tok)


def do_encoder_allrl(cfg, enc, subset, tag, tok, xe_ckpt):
    ld = _ld(cfg, tag, enc, subset)
    if xe_ckpt:
        # Đã có best_xe.pt từ session trước -> chỉ chạy RL cho cả 5 algo (kể cả scst), KHÔNG train XE.
        for algo in ALL_ALGOS:
            _step(f"{enc}_{algo}_rl", ld, run_experiment,
                  cfg, subset, enc, algo, "rl", tag, xe_ckpt_override=xe_ckpt, tokenizer=tok)
            _step(f"{enc}_{algo}_eval", ld, run_experiment,
                  cfg, subset, enc, algo, "eval", tag, xe_ckpt_override=xe_ckpt, tokenizer=tok)
    else:
        # XE + scst (phase all) tạo best_xe.pt, rồi 4 algo còn lại tái dùng chính nó.
        _step(f"{enc}_scst_all", ld, run_experiment,
              cfg, subset, enc, "scst", "all", tag, tokenizer=tok)
        for algo in OTHER_ALGOS:
            _step(f"{enc}_{algo}_rl", ld, run_experiment,
                  cfg, subset, enc, algo, "rl", tag, tokenizer=tok)
            _step(f"{enc}_{algo}_eval", ld, run_experiment,
                  cfg, subset, enc, algo, "eval", tag, tokenizer=tok)


def do_rl_allenc(cfg, algo, subset, tag, tok):
    for enc in ALL_ENCODERS:
        ld = _ld(cfg, tag, enc, subset)
        _step(f"{enc}_{algo}_all", ld, run_experiment,
              cfg, subset, enc, algo, "all", tag, tokenizer=tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["single", "encoder_allrl", "rl_allenc", "all"])
    ap.add_argument("--encoder", default="transformer", choices=ALL_ENCODERS)
    ap.add_argument("--algo", default="scst", choices=ALL_ALGOS)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--xe_ckpt", default=None,
                    help="Tái dùng best_xe.pt có sẵn (vd từ dataset) thay vì train lại XE — "
                         "dùng cho single/encoder_allrl khi tách XE và RL ra 2 session.")
    # --- reward override (không cần sửa config.py/.ipynb) — chọn 1 trong 4 reward cũ cho run này.
    #     4 reward = tổ hợp bật/tắt (w_rep, w_len) ∈ {0, 0.5}. KHÔNG dùng BERTScore. ---
    ap.add_argument("--w_bleu", type=float, default=None, help="Ghi đè reward_bleu_weight (mặc định 1.0)")
    ap.add_argument("--w_rep",  type=float, default=None,
                    help="Ghi đè reward_repetition_penalty (0 hoặc 0.5). Số liệu 5%: rep≈0 nên penalty "
                         "này gần như không giúp.")
    ap.add_argument("--w_len",  type=float, default=None,
                    help="Ghi đè reward_length_penalty (0 hoặc 0.5). Số liệu 5%: len penalty làm câu "
                         "ngắn thêm và GIẢM BLEU -> nên để 0.0.")
    args = ap.parse_args()

    if args.mode == "all":
        subprocess.run([sys.executable, "run_all.py", "--subset", str(args.subset)], check=True)
        return

    cfg = CFG
    # Ghi đè reward weights từ CLI (nếu có) TRƯỚC khi build tokenizer/model — áp cho mọi run RL.
    for attr, val in [("reward_bleu_weight", args.w_bleu),
                      ("reward_repetition_penalty", args.w_rep),
                      ("reward_length_penalty", args.w_len)]:
        if val is not None:
            setattr(cfg.train, attr, val)

    tok = _ensure_tokenizer(cfg)
    t0 = time.time()
    print(f"### mode={args.mode} encoder={args.encoder} algo={args.algo} "
          f"subset={args.subset} tag={args.tag} ###")
    print(f"### reward: w_bleu={cfg.train.reward_bleu_weight} w_rep={cfg.train.reward_repetition_penalty} "
          f"w_len={cfg.train.reward_length_penalty} ###")

    if args.mode == "single":
        do_single(cfg, args.encoder, args.algo, args.subset, args.tag, tok, args.xe_ckpt)
    elif args.mode == "encoder_allrl":
        do_encoder_allrl(cfg, args.encoder, args.subset, args.tag, tok, args.xe_ckpt)
    elif args.mode == "rl_allenc":
        do_rl_allenc(cfg, args.algo, args.subset, args.tag, tok)

    print(f"\n### XONG mode={args.mode} sau {(time.time()-t0)/60:.1f} phút ###")


if __name__ == "__main__":
    main()
