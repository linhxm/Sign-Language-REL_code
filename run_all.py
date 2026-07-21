"""
Orchestrator — chạy TOÀN BỘ ma trận thí nghiệm (baseline sàn + Exp.1/4/7/8/9/11/15 + P7 + ablation)
cho MỘT subset ratio, trong MỘT process Python duy nhất (không gọi rời `!python ...` từng cell).

Đây là điểm vào DUY NHẤT khuyến nghị dùng trên Kaggle:

    python run_all.py --subset 0.25
    python run_all.py --subset 0.5      # chạy khi nào sẵn sàng, không bắt buộc ngay sau 0.25
    python run_all.py --subset 1.0      # chạy sau cùng — có thể cần NHIỀU session (xem dưới)

RESUMABLE: mỗi bước con ghi 1 file marker `<log_dir>/.done_<key>` khi xong. Nếu Kaggle hết giờ
session (~12h) giữa chừng, chỉ cần bấm chạy LẠI ĐÚNG LỆNH TRÊN — mọi bước đã xong tự động bị bỏ
qua, chỉ bước dở dang/còn lại chạy tiếp. Vì epoch KHÔNG giảm theo subset (xe_epochs=80/rl_epochs=20
cố định, đã xác nhận), subset 100% nhiều khả năng cần >1 session — đây là hành vi DỰ KIẾN, không
phải lỗi.

Một bước lỗi (vd `graph_transformer` OOM — đã cảnh báo trong docs) chỉ bị log lại và bỏ qua, KHÔNG
làm hỏng toàn bộ ma trận còn lại — xem hàm `step()`.

Muốn giới hạn phạm vi (vd chỉ core, chạy nhanh để kiểm tra luồng), dùng --groups:
    python run_all.py --subset 0.25 --groups core,encoders
Mặc định --groups all = chạy hết (đúng tinh thần "1 lệnh cho tất cả" của toàn bộ ma trận).

Cuối MỖI lần chạy (dù --groups gì, dù có lỗi ở vài bước), tự động sinh lại:
    <work_dir>/comparison_table.csv/.md   -- 1 bảng gộp mọi run (scripts/aggregate_results.py)
    <work_dir>/report/tables/*.csv/.md/.tex -- 6 bảng đã lọc theo từng câu hỏi so sánh + 3 bảng
                                              LaTeX dán thẳng vào paper (scripts/make_report.py)
    <work_dir>/report/figures/*.png/.pdf  -- 5 biểu đồ so sánh
Không cần chạy thêm lệnh nào khác để có bảng/hình dùng viết báo cáo.
"""
import argparse, os, sys, time, json, traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import CFG
from data.tokenizer import Tokenizer
from data.dataset import make_loaders
from models.slt_transformer import SLTTransformer
from main import run_experiment, _ensure_tokenizer, set_seed
from main_twostage import run_twostage
from training.train_scst import train_scst
from training.train_ppo import train_ppo
from training.train_xe import evaluate
from training.train_selection_policy import train_selection_policy
from training.train_decode_policy import train_decode_policy
from scripts.eval_baselines import run_baseline_trivial, run_baseline_selection, run_baseline_temp
from scripts import aggregate_results as agg
from scripts.measure_latency import measure
from scripts.make_report import generate_report

ALL_ENCODERS = ["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"]
OTHER_ALGOS = ["ppo", "mrt", "raml", "dpo"]
GROUP_ORDER = ["core", "encoders", "algos", "ablations", "reward", "twostage",
              "selection", "decode", "latency"]


# --------------------------------------------------------------------------- resumable step runner
def step(key: str, log_dir: str, fn, *a, **kw):
    """Chạy 1 bước con, tự skip nếu marker đã tồn tại, tự bắt lỗi để 1 bước hỏng không phá cả ma
    trận (vd graph_transformer OOM). `key` phải DUY NHẤT trong `log_dir` (vd "xe_scst_all",
    "ppo_rl", "ppo_eval") vì nhiều bước có thể dùng CHUNG log_dir (algo khác nhau, cùng thư mục
    tag "run1" để tái dùng best_xe.pt)."""
    os.makedirs(log_dir, exist_ok=True)
    marker = os.path.join(log_dir, f".done_{key}")
    if os.path.exists(marker):
        print(f"[SKIP] {key} (đã xong — {marker})")
        return True
    print(f"\n{'='*78}\n[RUN] {key}\n  log_dir = {log_dir}\n{'='*78}")
    t0 = time.time()
    try:
        fn(*a, **kw)
        open(marker, "w").close()
        print(f"[OK] {key} ({(time.time()-t0)/60:.1f} phút)")
        return True
    except Exception as e:
        print(f"[LỖI] {key}: {e!r}")
        traceback.print_exc()
        print(f"[!] Bỏ qua {key}, tiếp tục các bước kế tiếp (không phá toàn bộ ma trận).")
        return False


def _eval_and_merge(log_dir, ckpt_path, key, cfg, tokenizer, test_loader, encoder="transformer"):
    """Load 1 checkpoint rời (ablation REINFORCE/A2C/Curriculum không đi qua run_experiment nên
    không tự có test_results.json) rồi merge kết quả test BLEU4 vào test_results.json của log_dir,
    cùng convention với main.py::_merge_json / scripts/eval_baselines.py::_merge_results."""
    import torch
    if not os.path.exists(ckpt_path):
        print(f"[i] Không có {ckpt_path} — bỏ qua eval cho {key} (RL có thể chưa vượt XE ở đây).")
        return
    cfg.model.encoder_type = encoder
    model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                           encoder_type=encoder)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"]); model = model.to(cfg.device)
    bleu, loss, _ = evaluate(model, test_loader, tokenizer, cfg)
    print(f"[{key}] Test BLEU4 = {bleu:.2f}")
    path = os.path.join(log_dir, "test_results.json")
    results = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
    results[key] = {"test_bleu4": bleu, "test_loss": loss}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _run_latency(encoder, ckpt_path, tokenizer, cfg):
    import torch
    cfg.model.encoder_type = encoder
    model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                           encoder_type=encoder)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"]); model = model.to(cfg.device).eval()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    results = []
    for bs in (1, 16):
        r = measure(model, cfg.data.pose_dim, cfg.data.max_frames, cfg.data.max_text_len,
                   bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, batch_size=bs, n_runs=30,
                   device=device)
        r["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        r["encoder"] = encoder
        results.append(r)
        print(r)
    out_path = os.path.join(os.path.dirname(ckpt_path), f"latency_{encoder}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------------------- nhóm thí nghiệm
def run_core(cfg, subset, pct, wd, tokenizer):
    log_dir = os.path.join(wd, f"run1_transformer_subset{pct}")
    step("xe_scst_all", log_dir, run_experiment, cfg, subset, "transformer", "scst", "all",
        "run1", tokenizer=tokenizer)
    return log_dir


def run_encoders(cfg, subset, pct, wd, tokenizer):
    for enc in ALL_ENCODERS:
        if enc == "transformer":
            continue  # đã chạy ở run_core
        log_dir = os.path.join(wd, f"run1_{enc}_subset{pct}")
        step(f"xe_scst_all_{enc}", log_dir, run_experiment, cfg, subset, enc, "scst", "all",
            "run1", tokenizer=tokenizer)


def run_algos(cfg, subset, pct, wd, core_log_dir, tokenizer):
    for algo in OTHER_ALGOS:
        step(f"{algo}_rl", core_log_dir, run_experiment, cfg, subset, "transformer", algo, "rl",
            "run1", tokenizer=tokenizer)
        step(f"{algo}_eval", core_log_dir, run_experiment, cfg, subset, "transformer", algo, "eval",
            "run1", tokenizer=tokenizer)


def run_ablations(cfg, subset, pct, wd, core_xe_ckpt, tokenizer, train_loader, dev_loader, test_loader):
    def _reinforce():
        cfg.train.rl_use_baseline = False
        try:
            log_dir = os.path.join(wd, f"run1_transformer_subset{pct}_reinforce_nobaseline")
            set_seed(cfg.seed)
            model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                                   encoder_type="transformer")
            train_scst(model, train_loader, dev_loader, tokenizer, cfg, log_dir, core_xe_ckpt)
            best = os.path.join(log_dir, "best_rl.pt")
            ckpt = best if os.path.exists(best) else os.path.join(log_dir, "last_rl.pt")
            _eval_and_merge(log_dir, ckpt, "reinforce_nobaseline", cfg, tokenizer, test_loader)
        finally:
            cfg.train.rl_use_baseline = True

    def _a2c():
        cfg.train.ppo_use_clip = False
        try:
            log_dir = os.path.join(wd, f"run1_transformer_subset{pct}_a2c")
            set_seed(cfg.seed)
            model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                                   encoder_type="transformer")
            train_ppo(model, train_loader, dev_loader, tokenizer, cfg, log_dir, core_xe_ckpt)
            _eval_and_merge(log_dir, os.path.join(log_dir, "best_ppo.pt"), "a2c", cfg, tokenizer,
                            test_loader)
        finally:
            cfg.train.ppo_use_clip = True

    def _curriculum():
        cfg.train.rl_curriculum_epochs = 5
        cfg.train.rl_curriculum_length_sort = True
        try:
            log_dir = os.path.join(wd, f"run1_transformer_subset{pct}_curriculum")
            set_seed(cfg.seed)
            model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                                   encoder_type="transformer")
            train_scst(model, train_loader, dev_loader, tokenizer, cfg, log_dir, core_xe_ckpt)
            best = os.path.join(log_dir, "best_rl.pt")
            ckpt = best if os.path.exists(best) else os.path.join(log_dir, "last_rl.pt")
            _eval_and_merge(log_dir, ckpt, "curriculum", cfg, tokenizer, test_loader)
        finally:
            cfg.train.rl_curriculum_epochs = 0
            cfg.train.rl_curriculum_length_sort = False

    step("reinforce_nobaseline", os.path.join(wd, f"run1_transformer_subset{pct}_reinforce_nobaseline"),
        _reinforce)
    step("a2c", os.path.join(wd, f"run1_transformer_subset{pct}_a2c"), _a2c)
    step("curriculum", os.path.join(wd, f"run1_transformer_subset{pct}_curriculum"), _curriculum)


REWARD_COMBOS = [  # (tag, rep_penalty, len_penalty) -- xem docs/1_Thuyet_Trinh_Tong_Hop.md §K.2
    ("rw_bleu_only", 0.0, 0.0),
    ("rw_default",   0.5, 0.0),
    ("rw_len_only",  0.0, 0.5),
    ("rw_both",      0.5, 0.5),
]


def run_reward_ablation(cfg, subset, pct, wd, core_xe_ckpt, tokenizer):
    orig_rep = cfg.train.reward_repetition_penalty
    orig_len = cfg.train.reward_length_penalty
    try:
        for tag, rep_w, len_w in REWARD_COMBOS:
            cfg.train.reward_repetition_penalty = rep_w
            cfg.train.reward_length_penalty = len_w
            log_dir = os.path.join(wd, f"{tag}_transformer_subset{pct}")
            step(f"{tag}_rl", log_dir, run_experiment, cfg, subset, "transformer", "scst", "rl",
                tag, xe_ckpt_override=core_xe_ckpt, tokenizer=tokenizer)
            step(f"{tag}_eval", log_dir, run_experiment, cfg, subset, "transformer", "scst", "eval",
                tag, xe_ckpt_override=core_xe_ckpt, tokenizer=tokenizer)
    finally:
        cfg.train.reward_repetition_penalty = orig_rep
        cfg.train.reward_length_penalty = orig_len


def run_twostage_group(cfg, subset, pct, wd):
    log_dir = os.path.join(wd, f"p7_twostage_transformer_subset{pct}")
    step("p7_twostage", log_dir, run_twostage, cfg, subset, "transformer", "p7_twostage")


def run_selection_group(cfg, subset, pct, wd, core_xe_ckpt, tokenizer, train_loader, dev_loader):
    baseline_dir = os.path.join(wd, f"baseline_transformer_subset{pct}")
    step("baseline_selection", baseline_dir, run_baseline_selection, subset, "transformer",
        core_xe_ckpt)

    variants = [("frame", "topk"), ("frame", "adaptive"), ("landmark", "topk")]
    log_dir = os.path.join(wd, f"run1_transformer_subset{pct}_selectpolicy")
    for target, mode in variants:
        def _fn(target=target, mode=mode):
            model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                                   encoder_type="transformer")
            train_selection_policy(model, train_loader, dev_loader, tokenizer, cfg, log_dir,
                                   core_xe_ckpt, target=target, mode=mode)
        step(f"selectpolicy_{target}_{mode}", log_dir, _fn)


def run_decode_group(cfg, subset, pct, wd, core_xe_ckpt, tokenizer, train_loader, dev_loader):
    baseline_dir = os.path.join(wd, f"baseline_transformer_subset{pct}")
    step("baseline_temp", baseline_dir, run_baseline_temp, subset, "transformer", core_xe_ckpt)

    log_dir = os.path.join(wd, f"run1_transformer_subset{pct}_decodepolicy")
    def _fn():
        model = SLTTransformer(cfg, vocab_size=tokenizer.vocab_size, pose_dim=cfg.data.pose_dim,
                               encoder_type="transformer")
        train_decode_policy(model, train_loader, dev_loader, tokenizer, cfg, log_dir, core_xe_ckpt)
    step("decodepolicy", log_dir, _fn)


def run_latency_group(cfg, subset, pct, wd, tokenizer):
    for enc in ALL_ENCODERS:
        ckpt = os.path.join(wd, f"run1_{enc}_subset{pct}", "best_xe.pt")
        log_dir = os.path.dirname(ckpt)
        if not os.path.exists(ckpt):
            print(f"[i] Bỏ qua latency {enc}: chưa có {ckpt}")
            continue
        step(f"latency_{enc}", log_dir, _run_latency, enc, ckpt, tokenizer, cfg)


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=float, required=True, choices=[0.05, 0.25, 0.5, 1.0],
                    help="Tỉ lệ train subset -- 0.05 (smoke-test toàn ma trận, KHÔNG dùng cho kết quả) "
                         "/ 0.25/0.5/1.0 (chạy tay khi bạn sẵn sàng)")
    ap.add_argument("--groups", type=str, default="all",
                    help="Danh sách nhóm cách nhau bởi dấu phẩy, hoặc 'all' (mặc định). "
                         f"Nhóm hợp lệ: {','.join(GROUP_ORDER)}")
    args = ap.parse_args()

    groups = GROUP_ORDER if args.groups == "all" else [g.strip() for g in args.groups.split(",")]
    unknown = set(groups) - set(GROUP_ORDER)
    if unknown:
        raise SystemExit(f"Nhóm không hợp lệ: {unknown} -- xem --help")

    cfg = CFG
    subset = args.subset
    pct = int(subset * 100)
    wd = cfg.data.work_dir
    os.makedirs(wd, exist_ok=True)
    t_start = time.time()
    print(f"\n{'#'*78}\n# run_all.py --subset {subset} ({pct}%)  groups={groups}\n{'#'*78}")

    tokenizer = _ensure_tokenizer(cfg)

    # Sàn không-cần-model -- rẻ, luôn chạy trước (marker tự bỏ qua nếu đã có).
    step(f"baseline_trivial_{pct}", os.path.join(wd, f"baseline_data_subset{pct}"),
        run_baseline_trivial, subset)

    core_log_dir = os.path.join(wd, f"run1_transformer_subset{pct}")
    core_xe_ckpt = os.path.join(core_log_dir, "best_xe.pt")

    if "core" in groups:
        run_core(cfg, subset, pct, wd, tokenizer)

    if "encoders" in groups:
        run_encoders(cfg, subset, pct, wd, tokenizer)

    if "algos" in groups:
        if not os.path.exists(core_xe_ckpt):
            print("[!] Bỏ qua nhóm 'algos': chưa có best_xe.pt của core (chạy --groups core trước).")
        else:
            run_algos(cfg, subset, pct, wd, core_log_dir, tokenizer)

    # Các nhóm dưới đây cần train/dev/test loader "thô" (không qua run_experiment) + core_xe_ckpt.
    need_raw_loader_groups = {"ablations", "reward", "selection", "decode"}
    if need_raw_loader_groups & set(groups) and not os.path.exists(core_xe_ckpt):
        print(f"[!] Bỏ qua {need_raw_loader_groups & set(groups)}: chưa có {core_xe_ckpt} "
              f"(chạy --groups core trước).")
    else:
        if need_raw_loader_groups & set(groups):
            train_loader, dev_loader, test_loader = make_loaders(cfg, tokenizer, subset_ratio=subset)

        if "ablations" in groups:
            run_ablations(cfg, subset, pct, wd, core_xe_ckpt, tokenizer,
                         train_loader, dev_loader, test_loader)

        if "reward" in groups:
            run_reward_ablation(cfg, subset, pct, wd, core_xe_ckpt, tokenizer)

        if "selection" in groups:
            run_selection_group(cfg, subset, pct, wd, core_xe_ckpt, tokenizer,
                               train_loader, dev_loader)

        if "decode" in groups:
            run_decode_group(cfg, subset, pct, wd, core_xe_ckpt, tokenizer, train_loader, dev_loader)

    if "twostage" in groups:
        run_twostage_group(cfg, subset, pct, wd)

    if "latency" in groups:
        run_latency_group(cfg, subset, pct, wd, tokenizer)

    # Luôn tổng hợp lại bảng so sánh + bảng/hình báo cáo ở cuối (rẻ, không cần marker -- quét mọi
    # subset đã chạy, không chỉ subset lần này) -- xong 1 lần run_all.py là có ngay tables/ + figures/
    # trong report/ để dùng viết báo cáo, không cần chạy thêm lệnh nào khác.
    rows = agg.collect(wd)
    agg.write_csv(rows, os.path.join(wd, "comparison_table.csv"))
    agg.write_markdown(rows, os.path.join(wd, "comparison_table.md"))
    try:
        generate_report(wd)
    except Exception as e:
        print(f"[!] generate_report lỗi (không phá run_all.py): {e!r} -- chạy tay "
              f"`python scripts/make_report.py --work_dir {wd}` sau để thử lại.")

    print(f"\n{'#'*78}\n# Xong subset {pct}% sau {(time.time()-t_start)/60:.1f} phút "
          f"({(time.time()-t_start)/3600:.2f}h)\n{'#'*78}")


if __name__ == "__main__":
    main()
