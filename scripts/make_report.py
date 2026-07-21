"""
Sinh BẢNG + BIỂU ĐỒ so sánh từ kết quả `run_all.py`/`main.py` đã chạy -- KHÔNG train gì thêm,
chỉ đọc lại `test_results.json` / `*_history.json` / `latency_*.json` đã có trong `--work_dir`
(dùng lại `scripts/aggregate_results.py::collect()`), rồi xuất ra dạng dễ dùng cho báo cáo/paper
thay vì phải tự lọc tay từ `comparison_table.csv` gộp chung mọi run.

Output trong `--out_dir` (mặc định `<work_dir>/report/`):
    tables/*.csv, *.md   -- 6 bảng đã lọc/nhóm sẵn theo từng câu hỏi so sánh (main/encoder/algo/
                            reward ablation/ablation khác/baseline/latency)
    tables/*.tex         -- 3 bảng LaTeX dán thẳng vào paper/sn-article.tex
                            (khớp \\label{tab:main}/\\label{tab:reward}/\\label{tab:encresults})
    figures/*.png + *.pdf -- 4 biểu đồ (PNG xem nhanh, PDF vector để \\includegraphics):
                             BLEU theo epoch (RL rẽ nhánh từ điểm warm-start XE) · reward ablation ·
                             so sánh 6 encoder · so sánh thuật toán. Biểu đồ cột đều có số trên cột.
                             (Đã bỏ 'ΔBLEU theo subset' vì so sánh giữa các subset không cùng điều
                             kiện train nên vô nghĩa.)

Usage (sau khi đã chạy `run_all.py --subset ...` ít nhất 1 lần):
    python scripts/make_report.py --work_dir /kaggle/working
    python scripts/make_report.py --work_dir /kaggle/working --subset 25   # ép 1 subset cụ thể
                                                                            # (mặc định: subset LỚN
                                                                            # NHẤT đã có dữ liệu)

GIỚI HẠN QUAN TRỌNG: pipeline hiện tại (training/train_xe.py::evaluate) chỉ tính BLEU-4 (sacrebleu)
+ rep_rate + len_ratio -- KHÔNG có BLEU-1/ROUGE-L. Bảng LaTeX `tab:main` trong paper có 2 cột đó;
script này để NGUYÊN "--" cho 2 cột đó thay vì bịa số -- muốn có thật thì phải thêm code tính
BLEU-1/ROUGE-L vào evaluate() (chưa làm, ngoài phạm vi script này).

Script này an toàn để chạy nhiều lần, kể cả khi CHƯA có run nào (in "không có dữ liệu" cho từng
bảng/hình thay vì crash) -- dùng để test luồng trước khi có số liệu thật từ Kaggle.
"""
import argparse, os, sys, csv, glob, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import aggregate_results as agg

import matplotlib
matplotlib.use("Agg")  # headless (Kaggle/CI) -- không cần display
import matplotlib.pyplot as plt

ALL_ENCODERS = ["transformer", "gcn", "stgcn", "graph_transformer", "tcn", "perceiver"]  # P1-P6
ENCODER_LABELS = {"transformer": "P1 Transformer", "gcn": "P2 GCN", "stgcn": "P3 ST-GCN",
                  "graph_transformer": "P4 Graph Transformer", "tcn": "P5 TCN",
                  "perceiver": "P6 Perceiver IO"}
ALGO_ORDER = ["xe", "scst", "ppo", "mrt", "raml", "dpo"]
ALGO_LABELS = {"xe": "Cross-entropy only", "scst": "CE plus SCST", "ppo": "CE plus PPO",
              "mrt": "CE plus MRT", "raml": "CE plus RAML", "dpo": "CE plus DPO"}
HISTORY_FILES = {"xe": "xe_history.json", "scst": "rl_history.json", "ppo": "ppo_history.json",
                 "mrt": "mrt_history.json", "raml": "raml_history.json", "dpo": "dpo_history.json"}
REWARD_TAGS = ["rw_bleu_only", "rw_default", "rw_len_only", "rw_both"]
REWARD_WEIGHTS = {  # (w_rep, w_len) -- khớp run_all.py::REWARD_COMBOS
    "rw_bleu_only": (0.0, 0.0), "rw_default": (0.5, 0.0),
    "rw_len_only": (0.0, 0.5), "rw_both": (0.5, 0.5),
}


# --------------------------------------------------------------------------------------- helpers
def _pick_subset(rows, subset_arg):
    subs = sorted({r["subset_pct"] for r in rows if r["subset_pct"] is not None}, reverse=True)
    if not subs:
        return None
    if subset_arg is not None:
        if subset_arg not in subs:
            print(f"[!] Không có dữ liệu subset={subset_arg}%. Các subset đã có: {subs} "
                  f"-> dùng {subs[0]}% thay thế.")
            return subs[0]
        return subset_arg
    return subs[0]


def _write_table(rows, cols, out_dir, name, title=None):
    if not rows:
        print(f"[i] Bỏ qua bảng '{name}': không có dữ liệu.")
        return
    prefix = os.path.join(out_dir, "tables", name)
    with open(prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    lines = [f"# {title}\n" if title else "", "| " + " | ".join(cols) + " |",
            "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    with open(prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(l for l in lines if l is not None) + "\n")
    print(f"[table] {name}: {len(rows)} dòng -> {prefix}.csv/.md")


def _fnum(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "--"


# ---------------------------------------------------------------------------------- xây từng bảng
def build_main(rows):
    out = [r for r in rows if r["tag"] == "run1" and r["encoder"] == "transformer"
          and r["method"] in ALGO_ORDER]
    out.sort(key=lambda r: (r["subset_pct"] or 0, ALGO_ORDER.index(r["method"])))
    return out


def build_encoders(rows, subset_pct):
    out = [r for r in rows if r["tag"] == "run1" and r["subset_pct"] == subset_pct
          and r["encoder"] in ALL_ENCODERS and r["method"] in ("xe", "scst")]
    out.sort(key=lambda r: (ALL_ENCODERS.index(r["encoder"]), r["method"]))
    return out


def build_reward(rows):
    out = [r for r in rows if r["tag"] in REWARD_TAGS and r["method"] == "scst"]
    out.sort(key=lambda r: (r["subset_pct"] or 0, REWARD_TAGS.index(r["tag"])))
    return out


def build_ablations(rows):
    ablation_tags = {"run1_reinforce_nobaseline", "run1_a2c", "run1_curriculum"}
    out = [r for r in rows if r["tag"] in ablation_tags]
    out.sort(key=lambda r: (r["subset_pct"] or 0, r["tag"]))
    return out


def build_baseline(rows):
    out = [r for r in rows if r["method"] in ("base_empty", "base_most_frequent")]
    out.sort(key=lambda r: (r["subset_pct"] or 0, str(r["method"])))
    return out


def build_latency(rows):
    seen, out = set(), []
    for r in rows:
        if r["n_params"] is None or r["latency_ms_per_sentence"] is None:
            continue
        key = r["encoder"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda r: ALL_ENCODERS.index(r["encoder"]) if r["encoder"] in ALL_ENCODERS else 99)
    return out


# -------------------------------------------------------------------------------------- LaTeX
def write_latex_main(main_rows, baseline_rows, subset_pct, out_dir):
    sub_rows = [r for r in main_rows if r["subset_pct"] == subset_pct]
    if not sub_rows:
        print("[i] Bỏ qua tab:main LaTeX: không có dữ liệu ở subset đã chọn.")
        return
    by_method = {r["method"]: r for r in sub_rows}
    base = {r["method"]: r for r in baseline_rows if r["subset_pct"] == subset_pct}

    lines = [
        r"% Tự sinh bởi scripts/make_report.py -- KHÔNG có BLEU-1/ROUGE-L (pipeline chỉ tính BLEU-4),",
        r"% 2 cột đó để \"--\" thay vì bịa số. Sửa \\caption cho đúng % subset nếu cần.",
        r"\begin{table}[h]",
        rf"\caption{{Main results on PHOENIX-2014T at the ${subset_pct}\%$ subset. Read relative to the floor.}}\label{{tab:main}}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Configuration & BLEU-1 & BLEU-4 & ROUGE-L & Rep. \% \\",
        r"\midrule",
    ]
    if "base_empty" in base:
        lines.append(rf"Floor, empty          & -- & {_fnum(base['base_empty']['test_bleu4'])} & -- & -- \\")
    if "base_most_frequent" in base:
        lines.append(rf"Floor, most frequent  & -- & {_fnum(base['base_most_frequent']['test_bleu4'])} & -- & -- \\")
    lines.append(r"\midrule")
    for algo in ALGO_ORDER:
        if algo not in by_method:
            continue
        r = by_method[algo]
        rep_pct = r["final_avg_rep_rate"] * 100 if isinstance(r["final_avg_rep_rate"], float) else None
        lines.append(rf"{ALGO_LABELS[algo]:<22}& -- & {_fnum(r['test_bleu4'])} & -- & {_fnum(rep_pct, 1)} \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(out_dir, "tables", "tab_main.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[latex] tab:main -> {path}")


def write_latex_reward(reward_rows, subset_pct, out_dir):
    sub_rows = {r["tag"]: r for r in reward_rows if r["subset_pct"] == subset_pct}
    if not sub_rows:
        print("[i] Bỏ qua tab:reward LaTeX: không có dữ liệu ở subset đã chọn.")
        return
    lines = [
        r"% Tự sinh bởi scripts/make_report.py",
        r"\begin{table}[h]",
        rf"\caption{{Reward ablation of Experiment 9 at the ${subset_pct}\%$ subset.}}\label{{tab:reward}}",
        r"\begin{tabular}{@{}cccccc@{}}",
        r"\toprule",
        r"$w_{\text{bleu}}$ & $w_{\text{rep}}$ & $w_{\text{len}}$ & BLEU-4 & Rep. \% & Length \\",
        r"\midrule",
    ]
    for tag in REWARD_TAGS:
        if tag not in sub_rows:
            continue
        r = sub_rows[tag]
        w_rep, w_len = REWARD_WEIGHTS[tag]
        rep_pct = r["final_avg_rep_rate"] * 100 if isinstance(r["final_avg_rep_rate"], float) else None
        lines.append(rf"1 & {w_rep} & {w_len} & {_fnum(r['test_bleu4'])} & {_fnum(rep_pct, 1)} & "
                    rf"{_fnum(r['final_avg_len_ratio'], 3)} \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(out_dir, "tables", "tab_reward.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[latex] tab:reward -> {path}")


def write_latex_encoders(encoder_rows, subset_pct, out_dir):
    sub_rows = [r for r in encoder_rows if r["subset_pct"] == subset_pct]
    if not sub_rows:
        print("[i] Bỏ qua tab:encresults LaTeX: không có dữ liệu ở subset đã chọn.")
        return
    by_enc = {}
    for r in sub_rows:  # ưu tiên scst, rơi về xe nếu chưa có scst (RL chưa vượt XE)
        if r["encoder"] not in by_enc or r["method"] == "scst":
            by_enc[r["encoder"]] = r
    lines = [
        r"% Tự sinh bởi scripts/make_report.py",
        r"\begin{table}[h]",
        rf"\caption{{Six-encoder comparison of Experiments 4 and 15 at the ${subset_pct}\%$ subset.}}\label{{tab:encresults}}",
        r"\begin{tabular}{@{}lrcc@{}}",
        r"\toprule",
        r"Encoder & Params & BLEU-4 & Latency, batch 1 \\",
        r"\midrule",
    ]
    for enc in ALL_ENCODERS:
        if enc not in by_enc:
            continue
        r = by_enc[enc]
        params = f"{r['n_params']/1e6:.2f}M" if r["n_params"] else "--"
        lat = f"{r['latency_ms_per_sentence']:.1f}ms" if r["latency_ms_per_sentence"] else "--"
        lines.append(rf"{ENCODER_LABELS[enc]:<20} & {params} & {_fnum(r['test_bleu4'])} & {lat} \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(out_dir, "tables", "tab_encresults.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[latex] tab:encresults -> {path}")


# ------------------------------------------------------------------------------------ Figures
def _savefig(fig, out_dir, name):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[figure] {name}.png/.pdf -> {fig_dir}")


def fig_bleu_vs_epoch(work_dir, subset_pct, out_dir):
    """XE full + các nhánh RL BẮT ĐẦU TỪ ĐÚNG epoch mà best_xe.pt được chọn (warm-start), thay vì
    mỗi đường vẽ từ epoch 0 riêng (gây rời rạc, khó đọc). Nhờ vậy thấy trực quan: từ điểm warm-start,
    'train XE TIẾP' (đường XE chạy tiếp) so với 'CHUYỂN sang RL' (các nhánh RL rẽ ra). RL epoch 0
    được nối thẳng từ điểm warm-start của XE để nhánh liền mạch."""
    run_dir = os.path.join(work_dir, f"run1_transformer_subset{subset_pct}")

    def _load_ys(fname):
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
        ys = [h.get("dev_bleu4") for h in hist if h.get("dev_bleu4") is not None]
        return ys or None

    xe_ys = _load_ys(HISTORY_FILES["xe"])
    # epoch mà best_xe được LẤY = epoch có dev BLEU cao nhất (train_xe.py lưu best theo dev BLEU).
    warm = int(max(range(len(xe_ys)), key=lambda i: xe_ys[i])) if xe_ys else 0
    xe_best = xe_ys[warm] if xe_ys else None

    rl_curves = {a: _load_ys(f) for a, f in HISTORY_FILES.items() if a != "xe"}
    rl_curves = {a: ys for a, ys in rl_curves.items() if ys}

    if not xe_ys and not rl_curves:
        print(f"[i] Bỏ qua fig_bleu_vs_epoch: không có history nào trong {run_dir}.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if xe_ys:
        ax.plot(range(len(xe_ys)), xe_ys, marker="o", markersize=3,
                label=ALGO_LABELS["xe"], color="#4C72B0", zorder=2)
        ax.axvline(warm, ls="--", lw=1, color="gray", alpha=0.7, zorder=1)  # mốc warm-start
        ax.scatter([warm], [xe_best], s=70, color="black", zorder=4)
        ax.annotate(f"XE warm-start\n(ep {warm}, BLEU {xe_best:.2f})", (warm, xe_best),
                    xytext=(6, -30), textcoords="offset points", fontsize=8)
    for algo, ys in rl_curves.items():
        if xe_best is not None:   # nối nhánh RL từ điểm warm-start của XE
            xs = list(range(warm, warm + len(ys) + 1)); yy = [xe_best] + ys
        else:
            xs = list(range(len(ys))); yy = ys
        ax.plot(xs, yy, marker="o", markersize=3, label=ALGO_LABELS.get(algo, algo), zorder=3)
    ax.set_xlabel("Epoch (XE, rồi RL nối tiếp từ điểm warm-start)")
    ax.set_ylabel("Dev BLEU-4")
    ax.set_title(f"BLEU theo epoch -- subset {subset_pct}% (RL rẽ nhánh từ điểm chọn XE)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    _savefig(fig, out_dir, "fig_bleu_vs_epoch")


def _annotate_bars(ax, bars, vals, fmt="{:.2f}"):
    """Ghi số lên đỉnh mỗi cột (yêu cầu: biểu đồ cột phải có số ngay trên biểu đồ)."""
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
# (Đã bỏ fig_delta_bleu_by_subset: mỗi subset train ĐỦ epoch độc lập nên so sánh ΔBLEU GIỮA các
#  subset không cùng điều kiện -> không có ý nghĩa. Muốn data-size ablation đúng cần thiết kế riêng.)


def fig_reward_ablation(reward_rows, subset_pct, out_dir):
    sub_rows = {r["tag"]: r for r in reward_rows if r["subset_pct"] == subset_pct}
    pts = [(tag, sub_rows[tag]) for tag in REWARD_TAGS if tag in sub_rows
          and sub_rows[tag]["test_bleu4"] is not None
          and sub_rows[tag]["final_avg_rep_rate"] is not None]
    if not pts:
        print("[i] Bỏ qua fig_reward_ablation: cần test_bleu4 + avg_rep_rate cho >=1 tổ hợp reward.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for tag, r in pts:
        ax.scatter(r["final_avg_rep_rate"], r["test_bleu4"], s=60)
        ax.annotate(f"{tag}\nBLEU {r['test_bleu4']:.2f}", (r["final_avg_rep_rate"], r["test_bleu4"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("avg_rep_rate (tỉ lệ n-gram lặp)"); ax.set_ylabel("Test BLEU-4")
    ax.set_title(f"Exp.9 -- trade-off reward ablation (subset {subset_pct}%)")
    ax.grid(alpha=0.3)
    _savefig(fig, out_dir, "fig_reward_ablation")


def fig_encoder_comparison(encoder_rows, subset_pct, out_dir):
    by_enc = {}
    for r in encoder_rows:
        if r["test_bleu4"] is None:
            continue
        if r["encoder"] not in by_enc or r["method"] == "scst":
            by_enc[r["encoder"]] = r
    encs = [e for e in ALL_ENCODERS if e in by_enc]
    if not encs:
        print("[i] Bỏ qua fig_encoder_comparison: chưa có test_bleu4 cho encoder nào.")
        return
    vals = [by_enc[e]["test_bleu4"] for e in encs]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([ENCODER_LABELS[e] for e in encs], vals, color="#55A868")
    _annotate_bars(ax, bars, vals)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Test BLEU-4"); ax.set_title(f"Exp.4 -- so sánh 6 encoder (subset {subset_pct}%)")
    plt.xticks(rotation=30, ha="right"); ax.grid(alpha=0.3, axis="y")
    _savefig(fig, out_dir, "fig_encoder_comparison")


def fig_algo_comparison(main_rows, subset_pct, out_dir):
    sub_rows = {r["method"]: r for r in main_rows if r["subset_pct"] == subset_pct
               and r["test_bleu4"] is not None}
    algos = [a for a in ALGO_ORDER if a in sub_rows]
    if not algos:
        print("[i] Bỏ qua fig_algo_comparison: chưa có test_bleu4 cho thuật toán nào.")
        return
    vals = [sub_rows[a]["test_bleu4"] for a in algos]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([ALGO_LABELS[a] for a in algos], vals, color="#C44E52")
    _annotate_bars(ax, bars, vals)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Test BLEU-4"); ax.set_title(f"Exp.1/7 -- so sánh thuật toán (subset {subset_pct}%)")
    plt.xticks(rotation=20, ha="right"); ax.grid(alpha=0.3, axis="y")
    _savefig(fig, out_dir, "fig_algo_comparison")


# --------------------------------------------------------------------------------------- main
def generate_report(work_dir: str, out_dir: str = None, subset: int = None):
    """Hàm tái sử dụng được -- run_all.py gọi thẳng cái này ở cuối mỗi lần chạy 1 subset để bảng/
    hình luôn cập nhật, không cần bước riêng. CLI bên dưới chỉ parse arg rồi gọi hàm này."""
    out_dir = out_dir or os.path.join(work_dir, "report")
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    rows = agg.collect(work_dir)
    if not rows:
        print(f"[!] Không tìm thấy kết quả nào trong {work_dir} -- chạy run_all.py trước.")
        return
    subset_pct = _pick_subset(rows, subset)
    print(f"Dùng subset={subset_pct}% cho các bảng/hình 'tại 1 subset'.")

    main_rows = build_main(rows)
    encoder_rows_all = [r for r in rows if r["tag"] == "run1" and r["encoder"] in ALL_ENCODERS
                        and r["method"] in ("xe", "scst")]
    reward_rows = build_reward(rows)
    baseline_rows = build_baseline(rows)

    _write_table(main_rows, ["subset_pct", "method", "test_bleu4", "best_dev_bleu4",
                             "final_avg_rep_rate", "final_avg_len_ratio"],
                out_dir, "table_main", "Exp.1/7/11 -- XE vs mọi thuật toán RL (Transformer)")
    _write_table(build_encoders(rows, subset_pct),
                ["subset_pct", "encoder", "method", "n_params", "test_bleu4",
                 "latency_ms_per_sentence"],
                out_dir, "table_encoders", f"Exp.4/15 -- so sánh 6 encoder (subset {subset_pct}%)")
    _write_table(reward_rows,
                ["subset_pct", "tag", "test_bleu4", "final_avg_rep_rate", "final_avg_len_ratio"],
                out_dir, "table_reward", "Exp.9 -- reward ablation")
    _write_table(build_ablations(rows),
                ["subset_pct", "tag", "method", "test_bleu4", "final_avg_rep_rate",
                 "final_avg_len_ratio"],
                out_dir, "table_ablations", "REINFORCE/A2C/Curriculum -- ablation")
    _write_table(baseline_rows, ["subset_pct", "encoder", "method", "test_bleu4"],
                out_dir, "table_baseline", "Baseline sàn (empty / most-frequent)")
    _write_table(build_latency(rows),
                ["encoder", "n_params", "latency_ms_per_sentence", "throughput_sentences_per_s",
                 "peak_memory_mb"],
                out_dir, "table_latency", "Exp.15 -- latency/memory/params")

    write_latex_main(main_rows, baseline_rows, subset_pct, out_dir)
    write_latex_reward(reward_rows, subset_pct, out_dir)
    write_latex_encoders(encoder_rows_all, subset_pct, out_dir)

    fig_bleu_vs_epoch(work_dir, subset_pct, out_dir)
    fig_reward_ablation(reward_rows, subset_pct, out_dir)
    fig_encoder_comparison(encoder_rows_all, subset_pct, out_dir)
    fig_algo_comparison(main_rows, subset_pct, out_dir)

    print(f"\nXong. Xem kết quả trong {out_dir}/tables và {out_dir}/figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default="/kaggle/working")
    ap.add_argument("--out_dir", default=None, help="Mặc định <work_dir>/report")
    ap.add_argument("--subset", type=int, default=None,
                    help="Subset %% dùng cho các bảng/hình 'tại 1 subset' (mặc định: lớn nhất đã có)")
    args = ap.parse_args()
    generate_report(args.work_dir, args.out_dir, args.subset)


if __name__ == "__main__":
    main()
