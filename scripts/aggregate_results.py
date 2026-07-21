"""Tổng hợp toàn bộ kết quả từ mọi lần chạy (main.py, main_twostage.py, measure_latency.py,
train_selection_policy.py, train_decode_policy.py) thành 1 BẢNG SO SÁNH DUY NHẤT giữa các
phương pháp — phục vụ trực tiếp yêu cầu so sánh pipeline/thuật toán RL/kiến trúc encoder với nhau
(docs/1_Thuyet_Trinh_Tong_Hop.md §A §4.2, docs/1_Thuyet_Trinh_Tong_Hop.md §E). Quét đệ quy `--work_dir` (mặc định
/kaggle/working) tìm mọi test_results.json/*_history.json/latency_*.json đã có — KHÔNG cần khai
báo trước danh sách run, chạy lại script này sau mỗi lần train xong 1 pipeline mới là bảng tự cập
nhật (đúng tinh thần "chưa code liền không sao, miễn verify được khi cần" — ở đây là verify bằng
số liệu thật thay vì bảng liệt kê lý thuyết).

Usage (trên Kaggle, cuối notebook sau khi đã chạy các cell train):
    python scripts/aggregate_results.py --work_dir /kaggle/working --out /kaggle/working/comparison_table
"""
import argparse, os, glob, json, csv, re

METHOD_ALIASES = {"rl": "scst"}  # train_scst.py luôn ghi "rl_history.json" (tên lịch sử) bất kể
                                 # --algo=scst đặt key "scst" trong test_results.json của main.py


def _parse_run_name(dirname: str):
    """`{tag}_{encoder}_subset{pct}[_hậu-tố]` (quy ước main.py/main_twostage.py, cộng hậu tố tự do
    run_all.py gắn thêm cho ablation -- vd `_reinforce_nobaseline`, `_a2c`, `_curriculum`,
    `_selectpolicy`, `_decodepolicy`) -> (tag[+hậu tố], encoder, subset_pct).
    Best-effort — không raise nếu không khớp, trả None cho phần không parse được.

    BUG cũ (sửa ở đây): `graph_transformer` chứa "transformer" làm hậu tố. Với `(.*)` GREEDY,
    backtrack từ phải sang trái dừng ở prefix DÀI NHẤT có thể khớp -- với dirname
    `run1_graph_transformer_subset100`, prefix "run1_graph" (rồi "transformer_subset100" khớp
    alternative "transformer") được tìm thấy TRƯỚC prefix "run1" (rồi "graph_transformer_subset100"
    khớp "graph_transformer"), bất kể thứ tự alternation -- vì độ dài prefix mới là thứ quyết định,
    không phải thứ tự alternative. Hậu quả: mọi thư mục `graph_transformer` bị gán nhầm
    encoder="transformer" (tag nuốt mất "graph_"), khiến bảng so sánh Exp.4 gộp lẫn 2 kiến trúc
    khác nhau vào 1 dòng. Sửa bằng `(.*?)` NON-greedy: thử prefix NGẮN NHẤT trước, nên tại
    prefix="run1" nó khớp "graph_transformer" ngay (chuỗi còn lại bắt đầu bằng "graph_...", alternative
    "transformer" không khớp được ở vị trí này vì thiếu "graph_" đứng trước) trước khi thử prefix
    dài hơn."""
    m = re.match(r"^(.*?)_(graph_transformer|transformer|stgcn|gcn|tcn|perceiver)_subset(\d+)(_.+)?$",
                 dirname)
    if m:
        tag = m.group(1) + (m.group(4) or "")  # gộp hậu tố vào tag để không mất thông tin biến thể
        return tag, m.group(2), int(m.group(3))
    return dirname, None, None


def _test_bleu(entry: dict):
    """Lấy test_bleu4 (hoặc test_bleu4_e2e của P7). KHÔNG dùng `a or b` -- BLEU 0.0 là giá trị
    hợp lệ (vd base_empty của scripts/eval_baselines.py) sẽ bị `or` nuốt thành None."""
    v = entry.get("test_bleu4")
    return v if v is not None else entry.get("test_bleu4_e2e")


def _best_dev_bleu(history):
    vals = [h.get("dev_bleu4") for h in history if h.get("dev_bleu4") is not None]
    return round(max(vals), 3) if vals else None


def _last(history, key):
    for h in reversed(history):
        if key in h and h[key] is not None:
            return h[key]
    return None


def _latency_for(latency_rows, bs=1):
    for r in latency_rows:
        if r.get("batch_size") == bs:
            return r
    return latency_rows[0] if latency_rows else {}


def collect(work_dir: str):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(work_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        dirname = os.path.basename(run_dir)
        tag, encoder, subset_pct = _parse_run_name(dirname)

        test_results = {}
        trp = os.path.join(run_dir, "test_results.json")
        if os.path.exists(trp):
            with open(trp, encoding="utf-8") as f:
                test_results = json.load(f)

        latency_rows = []
        for lp in glob.glob(os.path.join(run_dir, "latency_*.json")):
            with open(lp, encoding="utf-8") as f:
                latency_rows.extend(json.load(f))
        lat = _latency_for(latency_rows, bs=1)

        matched_canonical = set()
        for hist_path in glob.glob(os.path.join(run_dir, "*_history.json")):
            method = os.path.basename(hist_path).replace("_history.json", "")
            canonical = METHOD_ALIASES.get(method, method)
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)

            test_bleu4 = None
            if canonical in test_results:
                v = test_results[canonical]
                test_bleu4 = _test_bleu(v)
                matched_canonical.add(canonical)

            rows.append({
                "run_dir": dirname, "tag": tag, "encoder": encoder, "subset_pct": subset_pct,
                "method": canonical,
                "n_epochs_run": len(history),
                "best_dev_bleu4": _best_dev_bleu(history),
                "dev_wer": _last(history, "dev_wer"),
                "final_avg_rep_rate": _last(history, "avg_rep_rate"),
                "final_avg_len_ratio": _last(history, "avg_len_ratio"),
                "test_bleu4": test_bleu4,
                "n_params": lat.get("n_params"),
                "latency_ms_per_sentence": lat.get("latency_ms_per_sentence"),
                "throughput_sentences_per_s": lat.get("throughput_sentences_per_s"),
                "peak_memory_mb": lat.get("peak_memory_mb"),
            })

        # test_results.json có key chưa khớp history nào (vd P7 chỉ log test_bleu4_e2e tổng hợp,
        # không có 1 vòng epoch riêng để tạo history).
        for k, v in test_results.items():
            if k in matched_canonical:
                continue
            rows.append({
                "run_dir": dirname, "tag": tag, "encoder": encoder, "subset_pct": subset_pct,
                "method": k, "n_epochs_run": None, "best_dev_bleu4": None, "dev_wer": None,
                "final_avg_rep_rate": None, "final_avg_len_ratio": None,
                "test_bleu4": _test_bleu(v),
                "n_params": lat.get("n_params"), "latency_ms_per_sentence": None,
                "throughput_sentences_per_s": None, "peak_memory_mb": None,
            })
    return rows


def write_csv(rows, out_csv: str):
    if not rows:
        print("Không tìm thấy kết quả nào -- chạy pipeline trước rồi mới aggregate.")
        return
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Đã ghi {out_csv} ({len(rows)} dòng)")


def write_markdown(rows, out_md: str):
    if not rows:
        return
    cols = ["run_dir", "encoder", "subset_pct", "method", "best_dev_bleu4", "test_bleu4",
           "final_avg_rep_rate", "final_avg_len_ratio", "dev_wer",
           "n_params", "latency_ms_per_sentence", "throughput_sentences_per_s"]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (str(r["run_dir"]), str(r["method"]))):
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Đã ghi {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/kaggle/working")
    ap.add_argument("--out", default=None, help="Prefix output (mặc định <work_dir>/comparison_table)")
    args = ap.parse_args()
    out_prefix = args.out or os.path.join(args.work_dir, "comparison_table")

    rows = collect(args.work_dir)
    write_csv(rows, out_prefix + ".csv")
    write_markdown(rows, out_prefix + ".md")


if __name__ == "__main__":
    main()
