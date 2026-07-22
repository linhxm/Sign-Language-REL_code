"""BẢNG TỔNG (pivot) so sánh MỌI lần train cùng lúc -- gộp NHIỀU subset (5/10/25%) và NHIỀU dataset
(PHOENIX vs How2Sign) vào các bảng đặt cạnh nhau, thay vì phải mở từng `comparison_table` một.
Đây là bảng để DÁN THẲNG lên slide/report khi thuyết trình ("show cái bảng kia làm kết quả").

KHÁC 2 script cũ:
  · aggregate_results.py -> 1 bảng PHẲNG mọi run (mỗi run 1 dòng), 1 work_dir, KHÔNG có trục dataset.
  · make_report.py       -> bảng LỌC theo từng câu hỏi + LaTeX + hình, cũng chỉ 1 work_dir/1 subset.
  · make_overview.py (đây)-> bảng PIVOT: hàng = method/encoder, cột = subset%, TÁCH theo dataset ->
                             nhìn 5→10→25% tăng/giảm ra sao trong 1 lần liếc, và PHOENIX cạnh How2Sign.
Chỉ ĐỌC, không train, không sửa gì trong thư mục output -- chạy lại bao nhiêu lần cũng được.

--------------------------------------------------------------------------------------------------
TRỤC DATASET -- không cần sửa code train
--------------------------------------------------------------------------------------------------
Pipeline train (run_all.py) đặt tên thư mục GIỐNG NHAU cho mọi dataset (`run1_<enc>_subset<pct>`),
nên dataset KHÔNG suy ra được từ tên thư mục. Cách sạch nhất: train mỗi dataset vào 1 `work_dir`
RIÊNG rồi trỏ script vào từng root kèm nhãn:

    python scripts/make_overview.py \
        --root phoenix=/kaggle/input/phoenix-runs \
        --root how2sign=/kaggle/input/how2sign-runs \
        --out results/overview

Không truyền --root nào thì mặc định `--root phoenix=<work_dir>` (work_dir lấy từ configs.config).
Nếu 1 arg root KHÔNG có "nhãn=" (chỉ là path), nhãn dataset đoán từ tên đường dẫn: chứa "how2sign"
/"h2s" -> how2sign, còn lại -> phoenix.

--------------------------------------------------------------------------------------------------
SAU KHI TRAIN XONG TRÊN KAGGLE, TẢI VỀ NHỮNG FILE NÀO? (output rất nặng -- ĐỪNG tải hết)
--------------------------------------------------------------------------------------------------
Với MỖI thư mục run (`run1_*`, `rw_*`, `baseline_data_*`), script này chỉ cần các file JSON NHỎ:
    ✅ test_results.json      -- BLEU-4 test cuối của từng method  (BẮT BUỘC)
    ✅ *_history.json         -- xe/rl/ppo/mrt/raml/dpo: best dev BLEU, rep_rate, len_ratio, #epoch
    ✅ latency_*.json         -- #params, latency, throughput, peak memory
    ⚪ .done_*                -- marker resume, KHÔNG cần để đọc số (bỏ qua được)
    ❌ best_*.pt / last_*.pt  -- checkpoint ~43MB/cái, CHỈ tải nếu muốn chạy lại inference; KHÔNG cần
                                 cho bảng biểu -> BỎ để tiết kiệm dung lượng.
Tức là chỉ cần zip toàn bộ *.json (+ cây thư mục) là đủ dựng lại mọi bảng. Chạy với --manifest để
in ra chính xác file nào đang có / thiếu và ước lượng dung lượng cần tải cho từng root.
"""
import argparse, os, sys, glob, csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import aggregate_results as agg  # tái dùng parse tên run + đọc json (collect())

# thứ tự hiển thị cố định -- khớp make_report.py để bảng đọc quen mắt
ALGO_ORDER = ["xe", "scst", "ppo", "a2c", "mrt", "raml", "dpo",
              "reinforce_nobaseline", "curriculum"]
ALGO_LABELS = {"xe": "XE (CE-only)", "scst": "SCST", "ppo": "PPO", "a2c": "A2C", "mrt": "MRT",
               "raml": "RAML", "dpo": "DPO", "reinforce_nobaseline": "REINFORCE (no baseline)",
               "curriculum": "Curriculum RL"}
ENCODER_ORDER = ["transformer", "gcn", "stgcn", "graph_transformer", "tcn", "perceiver"]
ENCODER_LABELS = {"transformer": "P1 Transformer", "gcn": "P2 GCN", "stgcn": "P3 ST-GCN",
                  "graph_transformer": "P4 Graph-Transformer", "tcn": "P5 TCN",
                  "perceiver": "P6 Perceiver IO"}
REWARD_TAGS = ["rw_bleu_only", "rw_default", "rw_len_only", "rw_both"]
REWARD_LABELS = {"rw_bleu_only": "BLEU only", "rw_default": "BLEU − rep", "rw_len_only": "BLEU − len",
                 "rw_both": "BLEU − rep − len"}
PENDING = "–"  # ô chưa có số (chưa train subset/dataset đó) -- KHÔNG bịa, để rõ là còn thiếu.


# ------------------------------------------------------------------------------- thu thập đa-root
def parse_roots(root_args, default_work_dir):
    """['phoenix=/a', '/b/how2sign-runs'] -> [('phoenix','/a'), ('how2sign','/b/how2sign-runs')].
    Không có root nào -> [('phoenix', default_work_dir)]."""
    if not root_args:
        return [("phoenix", default_work_dir)]
    roots = []
    for a in root_args:
        if "=" in a:
            name, path = a.split("=", 1)
            name = name.strip().lower()
        else:
            path = a
            low = a.lower()
            name = "how2sign" if ("how2sign" in low or "h2s" in low) else "phoenix"
        roots.append((name, os.path.expanduser(path)))
    return roots


def collect_all(roots):
    """Gọi agg.collect() cho từng root rồi gắn thêm cột dataset. Bỏ qua root không tồn tại (in cảnh
    báo) thay vì crash -- tiện chạy khi mới có 1 dataset."""
    rows = []
    for dataset, path in roots:
        if not os.path.isdir(path):
            print(f"[!] Bỏ qua root '{dataset}': không thấy thư mục {path}")
            continue
        got = agg.collect(path)
        for r in got:
            r["dataset"] = dataset
            rows.append(r)
        print(f"[i] {dataset}: {len(got)} dòng từ {path}")
    return rows


# ------------------------------------------------------------------------------------ pivot core
def _fmt(v, nd=2):
    if v is None:
        return PENDING
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _subsets_present(rows):
    return sorted({r["subset_pct"] for r in rows if r["subset_pct"] is not None})


def pivot_table(rows, row_defs, subsets, cell_fn):
    """row_defs = list (row_key, label). Trả md string: cột đầu = label, các cột sau = mỗi subset%.
    cell_fn(row_key, subset) -> chuỗi 1 ô (đã format). Hàng toàn PENDING vẫn giữ (cho thấy chỗ chờ
    số) -- trừ khi không subset nào có, thì bỏ hàng đó."""
    header = ["", *[f"{s}%" for s in subsets]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row_key, label in row_defs:
        cells = [cell_fn(row_key, s) for s in subsets]
        if all(c == PENDING for c in cells):
            continue
        lines.append("| " + " | ".join([f"**{label}**", *cells]) + " |")
    if len(lines) <= 2:
        return None
    return "\n".join(lines)


def _lookup(rows, **filt):
    """1 giá trị test_bleu4 cho bộ lọc (dataset/tag/encoder/method/subset). None nếu không có."""
    for r in rows:
        if all(r.get(k) == v for k, v in filt.items()):
            return r
    return None


# --------------------------------------------------------------------------- các bảng tổng cụ thể
def overview_main(rows, dataset, subsets):
    """Transformer core: XE vs mọi thuật toán RL, pivot theo subset. tag='run1' cho xe/scst/ppo/
    mrt/raml/dpo; ablation reinforce/a2c/curriculum nằm ở tag khác (run1_a2c...) -> gộp bằng method."""
    sub = [r for r in rows if r["dataset"] == dataset and r["encoder"] == "transformer"]

    def cell(method, s):
        r = _lookup(sub, method=method, subset_pct=s)
        return _fmt(r["test_bleu4"]) if r else PENDING

    row_defs = [(m, ALGO_LABELS.get(m, m)) for m in ALGO_ORDER]
    return pivot_table(sub, row_defs, subsets, cell)


def overview_encoders(rows, dataset, subsets):
    """6 encoder, giá trị = BLEU-4 test của SCST (rơi về XE nếu chưa có SCST). tag='run1'."""
    sub = [r for r in rows if r["dataset"] == dataset and r["tag"] == "run1"
           and r["encoder"] in ENCODER_ORDER]

    def cell(enc, s):
        scst = _lookup(sub, encoder=enc, method="scst", subset_pct=s)
        if scst and scst["test_bleu4"] is not None:
            return _fmt(scst["test_bleu4"])
        xe = _lookup(sub, encoder=enc, method="xe", subset_pct=s)
        return _fmt(xe["test_bleu4"]) if xe and xe["test_bleu4"] is not None else PENDING

    row_defs = [(e, ENCODER_LABELS[e]) for e in ENCODER_ORDER]
    return pivot_table(sub, row_defs, subsets, cell)


def overview_reward(rows, dataset, subsets):
    """4 tổ hợp reward (SCST), pivot theo subset."""
    sub = [r for r in rows if r["dataset"] == dataset and r["tag"] in REWARD_TAGS
           and r["method"] == "scst"]

    def cell(tag, s):
        r = _lookup(sub, tag=tag, subset_pct=s)
        return _fmt(r["test_bleu4"]) if r else PENDING

    row_defs = [(t, REWARD_LABELS[t]) for t in REWARD_TAGS]
    return pivot_table(sub, row_defs, subsets, cell)


def cross_dataset(rows, datasets):
    """So sánh CHÉO dataset: XE vs SCST của Transformer core tại các subset dùng CHUNG (giao nhau).
    Cột = <dataset> <subset%>. Giúp thấy PHOENIX (DGS) vs How2Sign (ASL) ở cùng mức data."""
    core = [r for r in rows if r["encoder"] == "transformer" and r["tag"] == "run1"
            and r["method"] in ("xe", "scst")]
    cols = []  # (dataset, subset)
    for d in datasets:
        for s in _subsets_present([r for r in core if r["dataset"] == d]):
            cols.append((d, s))
    if not cols:
        return None
    header = ["", *[f"{d} {s}%" for d, s in cols]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method, label in (("xe", "XE (CE-only)"), ("scst", "SCST")):
        cells = []
        for d, s in cols:
            r = _lookup(core, dataset=d, method=method, subset_pct=s)
            cells.append(_fmt(r["test_bleu4"]) if r else PENDING)
        lines.append("| " + " | ".join([f"**{label}**", *cells]) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ manifest
def print_manifest(roots):
    """In file JSON nào ĐANG CÓ / còn thiếu ở mỗi run, và ước lượng dung lượng cần tải (chỉ json) so
    với dung lượng checkpoint .pt (không cần tải). Trả lời trực tiếp: 'nên tải file nào'."""
    print("\n" + "=" * 90)
    print("MANIFEST -- file cần để dựng bảng (tải json, BỎ .pt)")
    print("=" * 90)
    for dataset, path in roots:
        if not os.path.isdir(path):
            print(f"\n[{dataset}] {path}  -- KHÔNG tồn tại")
            continue
        json_bytes = pt_bytes = 0
        n_runs = 0
        print(f"\n[{dataset}] {path}")
        for run_dir in sorted(glob.glob(os.path.join(path, "*"))):
            if not os.path.isdir(run_dir):
                continue
            jsons = glob.glob(os.path.join(run_dir, "*.json"))
            pts = glob.glob(os.path.join(run_dir, "*.pt"))
            if not jsons and not pts:
                continue
            n_runs += 1
            jb = sum(os.path.getsize(p) for p in jsons)
            pb = sum(os.path.getsize(p) for p in pts)
            json_bytes += jb
            pt_bytes += pb
            has_test = os.path.exists(os.path.join(run_dir, "test_results.json"))
            flag = "" if has_test else "  ⚠️ THIẾU test_results.json"
            print(f"  {os.path.basename(run_dir):48s} json={len(jsons):2d} pt={len(pts):2d}"
                  f"  ({jb/1024:6.1f} KB){flag}")
        print(f"  -> {n_runs} run · TẢI json ≈ {json_bytes/1024/1024:.2f} MB · "
              f"(bỏ .pt ≈ {pt_bytes/1024/1024:.1f} MB không cần tải)")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------------------- main
def write_flat_csv(rows, out_csv):
    cols = ["dataset", "run_dir", "encoder", "subset_pct", "tag", "method", "best_dev_bleu4",
            "test_bleu4", "final_avg_rep_rate", "final_avg_len_ratio", "n_params",
            "latency_ms_per_sentence", "throughput_sentences_per_s"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (str(r.get("dataset")), str(r.get("run_dir")),
                                             str(r.get("method")))):
            w.writerow(r)
    print(f"[csv] {out_csv} ({len(rows)} dòng)")


def build_markdown(rows, datasets):
    blocks = ["# Bảng tổng kết quả (auto — scripts/make_overview.py)",
              "",
              f"Ô `{PENDING}` = subset/dataset đó CHƯA train xong (không phải số 0). "
              "Số = BLEU-4 test (sacreBLEU corpus). Chạy lại script sau mỗi lần train để tự điền.",
              ""]
    for d in datasets:
        drows = [r for r in rows if r["dataset"] == d]
        subs = _subsets_present(drows)
        if not subs:
            continue
        dlabel = {"phoenix": "PHOENIX-2014T (DGS — chính)",
                  "how2sign": "How2Sign (ASL — thí nghiệm phụ / train thử)"}.get(d, d)
        blocks.append(f"## {dlabel}")
        blocks.append("")
        for title, tbl in [
            ("Bảng 1 — XE vs RL (Transformer core), theo subset", overview_main(drows, d, subs)),
            ("Bảng 2 — 6 encoder (BLEU-4 SCST, rơi về XE), theo subset", overview_encoders(drows, d, subs)),
            ("Bảng 3 — reward ablation (SCST), theo subset", overview_reward(drows, d, subs)),
        ]:
            blocks.append(f"### {title}")
            blocks.append(tbl if tbl else f"_(chưa có dữ liệu — {PENDING})_")
            blocks.append("")
    cd = cross_dataset(rows, datasets)
    if cd:
        blocks.append("## So sánh chéo dataset (Transformer core, subset chung)")
        blocks.append("")
        blocks.append(cd)
        blocks.append("")
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="nhãn=path (lặp lại được). Vd --root phoenix=/a --root how2sign=/b. "
                         "Không có -> mặc định phoenix=<work_dir>.")
    ap.add_argument("--out", default=None,
                    help="prefix output (mặc định <first_root>/overview) -> .md + .csv")
    ap.add_argument("--manifest", action="store_true",
                    help="in danh sách file json có/thiếu + dung lượng cần tải rồi thoát")
    args = ap.parse_args()

    try:
        from configs.config import CFG
        default_wd = CFG.data.work_dir
    except Exception:
        default_wd = "/kaggle/working"

    roots = parse_roots(args.root, default_wd)
    print(f"Roots: {roots}")

    if args.manifest:
        print_manifest(roots)
        return

    rows = collect_all(roots)
    if not rows:
        print("Không có dữ liệu ở bất kỳ root nào — train trước rồi chạy lại.")
        return

    datasets = []
    for d, _ in roots:  # giữ thứ tự người dùng truyền vào, không trùng
        if d not in datasets:
            datasets.append(d)

    out_prefix = args.out or os.path.join(roots[0][1], "overview")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    md = build_markdown(rows, datasets)
    with open(out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"[md]  {out_prefix}.md")
    write_flat_csv(rows, out_prefix + ".csv")
    print_manifest(roots)


if __name__ == "__main__":
    main()
