"""Baseline CƠ BẢN NHẤT (sàn không-cần-model) cho bảng so sánh — mọi phương pháp trong
comparison_table phải được đọc TƯƠNG ĐỐI so với các "sàn" này, không phải con số tuyệt đối
(docs/1_Thuyet_Trinh_Tong_Hop.md §E, §F).

    - base_empty         : hypothesis rỗng cho mọi câu (sàn tuyệt đối của BLEU).
    - base_most_frequent : lặp lại câu train xuất hiện NHIỀU NHẤT cho mọi câu test. PHOENIX-2014T
      là domain thời tiết lặp nhiều — sàn này có thể > 0 đáng kể; model nào không vượt qua nó
      thì BLEU của model đó vô nghĩa.

Kết quả merge vào <work_dir>/baseline_data_subset{pct}/test_results.json — đúng định dạng mà
scripts/aggregate_results.py đã quét, nên các dòng baseline TỰ xuất hiện trong
comparison_table.csv/.md không cần sửa gì thêm.

(Các baseline cho frame-selection / decode-policy đã gỡ cùng nhánh RL-ngoài-decoder —
xem docs/2_Huong_Phat_Trien.md.)

Usage (Kaggle):
    python scripts/eval_baselines.py --subset 0.05
"""
import argparse, os, sys, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CFG
from data.tokenizer import Tokenizer
from data.dataset import make_loaders


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


def eval_trivial(train_loader, test_loader):
    refs = [r for batch in test_loader for r in batch["text_raw"]]
    most_freq = train_loader.dataset.df["translation"].astype(str).value_counts().idxmax()
    print(f"Câu train phổ biến nhất ({int(train_loader.dataset.df['translation'].astype(str).value_counts().max())} lần): {most_freq!r}")
    return {
        "base_empty": {"test_bleu4": _corpus_bleu([""] * len(refs), refs)},
        "base_most_frequent": {"test_bleu4": _corpus_bleu([most_freq] * len(refs), refs),
                               "hypothesis": most_freq},
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=float, default=0.05)
    args = ap.parse_args()
    run_baseline_trivial(args.subset)


if __name__ == "__main__":
    main()
