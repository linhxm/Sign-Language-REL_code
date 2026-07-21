"""Sinh danh sách TÊN sequence (cột `name`) mà loader sẽ THỰC SỰ cần cho một subset ratio,
để `data/extract_poses.py --names_file` chỉ trích đúng phần đó — phục vụ chạy toàn pipeline ở
subset nhỏ (vd 5%) mà KHÔNG phải extract cả 8257 sequence (~14h vô nghĩa).

Danh sách = TOÀN BỘ dev + test (data/dataset.py KHÔNG subset 2 split này — luôn eval full)
          + train được sample ĐÚNG như dataset.py: df.dropna('translation').sample(frac, seed).

    python scripts/make_subset_names.py --subset 0.05 --out /kaggle/working/subset_names.txt
"""
import argparse, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from data.dataset import find_annotation_csv
from configs.config import CFG


def _read(csv: str) -> pd.DataFrame:
    # KHỚP CHÍNH XÁC dataset.py: sep="|", bỏ NaN translation, reset index trước khi sample.
    return pd.read_csv(csv, sep="|").dropna(subset=["translation"]).reset_index(drop=True)


def _train_names(csv: str, frac: float, seed: int) -> set:
    df = _read(csv)
    if frac >= 1.0:
        return set(df["name"])
    return set(df.sample(frac=frac, random_state=seed)["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=float, required=True, help="Tỉ lệ train subset (vd 0.05)")
    ap.add_argument("--out", default="/kaggle/working/subset_names.txt")
    ap.add_argument("--phoenix_root", default=CFG.data.phoenix_root,
                    help="Thư mục chứa 3 file PHOENIX-2014-T.{train,dev,test}.corpus.csv")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    args = ap.parse_args()

    tr = find_annotation_csv(args.phoenix_root, "train")
    dv = find_annotation_csv(args.phoenix_root, "dev")
    te = find_annotation_csv(args.phoenix_root, "test")

    names = _train_names(tr, args.subset, args.seed)
    names |= set(_read(dv)["name"])   # dev full
    names |= set(_read(te)["name"])   # test full

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(names)) + "\n")
    print(f"subset={args.subset}: train(sample)+dev(full)+test(full) = {len(names)} tên -> {args.out}")


if __name__ == "__main__":
    main()
