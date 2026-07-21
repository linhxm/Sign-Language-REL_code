"""Entry point cho P7 (Two-stage: Pose -> Gloss -> NMT -> Text, docs/1_Thuyet_Trinh_Tong_Hop.md §A).
Trên Kaggle: !python main_twostage.py --subset 0.25
(run_all.py gọi thẳng run_twostage() cho toàn ma trận thí nghiệm 1 subset -- file này vẫn giữ CLI
để chạy tay riêng lẻ khi cần debug.)

Bước verify BẮT BUỘC trước khi chạy (docs/1_Thuyet_Trinh_Tong_Hop.md §A §4.1, P7 "cần verify trước khi code"):
cột `orth` (gloss) phải tồn tại trong PHOENIX-2014-T*.corpus.csv — `GlossVocab.build_from_csv()`
raise lỗi rõ ràng ngay từ đầu nếu thiếu, thay vì fail âm thầm giữa chừng.

Luồng:
  Stage 1: pose -> gloss (CTC, training/train_ctc_gloss.py)
  Stage 2: gloss -> text (NMT thuần text, training/train_gloss2text.py)
  Eval end-to-end: pose -> (CTC greedy) -> gloss -> (Gloss2Text greedy) -> text, so BLEU trực tiếp
  với pipeline single-stage P1 (main.py) để trả lời câu hỏi "gloss-based vs gloss-free"
  (docs/1_Thuyet_Trinh_Tong_Hop.md §A §4.3, docs/1_Thuyet_Trinh_Tong_Hop.md §H RQ4-adjacent).
"""
import argparse, os, sys, random, json
from functools import partial
import numpy as np
import torch
from sacrebleu import corpus_bleu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import CFG
from data.tokenizer import build_tokenizer_from_train, Tokenizer
from data.gloss_vocab import GlossVocab
from data.dataset import PhoenixSLTDataset, collate_fn, find_annotation_csv
from torch.utils.data import DataLoader
from models.gloss_ctc_head import GlossCTCModel
from models.gloss2text_nmt import Gloss2TextNMT
from training.train_ctc_gloss import train_ctc_gloss, _ctc_greedy_decode
from training.train_gloss2text import train_gloss2text


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_twostage(cfg, subset: float, encoder: str = "transformer", tag: str = "p7_twostage") -> dict:
    """Toàn bộ luồng P7 cho 1 subset/encoder, tách thành hàm để run_all.py gọi thẳng trong process."""
    cfg.model.encoder_type = encoder
    set_seed(cfg.seed)
    log_dir = os.path.join(cfg.data.work_dir, f"{tag}_{encoder}_subset{int(subset*100)}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log dir: {log_dir}")

    # find_annotation_csv: thử 3 layout (annotations/manual, annotations/, gốc) + quét đệ quy --
    # KHÔNG hard-code "annotations/" để khớp cả khi CSV để phẳng (vd data/archive/*.corpus.csv).
    # Đồng bộ với data/dataset.py::make_loaders (tránh lệch: loader tìm thấy nhưng tokenizer thì không).
    train_csv = find_annotation_csv(cfg.data.phoenix_root, "train")
    dev_csv = find_annotation_csv(cfg.data.phoenix_root, "dev")
    test_csv = find_annotation_csv(cfg.data.phoenix_root, "test")

    # 0. Verify + build gloss vocab (bước bắt buộc, xem docstring đầu file).
    gloss_vocab = GlossVocab.build_from_csv(train_csv, max_vocab=cfg.data.gloss_vocab_size)
    gloss_vocab.save(os.path.join(log_dir, "gloss_vocab.json"))
    print(f"Gloss vocab size: {gloss_vocab.vocab_size} (verify OK — cột 'orth' tồn tại)")

    # 1. Text tokenizer (dùng chung với pipeline single-stage nếu đã có, tránh train lại)
    spm_model = os.path.join(cfg.data.work_dir, "spm.model")
    if not os.path.exists(spm_model):
        tokenizer = build_tokenizer_from_train(train_csv, cfg.data.work_dir, cfg.data.vocab_size)
    else:
        tokenizer = Tokenizer(spm_model)
    print(f"Text vocab size: {tokenizer.vocab_size}")

    # 2. Data (có gloss)
    def make_ds(csv_path, subset_ratio=1.0):
        return PhoenixSLTDataset(csv_path, cfg.data.pose_cache_dir, tokenizer,
                                 cfg.data.max_frames, cfg.data.max_text_len,
                                 subset_ratio=subset_ratio, seed=cfg.seed, gloss_vocab=gloss_vocab)

    train_ds = make_ds(train_csv, subset)
    dev_ds = make_ds(dev_csv)
    test_ds = make_ds(test_csv)
    # partial thay vì lambda: picklable khi num_workers>0 trên start method "spawn" (Windows/macOS)
    collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.xe_batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.train.xe_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.xe_batch_size, shuffle=False,
                             collate_fn=collate, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}  Test: {len(test_ds)}")

    # 3. Stage 1: pose -> gloss (CTC)
    print(f"\n=== Stage 1: CTC gloss recognition (encoder={encoder}) ===")
    ctc_model = GlossCTCModel(cfg, gloss_vocab.vocab_size, cfg.data.pose_dim, encoder)
    train_ctc_gloss(ctc_model, train_loader, dev_loader, gloss_vocab, cfg, log_dir)

    # 4. Stage 2: gloss -> text (NMT thuần text, TRAIN TRÊN GLOSS THẬT — teacher forcing, không
    #    phụ thuộc chất lượng CTC ở bước train, giống cách stage 2 kinh điển tách rời lỗi thị giác
    #    khỏi lỗi ngôn ngữ — docs/1_Thuyet_Trinh_Tong_Hop.md §A P7).
    print("\n=== Stage 2: Gloss -> Text NMT ===")
    g2t_model = Gloss2TextNMT(gloss_vocab.vocab_size, tokenizer.vocab_size,
                              d_model=cfg.train.g2t_d_model, n_layers=cfg.train.g2t_n_layers)
    train_gloss2text(g2t_model, train_loader, dev_loader, tokenizer, cfg, log_dir)

    # 5. Eval END-TO-END trên test set: pose -> CTC greedy -> gloss -> Gloss2Text greedy -> text.
    #    Đây là số liệu công bằng để so P7 với pipeline single-stage P1 (main.py --phase eval).
    print("\n=== Eval end-to-end (test set) ===")
    ctc_ckpt = torch.load(os.path.join(log_dir, "best_ctc.pt"), map_location=cfg.device)
    ctc_model.load_state_dict(ctc_ckpt["model"]); ctc_model = ctc_model.to(cfg.device).eval()
    g2t_ckpt = torch.load(os.path.join(log_dir, "best_gloss2text.pt"), map_location=cfg.device)
    g2t_model.load_state_dict(g2t_ckpt["model"]); g2t_model = g2t_model.to(cfg.device).eval()

    hyps, refs = [], []
    with torch.no_grad():
        for batch in test_loader:
            pose = batch["pose"].to(cfg.device); pose_mask = batch["pose_mask"].to(cfg.device)
            log_probs = ctc_model(pose, pose_mask)
            pred_gloss_ids = _ctc_greedy_decode(log_probs, gloss_vocab.BLANK)
            gloss_texts = [gloss_vocab.decode(ids) for ids in pred_gloss_ids]
            gloss_ids_batch = [torch.tensor(gloss_vocab.encode(t) or [gloss_vocab.UNK], dtype=torch.long)
                               for t in gloss_texts]
            max_L = max(g.size(0) for g in gloss_ids_batch)
            gloss_pad = torch.zeros(len(gloss_ids_batch), max_L, dtype=torch.long)
            for i, g in enumerate(gloss_ids_batch):
                gloss_pad[i, :g.size(0)] = g
            gloss_pad = gloss_pad.to(cfg.device)
            gloss_mask_e2e = (gloss_pad == 0)
            gen = g2t_model.greedy_decode(gloss_pad, gloss_mask_e2e, tokenizer.bos_id,
                                          tokenizer.eos_id, max_len=cfg.data.max_text_len)
            for i in range(gen.size(0)):
                hyps.append(tokenizer.decode(gen[i].tolist()))
            refs.extend(batch["text_raw"])

    e2e_bleu = corpus_bleu(hyps, [refs]).score
    print(f"[P7 two-stage] Test BLEU4 end-to-end = {e2e_bleu:.2f}")
    results = {"p7_twostage": {"test_bleu4_e2e": e2e_bleu}}
    with open(os.path.join(log_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return {"log_dir": log_dir, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=0.25)
    parser.add_argument("--tag", type=str, default="p7_twostage")
    parser.add_argument("--encoder", choices=["transformer", "stgcn", "gcn", "graph_transformer", "tcn", "perceiver"],
                        default="transformer", help="Encoder cho stage 1 CTC (docs/1_Thuyet_Trinh_Tong_Hop.md §A P1-P6)")
    args = parser.parse_args()
    run_twostage(CFG, args.subset, args.encoder, args.tag)


if __name__ == "__main__":
    main()
