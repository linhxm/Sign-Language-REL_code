"""Stage 2 của P7 (docs/1_Thuyet_Trinh_Tong_Hop.md §A): gloss -> text, NMT thuần text (Cross-Entropy, không đụng
pose nữa). Cấu trúc training loop giống hệt training/train_xe.py (Phase 1 của pipeline single-stage)
— khác duy nhất input là `gloss_ids`/`gloss_mask` thay vì `pose`/`pose_mask`."""
import os, time, json, math
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from sacrebleu import corpus_bleu
from tqdm import tqdm
from .train_xe import make_warmup_scheduler


@torch.no_grad()
def evaluate_gloss2text(model, loader, tokenizer, cfg, max_samples_log: int = 5):
    model.eval()
    device = cfg.device
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    hyps, refs = [], []
    total_loss, n = 0.0, 0
    samples = []
    for batch in loader:
        gloss = batch["gloss_ids"].to(device); gloss_mask = batch["gloss_mask"].to(device)
        text = batch["text_ids"].to(device); text_mask = batch["text_mask"].to(device)
        inp = text[:, :-1]; tgt = text[:, 1:]; inp_mask = text_mask[:, :-1]
        logits = model(gloss, gloss_mask, inp, inp_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        total_loss += loss.item(); n += 1

        gen = model.greedy_decode(gloss, gloss_mask, tokenizer.bos_id, tokenizer.eos_id,
                                  max_len=cfg.data.max_text_len)
        for i in range(gen.size(0)):
            pred_text = tokenizer.decode(gen[i].tolist())
            ref_text = batch["text_raw"][i]
            hyps.append(pred_text); refs.append(ref_text)
            if len(samples) < max_samples_log:
                samples.append((ref_text, pred_text))
    bleu = corpus_bleu(hyps, [refs]).score
    return bleu, total_loss / max(1, n), samples


def train_gloss2text(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str):
    device = cfg.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=cfg.model.label_smoothing)
    opt = AdamW(model.parameters(), lr=cfg.train.g2t_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))
    total_steps = cfg.train.g2t_epochs * len(train_loader)
    scheduler = make_warmup_scheduler(opt, cfg.train.xe_warmup_steps, total_steps)
    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_bleu = -1.0
    patience = 0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.g2t_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, n_batches = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Gloss2Text Ep{epoch}"):
            gloss = batch["gloss_ids"].to(device); gloss_mask = batch["gloss_mask"].to(device)
            text = batch["text_ids"].to(device); text_mask = batch["text_mask"].to(device)
            inp = text[:, :-1]; tgt = text[:, 1:]; inp_mask = text_mask[:, :-1]

            with autocast(enabled=amp_enabled):
                logits = model(gloss, gloss_mask, inp, inp_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            scheduler.step()
            epoch_loss += loss.item(); n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        dev_bleu, dev_loss, samples = evaluate_gloss2text(model, dev_loader, tokenizer, cfg)
        elapsed = time.time() - t0
        log = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
               "dev_bleu4": dev_bleu, "time_s": elapsed}
        history.append(log)
        print(f"[Gloss2Text Ep{epoch}] loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
              f"BLEU4={dev_bleu:.2f} time={elapsed:.0f}s")
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}\n  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_gloss2text.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop Gloss2Text ep {epoch}"); break

    with open(os.path.join(log_dir, "gloss2text_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history
