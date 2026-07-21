"""Phase 1: Cross-entropy pretraining. PHẢI chạy phase này trước khi RL."""
import os, time, math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from sacrebleu import corpus_bleu
from tqdm import tqdm

def make_warmup_scheduler(opt, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(opt, lr_lambda)


def train_xe(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str):
    device = cfg.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id,
                                     label_smoothing=cfg.model.label_smoothing)
    opt = AdamW(model.parameters(), lr=cfg.train.xe_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))

    total_steps = cfg.train.xe_epochs * len(train_loader)
    scheduler = make_warmup_scheduler(opt, cfg.train.xe_warmup_steps, total_steps)

    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_bleu = -1.0
    patience = 0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.xe_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0; n_batches = 0

        for batch in tqdm(train_loader, desc=f"XE Ep{epoch}"):
            pose = batch["pose"].to(device)
            pose_mask = batch["pose_mask"].to(device)
            text = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)

            # Teacher forcing: input = text[:-1], target = text[1:]
            inp = text[:, :-1]
            tgt = text[:, 1:]
            inp_mask = text_mask[:, :-1]

            with autocast(enabled=amp_enabled):
                logits = model(pose, pose_mask, inp, inp_mask)  # [B, L-1, V]
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            scheduler.step()

            epoch_loss += loss.item(); n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        dev_bleu, dev_loss, samples = evaluate(model, dev_loader, tokenizer, cfg)
        elapsed = time.time() - t0

        log = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
               "dev_bleu4": dev_bleu, "time_s": elapsed,
               "lr": scheduler.get_last_lr()[0]}
        history.append(log)
        print(f"[Ep{epoch}] loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
              f"BLEU4={dev_bleu:.2f} time={elapsed:.0f}s")

        # Print 2 sample translations
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}")
            print(f"  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_xe.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop at epoch {epoch}"); break

    import json
    with open(os.path.join(log_dir, "xe_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history


@torch.no_grad()
def evaluate(model, loader, tokenizer, cfg, max_samples_log: int = 5):
    model.eval()
    device = cfg.device
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    hyps, refs = [], []
    total_loss, n = 0.0, 0
    samples_for_log = []

    for batch in loader:
        pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
        text = batch["text_ids"].to(device); text_mask = batch["text_mask"].to(device)
        # Loss (teacher forcing)
        inp = text[:, :-1]; tgt = text[:, 1:]; inp_mask = text_mask[:, :-1]
        logits = model(pose, pose_mask, inp, inp_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        total_loss += loss.item(); n += 1

        # Greedy decode for BLEU
        gen = model.greedy_decode(pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                                   max_len=cfg.data.max_text_len)
        for i in range(gen.size(0)):
            pred_text = tokenizer.decode(gen[i].tolist())
            ref_text = batch["text_raw"][i]
            hyps.append(pred_text); refs.append(ref_text)
            if len(samples_for_log) < max_samples_log:
                samples_for_log.append((ref_text, pred_text))

    bleu = corpus_bleu(hyps, [refs]).score
    return bleu, total_loss / max(1, n), samples_for_log
