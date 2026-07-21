"""Stage 1 của P7 (docs/1_Thuyet_Trinh_Tong_Hop.md §A): pose -> gloss bằng CTC loss (Graves et al., 2006).
Metric chính = WER (Word Error Rate, chuẩn cho gloss recognition trong literature SLT, khác BLEU
dùng cho stage 2/end-to-end) — xem docs/1_Thuyet_Trinh_Tong_Hop.md §F.
"""
import os, time, json
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from tqdm import tqdm
from .train_xe import make_warmup_scheduler


def _ctc_greedy_decode(log_probs, blank_id: int = 0):
    """Greedy CTC decode chuẩn: argmax mỗi bước -> xoá token lặp liên tiếp -> xoá blank."""
    ids = log_probs.argmax(dim=-1)  # [B,T]
    results = []
    for row in ids.tolist():
        out, prev = [], None
        for tok in row:
            if tok != prev and tok != blank_id:
                out.append(tok)
            prev = tok
        results.append(out)
    return results


def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


@torch.no_grad()
def evaluate_ctc(model, loader, gloss_vocab, cfg, max_samples_log: int = 5):
    model.eval()
    device = cfg.device
    total_edits, total_len = 0, 0
    samples = []
    for batch in loader:
        pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
        log_probs = model(pose, pose_mask)
        pred_ids = _ctc_greedy_decode(log_probs, gloss_vocab.BLANK)
        for i, pred in enumerate(pred_ids):
            gt = batch["gloss_raw"][i].split()
            pred_tokens = gloss_vocab.decode(pred).split()
            total_edits += _edit_distance(gt, pred_tokens)
            total_len += max(1, len(gt))
            if len(samples) < max_samples_log:
                samples.append((" ".join(gt), " ".join(pred_tokens)))
    wer = total_edits / max(1, total_len)
    return wer, samples


def train_ctc_gloss(model, train_loader, dev_loader, gloss_vocab, cfg, log_dir: str):
    device = cfg.device
    model = model.to(device)
    criterion = nn.CTCLoss(blank=gloss_vocab.BLANK, zero_infinity=True)
    opt = AdamW(model.parameters(), lr=cfg.train.ctc_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))
    total_steps = cfg.train.ctc_epochs * len(train_loader)
    scheduler = make_warmup_scheduler(opt, cfg.train.xe_warmup_steps, total_steps)
    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_wer = float("inf")
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.ctc_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"CTC Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            gloss = batch["gloss_ids"].to(device)
            gloss_lens = batch["gloss_lens"]      # CTCLoss cần lengths trên CPU
            pose_lens = batch["pose_lens"]

            with autocast(enabled=amp_enabled):
                log_probs = model(pose, pose_mask)  # [B,T,V]
            # CTCLoss không ổn định số học dưới fp16 -> luôn tính loss ở fp32, ngoài autocast
            # (chỉ phần forward encoder/head hưởng lợi AMP, không phải bản thân loss).
            log_probs_ctc = log_probs.float().transpose(0, 1)  # [T,B,V]
            loss = criterion(log_probs_ctc, gloss, pose_lens, gloss_lens)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            scheduler.step()
            epoch_loss += loss.item(); n += 1

        dev_wer, samples = evaluate_ctc(model, dev_loader, gloss_vocab, cfg)
        log = {"epoch": epoch, "train_loss": epoch_loss / n, "dev_wer": dev_wer, "time_s": time.time() - t0}
        history.append(log)
        print(f"[CTC Ep{epoch}] loss={epoch_loss/n:.4f} dev_WER={dev_wer:.3f}")
        for gt, pred in samples[:2]:
            print(f"  GT gloss  : {gt}\n  PRED gloss: {pred}")

        if dev_wer < best_wer:
            best_wer = dev_wer
            torch.save({"model": model.state_dict(), "epoch": epoch, "wer": dev_wer},
                       os.path.join(log_dir, "best_ctc.pt"))

    with open(os.path.join(log_dir, "ctc_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_wer, history
