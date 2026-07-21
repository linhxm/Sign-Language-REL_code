"""RAML — Reward Augmented Maximum Likelihood (Norouzi et al., NeurIPS 2016) — C.10,
docs/1_Thuyet_Trinh_Tong_Hop.md. KHÔNG cần rollout/decode tự hồi quy trong lúc train (khác
SCST/PPO/MRT) — sample target NHIỄU quanh ground-truth theo phân phối exp(R(y)/tau) rồi train bằng
MLE THƯỜNG trên các target đã nhiễu, y hệt Cross-Entropy phase về mặt kỹ thuật (chỉ khác target) —
rẻ hơn nhiều vì không cần forward decode từng bước.

Dùng reward Hamming-distance đơn giản (không phải BLEU/reward tổng hợp của compute_reward() trong
train_scst.py) vì closed-form sampling số lượng edit `q(m) ∝ C(L,m)(V-1)^m·exp(-m/tau)` (Norouzi
et al. 2016, mục 3.2) chỉ đúng với reward dạng Hamming-distance — đây là lý do RAML "gián tiếp hơn"
SCST thật (đã nêu ở docs/1_Thuyet_Trinh_Tong_Hop.md §C.10)."""
import os, time, json, math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from tqdm import tqdm
from .train_xe import evaluate


def _sample_num_edits(L: int, vocab_size: int, tau: float, max_edits: int,
                      rng: np.random.RandomState) -> int:
    """q(m) ∝ C(L,m) * (V-1)^m * exp(-m/tau), m = 0..max_edits."""
    log_w = np.empty(max_edits + 1)
    for m in range(max_edits + 1):
        log_w[m] = (math.lgamma(L + 1) - math.lgamma(m + 1) - math.lgamma(L - m + 1)
                    + m * math.log(max(vocab_size - 1, 1)) - m / tau)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    return int(rng.choice(max_edits + 1, p=w))


def _perturb(ids, vocab_size: int, pad_id: int, bos_id: int, eos_id: int, unk_id: int,
            tau: float, max_edits_ratio: float, rng: np.random.RandomState):
    """ids: list[int] gồm cả <bos>/<eos>. Chỉ nhiễu token NỘI DUNG (bỏ qua bos/eos/pad)."""
    special = {pad_id, bos_id, eos_id}
    content_pos = [i for i, t in enumerate(ids) if t not in special]
    L = len(content_pos)
    if L == 0:
        return list(ids)
    max_edits = max(1, int(math.ceil(L * max_edits_ratio)))
    max_edits = min(max_edits, L)
    m = _sample_num_edits(L, vocab_size, tau, max_edits, rng)
    out = list(ids)
    if m > 0:
        edit_positions = rng.choice(content_pos, size=m, replace=False)
        lo = max(4, unk_id + 1)  # bỏ qua các special token đầu bảng (pad/bos/eos/unk)
        for p in edit_positions:
            out[p] = int(rng.randint(lo, vocab_size))
    return out


def train_raml(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str, xe_ckpt_path: str):
    device = cfg.device
    ckpt = torch.load(xe_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded XE checkpoint từ ep{ckpt['epoch']} BLEU={ckpt['bleu']:.2f}")

    opt = AdamW(model.parameters(), lr=cfg.train.rl_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))
    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id,
                                    label_smoothing=cfg.model.label_smoothing)

    tau = cfg.train.raml_tau
    n_samples = max(1, cfg.train.raml_n_samples)
    max_edits_ratio = cfg.train.raml_max_edits_ratio
    rng = np.random.RandomState(cfg.seed)

    best_bleu = ckpt["bleu"]; patience = 0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.rl_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"RAML Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            text_ids_cpu = batch["text_ids"]
            text_lens = batch["text_lens"]
            B = pose.size(0)

            # Sinh n_samples target nhiễu/câu -- thuần CPU, không forward model (điểm mạnh của RAML).
            perturbed = []
            for i in range(B):
                ids = text_ids_cpu[i, :text_lens[i]].tolist()
                for _ in range(n_samples):
                    perturbed.append(_perturb(ids, tokenizer.vocab_size, tokenizer.pad_id,
                                              tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id,
                                              tau, max_edits_ratio, rng))
            max_L = max(len(p) for p in perturbed)
            tgt = torch.full((len(perturbed), max_L), tokenizer.pad_id, dtype=torch.long)
            for i, p in enumerate(perturbed):
                tgt[i, :len(p)] = torch.tensor(p, dtype=torch.long)
            tgt = tgt.to(device)

            pose_rep = pose.repeat_interleave(n_samples, dim=0)
            pose_mask_rep = pose_mask.repeat_interleave(n_samples, dim=0)
            tgt_inp = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            tgt_inp_mask = (tgt_inp == tokenizer.pad_id)

            with autocast(enabled=amp_enabled):
                logits = model(pose_rep, pose_mask_rep, tgt_inp, tgt_inp_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()

            epoch_loss += loss.item(); n += 1

        dev_bleu, dev_loss, samples = evaluate(model, dev_loader, tokenizer, cfg)
        log = {"epoch": epoch, "raml_loss": epoch_loss / n, "dev_bleu4": dev_bleu,
               "time_s": time.time() - t0}
        history.append(log)
        print(f"[RAML Ep{epoch}] loss={epoch_loss/n:.4f} BLEU4={dev_bleu:.2f}")
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}\n  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_raml.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop RAML ep {epoch}"); break

    with open(os.path.join(log_dir, "raml_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history
