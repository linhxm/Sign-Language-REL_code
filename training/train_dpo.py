"""DPO — Direct Preference Optimization (Rafailov et al., 2023) — C.7, docs/1_Thuyet_Trinh_Tong_Hop.md.

SLT không có preference con người thật -> tự sinh cặp (win, lose) bằng cách sample `dpo_n_samples`
câu/input từ chính policy hiện tại, xếp hạng bằng `compute_reward()` (train_scst.py) -- sample điểm
cao nhất làm y_w, thấp nhất làm y_l (tận dụng thẳng hạ tầng multi-sample đã có, không cần thu thập
preference ngoài). Không cần rollout nhiều epoch/batch như PPO, không cần reward model riêng như
RLHF thật -- chỉ 1 loss dạng cross-entropy trên tỉ lệ log-likelihood so với 1 policy THAM CHIẾU cố
định (bản sao đóng băng của checkpoint XE, không cập nhật trong suốt quá trình DPO)."""
import os, copy, time, json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from tqdm import tqdm
from .train_xe import evaluate
from .train_scst import compute_reward, repetition_penalty, length_penalty
from .train_mrt import _candidate_logprobs


def train_dpo(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str, xe_ckpt_path: str):
    device = cfg.device
    ckpt = torch.load(xe_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded XE checkpoint từ ep{ckpt['epoch']} BLEU={ckpt['bleu']:.2f}")

    # Policy tham chiếu CỐ ĐỊNH (Rafailov et al. 2023) -- bản sao đóng băng ngay tại checkpoint XE,
    # KHÔNG cập nhật trong suốt quá trình DPO. Nếu không có policy tham chiếu, loss DPO suy biến
    # về việc tối đa hoá thẳng (logp_w - logp_l) không giới hạn -> drift ngôn ngữ mất kiểm soát,
    # đúng vai trò "KL-implicit regularization" mà DPO thừa hưởng từ RLHF.
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    opt = AdamW(model.parameters(), lr=cfg.train.rl_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))
    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    n_samples = max(2, cfg.train.dpo_n_samples)
    beta = cfg.train.dpo_beta

    best_bleu = ckpt["bleu"]; patience = 0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.rl_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_margin, epoch_rep, epoch_lenratio, n = 0.0, 0.0, 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"DPO Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            refs = batch["text_raw"]
            B = pose.size(0)

            # 1. Sample n_samples câu/input từ policy hiện tại, xếp hạng theo reward -> (win, lose).
            with torch.no_grad():
                memory_ng = model.encode(pose, pose_mask)
                draws = [model.sample_decode(pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                                             max_len=cfg.data.max_text_len,
                                             temperature=cfg.train.rl_sample_temp,
                                             memory=memory_ng)[0]
                        for _ in range(n_samples)]
            draw_texts = [[tokenizer.decode(draws[j][i].tolist()) for j in range(n_samples)]
                         for i in range(B)]
            draw_rewards = [[compute_reward(draw_texts[i][j], refs[i], cfg) for j in range(n_samples)]
                            for i in range(B)]

            win_ids, lose_ids, win_texts, lose_texts = [], [], [], []
            for i in range(B):
                order = sorted(range(n_samples), key=lambda j: draw_rewards[i][j])
                lose_j, win_j = order[0], order[-1]
                win_ids.append([draws[win_j][i]])
                lose_ids.append([draws[lose_j][i]])
                win_texts.append(draw_texts[i][win_j]); lose_texts.append(draw_texts[i][lose_j])

            # 2. log pi_theta(y|x) (CÓ gradient) và log pi_ref(y|x) (KHÔNG gradient, model đóng băng).
            with autocast(enabled=amp_enabled):
                memory = model.encode(pose, pose_mask)
                logp_w = _candidate_logprobs(model, memory, pose_mask, win_ids,
                                             tokenizer.pad_id, tokenizer.eos_id).squeeze(1)
                logp_l = _candidate_logprobs(model, memory, pose_mask, lose_ids,
                                             tokenizer.pad_id, tokenizer.eos_id).squeeze(1)
                with torch.no_grad():
                    ref_memory = ref_model.encode(pose, pose_mask)
                    ref_logp_w = _candidate_logprobs(ref_model, ref_memory, pose_mask, win_ids,
                                                     tokenizer.pad_id, tokenizer.eos_id).squeeze(1)
                    ref_logp_l = _candidate_logprobs(ref_model, ref_memory, pose_mask, lose_ids,
                                                     tokenizer.pad_id, tokenizer.eos_id).squeeze(1)

                logits_diff = beta * ((logp_w - ref_logp_w) - (logp_l - ref_logp_l))
                loss = -F.logsigmoid(logits_diff).mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()

            rep_rate = sum(repetition_penalty(t) for t in win_texts + lose_texts) / (2 * B)
            len_ratio = sum(length_penalty(win_texts[i], refs[i]) for i in range(B)) / B

            epoch_loss += loss.item()
            epoch_margin += (logits_diff.detach().mean().item() / max(beta, 1e-8))
            epoch_rep += rep_rate; epoch_lenratio += len_ratio; n += 1

        dev_bleu, dev_loss, samples = evaluate(model, dev_loader, tokenizer, cfg)
        log = {"epoch": epoch, "dpo_loss": epoch_loss / n, "avg_margin": epoch_margin / n,
               "avg_rep_rate": epoch_rep / n, "avg_len_ratio": epoch_lenratio / n,
               "dev_bleu4": dev_bleu, "time_s": time.time() - t0}
        history.append(log)
        print(f"[DPO Ep{epoch}] loss={epoch_loss/n:.4f} margin={epoch_margin/n:.4f} "
              f"rep={epoch_rep/n:.3f} len_ratio={epoch_lenratio/n:.3f} BLEU4={dev_bleu:.2f}")
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}\n  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_dpo.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop DPO ep {epoch}"); break

    with open(os.path.join(log_dir, "dpo_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history
