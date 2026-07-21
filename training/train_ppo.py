"""Phase 2 (thay thế SCST) — PPO fine-tuning cho SLT.
Value head (Critic) + GAE + clipped surrogate objective (Schulman et al., 2017).
Xem docs/1_Thuyet_Trinh_Tong_Hop.md §C.3/C.4/C.6, docs/1_Thuyet_Trinh_Tong_Hop.md §E Experiment 7.

`cfg.train.ppo_use_clip=False` chuyển thuật toán này thành A2C (C.5, docs/1_Thuyet_Trinh_Tong_Hop.md)
-- bỏ clipped surrogate + ép về 1 epoch/rollout (không tận dụng lại batch nhiều lần như PPO thật).

Đơn giản hoá có chủ đích cho lần triển khai đầu (giảm rủi ro debug — docs/1_Thuyet_Trinh_Tong_Hop.md §I §11.3):
- Reward chỉ đặt ở bước cuối episode (sequence-level BLEU/penalty, giống SCST), KHÔNG dùng
  reward shaping incremental-BLEU từng bước (Reward 9, docs/1_Thuyet_Trinh_Tong_Hop.md §E)
  — có thể bật sau khi bản PPO cơ bản này đã ổn định.
- GAE mặc định gamma=1.0 (episode ngắn <=60 token, không cần discount), lam chỉnh qua config.
"""
import os, time, json
import torch
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from tqdm import tqdm
from .train_xe import evaluate
from .train_scst import compute_reward, repetition_penalty, length_penalty
from models.slt_transformer import ValueHead


def _terminal_reward_tensor(sample_ids, eos_id, rewards):
    """Đặt reward vào đúng bước EOS được sinh ra (cuối episode), 0 ở các bước khác.
    Trả về (reward_tensor [B,L-1], lengths [B] = số bước hợp lệ, 1-indexed)."""
    B, L = sample_ids.shape
    device = sample_ids.device
    reward_tensor = torch.zeros(B, L - 1, device=device)
    lengths = torch.full((B,), L - 1, dtype=torch.long, device=device)
    for i in range(B):
        row = sample_ids[i, 1:]
        eos_pos = (row == eos_id).nonzero(as_tuple=True)[0]
        t = eos_pos[0].item() if len(eos_pos) > 0 else L - 2
        lengths[i] = t + 1
        reward_tensor[i, t] = rewards[i]
    return reward_tensor, lengths


def _masked_gae(rewards, values, lengths, gamma: float = 1.0, lam: float = 1.0):
    """GAE (Schulman et al., 2015) có mask theo độ dài thực từng sample.
    rewards/values: [B, T]; lengths: [B]. Trả về advantages, returns, valid_mask cùng shape [B,T]."""
    B, T = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=device)
    idx = torch.arange(T, device=device).unsqueeze(0)
    valid_mask = (idx < lengths.unsqueeze(1)).float()
    is_terminal = (idx == (lengths.unsqueeze(1) - 1)).float()
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=device)
        next_value = next_value * (1 - is_terminal[:, t])
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae * (1 - is_terminal[:, t])
        advantages[:, t] = gae
    advantages = advantages * valid_mask
    returns = advantages + values * valid_mask
    return advantages, returns, valid_mask


def train_ppo(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str, xe_ckpt_path: str):
    device = cfg.device
    ckpt = torch.load(xe_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded XE checkpoint từ ep{ckpt['epoch']} BLEU={ckpt['bleu']:.2f}")

    value_head = ValueHead(model.d_model).to(device)
    params = list(model.parameters()) + list(value_head.parameters())
    opt = AdamW(params, lr=cfg.train.rl_lr, weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))

    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    # Ràng buộc tính đúng đắn của importance ratio (chặn một bug tiềm ẩn):
    # `old_log_probs` lấy từ sample_decode, tức phân phối ĐÃ CHIA temperature.
    # `new_log_probs` ở bước update lại lấy từ log_softmax(logits) KHÔNG chia temperature.
    # Nếu rl_sample_temp != 1.0 thì hai phân phối khác nhau -> ratio = exp(new − old) sai lệch
    # có hệ thống và clipped surrogate mất ý nghĩa. Với temp = 1.0 (mặc định) thì chúng trùng nhau.
    # Chặn sớm còn hơn để PPO chạy 20 epoch rồi ra một bảng số vô nghĩa.
    if abs(cfg.train.rl_sample_temp - 1.0) > 1e-6:
        raise ValueError(
            f"train_ppo yêu cầu rl_sample_temp = 1.0 (đang là {cfg.train.rl_sample_temp}).\n"
            f"Lý do: old_log_probs lấy từ phân phối CÓ temperature, new_log_probs lấy từ phân "
            f"phối KHÔNG temperature -> importance ratio sai. Muốn dùng temperature khác thì "
            f"phải chia logits cho temperature ở CẢ HAI chỗ trước khi log_softmax."
        )

    clip_eps = cfg.train.ppo_clip_eps
    use_clip = bool(getattr(cfg.train, "ppo_use_clip", True))
    # A2C (C.5, docs/1_Thuyet_Trinh_Tong_Hop.md) = PPO bỏ clipped surrogate + chỉ 1 epoch/rollout
    # (không có trust-region, không "tận dụng lại" cùng batch nhiều lần như PPO) -- cùng code path,
    # chỉ khác 2 cờ này, đúng như khuyến nghị "gộp vào ablation PPO (bỏ clip)" trong tài liệu.
    ppo_epochs = cfg.train.ppo_epochs if use_clip else 1
    gamma, lam = cfg.train.ppo_gamma, cfg.train.ppo_gae_lambda
    vf_coef = cfg.train.ppo_value_coef

    best_bleu = ckpt["bleu"]; patience = 0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.rl_epochs):
        model.train(); value_head.train()
        t0 = time.time()
        epoch_loss, epoch_reward, epoch_rep, epoch_lenratio, n = 0.0, 0.0, 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"PPO Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            refs = batch["text_raw"]
            B = pose.size(0)

            # 1. Rollout (no grad): sample từ policy hiện tại (pi_old), lưu log_prob + hidden
            #    để tính V_old(s_t) -- KHÔNG tái sử dụng memory này để backward (xem bước 2).
            with torch.no_grad():
                rollout_memory = model.encode(pose, pose_mask)
                sample_ids, old_log_probs, _, hiddens = model.sample_decode(
                    pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                    max_len=cfg.data.max_text_len, temperature=cfg.train.rl_sample_temp,
                    memory=rollout_memory, return_hidden=True)
                old_values = value_head(hiddens)  # [B, L-1]

            sample_texts = [tokenizer.decode(sample_ids[i].tolist()) for i in range(B)]
            rewards_seq = torch.tensor([compute_reward(sample_texts[i], refs[i], cfg) for i in range(B)],
                                        dtype=torch.float32, device=device)
            reward_tensor, lengths = _terminal_reward_tensor(sample_ids, tokenizer.eos_id, rewards_seq)
            advantages, returns, valid_mask = _masked_gae(reward_tensor, old_values.detach(), lengths, gamma, lam)
            adv_valid = advantages[valid_mask.bool()]
            if adv_valid.numel() > 1:
                advantages = (advantages - adv_valid.mean()) / adv_valid.std().clamp_min(1e-6) * valid_mask

            tgt_inp = sample_ids[:, :-1]
            tgt_out = sample_ids[:, 1:]

            # 2. Nhiều epoch update trên CÙNG 1 rollout (điểm mạnh chính của PPO so với SCST) --
            #    memory/hidden PHẢI tính lại (có gradient) mỗi ppo_epoch vì tham số model đã đổi
            #    sau mỗi opt.step(); nếu tái dùng rollout_memory (no_grad) thì encoder sẽ KHÔNG
            #    bao giờ nhận gradient từ PPO loss -- đây là lỗi dễ mắc khi cài PPO cho seq2seq.
            last_loss = None
            for _ in range(ppo_epochs):
                with autocast(enabled=amp_enabled):
                    memory = model.encode(pose, pose_mask)
                    logits, hidden = model.decode_step(tgt_inp, memory, pose_mask, return_hidden=True)
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs_all.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
                    new_values = value_head(hidden)

                    ratio = torch.exp((new_log_probs - old_log_probs) * valid_mask)
                    denom = valid_mask.sum().clamp_min(1)
                    if use_clip:
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                        policy_loss = -(torch.min(surr1, surr2) * valid_mask).sum() / denom
                    else:
                        # A2C: không clip. Chỉ 1 epoch/rollout (ppo_epochs=1) nên tham số CHƯA đổi khi
                        # tính new_log_probs -> ratio ≈ 1 (KHÔNG chính xác =1 vì dropout bật khác mask
                        # giữa rollout và update); gradient của ratio*A ≈ A*grad(log pi), tương đương
                        # update policy-gradient trực tiếp theo advantage như A2C nguyên bản.
                        policy_loss = -(ratio * advantages * valid_mask).sum() / denom
                    value_loss = (((new_values - returns) ** 2) * valid_mask).sum() / denom
                    loss = policy_loss + vf_coef * value_loss

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                scaler.step(opt); scaler.update()
                last_loss = loss.item()

            rep_rate = sum(repetition_penalty(t) for t in sample_texts) / B
            len_ratio = sum(length_penalty(sample_texts[i], refs[i]) for i in range(B)) / B
            epoch_loss += last_loss; epoch_reward += rewards_seq.mean().item()
            epoch_rep += rep_rate; epoch_lenratio += len_ratio; n += 1

        dev_bleu, dev_loss, samples = evaluate(model, dev_loader, tokenizer, cfg)
        log = {"epoch": epoch, "ppo_loss": epoch_loss / n, "avg_reward": epoch_reward / n,
               "avg_rep_rate": epoch_rep / n, "avg_len_ratio": epoch_lenratio / n,
               "dev_bleu4": dev_bleu, "time_s": time.time() - t0}
        history.append(log)
        print(f"[PPO Ep{epoch}] loss={epoch_loss/n:.4f} reward={epoch_reward/n:.4f} "
              f"rep={epoch_rep/n:.3f} len_ratio={epoch_lenratio/n:.3f} BLEU4={dev_bleu:.2f}")
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}\n  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0
            torch.save({"model": model.state_dict(), "value_head": value_head.state_dict(),
                       "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_ppo.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop PPO ep {epoch}"); break

    with open(os.path.join(log_dir, "ppo_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history
