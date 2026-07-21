"""RL for decoding strategy — Ý tưởng F.5, docs/1_Thuyet_Trinh_Tong_Hop.md §F.

Policy nhỏ chọn temperature sample_decode PHÙ HỢP THEO TỪNG INPUT (từ tập rời rạc
`cfg.train.decode_policy_temp_choices`) thay vì dùng 1 giá trị cố định toàn cục
(`rl_sample_temp` trong SCST/PPO) — Action = chọn 1 trong K temperature; Reward = BLEU câu kết quả.
Model SLT chính ĐÓNG BĂNG hoàn toàn (đã train xong CE/RL), chỉ policy này được cập nhật bằng
REINFORCE (baseline = trung bình reward trong batch) — tách biệt khỏi decoder fine-tuning, đúng
tinh thần "RL vượt ra khỏi fine-tuning decoder" (mục F)."""
import os, time, json
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from .train_scst import compute_reward


class DecodePolicy(nn.Module):
    """Nhận memory đã encode (pool theo thời gian) -> phân phối rời rạc trên các lựa chọn temperature."""
    def __init__(self, d_model: int, n_choices: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_choices)
        )

    def forward(self, memory, memory_mask):
        valid = (~memory_mask).float().unsqueeze(-1) if memory_mask is not None else torch.ones_like(memory[..., :1])
        pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.net(pooled)  # [B, n_choices] logits


def train_decode_policy(slt_model, train_loader, dev_loader, tokenizer, cfg, log_dir: str,
                        slt_ckpt_path: str):
    device = cfg.device
    ckpt = torch.load(slt_ckpt_path, map_location=device)
    slt_model.load_state_dict(ckpt["model"])
    slt_model = slt_model.to(device)
    slt_model.eval()
    for p in slt_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded SLT checkpoint (đóng băng) từ {slt_ckpt_path}, BLEU gốc={ckpt['bleu']:.2f}")

    temp_choices = cfg.train.decode_policy_temp_choices
    policy = DecodePolicy(slt_model.d_model, len(temp_choices)).to(device)
    opt = Adam(policy.parameters(), lr=cfg.train.decode_policy_lr)

    best_bleu = -1.0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.decode_policy_epochs):
        policy.train()
        t0 = time.time()
        epoch_loss, epoch_reward, n = 0.0, 0.0, 0
        choice_counts = [0] * len(temp_choices)

        for batch in tqdm(train_loader, desc=f"DecodePolicy Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            refs = batch["text_raw"]
            B = pose.size(0)

            with torch.no_grad():
                memory = slt_model.encode(pose, pose_mask)

            logits = policy(memory, pose_mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()                 # [B] index vào temp_choices
            log_prob = dist.log_prob(action)
            for a in action.tolist():
                choice_counts[a] += 1

            # Mỗi sample trong batch dùng ĐÚNG temperature theo action của chính nó -- sample_decode
            # hiện chỉ nhận 1 temperature/batch, nên nhóm theo action rồi decode riêng từng nhóm.
            hyp_texts = [None] * B
            with torch.no_grad():
                for choice_idx, temp in enumerate(temp_choices):
                    sel = (action == choice_idx).nonzero(as_tuple=True)[0]
                    if len(sel) == 0:
                        continue
                    sub_pose = pose[sel]; sub_mask = pose_mask[sel]; sub_mem = memory[sel]
                    ids, _, _ = slt_model.sample_decode(sub_pose, sub_mask, tokenizer.bos_id,
                                                        tokenizer.eos_id,
                                                        max_len=cfg.data.max_text_len,
                                                        temperature=temp, memory=sub_mem)
                    for k, idx in enumerate(sel.tolist()):
                        hyp_texts[idx] = tokenizer.decode(ids[k].tolist())

            reward = torch.tensor([compute_reward(hyp_texts[i], refs[i], cfg) for i in range(B)],
                                  dtype=torch.float32, device=device)
            baseline = reward.mean().detach()
            advantage = (reward - baseline).detach()
            loss = -(advantage * log_prob).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item(); epoch_reward += reward.mean().item(); n += 1

        policy.eval()
        dev_bleu = _evaluate_with_policy(slt_model, policy, dev_loader, tokenizer, cfg, temp_choices)
        dist_str = {temp_choices[i]: choice_counts[i] for i in range(len(temp_choices))}
        log = {"epoch": epoch, "loss": epoch_loss / n, "avg_reward": epoch_reward / n,
               "dev_bleu4": dev_bleu, "choice_counts": dist_str, "time_s": time.time() - t0}
        history.append(log)
        print(f"[DecodePolicy Ep{epoch}] loss={epoch_loss/n:.4f} reward={epoch_reward/n:.4f} "
              f"BLEU4={dev_bleu:.2f} choices={dist_str}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            torch.save({"policy": policy.state_dict(), "epoch": epoch, "bleu": dev_bleu,
                       "temp_choices": temp_choices},
                       os.path.join(log_dir, "best_decode_policy.pt"))

    with open(os.path.join(log_dir, "decode_policy_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history


@torch.no_grad()
def _evaluate_with_policy(slt_model, policy, loader, tokenizer, cfg, temp_choices):
    from sacrebleu import corpus_bleu
    hyps, refs = [], []
    for batch in loader:
        pose = batch["pose"].to(cfg.device); pose_mask = batch["pose_mask"].to(cfg.device)
        memory = slt_model.encode(pose, pose_mask)
        logits = policy(memory, pose_mask)
        action = logits.argmax(dim=-1)  # deterministic -- chọn temperature điểm kỳ vọng cao nhất
        B = pose.size(0)
        hyp_texts = [None] * B
        for choice_idx, temp in enumerate(temp_choices):
            sel = (action == choice_idx).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            sub_pose = pose[sel]; sub_mask = pose_mask[sel]; sub_mem = memory[sel]
            ids, _, _ = slt_model.sample_decode(sub_pose, sub_mask, tokenizer.bos_id, tokenizer.eos_id,
                                                max_len=cfg.data.max_text_len, temperature=temp,
                                                memory=sub_mem)
            for k, idx in enumerate(sel.tolist()):
                hyp_texts[idx] = tokenizer.decode(ids[k].tolist())
        hyps.extend(hyp_texts); refs.extend(batch["text_raw"])
    return corpus_bleu(hyps, [refs]).score
