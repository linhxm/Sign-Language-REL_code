"""RL cho lựa chọn CV rời rạc NGOÀI phạm vi decoder text — hợp nhất 3 ý tưởng cùng 1 cơ chế
REINFORCE (docs/1_Thuyet_Trinh_Tong_Hop.md §F), chọn qua tham số `target`/`mode` thay vì viết
3 script riêng biệt:
  - F.6 (RL for frame selection, Experiment 8, docs/1_Thuyet_Trinh_Tong_Hop.md §E): target="frame", mode="topk"
  - F.9 (RL for adaptive temporal sampling): target="frame", mode="adaptive"
  - F.8 (RL for dynamic landmark selection): target="landmark"

Policy nhỏ (GRU 2 chiều) huấn luyện RIÊNG bằng REINFORCE (baseline = trung bình reward trong batch,
kiểu SCST rẻ) — reward = BLEU của model SLT CHÍNH (đã train xong, load từ checkpoint có sẵn, ĐÓNG
BĂNG hoàn toàn trong suốt quá trình này) khi decode trên pose đã qua policy lọc, trừ phạt theo tỉ lệ
đơn vị (frame/nhóm-landmark) giữ lại để khuyến khích nén. Tách biệt hoàn toàn khỏi
SLTTransformer/SCST/PPO — đúng tinh thần "RL vượt ra khỏi fine-tuning decoder" (mục F).

Đơn giản hoá có chủ đích (ghi rõ để không nhầm là thiếu sót):
- **Frame bị loại được ZERO-HOÁ, KHÔNG bị xoá khỏi chuỗi** (`_apply_frame_mask`) — độ dài chuỗi giữ
  nguyên, encoder vẫn xử lý đủ T frame. Vì vậy experiment này đo được "tín hiệu chọn frame có
  giúp/hại BLEU không" (soft frame-masking) NHƯNG **KHÔNG giảm compute thực tế** và KHÔNG chứng minh
  được "nén hiệu quả" — đúng như đã đính chính ở docs/8 Experiment 8 và H5 (docs/10). Muốn giảm
  compute thật cần re-index chuỗi theo frame đã chọn trước khi vào encoder (hướng mở rộng).
- Hoạt động trên pose ĐÃ qua truncate `max_frames=300` của loader chuẩn (data/dataset.py), không
  phải chuỗi thô chưa cắt — tránh phải xây lại toàn bộ đường dữ liệu chỉ cho phần thử nghiệm này.
- target="landmark": policy gộp thông tin theo thời gian (mean-pool hidden GRU) rồi quyết định
  MỘT LẦN/câu cho mỗi nhóm (body/tay trái/tay phải) thay vì quyết định lại mỗi frame — bản đơn giản
  hoá tractable của ý tưởng gốc "theo từng frame/đoạn" (F.8), vẫn phản ánh đúng cơ chế "ưu tiên tập
  con landmark còn tin cậy" khi 1 nhóm bị occlude nặng toàn video.
"""
import os, time, json
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from .train_scst import compute_reward

N_BODY_DIM, N_LHAND_DIM, N_RHAND_DIM = 99, 42, 42  # khớp layout 183-d của data/extract_poses.py


class SelectionPolicy(nn.Module):
    def __init__(self, pose_dim: int, hidden: int = 64, target: str = "frame"):
        super().__init__()
        self.target = target
        self.gru = nn.GRU(pose_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1 if target == "frame" else 3)

    def forward(self, pose, pose_mask):
        h, _ = self.gru(pose)
        if self.target == "frame":
            logits = self.head(h).squeeze(-1)            # [B,T]
            if pose_mask is not None:
                logits = logits.masked_fill(pose_mask, -1e9)
            return logits
        pooled = h.mean(dim=1)                            # [B,2H] -- gộp thời gian
        return self.head(pooled)                          # [B,3]


def _apply_frame_mask(pose, keep_mask):  # keep_mask: [B,T] bool, True=giữ
    return pose * keep_mask.unsqueeze(-1).float()


def _apply_landmark_mask(pose, keep_group):  # keep_group: [B,3] bool
    body = pose[..., :N_BODY_DIM] * keep_group[:, 0:1].unsqueeze(1).float()
    lhand = pose[..., N_BODY_DIM:N_BODY_DIM + N_LHAND_DIM] * keep_group[:, 1:2].unsqueeze(1).float()
    rhand = pose[..., N_BODY_DIM + N_LHAND_DIM:] * keep_group[:, 2:3].unsqueeze(1).float()
    return torch.cat([body, lhand, rhand], dim=-1)


def train_selection_policy(slt_model, train_loader, dev_loader, tokenizer, cfg, log_dir: str,
                           slt_ckpt_path: str, target: str = "frame", mode: str = "topk"):
    assert target in ("frame", "landmark")
    assert mode in ("topk", "adaptive")
    device = cfg.device
    ckpt = torch.load(slt_ckpt_path, map_location=device)
    slt_model.load_state_dict(ckpt["model"])
    slt_model = slt_model.to(device)
    slt_model.eval()
    for p in slt_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded SLT checkpoint (đóng băng) từ {slt_ckpt_path}, BLEU gốc={ckpt['bleu']:.2f}")

    policy = SelectionPolicy(cfg.data.pose_dim, target=target).to(device)
    opt = Adam(policy.parameters(), lr=cfg.train.selection_policy_lr)

    keep_ratio = cfg.train.selection_policy_keep_ratio
    frame_penalty_w = cfg.train.selection_policy_frame_penalty

    best_bleu = -1.0
    history = []
    os.makedirs(log_dir, exist_ok=True)
    tag = f"{target}_{mode}" if target == "frame" else target

    for epoch in range(cfg.train.selection_policy_epochs):
        policy.train()
        t0 = time.time()
        epoch_loss, epoch_reward, epoch_keep_frac, n = 0.0, 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"SelectPolicy[{tag}] Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            refs = batch["text_raw"]
            B = pose.size(0)

            logits = policy(pose, pose_mask)

            if target == "frame":
                probs = torch.sigmoid(logits)
                valid_len = (~pose_mask).float().sum(dim=1)
                if mode == "topk":
                    dist = torch.distributions.Bernoulli(probs=probs)
                    action = dist.sample().masked_fill(pose_mask, 0.0)
                    log_prob = (dist.log_prob(action) * (~pose_mask).float()).sum(dim=1)
                    keep_mask = action.bool()
                else:  # "adaptive" (F.9) -- resample budget K theo mật độ probs, dày ở đoạn xác suất cao
                    K = (valid_len * keep_ratio).clamp_min(1).long()
                    Kmax = max(1, int(K.max().item()))
                    sample_probs = probs.masked_fill(pose_mask, 0.0) + 1e-8
                    idx = torch.multinomial(sample_probs, Kmax, replacement=True)  # [B,Kmax]
                    chosen_p = torch.gather(probs, 1, idx).clamp_min(1e-8)
                    log_prob = torch.log(chosen_p).sum(dim=1)
                    keep_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, idx, True)
                    keep_mask = keep_mask & (~pose_mask)
                keep_frac = keep_mask.float().sum(dim=1) / valid_len.clamp_min(1)
                filtered_pose = _apply_frame_mask(pose, keep_mask)
            else:  # target == "landmark" (F.8)
                probs = torch.sigmoid(logits)  # [B,3]
                dist = torch.distributions.Bernoulli(probs=probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=1)
                keep_group = action.bool()
                keep_frac = keep_group.float().mean(dim=1)
                filtered_pose = _apply_landmark_mask(pose, keep_group)

            with torch.no_grad():
                gen = slt_model.greedy_decode(filtered_pose, pose_mask, tokenizer.bos_id,
                                              tokenizer.eos_id, max_len=cfg.data.max_text_len)
                hyp_texts = [tokenizer.decode(gen[i].tolist()) for i in range(B)]
                bleu_reward = torch.tensor(
                    [compute_reward(hyp_texts[i], refs[i], cfg) for i in range(B)],
                    dtype=torch.float32, device=device)

            reward = bleu_reward - frame_penalty_w * keep_frac
            baseline = reward.mean().detach()  # baseline batch-mean (giảm variance, kiểu SCST rẻ)
            advantage = (reward - baseline).detach()
            loss = -(advantage * log_prob).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item(); epoch_reward += reward.mean().item()
            epoch_keep_frac += keep_frac.mean().item(); n += 1

        policy.eval()
        dev_bleu = _evaluate_with_policy(slt_model, policy, dev_loader, tokenizer, cfg, target, keep_ratio)
        log = {"epoch": epoch, "loss": epoch_loss / n, "avg_reward": epoch_reward / n,
               "avg_keep_frac": epoch_keep_frac / n, "dev_bleu4": dev_bleu, "time_s": time.time() - t0}
        history.append(log)
        print(f"[SelectPolicy {tag} Ep{epoch}] loss={epoch_loss/n:.4f} reward={epoch_reward/n:.4f} "
              f"keep_frac={epoch_keep_frac/n:.3f} BLEU4={dev_bleu:.2f}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            torch.save({"policy": policy.state_dict(), "epoch": epoch, "bleu": dev_bleu,
                       "target": target, "mode": mode},
                       os.path.join(log_dir, f"best_selection_policy_{tag}.pt"))

    with open(os.path.join(log_dir, f"selection_policy_{tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history


@torch.no_grad()
def _evaluate_with_policy(slt_model, policy, loader, tokenizer, cfg, target, keep_ratio):
    """Eval LUÔN dùng top-K xác suất tất định (không sample) bất kể mode train là gì (topk/adaptive)
    -- cần tính tái lập được cho so sánh cuối, không phải vì mode đó "tốt hơn"."""
    from sacrebleu import corpus_bleu
    hyps, refs = [], []
    for batch in loader:
        pose = batch["pose"].to(cfg.device); pose_mask = batch["pose_mask"].to(cfg.device)
        logits = policy(pose, pose_mask)
        if target == "frame":
            probs = torch.sigmoid(logits).masked_fill(pose_mask, -1.0)
            valid_len = (~pose_mask).float().sum(dim=1)
            K = (valid_len * keep_ratio).clamp_min(1).long()
            Kmax = max(1, int(K.max().item()))
            topk_idx = probs.topk(Kmax, dim=1).indices
            keep_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, topk_idx, True)
            keep_mask = keep_mask & (~pose_mask)
            filtered_pose = _apply_frame_mask(pose, keep_mask)
        else:
            keep_group = torch.sigmoid(logits) > 0.5
            filtered_pose = _apply_landmark_mask(pose, keep_group)
        gen = slt_model.greedy_decode(filtered_pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                                      max_len=cfg.data.max_text_len)
        for i in range(gen.size(0)):
            hyps.append(tokenizer.decode(gen[i].tolist())); refs.append(batch["text_raw"][i])
    return corpus_bleu(hyps, [refs]).score
