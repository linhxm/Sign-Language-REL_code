"""Minimum Risk Training (MRT, Shen et al. ACL 2016) — C.9, docs/1_Thuyet_Trinh_Tong_Hop.md.

L_MRT = sum_{y in S} Q(y|x) * (1 - R(y)),  Q(y|x) = softmax(alpha * log pi_theta(y|x)) trong tập
candidate S (N-best lấy từ sample HOẶC beam search — cfg.train.mrt_candidate_source). Khác SCST:
không cần baseline riêng (variance thấp hơn nhờ trung bình trên nhiều candidate cùng lúc thay vì
1 sample so với greedy).

`mrt_candidate_source="beam"` = cách hiện thực Ý tưởng F.13 (RL/MRT cho beam search policy,
docs/1_Thuyet_Trinh_Tong_Hop.md §F) — tối ưu các nhánh beam theo reward thay vì log-prob thô,
tái dùng `beam_search_decode(..., return_all_beams=True)` thay vì viết riêng 1 vòng lặp beam khác.
"""
import os, time, json
import torch
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from tqdm import tqdm
from .train_xe import evaluate
from .train_scst import compute_reward, repetition_penalty, length_penalty


def _true_length(seq_1d: torch.Tensor, eos_id: int) -> int:
    """Độ dài THỰC (tính đến và bao gồm <eos> đầu tiên) của 1 chuỗi sinh bởi sample_decode/
    beam_search_decode -- các chuỗi này được nhồi LẶP LẠI eos_id sau khi kết thúc (không phải
    pad_id=0), nên KHÔNG thể dùng thẳng `.size(0)` làm độ dài, sẽ tính dư log-prob của các bước
    eos lặp lại phía sau. Trả `size(0)` nếu không tìm thấy eos (câu bị cắt ở max_len)."""
    eos_pos = (seq_1d == eos_id).nonzero(as_tuple=True)[0]
    return int(eos_pos[0].item()) + 1 if len(eos_pos) > 0 else int(seq_1d.size(0))


def _candidate_logprobs(model, memory, pose_mask, cand_ids_per_sample, pad_id: int, eos_id: int):
    """cand_ids_per_sample: list độ dài B, mỗi phần tử là list N tensor 1D (có <bos> ở đầu).
    Trả log-prob tổng theo từng candidate [B, N] (CÓ gradient) bằng đúng 1 forward teacher-forcing
    trên toàn bộ B*N candidate cùng lúc (thay vì B*N forward riêng lẻ). Dùng `_true_length` (không
    phải `.size(0)`) để không tính dư log-prob của các bước sau <eos> đầu tiên."""
    B = len(cand_ids_per_sample)
    N = len(cand_ids_per_sample[0])
    flat = [seq for sample in cand_ids_per_sample for seq in sample]
    lengths = torch.tensor([_true_length(s, eos_id) for s in flat], device=memory.device)
    max_L = max(2, max(s.size(0) for s in flat))
    padded = torch.full((B * N, max_L), pad_id, dtype=torch.long, device=memory.device)
    for i, s in enumerate(flat):
        padded[i, :s.size(0)] = s
    tgt_inp = padded[:, :-1]
    tgt_out = padded[:, 1:]
    valid = (torch.arange(max_L - 1, device=memory.device)[None, :] < (lengths[:, None] - 1)).float()

    mem_rep = memory.repeat_interleave(N, dim=0)
    mask_rep = pose_mask.repeat_interleave(N, dim=0) if pose_mask is not None else None
    logits = model.decode_step(tgt_inp, mem_rep, mask_rep)
    log_probs_all = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs_all.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1) * valid
    return token_lp.sum(dim=1).view(B, N)


def train_mrt(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str, xe_ckpt_path: str):
    device = cfg.device
    ckpt = torch.load(xe_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded XE checkpoint từ ep{ckpt['epoch']} BLEU={ckpt['bleu']:.2f}")

    opt = AdamW(model.parameters(), lr=cfg.train.rl_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))
    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    n_cand = cfg.train.mrt_n_candidates
    alpha = cfg.train.mrt_alpha
    source = cfg.train.mrt_candidate_source

    best_bleu = ckpt["bleu"]; patience = 0
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.rl_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_reward, epoch_rep, epoch_lenratio, n = 0.0, 0.0, 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"MRT Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            refs = batch["text_raw"]
            B = pose.size(0)

            # 1. Sinh N candidate/input (KHÔNG cần gradient — chỉ cần chuỗi token cố định).
            with torch.no_grad():
                memory_ng = model.encode(pose, pose_mask)
                if source == "beam":
                    _, all_beams = model.beam_search_decode(
                        pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                        max_len=cfg.data.max_text_len, beam_size=n_cand, memory=memory_ng,
                        return_all_beams=True)
                    cand_ids = [[ys for ys, _ in beams_b[:n_cand]] for beams_b in all_beams]
                    for c in cand_ids:  # beam có thể kết thúc sớm với < n_cand nhánh còn sống
                        while len(c) < n_cand:
                            c.append(c[-1])
                else:
                    draws = [model.sample_decode(pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                                                 max_len=cfg.data.max_text_len,
                                                 temperature=cfg.train.rl_sample_temp,
                                                 memory=memory_ng)[0]
                            for _ in range(n_cand)]
                    cand_ids = [[draws[j][i] for j in range(n_cand)] for i in range(B)]

            cand_texts = [[tokenizer.decode(c.tolist()) for c in cand_ids[i]] for i in range(B)]
            rewards = torch.tensor(
                [[compute_reward(cand_texts[i][j], refs[i], cfg) for j in range(n_cand)] for i in range(B)],
                dtype=torch.float32, device=device)  # [B, N]

            # 2. Q(y|x) và risk loss — CÓ gradient (encode lại, khác nhánh sinh candidate ở trên).
            with autocast(enabled=amp_enabled):
                memory = model.encode(pose, pose_mask)
                seq_logp = _candidate_logprobs(model, memory, pose_mask, cand_ids,
                                               tokenizer.pad_id, tokenizer.eos_id)
                Q = torch.softmax(alpha * seq_logp, dim=1)  # [B,N]
                risk = (Q * (1.0 - rewards)).sum(dim=1)
                loss = risk.mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()

            flat_texts = [t for row in cand_texts for t in row]
            flat_refs = [refs[i] for i in range(B) for _ in range(n_cand)]
            rep_rate = sum(repetition_penalty(t) for t in flat_texts) / len(flat_texts)
            len_ratio = sum(length_penalty(flat_texts[k], flat_refs[k])
                            for k in range(len(flat_texts))) / len(flat_texts)

            epoch_loss += loss.item(); epoch_reward += rewards.mean().item()
            epoch_rep += rep_rate; epoch_lenratio += len_ratio; n += 1

        dev_bleu, dev_loss, samples = evaluate(model, dev_loader, tokenizer, cfg)
        log = {"epoch": epoch, "mrt_loss": epoch_loss / n, "avg_reward": epoch_reward / n,
               "avg_rep_rate": epoch_rep / n, "avg_len_ratio": epoch_lenratio / n,
               "dev_bleu4": dev_bleu, "candidate_source": source, "time_s": time.time() - t0}
        history.append(log)
        print(f"[MRT Ep{epoch}] loss={epoch_loss/n:.4f} reward={epoch_reward/n:.4f} "
              f"rep={epoch_rep/n:.3f} len_ratio={epoch_lenratio/n:.3f} BLEU4={dev_bleu:.2f}")
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}\n  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_mrt.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop MRT ep {epoch}"); break

    with open(os.path.join(log_dir, "mrt_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history
