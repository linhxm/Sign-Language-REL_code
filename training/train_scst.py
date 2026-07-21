"""Phase 2: Self-Critical Sequence Training (SCST).
LOAD checkpoint XE tốt nhất → fine-tune bằng RL.

Reward: BLEU(sample) - BLEU(greedy_baseline)  [tính per-sample]
Loss:   -(reward) * sum(log_prob_sample)

Hỗ trợ Multi-sample SCST (cfg.train.rl_n_samples > 1, xem docs/1_Thuyet_Trinh_Tong_Hop.md §E Experiment 3)
và Automatic Mixed Precision (cfg.train.use_amp, xem docs/1_Thuyet_Trinh_Tong_Hop.md §J điểm yếu B.7).
"""
import os, time, json
import torch
from torch.optim import AdamW
from utils.amp_compat import autocast, GradScaler  # shim: torch.cuda.amp bị deprecate từ PyTorch 2.4
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from .train_xe import evaluate
from data.dataset import make_curriculum_loader

_bleu = BLEU(effective_order=True)  # sentence-level

def sentence_bleu(hyp: str, ref: str) -> float:
    return _bleu.sentence_score(hyp, [ref]).score / 100.0

def repetition_penalty(text: str, n: int = 3) -> float:
    """Trả về tỉ lệ n-gram lặp. 0 = không lặp, 1 = lặp toàn bộ."""
    toks = text.split()
    if len(toks) < n: return 0.0
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return 1 - len(set(grams)) / max(1, len(grams))

def length_penalty(hyp: str, ref: str) -> float:
    """Phạt khi câu sinh NGẮN hơn reference (∈ [0,1]).
    0 = đủ dài (>= ref); ->1 khi câu rỗng. Chống reward hacking bằng output cụt.
    Câu rỗng bị phạt tối đa = 1.0; BLEU brevity penalty không đủ vì nó nằm trong
    log và bị các p_n=0 nuốt mất tín hiệu."""
    h = len(hyp.split()); r = len(ref.split())
    if r == 0: return 0.0
    if h >= r: return 0.0
    return (r - h) / r

_bertscorer_cache = {}

def bertscore_reward(hyp: str, ref: str, model_name: str) -> float:
    """Reward 5 (docs/1_Thuyet_Trinh_Tong_Hop.md §E) — F1 BERTScore giữa hyp/ref.
    Lazy-import + cache scorer theo model_name (tránh load lại BERT mỗi lần gọi).
    CHI PHÍ CAO (1 forward BERT/lần gọi, không batch hoá) — chỉ bật (reward_bert_weight>0) ở
    subset nhỏ khi chạy Experiment 9 (docs/1_Thuyet_Trinh_Tong_Hop.md §E), không dùng mặc định."""
    try:
        from bert_score import BERTScorer
    except ImportError as e:
        raise RuntimeError(
            "Cần `pip install bert-score` để dùng reward_bert_weight > 0 (xem KAGGLE_NOTEBOOK.ipynb)."
        ) from e
    if model_name not in _bertscorer_cache:
        _bertscorer_cache[model_name] = BERTScorer(model_type=model_name, lang="de",
                                                    rescale_with_baseline=False)
    _, _, f1 = _bertscorer_cache[model_name].score([hyp], [ref])
    return f1.item()

def compute_reward(hyp: str, ref: str, cfg, rep_weight: float = None, len_weight: float = None) -> float:
    """R = w_bleu*BLEU - w_rep*rep_penalty - w_len*length_penalty + w_bert*BERTScore.
    Mỗi weight đặt 0.0 để tắt thành phần tương ứng (dùng cho reward ablation, Experiment 2/9).
    rep_weight/len_weight: override cfg.train.reward_* -- dùng cho Curriculum reward (Reward 10,
    docs/1_Thuyet_Trinh_Tong_Hop.md §E, C.12) khi cần "ramp" trọng số theo epoch thay vì cố định."""
    if rep_weight is None:
        rep_weight = cfg.train.reward_repetition_penalty
    if len_weight is None:
        len_weight = cfg.train.reward_length_penalty
    r = cfg.train.reward_bleu_weight * sentence_bleu(hyp, ref)
    r -= rep_weight * repetition_penalty(hyp)
    r -= len_weight * length_penalty(hyp, ref)
    if getattr(cfg.train, "reward_bert_weight", 0.0) > 0:
        r += cfg.train.reward_bert_weight * bertscore_reward(hyp, ref, cfg.train.reward_bert_model)
    return r


def train_scst(model, train_loader, dev_loader, tokenizer, cfg, log_dir: str,
               xe_ckpt_path: str):
    device = cfg.device
    # Load XE best checkpoint
    ckpt = torch.load(xe_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded XE checkpoint từ ep{ckpt['epoch']} BLEU={ckpt['bleu']:.2f}")

    opt = AdamW(model.parameters(), lr=cfg.train.rl_lr,
                weight_decay=cfg.train.xe_weight_decay, betas=(0.9, 0.98))

    amp_enabled = bool(getattr(cfg.train, "use_amp", False)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    n_samples = max(1, cfg.train.rl_n_samples)
    use_baseline = bool(getattr(cfg.train, "rl_use_baseline", True))
    curriculum_epochs = int(getattr(cfg.train, "rl_curriculum_epochs", 0))
    curriculum_sort = bool(getattr(cfg.train, "rl_curriculum_length_sort", False))
    # Curriculum RL (C.12) + Ý tưởng F.18: loader riêng cho N epoch đầu, duyệt batch theo câu
    # ngắn->dài (dựa độ dài text tham chiếu) thay vì shuffle ngẫu nhiên -- dễ đạt reward dương sớm,
    # giảm cold-start RL. Xây 1 lần, dùng lại mỗi epoch curriculum (thứ tự batch vẫn xáo trộn qua seed).
    curriculum_loader = (make_curriculum_loader(train_loader, seed=cfg.seed)
                         if curriculum_sort and curriculum_epochs > 0 else None)

    # `best_bleu` khởi tạo bằng dev BLEU của checkpoint XE, nên best_rl.pt chỉ được lưu khi RL
    # THỰC SỰ vượt XE. Nhưng nếu RL không bao giờ vượt thì sẽ KHÔNG có file nào -> main.py bỏ
    # qua ở bước eval -> test_results.json chỉ có "xe" -> KHÔNG BÁO CÁO ĐƯỢC "RL kém hơn CE".
    # Mà "RL kém hơn" là kết quả hợp lệ và bắt buộc phải đo được (H3 trung lập, Experiment 7).
    # Vì vậy: giữ nguyên tiêu chí best, nhưng LUÔN lưu thêm last_rl.pt ở cuối để mọi run đều có
    # checkpoint đem đi eval trên test.
    xe_bleu = ckpt["bleu"]
    best_bleu = xe_bleu; patience = 0
    saved_best = False
    history = []
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg.train.rl_epochs):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_reward, epoch_entropy = 0.0, 0.0, 0.0
        epoch_rep, epoch_lenratio, n = 0.0, 0.0, 0

        # Reward curriculum (Reward 10, mục E + C.12): ramp tuyến tính w_rep/w_len từ 0 -> giá trị
        # cấu hình thật trong `rl_curriculum_epochs` epoch đầu, thay vì bật full ngay từ epoch 0.
        if curriculum_epochs > 0:
            ramp = min(1.0, (epoch + 1) / curriculum_epochs)
        else:
            ramp = 1.0
        rep_w = cfg.train.reward_repetition_penalty * ramp
        len_w = cfg.train.reward_length_penalty * ramp

        if curriculum_loader is not None and epoch < curriculum_epochs:
            curriculum_loader.sampler.set_epoch(epoch)  # đổi seed xáo trộn thứ tự batch mỗi epoch
            loader_this_epoch = curriculum_loader
        else:
            loader_this_epoch = train_loader

        for batch in tqdm(loader_this_epoch, desc=f"RL Ep{epoch}"):
            pose = batch["pose"].to(device); pose_mask = batch["pose_mask"].to(device)
            refs = batch["text_raw"]
            B = pose.size(0)

            # 1. Greedy baseline (no grad) — CHỈ tính khi rl_use_baseline=True (SCST, mặc định).
            #    rl_use_baseline=False = REINFORCE thuần (C.1, docs/1_Thuyet_Trinh_Tong_Hop.md):
            #    bỏ hẳn baseline, advantage = R(sample) trực tiếp -- variance cao hơn nhiều, chỉ
            #    dùng để ablation chứng minh vai trò baseline, không phải mặc định.
            #    rl_baseline_eval_mode=True (mặc định): baseline PHẢI encode() ở eval-mode
            #    (dropout tắt) -> KHÔNG thể share memory với nhánh sample (train-mode), cần 2
            #    lần encode() riêng biệt -- đây là yêu cầu đúng đắn, không phải lãng phí
            #    (xem docs/1_Thuyet_Trinh_Tong_Hop.md §G.22).
            #    rl_baseline_eval_mode=False: cả 2 nhánh cùng train-mode -> AN TOÀN share 1 lần
            #    encode() duy nhất, tiết kiệm ~1 forward encoder/batch
            #    (xem docs/1_Thuyet_Trinh_Tong_Hop.md §J điểm yếu B.6).
            sample_memory = None
            if use_baseline:
                if cfg.train.rl_baseline_eval_mode:
                    model.eval()
                    with torch.no_grad():
                        greedy_memory = model.encode(pose, pose_mask)
                        greedy_ids = model.greedy_decode(pose, pose_mask,
                                                          tokenizer.bos_id, tokenizer.eos_id,
                                                          max_len=cfg.data.max_text_len,
                                                          memory=greedy_memory)
                    model.train()
                    sample_memory = None  # sample_decode tự encode() lại ở train-mode bên dưới
                else:
                    # SỬA BUG (quan trọng): trước đây nhánh này share `shared_memory` — vốn được
                    # tính trong `torch.no_grad()` — sang cho sample_decode. Hậu quả: ENCODER
                    # KHÔNG BAO GIỜ NHẬN GRADIENT từ policy loss, chỉ decoder được fine-tune.
                    # Đây đúng là loại lỗi mà §36 cảnh báo cho PPO, chỉ khác là nó nằm im ở
                    # nhánh này của SCST nên không ai để ý.
                    #
                    # Tác hại cụ thể: ablation `rl_baseline_eval_mode` True vs False (§22) đáng
                    # lẽ đo "baseline eval-mode vs train-mode", nhưng thực tế đang so "encoder
                    # được train" vs "encoder bị đóng băng" -> kết luận rút ra sẽ sai hoàn toàn.
                    #
                    # Cách sửa: chỉ share memory cho nhánh GREEDY (vốn no_grad, không cần
                    # gradient); nhánh sample để sample_decode tự encode lại CÓ gradient. Tốn
                    # thêm 1 forward encoder/batch, đổi lại ablation mới có ý nghĩa.
                    with torch.no_grad():
                        greedy_memory = model.encode(pose, pose_mask)
                        greedy_ids = model.greedy_decode(pose, pose_mask,
                                                          tokenizer.bos_id, tokenizer.eos_id,
                                                          max_len=cfg.data.max_text_len,
                                                          memory=greedy_memory)
                    sample_memory = None

                greedy_texts = [tokenizer.decode(greedy_ids[i].tolist()) for i in range(B)]
                r_greedy = torch.tensor(
                    [compute_reward(greedy_texts[i], refs[i], cfg, rep_w, len_w) for i in range(B)],
                    dtype=torch.float32, device=device)
            else:
                r_greedy = torch.zeros(B, dtype=torch.float32, device=device)

            # 2. Sample (with grad on log_probs). Lặp n_samples lần (Multi-sample SCST,
            #    Experiment 3): mỗi sample dùng chung `sample_memory` (nếu có), chỉ decoder
            #    sampling là nguồn đa dạng giữa các sample -> giảm variance so với 1-sample.
            with autocast(enabled=amp_enabled):
                pg_terms, adv_terms, ent_terms = [], [], []
                sample_texts_all = []
                for _ in range(n_samples):
                    sample_ids, log_probs, entropies = model.sample_decode(
                        pose, pose_mask, tokenizer.bos_id, tokenizer.eos_id,
                        max_len=cfg.data.max_text_len, temperature=cfg.train.rl_sample_temp,
                        memory=sample_memory)
                    sample_texts = [tokenizer.decode(sample_ids[i].tolist()) for i in range(B)]
                    sample_texts_all.extend(sample_texts)
                    r_sample = torch.tensor(
                        [compute_reward(sample_texts[i], refs[i], cfg, rep_w, len_w) for i in range(B)],
                        dtype=torch.float32, device=device)
                    advantages = r_sample - r_greedy  # [B] -- r_greedy=0 khi rl_use_baseline=False (REINFORCE)
                    seq_log_prob = log_probs.sum(dim=1)  # [B]
                    pg_terms.append(-(advantages * seq_log_prob))
                    adv_terms.append(advantages)
                    ent_terms.append(entropies.sum(dim=1))

                pg_loss = torch.stack(pg_terms, dim=0).mean()
                ent_bonus = torch.stack(ent_terms, dim=0).mean()
                loss = pg_loss - cfg.train.rl_entropy_coef * ent_bonus

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()

            avg_adv = torch.stack(adv_terms, dim=0).mean().item()
            # Giám sát reward-hacking: rep-rate/length-ratio đo trực tiếp trên text sinh ra,
            # KHÔNG chỉ dựa vào reward tổng hợp (điểm yếu C.13, docs/1_Thuyet_Trinh_Tong_Hop.md §J) --
            # BLEU có thể "trông tốt" trong khi model đang lặp/cụt câu.
            rep_rate = sum(repetition_penalty(t) for t in sample_texts_all) / len(sample_texts_all)
            len_ratio = sum(length_penalty(sample_texts_all[i], refs[i % B])
                            for i in range(len(sample_texts_all))) / len(sample_texts_all)

            epoch_loss += loss.item(); epoch_reward += avg_adv
            epoch_entropy += ent_bonus.item()
            epoch_rep += rep_rate; epoch_lenratio += len_ratio
            n += 1

        # Eval
        dev_bleu, dev_loss, samples = evaluate(model, dev_loader, tokenizer, cfg)
        log = {"epoch": epoch, "rl_loss": epoch_loss/n, "avg_advantage": epoch_reward/n,
               "avg_entropy": epoch_entropy/n, "avg_rep_rate": epoch_rep/n,
               "avg_len_ratio": epoch_lenratio/n, "dev_bleu4": dev_bleu,
               "curriculum_ramp": ramp, "use_baseline": use_baseline,
               "time_s": time.time()-t0}
        history.append(log)
        print(f"[RL Ep{epoch}] loss={epoch_loss/n:.4f} adv={epoch_reward/n:.4f} "
              f"H={epoch_entropy/n:.3f} rep={epoch_rep/n:.3f} len_ratio={epoch_lenratio/n:.3f} "
              f"BLEU4={dev_bleu:.2f}")
        for gt, pred in samples[:2]:
            print(f"  GT  : {gt}\n  PRED: {pred}")

        if dev_bleu > best_bleu:
            best_bleu = dev_bleu; patience = 0; saved_best = True
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": dev_bleu},
                       os.path.join(log_dir, "best_rl.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop RL ep {epoch}"); break

    # Luôn có checkpoint cuối, kể cả khi RL không vượt XE (xem ghi chú ở chỗ khởi tạo best_bleu).
    torch.save({"model": model.state_dict(), "epoch": len(history) - 1,
                "bleu": history[-1]["dev_bleu4"] if history else xe_bleu},
               os.path.join(log_dir, "last_rl.pt"))
    if not saved_best:
        print(f"[!] RL KHÔNG vượt XE (dev BLEU của XE = {xe_bleu:.2f}). Không có best_rl.pt — "
              f"dùng last_rl.pt để eval và HÃY BÁO CÁO kết quả âm này thay vì bỏ qua: đó là "
              f"finding hợp lệ (H3, docs/1_Thuyet_Trinh_Tong_Hop.md §E Experiment 7).")

    with open(os.path.join(log_dir, "rl_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_bleu, history
