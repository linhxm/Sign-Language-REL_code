"""Transformer-based SLT. Encoder = pose features (P1-P6, docs/1_Thuyet_Trinh_Tong_Hop.md §A); Decoder =
autoregressive text (dùng chung cho mọi lựa chọn encoder)."""
import math
import torch
import torch.nn as nn
from .encoders import PositionalEncoding, PoseEmbed, build_pose_encoder

# Re-export để code cũ (nếu còn import trực tiếp từ module này) không bị vỡ.
__all__ = ["PositionalEncoding", "PoseEmbed", "SLTTransformer", "ValueHead"]


class SLTTransformer(nn.Module):
    def __init__(self, cfg, vocab_size: int, pose_dim: int = 183, encoder_type: str = "transformer"):
        """encoder_type: "transformer"|"stgcn"|"gcn"|"graph_transformer"|"tcn"|"perceiver"
        (P1/P3/P2/P4/P5/P6 — docs/1_Thuyet_Trinh_Tong_Hop.md §A), dựng qua factory `models.encoders.build_pose_encoder`
        để mọi biến thể cùng interface `forward(pose, pose_mask) -> [B,T,d_model]`. Decoder giữ
        nguyên ở mọi lựa chọn encoder để cô lập đúng 1 biến so sánh (kiến trúc encoder)."""
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.encoder_type = encoder_type
        self.pos_enc = PositionalEncoding(m.d_model)
        self.tok_embed = nn.Embedding(vocab_size, m.d_model, padding_idx=0)

        self.pose_encoder = build_pose_encoder(cfg, pose_dim, encoder_type)

        dec_layer = nn.TransformerDecoderLayer(m.d_model, m.n_heads, m.d_ff,
                                               m.dropout, batch_first=True, norm_first=True)
        # BẮT BUỘC có `norm=` khi norm_first=True (Pre-LN). nn.TransformerDecoder mặc định
        # norm=None; với Post-LN thì không sao vì mỗi layer đã kết thúc bằng LayerNorm, nhưng
        # với Pre-LN thì LayerNorm nằm ở ĐẦU mỗi sub-layer nên output cuối cùng là residual
        # stream CHƯA chuẩn hoá -- độ lớn cộng dồn qua từng layer.
        # Đo thực tế trước khi sửa: hidden std = 17.8 (đáng lẽ ~1) -> logits std = 304.
        # Hậu quả: softmax bão hoà thành one-hot -> log_prob = 0.0 CHÍNH XÁC -> gradient của
        # policy gradient bằng 0 -> RL không học được gì. Xem docs/1_Thuyet_Trinh_Tong_Hop.md §K.
        self.decoder = nn.TransformerDecoder(dec_layer, m.n_dec_layers,
                                             norm=nn.LayerNorm(m.d_model))

        self.out_proj = nn.Linear(m.d_model, vocab_size, bias=False)
        # Weight tying (Press & Wolf 2017) — out_proj DÙNG CHUNG tensor với tok_embed.
        self.out_proj.weight = self.tok_embed.weight

        # Init embedding theo std = d_model^-0.5 thay vì N(0,1) mặc định của nn.Embedding.
        # Lý do kép:
        #  (a) decode_step nhân embedding với sqrt(d_model) (chuẩn Vaswani 2017) — chuẩn đó giả
        #      định embedding có phương sai 1/d_model để sau khi nhân thì về đúng scale ~1.
        #      Với N(0,1) mặc định thì input decoder bị thổi lên 16 lần.
        #  (b) Do weight tying, CHÍNH tensor này cũng là ma trận out_proj: rows có norm ~sqrt(256)
        #      làm logits lớn thêm một lần nữa.
        # Sau khi sửa (a)+(b)+final norm: logits std ≈ 1.0, entropy ban đầu ≈ 7.5/8.0 nats —
        # policy còn đủ entropy để explore khi vào phase RL (khớp chủ đích của label_smoothing, §G.8).
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=m.d_model ** -0.5)
        with torch.no_grad():
            self.tok_embed.weight[0].fill_(0.0)   # giữ padding_idx=0 là vector 0

        self.vocab_size = vocab_size

    def encode(self, pose, pose_mask):
        return self.pose_encoder(pose, pose_mask)  # [B, T, D]

    def decode_step(self, tgt_ids, memory, memory_mask, tgt_mask=None, return_hidden: bool = False):
        """Dùng cho cả training (teacher forcing) lẫn inference.
        return_hidden=True: trả thêm hidden state trước out_proj (dùng cho value head của PPO,
        xem training/train_ppo.py) — không ảnh hưởng các lời gọi cũ (mặc định False)."""
        y = self.tok_embed(tgt_ids) * math.sqrt(self.d_model)
        y = self.pos_enc(y)
        L = tgt_ids.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt_ids.device), diagonal=1).bool()
        out = self.decoder(y, memory, tgt_mask=causal,
                           tgt_key_padding_mask=tgt_mask,
                           memory_key_padding_mask=memory_mask)
        logits = self.out_proj(out)  # [B, L, V]
        if return_hidden:
            return logits, out  # out: [B, L, D] — hidden trước out_proj
        return logits

    def forward(self, pose, pose_mask, tgt_ids, tgt_mask=None):
        memory = self.encode(pose, pose_mask)
        return self.decode_step(tgt_ids, memory, pose_mask, tgt_mask)

    @torch.no_grad()
    def greedy_decode(self, pose, pose_mask, bos_id: int, eos_id: int, max_len: int = 60,
                      memory=None):
        """memory: nếu đã có sẵn (encode() gọi trước đó), truyền vào để tránh forward encoder
        lại lần nữa — quan trọng trong SCST/PPO nơi greedy+sample decode dùng chung 1 batch input
        (xem docs/1_Thuyet_Trinh_Tong_Hop.md §J điểm yếu B.6)."""
        if memory is None:
            memory = self.encode(pose, pose_mask)
        B = memory.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.decode_step(ys, memory, pose_mask)
            next_tok = logits[:, -1].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if finished.all(): break
        return ys

    def sample_decode(self, pose, pose_mask, bos_id: int, eos_id: int,
                      max_len: int = 60, temperature: float = 1.0,
                      memory=None, return_hidden: bool = False):
        """Sample sequence cho SCST/PPO. Trả về:
            ys         [B, L]    : token đã sample (sau EOS được nhồi EOS).
            log_probs  [B, L-1]  : log π của token đã bốc (0 ở vị trí sau khi câu kết thúc).
            entropies  [B, L-1]  : entropy H(π) mỗi bước (0 sau khi kết thúc) — cho entropy bonus.
            hiddens    [B, L-1, D] (chỉ nếu return_hidden=True): hidden state trước out_proj mỗi
                       bước — input cho value head PPO (training/train_ppo.py).
        Các vị trí sau EOS để 0 nên không đóng góp vào gradient/loss.
        memory: xem ghi chú ở greedy_decode."""
        if memory is None:
            memory = self.encode(pose, pose_mask)
        B = memory.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(B, dtype=torch.bool, device=memory.device)
        log_probs_list, entropy_list, hidden_list = [], [], []
        for _ in range(max_len - 1):
            logits, hidden = self.decode_step(ys, memory, pose_mask, return_hidden=True)
            logits_last = logits[:, -1] / temperature
            dist = torch.distributions.Categorical(logits=logits_last)
            next_tok = dist.sample()
            lp = dist.log_prob(next_tok)
            ent = dist.entropy()
            # Mask out finished sequences (sau EOS không còn action thật)
            lp = lp.masked_fill(finished, 0.0)
            ent = ent.masked_fill(finished, 0.0)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            log_probs_list.append(lp); entropy_list.append(ent)
            if return_hidden:
                hidden_list.append(hidden[:, -1])
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if finished.all(): break
        log_probs = torch.stack(log_probs_list, dim=1)  # [B, L-1]
        entropies = torch.stack(entropy_list, dim=1)    # [B, L-1]
        if return_hidden:
            hiddens = torch.stack(hidden_list, dim=1)   # [B, L-1, D]
            return ys, log_probs, entropies, hiddens
        return ys, log_probs, entropies

    @torch.no_grad()
    def beam_search_decode(self, pose, pose_mask, bos_id: int, eos_id: int,
                           max_len: int = 60, beam_size: int = 4,
                           length_penalty: float = 0.6, memory=None,
                           return_all_beams: bool = False):
        """Beam search chuẩn (per-sample, không batch-song song giữa các sample để đơn giản hoá
        — chấp nhận chậm hơn vì chỉ dùng lúc eval/so sánh cuối, không nằm trong training loop).
        Điểm yếu C.12 (docs/1_Thuyet_Trinh_Tong_Hop.md §J): trước bản này, evaluate() chỉ có greedy, không
        so sánh công bằng được với số liệu literature (đa số báo cáo beam=4-5).

        return_all_beams=True: trả thêm `all_beams` (list độ dài B, mỗi phần tử là list các
        (ys_1d, score) của TẤT CẢ beam cuối cùng, không chỉ beam tốt nhất) — dùng làm nguồn candidate
        cho MRT (C.9, training/train_mrt.py) khi `mrt_candidate_source="beam"`, cũng chính là cách
        hiện thực hoá Ý tưởng F.13 (RL/MRT cho beam search policy, docs/1_Thuyet_Trinh_Tong_Hop.md
        mục F) mà không cần viết riêng 1 vòng lặp beam thứ 2."""
        if memory is None:
            memory = self.encode(pose, pose_mask)
        B = memory.size(0)
        outputs = []
        all_beams = []
        for b in range(B):
            mem_b = memory[b:b+1]
            mask_b = pose_mask[b:b+1] if pose_mask is not None else None
            beams = [(torch.tensor([[bos_id]], device=memory.device), 0.0, False)]  # (ys, logprob_sum, finished)
            for _ in range(max_len - 1):
                candidates = []
                for ys, score, fin in beams:
                    if fin:
                        candidates.append((ys, score, fin))
                        continue
                    mem_rep = mem_b.expand(1, -1, -1)
                    mask_rep = mask_b
                    logits = self.decode_step(ys, mem_rep, mask_rep)
                    log_probs = torch.log_softmax(logits[0, -1], dim=-1)  # [V]
                    topk_lp, topk_idx = log_probs.topk(beam_size)
                    for k in range(beam_size):
                        tok = topk_idx[k].view(1, 1)
                        new_ys = torch.cat([ys, tok], dim=1)
                        new_score = score + topk_lp[k].item()
                        new_fin = fin or (tok.item() == eos_id)
                        candidates.append((new_ys, new_score, new_fin))
                # length-normalized score để chọn top beam_size
                def norm_score(item):
                    ys, score, _ = item
                    L = ys.size(1)
                    return score / (L ** length_penalty)
                candidates.sort(key=norm_score, reverse=True)
                beams = candidates[:beam_size]
                if all(fin for _, _, fin in beams):
                    break
            best_ys = max(beams, key=norm_score)[0]
            outputs.append(best_ys[0])
            if return_all_beams:
                all_beams.append([(ys[0], score) for ys, score, _ in beams])
        max_L = max(o.size(0) for o in outputs)
        padded = torch.full((B, max_L), eos_id, dtype=torch.long, device=memory.device)
        for i, o in enumerate(outputs):
            padded[i, :o.size(0)] = o
        if return_all_beams:
            return padded, all_beams
        return padded


class ValueHead(nn.Module):
    """Critic V(s_t) cho PPO (training/train_ppo.py) — nhận hidden state decoder mỗi bước
    (trước out_proj, xem SLTTransformer.decode_step(..., return_hidden=True)) và dự đoán giá trị
    kỳ vọng của reward cuối câu tính từ bước đó. Tách riêng khỏi SLTTransformer (không share
    tham số) để không ảnh hưởng checkpoint XE/SCST đã có — chỉ cần thêm module này khi chạy PPO."""
    def __init__(self, d_model: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_states):  # [B, L, D] -> [B, L]
        return self.net(hidden_states).squeeze(-1)
