"""Stage 2 của P7 (Two-stage: Pose -> Gloss -> NMT -> Text, docs/1_Thuyet_Trinh_Tong_Hop.md §A): gloss -> text,
NMT THUẦN TEXT (không đụng tới pose/video nữa) — Transformer encoder-decoder nhỏ hơn model chính
(gloss vocab nhỏ + câu ngắn hơn nhiều so với pose sequence), độc lập hoàn toàn với
SLTTransformer/GlossCTCModel."""
import math
import torch
import torch.nn as nn
from .encoders import PositionalEncoding


class Gloss2TextNMT(nn.Module):
    def __init__(self, gloss_vocab_size: int, text_vocab_size: int,
                d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                d_ff: int = None, dropout: float = 0.2):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.gloss_embed = nn.Embedding(gloss_vocab_size, d_model, padding_idx=0)
        self.text_embed = nn.Embedding(text_vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)

        # BẮT BUỘC có norm=nn.LayerNorm khi norm_first=True (Pre-LN) — giống SLTTransformer
        # (models/slt_transformer.py). Thiếu norm cuối thì residual stream KHÔNG được chuẩn hoá,
        # hidden std phình to qua từng layer -> logits khổng lồ -> CE loss ~10^3 (đo thực tế 1314),
        # softmax bão hoà, gradient policy ~0. Đây đúng bug đã sửa ở model chính nhưng trước đây
        # chưa áp cho nhánh P7 stage 2 này.
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers, norm=nn.LayerNorm(d_model))
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout,
                                               batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers, norm=nn.LayerNorm(d_model))

        self.out_proj = nn.Linear(d_model, text_vocab_size, bias=False)
        self.out_proj.weight = self.text_embed.weight  # weight tying, giống SLTTransformer

        # Init embedding std=d_model^-0.5 (không phải N(0,1) mặc định): decode_step nhân embedding
        # với sqrt(d_model), và do weight tying text_embed CHÍNH là ma trận out_proj — cùng lý do
        # (a)+(b) đã ghi ở SLTTransformer. Sau khi thêm final-norm + init này, logits std ≈ 1.
        for emb in (self.gloss_embed, self.text_embed):
            nn.init.normal_(emb.weight, mean=0.0, std=d_model ** -0.5)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)  # giữ padding_idx=0 là vector 0

    def encode(self, gloss_ids, gloss_mask=None):
        x = self.gloss_embed(gloss_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=gloss_mask)

    def decode_step(self, tgt_ids, memory, memory_mask, tgt_mask=None):
        y = self.text_embed(tgt_ids) * math.sqrt(self.d_model)
        y = self.pos_enc(y)
        L = tgt_ids.size(1)
        causal = torch.triu(torch.ones(L, L, device=tgt_ids.device), diagonal=1).bool()
        out = self.decoder(y, memory, tgt_mask=causal,
                           tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        return self.out_proj(out)

    def forward(self, gloss_ids, gloss_mask, tgt_ids, tgt_mask=None):
        memory = self.encode(gloss_ids, gloss_mask)
        return self.decode_step(tgt_ids, memory, gloss_mask, tgt_mask)

    @torch.no_grad()
    def greedy_decode(self, gloss_ids, gloss_mask, bos_id: int, eos_id: int, max_len: int = 60):
        memory = self.encode(gloss_ids, gloss_mask)
        B = memory.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.decode_step(ys, memory, gloss_mask)
            next_tok = logits[:, -1].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if finished.all(): break
        return ys
